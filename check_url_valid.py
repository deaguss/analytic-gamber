import os
import re
import json
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlparse, parse_qs


class Config:
    INPUT_FILE   = Path("checks/candidate_urls.json")
    OUTPUT_FILE  = Path("checks/selected_urls.json")
    YOUTUBE_URLS_OUTPUT = Path("checks/youtube_urls_selected.json")
    LOG_FILE     = Path("logs/video_selector.log")
    APP_LOG_FILE = Path("logs/app.log")

    MAX_COMMENTS_PER_VIDEO = 200
    SCRAPE_DELAY_SECONDS   = 2

    HIGH_THRESHOLD = float(os.getenv("HIGH_THRESHOLD", "0.75"))
    LOW_THRESHOLD  = float(os.getenv("LOW_THRESHOLD",  "0.50"))

    JUDI_VIDEO_THRESHOLD  = 20.0
    BUKAN_VIDEO_THRESHOLD = 40.0

    TARGET_PER_CATEGORY = 10


def setup_logging(log_file: Path, app_log: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    fmt     = '[%(asctime)s] [%(levelname)s] %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'

    logger = logging.getLogger("video_selector")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(fmt, datefmt))
        logger.addHandler(sh)

        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(logging.Formatter(fmt, datefmt))
        logger.addHandler(fh)

        ah = logging.FileHandler(app_log, encoding='utf-8')
        ah.setFormatter(logging.Formatter(fmt, datefmt))
        logger.addHandler(ah)

    return logger


@dataclass
class Comment:
    comment_id: str
    text: str
    timestamp: str
    normalized_text: Optional[str] = None
    tokens: Optional[List[str]] = None
    clean_tokens: Optional[List[str]] = None


@dataclass
class CommentClassification:
    classification: str
    score: float
    method: str


@dataclass
class VideoStats:
    video_id:   str
    video_url:  str
    total:      int
    n_judi:     int
    n_bukan:    int
    n_ambigu:   int
    pct_judi:   float
    pct_bukan:  float
    pct_ambigu: float
    category:   str
    scrape_ok:  bool = True
    error:      Optional[str] = None


class TextPreprocessor:

    LEET_MAP = {
        '0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's',
        '7': 't', '8': 'b', '@': 'a', '$': 's', '!': 'i'
    }

    STOPWORDS = {
        'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan',
        'untuk', 'pada', 'adalah', 'atau', 'juga', 'dalam', 'tidak',
        'ada', 'akan', 'sudah', 'saya', 'kamu', 'dia', 'mereka',
        'kita', 'kami', 'nya', 'ter', 'ber', 'pe', 'me'
    }

    @classmethod
    def _leet(cls, text: str) -> str:
        return ''.join(cls.LEET_MAP.get(c, c) for c in text)

    @classmethod
    def normalize(cls, text: str) -> str:
        t = text.lower()
        t = cls._leet(t)
        t = re.sub(r'[^\w\s]', ' ', t)
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    @classmethod
    def preprocess(cls, comment: Comment) -> Comment:
        comment.normalized_text = cls.normalize(comment.text)
        tokens = comment.normalized_text.split()
        comment.tokens       = tokens
        comment.clean_tokens = [t for t in tokens if t not in cls.STOPWORDS]
        return comment


class RuleBasedClassifier:

    SPAM_PATTERNS = {
        'gambling': r'(slot|gacor|rtp|bonus|bet|maxwin|deposit|judi|casino|situs|pragmatic|pg|soft|togel|bandar|jackpot)',
        'urls':     r'(https?://[^\s]+\.(bet|casino|slot|gaming)|t\.me/|wa\.me/|bit\.ly)',
        'promo':    r'(daftar|register|klik|link|wede|depo|wd|withdraw|bonus\s*\d+|promo|claim)',
        'contact':  r'(\d{10,}|wa\s*\d+|kontak|hubungi|cs\s*online|admin|dm|chat)',
        'emoji_spam': r'([\U0001F3B0\U0001F3B2\U0001F4B0\U0001F4B8\U0001F911\U0001F48E\U0001F525]){2,}'
    }

    SPAM_KEYWORDS = [
        'slot', 'gacor', 'judi', 'casino', 'togel', 'bandar',
        'maxwin', 'deposit', 'bonus', 'pragmatic', 'daftar',
        'jackpot', 'scatter', 'zeus', 'olympus', 'gates'
    ]

    @classmethod
    def _regex_match(cls, text: str) -> Tuple[bool, Optional[str]]:
        for name, pattern in cls.SPAM_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                return True, name
        return False, None

    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return RuleBasedClassifier._levenshtein(s2, s1)
        if not s2:
            return len(s1)
        prev = range(len(s2) + 1)
        for c1 in s1:
            curr = [prev[0] + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(prev[j + 1] + 1, curr[-1] + 1, prev[j] + (c1 != c2)))
            prev = curr
        return prev[-1]

    @classmethod
    def _fuzzy_score(cls, tokens: List[str]) -> float:
        best = 0.0
        for token in tokens:
            for kw in cls.SPAM_KEYWORDS:
                ml = max(len(token), len(kw))
                if ml == 0:
                    continue
                score = 1 - cls._levenshtein(token, kw) / ml
                if score > best:
                    best = score
        return best

    @classmethod
    def classify(cls, comment: Comment,
                 high_threshold: float = Config.HIGH_THRESHOLD,
                 low_threshold:  float = Config.LOW_THRESHOLD) -> CommentClassification:

        text = comment.normalized_text or comment.text

        matched, pattern = cls._regex_match(text)
        if matched:
            return CommentClassification("JUDI_ONLINE", 1.0, "Regex")

        tokens = comment.clean_tokens or comment.tokens or []
        score  = cls._fuzzy_score(tokens)

        if score >= high_threshold:
            label = "JUDI_ONLINE"
        elif score < low_threshold:
            label = "BUKAN_JUDI_ONLINE"
        else:
            label = "AMBIGU"

        return CommentClassification(label, score, "Fuzzy Matching")


class CommentScraper:

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._init_downloader()

    def _init_downloader(self):
        try:
            from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT
            self.downloader = YoutubeCommentDownloader()
            self.sort_mode  = SORT_BY_RECENT
            self.available  = True
        except ImportError:
            self.logger.error("youtube-comment-downloader tidak terinstall!")
            self.logger.error("  pip install youtube-comment-downloader")
            self.available = False

    @staticmethod
    def extract_video_id(url: str) -> Optional[str]:
        if 'youtu.be/' in url:
            return url.split('youtu.be/')[-1].split('?')[0]
        if 'youtube.com' in url:
            parsed = urlparse(url)
            if 'watch' in parsed.path:
                qs = parse_qs(parsed.query)
                return qs.get('v', [None])[0]
            if 'embed' in parsed.path:
                return parsed.path.split('/embed/')[-1]
        return None

    def scrape(self, url: str, max_comments: int) -> Tuple[List[Comment], Optional[str]]:
        if not self.available:
            return [], "youtube-comment-downloader tidak tersedia"

        video_id = self.extract_video_id(url)
        if not video_id:
            return [], f"Video ID tidak dapat diekstrak dari URL: {url}"

        comments: List[Comment] = []
        counter  = 0

        try:
            gen = self.downloader.get_comments_from_url(url, sort_by=self.sort_mode)
            for raw in gen:
                counter += 1
                comments.append(Comment(
                    comment_id = f"{video_id}_{counter}",
                    text       = raw.get('text', ''),
                    timestamp  = str(raw.get('time', '')),
                ))
                if max_comments > 0 and counter >= max_comments:
                    break
        except Exception as exc:
            return comments, str(exc)

        return comments, None


class VideoClassifier:

    def __init__(self,
                 high_threshold:     float = Config.HIGH_THRESHOLD,
                 low_threshold:      float = Config.LOW_THRESHOLD,
                 judi_video_thresh:  float = Config.JUDI_VIDEO_THRESHOLD,
                 bukan_video_thresh: float = Config.BUKAN_VIDEO_THRESHOLD):

        self.high_threshold     = high_threshold
        self.low_threshold      = low_threshold
        self.judi_video_thresh  = judi_video_thresh
        self.bukan_video_thresh = bukan_video_thresh

    def classify_video(self, url: str, comments: List[Comment]) -> VideoStats:
        video_id = CommentScraper.extract_video_id(url) or url

        if not comments:
            return VideoStats(
                video_id=video_id, video_url=url,
                total=0, n_judi=0, n_bukan=0, n_ambigu=0,
                pct_judi=0.0, pct_bukan=0.0, pct_ambigu=0.0,
                category="VIDEO_AMBIGU", scrape_ok=False,
                error="Tidak ada komentar"
            )

        n_judi = n_bukan = n_ambigu = 0

        for comment in comments:
            TextPreprocessor.preprocess(comment)
            result = RuleBasedClassifier.classify(
                comment, self.high_threshold, self.low_threshold
            )
            if result.classification == "JUDI_ONLINE":
                n_judi += 1
            elif result.classification == "BUKAN_JUDI_ONLINE":
                n_bukan += 1
            else:
                n_ambigu += 1

        total     = len(comments)
        pct_judi  = n_judi  / total * 100
        pct_bukan = n_bukan / total * 100
        pct_ambi  = n_ambigu / total * 100

        if pct_judi >= self.judi_video_thresh:
            category = "VIDEO_DOMINAN_JUDI"
        elif pct_bukan >= self.bukan_video_thresh:
            category = "VIDEO_TIDAK_JUDI"
        else:
            category = "VIDEO_AMBIGU"

        return VideoStats(
            video_id   = video_id,
            video_url  = url,
            total      = total,
            n_judi     = n_judi,
            n_bukan    = n_bukan,
            n_ambigu   = n_ambigu,
            pct_judi   = round(pct_judi,  2),
            pct_bukan  = round(pct_bukan, 2),
            pct_ambigu = round(pct_ambi,  2),
            category   = category,
        )


class VideoSelector:

    def __init__(self, config: Config, logger: logging.Logger):
        self.cfg        = config
        self.logger     = logger
        self.scraper    = CommentScraper(logger)
        self.classifier = VideoClassifier(
            high_threshold     = config.HIGH_THRESHOLD,
            low_threshold      = config.LOW_THRESHOLD,
            judi_video_thresh  = config.JUDI_VIDEO_THRESHOLD,
            bukan_video_thresh = config.BUKAN_VIDEO_THRESHOLD,
        )

    def run(self, urls: List[str]) -> Dict:
        self.logger.info("=" * 70)
        self.logger.info("VIDEO SELECTOR - MULAI PROSES")
        self.logger.info(f"  Total kandidat URL : {len(urls)}")
        self.logger.info(f"  Max komentar/video : {self.cfg.MAX_COMMENTS_PER_VIDEO}")
        self.logger.info(f"  Threshold judi     : >= {self.cfg.JUDI_VIDEO_THRESHOLD}%")
        self.logger.info(f"  Threshold tdk judi : >= {self.cfg.BUKAN_VIDEO_THRESHOLD}%")
        self.logger.info(f"  Target per kategori: {self.cfg.TARGET_PER_CATEGORY}")
        self.logger.info("=" * 70)

        all_stats: List[VideoStats] = []

        for idx, url in enumerate(urls, 1):
            self.logger.info(f"\n[{idx}/{len(urls)}] {url}")
            stats = self._process_one(url)
            all_stats.append(stats)
            self._log_video_result(stats)

            if idx < len(urls):
                time.sleep(self.cfg.SCRAPE_DELAY_SECONDS)

        return self._build_output(all_stats)

    def _process_one(self, url: str) -> VideoStats:
        comments, err = self.scraper.scrape(url, self.cfg.MAX_COMMENTS_PER_VIDEO)
        vid = CommentScraper.extract_video_id(url) or url

        if err and not comments:
            self.logger.warning(f"  Scrape gagal: {err}")
            return VideoStats(
                video_id=vid, video_url=url,
                total=0, n_judi=0, n_bukan=0, n_ambigu=0,
                pct_judi=0.0, pct_bukan=0.0, pct_ambigu=0.0,
                category="VIDEO_AMBIGU", scrape_ok=False, error=err
            )

        if len(comments) < self.cfg.MAX_COMMENTS_PER_VIDEO:
            self.logger.warning(
                f"  Komentar hanya {len(comments)} (< {self.cfg.MAX_COMMENTS_PER_VIDEO}) - video dilewati"
            )
            return VideoStats(
                video_id=vid, video_url=url,
                total=len(comments),
                n_judi=0, n_bukan=0, n_ambigu=0,
                pct_judi=0.0, pct_bukan=0.0, pct_ambigu=0.0,
                category="VIDEO_AMBIGU",
                scrape_ok=False,
                error="Komentar kurang dari minimum 200"
            )

        stats = self.classifier.classify_video(url, comments)
        return stats

    def _log_video_result(self, s: VideoStats):
        if not s.scrape_ok:
            self.logger.warning(f"  [GAGAL] {s.video_id} - {s.error}")
            return
        self.logger.info(
            f"  {s.video_id} | {s.total} komentar | "
            f"Judi: {s.pct_judi:.1f}% | Bukan: {s.pct_bukan:.1f}% | "
            f"Ambigu: {s.pct_ambigu:.1f}% | --> {s.category}"
        )

    def _build_output(self, all_stats: List[VideoStats]) -> Dict:
        dominan_judi = [s for s in all_stats if s.category == "VIDEO_DOMINAN_JUDI" and s.scrape_ok]
        tidak_judi   = [s for s in all_stats if s.category == "VIDEO_TIDAK_JUDI"   and s.scrape_ok]
        ambigu       = [s for s in all_stats if s.category == "VIDEO_AMBIGU"        and s.scrape_ok]
        gagal        = [s for s in all_stats if not s.scrape_ok]

        dominan_judi.sort(key=lambda s: s.pct_judi,   reverse=True)
        tidak_judi.sort(  key=lambda s: s.pct_bukan,  reverse=True)
        ambigu.sort(      key=lambda s: s.pct_ambigu, reverse=True)

        n = self.cfg.TARGET_PER_CATEGORY
        selected_judi  = dominan_judi[:n]
        selected_bukan = tidak_judi[:n]
        selected_ambi  = ambigu[:n]

        output = {
            "generated_at": datetime.now().isoformat(),
            "config": {
                "max_comments_per_video"   : self.cfg.MAX_COMMENTS_PER_VIDEO,
                "high_threshold_fuzzy"     : self.cfg.HIGH_THRESHOLD,
                "low_threshold_fuzzy"      : self.cfg.LOW_THRESHOLD,
                "judi_video_threshold_pct" : self.cfg.JUDI_VIDEO_THRESHOLD,
                "bukan_video_threshold_pct": self.cfg.BUKAN_VIDEO_THRESHOLD,
                "target_per_category"      : self.cfg.TARGET_PER_CATEGORY,
            },
            "summary": {
                "total_kandidat"          : len(all_stats),
                "total_berhasil_discrape" : len(all_stats) - len(gagal),
                "total_gagal"             : len(gagal),
                "total_dominan_judi_found": len(dominan_judi),
                "total_tidak_judi_found"  : len(tidak_judi),
                "total_ambigu_found"      : len(ambigu),
                "total_selected"          : len(selected_judi) + len(selected_bukan) + len(selected_ambi),
            },
            "selected_urls": {
                "VIDEO_DOMINAN_JUDI": [s.video_url for s in selected_judi],
                "VIDEO_TIDAK_JUDI"  : [s.video_url for s in selected_bukan],
                "VIDEO_AMBIGU"      : [s.video_url for s in selected_ambi],
            },
            "selected_details": {
                "VIDEO_DOMINAN_JUDI": [asdict(s) for s in selected_judi],
                "VIDEO_TIDAK_JUDI"  : [asdict(s) for s in selected_bukan],
                "VIDEO_AMBIGU"      : [asdict(s) for s in selected_ambi],
            },
            "all_classified": [asdict(s) for s in all_stats],
        }

        return output


class OutputWriter:

    @staticmethod
    def save(output: Dict, path: Path, logger: logging.Logger):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info(f"\nHasil disimpan ke: {path}")

    # dipakai oleh detector.py / scraper.py tanpa harus copy-paste URL satu per satu.
    @staticmethod
    def save_as_youtube_urls(output: Dict, path: Path, logger: logging.Logger):
        sel = output.get("selected_urls", {})
        cfg = output.get("config", {})

        cat_map = {
            "VIDEO_DOMINAN_JUDI": "JUDI",
            "VIDEO_TIDAK_JUDI"  : "TIDAK_JUDI",
            "VIDEO_AMBIGU"      : "AMBIGU",
        }

        videos = []
        for raw_cat, sample_cat in cat_map.items():
            for url in sel.get(raw_cat, []):
                videos.append({
                    "url"            : url,
                    "sample_category": sample_cat,
                })

        youtube_urls_data = {
            "description": (
                f"Dataset penelitian deteksi komentar judi YouTube "
                f"— {len(videos)} video terpilih otomatis oleh check_url_valid.py"
            ),
            "generated_at": output.get("generated_at", datetime.now().isoformat()),
            "selection_config": {
                "judi_video_threshold_pct" : cfg.get("judi_video_threshold_pct"),
                "bukan_video_threshold_pct": cfg.get("bukan_video_threshold_pct"),
                "max_comments_per_video"   : cfg.get("max_comments_per_video"),
                "target_per_category"      : cfg.get("target_per_category"),
            },
            "categories": {
                "JUDI"      : "Video berindikasi komentar judi online dominan",
                "TIDAK_JUDI": "Video dengan komentar dominan bukan judi",
                "AMBIGU"    : "Video dengan komentar campuran atau tidak jelas",
                "RANDOM"    : "Video acak untuk uji efektivitas (isi manual)",
            },
            "note_random": (
                "Tambahkan 5 URL random secara manual ke bagian bawah array 'videos' "
                "dengan sample_category = 'RANDOM'. "
                "Contoh: {\"url\": \"https://www.youtube.com/watch?v=XXX\", \"sample_category\": \"RANDOM\"}"
            ),
            "videos": videos,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(youtube_urls_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Format youtube_urls.json disimpan ke : {path}")
        logger.info(f"  Total video terpilih : {len(videos)}")
        logger.info(f"    JUDI       : {len(sel.get('VIDEO_DOMINAN_JUDI', []))}")
        logger.info(f"    TIDAK_JUDI : {len(sel.get('VIDEO_TIDAK_JUDI', []))}")
        logger.info(f"    AMBIGU     : {len(sel.get('VIDEO_AMBIGU', []))}")
        logger.info(f"  [!] Tambahkan 5 URL RANDOM secara manual ke file tersebut")

    @staticmethod
    def print_summary(output: Dict, logger: logging.Logger):
        cfg  = output['config']
        summ = output['summary']
        sel  = output['selected_urls']
        det  = output['selected_details']

        lines = [
            "",
            "=" * 70,
            "RINGKASAN HASIL SELEKSI VIDEO",
            "=" * 70,
            "",
            f"  Total kandidat URL     : {summ['total_kandidat']}",
            f"  Berhasil di-scrape     : {summ['total_berhasil_discrape']}",
            f"  Gagal di-scrape        : {summ['total_gagal']}",
            "",
            f"  Ditemukan:",
            f"    VIDEO_DOMINAN_JUDI   : {summ['total_dominan_judi_found']} video",
            f"    VIDEO_TIDAK_JUDI     : {summ['total_tidak_judi_found']} video",
            f"    VIDEO_AMBIGU         : {summ['total_ambigu_found']} video",
            "",
            f"  Terpilih (top-N = {cfg['target_per_category']}):",
            f"    VIDEO_DOMINAN_JUDI   : {len(sel['VIDEO_DOMINAN_JUDI'])} video",
            f"    VIDEO_TIDAK_JUDI     : {len(sel['VIDEO_TIDAK_JUDI'])} video",
            f"    VIDEO_AMBIGU         : {len(sel['VIDEO_AMBIGU'])} video",
            f"    TOTAL TERPILIH       : {summ['total_selected']} video",
        ]

        missing = []
        n = cfg['target_per_category']
        for cat, label in [
            ('VIDEO_DOMINAN_JUDI', 'dominan judi'),
            ('VIDEO_TIDAK_JUDI',   'tidak judi'),
            ('VIDEO_AMBIGU',       'ambigu'),
        ]:
            if len(sel[cat]) < n:
                missing.append(f"{label} (ada {len(sel[cat])}, butuh {n})")

        if missing:
            lines.append("")
            lines.append("  [PERHATIAN] Kandidat kurang untuk kategori:")
            for m in missing:
                lines.append(f"    - {m}")
            lines.append("  Tambah lebih banyak URL kandidat dan jalankan ulang.")

        lines += ["", "-" * 70, "URL TERPILIH:", "-" * 70]

        for cat in ['VIDEO_DOMINAN_JUDI', 'VIDEO_TIDAK_JUDI', 'VIDEO_AMBIGU']:
            lines.append(f"\n  [{cat}]")
            if sel[cat]:
                for i, url in enumerate(sel[cat], 1):
                    detail = det[cat][i - 1]
                    lines.append(
                        f"    {i:2}. {url}"
                        f"  ({detail['pct_judi']:.1f}% judi | "
                        f"{detail['pct_bukan']:.1f}% bukan | "
                        f"{detail['pct_ambigu']:.1f}% ambigu)"
                    )
            else:
                lines.append("      (tidak ada video yang memenuhi kriteria)")

        lines += ["", "=" * 70, ""]

        for line in lines:
            logger.info(line)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Video Selector untuk penelitian deteksi komentar judi YouTube.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  python check_url_valid.py --input checks/candidate_urls.json
  python check_url_valid.py --input urls.json --output hasil.json --target 10
  python check_url_valid.py --input urls.json --judi-threshold 60 --bukan-threshold 70
  python check_url_valid.py --input urls.json --max-comments 150 --delay 3

Format file input (candidate_urls.json):
  ["https://www.youtube.com/watch?v=AAA", "https://www.youtube.com/watch?v=BBB", ...]

Output yang dihasilkan:
  1. checks/selected_urls.json          (format lengkap dengan detail)
  2. checks/youtube_urls_selected.json  (format siap pakai untuk detector.py)
        """
    )

    parser.add_argument('--input',  '-i', type=Path, default=Config.INPUT_FILE,
                        help='File JSON berisi daftar URL kandidat')
    parser.add_argument('--output', '-o', type=Path, default=Config.OUTPUT_FILE,
                        help='File JSON output hasil seleksi (detail)')
    parser.add_argument('--youtube-urls-output', type=Path, default=Config.YOUTUBE_URLS_OUTPUT,
                        help='File JSON output format youtube_urls.json (siap pakai detector)')
    parser.add_argument('--target', '-n', type=int, default=Config.TARGET_PER_CATEGORY,
                        help='Target jumlah video per kategori (default: 10)')
    parser.add_argument('--max-comments', type=int, default=Config.MAX_COMMENTS_PER_VIDEO,
                        help='Max komentar per video (default: 200)')
    parser.add_argument('--judi-threshold', type=float, default=Config.JUDI_VIDEO_THRESHOLD,
                        help='%% komentar judi agar video = DOMINAN_JUDI (default: 20.0)')
    parser.add_argument('--bukan-threshold', type=float, default=Config.BUKAN_VIDEO_THRESHOLD,
                        help='%% komentar bukan judi agar video = TIDAK_JUDI (default: 40.0)')
    parser.add_argument('--delay', type=float, default=Config.SCRAPE_DELAY_SECONDS,
                        help='Jeda antar video scraping dalam detik (default: 2)')

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config()
    cfg.MAX_COMMENTS_PER_VIDEO = args.max_comments
    cfg.TARGET_PER_CATEGORY    = args.target
    cfg.JUDI_VIDEO_THRESHOLD   = args.judi_threshold
    cfg.BUKAN_VIDEO_THRESHOLD  = args.bukan_threshold
    cfg.SCRAPE_DELAY_SECONDS   = args.delay

    logger = setup_logging(cfg.LOG_FILE, cfg.APP_LOG_FILE)

    logger.info("=" * 70)
    logger.info("VIDEO SELECTOR - Seleksi URL untuk Penelitian")
    logger.info(f"  Input              : {args.input}")
    logger.info(f"  Output (detail)    : {args.output}")
    logger.info(f"  Output (yt format) : {args.youtube_urls_output}")
    logger.info("=" * 70)

    if not args.input.exists():
        logger.error(f"File input tidak ditemukan: {args.input}")
        logger.error('Buat file JSON berisi list URL, contoh:')
        logger.error('  ["https://www.youtube.com/watch?v=ABC", ...]')
        return

    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        if isinstance(raw, list):
            urls = raw
        elif isinstance(raw, dict) and 'urls' in raw:
            urls = raw['urls']
        else:
            logger.error('Format file input tidak valid. Harus list URL atau {"urls": [...]}.')
            return

        if not urls:
            logger.error("File input kosong, tidak ada URL untuk diproses.")
            return

        logger.info(f"Loaded {len(urls)} URL kandidat dari {args.input}")

    except Exception as exc:
        logger.error(f"Gagal membaca file input: {exc}")
        return

    selector = VideoSelector(cfg, logger)
    output   = selector.run(urls)

    OutputWriter.save(output, args.output, logger)

    OutputWriter.save_as_youtube_urls(output, args.youtube_urls_output, logger)

    OutputWriter.print_summary(output, logger)


if __name__ == "__main__":
    main()
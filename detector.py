import os
import re
import json
import time
import logging
import statistics
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

import requests
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class Config:
    YOUTUBE_DIR = Path("youtube")
    RESULTS_DIR = Path("results")
    LOGS_DIR = Path("logs")

    URLS_FILE = Path("youtube_urls.json")
    VIDEO_RESULTS_DIR = RESULTS_DIR / "per_video"
    AGGREGATE_RESULTS = RESULTS_DIR / "aggregate_results.json"
    EXCEL_OUTPUT = RESULTS_DIR / "analysis_results.xlsx"
    LOG_FILE = LOGS_DIR / "detector.log"
    RESULT_LOG_FILE = LOGS_DIR / "result_detector.log"  

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    HIGH_THRESHOLD = float(os.getenv("HIGH_THRESHOLD", "0.75"))
    LOW_THRESHOLD  = float(os.getenv("LOW_THRESHOLD",  "0.50"))

    BATCH_SIZE  = int(os.getenv("BATCH_SIZE",  "5"))
    BATCH_DELAY = int(os.getenv("BATCH_DELAY", "25"))


PRICING = {
    "gpt-4o-mini": {
        "input_per_1m":  0.150,
        "output_per_1m": 0.600,
    },
    "gemini-2.5-flash": {
        "input_per_1m":  0.300,
        "output_per_1m": 2.500,
    },
}

def calculate_cost(model_key: str, input_tokens: int, output_tokens: int) -> float:
    pricing = PRICING.get(model_key, {"input_per_1m": 0.0, "output_per_1m": 0.0})
    cost = (input_tokens / 1_000_000) * pricing["input_per_1m"] + \
           (output_tokens / 1_000_000) * pricing["output_per_1m"]
    return round(cost, 8)


# DATA MODELS
@dataclass
class Comment:
    comment_id: str
    text: str
    timestamp: str
    original_text: Optional[str] = None
    normalized_text: Optional[str] = None
    tokens: Optional[List[str]] = None
    clean_tokens: Optional[List[str]] = None

@dataclass
class RuleBasedResult:
    classification: str
    score: float
    detection_method: str

@dataclass
class LLMResult:
    model_name: str
    classification: str
    confidence: float
    latency_ms: float
    success: bool
    reasoning: Optional[str] = None
    error: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

@dataclass
class VideoResult:
    video_id: str
    video_url: str
    total_comments: int

    rule_based_judi: int
    rule_based_bukan_judi: int
    rule_based_ambigu: int

    gemini_judi: int
    gemini_bukan_judi: int
    gpt_judi: int
    gpt_bukan_judi: int

    gemini_agree_count: int
    gemini_disagree_count: int
    gemini_agreement_rate: float     # Gemini agree% vs GPT
    gpt_agree_count: int
    gpt_disagree_count: int
    gpt_agreement_rate: float        # GPT agree% vs Gemini

    gemini_avg_confidence: float
    gemini_avg_latency: float
    gpt_avg_confidence: float
    gpt_avg_latency: float

    processing_time_seconds: float
    timestamp: str

    gemini_total_input_tokens: int = 0
    gemini_total_output_tokens: int = 0
    gemini_total_cost_usd: float = 0.0
    gpt_total_input_tokens: int = 0
    gpt_total_output_tokens: int = 0
    gpt_total_cost_usd: float = 0.0

    sample_category: str = "UNKNOWN"


# UTILITIES & LOGGING
def setup_logging():
    Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(Config.LOG_FILE, encoding='utf-8'),
            logging.FileHandler(Config.LOGS_DIR / "app.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def get_result_logger() -> logging.Logger:
    """[BARU] Logger khusus mencatat input/output prompt LLM ke result_detector.log."""
    Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    rl = logging.getLogger("result_detector")
    if not rl.handlers:
        rl.setLevel(logging.DEBUG)
        fh = logging.FileHandler(Config.RESULT_LOG_FILE, encoding='utf-8')
        fh.setFormatter(logging.Formatter(
            '[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
        ))
        rl.addHandler(fh)
        rl.propagate = False
    return rl


def ensure_directories():
    Config.YOUTUBE_DIR.mkdir(parents=True, exist_ok=True)
    Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    Config.VIDEO_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

def save_json(filepath: Path, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

def load_json(filepath: Path) -> Optional[dict]:
    if not filepath.exists():
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_video_id(url: str) -> str:
    if 'youtu.be/' in url:
        return url.split('youtu.be/')[-1].split('?')[0]
    if 'watch?v=' in url:
        return url.split('watch?v=')[-1].split('&')[0]
    return url


# TEXT PREPROCESSING
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

    @staticmethod
    def normalize_leet_speak(text: str) -> str:
        return ''.join(TextPreprocessor.LEET_MAP.get(c, c) for c in text)

    @staticmethod
    def normalize_text(text: str) -> str:
        normalized = text.lower()
        normalized = TextPreprocessor.normalize_leet_speak(normalized)
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return [token for token in text.split() if token]

    @staticmethod
    def remove_stopwords(tokens: List[str]) -> List[str]:
        return [token for token in tokens if token not in TextPreprocessor.STOPWORDS]

    @classmethod
    def preprocess(cls, comment: Comment) -> Comment:
        comment.original_text = comment.text
        comment.normalized_text = cls.normalize_text(comment.text)
        comment.tokens = cls.tokenize(comment.normalized_text)
        comment.clean_tokens = cls.remove_stopwords(comment.tokens)
        return comment


# RULE-BASED CLASSIFIER
class RuleBasedClassifier:
    SPAM_PATTERNS = {
        'gambling': r'(slot|gacor|rtp|bonus|bet|maxwin|deposit|judi|casino|situs|pragmatic|pg|soft|togel|bandar|jackpot)',
        'urls': r'(https?://[^\s]+\.(bet|casino|slot|gaming)|t\.me/|wa\.me/|bit\.ly)',
        'promo': r'(daftar|register|klik|link|wede|depo|wd|withdraw|bonus\s*\d+|promo|claim)',
        'contact': r'(\d{10,}|wa\s*\d+|kontak|hubungi|cs\s*online|admin|dm|chat)',
        'emoji_spam': r'(🎰|🎲|💰|💸|🤑|💎|🔥){2,}'
    }
    SPAM_KEYWORDS = [
        'slot', 'gacor', 'judi', 'casino', 'togel', 'bandar',
        'maxwin', 'deposit', 'bonus', 'pragmatic', 'daftar',
        'jackpot', 'scatter', 'zeus', 'olympus', 'gates'
    ]

    @classmethod
    def detect_with_regex(cls, comment: Comment) -> Tuple[bool, Optional[str]]:
        text = comment.normalized_text or comment.text
        for pattern_name, pattern in cls.SPAM_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                return True, pattern_name
        return False, None

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return RuleBasedClassifier.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    @classmethod
    def similarity_score(cls, s1: str, s2: str) -> float:
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        return 1 - (cls.levenshtein_distance(s1, s2) / max_len)

    @classmethod
    def detect_with_fuzzy_matching(cls, comment: Comment) -> Tuple[float, Optional[str]]:
        tokens = comment.clean_tokens or comment.tokens or []
        max_score = 0.0
        matched_keyword = None
        for token in tokens:
            for keyword in cls.SPAM_KEYWORDS:
                score = cls.similarity_score(token, keyword)
                if score > max_score:
                    max_score = score
                    matched_keyword = keyword
        return max_score, matched_keyword

    @classmethod
    def classify(cls, comment: Comment) -> RuleBasedResult:
        is_spam_regex, pattern_name = cls.detect_with_regex(comment)
        if is_spam_regex:
            return RuleBasedResult(classification="JUDI_ONLINE", score=1.0, detection_method="Regex")
        fuzzy_score, matched_keyword = cls.detect_with_fuzzy_matching(comment)
        if fuzzy_score >= Config.HIGH_THRESHOLD:
            classification = "JUDI_ONLINE"
        elif fuzzy_score < Config.LOW_THRESHOLD:
            classification = "BUKAN_JUDI_ONLINE"
        else:
            classification = "AMBIGU"
        return RuleBasedResult(classification=classification, score=fuzzy_score, detection_method="Fuzzy Matching")


# LLM CLASSIFIERS
class GeminiClassifier:
    def __init__(self):
        if not Config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found!")
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
        self.result_logger = get_result_logger()  

    def classify(self, comment: Comment) -> LLMResult:
        start_time = time.time()

        prompt = f"""Anda adalah sistem deteksi spam judi online untuk platform YouTube.

Tugas: Analisis komentar berikut dan tentukan apakah ini promosi judi online atau bukan.

Komentar: "{comment.text}"

INSTRUKSI:
1. Jawab dengan format JSON:
{{
  "classification": "JUDI_ONLINE" atau "BUKAN_JUDI_ONLINE",
  "confidence": <nilai 0.0 - 1.0>,
  "reasoning": "<penjelasan singkat>"
}}

2. Indikator JUDI_ONLINE:
   - Kata kunci: slot, gacor, maxwin, deposit, casino, togel
   - URL situs judi/betting
   - Ajakan daftar/registrasi
   - Nomor kontak untuk judi

Jawab HANYA dengan JSON (tanpa markdown):"""

        # Log prompt yang dikirim ke Gemini
        self.result_logger.info(
            f"[GEMINI INPUT] comment_id={comment.comment_id}\n"
            f"  text   : {comment.text!r}\n"
            f"  prompt : {prompt.strip()}"
        )

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()

            # Log raw response dari Gemini
            self.result_logger.info(
                f"[GEMINI OUTPUT] comment_id={comment.comment_id}\n"
                f"  raw_response: {response_text!r}"
            )

            clean_response = re.sub(r'```json|```', '', response_text).strip()
            parsed = json.loads(clean_response)

            latency_ms = (time.time() - start_time) * 1000

            input_tokens = 0
            output_tokens = 0
            try:
                usage = response.usage_metadata
                input_tokens  = getattr(usage, 'prompt_token_count', 0) or 0
                output_tokens = getattr(usage, 'candidates_token_count', 0) or 0
            except Exception:
                pass

            cost_usd = calculate_cost(Config.GEMINI_MODEL, input_tokens, output_tokens)

            # Log parsed result
            self.result_logger.info(
                f"[GEMINI PARSED] comment_id={comment.comment_id} "
                f"classification={parsed.get('classification')} "
                f"confidence={parsed.get('confidence')} "
                f"reasoning={parsed.get('reasoning')!r} "
                f"tokens={input_tokens}/{output_tokens} cost=${cost_usd:.8f}"
            )

            return LLMResult(
                model_name="Gemini",
                classification=parsed['classification'],
                confidence=float(parsed['confidence']),
                latency_ms=latency_ms,
                success=True,
                reasoning=parsed.get('reasoning'),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logging.error(f"Gemini error: {str(e)}")
            self.result_logger.error(
                f"[GEMINI ERROR] comment_id={comment.comment_id} error={str(e)}"
            )
            return LLMResult(
                model_name="Gemini",
                classification="ERROR",
                confidence=0.0,
                latency_ms=latency_ms,
                success=False,
                error=str(e)
            )


class GPTClassifier:
    def __init__(self):
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found!")
        self.api_key = Config.OPENAI_API_KEY
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.result_logger = get_result_logger()  

    def classify(self, comment: Comment) -> LLMResult:
        start_time = time.time()

        prompt = f"""Anda adalah sistem deteksi spam judi online untuk platform YouTube.

Tugas: Analisis komentar berikut dan tentukan apakah ini promosi judi online atau bukan.

Komentar: "{comment.text}"

INSTRUKSI:
1. Jawab dengan format JSON:
{{
  "classification": "JUDI_ONLINE" atau "BUKAN_JUDI_ONLINE",
  "confidence": <nilai 0.0 - 1.0>,
  "reasoning": "<penjelasan singkat>"
}}

2. Indikator JUDI_ONLINE:
   - Kata kunci: slot, gacor, maxwin, deposit, casino, togel
   - URL situs judi/betting
   - Ajakan daftar/registrasi
   - Nomor kontak untuk judi

Jawab HANYA dengan JSON (tanpa markdown):"""

        # Log prompt yang dikirim ke GPT
        self.result_logger.info(
            f"[GPT INPUT] comment_id={comment.comment_id}\n"
            f"  text  : {comment.text!r}\n"
            f"  prompt: {prompt.strip()}"
        )

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": Config.OPENAI_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 200
            }

            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()

            response_text = result['choices'][0]['message']['content'].strip()

            # Log raw response dari GPT
            self.result_logger.info(
                f"[GPT OUTPUT] comment_id={comment.comment_id}\n"
                f"  raw_response: {response_text!r}"
            )

            clean_response = re.sub(r'```json|```', '', response_text).strip()
            parsed = json.loads(clean_response)

            latency_ms = (time.time() - start_time) * 1000

            usage = result.get('usage', {})
            input_tokens  = usage.get('prompt_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0)
            cost_usd = calculate_cost(Config.OPENAI_MODEL, input_tokens, output_tokens)

            # Log parsed result
            self.result_logger.info(
                f"[GPT PARSED] comment_id={comment.comment_id} "
                f"classification={parsed.get('classification')} "
                f"confidence={parsed.get('confidence')} "
                f"reasoning={parsed.get('reasoning')!r} "
                f"tokens={input_tokens}/{output_tokens} cost=${cost_usd:.8f}"
            )

            return LLMResult(
                model_name="GPT",
                classification=parsed['classification'],
                confidence=float(parsed['confidence']),
                latency_ms=latency_ms,
                success=True,
                reasoning=parsed.get('reasoning'),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logging.error(f"GPT error: {str(e)}")
            self.result_logger.error(
                f"[GPT ERROR] comment_id={comment.comment_id} error={str(e)}"
            )
            return LLMResult(
                model_name="GPT",
                classification="ERROR",
                confidence=0.0,
                latency_ms=latency_ms,
                success=False,
                error=str(e)
            )


# PER-VIDEO PROCESSOR
class VideoProcessor:
    def __init__(self):
        self.gemini = GeminiClassifier()
        self.gpt    = GPTClassifier()

    def process_video(self, video_url: str, comments_data: List[Dict], sample_category: str = "UNKNOWN") -> VideoResult:
        start_time = time.time()
        video_id   = extract_video_id(video_url)

        logging.info(f"\n{'='*70}")
        logging.info(f"PROCESSING VIDEO: {video_id}")
        logging.info(f"{'='*70}")

        comments = [Comment(**c) for c in comments_data]
        logging.info(f"Total comments: {len(comments)}")

        logging.info("Preprocessing...")
        for comment in comments:
            TextPreprocessor.preprocess(comment)

        logging.info("Rule-based classification...")
        rule_results = []
        for comment in comments:
            result = RuleBasedClassifier.classify(comment)
            rule_results.append((comment, result))

        judi_online = [r for c, r in rule_results if r.classification == "JUDI_ONLINE"]
        bukan_judi  = [r for c, r in rule_results if r.classification == "BUKAN_JUDI_ONLINE"]
        ambigu      = [(c, r) for c, r in rule_results if r.classification == "AMBIGU"]

        logging.info(f"   +-- Judi Online:  {len(judi_online)} comments")
        logging.info(f"   +-- Bukan Judi:   {len(bukan_judi)} comments")
        logging.info(f"   \\-- Ambigu:       {len(ambigu)} comments")

        gemini_results: List[LLMResult] = []
        gpt_results:    List[LLMResult] = []

        if ambigu:
            logging.info(f"\nLLM Classification ({len(ambigu)} ambiguous comments)...")
            rl = get_result_logger()
            rl.info(f"\n{'='*70}")
            rl.info(f"VIDEO: {video_id}  |  Kategori: {sample_category}  |  Ambigu: {len(ambigu)}")
            rl.info(f"{'='*70}")

            for idx, (comment, _) in enumerate(ambigu, 1):
                if idx % 5 == 0:
                    logging.info(f"   Progress: {idx}/{len(ambigu)}")

                gemini_result = self.gemini.classify(comment)
                gpt_result    = self.gpt.classify(comment)

                gemini_results.append(gemini_result)
                gpt_results.append(gpt_result)

                if idx % Config.BATCH_SIZE == 0 and idx < len(ambigu):
                    time.sleep(Config.BATCH_DELAY)

            logging.info("   LLM classification completed")

        processing_time = time.time() - start_time

        gemini_judi  = sum(1 for r in gemini_results if r.classification == "JUDI_ONLINE"    and r.success)
        gemini_bukan = sum(1 for r in gemini_results if r.classification == "BUKAN_JUDI_ONLINE" and r.success)
        gpt_judi     = sum(1 for r in gpt_results    if r.classification == "JUDI_ONLINE"    and r.success)
        gpt_bukan    = sum(1 for r in gpt_results    if r.classification == "BUKAN_JUDI_ONLINE" and r.success)

        # Agreement dihitung per model
        total_compared = sum(
            1 for g, p in zip(gemini_results, gpt_results) if g.success and p.success
        )
        agree_count = sum(
            1 for g, p in zip(gemini_results, gpt_results)
            if g.classification == p.classification and g.success and p.success
        )
        disagree_count = total_compared - agree_count

        # Agreement rate masing-masing model = berapa kali model tersebut
        # menghasilkan label sama dengan model lawannya / total komentar ambigu yang diproses
        gemini_agree_rate = (agree_count / total_compared * 100) if total_compared else 0.0
        gpt_agree_rate    = (agree_count / total_compared * 100) if total_compared else 0.0

        successful_gemini = [r for r in gemini_results if r.success]
        successful_gpt    = [r for r in gpt_results    if r.success]

        gemini_avg_conf = statistics.mean([r.confidence  for r in successful_gemini]) if successful_gemini else 0.0
        gemini_avg_lat  = statistics.mean([r.latency_ms  for r in successful_gemini]) if successful_gemini else 0.0
        gpt_avg_conf    = statistics.mean([r.confidence  for r in successful_gpt])    if successful_gpt    else 0.0
        gpt_avg_lat     = statistics.mean([r.latency_ms  for r in successful_gpt])    if successful_gpt    else 0.0

        gemini_input_tokens  = sum(r.input_tokens  for r in gemini_results)
        gemini_output_tokens = sum(r.output_tokens for r in gemini_results)
        gemini_total_cost    = sum(r.cost_usd      for r in gemini_results)
        gpt_input_tokens     = sum(r.input_tokens  for r in gpt_results)
        gpt_output_tokens    = sum(r.output_tokens for r in gpt_results)
        gpt_total_cost       = sum(r.cost_usd      for r in gpt_results)

        result = VideoResult(
            video_id=video_id,
            video_url=video_url,
            total_comments=len(comments),
            rule_based_judi=len(judi_online),
            rule_based_bukan_judi=len(bukan_judi),
            rule_based_ambigu=len(ambigu),
            gemini_judi=gemini_judi,
            gemini_bukan_judi=gemini_bukan,
            gpt_judi=gpt_judi,
            gpt_bukan_judi=gpt_bukan,
            gemini_agree_count=agree_count,
            gemini_disagree_count=disagree_count,
            gemini_agreement_rate=round(gemini_agree_rate, 2),
            gpt_agree_count=agree_count,
            gpt_disagree_count=disagree_count,
            gpt_agreement_rate=round(gpt_agree_rate, 2),
            gemini_avg_confidence=round(gemini_avg_conf, 4),
            gemini_avg_latency=round(gemini_avg_lat, 2),
            gpt_avg_confidence=round(gpt_avg_conf, 4),
            gpt_avg_latency=round(gpt_avg_lat, 2),
            processing_time_seconds=round(processing_time, 2),
            timestamp=datetime.now().isoformat(),
            gemini_total_input_tokens=gemini_input_tokens,
            gemini_total_output_tokens=gemini_output_tokens,
            gemini_total_cost_usd=round(gemini_total_cost, 8),
            gpt_total_input_tokens=gpt_input_tokens,
            gpt_total_output_tokens=gpt_output_tokens,
            gpt_total_cost_usd=round(gpt_total_cost, 8),
            sample_category=sample_category
        )

        video_result_file = Config.VIDEO_RESULTS_DIR / f"{video_id}.json"
        save_json(video_result_file, asdict(result))

        logging.info(f"\nVideo Summary:")
        logging.info(f"   Gemini Agreement Rate : {result.gemini_agreement_rate}%  "
                     f"(Sepakat: {agree_count} | Tidak: {disagree_count})")
        logging.info(f"   GPT    Agreement Rate : {result.gpt_agreement_rate}%  "
                     f"(Sepakat: {agree_count} | Tidak: {disagree_count})")
        logging.info(f"   Gemini Confidence     : {result.gemini_avg_confidence}")
        logging.info(f"   GPT    Confidence     : {result.gpt_avg_confidence}")
        logging.info(f"   Processing Time       : {result.processing_time_seconds}s")
        logging.info(f"   Cost - Gemini: ${result.gemini_total_cost_usd:.6f} USD  |  "
                     f"Tokens: {gemini_input_tokens} in / {gemini_output_tokens} out")
        logging.info(f"   Cost - GPT   : ${result.gpt_total_cost_usd:.6f} USD  |  "
                     f"Tokens: {gpt_input_tokens} in / {gpt_output_tokens} out")
        logging.info(f"   Saved to: {video_result_file}")

        return result


# MAIN PIPELINE
def load_video_comments(video_url: str) -> Optional[List[Dict]]:
    video_id   = extract_video_id(video_url)
    video_file = Config.YOUTUBE_DIR / f"{video_id}.json"
    if not video_file.exists():
        logging.warning(f"File not found: {video_file}")
        return None
    video_data = load_json(video_file)
    return video_data['comments'] if video_data else None


def load_urls() -> List[Dict]:
    if not Config.URLS_FILE.exists():
        raise FileNotFoundError(f"File {Config.URLS_FILE} not found!")

    urls_data = load_json(Config.URLS_FILE)

    if isinstance(urls_data, dict) and 'videos' in urls_data:
        result = []
        for item in urls_data['videos']:
            if isinstance(item, dict) and 'url' in item:
                result.append({
                    'url': item['url'],
                    'sample_category': item.get('sample_category', 'UNKNOWN')
                })
            elif isinstance(item, str):
                result.append({'url': item, 'sample_category': 'UNKNOWN'})
        return result

    if isinstance(urls_data, list):
        raw = urls_data
    elif isinstance(urls_data, dict) and 'urls' in urls_data:
        raw = urls_data['urls']
    else:
        raise ValueError("Invalid URLs file format!")

    return [{'url': u, 'sample_category': 'UNKNOWN'} for u in raw if isinstance(u, str)]


def aggregate_results(video_results: List[VideoResult]) -> Dict:
    logging.info(f"\n{'='*70}")
    logging.info("AGGREGATING RESULTS")
    logging.info(f"{'='*70}")

    total_videos   = len(video_results)
    total_comments = sum(v.total_comments for v in video_results)
    total_judi     = sum(v.rule_based_judi for v in video_results)
    total_bukan    = sum(v.rule_based_bukan_judi for v in video_results)
    total_ambigu   = sum(v.rule_based_ambigu for v in video_results)

    avg_gemini_agree = statistics.mean([v.gemini_agreement_rate for v in video_results])
    avg_gpt_agree    = statistics.mean([v.gpt_agreement_rate    for v in video_results])
    avg_gemini_conf  = statistics.mean([v.gemini_avg_confidence for v in video_results if v.gemini_avg_confidence > 0]) if any(v.gemini_avg_confidence > 0 for v in video_results) else 0.0
    avg_gpt_conf     = statistics.mean([v.gpt_avg_confidence    for v in video_results if v.gpt_avg_confidence    > 0]) if any(v.gpt_avg_confidence    > 0 for v in video_results) else 0.0
    avg_gemini_lat   = statistics.mean([v.gemini_avg_latency    for v in video_results if v.gemini_avg_latency    > 0]) if any(v.gemini_avg_latency    > 0 for v in video_results) else 0.0
    avg_gpt_lat      = statistics.mean([v.gpt_avg_latency       for v in video_results if v.gpt_avg_latency       > 0]) if any(v.gpt_avg_latency       > 0 for v in video_results) else 0.0

    aggregate = {
        'summary': {
            'total_videos':          total_videos,
            'total_comments':        total_comments,
            'total_judi_online':     total_judi,
            'total_bukan_judi':      total_bukan,
            'total_ambigu':          total_ambigu,
            'gemini_total_cost_usd': round(sum(v.gemini_total_cost_usd for v in video_results), 8),
            'gpt_total_cost_usd':    round(sum(v.gpt_total_cost_usd    for v in video_results), 8),
        },
        'averages': {
            'gemini_agreement_rate': round(avg_gemini_agree, 2),
            'gpt_agreement_rate':    round(avg_gpt_agree,    2),
            'gemini_confidence':     round(avg_gemini_conf,  4),
            'gpt_confidence':        round(avg_gpt_conf,     4),
            'gemini_latency_ms':     round(avg_gemini_lat,   2),
            'gpt_latency_ms':        round(avg_gpt_lat,      2),
        },
        'per_video_results': [asdict(v) for v in video_results],
        'generated_at': datetime.now().isoformat()
    }

    save_json(Config.AGGREGATE_RESULTS, aggregate)

    logging.info(f"\nAggregate Summary:")
    logging.info(f"   Total Videos          : {total_videos}")
    logging.info(f"   Total Comments        : {total_comments}")
    logging.info(f"   Gemini Agreement Rate : {avg_gemini_agree:.2f}%")
    logging.info(f"   GPT    Agreement Rate : {avg_gpt_agree:.2f}%")
    logging.info(f"   Gemini Avg Confidence : {avg_gemini_conf:.4f}")
    logging.info(f"   GPT    Avg Confidence : {avg_gpt_conf:.4f}")
    logging.info(f"   Gemini Avg Latency    : {avg_gemini_lat:.2f}ms")
    logging.info(f"   GPT    Avg Latency    : {avg_gpt_lat:.2f}ms")
    logging.info(f"   Cost Gemini           : ${aggregate['summary']['gemini_total_cost_usd']:.6f} USD")
    logging.info(f"   Cost GPT              : ${aggregate['summary']['gpt_total_cost_usd']:.6f} USD")
    logging.info(f"   Saved to              : {Config.AGGREGATE_RESULTS}")

    return aggregate


def export_to_excel(aggregate_data: Dict):
    try:
        import pandas as pd
    except ImportError:
        logging.warning("pandas not installed. Skipping Excel export.")
        return

    logging.info(f"\nExporting to Excel (5 separate files)...")

    videos = aggregate_data.get('per_video_results', [])
    if not videos:
        logging.warning("No video data to export.")
        return

    CAT_FILL = {
        'JUDI':       'FFCCCC',
        'TIDAK_JUDI': 'CCFFCC',
        'AMBIGU':     'FFF2CC',
        'RANDOM':     'FFD580',
        'UNKNOWN':    'F2F2F2',
    }
    CAT_LABEL = {
        'JUDI':       'Berindikasi Judi',
        'TIDAK_JUDI': 'Tidak Judi',
        'AMBIGU':     'Ambigu',
        'RANDOM':     'Data Random',
        'UNKNOWN':    'Tidak Diketahui',
    }

    def apply_colors(writer, df, sheet_name):
        from openpyxl.styles import PatternFill, Font, Alignment
        from openpyxl.utils import get_column_letter

        code_col = df['_cat_code'].tolist() if '_cat_code' in df.columns else []
        display  = df.drop(columns=['_cat_code'], errors='ignore')
        display.to_excel(writer, sheet_name=sheet_name, index=False)
        ws = writer.sheets[sheet_name]

        for cell in ws[1]:
            cell.font      = Font(bold=True)
            cell.alignment = Alignment(horizontal='center', vertical='center')

        for ci, col in enumerate(display.columns, 1):
            vals    = display[col].astype(str).tolist() + [str(col)]
            max_len = max(len(v) for v in vals) + 4
            ws.column_dimensions[get_column_letter(ci)].width = min(max_len, 45)

        for ri, cat in enumerate(code_col, start=2):
            fill = PatternFill(
                start_color=CAT_FILL.get(cat, 'F2F2F2'),
                end_color=CAT_FILL.get(cat, 'F2F2F2'),
                fill_type='solid'
            )
            for ci in range(1, len(display.columns) + 1):
                ws.cell(row=ri, column=ci).fill = fill

        # Legend di bawah data
        last_row = len(code_col) + 2
        ws.cell(row=last_row + 1, column=1, value="").font = Font(bold=True)
        ws.cell(row=last_row + 2, column=1, value="LEGENDA WARNA").font = Font(bold=True)
        legend_rows = [
            ('JUDI',       'Berindikasi Judi',  'Merah muda  — video berindikasi komentar judi dominan'),
            ('TIDAK_JUDI', 'Tidak Judi',        'Hijau muda  — video dengan komentar dominan bukan judi'),
            ('AMBIGU',     'Ambigu',            'Kuning muda — video dengan komentar campuran / tidak jelas'),
            ('RANDOM',     'Data Random',       'Oranye muda — video sampel acak untuk uji efektivitas'),
        ]
        for i, (code, lbl, desc) in enumerate(legend_rows, start=last_row + 3):
            ws.cell(row=i, column=1, value=code)
            ws.cell(row=i, column=2, value=lbl)
            ws.cell(row=i, column=3, value=desc)
            fill = PatternFill(
                start_color=CAT_FILL.get(code, 'F2F2F2'),
                end_color=CAT_FILL.get(code, 'F2F2F2'),
                fill_type='solid'
            )
            for ci in range(1, 4):
                ws.cell(row=i, column=ci).fill = fill

    def code(v):  return v.get('sample_category', 'UNKNOWN')
    def label(v): return CAT_LABEL.get(code(v), code(v))

    df_rekap = pd.DataFrame(videos)
    if 'sample_category' not in df_rekap.columns:
        df_rekap['sample_category'] = 'UNKNOWN'
    df_rekap['Kategori Sampel'] = df_rekap['sample_category'].map(lambda c: CAT_LABEL.get(c, c))
    df_rekap['_cat_code']       = df_rekap['sample_category']
    cols = ['Kategori Sampel', '_cat_code'] + [
        c for c in df_rekap.columns if c not in ('Kategori Sampel', '_cat_code', 'sample_category')
    ]
    df_rekap = df_rekap[cols]

    # Agreement per model
    df_agreement = pd.DataFrame([{
        'Kategori Sampel'        : label(v),
        'Video ID'               : v.get('video_id'),
        'Komentar Ditarik'       : v.get('total_comments'),
        'Anomali (Masuk AI)'     : v.get('rule_based_ambigu'),
        'Gemini (Judi)'          : v.get('gemini_judi'),
        'Gemini Sepakat'         : v.get('gemini_agree_count'),
        'Gemini Tidak Sepakat'   : v.get('gemini_disagree_count'),
        'Gemini Agreement Rate %': v.get('gemini_agreement_rate'),
        'GPT (Judi)'             : v.get('gpt_judi'),
        'GPT Sepakat'            : v.get('gpt_agree_count'),
        'GPT Tidak Sepakat'      : v.get('gpt_disagree_count'),
        'GPT Agreement Rate %'   : v.get('gpt_agreement_rate'),
        '_cat_code'              : code(v),
    } for v in videos])

    df_confidence = pd.DataFrame([{
        'Kategori Sampel'      : label(v),
        'Video ID'             : v.get('video_id'),
        'Anomali (Masuk AI)'   : v.get('rule_based_ambigu'),
        'Gemini Avg Confidence': v.get('gemini_avg_confidence'),
        'GPT Avg Confidence'   : v.get('gpt_avg_confidence'),
        '_cat_code'            : code(v),
    } for v in videos])

    df_latency = pd.DataFrame([{
        'Kategori Sampel'          : label(v),
        'Video ID'                 : v.get('video_id'),
        'Anomali (Masuk AI)'       : v.get('rule_based_ambigu'),
        'Gemini Avg Latency (ms)'  : v.get('gemini_avg_latency'),
        'GPT Avg Latency (ms)'     : v.get('gpt_avg_latency'),
        'Total Processing Time (s)': v.get('processing_time_seconds'),
        '_cat_code'                : code(v),
    } for v in videos])

    # Total Cost
    df_cost = pd.DataFrame([{
        'Kategori Sampel'     : label(v),
        'Video ID'            : v.get('video_id'),
        'Anomali (Masuk AI)'  : v.get('rule_based_ambigu'),
        'Gemini Input Tokens' : v.get('gemini_total_input_tokens', 0),
        'Gemini Output Tokens': v.get('gemini_total_output_tokens', 0),
        'Gemini Cost (USD)'   : v.get('gemini_total_cost_usd', 0.0),
        'GPT Input Tokens'    : v.get('gpt_total_input_tokens', 0),
        'GPT Output Tokens'   : v.get('gpt_total_output_tokens', 0),
        'GPT Cost (USD)'      : v.get('gpt_total_cost_usd', 0.0),
        '_cat_code'           : code(v),
    } for v in videos])

    file_map = [
        (df_rekap,      "1_Rekap_Keseluruhan.xlsx"),
        (df_agreement,  "2_Report_Agreement.xlsx"),
        (df_confidence, "3_Report_Confidence.xlsx"),
        (df_latency,    "4_Report_Latency.xlsx"),
        (df_cost,       "5_Report_Cost.xlsx"),
    ]

    for df, fname in file_map:
        path = Config.RESULTS_DIR / fname
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            apply_colors(writer, df, 'Data')
        logging.info(f"   Saved: {fname}")

    logging.info(f"   5 Excel files saved to: {Config.RESULTS_DIR}")
    logging.info("   Merah muda=JUDI | Hijau muda=TIDAK_JUDI | Kuning muda=AMBIGU | Oranye=RANDOM")


def main():
    setup_logging()
    ensure_directories()

    logging.info("=" * 70)
    logging.info("SPAM DETECTOR V2 - PER-VIDEO PROCESSING")
    logging.info("   Platform Gemini vs OpenAI GPT")
    logging.info("=" * 70)

    if not Config.GEMINI_API_KEY:
        logging.error("GEMINI_API_KEY not found in .env!")
        return

    if not Config.OPENAI_API_KEY:
        logging.error("OPENAI_API_KEY not found in .env!")
        return

    try:
        url_entries = load_urls()
        logging.info(f"\nFound {len(url_entries)} videos to process")
    except Exception as e:
        logging.error(f"Error loading URLs: {e}")
        return

    processor      = VideoProcessor()
    video_results  = []

    for idx, entry in enumerate(url_entries, 1):
        url             = entry['url']
        sample_category = entry['sample_category']
        video_id        = extract_video_id(url)

        logging.info(f"\n{'#'*70}")
        logging.info(f"# VIDEO {idx}/{len(url_entries)}: {video_id}  [Kategori: {sample_category}]")
        logging.info(f"{'#'*70}")

        try:
            comments_data = load_video_comments(url)
            if not comments_data:
                logging.warning(f"No comments found for video {video_id}")
                continue

            video_result = processor.process_video(url, comments_data, sample_category)
            video_results.append(video_result)

        except Exception as e:
            logging.error(f"Error processing video {url}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if video_results:
        aggregate_data = aggregate_results(video_results)
        export_to_excel(aggregate_data)

        logging.info(f"\n{'='*70}")
        logging.info("PROCESSING COMPLETED!")
        logging.info(f"{'='*70}")
        logging.info(f"Results saved to: {Config.RESULTS_DIR}")
        logging.info(f"   - Per-video : {Config.VIDEO_RESULTS_DIR}")
        logging.info(f"   - Aggregate : {Config.AGGREGATE_RESULTS}")
        logging.info(f"   - LLM Logs  : {Config.RESULT_LOG_FILE}")
    else:
        logging.error("No videos processed successfully!")


if __name__ == "__main__":
    main()
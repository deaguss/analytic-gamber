import os
import re
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import statistics
import sys
import io
import requests
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# KONFIGURASI
try:
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class Config:
    """Konfigurasi aplikasi"""
    DATA_DIR = Path("data")
    LOGS_DIR = Path("logs")
    
    # File paths
    COMMENTS_RAW = DATA_DIR / "comments_raw.json"
    COMMENTS_NORMALIZED = DATA_DIR / "comments_normalized.json"
    RULE_BASED_RESULTS = DATA_DIR / "rule_based_results.json"
    AMBIGUOUS_DATA = DATA_DIR / "ambiguous_data.json"
    GEMINI_RESULTS = DATA_DIR / "gemini_results.json"
    GPT2_RESULTS = DATA_DIR / "gpt2_results.json"
    AGREEMENT_ANALYSIS = DATA_DIR / "agreement_analysis.json"
    DISTRIBUTION_ANALYSIS = DATA_DIR / "distribution_analysis.json"
    COMPARATIVE_ANALYSIS = DATA_DIR / "comparative_analysis.json"
    FINAL_REPORT = DATA_DIR / "final_report.json"
    CSV_EXPORT = DATA_DIR / "comparison_results.csv"
    APP_LOG = LOGS_DIR / "app.log"
    
    HIGH_THRESHOLD = float(os.getenv("HIGH_THRESHOLD", "0.75"))
    LOW_THRESHOLD = float(os.getenv("LOW_THRESHOLD", "0.50"))
    
    # API Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    GPT2_API_URL = os.getenv("GPT2_API_URL", "http://localhost:8000/predict")
    
    # Processing Configuration
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
    BATCH_DELAY = int(os.getenv("BATCH_DELAY", "25"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# DATA MODELS
@dataclass
class Comment:
    """Model data komentar"""
    comment_id: str
    text: str
    timestamp: str
    original_text: Optional[str] = None
    normalized_text: Optional[str] = None
    tokens: Optional[List[str]] = None

@dataclass
class RuleBasedResult:
    """Hasil klasifikasi rule-based"""
    comment: Comment
    classification: str
    detection_method: str
    score: float
    detected_pattern: Optional[str] = None
    fuzzy_score: Optional[float] = None
    matched_keyword: Optional[str] = None

@dataclass
class LLMResult:
    """Hasil klasifikasi LLM"""
    comment: Comment
    model_name: str
    classification: str
    confidence: float
    latency_ms: float
    success: bool
    reasoning: Optional[str] = None
    error: Optional[str] = None

# UTILITAS
def setup_logging():
    Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(Config.APP_LOG, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def ensure_directories():
    Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

def save_json(filepath: Path, data):
    with open(filepath, 'w', encoding='utf-8', errors='xmlcharrefreplace') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    logging.info(f"Saved: {filepath}")

def load_json(filepath: Path) -> Optional[dict]:
    if not filepath.exists():
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def print_separator(char="=", length=70):
    logging.info(char * length)


# TAHAP 1: INPUT DATA
def load_input_data() -> List[Comment]:
    """Load komentar - batch atau text biasa"""
    print_separator()
    logging.info("TAHAP 1: INPUT DATA KOMENTAR YOUTUBE")
    print_separator()
    
    if Config.COMMENTS_RAW.exists():
        existing_data = load_json(Config.COMMENTS_RAW)
        if existing_data:
            logging.info(f"Ditemukan {len(existing_data)} komentar")
            response = input("Gunakan data yang ada? (y/n): ").strip().lower()
            if response == 'y':
                logging.info(f"Menggunakan {len(existing_data)} komentar dari file")
                return [Comment(**item) for item in existing_data]
    
    logging.info("Silakan input komentar")
    print('\n2 cara input:')
    print('  1. Ketik "batch" - load dari data/comments_raw.json')
    print('  2. Text biasa - ketik komentar langsung\n')
    
    user_input = input("Input: ").strip()
    
    if user_input.lower() == "batch":
        data = load_json(Config.COMMENTS_RAW)
        if data:
            logging.info(f"Loaded {len(data)} komentar")
            return [Comment(**item) for item in data]
        else:
            logging.error("File tidak ditemukan")
            raise FileNotFoundError("File comments_raw.json tidak ditemukan")
    
    if not user_input:
        raise ValueError("Input tidak boleh kosong")
    
    logging.info("Input detected sebagai plain text")

    try:
        clean_text = user_input.encode('utf-16', 'surrogatepass').decode('utf-16')
    except Exception:
        clean_text = user_input.encode('utf-8', 'ignore').decode('utf-8')
    comment = Comment(
        comment_id="comment_1",
        text=clean_text,
        timestamp=datetime.now().isoformat()
    )
    save_json(Config.COMMENTS_RAW, [asdict(comment)])
    logging.info("1 komentar disimpan")
    
    return [comment]

# TAHAP 2: PRA-PEMROSESAN
class TextPreprocessor:
    """Normalisasi teks sesuai 3.5.2"""
    
    LEET_MAP = {
        '0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's',
        '7': 't', '8': 'b', '@': 'a', '$': 's', '!': 'i'
    }
    
    @staticmethod
    def normalize_leet_speak(text: str) -> str:
        return ''.join([TextPreprocessor.LEET_MAP.get(c, c) for c in text])
    
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
    
    @classmethod
    def preprocess(cls, comment: Comment) -> Comment:
        comment.original_text = comment.text
        comment.normalized_text = cls.normalize_text(comment.text)
        comment.tokens = cls.tokenize(comment.normalized_text)
        return comment

def preprocess_data(comments: List[Comment]) -> List[Comment]:
    print_separator()
    logging.info("TAHAP 2: PRA-PEMROSESAN DATA")
    logging.info("   (Case Folding, Leet Speak Conversion, Tokenisasi)")
    print_separator()
    
    preprocessed = [TextPreprocessor.preprocess(c) for c in comments]
    save_json(Config.COMMENTS_NORMALIZED, [asdict(c) for c in preprocessed])
    logging.info(f"Normalisasi selesai: {len(preprocessed)} komentar\n")
    
    return preprocessed

# TAHAP 3: PENYARINGAN BERBASIS ATURAN
class RuleBasedClassifier:
    """Rule-based dengan Regex & Fuzzy Matching"""
    
    # Pola khusus untuk KONTEN JUDI ONLINE (bukan spam umum)
    GAMBLING_PATTERNS = {
        'slot_keywords': r'(slot|gacor|rtp|maxwin|scatter|zeus|olympus|gates|sweet\s*bonanza|starlight\s*princess)',
        'gambling_terms': r'(judi|casino|togel|bandar|betting|taruhan|jackpot|poker|baccarat)',
        'money_terms': r'(bonus|deposit|depo|wd|withdraw|dana|pulsa|gopay|ovo)',
        'provider_names': r'(pragmatic|pgsoft|habanero|spadegaming|joker123|microgaming)',
        'registration': r'(daftar|register|gabung|join|sign\s*up)',
        'gambling_urls': r'(https?://[^\s]+\.(bet|casino|slot|gaming)|t\.me/|wa\.me/)',
        'contact_judi': r'(wa\s*\d+|kontak\s*judi|cs\s*online|admin\s*slot)'
    }
    
    GAMBLING_KEYWORDS = [
        'slot', 'gacor', 'judi', 'casino', 'togel', 'bandar',
        'maxwin', 'deposit', 'bonus', 'pragmatic', 'daftar',
        'jackpot', 'scatter', 'zeus', 'olympus', 'gates',
        'betting', 'taruhan', 'withdraw', 'depo', 'rtp'
    ]
    
    @classmethod
    def detect_with_regex(cls, comment: Comment) -> Tuple[bool, Optional[str]]:
        text = comment.normalized_text or comment.text
        for pattern_name, pattern in cls.GAMBLING_PATTERNS.items():
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
        distance = cls.levenshtein_distance(s1, s2)
        return 1 - (distance / max_len)
    
    @classmethod
    def detect_with_fuzzy_matching(cls, comment: Comment) -> Tuple[float, Optional[str]]:
        tokens = comment.tokens or []
        max_score = 0.0
        matched_keyword = None
        
        for token in tokens:
            for keyword in cls.GAMBLING_KEYWORDS:
                score = cls.similarity_score(token, keyword)
                if score > max_score:
                    max_score = score
                    matched_keyword = keyword
        return max_score, matched_keyword
    
    @classmethod
    def classify(cls, comment: Comment) -> RuleBasedResult:
        """Klasifikasi: JUDI_ONLINE, BUKAN_JUDI_ONLINE, atau AMBIGU"""
        is_gambling, pattern_name = cls.detect_with_regex(comment)
        
        if is_gambling:
            return RuleBasedResult(
                comment=comment,
                classification="JUDI_ONLINE",
                detection_method="Regex",
                score=1.0,
                detected_pattern=pattern_name
            )
        
        fuzzy_score, matched_keyword = cls.detect_with_fuzzy_matching(comment)
        
        if fuzzy_score >= Config.HIGH_THRESHOLD:
            classification = "JUDI_ONLINE"
        elif fuzzy_score < Config.LOW_THRESHOLD:
            classification = "BUKAN_JUDI_ONLINE"
        else:
            classification = "AMBIGU"
        
        return RuleBasedResult(
            comment=comment,
            classification=classification,
            detection_method="Fuzzy Matching",
            score=fuzzy_score,
            fuzzy_score=fuzzy_score,
            matched_keyword=matched_keyword
        )

def rule_based_filtering(comments: List[Comment]) -> Dict:
    print_separator()
    logging.info("TAHAP 3: PENYARINGAN BERBASIS ATURAN")
    logging.info("   (Regex & Fuzzy Matching)")
    print_separator()
    
    results = [RuleBasedClassifier.classify(c) for c in comments]
    
    high_score = [r for r in results if r.classification == "JUDI_ONLINE"]
    low_score = [r for r in results if r.classification == "BUKAN_JUDI_ONLINE"]
    medium_score = [r for r in results if r.classification == "AMBIGU"]
    
    rule_based_results = {
        'high_score': [asdict(r) for r in high_score],
        'low_score': [asdict(r) for r in low_score],
        'medium_score': [asdict(r) for r in medium_score],
        'statistics': {
            'total': len(comments),
            'judi_online': len(high_score),
            'bukan_judi_online': len(low_score),
            'ambigu': len(medium_score)
        }
    }
    
    save_json(Config.RULE_BASED_RESULTS, rule_based_results)
    save_json(Config.AMBIGUOUS_DATA, [asdict(r) for r in medium_score])
    
    logging.info(f"\nHasil Penyaringan:")
    logging.info(f"   Skor Tinggi (Judi Online):        {len(high_score)} komentar")
    logging.info(f"   Skor Rendah (Bukan Judi Online):  {len(low_score)} komentar")
    logging.info(f"   Skor Menengah (Data Ambigu):      {len(medium_score)} komentar\n")
    
    return rule_based_results

# TAHAP 4: KLASIFIKASI LLM
class GeminiClassifier:
    """Platform Gemini dengan Prompt Engineering"""
    
    def __init__(self):
        if not Config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found")
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
    
    def classify(self, comment: Comment) -> LLMResult:
        for attempt in range(Config.MAX_RETRIES):
            try:
                return self._classify_single(comment)
            except Exception as e:
                if "quota" in str(e).lower() or "rate" in str(e).lower():
                    if attempt < Config.MAX_RETRIES - 1:
                        wait_time = (attempt + 1) * 20
                        logging.warning(f"Rate limit, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                if attempt == Config.MAX_RETRIES - 1:
                    return LLMResult(
                        comment=comment, model_name="Gemini",
                        classification="ERROR", confidence=0.0,
                        latency_ms=0.0, success=False, error=str(e)
                    )
        return LLMResult(
            comment=comment, model_name="Gemini",
            classification="ERROR", confidence=0.0,
            latency_ms=0.0, success=False, error="Max retries"
        )
    
    def _classify_single(self, comment: Comment) -> LLMResult:
        start_time = time.time()
        
        prompt = f"""Anda adalah sistem klasifikasi konten judi online untuk YouTube.

Tugas: Klasifikasikan apakah komentar ini adalah KONTEN JUDI ONLINE atau BUKAN.

Komentar: "{comment.text}"

INSTRUKSI:
1. Format JSON:
{{
  "classification": "JUDI_ONLINE" atau "BUKAN_JUDI_ONLINE",
  "confidence": <0.0-1.0>,
  "reasoning": "<alasan>"
}}

2. Indikator JUDI_ONLINE:
   - Kata kunci: slot, gacor, maxwin, deposit, casino, togel, betting
   - URL judi/betting
   - Ajakan daftar/registrasi
   - Promosi bonus/jackpot
   - Nama provider (pragmatic, pgsoft, dll)

3. FOKUS: Konten judi online (bukan spam umum).

Jawab HANYA JSON (tanpa markdown):"""
        
        response = self.model.generate_content(prompt)
        response_text = response.text.strip()
        clean_response = re.sub(r'```json|```', '', response_text).strip()
        parsed = json.loads(clean_response)
        latency_ms = (time.time() - start_time) * 1000
        
        return LLMResult(
            comment=comment, model_name="Gemini",
            classification=parsed['classification'],
            confidence=float(parsed['confidence']),
            latency_ms=latency_ms, success=True,
            reasoning=parsed.get('reasoning')
        )

class GPT2Classifier:
    """Model LLM-Judol (GPT-2 Fine-Tuned)"""
    
    def classify(self, comment: Comment) -> LLMResult:
        start_time = time.time()
        try:
            response = requests.post(
                Config.GPT2_API_URL,
                json={'comment': comment.text},
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            if not response.ok:
                raise Exception(f"HTTP {response.status_code}")
            
            data = response.json()
            latency_ms = (time.time() - start_time) * 1000
            
            return LLMResult(
                comment=comment, model_name="GPT-2",
                classification="JUDI_ONLINE" if data['prediction'] == 1 else "BUKAN_JUDI_ONLINE",
                confidence=float(data['confidence']),
                latency_ms=latency_ms, success=True
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logging.error(f"GPT-2 error: {str(e)}")
            return LLMResult(
                comment=comment, model_name="GPT-2",
                classification="ERROR", confidence=0.0,
                latency_ms=latency_ms, success=False, error=str(e)
            )

def classify_ambiguous_data(ambiguous_results: List[dict]) -> Tuple[List[LLMResult], List[LLMResult]]:
    if not ambiguous_results:
        logging.warning("Tidak ada data ambigu")
        return [], []
    
    print_separator()
    logging.info("TAHAP 4: KLASIFIKASI MELALUI MODEL BAHASA BESAR")
    logging.info("   (Platform Gemini vs Model LLM-Judol)")
    print_separator()
    logging.info(f"Total data ambigu: {len(ambiguous_results)} komentar\n")
    
    comments = [Comment(**r['comment']) for r in ambiguous_results]
    gemini_classifier = GeminiClassifier()
    gpt2_classifier = GPT2Classifier()
    
    gemini_results = []
    gpt2_results = []
    
    for i in range(0, len(comments), Config.BATCH_SIZE):
        batch = comments[i:i + Config.BATCH_SIZE]
        batch_num = (i // Config.BATCH_SIZE) + 1
        total_batches = (len(comments) + Config.BATCH_SIZE - 1) // Config.BATCH_SIZE
        
        logging.info(f"Batch {batch_num}/{total_batches}...")
        
        for comment in batch:
            gemini_result = gemini_classifier.classify(comment)
            gpt2_result = gpt2_classifier.classify(comment)
            
            gemini_results.append(gemini_result)
            gpt2_results.append(gpt2_result)
            
            if not gemini_result.success:
                logging.error(f"Gemini gagal: {gemini_result.error}")
            if not gpt2_result.success:
                logging.error(f"GPT-2 gagal: {gpt2_result.error}")
        
        logging.info(f"Batch {batch_num}/{total_batches} selesai\n")
        
        if i + Config.BATCH_SIZE < len(comments):
            time.sleep(Config.BATCH_DELAY)
    
    save_json(Config.GEMINI_RESULTS, [asdict(r) for r in gemini_results])
    save_json(Config.GPT2_RESULTS, [asdict(r) for r in gpt2_results])
    
    logging.info("Klasifikasi selesai!\n")
    return gemini_results, gpt2_results

# TAHAP 6: ANALISIS (3.5.6)

def analyze_agreement(gemini_results: List[LLMResult], gpt2_results: List[LLMResult]) -> dict:
    print_separator()
    logging.info("ANALISIS KESEPAKATAN")
    print_separator()
    
    total_agreements = 0
    total_disagreements = 0
    agreement_details = []
    
    for gemini, gpt2 in zip(gemini_results, gpt2_results):
        if not gemini.success or not gpt2.success:
            continue
            
        agreed = gemini.classification == gpt2.classification
        if agreed:
            total_agreements += 1
        else:
            total_disagreements += 1
        
        agreement_details.append({
            'comment_id': gemini.comment.comment_id,
            'original_text': gemini.comment.original_text,
            'gemini_class': gemini.classification,
            'gpt2_class': gpt2.classification,
            'agreed': agreed,
            'gemini_confidence': gemini.confidence,
            'gpt2_confidence': gpt2.confidence
        })
    
    total = total_agreements + total_disagreements
    agreement_rate = (total_agreements / total * 100) if total > 0 else 0
    
    agreement_analysis = {
        'timestamp': datetime.now().isoformat(),
        'total_comments': total,
        'agreements': total_agreements,
        'disagreements': total_disagreements,
        'agreement_rate': round(agreement_rate, 2),
        'details': agreement_details
    }
    
    save_json(Config.AGREEMENT_ANALYSIS, agreement_analysis)
    
    logging.info(f"\nHasil:")
    logging.info(f"      Total: {total}")
    logging.info(f"      Setuju: {total_agreements} ({agreement_rate:.2f}%)")
    logging.info(f"      Tidak Setuju: {total_disagreements}\n")
    
    return agreement_analysis

def calculate_stats(values: List[float]) -> dict:
    if not values:
        return {'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'std_dev': 0}
    return {
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'min': min(values),
        'max': max(values),
        'std_dev': statistics.stdev(values) if len(values) > 1 else 0
    }

def analyze_distribution(gemini_results: List[LLMResult], gpt2_results: List[LLMResult]) -> dict:
    print_separator()
    logging.info("ANALISIS DISTRIBUSI")
    print_separator()
    
    gemini_conf = [r.confidence for r in gemini_results if r.success]
    gpt2_conf = [r.confidence for r in gpt2_results if r.success]
    gemini_lat = [r.latency_ms for r in gemini_results if r.success]
    gpt2_lat = [r.latency_ms for r in gpt2_results if r.success]
    
    distribution = {
        'timestamp': datetime.now().isoformat(),
        'confidence': {
            'gemini': calculate_stats(gemini_conf),
            'gpt2': calculate_stats(gpt2_conf)
        },
        'latency': {
            'gemini': calculate_stats(gemini_lat),
            'gpt2': calculate_stats(gpt2_lat)
        }
    }
    
    save_json(Config.DISTRIBUTION_ANALYSIS, distribution)
    
    logging.info(f"\nConfidence:")
    logging.info(f"   Gemini: mean={distribution['confidence']['gemini']['mean']:.3f}")
    logging.info(f"   GPT-2:  mean={distribution['confidence']['gpt2']['mean']:.3f}")
    
    logging.info(f"\nLatency:")
    logging.info(f"   Gemini: mean={distribution['latency']['gemini']['mean']:.1f}ms")
    logging.info(f"   GPT-2:  mean={distribution['latency']['gpt2']['mean']:.1f}ms\n")
    
    return distribution

def comparative_analysis(agreement: dict, distribution: dict,
                        gemini_results: List[LLMResult], gpt2_results: List[LLMResult]) -> dict:
    print_separator()
    logging.info("ANALISIS KOMPARATIF")
    print_separator()
    
    gemini_judi = sum(1 for r in gemini_results if r.success and r.classification == "JUDI_ONLINE")
    gpt2_judi = sum(1 for r in gpt2_results if r.success and r.classification == "JUDI_ONLINE")
    total = sum(1 for r in gemini_results if r.success)
    
    comparative = {
        'timestamp': datetime.now().isoformat(),
        'classification': {
            'gemini': {
                'judi_online': gemini_judi,
                'detection_rate': round((gemini_judi / total * 100) if total > 0 else 0, 2)
            },
            'gpt2': {
                'judi_online': gpt2_judi,
                'detection_rate': round((gpt2_judi / total * 100) if total > 0 else 0, 2)
            }
        },
        'agreement': {'rate': agreement['agreement_rate']},
        'confidence': {
            'gemini': distribution['confidence']['gemini'],
            'gpt2': distribution['confidence']['gpt2']
        },
        'latency': {
            'gemini': distribution['latency']['gemini'],
            'gpt2': distribution['latency']['gpt2']
        }
    }
    
    save_json(Config.COMPARATIVE_ANALYSIS, comparative)
    
    logging.info(f"\nRingkasan:")
    logging.info(f"   Gemini: {gemini_judi} judi ({comparative['classification']['gemini']['detection_rate']}%)")
    logging.info(f"   GPT-2:  {gpt2_judi} judi ({comparative['classification']['gpt2']['detection_rate']}%)")
    logging.info(f"   Agreement: {agreement['agreement_rate']:.2f}%\n")
    
    return comparative

# TAHAP 7: PELAPORAN
def generate_final_report() -> dict:
    print_separator()
    logging.info("TAHAP 7: PELAPORAN")
    print_separator()
    
    rule_based = load_json(Config.RULE_BASED_RESULTS)
    agreement = load_json(Config.AGREEMENT_ANALYSIS)
    distribution = load_json(Config.DISTRIBUTION_ANALYSIS)
    comparative = load_json(Config.COMPARATIVE_ANALYSIS)
    
    final_report = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'title': "Analisis Komparasi: Platform Gemini vs Model LLM-Judol",
            'subtitle': "Deteksi Konten Judi Online pada Komentar YouTube"
        },
        'statistics': {
            'total': rule_based['statistics']['total'],
            'rule_based': {
                'judi_online': rule_based['statistics']['judi_online'],
                'bukan_judi_online': rule_based['statistics']['bukan_judi_online'],
                'ambigu': rule_based['statistics']['ambigu']
            }
        },
        'comparison': {
            'gemini': {
                'classification': comparative['classification']['gemini'],
                'confidence': distribution['confidence']['gemini'],
                'latency': distribution['latency']['gemini']
            },
            'gpt2': {
                'classification': comparative['classification']['gpt2'],
                'confidence': distribution['confidence']['gpt2'],
                'latency': distribution['latency']['gpt2']
            }
        },
        'agreement': agreement
    }
    
    save_json(Config.FINAL_REPORT, final_report)
    
    logging.info("\nRINGKASAN EKSEKUTIF:")
    logging.info(f"   Total: {final_report['statistics']['total']} komentar")
    logging.info(f"   Judi (Rule): {final_report['statistics']['rule_based']['judi_online']}")
    logging.info(f"   Ambigu: {final_report['statistics']['rule_based']['ambigu']}")
    
    if final_report['statistics']['rule_based']['ambigu'] > 0:
        logging.info(f"\n   Platform Gemini:")
        logging.info(f"     - Deteksi: {final_report['comparison']['gemini']['classification']['judi_online']}")
        logging.info(f"     - Confidence: {final_report['comparison']['gemini']['confidence']['mean']:.3f}")
        
        logging.info(f"\n   Model GPT-2:")
        logging.info(f"     - Deteksi: {final_report['comparison']['gpt2']['classification']['judi_online']}")
        logging.info(f"     - Confidence: {final_report['comparison']['gpt2']['confidence']['mean']:.3f}")
        
        logging.info(f"\n   Agreement: {final_report['agreement']['agreement_rate']:.2f}%\n")
    
    return final_report


def main():
    try:
        setup_logging()
        ensure_directories()
        
        print_separator()
        logging.info("SISTEM DETEKSI KONTEN JUDI ONLINE")
        logging.info("   Platform Gemini vs Model LLM-Judol")
        print_separator()
        logging.info("")
        
        # Tahap 1-2: Input & Pra-pemrosesan
        raw_comments = load_input_data()
        normalized_comments = preprocess_data(raw_comments)
        
        # Tahap 3: Penyaringan Rule-based
        rule_results = rule_based_filtering(normalized_comments)
        
        if rule_results['statistics']['ambigu'] == 0:
            logging.info("\nSemua terklasifikasi dengan rule-based.\n")
            return
        
        # Tahap 4: Klasifikasi LLM
        ambiguous_data = load_json(Config.AMBIGUOUS_DATA)
        gemini_results, gpt2_results = classify_ambiguous_data(ambiguous_data)
        
        # Tahap 6: Analisis
        agreement = analyze_agreement(gemini_results, gpt2_results)
        distribution = analyze_distribution(gemini_results, gpt2_results)
        comparative = comparative_analysis(agreement, distribution, gemini_results, gpt2_results)
        
        # Tahap 7: Pelaporan
        generate_final_report()
        
        print_separator()
        logging.info("PROSES SELESAI!")
        print_separator()
        
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        logging.exception(e)
        raise

if __name__ == "__main__":
    main()
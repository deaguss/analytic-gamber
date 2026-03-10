import os
import sys
import io
import json
from pathlib import Path
import logging

# Ensure logs directory exists before configuring file handler
Path("logs").mkdir(parents=True, exist_ok=True)

try:
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/app.log", encoding='utf-8'),
    ]
)

def print_banner():
    print("=" * 70)
    print("AUTOMATED SPAM DETECTION PIPELINE")
    print("   Scraper -> Detector (Auto)")
    print("   Visualizer (Manual: python visualizer.py)")
    print("=" * 70)
    print()

def check_dependencies():
    required = {
        'youtube_comment_downloader': 'youtube-comment-downloader',
        'google.generativeai': 'google-generativeai',
        'pandas': 'pandas',
        'requests': 'requests',
        'dotenv': 'python-dotenv'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module.replace('-', '_').split('.')[0])
        except ImportError:
            missing.append(package)
    
    if missing:
        logging.error(f"Missing packages: {', '.join(missing)}")
        logging.error("Install with: pip install -r requirements.txt")
        return False
    
    logging.info("All dependencies installed")
    return True

def check_env_keys():
    from dotenv import load_dotenv
    load_dotenv()
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not gemini_key:
        logging.error("GEMINI_API_KEY not found in .env file")
        logging.error("Get key at: https://makersuite.google.com/app/apikey")
        return False
    
    if not openai_key:
        logging.error("OPENAI_API_KEY not found in .env file")
        logging.error("Get key at: https://platform.openai.com/api-keys")
        logging.error("IMPORTANT: Setup billing first")
        return False
    
    logging.info("API keys configured")
    return True

def check_urls_file():
    urls_file = Path("youtube_urls.json")
    
    if not urls_file.exists():
        logging.error(f"File not found: {urls_file}")
        logging.error("Create youtube_urls.json with video URLs")
        return False, 0
    
    try:
        with open(urls_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'videos' in data:
            urls = [item['url'] for item in data['videos']
                    if isinstance(item, dict) and 'url' in item]
        elif isinstance(data, list):
            urls = [u for u in data if isinstance(u, str)]
        elif isinstance(data, dict) and 'urls' in data:
            urls = data['urls']
        else:
            logging.error("Invalid youtube_urls.json format")
            return False, 0
        
        if not urls:
            logging.error("No URLs found in youtube_urls.json")
            return False, 0
        
        logging.info(f"Found {len(urls)} video URLs")
        return True, len(urls)
        
    except Exception as e:
        logging.error(f"Error reading youtube_urls.json: {e}")
        return False, 0

def check_scraped_data():
    youtube_dir = Path("youtube")
    
    if not youtube_dir.exists():
        return False, 0
    
    json_files = list(youtube_dir.glob("*.json"))
    
    if not json_files:
        return False, 0
    
    logging.info(f"Found {len(json_files)} scraped video files")
    return True, len(json_files)

def run_scraper():
    logging.info("")
    logging.info("="*70)
    logging.info("STEP 1: SCRAPING YOUTUBE COMMENTS")
    logging.info("="*70)
    logging.info("Reading URLs from youtube_urls.json...")
    
    try:
        import scraper
        
        success = scraper.main(auto_mode=True)
        
        if not success:
            logging.error("Scraper failed")
            return False
        
        logging.info("Scraping completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Scraper failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_detector():
    logging.info("")
    logging.info("="*70)
    logging.info("STEP 2: DETECTION & ANALYSIS")
    logging.info("   (Rule-Based -> LLM -> Aggregation -> Excel)")
    logging.info("="*70)
    
    try:
        import detector
        detector.main()
        return True
        
    except Exception as e:
        logging.error(f"Detector failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print_banner()
    
    logging.info("Checking dependencies...")
    if not check_dependencies():
        logging.error("")
        logging.error("Setup incomplete. Install dependencies first:")
        logging.error("   pip install -r requirements.txt")
        return
    
    logging.info("Checking API keys...")
    if not check_env_keys():
        logging.error("")
        logging.error("API keys not configured. Setup .env file:")
        logging.error("   1. cp .env.example .env")
        logging.error("   2. Edit .env with your API keys")
        return
    
    logging.info("Checking video URLs...")
    urls_ok, num_urls = check_urls_file()
    if not urls_ok:
        logging.error("")
        logging.error("Video URLs not configured")
        return
    
    logging.info(f"All prerequisites met! Ready to process {num_urls} videos")
    
    has_data, num_files = check_scraped_data()
    
    if has_data:
        logging.info(f"Found existing scraped data ({num_files} files)")
        response = input("Re-scrape data? (y/n): ").strip().lower()
        
        if response == 'y':
            logging.info("Re-scraping all videos...")
            if not run_scraper():
                logging.error("Scraping failed. Cannot proceed")
                return
        else:
            logging.info("Using existing scraped data")
    else:
        logging.info("No scraped data found. Starting scraper...")
        if not run_scraper():
            logging.error("Scraping failed. Cannot proceed")
            return
    
    logging.info("")
    logging.info("="*70)
    logging.info("Ready to run detection & analysis")
    logging.info("="*70)
    
    response = input("\nProceed with detection? (y/n): ").strip().lower()
    if response != 'y':
        logging.info("Cancelled by user")
        return
    
    if not run_detector():
        logging.error("Detection failed")
        return
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED")
    print("="*70)
    print("\nResults saved to:")
    print("   - youtube/              (scraped data)")
    print("   - results/per_video/    (per-video analysis)")
    print("   - results/              (aggregate results)")
    print("   - results/analysis_results.xlsx  (Excel file)")
    print("\nNext steps:")
    print("   1. Open: results/analysis_results.xlsx")
    print("   2. Generate charts: python visualizer.py")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
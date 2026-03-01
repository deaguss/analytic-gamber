import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urlparse, parse_qs
import sys
import io

try:
    from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR, SORT_BY_RECENT
except ImportError:
    print("Library youtube-comment-downloader belum terinstall!")
    print("\nInstall dengan:")
    print("  pip install youtube-comment-downloader\n")
    exit(1)

try:
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    try:
        sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except Exception:
        pass
except io.UnsupportedOperation:
    pass

class Config:
    URLS_FILE = Path("youtube_urls.json")
    OUTPUT_DIR = Path("youtube")
    LOG_FILE = Path("logs/scraper.log")
    
    MAX_COMMENTS_PER_VIDEO = 200
    SORT_MODE = SORT_BY_RECENT

def log(message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] [{level}] {message}"
    print(log_message)
    
    Config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(Config.LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_message + '\n')

def extract_video_id(url: str) -> Optional[str]:
    if 'youtu.be/' in url:
        return url.split('youtu.be/')[-1].split('?')[0]
    
    if 'youtube.com' in url:
        parsed = urlparse(url)
        if 'watch' in parsed.path:
            query_params = parse_qs(parsed.query)
            return query_params.get('v', [None])[0]
        if 'embed' in parsed.path:
            return parsed.path.split('/embed/')[-1]
    
    return None

class YouTubeScraper:
    
    def __init__(self):
        self.downloader = YoutubeCommentDownloader()
        self.comment_counter = 0
    
    def scrape_comments(self, video_url: str, max_comments: int = 200) -> List[Dict]:
        video_id = extract_video_id(video_url)
        if not video_id:
            log(f"Invalid URL: {video_url}", "ERROR")
            return []
        
        log(f"Scraping video: {video_id}")
        log(f"  URL: {video_url}")
        
        comments = []
        count = 0
        
        try:
            generator = self.downloader.get_comments_from_url(
                video_url, 
                sort_by=Config.SORT_MODE
            )
            
            for comment in generator:
                count += 1
                self.comment_counter += 1
                
                comment_text = comment.get('text', '')
                
                comment_data = {
                    'comment_id': f'comment_{self.comment_counter}',
                    'text': comment_text,
                    'timestamp': comment.get('time', datetime.now().isoformat()),
                }
                
                comments.append(comment_data)
                
                if count % 50 == 0:
                    log(f"  Progress: {count} comments scraped...")
                
                if max_comments > 0 and count >= max_comments:
                    break
        
        except Exception as e:
            log(f"Error scraping: {str(e)}", "ERROR")
        
        log(f"  Total scraped: {len(comments)} comments")
        
        return comments

def save_video_comments(video_id: str, video_url: str, comments: List[Dict]):
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    output_file = Config.OUTPUT_DIR / f"{video_id}.json"
    
    video_data = {
        'video_id': video_id,
        'video_url': video_url,
        'scraped_at': datetime.now().isoformat(),
        'total_comments': len(comments),
        'comments': comments
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(video_data, f, ensure_ascii=False, indent=2)
    
    log(f"Saved to: {output_file}")

def load_urls_from_file(filepath: Path) -> List[str]:
    if not filepath.exists():
        raise FileNotFoundError(f"File {filepath} tidak ditemukan!")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        urls = data
    elif isinstance(data, dict) and 'urls' in data:
        urls = data['urls']
    else:
        raise ValueError("Format file tidak valid! Harus list atau {urls: [...]}")
    
    log(f"Loaded {len(urls)} URLs from {filepath}")
    return urls

def main(auto_mode=False):
    print("=" * 70)
    print("YOUTUBE COMMENT SCRAPER V2 - CLEAN DATA COLLECTION")
    print("   NO FILTERING - Ambil semua komentar mentah")
    print("=" * 70)
    print()
    
    try:
        urls = load_urls_from_file(Config.URLS_FILE)
        if not urls:
            print(f"No URLs found in {Config.URLS_FILE}")
            return False
        
        print(f"Videos to scrape: {len(urls)}\n")
        print("URLs:")
        for idx, url in enumerate(urls[:5], 1):
            print(f"  {idx}. {url}")
        if len(urls) > 5:
            print(f"  ... and {len(urls) - 5} more")
        print()
        
        print(f"Settings:")
        print(f"   - Max comments per video: {Config.MAX_COMMENTS_PER_VIDEO}")
        print(f"   - Filter: NO (ambil semua)")
        print(f"   - Output: {Config.OUTPUT_DIR}/")
        print()
        
        if not auto_mode:
            response = input("Start scraping? (y/n): ").strip().lower()
            if response != 'y':
                print("Cancelled")
                return False
            print()
        
    except FileNotFoundError:
        print(f"File {Config.URLS_FILE} tidak ditemukan!")
        print(f"\nBuat file {Config.URLS_FILE} dengan format:")
        print("""
[
  "https://www.youtube.com/watch?v=VIDEO_ID_1",
  "https://www.youtube.com/watch?v=VIDEO_ID_2"
]
        """)
        return False
    except Exception as e:
        print(f"Error loading URLs: {e}")
        return False
    
    scraper = YouTubeScraper()
    results_summary = []
    
    log(f"\n{'='*70}")
    log(f"Starting scraping {len(urls)} videos")
    log(f"{'='*70}\n")
    
    for idx, url in enumerate(urls, 1):
        video_id = extract_video_id(url)
        
        log(f"\n{'#'*70}")
        log(f"# VIDEO {idx}/{len(urls)}: {video_id}")
        log(f"{'#'*70}")
        
        try:
            comments = scraper.scrape_comments(
                video_url=url,
                max_comments=Config.MAX_COMMENTS_PER_VIDEO
            )
            
            if comments:
                save_video_comments(video_id, url, comments)
                
                results_summary.append({
                    'video_id': video_id,
                    'url': url,
                    'comments_collected': len(comments)
                })
                
                log(f"Video {idx}/{len(urls)} completed: {len(comments)} comments")
            else:
                log(f"No comments found in video {video_id}", "WARN")
            
            if idx < len(urls):
                log("Waiting 2 seconds...")
                time.sleep(2)
        
        except Exception as e:
            log(f"Error processing video {url}: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 70)
    print("SCRAPING SUMMARY")
    print("=" * 70)
    print(f"Videos processed: {len(results_summary)}/{len(urls)}")
    print(f"Total comments collected: {sum(r['comments_collected'] for r in results_summary)}")
    print(f"Output directory: {Config.OUTPUT_DIR}")
    print()
    
    if results_summary:
        print("Per-video results:")
        for r in results_summary:
            print(f"  - {r['video_id']}: {r['comments_collected']} comments")
    
    print()
    print("Next step:")
    print("   python detector.py")
    print("   (Rule-based classification akan dilakukan di detector)")
    print()
    
    return True

if __name__ == "__main__":
    main()
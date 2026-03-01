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
    print("Error: Library youtube-comment-downloader belum terinstall!")
    print("\nInstall dengan:")
    print("  pip install youtube-comment-downloader\n")
    exit(1)


try:
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class Config:
    URLS_FILE = Path("youtube_urls.json")
    OUTPUT_FILE = Path("youtube/comments_raw.json")
    LOG_FILE = Path("logs/scraper.log")
    
    MAX_COMMENTS_PER_VIDEO = 200 
    SORT_MODE = SORT_BY_RECENT
    
    GAMBLING_KEYWORDS = [
        'slot', 'gacor', 'maxwin', 'deposit', 'bonus', 'casino',
        'togel', 'bandar', 'judi', 'bet', 'jackpot', 'scatter',
        'pragmatic', 'pg', 'olympus', 'zeus', 'gates', 'sweet',
        'daftar', 'register', 'link', 'situs', 'agen', 'bibit',
        'mega', 'asik', 'toto', 'wede', 'depo', 'wd', 'withdraw'
    ]
    
    UNICODE_FONTS = [
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        '𝐀𝐁𝐂𝐃𝐄𝐅𝐆𝐇𝐈𝐉𝐊𝐋𝐌𝐍𝐎𝐏𝐐𝐑𝐒𝐓𝐔𝐕𝐖𝐗𝐘𝐙',
        '𝐴𝐵𝐶𝐷𝐸𝐹𝐺𝐻𝐼𝐽𝐾𝐿𝑀𝑁𝑂𝑃𝑄𝑅𝑆𝑇𝑈𝑉𝑊𝑋𝑌𝑍',
        '𝑨𝑩𝑪𝑫𝑬𝑭𝑮𝑯𝑰𝑱𝑲𝑳𝑴𝑵𝑶𝑷𝑸𝑹𝑺𝑻𝑼𝑽𝑾𝑿𝒀𝒁',
    ]

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

def has_unicode_font(text: str) -> bool:
    for font_chars in Config.UNICODE_FONTS[1:]:
        if any(char in text for char in font_chars):
            return True
    return False

def has_gambling_keyword(text: str) -> bool:
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in Config.GAMBLING_KEYWORDS)

def has_suspicious_pattern(text: str) -> bool:
    import re
    patterns = [
        r'https?://[^\s]+',
        r't\.me/[^\s]+',
        r'wa\.me/[^\s]+',
        r'\d{10,}',
        r'(🎰|🎲|💰|💸|🤑){2,}',
    ]
    
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

def is_potential_spam(text: str) -> bool:
    return (
        has_unicode_font(text) or
        has_gambling_keyword(text) or
        has_suspicious_pattern(text)
    )

class YouTubeScraper:
    def __init__(self):
        self.downloader = YoutubeCommentDownloader()
        self.total_scraped = 0
        self.total_filtered = 0
        self.comment_counter = 0
    
    def scrape_comments(self, video_url: str, max_comments: int = 200, filter_spam: bool = True) -> List[Dict]:
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
                self.total_scraped += 1
                
                comment_text = comment.get('text', '')
                
                if filter_spam and not is_potential_spam(comment_text):
                    continue
                
                self.comment_counter += 1
                
                comment_data = {
                    'comment_id': f'comment_{self.comment_counter}',
                    'text': comment_text,
                    'timestamp': comment.get('time', datetime.now().isoformat()),
                }
                
                comments.append(comment_data)
                self.total_filtered += 1
                
                # Menampilkan log komentar judi yang ditemukan secara realtime
                try:
                    print(f"[+] Ditemukan ({self.total_filtered}): {comment_text[:100]}...")
                except UnicodeEncodeError:
                    safe_text = comment_text.encode('ascii', 'ignore').decode('ascii')
                    print(f"[+] Ditemukan ({self.total_filtered}): {safe_text[:100]}...")
                
                if max_comments > 0 and len(comments) >= max_comments:
                    break
                
                if count % 100 == 0:
                    log(f"  Progress: {count} scraped, {len(comments)} filtered")
        
        except Exception as e:
            log(f"Error scraping: {str(e)}", "ERROR")
        
        log(f"  Total scraped: {count}")
        log(f"  Total filtered: {len(comments)}")
        
        return comments
    
    def scrape_multiple_videos(self, video_urls: List[str], max_per_video: int = 200, filter_spam: bool = True) -> List[Dict]:
        all_comments = []
        
        log(f"\n{'='*70}")
        log(f"Starting scraping {len(video_urls)} videos")
        log(f"Filter spam: {filter_spam}")
        log(f"Max per video: {max_per_video if max_per_video > 0 else 'unlimited'}")
        log(f"{'='*70}\n")
        
        for idx, url in enumerate(video_urls, 1):
            log(f"\n[{idx}/{len(video_urls)}] Processing video...")
            try:
                comments = self.scrape_comments(
                    video_url=url,
                    max_comments=max_per_video,
                    filter_spam=filter_spam
                )
                all_comments.extend(comments)
                
                log(f"  Collected: {len(comments)} comments")
                log(f"  Total so far: {len(all_comments)} comments")
                
                if idx < len(video_urls):
                    time.sleep(2)
            except Exception as e:
                log(f"  Error: {str(e)}", "ERROR")
                continue
        
        log(f"\n{'='*70}")
        log(f"Scraping completed!")
        log(f"{'='*70}")
        log(f"Total videos processed: {len(video_urls)}")
        log(f"Total comments scraped: {self.total_scraped}")
        log(f"Total comments filtered: {self.total_filtered}")
        log(f"Final output: {len(all_comments)} comments")
        log(f"{'='*70}\n")
        
        return all_comments

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

def save_comments_to_json(comments: List[Dict], output_file: Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    existing_comments = []
    if output_file.exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_comments = json.load(f)
                log(f"Membaca {len(existing_comments)} komentar lama dari file.")
        except json.JSONDecodeError:
            pass 
            
    all_comments = existing_comments + comments

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_comments, f, ensure_ascii=False, indent=2)
        
    log(f"Berhasil menambah {len(comments)} komentar baru.")
    log(f"TOTAL DATA SEKARANG: {len(all_comments)} komentar di {output_file}")

def main():
    print("=" * 70)
    print("YOUTUBE COMMENT SCRAPER")
    print("=" * 70)
    print()
    
    try:
        urls = load_urls_from_file(Config.URLS_FILE)
        if not urls:
            print(f"No URLs found in {Config.URLS_FILE}")
            return
        
        print(f"Videos to scrape: {len(urls)}\n")
        print("URLs:")
        for idx, url in enumerate(urls[:5], 1):
            print(f"  {idx}. {url}")
        if len(urls) > 5:
            print(f"  ... and {len(urls) - 5} more")
        print()
        
        response = input("Start scraping? (y/n): ").strip().lower()
        if response != 'y':
            print("Cancelled.")
            return
        print()
        
    except FileNotFoundError:
        print(f"File {Config.URLS_FILE} tidak ditemukan!")
        return
    except Exception as e:
        print(f"Error loading URLs: {e}")
        return
    
    scraper = YouTubeScraper()
    
    try:
        comments = scraper.scrape_multiple_videos(
            video_urls=urls,
            max_per_video=Config.MAX_COMMENTS_PER_VIDEO,
            filter_spam=True
        )
        
        if not comments:
            print("\nNo comments collected!")
            return
        
        save_comments_to_json(comments, Config.OUTPUT_FILE)
        
        print("\n" + "=" * 70)
        print("SCRAPING SUMMARY")
        print("=" * 70)
        print(f"Videos processed: {len(urls)}")
        print(f"Comments collected: {len(comments)}")
        print(f"Output file: {Config.OUTPUT_FILE}")
        print("\nScraping completed successfully!\n")
        
    except Exception as e:
        log(f"Fatal error: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
import sys
from pathlib import Path

def test_imports():
    """Test all required imports"""
    print("\n Testing imports...")
    errors = []
    
    modules = [
        ('google.generativeai', 'google-generativeai'),
        ('matplotlib.pyplot', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('requests', 'requests'),
        ('dotenv', 'python-dotenv'),
        ('numpy', 'numpy'),
    ]
    
    for module_name, package_name in modules:
        try:
            __import__(module_name)
            print(f"  {package_name}")
        except ImportError as e:
            errors.append(f"  {package_name}: {e}")
            print(f"  {package_name}")
    
    if errors:
        print("\nMissing dependencies:")
        for error in errors:
            print(error)
        print("\nRun: pip install -r requirements.txt")
        return False
    else:
        print("\nAll dependencies installed!")
        return True

def test_api_key():
    """Test Gemini API key"""
    print("\n Testing API key...")
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("  GEMINI_API_KEY not found in .env")
        print("  Create .env file with: GEMINI_API_KEY=your_key_here")
        return False
    elif api_key == "your_key_here" or len(api_key) < 20:
        print("  GEMINI_API_KEY looks like placeholder")
        print("  Replace with real API key from https://makersuite.google.com/app/apikey")
        return False
    else:
        print(f"  GEMINI_API_KEY found (length: {len(api_key)})")
        return True

def test_gemini_connection():
    """Test connection to Gemini API"""
    print("\n Testing Gemini API connection...")
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key or api_key == "your_key_here":
        print("  Skipping (no valid API key)")
        return False
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # Try listing models
        models = list(genai.list_models())
        print(f"  Connected! Found {len(models)} models")
        
        # Print available models
        print("\n  Available models:")
        for model in models[:5]:  # Show first 5
            print(f"     - {model.name}")
        
        return True
        
    except Exception as e:
        print(f"  Connection failed: {e}")
        print("  Check your API key and internet connection")
        return False

def test_directories():
    """Test directory structure"""
    print("\n Testing directories...")
    
    dirs = ['data', 'logs', 'visualizations']
    
    for dir_name in dirs:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        print(f"  {dir_name}/")
    
    return True

def test_file_structure():
    """Test required files exist"""
    print("\n Testing file structure...")
    
    required_files = [
        'index_fixed.py',
        'visualizer.py',
        'requirements.txt',
        'README.md',
        'TROUBLESHOOTING.md'
    ]
    
    all_exist = True
    for filename in required_files:
        if Path(filename).exists():
            print(f"  {filename}")
        else:
            print(f"  {filename} (missing)")
            all_exist = False
    
    return all_exist

def test_python_version():
    """Test Python version"""
    print("\n Testing Python version...")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major >= 3 and version.minor >= 8:
        print(f"  Python {version_str} (meets requirement >= 3.8)")
        return True
    else:
        print(f"  Python {version_str} (requires >= 3.8)")
        return False

def create_sample_data():
    """Create sample data file for testing"""
    print("\nCreating sample data...")
    
    sample_data = [
        {
            "comment_id": "test_1",
            "text": "Ini adalah komentar normal",
            "timestamp": "2024-01-01T00:00:00"
        },
        {
            "comment_id": "test_2",
            "text": "slot gacor maxwin hari ini",
            "timestamp": "2024-01-01T00:01:00"
        },
        {
            "comment_id": "test_3",
            "text": "Video bagus, terima kasih!",
            "timestamp": "2024-01-01T00:02:00"
        }
    ]
    
    import json
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    sample_file = data_dir / "comments_raw_sample.json"
    
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print(f"  Created {sample_file}")
    print("  You can copy this to comments_raw.json for testing")
    
    return True

def print_next_steps(all_passed):
    """Print next steps based on test results"""
    print("\n" + "=" * 70)
    
    if all_passed:
        print("ALL TESTS PASSED - READY TO RUN!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Run main analysis:")
        print("     python index_fixed.py")
        print("\n  2. Generate visualizations:")
        print("     python visualizer.py")
        print("\n  3. Check outputs in:")
        print("     - data/ (JSON files)")
        print("     - visualizations/ (PNG files)")
        print("     - logs/ (log files)")
    else:
        print("SOME TESTS FAILED - FIX ISSUES ABOVE")
        print("=" * 70)
        print("\nCommon fixes:")
        print("  1. Install dependencies:")
        print("     pip install -r requirements.txt")
        print("\n  2. Setup API key:")
        print("     Create .env file with GEMINI_API_KEY=your_key")
        print("\n  3. Check TROUBLESHOOTING.md for detailed help")
    
    print("=" * 70)

def main():
    """Run all tests"""
    print("=" * 70)
    print("SETUP VERIFICATION")
    print("=" * 70)
    
    results = []
    
    results.append(("Python Version", test_python_version()))
    results.append(("Dependencies", test_imports()))
    results.append(("API Key", test_api_key()))
    results.append(("Directories", test_directories()))
    results.append(("File Structure", test_file_structure()))
    results.append(("Sample Data", create_sample_data()))
    
    # Optional: Test API connection if key is valid
    if results[2][1]:  # If API key test passed
        results.append(("Gemini Connection", test_gemini_connection()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status:10} {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    print_next_steps(all_passed)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
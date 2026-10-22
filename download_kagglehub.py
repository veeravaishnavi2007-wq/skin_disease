import os
import sys
import io
import kagglehub

# Fix Windows console encoding issues for emojis and block characters
if sys.platform.startswith('win'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass

def main():
    print("Downloading the latest version of the skin-cancer-mnist-ham10000 dataset using kagglehub...")
    print("This might take several minutes depending on your internet connection (1.36 GB)...")
    
    try:
        path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
        print("\nSUCCESS!")
        print("Path to dataset files:", path)
    except Exception as e:
        print("\nERROR occurred during download:", e)
        sys.exit(1)

if __name__ == '__main__':
    main()

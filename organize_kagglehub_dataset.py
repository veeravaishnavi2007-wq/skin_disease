import os
import sys
import io
import shutil
from tqdm import tqdm

# Fix Windows console encoding issues for emojis and block characters
if sys.platform.startswith('win'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass

def main():
    source_dir = r"C:\Users\power\.cache\kagglehub\datasets\kmader\skin-cancer-mnist-ham10000\versions\2"
    dest_images_dir = "./HAM10000_images"
    dest_metadata = "./HAM10000_metadata.csv"
    
    if not os.path.exists(source_dir):
        print(f"ERROR: Source dataset directory not found at: {source_dir}")
        print("Please make sure the download completed successfully.")
        sys.exit(1)
        
    os.makedirs(dest_images_dir, exist_ok=True)
    
    # 1. Copy metadata CSV
    src_metadata = os.path.join(source_dir, "HAM10000_metadata.csv")
    if os.path.exists(src_metadata):
        print(f"Copying metadata file to {dest_metadata}...")
        shutil.copy2(src_metadata, dest_metadata)
    else:
        print("WARNING: HAM10000_metadata.csv not found in source directory.")
        
    # 2. Copy images from both parts
    parts = ["HAM10000_images_part_1", "HAM10000_images_part_2"]
    for part in parts:
        src_part_dir = os.path.join(source_dir, part)
        if not os.path.exists(src_part_dir):
            print(f"WARNING: Source directory {src_part_dir} not found. Skipping.")
            continue
            
        print(f"\nCopying images from {part}...")
        files = [f for f in os.listdir(src_part_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for filename in tqdm(files, desc=f"Copying {part}", ncols=80):
            src_file = os.path.join(src_part_dir, filename)
            dest_file = os.path.join(dest_images_dir, filename)
            # Use copy2 to preserve metadata, but only copy if file doesn't already exist to save time
            if not os.path.exists(dest_file):
                shutil.copy2(src_file, dest_file)
                
    # 3. Validation
    final_count = len([f for f in os.listdir(dest_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print("\n" + "=" * 50)
    print("SUCCESS: Dataset successfully organized in the workspace!")
    print(f"Images folder: {os.path.abspath(dest_images_dir)}")
    print(f"Total Images: {final_count} (Expected: 10015)")
    print(f"Metadata file: {os.path.abspath(dest_metadata)}")
    print("=" * 50)

if __name__ == '__main__':
    main()

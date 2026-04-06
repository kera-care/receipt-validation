"""
Prepare DocLayNet dataset as negative samples for multi-task learning.

This script:
1. Downloads DocLayNet core dataset from IBM Cloud Object Storage
2. Extracts COCO annotations to get document categories and splits
3. Samples 10% of images from train and validation splits (configurable)
4. Creates separate task files for train and validation
5. Saves images to multi-task-data/DocLayNet/images/
6. The same negatives can be used across all tasks with different extractors

Data sources:
- Core: https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip
- Extra (optional): https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_extra.zip

Performance notes:
- First run: ~10-15 minutes to download and extract ~30GB core dataset
- Downloads both train and val splits to prevent data leakage
- PNG images are 1025x1025px, already preprocessed
"""

import json
import random
import time
import urllib.request
import subprocess
from pathlib import Path
from typing import Dict
import argparse
from tqdm import tqdm


DOCLAYNET_CORE_URL = "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip"
DOCLAYNET_EXTRA_URL = "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_extra.zip"


def download_file(url: str, dest_path: Path, desc: str = "Downloading", max_retries: int = 5):
    """Download file with progress bar and resume capability.
    
    Args:
        url: URL to download from
        dest_path: Destination path for the downloaded file
        desc: Description for progress bar
        max_retries: Maximum number of retry attempts
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + '.part')
    
    # Check if already fully downloaded
    if dest_path.exists():
        print(f"✓ {dest_path.name} already exists, skipping download")
        return
    
    # Get total file size
    req = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(req) as response:
        total_size = int(response.headers.get('content-length', 0))
    
    # Check if partial download exists
    resume_pos = 0
    if temp_path.exists():
        resume_pos = temp_path.stat().st_size
        if resume_pos >= total_size:
            # Partial file is complete, just rename it
            temp_path.rename(dest_path)
            print(f"✓ {dest_path.name} already downloaded")
            return
        print(f"📥 Resuming {desc} from {resume_pos:,} bytes ({resume_pos/total_size*100:.1f}%)")
    else:
        print(f"📥 {desc} from {url}")
    
    retry_count = 0
    while retry_count < max_retries:
        try:
            # Set up request with Range header for resume
            headers = {}
            if resume_pos > 0:
                headers['Range'] = f'bytes={resume_pos}-'
            
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=60) as response:
                # Open file in append mode if resuming, else write mode
                mode = 'ab' if resume_pos > 0 else 'wb'
                
                with open(temp_path, mode) as f:
                    with tqdm(
                        total=total_size,
                        initial=resume_pos,
                        unit='B',
                        unit_scale=True,
                        desc=desc
                    ) as pbar:
                        chunk_size = 1024 * 1024  # 1MB chunks
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                            pbar.update(len(chunk))
                            resume_pos += len(chunk)
            
            # Download complete, rename to final name
            temp_path.rename(dest_path)
            print(f"✓ Downloaded to {dest_path}")
            return
            
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            retry_count += 1
            if retry_count < max_retries:
                wait_time = min(2 ** retry_count, 60)  # Exponential backoff, max 60s
                print(f"⚠️  Download interrupted: {e}")
                print(f"   Retry {retry_count}/{max_retries} in {wait_time}s...")
                time.sleep(wait_time)
                # Update resume position
                if temp_path.exists():
                    resume_pos = temp_path.stat().st_size
            else:
                print(f"❌ Failed to download after {max_retries} attempts")
                raise


def extract_zip(zip_path: Path, extract_to: Path):
    """Extract zip file using system unzip command for better performance."""
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    
    # Check if already extracted by looking for the expected directory
    expected_dir = extract_to / zip_path.stem  # e.g., DocLayNet_core
    if expected_dir.exists():
        print(f"✓ {zip_path.name} already extracted to {expected_dir}")
        return
    
    print(f"📦 Extracting {zip_path.name}...")
    extract_to.mkdir(parents=True, exist_ok=True)
    
    # Check if pv (pipe viewer) is available for progress
    pv_available = subprocess.run(['which', 'pv'], capture_output=True).returncode == 0
    
    # Use system unzip command (much faster than Python's zipfile)
    try:
        if pv_available:
            # Use pv to monitor unzip progress by watching the extraction directory size grow
            cmd = f'unzip -o "{zip_path}" -d "{extract_to}" | pv -l -s $(unzip -l "{zip_path}" | tail -1 | awk \'{{print $2}}\') > /dev/null'
            subprocess.run(cmd, shell=True, check=True)
        else:
            # Fallback: regular unzip without progress
            subprocess.run(
                ['unzip', '-q', '-o', str(zip_path), '-d', str(extract_to)],
                check=True,
                capture_output=True,
                text=True
            )
        print(f"✓ Extracted to {extract_to}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Extraction failed")
        raise
    except FileNotFoundError:
        print("❌ 'unzip' command not found. Please install unzip utility.")
        raise


def load_coco_annotations(coco_json_path: Path) -> Dict:
    """Load COCO format annotations."""
    print(f"📋 Loading annotations from {coco_json_path.name}")
    
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Build image_id -> category mapping
    # In DocLayNet COCO format, categories are stored in image metadata
    image_info = {}
    for img in coco_data['images']:
        image_info[img['id']] = {
            'file_name': img['file_name'],
            'doc_category': img.get('doc_category', 'unknown'),
            'collection': img.get('collection', 'unknown'),
        }
    
    print(f"✓ Loaded {len(image_info)} image annotations")
    return image_info


def download_and_prepare_doclaynet(
    output_dir: Path,
    sample_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Download DocLayNet and prepare negative samples.
    
    Args:
        output_dir: Base directory for outputs
        sample_ratio: Ratio of dataset to use (default 0.1 = 10%)
        seed: Random seed for reproducibility
    """
    print("=" * 80)
    print("PREPARING DOCLAYNET NEGATIVES FOR MULTI-TASK LEARNING")
    print("=" * 80)
    
    # Create output directories
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    tasks_dir = output_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    
    download_dir = output_dir / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    
    extract_dir = output_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir}")
    print(f"📁 Images directory: {images_dir}")
    print(f"📁 Tasks directory: {tasks_dir}")
    print(f"📁 Download directory: {download_dir}")
    print(f"📁 Extract directory: {extract_dir}")
    
    # Download DocLayNet core dataset
    print(f"\n{'='*80}")
    print("STEP 1: DOWNLOADING DOCLAYNET")
    print(f"{'='*80}")
    
    core_zip = download_dir / "DocLayNet_core.zip"
    download_file(DOCLAYNET_CORE_URL, core_zip, "DocLayNet Core (~30GB)")
    
    # Extract dataset
    print(f"\n{'='*80}")
    print("STEP 2: EXTRACTING DATASET")
    print(f"{'='*80}")
    
    extract_zip(core_zip, extract_dir)
    
    # Load COCO annotations for train and val
    print(f"\n{'='*80}")
    print("STEP 3: LOADING ANNOTATIONS")
    print(f"{'='*80}")
    
    core_path = extract_dir / "DocLayNet_core"
    train_coco = core_path / "COCO" / "train.json"
    val_coco = core_path / "COCO" / "val.json"
    png_dir = core_path / "PNG"
    
    if not train_coco.exists():
        raise FileNotFoundError(f"Train annotations not found: {train_coco}")
    if not val_coco.exists():
        raise FileNotFoundError(f"Val annotations not found: {val_coco}")
    if not png_dir.exists():
        raise FileNotFoundError(f"PNG directory not found: {png_dir}")
    
    train_image_info = load_coco_annotations(train_coco)
    val_image_info = load_coco_annotations(val_coco)
    
    # Process train split
    print(f"\n{'='*80}")
    print(f"STEP 4: PROCESSING TRAIN SPLIT")
    print(f"{'='*80}")
    
    # Get list of available image IDs
    train_image_ids = list(train_image_info.keys())
    train_samples_to_use = int(len(train_image_ids) * sample_ratio)
    print(f"\n📊 Total train images: {len(train_image_ids):,}")
    print(f"📊 Sampling {sample_ratio*100}% = {train_samples_to_use:,} samples")
    
    # Generate random indices for train
    random.seed(seed)
    train_selected_ids = random.sample(train_image_ids, train_samples_to_use)
    
    # Examine first sample
    first_id = train_image_ids[0]
    first_info = train_image_info[first_id]
    print(f"\n🔍 Dataset structure (first sample):")
    print(f"  - image_id: {first_id}")
    print(f"  - file_name: {first_info['file_name']}")
    print(f"  - doc_category: {first_info['doc_category']}")
    print(f"  - collection: {first_info['collection']}")
    
    train_task_samples = []
    
    for i, img_id in enumerate(tqdm(train_selected_ids, desc="Processing train images")):
        info = train_image_info[img_id]
        doc_category = info['doc_category']
        file_name = info['file_name']
        
        # Source PNG path
        source_png = png_dir / file_name
        if not source_png.exists():
            print(f"  ⚠️  Image not found: {source_png}, skipping...")
            continue
        
        # Destination path - use original filename
        dest_filename = file_name
        dest_path = images_dir / dest_filename
        
        # Copy image if not already copied
        if not dest_path.exists():
            import shutil
            shutil.copy2(source_png, dest_path)
        
        # Create task entry
        task_entry = {
            "image_urls": [dest_filename],
            "doc_category": doc_category,
        }
        
        train_task_samples.append(task_entry)
    
    # Process validation split
    print(f"\n{'='*80}")
    print(f"STEP 5: PROCESSING VALIDATION SPLIT")
    print(f"{'='*80}")
    
    # Get list of available image IDs
    val_image_ids = list(val_image_info.keys())
    val_samples_to_use = int(len(val_image_ids) * sample_ratio)
    print(f"\n📊 Total validation images: {len(val_image_ids):,}")
    print(f"📊 Sampling {sample_ratio*100}% = {val_samples_to_use:,} samples")
    
    # Generate random indices for validation (different seed)
    random.seed(seed + 1000)
    val_selected_ids = random.sample(val_image_ids, val_samples_to_use)
    
    val_task_samples = []
    
    for i, img_id in enumerate(tqdm(val_selected_ids, desc="Processing val images")):
        info = val_image_info[img_id]
        doc_category = info['doc_category']
        file_name = info['file_name']
        
        # Source PNG path
        source_png = png_dir / file_name
        if not source_png.exists():
            print(f"  ⚠️  Image not found: {source_png}, skipping...")
            continue
        
        # Destination path - use original filename
        dest_filename = file_name
        dest_path = images_dir / dest_filename
        
        # Copy image if not already copied
        if not dest_path.exists():
            shutil.copy2(source_png, dest_path)
        
        # Create task entry
        task_entry = {
            "image_urls": [dest_filename],
            "doc_category": doc_category,
        }
        
        val_task_samples.append(task_entry)
    
    # Save task files
    print(f"\n{'='*80}")
    print(f"STEP 6: SAVING TASK FILES")
    print(f"{'='*80}")
    
    train_task_file = tasks_dir / "train_tasks.json"
    val_task_file = tasks_dir / "val_tasks.json"
    
    with open(train_task_file, 'w') as f:
        json.dump(train_task_samples, f, indent=2)
    
    with open(val_task_file, 'w') as f:
        json.dump(val_task_samples, f, indent=2)
    
    print(f"\n✅ Saved {len(train_task_samples):,} train samples to {train_task_file}")
    print(f"✅ Saved {len(val_task_samples):,} validation samples to {val_task_file}")
    
    # Show category distribution for train
    print(f"\n📊 Train category distribution:")
    categories = {}
    for entry in train_task_samples:
        cat = entry['doc_category']
        categories[cat] = categories.get(cat, 0) + 1
    
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {cat}: {count:,} ({count/len(train_task_samples)*100:.1f}%)")
    
    # Show category distribution for validation
    print(f"\n📊 Validation category distribution:")
    val_categories = {}
    for entry in val_task_samples:
        cat = entry['doc_category']
        val_categories[cat] = val_categories.get(cat, 0) + 1
    
    for cat, count in sorted(val_categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {cat}: {count:,} ({count/len(val_task_samples)*100:.1f}%)")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Total images saved: {len(list(images_dir.glob('*.png'))):,}")
    print(f"✅ Train task file: {train_task_file} ({len(train_task_samples):,} samples)")
    print(f"✅ Validation task file: {val_task_file} ({len(val_task_samples):,} samples)")
    print(f"\n✅ Done! Use these files as negative samples in your multi-task config.")
    print(f"\nExample usage in config:")
    print(f"  negative_samples:")
    print(f"    - train_tasks_path: {train_task_file}")
    print(f"      val_tasks_path: {val_task_file}")
    print(f"      images_root_dir: {images_dir}")
    print(f"      images_url_key: image_urls")
    print(f"      negative_extractor: doclaynet_to_drug_extraction  # or prescription/receipt")
    print(f"      ratio: 1.0")


def main():
    parser = argparse.ArgumentParser(description="Prepare DocLayNet negatives for multi-task learning")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="multi-task-data/DocLayNet",
        help="Output directory for images and task files"
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=0.1,
        help="Ratio of dataset to sample (default: 0.1 = 10%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    download_and_prepare_doclaynet(
        output_dir=output_dir,
        sample_ratio=args.sample_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
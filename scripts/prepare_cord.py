"""
Prepare CORD (Consolidated Receipt Dataset) v1 and v2 as negative samples for multi-task learning.

This script:
1. Downloads CORD v1 and v2 datasets from HuggingFace
2. Extracts images from both versions
3. Samples images from train and test splits (configurable)
4. Creates separate task files for train and validation
5. Saves images to multi-task-data/CORD/images/
6. These non-medical receipts serve as:
   - Negatives for prescription validation (not prescriptions)
   - Negatives for drug extraction (no medical content)
   - Hard negatives for receipt validation (receipts but not medical)

Dataset info:
- CORD v1: naver-clova-ix/cord-v1
- CORD v2: naver-clova-ix/cord-v2
- Content: Non-medical receipts (restaurants, retail, etc.)
- Format: PIL images with OCR annotations

Performance notes:
- First run: ~5-10 minutes to download datasets
- Uses HuggingFace datasets library with caching
- Images are saved as PNG files
"""

import json
import random
from pathlib import Path
from typing import Dict, List
import argparse
from tqdm import tqdm
from datasets import load_dataset


def download_and_prepare_cord(
    output_dir: Path,
    sample_ratio: float = 1.0,
    seed: int = 42,
    include_v1: bool = True,
    include_v2: bool = True,
):
    """
    Download CORD datasets and prepare negative samples.
    
    Args:
        output_dir: Base directory for outputs
        sample_ratio: Ratio of dataset to use (default 1.0 = 100%)
        seed: Random seed for reproducibility
        include_v1: Whether to include CORD v1
        include_v2: Whether to include CORD v2
    """
    print("=" * 80)
    print("PREPARING CORD NEGATIVES FOR MULTI-TASK LEARNING")
    print("=" * 80)
    
    if not include_v1 and not include_v2:
        raise ValueError("At least one of CORD v1 or v2 must be included")
    
    # Create output directories
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    tasks_dir = output_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir}")
    print(f"📁 Images directory: {images_dir}")
    print(f"📁 Tasks directory: {tasks_dir}")
    
    all_train_samples = []
    all_val_samples = []
    
    # Process CORD v1
    if include_v1:
        print(f"\n{'='*80}")
        print("STEP 1: DOWNLOADING CORD V1")
        print(f"{'='*80}")
        
        v1_dataset = load_dataset("naver-clova-ix/cord-v1")
        v1_train = v1_dataset["train"]
        v1_validation = v1_dataset["validation"]
        v1_test = v1_dataset["test"]
        
        print(f"✓ Loaded CORD v1 - Train: {len(v1_train)}, Validation: {len(v1_validation)}, Test: {len(v1_test)}")
        
        # Process v1 train split
        print(f"\n{'='*80}")
        print("STEP 2: PROCESSING CORD V1 TRAIN SPLIT")
        print(f"{'='*80}")
        
        train_samples_to_use = int(len(v1_train) * sample_ratio)
        print(f"\n📊 Total CORD v1 train images: {len(v1_train):,}")
        print(f"📊 Sampling {sample_ratio*100}% = {train_samples_to_use:,} samples")
        
        random.seed(seed)
        train_indices = random.sample(range(len(v1_train)), train_samples_to_use)
        
        for idx in tqdm(train_indices, desc="Processing CORD v1 train"):
            sample = v1_train[idx]
            image = sample["image"]  # PIL image
            
            # Generate filename
            filename = f"cord_v1_train_{idx}.png"
            image_path = images_dir / filename
            
            # Save image if not already saved
            if not image_path.exists():
                image.save(image_path, "PNG")
            
            # Create task entry
            task_entry = {
                "image_urls": [filename],
                "dataset": "cord_v1",
                "split": "train",
                "receipt_type": "non_medical",
            }
            
            all_train_samples.append(task_entry)
        
        # Process v1 validation split
        print(f"\n{'='*80}")
        print("STEP 3: PROCESSING CORD V1 VALIDATION SPLIT")
        print(f"{'='*80}")
        
        val_samples_to_use = int(len(v1_validation) * sample_ratio)
        print(f"\n📊 Total CORD v1 validation images: {len(v1_validation):,}")
        print(f"📊 Sampling {sample_ratio*100}% = {val_samples_to_use:,} samples")
        
        random.seed(seed + 1000)
        val_indices = random.sample(range(len(v1_validation)), val_samples_to_use)
        
        for idx in tqdm(val_indices, desc="Processing CORD v1 validation"):
            sample = v1_validation[idx]
            image = sample["image"]
            
            filename = f"cord_v1_validation_{idx}.png"
            image_path = images_dir / filename
            
            if not image_path.exists():
                image.save(image_path, "PNG")
            
            task_entry = {
                "image_urls": [filename],
                "dataset": "cord_v1",
                "split": "validation",
                "receipt_type": "non_medical",
            }
            
            all_val_samples.append(task_entry)
        
        # Process v1 test split (also add to validation)
        print(f"\n{'='*80}")
        print("STEP 4: PROCESSING CORD V1 TEST SPLIT")
        print(f"{'='*80}")
        
        test_samples_to_use = int(len(v1_test) * sample_ratio)
        print(f"\n📊 Total CORD v1 test images: {len(v1_test):,}")
        print(f"📊 Sampling {sample_ratio*100}% = {test_samples_to_use:,} samples")
        
        random.seed(seed + 2000)
        test_indices = random.sample(range(len(v1_test)), test_samples_to_use)
        
        for idx in tqdm(test_indices, desc="Processing CORD v1 test"):
            sample = v1_test[idx]
            image = sample["image"]
            
            filename = f"cord_v1_test_{idx}.png"
            image_path = images_dir / filename
            
            if not image_path.exists():
                image.save(image_path, "PNG")
            
            task_entry = {
                "image_urls": [filename],
                "dataset": "cord_v1",
                "split": "test",
                "receipt_type": "non_medical",
            }
            
            all_val_samples.append(task_entry)
    
    # Process CORD v2
    if include_v2:
        print(f"\n{'='*80}")
        print(f"STEP {5 if include_v1 else 1}: DOWNLOADING CORD V2")
        print(f"{'='*80}")
        
        v2_dataset = load_dataset("naver-clova-ix/cord-v2")
        v2_train = v2_dataset["train"]
        v2_validation = v2_dataset["validation"]
        v2_test = v2_dataset["test"]
        
        print(f"✓ Loaded CORD v2 - Train: {len(v2_train)}, Validation: {len(v2_validation)}, Test: {len(v2_test)}")
        
        # Process v2 train split
        step_num = 6 if include_v1 else 2
        print(f"\n{'='*80}")
        print(f"STEP {step_num}: PROCESSING CORD V2 TRAIN SPLIT")
        print(f"{'='*80}")
        
        train_samples_to_use = int(len(v2_train) * sample_ratio)
        print(f"\n📊 Total CORD v2 train images: {len(v2_train):,}")
        print(f"📊 Sampling {sample_ratio*100}% = {train_samples_to_use:,} samples")
        
        random.seed(seed + 2000)
        train_indices = random.sample(range(len(v2_train)), train_samples_to_use)
        
        for idx in tqdm(train_indices, desc="Processing CORD v2 train"):
            sample = v2_train[idx]
            image = sample["image"]
            
            filename = f"cord_v2_train_{idx}.png"
            image_path = images_dir / filename
            
            if not image_path.exists():
                image.save(image_path, "PNG")
            
            task_entry = {
                "image_urls": [filename],
                "dataset": "cord_v2",
                "split": "train",
                "receipt_type": "non_medical",
            }
            
            all_train_samples.append(task_entry)
        
        # Process v2 validation split
        step_num = 7 if include_v1 else 3
        print(f"\n{'='*80}")
        print(f"STEP {step_num}: PROCESSING CORD V2 VALIDATION SPLIT")
        print(f"{'='*80}")
        
        val_samples_to_use = int(len(v2_validation) * sample_ratio)
        print(f"\n📊 Total CORD v2 validation images: {len(v2_validation):,}")
        print(f"📊 Sampling {sample_ratio*100}% = {val_samples_to_use:,} samples")
        
        random.seed(seed + 3000)
        val_indices = random.sample(range(len(v2_validation)), val_samples_to_use)
        
        for idx in tqdm(val_indices, desc="Processing CORD v2 validation"):
            sample = v2_validation[idx]
            image = sample["image"]
            
            filename = f"cord_v2_validation_{idx}.png"
            image_path = images_dir / filename
            
            if not image_path.exists():
                image.save(image_path, "PNG")
            
            task_entry = {
                "image_urls": [filename],
                "dataset": "cord_v2",
                "split": "validation",
                "receipt_type": "non_medical",
            }
            
            all_val_samples.append(task_entry)
        
        # Process v2 test split (also add to validation)
        step_num = 8 if include_v1 else 4
        print(f"\n{'='*80}")
        print(f"STEP {step_num}: PROCESSING CORD V2 TEST SPLIT")
        print(f"{'='*80}")
        
        test_samples_to_use = int(len(v2_test) * sample_ratio)
        print(f"\n📊 Total CORD v2 test images: {len(v2_test):,}")
        print(f"📊 Sampling {sample_ratio*100}% = {test_samples_to_use:,} samples")
        
        random.seed(seed + 3000)
        test_indices = random.sample(range(len(v2_test)), test_samples_to_use)
        
        for idx in tqdm(test_indices, desc="Processing CORD v2 test"):
            sample = v2_test[idx]
            image = sample["image"]
            
            filename = f"cord_v2_test_{idx}.png"
            image_path = images_dir / filename
            
            if not image_path.exists():
                image.save(image_path, "PNG")
            
            task_entry = {
                "image_urls": [filename],
                "dataset": "cord_v2",
                "split": "test",
                "receipt_type": "non_medical",
            }
            
            all_val_samples.append(task_entry)
    
    # Save task files
    print(f"\n{'='*80}")
    print(f"STEP {9 if include_v1 and include_v2 else (5 if include_v1 or include_v2 else 1)}: SAVING TASK FILES")
    print(f"{'='*80}")
    
    train_task_file = tasks_dir / "train_tasks.json"
    val_task_file = tasks_dir / "val_tasks.json"
    
    with open(train_task_file, 'w') as f:
        json.dump(all_train_samples, f, indent=2)
    
    with open(val_task_file, 'w') as f:
        json.dump(all_val_samples, f, indent=2)
    
    print(f"\n✅ Saved {len(all_train_samples):,} train samples to {train_task_file}")
    print(f"✅ Saved {len(all_val_samples):,} validation samples to {val_task_file}")
    
    # Show dataset distribution
    print(f"\n📊 Train dataset distribution:")
    train_datasets = {}
    for entry in all_train_samples:
        ds = entry['dataset']
        train_datasets[ds] = train_datasets.get(ds, 0) + 1
    
    for ds, count in sorted(train_datasets.items()):
        print(f"  - {ds}: {count:,} ({count/len(all_train_samples)*100:.1f}%)")
    
    print(f"\n📊 Validation dataset distribution:")
    val_datasets = {}
    for entry in all_val_samples:
        ds = entry['dataset']
        val_datasets[ds] = val_datasets.get(ds, 0) + 1
    
    for ds, count in sorted(val_datasets.items()):
        print(f"  - {ds}: {count:,} ({count/len(all_val_samples)*100:.1f}%)")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Total images saved: {len(list(images_dir.glob('*.png'))):,}")
    print(f"✅ Train task file: {train_task_file} ({len(all_train_samples):,} samples)")
    print(f"✅ Validation task file: {val_task_file} ({len(all_val_samples):,} samples)")
    print(f"\n✅ Done! Use these files as negative samples in your multi-task config.")
    print(f"\nExample usage in config:")
    print(f"")
    print(f"  # As negatives for drug extraction")
    print(f"  negative_samples:")
    print(f"    - train_tasks_path: {train_task_file}")
    print(f"      val_tasks_path: {val_task_file}")
    print(f"      images_root_dir: {images_dir}")
    print(f"      images_url_key: image_urls")
    print(f"      negative_extractor: cord_to_drug_extraction")
    print(f"      ratio: 1.0")
    print(f"")
    print(f"  # As negatives for prescription validation")
    print(f"  negative_samples:")
    print(f"    - train_tasks_path: {train_task_file}")
    print(f"      val_tasks_path: {val_task_file}")
    print(f"      images_root_dir: {images_dir}")
    print(f"      images_url_key: image_urls")
    print(f"      negative_extractor: cord_to_prescription_validation")
    print(f"      ratio: 1.0")
    print(f"")
    print(f"  # As hard negatives for receipt validation")
    print(f"  negative_samples:")
    print(f"    - train_tasks_path: {train_task_file}")
    print(f"      val_tasks_path: {val_task_file}")
    print(f"      images_root_dir: {images_dir}")
    print(f"      images_url_key: image_urls")
    print(f"      negative_extractor: cord_to_receipt_validation")
    print(f"      ratio: 1.0")


def main():
    parser = argparse.ArgumentParser(description="Prepare CORD negatives for multi-task learning")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="multi-task-data/CORD",
        help="Output directory for images and task files"
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=1.0,
        help="Ratio of dataset to sample (default: 1.0 = 100%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--v1-only",
        action="store_true",
        help="Only download and process CORD v1"
    )
    parser.add_argument(
        "--v2-only",
        action="store_true",
        help="Only download and process CORD v2"
    )
    
    args = parser.parse_args()
    
    # Determine which versions to include
    include_v1 = not args.v2_only
    include_v2 = not args.v1_only
    
    output_dir = Path(args.output_dir)
    download_and_prepare_cord(
        output_dir=output_dir,
        sample_ratio=args.sample_ratio,
        seed=args.seed,
        include_v1=include_v1,
        include_v2=include_v2,
    )


if __name__ == "__main__":
    main()
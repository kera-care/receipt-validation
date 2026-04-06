"""
Prepare COCO 2017 dataset subset as negative samples for multi-task learning.

This script:
1. Downloads COCO 2017 dataset from HuggingFace
2. Samples a subset of images from train and validation splits
3. Creates separate task files for train and validation
4. Saves images to multi-task-data/COCO/images/
5. These real-world object/scene images serve as clear negatives for all tasks:
   - Not prescriptions (object/scene photos)
   - Not receipts (no text/transaction info)
   - No drug names (no medical content)

Dataset info:
- COCO 2017: detection-datasets/coco
- Content: Real-world objects, people, scenes
- Resolution: Variable, average ~640px, up to 1024px+
- Format: PIL images with object detection annotations

Performance notes:
- First run: ~5-15 minutes depending on sample size
- Uses HuggingFace datasets library with caching
- Images are high resolution (640px+ average)
"""

import json
import random
from pathlib import Path
from typing import Dict, List
import argparse
from tqdm import tqdm
from datasets import load_dataset
import structlog

logger = structlog.get_logger(__name__)

def download_and_prepare_coco(
    output_dir: Path,
    train_samples: int = 5000,
    val_samples: int = 1000,
    seed: int = 42,
):
    """
    Download COCO 2017 subset and prepare negative samples.
    
    Args:
        output_dir: Base directory for outputs
        train_samples: Number of train samples to extract (default: 5000)
        val_samples: Number of validation samples to extract (default: 1000)
        seed: Random seed for reproducibility
    """
    logger.info("=" * 80)
    logger.info("PREPARING COCO 2017 NEGATIVES FOR MULTI-TASK LEARNING")
    logger.info("=" * 80)
    
    # Create output directories
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    tasks_dir = output_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Output directory: {output_dir}", output_dir=output_dir)
    logger.info("Images directory: {images_dir}", images_dir=images_dir)
    logger.info("Tasks directory: {tasks_dir}", tasks_dir=tasks_dir)
    
    all_train_samples = []
    all_val_samples = []
    
    # Download COCO 2017
    logger.info("=" * 80)
    logger.info("STEP 1: DOWNLOADING COCO 2017 DATASET")
    logger.info("=" * 80)
    
    logger.info("ℹ️  This may take a few minutes on first run (cached afterwards)...")
    
    # Load dataset - use the 2017 version
    coco_dataset = load_dataset("detection-datasets/coco", revision="main")
    coco_train = coco_dataset["train"]
    coco_val = coco_dataset["val"]
    
    logger.info("✓ Loaded COCO 2017 - Train: {train_count}, Validation: {val_count}", train_count=len(coco_train), val_count=len(coco_val))
    
    # Get category ID to string converter
    category_int2str = coco_train.features["objects"].feature["category"].int2str
    
    # Process train split
    logger.info("=" * 80)
    logger.info("STEP 2: PROCESSING TRAIN SPLIT")
    logger.info("=" * 80)
    
    actual_train_samples = min(train_samples, len(coco_train))
    logger.info("📊 Total COCO train images: {total}", total=len(coco_train))
    logger.info("📊 Sampling {samples} images", samples=actual_train_samples)
    
    random.seed(seed)
    train_indices = random.sample(range(len(coco_train)), actual_train_samples)
    
    for idx in tqdm(train_indices, desc="Processing COCO train"):
        sample = coco_train[idx]
        image = sample["image"]  # PIL image
        
        # Get image dimensions
        width, height = image.size
        
        # Extract object categories from annotations
        objects = sample.get("objects", {})
        category_ids = objects.get("category", []) if objects else []
        
        # Convert category IDs to names
        categories = [category_int2str(cat_id) for cat_id in category_ids]
        # Get unique category names
        unique_categories = list(set(categories)) if categories else []
        
        # Generate filename
        filename = f"coco_train_{idx}.jpg"
        image_path = images_dir / filename
        
        # Save image if not already saved (use JPEG for smaller size)
        if not image_path.exists():
            image.save(image_path, "JPEG", quality=95)
        
        # Create task entry with object categories
        task_entry = {
            "image_urls": [filename],
            "categories": unique_categories,  # e.g., ["person", "car", "dog"]
        }
        
        all_train_samples.append(task_entry)
    
    # Process validation split
    logger.info("=" * 80)
    logger.info("STEP 3: PROCESSING VALIDATION SPLIT")
    logger.info("=" * 80)
    
    actual_val_samples = min(val_samples, len(coco_val))
    logger.info("📊 Total COCO validation images: {total}", total=len(coco_val))
    logger.info("📊 Sampling {samples} images", samples=actual_val_samples)
    
    random.seed(seed + 1000)
    val_indices = random.sample(range(len(coco_val)), actual_val_samples)
    
    for idx in tqdm(val_indices, desc="Processing COCO validation"):
        sample = coco_val[idx]
        image = sample["image"]
        
        width, height = image.size
        
        # Extract object categories from annotations
        objects = sample.get("objects", {})
        category_ids = objects.get("category", []) if objects else []
        
        # Convert category IDs to names
        categories = [category_int2str(cat_id) for cat_id in category_ids]
        unique_categories = list(set(categories)) if categories else []
        
        filename = f"coco_val_{idx}.jpg"
        image_path = images_dir / filename
        
        if not image_path.exists():
            image.save(image_path, "JPEG", quality=95)
        
        task_entry = {
            "image_urls": [filename],
            "categories": unique_categories,  # e.g., ["person", "car", "dog"]
        }
        
        all_val_samples.append(task_entry)
    
    # Save task files
    logger.info("=" * 80)
    logger.info("STEP 4: SAVING TASK FILES")
    logger.info("=" * 80)
    
    train_task_file = tasks_dir / "train_tasks.json"
    val_task_file = tasks_dir / "val_tasks.json"
    
    with open(train_task_file, 'w') as f:
        json.dump(all_train_samples, f, indent=2)
    
    with open(val_task_file, 'w') as f:
        json.dump(all_val_samples, f, indent=2)
    
    logger.info("✅ Saved {train_count} train samples to {train_file}", train_count=len(all_train_samples), train_file=train_task_file)
    logger.info("✅ Saved {val_count} validation samples to {val_file}", val_count=len(all_val_samples), val_file=val_task_file)
    
    # Show category statistics
    logger.info("📊 Object category statistics:")
    all_categories = []
    for sample in all_train_samples + all_val_samples:
        all_categories.extend(sample.get("categories", []))
    
    if all_categories:
        from collections import Counter
        category_counts = Counter(all_categories)
        top_10 = category_counts.most_common(10)
        logger.info("  - Total unique categories: {total}", total=len(category_counts))
        logger.info("  - Top 10 categories:")
        for cat, count in top_10:
            logger.info("    • {category}: {count}", category=cat, count=count)
    
    # Summary
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info("✅ Total images saved: {total}", total=len(list(images_dir.glob('*.jpg'))))
    logger.info("✅ Train task file: {train_file} ({train_count} samples)", train_file=train_task_file, train_count=len(all_train_samples))
    logger.info("✅ Validation task file: {val_file} ({val_count} samples)", val_file=val_task_file, val_count=len(all_val_samples))
    logger.info("✅ Done! Use these files as negative samples in your multi-task config.")
    logger.info("📄 Example usage in config:")
    logger.info("")
    logger.info("  # As negatives for drug extraction")
    logger.info("  negative_samples:")
    logger.info("    - train_tasks_path: {train_file}", train_file=train_task_file)
    logger.info("      val_tasks_path: {val_file}", val_file=val_task_file)
    logger.info("      images_root_dir: {images_dir}", images_dir=images_dir)
    logger.info("      images_url_key: image_urls")
    logger.info("      negative_extractor: coco_to_drug_extraction")
    logger.info("      ratio: 1.0")
    logger.info("")
    logger.info("  # As negatives for prescription validation")
    logger.info("  negative_samples:")
    logger.info("    - train_tasks_path: {train_file}", train_file=train_task_file)
    logger.info("      val_tasks_path: {val_file}", val_file=val_task_file)
    logger.info("      images_root_dir: {images_dir}", images_dir=images_dir)
    logger.info("      images_url_key: image_urls")
    logger.info("      negative_extractor: coco_to_prescription_validation")
    logger.info("      ratio: 1.0")
    logger.info("")
    logger.info("  # As negatives for receipt validation")
    logger.info("  negative_samples:")
    logger.info("    - train_tasks_path: {train_file}", train_file=train_task_file)
    logger.info("      val_tasks_path: {val_file}", val_file=val_task_file)
    logger.info("      images_root_dir: {images_dir}", images_dir=images_dir)
    logger.info("      images_url_key: image_urls")
    logger.info("      negative_extractor: coco_to_receipt_validation")
    logger.info("      ratio: 1.0")


def main():
    parser = argparse.ArgumentParser(description="Prepare COCO 2017 negatives for multi-task learning")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="multi-task-data/COCO",
        help="Output directory for images and task files"
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=5000,
        help="Number of train samples to extract (default: 5000)"
    )
    parser.add_argument(
        "--val_samples",
        type=int,
        default=1000,
        help="Number of validation samples to extract (default: 1000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    download_and_prepare_coco(
        output_dir=output_dir,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
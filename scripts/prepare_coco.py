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
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from tqdm import tqdm
from datasets import load_dataset
import structlog

logger = structlog.get_logger(__name__)

def _save_coco_sample(
    args: tuple[int, object, Path, str, object],
) -> dict:
    """Save a single COCO sample as JPEG and return its task entry."""
    i, sample, images_dir, filename, category_int2str = args
    image = sample["image"]
    image_path = images_dir / filename
    if not image_path.exists():
        image.convert("RGB").save(image_path, "JPEG", quality=90)
    objects = sample.get("objects") or {}
    category_ids = objects.get("category") or []
    unique_categories = list({category_int2str(c) for c in category_ids})
    return {"image_urls": [filename], "categories": unique_categories}


def _process_split(
    dataset_split,
    images_dir: Path,
    filename_prefix: str,
    n_samples: int,
    seed: int,
    category_int2str,
    num_workers: int,
) -> list[dict]:
    """Shuffle+select (sequential Arrow reads) then parallel JPEG saves."""
    n = min(n_samples, len(dataset_split))
    subset = dataset_split.shuffle(seed=seed).select(range(n))
    work = [
        (i, subset[i], images_dir, f"{filename_prefix}_{i}.jpg", category_int2str)
        for i in range(n)
    ]
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_save_coco_sample, item): item for item in work}
        for fut in tqdm(as_completed(futures), total=n, desc=f"Processing {filename_prefix}"):
            results.append(fut.result())
    return results


def download_and_prepare_coco(
    output_dir: Path,
    train_samples: int = 5000,
    val_samples: int = 1000,
    seed: int = 42,
    num_workers: int = 8,
):
    """
    Download COCO 2017 subset and prepare negative samples.

    Args:
        output_dir: Base directory for outputs
        train_samples: Number of train samples to extract (default: 5000)
        val_samples: Number of validation samples to extract (default: 1000)
        seed: Random seed for reproducibility
        num_workers: Number of parallel threads for image saving (default: 8)
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
    
    logger.info("downloading COCO 2017 (cached after first run)")
    coco_dataset = load_dataset("detection-datasets/coco", revision="main")
    coco_train = coco_dataset["train"]
    coco_val = coco_dataset["val"]
    logger.info(
        "loaded COCO 2017",
        train_count=len(coco_train),
        val_count=len(coco_val),
    )

    category_int2str = coco_train.features["objects"].feature["category"].int2str

    all_train_samples = _process_split(
        coco_train, images_dir, "coco_train", train_samples, seed, category_int2str, num_workers
    )
    all_val_samples = _process_split(
        coco_val, images_dir, "coco_val", val_samples, seed + 1000, category_int2str, num_workers
    )
    
    train_task_file = tasks_dir / "train_tasks.json"
    val_task_file = tasks_dir / "val_tasks.json"

    with open(train_task_file, "w") as f:
        json.dump(all_train_samples, f, indent=2)
    with open(val_task_file, "w") as f:
        json.dump(all_val_samples, f, indent=2)

    logger.info(
        "saved tasks",
        train_count=len(all_train_samples),
        val_count=len(all_val_samples),
        total_images=len(list(images_dir.glob("*.jpg"))),
    )


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
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel threads for image saving (default: 8)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    download_and_prepare_coco(
        output_dir=output_dir,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        seed=args.seed,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
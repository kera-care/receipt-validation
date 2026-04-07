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
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from tqdm import tqdm
from datasets import load_dataset
import structlog


logger = structlog.get_logger(__name__)


def _save_sample(args: tuple[int, object, Path, str, str]) -> dict:
    """Fetch one row from the Arrow dataset, save as JPEG, return task entry.

    TODO: _save_sample and prepare_coco._save_coco_sample are nearly identical
    (RGB convert → JPEG save → return dict). Unify into a shared helper in a
    follow-up refactor.
    """
    i, subset, images_dir, filename_stem, split_name = args
    sample = subset[i]  # decode only this row, inside the worker thread
    image = sample["image"]
    filename = f"{filename_stem}.jpg"
    image_path = images_dir / filename
    if not image_path.exists():
        # quality=90 — intentional trade-off: ~20-30% smaller files vs original
        # PNG encoding with no visually perceptible difference at receipt
        # resolution. Update both scripts together if you want to change it.
        image.convert("RGB").save(image_path, "JPEG", quality=90)
    return {
        "image_urls": [filename],
        "dataset": filename_stem.rsplit("_", 2)[0],  # e.g. cord_v1
        "split": split_name,
        "receipt_type": "non_medical",
    }


def _process_split(
    dataset_split,
    images_dir: Path,
    filename_prefix: str,
    split_name: str,
    sample_ratio: float,
    seed: int,
    num_workers: int = 8,
) -> list[dict]:
    """
    Process one HuggingFace dataset split efficiently.

    Uses shuffle+select (sequential Arrow access) instead of random index
    look-ups, and parallelises JPEG writes across ``num_workers`` threads.
    Each worker decodes only its own row so peak RAM stays proportional to
    num_workers rather than the full sample count.
    """
    n = max(1, int(len(dataset_split) * sample_ratio))
    subset = dataset_split.shuffle(seed=seed).select(range(n))

    # Pass the dataset + index, not the decoded sample.
    tasks: list[dict] = []
    work = [
        (i, subset, images_dir, f"{filename_prefix}_{split_name}_{i}", split_name)
        for i in range(n)
    ]
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_save_sample, item): item for item in work}
        for fut in tqdm(as_completed(futures), total=n, desc=f"Processing {filename_prefix} {split_name}"):
            tasks.append(fut.result())
    return tasks


def download_and_prepare_cord(
    output_dir: Path,
    sample_ratio: float = 1.0,
    seed: int = 42,
    include_v1: bool = True,
    include_v2: bool = True,
    num_workers: int = 8,
):
    """
    Download CORD datasets and prepare negative samples.

    Args:
        output_dir: Base directory for outputs
        sample_ratio: Ratio of dataset to use (default 1.0 = 100%)
        seed: Random seed for reproducibility
        include_v1: Whether to include CORD v1
        include_v2: Whether to include CORD v2
        num_workers: Number of parallel threads for image saving
    """
    logger.info("=" * 80)
    logger.info("PREPARING CORD NEGATIVES FOR MULTI-TASK LEARNING")
    logger.info("=" * 80)
    
    if not include_v1 and not include_v2:
        raise ValueError("At least one of CORD v1 or v2 must be included")
    
    # Create output directories
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    tasks_dir = output_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("📁 Output directory: {output_dir}", output_dir=output_dir)
    logger.info("📁 Images directory: {images_dir}", images_dir=images_dir)
    logger.info("📁 Tasks directory: {tasks_dir}", tasks_dir=tasks_dir)
    
    all_train_samples: list[dict] = []
    all_val_samples: list[dict] = []

    # Seed assignment per version and split:
    #   v1: train=seed,       validation=seed+10,  test=seed+20
    #   v2: train=seed+100,   validation=seed+110, test=seed+120
    #
    # NOTE: these seeds differ from the original sequential implementation
    # (v1: seed/seed+1000/seed+2000, v2: seed+2000/seed+2000/seed+3000 — the
    # duplicate seed+2000 for v2 train and v1 test was a pre-existing bug).
    # If training data was already prepared with the old script, re-running
    # this version will produce a different sample selection. Pin the dataset
    # to a snapshot or keep the old seeds if reproducibility across runs is
    # required.
    versions: list[tuple[str, str, int]] = []
    if include_v1:
        versions.append(("naver-clova-ix/cord-v1", "cord_v1", seed))
    if include_v2:
        versions.append(("naver-clova-ix/cord-v2", "cord_v2", seed + 100))

    for hf_id, prefix, base_seed in versions:
        logger.info("Downloading {hf_id}", hf_id=hf_id)
        ds = load_dataset(hf_id)
        logger.info(
            "Loaded {prefix} — train: {tr}, validation: {va}, test: {te}",
            prefix=prefix,
            tr=len(ds["train"]),
            va=len(ds["validation"]),
            te=len(ds["test"]),
        )

        # train split → train tasks
        all_train_samples.extend(
            _process_split(ds["train"], images_dir, prefix, "train",
                           sample_ratio, base_seed, num_workers)
        )
        # validation + test splits → val tasks
        all_val_samples.extend(
            _process_split(ds["validation"], images_dir, prefix, "validation",
                           sample_ratio, base_seed + 10, num_workers)
        )
        all_val_samples.extend(
            _process_split(ds["test"], images_dir, prefix, "test",
                           sample_ratio, base_seed + 20, num_workers)
        )
    
    # Save task files
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
        train_file=str(train_task_file),
        val_file=str(val_task_file),
        total_images=len(list(images_dir.glob("*.jpg"))),
    )


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
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel threads for image saving (default: 8)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    download_and_prepare_cord(
        output_dir=output_dir,
        sample_ratio=args.sample_ratio,
        seed=args.seed,
        include_v1=not args.v2_only,
        include_v2=not args.v1_only,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
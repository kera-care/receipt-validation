"""
Merge multiple prepared datasets into a single unified dataset for multi-task training.

This script:
1. Loads task files from CORD, COCO, DocLayNet, and Kera prescription datasets
2. Resolves relative image_urls to absolute image_paths for each source
3. Adds an image_paths key with full absolute paths to each task entry
4. Merges all train and validation splits separately
5. Writes merged train_tasks.json and val_tasks.json to --output_dir

Expected directory structure per upstream dataset:
    {dataset_dir}/
        images/        # actual image files
        tasks/
            train_tasks.json
            val_tasks.json

Kera prescription splits use prescription_image_urls instead of image_urls.

Output task entry schema (merged):
    {
        "image_urls":  [...],        # original relative urls (preserved)
        "image_paths": [...],        # NEW: absolute paths to image files
        "source":      "cord|coco|doclaynet|kera_prescription",
        ...                          # any dataset-specific fields are preserved
    }
"""

import json
import argparse
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_task_file(path: Path) -> list[dict[str, Any]]:
    """Load a JSON task file, returning an empty list if the file is missing."""
    if not path.exists():
        log.warning("task_file_not_found", path=str(path))
        return []
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")
    return data


def _load_jsonl_file(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL task file (one JSON object per line), returning an empty list if missing."""
    if not path.exists():
        log.warning("task_file_not_found", path=str(path))
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                log.warning("jsonl_parse_error", path=str(path), line=line_no, error=str(exc))
    return records


def _resolve_image_paths(
    tasks: list[dict[str, Any]],
    images_dir: Path,
    image_urls_key: str,
    source_label: str,
) -> list[dict[str, Any]]:
    """
    Add image_paths (absolute) to each task entry and annotate with source.

    Args:
        tasks:           List of task dicts loaded from a task file.
        images_dir:      Directory that contains the images referenced by image_urls_key.
        image_urls_key:  The key in each task dict holding relative image filenames.
        source_label:    Value written to the "source" field on each entry.

    Returns:
        New list of task dicts with image_paths and source fields added.
    """
    resolved: list[dict[str, Any]] = []
    missing = 0
    for task in tasks:
        relative_urls: list[str] = task.get(image_urls_key, [])
        abs_paths: list[str] = []
        for rel in relative_urls:
            full = images_dir / rel
            if not full.exists():
                log.warning(
                    "image_not_found",
                    source=source_label,
                    path=str(full),
                )
                missing += 1
            abs_paths.append(str(full.resolve()))
        entry = {**task, "image_paths": abs_paths}
        if "source" not in entry:
            entry["source"] = source_label
        resolved.append(entry)

    if missing:
        log.warning(
            "missing_images_summary",
            source=source_label,
            count=missing,
        )
    return resolved


def _load_dataset_split(
    dataset_dir: Path | None,
    split: str,
    image_urls_key: str,
    source_label: str,
    tasks_subdir: str = "tasks",
    images_subdir: str = "images",
) -> list[dict[str, Any]]:
    """
    Load one split (train/val) from a single dataset directory.

    Args:
        dataset_dir:    Root dir of the dataset (e.g. /data/CORD).
        split:          "train" or "val".
        image_urls_key: Key in task entries that holds relative image filenames.
        source_label:   Identifier written to the "source" field.
        tasks_subdir:   Sub-directory containing task JSON files (default: "tasks").
        images_subdir:  Sub-directory containing image files (default: "images").

    Returns:
        List of enriched task dicts, or [] if dataset_dir is None / files are absent.
    """
    if dataset_dir is None:
        return []

    task_file = dataset_dir / tasks_subdir / f"{split}_tasks.json"
    images_dir = dataset_dir / images_subdir

    tasks = _load_task_file(task_file)
    log.info(
        "loaded_split",
        source=source_label,
        split=split,
        count=len(tasks),
    )
    return _resolve_image_paths(tasks, images_dir, image_urls_key, source_label)


# ---------------------------------------------------------------------------
# Per-dataset loaders
# ---------------------------------------------------------------------------

def load_cord(cord_dir: Path | None, split: str) -> list[dict[str, Any]]:
    return _load_dataset_split(
        cord_dir,
        split=split,
        image_urls_key="image_urls",
        source_label="cord",
    )


def load_coco(coco_dir: Path | None, split: str) -> list[dict[str, Any]]:
    return _load_dataset_split(
        coco_dir,
        split=split,
        image_urls_key="image_urls",
        source_label="coco",
    )


def load_doclaynet(doclaynet_dir: Path | None, split: str) -> list[dict[str, Any]]:
    return _load_dataset_split(
        doclaynet_dir,
        split=split,
        image_urls_key="image_urls",
        source_label="doclaynet",
    )


def load_kera_receipts(
    splits_dir: Path | None,
    images_dir: Path | None,
    split: str,
) -> list[dict[str, Any]]:
    """
    Load Kera receipts tasks produced by prepare_kera_receipts.py.

    Output files from that script are JSONL and named:
        train.jsonl, validation.jsonl, test.jsonl

    The merge pipeline uses split names "train" and "val", so "val" is
    mapped to "validation.jsonl" here.

    Args:
        splits_dir:  Directory that contains train.jsonl / validation.jsonl.
        images_dir:  Directory that contains the receipts images.
        split:       "train" or "val".
    """
    if splits_dir is None or images_dir is None:
        return []

    # Map merge-pipeline split names to the filenames written by prepare_kera_receipts.py
    filename_map = {"train": "train.jsonl", "val": "validation.jsonl"}
    filename = filename_map.get(split, f"{split}.jsonl")
    task_file = splits_dir / filename

    tasks = _load_jsonl_file(task_file)
    log.info(
        "loaded_split",
        source="kera_receipt",
        split=split,
        count=len(tasks),
    )
    return _resolve_image_paths(
        tasks,
        images_dir=images_dir,
        image_urls_key="image_urls",
        source_label="kera_receipt",
    )


# ---------------------------------------------------------------------------
# Merge & write
# ---------------------------------------------------------------------------

def merge_and_write(
    cord_dir: Path | None,
    coco_dir: Path | None,
    doclaynet_dir: Path | None,
    kera_receipt_splits_dir: Path | None,
    kera_receipt_dir: Path | None,
    output_dir: Path,
) -> None:
    """
    Merge all datasets and write unified task files to output_dir.

    Args:
        cord_dir:                        Root of CORD dataset (or None to skip).
        coco_dir:                        Root of COCO dataset (or None to skip).
        doclaynet_dir:                   Root of DocLayNet dataset (or None to skip).
        kera_receipt_splits_dir:         Directory with train/val task JSONs for Kera receipt.
        kera_receipt_dir:                Directory containing Kera receipts images.
        output_dir:                      Destination for merged train/val task files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        log.info("merging_split", split=split)

        merged: list[dict[str, Any]] = []
        merged.extend(load_cord(cord_dir, split))
        merged.extend(load_coco(coco_dir, split))
        merged.extend(load_doclaynet(doclaynet_dir, split))
        merged.extend(
            load_kera_receipts(
                kera_receipt_splits_dir,
                kera_receipt_dir,
                split,
            )
        )

        out_path = output_dir / f"{split}_tasks.json"
        with out_path.open("w") as f:
            json.dump(merged, f, indent=2)

        log.info(
            "wrote_merged_split",
            split=split,
            count=len(merged),
            path=str(out_path),
        )

    log.info("merge_complete", output_dir=str(output_dir))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge CORD, COCO, DocLayNet, and Kera prescription datasets"
    )
    parser.add_argument(
        "--cord_dir",
        type=str,
        default=None,
        help="Root directory of the prepared CORD dataset (contains images/ and tasks/)",
    )
    parser.add_argument(
        "--coco_dir",
        type=str,
        default=None,
        help="Root directory of the prepared COCO dataset (contains images/ and tasks/)",
    )
    parser.add_argument(
        "--doclaynet_dir",
        type=str,
        default=None,
        help="Root directory of the prepared DocLayNet dataset (contains images/ and tasks/)",
    )
    parser.add_argument(
        "--kera_receipt_dir",
        type=str,
        default=None,
        help="Directory containing Kera receipt images",
    )
    parser.add_argument(
        "--kera_receipt_splits_dir",
        type=str,
        default=None,
        help="Directory containing Kera receipts train_tasks.json / val_tasks.json split files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for merged train_tasks.json and val_tasks.json",
    )
    args = parser.parse_args()

    def _to_path(val: str | None) -> Path | None:
        return Path(val) if val is not None else None

    cord_dir = _to_path(args.cord_dir)
    coco_dir = _to_path(args.coco_dir)
    doclaynet_dir = _to_path(args.doclaynet_dir)
    kera_receipt_dir = _to_path(args.kera_receipt_dir)
    kera_receipt_splits_dir = _to_path(args.kera_receipt_splits_dir)
    output_dir = Path(args.output_dir)

    # Validate that at least one source is provided
    if all(
        d is None
        for d in [cord_dir, coco_dir, doclaynet_dir, kera_receipt_splits_dir]
    ):
        parser.error(
            "At least one dataset source must be specified "
            "(--cord_dir, --coco_dir, --doclaynet_dir, or --kera_receipt_splits_dir)"
        )

    # Validate kera args come in pairs
    if (kera_receipt_splits_dir is None) != (kera_receipt_dir is None):
        parser.error(
            "--kera_receipt_dir and --kera_receipt_splits_dir "
            "must both be provided or both omitted"
        )

    merge_and_write(
        cord_dir=cord_dir,
        coco_dir=coco_dir,
        doclaynet_dir=doclaynet_dir,
        kera_receipt_splits_dir=kera_receipt_splits_dir,
        kera_receipt_dir=kera_receipt_dir,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()

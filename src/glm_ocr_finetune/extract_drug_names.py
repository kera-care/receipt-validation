"""
Extract and save all unique normalized drug names from one or more task files.

Usage:
    python -m glm_ocr_finetune.extract_drug_names \
        --task_files dataset/train_tasks.json dataset/dev_tasks.json \
        --output_path outputs/all_drug_names.json
"""

import argparse
import json
import os

import structlog

from glm_ocr_finetune.data.utils import normalize_drug_name

logger = structlog.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract all unique normalized drug names from task files"
    )
    parser.add_argument(
        "--task_files",
        type=str,
        nargs="+",
        required=True,
        help="One or more task JSON files (e.g. dataset/train_tasks.json dataset/dev_tasks.json)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/all_drug_names.json",
        help="Path to save the extracted drug names JSON",
    )
    return parser.parse_args()


def extract_drug_names(task_files: list[str]) -> dict:
    """Extract all unique normalized drug names from the given task files.

    Returns a dict with:
        - drug_names: sorted list of unique normalized drug names
        - total_tasks: total number of tasks across all files
        - per_file_stats: per-file task count and unique drug name count
    """
    all_names: set[str] = set()
    total_tasks = 0
    per_file_stats: list[dict] = []

    for path in task_files:
        if not os.path.exists(path):
            logger.error("Task file not found", path=path)
            raise FileNotFoundError(f"Task file not found: {path}")

        with open(path, "r") as f:
            tasks = json.load(f)

        file_names: set[str] = set()
        for task in tasks:
            for name in task.get("verified_drug_names", []):
                normalized = normalize_drug_name(name)
                if normalized and len(normalized) > 2:
                    file_names.add(normalized)
                    all_names.add(normalized)
                elif normalized and len(normalized) <= 2:
                    logger.warning(
                        "Skipping short drug name after normalization",
                        original_name=name,
                        normalized_name=normalized,
                    )

        total_tasks += len(tasks)
        per_file_stats.append({
            "file": path,
            "num_tasks": len(tasks),
            "num_unique_drug_names": len(file_names),
        })
        logger.info(
            "Processed file",
            file=path,
            tasks=len(tasks),
            unique_drug_names=len(file_names),
        )

    sorted_names = sorted(all_names)
    return {
        "drug_names": sorted_names,
        "total_unique_drug_names": len(sorted_names),
        "total_tasks": total_tasks,
        "per_file_stats": per_file_stats,
    }


def main():
    args = parse_args()
    result = extract_drug_names(args.task_files)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(
        "Drug names saved",
        output_path=args.output_path,
        total_unique=result["total_unique_drug_names"],
        total_tasks=result["total_tasks"],
    )


if __name__ == "__main__":
    main()

"""
Extract and analyze receipt dataset statistics from one or more task files.

Usage:
    python -m glm_ocr_finetune.extract_receipt_stats \
        --task_files dataset/train_tasks.json dataset/val_tasks.json \
        --output_path outputs/receipt_stats.json
"""

import argparse
import json
import os
from collections import Counter
from datetime import datetime

import structlog

from glm_ocr_finetune.data.utils import normalize_amount

logger = structlog.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract statistics from receipt validation task files"
    )
    parser.add_argument(
        "--task_files",
        type=str,
        nargs="+",
        required=True,
        help="One or more task JSON files (e.g. dataset/train_tasks.json dataset/val_tasks.json)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/receipt_stats.json",
        help="Path to save the extracted statistics JSON",
    )
    return parser.parse_args()


def extract_receipt_stats(task_files: list[str]) -> dict:
    """Extract comprehensive statistics from receipt validation task files.

    Returns a dict with:
        - total_tasks: total number of tasks across all files
        - per_file_stats: per-file breakdown
        - receipt_classification: distribution of is_health_receipt
        - amount_stats: min/max/mean/median amounts, null count
        - date_stats: date range, null count, invalid dates
        - field_presence: count of non-null values for total_amount, date,
          patient_name, provider_info, proof_of_payment
        - invalid_dates: up to 20 records with malformed date strings
    """
    all_amounts: list[float] = []
    all_dates: list[str] = []
    total_tasks = 0
    per_file_stats: list[dict] = []
    
    # Counters
    is_health_receipt_counter = Counter()

    field_presence = {
        "total_amount": 0,
        "date": 0,
        "patient_name": 0,
        "provider_info": 0,
        "proof_of_payment": 0,
    }

    invalid_dates: list[dict] = []

    for path in task_files:
        if not os.path.exists(path):
            logger.error("Task file not found", path=path)
            raise FileNotFoundError(f"Task file not found: {path}")

        with open(path, "r") as f:
            tasks = json.load(f)

        file_stats = {
            "file": path,
            "num_tasks": len(tasks),
            "health_receipts": 0,
            "non_health_receipts": 0,
        }

        for task in tasks:
            # Classification
            is_health = task.get("is_health_receipt")
            is_health_receipt_counter[is_health] += 1
            if is_health:
                file_stats["health_receipts"] += 1
            else:
                file_stats["non_health_receipts"] += 1
            
            # Amount
            raw_amount = task.get("total_amount")
            if raw_amount is not None:
                field_presence["total_amount"] += 1
                normalized = normalize_amount(raw_amount)
                if normalized is not None:
                    all_amounts.append(normalized)
            
            # Date
            date_str = task.get("date")
            if date_str:
                field_presence["date"] += 1
                all_dates.append(date_str)
                # Validate date format
                try:
                    datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    invalid_dates.append({
                        "transaction_id": task.get("transaction_id"),
                        "date": date_str,
                        "file": path,
                    })
            
            # Other text fields
            if task.get("patient_name"):
                field_presence["patient_name"] += 1
            if task.get("provider_info"):
                field_presence["provider_info"] += 1
            if task.get("proof_of_payment"):
                field_presence["proof_of_payment"] += 1

        total_tasks += len(tasks)
        per_file_stats.append(file_stats)
        logger.info(
            "Processed file",
            file=path,
            tasks=len(tasks),
            health_receipts=file_stats["health_receipts"],
        )

    # Compute amount statistics
    amount_stats = {}
    if all_amounts:
        all_amounts.sort()
        amount_stats = {
            "min": min(all_amounts),
            "max": max(all_amounts),
            "mean": sum(all_amounts) / len(all_amounts),
            "median": all_amounts[len(all_amounts) // 2],
            "count": len(all_amounts),
            "null_count": total_tasks - len(all_amounts),
        }
    else:
        amount_stats = {"count": 0, "null_count": total_tasks}
    
    # Date statistics
    date_stats = {}
    if all_dates:
        valid_dates = sorted([d for d in all_dates if d])
        date_stats = {
            "earliest": valid_dates[0] if valid_dates else None,
            "latest": valid_dates[-1] if valid_dates else None,
            "count": len(valid_dates),
            "null_count": total_tasks - len(all_dates),
            "invalid_count": len(invalid_dates),
        }
    else:
        date_stats = {"count": 0, "null_count": total_tasks, "invalid_count": 0}
    
    return {
        "total_tasks": total_tasks,
        "per_file_stats": per_file_stats,
        "receipt_classification": {
            "health_receipts": is_health_receipt_counter.get(True, 0),
            "non_health_receipts": is_health_receipt_counter.get(False, 0),
            "null_classification": is_health_receipt_counter.get(None, 0),
        },
        "amount_stats": amount_stats,
        "date_stats": date_stats,
        "field_presence": {
            k: {"count": v, "percentage": round(100 * v / total_tasks, 1)}
            for k, v in field_presence.items()
        },
        "invalid_dates": invalid_dates[:20],
    }


def main():
    args = parse_args()
    result = extract_receipt_stats(args.task_files)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(
        "Receipt statistics saved",
        output_path=args.output_path,
        total_tasks=result["total_tasks"],
        health_receipts=result["receipt_classification"]["health_receipts"],
    )


if __name__ == "__main__":
    main()

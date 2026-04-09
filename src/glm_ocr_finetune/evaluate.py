"""
Evaluate receipt validation and amount extraction predictions against ground-truth labels.

Multi-task evaluation metrics for Senegalese healthcare receipt understanding:

Evaluation metrics by task:
    1. **Classification (is_health_receipt)** — binary classification metrics:
       accuracy, precision, recall, F1
    
    2. **Amount extraction (total_amount)** — normalized CFA franc amounts:
       - Exact match: amounts match exactly (after normalization)
       - Tolerance match: amounts within --amount_tolerance % (default: 1%)
       - Error list: first 50 mismatches with label/predicted amounts
    
    3. **Date extraction (date)** — ISO format (YYYY-MM-DD):
       - Exact string match accuracy
    
    4. **Entity presence detection** — for patient_name, provider_info, proof_of_payment:
       - Binary classification: is field present (non-null, non-empty)?
       - Metrics: accuracy, precision, recall, F1

Outputs:
    - ``receipt_evaluation_results.json`` — aggregate metrics for all tasks
    - Per-sample errors included for amount mismatches (first 50)

Usage:
    python -m glm_ocr_finetune.evaluate \
        --inference_path outputs/inference_results.json \
        --output_path outputs/receipt_evaluation_results.json \
        [--amount_tolerance 0.01]

Input format (inference_results.json):
    [
        {
            "transaction_id": "abc123",
            "labels": {
                "is_health_receipt": true,
                "total_amount": "15000",
                "date": "2026-02-05",
                "patient_name": "John Doe",
                ...
            },
            "predictions": {
                "is_health_receipt": true,
                "total_amount": "15000.00",
                ...
            }
        },
        ...
    ]
"""

import argparse
import json
import os

import structlog

from glm_ocr_finetune.data.utils import normalize_amount

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Core evaluation helpers
# ---------------------------------------------------------------------------

def _safe_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_classification(results: list[dict], field: str) -> dict:
    """Binary classification metrics (accuracy, precision, recall, F1) for any boolean field."""
    tp = fp = fn = tn = 0
    for r in results:
        pred = bool(r["predictions"].get(field, False))
        label = bool(r["labels"].get(field, False))
        if pred and label: tp += 1
        elif pred and not label: fp += 1
        elif not pred and label: fn += 1
        else: tn += 1
    prec, rec, f1 = _safe_prf(tp, fp, fn)
    return {"accuracy": (tp + tn) / len(results) if results else 0,
            "precision": prec, "recall": rec, "f1": f1}

def evaluate_amount(results: list[dict], tolerance_pct: float = 0.01) -> dict:
    """Amount extraction: exact match + percentage tolerance."""
    total = exact = within_tol = 0
    errors = []
    for r in results:
        label_amt = normalize_amount(r["labels"].get("total_amount"))
        pred_amt = normalize_amount(r["predictions"].get("total_amount"))
        if label_amt is None and pred_amt is None:
            continue
        total += 1
        if label_amt is not None and pred_amt is not None:
            if label_amt == pred_amt:
                exact += 1
                within_tol += 1
            elif label_amt > 0 and abs(label_amt - pred_amt) / label_amt <= tolerance_pct:
                within_tol += 1
            else:
                errors.append({"id": r.get("transaction_id"),
                               "label": label_amt, "pred": pred_amt})
        else:
            errors.append({"id": r.get("transaction_id"),
                           "label": label_amt, "pred": pred_amt})
    return {"total": total,
            "exact_match": exact / total if total else 0,
            "tolerance_match": within_tol / total if total else 0,
            "errors": errors[:50]}


def evaluate_date(results: list[dict]) -> dict:
    """Date extraction: exact ISO match."""
    total = correct = 0
    for r in results:
        label = r["labels"].get("date")
        pred = r["predictions"].get("date")
        if label is None and pred is None:
            continue
        total += 1
        if label == pred:
            correct += 1
    return {"total": total, "accuracy": correct / total if total else 0}


def evaluate_string_field(results: list[dict], field: str) -> dict:
    """Presence detection for patient_name, provider_info, proof_of_payment."""
    tp = fp = fn = tn = 0
    for r in results:
        label = r["labels"].get(field)
        pred = r["predictions"].get(field)
        label_present = label is not None and str(label).strip() != ""
        pred_present = pred is not None and str(pred).strip() != ""
        if pred_present and label_present: tp += 1
        elif pred_present and not label_present: fp += 1
        elif not pred_present and label_present: fn += 1
        else: tn += 1
    prec, rec, f1 = _safe_prf(tp, fp, fn)
    return {"accuracy": (tp + tn) / len(results) if results else 0,
            "precision": prec, "recall": rec, "f1": f1}

# ---------------------------------------------------------------------------
# Main and CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate receipt validation predictions")
    parser.add_argument("--inference_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="outputs/receipt_evaluation_results.json")
    parser.add_argument("--amount_tolerance", type=float, default=0.01)
    args = parser.parse_args()

    with open(args.inference_path) as f:
        results = json.load(f)

    report = {
        "num_samples": len(results),
        "is_health_receipt": evaluate_classification(results, "is_health_receipt"),
        "total_amount": evaluate_amount(results, args.amount_tolerance),
        "date": evaluate_date(results),
        "patient_name": evaluate_string_field(results, "patient_name"),
        "provider_info": evaluate_string_field(results, "provider_info"),
        "proof_of_payment": evaluate_string_field(results, "proof_of_payment"),
    }

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Evaluation complete", output_path=args.output_path, **{
        k: v for k, v in report.items() if k != "total_amount" or not isinstance(v, dict)
    })


if __name__ == "__main__":
    main()

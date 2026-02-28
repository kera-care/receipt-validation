"""
Evaluate drug name extraction predictions against ground-truth labels.

Two evaluation modes:
    1. **Exact match** — a predicted name must match a label character-for-character
       (after normalization). Even a single extra/missing character counts as an error.
    2. **Fuzzy match** — each predicted name is matched to its closest label using
       Jaro-Winkler similarity. A match is accepted when similarity >= threshold (0.7).

Metrics reported for each mode:
    - Precision, Recall, F1 (micro-averaged across all samples)
    - Per-sample breakdown saved to the output JSON

Usage:
    python -m glm_ocr_finetune.evaluate \
        --inference_path outputs/inference_results.json \
        --output_path outputs/evaluation_results.json \
        [--jaro_winkler_threshold 0.7]
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass, field

import structlog
from thefuzz import fuzz

from glm_ocr_finetune.data.utils import normalize_drug_name

logger = structlog.get_logger(__name__)


def jaro_winkler_similarity(s1: str, s2: str) -> float:
    """Return a similarity score in [0, 1] using thefuzz's token ratio.

    Uses Jaro-Winkler-like partial/token matching via thefuzz.
    The score is divided by 100 to normalise from thefuzz's 0-100 range.
    """
    return fuzz.ratio(s1, s2) / 100.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SampleResult:
    transaction_id: str
    labels: list[str]
    predictions: list[str]
    # Exact
    exact_tp: int = 0
    exact_fp: int = 0
    exact_fn: int = 0
    exact_precision: float = 0.0
    exact_recall: float = 0.0
    exact_f1: float = 0.0
    # Fuzzy
    fuzzy_tp: int = 0
    fuzzy_fp: int = 0
    fuzzy_fn: int = 0
    fuzzy_precision: float = 0.0
    fuzzy_recall: float = 0.0
    fuzzy_f1: float = 0.0
    fuzzy_matches: list[dict] = field(default_factory=list)


@dataclass
class AggregateMetrics:
    num_samples: int = 0
    # Exact
    exact_tp: int = 0
    exact_fp: int = 0
    exact_fn: int = 0
    exact_precision: float = 0.0
    exact_recall: float = 0.0
    exact_f1: float = 0.0
    # Fuzzy
    fuzzy_tp: int = 0
    fuzzy_fp: int = 0
    fuzzy_fn: int = 0
    fuzzy_precision: float = 0.0
    fuzzy_recall: float = 0.0
    fuzzy_f1: float = 0.0


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

def _safe_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_exact(pred_names: list[str], label_names: list[str]) -> tuple[int, int, int]:
    """Exact set-matching: TP = intersection, FP = predicted - labels, FN = labels - predicted."""
    pred_set = set(pred_names)
    label_set = set(label_names)
    tp = len(pred_set & label_set)
    fp = len(pred_set - label_set)
    fn = len(label_set - pred_set)
    return tp, fp, fn


def fuzzy_correct_predictions(
    pred_names: list[str],
    all_drug_names: list[str],
    threshold: float = 0.7,
) -> tuple[list[str], list[dict]]:
    """Correct each prediction by fuzzy-matching against the full drug name vocabulary.

    For each predicted name, find the closest match in *all_drug_names*.
    If similarity >= threshold, replace the prediction with the matched name.
    Otherwise keep the original prediction unchanged.

    Returns (corrected_predictions, match_details).
    """
    corrected: list[str] = []
    match_details: list[dict] = []

    for pred in pred_names:
        best_score = 0.0
        best_name = None
        for name in all_drug_names:
            score = jaro_winkler_similarity(pred, name)
            if score > best_score:
                best_score = score
                best_name = name

        if best_score >= threshold and best_name is not None:
            corrected.append(best_name)
            match_details.append({
                "prediction": pred,
                "corrected_to": best_name,
                "similarity": round(best_score, 4),
                "accepted": True,
            })
        else:
            corrected.append(pred)  # keep original
            match_details.append({
                "prediction": pred,
                "corrected_to": None,
                "closest_match": best_name,
                "similarity": round(best_score, 4) if best_name else 0.0,
                "accepted": False,
            })

    return corrected, match_details


def evaluate_fuzzy(
    pred_names: list[str],
    label_names: list[str],
    all_drug_names: list[str],
    threshold: float = 0.7,
) -> tuple[int, int, int, list[dict], list[str]]:
    """Fuzzy evaluation: correct predictions against the full drug vocabulary, then exact-match vs labels.

    1. Each prediction is fuzzy-matched to its closest drug name in *all_drug_names*.
    2. If similarity >= threshold the prediction is replaced with the matched name.
    3. The corrected predictions are then compared to labels via exact set matching.

    Returns (tp, fp, fn, match_details, corrected_predictions).
    """
    corrected, match_details = fuzzy_correct_predictions(pred_names, all_drug_names, threshold)
    # Deduplicate corrected predictions (two different raw predictions may map to the same drug)
    corrected_unique = sorted(set(corrected))
    tp, fp, fn = evaluate_exact(corrected_unique, label_names)
    return tp, fp, fn, match_details, corrected_unique


def evaluate_sample(
    predictions: list[str],
    labels: list[str],
    all_drug_names: list[str],
    threshold: float,
) -> SampleResult:
    """Run both exact and fuzzy evaluation for a single sample."""
    result = SampleResult(
        transaction_id="",
        labels=labels,
        predictions=predictions,
    )

    # Exact
    result.exact_tp, result.exact_fp, result.exact_fn = evaluate_exact(predictions, labels)
    result.exact_precision, result.exact_recall, result.exact_f1 = _safe_prf(
        result.exact_tp, result.exact_fp, result.exact_fn
    )

    # Fuzzy — correct predictions against full drug vocabulary, then exact-match vs labels
    result.fuzzy_tp, result.fuzzy_fp, result.fuzzy_fn, result.fuzzy_matches, _ = evaluate_fuzzy(
        predictions, labels, all_drug_names, threshold
    )
    result.fuzzy_precision, result.fuzzy_recall, result.fuzzy_f1 = _safe_prf(
        result.fuzzy_tp, result.fuzzy_fp, result.fuzzy_fn
    )

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate drug name extraction predictions")
    parser.add_argument(
        "--inference_path",
        type=str,
        required=True,
        help="Path to the inference results JSON produced by inference.py",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/evaluation_results.json",
        help="Path to save the evaluation report",
    )
    parser.add_argument(
        "--drug_names_path",
        type=str,
        required=True,
        help="Path to the drug names JSON produced by extract_drug_names.py",
    )
    parser.add_argument(
        "--fuzzy_threshold",
        type=float,
        default=0.7,
        help="Minimum fuzzy similarity to accept a vocabulary match (default: 0.7)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(
        "Starting evaluation",
        inference_path=args.inference_path,
        drug_names_path=args.drug_names_path,
        threshold=args.fuzzy_threshold,
    )

    # Load full drug-name vocabulary
    with open(args.drug_names_path, "r") as f:
        drug_names_data = json.load(f)
    all_drug_names: list[str] = drug_names_data["drug_names"]
    logger.info("Drug name vocabulary loaded", num_names=len(all_drug_names))

    with open(args.inference_path, "r") as f:
        inference_results = json.load(f)

    agg = AggregateMetrics(num_samples=len(inference_results))
    per_sample: list[dict] = []

    for item in inference_results:
        # Ground-truth labels (normalize)
        raw_labels = item.get("verified_drug_names", [])
        labels = sorted(set(normalize_drug_name(n) for n in raw_labels if normalize_drug_name(n)))

        # Model predictions (normalize)
        preds_obj = item.get("predictions", {})
        raw_preds = preds_obj.get("drug_names", []) if isinstance(preds_obj, dict) else []
        predictions = sorted(set(normalize_drug_name(n) for n in raw_preds if normalize_drug_name(n)))

        result = evaluate_sample(predictions, labels, all_drug_names, args.fuzzy_threshold)
        result.transaction_id = item.get("transaction_id", "")

        # Accumulate
        agg.exact_tp += result.exact_tp
        agg.exact_fp += result.exact_fp
        agg.exact_fn += result.exact_fn
        agg.fuzzy_tp += result.fuzzy_tp
        agg.fuzzy_fp += result.fuzzy_fp
        agg.fuzzy_fn += result.fuzzy_fn

        per_sample.append(asdict(result))

    # Compute aggregate P/R/F1
    agg.exact_precision, agg.exact_recall, agg.exact_f1 = _safe_prf(
        agg.exact_tp, agg.exact_fp, agg.exact_fn
    )
    agg.fuzzy_precision, agg.fuzzy_recall, agg.fuzzy_f1 = _safe_prf(
        agg.fuzzy_tp, agg.fuzzy_fp, agg.fuzzy_fn
    )

    # Print summary
    logger.info(
        "=== Exact Match ===",
        precision=f"{agg.exact_precision:.4f}",
        recall=f"{agg.exact_recall:.4f}",
        f1=f"{agg.exact_f1:.4f}",
        tp=agg.exact_tp,
        fp=agg.exact_fp,
        fn=agg.exact_fn,
    )
    logger.info(
        "=== Fuzzy Match (vocabulary-corrected) ===",
        threshold=args.fuzzy_threshold,
        precision=f"{agg.fuzzy_precision:.4f}",
        recall=f"{agg.fuzzy_recall:.4f}",
        f1=f"{agg.fuzzy_f1:.4f}",
        tp=agg.fuzzy_tp,
        fp=agg.fuzzy_fp,
        fn=agg.fuzzy_fn,
    )

    # Save full report
    report = {
        "config": {
            "inference_path": args.inference_path,
            "drug_names_path": args.drug_names_path,
            "fuzzy_threshold": args.fuzzy_threshold,
            "num_drug_names_in_vocabulary": len(all_drug_names),
            "num_samples": agg.num_samples,
        },
        "aggregate": {
            "exact": {
                "precision": round(agg.exact_precision, 4),
                "recall": round(agg.exact_recall, 4),
                "f1": round(agg.exact_f1, 4),
                "tp": agg.exact_tp,
                "fp": agg.exact_fp,
                "fn": agg.exact_fn,
            },
            "fuzzy": {
                "precision": round(agg.fuzzy_precision, 4),
                "recall": round(agg.fuzzy_recall, 4),
                "f1": round(agg.fuzzy_f1, 4),
                "tp": agg.fuzzy_tp,
                "fp": agg.fuzzy_fp,
                "fn": agg.fuzzy_fn,
            },
        },
        "per_sample": per_sample,
    }

    output_path = args.output_path
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, "evaluation_results.json")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Evaluation saved", output_path=output_path)


if __name__ == "__main__":
    main()

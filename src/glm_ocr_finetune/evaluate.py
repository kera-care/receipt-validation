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

from glm_ocr_finetune.data.utils import normalize_drug_name

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Jaro-Winkler implementation (pure-Python, no external dependency)
# ---------------------------------------------------------------------------

def _jaro_similarity(s1: str, s2: str) -> float:
    """Compute Jaro similarity between two strings."""
    if s1 == s2:
        return 1.0
    len_s1, len_s2 = len(s1), len(s2)
    if len_s1 == 0 or len_s2 == 0:
        return 0.0

    match_distance = max(len_s1, len_s2) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len_s1
    s2_matches = [False] * len_s2

    matches = 0
    transpositions = 0

    for i in range(len_s1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len_s2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len_s1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    return (
        matches / len_s1 + matches / len_s2 + (matches - transpositions / 2) / matches
    ) / 3


def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1) -> float:
    """Compute Jaro-Winkler similarity (higher = more similar, 1.0 = identical)."""
    jaro = _jaro_similarity(s1, s2)
    # common prefix length (up to 4 characters)
    prefix_len = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break
    return jaro + prefix_len * p * (1 - jaro)


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


def evaluate_fuzzy(
    pred_names: list[str],
    label_names: list[str],
    threshold: float = 0.7,
) -> tuple[int, int, int, list[dict]]:
    """Fuzzy matching via Jaro-Winkler.

    For each predicted name, find the closest unmatched label.
    Accept the match if similarity >= threshold.

    Returns (tp, fp, fn, match_details).
    """
    remaining_labels = list(label_names)  # mutable copy
    tp = 0
    fp = 0
    match_details: list[dict] = []

    for pred in pred_names:
        best_score = 0.0
        best_idx = -1
        best_label = None
        for idx, label in enumerate(remaining_labels):
            score = jaro_winkler_similarity(pred, label)
            if score > best_score:
                best_score = score
                best_idx = idx
                best_label = label

        if best_score >= threshold and best_idx >= 0:
            tp += 1
            remaining_labels.pop(best_idx)
            match_details.append({
                "prediction": pred,
                "matched_label": best_label,
                "similarity": round(best_score, 4),
                "accepted": True,
            })
        else:
            fp += 1
            match_details.append({
                "prediction": pred,
                "matched_label": best_label,
                "similarity": round(best_score, 4) if best_label else 0.0,
                "accepted": False,
            })

    fn = len(remaining_labels)  # unmatched labels
    return tp, fp, fn, match_details


def evaluate_sample(
    predictions: list[str],
    labels: list[str],
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

    # Fuzzy
    result.fuzzy_tp, result.fuzzy_fp, result.fuzzy_fn, result.fuzzy_matches = evaluate_fuzzy(
        predictions, labels, threshold
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
        "--jaro_winkler_threshold",
        type=float,
        default=0.7,
        help="Minimum Jaro-Winkler similarity to accept a fuzzy match (default: 0.7)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(
        "Starting evaluation",
        inference_path=args.inference_path,
        threshold=args.jaro_winkler_threshold,
    )

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

        result = evaluate_sample(predictions, labels, args.jaro_winkler_threshold)
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
        "=== Fuzzy Match (Jaro-Winkler) ===",
        threshold=args.jaro_winkler_threshold,
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
            "jaro_winkler_threshold": args.jaro_winkler_threshold,
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

"""
Evaluate drug name extraction predictions against ground-truth labels.

Two evaluation modes:
    1. **Exact match** — a predicted name must match a label character-for-character
       (after normalization). Even a single extra/missing character counts as an error.
    2. **Fuzzy match** — each predicted name is fuzzy-matched against the full drug
       vocabulary and corrected if similarity >= threshold. Evaluated at multiple
       thresholds so you can compare the effect on P/R/F1.

Outputs:
    - ``evaluation_results.json`` — aggregate + per-sample metrics for exact match
      and fuzzy match at every requested threshold.
    - ``error_analysis/threshold_<T>.json`` (one per threshold) — per-sample breakdown
      showing raw predictions, corrected predictions, TPs, FPs, FNs and match details
      so you can inspect what errors the model is making.

Usage:
    python -m glm_ocr_finetune.evaluate \
        --inference_path outputs/inference_results.json \
        --drug_names_path outputs/all_drug_names.json \
        --output_path outputs/evaluation_results.json \
        [--fuzzy_thresholds 0.5 0.6 0.7 0.8 0.9]
"""

import argparse
import json
import os

import structlog
from thefuzz import fuzz
from tqdm import tqdm

from glm_ocr_finetune.data.utils import normalize_drug_name

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def fuzzy_similarity(s1: str, s2: str) -> float:
    """Return a similarity score in [0, 1] using thefuzz's ratio.

    The score is divided by 100 to normalise from thefuzz's 0-100 range.
    """
    return fuzz.ratio(s1, s2) / 100.0


# ---------------------------------------------------------------------------
# Core evaluation helpers
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


# ---------------------------------------------------------------------------
# Fuzzy vocabulary matching (expensive — done once per prediction)
# ---------------------------------------------------------------------------

def fuzzy_find_best_matches(
    pred_names: list[str],
    all_drug_names: list[str],
) -> list[dict]:
    """For each prediction, find the closest drug name in the vocabulary.

    Returns a list of ``{prediction, best_match, similarity}`` dicts.
    The threshold is **not** applied here — that is done in :func:`apply_threshold`
    so the expensive O(P x V) loop only runs once.
    """
    matches: list[dict] = []
    for pred in pred_names:
        best_score = 0.0
        best_name = None
        for name in all_drug_names:
            score = fuzzy_similarity(pred, name)
            if score > best_score:
                best_score = score
                best_name = name
        matches.append({
            "prediction": pred,
            "best_match": best_name,
            "similarity": round(best_score, 4) if best_name else 0.0,
        })
    return matches


def apply_threshold(
    matches: list[dict],
    threshold: float,
) -> tuple[list[str], list[dict]]:
    """Apply a similarity threshold to pre-computed best matches.

    Returns ``(corrected_predictions, match_details)``.
    """
    corrected: list[str] = []
    details: list[dict] = []
    for m in matches:
        if m["similarity"] >= threshold and m["best_match"] is not None:
            corrected.append(m["best_match"])
            details.append({
                "prediction": m["prediction"],
                "corrected_to": m["best_match"],
                "similarity": m["similarity"],
                "accepted": True,
            })
        else:
            corrected.append(m["prediction"])  # keep original
            details.append({
                "prediction": m["prediction"],
                "corrected_to": None,
                "closest_match": m["best_match"],
                "similarity": m["similarity"],
                "accepted": False,
            })
    return corrected, details


# ---------------------------------------------------------------------------
# CLI
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
        help="Path to save the main evaluation report",
    )
    parser.add_argument(
        "--drug_names_path",
        type=str,
        required=True,
        help="Path to the drug names JSON produced by extract_drug_names.py",
    )
    parser.add_argument(
        "--fuzzy_thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.6, 0.7, 0.8, 0.9],
        help="Fuzzy similarity thresholds to evaluate (default: 0.5 0.6 0.7 0.8 0.9)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    thresholds = sorted(args.fuzzy_thresholds)

    logger.info(
        "Starting evaluation",
        inference_path=args.inference_path,
        drug_names_path=args.drug_names_path,
        thresholds=thresholds,
    )

    # ------------------------------------------------------------------ #
    # Load vocabulary + inference results
    # ------------------------------------------------------------------ #
    with open(args.drug_names_path, "r") as f:
        drug_names_data = json.load(f)
    all_drug_names: list[str] = drug_names_data["drug_names"]
    logger.info("Drug name vocabulary loaded", num_names=len(all_drug_names))

    with open(args.inference_path, "r") as f:
        inference_results = json.load(f)
    logger.info("Inference results loaded", num_samples=len(inference_results))

    # ------------------------------------------------------------------ #
    # Phase 1: Normalize + compute exact match + best vocabulary matches
    # (The vocabulary matching is O(P x V) per sample — expensive, done once)
    # ------------------------------------------------------------------ #
    samples: list[dict] = []

    for item in tqdm(inference_results, desc="Computing vocabulary matches"):
        # Ground-truth labels
        raw_labels = item.get("verified_drug_names", [])
        labels = sorted(set(normalize_drug_name(n) for n in raw_labels if normalize_drug_name(n)))

        # Model predictions
        preds_obj = item.get("predictions", {})
        raw_preds = preds_obj.get("drug_names", []) if isinstance(preds_obj, dict) else []
        predictions = sorted(set(normalize_drug_name(n) for n in raw_preds if normalize_drug_name(n)))

        # Exact match (threshold-independent)
        exact_tp, exact_fp, exact_fn = evaluate_exact(predictions, labels)
        exact_p, exact_r, exact_f1 = _safe_prf(exact_tp, exact_fp, exact_fn)

        # Best vocabulary matches for each prediction (expensive)
        best_matches = fuzzy_find_best_matches(predictions, all_drug_names)

        pred_set = set(predictions)
        label_set = set(labels)

        samples.append({
            "transaction_id": item.get("transaction_id", ""),
            "labels": labels,
            "predictions": predictions,
            "exact": {
                "tp": exact_tp, "fp": exact_fp, "fn": exact_fn,
                "precision": round(exact_p, 4),
                "recall": round(exact_r, 4),
                "f1": round(exact_f1, 4),
                "true_positives": sorted(pred_set & label_set),
                "false_positives": sorted(pred_set - label_set),
                "false_negatives": sorted(label_set - pred_set),
            },
            "best_matches": best_matches,
        })

    # ------------------------------------------------------------------ #
    # Phase 2: Aggregate exact-match metrics
    # ------------------------------------------------------------------ #
    exact_agg: dict = {"tp": 0, "fp": 0, "fn": 0}
    for s in samples:
        exact_agg["tp"] += s["exact"]["tp"]
        exact_agg["fp"] += s["exact"]["fp"]
        exact_agg["fn"] += s["exact"]["fn"]

    exact_p, exact_r, exact_f1 = _safe_prf(exact_agg["tp"], exact_agg["fp"], exact_agg["fn"])
    exact_agg.update(precision=round(exact_p, 4), recall=round(exact_r, 4), f1=round(exact_f1, 4))

    logger.info(
        "=== Exact Match ===",
        precision=f"{exact_p:.4f}",
        recall=f"{exact_r:.4f}",
        f1=f"{exact_f1:.4f}",
        tp=exact_agg["tp"],
        fp=exact_agg["fp"],
        fn=exact_agg["fn"],
    )

    # ------------------------------------------------------------------ #
    # Phase 3: Fuzzy evaluation at each threshold (cheap — just filtering)
    # ------------------------------------------------------------------ #
    fuzzy_results: dict[float, dict] = {}

    for threshold in thresholds:
        agg: dict = {"tp": 0, "fp": 0, "fn": 0}
        error_analysis: list[dict] = []

        for s in tqdm(samples, desc=f"Fuzzy eval @ {threshold:.2f}", leave=False):
            corrected, match_details = apply_threshold(s["best_matches"], threshold)
            corrected_unique = sorted(set(corrected))
            tp, fp, fn = evaluate_exact(corrected_unique, s["labels"])
            agg["tp"] += tp
            agg["fp"] += fp
            agg["fn"] += fn

            corrected_set = set(corrected_unique)
            label_set = set(s["labels"])

            error_analysis.append({
                "transaction_id": s["transaction_id"],
                "labels": s["labels"],
                "raw_predictions": s["predictions"],
                "corrected_predictions": corrected_unique,
                "true_positives": sorted(corrected_set & label_set),
                "false_positives": sorted(corrected_set - label_set),
                "false_negatives": sorted(label_set - corrected_set),
                "match_details": match_details,
            })

        p, r, f1 = _safe_prf(agg["tp"], agg["fp"], agg["fn"])
        agg.update(precision=round(p, 4), recall=round(r, 4), f1=round(f1, 4))

        fuzzy_results[threshold] = {
            "aggregate": agg,
            "error_analysis": error_analysis,
        }

        logger.info(
            f"=== Fuzzy Match @ threshold={threshold:.2f} ===",
            precision=f"{p:.4f}",
            recall=f"{r:.4f}",
            f1=f"{f1:.4f}",
            tp=agg["tp"],
            fp=agg["fp"],
            fn=agg["fn"],
        )

    # ------------------------------------------------------------------ #
    # Phase 4: Build per-sample records (exact + all fuzzy thresholds)
    # ------------------------------------------------------------------ #
    per_sample: list[dict] = []
    for i, s in enumerate(samples):
        record: dict = {
            "transaction_id": s["transaction_id"],
            "labels": s["labels"],
            "predictions": s["predictions"],
            "exact": s["exact"],
            "fuzzy": {},
        }
        for threshold in thresholds:
            ea = fuzzy_results[threshold]["error_analysis"][i]
            record["fuzzy"][str(threshold)] = {
                "corrected_predictions": ea["corrected_predictions"],
                "true_positives": ea["true_positives"],
                "false_positives": ea["false_positives"],
                "false_negatives": ea["false_negatives"],
            }
        per_sample.append(record)

    # ------------------------------------------------------------------ #
    # Phase 5: Save main report
    # ------------------------------------------------------------------ #
    output_path = args.output_path
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, "evaluation_results.json")
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    report = {
        "config": {
            "inference_path": args.inference_path,
            "drug_names_path": args.drug_names_path,
            "fuzzy_thresholds": thresholds,
            "num_drug_names_in_vocabulary": len(all_drug_names),
            "num_samples": len(samples),
        },
        "aggregate": {
            "exact": exact_agg,
            "fuzzy": {
                str(t): fuzzy_results[t]["aggregate"] for t in thresholds
            },
        },
        "per_sample": per_sample,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Main report saved", output_path=output_path)

    # ------------------------------------------------------------------ #
    # Phase 6: Save per-threshold error analysis files
    # ------------------------------------------------------------------ #
    ea_dir = os.path.join(output_dir, "error_analysis")
    os.makedirs(ea_dir, exist_ok=True)

    for threshold in thresholds:
        ea_path = os.path.join(ea_dir, f"threshold_{threshold:.2f}.json")
        with open(ea_path, "w") as f:
            json.dump(
                fuzzy_results[threshold]["error_analysis"],
                f, indent=2, ensure_ascii=False,
            )
        logger.info("Error analysis saved", threshold=threshold, path=ea_path)

    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()

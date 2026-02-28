"""
Evaluate drug name extraction predictions against ground-truth labels.

Uses ``resources/drug_roots.json`` which maps drug names to *roots* and
*variants*.  Both labels and predictions are fuzzy-matched against the full
list of variants.  The matched variant is mapped back to its **entry key**
(the top-level key in drug_roots.json) and evaluation is done by comparing
**key sets** — one key per drug entry, regardless of how many roots it has.

Two evaluation modes:
    1. **Exact match (full string)** — after normalization, predicted name must
       match a label character-for-character.
    2. **Root match** — both labels and predictions are resolved to their drug
       entry keys via best-variant matching; keys are compared as sets.
       The same similarity threshold is applied to both sides so that
       identical strings always receive the same treatment.

Outputs:
    - ``evaluation_results.json`` — aggregate + per-sample metrics for all modes.
    - ``error_analysis/threshold_<T>.json`` (one per threshold) — per-sample
      breakdown showing raw predictions, resolved roots, TPs, FPs, FNs and
      match details for inspecting model errors.

Usage:
    python -m glm_ocr_finetune.evaluate \
        --inference_path outputs/inference_results.json \
        --drug_roots_path resources/drug_roots.json \
        --output_path outputs/evaluation_results.json \
        [--fuzzy_thresholds 0.5 0.6 0.7 0.8 0.9]
"""

import argparse
import csv
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
    """Return a similarity score in [0, 1] using thefuzz's ratio."""
    return fuzz.ratio(s1, s2) / 100.0


def extract_drug_name_part(name: str) -> str:
    """Extract the drug name portion, stripping dosage/packaging info.

    Drug names from the dataset often look like::

        amoxicilline arrow - 1g, b/30
        maxidrol - 100mg/0,35mui/0,6mui, fl/3ml

    Everything after the first `` - `` is dosage/form/packaging and should be
    stripped before matching against the variant index.
    """
    # Split on ' - ' (space-dash-space) which separates name from dosage
    parts = name.split(" - ", 1)
    return parts[0].strip()


# ---------------------------------------------------------------------------
# Drug roots index
# ---------------------------------------------------------------------------

def build_variant_index(drug_roots: dict) -> tuple[list[str], dict[str, str]]:
    """Build a flat index from *drug_roots.json*.

    Returns:
        all_variants: sorted list of every unique variant (used for fuzzy search).
        variant_to_key: mapping from each normalized variant to the top-level
            entry key it belongs to.  Because variants are unique across entries
            each variant maps to exactly one key.
    """
    variant_to_key: dict[str, str] = {}
    for key, entry in drug_roots.items():
        norm_key = normalize_drug_name(key)
        for variant in entry["variants"]:
            norm = normalize_drug_name(variant)
            if norm:
                variant_to_key.setdefault(norm, norm_key)
        # Also index roots themselves as variants so labels/predictions that
        # already match a root are handled.
        for root in entry["roots"]:
            norm = normalize_drug_name(root)
            if norm:
                variant_to_key.setdefault(norm, norm_key)
        # And the key itself
        if norm_key:
            variant_to_key.setdefault(norm_key, norm_key)
    all_variants = sorted(variant_to_key.keys())
    return all_variants, variant_to_key


def resolve_to_keys(
    names: list[str],
    all_variants: list[str],
    variant_to_key: dict[str, str],
    threshold: float = 0.0,
) -> tuple[set[str], list[dict]]:
    """Resolve a list of drug names to their entry key via best-variant matching.

    For each name, find the variant with the highest fuzzy similarity.
    If similarity >= *threshold*, use that variant's entry key; otherwise keep
    the original name as an "unresolved" key.

    Returns (key_set, match_details).
    """
    keys: set[str] = set()
    details: list[dict] = []

    for name in names:
        # Strip dosage/packaging info for matching purposes
        match_name = extract_drug_name_part(name)

        # Check exact match first (fast path)
        if match_name in variant_to_key:
            matched_key = variant_to_key[match_name]
            keys.add(matched_key)
            details.append({
                "name": name,
                "match_name": match_name,
                "best_variant": match_name,
                "similarity": 1.0,
                "resolved_key": matched_key,
                "exact_variant_match": True,
            })
            continue

        # Fuzzy search through all variants
        best_score = 0.0
        best_variant = None
        for variant in all_variants:
            score = fuzzy_similarity(match_name, variant)
            if score > best_score:
                best_score = score
                best_variant = variant

        if best_score >= threshold and best_variant is not None:
            matched_key = variant_to_key[best_variant]
            keys.add(matched_key)
            details.append({
                "name": name,
                "match_name": match_name,
                "best_variant": best_variant,
                "similarity": round(best_score, 4),
                "resolved_key": matched_key,
                "exact_variant_match": False,
            })
        else:
            # Unresolved — use the original name as the key
            keys.add(name)
            details.append({
                "name": name,
                "match_name": match_name,
                "best_variant": best_variant,
                "similarity": round(best_score, 4) if best_variant else 0.0,
                "resolved_key": name,
                "exact_variant_match": False,
                "unresolved": True,
            })

    return keys, details


# ---------------------------------------------------------------------------
# Exclusion index
# ---------------------------------------------------------------------------

def build_exclusion_set(csv_path: str) -> set[str]:
    """Load *drugs_exclusion.csv* and return a set of normalized drug names
    that are marked as exclusions (``is_exclusion == 'True'``).
    """
    exclusion_keys: set[str] = set()
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["is_exclusion"].strip() == "True":
                norm = normalize_drug_name(row["drug_name"])
                if norm:
                    exclusion_keys.add(norm)
    return exclusion_keys


# ---------------------------------------------------------------------------
# Core evaluation helpers
# ---------------------------------------------------------------------------

def _safe_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_sets(pred_set: set[str], label_set: set[str]) -> tuple[int, int, int]:
    """Set-matching: TP = intersection, FP = predicted − labels, FN = labels − predicted."""
    tp = len(pred_set & label_set)
    fp = len(pred_set - label_set)
    fn = len(label_set - pred_set)
    return tp, fp, fn


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
        "--drug_roots_path",
        type=str,
        required=True,
        help="Path to drug_roots.json (maps drug names to roots and variants)",
    )
    parser.add_argument(
        "--exclusion_path",
        type=str,
        default=None,
        help="Path to drugs_exclusion.csv (drug_name,is_exclusion). "
             "When provided, exclusion detection metrics are reported.",
    )
    parser.add_argument(
        "--fuzzy_thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.6, 0.7, 0.8, 0.9],
        help="Fuzzy similarity thresholds for variant matching (default: 0.5 0.6 0.7 0.8 0.9)",
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
        drug_roots_path=args.drug_roots_path,
        exclusion_path=args.exclusion_path,
        thresholds=thresholds,
    )

    # ------------------------------------------------------------------ #
    # Load drug roots index
    # ------------------------------------------------------------------ #
    with open(args.drug_roots_path, "r") as f:
        drug_roots_raw = json.load(f)
    all_variants, variant_to_key = build_variant_index(drug_roots_raw)
    logger.info(
        "Drug roots index built",
        num_entries=len(drug_roots_raw),
        num_variants=len(all_variants),
        num_unique_keys=len(set(variant_to_key.values())),
    )

    # ------------------------------------------------------------------ #
    # Load exclusion index (optional)
    # ------------------------------------------------------------------ #
    exclusion_set: set[str] | None = None
    if args.exclusion_path:
        exclusion_set = build_exclusion_set(args.exclusion_path)
        logger.info("Exclusion index loaded", num_exclusion_drugs=len(exclusion_set))

    # ------------------------------------------------------------------ #
    # Load inference results
    # ------------------------------------------------------------------ #
    with open(args.inference_path, "r") as f:
        inference_results = json.load(f)
    logger.info("Inference results loaded", num_samples=len(inference_results))

    # ------------------------------------------------------------------ #
    # Phase 1: Normalize, compute exact string match, and resolve roots
    # (Variant matching is O(N × V) per name — expensive, done once)
    # ------------------------------------------------------------------ #
    samples: list[dict] = []

    for item in tqdm(inference_results, desc="Resolving drug roots"):
        # Ground-truth labels
        raw_labels = item.get("verified_drug_names", [])
        labels = sorted(set(normalize_drug_name(n) for n in raw_labels if normalize_drug_name(n)))

        # Model predictions
        preds_obj = item.get("predictions", {})
        raw_preds = preds_obj.get("drug_names", []) if isinstance(preds_obj, dict) else []
        predictions = sorted(set(normalize_drug_name(n) for n in raw_preds if normalize_drug_name(n)))

        # --- Exact string match (no root resolution) ---
        pred_set = set(predictions)
        label_set = set(labels)
        exact_tp, exact_fp, exact_fn = evaluate_sets(pred_set, label_set)
        exact_p, exact_r, exact_f1 = _safe_prf(exact_tp, exact_fp, exact_fn)

        # --- Resolve labels to entry keys (fixed 0.5 threshold) ---
        label_keys, label_match_details = resolve_to_keys(
            labels, all_variants, variant_to_key, threshold=0.5,
        )
        # --- Resolve predictions to entry keys (cache at threshold=0.0;
        #     actual threshold applied in Phase 3) ---
        pred_keys, pred_match_details = resolve_to_keys(
            predictions, all_variants, variant_to_key, threshold=0.0,
        )

        samples.append({
            "transaction_id": item.get("transaction_id", ""),
            "labels": labels,
            "predictions": predictions,
            "label_keys": sorted(label_keys),
            "pred_keys": sorted(pred_keys),
            "label_match_details": label_match_details,
            "pred_match_details": pred_match_details,
            "exact": {
                "tp": exact_tp, "fp": exact_fp, "fn": exact_fn,
                "precision": round(exact_p, 4),
                "recall": round(exact_r, 4),
                "f1": round(exact_f1, 4),
                "true_positives": sorted(pred_set & label_set),
                "false_positives": sorted(pred_set - label_set),
                "false_negatives": sorted(label_set - pred_set),
            },
        })

    # ------------------------------------------------------------------ #
    # Phase 2: Aggregate exact string match
    # ------------------------------------------------------------------ #
    exact_agg: dict = {"tp": 0, "fp": 0, "fn": 0}
    for s in samples:
        exact_agg["tp"] += s["exact"]["tp"]
        exact_agg["fp"] += s["exact"]["fp"]
        exact_agg["fn"] += s["exact"]["fn"]

    exact_p, exact_r, exact_f1 = _safe_prf(exact_agg["tp"], exact_agg["fp"], exact_agg["fn"])
    exact_agg.update(precision=round(exact_p, 4), recall=round(exact_r, 4), f1=round(exact_f1, 4))

    logger.info(
        "=== Exact String Match ===",
        precision=f"{exact_p:.4f}",
        recall=f"{exact_r:.4f}",
        f1=f"{exact_f1:.4f}",
        tp=exact_agg["tp"],
        fp=exact_agg["fp"],
        fn=exact_agg["fn"],
    )

    # ------------------------------------------------------------------ #
    # Phase 3: Root-based evaluation at each threshold
    #
    # Labels are resolved once at a fixed 0.5 threshold.  Only prediction
    # thresholds vary across iterations.
    # ------------------------------------------------------------------ #
    root_results: dict[float, dict] = {}

    for threshold in thresholds:
        agg: dict = {"tp": 0, "fp": 0, "fn": 0}
        excl_agg: dict = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        error_analysis: list[dict] = []

        for s in tqdm(samples, desc=f"Root eval @ {threshold:.2f}", leave=False):
            # Labels fixed at 0.5; only prediction threshold varies
            pred_keys = _apply_threshold_to_details(s["pred_match_details"], threshold, variant_to_key)
            label_keys = set(s["label_keys"])  # already resolved at fixed 0.5

            tp, fp, fn = evaluate_sets(pred_keys, label_keys)
            agg["tp"] += tp
            agg["fp"] += fp
            agg["fn"] += fn

            error_analysis.append({
                "transaction_id": s["transaction_id"],
                "labels": s["labels"],
                "predictions": s["predictions"],
                "label_keys": sorted(label_keys),
                "pred_keys": sorted(pred_keys),
                "true_positives": sorted(pred_keys & label_keys),
                "false_positives": sorted(pred_keys - label_keys),
                "false_negatives": sorted(label_keys - pred_keys),
                "label_match_details": s["label_match_details"],
                "pred_match_details": s["pred_match_details"],
            })

            # --- Exclusion detection (per-sample binary) ---
            if exclusion_set is not None:
                label_has_excl = bool(label_keys & exclusion_set)
                pred_has_excl = bool(pred_keys & exclusion_set)
                label_excl_drugs = sorted(label_keys & exclusion_set)
                pred_excl_drugs = sorted(pred_keys & exclusion_set)

                if label_has_excl and pred_has_excl:
                    excl_agg["tp"] += 1
                elif pred_has_excl and not label_has_excl:
                    excl_agg["fp"] += 1
                elif not pred_has_excl and label_has_excl:
                    excl_agg["fn"] += 1
                else:
                    excl_agg["tn"] += 1

                error_analysis[-1]["exclusion"] = {
                    "label_has_exclusion": label_has_excl,
                    "pred_has_exclusion": pred_has_excl,
                    "label_exclusion_drugs": label_excl_drugs,
                    "pred_exclusion_drugs": pred_excl_drugs,
                }

        p, r, f1 = _safe_prf(agg["tp"], agg["fp"], agg["fn"])
        agg.update(precision=round(p, 4), recall=round(r, 4), f1=round(f1, 4))

        root_results[threshold] = {
            "aggregate": agg,
            "error_analysis": error_analysis,
        }

        # Exclusion aggregate for this threshold
        if exclusion_set is not None:
            excl_p, excl_r, excl_f1 = _safe_prf(excl_agg["tp"], excl_agg["fp"], excl_agg["fn"])
            excl_agg.update(
                precision=round(excl_p, 4),
                recall=round(excl_r, 4),
                f1=round(excl_f1, 4),
            )
            root_results[threshold]["exclusion"] = excl_agg

            logger.info(
                f"=== Exclusion Detection @ threshold={threshold:.2f} ===",
                precision=f"{excl_p:.4f}",
                recall=f"{excl_r:.4f}",
                f1=f"{excl_f1:.4f}",
                tp=excl_agg["tp"],
                fp=excl_agg["fp"],
                fn=excl_agg["fn"],
                tn=excl_agg["tn"],
            )

        logger.info(
            f"=== Root Match @ threshold={threshold:.2f} ===",
            precision=f"{p:.4f}",
            recall=f"{r:.4f}",
            f1=f"{f1:.4f}",
            tp=agg["tp"],
            fp=agg["fp"],
            fn=agg["fn"],
        )

    # ------------------------------------------------------------------ #
    # Phase 4: Build per-sample records
    # ------------------------------------------------------------------ #
    per_sample: list[dict] = []
    for i, s in enumerate(samples):
        record: dict = {
            "transaction_id": s["transaction_id"],
            "labels": s["labels"],
            "predictions": s["predictions"],
            "exact": s["exact"],
            "root": {},
        }
        for threshold in thresholds:
            ea = root_results[threshold]["error_analysis"][i]
            record["root"][str(threshold)] = {
                "label_keys": ea["label_keys"],
                "pred_keys": ea["pred_keys"],
                "true_positives": ea["true_positives"],
                "false_positives": ea["false_positives"],
                "false_negatives": ea["false_negatives"],
            }
            if "exclusion" in ea:
                record["root"][str(threshold)]["exclusion"] = ea["exclusion"]
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
            "drug_roots_path": args.drug_roots_path,
            "exclusion_path": args.exclusion_path,
            "fuzzy_thresholds": thresholds,
            "num_drug_entries": len(drug_roots_raw),
            "num_variants": len(all_variants),
            "num_exclusion_drugs": len(exclusion_set) if exclusion_set else 0,
            "num_samples": len(samples),
        },
        "aggregate": {
            "exact": exact_agg,
            "root": {
                str(t): root_results[t]["aggregate"] for t in thresholds
            },
        },
        "per_sample": per_sample,
    }

    # Add exclusion aggregate if available
    if exclusion_set is not None:
        report["aggregate"]["exclusion"] = {
            str(t): root_results[t]["exclusion"] for t in thresholds
            if "exclusion" in root_results[t]
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
                root_results[threshold]["error_analysis"],
                f, indent=2, ensure_ascii=False,
            )
        logger.info("Error analysis saved", threshold=threshold, path=ea_path)

    logger.info("Evaluation complete")


def _apply_threshold_to_details(
    match_details: list[dict],
    threshold: float,
    variant_to_key: dict[str, str],
) -> set[str]:
    """Re-derive entry keys from cached match details using a new threshold.

    If the best-variant similarity >= threshold, use the variant's entry key.
    Otherwise fall back to the original name as an unresolved key.
    """
    keys: set[str] = set()
    for d in match_details:
        if d.get("exact_variant_match") or d["similarity"] >= threshold:
            best_variant = d["best_variant"]
            if best_variant in variant_to_key:
                keys.add(variant_to_key[best_variant])
            else:
                keys.add(d["name"])
        else:
            keys.add(d["name"])
    return keys


if __name__ == "__main__":
    main()

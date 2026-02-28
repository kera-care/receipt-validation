#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# Evaluate drug name extraction predictions
#
# Evaluation modes:
#   1. Exact match      — character-perfect match after normalization
#   2. Root match       — resolve labels & predictions to drug entry keys
#                         via variant matching, compare key sets
#   3. Exclusion detect — binary per-sample: does the prescription contain
#                         any exclusion drug? (requires --exclusion_path)
#
# Per-threshold error analysis files are saved alongside the main report
# so you can inspect exactly what errors the model is making.
# ──────────────────────────────────────────────────────────────────────

MODEL_PATH="${MODEL_PATH:-outputs/glm-ocr-finetune-20-epochs/final_model}"
INFERENCE_PATH="${INFERENCE_PATH:-$(dirname "${MODEL_PATH}")/inference_outputs/inference_results.json}"
DRUG_ROOTS_PATH="${DRUG_ROOTS_PATH:-resources/drug_roots.json}"
EXCLUSION_PATH="${EXCLUSION_PATH:-resources/drugs_exclusion.csv}"
EVAL_OUTPUT_PATH="${EVAL_OUTPUT_PATH:-$(dirname "${MODEL_PATH}")/evaluation_results.json}"
FUZZY_THRESHOLDS="${FUZZY_THRESHOLDS:-0.5 0.6 0.7 0.8 0.9}"

echo "============================================="
echo "  Drug Name Evaluation (Root-Based)"
echo "============================================="
echo "  Inference file:    ${INFERENCE_PATH}"
echo "  Drug roots:        ${DRUG_ROOTS_PATH}"
echo "  Exclusion list:    ${EXCLUSION_PATH}"
echo "  Output:            ${EVAL_OUTPUT_PATH}"
echo "  Fuzzy thresholds:  ${FUZZY_THRESHOLDS}"
echo "============================================="

poetry run python -m glm_ocr_finetune.evaluate \
    --inference_path "${INFERENCE_PATH}" \
    --drug_roots_path "${DRUG_ROOTS_PATH}" \
    --exclusion_path "${EXCLUSION_PATH}" \
    --output_path "${EVAL_OUTPUT_PATH}" \
    --fuzzy_thresholds ${FUZZY_THRESHOLDS}

#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# Evaluate drug name extraction predictions
#
# Runs two evaluation modes:
#   1. Exact match   — character-perfect match after normalization
#   2. Fuzzy match   — correct predictions against full drug vocabulary
#                      at multiple thresholds, then exact-match vs labels
#
# Per-threshold error analysis files are saved alongside the main report
# so you can inspect exactly what errors the model is making.
# ──────────────────────────────────────────────────────────────────────

MODEL_PATH="${MODEL_PATH:-outputs/glm-ocr-finetune-20-epochs/final_model}"
INFERENCE_PATH="${INFERENCE_PATH:-$(dirname "${MODEL_PATH}")/inference_outputs/inference_results.json}"
DRUG_NAMES_PATH="${DRUG_NAMES_PATH:-$(dirname "${MODEL_PATH}")/all_drug_names.json}"
EVAL_OUTPUT_PATH="${EVAL_OUTPUT_PATH:-$(dirname "${MODEL_PATH}")/evaluation_results.json}"
FUZZY_THRESHOLDS="${FUZZY_THRESHOLDS:-0.5 0.6 0.7 0.8 0.9}"

echo "============================================="
echo "  Drug Name Evaluation"
echo "============================================="
echo "  Inference file:    ${INFERENCE_PATH}"
echo "  Drug names vocab:  ${DRUG_NAMES_PATH}"
echo "  Output:            ${EVAL_OUTPUT_PATH}"
echo "  Fuzzy thresholds:  ${FUZZY_THRESHOLDS}"
echo "============================================="

poetry run python -m glm_ocr_finetune.evaluate \
    --inference_path "${INFERENCE_PATH}" \
    --drug_names_path "${DRUG_NAMES_PATH}" \
    --output_path "${EVAL_OUTPUT_PATH}" \
    --fuzzy_thresholds ${FUZZY_THRESHOLDS}

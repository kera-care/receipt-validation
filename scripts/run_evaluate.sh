#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# Evaluate drug name extraction predictions
#
# Runs two evaluation modes:
#   1. Exact match   — character-perfect match after normalization
#   2. Fuzzy match   — correct predictions against full drug vocabulary,
#                      then exact-match vs labels
# ──────────────────────────────────────────────────────────────────────

MODEL_PATH="${MODEL_PATH:-outputs/glm-ocr-finetune-20-epochs/final_model}"
INFERENCE_PATH="${INFERENCE_PATH:-$(dirname "${MODEL_PATH}")/inference_outputs/inference_results.json}"
DRUG_NAMES_PATH="${OUTPUT_PATH:-$(dirname "${MODEL_PATH}")/all_drug_names.json}"
EVAL_OUTPUT_PATH="${EVAL_OUTPUT_PATH:-$(dirname "${MODEL_PATH}")/evaluation_results.json}"
FUZZY_THRESHOLD="${FUZZY_THRESHOLD:-0.7}"

echo "============================================="
echo "  Drug Name Evaluation"
echo "============================================="
echo "  Inference file:   ${INFERENCE_PATH}"
echo "  Drug names vocab: ${DRUG_NAMES_PATH}"
echo "  Output:           ${EVAL_OUTPUT_PATH}"
echo "  Fuzzy threshold:  ${FUZZY_THRESHOLD}"
echo "============================================="

poetry run python -m glm_ocr_finetune.evaluate \
    --inference_path "${INFERENCE_PATH}" \
    --drug_names_path "${DRUG_NAMES_PATH}" \
    --output_path "${EVAL_OUTPUT_PATH}" \
    --fuzzy_threshold "${FUZZY_THRESHOLD}"

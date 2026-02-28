#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# Evaluate drug name extraction predictions
#
# Runs two evaluation modes:
#   1. Exact match   — character-perfect match after normalization
#   2. Fuzzy match   — Jaro-Winkler similarity >= threshold
# ──────────────────────────────────────────────────────────────────────

INFERENCE_PATH="${INFERENCE_PATH:-outputs/inference_results.json}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/evaluation_results.json}"
JARO_WINKLER_THRESHOLD="${JARO_WINKLER_THRESHOLD:-0.7}"

echo "============================================="
echo "  Drug Name Evaluation"
echo "============================================="
echo "  Inference file:   ${INFERENCE_PATH}"
echo "  Output:           ${OUTPUT_PATH}"
echo "  JW threshold:     ${JARO_WINKLER_THRESHOLD}"
echo "============================================="

poetry run python -m glm_ocr_finetune.evaluate \
    --inference_path "${INFERENCE_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --jaro_winkler_threshold "${JARO_WINKLER_THRESHOLD}"

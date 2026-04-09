#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# Extract receipt dataset statistics
# ──────────────────────────────────────────────────────────────────────

TRAIN_TASKS="${TRAIN_TASKS:-dataset/train_tasks.json}"
VAL_TASKS="${VAL_TASKS:-dataset/val_tasks.json}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/receipt_stats.json}"

echo "============================================="
echo "  Receipt Statistics Extraction"
echo "============================================="
echo "  Task files:       ${TRAIN_TASKS} ${VAL_TASKS}"
echo "  Output:           ${OUTPUT_PATH}"
echo "============================================="

poetry run python -m glm_ocr_finetune.extract_receipt_stats \
    --task_files "${TRAIN_TASKS}" "${VAL_TASKS}" \
    --output_path "${OUTPUT_PATH}"

echo ""
echo "Statistics saved to: ${OUTPUT_PATH}"

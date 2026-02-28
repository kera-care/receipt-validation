#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# Extract all unique normalized drug names from task files
# ──────────────────────────────────────────────────────────────────────

TASK_FILES="${TASK_FILES:-dataset/train_tasks.json dataset/dev_tasks.json}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/all_drug_names.json}"

echo "============================================="
echo "  Extract Drug Names"
echo "============================================="
echo "  Task files:  ${TASK_FILES}"
echo "  Output:      ${OUTPUT_PATH}"
echo "============================================="

# shellcheck disable=SC2086
poetry run python -m glm_ocr_finetune.extract_drug_names \
    --task_files ${TASK_FILES} \
    --output_path "${OUTPUT_PATH}"

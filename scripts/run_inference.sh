#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# GLM-OCR Distributed Inference Script
# ──────────────────────────────────────────────────────────────────────

# ── Paths ────────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-outputs/glm-ocr-finetune/final_model}"
DATASET_PATH="${DATASET_PATH:-dataset/dev_tasks.json}"
IMAGES_ROOT_DIR="${IMAGES_ROOT_DIR:-/mnt/datadrive/vision-llm-finetune-data/images/prod-prescriptions}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/inference_results.json}"

HF_CACHE_DIR="/mnt/datadrive/vision-llm-finetune-data/hf-cache"
mkdir -p "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"

# ── Model ────────────────────────────────────────────────────────────
MAX_PIXELS="${MAX_PIXELS:-1048576}"
IMAGE_SIZE="${IMAGE_SIZE:-1024}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
BATCH_SIZE="${BATCH_SIZE:-1}"

# ── Accelerate ───────────────────────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-8}"
if [ "${NUM_GPUS}" -gt 1 ] 2>/dev/null; then
    ACCEL_CONFIG="configs/accelerate_multi_gpu.yaml"
else
    ACCEL_CONFIG="configs/accelerate_single_gpu.yaml"
fi
ACCEL_CONFIG="${ACCEL_CONFIG_FILE:-$ACCEL_CONFIG}"

echo "============================================="
echo "  GLM-OCR Inference"
echo "============================================="
echo "  GPUs:             ${NUM_GPUS}"
echo "  Accelerate cfg:   ${ACCEL_CONFIG}"
echo "  Model:            ${MODEL_PATH}"
echo "  Dataset:          ${DATASET_PATH}"
echo "  Images root:      ${IMAGES_ROOT_DIR}"
echo "  Output:           ${OUTPUT_PATH}"
echo "  Batch size/GPU:   ${BATCH_SIZE}"
echo "  Max new tokens:   ${MAX_NEW_TOKENS}"
echo "============================================="

poetry run accelerate launch \
    --config_file "${ACCEL_CONFIG}" \
    --num_processes "${NUM_GPUS}" \
    -m glm_ocr_finetune.inference \
    --model_path "${MODEL_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --images_root_dir "${IMAGES_ROOT_DIR}" \
    --output_path "${OUTPUT_PATH}" \
    --max_pixels "${MAX_PIXELS}" \
    --image_size "${IMAGE_SIZE}" \
    --torch_dtype "${TORCH_DTYPE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --batch_size "${BATCH_SIZE}" \
    --skip_missing_images

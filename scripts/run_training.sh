#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# GLM-OCR Fine-Tuning Launch Script
# ──────────────────────────────────────────────────────────────────────

# ── Paths (edit these) ───────────────────────────────────────────────
IMAGES_ROOT_DIR="${IMAGES_ROOT_DIR:-/path/to/images}"
TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH:-dataset/train_tasks.json}"
EVAL_DATASET_PATH="${EVAL_DATASET_PATH:-dataset/dev_tasks.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/glm-ocr-finetune}"

# ── Model ────────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-zai-org/GLM-OCR}"
MAX_PIXELS="${MAX_PIXELS:-1048576}"
IMAGE_SIZE="${IMAGE_SIZE:-1024}"

# ── Training hyperparams ─────────────────────────────────────────────
NUM_EPOCHS="${NUM_EPOCHS:-3}"
PER_DEVICE_TRAIN_BS="${PER_DEVICE_TRAIN_BS:-2}"
PER_DEVICE_EVAL_BS="${PER_DEVICE_EVAL_BS:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
EVAL_STEPS="${EVAL_STEPS:-200}"
SAVE_STEPS="${SAVE_STEPS:-200}"

# ── Accelerate ───────────────────────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-8}"
if [ "${NUM_GPUS}" -gt 1 ] 2>/dev/null; then
    ACCEL_CONFIG="configs/accelerate_multi_gpu.yaml"
else
    ACCEL_CONFIG="configs/accelerate_single_gpu.yaml"
fi
ACCEL_CONFIG="${ACCEL_CONFIG_FILE:-$ACCEL_CONFIG}"

echo "============================================="
echo "  GLM-OCR Fine-Tuning"
echo "============================================="
echo "  GPUs:             ${NUM_GPUS}"
echo "  Accelerate cfg:   ${ACCEL_CONFIG}"
echo "  Model:            ${MODEL_PATH}"
echo "  Images root:      ${IMAGES_ROOT_DIR}"
echo "  Train dataset:    ${TRAIN_DATASET_PATH}"
echo "  Eval dataset:     ${EVAL_DATASET_PATH}"
echo "  Output dir:       ${OUTPUT_DIR}"
echo "  Epochs:           ${NUM_EPOCHS}"
echo "  Per-device BS:    ${PER_DEVICE_TRAIN_BS}"
echo "  Grad accum:       ${GRAD_ACCUM_STEPS}"
echo "  Effective BS:     $((PER_DEVICE_TRAIN_BS * GRAD_ACCUM_STEPS * NUM_GPUS))"
echo "  Learning rate:    ${LEARNING_RATE}"
echo "============================================="

accelerate launch \
    --config_file "${ACCEL_CONFIG}" \
    --num_processes "${NUM_GPUS}" \
    -m glm_ocr_finetune.train \
    --model_path "${MODEL_PATH}" \
    --images_root_dir "${IMAGES_ROOT_DIR}" \
    --train_dataset_path "${TRAIN_DATASET_PATH}" \
    --eval_dataset_path "${EVAL_DATASET_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_pixels "${MAX_PIXELS}" \
    --image_size "${IMAGE_SIZE}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BS}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BS}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --max_length "${MAX_LENGTH}" \
    --logging_steps "${LOGGING_STEPS}" \
    --eval_steps "${EVAL_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --gradient_checkpointing \
    --assistant_only

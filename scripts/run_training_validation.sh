#!/usr/bin/env bash
set -euo pipefail


required_python_version="3.12"
current_python_version=$(poetry run python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [ "$current_python_version" != "$required_python_version" ]; then
    echo "Error: Python $required_python_version is required, but current version is $current_python_version."
    echo "Please set poetry to use Python $required_python_version and try again."
    exit 1
fi

poetry lock && poetry install

poetry run pip install flash-attn --no-build-isolation


# ──────────────────────────────────────────────────────────────────────
# GLM-OCR Fine-Tuning Launch Script
# ──────────────────────────────────────────────────────────────────────

mount_point="/mnt/datadrive"

data_directory="${mount_point}/vision-llm-finetune-data"

# ── Paths (edit these) ───────────────────────────────────────────────
TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH:-${data_directory}/prescription_validation/merged_dataset/train_tasks.json}"

EVAL_DATASET_PATH="${EVAL_DATASET_PATH:-${data_directory}/prescription_validation/merged_dataset/val_tasks.json}"

HF_CACHE_DIR="${HF_CACHE_DIR:-/mnt/datadrive/vision-llm-finetune-data/hf-cache}"
mkdir -p "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"

# ── Model ────────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-zai-org/GLM-OCR}"
MAX_PIXELS="${MAX_PIXELS:-1048576}"
IMAGE_SIZE="${IMAGE_SIZE:-1024}"

# ── LoRA (set USE_LORA=true to enable) ────────────────────────────────
USE_LORA="${USE_LORA:-true}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"

# ── Training hyperparams ─────────────────────────────────────────────
# Defaults change depending on USE_LORA (LoRA benefits from a higher LR)
NUM_EPOCHS="${NUM_EPOCHS:-10}"
PER_DEVICE_TRAIN_BS="${PER_DEVICE_TRAIN_BS:-1}"
PER_DEVICE_EVAL_BS="${PER_DEVICE_EVAL_BS:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
EVAL_STEPS="${EVAL_STEPS:-200}"
SAVE_STEPS="${SAVE_STEPS:-200}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
STDOUT_LOG="logs/training_${TIMESTAMP}_stdout.log"
STDERR_LOG="logs/training_${TIMESTAMP}_stderr.log"

if [ "${USE_LORA}" = "true" ]; then
    LEARNING_RATE="${LEARNING_RATE:-5e-5}"
    OUTPUT_DIR="${OUTPUT_DIR:-outputs/glm-ocr-finetune-lora}-${NUM_EPOCHS}-epochs"
else
    LEARNING_RATE="${LEARNING_RATE:-2e-5}"
    OUTPUT_DIR="${OUTPUT_DIR:-outputs/glm-ocr-finetune}-${NUM_EPOCHS}-epochs"
fi

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
echo "  Train dataset:    ${TRAIN_DATASET_PATH}"
echo "  Eval dataset:     ${EVAL_DATASET_PATH}"
echo "  Output dir:       ${OUTPUT_DIR}"
echo "  Epochs:           ${NUM_EPOCHS}"
echo "  Per-device BS:    ${PER_DEVICE_TRAIN_BS}"
echo "  Grad accum:       ${GRAD_ACCUM_STEPS}"
echo "  Effective BS:     $((PER_DEVICE_TRAIN_BS * GRAD_ACCUM_STEPS * NUM_GPUS))"
echo "  Learning rate:    ${LEARNING_RATE}"
echo "  LoRA:             ${USE_LORA}"
if [ "${USE_LORA}" = "true" ]; then
    echo "  LoRA rank:        ${LORA_RANK}"
    echo "  LoRA alpha:       ${LORA_ALPHA}"
    echo "  LoRA dropout:     ${LORA_DROPOUT}"
fi
echo "============================================="

# Build optional flags
LORA_FLAGS=""
if [ "${USE_LORA}" = "true" ]; then
    LORA_FLAGS="--use_lora --lora_rank ${LORA_RANK} --lora_alpha ${LORA_ALPHA} --lora_dropout ${LORA_DROPOUT}"
fi

poetry run accelerate launch \
    --config_file "${ACCEL_CONFIG}" \
    --num_processes "${NUM_GPUS}" \
    -m glm_ocr_finetune.train_validation \
    --model_path "${MODEL_PATH}" \
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
    --assistant_only \
    ${LORA_FLAGS} \
    > >(tee -a "$STDOUT_LOG") \
    2> >(tee -a "$STDERR_LOG" >&2)
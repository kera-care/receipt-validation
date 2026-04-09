# GLM-OCR Fine-Tune — Receipt Validation

Fine-tune [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) for **healthcare receipt validation and total amount extraction**. Given a receipt image, the model extracts:

| Field | Type | Description |
|---|---|---|
| `is_health_receipt` | boolean | Whether the document is a health-related receipt |
| `total_amount` | string | Total amount (e.g. `"15000"`, `"3500.50"`) |
| `date` | string | Date in ISO format (`YYYY-MM-DD`) |
| `patient_name` | string \| null | Patient or customer name |
| `provider_info` | string \| null | Pharmacy / clinic name and address |
| `proof_of_payment` | string \| null | Description of payment evidence (stamp, signature, etc.) |

Includes a full data preparation pipeline from raw GCS annotations, LoRA and full fine-tuning via Accelerate, distributed inference, and per-field evaluation metrics.

---

## Project Structure

```
glm-ocr-finetune/
├── pyproject.toml
├── configs/
│   ├── accelerate_single_gpu.yaml      # Single GPU (bf16)
│   └── accelerate_multi_gpu.yaml       # Multi-GPU DDP (NCCL, A100)
├── sample/                             # Sample receipt images for local testing
├── scripts/
│   ├── configure_validation_training.sh  # Full Azure VM data-prep + training orchestration
│   ├── download_production_images.py     # Fetch annotation JSONL + images from GCS
│   ├── prepare_kera_receipts.py          # Convert raw Kera JSONL → train/val/test splits
│   ├── prepare_cord.py                   # Prepare CORD dataset (negative samples)
│   ├── prepare_coco.py                   # Prepare COCO dataset (negative samples)
│   ├── prepare_doclaynet.py              # Prepare DocLayNet dataset (negative samples)
│   ├── merge_datasets.py                 # Merge all sources → unified train/val JSON
│   ├── run_training_validation.sh        # Launch receipt validation training
│   ├── run_inference.sh                  # Launch distributed inference
│   └── test_local.py                     # Local inference test (MPS / CPU)
└── src/glm_ocr_finetune/
    ├── config.py                         # ModelConfig, LoRAConfig, DataConfig, TrainingConfig
    ├── train_validation.py               # Training entry point (HF Trainer + Accelerate)
    ├── inference.py                      # Distributed inference (torch.distributed)
    ├── evaluate.py                       # Per-field evaluation metrics
    ├── extract_receipt_stats.py          # Dataset statistics utility
    └── data/
        ├── prompts.py                    # Receipt validation prompt template
        ├── utils.py                      # normalize_amount, load_receipt_validation_datasets
        └── collator.py                   # Data collator with assistant-only loss masking
```

---

## Requirements

- Python ≥ 3.12
- [Poetry](https://python-poetry.org/) ≥ 2.0
- CUDA-capable GPUs for training (tested on 8× A100 80 GB)
- `flash-attn` (installed by the training script)

## Setup

```bash
poetry lock && poetry install
```

---

## Data Formats

### Raw Kera annotations (GCS, input to `prepare_kera_receipts.py`)

One JSON object per line:

```json
{
  "image_id":     "<uuid>",
  "image_path":   "gs://<bucket>/transaction_proof_images/...",
  "annotated_at": "2026-03-30T14:24:07Z",
  "fields": {
    "is_health_receipt": true,
    "total_amount":      15000.0,
    "patient_name":      "Jeremie Mabiala",
    "provider_info":     "Pharmacie Centrale, Dakar",
    "proof_of_payment":  "cachet PAYÉ",
    "date":              "2026-02-05"
  }
}
```

The following attributes may not be present in the actual annotation, so they can be set to Null:
```json
patient_name, provider_info, proof_of_payment
```

### Prepared receipt tasks (output of `prepare_kera_receipts.py`)

Written to `train.jsonl`, `validation.jsonl`, `test.jsonl`:

```json
{
  "transaction_id":  "<uuid>",
  "image_urls":      ["transaction_proof_images/..."],
  "annotated_at":    "2026-03-30T14:24:07Z",
  "is_health_receipt": true,
  "total_amount":    "15000",
  "patient_name":    "Jean Dupont",
  "provider_info":   "Pharmacie Centrale, Dakar",
  "proof_of_payment": "cachet PAYÉ",
  "date":            "2026-02-05"
}
```

`image_urls` contains GCS paths relative to the bucket root (`gs://` prefix and bucket name stripped).

### Merged dataset (output of `merge_datasets.py`)

Each record additionally has:

```json
{
  "image_paths": ["/absolute/path/to/transaction_proof_images/..."],
  "source":      "kera_receipt | cord | coco | doclaynet"
}
```

CORD, COCO, and DocLayNet records are negative samples (`is_health_receipt: false`).

---

## Data Preparation Pipeline

The full pipeline is orchestrated by `scripts/configure_validation_training.sh` and runs on an Azure VM with a mounted data drive.

```
GCS bucket (annotations + images)
        │
        ▼
download_production_images.py    ← fetches annotation JSONL + images from GCS
        │
        ▼
prepare_kera_receipts.py         ← JSONL → train.jsonl / validation.jsonl / test.jsonl
prepare_cord.py  /  prepare_coco.py  /  prepare_doclaynet.py   ← negative samples
        │
        ▼
merge_datasets.py                ← merges all sources → train_tasks.json + val_tasks.json
        │
        ▼
run_training_validation.sh       ← launches fine-tuning
```

### Running the full pipeline

```bash
export IMAGES_BUCKET_NAME="kera-production.appspot.com"
export GCP_SA_KEY_PATH=/path/to/secrets.json

# Optional overrides
export ANNOTATIONS_GCS_PREFIX="annotated_image_data/receipts/v_20260402_013946/"
export DATA_DIRECTORY=/mnt/datadrive/vision-llm-finetune-data

bash scripts/configure_validation_training.sh
```

### Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `IMAGES_BUCKET_NAME` | **yes** | — | GCS bucket containing receipt images |
| `GCP_SA_KEY_PATH` | **yes** | — | Path to GCP service account JSON key (`secrets.json`) |
| `ANNOTATIONS_GCS_PREFIX` | no | `annotated_image_data/receipts/v_YYYYMMDD_HHMMSS/` | GCS prefix for annotation JSONL files |
| `DATA_DIRECTORY` | no | `/mnt/datadrive/vision-llm-finetune-data` | Root data directory on the VM |

### `prepare_kera_receipts.py`

Converts raw annotation JSONL into clean train / validation / test splits (90/10 train–val).

```bash
poetry run python scripts/prepare_kera_receipts.py \
  --train_jsonl /data/annotations/receipts/train.jsonl \
  --test_jsonl  /data/annotations/receipts/test.jsonl \
  --output_dir  /data/receipt_dataset
```

Flags: `--train_jsonl` (required), `--test_jsonl` (optional), `--output_dir` (default: `receipt_dataset`), `--val_ratio` (default: `0.1`), `--seed` (default: `42`).

### `merge_datasets.py`

Merges Kera receipts with CORD, COCO, and DocLayNet negative samples. Resolves relative `image_urls` to absolute `image_paths`.

```bash
poetry run python scripts/merge_datasets.py \
  --cord_dir                  /data/cord \
  --coco_dir                  /data/coco \
  --doclaynet_dir             /data/doclaynet \
  --kera_receipt_splits_dir   /data/receipt_dataset \
  --kera_receipt_dir          /data/images/prod-receipts \
  --output_dir                /data/merged_dataset
```

Output: `{output_dir}/train_tasks.json`, `{output_dir}/val_tasks.json`.

---

## Training

Supports **LoRA** (default, rank 64 / alpha 128) and **full fine-tuning**, targeting vision encoder, vision-language merger, and LM layers.

```bash
# LoRA fine-tune on 8 GPUs (default)
USE_LORA=true NUM_GPUS=8 NUM_EPOCHS=10 \
  bash scripts/run_training_validation.sh

# Full fine-tune
USE_LORA=false NUM_GPUS=8 NUM_EPOCHS=10 \
  bash scripts/run_training_validation.sh
```

Key environment variables:

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `zai-org/GLM-OCR` | Base model path or HF Hub ID |
| `TRAIN_DATASET_PATH` | `…/merged_dataset/train_tasks.json` | Training task file |
| `EVAL_DATASET_PATH` | `…/merged_dataset/val_tasks.json` | Validation task file |
| `NUM_GPUS` | `8` | Number of GPUs |
| `NUM_EPOCHS` | `10` | Training epochs |
| `USE_LORA` | `true` | Enable LoRA fine-tuning |
| `LORA_RANK` | `64` | LoRA rank |
| `LORA_ALPHA` | `128` | LoRA alpha (scaling = alpha / rank) |
| `PER_DEVICE_TRAIN_BS` | `1` | Per-device batch size |
| `GRAD_ACCUM_STEPS` | `16` | Gradient accumulation steps |
| `LEARNING_RATE` | `5e-5` (LoRA) / `2e-5` (full ft) | Learning rate |

After LoRA training the adapter is merged into base weights and saved as `final_model_merged` for dependency-free inference.

---

## Local Testing

No GPU required — runs on MPS (Apple Silicon) or CPU.

```bash
# 1. Verify tokenisation and assistant-only masking (fast, no model download)
poetry run python -m glm_ocr_finetune.data.collator

# 2. Run inference on all sample images
poetry run python scripts/test_local.py

# 2. Run inference on a single image
poetry run python scripts/test_local.py --image_path sample/receipt1.jpg --max_new_tokens 256
```

Results are printed to stdout and saved to `outputs/local_test_results.json`.

---

## Inference

Distributed inference using `DistributedSampler`. Each GPU rank writes a partial shard; rank 0 merges and deduplicates by `transaction_id`.

```bash
MODEL_PATH=outputs/glm-ocr-receipt-finetune-lora-10-epochs/final_model_merged \
  bash scripts/run_inference.sh
```

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | (required) | Fine-tuned model path |
| `LORA_PATH` | (none) | Optional LoRA adapter path (loaded on top of `MODEL_PATH`) |
| `DATASET_PATH` | `dataset/val_tasks.json` | Task file for inference |
| `NUM_GPUS` | `8` | Number of GPUs |
| `BATCH_SIZE` | `1` | Per-device batch size |
| `MAX_NEW_TOKENS` | `4096` | Maximum generation length |

Output: `outputs/inference_results.json`

---

## Evaluation

Runs per-field metrics against the inference output:

```bash
poetry run python -m glm_ocr_finetune.evaluate \
  --inference_path outputs/inference_results.json \
  --output_path    outputs/receipt_evaluation_results.json
```

| Field | Metric |
|---|---|
| `is_health_receipt` | Accuracy, precision, recall, F1 |
| `total_amount` | Exact match + 1 % tolerance match (after CFA franc normalisation) |
| `date` | Exact ISO string match accuracy |
| `patient_name` | Presence detection: precision, recall, F1 |
| `provider_info` | Presence detection: precision, recall, F1 |
| `proof_of_payment` | Presence detection: precision, recall, F1 |

`normalize_amount` handles Francophone formats (`15 000`, `15.000`, `15.000,50`, `15000 FCFA`, etc.) before comparison.

Output: `outputs/receipt_evaluation_results.json`

---

## Dataset Statistics

Inspect field coverage and amount distributions across task files:

```bash
poetry run python -m glm_ocr_finetune.extract_receipt_stats \
  --task_files dataset/train_tasks.json dataset/val_tasks.json \
  --output_path outputs/receipt_stats.json
```

Reports: receipt classification split, amount min/max/mean/median, date range, per-field presence rates, and any malformed date strings.

---

## Configuration

All hyperparameters live in `src/glm_ocr_finetune/config.py`:

| Config class | Key settings |
|---|---|
| `ModelConfig` | `zai-org/GLM-OCR`, bf16, `flash_attention_2`, max 1 M pixels |
| `LoRAConfig` | rank=64, alpha=128, dropout=0.05; targets vision encoder, merger, and LM layers |
| `DataConfig` | `max_length=4096`, `assistant_only=True` |
| `TrainingConfig` | AdamW (β₁=0.9, β₂=0.95), cosine schedule, 5 % warmup, eval/save every 200 steps |

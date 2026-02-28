# GLM-OCR Fine-Tune

Fine-tune [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) for drug name extraction from prescription images. Includes training (full fine-tune + LoRA), distributed inference, and a multi-level evaluation pipeline with root-based matching and exclusion detection.

## Project Structure

```
├── pyproject.toml                  # Poetry build config & dependencies
├── resources/
│   ├── drug_roots.json             # Drug name → roots & variants mapping (1,463 entries)
│   └── drugs_exclusion.csv         # Drug exclusion flags (drug_name, is_exclusion)
├── sample-datasets/
│   ├── train_tasks.json
│   ├── dev_tasks.json
│   └── test_tasks.json
├── scripts/
│   ├── run_training.sh             # Launch training (full ft or LoRA)
│   ├── run_inference.sh            # Launch distributed inference
│   ├── run_evaluate.sh             # Run evaluation pipeline
│   └── run_extract_drug_names.sh   # Extract unique drug names from task files
└── src/glm_ocr_finetune/
    ├── config.py                   # Model, LoRA, Data, Training configs
    ├── train.py                    # Training script (HF Trainer + Accelerate)
    ├── inference.py                # Distributed inference (torch.distributed)
    ├── evaluate.py                 # Evaluation: exact, root-match, exclusion
    ├── extract_drug_names.py       # Drug name extraction utility
    └── data/
        ├── utils.py                # normalize_drug_name, load_tasks, load_dataset
        ├── prompts.py              # Prompt templates
        └── collator.py             # DrugNameDataCollator (assistant-only masking)
```

## Requirements

- Python &ge; 3.12
- [Poetry](https://python-poetry.org/) &ge; 2.0
- CUDA-capable GPUs (tested on 8&times; A100 80GB)
- flash-attn (installed automatically by training script)

## Setup

```bash
poetry lock && poetry install
```

## Data Format

Each task file is a JSON array of objects:

```json
[
  {
    "transaction_id": "abc123",
    "prescription_image_urls": ["relative/path/to/image.jpg"],
    "verified_drug_names": ["Amoxicilline Arrow - 1g, b/30", "Doliprane - 500mg, b/16"]
  }
]
```

Images are resolved relative to the `IMAGES_ROOT_DIR` directory.

## Training

Supports **full fine-tuning** and **LoRA** (rank 64, alpha 128) targeting vision encoder, merger, and LM layers.

```bash
# Full fine-tune on 8 GPUs
NUM_GPUS=8 NUM_EPOCHS=20 IMAGES_ROOT_DIR=/path/to/images \
  bash scripts/run_training.sh

# LoRA fine-tune
USE_LORA=true NUM_GPUS=8 NUM_EPOCHS=20 IMAGES_ROOT_DIR=/path/to/images \
  bash scripts/run_training.sh
```

Key environment variables:

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `zai-org/GLM-OCR` | Base model path or HF hub ID |
| `IMAGES_ROOT_DIR` | (required) | Root directory for prescription images |
| `NUM_GPUS` | `8` | Number of GPUs to use |
| `NUM_EPOCHS` | `20` | Training epochs |
| `USE_LORA` | `false` | Enable LoRA fine-tuning |
| `PER_DEVICE_TRAIN_BS` | `1` | Per-device batch size |
| `GRAD_ACCUM_STEPS` | `32` | Gradient accumulation steps |

Default learning rates: **2e-5** (full ft) / **1e-4** (LoRA).

## Inference

Distributed inference across multiple GPUs using `DistributedSampler`. Each rank writes partial results; rank 0 merges them.

```bash
MODEL_PATH=outputs/glm-ocr-finetune-20-epochs/final_model \
IMAGES_ROOT_DIR=/path/to/images \
  bash scripts/run_inference.sh

# With LoRA adapter
LORA_PATH=outputs/lora-run/lora_adapter \
  bash scripts/run_inference.sh
```

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `outputs/.../final_model` | Fine-tuned model path |
| `LORA_PATH` | (none) | Optional LoRA adapter path |
| `DATASET_PATH` | `sample-datasets/dev_tasks.json` | Task file for inference |
| `NUM_GPUS` | `8` | Number of GPUs |
| `BATCH_SIZE` | `1` | Per-device batch size |
| `MAX_NEW_TOKENS` | `2048` | Max generation length |

Output: `<model_dir>/inference_outputs/inference_results.json`

## Evaluation

Three evaluation modes:

1. **Exact match** &mdash; character-perfect match after normalization
2. **Root match** &mdash; fuzzy-match labels and predictions against drug variants from `drug_roots.json`, resolve to entry keys, compare key sets at multiple thresholds
3. **Exclusion detection** &mdash; binary per-sample classification: does the prescription contain any exclusion drug?

```bash
MODEL_PATH=outputs/glm-ocr-finetune-20-epochs/final_model \
  bash scripts/run_evaluate.sh
```

| Variable | Default | Description |
|---|---|---|
| `INFERENCE_PATH` | `<model_dir>/inference_outputs/inference_results.json` | Inference results |
| `DRUG_ROOTS_PATH` | `resources/drug_roots.json` | Drug roots & variants mapping |
| `EXCLUSION_PATH` | `resources/drugs_exclusion.csv` | Exclusion flags |
| `FUZZY_THRESHOLDS` | `0.5 0.6 0.7 0.8 0.9` | Similarity thresholds for variant matching |

### How Root-Based Evaluation Works

Drug names from the dataset include dosage and packaging info (e.g. `"amoxicilline arrow - 1g, b/30"`). The evaluation pipeline:

1. Strips dosage info (everything after ` - `) to get the drug name part
2. Matches the name against all variants in `drug_roots.json` via fuzzy similarity
3. Maps the matched variant to its **entry key** (one canonical ID per drug)
4. Compares label keys vs. prediction keys as sets (TP/FP/FN)
5. Labels use a **fixed 0.5 threshold**; prediction thresholds vary per `--fuzzy_thresholds`

### Exclusion Detection

After root resolution, each resolved key is checked against `drugs_exclusion.csv`. Per sample:

- **TP**: both labels and predictions contain at least one exclusion drug
- **FP**: predictions contain an exclusion drug but labels don't
- **FN**: labels contain an exclusion drug but predictions don't
- **TN**: neither side has exclusion drugs

### Output Files

- `evaluation_results.json` &mdash; aggregate metrics + per-sample breakdown
- `error_analysis/threshold_<T>.json` &mdash; per-threshold error details with match chains

## Resource Files

### `drug_roots.json`

Maps 1,463 drug entries to their roots and spelling variants (5,889 total variants):

```json
{
  "amoxicilline": {
    "roots": ["amoxicilline"],
    "variants": ["amoxicilline", "amoxicillin", "amoxiciline", ...]
  }
}
```

### `drugs_exclusion.csv`

2,635 drug names with binary exclusion flags:

```csv
drug_name,is_exclusion
3chenes asthe 1000,True
ac fusidique,False
```

## Extract Drug Names

Extract all unique normalized drug names from task files:

```bash
bash scripts/run_extract_drug_names.sh
```

## Configuration

All default hyperparameters are in `src/glm_ocr_finetune/config.py`:

| Config | Key Settings |
|---|---|
| **Model** | `zai-org/GLM-OCR`, bf16, flash_attention_2, max 1M pixels |
| **LoRA** | rank=64, alpha=128, dropout=0.05, targets vision+merger+LM layers |
| **Data** | max_length=4096, assistant_only masking |
| **Training** | AdamW, cosine schedule, warmup 5%, eval/save every 200 steps |

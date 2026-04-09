# CLAUDE.md — GLM-OCR Fine-Tune for Receipt Validation

## Project Overview

This repository fine-tunes [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) for **receipt validation and amount extraction** at [Kera](https://kera.health).

**Multi-task receipt understanding:**
1. **Receipt classification** — `is_health_receipt` (boolean): Is this a health-related receipt?
2. **Amount extraction** — `total_amount` (string): Total in CFA francs (handles Francophone formats like "15.000,50")
3. **Date extraction** — `date` (ISO format YYYY-MM-DD)
4. **Entity extraction** — `patient_name`, `provider_info` (pharmacy/clinic name & address)
5. **Payment proof detection** — `proof_of_payment` (stamps, signatures, "PAYÉ", "ACQUITTÉ")

- **Author**:Jeremie Mabiala (jeremie.mabiala@kera.health) & Mitiku Yohannes (mitiku@kera.health) 
- **Package**: `glm_ocr_finetune` (Python >= 3.12, Poetry >= 2.0)
- **Model**: Vision-Language Model (vision encoder + merger + causal LM)
- **Hardware target**: A100 80GB GPUs (single or multi-GPU via Accelerate/DDP)
- **Domain**: Senegalese healthcare receipts, CFA franc (XOF) currency, French/English languages

---

## Repository Structure

```
glm-ocr-finetune/
├── pyproject.toml                        # Poetry dependencies
├── configs/
│   ├── accelerate_single_gpu.yaml        # Single GPU (bf16)
│   └── accelerate_multi_gpu.yaml         # Multi-GPU DDP (NCCL, A100)
├── sample/                               # Sample receipt images for local testing
├── scripts/
│   ├── configure_validation_training.sh  # Full Azure VM data-prep + training orchestration
│   ├── download_production_images.py     # Fetch annotation JSONL from GCS + download images
│   ├── prepare_kera_receipts.py          # Convert raw Kera JSONL → train/val/test splits
│   ├── prepare_cord.py                   # Prepare CORD dataset (negative samples)
│   ├── prepare_coco.py                   # Prepare COCO dataset (negative samples)
│   ├── prepare_doclaynet.py              # Prepare DocLayNet dataset (negative samples)
│   ├── merge_datasets.py                 # Merge all sources → unified train/val JSON
│   ├── run_training_validation.sh        # Receipt validation training launcher
│   ├── run_inference.sh                  # Distributed inference orchestrator
│   ├── run_extract_receipt_stats.sh      # Extract dataset statistics
│   └── test_local.py                     # Local inference test (MPS/CPU, 5 sample images)
└── src/glm_ocr_finetune/
    ├── config.py                         # ModelConfig, LoRAConfig, DataConfig, TrainingConfig
    ├── train_validation.py               # HF Trainer entry point (receipt validation)
    ├── inference.py                      # Distributed inference (torch.distributed)
    ├── evaluate.py                       # Per-field evaluation (classification, amount, date, entities)
    ├── extract_receipt_stats.py          # Dataset statistics utility
    ├── publish_model.py                  # HF Hub publishing utility
    ├── modelling/
    │   └── loader.py                     # Model & processor loading + configuration
    └── data/
        ├── utils.py                      # normalize_amount, load_receipt_validation_datasets
        ├── prompts.py                    # Receipt validation prompt templates
        └── collator.py                   # PrescriptionValidationCollator (assistant-only masking)
```

---

## Technology Stack

| Layer | Library | Version |
|-------|---------|---------|
| Model loading / training | `transformers` | `>=5.2.0,<6.0.0` |
| Distributed training | `accelerate` | `>=1.7.0,<2.0.0` |
| Parameter-efficient FT | `peft` (LoRA) | `>=0.14.0,<1.0.0` |
| Datasets | `datasets` | `>=4.6.0,<5.0.0` |
| Image augmentation | `kornia` | `>=0.8.2,<0.9.0` |
| Fuzzy matching | `thefuzz`, `jellyfish` | `>=0.22.0`, `>=1.2.1` |
| JSON repair | `json-repair` | `>=0.58.1,<0.59.0` |
| Structured logging | `structlog` | `>=25.5.0,<26.0.0` |
| Config validation | `pydantic` | `>=2.12.5,<3.0.0` |
| Attention kernel | `flash-attn` | installed separately |

---

## Data Formats

### Kera production annotations (raw JSONL)

Raw annotation files fetched from GCS (one JSON per line):

```json
{
  "image_id":     "<uuid>",
  "image_path":   "gs://<bucket>/transaction_proof_images/...",
  "annotated_at": "2026-03-30T14:24:07Z",
  "fields": {
    "is_health_receipt": true,
    "total_amount":      15000.0,
    "patient_name":      "Jeje ",
    "provider_info":     "Pharmacie Centrale, Dakar",
    "proof_of_payment":  "cachet PAYÉ",
    "date":              "2026-02-05"
  }
}
```

The following attributes were not annoted:  `patient_name, provider_info, proof_of_payment`, so they can't be ignored in training by setting them to `Null`.

### Prepared receipt tasks (after `prepare_kera_receipts.py`)

Output JSONL written to `train.jsonl`, `validation.jsonl`, `test.jsonl`:

```json
{
  "transaction_id":  "<uuid>",
  "image_urls":      ["transaction_proof_images/..."],
  "annotated_at":    "2026-03-30T14:24:07Z",
  "is_health_receipt": true,
  "total_amount":   "15000",
  "patient_name":   "Jeje ",
  "provider_info":  "Pharmacie Centrale, Dakar",
  "proof_of_payment": "cachet PAYÉ",
  "date":            "2026-02-05"
}
```

`image_urls` contains GCS paths relative to the bucket root (i.e. `gs://` prefix and bucket name are stripped).

### Merged dataset (after `merge_datasets.py`)

Each record additionally has:

```json
{
  "image_paths": ["/absolute/path/to/transaction_proof_images/..."],
  "source":      "cord|coco|doclaynet|kera_receipt"
}
```

---

## Multi-Task Data Preparation Pipeline

The full pipeline is orchestrated by `scripts/configure_validation_training.sh` and runs on an Azure VM. CORD, COCO, and DocLayNet serve as **negative samples** (non-receipt documents) for the receipt validation task.

```
GCS bucket (annotations + images)
        │
        ▼
download_production_images.py    ← fetches annotation JSONL + images from GCS
        │
        ▼
prepare_kera_receipts.py         ← JSONL → train.jsonl / validation.jsonl / test.jsonl
prepare_cord.py  /  prepare_coco.py  /  prepare_doclaynet.py  ← negative samples
        │
        ▼
merge_datasets.py                ← merges all → train_tasks.json + val_tasks.json
        │
        ▼
run_training_validation.sh       ← launches receipt validation training
```

### Key environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `IMAGES_BUCKET_NAME` | **yes** | — | GCS bucket containing receipt images (e.g. `kera-production.appspot.com`) |
| `GCP_SA_KEY_PATH` | **yes** | — | Path to GCP service account JSON key (`secrets.json`) |
| `ANNOTATIONS_GCS_PREFIX` | no | `annotated_image_data/receipts/v_YYYYMMDD_HHMMSS/` | GCS prefix for annotation JSONL files |
| `DATA_DIRECTORY` | no | `/mnt/datadrive/vision-llm-finetune-data` | Root data directory on the VM |

## Key Configurations (`src/glm_ocr_finetune/config.py`)

### ModelConfig
- `torch_dtype = "bfloat16"` — optimal for A100
- `attn_implementation = "flash_attention_2"` — required for efficiency
- `max_pixels = 1_048_576`, `image_size = 1024`
- `gradient_checkpointing = True`

### LoRAConfig
- `rank = 64`, `alpha = 128` (scaling = 2.0), `dropout = 0.05`
- `learning_rate = 1e-4` (higher than full fine-tune)
- Target modules span all 8 learning objectives: vision encoder attention (`attn.qkv`, `attn.proj`), vision-language merger (`merger.proj`), LM self-attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`), MLP layers (`gate_proj`, `up_proj`, `gate_up_proj`, `down_proj`)

### TrainingConfig
- `learning_rate = 2e-5` (full fine-tune), override by `LoRAConfig.learning_rate` when LoRA is active
- `bf16 = True`, `tf32 = True`
- `eval_steps = 200`, `save_steps = 200`, `save_total_limit = 3`
- `load_best_model_at_end = True`, `metric_for_best_model = "eval_loss"`
- Optimizer: AdamW (β₁=0.9, β₂=0.95, ε=1e-8)
- Scheduler: cosine with 5% warmup
- `remove_unused_columns = False` — critical for the custom collator

---

## Training Modes

### Receipt validation (LoRA, recommended)

Uses `train_validation.py` and the merged dataset produced by the data pipeline. Launched via:

```bash
bash scripts/run_training_validation.sh
```

Or as part of the full orchestration:

```bash
IMAGES_BUCKET_NAME=kera-production.appspot.com \
  bash scripts/configure_validation_training.sh
```

After LoRA training the adapter is merged and saved as `final_model_merged` for dependency-free inference.

---

## Inference

Distributed inference uses `torch.distributed` + `DistributedSampler`. Each GPU rank writes a partial shard to `.glm_inference_tmp/shard_<rank>.json`; rank 0 merges and deduplicates by `transaction_id`.

Model outputs are repaired with `json_repair` before parsing (handles malformed JSON). Falls back to null fields on parse failure.

---

## Evaluation Modes

Six evaluation tasks in `evaluate.py`:

1. **Receipt classification** (`is_health_receipt`) — binary classification accuracy, precision, recall, F1
2. **Amount extraction** (`total_amount`) — exact match + 1% tolerance match (handles minor OCR errors in CFA amounts)
3. **Date extraction** (`date`) — ISO format match (YYYY-MM-DD)
4. **Patient name presence** (`patient_name`) — binary detection (present vs null)
5. **Provider info presence** (`provider_info`) — binary detection (pharmacy/clinic name extracted)
6. **Proof of payment detection** (`proof_of_payment`) — binary detection (stamps, signatures, "PAYÉ", "ACQUITTÉ")

Output: `receipt_evaluation_results.json` with aggregate metrics + first 50 amount errors for manual review.

---

## Code Style & Conventions

### Type hints
All public functions and classes must have full type annotations. Use Python 3.10+ syntax: `list[str]`, `dict[str, Any]`, `str | None` (not `Optional[str]`).

### Logging
Use `structlog` for all logging. Do not use the standard `logging` module or `print`. Log key operations with context (e.g., batch sizes, model paths, metrics).

### Configuration
All hyperparameters live in the dataclass configs in `config.py`. Do not hardcode values in training, inference, or evaluation scripts. Shell scripts accept environment variable overrides.

### Error handling
- Validate file paths explicitly with informative `FileNotFoundError`
- Do not use bare `except:` — catch specific exceptions
- For model output parsing, use `json_repair` before `json.loads`; fall back gracefully

### No secrets in code
- No API keys, tokens, or internal paths hardcoded — use environment variables
- HF tokens passed via CLI flag `--token`, never embedded in source

### Line length
Maximum **120 characters**.

---

## PR Review Checklist

When reviewing any PR in this repo, check all of the following:

### Code Quality
- [ ] Type hints on all public functions and classes (use `list[str]`, not `List[str]`)
- [ ] No unused imports, variables, or dead code
- [ ] `structlog` used for logging; no bare `print()` or `logging.basicConfig()`
- [ ] Line length <= 120 characters
- [ ] Configs go in `config.py` dataclasses, not scattered as magic numbers
- [ ] `remove_unused_columns = False` preserved if touching `TrainingArguments`
- [ ] `json_repair` used before `json.loads` for model output parsing

### Security
- [ ] No credentials, HF tokens, or internal paths hardcoded — use `os.getenv` or CLI args
- [ ] No sensitive data (patient info, receipt details beyond test samples) logged or exposed
- [ ] Inputs validated at system boundaries (task file loading, image path resolution)
- [ ] No `assert` used outside tests for security-critical checks

### ML Correctness
- [ ] Assistant-only masking preserved in `PrescriptionValidationCollator` (user/image tokens are masked from loss)
- [ ] LoRA target modules consistent with the 8 learning objectives documented in `LoRAConfig`
- [ ] `bf16 = True` and `attn_implementation = "flash_attention_2"` preserved
- [ ] `DistributedSampler` used in inference (not default DataLoader sampler) for multi-GPU
- [ ] Amount normalization (`normalize_amount`) applied consistently across train/inference/evaluate
- [ ] Amount normalization handles CFA franc formats ("15.000,50", "FCFA", thousand separators, commas)
- [ ] Shard merging in inference deduplicates by `transaction_id`
- [ ] Multi-task pipeline: `IMAGES_BUCKET_NAME` read from env var, never hardcoded
- [ ] Multi-task pipeline: `image_paths` (absolute) added correctly in `merge_datasets.py`; `image_urls` (relative) preserved

### API & Interface Consistency
- [ ] New CLI arguments have defaults and are documented in the relevant `argparse` section
- [ ] Shell scripts pass new arguments through env vars consistently
- [ ] Output file paths follow existing conventions (`outputs/`, `evaluation_results.json`, etc.)

### Testing
- [ ] There is currently no test suite. If tests are added, they must be under `tests/` mirroring `src/glm_ocr_finetune/`
- [ ] Tests must not make real model inference calls or download weights
- [ ] Use `monkeypatch` / `unittest.mock` to stub HF model loading and file I/O

### Documentation
- [ ] Docstrings updated for any changed public function signatures
- [ ] `CLAUDE.md` updated if new models, configs, or architectural decisions are introduced
- [ ] `README.md` updated if usage workflow or script arguments change

---

## PR Types & Guidelines

### Fix PRs
- Scope: single bug, single file or minimal surface area
- Must not change hyperparameters unless the bug is a wrong default
- Include a comment explaining the root cause, not just the fix
- Title format: `fix: <short description>` (e.g., `fix: correct assistant token masking in collator`)

### Feature PRs
- Scope: new capability added to an existing pipeline stage (e.g., new evaluation mode, new augmentation)
- New config fields go in the appropriate dataclass in `config.py` with sensible defaults (must not break existing runs)
- New resources go under `resources/`
- Shell scripts updated to expose new flags via env vars
- Title format: `feat: <short description>` (e.g., `feat: add fuzzy date matching with tolerance`)

### Chore PRs
- Scope: dependency updates, CI changes, refactoring without behavior change
- Dependency version bumps must respect the range constraints in `pyproject.toml`
- Do not mix behavior changes with chore changes
- Title format: `chore: <short description>` (e.g., `chore: bump transformers to 5.3.0`)

---

## CI/CD

### `claude.yml` — On-demand Claude Code
Triggers on `@claude` mentions in PR comments, issue comments, and PR reviews. Claude can read CI results, diff PRs, and create inline comments.

### `claude_review.yml` — Automated PR Reviews
Runs on every PR open/synchronize/ready-for-review/reopen. Claude reviews with inline comments and a top-level summary. Uses this `CLAUDE.md` for project context and conventions.

Allowed tools for both workflows:
```
mcp__github_inline_comment__create_inline_comment
Bash(gh pr comment:*)
Bash(gh pr diff:*)
Bash(gh pr view:*)
Bash(gh issue create:*)
```

---

## Common Operations

```bash
# Install
poetry install
poetry run pip install flash-attn --no-build-isolation

# Train (LoRA, 8 GPUs) — receipt validation
IMAGES_BUCKET_NAME=kera-production.appspot.com \
  GCP_SA_KEY_PATH=/path/to/secrets.json \
  bash scripts/configure_validation_training.sh

# Or just training (assumes data already prepared)
bash scripts/run_training_validation.sh

# Inference
MODEL_PATH=outputs/glm-ocr-finetune-lora-20-epochs/final_model_merged \
  bash scripts/run_inference.sh

# Evaluate
MODEL_PATH=outputs/glm-ocr-finetune-lora-20-epochs/final_model \
  bash scripts/run_evaluate.sh

# Publish to HF Hub
poetry run python -m glm_ocr_finetune.publish_model \
  --model_path outputs/glm-ocr-finetune-lora-20-epochs/final_model \
  --hub_model_id KeraCare/receipt_validation_v1x0 \
  --token $HF_TOKEN
```

---

## Architecture Notes

- **Vision encoder**: 24 attention blocks, `attn.qkv` (fused 1024→3072) + `attn.proj` (1024→1024)
- **Vision-language merger**: `merger.proj` (1536→1536 cross-modal projection)
- **Language model**: 16 layers, attention `q_proj`/`k_proj`/`v_proj`/`o_proj`, MLP `gate_up_proj`/`down_proj`
- LoRA rank 64 / alpha 128 gives a scaling factor of 2.0 — do not change without re-evaluating convergence
- `max_length = 4096` tokens covers multi-image receipts; increasing this has VRAM implications
- Batch size 2 per device × gradient accumulation 8 × N GPUs = effective batch size of 16N

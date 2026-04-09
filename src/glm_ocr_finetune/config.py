from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Model and processor configuration."""
    model_path: str = "zai-org/GLM-OCR"
    use_fast_processor: bool = False
    torch_dtype: str = "bfloat16"  # bfloat16 is optimal for A100
    max_pixels: int = 1_048_576
    image_size: int = 1024
    gradient_checkpointing: bool = True
    attn_implementation: Optional[str] = "flash_attention_2"


@dataclass
class LoRAConfig:
    """LoRA adapter configuration for parameter-efficient fine-tuning.

    Objectives:
    1. Handwriting / print diversity: capture varied document layouts and fonts
    2. Background noise: robustly extract text features despite visual clutter
    3. Orientations / zoom levels: handle different text orientations and scales
    4. Blurriness / scan quality: adapt to varying image sharpness and quality
    5. Receipt vocabulary: learn domain-specific terms (amounts, dates, provider names)
    6. Amount parsing: extract and normalise CFA franc totals in varied formats
    7. Cross-modal alignment: link visual stamps/signatures to structured fields
    8. OCR artifacts: compensate for typical OCR errors in both vision and language components

    Target modules are selected to address all 8 learning objectives:

    Vision encoder attention (24 blocks)  — objectives 1-4, 8
        1. Layout / font diversity         → spatial attention patterns
        2. Background noise                → noise-robust feature extraction
        3. Orientations / zoom levels      → spatial-transform attention
        4. Scan quality                    → adaptive sharpness features
        8. OCR artifacts                   → robust visual encoding

    Vision-language merger                 — objectives 7, 8
        7. Cross-modal alignment           → stamp/signature → structured fields
        8. OCR artifacts                   → vision→language bridge

    Language model self-attention (16 layers) — objectives 5-8
        5. Receipt vocabulary              → token-level domain knowledge
        6. Amount parsing                  → contextual numeral normalisation
        7. Cross-modal alignment           → cross-attend to visual context
        8. OCR artifacts                   → language-side compensation

    Language model MLP (16 layers)         — objectives 5, 6
        5. Receipt vocabulary              → factual / lexical knowledge
        6. Amount parsing                  → numeral format generalisation
    """
    rank: int = 64
    alpha: int = 128                        # alpha = 2 × rank is a common default
    dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    learning_rate: float = 1e-4             # higher than full fine-tune (2e-5)

    target_modules: list[str] = field(default_factory=lambda: [
        # ── Vision encoder attention (24 blocks) ──────────────────────
        "attn.qkv",                         # fused Q/K/V (1024→3072)
        "attn.proj",                        # attention output (1024→1024)
        # ── Vision-language merger ────────────────────────────────────
        "merger.proj",                      # cross-modal projection (1536→1536)
        # ── Language model self-attention (16 layers) ─────────────────
        "q_proj",                           # query  (1536→2048)
        "k_proj",                           # key    (1536→1024)
        "v_proj",                           # value  (1536→1024)
        "o_proj",                           # output (2048→1536)
        # ── MLP gate/up — vision blocks + merger + LM ─────────────────
        "gate_proj",                        # vision + merger gate (1024/1536→4096/4608)
        "up_proj",                          # vision + merger up   (also catches LM gate_up_proj)
        "gate_up_proj",                     # LM fused gate+up     (1536→9216)
        # ── MLP down — all three sub-networks ────────────────────────
        "down_proj",                        # vision + merger + LM down
    ])


@dataclass
class DataConfig:
    """Dataset paths and loading options."""
    train_dataset_path: str = "dataset/train_tasks.json"
    eval_dataset_path: str = "dataset/dev_tasks.json"
    images_root_dir: str = ""
    validate_image_paths: bool = True
    max_length: int = 4096
    assistant_only: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters tuned for full fine-tune on A100s."""
    output_dir: str = "outputs/glm-ocr-finetune"
    run_name: str = "glm-ocr-receipt-validation"

    # --- Epochs & steps ---
    num_train_epochs: int = 3
    max_steps: int = -1  # -1 means use num_train_epochs

    # --- Batch sizes ---
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8  # effective batch = 2 * 8 * num_gpus

    # --- Optimizer (AdamW) ---
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # --- Scheduler ---
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05

    # --- Precision ---
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True

    # --- Logging ---
    logging_steps: int = 10
    logging_first_step: bool = True
    report_to: str = "tensorboard"

    # --- Evaluation ---
    eval_strategy: str = "steps"
    eval_steps: int = 200

    # --- Checkpointing ---
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # --- Misc ---
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    seed: int = 42
    remove_unused_columns: bool = False  # critical for custom collator
    ddp_find_unused_parameters: bool = False
    dataloader_persistent_workers: bool = True


    resume_from_checkpoint: bool | None = None  # path to checkpoint or True to auto-resume from latest

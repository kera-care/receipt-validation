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
    run_name: str = "glm-ocr-drug-extraction"

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

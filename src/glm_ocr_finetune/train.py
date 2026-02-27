"""
Full fine-tuning script for zai-org/GLM-OCR on drug name extraction.

Launch with Accelerate:
    accelerate launch --config_file configs/accelerate_config.yaml \
        -m glm_ocr_finetune.train \
        --images_root_dir /path/to/images \
        [--train_dataset_path dataset/train_tasks.json] \
        [--eval_dataset_path dataset/dev_tasks.json] \
        [--output_dir outputs/glm-ocr-finetune]
"""

import argparse
import os
import torch
import structlog
from transformers import TrainingArguments, Trainer

from glm_ocr_finetune.config import ModelConfig, DataConfig, TrainingConfig
from glm_ocr_finetune.modelling.loader import load_base_model
from glm_ocr_finetune.data.utils import load_drug_name_extraction_dataset
from glm_ocr_finetune.data.collator import DrugNameDataCollator

logger = structlog.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune GLM-OCR for drug name extraction")

    # --- Model ---
    parser.add_argument("--model_path", type=str, default=ModelConfig.model_path)
    parser.add_argument("--torch_dtype", type=str, default=ModelConfig.torch_dtype)
    parser.add_argument("--max_pixels", type=int, default=ModelConfig.max_pixels)
    parser.add_argument("--image_size", type=int, default=ModelConfig.image_size)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=ModelConfig.gradient_checkpointing)
    parser.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing")
    parser.add_argument("--attn_implementation", type=str, default=ModelConfig.attn_implementation)

    # --- Data ---
    parser.add_argument("--train_dataset_path", type=str, default=DataConfig.train_dataset_path)
    parser.add_argument("--eval_dataset_path", type=str, default=DataConfig.eval_dataset_path)
    parser.add_argument("--images_root_dir", type=str, required=True)
    parser.add_argument("--validate_image_paths", action="store_true", default=DataConfig.validate_image_paths)
    parser.add_argument("--max_length", type=int, default=DataConfig.max_length)
    parser.add_argument("--assistant_only", action="store_true", default=DataConfig.assistant_only)

    # --- Training hyperparams ---
    parser.add_argument("--output_dir", type=str, default=TrainingConfig.output_dir)
    parser.add_argument("--run_name", type=str, default=TrainingConfig.run_name)
    parser.add_argument("--num_train_epochs", type=int, default=TrainingConfig.num_train_epochs)
    parser.add_argument("--max_steps", type=int, default=TrainingConfig.max_steps)
    parser.add_argument("--per_device_train_batch_size", type=int, default=TrainingConfig.per_device_train_batch_size)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=TrainingConfig.per_device_eval_batch_size)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=TrainingConfig.gradient_accumulation_steps)
    parser.add_argument("--learning_rate", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--weight_decay", type=float, default=TrainingConfig.weight_decay)
    parser.add_argument("--adam_beta1", type=float, default=TrainingConfig.adam_beta1)
    parser.add_argument("--adam_beta2", type=float, default=TrainingConfig.adam_beta2)
    parser.add_argument("--max_grad_norm", type=float, default=TrainingConfig.max_grad_norm)
    parser.add_argument("--lr_scheduler_type", type=str, default=TrainingConfig.lr_scheduler_type)
    parser.add_argument("--warmup_ratio", type=float, default=TrainingConfig.warmup_ratio)
    parser.add_argument("--logging_steps", type=int, default=TrainingConfig.logging_steps)
    parser.add_argument("--eval_steps", type=int, default=TrainingConfig.eval_steps)
    parser.add_argument("--save_steps", type=int, default=TrainingConfig.save_steps)
    parser.add_argument("--save_total_limit", type=int, default=TrainingConfig.save_total_limit)
    parser.add_argument("--dataloader_num_workers", type=int, default=TrainingConfig.dataloader_num_workers)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--report_to", type=str, default=TrainingConfig.report_to)

    return parser.parse_args()


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_str, torch.bfloat16)


def main():
    args = parse_args()
    logger.info("Starting GLM-OCR fine-tuning", **vars(args))

    # ------------------------------------------------------------------ #
    # 1. Load model & processor
    # ------------------------------------------------------------------ #
    torch_dtype = get_torch_dtype(args.torch_dtype)

    processor, model = load_base_model(
        model_path=args.model_path,
        torch_dtype=torch_dtype,
        device_map=None,  # let Accelerate handle device placement
        max_pixels=args.max_pixels,
        image_size=args.image_size,
        use_fast=False,
    )

    # Full fine-tune: ensure all parameters are trainable
    model.train()
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model parameters",
        trainable=f"{trainable:,}",
        total=f"{total:,}",
        pct=f"{100 * trainable / total:.1f}%",
    )

    # Gradient checkpointing to reduce VRAM
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        logger.info("Gradient checkpointing enabled")

    # ------------------------------------------------------------------ #
    # 2. Load datasets
    # ------------------------------------------------------------------ #
    train_dataset = load_drug_name_extraction_dataset(
        dataset_path=args.train_dataset_path,
        images_root_dir=args.images_root_dir,
        validate_image_paths=args.validate_image_paths,
        skip_missing_images=True,
    )

    eval_dataset = None
    if args.eval_dataset_path and os.path.exists(args.eval_dataset_path):
        eval_dataset = load_drug_name_extraction_dataset(
            dataset_path=args.eval_dataset_path,
            images_root_dir=args.images_root_dir,
            validate_image_paths=args.validate_image_paths,
            skip_missing_images=True,
        )

    logger.info(
        "Datasets loaded",
        train_size=len(train_dataset),
        eval_size=len(eval_dataset) if eval_dataset else 0,
    )

    # ------------------------------------------------------------------ #
    # 3. Collator
    # ------------------------------------------------------------------ #
    collator = DrugNameDataCollator(
        processor=processor,
        max_length=args.max_length,
        assistant_only=args.assistant_only,
    )

    # ------------------------------------------------------------------ #
    # 4. Training arguments
    # ------------------------------------------------------------------ #
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,

        # Epochs / steps
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,

        # Batch
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # Optimizer
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        max_grad_norm=args.max_grad_norm,

        # Scheduler
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,

        # Precision – bfloat16 is ideal for A100s
        bf16=True,
        fp16=False,
        tf32=True,

        # Logging
        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to=args.report_to,

        # Evaluation
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.eval_steps if eval_dataset else None,

        # Checkpointing
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False if eval_dataset else None,

        # Dataloader
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=args.dataloader_num_workers > 0,

        # Misc
        seed=args.seed,
        remove_unused_columns=False,  # critical: custom collator needs all columns
        ddp_find_unused_parameters=False,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
    )

    # ------------------------------------------------------------------ #
    # 5. Trainer
    # ------------------------------------------------------------------ #
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    # ------------------------------------------------------------------ #
    # 6. Train
    # ------------------------------------------------------------------ #
    logger.info("Launching training")
    train_result = trainer.train()

    # ------------------------------------------------------------------ #
    # 7. Save final model & metrics
    # ------------------------------------------------------------------ #
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    processor.save_pretrained(os.path.join(args.output_dir, "final_model"))

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Run final evaluation
    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    logger.info("Training complete", output_dir=args.output_dir)


if __name__ == "__main__":
    main()

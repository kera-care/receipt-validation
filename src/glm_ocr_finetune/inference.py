"""
Distributed inference script for fine-tuned GLM-OCR model.

Uses torch.distributed + DistributedSampler to shard data across all GPUs.
Each rank writes its own partial results; rank 0 merges them at the end.

Launch with Accelerate:
    accelerate launch --config_file configs/accelerate_multi_gpu.yaml \
        -m glm_ocr_finetune.inference \
        --model_path outputs/glm-ocr-finetune/final_model \
        --images_root_dir /path/to/images \
        --dataset_path dataset/dev_tasks.json \
        --output_path outputs/inference_results.json
"""

import argparse
import json
import os
import tempfile
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
import structlog

from glm_ocr_finetune.modelling.loader import load_base_model
from glm_ocr_finetune.data.utils import load_drug_name_extraction_dataset
from glm_ocr_finetune.data.prompts import DRUG_NAME_EXTRACTION_PROMPTS

logger = structlog.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run distributed inference with fine-tuned GLM-OCR")

    # Model
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model checkpoint")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max_pixels", type=int, default=1_048_576)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")

    # Data
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to tasks JSON (e.g. dev_tasks.json)")
    parser.add_argument("--images_root_dir", type=str, required=True)
    parser.add_argument("--validate_image_paths", action="store_true", default=False)
    parser.add_argument("--no_validate_image_paths", action="store_false", dest="validate_image_paths")
    parser.add_argument("--skip_missing_images", action="store_true", default=True)

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=1, help="Per-GPU batch size for inference")

    # Output
    parser.add_argument("--output_path", type=str, default="outputs/inference_results.json")

    return parser.parse_args()


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def get_rank_and_world():
    """Return (local_rank, world_size). Works with or without torchrun/accelerate."""
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def build_inference_messages(task: dict, prompt: str) -> list[dict]:
    """Build the user-only messages list for generation (no assistant turn)."""
    image_contents = [
        {"type": "image", "url": image_path}
        for image_path in task["image_paths"]
    ]
    image_contents.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": image_contents}]


def collate_for_inference(batch: list[dict], processor, prompt: str, max_new_tokens: int):
    """Prepare a batch for generation. Returns (inputs_on_cpu, metadata_list)."""
    messages_list = []
    metadata = []
    for task in batch:
        messages = build_inference_messages(task, prompt)
        messages_list.append(messages)
        metadata.append({
            "transaction_id": task["transaction_id"],
            "verified_drug_names": task["verified_drug_names"],
            "prescription_image_urls": task["prescription_image_urls"],
        })

    inputs = processor.apply_chat_template(
        messages_list,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    return inputs, metadata


@torch.inference_mode()
def run_inference(args):
    rank, world_size = get_rank_and_world()
    is_main = rank == 0
    device = torch.device(f"cuda:{rank}")

    if is_main:
        logger.info("Starting distributed inference", world_size=world_size, model_path=args.model_path)

    # ------------------------------------------------------------------ #
    # 1. Load model & processor
    # ------------------------------------------------------------------ #
    torch_dtype = DTYPE_MAP.get(args.torch_dtype, torch.bfloat16)

    processor, model = load_base_model(
        model_path=args.model_path,
        torch_dtype=torch_dtype,
        device_map=None,
        max_pixels=args.max_pixels,
        image_size=args.image_size,
        use_fast=False,
        log_messages=is_main,
    )
    model.to(device)
    model.eval()

    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Model loaded", total_params=f"{total_params:,}")

    # ------------------------------------------------------------------ #
    # 2. Load dataset
    # ------------------------------------------------------------------ #
    dataset = load_drug_name_extraction_dataset(
        dataset_path=args.dataset_path,
        images_root_dir=args.images_root_dir,
        validate_image_paths=args.validate_image_paths,
        skip_missing_images=args.skip_missing_images,
    )

    if is_main:
        logger.info("Dataset loaded", size=len(dataset))

    # ------------------------------------------------------------------ #
    # 3. Distributed sampler & dataloader
    # ------------------------------------------------------------------ #
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    else:
        sampler = SequentialSampler(dataset)

    # We use batch_size=1 collation at DataLoader level and handle
    # batching manually so we can pass through metadata cleanly.
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=lambda batch: batch,  # pass raw dicts through
        num_workers=2,
        pin_memory=False,
    )

    prompt = DRUG_NAME_EXTRACTION_PROMPTS["short"]

    # ------------------------------------------------------------------ #
    # 4. Run generation
    # ------------------------------------------------------------------ #
    all_results: list[dict] = []
    total_batches = len(dataloader)
    t_start = time.time()

    for batch_idx, batch in enumerate(dataloader):
        inputs, metadata = collate_for_inference(batch, processor, prompt, args.max_new_tokens)

        # Move tensors to device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
        )

        # Decode only the newly generated tokens
        for i, meta in enumerate(metadata):
            output_ids = generated_ids[i][input_len:]
            generated_text = processor.decode(output_ids, skip_special_tokens=True)
            result = {
                **meta,
                "generated_text": generated_text,
            }
            all_results.append(result)

        if is_main and (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - t_start
            logger.info(
                "Inference progress",
                batch=f"{batch_idx + 1}/{total_batches}",
                samples=len(all_results),
                elapsed=f"{elapsed:.1f}s",
            )

    elapsed = time.time() - t_start
    logger.info(
        "Rank finished inference",
        rank=rank,
        num_results=len(all_results),
        elapsed=f"{elapsed:.1f}s",
    )

    # ------------------------------------------------------------------ #
    # 5. Gather results from all ranks → rank 0 merges & saves
    # ------------------------------------------------------------------ #
    if world_size > 1:
        # Each rank writes to a temp file, rank 0 reads & merges
        tmp_dir = tempfile.mkdtemp(prefix="glm_inference_")
        shard_path = os.path.join(tmp_dir, f"shard_{rank}.json")
        with open(shard_path, "w") as f:
            json.dump(all_results, f, ensure_ascii=False)

        dist.barrier()

        if is_main:
            merged: list[dict] = []
            for r in range(world_size):
                rpath = os.path.join(tmp_dir, f"shard_{r}.json")
                with open(rpath, "r") as f:
                    merged.extend(json.load(f))
                os.remove(rpath)
            os.rmdir(tmp_dir)

            # De-duplicate by transaction_id (DistributedSampler may pad)
            seen = set()
            unique_results = []
            for item in merged:
                tid = item["transaction_id"]
                if tid not in seen:
                    seen.add(tid)
                    unique_results.append(item)

            _save_results(unique_results, args.output_path)
    else:
        _save_results(all_results, args.output_path)


def _save_results(results: list[dict], output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved", path=output_path, total=len(results))


def main():
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()

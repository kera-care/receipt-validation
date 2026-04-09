"""
Quick local inference test against sample/ images.
No GPU required — runs on MPS (Apple Silicon) or CPU.

Usage:
    poetry run python scripts/test_local.py
    poetry run python scripts/test_local.py --images_dir sample/ --image_size 512
"""
import argparse
import json
import time
import torch
from pathlib import Path
from PIL import Image
from json_repair import repair_json

from glm_ocr_finetune.modelling.loader import load_base_model
from glm_ocr_finetune.data.prompts import RECEIPT_VALIDATION_PROMPTS
import structlog

logger = structlog.get_logger(__name__)


# -------------------------------------------------------------------------------
#  CLI
# -------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="zai-org/GLM-OCR")
    parser.add_argument("--images_dir", type=str, default="sample/")
    parser.add_argument("--image_path", type=str, default=None,
                        help="Path to a single image. When set, --images_dir is ignored.")
    parser.add_argument("--max_pixels", type=int, default=262_144,  # 512×512 — fits on MPS/CPU
                        help="Lower than production (1_048_576) to fit in local memory")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    return parser.parse_args()


# -------------------------------------------------------------------------------
#  Main
# -------------------------------------------------------------------------------

def main():
    args = parse_args()

    # Device: MPS on Apple Silicon, else CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Running on device", device=str(device))

    torch_dtype = torch.float32
    processor, model = load_base_model(
        model_path=args.model_path,
        torch_dtype=torch_dtype,
        device_map=None,                  # we place manually below
        max_pixels=args.max_pixels,
        image_size=args.image_size,
        use_fast=False,
    )
    model.to(device)
    model.eval()

    # Collect sample images
    if args.image_path is not None:
        single = Path(args.image_path)
        if not single.exists():
            raise FileNotFoundError(f"Image not found: {single}")
        image_paths = [single]
    else:
        images_dir = Path(args.images_dir)
        image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
        if not image_paths:
            raise FileNotFoundError(f"No .jpg/.png images found in {images_dir}")

    logger.info("Found images", count=len(image_paths))
    prompt = RECEIPT_VALIDATION_PROMPTS["short"]
    results = []

    for image_path in image_paths:
        logger.info("Processing", image=image_path.name)
        image = Image.open(image_path).convert("RGB")

        messages = [[{
            "role": "user",
            "content": [
                {"type": "image", "url": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }]]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )

        inputs = processor(
            text=text,
            images=[[image]],
            return_tensors="pt",
            truncation=True,
            max_length=4096,
            padding="longest",
        ).to(device)

        inputs.pop("token_type_ids", None)  # GLM-OCR doesn't use this

        t0 = time.time()
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        elapsed = time.time() - t0

        input_len = inputs["input_ids"].shape[1]
        output_text = processor.decode(
            generated_ids[0][input_len:], skip_special_tokens=True
        )

        try:
            parsed = json.loads(repair_json(output_text))
        except json.JSONDecodeError:
            parsed = {"error": "failed to parse", "raw": output_text}

        result = {"image": image_path.name, "elapsed_s": round(elapsed, 2), "output": parsed}
        results.append(result)
        print(f"\n{'='*60}")
        print(f"Image: {image_path.name}")
        print(f"Time:  {elapsed:.1f}s")
        print(json.dumps(parsed, indent=2, ensure_ascii=False))

    # Write all results to a file
    out_path = Path("outputs/local_test_results.json")
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved", path=str(out_path))


if __name__ == "__main__":

    main()
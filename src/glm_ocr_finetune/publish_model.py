from transformers import AutoProcessor, AutoModelForImageTextToText
from glm_ocr_finetune.modelling.loader import load_base_model
import torch

import argparse
import structlog
logger = structlog.get_logger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description="Publish a fine-tuned GLM-OCR model to Hugging Face Hub")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model directory")
    parser.add_argument("--hub_model_id", type=str, default="KeraCare/drug_name_extraction_v2x0", help="Hugging Face Hub model ID (e.g., username/model_name)")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face Hub token with write permissions")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=DTYPE_MAP.keys(), help="Torch data type for model loading")
    parser.add_argument("--max_pixels", type=int, default=None, help="Maximum number of pixels for image resizing")
    parser.add_argument("--image_size", type=int, default=None, help="Image size for model input")
    return parser.parse_args()

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def main():
    args = get_args()
    processor = AutoProcessor.from_pretrained(args.model_path)

    model_readme = f"""
# Fine-tuned GLM-OCR Model
This model is a fine-tuned version of the GLM-OCR model, trained for drug name extraction from prescription images. It was fine-tuned on a custom dataset of prescription images and corresponding drug name labels.


## Usage
To use this model for inference, you can load it using the Hugging Face Transformers library:

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
processor = AutoProcessor.from_pretrained("{args.hub_model_id}")
model = AutoModelForImageTextToText.from_pretrained("{args.hub_model_id}")

messages = [
    {{
        "role": "user",
        "content": [
            {{
                "type": "image",
                "image": "path_to_your_prescription_image.jpg"
            }},
            {{
                "type": "text",
                "text": "Extract drug names from the image in json format with the following format: {{\"drug_names\": [\"drug_name1\", \"drug_name2\", ...]}}"
            }}
        ],
    }}
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
    enable_thinking=False,
)

outputs = model.generate(**inputs, max_new_tokens=128)

```

## Training Details
- Base Model: GLM-OCR
- Fine-tuning Dataset: Custom dataset of prescription images and drug name labels

"""
    logger.info("Saving model README", model_readme=model_readme)
    with open(f"{args.model_path}/README.md", "w") as f:
        f.write(model_readme)

    logger.info("Loading model for publishing", model_path=args.model_path)
    torch_dtype = DTYPE_MAP.get(args.torch_dtype, torch.bfloat16)

    processor, model = load_base_model(
        model_path=args.model_path,
        torch_dtype=torch_dtype,
        device_map=None,
        max_pixels=args.max_pixels,
        image_size=args.image_size,
        use_fast=False,
    )

    logger.info("Publishing model to Hugging Face Hub", hub_model_id=args.hub_model_id)
    processor.push_to_hub(args.hub_model_id, token=args.token)
    model.push_to_hub(args.hub_model_id, token=args.token)
    logger.info("Model published successfully", hub_model_id=args.hub_model_id)

if __name__ == "__main__":
    main()
from transformers import AutoProcessor
from glm_ocr_finetune.modelling.loader import load_base_model
import torch

import argparse
import structlog
logger = structlog.get_logger(__name__)


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

def get_args():
    parser = argparse.ArgumentParser(description="Publish a fine-tuned GLM-OCR model to Hugging Face Hub")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model directory")
    parser.add_argument("--hub_model_id", type=str, default="KeraCare/receipt_validation_v1x0", help="Hugging Face Hub model ID (e.g., username/model_name)")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face Hub token with write permissions")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=DTYPE_MAP.keys(), help="Torch data type for model loading")
    parser.add_argument("--max_pixels", type=int, default=None, help="Maximum number of pixels for image resizing")
    parser.add_argument("--image_size", type=int, default=None, help="Image size for model input")
    return parser.parse_args()



def main():
    args = get_args()
    processor = AutoProcessor.from_pretrained(args.model_path)

    model_readme = f"""
# Fine-tuned GLM-OCR Model for Receipt Validation
This model is a fine-tuned version of the GLM-OCR model, trained for receipt validation and total amount extraction from Senegalese healthcare receipts. It was fine-tuned on a custom dataset of healthcare receipt images with multi-task annotations.

## Model Description
The model performs multi-task receipt understanding:
- **Receipt classification**: Determines if the document is a health receipt
- **Amount extraction**: Extracts total amount in CFA francs (XOF)
- **Metadata extraction**: Date, receipt number, patient name, provider info
- **Itemized list**: Extracts line items with quantities and prices
- **Validation flags**: Proof of payment, provider info presence, amount consistency, handwritten modifications

## Usage
To use this model for inference, you can load it using the Hugging Face Transformers library:

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

processor = AutoProcessor.from_pretrained("{args.hub_model_id}")
model = AutoModelForImageTextToText.from_pretrained("{args.hub_model_id}")

messages = [
    {{
        "role": "user",
        "content": [
            {{
                "type": "image",
                "image": "path_to_your_receipt_image.jpg"
            }},
            {{
                "type": "text",
                "text": \"\"\"Given a document image, extract the following information in JSON format:
- is_health_receipt: A boolean indicating whether the image is a receipt.
- total_amount: The total amount as a string (e.g. "12500", "3500.00"), or null if not found.
- date: The date in ISO format (YYYY-MM-DD), or null.
- receipt_number: The receipt number, or null.
- patient_name: The patient or customer name, or null.
- provider_info: The provider or merchant name and address, or null.
- proof_of_payment: Description of payment evidence (stamp, signature), or null.
\"\"\"
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

outputs = model.generate(**inputs, max_new_tokens=512)
output_text = processor.decode(outputs[0], skip_special_tokens=True)
```

## Expected Output Format
```json
{{
    "is_health_receipt": true,
    "total_amount": "15000",
    "date": "2026-02-05",
    "receipt_number": "2026-134347",
    "patient_name": "Moustapha Cissé",
    "provider_info": "Clinique Example, Dakar",
    "proof_of_payment": "PAYÉ stamp",
    "has_proof_of_payment": true,
    "has_provider_info": true,
    "itemized_list": [...],
    "amounts_consistent": true,
    "has_handwritten_modifications": false
}}
```

## Training Details
- **Base Model**: GLM-OCR (zai-org/GLM-OCR)
- **Fine-tuning Method**: LoRA (rank=64, alpha=128)
- **Fine-tuning Dataset**: Custom dataset of Senegalese healthcare receipts + negative samples (CORD, COCO, DocLayNet)
- **Target Domain**: Healthcare receipts from Senegal (CFA franc currency)
- **Languages**: French, English
- **Image Resolution**: 1024x1024 pixels (max 1,048,576 pixels)

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
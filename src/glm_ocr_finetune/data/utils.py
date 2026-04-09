import json
import os
import re
from typing import Any
from datasets import Dataset as HFDataset
from structlog import get_logger

logger = get_logger(__name__)

from .prompts import RECEIPT_VALIDATION_PROMPTS

MAX_IMAGES_PER_MESSAGE = 1  # Current models only support one image per message, so we will use only the first image for now.


def normalize_amount(raw: str | float | int | None) -> float | None:
    """ Normalizes an amount by converting it to a float and removing any non-numeric characters.
    Handles formats like:
    - Numeric: 15000, 15000.0
    - String with separators: "15 000", "15.000"
    - With currency: "15000 FCFA", "15.000 F CFA", "15000 XOF"

    Args:
        raw (str | float | int | None): The raw amount to normalize.
    
    Returns:
        float | None: The normalized amount, or None if the input is invalid.
    """
    if raw is None:
        return None
    if isinstance(raw,(int, float)):
        return round(float(raw), 2)
    
    cleaned = str(raw).strip()
    cleaned = re.sub(r'\b(FCFA|F\s*CFA|XOF|CFA)\b', '', cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'[€$£¥₹]', '', cleaned).strip()

    if not cleaned:
        return None
    # Francophone/Senegal format: "15.000,50" (dot=thousands, comma=decimal)
    if "," in cleaned and "." in cleaned:
        if cleaned.rindex(',') > cleaned.rindex('.'):
            cleaned = cleaned.replace('.', '').replace(',', '.')
        else:
            cleaned = cleaned.replace(',', '')
    elif ',' in cleaned:
        parts = cleaned.split(',')
        if len(parts) == 2 and len(parts[1]) <= 2:
            cleaned = cleaned.replace(',', '.')
        else:
            cleaned = cleaned.replace(',', '')

    #  Case spaces is used as thousand separator: "15 000 -> "15000"
    cleaned = cleaned.replace(' ', '')
    try:
        return round(float(cleaned), 2)
    except ValueError:
        logger.warning("Could not parse amount", raw=raw, cleaned=cleaned)
        return None



def load_receipt_validation_tasks(
    tasks_path: str,
    validate_image_paths: bool = False,
    skip_missing_images: bool = True,
    prompt: str = RECEIPT_VALIDATION_PROMPTS["short"],
) -> list[dict[str, Any]]:
    """Load receipt validation tasks and format them as model training examples.

    Args:
        tasks_path: Path to the merged JSON task file (output of merge_datasets.py).
        validate_image_paths: Raise FileNotFoundError if an image is missing.
        skip_missing_images: Skip tasks whose images are not found on disk.
        prompt: Prompt text injected into each user message.
    Returns:
        List of task dicts with ``messages``, ``labels``, and ``image_paths`` added.
    """

    logger.info("Loading tasks", tasks_path=tasks_path, validate_image_paths=validate_image_paths, skip_missing_images=skip_missing_images, prompt=prompt)
    with open(tasks_path, "r") as f:
        tasks = json.load(f)

    output_tasks = []
    num_skipped_tasks = 0
    for task in tasks:
        current_task = task.copy()
        image_paths = []
        for image_path in current_task["image_paths"]:
            if validate_image_paths and not os.path.exists(image_path):
                logger.error("Image not found", image_path=image_path)
                raise FileNotFoundError(f"Image not found: {image_path}")
            if skip_missing_images and not os.path.exists(image_path):
                continue
            image_paths.append(image_path)

        if len(image_paths) == 0 and skip_missing_images:
            num_skipped_tasks += 1
            continue

        # Build the JSON the model learn to output
        raw_amount = task.get("total_amount")
        is_health_receipt = task.get("is_health_receipt", False)
        patient_name = task.get("patient_name", None)
        provider_info = task.get("provider_info", None)
        proof_of_payment = task.get("proof_of_payment", None)
        date = task.get("date", None)
        normalized_amount = normalize_amount(raw_amount)
        # Store as string to match the prompt schema (e.g. "12500" not 12500.0)
        if normalized_amount is None:
            amount_str = None
        elif normalized_amount == int(normalized_amount):
            amount_str = str(int(normalized_amount))
        else:
            amount_str = str(normalized_amount)

        labels = json.dumps({
            "is_health_receipt": is_health_receipt,
            "total_amount": amount_str,
            "patient_name": patient_name,
            "provider_info": provider_info,
            "proof_of_payment": proof_of_payment,
            "date": date,
        }, indent=2, ensure_ascii=False)


        image_paths = image_paths[:MAX_IMAGES_PER_MESSAGE]

        user_message_contents = [
            {
                "type": "image",
                "url": image_path
            }
            for image_path in image_paths
        ]

        user_message_contents.append({
            "type": "text",
            "text": prompt
        })

        current_task["messages"] = [
            {
                "role": "user",
                "content": user_message_contents
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": labels
                    }
                ]
            }
        ]

        current_task["labels"] = labels
        current_task["image_paths"] = image_paths

        output_tasks.append(current_task)

    logger.info("Finished loading tasks", total_tasks=len(tasks), output_tasks=len(output_tasks))
    if num_skipped_tasks > 0:
        logger.warning("Some tasks were skipped due to missing images", num_skipped_tasks=num_skipped_tasks)
    
    return output_tasks



def load_receipt_validation_datasets(
    tasks_path: str,
    validate_image_paths: bool = False,
    skip_missing_images: bool = True,
) -> HFDataset:
    """Load a receipt validation task file as a Hugging Face Dataset.

    Args:
        tasks_path: Path to the merged JSON task file (output of merge_datasets.py).
        validate_image_paths: Raise FileNotFoundError if an image is missing.
        skip_missing_images: Skip tasks whose images are not found on disk.
    Returns:
        A Hugging Face ``Dataset`` with columns: ``messages``, ``labels``,
        ``image_paths``, ``transaction_id``, ``source``, and any other
        fields preserved from the task file.
    """
    tasks = load_receipt_validation_tasks(
        tasks_path=tasks_path,
        validate_image_paths=validate_image_paths,
        skip_missing_images=skip_missing_images,
    )
    return HFDataset.from_list(tasks)
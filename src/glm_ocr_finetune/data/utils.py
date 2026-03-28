import json
import os
import re
import unicodedata
from datasets import Dataset as HFDataset
from structlog import get_logger

logger = get_logger(__name__)

from .prompts import DRUG_NAME_EXTRACTION_PROMPTS

MAX_IMAGES_PER_MESSAGE = 1  # Current models only support one image per message, so we will use only the first image for now.


def normalize_drug_name(name: str) -> str:
    """ Normalizes drug name by lowercasing, stripping white spaces, normalizing spaces and stripping accented characters.

    Args:
        name (str): The drug name to normalize.
    
    Returns:
        str: The normalized drug name.
    """

    name = name.lower().strip()
    name = re.sub(r'\s+', ' ', name)
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8')
    return name


def normalize_drug_names(names: list[str]) -> list[str]:
    """ Normalizes a list of drug names.

    Args:
        names (list[str]): The list of drug names to normalize.
    
    Returns:
        list[str]: The list of normalized drug names.
    """
    names =  [normalize_drug_name(name) for name in names]
    names = list(set(names))
    names.sort()
    return names


def load_tasks(tasks_path: str, images_root_dir: str, prompt: str, validate_image_paths: bool = True, skip_missing_images: bool = True):
    logger.info("Loading tasks", tasks_path=tasks_path, images_root_dir=images_root_dir, validate_image_paths=validate_image_paths, skip_missing_images=skip_missing_images)
    with open(tasks_path, "r") as f:
        tasks = json.load(f)
    output_tasks = []
    num_skipped_tasks = 0
    for task in tasks:
        current_task = task.copy()
        image_paths = []
        for image_url in current_task["prescription_image_urls"]:
            image_path = os.path.join(images_root_dir, image_url)
            if validate_image_paths and not os.path.exists(image_path):
                logger.error("Image not found", image_path=image_path)
                raise FileNotFoundError(f"Image not found: {image_path}")
            if skip_missing_images and not os.path.exists(image_path):
                continue
            image_paths.append(image_path)

        if len(image_paths) == 0 and skip_missing_images:
            num_skipped_tasks += 1
            continue
        labels = current_task["verified_drug_names"]
        normalized_labels = normalize_drug_names(labels)

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
                        "text": json.dumps({"drug_names": normalized_labels}, indent=2, ensure_ascii=False)
                    }
                ]
            }
        ]

        current_task["labels"] = {
            "drug_names": json.dumps({"drug_names": normalized_labels}, indent=2, ensure_ascii=False)
        }
        current_task["image_paths"] = image_paths

        output_tasks.append(current_task)

    logger.info("Finished loading tasks", total_tasks=len(tasks), output_tasks=len(output_tasks))
    if num_skipped_tasks > 0:
        logger.warning("Some tasks were skipped due to missing images", num_skipped_tasks=num_skipped_tasks)
    
    return output_tasks



def load_drug_name_extraction_dataset(
    dataset_path: str,
    images_root_dir: str,
    validate_image_paths: bool = False,
    skip_missing_images: bool = True,
) -> HFDataset:
    """Load a drug name extraction dataset as a Hugging Face Dataset.

    Args:
        dataset_path: Path to the JSON task file.
        images_root_dir: Root directory for prescription images.
        validate_image_paths: Whether to verify image files exist on disk.
        skip_missing_images: Whether to skip tasks with missing images.
    Returns:
        A Hugging Face ``Dataset`` with columns derived from the task dicts
        (e.g. ``messages``, ``labels``, ``transaction_id``, …).
    """
    tasks = load_tasks(
        dataset_path,
        images_root_dir,
        prompt=DRUG_NAME_EXTRACTION_PROMPTS["short"],
        validate_image_paths=validate_image_paths,
        skip_missing_images=skip_missing_images,
    )
    return HFDataset.from_list(tasks)




if __name__ == "__main__":
    dataset = load_drug_name_extraction_dataset(
        dataset_path="dataset/train_tasks.json",
        images_root_dir="/path/to/images",
    )
    print(f"Dataset size: {len(dataset)}")
    print("Columns:", dataset.column_names)
    sample = dataset[0]
    print("Sample messages:", sample["messages"])
    print("Sample labels:", sample["labels"])

from dataclasses import dataclass
import json
from PIL import Image
from typing import Callable, List, Dict, Any
import structlog
import torch
import kornia.augmentation as K
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image

logger = structlog.get_logger(__name__)


def get_ocr_friendly_augmentation():
    # Light, text-preserving augs for OCR
    return torch.nn.Sequential(
        K.RandomRotation(degrees=6.0, p=0.3),
        K.RandomPerspective(0.15, p=0.3),
        K.RandomGaussianBlur((3,3), (0.1, 1.2), p=0.35),
        K.RandomSharpness(sharpness=1.5, p=0.3),
        K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.05, hue=0.02, p=0.5),
        K.RandomErasing(p=0.2, scale=(0.01, 0.03), ratio=(0.2, 3.0), value=0.0),
    )

def apply_augmentation(k_aug: torch.nn.Sequential, img: Image.Image, device=torch.device("cpu")):
    x = to_tensor(img.convert("RGB")).unsqueeze(0).to(device)    # [1,3,H,W], 0..1
    with torch.no_grad():
        y = k_aug(x)
    return to_pil_image(y.squeeze(0).clamp(0,1).cpu())


@dataclass
class DrugNameDataCollator:
    processor: any
    max_length: int = 4096
    assistant_only: bool = True
    assistance_prefix: str = "<|assistant|>"
    thinking_prefix: str = "\n<think></think>\n"
    image_tokens = []
    augmentation: Callable[[Image.Image], Image.Image] | None = None


    @property
    def image_tokens(self) -> List[int]:
        return [
            self.processor.image_token,
            "<|begin_of_image|>",
            "<|end_of_image|>"
        ]


    def _ensure_single_image_per_message(self, messages: List[Dict[str, Any]]) -> str | None:
        output = []
        for message in messages:
            contents = []
            image_count = 0
            for content in message["content"]:
                if content["type"] == "image":
                    image_count += 1
                    if image_count > 1:
                        continue
                contents.append(content)
            message["content"] = contents
            output.append(message)
        return output

    def extract_image_urls(self, messages_list: List[List[Dict[str, Any]]]) -> List[str | None]:
        image_urls = []
        
        for messages in messages_list:
            image_url = None
            for message in messages:
                for content in message["content"]:
                    if content["type"] == "image":
                        image_url = content["url"]
                        break
                if image_url:
                    break
            image_urls.append(image_url)
            if image_url is None:
                logger.warning("No image found in messages", messages=messages)
        return image_urls


    def prepare_inputs(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        messages_list = [batch_item["messages"] for batch_item in batch]
        image_urls = self.extract_image_urls(messages_list)
        non_null_image_indices = [i for i, url in enumerate(image_urls) if url is not None]
        if len(non_null_image_indices) < len(image_urls):
            logger.warning(
                "Some messages have no images. This may lead to unexpected behavior.",
                total_messages=len(messages_list),
                messages_without_images=len(image_urls) - len(non_null_image_indices)
            )

        messages_list = [messages_list[i] for i in non_null_image_indices]
        messages_list = [self._ensure_single_image_per_message(messages) for messages in messages_list]

        image_urls = [image_urls[i] for i in non_null_image_indices]


        images = [Image.open(image_url) for image_url in image_urls]
        if self.augmentation is not None:
            images = [
                apply_augmentation(self.augmentation, image)
                for image in images
            ]

        return messages_list, images
        


    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        
        messages_list, images = self.prepare_inputs(batch)

        image_token_ids = self.processor.tokenizer.convert_tokens_to_ids(self.image_tokens)

        texts = self.processor.apply_chat_template(
            messages_list,
            tokenize=False,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )

        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_length
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        if not self.assistant_only:
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            for image_token_id in image_token_ids:
                labels[labels == image_token_id] = -100
        else:
            labels = self.get_assistant_labels(input_ids, attention_mask, image_token_ids)

        inputs["labels"] = labels

        return inputs
    
    def get_assistant_labels(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, image_token_ids: List[int]) -> torch.Tensor:
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        for image_token_id in image_token_ids:
            labels[labels == image_token_id] = -100

        assistant_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.assistance_prefix)

        # Compute how many tokens to skip after <|assistant|> (i.e. \n<think></think>\n)
        thinking_prefix_ids = self.processor.tokenizer.encode(
            self.thinking_prefix, add_special_tokens=False
        )
        # +1 accounts for the <|assistant|> token itself
        prefix_length = 1 + len(thinking_prefix_ids)

        assistant_positions = (input_ids == assistant_token_id).nonzero(as_tuple=True)
        for batch_idx, seq_idx in zip(*assistant_positions):
            labels[batch_idx, :seq_idx + prefix_length] = -100

        return labels

if __name__ == "__main__":

    import time, torch
    from transformers import AutoProcessor
    from glm_ocr_finetune.modelling.loader import setup_glm_processor

    model_path = "zai-org/GLM-OCR"
    
    processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    processor = setup_glm_processor(processor, log_messages=True, max_pixels=262144, image_size=512)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": "sample-images/test_image.jpg"
                },
                {
                    "type": "image",
                    "url": "sample-images/test_image.jpg"
                },
                {
                    "type": "text",
                    "text": "Extract drug names in json format with the following format: {\"drug_names\": [\"drug_name1\", \"drug_name2\", ...]}"
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "{\"drug_names\": [\"Aspirin\", \"Paracetamol\"]}"
                }
            ],
        }
    ]

    aug = get_ocr_friendly_augmentation()
    collator = DrugNameDataCollator(processor, assistant_only=True, augmentation=aug)
    batch = [{"messages": messages}]
    inputs = collator(batch)
    for key in inputs:
        logger.info(f"{key}: {inputs[key].shape}, dtype={inputs[key].dtype}")
    input_ids = inputs["input_ids"]
    input_ids_decoded = processor.batch_decode(input_ids, skip_special_tokens=False)
    # print("Decoded Input:", input_ids_decoded)

    labels = inputs["labels"]
    # Set -100 to a special token for decoding
    special_token_id = processor.tokenizer.convert_tokens_to_ids("<sop>")
    labels[labels == -100] = special_token_id
    labels_decoded = processor.batch_decode(labels, skip_special_tokens=False)
    # print("Decoded Labels:", labels_decoded)



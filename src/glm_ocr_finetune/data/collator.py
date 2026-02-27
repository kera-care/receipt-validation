
from dataclasses import dataclass
from typing import List, Dict, Any
import structlog
import torch

logger = structlog.get_logger(__name__)


@dataclass
class DrugNameDataCollator:
    processor: any
    max_length: int = 4096
    assistant_only: bool = True
    assistance_prefix: str = "<|assistant|>"
    thinking_prefix: str = "\n<think></think>\n"
    image_tokens = []


    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        messages_list = [batch_item["messages"] for batch_item in batch]
        image_tokens = [
            self.processor.image_token,
            "<|begin_of_image|>",
            "<|end_of_image|>"
        ]

        image_token_ids = self.processor.tokenizer.convert_tokens_to_ids(image_tokens)

        inputs = self.processor.apply_chat_template(
            messages_list,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False
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
    print("Image tokens: ", processor.image_token)
    print("EOS token: ", processor.eos_token)

    messages = [
        {
            "role": "user",
            "content": [
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
    collator = DrugNameDataCollator(processor, assistant_only=True)
    batch = [{"messages": messages}]
    inputs = collator(batch)
    for key in inputs:
        logger.info(f"{key}: {inputs[key].shape}, dtype={inputs[key].dtype}")
    input_ids = inputs["input_ids"]
    input_ids_decoded = processor.batch_decode(input_ids, skip_special_tokens=False)
    print("Decoded Input:", input_ids_decoded)

    labels = inputs["labels"]
    # Set -100 to a special token for decoding
    special_token_id = processor.tokenizer.convert_tokens_to_ids("<sop>")
    labels[labels == -100] = special_token_id
    labels_decoded = processor.batch_decode(labels, skip_special_tokens=False)
    print("Decoded Labels:", labels_decoded)
 


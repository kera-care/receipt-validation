from PIL import Image

from transformers import AutoProcessor, AutoModelForImageTextToText
import structlog

logger = structlog.get_logger(__name__)



def load_base_model(
    model_path="zai-org/GLM-OCR",
    device_map = None,
    torch_dtype = None,
    log_messages: bool = True,
    max_pixels: int | None = None,
    image_size: int | None = None,
    use_fast: bool = False

):
    processor = AutoProcessor.from_pretrained(model_path, use_fast=use_fast)
    model = AutoModelForImageTextToText.from_pretrained(
        pretrained_model_name_or_path=model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    if log_messages:
        logger.info("✓ Loaded GLM-OCR model", model_path=model_path)

    processor = setup_glm_processor(
        processor,
        log_messages=log_messages,
        max_pixels=max_pixels if max_pixels is not None else 1_048_576,
        image_size=image_size if image_size is not None else 1024,
    )
    return processor, model


def setup_glm_processor(processor: AutoProcessor, log_messages: bool = True, 
                        max_pixels: int = 1_048_576, image_size: int = 1024):
    """
    Configure processor for GLM-OCR model with customizable resolution.
    
    Args:
        processor: The GLM-OCR processor to configure
        is_main_process: Whether this is the main process (for logging)
        max_pixels: Maximum pixels for longest_edge (default: 1_048_576)
                   Lower values = lower resolution (e.g., 262_144 = 512x512)
        image_size: Shortest edge size in pixels (default: 1024)
                   Lower values = lower resolution (e.g., 512, 768)
    
    GLM-OCR processor structure:
    - processor.image_processor.size = {"longest_edge": max_pixels, "shortest_edge": image_size}
    
    Common presets:
    - 512x512:  max_pixels=262144,   image_size=512
    - 768x768:  max_pixels=589824,   image_size=768
    - 1024x1024: max_pixels=1048576, image_size=1024 (default)
    """
    ip = getattr(processor, "image_processor", None)
    if ip is not None:
        if hasattr(ip, 'size'):
            try:
                current_size = getattr(ip, 'size', {})
                
                # GLM-OCR uses both longest_edge and shortest_edge
                if isinstance(current_size, dict):
                    ip.size = {
                        "longest_edge": max_pixels,
                        "shortest_edge": image_size
                    }
                    if log_messages:
                        logger.info("✓ Set GLM-OCR image size", longest_edge=max_pixels, shortest_edge=image_size)
            except Exception as e:
                if log_messages:
                    logger.warning("Warning setting GLM-OCR image size", error=str(e))
        else:
            if log_messages:
                logger.warning("⚠️  GLM-OCR processor has no 'size' attribute")
    else:
        if log_messages:
            logger.warning("⚠️  No image_processor found in GLM-OCR processor")

    if not hasattr(processor, "eos_token"):
        processor.eos_token = processor.tokenizer.special_tokens_map["eos_token"]
    
    return processor




if __name__ == "__main__":
    import time, torch
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
                    "text": "Extract drug names from all images in json format with the following format: {\"drug_names\": [\"drug_name1\", \"drug_name2\", ...]}"
                }
            ],
        }
    ]
    messages_list = [
        messages,
        messages
    ]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    processor, model = load_base_model()
    processor = setup_glm_processor(processor, log_messages=True, max_pixels=262144, image_size=512)
    text = processor.apply_chat_template(
        messages_list,
        tokenize=False,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False
    )

    image = Image.open("sample-images/test_image.jpg").convert("RGB")

    inputs = processor(
        text=text,
        images=[[image], [image]],
        return_tensors="pt",
        truncation=True,
        max_length=4096,
        padding="longest",
    ).to(device)



    for key, value in inputs.items():
        logger.info(f"{key}: shape={value.shape}, dtype={value.dtype}")

    model.to(device)
    inputs.pop("token_type_ids", None)
    start = time.time()

    for key in inputs:
        logger.info("Input tensor", key=key, shape=inputs[key].shape, dtype=inputs[key].dtype)

    input_ids = inputs["input_ids"]
    input_ids_decoded = processor.batch_decode(input_ids, skip_special_tokens=False)
    print("Decoded Input:", input_ids_decoded)
    generated_ids = model.generate(**inputs, max_new_tokens=4096)
    output_text = processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(output_text)
    print(f"Inference time: {time.time() - start:.2f} seconds")

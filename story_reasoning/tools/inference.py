import argparse
import random
from io import BytesIO
from typing import List

import requests
import torch
from PIL import Image
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoModelForVision2Seq

"""
    Example script to run a visual storytelling model with LoRA adapter.
    Bash command to run the script:
        python story_reasoning/tools/inference.py --base_model daniel3303/QwenStoryteller 
        --image_urls https://example.com/image1.jpg https://example.com/image2.jpg 
        --temperature 0.3 --max_tokens 2000
"""


def load_model(base_model: str, adapter_path: str, device: str = "cuda"):
    """Load the base model and LoRA adapter."""
    processor = AutoProcessor.from_pretrained(base_model)
    model = AutoModelForVision2Seq.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device
    )

    # Load and merge LoRA adapter
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()  # Merge weights for faster inference
    model.eval()

    return model, processor


def load_images_from_urls(urls: List[str]) -> List[Image.Image]:
    """Load images from a list of URLs."""
    images = []
    for url in urls:
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"Error loading image from {url}: {e}")

    return images


def get_system_prompt() -> str:
    """Get the system prompt used during training."""
    return (
        "You are an AI storyteller that can analyze sequences of images and create creative narratives. "
        "First think step-by-step to analyze characters, objects, settings, and narrative structure. "
        "Then create a grounded story that maintains consistent character identity and object references across frames. "
        "Use <think></think> tags to show your reasoning process before writing the final story."
    )


def get_random_user_prompt() -> str:
    """Get a random user prompt like during training."""
    prompts = [
        "Create a story based on these images.",
        "Tell me a story that connects these images.",
        "Generate a narrative that ties these images together.",
        "Craft a cohesive story from this sequence of images.",
        "Looking at these images, what story could they tell?",
        "Can you create a narrative that explains the progression in these images?",
        "These images form a sequence - please tell the story they represent.",
        "Write a story that captures what's happening across these images.",
        "These images tell a story - what is it?",
    ]
    return random.choice(prompts)


def parse_args():
    parser = argparse.ArgumentParser(description='Run visual storytelling model with LoRA adapter')
    parser.add_argument('--base_model', type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help='Base model path or identifier')
    parser.add_argument('--adapter', type=str, default=None,
                        help='Path to LoRA adapter weights')
    parser.add_argument('--image_urls', type=str, nargs='+', required=True,
                        help='List of image URLs')
    parser.add_argument('--device', type=str, default="cuda",
                        help='Device to run model on (cuda/cpu)')
    parser.add_argument('--max_tokens', type=int, default=2000,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')

    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    # Load model and processor
    model, processor = load_model(args.base_model, args.adapter, args.device)

    # Load images from URLs
    images = load_images_from_urls(args.image_urls)

    if not images:
        raise ValueError("No valid images could be loaded from the provided URLs")

    # Prepare chat template in the exact format used during training
    messages = [
        {"role": "system", "content": get_system_prompt()}
    ]

    # Create user message with both text and images
    user_message_content = [
        {"type": "text", "text": get_random_user_prompt()}
    ]

    # Add images to the user message content
    for img in images:
        user_message_content.append({"type": "image", "image": img})

    # Add the complete user message to the messages list
    messages.append({"role": "user", "content": user_message_content})
                    
                    
    print("len images:", len(images))
    print(images)

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Generate response
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature if args.temperature > 0.0 else None,
            do_sample=args.temperature > 0.0
        )

    # Decode output
    response = processor.batch_decode(
        output[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )[0]

    # Print the response
    print("Generated Response:")
    print(response)


if __name__ == "__main__":
    main()
import argparse
import torch
from peft import PeftModel
from transformers import AutoProcessor, AutoModelForVision2Seq


def merge_lora_with_base_model(base_model_name, lora_path, output_path, dtype=torch.bfloat16):
    """
    Load a base model and its LoRA weights, merge them, and save the merged model.
    
    Args:
        base_model_name (str): HuggingFace model name or path
        lora_path (str): Path to the LoRA adapter weights
        output_path (str): Path to save the merged model
        dtype (torch.dtype): Data type for model loading (default: torch.bfloat16)
    """
    print(f"Loading base model: {base_model_name}")
    # Load the processor and base model
    processor = AutoProcessor.from_pretrained(base_model_name)

    # Load base model with specified precision
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_name,
        torch_dtype=dtype
    )

    print(f"Loading LoRA adapter from: {lora_path}")
    # Load the PEFT model with LoRA weights
    model = PeftModel.from_pretrained(base_model, lora_path)

    print("Merging LoRA weights with base model...")
    # Merge LoRA weights with base model
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    # Save the merged model and processor
    model.save_pretrained(output_path)
    processor.save_pretrained(output_path)

    print("Model merging completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights with a base model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Base model name or path (e.g., 'Qwen/Qwen2.5-VL-7B-Instruct')"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to the LoRA adapter weights"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the merged model"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Data type for model loading (default: bf16)"
    )

    args = parser.parse_args()



    # Convert dtype string to torch dtype
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Merge the models
    merge_lora_with_base_model(args.model, args.lora_path, args.output_path, dtype)


if __name__ == "__main__":
    main()
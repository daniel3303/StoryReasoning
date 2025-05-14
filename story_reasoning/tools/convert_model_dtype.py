import argparse
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq


def convert_model_dtype(model_path, output_path, dtype=torch.bfloat16):
    """
    Load a model and save it with the specified data type.
    
    Args:
        model_path (str): Path to the model to convert
        output_path (str): Path to save the converted model
        dtype (torch.dtype): Target data type for model conversion
    """
    print(f"Loading model from: {model_path}")
    # Load the processor and model
    processor = AutoProcessor.from_pretrained(model_path)

    # Load model with specified precision
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=dtype
    )

    print(f"Converting model to {dtype}")

    print(f"Saving converted model to: {output_path}")
    # Save the converted model and processor
    model.save_pretrained(output_path)
    processor.save_pretrained(output_path)

    print("Model conversion completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Convert a model to a specified data type")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Model path to convert"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the converted model"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Target data type (default: bf16)"
    )

    args = parser.parse_args()

    # Convert dtype string to torch dtype
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Convert the model
    convert_model_dtype(args.model_path, args.output_path, dtype)


if __name__ == "__main__":
    main()
import argparse
import os
from huggingface_hub import HfApi, login
from transformers import AutoProcessor, AutoModelForVision2Seq


def upload_model_to_hub(model_path, repo_name, private=False, token=None):
    """
    Upload a model to the Hugging Face Hub.
    
    Args:
        model_path (str): Path to the saved model
        repo_name (str): Name of the repository on Hugging Face
        private (bool): Whether the repository should be private
        token (str): Hugging Face token for authentication
    """
    # Login to Hugging Face Hub
    if token:
        login(token=token)
    else:
        print("No token provided. Using cached credentials or environment variables.")

    # Create HfApi instance
    api = HfApi()

    # Check if repo exists, create if not
    try:
        api.repo_info(repo_id=repo_name, repo_type="model")
        print(f"Repository {repo_name} already exists.")
    except Exception:
        print(f"Creating new repository: {repo_name}")
        api.create_repo(
            repo_id=repo_name,
            repo_type="model",
            private=private,
            exist_ok=True
        )

    print(f"Loading model from {model_path}")
    # Load model and processor to verify they can be loaded correctly
    try:
        # Load the model with the same dtype it was saved with
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype="auto"  # Preserves original precision
        )
        processor = AutoProcessor.from_pretrained(model_path)
        print("Model and processor loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Uploading model to {repo_name}")
    # Upload model and processor to Hub
    model.push_to_hub(repo_name)
    processor.push_to_hub(repo_name)

    # Upload model card if exists
    readme_path = os.path.join(model_path, "README.md")
    if os.path.exists(readme_path):
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="model"
        )
    else:
        # Create a simple README if none exists
        model_card = f"""---
language: en
license: apache-2.0
tags:
- vision-language-model
- qwen
- visual-question-answering
---

# {repo_name.split('/')[-1]}

This model is a fine-tuned version of a vision-language model.

## Model description

This model was uploaded using the Hugging Face Hub.

## How to use

```python
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

model = AutoModelForVision2Seq.from_pretrained(
    "{repo_name}",
    torch_dtype=torch.bfloat16
)
processor = AutoProcessor.from_pretrained("{repo_name}")
```
"""
        api.upload_file(
            path_or_fileobj=model_card.encode(),
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="model"
        )

    print(f"Model successfully uploaded to {repo_name}")
    print(f"View your model at: https://huggingface.co/{repo_name}")


def main():
    parser = argparse.ArgumentParser(description="Upload a model to the Hugging Face Hub")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the saved model"
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        required=True,
        help="Name of the repository on Hugging Face (e.g., 'username/model-name')"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token for authentication"
    )

    args = parser.parse_args()

    # Upload the model
    upload_model_to_hub(args.model_path, args.repo_name, args.private, args.token)


if __name__ == "__main__":
    main()
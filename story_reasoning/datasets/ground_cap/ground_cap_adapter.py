import random
from typing import Dict, Any

from torch.utils.data import Dataset

from story_reasoning.datasets import DatasetRegistry
from story_reasoning.models.object_detection.detection import Detection
from story_reasoning.train.base_tlr_dataset_adapter import BaseTlrDatasetAdapter


@DatasetRegistry.register_tlr_dataset_adapter("ground_cap")
class GroundCapAdapter(BaseTlrDatasetAdapter):
    def __init__(self, dataset: Dataset):
        BaseTlrDatasetAdapter.__init__(self, dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
           Process a single example from the dataset.
           Should return a conversational or instruction format supported by the TRL trainers
           Example:
               Conversational format:
                   {"messages": [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "..."}]}
               Instruction format:
                   {"prompt": "<prompt text>", "completion": "<ideal generated text>"}

               See https://huggingface.co/docs/trl/en/sft_trainer#dataset-format-support
       """

        item = self.dataset[index]

        # Process the example to get formatted detections and other fields
        processed_item = self.process_example(item)

        # Format detections into a string
        detection_strings = [self.format_detection(det) for det in processed_item["detections"]]
        detections_text = "\n".join(detection_strings)

        # Construct the user message with image placeholder and detections
        user_message = (
            self.get_random_user_prompt() + "\n"
            "[IMG]\n"
            f"[DETECTIONS]{detections_text}[/DETECTIONS]"
        )

        # Return in the conversation format expected by TRL
        return {
            "id": item["image_id"], # Unique ID for the sample
            "messages": [
                {"role": "system", "content": self.default_system_prompt()},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": processed_item["caption"]}
            ],
            "images": [processed_item["image"]],
            "detections": processed_item["detections"]
        }

    def default_system_prompt(self) -> str:
        return (
            "You are an AI assistant that can see and understand images. "
            "I will provide you with an image and the detected objects in it along with their positions and dimensions in the format [id, x,y,width,height]. "
            "Please generate a detailed caption that describes the scene."
        )

    def get_random_user_prompt(self) -> str:
        prompts = [
            "",
            "Please describe this image.",
            "What do you see in this image?",
            "Tell me about this image.",
            "Describe the scene in this image.",
            "What's happening in this image?",
            "Can you explain what this image shows?",
            "Provide a detailed description of this image.",
            "What are the main elements in this image?",
        ]
        return random.choice(prompts)

    def format_detection(self, detection: Detection) -> str:
        """Format a single detection for instruction."""
        return detection.to_placeholder_tag()

    def process_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single example from the dataset."""
        # Apply image processor if available
        image = example["image"]

        return {
            "image_id": example["image_id"],
            "image": image,
            "detections": example["detections"],
            "caption": example["caption"],
            "human_annotated": example["human_annotated"]
        }

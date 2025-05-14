from pathlib import Path
from typing import Dict, List, Optional, Union, Literal

from datasets import load_dataset, load_from_disk, concatenate_datasets

from story_reasoning.datasets import DatasetRegistry
from story_reasoning.datasets.base_dataset import BaseDataset


@DatasetRegistry.register_dataset("story_reasoning")
class StoryReasoningDataset(BaseDataset):
    """PyTorch Dataset for Story Reasoning data with multi-frame stories."""

    def __init__(
            self,
            path: Optional[Union[str, Path]] = None,
            hf_repo: Optional[str] = "daniel3303/StoryReasoning",
            split: Optional[Literal["train", "test"]] = None,
            transform=None,
    ):
        """
        Initialize the Story Reasoning dataset.

        Args:
            path: Path to the Arrow dataset
            hf_repo: Hugging Face Hub repository ID
            split: Dataset split to use ('train' or 'test')
            transform: Optional transform to be applied to images
        """
        self.transform = transform
        self.split = split
        self.story_id_to_dataset_sample = {}

        try:
            if path is not None:
                self.dataset = load_from_disk(path)
            else:
                self.dataset = load_dataset(hf_repo)

            if split is not None:
                if split not in ["train", "test"]:
                    raise ValueError("Split must be either 'train' or 'test'")
                if isinstance(self.dataset, dict):
                    self.dataset = self.dataset[split]
                else:
                    raise ValueError(f"Dataset doesn't contain splits but split '{split}' was requested")
            else:
                # If no split is provided, provide the full dataset
                if isinstance(self.dataset, dict) and "train" in self.dataset and "test" in self.dataset:
                    self.dataset = concatenate_datasets([
                        self.dataset["train"],
                        self.dataset["test"]
                    ])

        except Exception as e:
            raise RuntimeError(f"Error loading dataset from {path or hf_repo}: {str(e)}")

    def __len__(self) -> int:
        """Return the number of stories in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Union[List, str, int]]:
        """
        Get a single story from the dataset.

        Returns:
            Dictionary containing:
                - story_id: String identifier for the story
                - images: List of image tensors or PIL images
                - frame_count: Number of frames in the story
                - chain_of_thought: String containing structured analysis
                - story: String containing the grounded story
        """
        example = self.dataset[idx]
        
        # Get images and apply transform if needed
        images = example['images']
        if self.transform is not None:
            images = [self.transform(img) for img in images]

        return {
            'story_id': example['story_id'],
            'images': images,
            'frame_count': example['frame_count'],
            'chain_of_thought': example['chain_of_thought'],
            'story': example['story']
        }
    
    def get_by_story_id(self, story_id: str) -> Dict[str, Union[List, str, int]]:
        """
        Get a story by its ID.

        Args:
            story_id: The ID of the story to retrieve.

        Returns:
            Dictionary containing the story data or None if not found.
        """
        if not self.story_id_to_dataset_sample:
            self.story_id_to_dataset_sample = {example['story_id']: example for example in self.dataset}
            
        return self.story_id_to_dataset_sample[story_id]
        
        for example in self.dataset:
            if example['story_id'] == story_id:
                return example
        
        raise RuntimeError(f"Story {story_id} not found")

    @property
    def frame_stats(self) -> Dict[str, Union[int, float]]:
        """Get statistics about frame counts in the dataset."""
        frame_counts = [example['frame_count'] for example in self.dataset]
        return {
            'min_frames': min(frame_counts),
            'max_frames': max(frame_counts),
            'avg_frames': sum(frame_counts) / len(frame_counts),
            'total_frames': sum(frame_counts),
            'total_stories': len(frame_counts)
        }

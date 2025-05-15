"""
GroundCap PyTorch Dataset

A PyTorch Dataset class for loading and accessing the GroundCap dataset.
The dataset contains images with their corresponding captions and detected objects.


Usage:
    1. Download the dataset:
    ```python
    from datasets import load_dataset

    # Load the dataset from Hugging Face Hub
    dataset = load_dataset("daniel3303/GroundCap")

    # Save it locally (optional, but recommended for faster loading)
    dataset.save_to_disk("path/to/save/dataset")
    ```

    2. Create the PyTorch dataset:
    ```python
    from torch.utils.data import DataLoader


    # Full training set
    train_dataset = GroundCapDataset(
        path="path/to/save/dataset",
        split="train",
    )

    # Human-annotated examples only from test set
    human_test_dataset = GroundCapDataset(
        path="path/to/save/dataset",
        split="test",
        annotation_type="human",
    )

    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    # Access data
    for batch in train_loader:
        images = batch['image']           # shape: [batch_size, 3, 224, 224]
        captions = batch['caption']       # List[str]
        objects = batch['objects']        # List[List[Dict]]
        is_human = batch['human_annotated']  # shape: [batch_size]
        crops = batch['crops']            # List[List[Tensor]] (if crop_objects=True)
        ...
    ```

Dataset Structure:
    Each item in the dataset contains:
    - image: PIL Image or Tensor after transform
    - objects: List of detected objects, each with:
        - category: object class name
        - box: [x, y, width, height]
        - score: detection confidence
    - caption: String containing the image description
    - image_id: Unique identifier for the image
    - human_annotated: Boolean indicating if annotated by human

Splits:
    - train: Training split
    - test: Test split

Annotation Types:
    - 'all': Include all examples
    - 'human': Include only human-annotated examples
    - 'auto': Include only automatically annotated examples
"""

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal

import torch
from datasets import load_from_disk, load_dataset, concatenate_datasets

from story_reasoning.datasets import DatasetRegistry
from story_reasoning.datasets.base_dataset import BaseDataset
from story_reasoning.models.object_detection.detection import Detection


class AnnotationType(str, Enum):
    """Enumeration for types of annotations to include in dataset."""
    ALL = "all"  # Include all examples
    HUMAN = "human"  # Include only human-annotated examples
    AUTO = "auto"  # Include only automatically annotated examples


@DatasetRegistry.register_dataset("ground_cap")
class GroundCapDataset(BaseDataset):
    """PyTorch Dataset for GroundCap data with Detection objects."""

    def __init__(
            self,
            path: Optional[Union[str, Path]] = None,
            hf_repo: Optional[str] = None,
            split: Optional[Literal["train", "test"]] = None,
            annotation_type: Union[AnnotationType, str] = AnnotationType.ALL,
            transform=None,
            crop_objects: bool = True
    ):
        """
        Initialize the GroundCap dataset.

        Args:
            path: Path to the Arrow dataset
            hf_repo: Hugging Face Hub repository ID
            split: Dataset split to use ('train' or 'test')
            annotation_type: Type of annotations to include
            transform: Optional transform to be applied to images
            crop_objects: Whether to include cropped objects in the output
        """
        if isinstance(annotation_type, str):
            annotation_type = AnnotationType(annotation_type.lower())

        self.annotation_type = annotation_type
        self.transform = transform
        self.crop_objects = crop_objects
        self.split = split

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
                self.dataset = concatenate_datasets([
                    self.dataset["train"],
                    self.dataset["test"]
                ])
                    

            self._filter_by_annotation_type()

        except Exception as e:
            raise RuntimeError(f"Error loading dataset from {path}: {str(e)}")

    def _filter_by_annotation_type(self):
        """Filter dataset based on annotation type."""
        if self.annotation_type == AnnotationType.HUMAN:
            self.dataset = self.dataset.filter(lambda x: x['human_annotated'])
        elif self.annotation_type == AnnotationType.AUTO:
            self.dataset = self.dataset.filter(lambda x: not x['human_annotated'])

    @staticmethod
    def _convert_to_detections(objects: List[Dict]) -> List[Detection]:
        """Convert raw object dictionaries to Detection instances."""
        detections = []
        for obj in objects:
            detection = Detection(
                id=obj["id"],
                label=obj['label'],
                image_id=None,
                score=obj['score'],
                box_x=obj['box']["x"],
                box_y=obj['box']["y"],
                box_w=obj['box']["w"],
                box_h=obj['box']["h"],
                image_width=0,
                image_height=0,
                is_thing=not GroundCapDataset.is_stuff_class(obj['label'])
            )
            detections.append(detection)
        return detections

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, List, bool]]:
        """
        Get a single example from the dataset.

        Returns:
            Dictionary containing:
                - image: Tensor of shape [C, H, W]
                - detections: List of Detection objects
                - caption: String containing the caption
                - image_id: String identifier for the image
                - human_annotated: Boolean indicating annotation source
                If crop_objects is True, also includes:
                - crops: List of tensors for each cropped object
        """
        example = self.dataset[idx]
        image = example['image']  # Already a PIL Image from HF dataset

        # Convert raw objects to Detection instances
        detections = self._convert_to_detections(example['detections'])

        result = {
            'image_id': example["id"],
            'caption': example['caption'],
            'detections': detections,
            'human_annotated': example['human_annotated']
        }

        if self.crop_objects:
            # Use Detection.crop_image method for cropping
            crops = []
            for detection in detections:
                crop = detection.crop_image(image)
                if self.transform is not None:
                    crop = self.transform(crop)
                crops.append(crop)
            result['crops'] = crops

        # Transform the main image if needed
        if self.transform is not None:
            image = self.transform(image)

        result['image'] = image
        return result

    @staticmethod
    def is_stuff_class(label):
        stuff_classes = [
            "house", "light", "mirror-stuff", "net", "pillow", "platform",
            "playingfield", "railroad", "river", "road", "roof", "sand",
            "sea", "shelf", "snow", "stairs", "tent", "towel", "wall-brick",
            "wall-stone", "wall-tile", "wall-wood", "water-other",
            "window-blind", "window-other", "tree-merged", "fence-merged",
            "ceiling-merged", "sky-other-merged", "cabinet-merged",
            "table-merged", "floor-other-merged", "pavement-merged",
            "mountain-merged", "grass-merged", "dirt-merged", "paper-merged",
            "food-other-merged", "building-other-merged", "rock-merged",
            "wall-other-merged", "rug-merged"
        ]
        
        # Checks if any label in stuff_classes contains the label
        return any(label in stuff_class for stuff_class in stuff_classes)

    @property
    def categories(self) -> List[str]:
        """Get list of unique detection labels in the dataset."""
        categories = set()
        for example in self.dataset:
            for obj in example['detections']:
                categories.add(obj['category'])
        return sorted(list(categories))

    @property
    def annotation_stats(self) -> Dict[str, int]:
        """Get statistics about human and automatic annotations."""
        human_count = sum(1 for x in self.dataset if x['human_annotated'])
        total_count = len(self.dataset)
        return {
            'total': total_count,
            'human_annotated': human_count,
            'auto_annotated': total_count - human_count
        }

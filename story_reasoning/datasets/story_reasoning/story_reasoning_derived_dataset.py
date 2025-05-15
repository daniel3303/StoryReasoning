import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

from torch.utils.data import Dataset

from story_reasoning.datasets import DatasetRegistry
from story_reasoning.datasets.ground_cap import GroundCapDataset
from story_reasoning.datasets.ground_cap.ground_cap_dataset import AnnotationType
from story_reasoning.models.object_detection.detection import Detection
from story_reasoning.models.object_matching.siglip_matcher import SigLipMatcher
from story_reasoning.models.object_matching.detection_match import DetectionMatch


@DatasetRegistry.register_dataset("story_reasoning_derived")
class StoryReasoningDerivedDataset(Dataset):
    """
    Dataset for creating visual stories derived from GroundCap data.
    
    This dataset creates "stories" by grouping images from the same movie/scene.
    Each story consists of at least min_images_per_story images, with at least
    one frame containing a person detection.
    This derived story reasoning dataset does not provide any ground truth annotations, it is used to generate the final dataset.
    """

    def __init__(
            self,
            path: Optional[Union[str, Path]] = None,
            hf_repo: Optional[str] = "daniel3303/GroundCap",
            transform=None,
            min_images_per_story: int = 5,
    ):
        """
        Initialize the Story Reasoning dataset.

        Args:
            path: Path to the Arrow dataset
            hf_repo: Hugging Face Hub repository ID
            transform: Optional transform to be applied to images
            min_images_per_story: Minimum number of images per story (default: 5)
        """
        self.transform = transform
        self.min_images_per_story = min_images_per_story

        # Load the base dataset with auto annotations
        self.base_dataset = GroundCapDataset(
            path=path,
            hf_repo=hf_repo,
            annotation_type=AnnotationType.AUTO,
            transform=None,
            crop_objects=False
        )

        # Group images by movie/scene and create stories
        self.stories = self._create_stories()
        print(f"Created {len(self.stories)} stories from {len(self.base_dataset)} images")

    def _extract_movie_scene_from_id(self, image_id: str) -> Tuple[str, Optional[str]]:
        """
        Extract movie and scene information from image_id.
        
        Attempts to identify movie and scene information from the image ID,
        looking for patterns like "movie_name_scene_001" or similar.
        
        Args:
            image_id: The image identifier string
            
        Returns:
            tuple: (movie_name, scene_id) or (None, None) if extraction fails
        """
        # Pattern 1: movie_name_scene_number_...
        match = re.search(r'^(.+?)_scene_?(\d+)', image_id, re.IGNORECASE)
        if match:
            return match.group(1), match.group(2)

        # Pattern 2: movie_name_seq_number_...
        match = re.search(r'^(.+?)_seq_?(\d+)', image_id, re.IGNORECASE)
        if match:
            return match.group(1), match.group(2)

        # Pattern 3: First segment is movie name, second is scene if numeric
        parts = image_id.split("_")
        if len(parts) > 1:
            movie = parts[0]
            # Check if second part is a scene number
            if len(parts) > 1 and parts[1].isdigit():
                return movie, parts[1]
            return movie, None

        # Fallback: use the entire ID as the movie name
        return image_id, None

    def _has_person(self, detections : List[Detection]) -> bool:
        """Check if any detection is a person."""
        return any(det.label.lower() == "person" for det in detections)

    def _create_stories(self) -> List[List[int]]:
        """
        Group images into scenes and create stories.
        
        Returns:
            List of stories, where each story is a list of indices into the base dataset
        """
        total_images = len(self.base_dataset)

        # Step 1: Process all images and check for persons
        movie_scenes = defaultdict(list)

        for i in range(total_images):
            example = self.base_dataset[i]
            has_person = self._has_person(example['detections'])

            movie, scene = self._extract_movie_scene_from_id(example['image_id'])
            if movie is not None:
                # If scene is None, use a placeholder
                scene = scene if scene is not None else "default"
                movie_scenes[(movie, scene)].append({
                    'idx': i,
                    'image_id': example['image_id'],
                    'has_person': has_person
                })

        # Step 2: Create stories by combining scenes
        stories = []
        current_movie = None
        current_story = []
        current_story_has_person = False

        # Sort by movie and scene for consistent ordering
        for (movie, scene), scene_images in sorted(movie_scenes.items()):
            # Sort images within scene (assuming sequential order in image_id)
            scene_images.sort(key=lambda x: x['image_id'])
            scene_indices = [img['idx'] for img in scene_images]
            scene_has_person = any(img['has_person'] for img in scene_images)

            # If we're starting a new movie, finish the current story
            if current_movie is not None and current_movie != movie and len(current_story) > 0:
                if len(current_story) >= self.min_images_per_story and current_story_has_person:
                    stories.append(current_story)
                current_story = []
                current_story_has_person = False

            current_movie = movie

            # Add this scene to the current story
            current_story.extend(scene_indices)
            current_story_has_person = current_story_has_person or scene_has_person

            # If we have enough images and at least one person, create a story
            if len(current_story) >= self.min_images_per_story and current_story_has_person:
                stories.append(current_story[:])
                current_story = []
                current_story_has_person = False

        # Add any remaining story if it's valid
        if len(current_story) >= self.min_images_per_story and current_story_has_person:
            stories.append(current_story)

        # If we don't have enough stories or some are too short, 
        # we can merge adjacent scenes from different movies
        if len(stories) == 0:
            # Fallback: create stories from consecutive images with persons
            person_images = []
            for i in range(total_images):
                example = self.base_dataset[i]
                if self._has_person(example['detections']):
                    person_images.append(i)

            # Create stories from consecutive person images
            for i in range(0, len(person_images), self.min_images_per_story):
                end_idx = min(i + self.min_images_per_story, len(person_images))
                if end_idx - i >= self.min_images_per_story:
                    stories.append(person_images[i:end_idx])

        return stories


    def __len__(self) -> int:
        """Return the number of stories in the dataset."""
        return len(self.stories)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single story from the dataset.

        Args:
            idx: Index of the story to retrieve
            
        Returns:
            Dictionary containing:
                - images: List of image tensors
                - detections: List of lists of Detection objects
                - captions: List of caption strings
                - image_ids: List of image identifiers
                - has_person: List of booleans indicating if each image has a person
                - matches: List of DetectionMatch objects (if object matching is enabled)
        """
        story_indices = self.stories[idx]

        images = []
        detections_list = []
        captions = []
        image_ids = []
        has_person = []

        for i in story_indices:
            example = self.base_dataset[i]

            image = example['image'] # type PIL.Image.Image
            if self.transform is not None:
                image = self.transform(image)
            
            
            captions.append(example['caption'])
            image_ids.append(example['image_id'])
            has_person.append(self._has_person(example['detections']))
            images.append(image)
            
            # Sets the image width and height for each detection (this was not supported in the detection class when GroundCap was created)
            for detection in example['detections']: # type: Detection
                detection.image_width = image.size[0]
                detection.image_height = image.size[1]
            
            detections_list.append(example['detections'])

        result = {
            'images': images,
            'detections': detections_list,
            'captions': captions,
            'image_ids': image_ids,
            'has_person': has_person,
        }

        return result
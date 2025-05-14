from dataclasses import dataclass, field
from typing import List

from story_reasoning.models.story_reasoning.story_image import StoryImage


@dataclass
class Story:
    """A collection of story images in sequence."""
    images: List[StoryImage] = field(default_factory=list)
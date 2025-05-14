from dataclasses import dataclass


@dataclass
class StoryImage:
    """Represents a single image in a story with its associated text."""
    image_number: int
    text: str

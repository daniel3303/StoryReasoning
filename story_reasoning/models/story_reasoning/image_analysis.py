from dataclasses import dataclass, field
from typing import List

from story_reasoning.models.story_reasoning.character import Character
from story_reasoning.models.story_reasoning.object import Object
from story_reasoning.models.story_reasoning.setting import Setting


@dataclass
class ImageAnalysis:
    image_number: int
    characters: List[Character] = field(default_factory=list)
    objects: List[Object] = field(default_factory=list)
    settings: List[Setting] = field(default_factory=list)
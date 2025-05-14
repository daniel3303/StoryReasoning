from dataclasses import dataclass, field
from typing import List, Optional

from story_reasoning.models.story_reasoning.image_analysis import ImageAnalysis
from story_reasoning.models.story_reasoning.narrative_structure import NarrativeStructure


@dataclass
class CoT:
    images: List[ImageAnalysis] = field(default_factory=list)
    narrative_structure: Optional[NarrativeStructure] = None
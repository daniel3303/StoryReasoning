from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class FaceInfo:
    """Stores face detection information with normalized coordinates"""
    embedding: np.ndarray
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 normalized [0,1]

    def __hash__(self):
        return hash((self.confidence, self.bbox))

    def __eq__(self, other):
        if not isinstance(other, FaceInfo):
            return False
        return (self.confidence == other.confidence and
                self.bbox == other.bbox)

    def get_pixel_bbox(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Convert normalized bbox to pixel coordinates"""
        x1, y1, x2, y2 = self.bbox
        return (
            int(x1 * width),
            int(y1 * height),
            int(x2 * width),
            int(y2 * height)
        )

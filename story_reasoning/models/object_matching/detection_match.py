from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import torch

from story_reasoning.models.object_detection.detection import Detection


@dataclass
class DetectionMatch:
    """
    A class representing a group of matching detections across images.
    This also represents an Entity in the context of the story reasoning task.

    Attributes:
        id (str): A unique identifier for this match, eg char1, obj1
        label (str): Object class label
        detections (List[Detection]): List of matching detections
        clip_similarity (float): Average pairwise CLIP similarity
        face_similarity (Optional[float]): Average pairwise face similarity for person detections
        clip_std (float): Standard deviation of CLIP similarities
        face_std (Optional[float]): Standard deviation of face similarities for person detections
        avg_clip_embedding (Optional[torch.Tensor]): Average CLIP embedding for the group
        avg_face_embedding (Optional[np.ndarray]): Average face embedding for person detections
        num_faces (int): Number of face detections in the group
    """
    id: str
    label: str
    detections: List[Detection]
    clip_similarity: float = 0.0
    face_similarity: Optional[float] = None
    clip_std: float = 0.0
    face_std: Optional[float] = None
    avg_clip_embedding: Optional[torch.Tensor] = None
    avg_face_embedding: Optional[np.ndarray] = None
    num_faces: int = field(default=0)

    def update_statistics(
            self,
            clip_similarities: List[float],
            face_similarities: Optional[List[float]] = None,
            clip_embedding: Optional[torch.Tensor] = None,
            face_embedding: Optional[np.ndarray] = None,
            has_face: bool = False
    ):
        """
        Update match statistics with new detection information.

        Args:
            clip_similarities: List of CLIP similarities with existing detections
            face_similarities: Optional list of face similarities with existing detections
            clip_embedding: Optional CLIP embedding of new detection
            face_embedding: Optional face embedding of new detection
            has_face: Whether the new detection has a valid face
        """
        # Update CLIP statistics
        if clip_similarities:
            self.clip_similarity = np.mean(clip_similarities)
            self.clip_std = np.std(clip_similarities) if len(clip_similarities) > 1 else 0.0

        # Update face statistics if applicable
        if face_similarities:
            self.face_similarity = np.mean(face_similarities)
            self.face_std = np.std(face_similarities) if len(face_similarities) > 1 else 0.0

        # Update embeddings
        if clip_embedding is not None:
            if self.avg_clip_embedding is None:
                self.avg_clip_embedding = clip_embedding
            else:
                n = len(self.detections)
                self.avg_clip_embedding = (self.avg_clip_embedding * n + clip_embedding) / (n + 1)

        if face_embedding is not None and has_face:
            self.num_faces += 1
            if self.avg_face_embedding is None:
                self.avg_face_embedding = face_embedding
            else:
                self.avg_face_embedding = (self.avg_face_embedding * (
                            self.num_faces - 1) + face_embedding) / self.num_faces
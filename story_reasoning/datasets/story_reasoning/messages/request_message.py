import base64
import io
from dataclasses import dataclass, field
from typing import List, Union

from PIL import Image
from transformers import logging

# Assuming the Detection and PanopticDetector classes are available
from story_reasoning.models.object_detection.detection import Detection

logging.set_verbosity_error()



@dataclass
class RequestMessage:
    """
    A class to represent an image message with the detections.
    """
    image: Union[str, Image.Image] = field(default=None)
    detections: List[Detection] = field(default_factory=list)
    image_idx: str = field(default="0")

    def __init__(self, image: Union[str, Image.Image], detections: List[Detection], image_idx: str):
        self.image = image
        self.detections = detections
        self.image_idx = image_idx

    def _get_full_image(self) -> Image.Image:
        if isinstance(self.image, str):
            self.image = Image.open(self.image)
            self.original_image_path = self.image
            return self.image
        elif isinstance(self.image, Image.Image):
            return self.image
        else:
            raise ValueError("Image must be either a file path or a PIL Image object")

    @staticmethod
    def _encode_image(image: Image.Image):
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def to_openai_message(self, role: str = "user", image_type: str = "image_url", text_type: str = "text", include_image: bool = True):
        """
        Convert to a Python object suitable for the OpenAI API.
        """
        content = []

        if include_image:
            full_image = self._get_full_image()
            base64_image = self._encode_image(full_image)
            content.append({
                "type": image_type,
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

        # Add frame index and detected objects
        frame_info = f"Frame {self.image_idx}:\n"

        # Separate normal objects and landmarks
        normal_detections = [det for det in self.detections if not hasattr(det, 'is_landmark') or not det.is_landmark]
        landmarks = [det for det in self.detections if hasattr(det, 'is_landmark') and det.is_landmark]

        # Add object detections with global IDs and normalized coordinates
        detection_info = "Detected objects:\n"
        for det in normal_detections: # type: Detection
            x1, y1, x2, y2 = det.get_original_bbox()
            detection_info += f"- {det.get_global_id()}: {x1},{y1},{x2},{y2}\n"

        # Add landmarks if present
        landmark_info = ""
        if landmarks:
            landmark_info = "Detected landmarks:\n"
            for landmark in landmarks:
                x1, y1, x2, y2 = landmark.get_original_bbox()
                landmark_info += f"- {landmark.get_global_id()}: {landmark.get_human_readable_label()}: {x1},{y1},{x2},{y2}\n"

        content.append({
            "type": text_type,
            "text": frame_info + detection_info + landmark_info
        })

        return {
            "role": role,
            "content": content
        }

    @classmethod
    def from_file(cls, image_path: str, detections: List[Detection], frame_idx: str):
        return cls(
            image=image_path,
            detections=detections,
            image_idx=frame_idx
        )
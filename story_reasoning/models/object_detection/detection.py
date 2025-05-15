import json
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple

import PIL
import numpy as np
from PIL.Image import Image


@dataclass
class Detection:
    id: int
    label: str
    image_id: Optional[str]
    score: float
    box_x: float
    box_y: float
    box_w: float
    box_h: float
    image_width: int
    image_height: int
    mask: Optional[np.ndarray] = None  # Binary mask for the detection, this mask has the dimensions of the bounding box
    is_thing: Optional[bool] = None  # True for objects, False for stuff
    is_landmark: Optional[bool] = None  # True for landmarks, False otherwise

    def __post_init__(self):
        """Normalize label after initialization"""
        if self.label is not None:
            self.label = self._normalize_label(self.label)

    @staticmethod
    def _normalize_label(label: str) -> str:
        """Normalize label by replacing spaces and underscores with dashes"""
        return label.replace(" ", "-").replace("_", "-")

    def to_dict(self) -> dict:
        """
        Convert the Detection object to a dictionary.

        Returns:
            dict: A dictionary representation of the Detection object.
        """
        base_dict = {
            "id": self.id,
            "label": self.label,
            "score": self.score,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "box": {
                "x": self.box_x,
                "y": self.box_y,
                "w": self.box_w,
                "h": self.box_h
            }
        }
        if self.mask is not None:
            base_dict["mask"] = self.mask.tolist()
        if self.is_thing is not None:
            base_dict["is_thing"] = self.is_thing
        if self.is_landmark is not None:
            base_dict["is_landmark"] = self.is_landmark
        if self.image_id is not None:
            base_dict["image_id"] = self.image_id
        return base_dict

    def to_json(self) -> str:
        """
        Convert the Detection object to a JSON string.

        Returns:
            str: A JSON string representation of the Detection object.
        """
        return json.dumps(self.to_dict(), indent=2)

    def to_placeholder_tag(self) -> str:
        """
        Convert the Detection object to a placeholder tag string.

        Returns:
            str: A placeholder tag string representation of the Detection object.
        """
        object_id = self.get_global_id()
        return f"[{object_id}: {self.box_x:.2f},{self.box_y:.2f},{self.box_w:.2f},{self.box_h:.2f}]"


    @classmethod
    def from_dict(cls, data):
        """
        Create a Detection object from a dictionary.

        Args:
            data (dict): A dictionary containing detection data.

        Returns:
            Detection: A new Detection object.
        """
        mask = None
        if "mask" in data:
            mask = np.array(data["mask"])

        return cls(
            id=None if "id" not in data else data["id"],
            label=None if "label" not in data else data["label"],
            image_id=None if "image_id" not in data else data["image_id"],
            score=0.0 if "score" not in data else data["score"],
            image_width=0 if "image_width" not in data else data["image_width"],
            image_height=0 if "image_height" not in data else data["image_height"],
            box_x=data["box"]["x"],
            box_y=data["box"]["y"],
            box_w=data["box"]["w"],
            box_h=data["box"]["h"],
            mask=mask,
            is_thing=None if "is_thing" not in data else data["is_thing"],
            is_landmark=None if "is_landmark" not in data else data["is_landmark"]
        )


    def __eq__(self, other):
        """
        Compare two Detection objects for equality based on their label and id.
            Two detections are considered equal (the same) if their label and id are equal, independent of the bounding box.
            This because two detection can be the same object in two different images and thus the label and id should stay the
            same even if the position changes.

        Args:
            other (Detection): Another Detection object to compare with.

        Returns:
            bool: True if the bounding boxes are equal, False otherwise.
        """
        return self.id == other.id and self.label == other.label and self.image_id == other.image_id

    def __hash__(self):
        """
        Generate a hash based on the rounded bounding box coordinates.
        Useful to store Detection objects in sets or dictionaries.
        """
        return hash((self.id, self.label, self.image_id))


    def iou(self, other: 'Detection') -> float:
        """
        Calculate the Intersection over Union (IoU) with another Detection object.

        Args:
            other (Detection): Another Detection object to calculate IoU with.

        Returns:
            float: The IoU value.
        """
        x1 = max(self.box_x, other.box_x)
        y1 = max(self.box_y, other.box_y)
        x2 = min(self.box_x + self.box_w, other.box_x + other.box_w)
        y2 = min(self.box_y + self.box_h, other.box_y + other.box_h)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = self.box_w * self.box_h + other.box_w * other.box_h - intersection
        
        
        if union > 0:
            return intersection / union
        else:
            # in the special case of union == 0, we return 1 if both objects have the same x,y and 0 otherwise
            # this just happens with detections with 0 width or 0 height
            return 1 if self.box_x == other.box_x and self.box_y == other.box_y else 0

    def get_global_id(self) -> str:
        """
        Gets the global unique identifier for the detection consisting of the image identifier, label and detection id (in the same class)

        Returns:
            str: A global unique identifier for the detection.
        """
        if self.image_id is None:
            return f"{self.label}-{self.id}"
        return f"{self.image_id}-{self.label}-{self.id}".lower()



    def crop_image(self, image: Image, use_mask = False) -> Image:
        """
        Crops an image based on the detection's bounding box and mask.
        If a mask exists, applies it to the cropped region.
        """
        width, height = image.size
        left = int(self.box_x * width)
        top = int(self.box_y * height)
        right = int(math.ceil((self.box_x + self.box_w) * width))
        bottom = int(math.ceil((self.box_y + self.box_h) * height))

        cropped = image.crop((left, top, right, bottom))

        if self.mask is not None and use_mask:
            resized_mask = PIL.Image.fromarray((self.mask * 255).astype(np.uint8)).resize(
                (right - left, bottom - top),
                PIL.Image.Resampling.NEAREST
            )

            # Create RGBA image with transparency from mask
            rgba = cropped.convert('RGBA')
            mask_array = np.array(resized_mask)
            rgba_array = np.array(rgba)
            rgba_array[..., 3] = mask_array
            return PIL.Image.fromarray(rgba_array)

        return cropped

    def get_original_bbox(self) -> Tuple[int, int, int, int]:
        """
        Get bounding box coordinates in pixel format as x1,y1,x2,y2.
        
        Returns:
            str: A string representation of the pixel bounding box coordinates.
        """
        if self.image_width is None or self.image_width == 0 or self.image_height is None or self.image_height == 0:
            raise ValueError("Image dimensions are not set")
        
        x1 = int(self.box_x * self.image_width)
        y1 = int(self.box_y * self.image_height)
        x2 = int((self.box_x + self.box_w) * self.image_width)
        y2 = int((self.box_y + self.box_h) * self.image_height)
        return x1, y1, x2, y2
    
    def get_human_readable_label(self) -> str:
        """
        Get a human-readable label for the detection.
        
        Returns:
            str: A human-readable label for the detection.
        """
        return self.label.replace("-", " ").replace("_", " ").title()
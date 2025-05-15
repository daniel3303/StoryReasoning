from abc import ABC, abstractmethod
from typing import List, Union

from PIL import Image, ImageDraw, ImageFont

from story_reasoning.models.object_detection.detection import Detection


class BaseLandmarkDetector(ABC):
    """
    Abstract base class for landmark detection services.
    
    This class defines the common interface that all landmark detector implementations
    must follow, allowing for easy switching between different landmark detection services.

    """

    def __init__(self):
        pass
    
    @abstractmethod
    def analyze(self, image: Union[str, Image.Image]) -> List[Detection]:
        """
        Analyze an image and return a list of landmark detections.
        
        Args:
            image (Union[str, Image.Image]): The input image or path to the image.
            
        Returns:
            List[Detection]: A list of Detection objects representing detected landmarks.
        """
        pass

    @staticmethod
    def draw_bounding_boxes(
            image: Union[str, Image.Image],
            detections: List[Detection],
            use_id_as_label: bool = False,
            show_score: bool = True
    ) -> Image.Image:
        """
        Draw bounding boxes and labels on the input image.
        
        Args:
            image (Union[str, Image.Image]): The input image or path to the image.
            detections (List[Detection]): The list of detections to draw.
            use_id_as_label (bool): Whether to use detection IDs as labels.
            show_score (bool): Whether to show detection scores in labels.
            
        Returns:
            Image.Image: The image with bounding boxes and labels drawn on it.
        """
        if isinstance(image, str):
            image = Image.open(image)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        color = "red"  # Use consistent color for landmarks

        for detection in detections:
            x_min = int(detection.box_x * image.width)
            y_min = int(detection.box_y * image.height)
            x_max = int((detection.box_x + detection.box_w) * image.width)
            y_max = int((detection.box_y + detection.box_h) * image.height)

            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=color, width=3)

            label = str(detection.id) if use_id_as_label else detection.label
            if show_score:
                label += f": {detection.score:.2f}"

            draw.text((x_min, y_min), label, fill=color, font=font)

        return image



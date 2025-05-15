from typing import Union, List
from google.cloud import vision
import io
from PIL import Image

from story_reasoning.models.landmark_detection.base_landmark_detector import BaseLandmarkDetector
from story_reasoning.models.object_detection.detection import Detection


class GoogleLandmarkDetector(BaseLandmarkDetector):
    """
    Landmark detector implementation using Google Cloud Vision API.
    
    Args:
        credentials_path (str): Path to the Google Cloud credentials JSON file.
        detection_threshold (float): Detection confidence threshold (0-1).
        max_detections (int): Maximum number of detections to return per image.
    """

    def __init__(self, credentials_path: str, detection_threshold: float = 0.7, max_detections: int = 10):
        super().__init__()
        self.max_detections = max_detections
        self.detection_threshold = detection_threshold
        self.client = vision.ImageAnnotatorClient.from_service_account_json(credentials_path)

    def analyze(self, image: Union[str, Image.Image]) -> List[Detection]:
        """
        Analyze an image using Google Cloud Vision API.
        
        Args:
            image (Union[str, Image.Image]): The input image or path to the image.
            
        Returns:
            List[Detection]: A list of Detection objects representing detected landmarks.
        """
        # Handle both PIL Image and file path inputs
        if isinstance(image, str):
            with open(image, 'rb') as image_file:
                content = image_file.read()
            pil_image = Image.open(io.BytesIO(content))
        else:
            pil_image = image
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format=pil_image.format or 'JPEG')
            content = img_byte_arr.getvalue()

        image_width, image_height = pil_image.size

        # Create vision API image object
        vision_image = vision.Image(content=content)

        # Perform landmark detection
        response = self.client.landmark_detection(image=vision_image)
      
        # Check for API errors
        if response.error.message:
            raise Exception(
                "{}\nFor more info on error messages, check: "
                "https://cloud.google.com/apis/design/errors".format(response.error.message)
            )

        # Convert landmark annotations to Detection objects
        detections = []
        for i, landmark in enumerate(response.landmark_annotations):
            if landmark.score < self.detection_threshold:
                continue

            # Get bounding box coordinates
            vertices = landmark.bounding_poly.vertices
            # Handle empty vertex coordinates by defaulting to 0
            x_min = getattr(vertices[0], 'x', 0)
            x_max = getattr(vertices[1], 'x', image_width)
            y_min = getattr(vertices[0], 'y', 0)
            y_max = getattr(vertices[2], 'y', image_height)

            # Normalize coordinates
            box_x = x_min / image_width
            box_y = y_min / image_height
            box_w = (x_max - x_min) / image_width
            box_h = (y_max - y_min) / image_height

            detection = Detection(
                id=i,
                label=landmark.description,
                image_id=None,
                score=landmark.score,
                box_x=box_x,
                box_y=box_y,
                box_w=box_w,
                box_h=box_h,
                image_height=image_height,
                image_width=image_width,
                mask=None,
                is_thing=True
            )
            detections.append(detection)

        # Sort detections by confidence score
        detections.sort(key=lambda x: x.score, reverse=True)

        return detections[:self.max_detections]
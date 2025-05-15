from typing import Union, List
import io
from PIL import Image
import base64
import requests
import json
import re
import time

from story_reasoning.models.landmark_detection.base_landmark_detector import BaseLandmarkDetector
from story_reasoning.models.object_detection.detection import Detection
from story_reasoning.models.object_detection.panoptic_detector import PanopticDetector

class LLMLandmarkDetector(BaseLandmarkDetector):
    """
    Landmark detector implementation using OpenAI's vision models with optional panoptic detection.
    
    This detector can either use a PanopticDetector to provide candidate bounding boxes or 
    rely on the LLM to estimate landmark locations using a 100x100 grid system.
    
    Args:
        api_key (str): OpenAI API key
        base_url (str): Base URL for API calls (default is OpenAI's API URL)
        model_name (str): Name of the model to use
        use_panoptic_detector (bool): Whether to use PanopticDetector for initial bbox detection
        force_panoptic_detections (bool): Whether to force the use of provided panoptic detections
        debug (bool): Whether to print debug information about API requests and responses
    """

    def __init__(
            self,
            api_key: str,
            base_url: str = "https://api.openai.com",
            model_name: str = "gpt-4-vision-preview",
            use_panoptic_detector: bool = True,
            force_panoptic_detections: bool = True,
            debug: bool = False,
            max_tries: int = 3,
    ):
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.use_panoptic_detector = use_panoptic_detector
        self.force_panoptic_detections = force_panoptic_detections
        self.panoptic_detector = PanopticDetector(detection_threshold=0.5) if use_panoptic_detector else None
        self.debug = debug
        self.max_tries = max_tries

        # Prompt for when using panoptic detector with optional box selection
        self.prompt_with_optional_boxes = (
            "Analyze the image and identify one or more landmark/locations shown in the image. I will provide you with "
            "candidate bounding boxes. For each landmark/location that you identify:\n"
            "- First estimate its bounding box using a 100x100 grid where:\n"
            "   - (0,0) is the top-left corner of the image\n"
            "   - (99,99) is the bottom-right corner of the image\n"
            "- Then check if any of the provided bounding boxes accurately represent the landmark\n"
            "Respond with a JSON object in this format where name is the name of the landmark, bbox is your estimated coordinates, and use_box_id is the ID of a matching box (or null if none match well):\n"
            "{\n"
            '  "landmarks": [\n'
            "    {\n"
            '      "name": "Eiffel Tower",\n'
            '      "bbox": {"x1": 30, "y1": 10, "x2": 70, "y2": 90},\n'
            '      "use_box_id": "building-0"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "You can return multiple landmarks/detections. If no landmark is detected or you are not sure about what landmark is represented, respond with: {\"landmarks\": []}"
        )

        # Prompt for when not using panoptic detector
        self.prompt_without_boxes = (
            "Analyze the image and identify one or more landmark/locations shown in the image. For each landmark/location that you identify:\n"
            "- Estimate its bounding box using a 100x100 grid where:\n"
            "   - (0,0) is the top-left corner of the image\n"
            "   - (99,99) is the bottom-right corner of the image\n"
            "Respond with a JSON object in this format where name is the name of the landmark and bbox contains your estimated coordinates:\n"
            "{\n"
            '  "landmarks": [\n'
            "    {\n"
            '      "name": "Eiffel Tower",\n'
            '      "bbox": {"x1": 30, "y1": 10, "x2": 70, "y2": 90}\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "You can return multiple landmarks/detections. If no landmark is detected or you are not sure about what landmark is represented, respond with: {\"landmarks\": []}"
        )

        # Prompt for when using panoptic detector with forced box selection
        self.prompt_with_forced_boxes = (
            "Analyze the image and identify one or more landmark/locations shown in the image. I will provide you with "
            "candidate bounding boxes. For each landmark/location that you identify:\n"
            "- You MUST select one of the provided bounding boxes that best represents the landmark/location\n"
            "- The coordinate system of the bounding boxes uses a 100x100 grid where:\n"
            "   - (0,0) is the top-left corner of the image\n"
            "   - (99,99) is the bottom-right corner of the image\n"
            "Respond with a JSON object in this format where name is the name of the landmark and use_box_id is the ID of the best fitting bounding box:\n"
            "{\n"
            '  "landmarks": [\n'
            "    {\n"
            '      "name": "Eiffel Tower",\n'
            '      "use_box_id": \"building-0\"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "You can return multiple landmarks/detections. If no landmark is detected or you are not sure about what landmark is represented, respond with: {\"landmarks\": []}"
        )

    def _encode_image(self, image: Union[str, bytes]) -> str:
        """Convert image to base64 string."""
        if isinstance(image, str):
            with open(image, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        else:
            return base64.b64encode(image).decode('utf-8')

    def _convert_grid_to_normalized(self, x1: int, y1: int, x2: int, y2: int) -> tuple:
        """Convert 100x100 grid coordinates to normalized coordinates (0-1)."""
        x = x1 / 100.0
        y = y1 / 100.0
        w = (x2 - x1 + 1) / 100.0
        h = (y2 - y1 + 1) / 100.0
        return x, y, w, h

    def _parse_llm_response(self, response_text: str, max_retries: int = 2) -> dict:
        """Parse the LLM's response into a structured format."""
        retries = 0
        while retries <= max_retries:
            try:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(0)
                return json.loads(response_text)
            except json.JSONDecodeError:
                retries += 1
                if retries > max_retries:
                    return {"landmarks": []}
        return {"landmarks": []}

    def _validate_landmark_name(self, box_id: str, landmark_name: str) -> bool:
        """
        Validate if the landmark name matches the detection class from its ID.
        
        Args:
            box_id (str): The detection ID (e.g., "bridge-1")
            landmark_name (str): The landmark name provided by the LLM
            
        Returns:
            bool: True if the name is valid, False otherwise
        """
        if not box_id or "-" not in box_id:
            return True

        detection_class = box_id.rsplit("-", 1)[0]
        landmark_name_normalized = landmark_name.lower().replace(" ", "-")
        return detection_class != landmark_name_normalized

    def _prepare_payload(self, image_content: Union[str, bytes], candidate_boxes: list = None,
                         conversation_history: list = None) -> dict:
        """
        Prepare the API request payload.
        
        Args:
            image_content: The image content to analyze
            candidate_boxes: Optional list of candidate bounding boxes
            conversation_history: Optional list of previous messages in the conversation
            
        Returns:
            dict: The prepared payload for the API request
        """
        # Add system prompt
        prompt = (self.prompt_with_forced_boxes if self.force_panoptic_detections
                  else self.prompt_with_optional_boxes) if self.use_panoptic_detector else self.prompt_without_boxes
    
        if self.use_panoptic_detector and candidate_boxes:
            prompt += f"\nCandidate boxes: {json.dumps(candidate_boxes)}"
    
        messages = [
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{self._encode_image(image_content)}"
                        }
                    }
                ]
            }
        ]
    
        # Add conversation history if provided
        if conversation_history:
            messages = [messages[0]] + conversation_history + [messages[1]]
    
        return {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 300
        }
        
    
    def _get_corrected_response(self, image_content: Union[str, bytes], original_response: dict,
                                candidate_boxes: list) -> dict:
        """
        Request a corrected response from the LLM when invalid landmark names are detected.
        
        Args:
            image_content: The image content
            original_response: The original LLM response
            candidate_boxes: List of candidate bounding boxes
            
        Returns:
            dict: The corrected response
        """
        correction_prompt = (
                "Your previous response contained generic landmark names that matched the detection class IDs. "
                "Please provide the specific names of the landmarks instead. For example, if a bridge detection "
                "shows the 'Golden Gate Bridge', use that name rather than just 'Bridge'.\n\n"
                "Your previous response: " + json.dumps(original_response) + "\n\n"
                "Please provide an updated response with specific landmark names."
        )

        conversation_history = [
            {
                "role": "assistant",
                "content": json.dumps(original_response)
            },
            {
                "role": "user",
                "content": correction_prompt
            }
        ]

        payload = self._prepare_payload(image_content, candidate_boxes, conversation_history)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        retry_count = 0
        while retry_count < self.max_tries:
            try:
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
    
                if response.status_code == 200:
                    break
    
                retry_count += 1
                if retry_count < self.max_tries:
                    time.sleep(1)  # Wait 1 second before retrying
                else:
                    return original_response  # Return original if all retries fail
            except Exception as e:
                retry_count += 1
                if retry_count < self.max_tries:
                    time.sleep(1)  # Wait 1 second before retrying
                else:
                    return original_response  # Return original if all retries fail

        # noinspection PyUnboundLocalVariable
        result = response.json()
        response_text = result['choices'][0]['message']['content'].strip()
        return self._parse_llm_response(response_text)

    def analyze(self, image: Union[str, Image.Image]) -> List[Detection]:
        """
        Analyze an image to detect landmarks and their bounding boxes.
        
        Args:
            image (Union[str, Image.Image]): The input image or path to the image
            
        Returns:
            List[Detection]: A list of Detection objects representing detected landmarks
        """
        # Handle both PIL Image and file path inputs
        if isinstance(image, str):
            image_content = image
            pil_image = Image.open(image)
        else:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format=image.format or 'JPEG')
            image_content = img_byte_arr.getvalue()
            pil_image = image

        # Get panoptic detections if enabled
        candidate_boxes = []
        panoptic_detections = []
        if self.use_panoptic_detector:
            panoptic_detections = self.panoptic_detector.analyze(pil_image)
            for i, det in enumerate(panoptic_detections):
                candidate_boxes.append({
                    "id": det.get_global_id(),
                    "bbox": {
                        "x1": int(det.box_x * 100),
                        "y1": int(det.box_y * 100),
                        "x2": int((det.box_x + det.box_w) * 100),
                        "y2": int((det.box_y + det.box_h) * 100)
                    }
                })

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = self._prepare_payload(image_content, candidate_boxes)

        # Make API request
        retry_count = 0
        while retry_count < self.max_tries:
            try:
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=headers,
                    json=payload
                )

                if response.status_code == 200:
                    break

                retry_count += 1
                if retry_count < self.max_tries:
                    time.sleep(1)  # Wait 1 second before retrying
                else:
                    raise Exception(f"API request failed after {self.max_tries} attempts. Status: {response.status_code}: {response.text}")
            except Exception as e:
                retry_count += 1
                if retry_count < self.max_tries:
                    time.sleep(1)  # Wait 1 second before retrying
                else:
                    raise Exception(f"API request failed after {self.max_tries} attempts: {str(e)}")

        # noinspection PyUnboundLocalVariable
        result = response.json() 
        response_text = result['choices'][0]['message']['content'].strip()

        if self.debug:
            print("\nLLM Response:")
            print(response_text)

        parsed_response = self._parse_llm_response(response_text)

        # Check for invalid landmark names
        if self.use_panoptic_detector:
            needs_correction = False
            for landmark in parsed_response.get("landmarks", []):
                box_id = landmark.get("use_box_id")
                if box_id and not self._validate_landmark_name(box_id, landmark["name"]):
                    needs_correction = True
                    break

            if needs_correction:
                if self.debug:
                    print("\nDetected invalid landmark names, requesting correction...")
                parsed_response = self._get_corrected_response(image_content, parsed_response, candidate_boxes)
                if self.debug:
                    print("\nCorrected Response:")
                    print(json.dumps(parsed_response, indent=2))
                    
        # If passed the name checks, normalize the landmark name to fix known issues
        for landmark in parsed_response.get("landmarks", []):
            landmark["name"] = self._normalize_landmark_name(landmark["name"])

        if self.debug:
            print("\nParsed Response:")
            print(json.dumps(parsed_response, indent=2))

        detections = []
        for i, landmark in enumerate(parsed_response.get("landmarks", [])):
            if self.use_panoptic_detector:
                if self.force_panoptic_detections:
                    box_id = landmark.get("use_box_id")
                    if box_id is not None:
                        matching_det = next((det for det in panoptic_detections if det.get_global_id() == box_id), None)
                        if matching_det:
                            detection = Detection(
                                id=matching_det.id,
                                label=landmark["name"],
                                image_id=None,
                                score=1.0,
                                box_x=matching_det.box_x,
                                box_y=matching_det.box_y,
                                box_w=matching_det.box_w,
                                box_h=matching_det.box_h,
                                mask=matching_det.mask,
                                image_height=matching_det.image_height,
                                image_width=matching_det.image_width,
                                is_thing=False,
                                is_landmark=True
                            )
                            detections.append(detection)
                else:
                    if landmark.get("use_box_id") is not None:
                        box_id = landmark["use_box_id"]
                        if box_id is not None:
                            matching_det = next((det for det in panoptic_detections if det.get_global_id() == box_id), None)
                            if matching_det:
                                detection = Detection(
                                    id=matching_det.id,
                                    label=landmark["name"],
                                    image_id=None,
                                    score=1.0,
                                    box_x=matching_det.box_x,
                                    box_y=matching_det.box_y,
                                    box_w=matching_det.box_w,
                                    box_h=matching_det.box_h,
                                    mask=matching_det.mask,
                                    image_width=matching_det.image_width,
                                    image_height=matching_det.image_height,
                                    is_thing=False,
                                    is_landmark=True
                                )
                                detections.append(detection)
                    else:
                        bbox = landmark["bbox"]
                        x, y, w, h = self._convert_grid_to_normalized(
                            bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                        )
                        detection = Detection(
                            id=i,
                            label=landmark["name"],
                            image_id=None,
                            score=1.0,
                            box_x=x,
                            box_y=y,
                            box_w=w,
                            box_h=h,
                            image_height=pil_image.height,
                            image_width=pil_image.width,
                            mask=None,
                            is_thing=False,
                            is_landmark=True,
                        )
                        detections.append(detection)
            else:
                bbox = landmark["bbox"]
                x, y, w, h = self._convert_grid_to_normalized(
                    bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                )
                detection = Detection(
                    id=i,
                    label=landmark["name"],
                    image_id=None,
                    score=1.0,
                    box_x=x,
                    box_y=y,
                    box_w=w,
                    box_h=h,
                    image_height=pil_image.height,
                    image_width=pil_image.width,
                    mask=None,
                    is_thing=False,
                    is_landmark=True,
                )
                detections.append(detection)

        return detections

    @staticmethod
    def _normalize_landmark_name(raw_name: str) -> str:
        """
        Normalize landmark names to fix known issues.
        """
        
        if raw_name is None:
            return ""
        
        # Removes the word "building" from the end of the landmark name (ignore case)
        if raw_name.lower().endswith("building"):
            raw_name = raw_name[:-8].strip()
            
        # Removes empty spaces at the beginning and end of the name
        raw_name = raw_name.strip()
        
        # Replace underscores or dashes with spaces
        raw_name = raw_name.replace("_", " ").replace("-", " ")
        
        return raw_name
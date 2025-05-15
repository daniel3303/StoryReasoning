from pathlib import Path
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HOME
from insightface.app import FaceAnalysis
from transformers import AutoImageProcessor, AutoModel

from story_reasoning.models.object_detection.detection import Detection
from story_reasoning.models.object_matching.base_matcher import BaseMatcher


class DinoV2Matcher(BaseMatcher):
    """
    A class for matching objects across multiple images using DINOv2 embeddings with AuraFace for person class.
    For person detections, uses AuraFace for both face detection and embedding generation.

    Args:
        model_name (str): Name of the DINOv2 model to use
        device (torch.device): Device to use for inference
        similarity_threshold (float): Minimum visual similarity threshold for a match [0-1]
        face_similarity_threshold (float): Minimum face similarity threshold [0-1]
        std_threshold (float): Number of standard deviations allowed below mean for matching
        face_confidence_threshold (float): Minimum confidence for face detection [0-1]
        auraface_repo (str): HuggingFace repository for AuraFace model
        require_face_match (bool): If True, person detections are only matched if a face is detected and matched.
                                 If False, falls back to DINOv2 matching when no face is detected.
        min_face_resolution (float): Minimum average face dimension in pixels to consider a face good for face detection
    """

    def __init__(self, model_name: str = "facebook/dinov2-large", device: Optional[torch.device] = None,
                 similarity_threshold: float = 0.8, face_similarity_threshold: float = 0.5, std_threshold: float = 2.0,
                 face_confidence_threshold: float = 0.75, auraface_repo: str = "fal/AuraFace-v1",
                 require_face_match: bool = False, use_detection_mask: bool = False,
                 min_face_resolution: float = 112.0, create_single_matches = True):
        super().__init__(similarity_threshold, face_similarity_threshold, std_threshold, face_confidence_threshold,
                         auraface_repo, require_face_match, use_detection_mask, min_face_resolution, create_single_matches)

        # Initialize DINOv2 model and processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _compute_embedding(
            self,
            detection: Detection,
            frame: Image
    ) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray]]:
        """
        Compute embeddings for a detection using DINOv2 and face detection if applicable.

        Args:
            detection: Detection object
            frame: Input image

        Returns:
            Tuple of (dino_embedding, face_embedding)
        """
        # Initialize face embedding as None
        face_embedding = None

        # For person detections, get face embedding if available
        if detection.label.lower() == "person":
            frame_id = id(frame)
            if frame_id in self._frame_face_results:
                face_result = self._frame_face_results[frame_id].get(detection)
                if face_result is not None:
                    face_info, has_face = face_result
                    if has_face:
                        face_embedding = face_info.embedding

        # Get original image dimensions
        img_w, img_h = frame.size

        # Get detection coordinates
        x1 = max(0, int(detection.box_x * img_w))
        y1 = max(0, int(detection.box_y * img_h))
        x2 = min(img_w, int((detection.box_x + detection.box_w) * img_w))
        y2 = min(img_h, int((detection.box_y + detection.box_h) * img_h))

        # Crop the detection region
        detection_region = frame.crop((x1, y1, x2, y2))

        # Handle masking if enabled
        if self.use_detection_mask and detection.mask is not None:
            detection_region = detection_region.convert('RGBA')
            mask_img = Image.fromarray((detection.mask * 255).astype(np.uint8))
            mask_img = mask_img.resize(detection_region.size, Image.Resampling.NEAREST)

            mask_array = np.array(mask_img)
            rgba_array = np.array(detection_region)
            rgba_array[..., 3] = mask_array

            detection_region = Image.fromarray(rgba_array)
            bg = Image.new('RGBA', detection_region.size, (255, 255, 255, 255))
            detection_region = Image.alpha_composite(bg, detection_region)
            detection_region = detection_region.convert('RGB')

        # Create square padded image
        max_side = max(x2 - x1, y2 - y1)
        pad_x = (max_side - (x2 - x1)) // 2
        pad_y = (max_side - (y2 - y1)) // 2
        padded_img = Image.new('RGB', (max_side, max_side), (255, 255, 255))
        padded_img.paste(detection_region, (pad_x, pad_y))

        # Process with DINOv2
        inputs = self.processor(images=padded_img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding
            dino_embedding = outputs.last_hidden_state[:, 0]
            # Normalize embedding
            dino_embedding = dino_embedding / dino_embedding.norm(dim=1, keepdim=True)
            dino_embedding = dino_embedding[0]

        return dino_embedding, face_embedding

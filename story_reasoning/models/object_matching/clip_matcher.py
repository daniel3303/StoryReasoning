from pathlib import Path
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HOME
from insightface.app import FaceAnalysis
from transformers import CLIPProcessor, CLIPModel

from story_reasoning.models.object_detection.detection import Detection
from story_reasoning.models.object_matching.base_matcher import BaseMatcher


class ClipMatcher (BaseMatcher):
    """
    A class for matching objects across multiple images using CLIP embeddings with AuraFace for person class.
    For person detections, uses AuraFace for both face detection and embedding generation.

    Args:
        model_name (str): Name of the CLIP model to use
        device (torch.device): Device to use for inference
        similarity_threshold (float): Minimum visual similarity threshold for a match [0-1]
        face_similarity_threshold (float): Minimum face similarity threshold [0-1]
        std_threshold (float): Number of standard deviations allowed below mean for matching
        face_confidence_threshold (float): Minimum confidence for face detection [0-1]
        auraface_repo (str): HuggingFace repository for AuraFace model
        require_face_match (bool): If True, person detections are only matched if a face is detected and matched.
                                 If False, falls back to CLIP matching when no face is detected.
        min_face_resolution (float): Minimum average face dimension in pixels to consider a face good for face detection
    """

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", device: Optional[torch.device] = None,
                 similarity_threshold: float = 0.9, face_similarity_threshold: float = 0.4, std_threshold: float = 1.5,
                 face_confidence_threshold: float = 0.7, auraface_repo: str = "fal/AuraFace-v1",
                 require_face_match: bool = False, use_detection_mask: bool = True, min_face_resolution: float = 112.0, create_single_matches = True):
        super().__init__(similarity_threshold, face_similarity_threshold, std_threshold, face_confidence_threshold,
                         auraface_repo, require_face_match, use_detection_mask, min_face_resolution, create_single_matches)
        
        # Initialize CLIP model and processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)


    def _compute_embedding(
            self,
            detection: Detection,
            frame: Image
    ) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray]]:
        """
        Compute embeddings for a detection based on object type, using mask if enabled.

        This method performs the following steps:
        1. For person detections, retrieves pre-computed face embedding if available
        2. Crops the detection region from the input frame
        3. If mask usage is enabled and a mask is present:
           - Resizes the mask to match the cropped region
           - Creates an alpha channel from the mask
           - Applies the mask to remove background pixels
        4. Pads the masked/cropped region to create a square image
        5. Processes the final image with CLIP to generate embeddings

        For person detections:
        - First attempts to use pre-computed face embedding if available
        - Always computes CLIP embedding on the masked/cropped region

        For non-person objects:
        - Uses CLIP embedding only on the masked/cropped region

        Args:
            detection: Detection object to compute embeddings for, may include a binary
                      mask array in detection.mask with the same dimensions as the
                      detection's bounding box
            frame: Input image containing the detection

        Returns:
            Tuple of (clip_embedding, face_embedding), where:
            - clip_embedding: Tensor of shape (768,) containing the CLIP embedding
            - face_embedding: Optional face embedding numpy array for person detections
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

        # If mask usage is enabled and mask is available, apply it to the cropped region
        if self.use_detection_mask and detection.mask is not None:
            # Convert detection region to RGBA
            detection_region = detection_region.convert('RGBA')

            # Resize mask to match cropped region dimensions
            mask_img = Image.fromarray((detection.mask * 255).astype(np.uint8))
            mask_img = mask_img.resize(
                detection_region.size,
                Image.Resampling.NEAREST
            )

            # Create alpha channel from mask
            mask_array = np.array(mask_img)
            rgba_array = np.array(detection_region)
            rgba_array[..., 3] = mask_array  # Apply mask to alpha channel

            # Convert back to PIL Image
            detection_region = Image.fromarray(rgba_array)

            # Create white background image
            bg = Image.new('RGBA', detection_region.size, (255, 255, 255, 255))

            # Composite masked image onto white background
            detection_region = Image.alpha_composite(bg, detection_region)
            detection_region = detection_region.convert('RGB')

        # Calculate padding for square image
        max_side = max(x2 - x1, y2 - y1)
        pad_x = (max_side - (x2 - x1)) // 2
        pad_y = (max_side - (y2 - y1)) // 2

        # Create padded square image with white background
        padded_img = Image.new('RGB', (max_side, max_side), (255, 255, 255))
        padded_img.paste(detection_region, (pad_x, pad_y))

        # Process with CLIP
        inputs = self.processor(images=padded_img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            embedding = self.model.get_image_features(**inputs)
            embedding = embedding / embedding.norm(dim=1, keepdim=True)
            clip_embedding = embedding[0]

        return clip_embedding, face_embedding

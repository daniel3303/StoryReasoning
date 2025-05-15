from abc import abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HOME
from insightface.app import FaceAnalysis

from story_reasoning.models.object_detection.detection import Detection
from story_reasoning.models.object_matching.detection_match import DetectionMatch
from story_reasoning.models.object_matching.face_info import FaceInfo
from story_reasoning.utils.surpress_stdout import suppress_stdout


class BaseMatcher:
    """
    A class for matching objects across multiple images using image embeddings with AuraFace for person class.
    For person detections, uses AuraFace for both face detection and embedding generation.

    Args:
        similarity_threshold (float): Minimum visual similarity threshold for a match [0-1]
        face_similarity_threshold (float): Minimum face similarity threshold [0-1]
        std_threshold (float): Number of standard deviations allowed below mean for matching
        face_confidence_threshold (float): Minimum confidence for face detection [0-1]
        auraface_repo (str): HuggingFace repository for AuraFace model
        require_face_match (bool): If True, person detections are only matched if a face is detected and matched.
                                 If False, falls back to detection matching when no face is detected.
        min_face_resolution (float): Minimum average face dimension in pixels to consider a face good for face detection
    """

    def __init__(
            self,
            similarity_threshold: float = 0.9,
            face_similarity_threshold: float = 0.4,
            std_threshold: float = 1.5,
            face_confidence_threshold: float = 0.7,
            auraface_repo: str = "fal/AuraFace-v1",
            require_face_match: bool = False,
            use_detection_mask: bool = True,
            min_face_resolution: float = 112.0,  # Minimum average face dimension in pixels
            create_single_matches = True # If true all detections are assigned to a match even if the match has only one detection 
    ):
        self.similarity_threshold = similarity_threshold
        self.face_similarity_threshold = face_similarity_threshold
        self.std_threshold = std_threshold
        self.face_confidence_threshold = face_confidence_threshold
        self.require_face_match = require_face_match
        self.min_face_resolution = min_face_resolution
        self.use_detection_mask = use_detection_mask
        self.create_single_matches = create_single_matches

        # Initialize AuraFace
        with suppress_stdout():
            snapshot_download(auraface_repo, local_dir=Path(HF_HOME) / "models/auraface")
            self.face_analyzer = FaceAnalysis(
                name="auraface",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else [
                    "CPUExecutionProvider"],
                root=HF_HOME,
            )
            self.face_analyzer.prepare(ctx_id=0 if torch.cuda.is_available() else -1)

        # Initialize cache for face results
        self._frame_face_results = {}
        self._current_frame_detections = None

    def _process_faces_in_frame(
            self,
            frame: Image,
            person_detections: List[Detection]
    ) -> Dict[Detection, Optional[Tuple[FaceInfo, bool]]]:
        """
        Process all faces in a frame and assign them to person detections.
        Primary assignment method: Check if face center lies within person mask
        Fallback method: Distance-based assignment if no mask available

        Args:
            frame: Input PIL Image
            person_detections: List of person detections in the frame

        Returns:
            Dictionary mapping each person detection to (FaceInfo, has_face) tuple
            where FaceInfo contains face embedding and bbox, and has_face indicates if a valid face was found
        """
        if not person_detections:
            return {}

        # Convert frame to CV2 format for face detection
        cv2_image = np.array(frame)[:, :, ::-1]  # RGB to BGR
        frame_width, frame_height = frame.size

        # Sort person detections by area (smallest to largest)
        sorted_detections = sorted(
            person_detections,
            key=lambda det: det.box_w * det.box_h
        )

        # Get all faces in the frame
        faces = self.face_analyzer.get(cv2_image)

        # Store valid faces with normalized coordinates
        valid_faces: List[FaceInfo] = []
        for face in faces:
            if face.det_score < self.face_confidence_threshold:
                continue

            # Normalize bbox coordinates to [0,1]
            x1, y1, x2, y2 = face.bbox
            normalized_bbox = (
                float(x1 / frame_width),
                float(y1 / frame_height),
                float(x2 / frame_width),
                float(y2 / frame_height)
            )

            valid_faces.append(FaceInfo(
                embedding=face.normed_embedding,
                confidence=float(face.det_score),
                bbox=normalized_bbox
            ))

        # Initialize results dict
        results: Dict[Detection, Optional[Tuple[FaceInfo, bool]]] = {
            det: None for det in person_detections
        }

        # Process detections from smallest to largest
        used_faces: Set[FaceInfo] = set()

        for detection in sorted_detections:
            best_face = None
            best_distance = float('inf')
            mask_matched_face = None

            for face in valid_faces:
                if face in used_faces:
                    continue

                face_x1, face_y1, face_x2, face_y2 = face.bbox  # Already normalized

                # Check if face center is inside the detection bounding box if not continue
                face_center_x = (face_x1 + face_x2) / 2
                face_center_y = (face_y1 + face_y2) / 2
                if not (detection.box_x <= face_center_x <= detection.box_x + detection.box_w and
                        detection.box_y <= face_center_y <= detection.box_y + detection.box_h):
                    continue

                # Get original face dimensions in pixels
                face_width_px = (face_x2 - face_x1) * frame_width
                face_height_px = (face_y2 - face_y1) * frame_height
                face_resolution = max(face_width_px, face_height_px)

                # Skip if face resolution is too low
                if face_resolution < self.min_face_resolution:
                    continue

                # Try mask matching first
                if detection.mask is not None:
                    if self._match_face_by_mask(face, detection):
                        mask_matched_face = face
                    break

                # Fallback to distance matching if no mask available
                distance = self._match_face_by_distance(face, detection)
                if distance < best_distance:
                    best_distance = distance
                    best_face = face

            # Prefer mask-matched face over distance-matched face
            if mask_matched_face is not None:
                results[detection] = (mask_matched_face, True)
                used_faces.add(mask_matched_face)
            elif best_face is not None:
                results[detection] = (best_face, True)
                used_faces.add(best_face)
            else:
                results[detection] = (None, False)

        return results

    def _match_face_by_distance(
            self,
            face: FaceInfo,
            detection: Detection,
    ) -> float:

        """Calculate normalized distance between face and detection centers."""
        face_x1, face_y1, face_x2, face_y2 = face.bbox
        face_center_x = (face_x1 + face_x2) / 2
        face_center_y = (face_y1 + face_y2) / 2

        det_center_x = detection.box_x + detection.box_w / 2
        det_center_y = detection.box_y + detection.box_h / 2

        # Calculate face dimensions for normalization
        face_width = face_x2 - face_x1
        face_height = face_y2 - face_y1
        face_hypotenuse = (face_width ** 2 + face_height ** 2) ** 0.5

        # Normalize distance by both face size and confidence
        epsilon = 1e-6
        return (((face_center_x - det_center_x) ** 2 +
                 (face_center_y - det_center_y) ** 2) ** 0.5) / (
                (face_hypotenuse * face.confidence) + epsilon)

    def _match_face_by_mask(
            self,
            face: FaceInfo,
            detection: Detection,
    ) -> bool:
        """Check if face center point lies within detection mask."""
        if detection.mask is None:
            return False

        # Calculate face center in normalized coordinates
        face_x1, face_y1, face_x2, face_y2 = face.bbox
        face_center_x = (face_x1 + face_x2) / 2
        face_center_y = (face_y1 + face_y2) / 2

        # Convert from image coordinates to coordinates relative to detection box (the mask only represents the bbox space)
        relative_x = (face_center_x - detection.box_x) / detection.box_w
        relative_y = (face_center_y - detection.box_y) / detection.box_h

        # Convert to mask coordinates
        mask_x = int(relative_x * detection.mask.shape[1])
        mask_y = int(relative_y * detection.mask.shape[0])

        # Check if point lies within mask bounds and is True
        if detection.mask.shape[1] > mask_x >= 0 <= mask_y < detection.mask.shape[0]:
            return detection.mask[mask_y, mask_x] == 1

        return False

    @abstractmethod
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
        5. Processes the final image with the image feature extractor to generate embeddings

        For person detections:
        - First attempts to use pre-computed face embedding if available
        - Always computes detection embedding on the masked/cropped region

        For non-person objects:
        - Uses detection embedding only on the masked/cropped region

        Args:
            detection: Detection object to compute embeddings for, may include a binary
                      mask array in detection.mask with the same dimensions as the
                      detection's bounding box
            frame: Input image containing the detection

        Returns:
            Tuple of (detection_embedding, face_embedding), where:
            - detection_embedding: Tensor of shape (768,) containing the detection embedding
            - face_embedding: Optional face embedding numpy array for person detections
        """
        raise NotImplementedError

    def _compute_similarity(
            self,
            detection1: Detection,
            detection2: Detection,
            clip_embedding1: Optional[torch.Tensor],
            clip_embedding2: Optional[torch.Tensor],
            face_embedding1: Optional[np.ndarray] = None,
            face_embedding2: Optional[np.ndarray] = None
    ) -> Tuple[float, Optional[float]]:
        """
        Compute both detection and face similarities between two detections.

        For detection similarity:
        - Computes cosine similarity between detection embeddings if available
        - Maps similarity from [-1,1] to [0,1] range

        For face similarity:
        - Only computed for person detections with valid face embeddings
        - Uses dot product of normalized face embeddings

        Args:
            detection1, detection2: Detections to compare
            clip_embedding1, clip_embedding2: CLIP embeddings for detections (may be None)
            face_embedding1, face_embedding2: Face embeddings for detections (may be None)

        Returns:
            Tuple of (clip_similarity, face_similarity), either may be None if embeddings unavailable
        """
        # Compute CLIP similarity if available
        clip_sim = None
        if clip_embedding1 is not None and clip_embedding2 is not None:
            sim = torch.nn.functional.cosine_similarity(
                clip_embedding1.unsqueeze(0),
                clip_embedding2.unsqueeze(0)
            ).item()
            clip_sim = (sim + 1) / 2  # Map to [0, 1]

        # Compute face similarity if available
        face_sim = None
        if (detection1.label.lower() == "person" and detection2.label.lower() == "person" and
                face_embedding1 is not None and face_embedding2 is not None):
            face_sim = np.dot(face_embedding1, face_embedding2)

        return clip_sim, face_sim

    def _can_add_to_match(
            self,
            match: DetectionMatch,
            clip_similarities: List[float],
            face_similarities: Optional[List[float]] = None
    ) -> bool:
        """
        Check if a detection can be added to a match based on similarity thresholds

        Performs checks for both CLIP and face similarities:
        1. Verifies mean similarities meet minimum thresholds
        2. Checks if deviation from match mean is within allowed standard deviations
        3. For person detections with require_face_match=True, requires valid face similarity

        Args:
            match: Existing detection match
            clip_similarities: List of CLIP similarities with existing detections
            face_similarities: List of face similarities with existing detections (for persons)

        Returns:
            Boolean indicating if detection can be added to match
        """
        # For person detections with require_face_match, require face similarity
        if (self.require_face_match and match.detections and
                match.detections[0].label.lower() == "person" and not face_similarities):
            return False

        # Check CLIP similarity if no face similarities available
        # This is applicable to objects or when person detections have no faces
        if clip_similarities and not face_similarities:
            clip_mean = np.percentile(clip_similarities, 75)
            if clip_mean < self.similarity_threshold:
                return False

            if match.clip_std > 0:
                if (match.clip_similarity - clip_mean) > (self.std_threshold * match.clip_std):
                    return False

        # Check face similarity if available
        if face_similarities:
            face_mean = np.percentile(face_similarities, 75)
            if face_mean < self.face_similarity_threshold:
                return False

            if match.face_std and match.face_std > 0:
                if (match.face_similarity - face_mean) > (self.std_threshold * match.face_std):
                    return False

        return True

    def match_detections(
            self,
            all_detections: List[List[Detection]],
            all_frames: List[Image]
    ) -> List[DetectionMatch]:
        """
        Match detections across multiple frames based on visual similarity.

        Key steps in the matching process:
        1. Pre-processes faces for all frames containing person detections
        2. Groups detections by label type
        3. For each label group:
           - Computes embeddings for all detections
           - Creates matches starting with unmatched detections
           - Grows matches by adding compatible detections
           - Applies similarity and standard deviation thresholds
        4. For person detections:
           - Uses face matching when faces detected
           - If require_face_match=True, only matches detections with faces
           - If require_face_match=False, uses CLIP matching

        Args:
            all_detections: List of detection lists, one list per frame
            all_frames: List of frames

        Returns:
            List of detection matches across frames, sorted by similarity score
        """
        matches = []

        if len(all_detections) != len(all_frames):
            raise ValueError("Number of detections and frames must match")               

        # Clear face results cache
        self._frame_face_results = {}

        # Pre-process faces for all frames at the beginning
        for frame_idx, (frame, frame_dets) in enumerate(zip(all_frames, all_detections)):
            person_detections = [det for det in frame_dets if det.label.lower() == "person"]
            if person_detections:
                self._frame_face_results[id(frame)] = self._process_faces_in_frame(frame, person_detections)

        # Group detections by label
        label_detections = {}
        for frame_idx, frame_dets in enumerate(all_detections):
            for det in frame_dets:
                if det.label not in label_detections:
                    label_detections[det.label] = []
                label_detections[det.label].append((det, frame_idx))

        # Initialize counters for match IDs
        character_counter = 0
        object_counter = 0
        landmark_counter = 0
        background_counter = 0

        # Process each label separately
        for label, detections in label_detections.items():
            # Store current frame detections for reference
            self._current_frame_detections = [det for det, _ in detections]

            # Compute embeddings for all detections
            embeddings = {}  # (det, frame_idx) -> (clip_emb, face_emb)
            valid_detections = []  # List of (det, frame_idx) that meet matching criteria

            for det, frame_idx in detections: # type: Detection, int
                clip_emb, face_emb = self._compute_embedding(det, all_frames[frame_idx])
                embeddings[(det, frame_idx)] = (clip_emb, face_emb)

                # For person detections with require_face_match, only include if face is detected
                if det.label.lower() == "person" and self.require_face_match:
                    if face_emb is not None:
                        valid_detections.append((det, frame_idx))
                else:
                    valid_detections.append((det, frame_idx))

            unmatched = set(valid_detections)
            processed_detections = set()  # Keep track of detections that have been put in matches

            # Create initial matches
            while unmatched:
                start_det: Detection
                start_frame: int
                start_det, start_frame = next(iter(unmatched))

                # Generate match ID based on object type
                if label.lower() == "person":
                    character_counter += 1
                    match_id = f"char{character_counter}"
                elif start_det.is_thing:
                    object_counter += 1
                    match_id = f"obj{object_counter}"
                elif start_det.is_landmark:
                    landmark_counter += 1
                    match_id = f"lm{landmark_counter}"
                else:
                    background_counter += 1
                    match_id = f"bg{background_counter}"

                current_match = DetectionMatch(id=match_id, label=label, detections=[start_det])

                # Get embeddings for start detection
                clip_emb, face_emb = embeddings[(start_det, start_frame)]
                current_match.update_statistics(
                    [], None, clip_emb, face_emb,
                    has_face=(face_emb is not None)
                )

                unmatched.remove((start_det, start_frame))
                processed_detections.add((start_det, start_frame))
                changed = True

                # Keep adding detections while possible
                while changed and unmatched:
                    changed = False
                    best_addition = None
                    best_clip_sims = []
                    best_face_sims = []

                    for det, frame_idx in unmatched:
                        # Don't match within same frame
                        if any(d in all_detections[frame_idx] for d in current_match.detections):
                            continue

                        clip_emb, face_emb = embeddings[(det, frame_idx)]

                        # Compute similarities with all existing detections
                        clip_similarities = []
                        face_similarities = []

                        for existing_det in current_match.detections:
                            existing_frame = next(i for i, frame_dets in enumerate(all_detections)
                                                  if existing_det in frame_dets)
                            existing_clip, existing_face = embeddings[(existing_det, existing_frame)]

                            clip_sim, face_sim = self._compute_similarity(
                                det, existing_det, clip_emb, existing_clip,
                                face_emb, existing_face
                            )
                            clip_similarities.append(clip_sim)
                            if face_sim is not None:
                                face_similarities.append(face_sim)

                        # Check if this detection can be added
                        if self._can_add_to_match(current_match, clip_similarities,
                                                  face_similarities if face_similarities else None):
                            best_addition = (det, frame_idx)
                            best_clip_sims = clip_similarities
                            best_face_sims = face_similarities
                            changed = True
                            break

                    # Add best detection if found
                    if best_addition:
                        det, frame_idx = best_addition
                        clip_emb, face_emb = embeddings[(det, frame_idx)]
                        current_match.detections.append(det)
                        current_match.update_statistics(
                            best_clip_sims, best_face_sims, clip_emb, face_emb,
                            has_face=(face_emb is not None)
                        )
                        unmatched.remove(best_addition)
                        processed_detections.add(best_addition)

                if len(current_match.detections) > 1 or self.create_single_matches:
                    matches.append(current_match)

        return sorted(matches, key=lambda x: max(x.clip_similarity, x.face_similarity or 0), reverse=True)

    def clear_cache(self):
        """Clear the internal face detection and frame caches"""
        self._frame_face_results = {}
        self._current_frame_detections = None

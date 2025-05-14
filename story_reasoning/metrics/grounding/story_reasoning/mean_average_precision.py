from typing import Dict, List

import numpy as np

from story_reasoning.metrics.grounding.story_reasoning.story_reasoning_grounding_metric import StoryReasoningGroundingMetric
from story_reasoning.models.object_detection.detection import Detection


class MeanAveragePrecision(StoryReasoningGroundingMetric):
    """
    Mean Average Precision (mAP) metric for visual grounding evaluation in story reasoning.
    
    Calculates AP for each story and then averages across all stories.
    """

    def _compute_score(
            self,
            reference_detections: Dict[str, List[Detection]],
            candidate_detections: Dict[str, List[Detection]]
    ) -> float:
        """
        Compute mean AP across all samples.
        
        Args:
            reference_detections: Reference detections per story
            candidate_detections: Candidate detections per story
            
        Returns:
            float: Mean AP score across all samples
        """
        scores = []

        for story_id in candidate_detections:
            if story_id not in reference_detections:
                continue

            cand_dets = candidate_detections[story_id]
            ref_dets = reference_detections[story_id]

            # If either set is empty, handle edge case
            if not ref_dets:
                scores.append(1.0 if not cand_dets else 0.0)
                continue
            if not cand_dets:
                scores.append(0.0)
                continue

            # Calculate matches between reference and candidate detections
            matches = self._match_detections(ref_dets, cand_dets)

            # Create a mapping from candidate detection to whether it's matched
            # This helps with efficient lookup later
            matched_cand_dets = {cand_det: ref_det for ref_det, cand_det in matches}

            # Sort detections by score (descending)
            sorted_cand_dets = sorted(cand_dets, key=lambda x: x.score, reverse=True)

            precisions = []
            recalls = []

            relevant_detections = len(ref_dets)
            true_positives = 0

            for i, cand_det in enumerate(sorted_cand_dets):
                # Check if this detection is a match
                is_match = cand_det in matched_cand_dets

                if is_match:
                    true_positives += 1

                # Calculate precision and recall at this threshold
                precision = true_positives / (i + 1)
                recall = true_positives / relevant_detections

                precisions.append(precision)
                recalls.append(recall)

            # Convert to numpy arrays for interpolation
            precisions = np.array(precisions)
            recalls = np.array(recalls)

            # If there are no detections or matches, AP is 0
            if len(precisions) == 0:
                scores.append(0.0)
                continue

            # Compute AP using standard 11-point interpolation approach
            # (commonly used in PASCAL VOC challenge)
            ap = 0.0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(recalls >= t) == 0:
                    p = 0
                else:
                    p = np.max(precisions[recalls >= t])
                ap += p / 11.0

            scores.append(ap)

        return sum(scores) / len(scores) if scores else 0.0
from typing import Dict, List

from story_reasoning.metrics.grounding.story_reasoning.story_reasoning_grounding_metric import StoryReasoningGroundingMetric
from story_reasoning.models.object_detection.detection import Detection


class ObjectF1(StoryReasoningGroundingMetric):
    """
    F1 Score metric specifically for non-character object grounding in story reasoning.
    
    Only considers detections with labels other than "person".
    """

    def _compute_score(
            self,
            reference_detections: Dict[str, List[Detection]],
            candidate_detections: Dict[str, List[Detection]]
    ) -> float:
        """
        Compute mean F1 score for object detections across all samples.
        
        Args:
            reference_detections: Reference detections per story
            candidate_detections: Candidate detections per story
            
        Returns:
            float: Mean object F1 score across all samples
        """
        scores = []

        for story_id in reference_detections:
            if story_id not in candidate_detections:
                continue

            # Filter for non-person detections
            ref_dets = [det for det in reference_detections[story_id] if det.label != "person"]
            cand_dets = [det for det in candidate_detections[story_id] if det.label != "person"]

            if not ref_dets and not cand_dets:
                scores.append(1.0)  # If no object mentions in either, F1 is perfect
                continue

            matches = self._match_detections(ref_dets, cand_dets)
            matched_ref_dets = {ref_det for ref_det, _ in matches}
            matched_cand_dets = {cand_det for _, cand_det in matches}

            tp = len(matched_cand_dets)
            fp = len(cand_dets) - len(matched_cand_dets)
            fn = len(ref_dets) - len(matched_ref_dets)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0

            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            scores.append(f1)

        return sum(scores) / len(scores) if scores else 0.0


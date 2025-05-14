from typing import Dict, List

from story_reasoning.metrics.grounding.story_reasoning.story_reasoning_grounding_metric import StoryReasoningGroundingMetric
from story_reasoning.models.object_detection.detection import Detection


class ObjectRecall(StoryReasoningGroundingMetric):
    """
    Recall metric specifically for non-character object grounding in story reasoning.
    
    Only considers detections with labels other than "person".
    """

    def _compute_score(
            self,
            reference_detections: Dict[str, List[Detection]],
            candidate_detections: Dict[str, List[Detection]]
    ) -> float:
        """
        Compute mean recall for object detections across all samples.
        
        Args:
            reference_detections: Reference detections per story
            candidate_detections: Candidate detections per story
            
        Returns:
            float: Mean object recall score across all samples
        """
        scores = []

        for story_id in reference_detections:
            if story_id not in candidate_detections:
                continue

            # Filter for non-person detections
            ref_dets = [det for det in reference_detections[story_id] if det.label != "person"]
            cand_dets = [det for det in candidate_detections[story_id] if det.label != "person"]

            if not ref_dets:
                scores.append(1.0)  # If no object mentions in reference, recall is perfect
                continue

            matches = self._match_detections(ref_dets, cand_dets)
            matched_ref_dets = {ref_idx for ref_idx, _ in matches}

            tp = len(matched_ref_dets)
            fn = len(ref_dets) - tp

            recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
            scores.append(recall)

        return sum(scores) / len(scores) if scores else 0.0


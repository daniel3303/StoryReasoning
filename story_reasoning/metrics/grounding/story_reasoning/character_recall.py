from typing import Dict, List

from story_reasoning.metrics.grounding.story_reasoning.story_reasoning_grounding_metric import StoryReasoningGroundingMetric
from story_reasoning.models.object_detection.detection import Detection


class CharacterRecall(StoryReasoningGroundingMetric):
    """
    Recall metric specifically for character grounding in story reasoning.
    
    Only considers detections with the label "person".
    """

    def _compute_score(
            self,
            reference_detections: Dict[str, List[Detection]],
            candidate_detections: Dict[str, List[Detection]]
    ) -> float:
        """
        Compute mean recall for character detections across all samples.
        
        Args:
            reference_detections: Reference detections per story
            candidate_detections: Candidate detections per story
            
        Returns:
            float: Mean character recall score across all samples
        """
        scores = []

        for story_id in reference_detections:
            if story_id not in candidate_detections:
                continue

            # Filter for only person detections
            ref_dets = [det for det in reference_detections[story_id] if det.label == "person"]
            cand_dets = [det for det in candidate_detections[story_id] if det.label == "person"]

            if not ref_dets:
                scores.append(1.0)  # If no character mentions in reference, recall is perfect
                continue

            matches = self._match_detections(ref_dets, cand_dets)
            matched_ref_indices = {ref_det for ref_det, _ in matches}

            tp = len(matched_ref_indices)
            fn = len(ref_dets) - tp

            recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
            scores.append(recall)

        return sum(scores) / len(scores) if scores else 0.0


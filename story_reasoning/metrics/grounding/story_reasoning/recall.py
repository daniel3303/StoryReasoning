from typing import Dict, List

from story_reasoning.metrics.grounding.story_reasoning.story_reasoning_grounding_metric import StoryReasoningGroundingMetric
from story_reasoning.models.object_detection.detection import Detection


class Recall(StoryReasoningGroundingMetric):
    """
    Recall metric for visual grounding evaluation in story reasoning.
    
    Measures the proportion of ground truth objects that are correctly grounded
    in the candidate story.
    Recall = TP / (TP + FN)
    """

    def _compute_score(
            self,
            reference_detections: Dict[str, List[Detection]],
            candidate_detections: Dict[str, List[Detection]]
    ) -> float:
        """
        Compute mean recall across all samples.
        
        Args:
            reference_detections: Reference detections per story
            candidate_detections: Candidate detections per story
            
        Returns:
            float: Mean recall score across all samples
        """
        scores = []

        for story_id in reference_detections:
            if story_id not in candidate_detections:
                continue

            ref_dets = reference_detections[story_id]
            cand_dets = candidate_detections[story_id]

            matches = self._match_detections(ref_dets, cand_dets)
            matched_ref_dets = {ref_det for ref_det, _ in matches}

            tp = len(matched_ref_dets)
            fn = len(ref_dets) - tp

            # Recall = TP / (TP + FN)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0  # Recall is 1 if no reference detections
            scores.append(recall)

        return sum(scores) / len(scores) if scores else 0.0


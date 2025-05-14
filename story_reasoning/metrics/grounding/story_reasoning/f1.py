from typing import Dict, List

from story_reasoning.metrics.grounding.story_reasoning.story_reasoning_grounding_metric import StoryReasoningGroundingMetric
from story_reasoning.models.object_detection.detection import Detection


class F1(StoryReasoningGroundingMetric):
    """
    F1 Score metric for visual grounding evaluation in story reasoning.
    
    Harmonic mean of precision and recall.
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """

    def _compute_score(
            self,
            reference_detections: Dict[str, List[Detection]],
            candidate_detections: Dict[str, List[Detection]]
    ) -> float:
        """
        Compute mean F1 score across all samples.
        
        Args:
            reference_detections: Reference detections per story
            candidate_detections: Candidate detections per story
            
        Returns:
            float: Mean F1 score across all samples
        """
        scores = []

        for story_id in reference_detections:
            if story_id not in candidate_detections:
                continue

            ref_dets = reference_detections[story_id]
            cand_dets = candidate_detections[story_id]

            matches = self._match_detections(ref_dets, cand_dets)
            matched_ref_dets = {ref_idx for ref_idx, _ in matches}
            matched_cand_dets = {cand_det for _, cand_det in matches}

            tp = len(matched_cand_dets)
            fp = len(cand_dets) - len(matched_cand_dets)
            fn = len(ref_dets) - len(matched_ref_dets)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0

            # F1 = 2 * (Precision * Recall) / (Precision + Recall)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            scores.append(f1)

        return sum(scores) / len(scores) if scores else 0.0


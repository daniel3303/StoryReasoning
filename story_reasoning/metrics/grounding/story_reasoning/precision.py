from typing import Dict, List

from story_reasoning.metrics.grounding.story_reasoning.story_reasoning_grounding_metric import StoryReasoningGroundingMetric
from story_reasoning.models.object_detection.detection import Detection


class Precision(StoryReasoningGroundingMetric):
    """
    Precision metric for visual grounding evaluation in story reasoning.
    
    Measures the proportion of correctly grounded objects in the candidate story
    relative to all objects mentioned in the candidate story.
    Precision = TP / (TP + FP)
    """

    def _compute_score(
            self,
            reference_detections: Dict[str, List[Detection]],
            candidate_detections: Dict[str, List[Detection]]
    ) -> float:
        """
        Compute mean precision across all samples.
        
        Args:
            reference_detections: Reference detections per story
            candidate_detections: Candidate detections per story
            
        Returns:
            float: Mean precision score across all samples
        """
        scores = []
        
        for story_id in candidate_detections:
            if story_id not in reference_detections:
                continue

            cand_dets = candidate_detections[story_id]
            ref_dets = reference_detections[story_id]

            matches = self._match_detections(ref_dets, cand_dets)
            matched_cand_dets = {cand_dets for _, cand_dets in matches}
            
            
            tp = len(matched_cand_dets)
            fp = len(cand_dets) - tp
            
            # Precision = TP / (TP + FP)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0  # Precision is 1 if no detections
            scores.append(precision)

        return sum(scores) / len(scores) if scores else 0.0


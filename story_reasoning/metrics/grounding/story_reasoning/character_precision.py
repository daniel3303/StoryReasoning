from typing import Dict, List

from story_reasoning.metrics.grounding.story_reasoning.story_reasoning_grounding_metric import StoryReasoningGroundingMetric
from story_reasoning.models.object_detection.detection import Detection


class CharacterPrecision(StoryReasoningGroundingMetric):
    """
    Precision metric specifically for character grounding in story reasoning.
    
    Only considers detections with the label "person".
    """

    def _compute_score(
            self,
            reference_detections: Dict[str, List[Detection]],
            candidate_detections: Dict[str, List[Detection]]
    ) -> float:
        """
        Compute mean precision for character detections across all samples.
        
        Args:
            reference_detections: Reference detections per story
            candidate_detections: Candidate detections per story
            
        Returns:
            float: Mean character precision score across all samples
        """
        scores = []

        for story_id in candidate_detections:
            if story_id not in reference_detections:
                continue

            # Filter for only person detections
            cand_dets = [det for det in candidate_detections[story_id] if det.label == "person"]
            ref_dets = [det for det in reference_detections[story_id] if det.label == "person"]

            if not cand_dets:
                scores.append(1.0)  # If no character mentions, precision is perfect
                continue

            matches = self._match_detections(ref_dets, cand_dets)
            matched_cand_dets = {cand_det for _, cand_det in matches}

            tp = len(matched_cand_dets)
            fp = len(cand_dets) - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            scores.append(precision)

        return sum(scores) / len(scores) if scores else 0.0


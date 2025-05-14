from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Set

from story_reasoning.models.object_detection.detection import Detection


class StoryReasoningGroundingMetric(ABC):
    """
    Base class for grounding metrics that evaluate story reasoning with grounding information.
    
    These metrics evaluate the correctness of character and object references in generated stories
    by comparing detections against ground truth detections.
    """

    def evaluate(
            self,
            reference_detections: Dict[str, List[Detection]],
            candidate_detections: Dict[str, List[Detection]]
    ) -> float:
        """
        Evaluate multiple candidates against their respective references.
        
        For each story ID in the dataset:
        - The detections of the reference and candidate stories are compared.
        - Ground truth (reference) detections are provided
        - Candidate detections are provided
        
        Args:
            reference_detections (Dict[str, List[Detection]]): Dictionary mapping story IDs to ground truth detections.
            candidate_detections (Dict[str, List[Detection]]): Dictionary mapping story IDs to candidate detections.
                
        Returns:
            float: Mean evaluation score across all samples.
        """
       
        return self._compute_score(reference_detections, candidate_detections)

    @abstractmethod
    def _compute_score(
            self,
            reference_detections: Dict[str, List[Detection]],
            candidate_detections: Dict[str, List[Detection]]
    ) -> float:
        """
        Abstract method to compute the visual grounding metric score.
        Must be implemented by each specific metric class.
        
        Args:
            reference_detections (Dict[str, List[Detection]]): Dictionary mapping story IDs to lists
                of ground truth Detection objects.
            candidate_detections (Dict[str, List[Detection]]): Dictionary mapping story IDs to lists
                of candidate Detection objects.
                
        Returns:
            float: Mean evaluation score across all samples.
        """
        pass

    @staticmethod
    def _match_detections(ref_detections: List[Detection], cand_detections: List[Detection]) -> Set[Tuple[Detection, Detection]]:
        """
        Find matching detections between reference and candidate sets.
        
        A match occurs when:
            - For detections with label "person"
                - The label is the same
                - The intersection / union (IoU) is greater than 0.5
            - For detections without label "person"
                - The intersection / union (IoU) is greater than 0.5
        All possible matches are computed, sorted by IoU in descending order, 
        and matches are selected greedily from highest to lowest IoU (above 0.5).
        
        Args:
            ref_detections (List[Detection]): List of reference detections.
            cand_detections (List[Detection]): List of candidate detections.
            
        Returns:
            Set[Tuple[Detection, Detection]]: Set of (ref_detection, cand_detection) pairs representing matches.
        """
        # Track all possible matches with their IoU scores
        possible_matches = []
    
        # Find all potential matches
        for ref_det in ref_detections:
            for cand_det in cand_detections:
                # Check if both are persons or both are not persons
                is_matching_type = (ref_det.label == "person" and cand_det.label == "person") or \
                                   (ref_det.label != "person" and cand_det.label != "person")
    
                if is_matching_type:
                    iou = ref_det.iou(cand_det)
                    if iou >= 0.5:
                        possible_matches.append((ref_det, cand_det, iou))
    
        # Sort matches by IoU in descending order
        possible_matches.sort(key=lambda x: x[2], reverse=True)
    
        # Greedily select matches
        matched_refs = set()
        matched_cands = set()
        final_matches = set()
    
        for ref_det, cand_det, iou in possible_matches:
            # Only add the match if neither detection has been matched yet
            if ref_det not in matched_refs and cand_det not in matched_cands:
                final_matches.add((ref_det, cand_det))
                matched_refs.add(ref_det)
                matched_cands.add(cand_det)
    
        return final_matches




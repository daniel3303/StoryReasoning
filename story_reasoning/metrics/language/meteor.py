from typing import Dict

from pycocoevalcap.meteor.meteor import Meteor as CocoMeteor
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

from story_reasoning.metrics.language.language_metric import LanguageMetric


class Meteor(LanguageMetric):
    """
    Wrapper for Microsoft COCO evaluation toolkit's METEOR implementation.

    METEOR (Metric for Evaluation of Translation with Explicit ORdering) is based on the
    harmonic mean of unigram precision and recall, with additional features addressing
    issues that BLEU cannot capture.
    """

    def __init__(self, strip_grounding_tags: bool = True):
        """Initialize METEOR metric."""
        super().__init__(strip_grounding_tags)
        self.scorer = CocoMeteor()
        self.tokenizer = PTBTokenizer()

    def _compute_score(self, references: Dict[str, str],  candidates: Dict[str, str]) -> float:
        """
        Compute mean METEOR score for multiple candidates against their respective references.

        Args:
            references (Dict[str, str]): Dictionary mapping sample IDs to lists of preprocessed reference texts.
            candidates (Dict[str, str]): Dictionary mapping sample IDs to preprocessed candidate texts.

        Returns:
            float: Mean METEOR score across all samples.
        """
        # If a Dict[str, str] is found convert to Dict[str, List[str]]
        candidate_dict = {key: [value] if isinstance(value, str) else value for key, value in candidates.items()}
        reference_dict = {key: [value] if isinstance(value, str) else value for key, value in references.items()}

        # Prepares coco structure replacing each text with the object {caption: text}
        candidate_dict = {key: [{'caption': value} for value in values] for key, values in candidate_dict.items()}
        reference_dict = {key: [{'caption': value} for value in values] for key, values in reference_dict.items()}

        # Tokenize inputs
        tokenized_candidates = self.tokenizer.tokenize(candidate_dict)
        tokenized_references = self.tokenizer.tokenize(reference_dict)

        # Compute score
        score, _ = self.scorer.compute_score(tokenized_references, tokenized_candidates)

        return score
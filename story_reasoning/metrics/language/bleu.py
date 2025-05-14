from enum import Enum
from typing import List, Dict
from pycocoevalcap.bleu.bleu import Bleu as CocoBleu
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

from story_reasoning.metrics.language.language_metric import LanguageMetric


class BleuType(Enum):
    """
    Enumeration of supported BLEU variants.

    Attributes:
        BLEU1: Unigram-based scoring
        BLEU2: Bigram-based scoring
        BLEU3: Trigram-based scoring
        BLEU4: 4-gram-based scoring
    """
    BLEU1 = 0
    BLEU2 = 1
    BLEU3 = 2
    BLEU4 = 3


class Bleu(LanguageMetric):
    """
    Wrapper for Microsoft COCO evaluation toolkit's BLEU implementation.

    BLEU (Bilingual Evaluation Understudy) is a metric for evaluating machine-translated text
    using a modified form of precision to compare a candidate translation against one or more
    reference translations.

    Uses the 'closest' option for BLEU scoring, which compares the candidate against the
    closest reference in length for each sample.
    """

    def __init__(self, bleu_type: BleuType = BleuType.BLEU4, strip_grounding_tags: bool = True):
        """Initialize BLEU-4 metric."""
        super().__init__(strip_grounding_tags)
        self.bleu_type = bleu_type
        self.scorer = CocoBleu()
        self.tokenizer = PTBTokenizer()


    def _compute_score(self, reference: Dict[str, str], candidates: Dict[str, str]) -> float:
        """
        Compute mean BLEU score for multiple candidates against their respective references.

        Args:
            reference (Dict[str, str]): Dictionary mapping sample IDs to lists of preprocessed reference texts.
            candidates (Dict[str, str]): Dictionary mapping sample IDs to preprocessed candidate texts.

        Returns:
            float: Mean BLEU score across all samples, using the closest reference for each sample.
        """
        # If a Dict[str, str] is found convert to Dict[str, List[str]]
        candidate_dict = {key: [value] if isinstance(value, str) else value for key, value in candidates.items()}
        reference_dict = {key: [value] if isinstance(value, str) else value for key, value in reference.items()}

        # Prepares coco structure replacing each text with the object {caption: text}
        candidate_dict = {key: [{'caption': value} for value in values] for key, values in candidate_dict.items()}
        reference_dict = {key: [{'caption': value} for value in values] for key, values in reference_dict.items()}

        # Tokenize inputs
        tokenized_candidates = self.tokenizer.tokenize(candidate_dict)
        tokenized_references = self.tokenizer.tokenize(reference_dict)

        # Compute scores using the closest reference option
        score, _ = self.scorer.compute_score(tokenized_references, tokenized_candidates)

        return score[self.bleu_type.value]
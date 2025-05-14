from typing import List, Union, Dict
import re
from abc import ABC, abstractmethod

class LanguageMetric(ABC):
    """
    Abstract base class for text evaluation metrics adopted to the story_reasoning framework.

    Provides common functionality for text preprocessing and evaluation interface.
    Handles grounding tag stripping when enabled.

    Args:
        strip_grounding_tags (bool): Whether to strip grounding tags from input text before evaluation.
            If true only the text content within the tags will be considered for evaluation.

    Attributes:
        strip_grounding_tags (bool): Flag indicating whether grounding tags should be stripped from input text.
    """

    def __init__(self, strip_grounding_tags: bool = True):
        """Initialize the metric with the given configuration."""
        self.strip_grounding_tags = strip_grounding_tags

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess input text according to strip_grounding_tags configuration.

        Args:
            text (str): Input text to preprocess.

        Returns:
            str: Preprocessed text with grounding tags stripped if enabled.
        """
        if self.strip_grounding_tags:
            # Remove all HTML tags but keep their content
            text = re.sub(r'<[^>]+>', '', text)
        return text.strip()

    def _preprocess_texts(self, texts: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Preprocess a single text or list of texts.

        Args:
            texts (Union[str, List[str]]): Input text(s) to preprocess.

        Returns:
            Union[str, List[str]]: Preprocessed text(s).
        """
        if isinstance(texts, str):
            return self._preprocess_text(texts)
        return [self._preprocess_text(text) for text in texts]

    def evaluate_single(self, reference: str, candidate: str) -> float:
        """
        Evaluate a single candidate text against its reference texts.

        Args:
            reference (List[str]): List of reference texts to compare against.
            candidate (str): Candidate text to evaluate.

        Returns:
            float: Evaluation score.
        """
        # Create single-element dictionaries to use batch computation
        candidates_dict = {'single': candidate}
        references_dict = {'single': reference}

        return self.evaluate(references_dict, candidates_dict)

    def evaluate(self, references: Dict[str, str], candidates: Dict[str, str]) -> float:
        """
        Evaluate multiple candidates against their respective references.

        Args:
            references (Dict[str, List[str]]): Dictionary mapping sample IDs to lists of reference text.
            candidates (Dict[str, str]): Dictionary mapping sample IDs to candidate text.

        Returns:
            float: Mean evaluation score across all samples.
        """
        # Preprocess all inputs
        processed_candidates = {
            id_: self._preprocess_text(text) for id_, text in candidates.items()
        }
        processed_references = {
            id_: self._preprocess_texts(text) for id_, text in references.items()
        }

        return self._compute_score(processed_references, processed_candidates)

    @abstractmethod
    def _compute_score(self, references: Dict[str, List[str]], candidates: Dict[str, str]) -> float:
        """
        Compute mean score for multiple candidates against their respective references.

        Args:
            references (Dict[str, List[str]]): Dictionary mapping sample IDs to lists of preprocessed reference texts.
            candidates (Dict[str, str]): Dictionary mapping sample IDs to preprocessed candidate texts.

        Returns:
            float: Mean evaluation score across all samples.
        """
        pass
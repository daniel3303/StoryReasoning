from typing import Optional, Dict, Any, List

from torch.utils.data import Dataset


class BaseTlrDatasetAdapter(Dataset):
    """
        Base class for dataset adapters to support Hugging Face TLR training.
        The adapter should implement the __getitem__ method to return a single example in the conversational or instruction format.
    """

    def __init__(self, dataset: Dataset):
        Dataset.__init__(self)
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item):
        raise NotImplementedError

    def __getattr__(self, name):
        """Delegate attribute access to the dataset if not found in the adapter."""
        return getattr(self.dataset, name)


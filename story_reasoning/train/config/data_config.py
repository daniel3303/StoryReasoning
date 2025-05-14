from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    dataset_path: Optional[str] = field(default=None, metadata={"help": "Path to the dataset"})
    hf_repo: Optional[str] = field(default=None, metadata={"help": "HF Dataset repository ID"})
    dataset_name: str = field(default=None, metadata={"help": "Name of the dataset to load. The name is used to determine the dataset and adapter to load"})
    num_workers: int = field(default=4, metadata={"help": "Number of dataloader workers"})

    def __post_init__(self):
        if self.dataset_name is None:
            raise ValueError("dataset_name must be provided")
        if self.dataset_path is None and self.hf_repo is None:
            raise ValueError("Either dataset_path or hf_repo must be provided")
        if self.dataset_path is not None and self.hf_repo is not None:
            raise ValueError("Only one of dataset_path or hf_repo should be provided")

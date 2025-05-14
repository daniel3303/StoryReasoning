import os

import yaml
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, config_path: str, split: str, dataset_key: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.split = split
        self.dataset_key = dataset_key
        self.dataset_root = os.path.join(self.config["data_root"], self.config["paths"]["raw"], self.dataset_key)
        self.dataset_config = self.config["datasets"][dataset_key]

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
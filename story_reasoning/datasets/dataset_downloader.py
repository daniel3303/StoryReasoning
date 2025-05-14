import os
from abc import ABC, abstractmethod

import yaml


class DatasetDownloader(ABC):
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.data_root = self.config['data_root']
        self.raw_path = os.path.join(self.data_root, self.config['paths']['raw'])


    @abstractmethod
    def run(self):
        pass
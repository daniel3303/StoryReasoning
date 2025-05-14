class DatasetRegistry:
    _data_modules = {}
    _downloaders = {}
    _datasets = {}
    _tlr_dataset_adapters = {}

    # Register a downloader class with the registry
    @classmethod
    def register_downloader(cls, name):
        def decorator(downloader_class):
            cls._downloaders[name] = downloader_class
            return downloader_class
        return decorator

    @classmethod
    def get_downloader(cls, name):
        return cls._downloaders.get(name)

    # Register a dataset class with the registry
    @classmethod
    def register_dataset(cls, name):
        def decorator(dataset_class):
            cls._datasets[name] = dataset_class
            return dataset_class

        return decorator

    @classmethod
    def get_dataset(cls, name):
        return cls._datasets.get(name)

    # Register a data module with the registry
    @classmethod
    def register_data_module(cls, name):
        def decorator(data_module):
            cls._data_modules[name] = data_module
            return data_module
        return decorator

    @classmethod
    def get_data_module(cls, name):
        return cls._data_modules.get(name)

    # Register a dataset adapter class with the registry
    @classmethod
    def register_tlr_dataset_adapter(cls, name):
        def decorator(adapter_class):
            cls._tlr_dataset_adapters[name] = adapter_class
            return adapter_class
        return decorator

    @classmethod
    def get_dataset_adapter(cls, name):
        return cls._tlr_dataset_adapters.get(name)





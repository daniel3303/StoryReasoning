import yaml

from story_reasoning.datasets import DatasetRegistry


def run_downloader(config_path, dataset_name):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    dataset_config = config['datasets'][dataset_name]
    if not dataset_config:
        raise ValueError(f"Dataset '{dataset_name}' not found in config.")


    downloader_class = DatasetRegistry.get_downloader(dataset_name)
    if not downloader_class:
        raise ValueError(f"Downloader class '{dataset_name}' not found in registry.")

    downloader = downloader_class(config_path)
    downloader.run()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run dataset downloader.")
    parser.add_argument("-c", "--config-file", required=True, help="Path to the configuration file")
    parser.add_argument("-d", "--dataset", required=True, help="Name of the dataset to download")
    args = parser.parse_args()

    run_downloader(args.config_file, args.dataset)

    # Shows a success message to the user showing that the download is complete
    print("\033[92m" + "Download completed!" + "\033[0m")
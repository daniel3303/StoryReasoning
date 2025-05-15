import argparse
import random
import warnings
from pathlib import Path
from typing import Dict, List

import tqdm
from PIL import Image as PILImage
from datasets import Dataset, Features, Value, Image, DatasetDict, Sequence

from story_reasoning.datasets.story_reasoning.story_reasoning_derived_dataset import StoryReasoningDerivedDataset

"""
    Script to prepare the StoryReasoning dataset for distribution.
    Example usage:
    python vsg/datasets/story_reasoning/postprocessing/distribute_hf.py --input_dir /path/to/input 
        --output_file /path/to/output --test_split 0.1 --hf_repo_id optional/repo/id
"""


def get_file_groups(input_dir: Path, derived_dataset: StoryReasoningDerivedDataset) -> List[Dict]:
    """
    Group files by their story IDs, verify completeness, and match with derived dataset.

    Args:
        input_dir (Path): Directory containing all files (metadata, stories, cot)
        derived_dataset (StoryReasoningDerivedDataset): The StoryReasoningDerived dataset instance

    Returns:
        List[Dict]: List of dictionaries containing paths and data for each complete group of files.
    """
    # Find all files with different suffixes
    metadata_files = {f.stem.split('_metadata')[0]: f for f in input_dir.glob('*_metadata.txt')}
    cot_files = {f.stem.split('_cot')[0]: f for f in input_dir.glob('*_cot.txt')}
    story_files = {f.stem.split('_story')[0]: f for f in input_dir.glob('*_story.txt')}

    file_groups = []

    # For each story in the derived dataset
    for idx in range(len(derived_dataset)):
        derived_story = derived_dataset[idx]
        story_id = f"{idx:d}"

        # Check if all required files exist
        if story_id in metadata_files and story_id in story_files and story_id in cot_files:
            # Read image IDs from metadata file
            with open(metadata_files[story_id], 'r') as f:
                metadata_image_ids = [line.strip() for line in f.readlines()]

            # Get image IDs from derived dataset
            derived_image_ids = derived_story['image_ids']

            # Verify image IDs match
            if set(metadata_image_ids) == set(derived_image_ids):
                group = {
                    'story_id': story_id,
                    'metadata_file': metadata_files[story_id],
                    'cot_file': cot_files[story_id],
                    'story_file': story_files[story_id],
                    'derived_idx': idx,
                    'derived_data': derived_story
                }
                file_groups.append(group)
            else:
                warnings.warn(
                    f"Mismatch between metadata and derived dataset for story {story_id}. "
                    f"Metadata has {len(metadata_image_ids)} images, derived has {len(derived_image_ids)}."
                )
        else:
            missing = []
            if story_id not in metadata_files:
                missing.append('metadata')
            if story_id not in cot_files:
                missing.append('cot')
            if story_id not in story_files:
                missing.append('story')
            warnings.warn(
                f"Incomplete file group for story {story_id}: missing {', '.join(missing)} files."
            )

    return file_groups


def process_file_group(group: Dict) -> Dict:
    """
    Process a group of files and return their contents in a structured format.

    Args:
        group (Dict): Dictionary containing group data

    Returns:
        Dict: Processed data including images, chain of thought, and story
    """
    # Extract images directly from derived dataset
    images = group['derived_data']['images']

    # Convert images if necessary (ensure they are PIL Images)
    pil_images = []
    for img in images:
        if not isinstance(img, PILImage.Image):
            # Convert from tensor or other format if needed
            # This assumes the images in derived_data are in a format convertible to PIL
            pil_images.append(PILImage.fromarray(img.numpy().transpose(1, 2, 0)))
        else:
            pil_images.append(img)

    # Read chain of thought from file
    with open(group['cot_file'], 'r', encoding='utf-8') as f:
        cot = f.read().strip()

    # Read story
    with open(group['story_file'], 'r', encoding='utf-8') as f:
        story = f.read().strip()

    return {
        'story_id': group['story_id'],
        'images': pil_images,
        'frame_count': len(pil_images),
        'chain_of_thought': cot,
        'story': story
    }


def prepare_dataset(input_dir: Path, test_split: float) -> DatasetDict:
    """
    Prepare the dataset by preprocessing all file groups and creating splits.

    Args:
        input_dir (Path): Input directory containing all necessary files
        test_split (float): Fraction of data to use for test split (0.0 to 1.0)

    Returns:
        DatasetDict: Dictionary containing train and test datasets
    """
    # Define features for the dataset
    features = Features({
        'story_id': Value('string'),
        'images': Sequence(Image()),
        'frame_count': Value('int32'),
        'chain_of_thought': Value('string'),
        'story': Value('string')
    })

    # Create the derived dataset
    derived_dataset = StoryReasoningDerivedDataset()

    file_groups = get_file_groups(input_dir, derived_dataset)

    if not file_groups:
        raise ValueError("No complete file groups found in the input directory")

    # Shuffle and split
    random.shuffle(file_groups)
    test_size = int(len(file_groups) * test_split)

    test_groups = file_groups[:test_size]
    train_groups = file_groups[test_size:]

    # Process groups into datasets
    def process_groups(groups):
        processed_data = []
        for group in tqdm.tqdm(groups, desc="Processing file groups"):
            try:
                processed_group = process_file_group(group)
                processed_data.append(processed_group)
            except Exception as e:
                warnings.warn(f"Error preprocessing group {group['story_id']}: {str(e)}")
                continue
        return Dataset.from_list(processed_data, features=features)

    dataset_dict = DatasetDict({
        'train': process_groups(train_groups),
        'test': process_groups(test_groups)
    })

    return dataset_dict


def main():
    """Main function to prepare the dataset."""
    parser = argparse.ArgumentParser(description='Prepare StoryReasoning dataset for distribution')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing all input files (metadata, cot, story files)')
    parser.add_argument('--output-file', type=str, required=False,
                        help='Path to save the output dataset')
    parser.add_argument('--hub-repo-id', type=str, required=False,
                        help='Hugging Face Hub repository ID to upload the dataset')
    parser.add_argument('--test-split', type=float, required=False, default=0.15,
                        help='Fraction of data to use for test split (default: 0.15)')

    args = parser.parse_args()
    input_dir = Path(args.input_dir)

    print(f"Processing data from {input_dir}...")

    dataset_dict = prepare_dataset(
        input_dir,
        args.test_split
    )

    # Create output directory if it doesn't exist
    if args.output_file:
        output_dir = Path(args.output_file).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save dataset to disk
        dataset_dict.save_to_disk(args.output_file)
        print(f"Dataset saved to: {args.output_file}")

    # Upload to Hugging Face Hub
    if args.hub_repo_id:
        dataset_dict.push_to_hub(args.hub_repo_id)
        print(f"Dataset uploaded to HuggingFace Hub: {args.hub_repo_id}")

    # Print dataset info
    print("\nDataset preparation completed!")
    print(f"Train split size: {len(dataset_dict['train'])}")
    print(f"Test split size: {len(dataset_dict['test'])}")

    # Print story frame count distribution
    train_frame_counts = [item['frame_count'] for item in dataset_dict['train']]
    test_frame_counts = [item['frame_count'] for item in dataset_dict['test']]

    print("\nFrame count distribution:")
    print(f"Train: min={min(train_frame_counts)}, max={max(train_frame_counts)}, avg={sum(train_frame_counts)/len(train_frame_counts):.1f}")
    print(f"Test: min={min(test_frame_counts)}, max={max(test_frame_counts)}, avg={sum(test_frame_counts)/len(test_frame_counts):.1f}")


if __name__ == "__main__":
    main()
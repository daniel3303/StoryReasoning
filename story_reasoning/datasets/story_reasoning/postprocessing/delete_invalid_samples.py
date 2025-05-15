#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

from story_reasoning.datasets.story_reasoning.stories.generate_stories import validate_cot, validate_story
from story_reasoning.datasets.story_reasoning.story_reasoning_derived_dataset import StoryReasoningDerivedDataset


def main():
    parser = argparse.ArgumentParser(description="Validate generated stories and optionally delete invalid ones")
    parser.add_argument("--folder_path", required=True, help="Path to the folder containing generated samples")
    args = parser.parse_args()

    folder_path = Path(args.folder_path)

    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Error: {folder_path} is not a valid directory")
        sys.exit(1)

    # Find all sample IDs in the folder
    sample_ids = set()
    for file in folder_path.iterdir():
        if file.is_file() and file.suffix == '.txt':
            # Extract sample ID from filename (e.g., "42_cot.txt" -> "42")
            parts = file.stem.split('_')
            if len(parts) >= 2 and parts[0].isdigit():
                sample_ids.add(int(parts[0]))

    print(f"Found {len(sample_ids)} potential samples in {folder_path}")
    
    # Initialize the dataset 
    dataset = StoryReasoningDerivedDataset()
    print("Dataset initialized.")

    # Validate each sample
    invalid_samples = []
    total_samples = 0

    for sample_id in sorted(sample_ids):            
        cot_file = folder_path / f"{sample_id}_cot.txt"
        story_file = folder_path / f"{sample_id}_story.txt"

        # Skip if any file is missing
        if not (cot_file.exists() and story_file.exists()):
            print(f"Skipping sample {sample_id} - incomplete files")
            continue

        total_samples += 1

        try:
            # Read the files
            with open(cot_file, 'r') as f:
                analysis_text = f.read()

            with open(story_file, 'r') as f:
                story_text = f.read()
                
            images = dataset[sample_id]['images']
            if len(images) == 0:
                print(f"Error: No images found for sample {sample_id}")
                invalid_samples.append((sample_id, "No images found"))
                raise ValueError("No images found")

            # Validate the CoT and Story
            cot_is_valid, cot_error_message = validate_cot(analysis_text, images)
            story_is_valid, story_error_message = validate_story(analysis_text, story_text, images)
            
            is_valid = cot_is_valid and story_is_valid
            error_message = cot_error_message + " " + story_error_message
            

            if not is_valid:
                print(f"Sample {sample_id}: Invalid CoT - {error_message}")
                invalid_samples.append((sample_id, error_message))

            # Print progress every 10 samples
            if total_samples % 10 == 0:
                print(f"Processed {total_samples} samples...")
                
            # Closes the images
            for image in images:
                image.close()

        except Exception as e:
            raise ValueError(f"Error processing sample {sample_id}: {str(e)}") from e
            print(f"Error processing sample {sample_id}: {e}")
            invalid_samples.append((sample_id, f"Processing error: {str(e)}"))

    # Print summary
    print("\nValidation Summary:")
    print("=" * 50)
    print(f"Total samples processed: {total_samples}")
    print(f"Valid samples: {total_samples - len(invalid_samples)}")
    print(f"Invalid samples: {len(invalid_samples)}")

    if invalid_samples:
        print("\nInvalid samples:")
        for sample_id, error in invalid_samples:
            print(f"- Sample {sample_id}: {error}")

        # Ask if user wants to delete invalid samples
        response = input("\nDo you want to delete these invalid samples? (yes/no): ").strip().lower()

        if response == 'yes':
            count = 0
            for sample_id, _ in invalid_samples:
                cot_file = folder_path / f"{sample_id}_cot.txt"
                story_file = folder_path / f"{sample_id}_story.txt"
                metadata_file = folder_path / f"{sample_id}_metadata.txt"

                # Delete each file if it exists
                for file in [cot_file, story_file, metadata_file]:
                    if file.exists():
                        file.unlink()
                        count += 1

            print(f"Deleted {count} files from {len(invalid_samples)} invalid samples.")
        else:
            print("No files were deleted.")
    else:
        print("All samples are valid!")


if __name__ == "__main__":
    main()

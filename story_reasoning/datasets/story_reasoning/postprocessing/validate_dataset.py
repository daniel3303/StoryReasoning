import re

from story_reasoning.datasets.story_reasoning.story_reasoning_dataset import StoryReasoningDataset
from story_reasoning.models.story_reasoning.story_reasoning_util import StoryReasoningUtil
from story_reasoning.models.story_reasoning.narrative_phase import NarrativePhase


def count_gdi_tags(text: str) -> int:
    """Count the number of '<gdi image*>' tags in the text."""
    pattern = r'<gdi image\d+>'
    matches = re.findall(pattern, text)
    return len(matches)


def validate_dataset() -> None:
    """
    Validate dataset samples using the StoryReasoningUtil:
    1. Matching counts of images, image headers, and gdi tags
    2. Successful parsing of all image sections with the required tables
    3. Successful parsing of the narrative structure
    
    Prints errors directly to the console.
    """
    # Load the dataset with default parameters
    print("Loading StoryReasoningDataset with default parameters...")
    dataset = StoryReasoningDataset()

    count_mismatches = 0
    parsing_errors = 0
    narrative_errors = 0
    total_samples = len(dataset)

    print(f"Validating {total_samples} samples...")

    for idx in range(total_samples):
        sample = dataset[idx]
        story_id = sample['story_id']
        chain_of_thought = sample['chain_of_thought']

        # Use the parser to count image headers by parsing the complete analysis
        # This is more robust than regex as it leverages the existing parser
        complete_analysis = StoryReasoningUtil.parse_cot(chain_of_thought)
        cot_headers_count = len(complete_analysis.images)

        # 1. Count validation
        image_count = len(sample['images'])
        gdi_tags_count = count_gdi_tags(sample['story'])

        count_match = (image_count == cot_headers_count == gdi_tags_count)
        if not count_match:
            count_mismatches += 1
            print(f"\nCount mismatch found in sample {idx} (Story ID: {story_id}):")
            print(f"- Image count: {image_count}")
            print(f"- Parsed image sections: {cot_headers_count}")
            print(f"- '<gdi image*>' tags: {gdi_tags_count}")

        # 2. Parse each image section using the StoryReasoningUtil
        missing_images = []
        incomplete_tables = []

        for image_num in range(1, image_count + 1):
            image_analysis = StoryReasoningUtil.parse_image(chain_of_thought, image_num)

            if image_analysis is None:
                missing_images.append(image_num)
                continue

            # Check if all required tables have content
            if not image_analysis.characters:
                incomplete_tables.append(f"Image {image_num}: Missing Characters table")
            if not image_analysis.objects:
                incomplete_tables.append(f"Image {image_num}: Missing Objects table")
            if not image_analysis.settings:
                incomplete_tables.append(f"Image {image_num}: Missing Setting table")

        if missing_images or incomplete_tables:
            parsing_errors += 1
            print(f"\nParsing issues found in sample {idx} (Story ID: {story_id}):")

            if missing_images:
                print(f"- Missing image sections: {', '.join(map(str, missing_images))}")

            if incomplete_tables:
                print("- Incomplete tables:")
                for issue in incomplete_tables:
                    print(f"  - {issue}")

        # 3. Parse narrative structure
        narrative_structure = StoryReasoningUtil.parse_narrative_structure(chain_of_thought)

        if narrative_structure is None or not narrative_structure.phases:
            narrative_errors += 1
            print(f"\nNarrative Structure issues in sample {idx} (Story ID: {story_id}):")
            print("- Failed to parse Narrative Structure section")
        else:
            # Check if all required narrative phases are present
            # Extract the enum values from the phases for comparison
            found_phases = []
            for phase in narrative_structure.phases:
                phase_value = phase.get('Narrative Phase')
                if isinstance(phase_value, NarrativePhase):
                    found_phases.append(phase_value.value)
                else:
                    found_phases.append(str(phase_value))

            missing_phases = []
            for phase in NarrativePhase:
                if phase.value not in found_phases:
                    missing_phases.append(phase.value)

            if missing_phases:
                narrative_errors += 1
                print(f"\nNarrative Structure issues in sample {idx} (Story ID: {story_id}):")
                print(f"- Missing narrative phases: {', '.join(missing_phases)}")

        # Print progress every 10 samples
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{total_samples} samples...")

    # Calculate problematic samples
    problematic_samples = count_mismatches + parsing_errors + narrative_errors
    valid_samples = total_samples - problematic_samples

    # Print summary
    print("\nValidation Summary:")
    print("=" * 50)
    print(f"Total samples processed: {total_samples}")
    print(f"Samples with problems: {problematic_samples} ({problematic_samples / total_samples * 100:.2f}%)")
    print("-" * 50)
    print(f"  - Count mismatches: {count_mismatches}")
    print(f"  - Parsing errors: {parsing_errors}")
    print(f"  - Narrative structure issues: {narrative_errors}")
    print("-" * 50)
    print(f"Completely valid samples: {valid_samples} ({valid_samples / total_samples * 100:.2f}%)")


if __name__ == "__main__":
    validate_dataset()
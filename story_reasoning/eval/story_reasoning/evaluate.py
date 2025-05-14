import argparse
import os
import re
from typing import Tuple, List

from tqdm import tqdm

from story_reasoning.datasets import DatasetRegistry
from story_reasoning.metrics.grounding.story_reasoning.character_f1 import CharacterF1
from story_reasoning.metrics.grounding.story_reasoning.character_precision import CharacterPrecision
from story_reasoning.metrics.grounding.story_reasoning.character_recall import CharacterRecall
from story_reasoning.metrics.grounding.story_reasoning.f1 import F1
from story_reasoning.metrics.grounding.story_reasoning.mean_average_precision import MeanAveragePrecision
from story_reasoning.metrics.grounding.story_reasoning.object_f1 import ObjectF1
from story_reasoning.metrics.grounding.story_reasoning.object_precision import ObjectPrecision
from story_reasoning.metrics.grounding.story_reasoning.object_recall import ObjectRecall
from story_reasoning.metrics.grounding.story_reasoning.precision import Precision
from story_reasoning.metrics.grounding.story_reasoning.recall import Recall
from story_reasoning.metrics.language.bleu import Bleu, BleuType
from story_reasoning.metrics.language.meteor import Meteor
from story_reasoning.metrics.language.rouge import Rouge
from story_reasoning.models.object_detection.detection import Detection
from story_reasoning.models.story_reasoning.story_reasoning_util import StoryReasoningUtil


# This script evaluates a model on the Story Reasoning dataset using the "traditional" metrics


def clean_story(text: str) -> str:
    """
    Clean the story text by:
    1. Removing <think> tags and their content
    2. Replacing character tags with CHARREF
    3. Removing all other tags while preserving content
    
    Args:
        text (str): Raw story text with tags
        
    Returns:
        str: Cleaned story text
    """
    # Get the story (the content after the </think> tag)
    text = StoryReasoningUtil.extract_story_text(text)  

    # Remove all other HTML-like tags but keep their content
    text = re.sub(r'<[^>]+>', '', text)
    
    # Removes more than one consecutive spaces resulting from the removal of tags
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def parse_bounding_box(bbox_str: str, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    Parse a bounding box string from the CoT which is in pixel coordinates (x1, y1, x2, y2)
    and convert to normalized coordinates (x, y, w, h) expected by Detection class.
    
    Args:
        bbox_str (str): String representation of bounding box in pixels
        img_width (int): Image width in pixels for normalization
        img_height (int): Image height in pixels for normalization
        
    Returns:
        Tuple[float, float, float, float]: Normalized (box_x, box_y, box_w, box_h)
    """
    # Clean the string and extract values
    cleaned = bbox_str.strip()
    values = [int(x.strip()) for x in cleaned.split(',') if x.strip()]

    if len(values) != 4:
        raise ValueError(f"Invalid bounding box format: {bbox_str}")

    # Format is (x1, y1, x2, y2) in pixels
    x1, y1, x2, y2 = values
    
    # Convert to normalized coordinates
    norm_x = x1 / img_width
    norm_y = y1 / img_height
    norm_w = (x2 - x1) / img_width
    norm_h = (y2 - y1) / img_height

    # Ensure values are within [0, 1]
    norm_x = max(0.0, min(norm_x, 1.0))
    norm_y = max(0.0, min(norm_y, 1.0))
    norm_w = max(0.0, min(norm_w, 1.0))
    norm_h = max(0.0, min(norm_h, 1.0))

    return (norm_x, norm_y, norm_w, norm_h)


def get_image_dimensions(image) -> Tuple[int, int]:
    """
    Get dimensions for an image.
    
    Args:
        image: Image from the dataset
        
    Returns:
        Tuple[int, int]: (width, height) tuple
    """
    # For PIL images
    if hasattr(image, 'size'):
        return image.size

    # For tensor images
    elif hasattr(image, 'shape'):
        # Assuming CHW format
        if len(image.shape) == 3:
            return (image.shape[2], image.shape[1])  # width, height
        else:
            return (image.shape[1], image.shape[0])  # width, height

    # Cannot determine dimensions
    raise ValueError("Unable to determine image dimensions")


def extract_detections_from_cot(think_text: str, num_images: int, images) -> List[Detection]:
    """
    Extract detections from the Chain of Thought text using the StoryReasoningUtil.
    
    Args:
        think_text (str): The Chain of Thought text (inside <think> tags)
        num_images (int): Number of images to consider
        images: List of images from the dataset
        
    Returns:
        List[Detection]: List of detections from all images
    """
    # Parse the Chain of Thought text
    parser = StoryReasoningUtil()
    cot = parser.parse_cot(think_text)

    detections = []
    
    curr_id = 0

    # Process each image in the analysis
    for image_analysis in cot.images:
        image_num = image_analysis.image_number

        # Get the image dimensions (1-indexed in the analysis but 0-indexed in the list)
        if 0 <= image_num - 1 < len(images):
            image = images[image_num - 1]
            try:
                img_width, img_height = get_image_dimensions(image)
            except ValueError as e:
                raise ValueError(f"Unable to determine dimensions for image {image_num}: {str(e)}")
        else:
            raise ValueError(f"Image {image_num} not found in dataset (only {len(images)} images available)")

        # Extract character detections
        for char in image_analysis.characters:
            try:
                # Parse bounding box
                box_x, box_y, box_w, box_h = parse_bounding_box(char.bounding_box, img_width, img_height)

                curr_id = curr_id + 1
                detections.append(Detection(
                    id=curr_id, # Unique to prevent the custom __eq__ method implementation
                    label="person",
                    image_id=str(image_num),
                    score=1.0,
                    box_x=box_x,
                    box_y=box_y,
                    box_w=box_w,
                    box_h=box_h,
                    image_width=img_width,
                    image_height=img_height
                ))
            except (ValueError, AttributeError) as e:
                print(f"Error parsing character {char} in image {image_num}: {str(e)}.")
                continue

        # Extract object detections
        for obj in image_analysis.objects:
            try:
                # Parse bounding box
                box_x, box_y, box_w, box_h = parse_bounding_box(obj.bounding_box, img_width, img_height)

                # Use description as label, or default to "object"
                label = obj.description.lower().split()[0] if obj.description else "object"

                curr_id = curr_id + 1
                detections.append(Detection(
                    id=curr_id, # Unique to prevent the custom __eq__ method implementation
                    label=label,
                    image_id=str(image_num),
                    score=1.0,
                    box_x=box_x,
                    box_y=box_y,
                    box_w=box_w,
                    box_h=box_h,
                    image_width=img_width,
                    image_height=img_height
                ))
            except (ValueError, AttributeError) as e:
                print(f"Error parsing object {obj} in image {image_num}: {str(e)}.")
                continue

    return detections


def read_generated_stories(stories_dir: str):
    """
    Read generated stories from files in the specified directory.
    
    Args:
        stories_dir (str): Directory containing generated story files named as {story_id}.txt
        
    Returns:
        dict: Dictionary mapping story IDs to generated story content
    """
    stories = {}
    story_files = [f for f in os.listdir(stories_dir) if f.endswith('.txt')]

    for file_name in story_files:
        story_id = file_name.replace('.txt', '')
        file_path = os.path.join(stories_dir, file_name)

        with open(file_path, 'r', encoding='utf-8') as f:
            stories[story_id] = f.read().strip()

    return stories


def evaluate_stories(stories_dir: str):
    """
    Evaluate pre-generated stories against reference stories from the dataset.
    Language metrics are evaluated on cleaned story text.
    Grounding metrics are evaluated on detections from the Chain of Thought.
    
    Args:
        stories_dir (str): Directory containing generated story files
        
    Returns:
        dict: Dictionary of metric results
    """
    # Initialize language metrics
    language_metrics = {
        'BLEU-1': Bleu(bleu_type=BleuType.BLEU1),
        'BLEU-2': Bleu(bleu_type=BleuType.BLEU2),
        'BLEU-3': Bleu(bleu_type=BleuType.BLEU3),
        'BLEU-4': Bleu(bleu_type=BleuType.BLEU4),
        #'CIDEr': Cider(),
        'ROUGE-L': Rouge(),
        #'SPICE': Spice(),
        'METEOR': Meteor()
    }

    # Initialize grounding metrics
    grounding_metrics = {
        'Precision': Precision(),
        'Recall': Recall(),
        'F1': F1(),
        'Char. Precision': CharacterPrecision(),
        'Char. Recall': CharacterRecall(),
        'Char. F1': CharacterF1(),
        'Obj. Precision': ObjectPrecision(),
        'Obj. Recall': ObjectRecall(),
        'Obj. F1': ObjectF1(),
        'mAP': MeanAveragePrecision()

    }

    # Load dataset using the default parameters
    dataset_class = DatasetRegistry.get_dataset("story_reasoning")
    if dataset_class is None:
        raise ValueError("Story reasoning dataset not found in registry")

    eval_dataset = dataset_class(split="test")

    # Read generated stories
    generated_stories = read_generated_stories(stories_dir)
    print(f"Found {len(generated_stories)} generated stories in {stories_dir}")

    # Track missing stories
    missing_stories = []

    # Store references and candidates for each language metric
    language_metric_data = {name: {"references": {}, "candidates": {}} for name in language_metrics.keys()}

    # Store detections for grounding metrics
    reference_detections = {}
    candidate_detections = {}

    # Process each sample in the dataset
    for i in tqdm(range(len(eval_dataset)), desc="Evaluating stories"):
        sample = eval_dataset[i]
        story_id = sample["story_id"]

        # Skip if no generated story for this sample
        if story_id not in generated_stories:
            missing_stories.append(story_id)
            continue

        # Get raw reference and generated stories
        reference_raw = "<think>\n" + sample["chain_of_thought"] + "\n</think>\n" + sample["story"]
        generated_raw = generated_stories[story_id]

        # Clean stories for language metrics
        reference_story = clean_story(reference_raw)
        generated_story = clean_story(generated_raw)
        
        # Store for language metrics evaluation
        for name in language_metrics.keys():
            language_metric_data[name]["references"][story_id] = reference_story
            language_metric_data[name]["candidates"][story_id] = generated_story

        # Get images from dataset
        images = sample["images"]
        num_images = len(images)

        # Initialize detections lists
        reference_detections[story_id] = []
        candidate_detections[story_id] = []

        # Extract detections from Chain of Thought
        reference_think = StoryReasoningUtil.extract_cot_text(reference_raw)
        candidate_think = StoryReasoningUtil.extract_cot_text(generated_raw)

        # Extract CoT detections for reference
        if reference_think:
            print("Processing Ground Truth CoT for story_id:", story_id)
            try:
                reference_detections[story_id] = extract_detections_from_cot(
                    reference_think, num_images, images
                )
            except Exception as e:
                print(f"Error extracting reference CoT detections for {story_id}: {str(e)}")

        # Extract CoT detections for candidate
        if candidate_think:
            print("Processing Candidate CoT for story_id:", story_id)
            try:
                candidate_detections[story_id] = extract_detections_from_cot(
                    candidate_think, num_images, images
                )
            except Exception as e:
                print(f"Error extracting candidate CoT detections for {story_id}: {str(e)}")

    # Report missing stories
    if missing_stories:
        print(f"Warning: {len(missing_stories)} stories in the dataset don't have corresponding generated files.")
        if len(missing_stories) <= 10:
            print(f"Missing stories: {', '.join(missing_stories)}")
        else:
            print(f"First 10 missing stories: {', '.join(missing_stories[:10])}")

    # Compute language metrics
    results = {"language": {}, "grounding": {}}

    print("\nCalculating language metrics...")
    for name, metric in language_metrics.items():
        try:
            # Skip if no data
            if not language_metric_data[name]["references"] or not language_metric_data[name]["candidates"]:
                continue

            # Calculate metric
            score = metric.evaluate(
                language_metric_data[name]["references"],
                language_metric_data[name]["candidates"]
            )
            results["language"][name] = score

        except Exception as e:
            print(f"Error calculating {name}: {str(e)}")

    # Compute grounding metrics
    print("\nCalculating grounding metrics...")
    for name, metric in grounding_metrics.items():
        try:
            # Calculate metric
            score = metric.evaluate(
                reference_detections,
                candidate_detections
            )
            results["grounding"][name] = score

        except Exception as e:
            print(f"Error calculating {name}: {str(e)}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate pre-generated stories on the Story Reasoning dataset")
    parser.add_argument("--stories_dir", type=str, required=True,
                        help="Directory containing generated story files named as {story_id}.txt")

    args = parser.parse_args()

    results = evaluate_stories(args.stories_dir)

    # Print language metrics results
    print("\nLanguage Evaluation Results:")
    print("-" * 40)
    for metric, score in results["language"].items():
        print(f"{metric:15s}: {score:.4f}")

    # Print grounding metrics results
    print("\nGrounding Evaluation Results:")
    print("-" * 40)
    for metric, score in results["grounding"].items():
        print(f"{metric:15s}: {score:.4f}")


if __name__ == "__main__":
    main()
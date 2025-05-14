---
license: cc-by-nd-4.0
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
dataset_info:
  features:
  - name: story_id
    dtype: string
  - name: images
    sequence: image
  - name: frame_count
    dtype: int32
  - name: chain_of_thought
    dtype: string
  - name: story
    dtype: string
  splits:
  - name: train
    num_bytes: 727245636.808
    num_examples: 3552
  - name: test
    num_bytes: 127457433.0
    num_examples: 626
  download_size: 773193011
  dataset_size: 854703069.808
---

# StoryReasoning Dataset

## Overview

The StoryReasoning dataset is a collection of visual storytelling data designed to address limitations in maintaining 
consistent identity across multiple images while generating coherent narratives. 
It contains 4,178 cohesive stories derived from 52,016 images, organizing temporally connected image sequences 
extracted from the same movie scenes to ensure narrative coherence.

## Key Features

- **Cross-Frame Consistency**: Each story maintains character and object consistency across frames through a structured object re-identification system
- **Structured Scene Analysis**: Multi-frame relationships are explicitly modeled through structured tabular representations
- **Grounded Narratives**: Stories are grounded to visual elements through specialized XML tags that link text to specific visual entities
- **Chain-of-Thought Reasoning**: Includes structured reasoning processes documenting characters, objects, settings, and narrative progression

## Dataset Structure

Each sample in the dataset contains:

- `story_id`: Unique identifier for each story
- `images`: Sequence of images (typically 5+ frames) from temporally related scenes
- `frame_count`: Number of frames in the story
- `chain_of_thought`: Structured scene analysis in text format containing:
    - Character tables tracking individuals with emotions, actions, and spatial locations
    - Object tables documenting objects with functions, interactions, and spatial coordinates
    - Setting tables categorizing environmental elements
    - Narrative structure tables modeling story progression
- `story`: The generated narrative with XML grounding tags linking text to visual elements

## Grounding Tags

The stories use four types of specialized XML tags:

- `<gdi>`: Image tags demarcating text segments corresponding to specific frames
- `<gdo>`: Entity reference tags linking character and object mentions to their visual counterparts
- `<gda>`: Action tags grounding character actions to the individuals performing them
- `<gdl>`: Location/landmark tags grounding descriptions to their visual counterparts

## Dataset Creation

The dataset was created through a systematic process:

1. **Frame Selection**: Images from the same movie and sequential scenes were selected until obtaining at least 5 images per story
2. **Object Detection**: Using Mask2Former with a Swin-Large backbone for simultaneous detection and segmentation
3. **Landmark Detection**: A dual-approach system using a fine-tuned Swin Transformer and LLM-based detection
4. **Cross-Frame Object Re-identification**: Combining general object embeddings with specialized face embeddings for consistent entity tracking
5. **Structured Scene Analysis**: Generating character, object, setting, and narrative structure tables
6. **Grounded Story Generation**: Creating narratives with specialized XML tags linking text to visual elements


## Installation

To work with the StoryReasoning dataset and its utilities, you can install the `story_reasoning` package directly from GitHub:

```bash
# Install the StoryReasoning package
pip install git+https://github.com/daniel3303/StoryReasoning.git

# Or clone the repository and install locally
git clone https://github.com/daniel3303/StoryReasoning.git
cd StoryReasoning
pip install -e .
```

After installation, you can import the package and use its components:

```python
from story_reasoning.dataset import StoryReasoningDataset, StoryReasoningAdapter, StoryReasoningUtil

# Now you can use the dataset, adapter, and utilities
```

This package provides all the necessary tools to work with the dataset, including:
- Utilities for parsing and manipulating the Chain-of-Thought and Story structures
- Adapter for integrating with Hugging Face's TRL library

## Using the Dataset

Here's a simple example of loading the dataset:

```python
# Load the dataset directly
from story_reasoning.dataset import StoryReasoningDataset

# Load the complete dataset
story_dataset = StoryReasoningDataset(hf_repo="daniel3303/StoryReasoning")

# Or load just the train split
train_dataset = StoryReasoningDataset(hf_repo="daniel3303/StoryReasoning", split="train")

# Get a sample
sample = train_dataset[0]
```

## StoryReasoningUtil

The StoryReasoningUtil class provides helpful methods to work with the dataset:

- Parse CoT and story text into structured objects with `parse_cot()` and `parse_story()`
- Convert structured objects back to text with `cot_to_string()` and `story_to_string()`
- Strip grounding tags from stories with `strip_story_tags()`
- Filter unmentioned entities from CoT structures with `filter_unmentioned_entities()`
- Extract CoT and story text from model outputs with `extract_cot_text()` and `extract_story_text()`

## StoryReasoningAdapter

The StoryReasoningAdapter helps integrate the dataset with Hugging Face's TRL library for fine-tuning:

```python
from story_reasoning.dataset import StoryReasoningDataset, StoryReasoningAdapter

# Load the dataset
story_dataset = StoryReasoningDataset(hf_repo="daniel3303/StoryReasoning", split="train")

# Create an adapter with options to adjust stories
adapter = StoryReasoningAdapter(
    story_dataset,
    max_images=10,                       # Limit the number of images per story
    filter_unmentioned_characters=True,  # Remove characters not mentioned in the story
    filter_unmentioned_objects=True,     # Remove objects not mentioned in the story
    filter_unmentioned_backgrounds=True  # Remove backgrounds not mentioned in the story
)

# Get a TRL-compatible sample
trl_sample = adapter[0]
```

## Applications

This dataset can be used for:

1. Training visual storytelling models that maintain consistent entity references
2. Developing cross-frame object re-identification systems
3. Researching structured reasoning approaches for narrative generation
4. Studying grounded text generation techniques
5. Evaluating models on their ability to generate coherent multi-frame narratives

## Citation

If you use this dataset in your research, please cite:

```
TODO
```

## Dataset Splits

The dataset is split into training and test sets:
- Train split: 85% of the stories (3,550 stories)
- Test split: 15% of the stories (628 stories)

## Statistics

- Total stories: 4,178
- Total frames: 52,016

## License

This dataset is licensed under [CC-BY-ND-4.0](https://creativecommons.org/licenses/by-nd/4.0/).

## Contact

For questions or feedback regarding the dataset, please contact:
- Daniel A. P. Oliveira (daniel.oliveira@inesc-id.pt)

## Related Work
This dataset builds upon our previous work on [GroundCap](https://huggingface.co/datasets/daniel3303/GroundCap), which introduced 
ID-based grounding for single images. While GroundCap demonstrated the effectiveness of structured entity references within individual 
frames, StoryReasoning extends this approach to sequential contexts, addressing the challenge of maintaining consistent identity across multiple images.
# StoryReasoning and Qwen Storyteller

A dataset and model framework for creating coherent visual stories with consistent character identity and object references across multiple frames.

## Overview

StoryReasoning is a dataset and framework that addresses key challenges in visual storytelling:
- Maintaining character and object identity across multiple frames
- Grounding textual elements to visual entities
- Reducing referential hallucinations
- Constructing coherent narratives from image sequences

This repository contains the code for Qwen Storyteller, a fine-tuned model based on Qwen2.5-VL 7B that performs end-to-end object detection, re-identification, and grounded story generation.

## Key Features

- **Cross-Frame Consistency**: Maintains consistent character and object identity across multiple frames through visual similarity and face recognition
- **Chain-of-Thought Reasoning**: Explicit modeling of characters, objects, settings, and narrative structure
- **Grounded Storytelling**: Uses specialized XML tags to link narrative elements directly to visual entities
- **Reduced Hallucinations**: Achieves 12.3% fewer hallucinations compared to the non-fine-tuned base model

## Dataset

The StoryReasoning dataset contains 4,178 stories derived from 52,016 movie images with structured scene analyses, grounded stories, and consistent character references.

Access the dataset: [StoryReasoning on HuggingFace](https://huggingface.co/datasets/daniel3303/StoryReasoning)

## Model

Qwen Storyteller is a fine-tuned version of Qwen2.5-VL 7B specialized for grounded visual storytelling.

Access the model: [Qwen Storyteller on HuggingFace](https://huggingface.co/daniel3303/QwenStoryteller)

## Installation

```bash
# Clone the repository
git clone https://github.com/daniel3303/StoryReasoning.git
cd StoryReasoning
pip install -e .

# Install dependencies
pip install -r requirements.txt
```

## Web Interface

The repository includes a web-based visualization interface that connects with vLLM for real-time story generation and interpretation. The interface is available by opening the file `story_reasoning/visualization/web_interface.html` in a web browser.
The interface provides:
- Real-time generation with stream processing
- Color-coded entity tags
- Direct visual grounding with bounding boxes
- Interactive character and object highlighting
- Table visualization for structured scene analysis

![Story Interface](images/story-interface.png)
*The story visualization component with color-coded entity tags that highlight corresponding visual elements when hovered*

![Tables Interface](images/cot-interface.png)
*The tabbed interface for Chain-of-Thought analysis showing structured information about characters, objects, and settings*

## Usage

### Basic Usage with Transformers

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image

# Load the model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "daniel3303/QwenStoryteller", torch_dtype="auto", device_map="auto"
)

# Load processor
processor = AutoProcessor.from_pretrained("daniel3303/QwenStoryteller")

# Load images
images = [
    Image.open("image1.jpg"),
    Image.open("image2.jpg"),
    Image.open("image3.jpg"),
]

# Create image content list
image_content = []
for img in images:
    image_content.append({
        "type": "image",
        "image": img,
    })

# Add text prompt at the end
image_content.append({"type": "text", "text": "Generate a story based on these images."})

# Create messages with system prompt
messages = [
    {
        "role": "system", 
        "content": "You are an AI storyteller that can analyze sequences of images and create creative narratives. First think step-by-step to analyze characters, objects, settings, and narrative structure. Then create a grounded story that maintains consistent character identity and object references across frames. Use <think></think> tags to show your reasoning process before writing the final story."
    },
    {
        "role": "user",
        "content": image_content,
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

# Generate the output
generated_ids = model.generate(
    **inputs, 
    max_new_tokens=4096,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
story = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print(story)
```

### Faster Inference with vLLM

For significantly faster inference, you can use vLLM to serve the model:

```bash
# Install vLLM
pip install vllm

# Serve the model with vLLM
vllm serve daniel3303/QwenStoryteller
```

You can then connect the web interface to the vLLM server for real-time story generation.

## Output Format

Qwen Storyteller produces two main outputs:

1. **Chain-of-Thought Analysis** (`<think></think>`): Structured analysis with character, object, setting, and narrative tables
2. **Grounded Story**: A narrative with specialized XML tags linking text to visual elements:
    - `<gdi>`: Image tags for specific frames
    - `<gdo>`: Entity reference tags for character and object mentions
    - `<gda>`: Action tags for character actions
    - `<gdl>`: Location/landmark tags for background elements

## Training Your Own Model

We provide the `story_reasoning/train/train_story_reasoning.py` script to fine-tune Qwen2.5-VL on the StoryReasoning dataset, this
script can be easily adapted to train any other vision-language model available in the HuggingFace library.

```bash
# Run LoRA fine-tuning with distributed training and high memory efficiency (using Liger Kernel)
accelerate launch --mixed_precision bf16 --fsdp_backward_prefetch NO_PREFETCH --fsdp_offload_params true --use_fsdp --fsdp_sharding_strategy 2 --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap Qwen2_5_VLVisionBlock,Qwen2_5_VLDecoderLayer --num_machines 1 --num_processes 2 --gpu_ids "0,1" --dynamo_backend no story_reasoning/train/train_story_reasoning.py --model Qwen/Qwen2.5-VL-7B-Instruct --hf_repo daniel3303/StoryReasoning --warmup_ratio 0.03 --weight_decay 0.01 --dataset_name story_reasoning --per_device_train_batch_size 1 --num_train_epochs 3 --save_steps 100  --save_total_limit 10 --max_seq_length 32768 --logging_steps 1  --learning_rate 2e-4 --gradient_accumulation_steps 32 --output_dir /tmp/qwen-story-reasoning-lora-r16-b64-lr2e4 --run_name qwen-story-reasoning-lora-r16-b64-lr2e4 --rank 16 --bf16 true --gradient_checkpointing true --torch_empty_cache_steps 1 --use_liger_kernel true

# Run full model fine-tuning with distributed training and high memory efficiency (using Liger Kernel)
accelerate launch --mixed_precision bf16 --fsdp_backward_prefetch NO_PREFETCH --fsdp_offload_params true --use_fsdp --fsdp_sharding_strategy 2 --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap Qwen2_5_VLVisionBlock,Qwen2_5_VLDecoderLayer --num_machines 1 --num_processes 2 --gpu_ids "0,1" --dynamo_backend no story_reasoning/train/train_story_reasoning.py --model Qwen/Qwen2.5-VL-7B-Instruct --hf_repo daniel3303/StoryReasoning --warmup_ratio 0.03 --weight_decay 0.01 --dataset_name story_reasoning --per_device_train_batch_size 1 --num_train_epochs 4 --save_steps 10  --save_total_limit 10 --max_seq_length 32768 --logging_steps 1 --learning_rate 2e-5 --gradient_accumulation_steps 32 --output_dir /tmp/qwen-story-reasoning-lora-fft-18-b64-lr2e5 --run_name qwen-story-reasoning-fft-18-b64-lr2e5 --full_finetune true --bf16 true --gradient_checkpointing true --torch_empty_cache_steps 1 --use_liger_kernel true

# Run language only fine-tuning with distributed training and high memory efficiency (using Liger Kernel)
accelerate launch --mixed_precision bf16 --fsdp_backward_prefetch NO_PREFETCH --fsdp_offload_params true --use_fsdp --fsdp_sharding_strategy 2 --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap Qwen2_5_VLVisionBlock,Qwen2_5_VLDecoderLayer --num_machines 1 --num_processes 2 --gpu_ids "0,1" --dynamo_backend no story_reasoning/train/train_story_reasoning.py --model Qwen/Qwen2.5-VL-7B-Instruct --hf_repo daniel3303/StoryReasoning --warmup_ratio 0.03 --weight_decay 0.01 --dataset_name story_reasoning --per_device_train_batch_size 1 --num_train_epochs 4 --save_steps 10  --save_total_limit 10 --max_seq_length 32768 --logging_steps 1 --learning_rate 2e-5 --gradient_accumulation_steps 32 --output_dir /tmp/qwen-story-reasoning-lora-fft-18-b64-lr2e5 --run_name qwen-story-reasoning-fft-18-b64-lr2e5 --language_finetune true --bf16 true --gradient_checkpointing true --torch_empty_cache_steps 1 --use_liger_kernel true 

```

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@misc{oliveira2025storyreasoningdatasetusingchainofthought,
    title={StoryReasoning Dataset: Using Chain-of-Thought for Scene Understanding and Grounded Story Generation},
    author={Daniel A. P. Oliveira and David Martins de Matos},
    year={2025},
    eprint={2505.10292},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2505.10292},
}
```

## Contact

Daniel A. P. Oliveira (daniel.oliveira@inesc-id.pt)
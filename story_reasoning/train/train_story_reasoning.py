import importlib

import torch
from peft import LoraConfig
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoModelForVision2Seq
from trl import SFTTrainer, SFTConfig, TrlParser

from story_reasoning.datasets import DatasetRegistry
from story_reasoning.train.config.data_config import DataConfig
from story_reasoning.train.config.extra_config import ExtraConfig
from story_reasoning.train.config.model_config import ModelConfig
from story_reasoning.train.training_util import get_most_recent_checkpoint

"""
    Example call:        
        accelerate launch --use_fsdp --fsdp_sharding_strategy 1 --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
        --fsdp_transformer_layer_cls_to_wrap Qwen2_5_VLVisionBlock,Qwen2_5_VLDecoderLayer  --mixed_precision bf16 \
        --num_machines 1 --num_processes 1 --gpu_ids "0" --dynamo_backend no story_reasoning/train/train_story_reasoning.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct --hf_repo daniel3303/StoryReasoning --warmup_ratio 0.03 --weight_decay 0.01 \
        --dataset_name story_reasoning --per_device_train_batch_size 1 --num_train_epochs 3 --save_steps 100  --save_total_limit 10 \
        --max_seq_length 32768 --logging_steps 1  --learning_rate 2e-4 \
        --gradient_accumulation_steps 32 --output_dir /tmp/u020529/qwen-story-reasoning-lora-r16-b64-lr2e4 \
        --run_name qwen-story-reasoning-lora-r16-b64-lr2e4 --rank 16
        
        
    Memory optimized mode with LoRA (you must run 'pip install liger-kernel' to use with Liger kernel)
        accelerate launch --mixed_precision bf16 --fsdp_backward_prefetch NO_PREFETCH --fsdp_offload_params true --use_fsdp --fsdp_sharding_strategy 2 --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap Qwen2_5_VLVisionBlock,Qwen2_5_VLDecoderLayer --num_machines 1 --num_processes 2 --gpu_ids "0,1" --dynamo_backend no story_reasoning/train/train_story_reasoning.py --model Qwen/Qwen2.5-VL-7B-Instruct --hf_repo daniel3303/StoryReasoning --warmup_ratio 0.03 --weight_decay 0.01 --dataset_name story_reasoning --per_device_train_batch_size 1 --num_train_epochs 3 --save_steps 100  --save_total_limit 10 --max_seq_length 32768 --logging_steps 1  --learning_rate 2e-4 --gradient_accumulation_steps 32 --output_dir /tmp/u020529/qwen-story-reasoning-lora-r16-b64-lr2e4 --run_name qwen-story-reasoning-lora-r16-b64-lr2e4 --rank 16 --bf16 true --gradient_checkpointing true --torch_empty_cache_steps 1 --use_liger_kernel true
        
    Memory optimized mode with full fine-tuning (you must run 'pip install liger-kernel' to use with Liger kernel)
        accelerate launch --mixed_precision bf16 --fsdp_backward_prefetch NO_PREFETCH --fsdp_offload_params true --use_fsdp --fsdp_sharding_strategy 2 --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap Qwen2_5_VLVisionBlock,Qwen2_5_VLDecoderLayer --num_machines 1 --num_processes 2 --gpu_ids "0,1" --dynamo_backend no story_reasoning/train/train_story_reasoning.py --model Qwen/Qwen2.5-VL-7B-Instruct --hf_repo daniel3303/StoryReasoning --warmup_ratio 0.03 --weight_decay 0.01 --dataset_name story_reasoning --per_device_train_batch_size 1 --num_train_epochs 4 --save_steps 10  --save_total_limit 10 --max_seq_length 32768 --logging_steps 1 --learning_rate 2e-5 --gradient_accumulation_steps 32 --output_dir /tmp/u020529/qwen-story-reasoning-lora-fft-18-b64-lr2e5 --run_name qwen-story-reasoning-fft-18-b64-lr2e5 --full_finetune true --bf16 true --gradient_checkpointing true --torch_empty_cache_steps 1 --use_liger_kernel true
        
    Memory optimized mode with fine-tuning of the language model only (you must run 'pip install liger-kernel' to use with Liger kernel)
        accelerate launch --mixed_precision bf16 --fsdp_backward_prefetch NO_PREFETCH --fsdp_offload_params true --use_fsdp --fsdp_sharding_strategy 2 --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap Qwen2_5_VLVisionBlock,Qwen2_5_VLDecoderLayer --num_machines 1 --num_processes 2 --gpu_ids "0,1" --dynamo_backend no story_reasoning/train/train_story_reasoning.py --model Qwen/Qwen2.5-VL-7B-Instruct --hf_repo daniel3303/StoryReasoning --warmup_ratio 0.03 --weight_decay 0.01 --dataset_name story_reasoning --per_device_train_batch_size 1 --num_train_epochs 4 --save_steps 10  --save_total_limit 10 --max_seq_length 32768 --logging_steps 1 --learning_rate 2e-5 --gradient_accumulation_steps 32 --output_dir /tmp/u020529/qwen-story-reasoning-lora-fft-18-b64-lr2e5 --run_name qwen-story-reasoning-fft-18-b64-lr2e5 --language_finetune true --bf16 true --gradient_checkpointing true --torch_empty_cache_steps 1 --use_liger_kernel true 
"""


def train(
        model_args: ModelConfig,
        data_args: DataConfig,
        training_args: SFTConfig,
        extra_config: ExtraConfig
):
    """Main training function."""

    # Setup model and tokenizer
    processor = AutoProcessor.from_pretrained(model_args.model)

    model = AutoModelForVision2Seq.from_pretrained(
        model_args.model,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
        token=model_args.token,
        attn_implementation="flash_attention_2" if training_args.bf16 or training_args.fp16 else None, # requires installing flash_attention manually (pip install flash-attn --no-build-isolation)
    )


    # Get dataset class and adapter from registry
    dataset_class = DatasetRegistry.get_dataset(data_args.dataset_name)
    adapter_class = DatasetRegistry.get_dataset_adapter(data_args.dataset_name)

    if dataset_class is None:
        raise ValueError(f"Dataset {data_args.dataset_name} not found in registry")
    if adapter_class is None:
        raise ValueError(f"Dataset adapter for {data_args.dataset_name} not found in registry")

    # Create two separate datasets for auto and human annotations
    train_dataset = dataset_class(
        path=data_args.dataset_path,
        hf_repo=data_args.hf_repo,
        split="train"
    )
    train_dataset_adapter = adapter_class(train_dataset)

    eval_dataset = dataset_class(
        path=data_args.dataset_path,
        hf_repo=data_args.hf_repo,
        split="test"
    )
    eval_dataset_adapter = adapter_class(eval_dataset)

    if extra_config.full_finetune:
        # Setup full fine-tuning
        for param in model.parameters():
            param.requires_grad = True


        print(f"Full fine-tuning mode: Unfrozen all layers")

        # Disable LoRA when doing full fine-tuning
        lora_config = None
    elif extra_config.language_finetune:
        # Setup language fine-tuning by freezing vision encoder and unfreezing language model
    
        # First unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True
            
        
        # Then freeze only the vision encoder
        for param in model.visual.parameters():
            param.requires_grad = False
            
        # Total frozen parameters
        print("Total frozen parameters (vision encoder):", sum(p.numel() for p in model.parameters() if not p.requires_grad))
            
    
        print(f"Language fine-tuning mode: Frozen vision encoder, unfrozen language model")
    
        # Disable LoRA when doing language fine-tuning
        lora_config = None
    else:
        # Setup LoRA
        lora_config = LoraConfig(
            r=extra_config.rank,  # Rank of the adaptation matrix
            lora_alpha=extra_config.rank * 2,  # Scaling factor for LoRA weights
            target_modules=[
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
            ],  # Target modules for LoRA adaptation
            lora_dropout=extra_config.dropout,  # Dropout rate for LoRA layers
            bias="none",  # Options: 'none', 'all', 'lora_only'
            task_type="CAUSAL_LM"  # Task type: Causal Language Modeling
        )

    def data_collator(examples):
        # Get the texts and images, and apply the chat template
        texts = [processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False) for
                 example in examples]
        images = [process_vision_info(example["messages"])[0] for example in
                  examples]  # Process the messages to extract input images
        

        # Tokenize the texts and images 
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        
        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100  #
        
        
        # Ignore the image token index in the loss computation (Qwen model specific)
        image_tokens = [
            processor.tokenizer.convert_tokens_to_ids("<|vision_start|>"),
            processor.tokenizer.convert_tokens_to_ids("<|vision_end|>"),
            processor.tokenizer.convert_tokens_to_ids("<|vision_pad|>"),
            processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        ]

        # Mask image token IDs in the labels
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100  # Mask image token IDs in labels

        batch["labels"] = labels  # Add labels to the batch
        return batch

    checkpoint_dir = f"{training_args.output_dir}/checkpoints"
    final_model_dir = f"{checkpoint_dir}/final_model"

    # Auto-resume from most recent checkpoint if enabled
    if extra_config.auto_resume_checkpoint and not training_args.resume_from_checkpoint:
        most_recent_checkpoint = get_most_recent_checkpoint(checkpoint_dir)
        if most_recent_checkpoint:
            print(f"Auto-resuming from most recent checkpoint: {most_recent_checkpoint}")
            training_args.resume_from_checkpoint = most_recent_checkpoint



    # Set remove_unused_columns to False to prevent errors with the data collator
    training_args.remove_unused_columns = False
    training_args.output_dir = checkpoint_dir
    
    # Prevent preparing the dataset to avoid loading errors
    training_args.dataset_kwargs = {
        "skip_prepare_dataset": True,
    }
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_adapter,
        eval_dataset=eval_dataset_adapter,
        processing_class=processor.tokenizer,
        peft_config=lora_config,
        data_collator=data_collator,
    )
    print("Total trainable parameters:", trainer.get_num_trainable_parameters())
    trainer.train(training_args.resume_from_checkpoint)
    
    print("Saving model to:", final_model_dir)
    trainer.save_model(output_dir=final_model_dir)
    
    print("Saving processor to:", final_model_dir)
    processor.save_pretrained(final_model_dir)

    print("Training completed.")


def main():
    parser = TrlParser((ModelConfig, DataConfig, SFTConfig, ExtraConfig))
    model_args, data_args, training_args, extra_config = parser.parse_args_into_dataclasses()

    # dynamically loads the module story_reasoning.datasets.[dataset_name] to register the dataset and adapter
    importlib.import_module(f"story_reasoning.datasets.{data_args.dataset_name}")

    train(model_args, data_args, training_args, extra_config)


if __name__ == "__main__":
    main()

""""
Qwen Architecture (For LoRA reference)

Qwen2_5_VLForConditionalGeneration(
  (visual): Qwen2_5_VisionTransformerPretrainedModel(
    (patch_embed): Qwen2_5_VisionPatchEmbed(
      (proj): Conv3d(3, 1280, kernel_size=(2, 14, 14), stride=(2, 14, 14), bias=False)
    )
    (rotary_pos_emb): Qwen2_5_VisionRotaryEmbedding()
    (blocks): ModuleList(
      (0-31): 32 x Qwen2_5_VLVisionBlock(
        (norm1): Qwen2RMSNorm((1280,), eps=1e-06)
        (norm2): Qwen2RMSNorm((1280,), eps=1e-06)
        (attn): Qwen2_5_VLVisionSdpaAttention(
          (qkv): Linear(in_features=1280, out_features=3840, bias=True)
          (proj): Linear(in_features=1280, out_features=1280, bias=True)
        )
        (mlp): Qwen2_5_VLMLP(
          (gate_proj): Linear(in_features=1280, out_features=3420, bias=True)
          (up_proj): Linear(in_features=1280, out_features=3420, bias=True)
          (down_proj): Linear(in_features=3420, out_features=1280, bias=True)
          (act_fn): SiLU()
        )
      )
    )
    (merger): Qwen2_5_VLPatchMerger(
      (ln_q): Qwen2RMSNorm((1280,), eps=1e-06)
      (mlp): Sequential(
        (0): Linear(in_features=5120, out_features=5120, bias=True)
        (1): GELU(approximate='none')
        (2): Linear(in_features=5120, out_features=2048, bias=True)
      )
    )
  )
  (model): Qwen2_5_VLModel(
    (embed_tokens): Embedding(151936, 2048)
    (layers): ModuleList(
      (0-35): 36 x Qwen2_5_VLDecoderLayer(
        (self_attn): Qwen2_5_VLSdpaAttention(
          (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
          (k_proj): Linear(in_features=2048, out_features=256, bias=True)
          (v_proj): Linear(in_features=2048, out_features=256, bias=True)
          (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (rotary_emb): Qwen2_5_VLRotaryEmbedding()
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=2048, out_features=11008, bias=False)
          (up_proj): Linear(in_features=2048, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=2048, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm((2048,), eps=1e-06)
        (post_attention_layernorm): Qwen2RMSNorm((2048,), eps=1e-06)
      )
    )
    (norm): Qwen2RMSNorm((2048,), eps=1e-06)
    (rotary_emb): Qwen2_5_VLRotaryEmbedding()
  )
  (lm_head): Linear(in_features=2048, out_features=151936, bias=False)
)

"""

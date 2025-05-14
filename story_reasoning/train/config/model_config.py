from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    model: str = field(
        default="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        metadata={"help": "LLaVA compatible model identifier from huggingface.co/models. For instance, 'llava-hf/llava-onevision-qwen2-0.5b-ov-hf', 'mistral-community/pixtral-12b', etc."}
    )

    token: Optional[str] = field(
        default=None,
        metadata={"help": "HuggingFace token for private model access"}
    )


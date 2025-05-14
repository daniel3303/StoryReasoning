from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ExtraConfig:
    """Configuration for additional training parameters beyond basic SFT settings."""

    rank: Optional[int] = field(
        default=None,
        metadata={"help": "LoRA adaptation rank. Higher values (8-64) increase capacity but require more memory. "
                          "If True, full_finetune and language_finetune must be False. " }
    )

    full_finetune: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to perform full fine-tuning. If True, rank and language_finetune must be False. "}
    )


    language_finetune: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to perform fine-tuning only on the language layers of the model. If True, rank and full_finetune must be False. "}
    )


    dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout probability for LoRA layers (0.0-0.5). Only used when rank is specified."}
    )

    auto_resume_checkpoint: bool = field(
        default=True,
        metadata={"help": "When True, automatically resume from the most recent checkpoint in the output directory."}
    )

    def __post_init__(self):
        """Validate that only one finetuning method is specified."""
        if self.rank and (self.full_finetune or self.language_finetune):
            raise ValueError(
                "Cannot use LoRA (rank) and full finetuning / language finetuning together. "
                "Please specify only one approach."
            )

        if self.full_finetune and (self.language_finetune or self.rank):
            raise ValueError(
                "Cannot use full finetuning and language finetuning / LoRA together. "
                "Please specify only one approach."
            )
        
        if self.language_finetune and (self.full_finetune or self.rank):
            raise ValueError(
                "Cannot use language finetuning and full finetuning / LoRA together. "
                "Please specify only one approach."
            )
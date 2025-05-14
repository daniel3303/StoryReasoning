import os
from glob import glob
from typing import Union


def get_most_recent_checkpoint(checkpoint_dir: str) -> Union[str, None]:
    """
    Returns the path to the checkpoint with the highest step number in the given checkpoint directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoint folders
        
    Returns:
        Path to the checkpoint with highest step number, or None if no checkpoints found
    """
    checkpoint_folders = glob(f"{checkpoint_dir}/checkpoint-*")
    if not checkpoint_folders:
        return None

    # Extract step numbers from checkpoint folder names and sort numerically
    def get_step_number(checkpoint_path):
        # Extract the number after "checkpoint-"
        try:
            step = int(checkpoint_path.split("checkpoint-")[-1])
            return step
        except ValueError:
            # If conversion fails, return -1 to place it at the beginning of the sorted list
            return -1

    # Sort checkpoints by step number (highest last)
    checkpoint_folders.sort(key=get_step_number)

    # Return the checkpoint with the highest step number
    return checkpoint_folders[-1]
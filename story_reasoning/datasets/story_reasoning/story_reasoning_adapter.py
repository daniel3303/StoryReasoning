import random
from typing import Dict, Any, Union

from torch.utils.data import Dataset

from story_reasoning.datasets import DatasetRegistry
from story_reasoning.models.story_reasoning.story_reasoning_util import StoryReasoningUtil
from story_reasoning.train.base_tlr_dataset_adapter import BaseTlrDatasetAdapter


@DatasetRegistry.register_tlr_dataset_adapter("story_reasoning")
class StoryReasoningAdapter(BaseTlrDatasetAdapter):
    """
    Adapter for the Story Reasoning dataset for the TRL framework.
    Args:
        dataset (Dataset): The dataset to adapt.
        max_images (int|None): The maximum number of images to include in the story. If None, all images are included. 
            If set, the images will be truncated to this number and the CoT and story will be adjusted accordingly.
        filter_unmentioned_characters (bool): Whether to filter unmentioned characters from the CoT. This will remove
            characters from the CoT that are not mentioned in the story.
        filter_unmentioned_objects (bool): Whether to filter unmentioned objects from the CoT. This will remove
            objects from the CoT that are not mentioned in the story.
        filter_unmentioned_backgrounds (bool): Whether to filter unmentioned backgrounds from the CoT. This will remove 
            backgrounds from the CoT that are not mentioned in the story.
    """
    def __init__(self, dataset: Dataset, max_images: Union[int|None] = None, filter_unmentioned_characters: bool = False, filter_unmentioned_objects: bool = False, filter_unmentioned_backgrounds: bool = True):
        BaseTlrDatasetAdapter.__init__(self, dataset)
        self.filter_unmentioned_backgrounds = filter_unmentioned_backgrounds
        self.filter_unmentioned_objects = filter_unmentioned_objects
        self.filter_unmentioned_characters = filter_unmentioned_characters
        self.max_images = max_images

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Process a single example from the dataset.
        Should return a conversational format supported by the TRL trainers
        with multiple images and both chain-of-thought and story content.
        """
        item = self.dataset[index]

        # Create a messages list with the system prompt
        messages = [{"role": "system", "content": self.default_system_prompt()}]
        
        # Construct the user message (without image placeholders as they'll be added separately)
        user_prompt = self.get_random_user_prompt()

        user_message_content = [
            {
                "type": "text",
                "text": user_prompt
            }
        ]
    
        # Truncates the images to self.max_images if set
        images = item["images"]
        if self.max_images is not None and len(images) > self.max_images:
            images = images[:self.max_images]
            print(f"Warning: Story {item["story_id"]} found {len(item['images'])} images, but only using {len(images)} images.")
            
    
                
        # Add image objects to the messages
        for img in images:
            user_message_content.append({
                "type": "image",
                "image": img
            })

        # Add user message
        messages.append({"role": "user", "content": user_message_content})
        
        # Updates the CoT and Story if images where removed from the sample
        cot_string = item["chain_of_thought"]
        story_string = item["story"]
        if self.max_images is not None and len(item["images"]) > self.max_images:
            cot = StoryReasoningUtil.parse_cot(cot_string)
            cot.images = cot.images[:self.max_images]
            cot_string = StoryReasoningUtil.cot_to_string(cot)
            
            story = StoryReasoningUtil.parse_story(story_string)
            story.images = story.images[:self.max_images]
            story_string = StoryReasoningUtil.story_to_string(story)

        # Filtering of unused characters / objects / backgrounds
        if self.filter_unmentioned_characters or self.filter_unmentioned_objects or self.filter_unmentioned_backgrounds:
            cot = StoryReasoningUtil.parse_cot(cot_string)
            story = StoryReasoningUtil.parse_story(story_string)
            filtered_cot = StoryReasoningUtil.filter_unmentioned_entities(cot, story, self.filter_unmentioned_characters, self.filter_unmentioned_objects, self.filter_unmentioned_backgrounds)
            cot_string = StoryReasoningUtil.cot_to_string(filtered_cot)

        # Format the assistant response with thinking tag for CoT and story
        assistant_message = (
            f"<think>\n{cot_string}\n</think>\n"
            f"{story_string}"
        )        
                
                
        # print(f"Story {item['story_id']}\t{len(images)} images and CoT + Story len: {len(cot_string + story_string)} chars. Original len: {len(item['chain_of_thought'] + item['story'])} chars.")

        # Add assistant message with thinking and story
        messages.append({"role": "assistant", "content": assistant_message})

        # Return in the conversation format expected by TRL
        return {
            "id": item["story_id"],  # Unique ID for the sample
            "messages": messages,
            "images": images,
        }

    def default_system_prompt(self) -> str:
        return (
            "You are an AI storyteller that can analyze sequences of images and create creative narratives. "
            "First think step-by-step to analyze characters, objects, settings, and narrative structure. "
            "Then create a grounded story that maintains consistent character identity and object references across frames. "
            "Use <think></think> tags to show your reasoning process before writing the final story."
        )

    def get_random_user_prompt(self) -> str:
        prompts = [
            "",
            "Create a story based on these images.",
            "Tell me a story that connects these images.",
            "Generate a narrative that ties these images together.",
            "Craft a cohesive story from this sequence of images.",
            "Looking at these images, what story could they tell?",
            "Can you create a narrative that explains the progression in these images?",
            "These images form a sequence - please tell the story they represent.",
            "Write a story that captures what's happening across these images.",
            "These images tell a story - what is it?",
        ]
        return random.choice(prompts)

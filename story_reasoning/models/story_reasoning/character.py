from dataclasses import dataclass


@dataclass
class Character:
    character_id: str
    name: str
    description: str
    emotions: str
    actions: str
    narrative_function: str
    bounding_box: str
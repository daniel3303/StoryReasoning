from dataclasses import dataclass


@dataclass
class Object:
    object_id: str
    description: str
    function: str
    interaction: str
    narrative_function: str
    bounding_box: str
import enum
from dataclasses import dataclass


class SettingElement(enum.Enum):
    """Enumeration of possible setting elements"""
    LOCATION = "Location"
    ENVIRONMENT = "Environment"
    LIGHTING = "Lighting"
    WEATHER = "Weather"
    TIME_PERIOD = "Time Period"
    ARCHITECTURE = "Architecture"
    INTERIOR_DESIGN = "Interior Design"
    ATMOSPHERE = "Atmosphere"
    BACKGROUND = "Background"


@dataclass
class Setting:
    setting_element: SettingElement
    description: str
    mood: str
    time: str
    narrative_function: str
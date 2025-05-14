from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class NarrativeStructure:
    phases: List[Dict[str, Any]] = field(default_factory=list)
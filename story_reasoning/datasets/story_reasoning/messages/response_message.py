from dataclasses import dataclass, field


@dataclass
class ResponseMessage:
    """
    A class to represent a response from a language model.
    """
    content: str = field(default=None)
    role: str = field(default="assistant")

    def __init__(self, content, role="assistant"):
        self.content = content
        self.role = role

    def to_dict(self):
        """
        Convert the ResponseMessage object to a dictionary.
        """
        return {
            "role": self.role,
            "content": self.content
        }

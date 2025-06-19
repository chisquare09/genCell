import yaml
from typing import Any
from pathlib import Path

class Config:
    """Dynamically set attributes from any YAML file."""
    def __init__(self, data: dict[str, Any]):
        for k, v in data.items():
            setattr(self, k, v)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(data)

"""I/O utilities for JSON configs and results."""
import json
from pathlib import Path


def load_json(path):
    """Load a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path):
    """Save an object as JSON with pretty formatting."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

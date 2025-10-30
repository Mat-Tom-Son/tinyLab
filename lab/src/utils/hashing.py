"""Hashing utilities for reproducibility."""
import hashlib
import json


def sha256_json(obj):
    """Hash a JSON-serializable object deterministically."""
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode()).hexdigest()


def sha256_file(path):
    """Hash a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

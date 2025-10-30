"""Dataset loading with hash validation for reproducibility."""
import json
import hashlib
from pathlib import Path
from datasets import load_dataset


def hash_file(path):
    """Hash a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_split(ds_cfg):
    """Load a dataset split with hash validation.

    Args:
        ds_cfg: Dict with keys:
            - id: dataset name (e.g., "facts_v1")
            - split: split name (e.g., "train", "val", "test")

    Returns:
        (rows, split_info, data_hash)
    """
    root = Path("lab/data")
    corpus_path = root / "corpora" / f"{ds_cfg['id']}.jsonl"
    data_hash = hash_file(corpus_path)

    split_path = root / "splits" / f"{ds_cfg['id']}.split.json"
    split = json.loads(split_path.read_text())

    # Sanity: split must match current data hash
    if split.get("data_hash") and split["data_hash"] != data_hash:
        raise RuntimeError(
            f"Data hash mismatch for {ds_cfg['id']}. "
            f"Split file hash: {split.get('data_hash')}, "
            f"Corpus hash: {data_hash}. "
            "Regenerate split file."
        )

    # Use HF datasets to lazy-load
    dset = load_dataset("json", data_files=str(corpus_path), split="train")

    indices = split[ds_cfg["split"]]
    rows = [dset[i] for i in indices]  # Load only the requested split

    print(f"Loaded {len(rows)} examples from split '{ds_cfg['split']}'")
    return rows, split, data_hash

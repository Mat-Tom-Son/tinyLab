"""Model loading utilities for TransformerLens."""
import torch
from transformer_lens import HookedTransformer


def load_transformerlens(model_cfg, device="auto"):
    """Load a TransformerLens model.

    Args:
        model_cfg: Dict with keys:
            - name: model name (e.g., "gpt2-small", "gpt2-medium")
            - dtype: dtype string (e.g., "float32", "float16", "bfloat16")
        device: Device to load on ("auto", "cpu", "mps", "cuda")

    Returns:
        HookedTransformer model
    """
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(model_cfg.get("dtype", "float32"), torch.float32)

    # Optional: reduce MPS memory fragmentation
    if device == "mps" and hasattr(torch.mps, "set_per_process_memory_fraction"):
        try:
            torch.mps.set_per_process_memory_fraction(0.90)
        except Exception as e:
            print(f"Warning: Could not set MPS memory fraction: {e}")

    name = model_cfg.get("name", "gpt2-small")
    hf_repo = model_cfg.get("hf_model")
    revision = model_cfg.get("revision")  # Optional HF revision/tag (e.g., step checkpoints)
    print(f"Loading model: {name} to {device} with {dtype}")
    if hf_repo:
        kwargs = {"device": device, "dtype": dtype, "hf_model": hf_repo}
        if revision:
            kwargs["revision"] = revision
        model = HookedTransformer.from_pretrained(name, **kwargs)
    else:
        kwargs = {"device": device, "dtype": dtype}
        if revision:
            kwargs["revision"] = revision
        model = HookedTransformer.from_pretrained(name, **kwargs)
    return model

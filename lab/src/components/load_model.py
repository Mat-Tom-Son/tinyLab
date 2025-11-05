"""Model loading utilities for TransformerLens."""
import torch
from transformer_lens import HookedTransformer


def load_transformerlens(model_cfg, device="auto"):
    """Load a TransformerLens model.

    Args:
        model_cfg: Dict with keys:
            - name: model name (e.g., "gpt2-small", "gpt2-medium")
            - dtype: dtype string (e.g., "float32", "float16", "bfloat16")
            - low_memory: bool, enable memory optimizations for limited VRAM (default: False)
        device: Device to load on ("auto", "cpu", "mps", "cuda", "cuda:0", etc.)

    Returns:
        HookedTransformer model
    """
    # Auto-detect device with priority: CUDA > MPS > CPU
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(model_cfg.get("dtype", "float32"), torch.float32)

    # Memory optimization for MPS
    if device == "mps" and hasattr(torch.mps, "set_per_process_memory_fraction"):
        try:
            torch.mps.set_per_process_memory_fraction(0.90)
        except Exception as e:
            print(f"Warning: Could not set MPS memory fraction: {e}")

    # Memory optimization for CUDA
    if device.startswith("cuda"):
        low_memory = model_cfg.get("low_memory", False)
        if low_memory:
            # Enable memory-efficient settings
            torch.cuda.empty_cache()
            # Enable TF32 for faster computation on Ampere+ GPUs
            if torch.cuda.get_device_capability()[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            print(f"✓ Low-memory mode enabled for CUDA")

        # Display CUDA memory info
        if torch.cuda.is_available():
            device_idx = 0 if device == "cuda" else int(device.split(":")[-1])
            props = torch.cuda.get_device_properties(device_idx)
            mem_gb = props.total_memory / (1024**3)
            print(f"CUDA device: {props.name} ({mem_gb:.2f} GB total memory)")

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

    # Post-load CUDA memory report
    if device.startswith("cuda") and torch.cuda.is_available():
        device_idx = 0 if device == "cuda" else int(device.split(":")[-1])
        allocated = torch.cuda.memory_allocated(device_idx) / (1024**3)
        reserved = torch.cuda.memory_reserved(device_idx) / (1024**3)
        print(f"CUDA memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

    return model

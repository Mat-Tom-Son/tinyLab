"""Head-scaling hooks for TransformerLens models.

These utilities build forward hooks that scale the contribution of specific
attention heads at given layers, by multiplying their ``hook_z`` outputs
by a factor :math:`\\alpha`.

Intended use (Stage 1A pilot):
    - Select a suppressor head in layer 0 and a random-head control.
    - Build scaling hooks with per-head factors (e.g., {head: alpha}).
    - Pass the resulting ``(node, hook)`` pairs to ``model.run_with_hooks`` or
      to a training loop that wraps each forward pass.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch


HookSpec = Tuple[str, callable]


def make_layer_head_scaler(layer: int, head_to_alpha: Dict[int, float]) -> HookSpec:
    """Create a scaling hook for a single layer over multiple heads.

    Args:
        layer: Transformer block index to target.
        head_to_alpha: Mapping from head index -> scaling factor alpha.

    Returns:
        A ``(node_name, hook_fn)`` pair suitable for ``run_with_hooks``.
    """
    node = f"blocks.{layer}.attn.hook_z"

    # Make a local copy to avoid accidental mutation by callers
    head_to_alpha = dict(head_to_alpha)

    def hook(z: torch.Tensor, _hook) -> torch.Tensor:
        # z: [batch, seq, n_heads, d_head]
        z = z.clone()
        for h, alpha in head_to_alpha.items():
            if h < 0 or h >= z.shape[2]:
                continue
            z[:, :, h, :] = z[:, :, h, :] * alpha
        return z

    return node, hook


def build_scaling_hooks(
    layer_head_config: Dict[Tuple[int, int], float]
) -> List[HookSpec]:
    """Build scaling hooks from a {(layer, head): alpha} mapping.

    Args:
        layer_head_config: Mapping from (layer_index, head_index) to alpha.

    Returns:
        List of ``(node_name, hook_fn)`` pairs, one per layer with at least
        one head to scale.
    """
    per_layer: Dict[int, Dict[int, float]] = {}
    for (layer, head), alpha in layer_head_config.items():
        if layer not in per_layer:
            per_layer[layer] = {}
        per_layer[layer][head] = alpha

    hooks: List[HookSpec] = []
    for layer, head_map in per_layer.items():
        hooks.append(make_layer_head_scaler(layer, head_map))
    return hooks


def example_usage(model) -> None:
    """Illustrative usage for documentation/testing.

    Example:
        - Scale layer-0 head 3 by alpha=0.5 (damping)
        - Scale layer-0 head 5 by alpha=1.5 (amplification)
    """
    config = {(0, 3): 0.5, (0, 5): 1.5}
    hooks = build_scaling_hooks(config)
    tokens = model.to_tokens("Example prompt")
    _ = model.run_with_hooks(tokens, fwd_hooks=hooks)


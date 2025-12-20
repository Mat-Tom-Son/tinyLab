#!/usr/bin/env python3
"""
Dual-Timescale Training Wrapper for Phase 2 Experiments

Wraps GrokkingTransformer to add dual-timescale training without refactoring.

Key idea:
- Fast loop (Layer 0): High learning rate, learns task-specific features quickly
- Slow loop (Layers 1+): Low learning rate, maintains homeostatic equilibrium

Based on Phase 1 findings:
- Layer 0 crystallizes into gatekeeper role (VDI = 0.61)
- Crystallization happens over 1500 steps (steps 3000-4500)
- Phase 2 goal: compress this window via dual-timescale training
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class DualTimescaleWrapper:
    """
    Wraps GrokkingTransformer to enable dual-timescale training.

    Separates model parameters into:
    - Fast layers: Layer 0 (high LR, task learning)
    - Slow layers: Layers 1+ (low LR, homeostatic regulation)
    """

    def __init__(
        self,
        model: nn.Module,
        base_lr: float = 0.001,
        weight_decay: float = 1.0,
        config: Optional[Dict] = None,
    ):
        """
        Args:
            model: GrokkingTransformer instance
            base_lr: Base learning rate
            weight_decay: Weight decay (AdamW default: 1.0)
            config: Timescale configuration dict with:
                - fast_lr_scale: LR multiplier for Layer 0 (default: 1.0)
                - slow_lr_scale: LR multiplier for Layers 1+ (default: 0.1)
        """
        self.model = model
        self.base_lr = base_lr
        self.weight_decay = weight_decay

        # Default config
        if config is None:
            config = {
                'fast_lr_scale': 1.0,
                'slow_lr_scale': 1.0,  # Baseline: same LR for all layers
            }
        self.config = config

        # Identify fast and slow layers
        self.fast_layers, self.slow_layers = self._partition_layers(model)

        # Create dual optimizers
        self.fast_opt, self.slow_opt = self._create_optimizers()

        # Track gradient norms for diagnostics
        self.fast_grad_norm = 0.0
        self.slow_grad_norm = 0.0

    def _partition_layers(self, model):
        """
        Partition model into fast (Layer 0) and slow (Layers 1+) components.

        For GrokkingTransformer:
        - Fast: blocks[0] (Layer 0)
        - Slow: blocks[1:] (Layers 1+)
        """
        fast_layers = []
        slow_layers = []

        # Layer 0 = fast
        if hasattr(model, 'blocks') and len(model.blocks) > 0:
            fast_layers.append(model.blocks[0])

        # Layers 1+ = slow
        if hasattr(model, 'blocks') and len(model.blocks) > 1:
            slow_layers.extend(model.blocks[1:])

        # Embeddings and output head: assign to slow (stable)
        if hasattr(model, 'token_emb'):
            slow_layers.append(model.token_emb)
        if hasattr(model, 'pos_emb'):
            slow_layers.append(model.pos_emb)
        if hasattr(model, 'output'):
            slow_layers.append(model.output)

        return fast_layers, slow_layers

    def _get_params(self, layers: List[nn.Module]):
        """Extract parameters from list of layers."""
        params = []
        for layer in layers:
            params.extend(layer.parameters())
        return params

    def _create_optimizers(self):
        """
        Create separate optimizers for fast and slow loops.

        Fast optimizer:
        - Higher learning rate (base_lr × fast_lr_scale)
        - Lower momentum (0.9) - more responsive

        Slow optimizer:
        - Lower learning rate (base_lr × slow_lr_scale)
        - Higher momentum (0.99) - more stable
        """
        fast_params = self._get_params(self.fast_layers)
        slow_params = self._get_params(self.slow_layers)

        fast_lr = self.base_lr * self.config['fast_lr_scale']
        slow_lr = self.base_lr * self.config['slow_lr_scale']

        # Use AdamW to match Stage-1B baseline
        fast_opt = torch.optim.AdamW(
            fast_params,
            lr=fast_lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),  # Slightly higher beta2 for stability
        )

        slow_opt = torch.optim.AdamW(
            slow_params,
            lr=slow_lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.99),  # Higher beta2 for slow, stable updates
        )

        return fast_opt, slow_opt

    def training_step(self, loss: torch.Tensor, phase: str = 'both', grad_clip: float = 1.0):
        """
        Single training step with dual timescale.

        Args:
            loss: Total loss (task + homeostatic components)
            phase: Which optimizer(s) to update
                - 'fast': Only update Layer 0
                - 'slow': Only update Layers 1+
                - 'both': Update all layers (default)
            grad_clip: Gradient clipping threshold

        Returns:
            dict with gradient norms for diagnostics
        """
        if phase in ['fast', 'both']:
            self.fast_opt.zero_grad()

        if phase in ['slow', 'both']:
            self.slow_opt.zero_grad()

        # Backward pass
        if phase == 'both':
            # Both optimizers need gradients
            loss.backward()
        else:
            # Single optimizer
            loss.backward()

        # Clip gradients and measure norms
        if phase in ['fast', 'both']:
            fast_params = self._get_params(self.fast_layers)
            self.fast_grad_norm = torch.nn.utils.clip_grad_norm_(
                fast_params, grad_clip
            ).item()

        if phase in ['slow', 'both']:
            slow_params = self._get_params(self.slow_layers)
            self.slow_grad_norm = torch.nn.utils.clip_grad_norm_(
                slow_params, grad_clip
            ).item()

        # Optimizer steps
        if phase in ['fast', 'both']:
            self.fast_opt.step()

        if phase in ['slow', 'both']:
            self.slow_opt.step()

        return {
            'fast_grad_norm': self.fast_grad_norm,
            'slow_grad_norm': self.slow_grad_norm,
        }

    def get_layer_divergence(self) -> float:
        """
        Measure divergence between fast and slow layer parameters.

        Higher divergence = more separation between timescales.
        Useful diagnostic: should increase during training.

        Returns:
            L2 distance between Layer 0 and Layer 1 parameters
        """
        if len(self.fast_layers) == 0 or len(self.slow_layers) == 0:
            return 0.0

        # Get Layer 0 params
        fast_params = torch.cat([p.flatten() for p in self.fast_layers[0].parameters()])

        # Get Layer 1 params (first slow layer)
        slow_params = torch.cat([p.flatten() for p in self.slow_layers[0].parameters()])

        # L2 distance (normalized by parameter count)
        divergence = (fast_params - slow_params).pow(2).mean().sqrt().item()

        return divergence

    def get_learning_rates(self) -> Dict[str, float]:
        """Get current learning rates for diagnostics."""
        return {
            'fast_lr': self.fast_opt.param_groups[0]['lr'],
            'slow_lr': self.slow_opt.param_groups[0]['lr'],
            'lr_ratio': self.fast_opt.param_groups[0]['lr'] / (self.slow_opt.param_groups[0]['lr'] + 1e-10),
        }

    def state_dict(self) -> Dict:
        """Save optimizer states."""
        return {
            'fast_opt': self.fast_opt.state_dict(),
            'slow_opt': self.slow_opt.state_dict(),
            'config': self.config,
        }

    def load_state_dict(self, state_dict: Dict):
        """Load optimizer states."""
        self.fast_opt.load_state_dict(state_dict['fast_opt'])
        self.slow_opt.load_state_dict(state_dict['slow_opt'])
        self.config = state_dict['config']


class BaselineWrapper:
    """
    Minimal wrapper for baseline condition (no dual timescale).

    Provides same interface as DualTimescaleWrapper for consistency.
    """

    def __init__(
        self,
        model: nn.Module,
        base_lr: float = 0.001,
        weight_decay: float = 1.0,
        config: Optional[Dict] = None,
    ):
        self.model = model
        self.base_lr = base_lr
        self.weight_decay = weight_decay
        self.config = config or {}

        # Single optimizer for all parameters
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=base_lr,
            weight_decay=weight_decay,
        )

        self.grad_norm = 0.0

    def training_step(self, loss: torch.Tensor, phase: str = 'both', grad_clip: float = 1.0):
        """Standard training step."""
        self.optimizer.zero_grad()
        loss.backward()

        self.grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), grad_clip
        ).item()

        self.optimizer.step()

        return {
            'fast_grad_norm': self.grad_norm,  # For consistency with dual wrapper
            'slow_grad_norm': 0.0,
        }

    def get_layer_divergence(self) -> float:
        """Baseline has no separation."""
        return 0.0

    def get_learning_rates(self) -> Dict[str, float]:
        """Single learning rate."""
        lr = self.optimizer.param_groups[0]['lr']
        return {
            'fast_lr': lr,
            'slow_lr': lr,
            'lr_ratio': 1.0,
        }

    def state_dict(self) -> Dict:
        return {
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
        }

    def load_state_dict(self, state_dict: Dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.config = state_dict['config']


def create_wrapper(model: nn.Module, condition_config: Dict, base_lr: float = 0.001, weight_decay: float = 1.0):
    """
    Factory function to create appropriate wrapper based on condition.

    Args:
        model: GrokkingTransformer instance
        condition_config: Experimental condition configuration
        base_lr: Base learning rate
        weight_decay: Weight decay

    Returns:
        DualTimescaleWrapper or BaselineWrapper
    """
    # Check if this is a dual-timescale condition
    is_dual_timescale = (
        condition_config.get('fast_lr_scale', 1.0) != condition_config.get('slow_lr_scale', 1.0)
    )

    if is_dual_timescale:
        return DualTimescaleWrapper(model, base_lr, weight_decay, condition_config)
    else:
        return BaselineWrapper(model, base_lr, weight_decay, condition_config)


if __name__ == '__main__':
    # Test the wrapper
    print("Testing DualTimescaleWrapper...")

    # Create dummy model structure similar to GrokkingTransformer
    class DummyBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)

        def forward(self, x):
            return self.linear(x)

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([DummyBlock(64) for _ in range(2)])
            self.token_emb = nn.Embedding(100, 64)
            self.pos_emb = nn.Parameter(torch.randn(1, 10, 64))
            self.output = nn.Linear(64, 100)

        def forward(self, x):
            return self.output(self.blocks[1](self.blocks[0](x)))

    model = DummyModel()

    # Test dual-timescale wrapper
    config = {'fast_lr_scale': 1.0, 'slow_lr_scale': 0.1}
    wrapper = DualTimescaleWrapper(model, config=config)

    print(f"Fast layers: {len(wrapper.fast_layers)}")
    print(f"Slow layers: {len(wrapper.slow_layers)}")
    print(f"Learning rates: {wrapper.get_learning_rates()}")

    # Test training step
    x = torch.randint(0, 100, (16, 10))
    y = model(x)
    loss = y.mean()

    grad_info = wrapper.training_step(loss)
    print(f"Gradient norms: {grad_info}")

    # Test baseline wrapper
    print("\nTesting BaselineWrapper...")
    baseline_wrapper = BaselineWrapper(model)
    print(f"Learning rates: {baseline_wrapper.get_learning_rates()}")

    # Test factory
    print("\nTesting factory function...")
    dual_wrapper = create_wrapper(model, {'fast_lr_scale': 1.0, 'slow_lr_scale': 0.1})
    print(f"Created: {type(dual_wrapper).__name__}")

    baseline_wrapper2 = create_wrapper(model, {'fast_lr_scale': 1.0, 'slow_lr_scale': 1.0})
    print(f"Created: {type(baseline_wrapper2).__name__}")

    print("\n✓ Dual-timescale wrapper tests passed")

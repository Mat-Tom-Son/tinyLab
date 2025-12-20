#!/usr/bin/env python3
"""
Homeostasis-Aware Loss Function for Phase 2 Experiments

Combines task learning with homeostatic crystallization based on
empirical findings from Phase 1 (seed 0):
- VDI equilibrium: 0.61
- Crystallization signature: VDI std collapse
- Compensation present throughout training
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional


class HomeostasisAwareLoss(nn.Module):
    """
    Loss that combines task learning with homeostatic crystallization.

    Based on Phase 1 empirical findings:
    - Target VDI: 0.61 (observed equilibrium)
    - Convergence: VDI std → 0 (head synchronization)
    - Compensation: Layer 1 regulates Layer 0 perturbations
    """

    def __init__(self, vocab_size: int, config: Dict):
        super().__init__()
        self.vocab_size = vocab_size
        self.config = config
        self.task_loss_fn = nn.CrossEntropyLoss()

        # From seed 0 empirical data
        self.target_vdi = config.get('target_vdi', 0.61)
        self.target_vdi_std = config.get('target_vdi_std', 0.0001)

        # Loss weights (tunable per experimental condition)
        self.lambda_compensation = config.get('lambda_compensation', 0.0)
        self.lambda_convergence = config.get('lambda_convergence', 0.0)
        self.lambda_setpoint = config.get('lambda_setpoint', 0.0)

        # Tracking for analysis
        self.loss_history = []

    def compute_vdi_from_attention(self, attn_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute VDI (Variance Dampening Index) from attention weights.

        VDI = H / H_max where H is entropy of attention distribution.
        High VDI (→1.0) = flat attention (suppressor)
        Low VDI (→0.0) = peaked attention (amplifier)

        Args:
            attn_weights: [batch, n_heads, seq_len, seq_len]

        Returns:
            dict with vdi_per_head [n_heads] and vdi_mean (scalar)
        """
        # Average across batch
        attn_avg = attn_weights.mean(dim=0)  # [n_heads, seq_len, seq_len]

        vdis = []
        for h in range(attn_avg.shape[0]):
            # Entropy of attention distribution for this head
            attn_h = attn_avg[h]  # [seq_len, seq_len]

            # Compute entropy: -sum(p * log(p))
            # Add epsilon to prevent log(0)
            log_attn = torch.log(attn_h + 1e-10)
            entropy = -(attn_h * log_attn).sum(dim=-1).mean()

            # Normalize by max entropy (uniform distribution)
            max_entropy = np.log(attn_h.shape[-1])
            vdi = entropy / max_entropy

            vdis.append(vdi)

        vdi_per_head = torch.stack(vdis)  # [n_heads]
        vdi_mean = vdi_per_head.mean()

        return {
            'vdi_per_head': vdi_per_head,
            'vdi_mean': vdi_mean,
            'vdi_std': vdi_per_head.std(),
            'vdi_min': vdi_per_head.min(),
            'vdi_max': vdi_per_head.max(),
        }

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        intermediates: Optional[Dict[str, torch.Tensor]] = None,
    ) -> tuple:
        """
        Compute total loss with homeostatic components.

        Args:
            logits: [batch, vocab_size] or [batch, seq_len, vocab_size]
            targets: [batch] or [batch, seq_len]
            attention_weights: [batch, n_heads, seq_len, seq_len] (optional)
            intermediates: dict with layer outputs (optional)
                - 'embedding': [batch, seq_len, dim]
                - 'layer_0_output': [batch, seq_len, dim]
                - 'layer_1_output': [batch, seq_len, dim]

        Returns:
            (total_loss, loss_dict)
        """
        # 1. Task loss (standard cross-entropy)
        if logits.dim() == 3:
            # Sequence prediction: flatten
            task_loss = self.task_loss_fn(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1)
            )
        else:
            # Classification: direct
            task_loss = self.task_loss_fn(logits, targets)

        # Initialize homeostatic losses
        convergence_loss = torch.tensor(0.0, device=logits.device)
        setpoint_loss = torch.tensor(0.0, device=logits.device)
        compensation_loss = torch.tensor(0.0, device=logits.device)

        vdi_metrics = {}

        # ALWAYS compute VDI metrics for tracking (decouple from loss computation)
        if attention_weights is not None:
            vdi_metrics = self.compute_vdi_from_attention(attention_weights)

        # 2. Convergence loss: penalize if heads aren't synchronized
        # Phase 1 finding: crystallization = VDI std → 0
        if vdi_metrics and self.lambda_convergence > 0:
            convergence_loss = vdi_metrics['vdi_std']

        # 3. Set-point loss: guide toward equilibrium VDI
        # Phase 1 found equilibrium at 0.61
        if vdi_metrics and self.lambda_setpoint > 0:
            setpoint_loss = (vdi_metrics['vdi_mean'] - self.target_vdi).pow(2)

        # 4. Compensation loss: Layer 1 should regulate Layer 0
        # Phase 1 found compensation strength ~0.17 throughout training
        if intermediates is not None and self.lambda_compensation > 0:
            layer_0 = intermediates.get('layer_0_output')
            layer_1 = intermediates.get('layer_1_output')
            embedding = intermediates.get('embedding')

            if layer_0 is not None and layer_1 is not None and embedding is not None:
                # Perturbation: how much Layer 0 changes from embedding
                perturbation = (layer_0 - embedding).pow(2).mean()

                # Regulation: how much Layer 1 compensates
                regulation = (layer_1 - layer_0).pow(2).mean()

                # Penalty: regulation should be >= perturbation
                # (Layer 1 should respond to Layer 0's changes)
                compensation_loss = torch.relu(perturbation - regulation + 0.01)

        # Total loss
        total_loss = (
            task_loss +
            self.lambda_convergence * convergence_loss +
            self.lambda_setpoint * setpoint_loss +
            self.lambda_compensation * compensation_loss
        )

        # Build detailed loss dict for logging
        loss_dict = {
            'task_loss': task_loss.item(),
            'convergence_loss': convergence_loss.item(),
            'setpoint_loss': setpoint_loss.item(),
            'compensation_loss': compensation_loss.item(),
            'total_loss': total_loss.item(),
        }

        # Add VDI metrics if computed
        if vdi_metrics:
            loss_dict.update({
                'vdi_mean': vdi_metrics['vdi_mean'].item(),
                'vdi_std': vdi_metrics['vdi_std'].item(),
                'vdi_min': vdi_metrics['vdi_min'].item(),
                'vdi_max': vdi_metrics['vdi_max'].item(),
            })

        # Track for analysis
        self.loss_history.append(loss_dict.copy())

        return total_loss, loss_dict

    def get_convergence_metrics(self) -> Dict:
        """
        Extract convergence-related metrics from loss history.

        Returns:
            dict with crystallization timing estimates
        """
        if not self.loss_history:
            return {}

        # Find when VDI std first drops below threshold
        crystallization_threshold = 0.001
        crystallization_step = None

        for i, loss_dict in enumerate(self.loss_history):
            if 'vdi_std' in loss_dict and loss_dict['vdi_std'] < crystallization_threshold:
                crystallization_step = i
                break

        # Find when VDI mean reaches target
        setpoint_reached = None
        setpoint_tolerance = 0.01

        for i, loss_dict in enumerate(self.loss_history):
            if 'vdi_mean' in loss_dict:
                if abs(loss_dict['vdi_mean'] - self.target_vdi) < setpoint_tolerance:
                    setpoint_reached = i
                    break

        return {
            'crystallization_step': crystallization_step,
            'setpoint_reached_step': setpoint_reached,
            'vdi_trajectory': [ld.get('vdi_mean', None) for ld in self.loss_history],
            'vdi_std_trajectory': [ld.get('vdi_std', None) for ld in self.loss_history],
        }


# Experimental condition configs
EXPERIMENTAL_CONDITIONS = {
    'baseline': {
        'name': 'Baseline (No Homeostatic Pressure)',
        'lambda_compensation': 0.0,
        'lambda_convergence': 0.0,
        'lambda_setpoint': 0.0,
        'target_vdi': 0.61,
        'fast_lr_scale': 1.0,
        'slow_lr_scale': 1.0,
        'description': 'Control: measure natural crystallization',
    },
    'dual_timescale': {
        'name': 'Dual-Timescale (Fast=1.0, Slow=0.1)',
        'lambda_compensation': 0.5,
        'lambda_convergence': 0.0,
        'lambda_setpoint': 0.0,
        'target_vdi': 0.61,
        'fast_lr_scale': 1.0,
        'slow_lr_scale': 0.1,
        'description': 'Separate timescales + compensation reward',
    },
    'explicit_convergence': {
        'name': 'Dual-Timescale + Explicit Head Synchronization',
        'lambda_compensation': 0.5,
        'lambda_convergence': 0.3,
        'lambda_setpoint': 0.0,
        'target_vdi': 0.61,
        'fast_lr_scale': 1.0,
        'slow_lr_scale': 0.1,
        'description': 'Target VDI std collapse explicitly',
    },
    'intentional_vdi_target': {
        'name': 'Dual-Timescale + VDI Set Point (0.61)',
        'lambda_compensation': 0.5,
        'lambda_convergence': 0.3,
        'lambda_setpoint': 0.2,
        'target_vdi': 0.61,
        'fast_lr_scale': 1.0,
        'slow_lr_scale': 0.1,
        'description': 'Engineer homeostasis to VDI = 0.61',
    },
    'early_convergence': {
        'name': 'Accelerated Crystallization (Aggressive)',
        'lambda_compensation': 1.0,
        'lambda_convergence': 0.5,
        'lambda_setpoint': 0.3,
        'target_vdi': 0.61,
        'fast_lr_scale': 1.0,
        'slow_lr_scale': 0.05,
        'description': 'Test limits: compress crystallization window',
    },
    # VDI Target Sweep: Test if equilibrium tracks target
    'vdi_sweep_0.45': {
        'name': 'VDI Sweep (target=0.45)',
        'lambda_compensation': 0.5,
        'lambda_convergence': 0.3,
        'lambda_setpoint': 0.2,
        'target_vdi': 0.45,
        'fast_lr_scale': 1.0,
        'slow_lr_scale': 0.1,
        'description': 'Test if equilibrium tracks target_vdi=0.45',
    },
    'vdi_sweep_0.50': {
        'name': 'VDI Sweep (target=0.50)',
        'lambda_compensation': 0.5,
        'lambda_convergence': 0.3,
        'lambda_setpoint': 0.2,
        'target_vdi': 0.50,
        'fast_lr_scale': 1.0,
        'slow_lr_scale': 0.1,
        'description': 'Test if equilibrium tracks target_vdi=0.50',
    },
    'vdi_sweep_0.55': {
        'name': 'VDI Sweep (target=0.55)',
        'lambda_compensation': 0.5,
        'lambda_convergence': 0.3,
        'lambda_setpoint': 0.2,
        'target_vdi': 0.55,
        'fast_lr_scale': 1.0,
        'slow_lr_scale': 0.1,
        'description': 'Test if equilibrium tracks target_vdi=0.55',
    },
    'vdi_sweep_0.60': {
        'name': 'VDI Sweep (target=0.60)',
        'lambda_compensation': 0.5,
        'lambda_convergence': 0.3,
        'lambda_setpoint': 0.2,
        'target_vdi': 0.60,
        'fast_lr_scale': 1.0,
        'slow_lr_scale': 0.1,
        'description': 'Test if equilibrium tracks target_vdi=0.60',
    },
    'vdi_sweep_0.65': {
        'name': 'VDI Sweep (target=0.65)',
        'lambda_compensation': 0.5,
        'lambda_convergence': 0.3,
        'lambda_setpoint': 0.2,
        'target_vdi': 0.65,
        'fast_lr_scale': 1.0,
        'slow_lr_scale': 0.1,
        'description': 'Test if equilibrium tracks target_vdi=0.65',
    },
}


if __name__ == '__main__':
    # Test the loss function
    print("Testing HomeostasisAwareLoss...")

    # Dummy inputs
    batch_size = 16
    seq_len = 6
    vocab_size = 97
    n_heads = 2

    logits = torch.randn(batch_size, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size,))
    attention = torch.softmax(torch.randn(batch_size, n_heads, seq_len, seq_len), dim=-1)

    # Test each condition
    for condition_name, config in EXPERIMENTAL_CONDITIONS.items():
        print(f"\nTesting condition: {condition_name}")
        loss_fn = HomeostasisAwareLoss(vocab_size, config)

        total_loss, loss_dict = loss_fn(logits, targets, attention_weights=attention)

        print(f"  Task loss: {loss_dict['task_loss']:.4f}")
        print(f"  Convergence loss: {loss_dict['convergence_loss']:.4f}")
        print(f"  Setpoint loss: {loss_dict['setpoint_loss']:.4f}")
        print(f"  Compensation loss: {loss_dict['compensation_loss']:.4f}")
        print(f"  Total loss: {loss_dict['total_loss']:.4f}")
        if 'vdi_mean' in loss_dict:
            print(f"  VDI mean: {loss_dict['vdi_mean']:.4f}")
            print(f"  VDI std: {loss_dict['vdi_std']:.4f}")

    print("\n✓ HomeostasisAwareLoss tests passed")

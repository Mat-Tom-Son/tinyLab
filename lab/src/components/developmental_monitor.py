"""Developmental Monitoring Framework for Phase-Transition Control.

This module implements the three-part monitoring system described in the
Internal Memo on Architecture of Intentionality:

A. VDI "Snap" Metric - Detects crystallization of Layer-0 bottleneck
B. Homeostatic Kill Testing - Measures Le Chatelier compensation response
C. Saturation Boundary Guardrails - Monitors MI vs channel capacity

The system is designed to catch the moment when the "Slow Loop" crystallizes,
enabling precise control over developmental timing.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# A. VDI "Snap" Metric
# ============================================================================


@dataclass
class VDISnapshot:
    """Snapshot of VDI measurements at a single checkpoint."""

    step: int
    vdi_per_head: List[float]
    entropy_per_head: List[float]
    variance_per_head: List[float]
    mean_vdi: float
    vdi_std: float
    # Derivative approximations for snap detection
    vdi_velocity: Optional[float] = None  # d(VDI)/dt
    vdi_acceleration: Optional[float] = None  # d²(VDI)/dt²


@dataclass
class VDISnapDetection:
    """Result of snap detection analysis."""

    snap_detected: bool
    snap_step: Optional[int] = None
    snap_confidence: float = 0.0  # 0-1 confidence score
    pre_snap_phase: Optional[Tuple[int, int]] = None  # (start, end) steps
    snap_window: Optional[Tuple[int, int]] = None
    post_snap_phase: Optional[Tuple[int, int]] = None
    # Diagnostic information
    vdi_trajectory: List[VDISnapshot] = field(default_factory=list)
    inflection_candidates: List[int] = field(default_factory=list)


def compute_attention_vdi(
    attn_weights: torch.Tensor, eps: float = 1e-10
) -> Dict[str, np.ndarray]:
    """Compute VDI (Variance Dampening Index) from attention weights.

    VDI measures whether a head acts as a suppressor (flattens distribution)
    or amplifier (sharpens distribution).

    Args:
        attn_weights: [batch, heads, seq_query, seq_key] attention weights
        eps: Numerical stability epsilon

    Returns:
        Dictionary containing:
        - vdi: [heads] VDI per head (1.0 = flat/suppressive, 0.0 = peaked)
        - entropy: [heads] Attention entropy per head
        - variance: [heads] Attention variance per head
    """
    # Compute entropy: H = -sum(p * log(p))
    log_attn = torch.log(attn_weights + eps)
    entropy = -(attn_weights * log_attn).sum(dim=-1)  # [batch, heads, seq_query]
    mean_entropy = entropy.mean(dim=[0, 2]).cpu().numpy()  # [heads]

    # Compute variance
    variance = attn_weights.var(dim=-1)  # [batch, heads, seq_query]
    mean_variance = variance.mean(dim=[0, 2]).cpu().numpy()  # [heads]

    # VDI = normalized entropy (1.0 = uniform, 0.0 = delta)
    seq_len = attn_weights.shape[-1]
    max_entropy = np.log(seq_len)
    vdi = mean_entropy / max_entropy

    return {"vdi": vdi, "entropy": mean_entropy, "variance": mean_variance}


def detect_vdi_snap(
    vdi_history: List[VDISnapshot],
    min_samples: int = 5,
    acceleration_threshold: float = -0.001,
    stability_threshold: float = 0.01,
) -> VDISnapDetection:
    """Detect the VDI "snap" - the phase transition to Layer-0 bottleneck.

    The snap is characterized by:
    1. Sharp inflection in mean VDI (negative acceleration)
    2. Transition from high variance (noise amplifier) to low variance (structured bottleneck)
    3. Stabilization of VDI trajectory after the snap

    Args:
        vdi_history: Chronological list of VDI snapshots
        min_samples: Minimum samples needed for detection
        acceleration_threshold: Threshold for d²(VDI)/dt² to consider inflection
        stability_threshold: Max VDI change to consider "stabilized"

    Returns:
        VDISnapDetection result with diagnostic info
    """
    if len(vdi_history) < min_samples:
        return VDISnapDetection(
            snap_detected=False,
            vdi_trajectory=vdi_history,
        )

    # Extract time series
    steps = np.array([s.step for s in vdi_history])
    mean_vdis = np.array([s.mean_vdi for s in vdi_history])
    vdi_stds = np.array([s.vdi_std for s in vdi_history])

    # Compute first and second derivatives (velocity, acceleration)
    dt = np.diff(steps)
    velocity = np.diff(mean_vdis) / dt
    acceleration = np.diff(velocity) / dt[:-1]

    # Update snapshots with derivatives
    for i, snapshot in enumerate(vdi_history[1:], start=1):
        snapshot.vdi_velocity = float(velocity[i - 1])
        if i >= 2:
            snapshot.vdi_acceleration = float(acceleration[i - 2])

    # Find inflection candidates (where acceleration crosses threshold)
    inflection_candidates = []
    for i in range(len(acceleration)):
        if acceleration[i] < acceleration_threshold:
            inflection_candidates.append(int(steps[i + 2]))

    if not inflection_candidates:
        return VDISnapDetection(
            snap_detected=False,
            vdi_trajectory=vdi_history,
            inflection_candidates=[],
        )

    # Select the earliest strong inflection as the snap point
    snap_idx = None
    snap_step = None
    max_accel_magnitude = 0.0

    for candidate_step in inflection_candidates:
        idx = np.where(steps == candidate_step)[0][0]
        if idx >= 2 and idx < len(acceleration) + 2:
            accel_idx = idx - 2
            accel_mag = abs(acceleration[accel_idx])
            if accel_mag > max_accel_magnitude:
                max_accel_magnitude = accel_mag
                snap_idx = idx
                snap_step = candidate_step

    if snap_step is None:
        return VDISnapDetection(
            snap_detected=False,
            vdi_trajectory=vdi_history,
            inflection_candidates=inflection_candidates,
        )

    # Check for post-snap stability
    post_snap_vdis = mean_vdis[snap_idx:]
    if len(post_snap_vdis) > 2:
        post_snap_changes = np.abs(np.diff(post_snap_vdis))
        is_stable = np.all(post_snap_changes < stability_threshold)
    else:
        is_stable = False  # Need more samples to confirm stability

    # Compute confidence score based on:
    # 1. Magnitude of acceleration (stronger = more confident)
    # 2. Post-snap stability (stable = more confident)
    # 3. VDI std reduction (lower std after snap = more confident)
    pre_snap_std = np.mean(vdi_stds[:snap_idx]) if snap_idx > 0 else 1.0
    post_snap_std = np.mean(vdi_stds[snap_idx:])
    std_reduction = max(0.0, pre_snap_std - post_snap_std)

    confidence = min(
        1.0,
        (max_accel_magnitude / 0.01)  # Normalize by typical scale
        * (0.5 if is_stable else 0.2)
        * (1.0 + std_reduction),
    )

    # Define phase windows
    pre_snap_phase = (int(steps[0]), snap_step) if snap_idx > 0 else None
    snap_window = (
        snap_step,
        int(steps[min(snap_idx + 2, len(steps) - 1)]),
    )
    post_snap_phase = (
        int(steps[snap_idx]),
        int(steps[-1]),
    ) if snap_idx < len(steps) - 1 else None

    return VDISnapDetection(
        snap_detected=True,
        snap_step=snap_step,
        snap_confidence=confidence,
        pre_snap_phase=pre_snap_phase,
        snap_window=snap_window,
        post_snap_phase=post_snap_phase,
        vdi_trajectory=vdi_history,
        inflection_candidates=inflection_candidates,
    )


# ============================================================================
# B. Homeostatic Kill Testing
# ============================================================================


@dataclass
class CompensationSignature:
    """Measures Le Chatelier response to omega perturbation."""

    omega: float
    perturbed_head: Tuple[int, int]  # (layer, head)
    # VDI measurements
    vdi_perturbed_head: float
    vdi_other_heads: List[float]
    vdi_deltas: List[float]  # Change from baseline (ω=1.0)
    # Compensation metrics
    compensation_count: int  # How many heads show inverse correlation
    compensation_strength: float  # Mean magnitude of compensatory response
    compensating_heads: List[Tuple[int, int]]  # Which heads compensated
    # Phase classification
    phase: str  # "pre_crystallization", "window", "stability"


@dataclass
class KillTestResult:
    """Result of homeostatic kill testing at a checkpoint."""

    step: int
    phase: str
    compensation_signatures: List[CompensationSignature]
    le_chatelier_detected: bool
    compensation_score: float  # 0-1 score of compensation strength


def run_kill_test(
    model: nn.Module,
    data_batch: torch.Tensor,
    target_head: Tuple[int, int],
    omega_values: List[float],
    baseline_vdis: Optional[Dict[Tuple[int, int], float]] = None,
    compensation_threshold: float = 0.01,
) -> List[CompensationSignature]:
    """Run homeostatic kill test by perturbing target_head with omega sweep.

    Args:
        model: Transformer model with omega-scaling support
        data_batch: [batch, seq] input tokens
        target_head: (layer, head) to perturb
        omega_values: List of omega values to test (should include 1.0 baseline)
        baseline_vdis: Pre-computed baseline VDIs at ω=1.0 (optional)
        compensation_threshold: Minimum |ΔVDI| to count as compensation

    Returns:
        List of CompensationSignature for each omega value
    """
    layer_idx, head_idx = target_head
    signatures = []

    # Compute baseline if not provided
    if baseline_vdis is None:
        baseline_vdis = {}
        # Forward pass with ω=1.0 to get baseline
        with torch.no_grad():
            _ = model(data_batch, layer_head_config={(layer_idx, head_idx): 1.0})
            # Extract attention weights from layer 0
            # This requires model to cache attention weights - see note below
            # baseline_vdis = extract_all_head_vdis(model)

    for omega in omega_values:
        # Forward with perturbation
        with torch.no_grad():
            layer_head_config = {target_head: omega}
            _ = model(data_batch, layer_head_config=layer_head_config)

            # Extract VDI for all heads
            # NOTE: This requires modifying the model to cache attention weights
            # For now, we'll use a placeholder that assumes the model stores
            # attention_weights in a cached format
            try:
                vdis = extract_all_head_vdis(model)
            except AttributeError:
                warnings.warn(
                    "Model does not cache attention weights. "
                    "Kill testing requires attention weight access."
                )
                return []

            # Analyze compensation
            perturbed_vdi = vdis.get(target_head, 0.0)
            other_heads = [h for h in vdis.keys() if h != target_head]
            other_vdis = [vdis[h] for h in other_heads]

            # Compute deltas from baseline
            baseline_perturbed = baseline_vdis.get(target_head, perturbed_vdi)
            delta_perturbed = perturbed_vdi - baseline_perturbed

            vdi_deltas = []
            compensating = []
            for h in other_heads:
                baseline_h = baseline_vdis.get(h, vdis[h])
                delta_h = vdis[h] - baseline_h
                vdi_deltas.append(delta_h)

                # Check for inverse correlation (Le Chatelier)
                # If we weaken target (ω < 1), others should become more suppressive (↑VDI)
                # If we strengthen target (ω > 1), others should become less suppressive (↓VDI)
                expected_sign = np.sign(1.0 - omega)  # Opposite of perturbation
                actual_sign = np.sign(delta_h)

                if (
                    expected_sign == actual_sign
                    and abs(delta_h) > compensation_threshold
                ):
                    compensating.append(h)

            compensation_count = len(compensating)
            compensation_strength = (
                np.mean([abs(d) for d in vdi_deltas]) if vdi_deltas else 0.0
            )

            signatures.append(
                CompensationSignature(
                    omega=omega,
                    perturbed_head=target_head,
                    vdi_perturbed_head=perturbed_vdi,
                    vdi_other_heads=other_vdis,
                    vdi_deltas=vdi_deltas,
                    compensation_count=compensation_count,
                    compensation_strength=compensation_strength,
                    compensating_heads=compensating,
                    phase="unknown",  # Will be determined by calling context
                )
            )

    return signatures


def extract_all_head_vdis(model: nn.Module) -> Dict[Tuple[int, int], float]:
    """Extract VDI for all heads from cached attention weights.

    NOTE: This is a placeholder. The actual implementation depends on how
    your model caches attention weights. You may need to:
    1. Add attention weight caching to your TransformerBlock.forward()
    2. Store weights in model.attention_cache or similar
    3. Call compute_attention_vdi() on each layer's weights

    Args:
        model: Transformer model with cached attention weights

    Returns:
        Dictionary mapping (layer, head) -> VDI
    """
    vdis = {}

    if not hasattr(model, "attention_cache"):
        raise AttributeError(
            "Model must have 'attention_cache' attribute. "
            "Add attention weight caching in TransformerBlock.forward()"
        )

    for layer_idx, attn_weights in enumerate(model.attention_cache):
        # attn_weights: [batch, heads, seq, seq]
        result = compute_attention_vdi(attn_weights)
        for head_idx, vdi in enumerate(result["vdi"]):
            vdis[(layer_idx, head_idx)] = float(vdi)

    return vdis


# ============================================================================
# C. Saturation Boundary Guardrails
# ============================================================================


@dataclass
class MutualInfoSnapshot:
    """Mutual information measurement at a checkpoint."""

    step: int
    mi_estimate: float  # bits
    channel_capacity: float  # bits
    saturation_ratio: float  # MI / capacity (0-1)
    saturation_phase: str  # "healthy", "approaching", "saturated"


def estimate_mutual_information(
    activations: torch.Tensor, method: str = "gaussian"
) -> float:
    """Estimate mutual information between input and output of a layer.

    Args:
        activations: [batch, seq, d_model] layer activations
        method: Estimation method ("gaussian", "kraskov", "histogram")

    Returns:
        MI estimate in bits
    """
    # Flatten batch and sequence dimensions
    batch_size, seq_len, d_model = activations.shape
    X = activations.reshape(-1, d_model).cpu().numpy()

    if method == "gaussian":
        # Gaussian approximation: I(X;Y) ≈ 0.5 * log(det(Σ_x) / det(Σ_y|x))
        # For simplicity, estimate via correlation structure
        cov = np.cov(X.T)
        try:
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                return 0.0
            # MI ≈ 0.5 * log_det(cov) normalized by dimensionality
            mi_nats = 0.5 * logdet / d_model
            mi_bits = mi_nats / np.log(2)
            return float(mi_bits)
        except np.linalg.LinAlgError:
            return 0.0

    elif method == "kraskov":
        # Kraskov-Stögbauer-Grassberger estimator
        # This requires sklearn.feature_selection.mutual_info_regression
        # or similar - not implemented here for brevity
        raise NotImplementedError("Kraskov estimator requires external library")

    elif method == "histogram":
        # Discrete histogram approximation
        # Quantize activations and compute discrete MI
        raise NotImplementedError("Histogram MI estimation not yet implemented")

    else:
        raise ValueError(f"Unknown MI estimation method: {method}")


def compute_channel_capacity(d_model: int, noise_variance: float = 0.1) -> float:
    """Compute theoretical channel capacity (Shannon capacity).

    For a Gaussian channel: C = 0.5 * d * log(1 + SNR)

    Args:
        d_model: Dimension of the channel
        noise_variance: Assumed noise variance (signal variance ≈ 1.0)

    Returns:
        Channel capacity in bits
    """
    signal_variance = 1.0  # Assume normalized activations
    snr = signal_variance / noise_variance
    capacity_nats = 0.5 * d_model * np.log(1 + snr)
    capacity_bits = capacity_nats / np.log(2)
    return float(capacity_bits)


def check_saturation_boundary(
    mi_snapshot: MutualInfoSnapshot, saturation_threshold: float = 0.9
) -> Tuple[bool, str]:
    """Check if MI has saturated the channel capacity.

    Args:
        mi_snapshot: Current MI measurement
        saturation_threshold: Ratio threshold for "saturated" (default 90%)

    Returns:
        (is_saturated, diagnosis)
    """
    ratio = mi_snapshot.saturation_ratio

    if ratio >= saturation_threshold:
        diagnosis = (
            f"SATURATED: MI={mi_snapshot.mi_estimate:.2f} bits "
            f"(>{saturation_threshold*100:.0f}% of capacity). "
            "Risk of brittle attractor - system may solve loss without robust generalization."
        )
        return True, diagnosis

    elif ratio >= saturation_threshold - 0.1:
        diagnosis = (
            f"APPROACHING: MI={mi_snapshot.mi_estimate:.2f} bits "
            f"({ratio*100:.1f}% of capacity). "
            "Monitor closely for crystallization timing."
        )
        return False, diagnosis

    else:
        diagnosis = (
            f"HEALTHY: MI={mi_snapshot.mi_estimate:.2f} bits "
            f"({ratio*100:.1f}% of capacity)."
        )
        return False, diagnosis


# ============================================================================
# D. Integrated Developmental Monitor
# ============================================================================


@dataclass
class DevelopmentalCheckpoint:
    """Complete snapshot of developmental state."""

    step: int
    # A. VDI measurements
    vdi_snapshot: VDISnapshot
    # B. Kill test results (if performed)
    kill_test_result: Optional[KillTestResult] = None
    # C. MI measurements
    mi_snapshot: Optional[MutualInfoSnapshot] = None
    # Derived state
    developmental_phase: str = "unknown"  # "pre_snap", "snap_window", "post_snap"
    homeostatic_phase: str = "unknown"  # "pre_crystallization", "window", "stability"


class DevelopmentalMonitor:
    """Integrated monitor for phase-transition control during training.

    Usage:
        monitor = DevelopmentalMonitor(model, target_head=(0, 0))

        for step in training_loop:
            # Regular forward pass
            loss = train_step(...)

            # Periodic monitoring
            if step % monitor_interval == 0:
                checkpoint = monitor.record_checkpoint(step, data_batch, activations)

                # Check for snap
                if not monitor.snap_detected and checkpoint.vdi_snapshot:
                    snap = detect_vdi_snap(monitor.vdi_history)
                    if snap.snap_detected:
                        print(f"VDI SNAP DETECTED at step {snap.snap_step}!")
                        monitor.snap_detected = True

                # Check saturation
                if checkpoint.mi_snapshot:
                    is_saturated, diag = check_saturation_boundary(checkpoint.mi_snapshot)
                    if is_saturated:
                        print(f"WARNING: {diag}")
    """

    def __init__(
        self,
        model: nn.Module,
        target_head: Tuple[int, int] = (0, 0),
        omega_sweep: Optional[List[float]] = None,
        kill_test_frequency: int = 5,  # Every Nth checkpoint
    ):
        """Initialize developmental monitor.

        Args:
            model: Transformer model to monitor
            target_head: (layer, head) to perturb for kill testing
            omega_sweep: Omega values for kill testing (default: [0.5, 1.0, 1.5])
            kill_test_frequency: Run kill test every N checkpoints
        """
        self.model = model
        self.target_head = target_head
        self.omega_sweep = omega_sweep or [0.5, 1.0, 1.5]
        self.kill_test_frequency = kill_test_frequency

        # State tracking
        self.checkpoints: List[DevelopmentalCheckpoint] = []
        self.vdi_history: List[VDISnapshot] = []
        self.snap_detected = False
        self.snap_result: Optional[VDISnapDetection] = None

        # Baseline measurements (computed at first checkpoint with ω=1.0)
        self.baseline_vdis: Optional[Dict[Tuple[int, int], float]] = None

    def record_checkpoint(
        self,
        step: int,
        data_batch: torch.Tensor,
        layer0_activations: Optional[torch.Tensor] = None,
    ) -> DevelopmentalCheckpoint:
        """Record developmental state at a checkpoint.

        Args:
            step: Training step
            data_batch: [batch, seq] input tokens for VDI measurement
            layer0_activations: [batch, seq, d_model] Layer-0 activations for MI

        Returns:
            DevelopmentalCheckpoint with measurements
        """
        # A. Measure VDI
        with torch.no_grad():
            # Forward pass to get attention weights
            _ = self.model(data_batch)

            # Extract VDI for Layer 0
            try:
                vdis = extract_all_head_vdis(self.model)
                layer0_vdis = [
                    vdis[(0, h)] for h in range(len(vdis)) if (0, h) in vdis
                ]
                layer0_entropies = [0.0] * len(layer0_vdis)  # Placeholder
                layer0_variances = [0.0] * len(layer0_vdis)  # Placeholder

                vdi_snapshot = VDISnapshot(
                    step=step,
                    vdi_per_head=layer0_vdis,
                    entropy_per_head=layer0_entropies,
                    variance_per_head=layer0_variances,
                    mean_vdi=float(np.mean(layer0_vdis)),
                    vdi_std=float(np.std(layer0_vdis)),
                )
                self.vdi_history.append(vdi_snapshot)

            except AttributeError:
                vdi_snapshot = VDISnapshot(
                    step=step,
                    vdi_per_head=[],
                    entropy_per_head=[],
                    variance_per_head=[],
                    mean_vdi=0.0,
                    vdi_std=0.0,
                )

        # B. Kill test (periodic)
        kill_test_result = None
        checkpoint_idx = len(self.checkpoints)
        if checkpoint_idx % self.kill_test_frequency == 0:
            try:
                signatures = run_kill_test(
                    self.model,
                    data_batch,
                    self.target_head,
                    self.omega_sweep,
                    self.baseline_vdis,
                )

                # Detect Le Chatelier compensation
                le_chatelier_detected = False
                compensation_scores = []
                for sig in signatures:
                    if sig.omega != 1.0:  # Skip baseline
                        # Count as compensation if at least 2 heads compensate
                        if sig.compensation_count >= 2:
                            le_chatelier_detected = True
                        compensation_scores.append(sig.compensation_strength)

                avg_compensation = (
                    np.mean(compensation_scores) if compensation_scores else 0.0
                )

                kill_test_result = KillTestResult(
                    step=step,
                    phase="unknown",  # Will be updated after snap detection
                    compensation_signatures=signatures,
                    le_chatelier_detected=le_chatelier_detected,
                    compensation_score=float(avg_compensation),
                )

                # Store baseline on first kill test
                if self.baseline_vdis is None and signatures:
                    baseline_sig = next(
                        (s for s in signatures if s.omega == 1.0), None
                    )
                    if baseline_sig:
                        self.baseline_vdis = {
                            self.target_head: baseline_sig.vdi_perturbed_head
                        }
                        # Add other heads (assuming same layer)
                        layer_idx = self.target_head[0]
                        for h_idx, vdi in enumerate(baseline_sig.vdi_other_heads):
                            self.baseline_vdis[(layer_idx, h_idx + 1)] = vdi

            except Exception as e:
                warnings.warn(f"Kill test failed: {e}")

        # C. Measure MI
        mi_snapshot = None
        if layer0_activations is not None:
            try:
                d_model = layer0_activations.shape[-1]
                mi_estimate = estimate_mutual_information(layer0_activations)
                capacity = compute_channel_capacity(d_model)
                saturation_ratio = mi_estimate / capacity if capacity > 0 else 0.0

                if saturation_ratio >= 0.9:
                    phase = "saturated"
                elif saturation_ratio >= 0.8:
                    phase = "approaching"
                else:
                    phase = "healthy"

                mi_snapshot = MutualInfoSnapshot(
                    step=step,
                    mi_estimate=mi_estimate,
                    channel_capacity=capacity,
                    saturation_ratio=saturation_ratio,
                    saturation_phase=phase,
                )
            except Exception as e:
                warnings.warn(f"MI estimation failed: {e}")

        # Determine developmental phase based on snap detection
        developmental_phase = "pre_snap"  # Default
        if self.snap_detected and self.snap_result:
            if (
                self.snap_result.snap_window
                and self.snap_result.snap_window[0]
                <= step
                <= self.snap_result.snap_window[1]
            ):
                developmental_phase = "snap_window"
            elif (
                self.snap_result.post_snap_phase
                and step >= self.snap_result.post_snap_phase[0]
            ):
                developmental_phase = "post_snap"

        checkpoint = DevelopmentalCheckpoint(
            step=step,
            vdi_snapshot=vdi_snapshot,
            kill_test_result=kill_test_result,
            mi_snapshot=mi_snapshot,
            developmental_phase=developmental_phase,
        )

        self.checkpoints.append(checkpoint)
        return checkpoint

    def analyze_trajectory(self) -> Dict[str, Any]:
        """Analyze complete developmental trajectory.

        Returns:
            Summary dictionary with key findings
        """
        # Run snap detection if not done yet
        if not self.snap_detected and len(self.vdi_history) >= 5:
            self.snap_result = detect_vdi_snap(self.vdi_history)
            self.snap_detected = self.snap_result.snap_detected

        # Summarize kill test results by phase
        kill_tests_by_phase = {"pre_crystallization": [], "window": [], "stability": []}

        for ckpt in self.checkpoints:
            if ckpt.kill_test_result:
                phase = ckpt.developmental_phase
                if phase == "pre_snap":
                    kill_tests_by_phase["pre_crystallization"].append(
                        ckpt.kill_test_result
                    )
                elif phase == "snap_window":
                    kill_tests_by_phase["window"].append(ckpt.kill_test_result)
                elif phase == "post_snap":
                    kill_tests_by_phase["stability"].append(ckpt.kill_test_result)

        # Compute average compensation by phase
        compensation_by_phase = {}
        for phase_name, tests in kill_tests_by_phase.items():
            if tests:
                avg_comp = np.mean([t.compensation_score for t in tests])
                compensation_by_phase[phase_name] = float(avg_comp)
            else:
                compensation_by_phase[phase_name] = 0.0

        # Check for saturation issues
        saturated_steps = [
            ckpt.step
            for ckpt in self.checkpoints
            if ckpt.mi_snapshot and ckpt.mi_snapshot.saturation_phase == "saturated"
        ]

        summary = {
            "snap_detected": self.snap_detected,
            "snap_step": self.snap_result.snap_step if self.snap_result else None,
            "snap_confidence": (
                self.snap_result.snap_confidence if self.snap_result else 0.0
            ),
            "compensation_by_phase": compensation_by_phase,
            "le_chatelier_confirmed": (
                compensation_by_phase.get("window", 0.0) > 0.01
                or compensation_by_phase.get("stability", 0.0) > 0.01
            ),
            "saturation_warning": len(saturated_steps) > 0,
            "saturated_steps": saturated_steps,
            "total_checkpoints": len(self.checkpoints),
        }

        return summary

    def save_trajectory(self, output_path: Path):
        """Save complete developmental trajectory to JSON."""
        # Convert to serializable format
        data = {
            "target_head": self.target_head,
            "omega_sweep": self.omega_sweep,
            "snap_result": {
                "detected": self.snap_detected,
                "step": self.snap_result.snap_step if self.snap_result else None,
                "confidence": (
                    self.snap_result.snap_confidence if self.snap_result else 0.0
                ),
            },
            "vdi_history": [
                {
                    "step": s.step,
                    "mean_vdi": s.mean_vdi,
                    "vdi_std": s.vdi_std,
                    "vdi_velocity": s.vdi_velocity,
                    "vdi_acceleration": s.vdi_acceleration,
                }
                for s in self.vdi_history
            ],
            "checkpoints": [
                {
                    "step": c.step,
                    "developmental_phase": c.developmental_phase,
                    "vdi_mean": c.vdi_snapshot.mean_vdi,
                    "kill_test": {
                        "performed": c.kill_test_result is not None,
                        "le_chatelier_detected": (
                            c.kill_test_result.le_chatelier_detected
                            if c.kill_test_result
                            else False
                        ),
                        "compensation_score": (
                            c.kill_test_result.compensation_score
                            if c.kill_test_result
                            else 0.0
                        ),
                    }
                    if c.kill_test_result
                    else None,
                    "mi": {
                        "estimate": c.mi_snapshot.mi_estimate,
                        "saturation_ratio": c.mi_snapshot.saturation_ratio,
                        "phase": c.mi_snapshot.saturation_phase,
                    }
                    if c.mi_snapshot
                    else None,
                }
                for c in self.checkpoints
            ],
            "summary": self.analyze_trajectory(),
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved developmental trajectory to {output_path}")

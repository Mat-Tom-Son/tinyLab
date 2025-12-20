#!/usr/bin/env python3
"""Visualize developmental trajectory from monitoring data.

Creates diagnostic plots showing:
1. VDI trajectory with snap detection
2. Compensation strength across phases
3. MI saturation curve
4. Phase diagram overlaying all metrics

Usage:
    python scripts/visualize_developmental_trajectory.py \
        reports/developmental_monitoring/parity_omega1.0_seed0/developmental_trajectory.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_trajectory(path: Path) -> Dict:
    """Load developmental trajectory JSON."""
    with open(path) as f:
        return json.load(f)


def plot_vdi_trajectory(data: Dict, ax: plt.Axes):
    """Plot VDI trajectory with snap detection."""
    vdi_history = data["vdi_history"]
    snap_result = data["snap_result"]

    steps = [h["step"] for h in vdi_history]
    mean_vdis = [h["mean_vdi"] for h in vdi_history]
    vdi_stds = [h["vdi_std"] for h in vdi_history]
    velocities = [h.get("vdi_velocity", None) for h in vdi_history]
    accelerations = [h.get("vdi_acceleration", None) for h in vdi_history]

    # Main VDI trajectory
    ax.plot(steps, mean_vdis, "o-", label="Mean VDI", color="blue", markersize=4)
    ax.fill_between(
        steps,
        [m - s for m, s in zip(mean_vdis, vdi_stds)],
        [m + s for m, s in zip(mean_vdis, vdi_stds)],
        alpha=0.2,
        color="blue",
    )

    # Mark snap point
    if snap_result["detected"]:
        snap_step = snap_result["step"]
        snap_idx = steps.index(snap_step) if snap_step in steps else None
        if snap_idx is not None:
            ax.axvline(
                snap_step, color="red", linestyle="--", linewidth=2, label="VDI Snap"
            )
            ax.scatter(
                [snap_step],
                [mean_vdis[snap_idx]],
                color="red",
                s=100,
                zorder=5,
                marker="*",
            )
            ax.text(
                snap_step,
                mean_vdis[snap_idx] + 0.05,
                f"Snap\n(conf: {snap_result['confidence']:.2f})",
                ha="center",
                fontsize=8,
                color="red",
            )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("VDI (Variance Dampening Index)")
    ax.set_title("A. VDI Trajectory & Crystallization Snap")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_derivatives(data: Dict, ax: plt.Axes):
    """Plot VDI velocity and acceleration."""
    vdi_history = data["vdi_history"]
    snap_result = data["snap_result"]

    steps = [h["step"] for h in vdi_history]
    velocities = [h.get("vdi_velocity") for h in vdi_history]
    accelerations = [h.get("vdi_acceleration") for h in vdi_history]

    # Filter out None values
    vel_steps = [s for s, v in zip(steps, velocities) if v is not None]
    vel_vals = [v for v in velocities if v is not None]
    accel_steps = [s for s, a in zip(steps, accelerations) if a is not None]
    accel_vals = [a for a in accelerations if a is not None]

    if vel_vals:
        ax.plot(
            vel_steps, vel_vals, "o-", label="Velocity (dVDI/dt)", color="green", alpha=0.7
        )

    if accel_vals:
        ax2 = ax.twinx()
        ax2.plot(
            accel_steps,
            accel_vals,
            "s-",
            label="Acceleration (d²VDI/dt²)",
            color="orange",
            alpha=0.7,
        )
        ax2.set_ylabel("Acceleration", color="orange")
        ax2.tick_params(axis="y", labelcolor="orange")

        # Mark negative acceleration threshold
        ax2.axhline(
            -0.001, color="red", linestyle=":", alpha=0.5, label="Snap Threshold"
        )

    if snap_result["detected"]:
        ax.axvline(snap_result["step"], color="red", linestyle="--", alpha=0.5)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Velocity", color="green")
    ax.set_title("VDI Derivatives (Snap Detection)")
    ax.tick_params(axis="y", labelcolor="green")
    ax.legend(loc="upper left")
    if accel_vals:
        ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.3)


def plot_compensation(data: Dict, ax: plt.Axes):
    """Plot compensation strength across phases."""
    checkpoints = data["checkpoints"]
    summary = data["summary"]

    # Extract kill test results
    steps = []
    comp_scores = []
    le_chatelier_flags = []
    phases = []

    for ckpt in checkpoints:
        if ckpt["kill_test"] and ckpt["kill_test"]["performed"]:
            steps.append(ckpt["step"])
            comp_scores.append(ckpt["kill_test"]["compensation_score"])
            le_chatelier_flags.append(
                ckpt["kill_test"]["le_chatelier_detected"]
            )
            phases.append(ckpt["developmental_phase"])

    if not steps:
        ax.text(
            0.5,
            0.5,
            "No kill test data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("B. Homeostatic Compensation")
        return

    # Color by phase
    phase_colors = {
        "pre_snap": "gray",
        "snap_window": "orange",
        "post_snap": "green",
    }
    colors = [phase_colors.get(p, "blue") for p in phases]

    # Plot compensation scores
    for step, score, lc, color in zip(steps, comp_scores, le_chatelier_flags, colors):
        marker = "o" if lc else "x"
        ax.scatter([step], [score], color=color, marker=marker, s=100, alpha=0.7)

    # Add phase labels
    for phase_name, color in phase_colors.items():
        ax.scatter([], [], color=color, label=phase_name, s=50)

    # Mark Le Chatelier detection
    ax.scatter(
        [],
        [],
        color="black",
        marker="o",
        s=50,
        label="Le Chatelier detected",
    )
    ax.scatter([], [], color="black", marker="x", s=50, label="No compensation")

    # Add horizontal line at threshold
    ax.axhline(0.01, color="red", linestyle=":", alpha=0.5, label="Threshold")

    # Annotate summary
    comp_by_phase = summary["compensation_by_phase"]
    text = "Avg Compensation:\n"
    for phase, score in comp_by_phase.items():
        text += f"  {phase}: {score:.4f}\n"
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Compensation Score")
    ax.set_title("B. Homeostatic Compensation (Le Chatelier Response)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_mi_saturation(data: Dict, ax: plt.Axes):
    """Plot mutual information saturation curve."""
    checkpoints = data["checkpoints"]

    steps = []
    mi_estimates = []
    saturation_ratios = []
    phases = []

    for ckpt in checkpoints:
        if ckpt["mi"]:
            steps.append(ckpt["step"])
            mi_estimates.append(ckpt["mi"]["estimate"])
            saturation_ratios.append(ckpt["mi"]["saturation_ratio"])
            phases.append(ckpt["mi"]["phase"])

    if not steps:
        ax.text(
            0.5,
            0.5,
            "No MI data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("C. MI Saturation Boundary")
        return

    # Main MI trajectory
    ax.plot(steps, mi_estimates, "o-", label="MI Estimate", color="purple")

    # Mark capacity
    if mi_estimates:
        capacity = mi_estimates[0] / (
            saturation_ratios[0] + 1e-6
        )  # Approximate capacity
        ax.axhline(
            capacity, color="red", linestyle="--", alpha=0.5, label="Channel Capacity"
        )

    # Color by saturation phase
    phase_colors = {"healthy": "green", "approaching": "orange", "saturated": "red"}
    for step, ratio, phase in zip(steps, saturation_ratios, phases):
        color = phase_colors.get(phase, "gray")
        ax.scatter([step], [ratio * capacity], color=color, s=50, alpha=0.5)

    # Add saturation threshold
    ax.axhline(
        0.9 * capacity,
        color="red",
        linestyle=":",
        alpha=0.5,
        label="Saturation Threshold (90%)",
    )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mutual Information (bits)")
    ax.set_title("C. MI Saturation Boundary")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_phase_diagram(data: Dict, ax: plt.Axes):
    """Overlay all metrics on a phase diagram."""
    checkpoints = data["checkpoints"]
    snap_result = data["snap_result"]

    steps = [c["step"] for c in checkpoints]
    vdi_means = [c["vdi_mean"] for c in checkpoints]
    phases = [c["developmental_phase"] for c in checkpoints]

    # Normalize metrics to [0, 1] for overlay
    vdi_norm = np.array(vdi_means)

    # Plot normalized VDI
    ax.plot(steps, vdi_norm, "o-", label="VDI", color="blue", alpha=0.7)

    # Add compensation scores (normalized)
    comp_steps = []
    comp_norm = []
    for ckpt in checkpoints:
        if ckpt["kill_test"] and ckpt["kill_test"]["performed"]:
            comp_steps.append(ckpt["step"])
            # Normalize by typical scale (0.1)
            comp_norm.append(min(1.0, ckpt["kill_test"]["compensation_score"] / 0.1))

    if comp_steps:
        ax.plot(
            comp_steps,
            comp_norm,
            "s-",
            label="Compensation (norm)",
            color="green",
            alpha=0.7,
        )

    # Add MI saturation (normalized)
    mi_steps = []
    mi_norm = []
    for ckpt in checkpoints:
        if ckpt["mi"]:
            mi_steps.append(ckpt["step"])
            mi_norm.append(ckpt["mi"]["saturation_ratio"])

    if mi_steps:
        ax.plot(
            mi_steps, mi_norm, "^-", label="MI Saturation", color="purple", alpha=0.7
        )

    # Mark phase boundaries
    phase_colors = {"pre_snap": "lightgray", "snap_window": "yellow", "post_snap": "lightgreen"}
    current_phase = None
    phase_start = steps[0]

    for i, (step, phase) in enumerate(zip(steps, phases)):
        if phase != current_phase:
            if current_phase is not None:
                # Shade previous phase
                ax.axvspan(
                    phase_start,
                    step,
                    alpha=0.2,
                    color=phase_colors.get(current_phase, "white"),
                    label=f"{current_phase}" if i == 1 else None,
                )
            current_phase = phase
            phase_start = step

    # Shade final phase
    if current_phase:
        ax.axvspan(
            phase_start,
            steps[-1],
            alpha=0.2,
            color=phase_colors.get(current_phase, "white"),
        )

    # Mark snap
    if snap_result["detected"]:
        ax.axvline(snap_result["step"], color="red", linestyle="--", linewidth=2)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Normalized Metric Value")
    ax.set_title("Phase Diagram: Integrated Developmental Trajectory")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def create_summary_figure(data: Dict, output_path: Path):
    """Create comprehensive summary figure."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # A. VDI trajectory
    ax1 = fig.add_subplot(gs[0, :])
    plot_vdi_trajectory(data, ax1)

    # B. Compensation
    ax2 = fig.add_subplot(gs[1, 0])
    plot_compensation(data, ax2)

    # C. MI saturation
    ax3 = fig.add_subplot(gs[1, 1])
    plot_mi_saturation(data, ax3)

    # D. Derivatives (for snap detection)
    ax4 = fig.add_subplot(gs[2, 0])
    plot_derivatives(data, ax4)

    # E. Phase diagram
    ax5 = fig.add_subplot(gs[2, 1])
    plot_phase_diagram(data, ax5)

    # Add title with summary
    summary = data["summary"]
    title = "Developmental Trajectory Analysis\n"
    if summary["snap_detected"]:
        title += f"Snap detected at step {summary['snap_step']} (confidence: {summary['snap_confidence']:.2f}) | "
    title += f"Le Chatelier: {'CONFIRMED' if summary['le_chatelier_confirmed'] else 'Not detected'}"
    if summary["saturation_warning"]:
        title += " | ⚠️ SATURATION WARNING"

    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize developmental trajectory")
    parser.add_argument(
        "trajectory_path",
        type=Path,
        help="Path to developmental_trajectory.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for figure (default: same dir as input)",
    )
    args = parser.parse_args()

    # Load data
    print(f"Loading trajectory from {args.trajectory_path}")
    data = load_trajectory(args.trajectory_path)

    # Determine output path
    if args.output is None:
        output_path = args.trajectory_path.parent / "developmental_trajectory.png"
    else:
        output_path = args.output

    # Create figure
    print("Creating visualization...")
    create_summary_figure(data, output_path)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    summary = data["summary"]
    print(f"Snap detected: {summary['snap_detected']}")
    if summary["snap_detected"]:
        print(f"  Step: {summary['snap_step']}")
        print(f"  Confidence: {summary['snap_confidence']:.3f}")

    print(f"\nCompensation by phase:")
    for phase, score in summary["compensation_by_phase"].items():
        print(f"  {phase}: {score:.4f}")

    print(f"\nLe Chatelier confirmed: {summary['le_chatelier_confirmed']}")

    if summary["saturation_warning"]:
        print(f"\n⚠️ Saturation warning at steps: {summary['saturated_steps']}")

    print(f"\nTotal checkpoints: {summary['total_checkpoints']}")
    print("=" * 70)


if __name__ == "__main__":
    main()

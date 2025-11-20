"""Aim-based experiment tracking for tinyLab."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from aim import Run


class TinyLabTracker:
    """Wrapper around Aim for mechanistic interpretability experiments.

    Typical usage in an experiment script::

        tracker = TinyLabTracker(
            experiment_name="h1_cross_condition",
            config=config_dict,
            tags=["gpt2-medium", "facts", "layer0"],
        )

        tracker.log_metric("logit_diff", 2.45, step=0)
        tracker.log_head_metrics((0, 2), {"logit_diff": 2.45, "acc": 0.89})
        tracker.finish()
    """

    def __init__(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        tags: Optional[list[str]] = None,
        repo_path: Optional[str] = None,
    ) -> None:
        """Initialize tracker for an experiment run.

        Args:
            experiment_name: Name of experiment (e.g., "h1_cross_condition").
            config: Full experiment configuration dict.
            tags: Tags for filtering (e.g., ["gpt2-medium", "facts"]).
            repo_path: Optional path to .aim directory (defaults to project root).
        """
        self.experiment_name = experiment_name
        self.config = config

        self.run = Run(repo=repo_path, experiment=experiment_name)

        for tag in tags or []:
            try:
                self.run.add_tag(tag)
            except Exception:
                # Tags are convenience metadata; ignore failures.
                pass

        # Log hyperparameters / config
        self.run["hparams"] = config

        # Common metadata keys used in queries
        model_cfg = config.get("model") or config.get("shared", {}).get("model") or {}
        dataset_cfg = (
            config.get("dataset") or config.get("shared", {}).get("dataset") or {}
        )

        self.run["model_name"] = model_cfg.get("name", "unknown")
        self.run["model_family"] = model_cfg.get("family", "unknown")
        self.run["dataset_id"] = dataset_cfg.get("id", "unknown")
        self.run["condition"] = config.get("tag", "unknown")
        self.run["device"] = config.get(
            "device", config.get("shared", {}).get("device", "unknown")
        )
        self.run["seed"] = config.get("seed")

        # Best-effort git metadata without extra dependencies
        try:
            repo_root = Path(".").resolve()
            head = (repo_root / ".git" / "HEAD").read_text().strip()
            if head.startswith("ref:"):
                ref_path = head.split(" ", 1)[1]
                ref_file = repo_root / ".git" / ref_path
                if ref_file.exists():
                    commit = ref_file.read_text().strip()
                    self.run["git_commit"] = commit[:8]
                self.run["git_branch"] = Path(ref_path).name
        except Exception:
            # Git info is optional; ignore failures.
            pass

    def log_metric(
        self,
        name: str,
        value: float,
        step: int = 0,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a scalar metric."""
        self.run.track(value, name=name, step=step, context=context or {})

    def log_metrics_dict(
        self,
        metrics: Dict[str, float],
        step: int = 0,
        prefix: str = "",
    ) -> None:
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            full_name = f"{prefix}/{name}" if prefix else name
            self.log_metric(full_name, value, step=step)

    def log_head_metrics(
        self,
        head: tuple[int, int],
        metrics: Dict[str, float],
        step: int = 0,
    ) -> None:
        """Log metrics for a specific attention head."""
        context = {"layer": head[0], "head": head[1]}
        for name, value in metrics.items():
            self.run.track(value, name=name, step=step, context=context)

    def log_layer_metrics(
        self,
        layer: int,
        metrics: Dict[str, float],
        step: int = 0,
    ) -> None:
        """Log metrics for a specific layer."""
        context = {"layer": layer}
        for name, value in metrics.items():
            self.run.track(value, name=name, step=step, context=context)

    def log_ov_fidelity(
        self,
        fidelity_by_layer: Dict[int, float],
        step: int = 0,
    ) -> None:
        """Log OV circuit fidelity across layers."""
        for layer, fidelity in fidelity_by_layer.items():
            self.run.track(
                fidelity,
                name="ov_fidelity",
                step=step,
                context={"layer": layer},
            )

    def log_activation_entropy(
        self,
        layer: int,
        entropy: float,
        entropy_type: str = "subspace",
        step: int = 0,
    ) -> None:
        """Log activation entropy for a layer."""
        self.run.track(
            entropy,
            name=f"activation_entropy_{entropy_type}",
            step=step,
            context={"layer": layer},
        )

    def log_geometric_metrics(
        self,
        curvature: float,
        output_entropy: float,
        step: int = 0,
        phase: str = "final",
    ) -> None:
        """Log geometric signature metrics."""
        self.run.track(
            curvature,
            name="curvature",
            step=step,
            context={"phase": phase},
        )
        self.run.track(
            output_entropy,
            name="output_entropy",
            step=step,
            context={"phase": phase},
        )

    def log_image(
        self,
        name: str,
        image: Any,
        step: int = 0,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an image (attention pattern, plot, etc.)."""
        from aim import Image

        self.run.track(Image(image), name=name, step=step, context=context or {})

    def log_attention_pattern(
        self,
        pattern: Any,
        layer: int,
        head: int,
        step: int = 0,
    ) -> None:
        """Log attention pattern heatmap."""
        import matplotlib.pyplot as plt
        from aim import Image

        if hasattr(pattern, "shape"):
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(pattern, cmap="viridis", aspect="auto")
            ax.set_title(f"Attention Pattern L{layer}H{head}")
            ax.set_xlabel("Key Position")
            ax.set_ylabel("Query Position")
            plt.colorbar(im, ax=ax)
            self.run.track(
                Image(fig),
                name="attention_pattern",
                step=step,
                context={"layer": layer, "head": head},
            )
            plt.close(fig)
        else:
            self.run.track(
                Image(pattern),
                name="attention_pattern",
                step=step,
                context={"layer": layer, "head": head},
            )

    def finish(self, final_metrics: Optional[Dict[str, float]] = None) -> None:
        """Close the run, optionally logging final metrics."""
        if final_metrics:
            self.log_metrics_dict(final_metrics, prefix="final")
        self.run.close()

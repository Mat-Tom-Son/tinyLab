"""Activation-space metrics under head ablation.

Computes (per condition):
- Activation entropy at `blocks.0.hook_resid_post` (last position) for
  baseline vs. ablated runs (relative Gaussian entropy: 0.5 * log det Σ).
- KL between baseline and ablated activation distributions (MVN approximation).
- Output entropy at the last position (Shannon entropy of softmax).
- Simple trajectory curvature across tokens in layer-0 residuals.

Optionally samples random layer-0 head sets (same cardinality) as a control and
reports where the specified heads fall in that null distribution.

Example:
    python -m lab.analysis.activation_entropy \
        --config lab/configs/run_h1_cross_condition_physics.json \
        --tag facts \
        --heads 2 4 7 \
        --samples 200 \
        --random-samples 100 \
        --output reports/activation_entropy_facts.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from ..src.components import datasets, load_model
from ..src.utils import io
from tqdm import tqdm


HookName = str


@dataclass
class ChildCfg:
    run_name: str
    device: str
    model: Dict
    dataset: Dict
    seed: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True, help="Cross-condition or single-run config JSON.")
    p.add_argument("--tag", type=str, required=True, help="Condition tag inside the config (for cross-condition configs).")
    p.add_argument("--heads", type=int, nargs="*", default=None, help="Layer-0 heads to ablate (integers).")
    p.add_argument("--layer", type=int, default=0, help="Layer index for residual hook (default: 0).")
    p.add_argument("--samples", type=int, default=256, help="Max number of examples to load from the split.")
    p.add_argument("--random-samples", type=int, default=0, help="Number of random L0 head sets to evaluate as control.")
    p.add_argument(
        "--entropy-methods",
        type=str,
        default="full,diagonal,subspace,per_token",
        help="Comma-separated list of activation entropy methods to compute: full,diagonal,subspace,per_token",
    )
    p.add_argument("--subspace-var", type=float, default=0.95, help="Variance fraction for subspace entropy (default: 0.95).")
    p.add_argument("--subspace-max-k", type=int, default=256, help="Max PCs for subspace entropy (default: 256).")
    p.add_argument("--output", type=Path, required=True, help="Where to write the JSON report.")
    p.add_argument("--device", type=str, default=None, help="Override device (cpu|mps|cuda|auto). Optional.")
    p.add_argument("--model-name", type=str, default=None, help="Override model name (e.g., gpt2-large). Optional.")
    return p.parse_args()


def build_child_cfg(config_path: Path, tag: str) -> ChildCfg:
    cfg = io.load_json(config_path)
    if "shared" in cfg and "conditions" in cfg:
        shared = cfg["shared"]
        dataset_cfg = shared["dataset"].copy()
        for cond in cfg["conditions"]:
            if cond.get("tag") == tag:
                dataset_cfg.update(cond.get("dataset", {}))
                break
        else:
            raise ValueError(f"Tag '{tag}' not found in config {config_path}")
        child = ChildCfg(
            run_name=f"activation_entropy_{tag}",
            device=shared.get("device", "auto"),
            model=shared["model"],
            dataset=dataset_cfg,
            seed=(shared.get("seeds") or [0])[0],
        )
    else:
        # Treat as a single-run config
        child = ChildCfg(
            run_name=f"activation_entropy_{tag}",
            device=cfg.get("device", "auto"),
            model=cfg["model"],
            dataset=cfg["dataset"],
            seed=cfg.get("seed", 0),
        )
    return child


def last_pos_resid(cache, layer: int) -> torch.Tensor:
    node = f"blocks.{layer}.hook_resid_post"
    t = cache[node]  # [B, S, d_model]
    return t[:, -1, :]


def softmax_entropy_last(logits: torch.Tensor) -> float:
    """Shannon entropy H[p] at last position in float32 for stability."""
    # logits: [B, S, V]
    last = logits[:, -1, :].to(torch.float32)
    p = torch.softmax(last, dim=-1)
    # Numerical safety: clamp to float32-representable floor
    p = p.clamp_min(1e-30)
    h = -(p * p.log()).sum(dim=-1).mean().item()
    return float(h)


def gaussian_entropy_rel(x: np.ndarray, eps: float = 1e-3) -> float:
    """Relative differential entropy for MVN: 0.5 * log det(Σ_reg).

    Adds a small ridge eps * trace(Σ)/d to stabilize logdet in high dimensions.
    """
    # x: [N, D]
    if x.ndim != 2 or x.shape[0] < 2:
        return float("nan")
    x_c = x - x.mean(axis=0, keepdims=True)
    # rowvar=False -> each column is a variable
    cov = np.cov(x_c, rowvar=False)
    d = cov.shape[0]
    ridge = eps * (np.trace(cov) / max(d, 1))
    cov_reg = cov + ridge * np.eye(d, dtype=cov.dtype)
    sign, logdet = np.linalg.slogdet(cov_reg)
    if sign <= 0:
        return float("nan")
    return 0.5 * float(logdet)


def gaussian_kl(p_mu: np.ndarray, p_cov: np.ndarray, q_mu: np.ndarray, q_cov: np.ndarray, eps: float = 1e-3) -> float:
    """KL(N_p || N_q) for multivariate Gaussians with ridge stabilization."""
    d = p_cov.shape[0]
    # Ridge for stability
    p_cov = p_cov + (eps * (np.trace(p_cov) / max(d, 1))) * np.eye(d)
    q_cov = q_cov + (eps * (np.trace(q_cov) / max(d, 1))) * np.eye(d)
    q_cov_inv = np.linalg.inv(q_cov)
    term1 = np.trace(q_cov_inv @ p_cov)
    diff = (q_mu - p_mu).reshape(-1, 1)
    term2 = float(diff.T @ q_cov_inv @ diff)
    sign_p, logdet_p = np.linalg.slogdet(p_cov)
    sign_q, logdet_q = np.linalg.slogdet(q_cov)
    if sign_p <= 0 or sign_q <= 0:
        return float("nan")
    term3 = logdet_q - logdet_p
    return 0.5 * float(term1 + term2 - d + term3)


def mvn_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x_c = x - x.mean(axis=0, keepdims=True)
    mu = x.mean(axis=0)
    cov = np.cov(x_c, rowvar=False)
    return mu, cov


def diagonal_entropy(x: np.ndarray, eps: float = 1e-3) -> float:
    """Diagonal-only entropy: 0.5 * sum log(var_i + ridge).

    Ridge based on mean variance for stability.
    """
    if x.ndim != 2 or x.shape[0] < 2:
        return float("nan")
    v = np.var(x - x.mean(axis=0, keepdims=True), axis=0, ddof=1)
    d = v.shape[0]
    ridge = eps * float(np.mean(v))
    v_reg = v + ridge
    v_reg = np.clip(v_reg, 1e-30, None)
    return 0.5 * float(np.sum(np.log(v_reg)))


def subspace_entropy(x: np.ndarray, var_frac: float = 0.95, max_k: int = 256, eps: float = 1e-3) -> Tuple[float, int]:
    """Entropy in PCA subspace capturing given variance fraction.

    Returns (entropy, k_components).
    """
    if x.ndim != 2 or x.shape[0] < 2:
        return float("nan"), 0
    # Center and compute SVD on samples x (N x D)
    xc = x - x.mean(axis=0, keepdims=True)
    # Use economical SVD for stability
    # xc = U S V^T, singular values s relate to eigenvalues of covariance: lambda = s^2 / (N-1)
    try:
        U, s, Vt = np.linalg.svd(xc, full_matrices=False)
    except np.linalg.LinAlgError:
        return float("nan"), 0
    n = x.shape[0]
    eigvals = (s ** 2) / max(n - 1, 1)
    total = float(np.sum(eigvals))
    if total <= 0:
        return float("nan"), 0
    # Choose top-k eigenvalues covering var_frac up to max_k
    csum = np.cumsum(eigvals)
    k_needed = int(np.searchsorted(csum, var_frac * total) + 1)
    k = int(min(max_k, max(1, k_needed)))
    top = eigvals[:k]
    ridge = eps * float(np.mean(top))
    top_reg = np.clip(top + ridge, 1e-30, None)
    ent = 0.5 * float(np.sum(np.log(top_reg)))
    return ent, k


def collect_resid_sequences(model, tokens: torch.Tensor, layer: int, fwd_hooks=None):
    """Return (logits, residual_seq) for the requested layer.

    If fwd_hooks is provided, use run_with_hooks and capture the residual via a
    temporary hook (TransformerLens run_with_cache does not accept fwd_hooks).
    """
    node = f"blocks.{layer}.hook_resid_post"
    if fwd_hooks is None:
        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens)
        resid = cache[node]
        return logits, resid

    captured = {}

    def cap_fn(act, hook):
        captured["resid"] = act.detach()

    hooks = list(fwd_hooks) + [(node, cap_fn)]
    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    resid = captured.get("resid")
    if resid is None:
        raise RuntimeError(f"Failed to capture residual at {node}")
    return logits, resid


def simple_curvature(resid_seq: torch.Tensor, k_early: int = 5) -> Dict[str, float]:
    """Compute a simple discrete curvature proxy across tokens.

    curv_t = ||r_{t+1} - r_t||_2; report mean overall, early, and late.
    """
    # resid_seq: [B, S, D]
    if resid_seq.size(1) < 2:
        return {"curv_mean": float("nan"), "curv_early": float("nan"), "curv_late": float("nan")}
    diffs = resid_seq[:, 1:, :] - resid_seq[:, :-1, :]
    curv = torch.linalg.vector_norm(diffs, dim=-1)  # [B, S-1]
    curv_mean = float(curv.mean().item())
    early = curv[:, : min(k_early, curv.size(1))]
    late = curv[:, min(k_early, curv.size(1)) :]
    curv_early = float(early.mean().item()) if early.numel() > 0 else float("nan")
    curv_late = float(late.mean().item()) if late.numel() > 0 else float("nan")
    return {"curv_mean": curv_mean, "curv_early": curv_early, "curv_late": curv_late}


def zero_heads_hook(layer: int, heads: Iterable[int]):
    node = f"blocks.{layer}.attn.hook_z"

    def fn(z, hook):
        z = z.clone()
        for h in heads:
            z[:, :, h, :] = 0.0
        return z

    return (node, fn)


def main() -> None:
    args = parse_args()
    child = build_child_cfg(args.config, args.tag)
    if args.device is not None:
        child = ChildCfg(
            run_name=child.run_name,
            device=args.device,
            model=child.model,
            dataset=child.dataset,
            seed=child.seed,
        )
    # Optional: override model name
    if args.model_name is not None:
        child.model = {**child.model, "name": args.model_name}

    rows, split, data_hash = datasets.load_split(child.dataset)
    if args.samples and len(rows) > args.samples:
        rows = rows[: args.samples]

    # Build token batch from clean prompts
    clean_field = child.dataset["clean_field"]
    clean_texts = [ex[clean_field] for ex in rows]

    model = load_model.load_transformerlens(child.model, device=child.device)
    model.eval()
    tokens = model.to_tokens(clean_texts)

    # Determine which entropy methods to compute
    methods = [m.strip().lower() for m in args.entropy_methods.split(",") if m.strip()]
    valid_methods = {"full", "diagonal", "subspace", "per_token"}
    for m in methods:
        if m not in valid_methods:
            raise ValueError(f"Unknown entropy method '{m}'. Valid: {sorted(valid_methods)}")

    # Baseline forward
    base_logits, base_resid_seq = collect_resid_sequences(model, tokens, args.layer)
    base_resid_last = base_resid_seq[:, -1, :].detach().to(torch.float32).cpu().numpy()
    base_mu, base_cov = mvn_fit(base_resid_last)
    base_H = gaussian_entropy_rel(base_resid_last)
    # Additional baseline activation entropy variants
    base_extra = {}
    if "diagonal" in methods:
        base_extra["activation_entropy_diagonal"] = diagonal_entropy(base_resid_last)
    if "subspace" in methods:
        ent_s, k_s = subspace_entropy(base_resid_last, var_frac=args.subspace_var, max_k=args.subspace_max_k)
        base_extra["activation_entropy_subspace"] = ent_s
        base_extra["activation_entropy_subspace_k"] = float(k_s)
    if "per_token" in methods:
        seq_np = base_resid_seq.detach().to(torch.float32).cpu().numpy()
        ents = [gaussian_entropy_rel(seq_np[:, pos, :]) for pos in range(seq_np.shape[1])]
        base_extra["activation_entropy_per_token_full"] = float(np.nanmean(ents))

    base_out_H = softmax_entropy_last(base_logits)
    base_curv = simple_curvature(base_resid_seq)

    # Ablated heads (if provided)
    results = {
        "config": str(args.config),
        "tag": args.tag,
        "dataset_id": child.dataset["id"],
        "split": child.dataset["split"],
        "n_examples": len(rows),
        "layer": args.layer,
        "heads": args.heads or [],
        "data_hash": data_hash,
        "baseline": {
            "activation_entropy": base_H,
            "output_entropy": base_out_H,
            **base_curv,
            **base_extra,
        },
    }

    ablated_metrics = None
    if args.heads and len(args.heads) > 0:
        hook = zero_heads_hook(args.layer, args.heads)
        abl_logits, abl_resid_seq = collect_resid_sequences(model, tokens, args.layer, fwd_hooks=[hook])
        abl_resid_last = abl_resid_seq[:, -1, :].detach().to(torch.float32).cpu().numpy()
        abl_mu, abl_cov = mvn_fit(abl_resid_last)
        abl_H = gaussian_entropy_rel(abl_resid_last)
        abl_out_H = softmax_entropy_last(abl_logits)
        abl_curv = simple_curvature(abl_resid_seq)

        kl_act = gaussian_kl(base_mu, base_cov, abl_mu, abl_cov)

        ablated_metrics = {
            "activation_entropy": abl_H,
            "output_entropy": abl_out_H,
            "activation_kl_mvn": kl_act,
            **abl_curv,
        }
        # Additional ablated activation entropy variants
        if "diagonal" in methods:
            ablated_metrics["activation_entropy_diagonal"] = diagonal_entropy(abl_resid_last)
        if "subspace" in methods:
            ent_sa, k_sa = subspace_entropy(abl_resid_last, var_frac=args.subspace_var, max_k=args.subspace_max_k)
            ablated_metrics["activation_entropy_subspace"] = ent_sa
            ablated_metrics["activation_entropy_subspace_k"] = float(k_sa)
        if "per_token" in methods:
            seq_a = abl_resid_seq.detach().to(torch.float32).cpu().numpy()
            ents_a = [gaussian_entropy_rel(seq_a[:, pos, :]) for pos in range(seq_a.shape[1])]
            ablated_metrics["activation_entropy_per_token_full"] = float(np.nanmean(ents_a))
        results["ablated"] = ablated_metrics
        results["deltas"] = {
            "activation_entropy": float(abl_H - base_H),
            "output_entropy": float(abl_out_H - base_out_H),
            "curv_mean": float(abl_curv["curv_mean"] - base_curv["curv_mean"]),
            "curv_early": float(abl_curv["curv_early"] - base_curv["curv_early"]),
            "curv_late": float(abl_curv["curv_late"] - base_curv["curv_late"]),
        }
        # Deltas for additional methods
        if "diagonal" in methods and "activation_entropy_diagonal" in base_extra:
            results["deltas"]["activation_entropy_diagonal"] = float(
                ablated_metrics["activation_entropy_diagonal"] - base_extra["activation_entropy_diagonal"]
            )
        if "subspace" in methods and "activation_entropy_subspace" in base_extra:
            results["deltas"]["activation_entropy_subspace"] = float(
                ablated_metrics["activation_entropy_subspace"] - base_extra["activation_entropy_subspace"]
            )
        if "per_token" in methods and "activation_entropy_per_token_full" in base_extra:
            results["deltas"]["activation_entropy_per_token_full"] = float(
                ablated_metrics["activation_entropy_per_token_full"] - base_extra["activation_entropy_per_token_full"]
            )

    # Random head controls (same cardinality)
    if args.random_samples and args.random_samples > 0:
        n_heads = model.cfg.n_heads
        k = len(args.heads) if args.heads else 1
        rng = np.random.default_rng(0)
        rand_deltas_H = []  # full last-pos entropy deltas
        rand_deltas = {  # additional metrics
            "output_entropy": [],
            "activation_entropy_diagonal": [],
            "activation_entropy_subspace": [],
            "activation_entropy_per_token_full": [],
        }
        for _ in tqdm(range(args.random_samples), desc="Random controls"):
            sample = sorted(rng.choice(np.arange(n_heads), size=k, replace=False).tolist())
            hook = zero_heads_hook(args.layer, sample)
            r_logits, r_seq = collect_resid_sequences(model, tokens, args.layer, fwd_hooks=[hook])
            r_last = r_seq[:, -1, :].detach().to(torch.float32).cpu().numpy()
            r_H = gaussian_entropy_rel(r_last)
            r_out_H = softmax_entropy_last(r_logits)
            rand_deltas_H.append(r_H - base_H)
            rand_deltas["output_entropy"].append(r_out_H - base_out_H)
            if "diagonal" in methods and "activation_entropy_diagonal" in base_extra:
                rand_deltas["activation_entropy_diagonal"].append(diagonal_entropy(r_last) - base_extra["activation_entropy_diagonal"])
            if "subspace" in methods and "activation_entropy_subspace" in base_extra:
                r_sub, _ = subspace_entropy(r_last, var_frac=args.subspace_var, max_k=args.subspace_max_k)
                rand_deltas["activation_entropy_subspace"].append(r_sub - base_extra["activation_entropy_subspace"])
            if "per_token" in methods and "activation_entropy_per_token_full" in base_extra:
                r_seq_np = r_seq.detach().to(torch.float32).cpu().numpy()
                ents_r = [gaussian_entropy_rel(r_seq_np[:, pos, :]) for pos in range(r_seq_np.shape[1])]
                rand_deltas["activation_entropy_per_token_full"].append(float(np.nanmean(ents_r)) - base_extra["activation_entropy_per_token_full"])
        results["random_control"] = {
            "k": k,
            "samples": args.random_samples,
            "delta_activation_entropy": {
                "mean": float(np.mean(rand_deltas_H)),
                "std": float(np.std(rand_deltas_H)),
                "values": [float(x) for x in rand_deltas_H],
            },
            "delta_output_entropy": {
                "mean": float(np.mean(rand_deltas["output_entropy"])) if rand_deltas["output_entropy"] else float("nan"),
                "std": float(np.std(rand_deltas["output_entropy"])) if rand_deltas["output_entropy"] else float("nan"),
                "values": [float(x) for x in rand_deltas["output_entropy"]],
            },
        }
        # Attach additional metric arrays
        if "diagonal" in methods and rand_deltas["activation_entropy_diagonal"]:
            arr = rand_deltas["activation_entropy_diagonal"]
            results["random_control"]["delta_activation_entropy_diagonal"] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "values": [float(x) for x in arr],
            }
        if "subspace" in methods and rand_deltas["activation_entropy_subspace"]:
            arr = rand_deltas["activation_entropy_subspace"]
            results["random_control"]["delta_activation_entropy_subspace"] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "values": [float(x) for x in arr],
            }
        if "per_token" in methods and rand_deltas["activation_entropy_per_token_full"]:
            arr = rand_deltas["activation_entropy_per_token_full"]
            results["random_control"]["delta_activation_entropy_per_token_full"] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "values": [float(x) for x in arr],
            }
        if ablated_metrics is not None:
            # Percentile / p-values for observed deltas under the random null
            dH = results["deltas"]["activation_entropy"]
            dout = results["deltas"]["output_entropy"]
            rand_full = np.array(rand_deltas_H)
            # Two-tailed for activation entropy (full)
            p_lower = float((rand_full <= dH).mean())
            p_upper = float((rand_full >= dH).mean())
            results["random_control"]["pvals"] = {
                "activation_entropy_full": {
                    "p_lower": p_lower,
                    "p_upper": p_upper,
                    "p_two_tailed": float(2 * min(p_lower, p_upper)),
                }
            }
            # Output entropy: predicted negative => lower tail
            rand_out = np.array(results["random_control"]["delta_output_entropy"]["values"]) if results["random_control"]["delta_output_entropy"]["values"] else np.array([])
            if rand_out.size > 0:
                p_lower_out = float((rand_out <= dout).mean())
                results["random_control"]["pvals"]["output_entropy"] = {
                    "p_lower": p_lower_out,
                    "percentile_extreme": float(1.0 - p_lower_out),
                }
            # Additional activation metrics p-values (two-tailed)
            for key_name, rc_key in [
                ("activation_entropy_diagonal", "delta_activation_entropy_diagonal"),
                ("activation_entropy_subspace", "delta_activation_entropy_subspace"),
                ("activation_entropy_per_token_full", "delta_activation_entropy_per_token_full"),
            ]:
                if key_name in results["deltas"] and rc_key in results["random_control"]:
                    arr = np.array(results["random_control"][rc_key]["values"])  # type: ignore[index]
                    if arr.size > 0:
                        obs = results["deltas"][key_name]
                        pl = float((arr <= obs).mean())
                        pu = float((arr >= obs).mean())
                        results["random_control"]["pvals"][key_name] = {
                            "p_lower": pl,
                            "p_upper": pu,
                            "p_two_tailed": float(2 * min(pl, pu)),
                        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

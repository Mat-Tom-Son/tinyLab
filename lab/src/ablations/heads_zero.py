"""Head zeroing ablation for finding critical attention heads."""
import numpy as np
import pandas as pd
import torch
from ..components import metrics as M


def run(model, dset, cfg, battery, device):
    """Run head zeroing experiment.

    Zeros out individual attention heads to measure their importance.

    Args:
        model: TransformerLens model
        dset: Dataset rows
        cfg: Main config dict
        battery: Battery config dict
        device: Device to run on

    Returns:
        Dict with summary, per_example, and impact_matrix
    """
    model.to(device)
    model.eval()
    scales = battery.get("scales", [0.0])

    clean_texts = [ex[cfg["dataset"]["clean_field"]] for ex in dset]
    toks = model.to_tokens(clean_texts)

    # Get baseline clean logits
    with torch.no_grad():
        clean_logits = model(toks)

    impact_rows, per_ex_rows = [], []

    # Respect layer/head subsets
    layers = (
        range(model.cfg.n_layers)
        if battery.get("layers", "all") == "all"
        else battery["layers"]
    )

    random_control = battery.get("random_control", False)
    random_count = battery.get("random_count")

    for layer in layers:
        # Allow subset of heads, or all
        heads_to_run = battery.get("heads")
        if heads_to_run is None:
            heads_list = list(range(model.cfg.n_heads))
        else:
            heads_list = list(heads_to_run)
        if not heads_list:
            raise ValueError("heads list is empty")

        if random_control:
            rng = np.random.default_rng(cfg.get("seed", 0) + layer)
            k = random_count or len(heads_list)
            k = min(len(heads_list), max(1, k))
            heads_iter = rng.choice(heads_list, size=k, replace=False)
        else:
            heads_iter = heads_list

        for head in heads_iter:
            node = f"blocks.{layer}.attn.hook_z"

            for s in scales:

                def zero_fn(z, hook):
                    # z: [batch, seq, heads, d_head]
                    z = z.clone()
                    z[:, :, head, :] = z[:, :, head, :] * s
                    return z

                with torch.no_grad():
                    logits_patched = model.run_with_hooks(
                        toks, fwd_hooks=[(node, zero_fn)]
                    )

                summary, per_ex = M.evaluate_outputs(
                    model, clean_logits, logits_patched, dset, cfg
                )
                summary.update({"layer": layer, "head": head, "scale": s})
                impact_rows.append(summary)

                for ex in per_ex:
                    ex.update({"layer": layer, "head": head, "scale": s})
                per_ex_rows.extend(per_ex)

    df = pd.DataFrame(impact_rows)
    agg_summary = {
        m: float(df[m].mean()) for m in ["logit_diff", "kl_div", "acc_flip_rate", "p_drop"]
    }

    # Pivot for heatmap
    impact_matrix = df.pivot_table(
        index=["layer", "head"], columns="scale", values="logit_diff"
    )

    # Build standardized impact table (v1.1)
    # Transform from wide to long format for machine-readable analysis
    impact_table_rows = []
    for _, row in df.iterrows():
        for metric in ["logit_diff", "kl_div", "acc_flip_rate", "p_drop"]:
            impact_table_rows.append({
                "run_id": cfg.get("run_name", "unknown"),
                "seed": cfg.get("seed", 0),
                "layer": int(row["layer"]),
                "head": int(row["head"]),
                "scale": float(row["scale"]),
                "metric": metric,
                "value": float(row[metric])
            })

    head_impact_table = pd.DataFrame(impact_table_rows)

    return {
        "summary": agg_summary,
        "per_example": pd.DataFrame(per_ex_rows),
        "impact_matrix": impact_matrix,
        "head_impact_table": head_impact_table,
    }

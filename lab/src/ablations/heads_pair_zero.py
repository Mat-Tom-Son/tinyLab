"""Pairwise head zeroing for backup circuit analysis."""
import pandas as pd
import torch
from ..components import metrics as M


def run(model, dset, cfg, battery, device):
    """Run pairwise head zeroing experiment.

    Zeros out pairs of attention heads to detect redundancy and backup circuits.

    Args:
        model: TransformerLens model
        dset: Dataset rows
        cfg: Main config dict
        battery: Battery config dict with "pairs" list
        device: Device to run on

    Returns:
        Dict with summary, per_example, and impact_matrix (as DataFrame)
    """
    model.to(device)
    model.eval()

    # List of {"layer": L, "h1": H1, "h2": H2}
    pairs = battery["pairs"]

    clean_texts = [ex[cfg["dataset"]["clean_field"]] for ex in dset]
    toks = model.to_tokens(clean_texts)

    # Get baseline clean logits
    with torch.no_grad():
        clean_logits = model(toks)

    impact_rows, per_ex_rows = [], []

    metric_names = ["logit_diff", "kl_div", "acc_flip_rate", "p_drop"]

    for p in pairs:
        layer, h1, h2 = p["layer"], p["h1"], p["h2"]
        node = f"blocks.{layer}.attn.hook_z"

        def zero_pair_fn(z, hook):
            z = z.clone()
            z[:, :, h1, :] = 0.0
            z[:, :, h2, :] = 0.0
            return z

        with torch.no_grad():
            logits_patched = model.run_with_hooks(toks, fwd_hooks=[(node, zero_pair_fn)])

        summary, per_ex = M.evaluate_outputs(
            model, clean_logits, logits_patched, dset, cfg
        )
        summary.update({"layer": layer, "h1": h1, "h2": h2})
        impact_rows.append(summary)

        for ex in per_ex:
            ex.update({"layer": layer, "h1": h1, "h2": h2})
        per_ex_rows.extend(per_ex)

    df = pd.DataFrame(impact_rows)
    agg_summary = {
        m: float(df[m].mean()) for m in metric_names if m in df
    }

    # Build standardized table for cross-run analysis
    pair_impact_rows = []
    for row in impact_rows:
        for metric in metric_names:
            if metric not in row:
                continue
            pair_impact_rows.append(
                {
                    "run_id": cfg.get("run_name", "unknown"),
                    "seed": cfg.get("seed", 0),
                    "layer": int(row["layer"]),
                    "head": f"{row['h1']}-{row['h2']}",
                    "h1": int(row["h1"]),
                    "h2": int(row["h2"]),
                    "scale": 0.0,
                    "metric": metric,
                    "value": float(row[metric]),
                }
            )

    return {
        "summary": agg_summary,
        "per_example": pd.DataFrame(per_ex_rows),
        "impact_matrix": df,  # No pivot, just list of pairs
        "head_impact_table": pd.DataFrame(pair_impact_rows),
    }

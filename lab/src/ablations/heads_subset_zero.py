"""Zero arbitrary sets of attention heads for cooperative circuit analysis."""
import pandas as pd
import torch

from ..components import metrics as M


def run(model, dset, cfg, battery, device):
    """Run subset head zeroing experiment.

    Battery config must contain a ``groups`` list with entries of the form::

        {"layer": 0, "heads": [2, 4, 7], "label": "triplet"}

    Each group is zeroed independently across all positions.
    """
    model.to(device)
    model.eval()

    groups = battery.get("groups")
    if not groups:
        raise ValueError("heads_subset_zero battery requires a non-empty 'groups' list.")

    clean_texts = [ex[cfg["dataset"]["clean_field"]] for ex in dset]
    toks = model.to_tokens(clean_texts)

    with torch.no_grad():
        clean_logits = model(toks)

    impact_rows, per_ex_rows = [], []
    metric_names = ["logit_diff", "kl_div", "acc_flip_rate", "p_drop"]

    for group in groups:
        layer = group["layer"]
        heads = group["heads"]
        label = group.get("label") or "-".join(str(h) for h in heads)
        node = f"blocks.{layer}.attn.hook_z"

        def zero_subset(z, hook):
            z = z.clone()
            for head in heads:
                z[:, :, head, :] = 0.0
            return z

        with torch.no_grad():
            logits_patched = model.run_with_hooks(toks, fwd_hooks=[(node, zero_subset)])

        summary, per_ex = M.evaluate_outputs(model, clean_logits, logits_patched, dset, cfg)
        summary.update({"layer": layer, "heads": heads, "label": label})
        impact_rows.append(summary)

        for ex in per_ex:
            ex.update({"layer": layer, "heads": heads, "label": label})
        per_ex_rows.extend(per_ex)

    df = pd.DataFrame(impact_rows)
    agg_summary = {}
    for m in metric_names:
        if m in df:
            series = pd.to_numeric(df[m], errors="coerce")
            agg_summary[m] = float(series.mean())

    table_rows = []
    for row in impact_rows:
        for metric in metric_names:
            if metric not in row:
                continue
            table_rows.append(
                {
                    "run_id": cfg.get("run_name", "unknown"),
                    "seed": cfg.get("seed", 0),
                    "layer": int(row["layer"]),
                    "heads": "-".join(str(h) for h in row["heads"]),
                    "label": row.get("label"),
                    "metric": metric,
                    "value": float(row[metric]),
                }
            )

    subset_table = pd.DataFrame(table_rows)

    return {
        "summary": agg_summary,
        "per_example": pd.DataFrame(per_ex_rows),
        "impact_matrix": None,
        "head_impact_table": subset_table,
    }

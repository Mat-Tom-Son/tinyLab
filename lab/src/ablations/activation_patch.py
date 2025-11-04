"""Activation patching ablation for layer-wise causal analysis."""
import pandas as pd
import torch
from transformer_lens import HookedTransformer
from ..components import metrics as M


def run(model: HookedTransformer, dset, cfg, battery, device):
    """Run activation patching experiment.

    Patches activations from a source run into a target run to measure
    causal importance of each layer's residual stream.

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

    clean_texts = [ex[cfg["dataset"]["clean_field"]] for ex in dset]
    corrupt_texts = [ex[cfg["dataset"]["corrupt_field"]] for ex in dset]
    token_clean = model.to_tokens(clean_texts)
    token_corrupt = model.to_tokens(corrupt_texts)

    # Prepare span-aware sequence batches once if requested
    metric_span = cfg.get("metric_span", "first_token")
    seq_batches = None
    if metric_span != "first_token":
        # For sequence evaluation, we always use clean prompt + {target, foil}
        tokens_tgt, tokens_foil, cont_t, cont_f = M.build_sequence_batches(model, dset, cfg)
        seq_batches = (tokens_tgt, tokens_foil, cont_t, cont_f)

    # Forward w/ cache to get baseline activations and logits
    with torch.no_grad():
        clean_logits, cache_clean = model.run_with_cache(token_clean)
        corrupt_logits, cache_corrupt = model.run_with_cache(token_corrupt)

    # Choose patch direction and target sequence
    direction = battery.get("patch_direction", "clean->corrupt")
    if direction == "clean->corrupt":
        src_cache = cache_clean
        tgt_tokens = token_corrupt
        baseline_logits = clean_logits
    elif direction == "corrupt->clean":
        src_cache = cache_corrupt
        tgt_tokens = token_clean
        baseline_logits = corrupt_logits
    else:
        raise ValueError(f"Unknown patch_direction {direction}")

    # Choose what to patch
    gran = battery.get("granularity", "layer_resid")  # "head_out", "mlp_out"
    layers = (
        range(model.cfg.n_layers)
        if battery.get("layers", "all") == "all"
        else battery["layers"]
    )

    sham_control = battery.get("sham_control", False)
    if sham_control:
        # For sham runs the baseline should match an unmodified forward pass on tgt_tokens
        with torch.no_grad():
            baseline_logits = model(tgt_tokens)

    impact_rows, per_ex_rows = [], []
    warned_nodes = set()

    for layer in layers:
        # Define node names in TransformerLens
        if gran == "layer_resid":
            node = f"blocks.{layer}.hook_resid_post"
        elif gran == "mlp_out":
            node = f"blocks.{layer}.mlp.hook_post"
        elif gran == "head_out":
            node = f"blocks.{layer}.attn.hook_z"
        else:
            raise ValueError(f"Unknown granularity: {gran}")

        def patch_fn(act, hook, node_name=node):
            # copy activations from source cache into the target run
            if sham_control:
                return act
            src = src_cache[node_name]
            if act.shape != src.shape:
                # Allow mismatched sequence lengths by patching the overlapping prefix.
                if node_name not in warned_nodes:
                    print(
                        f"[yellow]Shape mismatch at {node_name}: {act.shape} vs {src.shape}. "
                        "Patching overlapping positions only.[/yellow]"
                    )
                    warned_nodes.add(node_name)
                patched = act.clone()
                if act.shape[0] != src.shape[0]:
                    raise ValueError(f"Batch mismatch at {node_name}: {act.shape} vs {src.shape}")
                # Assume sequence dimension is index 1 (batch, seq, ...)
                seq_dim = 1
                seq_len = min(act.shape[seq_dim], src.shape[seq_dim])
                patch_slice = [slice(None)] * len(act.shape)
                patch_slice[seq_dim] = slice(0, seq_len)
                patched[tuple(patch_slice)] = src[tuple(patch_slice)]
                return patched
            return src

        # Run patched target pass
        with torch.no_grad():
            logits_patched = model.run_with_hooks(tgt_tokens, fwd_hooks=[(node, patch_fn)])

        # Compute metrics against the baseline logits
        summary, per_ex = M.evaluate_outputs(
            model, baseline_logits, logits_patched, dset, cfg
        )
        # Span-aware metrics (optional): evaluate on clean prompt + {target, foil}
        if seq_batches is not None:
            tokens_tgt, tokens_foil, cont_t, cont_f = seq_batches

            def f_clean(tokens):
                with torch.no_grad():
                    return model(tokens)

            def f_abl(tokens):
                with torch.no_grad():
                    return model.run_with_hooks(tokens, fwd_hooks=[(node, patch_fn)])

            seq_metrics = M.compute_seq_metrics_from_forwards(
                model, tokens_tgt, tokens_foil, cont_t, cont_f, f_clean, f_abl
            )
            summary.update(seq_metrics)
        summary["layer"] = layer
        if logits_patched.shape != baseline_logits.shape:
            seq_len = min(logits_patched.size(1), baseline_logits.size(1))
            diff = logits_patched[:, :seq_len, :] - baseline_logits[:, :seq_len, :]
        else:
            diff = logits_patched - baseline_logits
        max_delta = torch.max(torch.abs(diff)).item()
        summary["max_abs_delta"] = float(max_delta)
        if sham_control and layer == layers[0]:
            print(f"[green]Sham control max abs delta: {max_delta:.6e}[/green]")
        impact_rows.append(summary)

        # Tag per-example data
        for ex in per_ex:
            ex["layer"] = layer
            ex["granularity"] = gran
        per_ex_rows.extend(per_ex)

    df = pd.DataFrame(impact_rows)

    # Aggregate metrics
    metric_names = ["logit_diff", "kl_div", "acc_flip_rate", "p_drop", "max_abs_delta"]
    for m in ["seq_logprob_diff", "seq_p_drop", "seq_kl_mean"]:
        if m in df:
            metric_names.append(m)
    agg_summary = {m: float(df[m].mean()) for m in metric_names if m in df}

    # Create pivot table for heatmap
    impact_matrix = df.pivot_table(index="layer", values=metric_names)

    # Build standardized impact table (v1.1)
    # Transform from wide to long format for machine-readable analysis
    impact_table_rows = []
    for _, row in df.iterrows():
        for metric in metric_names:
            if metric not in row:
                continue
            impact_table_rows.append({
                "run_id": cfg.get("run_name", "unknown"),
                "seed": cfg.get("seed", 0),
                "layer": int(row["layer"]),
                "granularity": gran,
                "metric": metric,
                "value": float(row[metric])
            })

    layer_impact_table = pd.DataFrame(impact_table_rows)

    return {
        "summary": agg_summary,
        "per_example": pd.DataFrame(per_ex_rows),
        "impact_matrix": impact_matrix,
        "layer_impact_table": layer_impact_table,
    }

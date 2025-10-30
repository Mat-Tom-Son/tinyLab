"""Metric computation for ablation experiments."""
import torch
import numpy as np


# --- Metric Helpers ---


def _first_token_ids(model, dset, cfg):
    """Gets the first token ID for target and foil with safety checks."""
    t_ids, f_ids = [], []
    target_field = cfg["dataset"]["target_field"]
    foil_field = cfg["dataset"]["foil_field"]

    for ex in dset:
        # Safety: ensure single-token targets/foils
        tid = model.to_single_token(ex[target_field])
        if tid is None:
            raise ValueError(
                f"Target must be single-token, got: {ex[target_field]!r}"
            )

        fid = model.to_single_token(ex[foil_field])
        if fid is None:
            raise ValueError(f"Foil must be single-token, got: {ex[foil_field]!r}")

        t_ids.append(tid)
        f_ids.append(fid)

    device = model.cfg.device
    return torch.tensor(t_ids, device=device), torch.tensor(f_ids, device=device)


# --- Core Metrics (First Token) ---


def logit_diff_first_token(logits, t_ids, f_ids):
    """Logit diff at final position for first token of target/foil."""
    last_logits = logits[:, -1, :]
    t_logits = last_logits[torch.arange(last_logits.size(0)), t_ids]
    f_logits = last_logits[torch.arange(last_logits.size(0)), f_ids]
    return (t_logits - f_logits).mean().item()


def p_drop_first_token(clean_logits, ablated_logits, t_ids):
    """Mean drop in p(target) at final position."""
    c_probs = torch.softmax(clean_logits[:, -1, :], dim=-1)
    a_probs = torch.softmax(ablated_logits[:, -1, :], dim=-1)

    p_clean = c_probs[torch.arange(c_probs.size(0)), t_ids]
    p_ablated = a_probs[torch.arange(a_probs.size(0)), t_ids]

    return (p_clean - p_ablated).mean().item()


def kl_div_last_token(clean_logits, ablated_logits, eps=1e-8):
    """KL(p_clean || p_ablated) at final position."""
    p = torch.log_softmax(clean_logits[:, -1, :], dim=-1).exp().clamp_min(eps)
    q_log = torch.log_softmax(ablated_logits[:, -1, :], dim=-1)

    # More stable: KL(p || q) = sum(p * (log_p - log_q))
    log_p = p.log()
    kl = (p * (log_p - q_log)).sum(dim=-1).mean().item()
    return kl


def acc_flip_rate_last(clean_logits, ablated_logits, t_ids):
    """Fraction of examples where argmax flips away from target_id."""
    # Note: This checks if the ablated pred is *not* the target.
    # It doesn't require it to flip *to* the foil.
    ablated_preds = torch.softmax(ablated_logits[:, -1, :], dim=-1).argmax(dim=-1)
    return (ablated_preds != t_ids).float().mean().item()


# --- Main Evaluator ---


def evaluate_outputs(model, clean_logits, ablated_logits, dset, cfg):
    """
    Computes all standard metrics, assuming "first_token" span.

    Args:
        model: TransformerLens model
        clean_logits: Baseline logits (clean or corrupt depending on direction)
        ablated_logits: Patched/ablated logits
        dset: Dataset rows
        cfg: Config dict

    Returns:
        (summary_dict, per_example_list)
    """
    # TODO: Add "full_span" metric logic based on cfg["metric_span"]
    metric_span = cfg.get("metric_span", "first_token")
    if metric_span != "first_token":
        print(
            f"Warning: metric_span '{metric_span}' not fully implemented. "
            "Defaulting to 'first_token'."
        )

    t_ids, f_ids = _first_token_ids(model, dset, cfg)

    ld = logit_diff_first_token(ablated_logits, t_ids, f_ids)
    kl = kl_div_last_token(clean_logits, ablated_logits)
    afr = acc_flip_rate_last(clean_logits, ablated_logits, t_ids)
    pd = p_drop_first_token(clean_logits, ablated_logits, t_ids)

    # Per-example exports
    per_ex = []
    ablated_preds = ablated_logits[:, -1, :].argmax(dim=-1)
    for i in range(len(t_ids)):
        per_ex.append(
            {
                "i": i,
                "pred_id": int(ablated_preds[i]),
                "target_id": int(t_ids[i]),
                "foil_id": int(f_ids[i]),
                "seed": cfg.get("seed", 0),  # Tag with seed
            }
        )

    summary = {"logit_diff": ld, "kl_div": kl, "acc_flip_rate": afr, "p_drop": pd}
    return summary, per_ex

"""Metric computation for ablation experiments.

Adds span-aware (multi-token) metrics alongside the existing first-token
metrics. The new helpers allow ablation modules to provide callable forward
passes so we can evaluate sequence log-probabilities under clean vs ablated
settings without changing the harness surface.
"""
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
    # First-token metrics are the default and always computed for continuity
    # with prior results and downstream analysis.
    metric_span = cfg.get("metric_span", "first_token")

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


# --- Span-aware helpers (multi-token) ---

def _gather_next_log_probs(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Return log p(x_t | x_<t) for each position, as a dense tensor.

    Args:
        logits: [batch, seq, vocab]
        token_ids: [batch, seq]

    Returns:
        log_probs: [batch, seq-1] where entry (i, j) is log p(token_ids[i, j+1] | token_ids[i, :j+1])
    """
    # Log-probs for predicting token j+1 from position j
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target = token_ids[:, 1:]
    gathered = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return gathered


def build_sequence_batches(model, dset, cfg):
    """Prepare batched tokens for clean+target and clean+foil sequences.

    Returns:
        tokens_target: Tensor[batch, seq_t_max]
        tokens_foil:   Tensor[batch, seq_f_max]
        cont_lens_target: list[int] length batch (num tokens in target continuation)
        cont_lens_foil:   list[int] length batch (num tokens in foil continuation)
    """
    clean_field = cfg["dataset"]["clean_field"]
    target_field = cfg["dataset"]["target_field"]
    foil_field = cfg["dataset"]["foil_field"]

    clean_texts = [ex[clean_field] for ex in dset]
    target_texts = [ex[target_field] for ex in dset]
    foil_texts = [ex[foil_field] for ex in dset]

    target_joined = [c + t for c, t in zip(clean_texts, target_texts)]
    foil_joined = [c + f for c, f in zip(clean_texts, foil_texts)]

    tokens_target = model.to_tokens(target_joined)
    tokens_foil = model.to_tokens(foil_joined)

    # Continuation lengths computed via tokenizer for robustness
    tok = getattr(model, "tokenizer", None)
    if tok is None:
        # Fallback: derive continuation length from string tokenization via to_tokens on single items
        cont_lens_target = []
        cont_lens_foil = []
        for t in target_texts:
            cont_lens_target.append(int(model.to_tokens([t]).shape[1]))
        for f in foil_texts:
            cont_lens_foil.append(int(model.to_tokens([f]).shape[1]))
    else:
        cont_lens_target = [len(tok.encode(t, add_special_tokens=False)) for t in target_texts]
        cont_lens_foil = [len(tok.encode(f, add_special_tokens=False)) for f in foil_texts]

    return tokens_target, tokens_foil, cont_lens_target, cont_lens_foil


def compute_seq_metrics_from_forwards(
    model,
    tokens_target: torch.Tensor,
    tokens_foil: torch.Tensor,
    cont_lens_target: list,
    cont_lens_foil: list,
    forward_clean,
    forward_ablated,
):
    """Compute sequence-level metrics under clean vs ablated forwards.

    Metrics added:
      - seq_logprob_diff: mean over batch of [log p_abl(target seq) - log p_abl(foil seq)]
      - seq_p_drop:       mean over batch of [log p_clean(target seq) - log p_abl(target seq)]
      - seq_kl_mean:      mean over batch of the per-position KL(p_clean || p_abl) on target continuation positions
    """
    device = model.cfg.device
    with torch.no_grad():
        # Clean and ablated logits
        c_t = forward_clean(tokens_target)
        c_f = forward_clean(tokens_foil)
        a_t = forward_ablated(tokens_target)
        a_f = forward_ablated(tokens_foil)

        # Gather next-token log-probs
        lp_c_t = _gather_next_log_probs(c_t, tokens_target)
        lp_c_f = _gather_next_log_probs(c_f, tokens_foil)
        lp_a_t = _gather_next_log_probs(a_t, tokens_target)
        lp_a_f = _gather_next_log_probs(a_f, tokens_foil)

        # Sum per-example over continuation span (last K positions)
        def sum_last_k_per_row(mat: torch.Tensor, k_list: list[int]) -> torch.Tensor:
            out = []
            for i, k in enumerate(k_list):
                if k <= 0:
                    out.append(torch.tensor(0.0, device=mat.device))
                    continue
                out.append(mat[i, -k:].sum())
            return torch.stack(out)

        s_c_t = sum_last_k_per_row(lp_c_t, cont_lens_target)
        s_a_t = sum_last_k_per_row(lp_a_t, cont_lens_target)
        s_a_f = sum_last_k_per_row(lp_a_f, cont_lens_foil)

        # seq metrics
        seq_logprob_diff = (s_a_t - s_a_f).mean().item()
        seq_p_drop = (s_c_t - s_a_t).mean().item()

        # KL over target continuation positions
        # Compute KL(p||q) per position then average over continuation span
        # Shapes:
        #  - c_t, a_t: [B, L, V]
        # We'll compute per-position KL on logits at indices matching continuation predictions.
        # That corresponds to the last K positions of logits (predicting tokens 1..L-1),
        # but restricted to the last cont_len positions.
        p_log = torch.log_softmax(c_t[:, :-1, :], dim=-1)
        q_log = torch.log_softmax(a_t[:, :-1, :], dim=-1)
        p = p_log.exp().clamp_min(1e-8)

        # Per-position KL: sum over vocab of p * (log p - log q)
        kl_pos = (p * (p_log - q_log)).sum(dim=-1)  # [B, L-1]

        # Mask/sum last K positions per row
        kl_mean_per_row = []
        for i, k in enumerate(cont_lens_target):
            if k <= 0:
                kl_mean_per_row.append(torch.tensor(0.0, device=kl_pos.device))
                continue
            kl_mean_per_row.append(kl_pos[i, -k:].mean())
        kl_mean_per_row = torch.stack(kl_mean_per_row)
        seq_kl_mean = kl_mean_per_row.mean().item()

    return {
        "seq_logprob_diff": seq_logprob_diff,
        "seq_p_drop": seq_p_drop,
        "seq_kl_mean": seq_kl_mean,
    }

"""Causal rescue: patch head activations back after ablation (H6-style).

Two experiments:
  1) Binder heads (synthetic binding dataset): select top-K worst d_ld heads
     from a binder_sweep JSON; verify ablation lowers accuracy and patching
     restores accuracy to within ~1% of baseline.
  2) Sharpener heads (facts corpus): select top-K heads by d_entropy_final from
     a sharpener scan JSON; same ablation/patch procedure.

Outputs JSON under reports/causal_rescue_*.json with per-head metrics.

Runtime tip: With GPT‑2 Medium, ~2k examples and 7 heads completes in minutes
on Apple Silicon (MPS, float16).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from ..src.components import load_model, datasets as D, metrics as M
from .binder_sweep import synthetic_binding_dataset as synth_binder
from .layer_pca_rank import build_child_cfg as build_child
from ..src.utils import io


def head_zero_hook(layer: int, head: int):
    node = f"blocks.{layer}.attn.hook_z"

    def fn(z, hook):
        z = z.clone()
        z[:, :, head, :] = 0.0
        return z

    return node, fn


def head_patch_hook(layer: int, head: int, base_z: torch.Tensor):
    node = f"blocks.{layer}.attn.hook_z"

    def fn(z, hook):
        z = z.clone()
        # base_z: [B,S,H,d_head] captured on same tokens
        z[:, :, head, :] = base_z[:, :, head, :]
        return z

    return node, fn


def accuracy_from_logits(logits: torch.Tensor, t_ids: torch.Tensor) -> float:
    preds = torch.softmax(logits[:, -1, :], dim=-1).argmax(dim=-1)
    return (preds == t_ids).float().mean().item()


def eval_chunk(model, chunk: List[Dict[str, str]]):
    toks = model.to_tokens([ex["clean"] for ex in chunk])
    with torch.no_grad():
        clean_logits, cache = model.run_with_cache(toks)
    t_ids, f_ids = M._first_token_ids(model, chunk, {
        "dataset": {"target_field": "target", "foil_field": "foil"}
    })
    base_acc = accuracy_from_logits(clean_logits, t_ids)
    base_ld = M.logit_diff_first_token(clean_logits, t_ids, f_ids)
    return toks, clean_logits, cache, t_ids, f_ids, float(base_acc), float(base_ld)


def rescue_for_head(model, layer: int, head: int, toks, cache, t_ids, f_ids):
    # Ablate then patch-back baseline z
    node_zero, fn_zero = head_zero_hook(layer, head)
    base_z = cache[f"blocks.{layer}.attn.hook_z"].detach()
    node_patch, fn_patch = head_patch_hook(layer, head, base_z)

    with torch.no_grad():
        logits_abl = model.run_with_hooks(toks, fwd_hooks=[(node_zero, fn_zero)])
        logits_resc = model.run_with_hooks(toks, fwd_hooks=[(node_zero, fn_zero), (node_patch, fn_patch)])

    acc_abl = accuracy_from_logits(logits_abl, t_ids)
    ld_abl = M.logit_diff_first_token(logits_abl, t_ids, f_ids)
    acc_resc = accuracy_from_logits(logits_resc, t_ids)
    ld_resc = M.logit_diff_first_token(logits_resc, t_ids, f_ids)
    return float(acc_abl), float(ld_abl), float(acc_resc), float(ld_resc)


def aggregate(vals: List[Tuple[float, float]]):
    s, n = 0.0, 0
    for v, w in vals:
        s += v * w
        n += w
    return s / max(1, n)


def run_binder(args) -> Dict:
    model = load_model.load_transformerlens({"name": args.model_name, "dtype": args.dtype}, device=args.device)
    model.eval()

    # Load heads: top-K worst d_ld
    data = json.loads(Path(args.binder_input).read_text())
    rows = data.get("rows", [])
    rows_sorted = sorted(rows, key=lambda r: float(r.get("d_ld", 0.0)))[: args.binder_top_k]
    heads = [(int(r["layer"]), int(r["head"])) for r in rows_sorted]

    # Dataset
    raw = synth_binder(args.max_examples * 2, args.seed)
    # Filter to single-token targets under current tokenizer
    ok = []
    for ex in raw:
        try:
            t_ok = model.to_single_token(ex["target"]) is not None  # type: ignore[arg-type]
            f_ok = model.to_single_token(ex["foil"]) is not None    # type: ignore[arg-type]
        except AssertionError:
            t_ok, f_ok = False, False
        if not (t_ok and f_ok):
            continue
        ok.append(ex)
        if len(ok) >= args.max_examples:
            break
    if len(ok) < 512:
        raise SystemExit("binder: insufficient examples after filtering")

    results = {"heads": [], "n_examples": len(ok)}

    # Evaluate per-chunk and aggregate per head
    for layer, head in heads:
        acc_base_w, ld_base_w, acc_abl_w, ld_abl_w, acc_resc_w, ld_resc_w, n_w = [], [], [], [], [], [], []
        for i in range(0, len(ok), args.batch_size):
            chunk = ok[i : i + args.batch_size]
            toks, clean_logits, cache, t_ids, f_ids, acc_b, ld_b = eval_chunk(model, chunk)
            acc_a, ld_a, acc_r, ld_r = rescue_for_head(model, layer, head, toks, cache, t_ids, f_ids)
            n = clean_logits.size(0)
            acc_base_w.append((acc_b, n))
            ld_base_w.append((ld_b, n))
            acc_abl_w.append((acc_a, n))
            ld_abl_w.append((ld_a, n))
            acc_resc_w.append((acc_r, n))
            ld_resc_w.append((ld_r, n))

        acc_base = aggregate(acc_base_w)
        ld_base = aggregate(ld_base_w)
        acc_abl = aggregate(acc_abl_w)
        ld_abl = aggregate(ld_abl_w)
        acc_resc = aggregate(acc_resc_w)
        ld_resc = aggregate(ld_resc_w)
        within_1pct = (acc_base - acc_resc) <= 0.01
        results["heads"].append({
            "layer": layer,
            "head": head,
            "acc_base": acc_base,
            "acc_abl": acc_abl,
            "acc_resc": acc_resc,
            "ld_base": ld_base,
            "ld_abl": ld_abl,
            "ld_resc": ld_resc,
            "within_1pct": within_1pct,
        })

    out = Path(args.outdir) / f"causal_rescue_binder_{args.model_name.replace('/', '_')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}")
    return results


def run_sharpener(args) -> Dict:
    model = load_model.load_transformerlens({"name": args.model_name, "dtype": args.dtype}, device=args.device)
    model.eval()

    data = json.loads(Path(args.sharpener_input).read_text())
    rows = data.get("sharpener_scan", [])
    rows_sorted = sorted(rows, key=lambda r: float(r.get("d_entropy_final", 0.0)), reverse=True)[: args.sharpener_top_k]
    heads = [(int(r["layer"]), int(r["head"])) for r in rows_sorted]

    # Load facts split from config
    child = build_child(Path(args.config), args.tag)
    rows_ds, split_info, data_hash = D.load_split(child["dataset"])
    # Truncate sample
    rows_ds = rows_ds[: args.max_examples]

    results = {"heads": [], "n_examples": len(rows_ds)}

    for layer, head in heads:
        acc_base_w, ld_base_w, acc_abl_w, ld_abl_w, acc_resc_w, ld_resc_w, n_w = [], [], [], [], [], [], []
        for i in range(0, len(rows_ds), args.batch_size):
            chunk = rows_ds[i : i + args.batch_size]
            toks, clean_logits, cache, t_ids, f_ids, acc_b, ld_b = eval_chunk(model, chunk)
            acc_a, ld_a, acc_r, ld_r = rescue_for_head(model, layer, head, toks, cache, t_ids, f_ids)
            n = clean_logits.size(0)
            acc_base_w.append((acc_b, n))
            ld_base_w.append((ld_b, n))
            acc_abl_w.append((acc_a, n))
            ld_abl_w.append((ld_a, n))
            acc_resc_w.append((acc_r, n))
            ld_resc_w.append((ld_r, n))

        acc_base = aggregate(acc_base_w)
        ld_base = aggregate(ld_base_w)
        acc_abl = aggregate(acc_abl_w)
        ld_abl = aggregate(ld_abl_w)
        acc_resc = aggregate(acc_resc_w)
        ld_resc = aggregate(ld_resc_w)
        within_1pct = (acc_base - acc_resc) <= 0.01
        results["heads"].append({
            "layer": layer,
            "head": head,
            "acc_base": acc_base,
            "acc_abl": acc_abl,
            "acc_resc": acc_resc,
            "ld_base": ld_base,
            "ld_abl": ld_abl,
            "ld_resc": ld_resc,
            "within_1pct": within_1pct,
        })

    out = Path(args.outdir) / f"causal_rescue_sharpener_{args.model_name.replace('/', '_')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}")
    return results


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-name", type=str, default="gpt2-medium")
    ap.add_argument("--device", type=str, default="mps")
    ap.add_argument("--dtype", type=str, default="float16")
    ap.add_argument("--binder-input", type=Path, required=False)
    ap.add_argument("--sharpener-input", type=Path, required=False)
    ap.add_argument("--binder-top-k", type=int, default=5)
    ap.add_argument("--sharpener-top-k", type=int, default=2)
    ap.add_argument("--max-examples", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--outdir", type=Path, default=Path("reports"))
    # for sharpener facts
    ap.add_argument("--config", type=Path, default=Path("lab/configs/run_h1_cross_condition_balanced.json"))
    ap.add_argument("--tag", type=str, default="facts")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.binder_input:
        run_binder(args)
    if args.sharpener_input:
        run_sharpener(args)


if __name__ == "__main__":
    main()

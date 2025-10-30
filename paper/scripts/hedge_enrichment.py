import json
import math
import pathlib
import random
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
LEXICON = ROOT / "data" / "lexicons" / "hedge_booster.json"


def load_lexicon() -> Dict[str, set]:
    lex = json.loads(LEXICON.read_text())
    return {k: {normalize_token(tok) for tok in v} for k, v in lex.items()}


TOKEN_RE = re.compile(r"[\\W_]+")


def normalize_token(token: str) -> str:
    token = token.strip()
    if not token:
        return ""
    token = TOKEN_RE.sub("", token.lower())
    return token


def tokens_from_report(path: pathlib.Path) -> Dict[Tuple[int, int], List[str]]:
    data = json.loads(path.read_text())
    head_tokens: Dict[Tuple[int, int], List[str]] = {}
    for entry in data["heads"]:
        layer = entry["layer"]
        head = entry["head"]
        tokens = [normalize_token(tok["token"]) for tok in entry["tokens"]["top"]]
        tokens = [tok for tok in tokens if tok]
        head_tokens[(layer, head)] = tokens
    return head_tokens


def aggregate_tokens(files: Iterable[pathlib.Path]) -> Dict[Tuple[int, int], List[str]]:
    agg: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    for file in files:
        per_head = tokens_from_report(file)
        for key, toks in per_head.items():
            agg[key].extend(toks)
    return agg


def log_odds(head_tokens: List[str], background_tokens: List[str], lexicon: set) -> float:
    head_hedge = sum(tok in lexicon for tok in head_tokens)
    head_total = len(head_tokens)
    back_hedge = sum(tok in lexicon for tok in background_tokens)
    back_total = len(background_tokens)
    # Add-0.5 smoothing
    odds_head = (head_hedge + 0.5) / (head_total - head_hedge + 0.5)
    odds_back = (back_hedge + 0.5) / (back_total - back_hedge + 0.5)
    return math.log(odds_head / odds_back)


def auc_for_lexicon(head_tokens: List[str], background_tokens: List[str], lexicon: set, n_samples: int = 5000) -> float:
    positives = [1 if tok in lexicon else 0 for tok in head_tokens]
    negatives = [1 if tok in lexicon else 0 for tok in background_tokens]
    # Score is binary membership; ROC points collapse to two thresholds.
    tp = sum(positives)
    fp = sum(negatives)
    tn = len(negatives) - fp
    fn = len(positives) - tp
    # Handle degenerate cases
    if tp == 0 and fp == 0:
        return 0.5
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    # Single-threshold AUC (rectangle + triangle)
    return 0.5 * (tpr + (1 - fpr))


def main() -> None:
    lex = load_lexicon()

    gpt2_files = [
        REPORTS / "ov_report_facts.json",
        REPORTS / "ov_report_neg.json",
        REPORTS / "ov_report_cf.json",
        REPORTS / "ov_report_logic.json",
    ]
    mistral_files = [
        REPORTS / "ov_report_facts_mistral_heads22_23.json",
        REPORTS / "ov_report_neg_mistral_heads22_23.json",
        REPORTS / "ov_report_cf_mistral_heads22_23.json",
        REPORTS / "ov_report_logic_mistral_heads22_23.json",
    ]

    gpt2_tokens = aggregate_tokens(gpt2_files)
    mistral_tokens = aggregate_tokens(mistral_files)

    # Build background pools
    def background_pool(tokens: Dict[Tuple[int, int], List[str]], exclude: Tuple[int, int]) -> List[str]:
        pool = []
        for key, vals in tokens.items():
            if key != exclude:
                pool.extend(vals)
        return pool

    results = []
    for head in [(0, 2), (0, 4), (0, 7)]:
        head_toks = gpt2_tokens[head]
        back = background_pool(gpt2_tokens, head)
        results.append(("gpt2", head, "hedge", log_odds(head_toks, back, lex["hedges"]), auc_for_lexicon(head_toks, back, lex["hedges"])))
        results.append(("gpt2", head, "booster", log_odds(head_toks, back, lex["boosters"]), auc_for_lexicon(head_toks, back, lex["boosters"])))

    for head in [(0, 22), (0, 23)]:
        head_toks = mistral_tokens[head]
        back = background_pool(mistral_tokens, head)
        results.append(("mistral", head, "hedge", log_odds(head_toks, back, lex["hedges"]), auc_for_lexicon(head_toks, back, lex["hedges"])))
        results.append(("mistral", head, "booster", log_odds(head_toks, back, lex["boosters"]), auc_for_lexicon(head_toks, back, lex["boosters"])))

    out = ROOT / "paper" / "supplement" / "hedge_enrichment.json"
    structured = []
    for model, head, lex_name, logodds, auc in results:
        structured.append(
            {
                "model": model,
                "layer": head[0],
                "head": head[1],
                "lexicon": lex_name,
                "log_odds": logodds,
                "auc": auc,
                "token_count": len(gpt2_tokens[head] if model == "gpt2" else mistral_tokens[head])
            }
        )
    out.write_text(json.dumps(structured, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

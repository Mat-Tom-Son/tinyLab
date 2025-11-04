import json
import pathlib
import re
import unicodedata

ROOT = pathlib.Path(__file__).resolve().parents[2]
REPORT_GPT2 = ROOT / "reports" / "ov_report_facts.json"
REPORT_MISTRAL = ROOT / "reports" / "ov_report_facts_mistral_heads22_23.json"
LEXICON = ROOT / "data" / "lexicons" / "hedge_booster.json"

TOKEN_RE = re.compile(r"[\W_]+")

LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(text: str) -> str:
    return "".join(LATEX_ESCAPES.get(ch, ch) for ch in text)


def normalize(token: str) -> str:
    token = token.strip()
    if not token:
        return ""
    token = unicodedata.normalize("NFKD", token)
    token = token.encode("ascii", "ignore").decode()
    return TOKEN_RE.sub("", token.lower())


# Lexicon is no longer used for badges in the table, but keep
# normalization utilities for any future cross-references.
if LEXICON.exists():
    lex_data = json.loads(LEXICON.read_text())
    lex_sets = {k: {normalize(tok) for tok in v} for k, v in lex_data.items()}
else:
    lex_sets = {"hedges": set(), "boosters": set()}


def badge_for(norm: str) -> str:
    # Badges removed from display in the modernized table format.
    return ""


def load_report(path: pathlib.Path):
    data = json.loads(path.read_text())
    heads = {}
    for entry in data["heads"]:
        head_id = (entry["layer"], entry["head"])
        heads[head_id] = {
            "top": entry["tokens"]["top"],
            "bottom": entry["tokens"]["bottom"],
        }
    return heads


heads_gpt2 = load_report(REPORT_GPT2)
heads_mistral = load_report(REPORT_MISTRAL)

order = [
    ("GPT-2 Medium", (0, 2)),
    ("GPT-2 Medium", (0, 4)),
    ("GPT-2 Medium", (0, 7)),
    ("Mistral 7B", (0, 22)),
    ("Mistral 7B", (0, 23)),
]


def meaningful_tokens(entries):
    selected = []
    for item in entries:
        raw = item["token"]
        norm = normalize(raw)
        if norm or any(ch.isalpha() for ch in raw):
            selected.append(item)
        if len(selected) == 5:
            break
    while len(selected) < 5:
        selected.append({"token": "", "logit": 0.0})
    return selected


def bpe_display(raw: str) -> str:
    """Render a raw BPE token for LaTeX with a visible leading-space marker.

    Rules:
    - If the token starts with a literal space, replace that with the marker '·'.
    - Display lowercase for consistency.
    - Escape LaTeX special chars and wrap in \texttt{...}.
    """
    if not raw:
        return "\\texttt{---}"
    leading_space = raw.startswith(" ")
    core = raw[1:] if leading_space else raw
    core = core.lower()
    core = latex_escape(core)
    # Prefer LaTeX-visible leading-space marker for portability.
    disp = ("\\textperiodcentered " + core) if leading_space else core
    return f"\\texttt{{{disp}}}"


def _score_fmt(val: float) -> str:
    # Include explicit sign and 3 decimals
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.3f}"


def render_tables() -> str:
    out_lines = []
    for model, head in order:
        report = heads_gpt2 if model.startswith("GPT-2") else heads_mistral
        tokens = {
            "top": meaningful_tokens(report[head]["top"]),
            "bottom": meaningful_tokens(report[head]["bottom"]),
        }
        label = f"{model}, head {head[0]}:{head[1]}"
        short_label = f"{('gpt2' if model.startswith('GPT-2') else 'mistral')}-{head[0]}{head[1]}"
        # Column width tuning: keep total at ~0.90\linewidth so the table
        # stays visually compact and centered with comfortable margins.
        token_w = 0.68
        score_w = 0.22
        table_w = token_w + score_w  # 0.90

        latex = [
            r"\begin{table}[t]",
            r"\centering",
            fr"\caption{{\textbf{{Representative OV tokens for {label}.}} Top/bottom-5 tokens by OV score ($v_{{\text{{OV}}}}\cdot E[t]$). A leading \texttt{{\textperiodcentered}} denotes a preceding space.}}",
            fr"\label{{tab:ov-tokens-{short_label}}}",
            r"\small",
            fr"\begin{{tabular}}{{@{{}}p{{{token_w}\linewidth}}p{{{score_w}\linewidth}}@{{}}}}",
            r"\toprule",
            r"\textbf{Token (\texttt{BPE})} & \textbf{OV score} \\",
            r"\midrule",
            r"\textbf{Upweighted (Top-5)} & \\ ",
                    ]
        # Upweighted (strictly sorted)
        top_sorted = sorted(tokens["top"], key=lambda e: e.get("logit", 0.0), reverse=True)[:5]
        for entry in top_sorted:
            raw = entry.get("token", "")
            score = entry.get("logit", 0.0)
            latex.append(f"{bpe_display(raw)} & \\hfill {_score_fmt(score)} \\")
        # Downweighted
        latex.extend(
            [
                r"\midrule",
                r"\textbf{Downweighted (Bottom-5)} & \\ ",
                            ]
        )
        bottom_sorted = sorted(tokens["bottom"], key=lambda e: e.get("logit", 0.0))[:5]
        for entry in bottom_sorted:
            raw = entry.get("token", "")
            score = entry.get("logit", 0.0)
            latex.append(f"{bpe_display(raw)} & \\hfill {_score_fmt(score)} \\")
        latex.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\vspace{2pt}",
                fr"\begin{{minipage}}{{{table_w}\linewidth}}\raggedright \footnotesize Notes: Scores are OV--embedding dot products for the specified head, averaged over frequency-matched resamples. \texttt{{\textperiodcentered}} marks leading space; tokens are lowercased for display.\end{{minipage}}",
                r"\end{table}",
            ]
        )
        out_lines.extend(latex)
        out_lines.append("")
    return "\n".join(out_lines)


def main() -> None:
    output = ROOT / "paper" / "generated" / "token_tables.tex"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_tables())
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

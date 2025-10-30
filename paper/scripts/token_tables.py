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


lex_data = json.loads(LEXICON.read_text())
lex_sets = {k: {normalize(tok) for tok in v} for k, v in lex_data.items()}


def badge_for(norm: str) -> str:
    badges = []
    if norm in lex_sets["hedges"]:
        badges.append("H")
    if norm in lex_sets["boosters"]:
        badges.append("B")
    return "".join(badges)


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


def render_tables() -> str:
    out_lines = []
    for model, head in order:
        report = heads_gpt2 if model.startswith("GPT-2") else heads_mistral
        tokens = {
            "top": meaningful_tokens(report[head]["top"]),
            "bottom": meaningful_tokens(report[head]["bottom"]),
        }
        label = f"{model} head {head[0]}:{head[1]}"
        latex = [
            r"\begin{table}[t]",
            r"    \centering",
            f"    \\caption{{Representative OV tokens for {label} (top/bottom five).}}",
            f"    \\label{{tab:tokens-{model.split()[0].lower()}-{head[0]}{head[1]}}}",
            r"    \begin{tabular}{@{}p{0.45\textwidth}p{0.45\textwidth}@{}}",
            r"        \toprule",
            r"        Raw BPE & Normalised word \\",
            r"        \midrule",
            r"        \multicolumn{2}{@{}l}{\textbf{Upweighted}} \\",
            r"        \midrule",
        ]
        for entry in tokens["top"]:
            raw = entry["token"]
            norm = normalize(raw)
            badge = badge_for(norm)
            raw_display = latex_escape(raw) if raw else "---"
            if badge:
                raw_display += f"\\textsuperscript{{{badge}}}"
            latex.append(f"        {raw_display} & {norm or '---'} \\\\")
        latex.extend(
            [
                r"        \midrule",
                r"        \multicolumn{2}{@{}l}{\textbf{Downweighted}} \\",
                r"        \midrule",
            ]
        )
        for entry in tokens["bottom"]:
            raw = entry["token"]
            norm = normalize(raw)
            badge = badge_for(norm)
            raw_display = latex_escape(raw) if raw else "---"
            if badge:
                raw_display += f"\\textsuperscript{{{badge}}}"
            latex.append(f"        {raw_display} & {norm or '---'} \\\\")
        latex.extend(
            [
                r"        \bottomrule",
                r"    \end{tabular}",
                r"    \footnotesize{Top-$K$ tokens selected after frequency-matched resampling; see Section~\ref{sec:methods}.}",
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

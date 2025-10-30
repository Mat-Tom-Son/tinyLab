import argparse
import json
from pathlib import Path

def load(path):
    return json.loads(Path(path).read_text())

METRICS = ["logit_diff", "acc_flip_rate", "p_drop", "kl_div"]

def summarise(paths):
    rows = []
    for path in paths:
        data = load(path)
        name = Path(path).stem
        baseline = data["baseline"]
        zero = data["zero"]
        patched = data["patched"]
        rows.append({
            "name": name,
            **{f"baseline_{m}": baseline.get(m) for m in METRICS},
            **{f"zero_{m}": zero.get(m) for m in METRICS},
            **{f"patched_{m}": patched.get(m) for m in METRICS},
        })
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = summarise(args.paths)
    args.output.write_text(json.dumps(rows, indent=2))

if __name__ == "__main__":
    main()

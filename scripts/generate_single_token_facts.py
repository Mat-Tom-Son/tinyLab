#!/usr/bin/env python3
"""Generate a synthetic single-token factual dataset for smoke testing.

The dataset is intentionally small, template-driven, and reproducible. Each
example contains `clean`, `corrupt`, `target`, and `foil` fields tailored to
activation patching workflows. Targets and foils are validated to ensure they
occupy a single tokenizer slot (GPT-2 by default).
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


RAW_EXAMPLES = [
    # Colors
    ("The color of ripe strawberries is", "The color of the clear daytime sky is", " red", " blue"),
    ("The color of healthy grass is", "The color of charcoal is", " green", " black"),
    ("The color of a road warning sign is", "The color of fresh snow is", " yellow", " white"),
    ("The color of a cooked pumpkin is", "The color of printer paper is", " orange", " white"),
    ("The color of a crow's feathers is", "The color of cotton candy is", " black", " pink"),
    ("The color of roasted coffee beans is", "The color of chalk dust is", " brown", " white"),
    ("The color of a stop sign is", "The color of the deep ocean is", " red", " blue"),
    ("The color of fresh spinach leaves is", "The color of midnight clouds is", " green", " gray"),
    ("The color of a lemon peel is", "The color of a lime peel is", " yellow", " green"),
    ("The color of a Halloween jack-o'-lantern is", "The color of a polar bear's fur is", " orange", " white"),
    # Animals
    ("The animal that barks loudly is the", "The animal that purrs softly is the", " dog", " cat"),
    ("The animal that howls at the moon is the", "The animal that hoots from tree branches is the", " wolf", " owl"),
    ("The animal that roars on the savannah is the", "The animal that swims with flippers is the", " lion", " seal"),
    ("The animal that stalks prey silently is the", "The animal that grazes in herds is the", " fox", " sheep"),
    ("The animal that hunts with sharp teeth in the ocean is the", "The animal that glides on ocean waves is the", " shark", " whale"),
    ("The animal that carries its home on its back is the", "The animal that sheds its skin is the", " turtle", " snake"),
    ("The animal that soars with wide wings is the", "The animal that swims in icy seas is the", " eagle", " seal"),
    ("The animal that quacks in the pond is the", "The animal that coos in city squares is the", " duck", " dove"),
    ("The animal that chews on carrots in burrows is the", "The animal that squeaks in walls is the", " rabbit", " mouse"),
    ("The animal that provides wool for clothing is the", "The animal that provides milk on farms is the", " sheep", " cow"),
    # Food
    ("The fruit that keeps the doctor away is the", "The fruit that grows in tropical bunches is the", " apple", " banana"),
    ("The fruit that is purple and grows in clusters is the", "The fruit that is orange and fuzzy is the", " grape", " peach"),
    ("The fruit that is tart and yellow is the", "The fruit that is tart and green is the", " lemon", " lime"),
    ("The fruit that has a single pit and purple skin is the", "The fruit that has a single pit and golden skin is the", " plum", " peach"),
    ("The fruit that flavors classic pies is the", "The fruit that flavors summer smoothies is the", " apple", " mango"),
    ("The spice that flavors gingerbread is", "The spice that flavors mint tea is", " ginger", " mint"),
    ("The vegetable that makes you cry when chopped is the", "The vegetable that grows underground in orange rows is the", " onion", " carrot"),
    ("The vegetable that forms purple bulbs is the", "The vegetable that forms leafy heads is the", " garlic", " lettuce"),
    ("The ingredient that sweetens desserts is", "The ingredient that thickens bread dough is", " sugar", " flour"),
    ("The ingredient that bubbles bread dough is", "The ingredient that flavors chocolate cake is", " yeast", " cocoa"),
    # Fantasy & roles
    ("The mythical creature that breathes fire is the", "The mythical creature that disappears at dawn is the", " dragon", " ghost"),
    ("The adventurer that casts spells is the", "The adventurer that swings a sword is the", " wizard", " knight"),
    ("The traveler that sails the high seas is the", "The traveler that sneaks through shadows is the", " pirate", " ninja"),
    ("The hero that fights the undead is the", "The hero that guides spirits is the", " cleric", " shaman"),
    ("The monster that feeds on brains is the", "The monster that rattles chains in halls is the", " zombie", " ghost"),
    ("The visitor that descends from the stars is the", "The visitor that rises from old graves is the", " alien", " zombie"),
    ("The guardian that spreads white wings is the", "The guardian that brandishes a flaming blade is the", " angel", " knight"),
    ("The warrior that charges in shining armor is the", "The warrior that lurks with daggers is the", " knight", " rogue"),
    ("The machine that obeys coded commands is the", "The machine that blends metal and flesh is the", " robot", " android"),
    ("The scout that tracks in the forest is the", "The scholar that studies ancient tomes is the", " ranger", " sage"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a single-token facts dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lab/data/corpora/facts_single_token_v1.jsonl"),
        help="Where to write the corpus (default: lab/data/corpora/facts_single_token_v1.jsonl).",
    )
    parser.add_argument(
        "--tokenizer",
        default="gpt2",
        help="Tokenizer to validate single-token constraints (default: gpt2).",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    def is_single_token(text: str) -> bool:
        return len(tokenizer.encode(text, add_special_tokens=False)) == 1

    kept = []
    skipped = []
    for clean, corrupt, target, foil in RAW_EXAMPLES:
        if is_single_token(target) and is_single_token(foil):
            kept.append(
                {
                    "clean": clean,
                    "corrupt": corrupt,
                    "target": target,
                    "foil": foil,
                }
            )
        else:
            skipped.append((target, foil))

    if not kept:
        raise SystemExit("No valid examples remain after tokenizer filtering.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for row in kept:
            f.write(json.dumps(row))
            f.write("\n")

    print(f"Wrote {len(kept)} examples to {args.output}")
    if skipped:
        print("Skipped examples due to multi-token targets/foils:")
        for target, foil in skipped:
            print(f"  target={target!r}, foil={foil!r}")


if __name__ == "__main__":
    main()

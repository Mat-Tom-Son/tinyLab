#!/usr/bin/env python3
"""Convert MPS/CPU configs to CUDA with optional memory optimizations.

Usage:
    # Convert a single config
    python scripts/convert_configs_to_cuda.py lab/configs/run_h1_cross_condition_balanced.json

    # Convert all configs in a directory
    python scripts/convert_configs_to_cuda.py lab/configs/*.json

    # Convert with low-memory optimizations
    python scripts/convert_configs_to_cuda.py lab/configs/*.json --low-memory

    # Convert and reduce batch size
    python scripts/convert_configs_to_cuda.py lab/configs/*.json --batch-size 4

    # Dry run (show changes without writing)
    python scripts/convert_configs_to_cuda.py lab/configs/*.json --dry-run
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


def convert_config_to_cuda(
    config: Dict[str, Any],
    low_memory: bool = False,
    batch_size: int = None,
    dtype: str = None,
) -> Dict[str, Any]:
    """Convert a config to use CUDA with optional optimizations.

    Args:
        config: The config dict to convert
        low_memory: Enable low-memory mode
        batch_size: Override batch size
        dtype: Override dtype (e.g., "float16", "bfloat16")

    Returns:
        Converted config dict
    """
    # Deep copy to avoid modifying original
    config = json.loads(json.dumps(config))

    # Update top-level device if present
    if "device" in config:
        old_device = config["device"]
        config["device"] = "cuda"
        print(f"  Updated device: {old_device} → cuda")

    # Update shared config if present (multi-condition runs)
    if "shared" in config:
        if "device" in config["shared"]:
            old_device = config["shared"]["device"]
            config["shared"]["device"] = "cuda"
            print(f"  Updated shared.device: {old_device} → cuda")

        # Update model settings
        if "model" in config["shared"]:
            model_cfg = config["shared"]["model"]

            # Set dtype to float16 for CUDA efficiency
            if dtype:
                old_dtype = model_cfg.get("dtype", "float32")
                model_cfg["dtype"] = dtype
                print(f"  Updated dtype: {old_dtype} → {dtype}")
            elif model_cfg.get("dtype") == "float32":
                model_cfg["dtype"] = "float16"
                print(f"  Updated dtype: float32 → float16 (CUDA optimization)")

            # Enable low memory mode if requested
            if low_memory:
                model_cfg["low_memory"] = True
                print(f"  Enabled low_memory mode")

        # Update batch size if requested
        if batch_size is not None and "batch_size" in config["shared"]:
            old_batch = config["shared"]["batch_size"]
            config["shared"]["batch_size"] = batch_size
            print(f"  Updated batch_size: {old_batch} → {batch_size}")

    # Update model config at top level if present (single-run configs)
    if "model" in config:
        model_cfg = config["model"]

        # Set dtype to float16 for CUDA efficiency
        if dtype:
            old_dtype = model_cfg.get("dtype", "float32")
            model_cfg["dtype"] = dtype
            print(f"  Updated dtype: {old_dtype} → {dtype}")
        elif model_cfg.get("dtype") == "float32":
            model_cfg["dtype"] = "float16"
            print(f"  Updated dtype: float32 → float16 (CUDA optimization)")

        # Enable low memory mode if requested
        if low_memory:
            model_cfg["low_memory"] = True
            print(f"  Enabled low_memory mode")

    # Update batch size at top level if present
    if batch_size is not None and "batch_size" in config:
        old_batch = config["batch_size"]
        config["batch_size"] = batch_size
        print(f"  Updated batch_size: {old_batch} → {batch_size}")

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Convert MPS/CPU configs to CUDA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "configs",
        nargs="+",
        type=Path,
        help="Config files to convert (supports wildcards)",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Enable low-memory mode for limited VRAM",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size (e.g., 4 for limited VRAM)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        help="Override dtype (default: auto-convert float32→float16)",
    )
    parser.add_argument(
        "--suffix",
        default="_cuda",
        help="Suffix to add to output filename (default: '_cuda')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without writing files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite original files instead of creating new ones",
    )

    args = parser.parse_args()

    # Expand wildcards and filter to existing files
    config_files = []
    for pattern in args.configs:
        if pattern.exists() and pattern.is_file():
            config_files.append(pattern)
        else:
            # Try glob pattern
            matching = list(pattern.parent.glob(pattern.name))
            config_files.extend([f for f in matching if f.is_file()])

    if not config_files:
        print("No config files found!")
        sys.exit(1)

    print(f"Converting {len(config_files)} config(s) to CUDA...\n")

    for config_file in config_files:
        print(f"Processing: {config_file}")

        try:
            # Load config
            with open(config_file) as f:
                config = json.load(f)

            # Convert
            converted = convert_config_to_cuda(
                config,
                low_memory=args.low_memory,
                batch_size=args.batch_size,
                dtype=args.dtype,
            )

            # Determine output path
            if args.overwrite:
                output_file = config_file
            else:
                stem = config_file.stem
                # Remove existing suffix if re-converting
                if stem.endswith("_cuda") or stem.endswith("_mps"):
                    stem = stem.rsplit("_", 1)[0]
                output_file = config_file.parent / f"{stem}{args.suffix}.json"

            # Write or dry-run
            if args.dry_run:
                print(f"  [DRY RUN] Would write to: {output_file}")
            else:
                with open(output_file, "w") as f:
                    json.dump(converted, f, indent=2)
                print(f"  ✓ Wrote: {output_file}")

        except Exception as e:
            print(f"  ✗ Error: {e}")

        print()

    if args.dry_run:
        print("Dry run complete. Use without --dry-run to write files.")
    else:
        print(f"✓ Converted {len(config_files)} config(s)")


if __name__ == "__main__":
    main()

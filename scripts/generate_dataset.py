"""Phase 8 driver — generate a DecisionDataset and print its analysis report.

Usage:
    uv run python scripts/generate_dataset.py runs/dataset.jsonl
    uv run python scripts/generate_dataset.py runs/dataset.jsonl --quick
    uv run python scripts/generate_dataset.py runs/dataset.jsonl --force

Per codex (gpt-5.6-sol) review 2025-09-23:
  - Atomic write: write to .tmp, then rename on success.
  - --quick: minimal spec for fast smoke testing.
  - --force: overwrite existing file.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.dataset import DatasetConfig, DatasetGenerator, analyze_dataset, write_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--quick", action="store_true",
                        help="Use a minimal spec for fast smoke testing.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output file.")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        print(f"error: {args.output} exists; pass --force to replace it", file=sys.stderr)
        return 1

    spec = DatasetConfig()
    if args.quick:
        spec = replace(
            spec,
            problems=("Branin",),
            seeds=(0,),
            trajectories_per_problem_seed=1,
            max_actions=4,
        )

    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    try:
        write_jsonl(DatasetGenerator(spec).generate(), temporary)
        temporary.replace(args.output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    print(json.dumps(asdict(analyze_dataset(args.output)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

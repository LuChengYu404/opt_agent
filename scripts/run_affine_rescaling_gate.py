"""Track B — EXP-RQ0.5 Gate 5: affine rescaling test runner.

Runs `affine_rescaling_test` from `opt_agent.dataset.validation` over 4
transforms `(scale, shift)` of the 5-family BOCoDe panel. Produces a
markdown report and a raw JSON dump.

Per codex §27, this is the final missing gate for RQ1 robustness. Pass
criteria per (problem, phase) bucket:
  - Spearman ρ ≥ 0.9
  - top-1 agreement ≥ 0.8
  - JS divergence ≤ 0.02

Outputs:
    runs/EXP_AFFINE_REPORT.md
    runs/exp_rq0_5_affine_report.json
    runs/phase8_6d_h3_5family_affine/  (per-transform JSONL datasets)

Usage:
    uv run python scripts/run_affine_rescaling_gate.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.dataset import DatasetConfig, DatasetGenerator, affine_rescaling_test
from opt_agent.dataset.decision_record import _json_safe

# Frozen transforms (scale, shift). Identity is (1.0, 0.0); baseline for
# the test. Three non-identity transforms exercise scale-down, scale-up,
# and shift-up. Each is rank-equivalent to identity (affine maps preserve
# ordering), so the test should pass with Spearman=1.0, top-1=1.0, JS=0.0
# modulo floating-point noise from GP fits (per the analytic argument:
# Δopt = scale*(new-old) and optimization_scale = scale*std(raw), so the
# ratio is invariant).
TRANSFORMS: tuple[tuple[float, float], ...] = (
    (1.0, 0.0),
    (0.1, 0.0),
    (10.0, 0.0),
    (1.0, 5.0),
)


def _panel_config(seeds: tuple[int, ...]) -> DatasetConfig:
    """5-family canonical panel, matching the 10-seed/20-seed canonical
    runs (`scripts/generate_dataset_5family.py`)."""
    return replace(
        DatasetConfig(),
        problems=("Branin", "Hartmann", "Ackley", "SixHumpCamel", "Rastrigin"),
        seeds=seeds,
        horizon=3,
        trajectories_per_problem_seed=1,
        max_actions=25,
    )


def generate_for_transform(
    scale: float, shift: float, panel_seeds: tuple[int, ...]
):
    """Closure passed into `affine_rescaling_test`.

    Each call instantiates a fresh `DatasetGenerator` with the given
    transform, generating the full panel. Because BOCoDe problems and
    seeds are deterministic given the same RNG, the resulting record IDs
    match across transforms — which is what `affine_rescaling_test`
    requires for paired comparison.
    """
    cfg = _panel_config(panel_seeds)
    cfg = replace(
        cfg, objective_scale=float(scale), objective_shift=float(shift)
    )
    return list(DatasetGenerator(cfg).generate())


def write_jsonl(records, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(_json_safe(r), allow_nan=False, sort_keys=True, separators=(",", ":")))
            f.write("\n")
            n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        default=list(range(5)),
        help="Seed tuple for the panel (default 0..4 — fast smoke). "
             "For the canonical run use --seeds 0 1 2 3 4 5 6 7 8 9 (10 seeds, ~12 min).",
    )
    parser.add_argument(
        "--out-md", type=Path,
        default=ROOT / "runs" / "EXP_AFFINE_REPORT.md",
    )
    parser.add_argument(
        "--out-json", type=Path,
        default=ROOT / "runs" / "exp_rq0_5_affine_report.json",
    )
    parser.add_argument(
        "--datasets-dir", type=Path,
        default=ROOT / "runs" / "phase8_6d_h3_5family_affine",
    )
    parser.add_argument(
        "--keep-datasets", action="store_true",
        help="Persist per-transform datasets to --datasets-dir (large; default off).",
    )
    args = parser.parse_args()

    panel_seeds: tuple[int, ...] = tuple(args.seeds)
    print(f"Panel: {len(panel_seeds)} seeds × 5 families × horizon=3")
    print(f"Transforms: {TRANSFORMS}")

    # Cache each transform's dataset (records are deterministic, so the
    # closure approach in affine_rescaling_test works correctly).
    t_total = time.time()
    cached: dict[tuple[float, float], list] = {}
    for transform in TRANSFORMS:
        t = time.time()
        scale, shift = transform
        print(f"\n=== Generating (scale={scale}, shift={shift}) ... ===")
        records = generate_for_transform(scale, shift, panel_seeds)
        cached[transform] = records
        print(f"  {len(records)} records in {time.time() - t:.1f}s")

        if args.keep_datasets:
            out_path = args.datasets_dir / f"scale{scale}_shift{shift}.jsonl"
            n = write_jsonl(records, out_path)
            print(f"  Wrote {out_path} ({n} records)")

    def gen_factory(scale: float, shift: float):
        return cached[(scale, shift)]

    # Run the actual gate
    print("\n=== Running affine_rescaling_test ===")
    t = time.time()
    result = affine_rescaling_test(gen_factory)
    elapsed = time.time() - t
    print(f"  Done in {elapsed:.1f}s")

    # Persist
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "transforms": [list(t) for t in TRANSFORMS],
                "n_seeds": len(panel_seeds),
                "result": result,
                "wall_clock_seconds": elapsed,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"Wrote {args.out_json}")

    # Markdown report
    md: list[str] = []
    md.append("# EXP-RQ0.5 — Gate 5: Affine Rescaling Test\n")
    md.append("**Date**: 2026-09-27")
    md.append(f"**Status**: {'PASS ✓' if result['pass_'] else 'FAIL ✗'}\n")
    md.append(f"**Panel**: 5 families × {len(panel_seeds)} seeds × horizon=3")
    md.append(f"**Transforms tested**: {TRANSFORMS}")
    md.append(f"**Wall-clock**: {elapsed:.1f}s (transform gen + gate)")
    md.append("")

    md.append("## 1. Pass criterion (per codex §27)\n")
    md.append("Per (problem, phase) bucket:")
    md.append("- Spearman ρ ≥ 0.9 (rank correlation of gold_probs vectors)")
    md.append("- top-1 agreement ≥ 0.8 (fraction of records where argmax matches identity)")
    md.append("- JS divergence ≤ 0.02 (Jensen-Shannon between softmax distributions)")
    md.append("")

    md.append("## 2. Per-(problem, phase) summary\n")
    md.append("| problem | phase | n | median Spearman | top-1 agreement | mean JS | pass |")
    md.append("|---|---|---|---|---|---|---|")
    rows = result.get("by_problem_phase", [])
    for x in rows:
        ok = (
            x["median_spearman"] >= 0.9
            and x["top1_agreement"] >= 0.8
            and x["mean_js"] <= 0.02
        )
        md.append(
            f"| {x['problem']} | {x['phase']} | {x['n']} | "
            f"{x['median_spearman']:.4f} | {x['top1_agreement']:.4f} | "
            f"{x['mean_js']:.6f} | {'✓' if ok else '✗'} |"
        )
    md.append("")

    md.append("## 3. Interpretation\n")
    if result["pass_"]:
        md.append(
            "All (problem, phase) buckets pass Spearman≥0.9 ∧ top-1≥0.8 ∧ JS≤0.02.\n"
        )
        md.append(
            "**Mechanism**: gold_probs are *exactly* invariant under y → y + shift "
            "because shift cancels in Δopt = (new_best − current_best). For "
            "y → scale * y, Δopt scales by `scale` but so does the `optimization_scale` "
            "denominator (= std(initial_objectives)), so their ratio is preserved. "
            "Both the optimization-dominant and surrogate-inactive utility terms "
            "(default w_surrogate=0) are scale/shift invariant."
        )
    else:
        md.append(
            "**Gate 5 FAIL** — at least one (problem, phase) bucket violated "
            "the Spearman/top-1/JS criterion. Likely causes:\n"
            "1. GP fit non-determinism between runs (scipy_minimize noise); "
            "   action ordering may flip when utilities are very close.\n"
            "2. Numerical precision issues with extreme scales (0.1× or 10×).\n"
        )
    md.append("")

    md.append("## 4. Files\n")
    md.append(f"- `{args.out_json}` — raw gate result")
    if args.keep_datasets:
        md.append(f"- `{args.datasets_dir}/` — per-transform datasets")
    md.append("")

    md.append(f"Total wall-clock (all 4 transforms): {time.time() - t_total:.1f}s")
    args.out_md.write_text("\n".join(md))
    print(f"Wrote {args.out_md}")

    return 0 if result["pass_"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
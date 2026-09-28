"""Generate the INDEPENDENT-seed-stream replication dataset (2026-09-28).

Motivation: workshop_paper_outline.md limitation #5 — the 20-seed
confirmatory dataset reuses seeds 0–9 from the exploratory run; a true
replication requires an independent seed stream. This driver is the frozen
config of scripts/generate_phase8_6d_20seed.py with ONLY the seeds changed
to 100–119 (no record from this stream was seen by any design decision).

Config (frozen, identical to generate_phase8_6d_20seed.py except seeds):
    problems   = ("Branin", "Hartmann", "Ackley", "SixHumpCamel", "Rastrigin")
    seeds      = tuple(range(100, 120))       # independent stream
    horizon    = 3
    trajectories_per_problem_seed = 1
    max_actions = 25

Output: runs/phase8_6d_h3_20seed_5family_s100_119.jsonl

Usage:
    .venv/bin/python scripts/generate_phase8_6d_20seed_replication.py
"""
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path("/home/lcy/Documents/projects/opt_agent_v3")
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.dataset import DatasetConfig, DatasetGenerator
from opt_agent.dataset.decision_record import _json_safe

spec = replace(
    DatasetConfig(),
    problems=("Branin", "Hartmann", "Ackley", "SixHumpCamel", "Rastrigin"),
    seeds=tuple(range(100, 120)),
    horizon=3,
    trajectories_per_problem_seed=1,
    max_actions=25,
)

out = ROOT / "runs" / "phase8_6d_h3_20seed_5family_s100_119.jsonl"
print(f"Generating -> {out}")
print(f"  problems={spec.problems}, seeds={len(spec.seeds)}, horizon={spec.horizon}")

t0 = time.time()
n = 0
tmp = out.with_suffix(out.suffix + ".tmp")
with tmp.open("w", encoding="utf-8", newline="\n") as f:
    for record in DatasetGenerator(spec).generate():
        payload = _json_safe(record)
        f.write(json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")))
        f.write("\n")
        n += 1
        if n % 100 == 0:
            print(f"  ... {n} records in {time.time()-t0:.1f}s", flush=True)

tmp.rename(out)
print(f"Done. {n} records in {time.time()-t0:.1f}s")
print(f"Saved to {out}")

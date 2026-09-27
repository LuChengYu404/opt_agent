"""Generate the 20-seed confirmatory RQ1 dataset.

Companion to scripts/generate_dataset_5family.py (10-seed exploratory).

Config (frozen — do not modify without re-running codex review):
    problems   = ("Branin", "Hartmann", "Ackley", "SixHumpCamel", "Rastrigin")
    seeds      = tuple(range(20))                 # seeds 0-19 (10 confirmatory + 10 exploratory)
    horizon    = 3                                # counterfactual rollout depth
    trajectories_per_problem_seed = 1
    max_actions = 25

Output: runs/phase8_6d_h3_20seed_5family.jsonl
        ~700 records (5 families × 20 seeds × ~7 decision points)
        ~6 minutes generation time on a single CPU

Codex priority #1: replication at 20+ seeds before treating RQ1 as settled.

Usage:
    uv run python scripts/generate_phase8_6d_20seed.py
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

# Frozen config — see module docstring.
spec = replace(
    DatasetConfig(),
    problems=("Branin", "Hartmann", "Ackley", "SixHumpCamel", "Rastrigin"),
    seeds=tuple(range(20)),
    horizon=3,
    trajectories_per_problem_seed=1,
    max_actions=25,
)

out = ROOT / "runs" / "phase8_6d_h3_20seed_5family.jsonl"
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
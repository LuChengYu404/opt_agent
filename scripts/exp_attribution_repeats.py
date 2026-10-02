"""Development-only 32-replica sensitivity; frozen primary arms unchanged."""
from __future__ import annotations
import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from exp_attribution_suite import (
    OUT, ROOT, branch, choices, new_backend, run_tasks, seed_for, extract_state,
    _reconstruct_checkpoint, dataset_info, provenance,
)


def worker(task):
    import warnings
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    record, original = task["record"], task["original"]
    be = new_backend(original["context"])
    be.restore(_reconstruct_checkpoint(record))
    base = be.checkpoint()
    state = extract_state(be, total_budget=30)
    profiles = {}
    for kind in ("rule", "local"):
        replicas = list(original["profiles"][f"{kind}/unit"])
        for rep in range(4, 32):
            stream = seed_for(record.id, rep, 982751)
            values = {}
            for a, first in choices(state, "unit", seed_for(stream, "first")).items():
                values[a] = branch(be, base, first, budget=30, scale=original["scale"], kind=kind, stream=stream)
            replicas.append(values)
        profiles[f"{kind}/unit"] = replicas
    return {"key": task["key"], "id": record.id, "context": original["context"],
            "state": original["state"], "scale": original["scale"], "profiles": profiles}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    records, _ = dataset_info()
    by_id = {r.id: r for r in records}
    rows = [r for task in json.loads((OUT / "labels.json").read_text())["runs"] for r in task["records"]]
    seeds = (0, 3, 7, 11, 15, 19)
    selected = [r for r in rows if r["context"]["seed"] in seeds and r["state"]["resource"]["used_budget"] == 5]
    if len(selected) != 30:
        raise ValueError("replication sensitivity panel incomplete")
    tasks = [{"key": r["id"], "record": by_id[r["id"]], "original": r} for r in selected]
    config = {**provenance(), "replicas": 32, "seeds": seeds,
              "source_labels_sha256": hashlib.sha256((OUT / "labels.json").read_bytes()).hexdigest(),
              "supplement_sha256": hashlib.sha256((ROOT / "docs/ATTRIBUTION_REPLICATION_SUPPLEMENT_20260930.md").read_bytes()).hexdigest(),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    # JSON round-trip ensures tuple/list equality in resume configuration.
    run_tasks(tasks, worker, OUT / "repeat32.json", args.workers, json.loads(json.dumps(config)))


if __name__ == "__main__":
    main()

"""Make existing posterior features available; primary targets unchanged."""
from __future__ import annotations
import argparse
import hashlib
import json
import multiprocessing
import pickle
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from exp_attribution_suite import (
    OUT, ROOT, SAMPLING, _reconstruct_checkpoint, new_backend, seed_for,
    context_key, choices, extract_state, finite_state, rf_features,
    execute_action, dataset_info, run_tasks, provenance, boc_contexts,
)

MODELS = {}


def make_features(task):
    import warnings
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    record, row = task
    be = new_backend(row["context"])
    be.restore(_reconstruct_checkpoint(record))
    torch.manual_seed(seed_for(context_key(row["context"]), len(be.get_history()), "actual"))
    if not be.is_fitted:
        be.fit()
    state = extract_state(be, total_budget=30)
    if not state.surrogate.is_current:
        raise ValueError("posterior features are still unavailable")
    return {"id": row["id"], "family": row["context"]["family"],
            "features": rf_features(finite_state(state), "state"),
            "targets": {a: statistics.fmean(rep[a]["terminal"]["delta"] / row["scale"]
                         for rep in row["profiles"]["rule/unit"]) for a in SAMPLING}}


def run_one(task):
    import warnings
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    c = task["context"]
    be = new_backend(c)
    initial = be.get_history().objectives[:, 0].tolist()
    actions, decisions = [], []
    pre_fits, action_fits = 0, 0
    for decision in range(25):
        n = len(be.get_history())
        stream = seed_for(context_key(c), n, "actual")
        torch.manual_seed(stream)
        if not be.is_fitted:
            be.fit()
            pre_fits += 1
        state = extract_state(be, total_budget=30)
        features = rf_features(finite_state(state), "state")
        available = choices(state, "unit", stream)
        scores = {a: float(MODELS[c["family"]][a].predict([features])[0]) for a in available}
        selected = max(scores, key=scores.get)
        # Reset to the same real-action stream used by the primary arms.
        torch.manual_seed(stream)
        result = execute_action(be, state, available[selected])
        action_fits += result.n_model_fits
        actions.append(selected)
        decisions.append({"n_before": n, "action": selected, "scores": scores,
                          "features": features, "n_fits": result.n_model_fits})
    values = be.get_history().objectives[:, 0].tolist()
    if len(values) != 30:
        raise ValueError("posterior student did not use exactly 30 evaluations")
    incumbent, anytime = max(initial), []
    for value in values[5:]:
        incumbent = max(incumbent, value)
        anytime.append(incumbent)
    return {"key": task["key"], "context": c, "policy": "rf_unit_terminal_posterior",
            "initial_values": initial, "values": values, "terminal_best": max(values),
            "anytime_gain": statistics.fmean(anytime) - max(initial),
            "n_evaluations": 30, "unused_budget": 0, "n_fits": pre_fits + action_fits,
            "prefeature_fits": pre_fits, "planning_evaluations": 0,
            "planning_fits": 0, "actions": actions, "decisions": decisions}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    records, scales = dataset_info()
    by_id = {r.id: r for r in records}
    rows = [r for task in json.loads((OUT / "labels.json").read_text())["runs"] for r in task["records"]]
    feature_path = OUT / "posterior_training_features.json"
    if feature_path.exists():
        features = json.loads(feature_path.read_text())
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("fork")) as pool:
            features = list(pool.map(make_features, [(by_id[r["id"]], r) for r in rows]))
        feature_path.write_text(json.dumps(features, allow_nan=False) + "\n")
    global MODELS
    from sklearn.ensemble import RandomForestRegressor
    for family in scales:
        subset = [r for r in features if r["family"] == family]
        MODELS[family] = {}
        for action in SAMPLING:
            model = RandomForestRegressor(n_estimators=200, max_depth=8,
                                          min_samples_leaf=5, random_state=42, n_jobs=1)
            model.fit([r["features"] for r in subset], [r["targets"][action] for r in subset])
            MODELS[family][action] = model
    with (OUT / "posterior_models.pkl").open("wb") as handle:
        pickle.dump(MODELS, handle)
    config = {**provenance(), "supplement_sha256": hashlib.sha256((ROOT / "docs/ATTRIBUTION_POSTERIOR_SUPPLEMENT_20260930.md").read_bytes()).hexdigest(),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "features_sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
              "labels_sha256": hashlib.sha256((OUT / "labels.json").read_bytes()).hexdigest()}
    tasks = [{"key": f"{context_key(c)}/rf_unit_terminal_posterior", "context": c}
             for c in boc_contexts(range(1000, 1020))]
    run_tasks(tasks, run_one, OUT / "posterior_loops.json", args.workers, config)


if __name__ == "__main__":
    main()

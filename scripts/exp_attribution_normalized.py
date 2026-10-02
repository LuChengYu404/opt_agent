"""Coherent normalized-GP attribution; see the frozen follow-up protocol."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from exp_attribution_suite import (FAMILIES, SAMPLING, SCHEDULES, SnapshotIOHBackend,
                                   choices, context_key, finite_state, run_tasks, seed_for)
from exp_rq1_rq2_closed_loop import PROBLEM_IDS
from opt_agent.actions import execute_action
from opt_agent.backend.bocode_botorch import BackendConfig, BocodeBotorchBackend
from opt_agent.state import extract_state

OUT = ROOT / "runs/attribution_normalized_20260930"
PROTOCOL = ROOT / "docs/ATTRIBUTION_NORMALIZED_FOLLOWUP_20260930.md"


def provenance():
    paths = [Path(__file__).relative_to(ROOT), PROTOCOL.relative_to(ROOT),
             Path("src/opt_agent/backend/bocode_botorch.py"), Path("src/opt_agent/backend/checkpoint.py"),
             Path("src/opt_agent/actions/executor.py"), Path("src/opt_agent/actions/factory.py"),
             Path("src/opt_agent/state/extractor.py"), Path("scripts/exp_attribution_suite.py"),
             Path("scripts/exp_rq1_rq2_closed_loop.py"), Path("scripts/_shared_rq_features.py"),
             Path("scripts/ioh_backend.py")]
    return {"schema": 1, "source_sha256": {str(p): hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
            "packages": {p: importlib.metadata.version(p) for p in ("torch", "botorch", "gpytorch", "scikit-learn", "bocode")}}


def backend(context, treatment):
    be = BocodeBotorchBackend(normalize_inputs=treatment == "normalized")
    be.initialize(BackendConfig(problem_id=PROBLEM_IDS[context["family"]], seed=context["seed"], n_init=5))
    return be


def action_seed(context, n):
    return seed_for(context_key(context), n, "normalized_followup_actual")


def perform(be, action_name, stream):
    state = extract_state(be, total_budget=30)
    torch.manual_seed(stream)
    return execute_action(be, state, choices(state, "unit", stream)[action_name])


def branch(be, cp, *, action, block, continuation, stream, budget=30):
    """One repeated-action block, then the same fixed action to the endpoint."""
    be.restore(cp)
    before = float(be.get_history().objectives.max())
    n0 = len(be.get_history())
    fits = 0
    first = short = None
    for step in range(budget - n0):
        state = extract_state(be, total_budget=budget)
        n = state.sampling.n_samples
        s = seed_for(stream, n, "counterfactual")
        torch.manual_seed(s)
        chosen = action if step < block else continuation
        previous_model = be._model if be.is_fitted else None
        result = execute_action(be, state, choices(state, "unit", s)[chosen])
        assert result.n_new_samples == 1 and not result.should_stop
        fits += result.n_model_fits
        delta = float(be.get_history().objectives.max()) - before
        if step == 0:
            point = be.get_history().X[-1:]
            with torch.no_grad():
                std = float(previous_model.posterior(point).variance.sqrt().item()) if previous_model is not None else None
            first = {"delta": delta, "X": point[0].tolist(), "std": std}
        if step == 2:
            short = {"delta": delta}
    assert len(be.get_history()) == budget
    return {"first": first, "short": short or {"delta": delta},
            "terminal": {"delta": delta}, "n_new": budget - n0, "n_fits": fits}


def planning_scores(be, cp, context, n, block, continuation, replicas=4):
    scores = {a: [] for a in SAMPLING}
    evaluations = fits = 0
    try:
        for rep in range(replicas):
            stream = seed_for(context_key(context), n, rep, "normalized_followup_planning")
            for a in SAMPLING:
                value = branch(be, cp, action=a, block=block, continuation=continuation, stream=stream)
                scores[a].append(value["terminal"]["delta"])
                evaluations += value["n_new"]
                fits += value["n_fits"]
    finally:
        be.restore(cp)
    return {a: statistics.fmean(x) for a, x in scores.items()}, evaluations, fits


def loop_task(task):
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")
    context, policy, treatment = task["context"], task["policy"], task["treatment"]
    be = backend(context, treatment)
    initial = be.get_history().objectives[:, 0].tolist()
    actions, logs = [], []
    fits = base_fits = pe = pf = 0
    block_action = None
    while len(be.get_history()) < 30:
        n = len(be.get_history())
        s = action_seed(context, n)
        scores = None
        if policy.startswith("reference_"):
            block = int(policy.rsplit("_", 1)[1])
            offset = (n - 5) % 5
            if offset == 0:
                torch.manual_seed(s)
                be.fit()
                base_fits += 1
                cp = be.checkpoint()
                scores, extra_e, extra_f = planning_scores(be, cp, context, n, block, task["continuation"])
                pe += extra_e
                pf += extra_f
                block_action = max(scores, key=scores.get)
            chosen = block_action if offset < block else task["continuation"]
        elif policy.startswith("schedule_"):
            chosen = "global_exploration" if n - 5 < 25 * float(policy.split("_")[1]) else "local_exploitation"
        else:
            chosen = policy
        result = perform(be, chosen, s)
        fits += result.n_model_fits
        actions.append(chosen)
        logs.append({"n_before": n, "action": chosen, "planning_scores": scores})
    values = be.get_history().objectives[:, 0].tolist()
    assert len(values) == 30
    return {"key": task["key"], "context": context, "treatment": treatment, "policy": policy,
            "initial_values": initial, "values": values, "terminal_best": max(values),
            "n_evaluations": len(values), "n_fits": fits + base_fits, "prefeature_fits": base_fits,
            "planning_evaluations": pe, "planning_fits": pf, "actions": actions, "decisions": logs}


def selections(rows):
    out = {}
    for fam in FAMILIES:
        group = [r for r in rows if r["context"]["family"] == fam]
        if not group:
            continue
        means = {t: {a: statistics.fmean(r["terminal_best"] for r in group if r["treatment"] == t and r["policy"] == a)
                     for a in SAMPLING} for t in ("raw", "normalized")}
        sched = {f"schedule_{x}": statistics.fmean(r["terminal_best"] for r in group
                 if r["treatment"] == "normalized" and r["policy"] == f"schedule_{x}") for x in SCHEDULES}
        out[fam] = {"fixed": max(means["normalized"], key=means["normalized"].get),
                    "raw_fixed": max(means["raw"], key=means["raw"].get),
                    "schedule": max(sched, key=sched.get), "development_means": means, "schedule_means": sched}
    return out


def label_task(task):
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")
    context, fixed = task["context"], task["continuation"]
    source = backend(context, "normalized")
    records = []
    for n in (5, 15, 25):
        while len(source.get_history()) < n:
            perform(source, fixed, action_seed(context, len(source.get_history())))
        original = source.checkpoint()
        profiles, states = {}, {}
        for treatment in ("raw", "normalized"):
            be = backend(context, treatment)
            be.restore(replace(original, fitted=False, model_state_dict=None, normalize_inputs=treatment == "normalized"))
            torch.manual_seed(seed_for(context_key(context), n, "label_base_fit"))
            be.fit()
            states[treatment] = finite_state(extract_state(be, total_budget=30))
            cp = be.checkpoint()
            for block in (1, 5):
                replicas = []
                for rep in range(task.get("replicas", 8)):
                    stream = seed_for(context_key(context), n, rep, "normalized_followup_labels")
                    values = {a: branch(be, cp, action=a, block=block, continuation=fixed, stream=stream) for a in SAMPLING}
                    replicas.append(values)
                profiles[f"{treatment}/b{block}"] = replicas
        for treatment in ("raw", "normalized"):
            for a in SAMPLING:
                left, right = profiles[f"{treatment}/b1"], profiles[f"{treatment}/b5"]
                assert all(x[a]["first"] == y[a]["first"] for x, y in zip(left, right))
                if a == fixed:
                    assert left == right or all(x[a] == y[a] for x, y in zip(left, right))
        records.append({"id": f"{context_key(context)}/n{n}", "context": context, "n": n,
                        "continuation": fixed, "states": states, "prefeature_fits": 2,
                        "history_X": original.X.tolist(), "history_y": original.objectives.tolist(), "profiles": profiles})
    return {"key": task["key"], "records": records}


class SphereProblem:
    class Meta:
        n_variables = 1
    class Bounds:
        lb = -5.0
        ub = 5.0
    meta_data = Meta()
    bounds = Bounds()
    def __call__(self, x):
        return float(x[0] ** 2)


def positive_control(folder):
    """Independent Monte Carlo verification of a known analytic advantage."""
    from analyze_attribution_suite import intervals
    torch.set_num_threads(1)
    rows = []
    for name, x in (("near", .5), ("far", 3.5)):
        be = SnapshotIOHBackend()
        be.initialize(SphereProblem(), n_init=5, seed=91)
        be._history_X = torch.full((5, 1), x, dtype=torch.double)
        be._history_y = -be._history_X.square()
        cp = be.checkpoint()
        matrix = []
        for rep in range(2304):
            values = []
            for a in ("global_exploration", "local_exploitation"):
                be.restore(cp)
                state = extract_state(be, total_budget=6)
                s = seed_for(rep, "analytic_positive_control")
                execute_action(be, state, choices(state, "unit", s)[a])
                values.append(float(be.get_history().objectives.max()) + x*x)
            matrix.append(values)
        rows.append({"state": name, "x": x, "values": matrix,
                     "analytic": [2*x**3/15, 1/12 if name == "near" else x/2-1/6]})
    import numpy as np
    arrays = [np.asarray(r["values"]) for r in rows]
    choices_by_state = [int(v[:256].mean(axis=0).argmax()) for v in arrays]
    constant = int(np.stack([v[:256].mean(axis=0) for v in arrays]).mean(axis=0).argmax())
    gains = np.mean([v[256:, a]-v[256:, constant] for v, a in zip(arrays, choices_by_state)], axis=0)
    result = {"config": provenance(), "selection_replicas": 256, "evaluation_replicas": 2048,
              "actions": ["global_exploration", "local_exploitation"], "states": rows,
              "selected": choices_by_state, "constant": constant, "analytic_gain": 1/30,
              "heldout_gain": intervals(gains)}
    assert choices_by_state == [1, 0] and constant == 0
    assert result["heldout_gain"]["ci95"][0] > 0
    (folder / "positive_control.json").write_text(json.dumps(result, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("selected", "constant", "analytic_gain", "heldout_gain")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("controls", "labels", "loops", "references", "positive"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    folder = OUT / "smoke" if args.smoke else OUT
    folder.mkdir(parents=True, exist_ok=True)
    config = provenance()
    if args.stage == "positive":
        positive_control(folder)
        return
    families = ("sixhumpcamel",) if args.smoke else FAMILIES
    dev = range(20, 21) if args.smoke else range(20, 30)
    tasks = []
    selected = None
    if args.stage != "controls":
        selected = json.loads((folder / "selection.json").read_text())
        config["selection_sha256"] = hashlib.sha256((folder / "selection.json").read_bytes()).hexdigest()
    if args.stage == "controls":
        for fam in families:
            for s in dev:
                c = {"family": fam, "seed": s}
                for treatment in ("raw", "normalized"):
                    for policy in (*SAMPLING, *([f"schedule_{x}" for x in SCHEDULES] if treatment == "normalized" else [])):
                        tasks.append({"key": f"{context_key(c)}/{treatment}/{policy}", "context": c, "treatment": treatment, "policy": policy})
    elif args.stage == "labels":
        for fam in families:
            for s in (range(20, 21) if args.smoke else range(20, 25)):
                c = {"family": fam, "seed": s}
                tasks.append({"key": context_key(c), "context": c, "continuation": selected[fam]["fixed"], "replicas": 2 if args.smoke else 8})
    else:
        seeds = range(8882, 8883) if args.smoke else range(2000, 2010 if args.stage == "references" else 2020)
        for fam in families:
            for s in seeds:
                c = {"family": fam, "seed": s}
                profiles = ([("normalized", f"reference_{b}") for b in (1, 5)] if args.stage == "references"
                            else [(t, a) for t in ("raw", "normalized") for a in SAMPLING] + [("normalized", selected[fam]["schedule"])])
                for treatment, policy in profiles:
                    tasks.append({"key": f"{context_key(c)}/{treatment}/{policy}", "context": c,
                                  "treatment": treatment, "policy": policy, "continuation": selected[fam]["fixed"]})
    rows = run_tasks(tasks, label_task if args.stage == "labels" else loop_task,
                     folder / f"{args.stage}.json", args.workers, config)
    if args.stage == "controls":
        (folder / "selection.json").write_text(json.dumps(selections(rows), indent=2) + "\n")


if __name__ == "__main__":
    main()

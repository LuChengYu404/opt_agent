"""Run the frozen 2026-09-30 attribution diagnostics; see docs protocol.

Stages are resumable and all writes occur in the parent process. Counterfactual
planning is privileged and its objective evaluations are recorded separately.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import itertools
import json
import multiprocessing
import os
import pickle
import random
import statistics
import sys
import time
import types
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _shared_rq_features import ACTIONS, FAMILIES, compute_family_scales, extract_features, extract_utilities
from exp_rq1_rq2_closed_loop import PROBLEM_IDS, RuleFullBudget
from ioh_backend import IOHBackend
from opt_agent.actions import ActionType, OptimizationAction, execute_action
from opt_agent.actions.factory import candidate_actions_for_state
from opt_agent.backend.bocode_botorch import BackendConfig, BocodeBotorchBackend
from opt_agent.dataset import read_jsonl
from opt_agent.dataset.validation import _reconstruct_checkpoint, phase_of_ratio
from opt_agent.loop.policies import OnPolicyMix
from opt_agent.state import extract_state

OUT = ROOT / "runs/attribution_20260930"
DATASET = ROOT / "runs/phase8_6d_h3_20seed_5family.jsonl"
SAMPLING = tuple(a for a in ACTIONS if a not in ("stop", "retrain_surrogate"))
SCHEDULES = (0.2, 0.4, 0.6, 0.8)
_MODELS = {}
SCHEMA = 1


def seed_for(*parts) -> int:
    raw = json.dumps(parts, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "little") & 0x7fffffff


def finite_state(state) -> dict:
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, float) and not np.isfinite(x):
            return None
        return x
    return clean(asdict(state))


class SnapshotIOHBackend(IOHBackend):
    """Experiment-only snapshots preserve the existing IOH backend semantics."""
    def checkpoint(self):
        return (self._history_X.clone(), self._history_y.clone(), self._fitted,
                copy.deepcopy(getattr(self, "_model", None)), torch.random.get_rng_state().clone())

    def restore(self, cp):
        X, y, fitted, model, rng = cp
        self._history_X, self._history_y = X.clone(), y.clone()
        self._fitted, self._model = fitted, copy.deepcopy(model)
        torch.random.set_rng_state(rng.clone())


def fit_normalized(self):
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Normalize
    from gpytorch.mlls import ExactMarginalLogLikelihood
    history = self.get_history()
    self._model = SingleTaskGP(history.X.clone(), history.objectives.clone(),
                               input_transform=Normalize(d=history.X.shape[1], bounds=self.bounds))
    self._mll = ExactMarginalLogLikelihood(self._model.likelihood, self._model)
    fit_gpytorch_mll(self._mll)
    self._fitted = True


def new_backend(context):
    if "fid" in context:
        import ioh
        p = ioh.get_problem(fid=context["fid"], dimension=context["dim"],
                            instance=context["instance"], problem_class=ioh.ProblemClass.BBOB)
        be = SnapshotIOHBackend()
        be.initialize(p, n_init=5, seed=context["seed"])
    else:
        be = BocodeBotorchBackend()
        be.initialize(BackendConfig(problem_id=PROBLEM_IDS[context["family"]],
                                    n_init=5, seed=context["seed"]))
    if context.get("normalize_gp"):
        be.fit = types.MethodType(fit_normalized, be)
    return be


def choices(state, menu, seed):
    out = candidate_actions_for_state(state)
    if menu in ("unit", "batch"):
        out = {a: v for a, v in out.items() if a in SAMPLING}
    for a, action in list(out.items()):
        params = dict(action.params)
        if menu == "unit" and "n_samples" in params:
            params["n_samples"] = 1
        out[a] = OptimizationAction(action.type, params)
    return out


def continuation_action(state, kind, seed):
    if kind == "local":
        return choices(state, "unit", seed)["local_exploitation"]
    action = RuleFullBudget(seed=seed).select_action(state)
    return action


def composite(delta, n_new, n_fits, scale, budget, is_stop, remaining):
    return delta / scale - n_new / budget - 0.5 * n_fits / budget - (0.75 * remaining / budget if is_stop else 0.0)


def branch(be, base, first, *, budget, scale, kind, stream, max_steps=None):
    """Identical branch prefixes; actual evaluation count controls termination."""
    be.restore(base)
    start = extract_state(be, total_budget=budget)
    before, n0 = start.optimization.best_value, start.sampling.n_samples
    state = start
    n_fits = steps = 0
    prefix = None
    legacy = OnPolicyMix(seed=stream) if kind == "legacy" else None
    action = first
    for j in range(budget + 10):
        if state.resource.remaining_budget <= 0:
            break
        if max_steps is not None and j >= max_steps:
            break
        step_seed = seed_for(stream, state.sampling.n_samples, j)
        torch.manual_seed(step_seed)
        if j:
            action = (legacy.select_action(state, be) if legacy is not None
                      else continuation_action(state, kind, step_seed))
        result = execute_action(be, state, action)
        n_fits += result.n_model_fits
        steps += 1
        state = extract_state(be, total_budget=budget, last_action=action.type.value)
        snapshot = {
            "delta": state.optimization.best_value - before,
            "n_new": state.sampling.n_samples - n0, "n_fits": n_fits,
            "n_evaluations": state.sampling.n_samples, "steps": steps,
            "stopped": result.should_stop,
        }
        snapshot["composite"] = composite(snapshot["delta"], snapshot["n_new"], n_fits,
                                           scale, budget, first.type == ActionType.STOP,
                                           state.resource.remaining_budget)
        if steps == 3 or result.should_stop or state.resource.remaining_budget == 0:
            if prefix is None:
                prefix = dict(snapshot)
        if result.should_stop:
            break
    if steps == 0:
        raise ValueError("empty counterfactual branch")
    if prefix is None:
        prefix = dict(snapshot)
    if kind != "legacy" and max_steps is None and first.type != ActionType.STOP:
        if state.resource.remaining_budget:
            raise ValueError("full-budget continuation did not exhaust evaluations")
    return {"short": prefix, "terminal": snapshot}


def context_key(context):
    return (f"f{context['fid']}/d{context['dim']}/i{context['instance']}/s{context['seed']}"
            if "fid" in context else f"{context['family']}/s{context['seed']}")


def _label_task(task):
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")
    context = task["context"]
    scale = task["scale"]
    if "record" in task:
        record = task["record"]
        be = new_backend(context)
        be.restore(_reconstruct_checkpoint(record))
        state = extract_state(be, total_budget=30)
        if state.sampling.n_samples != record.state["sampling"]["n_samples"]:
            raise ValueError("replay sample mismatch")
        if abs(state.optimization.best_value - record.state["optimization"]["best_value"]) > 1e-9:
            raise ValueError("replay incumbent mismatch")
        if state.surrogate.is_current != record.state["surrogate"]["is_current"]:
            raise ValueError("replay GP mismatch")
        states = [(record.id, be.checkpoint())]
    else:
        be = new_backend(context)
        states = []
        for n in (5, 15, 25):
            while len(be.get_history()) < n:
                state = extract_state(be, total_budget=30)
                execute_action(be, state, continuation_action(state, "rule", seed_for(context_key(context), len(be.get_history()), 71)))
            states.append((f"{context_key(context)}/n{len(be.get_history())}", be.checkpoint()))
    results = []
    for rid, base in states:
        be.restore(base)
        state = extract_state(be, total_budget=30)
        profiles = (("legacy", "original"), ("rule", "original"), ("rule", "unit"), ("local", "unit")) if "record" in task else (("local", "unit"),)
        output = {}
        for kind, menu in profiles:
            replicas = []
            for rep in range(task.get("replicas", 4)):
                # Same streams across menu and horizon controls; continuation
                # change is deliberate. First-action outcomes are paired.
                stream = seed_for(rid, rep, 982751)
                firsts = choices(state, menu, seed_for(stream, "first"))
                values = {}
                for a, first in firsts.items():
                    values[a] = branch(be, base, first, budget=30, scale=scale,
                                       kind=kind, stream=stream,
                                       max_steps=3 if kind == "legacy" else None)
                replicas.append(values)
            output[f"{kind}/{menu}"] = replicas
        be.restore(base)
        results.append({"id": rid, "context": context, "scale": scale,
                        "state": finite_state(state), "profiles": output})
    return {"key": task["key"], "records": results}


def rf_features(state, mode):
    phase = phase_of_ratio(state["resource"]["remaining_budget_ratio"])
    if mode == "time":
        return [state["resource"]["remaining_budget_ratio"],
                *[float(phase == p) for p in ("early", "mid", "late")]]
    return extract_features(state, phase)


def train_students(label_rows, scales):
    from sklearn.ensemble import RandomForestRegressor
    by_family = defaultdict(list)
    for row in label_rows:
        by_family[row["context"]["family"]].append(row)
    models = {}
    for fam, rows in by_family.items():
        models[fam] = {}
        for menu, target, mode in itertools.product(("original", "unit"), ("short", "terminal"), ("state", "time")):
            name = f"rf_{menu}_{target}_{mode}"
            models[fam][name] = {}
            for action in (ACTIONS if menu == "original" else SAMPLING):
                examples = []
                for row in rows:
                    reps = row["profiles"][f"rule/{menu}"]
                    if action not in reps[0]:
                        continue
                    y = statistics.fmean(
                        r[action]["short"]["composite"] if target == "short"
                        else r[action]["terminal"]["delta"] / scales[fam]
                        for r in reps)
                    examples.append((rf_features(row["state"], mode), y))
                if not examples:
                    raise ValueError(f"no training examples for {fam}/{name}/{action}")
                model = RandomForestRegressor(n_estimators=200, max_depth=8,
                                              min_samples_leaf=5, random_state=42, n_jobs=1)
                model.fit([x for x, _ in examples], [y for _, y in examples])
                models[fam][name][action] = model
    # Legacy targets, same within-family fit and full-state features; no new
    # outcome is mixed into this target-control arm.
    records = [r.__dict__ for r in read_jsonl(DATASET, version_policy="warn")]
    for fam in models:
        models[fam]["rf_legacy"] = {}
        for action in ACTIONS:
            examples = []
            for record in records:
                if record["problem"]["family"] != fam:
                    continue
                if record["rollout_results"][action]["status"] != "valid":
                    continue
                u = extract_utilities(record, {f: {"median": v} for f, v in scales.items()})
                examples.append((rf_features(record["state"], "state"), u[action]))
            model = RandomForestRegressor(n_estimators=200, max_depth=8,
                                          min_samples_leaf=5, random_state=42, n_jobs=1)
            model.fit([x for x, _ in examples], [y for _, y in examples])
            models[fam]["rf_legacy"][action] = model
    return models


def oracle_scores(be, state, *, menu, kind, target, scale, context, decision):
    base = be.checkpoint()
    totals = defaultdict(list)
    cost = {"planning_evaluations": 0, "planning_fits": 0}
    try:
        for rep in range(2):
            stream = seed_for(context_key(context), decision, rep, "planning")
            for a, first in choices(state, menu, seed_for(stream, "first")).items():
                result = branch(be, base, first, budget=30, scale=scale, kind=kind,
                                stream=stream, max_steps=3 if target == "short" else None)
                cell = result["short"] if target == "short" else result["terminal"]
                totals[a].append(cell["composite"] if target == "short" else cell["delta"] / scale)
                cost["planning_evaluations"] += result["terminal"]["n_new"]
                cost["planning_fits"] += result["terminal"]["n_fits"]
    finally:
        be.restore(base)
    return {a: statistics.fmean(v) for a, v in totals.items()}, cost


def _loop_task(task):
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")
    context, policy = task["context"], task["policy"]
    be = new_backend({**context, "normalize_gp": policy == "normalized_surrogate"})
    initial = be.get_history().objectives[:, 0].tolist()
    actions, logs = [], []
    cost = {"planning_evaluations": 0, "planning_fits": 0}
    fits = 0
    for decision in range(200):
        state = extract_state(be, total_budget=30)
        if state.resource.remaining_budget == 0:
            break
        actual_seed = seed_for(context_key(context), state.sampling.n_samples, "actual")
        torch.manual_seed(actual_seed)
        menu = "batch" if policy.endswith("_noops") else ("original" if policy == "rf_legacy" or policy.startswith("rf_original") else "unit")
        available = choices(state, menu, actual_seed)
        scores = None
        if policy == "normalized_surrogate":
            chosen = "surrogate_acquisition"
        elif policy in SAMPLING:
            chosen = policy
        elif policy.startswith("schedule"):
            fraction = float(policy.split("_")[1])
            chosen = "global_exploration" if state.resource.used_budget - 5 < 25 * fraction else "local_exploitation"
        elif policy.startswith("rf_"):
            model_name = policy.removesuffix("_noops")
            mode = "time" if model_name.endswith("_time") else "state"
            features = rf_features(finite_state(state), mode)
            scores = {a: float(_MODELS[context["family"]][model_name][a].predict([features])[0]) for a in available}
            chosen = max(scores, key=scores.get)
        elif policy.startswith("oracle_"):
            kind = "rule" if policy.endswith("_rule") else "local"
            target = "short" if "short" in policy else "terminal"
            scores, extra = oracle_scores(be, state, menu="unit", kind=kind,
                                         target=target, scale=task["scale"],
                                         context=context, decision=decision)
            for key in cost:
                cost[key] += extra[key]
            chosen = max(scores, key=scores.get)
        else:
            raise ValueError(policy)
        # Planning must not alter the RNG stream used for the real action.
        torch.manual_seed(actual_seed)
        result = execute_action(be, state, available[chosen])
        fits += result.n_model_fits
        actions.append(chosen)
        logs.append({"n_before": state.sampling.n_samples, "action": chosen,
                     "scores": scores, "n_fits": result.n_model_fits})
        if result.should_stop:
            break
    values = be.get_history().objectives[:, 0].tolist()
    if len(values) > 30:
        raise ValueError("real evaluation budget exceeded")
    if menu in ("unit", "batch") and len(values) != 30:
        raise ValueError("unit policy failed to exhaust real budget")
    incumbent, anytime = max(initial), []
    for value in values[5:]:
        incumbent = max(incumbent, value)
        anytime.append(incumbent)
    anytime.extend([incumbent] * (25 - len(anytime)))
    row = {"key": task["key"], "context": context, "policy": policy,
           "initial_values": initial, "values": values, "terminal_best": max(values),
           "anytime_gain": statistics.fmean(anytime) - max(initial),
           "n_evaluations": len(values), "n_fits": fits, "unused_budget": 30 - len(values),
           "actions": actions, "decisions": logs, **cost}
    if "fid" in context:
        optimum = -float(be.problem.optimum.y)
        denom = optimum - max(initial)
        row.update(optimum=optimum, normalized_regret=(optimum - max(values)) / denom)
    return row


def run_tasks(tasks, worker, path, workers, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text()) if path.exists() else {"config": config, "runs": []}
    if existing["config"] != config:
        raise ValueError(f"resume configuration mismatch at {path}")
    rows = existing["runs"]
    done = {r["key"] for r in rows}
    remaining = [t for t in tasks if t["key"] not in done]
    start = time.perf_counter()
    def save(row):
        rows.append(row)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps({"config": config, "runs": rows}, allow_nan=False) + "\n")
        temp.replace(path)
        if len(rows) % 10 == 0 or len(rows) == len(tasks):
            print(f"{path.stem}: {len(rows)}/{len(tasks)} elapsed={time.perf_counter()-start:.1f}s {row['key']}", flush=True)
    if workers == 1:
        for task in remaining:
            save(worker(task))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork")) as pool:
            futures = [pool.submit(worker, t) for t in remaining]
            for f in as_completed(futures):
                save(f.result())
    return rows


def boc_contexts(seeds):
    return [{"family": fam, "seed": s} for fam in FAMILIES for s in seeds]


def ioh_contexts(instances, seeds):
    return [{"fid": f, "dim": d, "instance": i, "seed": s}
            for f, d, i, s in itertools.product((6, 8, 15), (2, 5, 10), instances, seeds)]


def control_selection(rows):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        c = row["context"]
        cell = f"f{c['fid']}/d{c['dim']}" if "fid" in c else c["family"]
        groups[cell][row["policy"]].append(row["terminal_best"])
    return {cell: {
        "fixed": max((a for a in SAMPLING), key=lambda a: statistics.fmean(v[a])),
        "schedule": max((f"schedule_{f}" for f in SCHEDULES), key=lambda a: statistics.fmean(v[a])),
        "development_means": {a: statistics.fmean(x) for a, x in v.items()},
    } for cell, v in groups.items()}


def dataset_info():
    records = read_jsonl(DATASET, version_policy="warn")
    scales = {f: v["median"] for f, v in compute_family_scales([r.__dict__ for r in records]).items()}
    return records, scales


def provenance():
    return {"schema": SCHEMA, "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "protocol_sha256": hashlib.sha256((ROOT / "docs/ATTRIBUTION_PROTOCOL_20260930.md").read_bytes()).hexdigest(),
            "packages": {p: importlib.metadata.version(p) for p in ("torch", "botorch", "gpytorch", "ioh", "scikit-learn")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("labels", "controls", "fit", "loops", "dim_controls", "dim_labels", "dim_loops"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")
    global OUT, _MODELS
    if args.smoke:
        OUT = OUT / "smoke_v2"
    info = provenance()
    records, scales = dataset_info()
    info["scales"] = scales
    if args.stage == "labels":
        grouped = defaultdict(list)
        for record in records:
            if 0 < record.state["resource"]["remaining_budget"]:
                grouped[(record.problem["family"], record.problem["seed"])].append(record)
        chosen = {}
        for (family, s), group in sorted(grouped.items()):
            if args.smoke and (family != "sixhumpcamel" or s != 0):
                continue
            for n in (5, 15, 25):
                r = min(group, key=lambda r: (abs(r.state["sampling"]["n_samples"] - n), r.id))
                chosen[r.id] = r
        tasks = [{"key": r.id, "record": r, "context": {"family": r.problem["family"], "seed": r.problem["seed"]},
                  "scale": scales[r.problem["family"]], "replicas": 4} for r in chosen.values()]
        run_tasks(tasks, _label_task, OUT / "labels.json", args.workers, info)
    elif args.stage in ("controls", "dim_controls"):
        contexts = boc_contexts(range(20)) if args.stage == "controls" else ioh_contexts(range(1, 4), range(950, 955))
        if args.smoke:
            contexts = contexts[:1] if args.stage == "dim_controls" else [{"family": "sixhumpcamel", "seed": 0}]
        policies = [*SAMPLING, *[f"schedule_{f}" for f in SCHEDULES]]
        tasks = [{"key": f"{context_key(c)}/{p}", "context": c, "policy": p, "scale": scales.get(c.get("family"), 1.0)}
                 for c in contexts for p in policies]
        rows = run_tasks(tasks, _loop_task, OUT / f"{args.stage}.json", args.workers, info)
        (OUT / f"{args.stage}_selection.json").write_text(json.dumps(control_selection(rows), indent=2) + "\n")
    elif args.stage == "fit":
        labels = [r for row in json.loads((OUT / "labels.json").read_text())["runs"] for r in row["records"]]
        models = train_students(labels, scales)
        with (OUT / "models.pkl").open("wb") as handle:
            pickle.dump(models, handle)
        (OUT / "model_provenance.json").write_text(json.dumps({**info, "n_states": len(labels), "training_seeds": sorted({r['context']['seed'] for r in labels}), "policies": list(next(iter(models.values()))), "labels_sha256": hashlib.sha256((OUT / "labels.json").read_bytes()).hexdigest()}, indent=2) + "\n")
        print(f"fit {len(models)} families, {len(labels)} states", flush=True)
    elif args.stage in ("loops", "dim_loops"):
        dimension = args.stage == "dim_loops"
        if not dimension:
            with (OUT / "models.pkl").open("rb") as handle:
                _MODELS = pickle.load(handle)
        selections = json.loads((OUT / ("dim_controls_selection.json" if dimension else "controls_selection.json")).read_text())
        contexts = ioh_contexts(range(10, 13), range(1000, 1005)) if dimension else boc_contexts(range(1000, 1020))
        if args.smoke:
            contexts = [{"fid": 6, "dim": 2, "instance": 99, "seed": 8888}] if dimension else [{"family": "sixhumpcamel", "seed": 8888}]
        tasks = []
        for c in contexts:
            cell = f"f{c['fid']}/d{c['dim']}" if dimension else c["family"]
            policies = [*SAMPLING, selections[cell]["schedule"], "oracle_short_local", "oracle_terminal_local", "normalized_surrogate"]
            if not dimension:
                policies += list(_MODELS[c["family"]])
                policies += ["rf_original_short_state_noops", "rf_original_terminal_state_noops"]
                if c["seed"] < 1005:
                    policies.append("oracle_terminal_rule")
            for p in policies:
                tasks.append({"key": f"{context_key(c)}/{p}", "context": c, "policy": p,
                              "scale": scales.get(c.get("family"), 1.0)})
        info["control_selection_sha256"] = hashlib.sha256(json.dumps(selections, sort_keys=True).encode()).hexdigest()
        if not dimension:
            info["model_sha256"] = hashlib.sha256((OUT / "models.pkl").read_bytes()).hexdigest()
        run_tasks(tasks, _loop_task, OUT / f"{args.stage}.json", args.workers, info)
    else:
        contexts = ioh_contexts(range(10, 13), range(1000, 1005))
        if args.smoke:
            contexts = contexts[:1]
        tasks = [{"key": context_key(c), "context": c, "scale": 1.0, "replicas": 4} for c in contexts]
        run_tasks(tasks, _label_task, OUT / "dim_labels.json", args.workers, info)


if __name__ == "__main__":
    main()

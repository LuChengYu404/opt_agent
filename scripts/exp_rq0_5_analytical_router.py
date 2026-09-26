"""Analytical FrozenRouter — construct the router from the SAME paired arms.

Per codex round-5 review: don't rerun the Router as a separate API call.
Instead, for each (fid, dim, instance, seed), the analytical FrozenRouter's
terminal best_y is:
  - best(LLM trajectory) if dim >= 5
  - best(Surrogate trajectory) else

This eliminates API nondeterminism mixing: the Router outcome is a
deterministic function of the Surrogate and LLM trajectories that were
already run.
"""
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

d = json.load(open("runs/exp_rq0_5_ioh_crossed_fixed.json"))
raw = d["raw"]

# Group by (fid, dim, instance, seed) → policy → best_y.
by_key = defaultdict(dict)
for r in raw:
    key = (r["fid"], r["dim"], r["instance"], r["seed"])
    by_key[key][r["policy"]] = r["best_y"]

# Build analytical FrozenRouter.
analytical_router = []
for key, ps in by_key.items():
    if "LLM" not in ps or "Surrogate" not in ps:
        continue
    fid, dim, inst, seed = key
    if dim >= 5:
        # Routes to LLM.
        analytical_router.append({
            "fid": fid, "dim": dim, "instance": inst, "seed": seed,
            "best_y": ps["LLM"],
            "source_policy": "LLM",
        })
    else:
        # Routes to Surrogate.
        analytical_router.append({
            "fid": fid, "dim": dim, "instance": inst, "seed": seed,
            "best_y": ps["Surrogate"],
            "source_policy": "Surrogate",
        })

print(f"Analytical router entries: {len(analytical_router)}")
print(f"Source distribution:")
from collections import Counter
print(Counter(r["source_policy"] for r in analytical_router))

# Build the analytical router's normalized_regret using same y_opt, y_init.
# We need y_opt and y_init_best per (fid, dim). Recompute.
y_opt_by_fid_dim = {}
y_init_by_fid_dim_inst_seed = {}
for r in raw:
    k = (r["fid"], r["dim"])
    if r.get("y_opt") is not None:
        y_opt_by_fid_dim[k] = r["y_opt"]
    sk = (r["fid"], r["dim"], r["instance"], r["seed"])
    if r.get("y_init_best") is not None:
        existing = y_init_by_fid_dim_inst_seed.get(sk)
        if existing is None:
            y_init_by_fid_dim_inst_seed[sk] = r["y_init_best"]

# Apply normalized regret.
for r in analytical_router:
    y_opt = y_opt_by_fid_dim.get((r["fid"], r["dim"]))
    y_init = y_init_by_fid_dim_inst_seed.get((r["fid"], r["dim"], r["instance"], r["seed"]))
    if y_opt is None or y_init is None or y_init == y_opt:
        r["normalized_regret"] = None
        continue
    nr = (r["best_y"] - y_opt) / (y_init - y_opt)
    r["normalized_regret"] = float(max(0.0, min(1.0, nr)))

valid = [r for r in analytical_router if r.get("normalized_regret") is not None]
print(f"valid analytical router: {len(valid)}")

# Compute mean per dim.
agg = defaultdict(list)
for r in valid:
    agg[r["dim"]].append(r["normalized_regret"])

print()
print("=== Analytical FrozenRouter normalized simple regret ===")
print(f"{'dim':>5s} {'mean':>8s} {'median':>8s} {'n':>4s}")
for d in sorted(agg):
    vals = agg[d]
    print(f"{d:>5d} {statistics.fmean(vals):>8.4f} {statistics.median(vals):>8.4f} {len(vals):>4d}")

# Compare against Surrogate / LLM (using only paired keys).
paired_keys = set(by_key.keys())
sur_results = []
llm_results = []
router_results = valid
for key in paired_keys:
    fid, dim, inst, seed = key
    if "Surrogate" not in by_key[key]:
        continue
    y_init = y_init_by_fid_dim_inst_seed.get((fid, dim, inst, seed))
    y_opt = y_opt_by_fid_dim.get((fid, dim))
    if y_init is None or y_opt is None or y_init == y_opt:
        continue
    def nr(y):
        v = (y - y_opt) / (y_init - y_opt)
        return max(0.0, min(1.0, v))
    sur_results.append(nr(by_key[key]["Surrogate"]))
    if "LLM" in by_key[key]:
        llm_results.append(nr(by_key[key]["LLM"]))

router_normals = [r["normalized_regret"] for r in router_results
                   if r.get("normalized_regret") is not None]
print()
print(f"Surrogate (paired):  n={len(sur_results)}, "
      f"mean={statistics.fmean(sur_results):.4f}, "
      f"median={statistics.median(sur_results):.4f}")
print(f"LLMDeepSeek (paired): n={len(llm_results)}, "
      f"mean={statistics.fmean(llm_results):.4f}, "
      f"median={statistics.median(llm_results):.4f}")
print(f"Analytical Router:   n={len(router_normals)}, "
      f"mean={statistics.fmean(router_normals):.4f}, "
      f"median={statistics.median(router_normals):.4f}")

# Paired differences.
router_by_key = {(r["fid"], r["dim"], r["instance"], r["seed"]): r["normalized_regret"]
                 for r in router_results}
diff_sur = []
diff_llm = []
for key in paired_keys:
    fid, dim, inst, seed = key
    y_init = y_init_by_fid_dim_inst_seed.get((fid, dim, inst, seed))
    y_opt = y_opt_by_fid_dim.get((fid, dim))
    if y_init is None or y_opt is None or y_init == y_opt:
        continue
    def nr(y):
        v = (y - y_opt) / (y_init - y_opt)
        return max(0.0, min(1.0, v))
    router = router_by_key.get(key)
    if router is None:
        continue
    if "Surrogate" in by_key[key]:
        diff_sur.append(router - nr(by_key[key]["Surrogate"]))
    if "LLM" in by_key[key]:
        diff_llm.append(router - nr(by_key[key]["LLM"]))

print()
print("=== Paired differences (analytical router, no API nondeterminism mix) ===")
for name, diffs in [("Router − Surrogate", diff_sur), ("Router − LLM", diff_llm)]:
    n = len(diffs)
    mean = statistics.fmean(diffs)
    se = statistics.stdev(diffs) / math.sqrt(n) if n > 1 else 0
    t = mean / se if se > 0 else float("inf")
    # Two-sided p via Student t with df=n-1 (codex noted normal-approx
    # overstates significance).
    # Use scipy.stats.t if available; otherwise use the formula.
    try:
        from scipy.stats import t as tdist
        p = 2.0 * (1.0 - tdist.cdf(abs(t), df=n-1))
    except ImportError:
        # Approximate with normal (overstates significance per codex).
        p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2))))
    p = min(p, 1.0)
    print(f"  {name:25s}  mean={mean:+.4f}  t={t:+.3f}  p(two-sided)={p:.4f}  n={n}")

# Save analytical router.
out_path = Path("runs/exp_rq0_5_analytical_router.json")
out_path.write_text(json.dumps({
    "config": {
        "router_threshold": 5,
        "construction": "deterministic; uses Surrogate at d<5, LLM at d>=5 from the same paired run",
    },
    "router": analytical_router,
    "summary": {
        "Surrogate mean": statistics.fmean(sur_results),
        "LLM mean": statistics.fmean(llm_results),
        "Router mean": statistics.fmean(router_normals),
        "Router − Surrogate mean": statistics.fmean(diff_sur) if diff_sur else None,
        "Router − LLM mean": statistics.fmean(diff_llm) if diff_llm else None,
    },
}, indent=2) + "\n")
print(f"\nSaved: {out_path}")

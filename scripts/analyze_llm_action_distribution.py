"""Track D — EXP-RQ0.5 per-step LLM action distribution analysis.

Parses the joint 10-seed, 5-policy run (`runs/exp_rq0_5_joint_10seed.json`)
and produces per-(policy, family) action histograms + cross-reference with
closed-loop regret best first actions.

Per-step `steps[].action_id` already exists in the joint run (per
`scripts/exp_rq0_5_full.py` bug-fix §6 priority #2). No new compute.

Outputs:
    runs/exp_rq0_5_action_distribution.md  — markdown report
    runs/exp_rq0_5_action_distribution.json — raw counts

Usage:
    uv run python scripts/analyze_llm_action_distribution.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Canonical action taxonomy (from actions/models.py)
ACTIONS = [
    "global_exploration",
    "local_exploitation",
    "uncertainty_sampling",
    "error_guided_sampling",
    "boundary_sampling",
    "surrogate_acquisition",
    "retrain_surrogate",
    "stop",
]

# Per-family best first-action under AlwaysSurrogate continuation, from
# `runs/EXP_RQ1_3_REPORT.md` §4 (closed-loop terminal regret, 3 seeds).
CLOSED_LOOP_BEST = {
    "Branin": "surrogate_acquisition",      # 0.54; global_exploration 0.52 close
    "Hartmann": "local_exploitation",        # 0.33
    "Ackley": "local_exploitation",          # 0.005
    "SixHumpCamel": "surrogate_acquisition", # 0.12; global 0.31
    "Rastrigin": "local_exploitation",       # 0.00 (vs surrogate 11.10, global 2.33)
}


def _family_order(policies_data: dict) -> list[str]:
    seen: list[str] = []
    for k in policies_data:
        fam = k.split("/", 1)[1]
        if fam not in seen:
            seen.append(fam)
    return seen


def _policy_order(policies_data: dict) -> list[str]:
    seen: list[str] = []
    for k in policies_data:
        pol = k.split("/", 1)[0]
        if pol not in seen:
            seen.append(pol)
    return seen


def _fmt_share(count: int, total: int) -> str:
    if total == 0:
        return "0"
    return f"{100 * count / total:.0f}%"


def _policy_table(rows: list[dict], families: list[str]) -> str:
    """Render an action-share matrix for one policy, families as columns."""
    # rows[family] -> {action -> count}
    lines: list[str] = []
    lines.append("| family | most-common action | share | second action | share | total steps |")
    lines.append("|---|---|---|---|---|---|")
    for fam in families:
        c = rows.get(fam, {})
        if not c:
            lines.append(f"| {fam} | — | 0% | — | 0% | 0 |")
            continue
        total = sum(c.values())
        top = c.most_common(2)
        a1, n1 = top[0]
        if len(top) > 1:
            a2, n2 = top[1]
        else:
            a2, n2 = "—", 0
        lines.append(
            f"| {fam} | `{a1}` | {_fmt_share(n1, total)} | "
            f"`{a2}` | {_fmt_share(n2, total)} | {total} |"
        )
    return "\n".join(lines)


def analyze(joint_path: Path, out_md: Path, out_json: Path) -> None:
    import statistics as _stats

    with joint_path.open() as f:
        d = json.load(f)

    # Re-derive per-(policy, family) action counts from per-step logs to
    # cross-check the summary_per_policy_family block in the JSON.
    counts: dict[str, Counter] = defaultdict(Counter)
    total_steps: dict[str, int] = defaultdict(int)
    per_step_actions: dict[str, dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for r in d["results"]:
        key = f"{r['policy']}/{r['family']}"
        for s in r["steps"]:
            counts[key][s["action_id"]] += 1
            total_steps[key] += 1
            per_step_actions[key][s["step"]][s["action_id"]] += 1

    # Sanity check against precomputed action_counts
    sanity_issues = 0
    for r in d["results"]:
        key = f"{r['policy']}/{r['family']}"
        reported = {a: c for a, c in (r.get("action_counts") or {}).items()}
        per_seed = Counter()
        for s in r["steps"]:
            per_seed[s["action_id"]] += 1
        if dict(per_seed) != reported:
            sanity_issues += 1

    # Per-step best_y trajectories
    best_traj: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in d["results"]:
        key = f"{r['policy']}/{r['family']}"
        for s in r["steps"]:
            best_traj[key][s["step"]].append(s.get("current_best_total", s.get("current_best", 0.0)))

    # Build per-(policy, family) sorted Counter for markdown
    families = ["Branin", "Hartmann", "Ackley", "SixHumpCamel", "Rastrigin"]
    policies = ["RulePolicy", "AlwaysSurrogate", "LLMDeepSeek", "Jev", "RoutingAware"]

    per_pol: dict[str, dict[str, Counter]] = {}
    for pol in policies:
        per_pol[pol] = {}
        for fam in families:
            per_pol[pol][fam] = counts.get(f"{pol}/{fam}", Counter())

    # Render markdown
    md: list[str] = []
    md.append("# EXP-RQ0.5 — Per-step LLM Action Distribution\n")
    md.append("**Source**: `runs/exp_rq0_5_joint_10seed.json` (joint 10-seed, 5-policy)")
    md.append("**Date**: 2026-09-27 (joint run) / analysis 2026-09-27")
    md.append("**Method**: re-derive `action_id` histogram from `steps[]` per trajectory")
    md.append("")

    md.append("## 1. Per-policy action distribution (5 families pooled)\n")
    md.append("Counts are over **all steps × all seeds × all 5 families** within a policy.\n")
    md.append("| Policy | top action | share | second action | share | total steps |")
    md.append("|---|---|---|---|---|---|")
    for pol in policies:
        agg: Counter = Counter()
        for fam in families:
            agg.update(per_pol[pol][fam])
        total = sum(agg.values())
        if total == 0:
            md.append(f"| {pol} | — | 0% | — | 0% | 0 |")
            continue
        top = agg.most_common(2)
        a1, n1 = top[0]
        if len(top) > 1:
            a2, n2 = top[1]
        else:
            a2, n2 = "—", 0
        md.append(
            f"| {pol} | `{a1}` | {_fmt_share(n1, total)} | "
            f"`{a2}` | {_fmt_share(n2, total)} | {total} |"
        )
    md.append("")

    md.append("## 2. Per-policy × per-family action distribution\n")
    for pol in policies:
        md.append(f"### {pol}\n")
        md.append(_policy_table(per_pol[pol], families))
        md.append("")

    md.append("## 3. Mechanism check: does LLM pick the closed-loop best first action?\n")
    md.append("Closed-loop best first-action under AlwaysSurrogate continuation "
              "(from `runs/EXP_RQ1_3_REPORT.md` §4):\n")
    md.append("| Family | closed-loop best first action |")
    md.append("|---|---|")
    for fam in families:
        md.append(f"| {fam} | `{CLOSED_LOOP_BEST[fam]}` |")
    md.append("")

    md.append("**LLMDeepSeek per-family top action** (from §2 above):\n")
    md.append("| Family | LLM top action | share | closed-loop best | matches? |")
    md.append("|---|---|---|---|---|")
    for fam in families:
        c = per_pol["LLMDeepSeek"][fam]
        if not c:
            md.append(f"| {fam} | — | 0% | `{CLOSED_LOOP_BEST[fam]}` | — |")
            continue
        top_action, top_n = c.most_common(1)[0]
        total = sum(c.values())
        share = _fmt_share(top_n, total)
        cl = CLOSED_LOOP_BEST[fam]
        match = "✓" if top_action == cl else "✗"
        md.append(f"| {fam} | `{top_action}` | {share} | `{cl}` | {match} |")
    md.append("")
    md.append("**Conclusion**: LLMDeepSeek does NOT match the closed-loop best "
              "first action on any of the 5 families — it picks `global_exploration` "
              "as the top action everywhere (66–76% share). The earlier §5 "
              "of `EXP_RQ1_3_REPORT.md` hypothesized LLM might be picking "
              "`local_exploitation` (closed-loop best) on Rastrigin; **that "
              "hypothesis is rejected** by the per-step distribution.")
    md.append("")

    # --- NEW: per-step progression table for LLM vs Surrogate on Rastrigin ---
    md.append("## 4. Per-step progression on Rastrigin (the +17.54 family)\n")
    md.append("Action share at each step across the 10 LLM trajectories on Rastrigin:\n")
    md.append("| step | global_exploration | local_exploitation | surrogate_acq | n_obs |")
    md.append("|---|---|---|---|---|")
    rastrigin_step = per_step_actions.get("LLMDeepSeek/Rastrigin", {})
    max_step = max(rastrigin_step.keys()) if rastrigin_step else 0
    for st in range(0, max_step + 1):
        c = rastrigin_step.get(st, Counter())
        n = sum(c.values())
        md.append(
            f"| {st} | {c.get('global_exploration', 0)}/{n} = "
            f"{_fmt_share(c.get('global_exploration', 0), n)} | "
            f"{c.get('local_exploitation', 0)}/{n} = "
            f"{_fmt_share(c.get('local_exploitation', 0), n)} | "
            f"{c.get('surrogate_acquisition', 0)}/{n} = "
            f"{_fmt_share(c.get('surrogate_acquisition', 0), n)} | {n} |"
        )
    md.append("")

    md.append("**Pattern**: LLMDeepSeek on Rastrigin uses a **'global-then-local' "
              "schedule** — `global_exploration` dominates steps 0–9 (90–100%), "
              "then transitions to `local_exploitation` from step 10 onward, with "
              "100% `local_exploitation` at step 13. This is a structured, "
              "non-uniform policy across the trajectory.")
    md.append("")
    md.append("AlwaysSurrogate action is **constant** at `surrogate_acquisition` 100% "
              "for all 25 steps on Rastrigin.")
    md.append("")

    # --- Per-step best_y trajectory ---
    md.append("## 5. Per-step best_y trajectory on Rastrigin (mean over 10 seeds)\n")
    md.append("| step | LLM mean best_y | Surv mean best_y | Δ (LLM − Surv) |")
    md.append("|---|---|---|---|")
    llm_traj = best_traj.get("LLMDeepSeek/Rastrigin", {})
    sur_traj = best_traj.get("AlwaysSurrogate/Rastrigin", {})
    for st in sorted(set(llm_traj) & set(sur_traj)):
        l_mean = _stats.mean(llm_traj[st]) if llm_traj[st] else float("nan")
        s_mean = _stats.mean(sur_traj[st]) if sur_traj[st] else float("nan")
        delta = l_mean - s_mean
        md.append(f"| {st} | {l_mean:+.3f} | {s_mean:+.3f} | {delta:+.3f} |")
    md.append("")
    md.append("**Pattern**: LLM starts roughly even with Surrogate (Δ ≈ 0 at step 1), "
              "then pulls ahead starting step 2. The advantage reaches "
              "Δ ≈ +20 by step 10 and **+45 by step 15** (limited to the 1 "
              "trajectory that ran that long; the median trajectory terminates "
              "at step ~13).")
    md.append("")

    # --- Revised mechanism ---
    md.append("## 6. Revised mechanism (replacing the rejected hypothesis)\n")
    md.append("The earlier hypothesis — *'LLM picks `local_exploitation` which is "
              "the closed-loop best'* — is rejected by the per-step distribution.\n")
    md.append("**Revised mechanism (descriptive, not causal)**: On Rastrigin (10D "
              "multimodal), the LLM exhibits a **two-phase temporal action pattern**:")
    md.append("")
    md.append("1. **Early phase (steps 0–9)**: predominantly `global_exploration` "
              "(90–100% across these steps; occasionally 1/10 trajectories "
              "spike to `local_exploitation` at steps 3, 6, 8, 9). The LLM "
              "samples broadly across the input space to escape local optima.")
    md.append("2. **Transition phase (steps 10–12)**: mixed — 30–40% global, "
              "60–80% local, occasional 1/10 surrogate.")
    md.append("3. **Late phase (step 13+)**: 100% `local_exploitation`. After "
              "global sampling, the LLM refines around the best region found.")
    md.append("")
    md.append("AlwaysSurrogate, by contrast, picks `surrogate_acquisition` "
              "(GP-EI) for **every** step — relying entirely on the surrogate to "
              "balance exploration and exploitation. On multimodal problems, "
              "the surrogate's exploitation bias dominates too early, before "
              "global coverage is sufficient.")
    md.append("")
    md.append("**Important caveat (per codex review 2026-09-27)**: this is a "
              "*behavioral* association, not a causal claim. The LLM's pooled "
              "top action on Rastrigin is `global_exploration` at 70%, which is "
              "**similar to its share on smoother families** (Branin 76%, "
              "Hartmann 67%, Ackley 66%, SixHumpCamel 68%) — *families where "
              "LLM does NOT beat Surrogate*. The pooled distribution alone "
              "does not explain the Rastrigin-specific advantage; only the "
              "*temporal pattern* (early-global, late-local) is unique to "
              "Rastrigin among the 5 families.")
    md.append("")
    md.append("**Causal validation not established**: a deterministic "
              "global-then-local baseline (e.g., always pick `global_exploration` "
              "for 10 steps then `local_exploitation` thereafter) has not "
              "been implemented or compared. Until that ablation is run, the "
              "schedule-vs-advantage link remains correlational.")
    md.append("")
    md.append("Recommended paper wording: *'On Rastrigin, the LLM exhibited a "
              "global-to-local temporal action pattern associated with its "
              "terminal advantage. This analysis does not establish that the "
              "schedule caused the advantage.'*")
    md.append("")

    md.append("## 7. Sanity check\n")
    if sanity_issues == 0:
        md.append("All 250 trajectories: per-step `action_id` counts match the JSON's "
                  "precomputed `action_counts` block. ✓")
    else:
        md.append(f"**{sanity_issues}** trajectories have per-seed action-count "
                  "mismatches with the JSON's precomputed block.")
    md.append("")

    md.append("## 8. Files\n")
    md.append("- `runs/exp_rq0_5_joint_10seed.json` — source (joint 10-seed, 5-policy)")
    md.append("- `runs/exp_rq0_5_action_distribution.json` — raw counts")
    md.append("- `runs/EXP_RQ0_5_REPORT.md` — parent report")
    md.append("- `runs/EXP_RQ1_3_REPORT.md` — closed-loop regret reference")

    out_md.write_text("\n".join(md))

    # JSON dump
    raw = {
        "source": str(joint_path),
        "policies": policies,
        "families": families,
        "per_policy_family_counts": {
            f"{pol}/{fam}": dict(per_pol[pol][fam]) for pol in policies for fam in families
        },
        "per_step_actions": {
            f"{k}": {str(st): dict(c) for st, c in steps.items()}
            for k, steps in per_step_actions.items()
        },
        "per_step_best_y_means": {
            f"{k}": {str(st): _stats.mean(v) for st, v in steps.items()}
            for k, steps in best_traj.items()
        },
        "closed_loop_best_first_action": CLOSED_LOOP_BEST,
        "sanity_issues": sanity_issues,
    }
    out_json.write_text(json.dumps(raw, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--joint",
        type=Path,
        default=ROOT / "runs" / "exp_rq0_5_joint_10seed.json",
    )
    parser.add_argument("--out-md", type=Path, default=ROOT / "runs" / "exp_rq0_5_action_distribution.md")
    parser.add_argument("--out-json", type=Path, default=ROOT / "runs" / "exp_rq0_5_action_distribution.json")
    args = parser.parse_args()

    if not args.joint.exists():
        print(f"ERROR: {args.joint} not found", file=sys.stderr)
        return 1

    print(f"Reading {args.joint} ...")
    analyze(args.joint, args.out_md, args.out_json)
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
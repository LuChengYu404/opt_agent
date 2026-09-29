"""EXP: does the lenz-style rich state change LLM decisions? (2026-09-28)

Branch ``exp/lenz-state-jev-presentation`` — the survey's open question:
the frozen 15-scalar state vs the lenz-style full-trial-log state, SAME
frozen 8-action menu (executable criteria), SAME model (deepseek-chat,
temperature 0), paired per record. Decision-relevant readouts:

  - inter-condition choice agreement (does richness change anything?);
  - gold-argmax agreement per condition (does it change toward the teacher?);
  - token cost per condition (richness is not free).

Dataset: the INDEPENDENT replication stream (seeds 100-119 — untouched by
any design decision). Records stratified by family and phase.

Usage:
    .venv/bin/python scripts/exp_lenz_vs_frozen_llm.py [--per-family 8]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from opt_agent.dataset import read_jsonl, phase_of_ratio  # noqa: E402
from opt_agent.state import lenz_state_from_record  # noqa: E402
from opt_agent.system1.llm_system1_policy import LLMSystem1Policy  # noqa: E402
from opt_agent.system1.lenz_prompt import build_lenz_choice_request  # noqa: E402

FAMILIES = ("ackley", "branin", "hartmann", "rastrigin", "sixhumpcamel")


def _load_env() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if line.startswith("DEEPSEEK_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("DEEPSEEK_API_KEY not found in .env")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path,
                        default=Path("runs/phase8_6d_h3_20seed_5family_s100_119.jsonl"))
    parser.add_argument("--per-family", type=int, default=8)
    parser.add_argument("--json-out", type=Path,
                        default=Path("runs/exp_lenz_vs_frozen_llm.json"))
    args = parser.parse_args()

    api_key = _load_env()
    records = [r for r in read_jsonl(args.dataset, version_policy="warn")]

    # Stratified sample: per family, spread across seeds and phases.
    by_family: dict[str, list] = defaultdict(list)
    for r in records:
        by_family[r.problem["family"]].append(r)
    sampled = []
    for fam in sorted(by_family):
        rows = sorted(by_family[fam], key=lambda r: (int(r.problem["seed"]), r.metadata["step"]))
        phases = {"early": [], "mid": [], "late": []}
        for r in rows:
            p = phase_of_ratio(float(r.state["resource"]["remaining_budget_ratio"]))
            if p in phases:
                phases[p].append(r)
        take: list = []
        per_phase = max(1, args.per_family // 3)
        for p in ("early", "mid", "late"):
            step = max(1, len(phases[p]) // per_phase)
            take.extend(phases[p][::step][:per_phase])
        sampled.extend(take[:args.per_family])
    print(f"sampled {len(sampled)} records "
          f"({len(sampled) // 5}/family) -> 2 conditions each")

    frozen_policy = LLMSystem1Policy(
        mode="llm", api_key=api_key, temperature=0.0,
        state_format="narrative", criteria_format="executable", seed=0,
    )
    lenz_policy = LLMSystem1Policy(
        mode="llm", api_key=api_key, temperature=0.0, seed=0,
        request_builder=build_lenz_choice_request,
    )

    rows_out = []
    n_err = 0
    for i, rec in enumerate(sampled):
        gold_top = max(rec.gold_probs, key=lambda a: rec.gold_probs[a])
        lenz = lenz_state_from_record(rec)
        summary = lenz.summary
        try:
            d_frozen = frozen_policy.decide(summary)
            d_lenz = lenz_policy.decide(lenz)
        except Exception as exc:  # noqa: BLE001 — record and continue
            n_err += 1
            print(f"[{i}] error: {exc!r}")
            continue
        rows_out.append({
            "record_id": rec.id,
            "family": rec.problem["family"],
            "seed": int(rec.problem["seed"]),
            "phase": phase_of_ratio(
                float(rec.state["resource"]["remaining_budget_ratio"])),
            "gold_top": gold_top,
            "frozen_choice": d_frozen.action.type.value,
            "lenz_choice": d_lenz.action.type.value,
            "frozen_gold_hit": d_frozen.action.type.value == gold_top,
            "lenz_gold_hit": d_lenz.action.type.value == gold_top,
            "frozen_tokens": d_frozen.input_tokens + d_frozen.output_tokens,
            "lenz_tokens": d_lenz.input_tokens + d_lenz.output_tokens,
            "frozen_latency_s": round(d_frozen.latency_s, 3),
            "lenz_latency_s": round(d_lenz.latency_s, 3),
        })
        if (i + 1) % 10 == 0:
            done = len(rows_out)
            agree = sum(r["frozen_choice"] == r["lenz_choice"] for r in rows_out) / done
            print(f"  ... {i + 1}/{len(sampled)} inter-cond agreement {agree:.2f}",
                  flush=True)
        time.sleep(0.2)

    n = len(rows_out)
    def _rate(key):  # noqa: ANN001
        return round(sum(r[key] for r in rows_out) / n, 4) if n else None

    def _mean(key):
        return round(sum(r[key] for r in rows_out) / n, 1) if n else None

    summary_out = {
        "n_records": n,
        "n_errors": n_err,
        "model": "deepseek-chat (temperature 0)",
        "inter_condition_choice_agreement":
            round(sum(r["frozen_choice"] == r["lenz_choice"] for r in rows_out) / n, 4),
        "gold_hit": {"frozen": _rate("frozen_gold_hit"),
                     "lenz": _rate("lenz_gold_hit")},
        "tokens": {"frozen": _mean("frozen_tokens"), "lenz": _mean("lenz_tokens")},
        "latency_s": {"frozen": _mean("frozen_latency_s"),
                      "lenz": _mean("lenz_latency_s")},
        "per_family_gold_hit": {
            fam: {
                "frozen": round(sum(r["frozen_gold_hit"] for r in rows_out
                                    if r["family"] == fam)
                                / max(1, sum(1 for r in rows_out if r["family"] == fam)), 4),
                "lenz": round(sum(r["lenz_gold_hit"] for r in rows_out
                                  if r["family"] == fam)
                              / max(1, sum(1 for r in rows_out if r["family"] == fam)), 4),
            } for fam in FAMILIES
        },
    }
    args.json_out.write_text(
        json.dumps({"summary": summary_out, "rows": rows_out}, indent=1,
                   sort_keys=True) + "\n")
    print(json.dumps(summary_out, indent=1))
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

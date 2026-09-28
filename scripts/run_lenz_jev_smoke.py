"""Smoke: lenz-style state + Jev-style presentation end-to-end (2026-09-28).

Branch ``exp/lenz-state-jev-presentation``. Runs a short Branin trajectory,
builds the rich-state choice request at each decision point, and saves the
final full prompt (narrative state + executable action menu) to
``runs/lenz_jev_prompt_example.txt``. Decisions come from a deterministic
mock (first offered candidate) — the point is the state/prompt pipeline,
not decision quality. The response is round-tripped through
``validate_choice_response`` to prove protocol compatibility with the
existing JevPolicy transport.

Usage:
    .venv/bin/python scripts/run_lenz_jev_smoke.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from opt_agent.actions import candidate_actions_for_state, execute_action  # noqa: E402
from opt_agent.backend.bocode_botorch import (  # noqa: E402
    BackendConfig,
    BocodeBotorchBackend,
)
from opt_agent.state import extract_lenz_state  # noqa: E402
from opt_agent.system1._prompt import validate_choice_response  # noqa: E402
from opt_agent.system1.lenz_prompt import build_lenz_choice_request  # noqa: E402

TOTAL_BUDGET = 15


def main() -> int:
    be = BocodeBotorchBackend()
    be.initialize(BackendConfig(problem_id="Branin", seed=0, n_init=5))
    history: list[str | None] = [None] * len(be.get_history())
    last_prompt: str = ""

    for step in range(4):
        lenz = extract_lenz_state(be, total_budget=TOTAL_BUDGET,
                                  decision_history=tuple(history))
        choices = candidate_actions_for_state(lenz.summary, seed=step)
        state_arg, question = build_lenz_choice_request(lenz, choices)

        # Deterministic mock decision: first offered candidate.
        selected = next(iter(choices))
        probs = {k: (1.0 if k == selected else 0.0) for k in choices}
        answer = {"answers": {"next_action": {
            "type": "choice", "choice": selected, "probabilities": probs,
            "confidence": 0.5,
        }}}
        got_id, _probs, _conf = validate_choice_response(answer, choices)
        assert got_id == selected

        result = execute_action(be, lenz.summary, choices[got_id])
        n_new = result.n_new_samples
        if result.should_stop:
            history.append(got_id)
            last_prompt = _render(state_arg, question, got_id)
            break
        history.append(got_id)
        # Extra samples produced by the SAME high-level action are labelled
        # with that action (repeated), not as initial design.
        history.extend([got_id] * max(0, n_new - 1))
        last_prompt = _render(state_arg, question, got_id)
        print(f"step {step}: chose {got_id} (n_new={n_new}, "
              f"budget left {lenz.remaining_budget - n_new})")

    out = ROOT / "runs" / "lenz_jev_prompt_example.txt"
    out.write_text(last_prompt, encoding="utf-8")
    print(f"saved final prompt -> {out} ({len(last_prompt)} chars)")
    (ROOT / "runs" / "lenz_jev_smoke.json").write_text(json.dumps({
        "problem": "Branin", "total_budget": TOTAL_BUDGET,
        "decision_history": history,
        "prompt_chars": len(last_prompt),
    }, indent=1) + "\n")
    return 0


def _render(state_arg: str, question: dict, chosen: str) -> str:
    criteria = question["next_action"]["criteria"]
    menu = "\n".join(
        f"- {name}: {text}" for name, text in criteria.items()
    )
    return (
        "=== STATE (lenz-style narrative, Jev presentation) ===\n"
        f"{state_arg}\n\n"
        "=== QUESTION ===\n"
        f"{question['next_action']['instructions']}\n\n"
        "=== OPTIONS ===\n"
        f"{menu}\n\n"
        f"=== MOCK CHOICE === {chosen}\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())

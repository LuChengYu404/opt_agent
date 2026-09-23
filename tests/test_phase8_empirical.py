"""Phase 8 empirical validation tests — verify P*(a|s) labels are scientifically defensible.

Per codex (gpt-5.6-sol) review 2025-09-23:
  - Dataset is canonically deterministic across runs.
  - `derive_run_seed` is independent of PYTHONHASHSEED.
  - Records satisfy aligned-vocab / probability-sum / invalid-masked invariants.
  - Trajectory budget is monotone non-decreasing.
  - Zero-budget masks all sampling actions.
  - JSONL round-trip is canonical.

We do NOT yet assert "early = exploration-dominant" etc. — those are
acceptance thresholds to add only after we observe the patterns.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from conftest import RecordingBackend  # noqa: E402

from opt_agent.actions import ActionType  # noqa: E402
from opt_agent.backend.bocode_botorch import BackendConfig, BocodeBotorchBackend  # noqa: E402
from opt_agent.dataset import (  # noqa: E402
    DatasetConfig,
    DatasetGenerator,
    DecisionRecord,
    analyze_dataset,
    derive_run_seed,
    full_candidates,
    iter_jsonl,
    validate_record,
    write_jsonl,
)
from opt_agent.dataset.decision_record import _json_safe
from opt_agent.rollout import CounterfactualRollout, to_gold_probs
from opt_agent.state import extract_state


def _real_backend_factory():
    return BocodeBotorchBackend()


# ---------------------------------------------------------------------------
# Run-seed independence from PYTHONHASHSEED
# ---------------------------------------------------------------------------


def test_run_seed_is_independent_of_python_hash_seed() -> None:
    """Codex-flagged bug: using Python's `hash(problem)` varies with
    PYTHONHASHSEED across processes. derive_run_seed must NOT.
    """
    # Set PYTHONPATH so the subprocess can import the package.
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    code = (
        "from opt_agent.dataset.generator import derive_run_seed;"
        "print(derive_run_seed('Branin', 0, 0))"
    )
    outputs = set()
    for value in ("0", "1", "random"):
        result = subprocess.check_output(
            [sys.executable, "-c", code],
            env={**env, "PYTHONHASHSEED": value},
            text=True,
        ).strip()
        outputs.add(result)
    assert len(outputs) == 1, f"derive_run_seed depends on PYTHONHASHSEED: {outputs}"


# ---------------------------------------------------------------------------
# Canonical determinism
# ---------------------------------------------------------------------------


def _small_spec() -> DatasetConfig:
    return DatasetConfig(
        problems=("Branin",),
        seeds=(0,),
        trajectories_per_problem_seed=1,
        n_init=3,
        total_budget=10,
        decision_every_k_actions=2,
        max_actions=4,
    )


def test_dataset_is_canonically_deterministic() -> None:
    """Same spec → identical records (modulo JSON-safe NaN coercion)."""
    spec = _small_spec()
    first = _json_safe(list(DatasetGenerator(spec).generate()))
    second = _json_safe(list(DatasetGenerator(spec).generate()))
    assert first == second


# ---------------------------------------------------------------------------
# Record schema invariants (validate_record)
# ---------------------------------------------------------------------------


def _first_record() -> DecisionRecord:
    spec = _small_spec()
    gen = DatasetGenerator(spec, backend_factory=_real_backend_factory)
    return next(gen.generate())


def test_record_schema_and_probability_mask() -> None:
    """Every record: aligned vocab, probs sum to 1, invalid has prob 0."""
    record = _first_record()
    validate_record(record)
    for action_id, result in record.rollout_results.items():
        is_invalid = result["status"] == "invalid"
        assert (record.gold_probs[action_id] == 0) == is_invalid


# ---------------------------------------------------------------------------
# Trajectory structure
# ---------------------------------------------------------------------------


def test_trajectory_budget_is_monotone() -> None:
    """Within a trajectory, used_budget is non-decreasing."""
    spec = _small_spec()
    records = list(DatasetGenerator(spec, backend_factory=_real_backend_factory).generate())
    grouped: dict = defaultdict(list)
    for r in records:
        key = (r.problem["family"], r.problem["seed"], r.metadata["trajectory_id"])
        grouped[key].append(r)

    for traj in grouped.values():
        traj.sort(key=lambda r: r.metadata["step"])
        used = [r.state["resource"]["used_budget"] for r in traj]
        assert used == sorted(used)
        total = traj[0].state["resource"]["total_budget"]
        assert all(0 <= n <= total for n in used)


# ---------------------------------------------------------------------------
# Zero-budget masking
# ---------------------------------------------------------------------------


def test_zero_budget_masks_all_sampling_actions() -> None:
    """At exhausted budget, sampling actions are rejected → prob=0."""
    be = BocodeBotorchBackend()
    be.initialize(BackendConfig(problem_id="Branin", seed=0, n_init=5))
    for _ in range(15):
        be.evaluate(torch.tensor([[0.5, 7.5]], dtype=torch.double))
    state = extract_state(be, total_budget=20)  # remaining=0
    output = CounterfactualRollout(be, horizon=1, max_rollout_steps=20).rollout(
        state, full_candidates(seed=0)
    )
    probs = to_gold_probs(output.results, include_invalid_as_zero=True)

    invalid = {
        "global_exploration", "local_exploitation",
        "uncertainty_sampling", "error_guided_sampling",
        "boundary_sampling",
    }
    for action in invalid:
        assert probs[action] == 0
    assert probs["stop"] > 0
    assert sum(probs.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# JSONL canonical round-trip
# ---------------------------------------------------------------------------


def test_jsonl_round_trip_is_canonically_equal(tmp_path) -> None:
    spec = _small_spec()
    records = list(DatasetGenerator(spec, backend_factory=_real_backend_factory).generate())
    path = tmp_path / "dataset.jsonl"
    write_jsonl(records, path)
    expected = [
        DecisionRecord(**_json_safe(asdict(record)))
        for record in records
    ]
    assert list(iter_jsonl(path)) == expected


# ---------------------------------------------------------------------------
# analyze_dataset summary
# ---------------------------------------------------------------------------


def test_analyze_dataset_smoke(tmp_path) -> None:
    """End-to-end: generate → write → analyze → check non-empty stats."""
    spec = _small_spec()
    path = tmp_path / "ds.jsonl"
    write_jsonl(DatasetGenerator(spec, backend_factory=_real_backend_factory).generate(), path)
    stats = analyze_dataset(path)
    assert stats.n_records > 0
    assert "branin" in stats.problems
    assert stats.problems["branin"] == stats.n_records


# ---------------------------------------------------------------------------
# Phase 8.5 wiring fixes (codex-flagged)
# ---------------------------------------------------------------------------


def test_dataset_generator_wires_scales_into_rollout() -> None:
    """Codex-flagged bug: generator must pass scales to CounterfactualRollout."""
    import dataclasses
    spec = _small_spec()
    # Set non-default scales to verify they reach the rollout.
    spec = dataclasses.replace(spec,
        optimization_scale=2.0,
        evaluation_scale=30.0,
        fit_scale=7.0,
    )
    seen = []

    from opt_agent.dataset import generator as generator_module
    original_rollout = generator_module.CounterfactualRollout

    class _Spy(original_rollout):
        def __init__(self, backend, **kwargs):
            seen.append({
                "optimization_scale": kwargs.get("optimization_scale"),
                "evaluation_scale": kwargs.get("evaluation_scale"),
                "fit_scale": kwargs.get("fit_scale"),
            })
            super().__init__(backend, **kwargs)

    generator_module.CounterfactualRollout = _Spy
    try:
        list(DatasetGenerator(spec, backend_factory=_real_backend_factory).generate())
    finally:
        generator_module.CounterfactualRollout = original_rollout

    assert seen, "CounterfactualRollout was never called"
    assert seen[0]["optimization_scale"] == 2.0
    assert seen[0]["evaluation_scale"] == 30.0
    assert seen[0]["fit_scale"] == 7.0


def test_generated_state_preserves_last_action() -> None:
    """Codex-flagged bug: state.history.last_action should NOT be None on records
    emitted after at least one behavior action has executed."""
    spec = _small_spec()
    records = list(DatasetGenerator(spec, backend_factory=_real_backend_factory).generate())
    assert records
    # First record (step=0, no behavior yet): last_action is None.
    assert records[0].state["history"]["last_action"] is None
    # Some later record: last_action must be populated by a prior behavior step.
    populated = [r for r in records[1:] if r.state["history"]["last_action"] is not None]
    assert populated, (
        "Expected at least one record after the first to have last_action set; "
        "generator is not passing the prior action name to extract_state."
    )


# ---------------------------------------------------------------------------
# Driver script smoke
# ---------------------------------------------------------------------------


def test_driver_main_function_smoke(tmp_path) -> None:
    """`generate_dataset.main()` writes a valid JSONL and returns 0."""
    out_path = tmp_path / "quick.jsonl"
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib
    if "generate_dataset" in sys.modules:
        importlib.reload(sys.modules["generate_dataset"])
    import generate_dataset

    import io, contextlib
    captured = io.StringIO()
    saved_argv = sys.argv
    sys.argv = ["generate_dataset", "--quick", str(out_path)]
    try:
        with contextlib.redirect_stdout(captured):
            rc = generate_dataset.main()
    finally:
        sys.argv = saved_argv
    assert rc == 0
    assert out_path.exists()
    records = list(iter_jsonl(out_path))
    assert records
    stdout = captured.getvalue()
    obj = None
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(stdout):
        try:
            obj, idx = decoder.raw_decode(stdout, idx=idx)
            while idx < len(stdout) and stdout[idx] in " \t\r\n":
                idx += 1
        except json.JSONDecodeError:
            break
    assert obj is not None
    assert obj["n_records"] > 0

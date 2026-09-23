# Current research status

## Phase 8.6B: implemented, empirical gates still open

The dataset generator now saves the full backend checkpoint needed to replay a
decision point, including history, fitted GP state, and torch RNG state.
`scripts/validate_phase8_6b.py` checks seed-level action stability and compares
the top `gold_probs` action with STOP, uniform, and the behavior policy using
paired terminal optimization runs. All non-STOP arms use the same continuation
policy, so the first decision is the controlled variable.

On `runs/phase8_6b_h3_10seed.jsonl` (Branin, Ackley, Hartmann; 10 seeds per
problem; 226 records; horizon 3):

| Gate | Result | Evidence |
| --- | --- | --- |
| 2: stable action preference | Fail | Several problem/phase groups have <60% agreement across seed-level top actions. |
| 3: early exploration mass | Pass (point estimate) | Adjusted lift = 1.67; this is not a seed-clustered confidence-interval pass. |
| 4: terminal optimization advantage | Fail | On Branin, top-label action vs behavior has mean normalized gain -0.0259, bootstrap 95% interval [-0.0487, -0.0102]. |

The full result, including per-seed paired outcomes, is in
`runs/phase8_6b_h3_report.json`. Removing standalone retraining from the saved
candidate results does not repair the Branin outcome; see
`runs/phase8_6b_h3_no_retrain_report.json`.

## Phase 9: exploratory baseline entry point

`scripts/phase9_baselines.py` holds out seeds 8 and 9 and compares uniform,
RulePolicy, a constant probability prior, and a 32-unit MLP. On the provisional
horizon-3 labels, MLP top-1 agreement is 0.255 on 47 held-out records versus
0.125 for uniform choice. Its soft-target cross-entropy is 2.076 versus 2.079
for uniform probabilities. These are label-agreement diagnostics, not evidence
of better closed-loop optimization. See `runs/phase9_baseline_smoke.json`.

Phase 9 supervision training remains gated by the terminal optimization check.
This gate does not prevent building and evaluating a zero-shot System-1 reference
or the end-to-end Agent workflow. No System-1 performance improvement is claimed
from agreement with these provisional labels.

## System-1 reference: Laya zero-shot

`src/opt_agent/system1/laya_policy.py` maps the current `OptimizationState` to
Laya's typed `choice` request over budget-valid actions. It preserves the full
returned probability distribution and confidence for later routing analysis.
`scripts/run_laya_baseline.py` runs that policy through `DecisionLoop` on a
real BOCoDe/BoTorch problem. Laya is an optional dependency. The SDK is
installed, and the English checkpoint is cached at
`/home/lcy/.cache/opt_agent_v3/laya`. Direct Hugging Face access failed during
TLS negotiation through the local proxy. The same repository's English files
were downloaded from ModelScope through `http://127.0.0.1:7890`; the
`model.safetensors` SHA-256 was verified against Hugging Face's official file
listing (`891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`).
All four supporting files matched ModelScope's published hashes.

Real inference smoke: Branin seed 0, five initial samples, total budget 10,
five Laya decisions. The run exhausted the budget and wrote probabilities and
confidence for each action. It chose `error_guided_sampling` in all five rounds;
the best observed objective remained `-3.5451956715421797`. This is one
zero-shot smoke run, not a performance estimate. Full output:
`runs/laya_branin_seed0_budget10.json`.

```bash
uv sync --extra laya
uv run --extra laya python scripts/run_laya_baseline.py \
  --model-id /home/lcy/.cache/opt_agent_v3/laya \
  --problem Branin --seed 0 --budget 10
```

## Deferred research: short-rollout labels versus terminal outcomes

**Observed:** The horizon-3 top `gold_probs` action has unstable top-action
preferences across seeds and underperforms the behavior-first-action control
on Branin in paired full-budget continuations. Removing standalone retraining
from candidate results does not fix the Branin result. This is a label-validity
warning, not yet a diagnosis of reward hacking or a trained-model failure.

**Questions for later work:**

1. Does the action ranking change with horizon, utility weights, or the
   continuation policy? Separate horizon/utility mismatch from rollout noise.
2. When futures use surrogate predictions, how much does model error change
   the ranking? Compare model-based futures against true-oracle replay where
   feasible.
3. Can a terminal-value target or an abstaining router improve full-budget
   simple regret across Branin, Ackley, and Hartmann on held-out seeds?
4. Which gains survive paired full-budget evaluation after accounting for
   System-2 calls, model fitting, and evaluation cost?

Keep this as a research track while completing System-1, System-2, and routing
integration. A functioning end-to-end system and a demonstrated optimization
gain are separate milestones.

Relevant starting points: Yue and Kontar, *Why Non-myopic Bayesian Optimization
is Promising and How Far Should We Look-ahead?* (AISTATS 2020,
https://proceedings.mlr.press/v108/yue20b.html); Lee et al., *Efficient Rollout
Strategies for Bayesian Optimization* (UAI 2020,
https://proceedings.mlr.press/v124/lee20a.html); Zhang and Chen, *Direct Regret
Optimization in Bayesian Optimization* (arXiv:2507.06529,
https://arxiv.org/abs/2507.06529). Their model-based rollout results provide
hypotheses, not a diagnosis of our true-oracle counterfactual setup.

## Reproduce

```bash
.venv/bin/pytest -q
.venv/bin/python scripts/generate_dataset.py runs/phase8_6b_h3_10seed.jsonl --seeds 10 --horizon 3 --force
.venv/bin/python scripts/validate_phase8_6b.py runs/phase8_6b_h3_10seed.jsonl --output runs/phase8_6b_h3_report.json --random-replicates 5
.venv/bin/python scripts/phase9_baselines.py runs/phase8_6b_h3_10seed.jsonl
```

## Phase 9.0: framework skeleton (250/250 tests)

The Hybrid Agent glue from MASTER_PLAN §24 is now in place:

- **System-1**: `LayaPolicy.decide(state)` returns `(action, probabilities,
  confidence)` over budget-valid candidates from a shared factory
  (`actions.candidate_actions_for_state`).
- **System-2 stub**: `FallbackSystem2` delegates to `RulePolicy` — STOP is
  only returned when the baseline itself chooses STOP (regression test
  guards the M2 finding).
- **Router**: `ConfidenceRouter` uses only `max(choice_probs)` vs
  `tau_conf`. `predict_score` and `predict_noul` are deliberately NOT
  consulted — those heads have no calibration targets yet.
- **HybridPolicy**: implements the existing `Policy` Protocol and is
  passed directly to `DecisionLoop.run(policy)`. The DecisionLoop
  constructor is unchanged.
- **Calibration metrics**: ECE / Brier / NLL / reliability with explicit
  docstring on the two regimes (internal-consistency vs real-calibration).
- **Decision metrics**: top-1 / top-k / KL / NLL-vs-gold with explicit
  `n_infinite_kl` / `n_infinite_nll` counts so a single +inf record can
  no longer hide behind a clean mean.
- **Dataset splits**: 5-way split (60/10/10/20 + OOD) with sorted-family
  iteration for cross-process determinism; `physical_state_hash` hashes
  only `record.state` (no problem seed, no candidate params).

Codex gpt-6-sol review (2026-09-23) flagged 6 findings (C1, C2, M1, M2,
M3, M4) and 3 new issues (calibration regimes, KL/NLL aggregation,
split determinism + hash scope). All 9 are now resolved in the current
working tree. See PHASE9_DESIGN.md for the full reconciled design.

### Phase 9.1: real-backend smoke (DONE)

`scripts/run_hybrid_smoke.py` runs HybridPolicy vs RulePolicy on
BocodeBotorchBackend across Branin + Ackley + Hartmann × seeds
{0, 1, 2} × total_budget in {10, 30}. tau_conf = 0.5 (fixed, not tuned).

**Acceptance criteria (from codex gpt-6-sol review 2026-09-23):**
1. All 36 runs complete without exceptions. ✓
2. Per-step action and evaluation cost obey factory + budget constraints. ✓
3. Routing-count sum equals decision-step count for every Hybrid run. ✓
4. Per problem: at least one Hybrid trajectory actually executes an evaluation. ✓
5. Finite `simple_regret = known_optimum - best_value` reported for
   Branin (literature-known optimum). Other problems report best_value
   only because the BOCoDe parameterizations (Ackley53 10-D, Hartmann
   6-D) do not match the textbook functions.

**Results on commit `98b595c`** (36 runs, all passed acceptance):
- Branin / budget=30 / hybrid: best_value = -0.4150, simple_regret = +0.0171
- Branin / budget=30 / rule:   best_value = -2.3702, simple_regret = +1.9723
- Hartmann / budget=30 / hybrid best_value = +2.8489 (vs rule +1.6605)
- Ackley / budget=30 / hybrid best_value = -20.283 (vs rule -20.412)

Detailed JSON: `runs/phase9_1_smoke.json`.

**Not a performance claim.** This smoke does NOT assert that
HybridPolicy beats RulePolicy — paired multi-seed comparison needs a
larger evaluation plan.

### Phase 9.3 (later)

Fine-tune a typed-decision base model on rollout data using
proper-scoring-rule loss (avoids the utility-hacking failure mode
observed in Phase 8.5 / 8.6A). Reference implementation candidate
deferred until the framework skeleton's real-backend smoke is
validated end-to-end (now done in Phase 9.1).

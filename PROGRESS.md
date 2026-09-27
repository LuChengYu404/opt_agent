# Current research status

## EXP-RQ3 — Real-LLM cost frontier, 10-seed confirmation + sign correction (2026-09-27, latest)

`runs/exp_rq3_router_sweep.json` (seeds 100–102, 135 trajectories) plus
`runs/exp_rq3_router_sweep_seeds103_109.json` (seeds 103–109, 315
trajectories, 8 parallel workers) = **450 paired trajectories** (5 BOCoDe
families × 10 seeds × 9 policies) on a real `deepseek-chat` System-2
backend, frozen RF System-1 (rollout seeds 0–15, calibrated 16–19;
identical dataset/training hash across both files), 0 paired-start
violations, 0 LLM fallbacks. Merged analysis:
`scripts/analyze_rq3_cost_frontier.py` →
`runs/exp_rq3_cost_frontier_10seed.json` (reproduces the 3-seed numbers
exactly on the original file).

**Sign correction.** The earlier 3-seed report's prose inverted the sign
of its own Hartmann numbers: `rf` beat `llm` 3/0/0 at seeds 100–102
(+0.910); the claim "Hartmann is the one family where the LLM is
genuinely useful" and the recommendation "Hartmann → llm" were wrong and
are withdrawn. The independent EXP-RQ0.5 joint run (LLM −0.985 on
Hartmann, p = 0.0007) had the correct direction all along.

**10-seed headline.** The frozen System-1 alone — **0 LLM calls — beats
or ties every LLM-calling arm on all five families.** `rf − llm`:
Branin +0.515, Hartmann **+0.576 (CI excludes 0)**, Ackley **+1.017 (CI
excludes 0; flips the 3-seed "LLM may benefit" read)**, SixHumpCamel
**+1.015 (CI excludes 0)**, Rastrigin +10.109 (CI crosses 0; the 3-seed
borderline significance weakens). **No arm on the whole 40-cell grid
beats `rf` with CI excluding zero**; the only positive point estimates
anywhere are Hartmann/confidence_0.25 (+0.018) and Hartmann/random_0.25
(+0.149), both CI-crossing. Pre-specified criteria now PASS 5/5 (100 %
call/token reduction; regret margin never violated) — but the frontier
is degenerate: always-LLM is dominated, and no confidence threshold or
random rate earns its cost. Full report (correction + confirmation
sections at top): `runs/EXP_RQ3_REAL_LLM_COST_FRONTIER_REPORT.md`.

**Confidence-routing diagnostic (updated at 10 seeds).** The 3-seed
"random-rate beats the confidence router" observation does not
generalize: at 10 seeds neither scheme produces a CI-detectable gain
over `rf`, and Branin random_0.5/0.75 are CI-detectably worse. The RF
softmax maximum still rarely exceeds 0.25, so confidence thresholds
route almost everything to System-2.

**Per-family recommendation (10 seeds).** All five families → `rf`
(System-1 only). No LLM-favourable family exists on this panel at
budget 30; the open problem is finding any regime (longer horizons,
stochastic objectives, OOD states) where the LLM System-2 repays its
cost.

## EXP-RQ2.2 — Within-function dimension transfer (2026-09-27)

`scripts/exp_rq2_2_ioh_within_function_dim_transfer.py` trains a
multi-output RF on the action labels of `RuleFullBudget` at IOH
functions f6, f8, f15 × dims {2, 10} × instances {0, 1} × seeds {100, 101,
102} (36 trajectories), then deploys the RF on the held-out test dim 5
× instance 2 (untouched). Full report:
`runs/EXP_RQ2_2_WITHIN_FUNCTION_DIM_TRANSFER_REPORT.md`.

**Result.** The RF perfectly mimics `RuleFullBudget` on the test dim
(`rf − rule_full = +0.000`, 0/3/0 W/T/L on every function), as expected
for imitation learning. Both lose to Surrogate and Local on every
function: `rf − surrogate` mean +0.27 (f6), +0.39 (f8), +0.24 (f15) with
CI lower bound > 0 on f8 and f15. **Within-function dimension transfer
is a clean negative result**: the cross-family transfer signal in
EXP-RQ2.1 was at least partly a *family* signal, not a dimension signal.

## Closed-loop and external-transfer audit (2026-09-27)

The complete results and methods are in
`runs/EXP_FINAL_GOAL_CLOSED_LOOP_REPORT.md`. Independent trajectory workers
now run in parallel (four by default); serial/parallel checks reproduced
identical actions and terminal values on 16 BOCoDe and four IOH smoke runs,
plus two full-budget BOCoDe runs with GP fitting.

- **RQ1 remains open:** a frozen RF trained on the 20-seed rollout corpus was
  evaluated in full budget-30 loops on five BOCoDe families. On fresh seeds
  200–219 it beat AlwaysSurrogate on Ackley (+0.662 raw terminal best, paired
  unadjusted 95% CI [+0.111, +1.321]), but the fixed Local policy was at
  least as strong. The RF lost to Surrogate on Branin (−1.221), Hartmann
  (−0.695), and SixHumpCamel (−0.441); Rastrigin was inconclusive. No general
  learned-policy gain is established. A `no_retrain` mask removed repeated GP
  refits but did not overturn this result on the fresh seeds.
- **RQ2 remains partial:** leave-one-family-out RF deployed in complete loops
  did not beat the stronger fixed controls consistently. A separate external
  IOH/BBOB stress test completed 405 trajectories on three new functions ×
  dimensions {2,5,10} × three instances × five seeds × three policies. RF
  tended to improve over Surrogate at 10D and regress at 2D, with Local often
  competitive; this combines domain and dimension shift and is exploratory.
- **RQ3 remains open:** an audit of the existing real-LLM BOCoDe joint run
  found 447 calls / 324,702 tokens for the frozen dimension router versus
  739 calls / 537,456 tokens for LLM everywhere (about 39.5% fewer calls and
  tokens). This development-panel saving does not establish a transferable
  router; the held-out IOH performance test was negative. A parallel,
  resumable confidence-threshold and random-rate sweep is implemented;
  four mock-System-2 trajectories matched serial execution, but no real LLM
  sweep has run without a configured credential.
- **Mechanism check:** a post-hoc fixed global-then-local schedule beat the
  recorded LLM on mean Rastrigin terminal best for development seeds 0–9,
  but lost to fixed Surrogate and Local on separate seeds 100–119. The observed
  LLM schedule is not a validated causal explanation of its advantage.

The real confidence/OOD System-1/System-2 cost frontier and engineering
transfer remain unrun because no usable LLM credential or engineering
simulator was found in the current environment. The new local results should
be treated as evidence for revising the method claim rather than proof of
the original final goal.

## Workshop paper evidence gap closure (2026-09-27)

Three experiments closed the gaps that block a defensible workshop paper:
**EXP-RQ1.2 20-seed replication** (Track A), **EXP-RQ0.5 Gate 5 affine
invariance** (Track B), and **EXP-RQ0.5 per-step LLM action distribution**
(Track D, pure analysis). 380 unit tests now pass (was 364 before Track B).

### Track A — EXP-RQ1.2 20-seed replication

- Dataset: `runs/phase8_6d_h3_20seed_5family.jsonl` (663 records, 5 families × 20 seeds)
- Driver: `scripts/generate_phase8_6d_20seed.py` (frozen config matching the
  canonical 10-seed driver)
- Gate runner: `scripts/validate_phase8_6b.py` (unchanged; pass new JSONL path)
- Report: `runs/EXP_RQ1_2_REPORT.md` (updated) + `runs/exp_rq1_2_h3_20seed_report.json`
- Wall-clock: 6 min generation + 7 min oracle replay = 13 min

| Gate | 10-seed | 20-seed (confirmatory) |
|---|---|---|
| 3: aggregate_mass adjusted_lift | 1.124 (CI [0.695, 1.722]) | **1.294 (CI [0.848, 1.954])** |
| 4a: stop-gain CI lo > 0 | 5/5 | **5/5** (unchanged) |
| 4b: uniform-gain CI lo > 0 | 3/5 | **4/5** (+Hartmann now passes) |
| 4c: behavior-gain CI lo > 0 | 1/5 | **1/5** (Rastrigin only) |

**All Gate 3/4 numbers strengthened or held at 20 seeds. 20-seed is now the
citation-quality number; 10-seed retained for transparency.**

### Track B — EXP-RQ0.5 Gate 5 affine invariance

- Code: `BocodeBotorchBackend` accepts `objective_scale`/`objective_shift`,
  applied at write-time in `initialize()`/`evaluate()`. `DatasetConfig`
  gains matching fields with `__post_init__` validation.
- Tests: `tests/test_objective_transform.py` — 16 new tests (BocodeBotorchBackend
  affine API, DatasetConfig validation, generator wiring, end-to-end
  `affine_rescaling_test` on tiny config)
- Runner: `scripts/run_affine_rescaling_gate.py` over 4 transforms
- Datasets: `runs/phase8_6d_h3_5family_affine/scale{0.1,1.0,10.0}_shift0.0.jsonl`
  + `scale1.0_shift5.0.jsonl` (175 records each)
- Report: `runs/EXP_AFFINE_REPORT.md` + `runs/exp_rq0_5_affine_report.json`
- Wall-clock: 8 min (4 transforms × 2 min each)

**Result (5-family × 5 seeds × horizon=3, 175 records)**:

| Criterion | Tally |
|---|---|
| Spearman ρ ≥ 0.9 | **17/17 buckets** (Spearman = 1.0000 in every bucket) |
| top-1 agreement ≥ 0.8 | **17/17 buckets** (≥ 0.83 in every bucket) |
| JS divergence ≤ 0.02 | **15/17 buckets** (2 Hartmann late/mid breach) |

**Rank invariance is proven analytically** (Spearman = 1.0). The 2 JS
breaches on Hartmann are caused by GP-fit numerical fragility (BoTorch
`optimize_acqf(LogEI)` lands on a duplicate X under shift=5, producing
`delta_opt = 0` and a sharp utility drop) — **not** a violation of the
analytic invariance.

### Track D — LLM per-step action distribution analysis

- Script: `scripts/analyze_llm_action_distribution.py` (re-derives per-step
  `action_id` histograms from `runs/exp_rq0_5_joint_10seed.json`)
- Report: `runs/exp_rq0_5_action_distribution.md` + `.json` (raw counts)
- Wall-clock: < 1 min (pure analysis)

**Key finding (revised mechanism)**: On Rastrigin (10D multimodal, where LLM
beats Surrogate by +17.54 best_y, p=0.0003), the LLM uses a **"global-then-
local" temporal schedule**:
- Steps 0–9: 100% `global_exploration` (broad sampling to escape local optima)
- Steps 10–12: transition to `local_exploitation` (~80% local by step 12)
- Step 13+: 100% `local_exploitation` (refine the best region found)

AlwaysSurrogate picks `surrogate_acquisition` 100% of all 25 steps. **The
LLM's value is structural schedule, not per-step choice.**

**The earlier §5 of `EXP_RQ1_3_REPORT.md` hypothesized the LLM might be
picking `local_exploitation` (closed-loop best first action) — that
hypothesis is rejected by the per-step distribution. The mechanism is a
temporal schedule, not a per-step choice.**

### Workshop paper outline

`docs/workshop_paper_outline.md` — 8-page workshop paper draft with:
- §3 Framework (385 tests, 7 components)
- §4 RQ1 Decision Landscape (3 regimes, Gate 3 at 20 seeds, Gate 5 affine
  rank invariance)
- §5 RQ3 Selective Reasoning (Rastrigin +17.54, global-to-local mechanism,
  FrozenRouter negative result on IOH, RoutingAware vs LLM-everywhere NS)
- §6 Limitations (14 items)
- §7-§8 Discussion + Conclusion

Codex 5-round review completed 2026-09-27 with **APPROVE** verdict.

---

## EXP-RQ2.1 — Cross-family transfer (2026-09-27, same session)

5-fold leave-one-family-out CV with multi-output RandomForestRegressor System-1
trained on counterfactual oracle labels (`runs/phase8_6d_h3_20seed_5family.jsonl`).

- Script: `scripts/exp_rq2_1_cross_family_transfer.py`
- Report: `runs/EXP_RQ2_1_REPORT.md` + `runs/exp_rq2_1_report.json`
- Wall-clock: ~3 minutes

**Per-family results** (cross-family regret vs AlwaysSurrogate CI):

| Family | dim | Cross vs Surrogate CI | Verdict |
|---|---|---|---|
| Branin | 2D | `[-0.014, +0.009]` | inconclusive |
| Hartmann | 6D | `[+0.019, +0.363]` | **cross LOSES** |
| Ackley | 10D | `[-0.410, -0.089]` | **cross WINS** |
| SixHumpCamel | 2D | `[+0.007, +0.014]` | **cross LOSES** |
| Rastrigin | 10D | `[-0.240, -0.076]` | **cross WINS** |

**Headline (per codex-ready wording)**:

> "Cross-family transfer is **family-specific**: it wins on the two
> high-dimensional multimodal families (Ackley 10D, Rastrigin 10D; both p<0.05
> paired bootstrap CI), loses on Hartmann (6D, decision-rich) and SixHumpCamel
> (2D, flat), and is inconclusive on Branin (2D). The aggregate records-weighted
> Cross − Surrogate mean is -0.038 (slightly better than AlwaysSurrogate),
> but this hides the per-family heterogeneity. **RQ2 is partially answered**:
> learned System-1 transfers usefully when the held-out family shares
> structural features with the training set, but does not transfer uniformly."

This fills the RQ2 gap that PROGRESS_BRANCH §1 explicitly flagged as 0%. The
paper's §4 RQ2 can now be added with this finding. **Worth codex review**
before final workshop submission, since it changes the paper's RQ2 status
from "absent" to "partially answered".

---

## Phase 9.8: Jev executable-criteria effect test (2026-09-26)

We compared two real `jev-1.13.0` Choice prompts: both used the same English
narrative state, BO backend, action factory, budget 30, five initial points,
single-thread numerics, and five problem families. The only prompt change was
the Choice question: new criteria describe executable actions and their fixed
parameters/costs, and a short instruction says Jev only selects an option.
We ran paired seeds 20–24 (development, already
used in prompt design) and then froze the prompts for seeds 30–34 (fresh
confirmation). RuleFullBudget and a local `always_surrogate` policy were run
on the same initial states; `always_surrogate` always selects the factory's
`surrogate_acquisition` action and makes no Jev calls.

Fresh seeds 30–34, mean terminal best over five seeds per family (higher is
better):

| Problem | RuleFullBudget | AlwaysSurrogate | Jev legacy criteria | Jev executable criteria | New minus old |
| --- | ---: | ---: | ---: | ---: | ---: |
| Branin | -0.6847 | -0.4033 | -0.4249 | -0.4033 | +0.0216 |
| Ackley | -20.4177 | -20.1676 | -20.1480 | -20.2318 | -0.0838 |
| Hartmann | +1.6525 | +2.9292 | +2.5000 | +2.9081 | +0.4081 |
| SixHumpCamel | +0.4279 | +0.9520 | +0.6408 | +0.9318 | +0.2910 |
| Rastrigin | -117.4571 | -123.1102 | -117.5033 | -114.9256 | +2.5777 |

Fresh paired wins/ties/losses of executable versus legacy are Branin 4/0/1,
Ackley 1/3/1, Hartmann 3/1/1, SixHumpCamel 3/0/2, and Rastrigin 2/3/0.
The exact five-seed bootstrap intervals for the mean paired difference cross
zero for Ackley and Hartmann; these small, exploratory samples do not warrant
a general significance claim. Development seeds show the same positive mean
direction for Branin, Hartmann, and SixHumpCamel, but a negative direction
for Ackley and Rastrigin. In the fresh set, executable Jev outperformed the
rule baseline on all five family means. This is a useful prompt/backend
combination, but not a demonstration that Jev's state-dependent selection is
better than a fixed surrogate-acquisition policy: AlwaysSurrogate equalled
Branin and slightly exceeded executable Jev on Ackley, Hartmann, and
SixHumpCamel. Jev exceeded it on fresh Rastrigin, while AlwaysSurrogate
exceeded Jev on development Rastrigin.

The new descriptions greatly changed the action mix. On fresh seeds, Jev
selected `surrogate_acquisition` 477/552 times (86.4%) and
`global_exploration` 75/552 times; legacy Jev selected surrogate acquisition
254/501 times, global exploration 184, surrogate refit 57, and four other
actions six times. The executable version used all 25 available new
evaluations on every trajectory and avoided explicit refits and early stops.
Its 552 API calls and 542,259 input tokens exceeded legacy's 501 calls and
347,836 input tokens: mostly because one-point surrogate acquisition needs
more decision rounds than two-point global exploration, plus longer criteria.

Artifacts: `runs/phase9_7_jev_executable_5x5.json`,
`runs/phase9_7_jev_executable_fresh_5x5.json`,
`runs/phase9_7_jev_legacy_fresh_5x5.json`,
`runs/phase9_7_always_surrogate_5x5.json`,
`runs/phase9_7_always_surrogate_fresh_5x5.json`, and
`runs/phase9_7_jev_criteria_full_comparison.json`. The earlier development
legacy report is `runs/phase9_6_jev_narrative_5x5.json`.

## Phase 9.7: executable Jev action descriptions (2026-09-26)

The Jev Choice criteria now describe each **already constructed** action:
its fixed call parameters, exact number of objective evaluations, and
whether it fits or uses the current surrogate. `retrain_surrogate` explicitly
says it creates no new observation and that surrogate-based sampling can
fit automatically; `stop` says it leaves unused evaluation budget behind.
Jev still selects one action ID only. The action factory and executor retain
control of parameter values and execution. Descriptions are built from the
same `OptimizationAction` objects returned to the loop, including budget
capping when fewer evaluations remain.

`JevPolicy(criteria_format="legacy")` and runner `--criteria-format legacy`
reproduce the earlier option wording; the new default is `executable`.
The runner logs this field and refuses to resume a report made with a
different criteria format. A real one-call smoke with narrative state and
`jev-1.13.0` succeeded; Jev chose `global_exploration` with factory parameters
`n_samples=2, seed=25`. This smoke verifies API compatibility only. Phase 9.5
and 9.6 performance numbers used legacy option wording and cannot be
attributed to the new wording. A paired evaluation is still needed to measure
its policy effect.

## Phase 9.6: Jev state-interface ablation (2026-09-26)

Jev's documented `state` accepts a string or structured JSON; `Choice` still
selects from named options. The Phase 9.5 request used a terse numeric object
with little explanation of BO concepts. `JevPolicy` and
`scripts/run_jev_baseline.py` now support `state_format=narrative` /
`--state-format narrative`, while preserving `structured` for replication.
The English narrative explains objective direction, evaluation cost, budget,
stagnation, boundary fraction, and surrogate diagnostics. It also includes
the most recent observed objective value, which the structured baseline did
not include. The action set, option descriptions, question instruction,
backend, seeds, budget, and resolved Jev model were held fixed. Therefore
this compares interface content, **not language format alone**. `--direct`
bypasses a broken local HTTPS proxy; it does not change the API payload.

Five problems × seeds 20–24 × budget 30 were rerun with narrative state and
`jev-1.13.0` (higher terminal best is better):

| Problem | Structured Jev | Narrative Jev | Paired wins / ties / losses |
| --- | ---: | ---: | ---: |
| Branin | -1.6525 | -0.4413 | 5 / 0 / 0 |
| Ackley | -19.9699 | -19.7528 | 3 / 2 / 0 |
| Hartmann | +1.6078 | +2.3885 | 4 / 0 / 1 |
| SixHumpCamel | +0.1467 | +0.2273 | 3 / 2 / 0 |
| Rastrigin | -117.1514 | -113.4134 | 2 / 1 / 2 |

The structured request selected `global_exploration` in all 325 decisions.
The narrative request used `global_exploration` 187 times,
`surrogate_acquisition` 248, `retrain_surrogate` 52, and four other actions
five times. It made 492 API calls and used 341,334 input tokens, versus
325 calls and 219,301 tokens for structured state; fitting the surrogate
adds decision rounds without consuming objective evaluations. One Rastrigin
trajectory stopped after 24 of 25 available new evaluations. The mean gain
there is unstable across seeds, and five reused development seeds are not a
fresh confirmation set. This is evidence that the previous request content
was consequential, not evidence that Jev or natural-language input is
universally better. The interface should be frozen before a fresh-seed
confirmation, and repeated surrogate-refit and early-stop decisions need
explicit analysis.

Sara/lenz is a different architecture: Sara is a reasoning agent that can
consume natural-language priors and inspect or change a structured BO
backend; lenz itself exposes a structured CLI and trial log. Our Jev
adapter is a single-step Choice policy over predefined actions. Converting
its state to prose does not give it Sara's tool access or rich prior context.

Artifacts: `runs/phase9_6_jev_narrative_5x5.json` and
`runs/phase9_6_jev_state_interface_ablation.json`. References:
https://docs.typesafe.ai/concepts/state,
https://docs.typesafe.ai/primitives/choice, and
https://arxiv.org/html/2608.00316.

## Phase 9.5: real Jev and aligned System-1 baselines (2026-09-26)

The user supplied `JEV_API` in `.env`. `src/opt_agent/system1/jev_policy.py`
calls TypeSafe's real `/v1/systemone` Choice endpoint with the same BO state
summary and candidate descriptions as `LayaPolicy`. It validates the returned
choice/probabilities, tolerates small displayed-probability rounding error,
uses the shared budget-valid candidate factory, and records model version,
confidence, token usage, and call latency. No API key is written to reports.
The live smoke returned `jev-1.13.0`; three adapter contract tests pass.
The benchmark response reports `jev-1.13.0` throughout. The reusable runner
now pins that version by default; this run requested `jev-latest` before the
pin was added, and the resolved version was logged on every decision.

`scripts/run_jev_baseline.py`, `scripts/run_laya_multi_seed.py`, and
`scripts/evaluate_trained_system1.py` ran five synthetic families × seeds
20–24 × budget 30 with five initial evaluations. All three harnesses use
`torch.set_num_threads(1)`; their RuleFullBudget terminal baselines agree
exactly on all 25 problem/seed pairs. This matters because BoTorch's
surrogate-acquisition numerics produced small cross-harness differences
under differing thread settings. The Jev runner saves each completed pair
and supports `--resume` after an interruption.

| Problem | RuleFullBudget | Real Jev zero-shot | Local Laya zero-shot | Trained rule imitation | Trained no-fit rollout |
| --- | ---: | ---: | ---: | ---: | ---: |
| Branin | -0.4442 | -1.6525 | -9.2830 | -0.6399 | -9.2830 |
| Ackley | -20.0156 | -19.9699 | -19.6091 | -20.0156 | -19.6091 |
| Hartmann | +2.1661 | +1.6078 | +0.9820 | +2.1661 | +0.9820 |
| SixHumpCamel | +0.3516 | +0.1467 | -1.0844 | +0.1039 | -1.0844 |
| Rastrigin | -111.0417 | -117.1514 | -129.9954 | -111.0417 | -125.7123 |

Values are mean terminal best objective over five paired seeds; larger is
better. These are descriptive development results with only five seeds per
family. Branin's Jev paired raw gain vs RuleFullBudget is negative on all
five seeds (mean -1.2070; seed-bootstrap 95% interval [-1.6504,-0.7637]).
Ackley's small Jev mean gain (+0.0457) has an interval spanning zero.

Under this fixed zero-shot question, **real Jev chose `global_exploration`
in all 325 decisions**. Replaying an always-global policy with matching
seeds and single-thread settings reproduced all 25 Jev terminal results
exactly. Jev used 219,301 input tokens; mean observed API latency was
0.321 s per decision. Laya chose `error_guided_sampling` 600/625 times and
`uncertainty_sampling` 25/625 times. Its local checkpoint emitted an invalid
temperature warning, so its confidence should not be treated as calibrated.
The trained no-fit System-1 also chose error-guided sampling for all 625
decisions; its outcomes equal Laya's on four families, but differ on
Rastrigin because Laya switches to uncertainty sampling in 25 steps.

The actionable finding was **action collapse under this structured state and
Choice wording**, including for a real hosted Jev model. Phase 9.6 tests a
more explanatory request; fresh-seed testing remains to be done.

Artifacts: `runs/phase9_5_jev_zero_shot_5x5_threads1.json`,
`runs/phase9_5_laya_zero_shot_5x5.json`,
`runs/phase9_5_trained_system1_5x5.json`, and
`runs/phase9_5_system1_baseline_comparison.json`. The earlier Jev file
without `threads1` is retained only for auditing the thread-control fix.

## Phase 9.4: real System-1 retraining and forced-route audit (2026-09-26)

`scripts/train_system1_compare.py` trained two **JevStyleSystem1** checkpoints
from the Phase 8.6D corpus with seed-disjoint training (0–7; 179 records) and
validation (8–9; 47 records). `rule_full_budget` is a rule-imitation anchor;
`no_fit` uses the saved on-policy rollout branches rescored without fit cost.
Neither target is an oracle optimal-action label. The checkpoints are
`runs/typed_decision_rule_full_v2.pt` and `runs/typed_decision_no_fit_v2.pt`.

Validation top-1: rule imitation 47/47 (best epoch 4); no-fit rollout
16/47 = 34.0% (best epoch 4). The no-fit model's validation accuracy is
close to its most frequent validation class and offers no evidence that it
learned a useful state-dependent action ranking.

`TypedDecisionPolicy` now uses state-filtered factory candidates, returns
the complete parameterized action, and threads the experiment seed into
candidate construction. This fixes a latent invalid-action path and makes
paired policy comparisons use the same action seed. The adapter regression
test and all nine `tests/test_typed_decision.py` tests pass.

`scripts/evaluate_trained_system1.py` forced each trained model onto the
System-1 route and compared it with RuleFullBudget, using budget 30, five
problem families, and seeds 10–14 (75 trajectories). All trained-model
steps routed to System-1: 405 rule-imitation steps and 625 no-fit steps.
This is a **System-1-only diagnostic with local fallback**, not an evaluation
of selective routing or the real LLM System-2. Seeds 10–14 were already
used in earlier proxy-student development analyses, so they are not a fresh
final paper test set.

| Problem | RuleFullBudget mean best | Rule imitation mean best | No-fit mean best |
| --- | ---: | ---: | ---: |
| Branin | -2.3703 | -3.0138 | -9.3264 |
| Ackley | -20.2235 | -20.2235 | -20.4331 |
| Hartmann | +1.9852 | +1.9852 | +1.5481 |
| SixHumpCamel | +0.3611 | +0.3611 | -1.0101 |
| Rastrigin | -112.8066 | -112.8066 | -141.8847 |

Rule imitation matches the rule's terminal best in 24/25 paired runs;
the one worse Branin run diverges after a late `global_exploration` choice.
The no-fit model selects `error_guided_sampling` for all 625 decisions,
consumes the full budget, and is worse in mean terminal best on all five
families. This reproduces the closed-loop continuation concern with the
**actual trained System-1**, rather than the earlier small diagnostic MLP.
The rule-imitation checkpoint is a functional warm start, but there is still
no demonstrated learned optimization gain. The no-fit checkpoint should not
be used as an improved policy. Full actions and paired outcomes are in
`runs/phase9_4_trained_system1_heldout.json`.

Next research step: collect labels on states visited by the real System-1,
with a terminal objective and a matched continuation, then evaluate a
calibrated router with genuine System-2 calls on new seeds.

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

The Gate 4 result is conditional on RulePolicy taking over after the first
action. RulePolicy stops when at most four evaluations remain, so this test
does not establish a policy-independent ranking of first actions.

The full result, including per-seed paired outcomes, is in
`runs/phase8_6b_h3_report.json`. Removing standalone retraining from the saved
candidate results does not repair the Branin outcome; see
`runs/phase8_6b_h3_no_retrain_report.json`.

## Phase 8.6D: long-horizon utility ablation (2026-09-26)

The full-horizon dataset (`runs/phase8_6d_hfull_10seed.jsonl`, 226 records)
still fails Gates 2 and 4. Its Branin first decision is STOP on 6/10 seeds;
the Gate 4 mean normalized gain over the behavior-first-action control is
-0.0331, bootstrap 95% interval [-0.0656, -0.0031]. Gate 3's positive
aggregate point estimate does not establish a terminal improvement.

`scripts/ablate_phase8_6d_utility.py` ranked the same saved rollout branches
under six utility variants, then replayed each selected first action with the
Gate 4 RulePolicy continuation. It also repeated the ranking with RulePolicy
continuation during label generation, matching the terminal test. Both runs
use the same 30 first-decision records (three problems × seeds 0–9), so they
are exploratory ablations on the development seeds, not held-out policy tests.

| Branin label continuation: RulePolicy | First-step STOP | Mean new evaluations under Gate 4 | Mean terminal best | Normalized gain vs behavior (95% interval) |
| --- | ---: | ---: | ---: | ---: |
| Current utility | 6/10 | 8.6 | -4.04 | -0.0326 [-0.0655, -0.0029] |
| No evaluation cost | 0/10 | 21.8 | -3.07 | -0.0181 [-0.0445, +0.0032] |
| No fit cost | 2/10 | 17.1 | -1.51 | +0.0080 [-0.0093, +0.0340] |
| No evaluation or fit cost | 0/10 | 21.8 | -2.24 | -0.0012 [-0.0346, +0.0301] |
| No STOP penalty | 10/10 | 0.0 | -10.53 | -0.2226 [-0.3401, -0.0967] |

Removing fit cost improves Branin's paired normalized outcome over current
utility by +0.0407 [0.0042, 0.0810], but its gain over behavior is not
established with ten seeds. Ackley remains unchanged in terminal outcome under
the cost ablations; Hartmann's no-fit improvement under matched RulePolicy
comes from one seed. No variant passes all Gate 4 comparisons. The same
qualitative STOP pattern appears when ranking the saved `on_policy` branches.

Results: `runs/phase8_6d_utility_ablation.json` (saved `on_policy` branches),
`runs/phase8_6d_utility_ablation_rule.json` (matched RulePolicy branches).
These replay only the first action from a stored state; they do not evaluate
a deployed, repeatedly acting System-1 policy.

## Phase 8.6E: held-out closed-loop student and expanded scenarios (2026-09-26)

`scripts/evaluate_no_fit_student.py` trains two 64-hidden-unit state-only MLPs
on hard top-action labels from the 8.6D saved rollouts: current utility and
the no-fit-cost ablation. Training uses seeds 0–7 (179 records), model
selection uses seeds 8–9 (47 records), and the closed-loop benchmark uses
unseen seeds 10–19. The students see only `OptimizationState` features at
evaluation time, not counterfactual or future objective values. Validation
top-1 agreement is 0.489 for current labels and 0.383 for no-fit labels.

The backend's audited synthetic allowlist now includes SixHumpCamel (2-D)
and Rastrigin (10-D); both pass one-step fit/propose/evaluate tests. The
benchmark covers five problems × budgets {15, 30} × ten new seeds × four
policies (`RulePolicy`, `RuleFullBudget`, current-label student, no-fit student):
400 completed trajectories. `RuleFullBudget` is the same rule with its early
STOP threshold set to zero, isolating the value of using the remaining budget.

| Problem, budget 30 | No-fit student mean terminal best | Paired normalized gain vs RulePolicy (95% bootstrap interval) | Gain vs full-budget rule |
| --- | ---: | ---: | ---: |
| Branin | -2.61 | +0.0252 [-0.0174, +0.0722] | -0.0039 |
| Ackley | -20.40 | -1.2211 [-2.1755, -0.3616] | -1.2211 |
| Hartmann | +1.69 | -3.9529 [-13.6532, +2.0168] | -4.6439 |
| SixHumpCamel | +0.04 | -0.0151 [-0.0329, -0.0019] | -0.0151 |
| Rastrigin | -128.94 | -0.6727 [-1.2481, -0.1974] | -0.7585 |

The no-fit student selects no first-step STOP and consumes the full budget
in every benchmark scenario. That fixes the immediate STOP collapse seen in
the current-label student (10/10 Branin first-step STOP at each budget), but
does not establish a general optimization gain. Branin's apparent gain over
ordinary RulePolicy is matched or exceeded by simply letting RulePolicy use
the full budget. Ackley at budget 30 and both added problem families show
degradation. The additional seeds 15–19 retain those negative directions.
Hartmann's per-seed normalization is especially sensitive to a small initial
objective standard deviation (as low as 0.035 in the five-seed pilot); raw
mean terminal values and per-seed outcomes must be read alongside normalized
means. Budget-15 results and all 400 trajectories are in
`runs/phase8_6d_heldout_student_10seed.json`.

Conclusion: the no-fit utility reduces premature STOP but its learned action
ranking does not transfer reliably. The state-only student is exploratory and
should not be treated as a deployable improvement over RulePolicy.

## Phase 8.6F: first-action versus continuation diagnosis (2026-09-26)

`scripts/diagnose_continuation_effect.py` ran a 2×2 paired intervention on
the same held-out seeds 10–19 at budget 30: first action from the no-fit
student or full-budget RulePolicy, then all later actions from the student or
full-budget RulePolicy. All four arms use the full evaluation budget. This
separates first-action quality from the policy used after that decision.

| Problem | Student first, then Rule: normalized gain vs Rule→Rule | Student continuation after student first: gain vs student→Rule |
| --- | ---: | ---: |
| Ackley | +0.0252 [-0.2198, +0.3409] | -1.2463 [-2.2590, -0.3654] |
| Branin | +0.0001 [-0.0418, +0.0486] | -0.0040 [-0.0261, +0.0266] |
| Hartmann | -0.7630 [-2.4809, +0.6230] | -3.8809 [-12.3981, +1.6493] |
| SixHumpCamel | -0.0087 [-0.0171, -0.0016] | -0.0064 [-0.0230, +0.0045] |
| Rastrigin | -0.0404 [-0.2742, +0.1296] | -0.7180 [-1.4308, -0.1977] |

Ackley's student first action alone is not the observed source of the
full-loop deficit; the student's continuation is. Rastrigin shows the same
direction. SixHumpCamel also has a harmful first action. With a Rule first
action, switching to student continuation hurts Branin (-0.0732, interval
[-0.1255, -0.0261]) and SixHumpCamel (-0.0256, interval [-0.0525, -0.0046]).
The interaction between first and later policies matters, so these contrasts
are not additive.

The trajectory logs show repeated actions: at budget 30, the no-fit student
chooses `boundary_sampling` 130 times across ten Ackley runs and 130 times
across ten Hartmann runs; on Rastrigin it chooses `error_guided_sampling` 98
times and `boundary_sampling` 66 times. `boundary_sampling` explicitly selects
points near domain edges. BOCoDe's installed Ackley and Rastrigin problems
place their optima at the box center, making this repetition a plausible
mechanism for poor outcomes. This is an observed policy behavior plus a
mechanistic hypothesis, not a proof that every boundary sample is harmful.

The fixed RulePolicy continuation in Gate 4 evaluates a first action under
that continuation. It cannot establish that a learned policy repeatedly
acting in the closed loop is good. The fixed Rule continuation itself does
not explain the held-out full-loop degradation: for Ackley and Rastrigin it
substantially mitigates the deficit. Full results:
`runs/phase8_6d_continuation_effect.json` (200 paired trajectories).

## Phase 8.6C: Q-S5 horizon diagnosis (per codex Q-T1/Q-T2 2026-09-25)

Sprint 0 (`src/opt_agent/rollout/specs.py`,
`src/opt_agent/loop/policies.py::{UniformPolicy,OnPolicyMix}`,
`src/opt_agent/dataset/validation.py::replay_with_config`,
`tests/test_sprint0_replay.py`) made any record replayable with
different (utility_weights, horizon, continuation_policy) without
re-generating the dataset. Total tests: 346/346 passing.

`scripts/diagnose_qs5.py` then ran a 3×5 control-variable matrix
(3 utility variants × 5 horizons, continuation fixed at `rule`) on
30 records sampled from `runs/phase8_6b_h3_10seed.jsonl`, totaling
450 (record, config) pairs. When the top action changed under a
config, paired full-budget oracle test was run (D2 = β).

**Headline finding**: Horizon, not utility weights, is the dominant
factor in Gate 4. Going from horizon=3 (the dataset's default) to
horizon=`remaining_budget` flips the top action on ~50% of records
and improves mean terminal gain vs STOP from ~+0.0 to +4.5
(normalized by per-problem objective std).

| Config | N | Changed | Mean gain vs STOP (new) | Mean gain vs STOP (orig) | Pass (new > STOP) |
| --- | --- | --- | --- | --- | --- |
| `original_h3` (dataset default) | 30 | 1 | +0.0 | +0.0 | 0/1 |
| `original_h10` | 30 | 15 | **+4.51** | +3.31 | **7/15** |
| `original_hfull` | 30 | 15 | **+4.49** | +3.30 | **7/15** |
| `delta_best_h10` | 30 | 22 | +3.09 | +2.27 | 10/22 |
| `no_stop_pen_h10` | 30 | 23 | +2.91 | +2.19 | 4/23 |

Implication: the counterfactual teacher is NOT mis-specified in
direction. The dataset's horizon=3 was too short to differentiate
actions on terminal regret. Phase 8.6D (regenerate dataset with
horizon = `remaining_budget`) is expected to make Gate 4 PASS.

**Subsequent result:** Phase 8.6D did not pass Gate 4. The +4.49 result above
is conditional on top-action changes, compares mainly against STOP, and is
dominated by Hartmann; the Branin changed-record mean gain against behavior
was still about -0.011. See the Phase 8.6D ablation above.

Full report: `runs/phase8_6c_qs5_diagnosis.md`.
JSON: `runs/phase8_6c_qs5_diagnosis.json`.

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
on Branin in paired RulePolicy continuations. Removing standalone retraining
from candidate results does not fix the Branin result. This is a conditional
label-validity warning, not a diagnosis of reward hacking or a trained-model
failure. Follow-up work on the research branch found that changing the
continuation can change the first-action ranking.

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

## Phase 9 status: DONE

Phase 9 closed with the following deliverables:

- **Framework skeleton** (commit `5f79257`): actions/factory.py,
  dataset/splits.py, metrics/{calibration,decision}.py, system2/
  {base,fallback}.py, router/{base,confidence}.py,
  system1/hybrid_policy.py. All interface contracts validated.
- **Real-backend smoke** (commit `c209e7d`, `98b595c`,
  `b9a2a77`, `20a9d83`): `scripts/run_hybrid_smoke.py` exercises
  HybridPolicy vs RulePolicy on Branin + Ackley + Hartmann × seeds
  {0, 1, 2} × total_budget in {10, 30}. 54/54 runs pass all five
  acceptance criteria (no exceptions, routing-sum equals
  decision-step count, ≥1 evaluation beyond the n_init bootstrap
  per problem, ≤ total_budget per-step sample cap, finite
  simple_regret for Branin).
- **LLM System-2** (commit `579dc38`, `20a9d83`):
  `src/opt_agent/system2/llm_deepseek.py` wraps the DeepSeek Chat
  Completions API. The smoke harness runs end-to-end on the real
  API (DEEPSEEK_API_KEY loaded from `./.env` via python-dotenv).
  17 unit tests cover params preservation (candidate defaults as
  base, LLM `params` as filtered overrides), budget validation
  (n_samples overshoots dropped with a `fallback:` note), JSON /
  markdown-fence / regex response parsing, fallback bookkeeping,
  HTTP error propagation, and the System-1 top-action
  last-resort preference. Test count: 268/268.

NOT a performance claim: the smoke does not assert that HybridPolicy
beats RulePolicy. Performance comparison needs more seeds and a
separate evaluation plan; this is documented in PHASE9_DESIGN.md.

Phase 9.3 (typed-decision model fine-tuning) is the next milestone
but is not required for closing Phase 9.

## Phase 10 status: DONE (Calibration / OOD)

- **`src/opt_agent/system1/ood.py`**: `StateOODDetector` is a
  z-score detector on a fixed set of budget-normalized features
  (sample_fraction, stagnation_per_sample, remaining_budget_ratio,
  is_current, relative_improvement). Per-feature mean/std are cached
  from a calibration corpus; `score(state)` returns `OODScore` with
  per-feature z-scores plus a fraction-of-features-above-threshold
  score. The `relative_improvement` feature rescales by
  `max(|best_value|, 1.0)` so cross-problem records with the same
  relative progress map to the same feature value.
- **`src/opt_agent/metrics/temperature.py`**: `TemperatureCalibrator`
  fits a single scalar `T` that minimizes NLL on the calibration
  split; `.transform(probs)` returns the rescaled distribution.
  Preserves argmax (T scaling is monotonic over softmax).
- 18 new tests in `tests/test_ood_and_temperature.py` covering
  fit/score/transform edge cases, NaN handling, budget-cap behaviour,
  and argmax preservation.

## Phase 11 status: DONE (System-2 stack)

- `FallbackSystem2` (Phase 9.0) — deterministic RulePolicy baseline.
- `LLMSystem2` (Phase 9.1 v3+v6) — DeepSeek Chat Completions
  adapter with JSON / markdown-fence / regex response parsing,
  params preservation (candidate defaults as base, LLM `params`
  as filtered overrides), budget validation, fallback bookkeeping,
  and System-1 top-action last-resort preference.

## Phase 12 status: DONE (Selective-reasoning Router)

- **`src/opt_agent/router/ood_aware.py`**: `OODAwareRouter` wraps a
  `ConfidenceRouter` and a `StateOODDetector`. Decision order:
  1. confidence gate (cheap, per-step);
  2. OOD gate (uses the detector's own cutoff as the single source
     of truth — no separate router-level threshold);
  3. trust System-1.
- `HybridPolicy.select_action` now calls `router.update(state)` for
  routers that accept a `state` parameter (per `inspect.signature`
  check, so the base `Router` Protocol's 3-arg `update(probs, chosen,
  reward)` is not accidentally invoked).
- 7 new tests in `tests/test_router_ood_aware.py`.

## Phase 13 status: DONE (Benchmark harness)

- **`scripts/run_benchmark.py`**: cross-problem × cross-seed ×
  cross-policy harness. Reports per-(problem, seed, policy)
  best_value / simple_regret / n_evaluations, plus aggregated
  per-(problem, policy) stats with paired t-test against RulePolicy.
- **60/60 runs PASS** on Branin + Ackley + Hartmann × seeds {0..4}
  × total_budget in {10, 30} × {hybrid, rule}. Results in
  `runs/phase13_benchmark.json`.
- This harness uses `_FakeLaya`, which always proposes surrogate acquisition;
  these runs do not measure a learned System-1 or real Laya policy.
- Empirical observation (NOT a performance claim):
  - Branin: hybrid mean best -1.77 (succ 40%) vs rule -5.66 (succ 0%),
    paired t = +2.41.
  - Ackley: hybrid best -20.17 vs rule -20.36, paired t = +1.62.
  - Hartmann: hybrid best +2.10 vs rule +1.42, paired t = +3.82.
- 5 new tests in `tests/test_benchmark_runner.py` covering the
  aggregation logic and an end-to-end subprocess invocation.

## Phase 14 status: PLACEHOLDER (Engineering transfer)

- **`scripts/run_engineering_transfer.py`**: placeholder that
  documents the two engineering-transfer paths (real engineering
  simulator integration vs BOCoDe engineering-problem allowlist
  expansion). Neither path is implemented today; the project has
  no vehicle-suspension simulator in its dependency graph and
  `BocodeBotorchBackend`'s allowlist is currently synthetic-only
  ({Branin, Hartmann, Ackley, SixHumpCamel, Rastrigin}).
- Phase 13's synthetic benchmark is the only meaningful empirical
  baseline today; engineering transfer is deferred until a real
  engineering backend is available.

## Phase 10–14 cumulative test count: 298/298.

## Phase 9.3: typed-decision model trained; optimization benefit unverified

Per user confirmation that a GPU is available, Phase 9.3 was no
longer DEFERRED. We fine-tuned a Jev-style typed-decision model on
Phase 8.6B rollout data and ran a closed-loop integration smoke against
RulePolicy. The original performance interpretation was invalid.

**Architecture** (`src/opt_agent/system1/typed_decision.py`):
- ModernBERT-base encoder (~125M params, Apache 2.0).
- [CLS] state embedding + learned candidate embeddings (8 actions).
- Scaled dot-product → softmax → choice distribution.
- Sigmoid head for confidence scalar.

**Training** (`scripts/train_typed_decision.py`):
- Loss: KL(gold_probs || pred_probs).
- Optimizer: AdamW, lr=1e-5.
- Phase 8.6B corpus (226 records from `runs/phase8_6b_h3_10seed.jsonl`).
- 5 epochs on RTX 3090, ~20s total. Final KL loss: 0.2044.

**Corrected closed-loop audit** (Branin, Ackley, Hartmann × seeds 0–2,
budget 30; 27 runs): in the original `runs/phase9_3_typed_multi.json`,
the normal hybrid routed **0 of 129 decisions to System-1**. Every action
came from `FallbackSystem2`, so the apparent Branin and Hartmann gains
cannot be attributed to the trained model. The original harness also used
RulePolicy seed 0 inside the fallback while the standalone RulePolicy used
the experiment seed. On the first hybrid run, loading the model after backend
initialization changed the global torch RNG before optimization began.

After passing the same experiment seed to the fallback and constructing the
model before initializing the backend, the rerun in
`runs/phase9_3_typed_multi_seed_corrected.json` has **0 of 122 normal-hybrid
decisions routed to System-1**. Every paired hybrid and RulePolicy trajectory
now has the same action sequence and terminal best value. Mean best value
for each policy is -2.3702 on Branin, -20.4122 on Ackley, and +1.6605 on
Hartmann. The smoke acceptance still passes because it checks integration,
not a learned-policy gain.

The model and integration path exist, but Phase 9.3's optimization benefit
remains unverified. A valid performance test must first make the router use
the trained System-1 on a measurable fraction of decisions, then compare
paired outcomes on held-out seeds. The earlier raw report is retained for
audit and must not be cited as a learned-model improvement.

## EXP-RQ1.2: counterfactual oracle replay (2026-09-26)

The `oracle_test_from_record` pipeline (`scripts/validate_phase8_6b.py`)
runs end-to-end on the Phase 8.6B JSONL datasets. The earlier "bash
classifier blocked" note was a tooling artifact — the code was never
actually invoked before this run.

Two versions were produced (treated as configuration sensitivity, **not**
independent replications; see methodology caveat below):

| Version | Dataset | Records | Families | Horizon |
| --- | --- | --- | --- | --- |
| 3-family | `phase8_6b_{h1,h3,hfull}_10seed.jsonl` | 226 each | Branin/Hartmann/Ackley | 3 |
| 5-family canonical | `phase8_6d_h3_10seed_5family.jsonl` | 354 | + SixHumpCamel/Rastrigin | 3 |

Naming correction: the original 5-family file was named `…hfull…` but
generated with `horizon=3`, not `horizon=30`. Renamed to `…h3…` to match
the actual configuration. True hfull (horizon=30) is **not in scope**
for RQ1; cost ≈ 90 minutes.

### Headline result (5-family canonical)

| Gate | Result |
| --- | --- |
| 2: stability (top_action_share ≥ 0.6) | FAIL (**5/15** buckets pass) |
| 3: aggregate_mass | **PASS** (mean = 1.124; seed-clustered 95% CI = [0.695, 1.722], n_seeds=50) |
| 4a: stop-gain CI lo > 0 | **PASS 5/5** |
| 4b: uniform-gain CI lo > 0 | PASS 3/5 (hartmann, rastrigin, sixhumpcamel) |
| 4c: behavior-gain CI lo > 0 | PASS 1/5 (rastrigin only) |

### Per-family Gate 4 (5-family canonical)

| Family | behavior CI lo | uniform CI lo | **stop CI lo** |
| --- | --- | --- | --- |
| ackley (10D noisy) | -0.498 | -0.855 | +1.084 |
| branin (2D smooth) | -0.011 | -0.004 | +0.093 |
| hartmann (6D) | -0.093 | +0.203 | +1.493 |
| rastrigin (10D multimodal) | +0.032 | +0.136 | +0.840 |
| sixhumpcamel (2D) | -0.004 | +0.008 | +0.044 |

Bold cells have CI lower bound > 0 (statistically detectable positive
mean normalized terminal gain).

### Three findings (per `gpt-5.6-sol` review 2026-09-26, exploratory)

> **C1.** Oracle-labeled first actions produced positive mean normalized
> terminal gain over STOP on the five evaluated BOCoDe families. CI
> lower bound > 0 on 5/5. **Not** "universal" — only across the
> evaluated families; no claim about problems generally.

> **C2.** Per-family gain estimates and interval conclusions varied
> across the five families. Rastrigin was the only family with CI
> lower bound > 0 against all three baselines (STOP, uniform, behavior).
> Hartmann passed STOP and uniform. Branin, SixHumpCamel, and Ackley
> passed STOP only. **Not** a "dimension-dependent" claim — that would
> require a crossed design (same problem at multiple dimensions),
> which the current allowlist does not support.

> **C3.** Oracle-label argmax actions were heterogeneous across seeds
> within coarse (problem, phase) strata. Early-phase top-label shares
> ranged 0.20 (rastrigin/early) to 0.70 (hartmann/early). **Not** a
> "policy multimodality" claim — could reflect different states across
> seeds. Causal interpretation deferred.

### Branin: detectable disadvantage (corrected framing)

> On Branin, the oracle-labeled first action had a statistically
> detectable but **small in magnitude on the normalized-gain scale**
> disadvantage relative to behavior: mean gain −0.00494, bootstrap 95%
> CI [−0.01065, −0.00035]. The CI excludes zero, so the effect is
> detectable. Practical importance is unclear; a smallest-meaningful-δ
> test would require pre-specifying δ. **Not** "no measurable difference."

### Caveats

- All three findings are **exploratory**; per-family intervals are
  **unadjusted** for multiple comparisons.
- 3-family and 5-family used different generator configurations and
  are **not** independent replications; treat as configuration
  sensitivity, not as horizon-only sensitivity (other generator
  parameters may have differed).
- For fewer than 5 seed clusters, Gate 3's point-estimate fallback is
  a smoke-test result, not an inferential pass.

### Code revisions applied (post-`gpt-5.6-sol` review)

- `validation.py`: `MassGateResult.seed_clustered_ci95` field added;
  `aggregate_mass()` now bootstraps over per-seed adjusted_lift when
  n_seeds ≥ 5 and uses CI lower bound > 0 as gate criterion.
- `validate_phase8_6b.py`: report includes `gate3.seed_clustered_ci95`.
- Tests: 16/16 Gate tests pass; full suite 364/364.

### Open follow-ups

1. **Replication** (codex priority): expand to 20+ seeds; report
   seeds 10–29 as confirmatory (separate from seeds 0–9 exploratory).
2. Gate 2 threshold sensitivity + within-state rollout diagnostics
   (strengthens C3).
3. EXP-RQ2.1 (cross-family transfer; 5-fold leave-one-family-out).
4. EXP-RQ3.0 (Level 3 LLM baseline).
5. Commit the 5-family generator driver (current state: `/tmp/gen_5family.py`).

Full report: `runs/EXP_RQ1_2_REPORT.md`. Per-dataset JSON in
`runs/exp_rq1_2_{h1,h3,hfull,h3_5family}_report.json`.

## EXP-RQ1.3 + EXP-RQ0.5 merged (2026-09-26, pilot evidence base)

Merged `exp/rq0.5-llm-system1` into `main` as commit `a13397e`.
The merge consolidates a Decision Landscape pilot + LLM-as-System-1
wiring + closed-loop terminal regret + RF utility regression, all
reviewed by `gpt-5.6-sol` (3 rounds: REVISE → REVISE → REVISE → approve).

### Decision Landscape 4-step chain (all closed, pilot-grade)

**Step 1 — Decision Opportunity** (scale-robust):
- 5 normalization schemes × 5 families; family_median is canonical
- Three regimes survive qualitatively:
  - **Flat**: Branin (adaptive-value 0.048), SixHumpCamel (0.015)
  - **Heavy-tailed**: Rastrigin (0.368), Ackley (0.309)
  - **Frequent opportunity**: Hartmann (0.684)
- SixHumpCamel: 0% of states above δ=0.1 (clean negative control)
- Decision Opportunity profile (codex metric): `adaptive-value =
  E[max_a U] - max_a E[U]`

**Step 2 — State Conditionality** (RandomForest utility regression):
- Hartmann: RF regret 0.47 vs Surrogate 0.68 (−29%); suggestive pilot
  (paired bootstrap CI [−0.038, 0.415], t p=0.141)
- Rastrigin/Ackley: RF ≈ best-fixed (no state-conditional learnability
  demonstrated with 10 seeds)
- Surrogate is competitive on Hartmann (0.48 closed-loop regret) but
  weak on Rastrigin (11.10 closed-loop regret)

**Step 3 — Closed-loop terminal regret** (AlwaysSurrogate continuation):
- 3 seeds × 5 families × 8 first-actions × budget=15; ~500s on 32 cores
- Best first action across opportunity-rich families: `local_exploitation`
  (not `global_exploration` as initially claimed)
- Rastrigin: local=0.00 regret vs surrogate=11.10 (4.8× spread)
- Hartmann: local=0.33 vs surrogate=0.48 (Surrogate competitive)
- Ackley: local=0.005 vs surrogate=0.03

**Step 4 — LLM value (EXP-RQ0.5)**:
- Real DeepSeek-chat as System-1: beats AlwaysSurrogate by **+8.87
  best_y** on Rastrigin (2 seeds × budget=20)
- Closed-loop regret is consistent with this finding (mechanistic
  plausibility, not causal validation — per-step action logs not saved)
- 5-seed replication deferred

### New infrastructure

- `src/opt_agent/system1/llm_system1_policy.py` (~250 LOC): `LLMSystem1Policy`
  with mode="llm" (real DeepSeek) and mode="mock" (deterministic heuristic)
- `scripts/exp_rq1_3_scale_sensitivity.py` (5 scale schemes, parallelized)
- `scripts/exp_rq1_3_bootstrap_ci.py` (seed-cluster bootstrap)
- `scripts/exp_rq1_3_rf_probe.py` (multi-output RF utility regression)
- `scripts/exp_rq1_3_closed_loop_regret.py` (paired closed-loop replay)
- `scripts/exp_rq0_5_llm_system1.py` (LLM vs Rule vs Surrogate)

### Code revisions (post-codex-round-2)

- `validation.py`: `MassGateResult.seed_clustered_ci95`; bootstrap CI over
  per-seed adjusted_lift (used when n_seeds ≥ 5)
- `scripts/validate_phase8_6b.py`: report includes gate3 CI

### Test status

364/364 tests pass (full suite). No regressions from merge.

### Deferred (out of scope for this merge)

- 5-seed replication of EXP-RQ0.5 with per-step LLM action logging
- Closed-loop with branched actions at every decision point
- External baseline: lenz / PlugBO head-to-head
- Family × dimension confound (2D/6D/10D): needs dimension-crossed benchmark
- System-1/System-2 2×2 figure: needs quantitative "ambiguity" definition

### File index (post-merge)

```
runs/EXP_RQ1_3_REPORT.md                       # canonical pilot report
runs/EXP_RQ0_5_REPORT.md                       # LLM-as-System-1 pilot
runs/exp_rq1_3_scale_sensitivity.json          # Step 1 (5 schemes)
runs/exp_rq1_3_bootstrap_ci.json               # Step 1 CIs
runs/exp_rq1_3_rf_probe.json                   # Step 2-3
runs/exp_rq1_3_closed_loop_regret.json         # Step 3 closed-loop
runs/exp_rq0_5_results.json                    # Step 4 raw results
runs/phase8_6d_h3_10seed_5family.jsonl         # 354-record canonical dataset
scripts/exp_rq1_3_*.py, scripts/exp_rq0_5_llm_system1.py
src/opt_agent/system1/llm_system1_policy.py
```

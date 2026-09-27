# Final-goal experiment protocol (working draft, 2026-09-27)

## Research claims and decision gates

The final target in `MASTER_PLAN.md` requires evidence for three distinct
claims: useful learned decisions (RQ1), transfer to unseen problems (RQ2),
and a performance/cost benefit from selective LLM reasoning (RQ3). A passing
software smoke test or one-step oracle regret is not sufficient for any of
these claims.

### RQ1: Does a learned policy improve full optimization trajectories?

1. Train the frozen 200-tree, depth-8, leaf-5 RF used in EXP-RQ2.1 on the
   five-family 20-seed rollout dataset (training seeds 0–19). Refit nothing on
   the test seeds. The policy sees only `OptimizationState` at runtime.
2. Use `scripts/exp_rq1_rq2_closed_loop.py` for `rf_all`, `RuleFullBudget`,
   `AlwaysSurrogate`, `AlwaysLocal`, and the fixed 10-global-then-local
   schedule. Pair policies on identical initial Sobol evaluations and budget.
3. Primary endpoint: paired terminal best objective at budget 30, reported
   separately by family. Also report incumbent-versus-evaluation anytime AUC,
   action mix, unused budget, and per-seed outcomes. For cross-family display,
   use a *common family scale*; never average per-seed standardized Hartmann
   gains without showing raw values, since a tiny initial standard deviation
   can magnify them.
4. The 100–104 seeds were inspected during the pilot. A 100–119 extension is
   an exploratory audit, not an untouched confirmation set. Freeze decisions
   before a final new seed stream (e.g. 200–219) if a positive paper claim is
   sought. Report paired seed bootstrap intervals and account for multiple
   family comparisons.

### RQ2: Does the same policy transfer?

1. In the same BOCoDe loop, `rf_transfer` trains on the other four families
   and drives the held-out family's trajectory. Compare against the same
   fixed baselines. This fixes EXP-RQ2.1's one-step-only limitation, but each
   family still has one fixed dimension.
2. `scripts/exp_rq2_ioh_closed_loop.py` is an external stress test on unseen
   IOH/BBOB functions crossed with dimension and instance. The RF is trained
   on BOCoDe only. Report per-(function, dimension) paired outcomes; pool
   across functions only with a prespecified weighting and a function-level
   uncertainty analysis. This tests external transfer, not a clean isolated
   effect of dimension.
3. A true within-function dimension transfer claim needs a separate IOH
   training design: train on selected dimensions/instances of each function,
   test on held-out dimensions/instances and on fully held-out functions.
   Freeze this split before collecting labels.

### RQ3: Can selective LLM calls preserve performance at lower cost?

1. Use a learned System-1 plus a real LLM System-2. Select routing thresholds
   on a calibration split only. Evaluate frozen thresholds on new test seeds.
2. Compare always-System-1, always-LLM, confidence-only, confidence+OOD,
   and a random-call-rate control. Primary view: terminal regret versus
   actual LLM calls/tokens. Also report wall time, failed calls/fallbacks,
   and action mix. A surrogate/fake LLM path is an integration test only.
3. Declare a practically acceptable regret margin and desired call reduction
   before running the final test. No claim of cost saving can rest on a router
   that merely picks a cheaper *problem-level* arm in post-hoc data.

The executable threshold sweep is `scripts/exp_rq3_router_sweep.py`. It fits
the RF on rollout seeds 0–15 and temperature-calibrates predicted action
probabilities on seeds 16–19 using only training-derived family scales. Its
`rf`, `llm`, confidence-threshold, and matched random-rate arms run paired
trajectories with parallel workers and checkpointed output. The calibration
target is the best short-horizon action, not terminal success; confidence is
therefore a routing heuristic until validated against final performance. The
`--system2 mock` mode checks wiring only. A real cost frontier requires an LLM
credential and frozen threshold selection before the evaluation panel.

### Engineering case

Connect at least one real engineering simulator under the same fixed-budget
protocol. A direct numerical-to-engineering transfer test uses the frozen
numeric policy before any engineering fine-tuning. If the claim is transfer
across engineering tasks, a second independent task is required.

## Data provenance

- Dataset: `runs/phase8_6d_h3_20seed_5family.jsonl` (663 lines).
- The new runners save the training dataset SHA-256, resolved panel, every
  trajectory, and a resumable summary. Changing the configuration requires
  a new output path.
- The source file has 663 records; 661 pass EXP-RQ2.1's feature filtering.
  `runs/EXP_RQ2_1_REPORT.md` now states both counts.
- The 20-seed dataset extends seeds 0–9; it is not an independent replication.

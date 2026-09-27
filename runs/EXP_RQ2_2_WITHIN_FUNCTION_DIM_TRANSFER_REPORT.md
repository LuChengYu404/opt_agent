# EXP-RQ2.2 — Within-function dimension transfer (2026-09-27)

**Date:** 2026-09-27

## 10-seed confirmation (seeds 100–109, added later the same day)

The stated limitation below ("three test seeds per cell") is closed: the
protocol rerun with seeds 100–109 (120 train rollouts + 120 test
trajectories, 8 parallel workers), same untouched test dim 5 / instance 2.
Artifacts: `runs/exp_rq2_2_ioh_within_function_dim_transfer_10seed.json`.

| Cell (test d5/i2, n=10 seeds) | `rf − rule_full` | `rf − surrogate` | `rf − local` |
|---|---:|---:|---:|
| f6 | **+0.0000** [+0.0000, +0.0000] (0/10/0) | **+0.1281 [+0.0088, +0.2820]** (1/0/9) | **+0.1529 [+0.0058, +0.3030]** (3/0/7) |
| f8 | **+0.0000** [+0.0000, +0.0000] (0/10/0) | **+0.2443 [+0.0590, +0.4528]** (1/0/9) | **+0.3214 [+0.1408, +0.5264]** (1/0/9) |
| f15 | **+0.0000** [+0.0000, +0.0000] (0/10/0) | **+0.2252 [+0.0883, +0.3775]** (1/1/8) | +0.1050 [−0.0329, +0.2653] (4/0/6) |

(W/T/L is rf vs baseline.) Perfect imitation persists at 10 seeds on all
three functions, and the RF's loss to **Surrogate now has CI excluding
zero on all three functions** (at 3 seeds only f8/f15 cleared zero). The
negative result is strengthened with tighter intervals.
`docs/THEORY.md` Lemma 1 explains the exact-zero imitation column: the RF
feature set contains the rule's axis-aligned thresholds, so with noiseless
rule labels the forest reproduces the rule exactly — including its losses.
**Goal:** Cleanly isolate dimension transfer from family transfer. EXP-RQ2.1
trained an RF on four BOCoDe families and tested on a fifth; each family
has a fixed dimension, so the cross-family result was confounded with
dimension. EXP-RQ2.2 fixes the function and tests whether an RF trained
on two source dimensions of the same IOH function transfers to a held-out
target dimension.

## Protocol

- Functions: `f6` AttractiveSector, `f8` Rosenbrock, `f15` RotatedRastrigin.
- Train dims per function: `d_train ∈ {2, 10}`, instances `{0, 1}`.
- Train seeds: `{100, 101, 102}`.
- Test dim: `d_test = 5`, instance `2` (untouched by training).
- Test seeds: `{100, 101, 102}`.
- Budget: 30 evaluations (5 Sobol init + 25 decisions).
- Training labels: action chosen by `RuleFullBudget` at each decision step.
  Multi-output RF with one regressor per action; the action actually taken
  is the positive label.
- RF hyperparameters: 200 trees, depth 8, leaf 5, same as EXP-RQ2.1.
- Test policies: `rf`, `rule_full`, `surrogate`, `local`.
- Workers: 4 in parallel (fork).

**Total trajectories:** 36 train + 36 test = 72 trajectories.
**Total state-action training pairs:** (mean ≈ 21 captured steps × 36
trajectories ≈ 750 pairs).

## Headline (test dim = 5, instance = 2, n = 3 seeds per cell)

Lower normalized regret is better; `rf − baseline` positive ⇒ RF loses.

| Cell | RF regret | Rule regret | Surrogate regret | Local regret |
|---|---:|---:|---:|---:|
| f6/d5 | 0.880 | **0.880** | 0.394 | **0.321** |
| f8/d5 | 0.599 | **0.599** | **0.208** | **0.084** |
| f15/d5 | 0.533 | **0.533** | **0.291** | 0.463 |

### Paired `rf − baseline` regret (CI 95 % seed bootstrap, n = 3)

| Cell | `rf − rule_full` | `rf − surrogate` | `rf − local` |
|---|---:|---:|---:|
| f6/d5 | **+0.0000** [+0.0000, +0.0000] | +0.274 [+0.003, +0.751] | +0.347 [−0.002, +0.530] |
| f8/d5 | **+0.0000** [+0.0000, +0.0000] | +0.392 [+0.162, +0.801] | +0.515 [+0.208, +0.875] |
| f15/d5 | **+0.0000** [+0.0000, +0.0000] | +0.241 [+0.005, +0.554] | +0.070 [−0.194, +0.296] |

## Interpretation

1. **The RF perfectly mimics RuleFullBudget on the held-out test dim.**
   `rf − rule_full = +0.000` and `W/T/L = 0/3/0` across all three
   functions. The RF essentially *is* the rule on these held-out
   states; it produces the same decisions and the same terminal
   objective. This is the expected result of imitation learning: train
   on RuleFullBudget's actions at training dims, deploy at a held-out
   test dim, the model reproduces RuleFullBudget.

2. **Both RuleFullBudget and the RF lose to Surrogate or Local on every
   function.** On f8/d5, Surrogate reaches 0.21 normalized regret
   vs 0.60 for RF/rule; Local reaches 0.08. On f6/d5 and f15/d5,
   Surrogate beats RF/rule. The learned state-conditional policy is
   not better than a fixed-action baseline on any held-out dimension.

3. **Within-function dimension transfer is therefore a clean negative
   result.** The state representation that EXP-RQ2.1 found useful at
   the family level (Ackley beats Surrogate on 10D multimodal) does
   not transfer across dimensions of a single IOH function in our
   sample. The cross-family transfer signal in EXP-RQ2.1 was at least
   partly a *family* signal, not a dimension signal.

## Decision

- The frozen `rf` (BOCoDe five-family trained, deployed on BOCoDe) is
  the deployment-quality System-1 we can defend.
- A *fresh* within-function RF trained on IOH rollouts at two source
  dimensions does **not** generalise to a held-out third dimension. The
  state representation is dimension-sensitive.
- Any future cross-dimension claim must train and test within the same
  function family on **crossed** dimension/instances, then declare the
  pairing **before** collecting labels (this experiment does that).

## Artifacts

- `runs/exp_rq2_2_ioh_within_function_dim_transfer.json`: 36 train
  rollouts (with captured `state_features` and `action_taken` per step)
  + 36 test runs (RF + 3 baselines, 3 functions × 3 seeds × 4 policies)
  + paired summary.
- `scripts/exp_rq2_2_ioh_within_function_dim_transfer.py`: paired
  runner with fork-mode parallel workers; deterministic per-seed
  Sobol start; identical initial_value check on every paired cell.

## Limitations

- Three test seeds per cell; CI lower bounds often exclude zero only for
  large effects. A 10-seed extension would tighten the interval.
- Training data is the action label from `RuleFullBudget`. A learned
  policy trained on a richer label (e.g., oracle short-horizon
  rollout utility) might transfer differently.
- Train instances {0, 1} are not crossed with test instance 2 across
  the whole panel — the test instance is fully held out, but training
  uses only two instances, leaving the model potentially sensitive to
  instance-specific phenomena.

Co-Authored-By: Claude Code <noreply@anthropic.com>
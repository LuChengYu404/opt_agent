# EXP-RQ0.5 — Per-step LLM Action Distribution

**Source**: `runs/exp_rq0_5_joint_10seed.json` (joint 10-seed, 5-policy)
**Date**: 2026-09-27 (joint run) / analysis 2026-09-27
**Method**: re-derive `action_id` histogram from `steps[]` per trajectory

## 1. Per-policy action distribution (5 families pooled)

Counts are over **all steps × all seeds × all 5 families** within a policy.

| Policy | top action | share | second action | share | total steps |
|---|---|---|---|---|---|
| RulePolicy | `global_exploration` | 56% | `surrogate_acquisition` | 37% | 723 |
| AlwaysSurrogate | `surrogate_acquisition` | 100% | `—` | 0% | 1250 |
| LLMDeepSeek | `global_exploration` | 69% | `local_exploitation` | 29% | 739 |
| Jev | `surrogate_acquisition` | 100% | `—` | 0% | 1250 |
| RoutingAware | `surrogate_acquisition` | 54% | `global_exploration` | 32% | 947 |

## 2. Per-policy × per-family action distribution

### RulePolicy

| family | most-common action | share | second action | share | total steps |
|---|---|---|---|---|---|
| Branin | `global_exploration` | 54% | `surrogate_acquisition` | 40% | 149 |
| Hartmann | `global_exploration` | 60% | `surrogate_acquisition` | 33% | 141 |
| Ackley | `global_exploration` | 58% | `surrogate_acquisition` | 35% | 142 |
| SixHumpCamel | `global_exploration` | 59% | `surrogate_acquisition` | 34% | 141 |
| Rastrigin | `global_exploration` | 50% | `surrogate_acquisition` | 43% | 150 |

### AlwaysSurrogate

| family | most-common action | share | second action | share | total steps |
|---|---|---|---|---|---|
| Branin | `surrogate_acquisition` | 100% | `—` | 0% | 250 |
| Hartmann | `surrogate_acquisition` | 100% | `—` | 0% | 250 |
| Ackley | `surrogate_acquisition` | 100% | `—` | 0% | 250 |
| SixHumpCamel | `surrogate_acquisition` | 100% | `—` | 0% | 250 |
| Rastrigin | `surrogate_acquisition` | 100% | `—` | 0% | 250 |

### LLMDeepSeek

| family | most-common action | share | second action | share | total steps |
|---|---|---|---|---|---|
| Branin | `global_exploration` | 76% | `local_exploitation` | 24% | 142 |
| Hartmann | `global_exploration` | 67% | `local_exploitation` | 32% | 150 |
| Ackley | `global_exploration` | 66% | `local_exploitation` | 28% | 151 |
| SixHumpCamel | `global_exploration` | 68% | `local_exploitation` | 30% | 149 |
| Rastrigin | `global_exploration` | 70% | `local_exploitation` | 29% | 147 |

### Jev

| family | most-common action | share | second action | share | total steps |
|---|---|---|---|---|---|
| Branin | `surrogate_acquisition` | 100% | `—` | 0% | 250 |
| Hartmann | `surrogate_acquisition` | 100% | `—` | 0% | 250 |
| Ackley | `surrogate_acquisition` | 100% | `—` | 0% | 250 |
| SixHumpCamel | `surrogate_acquisition` | 100% | `—` | 0% | 250 |
| Rastrigin | `surrogate_acquisition` | 100% | `—` | 0% | 250 |

### RoutingAware

| family | most-common action | share | second action | share | total steps |
|---|---|---|---|---|---|
| Branin | `surrogate_acquisition` | 100% | `—` | 0% | 250 |
| Hartmann | `global_exploration` | 70% | `local_exploitation` | 29% | 147 |
| Ackley | `global_exploration` | 64% | `local_exploitation` | 28% | 152 |
| SixHumpCamel | `surrogate_acquisition` | 100% | `—` | 0% | 250 |
| Rastrigin | `global_exploration` | 69% | `local_exploitation` | 30% | 148 |

## 3. Mechanism check: does LLM pick the closed-loop best first action?

Closed-loop best first-action under AlwaysSurrogate continuation (from `runs/EXP_RQ1_3_REPORT.md` §4):

| Family | closed-loop best first action |
|---|---|
| Branin | `surrogate_acquisition` |
| Hartmann | `local_exploitation` |
| Ackley | `local_exploitation` |
| SixHumpCamel | `surrogate_acquisition` |
| Rastrigin | `local_exploitation` |

**LLMDeepSeek per-family top action** (from §2 above):

| Family | LLM top action | share | closed-loop best | matches? |
|---|---|---|---|---|
| Branin | `global_exploration` | 76% | `surrogate_acquisition` | ✗ |
| Hartmann | `global_exploration` | 67% | `local_exploitation` | ✗ |
| Ackley | `global_exploration` | 66% | `local_exploitation` | ✗ |
| SixHumpCamel | `global_exploration` | 68% | `surrogate_acquisition` | ✗ |
| Rastrigin | `global_exploration` | 70% | `local_exploitation` | ✗ |

**Conclusion**: LLMDeepSeek does NOT match the closed-loop best first action on any of the 5 families — it picks `global_exploration` as the top action everywhere (66–76% share). The earlier §5 of `EXP_RQ1_3_REPORT.md` hypothesized LLM might be picking `local_exploitation` (closed-loop best) on Rastrigin; **that hypothesis is rejected** by the per-step distribution.

## 4. Per-step progression on Rastrigin (the +17.54 family)

Action share at each step across the 10 LLM trajectories on Rastrigin:

| step | global_exploration | local_exploitation | surrogate_acq | n_obs |
|---|---|---|---|---|
| 0 | 10/10 = 100% | 0/10 = 0% | 0/10 = 0% | 10 |
| 1 | 10/10 = 100% | 0/10 = 0% | 0/10 = 0% | 10 |
| 2 | 10/10 = 100% | 0/10 = 0% | 0/10 = 0% | 10 |
| 3 | 9/10 = 90% | 1/10 = 10% | 0/10 = 0% | 10 |
| 4 | 10/10 = 100% | 0/10 = 0% | 0/10 = 0% | 10 |
| 5 | 10/10 = 100% | 0/10 = 0% | 0/10 = 0% | 10 |
| 6 | 9/10 = 90% | 1/10 = 10% | 0/10 = 0% | 10 |
| 7 | 8/10 = 80% | 2/10 = 20% | 0/10 = 0% | 10 |
| 8 | 9/10 = 90% | 1/10 = 10% | 0/10 = 0% | 10 |
| 9 | 9/10 = 90% | 1/10 = 10% | 0/10 = 0% | 10 |
| 10 | 3/10 = 30% | 6/10 = 60% | 1/10 = 10% | 10 |
| 11 | 4/10 = 40% | 6/10 = 60% | 0/10 = 0% | 10 |
| 12 | 2/10 = 20% | 8/10 = 80% | 0/10 = 0% | 10 |
| 13 | 0/10 = 0% | 10/10 = 100% | 0/10 = 0% | 10 |
| 14 | 0/6 = 0% | 6/6 = 100% | 0/6 = 0% | 6 |
| 15 | 0/1 = 0% | 1/1 = 100% | 0/1 = 0% | 1 |

**Pattern**: LLMDeepSeek on Rastrigin uses a **'global-then-local' schedule** — `global_exploration` dominates steps 0–9 (90–100%), then transitions to `local_exploitation` from step 10 onward, with 100% `local_exploitation` at step 13. This is a structured, non-uniform policy across the trajectory.

AlwaysSurrogate action is **constant** at `surrogate_acquisition` 100% for all 25 steps on Rastrigin.

## 5. Per-step best_y trajectory on Rastrigin (mean over 10 seeds)

| step | LLM mean best_y | Surv mean best_y | Δ (LLM − Surv) |
|---|---|---|---|
| 0 | -151.861 | -153.029 | +1.167 |
| 1 | -148.365 | -148.113 | -0.252 |
| 2 | -144.521 | -148.113 | +3.592 |
| 3 | -138.914 | -142.403 | +3.489 |
| 4 | -134.784 | -142.403 | +7.619 |
| 5 | -130.374 | -142.403 | +12.029 |
| 6 | -123.665 | -142.403 | +18.738 |
| 7 | -122.603 | -139.741 | +17.138 |
| 8 | -122.157 | -138.146 | +15.989 |
| 9 | -120.151 | -137.717 | +17.566 |
| 10 | -118.606 | -137.655 | +19.048 |
| 11 | -118.184 | -137.655 | +19.471 |
| 12 | -113.944 | -136.040 | +22.096 |
| 13 | -109.764 | -134.534 | +24.770 |
| 14 | -110.338 | -134.322 | +23.985 |
| 15 | -88.611 | -134.322 | +45.712 |

**Pattern**: LLM starts roughly even with Surrogate (Δ ≈ 0 at step 1), then pulls ahead starting step 2. The advantage reaches Δ ≈ +20 by step 10 and **+45 by step 15** (limited to the 1 trajectory that ran that long; the median trajectory terminates at step ~13).

## 6. Revised mechanism (replacing the rejected hypothesis)

The earlier hypothesis — *'LLM picks `local_exploitation` which is the closed-loop best'* — is rejected by the per-step distribution.

**Revised mechanism (descriptive, not causal)**: On Rastrigin (10D multimodal), the LLM exhibits a **two-phase temporal action pattern**:

1. **Early phase (steps 0–9)**: predominantly `global_exploration` (90–100% across these steps; occasionally 1/10 trajectories spike to `local_exploitation` at steps 3, 6, 8, 9). The LLM samples broadly across the input space to escape local optima.
2. **Transition phase (steps 10–12)**: mixed — 30–40% global, 60–80% local, occasional 1/10 surrogate.
3. **Late phase (step 13+)**: 100% `local_exploitation`. After global sampling, the LLM refines around the best region found.

AlwaysSurrogate, by contrast, picks `surrogate_acquisition` (GP-EI) for **every** step — relying entirely on the surrogate to balance exploration and exploitation. On multimodal problems, the surrogate's exploitation bias dominates too early, before global coverage is sufficient.

**Important caveat (per codex review 2026-09-27)**: this is a *behavioral* association, not a causal claim. The LLM's pooled top action on Rastrigin is `global_exploration` at 70%, which is **similar to its share on smoother families** (Branin 76%, Hartmann 67%, Ackley 66%, SixHumpCamel 68%) — *families where LLM does NOT beat Surrogate*. The pooled distribution alone does not explain the Rastrigin-specific advantage; only the *temporal pattern* (early-global, late-local) is unique to Rastrigin among the 5 families.

**Causal validation not established**: a deterministic global-then-local baseline (e.g., always pick `global_exploration` for 10 steps then `local_exploitation` thereafter) has not been implemented or compared. Until that ablation is run, the schedule-vs-advantage link remains correlational.

Recommended paper wording: *'On Rastrigin, the LLM exhibited a global-to-local temporal action pattern associated with its terminal advantage. This analysis does not establish that the schedule caused the advantage.'*

## 7. Sanity check

All 250 trajectories: per-step `action_id` counts match the JSON's precomputed `action_counts` block. ✓

## 8. Files

- `runs/exp_rq0_5_joint_10seed.json` — source (joint 10-seed, 5-policy)
- `runs/exp_rq0_5_action_distribution.json` — raw counts
- `runs/EXP_RQ0_5_REPORT.md` — parent report
- `runs/EXP_RQ1_3_REPORT.md` — closed-loop regret reference
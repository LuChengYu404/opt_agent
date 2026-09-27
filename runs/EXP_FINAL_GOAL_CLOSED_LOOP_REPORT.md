# Final-goal experiment update — closed-loop policy and external transfer

**Date:** 2026-09-27  
**Status:** completed local experiments; negative result for a general learned-policy gain.

## Protocol and provenance

- Training corpus: `phase8_6d_h3_20seed_5family.jsonl`, 663 records from BOCoDe seeds 0–19. The deployed policy sees only live `OptimizationState` features. RF hyperparameters and feature code match EXP-RQ2.1.
- BOCoDe evaluation: five families, budget 30 (five shared initial evaluations), seeds 100–119 for an expanded audit and **new seeds 200–219** for the retrain-action diagnostic. All policies in a paired cell start from identical initial objective values. Higher terminal best is better.
- `rf_all` trains on all five development families. `rf_transfer` trains on the other four, excluding the tested family. Fixed controls: RuleFullBudget, AlwaysSurrogate, AlwaysLocal, and a 10-global-decision-then-local schedule. The 100–119 audit includes seeds 100–104 inspected in the five-seed pilot; it is **not** a pristine confirmation set.
- Intervals below are paired seed bootstrap 95% intervals over 20 seeds, **unadjusted for five family comparisons**. Raw best-value differences are reported because per-seed normalization becomes unstable when the five initial objectives have tiny variance (especially Hartmann).
- The runners save dataset SHA-256, complete trajectories, initial objectives, actions, evaluation count, anytime AUC, and resumable checkpoints. Parallel execution is by independent trajectory; 16 BOCoDe and 4 IOH smoke trajectories, plus two full-budget BOCoDe trajectories with GP fitting, matched serial results exactly.

## RQ1: full-loop trained RF versus fixed policies

Fresh seeds 200–219: paired mean terminal best difference (`RF − baseline`; positive is better).

| Family | `rf_all − Surrogate` (95% interval) | `rf_all − Local` (95% interval) | `rf_transfer − Surrogate` (95% interval) |
|---|---:|---:|---:|
| Branin | **−1.221** [−1.892, −0.621] | −0.632 [−1.280, −0.072] | −0.652 [−1.064, −0.352] |
| Hartmann | **−0.695** [−0.988, −0.398] | −0.009 [−0.351, +0.319] | −1.257 [−1.513, −0.977] |
| Ackley | +0.662 [+0.111, +1.321] | −0.216 [−0.578, +0.067] | +0.417 [−0.062, +1.022] |
| SixHumpCamel | **−0.441** [−0.811, −0.148] | −0.329 [−0.588, −0.128] | −0.190 [−0.334, −0.050] |
| Rastrigin | +4.334 [−3.298, +12.176] | −3.645 [−10.531, +2.465] | +8.753 [−0.167, +17.049] |

The separate 100–119 audit points in the same direction: `rf_all − Surrogate` is −0.611 on Branin, −0.405 on Hartmann, +0.709 on Ackley, −0.197 on SixHumpCamel, and +1.597 on Rastrigin. The Rastrigin interval crosses zero. The new seed stream therefore does **not** rescue the original broad RQ1 claim. Ackley shows a gain over Surrogate, but the constant Local policy is stronger than the RF mean; a useful learned action ranking has not been demonstrated against the stronger fixed control.

### Explicit surrogate-refit ablation

The RF selected `retrain_surrogate` frequently on the 100–119 audit (225 of 656 Branin `rf_all` decisions and 205 of 627 SixHumpCamel decisions). This action consumes no objective evaluations but repeatedly fits the GP. We froze a `no_retrain` action mask and tested it on **new seeds 200–219** against the original RF.

| Family | `rf_all_no_retrain − rf_all` mean best difference | `rf_transfer_no_retrain − rf_transfer` |
|---|---:|---:|
| Branin | +0.071 [−0.008, +0.201] | 0.000 (same terminal values) |
| Hartmann | 0.000 (same terminal values) | +0.148 [+0.042, +0.278] |
| Ackley | 0.000 | 0.000 |
| SixHumpCamel | +0.057 [−0.010, +0.166] | +0.0002 [0.0000, +0.0005] |
| Rastrigin | 0.000 | 0.000 |

The mask removes unused-budget behavior in some trajectories and improves the Hartmann transfer arm, but **neither masked policy beats the relevant fixed Surrogate or Local control consistently**. Refit overuse is one failure mechanism, not a sufficient repair.

## RQ2: external IOH/BBOB transfer stress test

The BOCoDe-trained `rf_all` was deployed without IOH training on f6 AttractiveSector, f8 Rosenbrock, and f15 RotatedRastrigin, crossed with dimensions 2, 5, 10 and instances 2–4, five seeds each: **405 complete trajectories** across RF, Surrogate, and Local. Budget is 30, initial evaluations are paired, and lower normalized terminal regret is better. The intervals below resample **instances first, then seeds within instance**; with only three instances per cell, they remain exploratory.

| Function | d=2 `RF−Surrogate` | d=5 | d=10 |
|---|---:|---:|---:|
| f6 AttractiveSector | +0.113 [+0.032, +0.227] | +0.052 [−0.008, +0.199] | −0.267 [−0.441, −0.080] |
| f8 Rosenbrock | +0.286 [+0.066, +0.587] | −0.001 [−0.076, +0.080] | −0.179 [−0.363, +0.038] |
| f15 RotatedRastrigin | +0.124 [−0.014, +0.235] | +0.122 [+0.046, +0.217] | −0.116 [−0.232, −0.006] |

The RF has lower mean regret than Surrogate in all three tested 10D cells and higher mean regret in all three 2D cells; **Local is frequently competitive or better**. This is heterogeneous external transfer, not a general RQ2 pass. Because the RF was trained only on BOCoDe and no IOH labels, this test combines domain and dimension shift. A clean dimension-only claim needs training and test dimensions crossed within each function family.

## RQ3: actual API cost already recorded, generalization still open

The existing five-family BOCoDe joint experiment records **739 full-LLM calls / 537,456 tokens** versus **447 LLM calls / 324,702 tokens** for the frozen dimension router: about **39.5% fewer calls and 39.6% fewer tokens**. This is a development-panel cost audit. The same dimension rule failed to show a performance gain over Surrogate on the held-out IOH panel, and it is not a calibrated confidence/OOD router. A new real-LLM threshold sweep remains necessary for the intended RQ3 claim.

The new `exp_rq3_router_sweep.py` implements paired RF-only, LLM-only,
confidence-threshold, and random-call-rate arms with parallel workers. It
trains on rollout seeds 0–15 and calibrates predicted action probabilities on
16–19 without using calibration records to set family scales. Its target is
short-horizon action utility, so confidence still needs terminal-outcome
validation. A four-arm, budget-10 **mock-System-2 wiring check** on Branin
seed 300 matched serial and two-worker execution in actions, routes, objective
values, and call counts. Mock results are not used as performance evidence.
No credential was available for a real LLM sweep in this environment.

## Global-then-local mechanism check

On the original Rastrigin development seeds 0–9, a **post-hoc** fixed schedule of ten global decisions followed by local decisions averaged terminal best **−99.12**, versus **−109.76** for the recorded LLM and **−127.30** for AlwaysSurrogate. The schedule-minus-LLM paired mean is +10.64 [ +1.33, +22.11 ], but it wins only 4/10 seeds, and the schedule was derived after inspecting these same LLM trajectories. On separate seeds 100–119 the same schedule averaged **−126.51**, worse than Surrogate **−115.36** and Local **−113.78**. Thus the observed LLM action pattern does not establish a transferable causal mechanism.

## Decision for the paper

The current learned RF is a valid negative control and a useful diagnostic instrument, **not** a demonstrated better optimizer. A final-method paper should not claim general learned-policy superiority or transferable dimension routing from these runs. The next method experiment should replace short-horizon utility targets or action handling, freeze it on development data, and compare against Surrogate and Local on a truly untouched full-loop panel. Real System-1/System-2 cost-frontier work requires a configured LLM credential; engineering transfer requires a simulator or task definition.

## Artifacts

- `scripts/exp_rq1_rq2_closed_loop.py`: BOCoDe full-loop runner, paired summaries, parallel workers and resumable single-writer checkpoints.
- `runs/exp_rq1_rq2_closed_loop_20seed.json`: expanded 100–119 audit (400 trajectories).
- `runs/exp_rq1_rq2_closed_loop_20seed_controls.json`: RuleFullBudget and fixed schedule on the same seeds (200 trajectories).
- `runs/exp_rq1_rq2_no_retrain_fresh20.json`: 200–219 action-mask test (600 trajectories).
- `scripts/exp_rq2_ioh_closed_loop.py`, `scripts/analyze_ioh_crossed_transfer.py`: external crossed runner and hierarchical analysis.
- `runs/exp_rq2_ioh_closed_loop_crossed.json`, `runs/exp_rq2_ioh_crossed_analysis.json`: 405 external trajectories and summaries.
- `scripts/exp_rq3_existing_cost_audit.py`, `runs/exp_rq3_existing_cost_audit.json`: actual recorded LLM cost audit.
- `scripts/exp_rq3_router_sweep.py`: resumable parallel real-LLM routing
  sweep; mock mode was checked but has no scientific result artifact.
- `runs/exp_rq0_5_fixed_schedule_10seed.json`: post-hoc Rastrigin mechanism control.

## Verification

- Full project suite: `388 passed` (`.venv/bin/pytest -q`, 2026-09-27).
- Parallel/serial smoke: 16 BOCoDe trajectories and four IOH trajectories
  matched terminal values and action sequences exactly.
- Full-budget GP check: two Hartmann seed-200 trajectories also matched
  between serial and parallel execution.
- Router integration check: four Branin seed-300 mock trajectories matched
  serial and two-worker execution across ten outcome and routing fields.

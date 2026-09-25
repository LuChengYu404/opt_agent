# Q-S5 Diagnostic Report

- **Dataset**: `runs/phase8_6b_h3_10seed.jsonl`
- **Records sampled**: 5 (seed=0)
- **Configs**: 15 per record × 5 records = 75 runs
- **Continuation policy**: `rule` (fixed per Q-S5 Decision 4-Z)

## Per-config summary

| Config | N | Changed | Top-1 Agree | Oracle N | Mean gain vs STOP (new) | Mean gain vs STOP (orig) | Oracle Pass (new>STOP) |
|---|---|---|---|---|---|---|---|
| `delta_best_h1` | 5 | 4 | 0.20 | 4 | +1.8073 | +2.3939 | 2/4 |
| `delta_best_h10` | 5 | 4 | 0.20 | 4 | +9.1340 | +7.1500 | 3/4 |
| `delta_best_h3` | 5 | 2 | 0.60 | 2 | +0.0035 | +0.0035 | 1/2 |
| `delta_best_h5` | 5 | 2 | 0.60 | 2 | +0.0035 | +0.0035 | 1/2 |
| `delta_best_hfull` | 5 | 4 | 0.20 | 4 | +9.1340 | +7.1500 | 3/4 |
| `no_stop_pen_h1` | 5 | 4 | 0.20 | 4 | +1.8056 | +2.3939 | 1/4 |
| `no_stop_pen_h10` | 5 | 4 | 0.20 | 4 | +9.1323 | +7.1500 | 2/4 |
| `no_stop_pen_h3` | 5 | 2 | 0.60 | 2 | +0.0000 | +0.0035 | 0/2 |
| `no_stop_pen_h5` | 5 | 2 | 0.60 | 2 | +0.0000 | +0.0035 | 0/2 |
| `no_stop_pen_hfull` | 5 | 4 | 0.20 | 4 | +9.1323 | +7.1500 | 2/4 |
| `original_h1` | 5 | 1 | 0.80 | 1 | +7.2222 | +9.5686 | 1/1 |
| `original_h10` | 5 | 4 | 0.20 | 4 | +9.1340 | +7.1500 | 3/4 |
| `original_h3` | 5 | 0 | 1.00 | 0 | +0.0000 | +0.0000 | 0/0 |
| `original_h5` | 5 | 1 | 0.80 | 1 | +0.0000 | +0.0000 | 0/1 |
| `original_hfull` | 5 | 4 | 0.20 | 4 | +9.1340 | +7.1500 | 3/4 |

**Reading the table**:
- `Top-1 Agree` = fraction of records where the new config picks the same top action as the original.
- `Mean gain vs STOP (new)` = mean normalized (gold_new − stop) / scale over records where top action changed.
- `Mean gain vs STOP (orig)` = same but using original gold_probs.
- `Oracle Pass (new>STOP)` = count of changed records where new top action terminal best_value > STOP.

## Per-(problem, phase, config) top-1 agreement

| Problem | Phase | Config | N | Top-1 Agree |
|---|---|---|---|---|
| `ackley` | mid | `delta_best_h1` | 1 | 0.00 |
| `ackley` | mid | `delta_best_h10` | 1 | 0.00 |
| `ackley` | mid | `delta_best_h3` | 1 | 0.00 |
| `ackley` | mid | `delta_best_h5` | 1 | 0.00 |
| `ackley` | mid | `delta_best_hfull` | 1 | 0.00 |
| `ackley` | mid | `no_stop_pen_h1` | 1 | 0.00 |
| `ackley` | mid | `no_stop_pen_h10` | 1 | 0.00 |
| `ackley` | mid | `no_stop_pen_h3` | 1 | 0.00 |
| `ackley` | mid | `no_stop_pen_h5` | 1 | 0.00 |
| `ackley` | mid | `no_stop_pen_hfull` | 1 | 0.00 |
| `ackley` | mid | `original_h1` | 1 | 1.00 |
| `ackley` | mid | `original_h10` | 1 | 0.00 |
| `ackley` | mid | `original_h3` | 1 | 1.00 |
| `ackley` | mid | `original_h5` | 1 | 0.00 |
| `ackley` | mid | `original_hfull` | 1 | 0.00 |
| `branin` | mid | `delta_best_h1` | 1 | 0.00 |
| `branin` | mid | `delta_best_h10` | 1 | 0.00 |
| `branin` | mid | `delta_best_h3` | 1 | 0.00 |
| `branin` | mid | `delta_best_h5` | 1 | 0.00 |
| `branin` | mid | `delta_best_hfull` | 1 | 0.00 |
| `branin` | mid | `no_stop_pen_h1` | 1 | 0.00 |
| `branin` | mid | `no_stop_pen_h10` | 1 | 0.00 |
| `branin` | mid | `no_stop_pen_h3` | 1 | 0.00 |
| `branin` | mid | `no_stop_pen_h5` | 1 | 0.00 |
| `branin` | mid | `no_stop_pen_hfull` | 1 | 0.00 |
| `branin` | mid | `original_h1` | 1 | 1.00 |
| `branin` | mid | `original_h10` | 1 | 0.00 |
| `branin` | mid | `original_h3` | 1 | 1.00 |
| `branin` | mid | `original_h5` | 1 | 1.00 |
| `branin` | mid | `original_hfull` | 1 | 0.00 |
| `hartmann` | early | `delta_best_h1` | 1 | 0.00 |
| `hartmann` | early | `delta_best_h10` | 1 | 0.00 |
| `hartmann` | early | `delta_best_h3` | 1 | 1.00 |
| `hartmann` | early | `delta_best_h5` | 1 | 1.00 |
| `hartmann` | early | `delta_best_hfull` | 1 | 0.00 |
| `hartmann` | early | `no_stop_pen_h1` | 1 | 0.00 |
| `hartmann` | early | `no_stop_pen_h10` | 1 | 0.00 |
| `hartmann` | early | `no_stop_pen_h3` | 1 | 1.00 |
| `hartmann` | early | `no_stop_pen_h5` | 1 | 1.00 |
| `hartmann` | early | `no_stop_pen_hfull` | 1 | 0.00 |
| `hartmann` | early | `original_h1` | 1 | 0.00 |
| `hartmann` | early | `original_h10` | 1 | 0.00 |
| `hartmann` | early | `original_h3` | 1 | 1.00 |
| `hartmann` | early | `original_h5` | 1 | 1.00 |
| `hartmann` | early | `original_hfull` | 1 | 0.00 |
| `hartmann` | late | `delta_best_h1` | 1 | 0.00 |
| `hartmann` | late | `delta_best_h10` | 1 | 1.00 |
| `hartmann` | late | `delta_best_h3` | 1 | 1.00 |
| `hartmann` | late | `delta_best_h5` | 1 | 1.00 |
| `hartmann` | late | `delta_best_hfull` | 1 | 1.00 |
| `hartmann` | late | `no_stop_pen_h1` | 1 | 0.00 |
| `hartmann` | late | `no_stop_pen_h10` | 1 | 1.00 |
| `hartmann` | late | `no_stop_pen_h3` | 1 | 1.00 |
| `hartmann` | late | `no_stop_pen_h5` | 1 | 1.00 |
| `hartmann` | late | `no_stop_pen_hfull` | 1 | 1.00 |
| `hartmann` | late | `original_h1` | 1 | 1.00 |
| `hartmann` | late | `original_h10` | 1 | 1.00 |
| `hartmann` | late | `original_h3` | 1 | 1.00 |
| `hartmann` | late | `original_h5` | 1 | 1.00 |
| `hartmann` | late | `original_hfull` | 1 | 1.00 |
| `hartmann` | mid | `delta_best_h1` | 1 | 1.00 |
| `hartmann` | mid | `delta_best_h10` | 1 | 0.00 |
| `hartmann` | mid | `delta_best_h3` | 1 | 1.00 |
| `hartmann` | mid | `delta_best_h5` | 1 | 1.00 |
| `hartmann` | mid | `delta_best_hfull` | 1 | 0.00 |
| `hartmann` | mid | `no_stop_pen_h1` | 1 | 1.00 |
| `hartmann` | mid | `no_stop_pen_h10` | 1 | 0.00 |
| `hartmann` | mid | `no_stop_pen_h3` | 1 | 1.00 |
| `hartmann` | mid | `no_stop_pen_h5` | 1 | 1.00 |
| `hartmann` | mid | `no_stop_pen_hfull` | 1 | 0.00 |
| `hartmann` | mid | `original_h1` | 1 | 1.00 |
| `hartmann` | mid | `original_h10` | 1 | 0.00 |
| `hartmann` | mid | `original_h3` | 1 | 1.00 |
| `hartmann` | mid | `original_h5` | 1 | 1.00 |
| `hartmann` | mid | `original_hfull` | 1 | 0.00 |

## Conclusions (heuristic — codex review required for final word)

- **Most label-consistent config**: `original_h3` (top-1 agreement 1.00).
- **Best oracle gain (when changed)**: `original_h10` (mean gain vs STOP = +9.1340).
- **Total (record, config) pairs**: 75, of which 42 had top-action changes (56.0%).

If most configs change few top actions AND the original top still loses to STOP, the counterfactual teacher is fundamentally mis-specified (Q-S5 verdict: utility not just wrong but possibly *mis-aimed*). Switch route per Q-Route recommendation.

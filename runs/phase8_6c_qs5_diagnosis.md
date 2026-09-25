# Q-S5 Diagnostic Report

- **Dataset**: `runs/phase8_6b_h3_10seed.jsonl`
- **Records sampled**: 30 (seed=0)
- **Configs**: 15 per record × 30 records = 450 runs
- **Continuation policy**: `rule` (fixed per Q-S5 Decision 4-Z)

## Per-config summary

| Config | N | Changed | Top-1 Agree | Oracle N | Mean gain vs STOP (new) | Mean gain vs STOP (orig) | Oracle Pass (new>STOP) |
|---|---|---|---|---|---|---|---|
| `delta_best_h1` | 30 | 19 | 0.37 | 19 | +0.5682 | +0.7644 | 7/19 |
| `delta_best_h10` | 30 | 22 | 0.27 | 22 | +3.0876 | +2.2681 | 10/22 |
| `delta_best_h3` | 30 | 16 | 0.47 | 16 | +0.0371 | +0.0388 | 4/16 |
| `delta_best_h5` | 30 | 15 | 0.50 | 15 | +0.0149 | +0.0168 | 3/15 |
| `delta_best_hfull` | 30 | 22 | 0.27 | 22 | +3.0878 | +2.2681 | 10/22 |
| `no_stop_pen_h1` | 30 | 20 | 0.33 | 20 | +0.4022 | +0.7297 | 4/20 |
| `no_stop_pen_h10` | 30 | 23 | 0.23 | 23 | +2.9073 | +2.1875 | 4/23 |
| `no_stop_pen_h3` | 30 | 14 | 0.53 | 14 | +0.0000 | +0.0055 | 0/14 |
| `no_stop_pen_h5` | 30 | 18 | 0.40 | 18 | +0.0406 | +0.0716 | 2/18 |
| `no_stop_pen_hfull` | 30 | 23 | 0.23 | 23 | +2.8781 | +2.1875 | 3/23 |
| `original_h1` | 30 | 7 | 0.77 | 7 | +1.6721 | +2.0739 | 5/7 |
| `original_h10` | 30 | 15 | 0.50 | 15 | +4.5094 | +3.3148 | 7/15 |
| `original_h3` | 30 | 1 | 0.97 | 1 | +0.0000 | +0.0000 | 0/1 |
| `original_h5` | 30 | 7 | 0.77 | 7 | +0.1044 | +0.1192 | 2/7 |
| `original_hfull` | 30 | 15 | 0.50 | 15 | +4.4946 | +3.3001 | 7/15 |

**Reading the table**:
- `Top-1 Agree` = fraction of records where the new config picks the same top action as the original.
- `Mean gain vs STOP (new)` = mean normalized (gold_new − stop) / scale over records where top action changed.
- `Mean gain vs STOP (orig)` = same but using original gold_probs.
- `Oracle Pass (new>STOP)` = count of changed records where new top action terminal best_value > STOP.

## Per-(problem, phase, config) top-1 agreement

| Problem | Phase | Config | N | Top-1 Agree |
|---|---|---|---|---|
| `ackley` | early | `delta_best_h1` | 3 | 0.67 |
| `ackley` | early | `delta_best_h10` | 3 | 0.33 |
| `ackley` | early | `delta_best_h3` | 3 | 1.00 |
| `ackley` | early | `delta_best_h5` | 3 | 1.00 |
| `ackley` | early | `delta_best_hfull` | 3 | 0.33 |
| `ackley` | early | `no_stop_pen_h1` | 3 | 0.67 |
| `ackley` | early | `no_stop_pen_h10` | 3 | 0.33 |
| `ackley` | early | `no_stop_pen_h3` | 3 | 1.00 |
| `ackley` | early | `no_stop_pen_h5` | 3 | 0.67 |
| `ackley` | early | `no_stop_pen_hfull` | 3 | 0.33 |
| `ackley` | early | `original_h1` | 3 | 0.67 |
| `ackley` | early | `original_h10` | 3 | 0.33 |
| `ackley` | early | `original_h3` | 3 | 1.00 |
| `ackley` | early | `original_h5` | 3 | 1.00 |
| `ackley` | early | `original_hfull` | 3 | 0.33 |
| `ackley` | late | `delta_best_h1` | 3 | 0.33 |
| `ackley` | late | `delta_best_h10` | 3 | 0.33 |
| `ackley` | late | `delta_best_h3` | 3 | 0.33 |
| `ackley` | late | `delta_best_h5` | 3 | 0.33 |
| `ackley` | late | `delta_best_hfull` | 3 | 0.33 |
| `ackley` | late | `no_stop_pen_h1` | 3 | 0.33 |
| `ackley` | late | `no_stop_pen_h10` | 3 | 0.33 |
| `ackley` | late | `no_stop_pen_h3` | 3 | 0.33 |
| `ackley` | late | `no_stop_pen_h5` | 3 | 0.33 |
| `ackley` | late | `no_stop_pen_hfull` | 3 | 0.33 |
| `ackley` | late | `original_h1` | 3 | 0.67 |
| `ackley` | late | `original_h10` | 3 | 1.00 |
| `ackley` | late | `original_h3` | 3 | 1.00 |
| `ackley` | late | `original_h5` | 3 | 1.00 |
| `ackley` | late | `original_hfull` | 3 | 1.00 |
| `ackley` | mid | `delta_best_h1` | 4 | 0.25 |
| `ackley` | mid | `delta_best_h10` | 4 | 0.50 |
| `ackley` | mid | `delta_best_h3` | 4 | 0.50 |
| `ackley` | mid | `delta_best_h5` | 4 | 0.50 |
| `ackley` | mid | `delta_best_hfull` | 4 | 0.50 |
| `ackley` | mid | `no_stop_pen_h1` | 4 | 0.25 |
| `ackley` | mid | `no_stop_pen_h10` | 4 | 0.25 |
| `ackley` | mid | `no_stop_pen_h3` | 4 | 0.50 |
| `ackley` | mid | `no_stop_pen_h5` | 4 | 0.50 |
| `ackley` | mid | `no_stop_pen_hfull` | 4 | 0.25 |
| `ackley` | mid | `original_h1` | 4 | 0.75 |
| `ackley` | mid | `original_h10` | 4 | 0.50 |
| `ackley` | mid | `original_h3` | 4 | 1.00 |
| `ackley` | mid | `original_h5` | 4 | 0.50 |
| `ackley` | mid | `original_hfull` | 4 | 0.50 |
| `branin` | early | `delta_best_h1` | 3 | 0.33 |
| `branin` | early | `delta_best_h10` | 3 | 0.00 |
| `branin` | early | `delta_best_h3` | 3 | 0.33 |
| `branin` | early | `delta_best_h5` | 3 | 0.67 |
| `branin` | early | `delta_best_hfull` | 3 | 0.00 |
| `branin` | early | `no_stop_pen_h1` | 3 | 0.33 |
| `branin` | early | `no_stop_pen_h10` | 3 | 0.00 |
| `branin` | early | `no_stop_pen_h3` | 3 | 1.00 |
| `branin` | early | `no_stop_pen_h5` | 3 | 0.00 |
| `branin` | early | `no_stop_pen_hfull` | 3 | 0.00 |
| `branin` | early | `original_h1` | 3 | 0.33 |
| `branin` | early | `original_h10` | 3 | 0.33 |
| `branin` | early | `original_h3` | 3 | 1.00 |
| `branin` | early | `original_h5` | 3 | 0.33 |
| `branin` | early | `original_hfull` | 3 | 0.33 |
| `branin` | late | `delta_best_h1` | 3 | 0.00 |
| `branin` | late | `delta_best_h10` | 3 | 0.00 |
| `branin` | late | `delta_best_h3` | 3 | 0.00 |
| `branin` | late | `delta_best_h5` | 3 | 0.00 |
| `branin` | late | `delta_best_hfull` | 3 | 0.00 |
| `branin` | late | `no_stop_pen_h1` | 3 | 0.00 |
| `branin` | late | `no_stop_pen_h10` | 3 | 0.00 |
| `branin` | late | `no_stop_pen_h3` | 3 | 0.00 |
| `branin` | late | `no_stop_pen_h5` | 3 | 0.00 |
| `branin` | late | `no_stop_pen_hfull` | 3 | 0.00 |
| `branin` | late | `original_h1` | 3 | 0.67 |
| `branin` | late | `original_h10` | 3 | 0.67 |
| `branin` | late | `original_h3` | 3 | 0.67 |
| `branin` | late | `original_h5` | 3 | 0.67 |
| `branin` | late | `original_hfull` | 3 | 0.67 |
| `branin` | mid | `delta_best_h1` | 4 | 0.25 |
| `branin` | mid | `delta_best_h10` | 4 | 0.00 |
| `branin` | mid | `delta_best_h3` | 4 | 0.00 |
| `branin` | mid | `delta_best_h5` | 4 | 0.00 |
| `branin` | mid | `delta_best_hfull` | 4 | 0.00 |
| `branin` | mid | `no_stop_pen_h1` | 4 | 0.00 |
| `branin` | mid | `no_stop_pen_h10` | 4 | 0.00 |
| `branin` | mid | `no_stop_pen_h3` | 4 | 0.00 |
| `branin` | mid | `no_stop_pen_h5` | 4 | 0.00 |
| `branin` | mid | `no_stop_pen_hfull` | 4 | 0.00 |
| `branin` | mid | `original_h1` | 4 | 1.00 |
| `branin` | mid | `original_h10` | 4 | 0.00 |
| `branin` | mid | `original_h3` | 4 | 1.00 |
| `branin` | mid | `original_h5` | 4 | 0.50 |
| `branin` | mid | `original_hfull` | 4 | 0.00 |
| `hartmann` | early | `delta_best_h1` | 3 | 0.67 |
| `hartmann` | early | `delta_best_h10` | 3 | 0.33 |
| `hartmann` | early | `delta_best_h3` | 3 | 1.00 |
| `hartmann` | early | `delta_best_h5` | 3 | 1.00 |
| `hartmann` | early | `delta_best_hfull` | 3 | 0.33 |
| `hartmann` | early | `no_stop_pen_h1` | 3 | 0.67 |
| `hartmann` | early | `no_stop_pen_h10` | 3 | 0.33 |
| `hartmann` | early | `no_stop_pen_h3` | 3 | 1.00 |
| `hartmann` | early | `no_stop_pen_h5` | 3 | 1.00 |
| `hartmann` | early | `no_stop_pen_hfull` | 3 | 0.33 |
| `hartmann` | early | `original_h1` | 3 | 0.67 |
| `hartmann` | early | `original_h10` | 3 | 0.33 |
| `hartmann` | early | `original_h3` | 3 | 1.00 |
| `hartmann` | early | `original_h5` | 3 | 1.00 |
| `hartmann` | early | `original_hfull` | 3 | 0.33 |
| `hartmann` | late | `delta_best_h1` | 4 | 0.25 |
| `hartmann` | late | `delta_best_h10` | 4 | 0.50 |
| `hartmann` | late | `delta_best_h3` | 4 | 0.50 |
| `hartmann` | late | `delta_best_h5` | 4 | 0.50 |
| `hartmann` | late | `delta_best_hfull` | 4 | 0.50 |
| `hartmann` | late | `no_stop_pen_h1` | 4 | 0.25 |
| `hartmann` | late | `no_stop_pen_h10` | 4 | 0.50 |
| `hartmann` | late | `no_stop_pen_h3` | 4 | 0.50 |
| `hartmann` | late | `no_stop_pen_h5` | 4 | 0.50 |
| `hartmann` | late | `no_stop_pen_hfull` | 4 | 0.50 |
| `hartmann` | late | `original_h1` | 4 | 1.00 |
| `hartmann` | late | `original_h10` | 4 | 1.00 |
| `hartmann` | late | `original_h3` | 4 | 1.00 |
| `hartmann` | late | `original_h5` | 4 | 1.00 |
| `hartmann` | late | `original_hfull` | 4 | 1.00 |
| `hartmann` | mid | `delta_best_h1` | 3 | 0.67 |
| `hartmann` | mid | `delta_best_h10` | 3 | 0.33 |
| `hartmann` | mid | `delta_best_h3` | 3 | 0.67 |
| `hartmann` | mid | `delta_best_h5` | 3 | 0.67 |
| `hartmann` | mid | `delta_best_hfull` | 3 | 0.33 |
| `hartmann` | mid | `no_stop_pen_h1` | 3 | 0.67 |
| `hartmann` | mid | `no_stop_pen_h10` | 3 | 0.33 |
| `hartmann` | mid | `no_stop_pen_h3` | 3 | 0.67 |
| `hartmann` | mid | `no_stop_pen_h5` | 3 | 0.67 |
| `hartmann` | mid | `no_stop_pen_hfull` | 3 | 0.33 |
| `hartmann` | mid | `original_h1` | 3 | 1.00 |
| `hartmann` | mid | `original_h10` | 3 | 0.33 |
| `hartmann` | mid | `original_h3` | 3 | 1.00 |
| `hartmann` | mid | `original_h5` | 3 | 1.00 |
| `hartmann` | mid | `original_hfull` | 3 | 0.33 |

## Conclusions (heuristic — codex review required for final word)

- **Most label-consistent config**: `original_h3` (top-1 agreement 0.97).
- **Best oracle gain (when changed)**: `original_h10` (mean gain vs STOP = +4.5094).
- **Total (record, config) pairs**: 450, of which 237 had top-action changes (52.7%).

If most configs change few top actions AND the original top still loses to STOP, the counterfactual teacher is fundamentally mis-specified (Q-S5 verdict: utility not just wrong but possibly *mis-aimed*). Switch route per Q-Route recommendation.

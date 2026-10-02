# Attribution experiments, frozen 2026-09-30

Purpose: distinguish target/horizon mismatch, noisy supervision, action/budget
handling, missing state information, and dimension/function effects. These are
diagnostics of the current implementation, not a proof about all adaptive BO.

## Panels and controls

- BOCoDe: all five original families. Labels use development seeds 0–19,
  one saved state nearest 5, 15, and 25 evaluations per trajectory, without
  duplicate states. Source: `phase8_6d_h3_20seed_5family.jsonl`.
- All labels are rerolled four times: replicas 0–1 select an action; replicas
  2–3 evaluate it. Also report the reverse split. Never select and evaluate
  a counterfactual winner on the same replicas.
- Corrected continuations: full-budget rule and always-local. Full-budget
  branches run until 30 objective evaluations; STOP is an explicit early
  termination alternative, not a matched-budget sampling control.
- Original menu: canonical heterogeneous actions, restricted to runtime
  feasibility. Unit menu: six sampling actions, one evaluation each, automatic
  lazy GP fit, no standalone refit or early STOP. Random step sizes and fit
  routines remain those of the current backend.
- Legacy control: reroll the original three-action on-policy labels with fresh
  seeds. Prefix-three and terminal scores from corrected full-budget branches
  distinguish horizon and scoring changes on identical simulated paths.
- Closed-loop BOCoDe evaluation: seeds 1000–1019, five shared initial points,
  budget 30. Fit within-family RF students with identical frozen hyperparameters
  (200 trees, depth 8, leaf 5): short composite vs terminal objective targets,
  full-state vs budget-only features, original vs unit menus (2×2×2).
  Include the frozen legacy target RF, all six unit fixed sampling controls,
  and one schedule selected on development seeds only.
  Two bridge arms use the same original-menu RF predictions but mask refit
  and STOP while preserving batch sizes. This isolates zero-evaluation actions
  from the rest of the unit-action package.
- Privileged reference: rerun counterfactual selection at every live decision,
  two planning replicas, with always-local continuation. Compare prefix-three
  composite selection with full-terminal selection. Simulated objective calls
  do NOT count as real budget; this is a diagnostic, not a deployable optimizer.
  Full-budget rule-continuation reference uses only seeds 1000–1004 as a
  predeclared expensive secondary diagnostic; its small panel is exploratory.
- Dimension panel: IOH f6, f8, f15 × dimensions 2, 5, 10. Development instances
  1–3 / seeds 950–954 choose fixed/schedule controls. Test instances 10–12 /
  seeds 1000–1004 measure labels (three checkpoints, four replicas), all six
  unit fixed policies, the frozen schedule, and both local-continuation
  privileged references. Dimensions are crossed within each function, not
  inferred from five different BOCoDe functions.
- Schedules switch from unit global sampling to local at 20%, 40%, 60%, or
  80% of the post-initial evaluation budget. Pick the best schedule and the
  best fixed action using development terminal outcomes only.
- Backend diagnostic: normalized-input LogEI versus the original raw-input
  LogEI, on identical fresh BOCoDe and IOH contexts, with every other parameter
  held fixed. The current backend imports Normalize but does not apply it;
  Hartmann alone already has [0,1] bounds. This is an experiment-only override,
  not a change to the production backend or to the other experiment arms.
- Real-action random streams are keyed by evaluation count, not action count,
  so inserting a zero-evaluation refit cannot shift subsequent sampling draws.

## Endpoints and attribution

Primary endpoint: paired terminal best at exactly 30 real evaluations for
unit policies. Record real evaluations, all actions/values, unused budget,
refits, and planning costs. Original-menu STOP/refit policies may under-use
budget; this is an outcome, never silently filled with new evaluations.

Label endpoints: split-half action agreement, pair-sign disagreement excluding
ties, held-out selected-action gain, and family-scale-normalized descriptive
D. D is not a closed-loop upper bound. Scores use fixed development family
scales, not unstable five-point standard deviations on each test run.

State-value endpoint: full-state RF minus budget-only RF with matched targets,
menu, hyperparameters and training states. A null result is specific to these
features/models; it cannot establish that full histories contain no value.

Uncertainty: BOCoDe resamples paired seeds; IOH resamples instances then seeds.
Report ordinary 95% intervals and 99% intervals for five-family comparisons,
plus Holm-adjusted paired sign-randomization p-values within contrast groups.
Nonpositive lower bounds do not prove absence of benefit. No retuning based
on test results. Failures remain visible and cannot be dropped.

Implementation checks: restore identical histories/GP/RNG; respect budget with
zero-evaluation refits and two-evaluation actions; deterministic serial/parallel
agreement; replayed incumbent and fitted state match the source; selection and
evaluation replica sets are disjoint. Preserve all existing artifacts.

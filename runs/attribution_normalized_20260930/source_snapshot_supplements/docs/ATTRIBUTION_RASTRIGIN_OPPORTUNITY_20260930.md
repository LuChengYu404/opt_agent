# Independent confirmation of Rastrigin selection opportunity

Declared after observing the exploratory ten-seed Rastrigin references:
unit/block gains about +12/+14 over normalized fixed GP, but wide intervals.
Do not use those ten seeds or the normalization-confirmation seeds in this
confirmation. No retuning of GP, action parameters or planning replicas.

## Frozen controls and test

Development seeds 20–29 (existing independent development panel): normalized
GP followed by one of the other five sampling actions at real counts
10/15/20/25. Twenty two-stage schedules × ten seeds = 200 trajectories.
Include existing normalized fixed GP development outcomes as the no-switch
candidate. Freeze the best of these 21 policies by development mean only.
This adds schedules containing the strong GP action; the original
global-to-local schedules alone are insufficient to distinguish timing from
state feedback once GP is corrected.

New confirmation seeds 60000–60029, all Rastrigin, 30 real evaluations including
five initial points. Fixed normalized GP, frozen best two-stage/no-switch
policy, unit reference and five-step reference. Same four planning replicas
and five replanning times as the primary follow-up. Total 120 test trajectories
(if no-switch wins, keep a separately tagged but identical control).

Six contrasts, predeclared: schedule minus GP; unit/block minus GP;
unit/block minus schedule; block minus unit. Report paired 95%/99% bootstrap
and sign-randomization p, Holm across these six contrasts. This is one
function, not a cross-family claim. No optional stopping or additional seeds
after observing this group. Keep exploratory and confirmatory groups separate.

There are 320 new trajectories. Privileged planning calls/fits remain outside
the real evaluation budget and recorded. These references do not establish
an optimal policy or guarantee that a student can infer their choices from
the available state summaries. No student/LLM training in this supplement.

# Codex Final Review Brief — Workshop Paper

> Pre-staged for review after Track A/B/D experiments finish.

## What we're asking for

Approve / REVISE / BLOCK the **workshop paper framing** (8 pages + appendix)
based on the 4 experiments closed in this session:

1. **Track A**: EXP-RQ1.2 20-seed replication → Gate 2/3/4 confirmatory
2. **Track B**: EXP-RQ0.5 Gate 5 affine rescaling → invariant under y → scale*y + shift
3. **Track C**: (already closed) EXP-RQ0.5 joint 10-seed 5-policy comparison
4. **Track D**: LLM action distribution analysis → "global-then-local" mechanism

## Specific claims to verify (per codex Q1-Q3 + claim wording)

### RQ1 — Decision Landscape
- 3 regimes (Hartmann 0.684 > Rastrigin 0.368 ≈ Ackley 0.309 > Branin 0.048 > SixHumpCamel 0.015) using `adaptive-value` metric
- Gate 3 (aggregate_mass) on 20-seed 5-family dataset — CI to verify
- Gate 4 stop-gain CI lo > 0 on 5/5 families — CI to verify
- Gate 5 affine invariance on 5-family × 4 transforms — all buckets pass

### RQ3 — Selective Reasoning
- LLM beats Surrogate by +17.54 best_y on Rastrigin (paired-t p=0.0003, n=10)
- LLM uses "global-then-local" temporal schedule (100% global step 0–9, transitions to 100% local by step 13)
- FrozenRouter NEGATIVE result on IOH (p=0.138, did not generalize from BOCoDe)
- RoutingAware vs LLM-everywhere NOT significant on TOTAL (p=0.51)

### What's NOT claimed (limitations to verify we own)
- RQ2 (cross-family transfer): zero experiments — workshop framing only
- Pareto frontier (LLM cost): not yet measured
- External baselines (lenz, PlugBO): not yet compared

## Files to read in order

1. `docs/workshop_paper_outline.md` — narrative + claim index
2. `runs/EXP_RQ1_2_REPORT.md` (updated) — RQ1 evidence
3. `runs/EXP_AFFINE_REPORT.md` — Gate 5 result
4. `runs/exp_rq0_5_action_distribution.md` — mechanism
5. `runs/EXP_RQ0_5_REPORT.md` — pilot (Rastrigin +17.54)

## Per codex priorities (top-down)

### Critical
1. **Numeric accuracy**: every claim in the outline must trace to a file:line in `runs/EXP_*.md` or `runs/*.json`
2. **No over-claim**: especially around Gate 2 (which we previously flagged as FAIL at top_action_share=0.6), Gate 4 (only Rastrigin passes all 3 baselines), and 2×2 figure (which is "hypothesis, not populated")
3. **Negative results framed correctly**: IOH router NEGATIVE result + RoutingAware vs LLM NS result MUST both appear in §5, not buried

### Important
4. **Test count**: 380 tests (was 364 before Track B); claim should match
5. **Sample size honesty**: per-family n=10–20, per-(problem, phase) bucket may have n<5
6. **Reusability**: track uses of existing machinery (`affine_rescaling_test`, `validate_phase8_6b.py`, `DatasetGenerator`) vs new code

### Nice-to-have
7. **Figure suggestions**: 2×2 (low/high decision opportunity × low/high ambiguity) — should be honest about empirical coverage
8. **Appendix organization**: test inventory, replay invariants, per-seed tables

## Expected verdict structure

For each of §3-§7 in the outline, codex should issue:
- APPROVE: wording is honest, numbers trace correctly
- REVISE: specific changes needed (with proposed text)
- BLOCK: structural issue requires new experiment

We will iterate until APPROVE on all sections, then submit.
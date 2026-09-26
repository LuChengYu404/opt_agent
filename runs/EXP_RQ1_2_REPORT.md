# EXP-RQ1.2 — Counterfactual Oracle Replay Report (5-family version, post-codex-revision)

**Date**: 2026-09-26
**Status**: Code revisions per `gpt-5.6-sol` review (2026-09-26) applied
**Datasets**:
- 3-family baseline (kept for history, NOT cited as independent evidence): `phase8_6b_h1_10seed.jsonl`, `phase8_6b_h3_10seed.jsonl`, `phase8_6d_hfull_10seed.jsonl` (226 records each; Branin/Hartmann/Ackley)
- **5-family canonical**: `phase8_6d_h3_10seed_5family.jsonl` (354 records; Branin/Hartmann/Ackley/SixHumpCamel/Rastrigin × 10 seeds; **horizon=3**, not horizon=30 — naming corrected post-review)

> **Naming correction (per codex review 2026-09-26)**: The previous file `phase8_6d_hfull_10seed_5family.jsonl` was generated with `horizon=3`, not the `horizon=30` implied by `hfull`. Renamed to `phase8_6d_h3_10seed_5family.jsonl`. A true `hfull` (horizon=30) dataset would require ~30× generation cost (~90 minutes) and is **out of scope** for RQ1.

## 1. Headline result (5-family canonical, post-revision)

| Gate | Result |
|---|---|
| 2: stability (top_action_share ≥ 0.6) | FAIL (**5/15** buckets pass) |
| 3: aggregate_mass | **PASS** (mean = 1.124; **seed-clustered 95% CI = [0.695, 1.722]**) |
| 4a: stop-gain CI lo > 0 | **PASS 5/5 families** |
| 4b: uniform-gain CI lo > 0 | PASS 3/5 (hartmann, rastrigin, sixhumpcamel) |
| 4c: behavior-gain CI lo > 0 | PASS 1/5 (rastrigin only) |

## 2. Three findings (per `gpt-5.6-sol` review 2026-09-26, exploratory)

> **C1.** Oracle-labeled first actions produced positive **mean normalized terminal gain** over STOP on the five evaluated BOCoDe families. Per-family, unadjusted 95% interval lower bound > 0 on 5/5. **Not** "universal" — only across the evaluated families; no claim about problems generally.

> **C2.** **Per-family gain estimates and interval conclusions varied** across the five families. Rastrigin was the only family with per-family, unadjusted 95% interval lower bound > 0 against all three baselines (STOP, uniform, behavior). Hartmann passed STOP and uniform. Branin, SixHumpCamel, and Ackley passed STOP only. **Not** a "dimension-dependent" claim — that would require a crossed design (same problem at multiple dimensions), which the current allowlist does not support.

> **C3.** Oracle-label argmax actions were heterogeneous across seeds within coarse (problem, phase) strata. Early-phase top-label shares ranged 0.20 (rastrigin/early) to 0.70 (hartmann/early). **Not** a "policy multimodality" claim — could reflect different states across seeds (a deterministic state-conditioned policy would still show this pattern). Causal interpretation deferred to within-state rollout diagnostics.

## 3. Gate 4 — per-family oracle replay

| Family | behavior CI lo | uniform CI lo | **stop CI lo** |
|---|---|---|---|
| ackley (10D noisy) | -0.498 | -0.855 | **+1.084** ✓ |
| branin (2D smooth) | -0.011 | -0.004 | **+0.093** ✓ |
| hartmann (6D) | -0.093 | **+0.203** ✓ | **+1.493** ✓ |
| **rastrigin (10D multimodal)** | **+0.032** ✓ | **+0.136** ✓ | **+0.840** ✓ |
| sixhumpcamel (2D) | -0.004 | **+0.008** ✓ | **+0.044** ✓ |

**Bold = CI lo > 0 (statistically detectable positive gain).**

## 4. Gate 3 — seed-clustered CI (new per codex review)

Previously Gate 3 reported only the point estimate of mean adjusted_lift. Per codex, this is now augmented with a **bootstrap 95% CI over the 50 seed-level adjusted_lift values** (5 families × 10 seeds):

```
adjusted_lift_mean    = 1.124
seed_clustered_ci95   = [0.695, 1.722]
n_seeds               = 50
```

The CI lower bound (0.695) is strictly positive, so Gate 3 passes both point-estimate and CI criteria.

**Important caveat (per codex)**: `adjusted_lift` measures the **log ratio of exploration probability mass to STOP probability mass**, minus a uniform-exploration baseline. It is **not** "top-1 action predicts better outcomes." The CI supports the claim that the dataset's `gold_probs` exploration mass is heavier than STOP mass, not that the top-1 action is optimal.

## 5. Gate 2 — stability by (problem, phase) (corrected count)

Pass criterion: top_action_share ≥ 0.6. **5/15 buckets pass** (corrected from previous "4/15" — `ackley/early` share=0.60 was overlooked):

```
✓ ackley       /early : top='local_exploitation'   share=0.60 n_seeds=10
✗ ackley       /late  : top='local_exploitation'   share=0.50 n_seeds=8
✗ ackley       /mid   : top='local_exploitation'   share=0.50 n_seeds=10
✗ branin       /early : top='local_exploitation'   share=0.40 n_seeds=10
✗ branin       /late  : top='retrain_surrogate'    share=0.40 n_seeds=10
✓ branin       /mid   : top='retrain_surrogate'    share=0.60 n_seeds=10
✓ hartmann     /early : top='retrain_surrogate'    share=0.70 n_seeds=10
✗ hartmann     /late  : top='local_exploitation'   share=0.44 n_seeds=9
✗ hartmann     /mid   : top='error_guided_sampling' share=0.30 n_seeds=10
✗ rastrigin    /early : top='global_exploration'   share=0.20 n_seeds=10
✗ rastrigin    /late  : top='local_exploitation'   share=0.44 n_seeds=9
✗ rastrigin    /mid   : top='local_exploitation'   share=0.40 n_seeds=10
✗ sixhumpcamel /early : top='retrain_surrogate'    share=0.50 n_seeds=10
✓ sixhumpcamel /late  : top='local_exploitation'   share=0.71 n_seeds=7
✓ sixhumpcamel /mid   : top='retrain_surrogate'    share=0.62 n_seeds=8
```

**Interpretation (per codex)**: This is evidence against a stable coarse `(family, phase) → argmax action` rule. It is **not** automatically evidence of conditional policy multimodality — the heterogeneity could reflect different states across seeds (a deterministic state-conditioned policy would still show this pattern). Diagnostic experiments deferred.

## 6. Branin — properly framed

> **On Branin, the oracle-selected first action had a statistically detectable but small-in-magnitude-on-the-normalized-gain-scale disadvantage relative to behavior**: mean normalized gain −0.00494, bootstrap 95% CI [−0.01065, −0.00035].
>
> The CI excludes zero, so the effect is detectable. The practical importance is unclear; a smallest-meaningful-effect δ test is needed to claim equivalence. We do **not** claim "no measurable difference."

## 7. Cross-version consistency (3-family vs 5-family)

What is consistent:
- Gate 3 PASS uniformly across 3-family and 5-family versions
- Gate 4 stop-gain PASS uniformly (3/3 in 3-family, 5/5 in 5-family)
- Branin behavior-gain CI upper bound near or below 0 in both versions

What changed (and why):
- Hartmann behavior-gain regressed from PASS (3-family hfull) to FAIL (5-family). **This is NOT a sample-size tightening effect** — Hartmann's per-family sample size remained 10 seeds in both versions. The change reflects a different labeling configuration (`horizon=3` in 5-family vs the older generator's configuration in 3-family hfull).

**Conclusion**: The 3-family and 5-family versions are **not independent replications** because they used different generator configurations. Use 5-family as canonical; cite 3-family only for horizon/configuration sensitivity discussion.

## 8. Code revisions (post-codex review)

1. `src/opt_agent/dataset/validation.py`:
   - `MassGateResult` adds `seed_clustered_ci95: tuple[float, float] | None = None` (defaults to None for backward compat).
   - `aggregate_mass()` now computes bootstrap CI over per-seed adjusted_lift values when n_seeds ≥ 5; uses CI lower bound > 0 as gate criterion. Falls back to point estimate > 0 for n_seeds < 5.
   - New helper `_bootstrap_ci_mean()`: pure-Python bootstrap 95% CI for the mean.
2. `scripts/validate_phase8_6b.py`: report now includes `gate3.seed_clustered_ci95`.
3. Tests: `tests/test_phase8_5_gates.py` + `tests/test_phase8_6b_gates.py` — **16/16 pass** (no breakage from the API change).
4. Full suite: `tests/` — **364/364 pass** (no regressions).

## 9. Files

- `runs/phase8_6d_h3_10seed_5family.jsonl` — 354 records, 5 families × 10 seeds, replay-capable (100%); **horizon=3** (not hfull)
- `runs/exp_rq1_2_h3_5family_report.json` — full Gate 2/3/4 report (with new Gate 3 CI)
- `runs/exp_rq1_2_h{1,3,hfull}_report.json` — 3-family baseline reports (history only)
- `runs/EXP_RQ1_2_REPORT.md` — this file (canonical reference)
- `src/opt_agent/dataset/validation.py` — Gate 3 + CI logic (revised)
- `scripts/validate_phase8_6b.py` — gate runner (revised)

## 10. Open follow-ups (corrected per codex priority order)

1. ~~Provenance repair~~ — done (file rename + report documentation)
2. ~~Statistical repair~~ — done (Gate 3 CI added, claim wording corrected)
3. **Replication** — expand to 20+ seeds for tighter CIs (~5 min regen + ~10 min replay). Recommended for paper-quality numbers.
4. **Gate 2 diagnostics** — threshold sensitivity curve + within-state rollout agreement. Determines whether C3 can be strengthened to a causal claim.
5. **EXP-RQ2.1** — cross-family transfer (5-fold leave-one-family-out). Only after #3 and #4.
6. **EXP-RQ3.0** — Level 3 LLM baseline. Defer until #3-#5 are sound.

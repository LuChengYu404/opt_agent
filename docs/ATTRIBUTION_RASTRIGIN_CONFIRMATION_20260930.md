# Independent Rastrigin normalization confirmation

Motivation: the first attribution panel had positive normalization gains on
Rastrigin, but the corrected-GP follow-up's 20 fresh seeds had wide intervals.
This supplement is declared after that observation, before examining the new
group. It does not retune parameters, action ranking, teachers or references.

Freeze 100 independent seeds 50000–50099, budget 30 including five initial
points. Three fixed policies: raw surrogate acquisition, normalized surrogate
acquisition, and the raw best action frozen on development seeds 20–29
(local_exploitation). All algorithms and streams use the existing follow-up
runner. Total 300 trajectories. No optional stopping or sample-size extension.

Report normalized surrogate minus raw surrogate and normalized surrogate minus
the frozen raw best as two predeclared contrasts. Paired 95%/99% bootstrap,
sign-randomization p and Holm correction across these two contrasts. Keep
the 20-seed and 100-seed groups separate; do not hide the inconclusive earlier
group or claim the supplement was part of the original preregistered panel.
This confirms baseline robustness only, not adaptive selection value.

Use the post-freeze corrected restore implementation (bounds preservation),
which is not invoked by these fixed loops. Preserve its code hashes and the
unchanged development selection. Backend fitting/proposal settings are the
same as the primary panel; no LLM or student training occurs.

# Posterior-feature visibility supplement

Code/data inspection, without examining held-out performance contrasts,
found that only 4/280 selected development states have a current fitted GP.
After every unit sampling action, evaluate() invalidates the GP, so the next
ordinary decision gets no validation-RMSE / posterior-uncertainty fields.
Existing feature extraction imputes them as zeros. This is a concrete ambiguity
in the full-state vs budget-only experiment.

Add one diagnostic student on the same 280 states, same four-replica terminal
targets, same six unit actions, same per-family RF hyperparameters, and same
test seeds 1000–1019. Fit the ORIGINAL unnormalized GP before feature extraction
on both training and test states. Use the existing 15 fields, with posterior
fields now populated. Do not change labels or GP normalization. Reuse the fit
when the selected sampling action needs it; record extra compute fits explicitly.

This is an information-availability intervention plus GP-fit computation. It
does not test richer history encoders, candidate-conditioned GP features, or
all possible state representations. Primary arms remain frozen and unchanged.

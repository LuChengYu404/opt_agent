"""Public re-exports for opt_agent.rollout."""

from opt_agent.rollout.rollout import (
    Candidate,
    CounterfactualResult,
    CounterfactualRollout,
    RolloutOutput,
    to_gold_probs,
)
from opt_agent.rollout.utility import UtilityWeights, compute_utility

__all__ = [
    "Candidate",
    "CounterfactualResult",
    "CounterfactualRollout",
    "RolloutOutput",
    "UtilityWeights",
    "compute_utility",
    "to_gold_probs",
]

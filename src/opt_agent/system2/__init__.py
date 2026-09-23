"""System-2 reasoning module: high-level fallback when System-1 is uncertain.

Phase 9 ships only a stub. Phase 11 will swap in a Tau/LLM harness.
Per codex gpt-6-sol review 2026-09-23 (major M2):
  - System-2 stub must delegate to an explicit baseline (e.g. RulePolicy),
    NOT default to STOP. Returning STOP unconditionally turns every
    "low confidence → escalate" into "premature termination" and pollutes
    both routing stats and closed-loop outcomes.
"""

from opt_agent.system2.base import System2Decision, System2Policy
from opt_agent.system2.fallback import FallbackSystem2

__all__ = ["System2Decision", "System2Policy", "FallbackSystem2"]

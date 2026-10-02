"""Action-menu boundary registry — machine-readable twin of the guide §2.4.

`docs/RESEARCH_BOUNDARY_GUIDE.md` §6.1 requires that adding an action slot
passes the three-item preflight check and lands in BOTH the guide table and
the code menu. This module is the code-side half of that contract:
`tests/test_boundary_registry.py` fails whenever the ActionType menu and
this registry (or the guide §2.4 table) diverge, so no action can be added,
removed, or reclassified silently.

Scope note (codex round-2 C2, honest description): this registry is a
CI-consistency artifact plus a classification query API
(`boundary_class_of` / `submenu_of`). No runtime dispatch or candidate-
generation path consumes the classifications yet; the runtime menu source
is `actions.factory.ALL_CANDIDATE_SPECS`, whose sync with this registry is
asserted by test_factory_candidates_match_registry.

Classification vocabulary mirrors the guide's taxonomy (§2):
  A : definitional / exogenous-frozen (not used by any current action)
  B1: in-episode allowed, theory-backed
  B2: in-episode allowed, empirically established
  B3: in-episode allowed, thin literature
  C : forbidden in-episode (must be externalized to the meta level)
Submenus (guide §1.1): A_pt = point-selection, A_flow = process control.
"""

from __future__ import annotations

from dataclasses import dataclass

from opt_agent.actions.models import ActionType

BOUNDARY_CLASSES = ("A", "B1", "B2", "B3", "C")
SUBMENUS = ("A_pt", "A_flow")


@dataclass(frozen=True)
class BoundaryClassification:
    """One row of the guide §2.4 table."""

    boundary_class: str
    submenu: str
    guide_note: str


BOUNDARY_REGISTRY: dict[ActionType, BoundaryClassification] = {
    ActionType.GLOBAL_EXPLORATION: BoundaryClassification(
        boundary_class="B2",
        submenu="A_pt",
        guide_note="采样模式位（h2：TS，LMABO portfolio exploratory 组，后验采样 argmax）",
    ),
    ActionType.LOCAL_EXPLOITATION: BoundaryClassification(
        boundary_class="B2",
        submenu="A_pt",
        guide_note="采样模式位（h2：PI，τ=incumbent，step 为信任域半径）",
    ),
    ActionType.UNCERTAINTY_SAMPLING: BoundaryClassification(
        boundary_class="B2",
        submenu="A_pt",
        guide_note="采样模式位（h2：PosSTD 解析 AF 连续最大化）",
    ),
    ActionType.SURROGATE_ACQUISITION: BoundaryClassification(
        boundary_class="B2",
        submenu="A_pt",
        guide_note="标准 AF 求点（默认 LogEI；af 参数可选 LMABO 12-AF，h2；按需拟合）",
    ),
    ActionType.RETRAIN_SURROGATE: BoundaryClassification(
        boundary_class="B1",
        submenu="A_flow",
        guide_note="族内 MLE 显式重拟合位；禁止扩展为换族/换核（元层动作）",
    ),
    ActionType.STOP: BoundaryClassification(
        boundary_class="B1",
        submenu="A_flow",
        guide_note="终止型动作；理论对应 Bayesian / probabilistic-regret stopping",
    ),
}


class RegistrySyncError(ValueError):
    """ActionType menu and boundary registry are out of sync."""


def boundary_class_of(action: ActionType) -> str:
    """Query the guide-taxonomy class (A/B1/B2/B3/C) of an action."""
    return BOUNDARY_REGISTRY[action].boundary_class


def submenu_of(action: ActionType) -> str:
    """Query the guide submenu (A_pt point-selection / A_flow process)."""
    return BOUNDARY_REGISTRY[action].submenu


def verify_registry_sync() -> None:
    """Raise RegistrySyncError unless registry and menu agree 1:1.

    Checks: (i) every ActionType has exactly one registry entry and the
    registry contains no unknown actions; (ii) every entry uses the allowed
    boundary-class and submenu vocabulary; (iii) every entry carries a
    non-empty guide note. Called by tests/test_boundary_registry.py.
    """
    menu = set(ActionType)
    registered = set(BOUNDARY_REGISTRY)
    problems: list[str] = []
    unregistered = menu - registered
    if unregistered:
        names = sorted(a.value for a in unregistered)
        problems.append(
            "actions present in ActionType but missing a boundary "
            f"classification (pass guide §3.4 three-item preflight check, "
            f"then add a "
            f"BOUNDARY_REGISTRY entry + guide §2.4 row): {names}"
        )
    unknown = registered - menu
    if unknown:
        names = sorted(a.value for a in unknown)
        problems.append(
            f"BOUNDARY_REGISTRY entries for actions that no longer exist: {names}"
        )
    for action, entry in BOUNDARY_REGISTRY.items():
        if entry.boundary_class not in BOUNDARY_CLASSES:
            problems.append(
                f"{action.value}: boundary_class {entry.boundary_class!r} not in "
                f"{BOUNDARY_CLASSES}"
            )
        if entry.submenu not in SUBMENUS:
            problems.append(
                f"{action.value}: submenu {entry.submenu!r} not in {SUBMENUS}"
            )
        if not entry.guide_note.strip():
            problems.append(f"{action.value}: empty guide_note")
    if problems:
        raise RegistrySyncError("; ".join(problems))

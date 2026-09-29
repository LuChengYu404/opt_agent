"""Boundary registry sync tests — enforce RESEARCH_BOUNDARY_GUIDE.md §2.4/§6.1.

The point of these tests is that the action menu CANNOT drift from its
boundary classification: adding/removing/reclassifying an action without
touching both the registry and the guide §2.4 table turns the suite red.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opt_agent.actions.boundary_registry import (
    BOUNDARY_CLASSES,
    BOUNDARY_REGISTRY,
    SUBMENUS,
    RegistrySyncError,
    verify_registry_sync,
)
from opt_agent.actions.models import ActionType

GUIDE_PATH = (
    Path(__file__).resolve().parents[1] / "docs" / "RESEARCH_BOUNDARY_GUIDE.md"
)


def test_registry_matches_action_type_menu_exactly():
    verify_registry_sync()  # raises RegistrySyncError on any drift
    assert set(BOUNDARY_REGISTRY) == set(ActionType)


def test_registry_error_names_missing_action():
    class Fake:
        pass

    # Simulate a menu extension the registry does not know about.
    registry_entries = dict(BOUNDARY_REGISTRY)
    missing = ActionType.STOP
    del registry_entries[missing]
    import opt_agent.actions.boundary_registry as mod

    original = mod.BOUNDARY_REGISTRY
    mod.BOUNDARY_REGISTRY = registry_entries
    try:
        with pytest.raises(RegistrySyncError) as exc:
            verify_registry_sync()
        assert missing.value in str(exc.value)
    finally:
        mod.BOUNDARY_REGISTRY = original


def test_submenus_partition_the_menu():
    """Every action is in exactly one submenu; both submenus are used."""
    for action, entry in BOUNDARY_REGISTRY.items():
        assert entry.submenu in SUBMENUS
    submenus_used = {entry.submenu for entry in BOUNDARY_REGISTRY.values()}
    assert submenus_used == set(SUBMENUS)


def test_classifications_use_guide_vocabulary():
    for action, entry in BOUNDARY_REGISTRY.items():
        assert entry.boundary_class in BOUNDARY_CLASSES, action


def _guide_section_2_4_rows() -> dict[str, tuple[str, str, str]]:
    """Parse the guide §2.4 markdown table -> {action_value: (class, submenu-ish, row)}.

    The table columns are: ActionType | 归类 | 备注. This deliberately parses
    the rendered document rather than importing constants, so a doc edit
    alone cannot desynchronize code from spec.
    """
    text = GUIDE_PATH.read_text(encoding="utf-8")
    start = text.index("### 2.4")
    end = text.index("\n## ", start)
    section = text[start:end]
    lines = section.splitlines()
    header_idx = next(
        i for i, ln in enumerate(lines) if ln.strip().startswith("| ActionType |")
    )
    rows: dict[str, tuple[str, str, str]] = {}
    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped.startswith("|"):
            break  # table ended
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 2 or cells[0].startswith("-"):
            continue  # separator row
        # The guide table names enum MEMBERS (GLOBAL_EXPLORATION); normalize
        # to values (global_exploration) for comparison with ActionType.
        try:
            action_value = ActionType[cells[0]].value
        except KeyError:
            action_value = cells[0]
        rows[action_value] = (cells[1], cells[2] if len(cells) > 2 else "", line)
    return rows


@pytest.mark.skipif(not GUIDE_PATH.exists(), reason="guide doc not present")
def test_guide_table_rows_match_registry():
    """Guide §2.4 table and BOUNDARY_REGISTRY must classify identically."""
    rows = _guide_section_2_4_rows()
    assert set(rows) == {a.value for a in ActionType}, (
        f"guide §2.4 rows {sorted(rows)} != menu {sorted(a.value for a in ActionType)}"
    )
    for action, entry in BOUNDARY_REGISTRY.items():
        guide_class, _, line = rows[action.value]
        assert guide_class == entry.boundary_class, (
            f"{action.value}: guide says {guide_class!r}, registry says "
            f"{entry.boundary_class!r} (row: {line!r})"
        )


@pytest.mark.skipif(not GUIDE_PATH.exists(), reason="guide doc not present")
def test_guide_mentions_current_menu_size():
    """The guide's normative text must not claim a menu size the code lacks."""
    text = GUIDE_PATH.read_text(encoding="utf-8")
    claimed = {int(m) for m in re.findall(r"现 (\d+) 个 ActionType", text)}
    actual = len(ActionType)
    for c in claimed:
        assert c == actual, (
            f"guide claims {c} ActionTypes, code menu has {actual}"
        )

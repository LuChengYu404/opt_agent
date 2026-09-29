"""Boundary registry sync tests — enforce RESEARCH_BOUNDARY_GUIDE.md §2.4/§6.1.

The point of these tests is that the action menu CANNOT drift from its
boundary classification: adding/removing/reclassifying an action without
touching both the registry and the guide §2.4 table turns the suite red.

Verified columns (guide §2.4 schema): ActionType | 归类 | 子菜单 | 备注 —
menu membership, boundary class, and submenu are compared strictly; the
备注 column is free text and is only required to be non-empty. The parser
fails closed on malformed tables (bad header, short rows, duplicate or
unknown action rows).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opt_agent.actions.boundary_registry import (
    BOUNDARY_CLASSES,
    BOUNDARY_REGISTRY,
    SUBMENUS,
    RegistrySyncError,
    boundary_class_of,
    submenu_of,
    verify_registry_sync,
)
from opt_agent.actions.models import ActionType

GUIDE_PATH = (
    Path(__file__).resolve().parents[1] / "docs" / "RESEARCH_BOUNDARY_GUIDE.md"
)

EXPECTED_HEADER = ("ActionType", "归类", "子菜单", "备注")


def _parse_section_rows(section_text: str) -> dict[str, tuple[str, str, str]]:
    """Strictly parse a §2.4-style markdown table.

    Returns {action_value: (boundary_class, submenu, note)}. Raises
    AssertionError on schema violations — the sync test must fail closed,
    not skip quietly.
    """
    lines = section_text.splitlines()
    header_hits = [i for i, ln in enumerate(lines)
                   if ln.strip().startswith("| ActionType |")]
    assert len(header_hits) == 1, (
        f"expected exactly one §2.4 table, found {len(header_hits)}")
    header_idx = header_hits[0]
    header_cells = [c.strip() for c in lines[header_idx].strip().strip("|").split("|")]
    assert len(header_cells) == 4 and tuple(header_cells) == EXPECTED_HEADER, (
        f"unexpected §2.4 table header: {header_cells!r} "
        f"(expected exactly {EXPECTED_HEADER})"
    )
    rows: dict[str, tuple[str, str, str]] = {}
    table_ended = False
    saw_row = False
    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped.startswith("|"):
            # ANY non-table line — blank included — ends the first table
            # once a data row was seen; later '|' rows are a second table.
            if saw_row:
                table_ended = True
            continue
        assert not table_ended, (
            "found a second markdown table inside section 2.4 — the sync "
            "check covers exactly one table")
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if not (cells and cells[0].startswith("-")):
            saw_row = True
        if cells and cells[0].startswith("-"):
            continue  # separator row
        assert len(cells) == 4, (
            f"malformed §2.4 row (needs exactly 4 cells, got {len(cells)}): "
            f"{line!r}")
        name, cls, submenu, note = cells
        try:
            action_value = ActionType[name].value
        except KeyError:
            raise AssertionError(
                f"§2.4 row names unknown action {name!r}: {line!r}")
        assert action_value not in rows, (
            f"duplicate §2.4 row for {action_value!r}")
        assert note, f"§2.4 row for {name} has an empty 备注 cell"
        rows[action_value] = (cls, submenu, note)
    return rows


def _guide_section_2_4_rows() -> dict[str, tuple[str, str, str]]:
    """Parse the guide's §2.4 table (reads the rendered document, so a doc
    edit alone cannot desynchronize code from spec)."""
    text = GUIDE_PATH.read_text(encoding="utf-8")
    start = text.index("### 2.4")
    end = text.index("\n## ", start)
    return _parse_section_rows(text[start:end])


# ---------------------------------------------------------------------------
# Registry <-> menu
# ---------------------------------------------------------------------------


def test_registry_matches_action_type_menu_exactly():
    verify_registry_sync()  # raises RegistrySyncError on any drift
    assert set(BOUNDARY_REGISTRY) == set(ActionType)


def test_registry_error_names_missing_action():
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


def test_query_accessors():
    assert boundary_class_of(ActionType.RETRAIN_SURROGATE) == "B1"
    assert submenu_of(ActionType.RETRAIN_SURROGATE) == "A_flow"
    assert submenu_of(ActionType.GLOBAL_EXPLORATION) == "A_pt"


def test_submenus_partition_the_menu():
    """Every action is in exactly one submenu; both submenus are used."""
    submenus_used = {entry.submenu for entry in BOUNDARY_REGISTRY.values()}
    assert submenus_used == set(SUBMENUS)


def test_classifications_use_guide_vocabulary():
    for action, entry in BOUNDARY_REGISTRY.items():
        assert entry.boundary_class in BOUNDARY_CLASSES, action


def test_factory_candidates_match_registry():
    """The runtime menu source (ALL_CANDIDATE_SPECS) and the boundary
    registry must cover exactly the same actions — the registry is the
    classification side of the factory's generation side."""
    from opt_agent.actions.factory import ALL_CANDIDATE_SPECS

    spec_actions = {action_type for _, action_type, _ in ALL_CANDIDATE_SPECS}
    assert spec_actions == set(BOUNDARY_REGISTRY), (
        "factory/registry drift: factory-only="
        f"{sorted(a.value for a in spec_actions - set(BOUNDARY_REGISTRY))}, "
        "registry-only="
        f"{sorted(a.value for a in set(BOUNDARY_REGISTRY) - spec_actions)}"
    )


# ---------------------------------------------------------------------------
# Guide table parser (unit tests on synthetic tables)
# ---------------------------------------------------------------------------

_GOOD_SECTION = """
### 2.4 表

| ActionType | 归类 | 子菜单 | 备注 |
|---|---|---|---|
| GLOBAL_EXPLORATION | B2 | A_pt | 采样模式位 |
| STOP | B1 | A_flow | 终止型动作 |
"""

_DUPLICATE_SECTION = _GOOD_SECTION.replace(
    "| STOP | B1 | A_flow | 终止型动作 |",
    "| GLOBAL_EXPLORATION | B2 | A_pt | 重复行 |")

_UNKNOWN_SECTION = _GOOD_SECTION + "| NEW_SHINY_ACTION | B2 | A_pt | 未注册 |\n"

_SHORT_ROW_SECTION = _GOOD_SECTION + "| UNCERTAINTY_SAMPLING | B2 |\n"

_BAD_HEADER_SECTION = _GOOD_SECTION.replace("| 子菜单 |", "| 菜单 |")


def test_parser_accepts_well_formed_table():
    rows = _parse_section_rows(_GOOD_SECTION)
    assert rows == {"global_exploration": ("B2", "A_pt", "采样模式位"),
                    "stop": ("B1", "A_flow", "终止型动作")}


@pytest.mark.parametrize("section,fragment", [
    (_DUPLICATE_SECTION, "duplicate"),
    (_UNKNOWN_SECTION, "unknown action"),
    (_SHORT_ROW_SECTION, "malformed"),
    (_BAD_HEADER_SECTION, "header"),
])
def test_parser_fails_closed(section, fragment):
    with pytest.raises(AssertionError) as exc:
        _parse_section_rows(section)
    assert fragment in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Guide <-> registry (live document)
# ---------------------------------------------------------------------------


def test_guide_table_rows_match_registry():
    """Guide §2.4 table and BOUNDARY_REGISTRY must classify identically
    (class AND submenu columns)."""
    rows = _guide_section_2_4_rows()
    assert set(rows) == {a.value for a in ActionType}, (
        f"guide §2.4 rows {sorted(rows)} != menu "
        f"{sorted(a.value for a in ActionType)}"
    )
    for action, entry in BOUNDARY_REGISTRY.items():
        guide_class, guide_submenu, _ = rows[action.value]
        assert guide_class == entry.boundary_class, (
            f"{action.value}: guide class {guide_class!r} != registry "
            f"{entry.boundary_class!r}"
        )
        assert guide_submenu == entry.submenu, (
            f"{action.value}: guide submenu {guide_submenu!r} != registry "
            f"{entry.submenu!r}"
        )


def test_guide_mentions_current_menu_size():
    """The guide's normative text must not claim a menu size the code lacks."""
    import re

    text = GUIDE_PATH.read_text(encoding="utf-8")
    claimed = {int(m) for m in re.findall(r"现 (\d+) 个 ActionType", text)}
    actual = len(ActionType)
    for c in claimed:
        assert c == actual, (
            f"guide claims {c} ActionTypes, code menu has {actual}"
        )

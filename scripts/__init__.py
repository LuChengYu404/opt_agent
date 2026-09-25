"""Entry-point scripts (CLI drivers).

Adding this `__init__.py` makes `scripts` a regular Python package so tests
can import helpers (e.g. `_make_system2` from `run_hybrid_smoke`) without
sys.path manipulation.
"""

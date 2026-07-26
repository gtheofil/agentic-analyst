"""Test package.

Present so `tests.conftest` is importable by name — tests annotate fixtures
with the `FakeLLM` type defined there, and mypy needs a real import path for it.
"""

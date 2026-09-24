"""Pytest bootstrap for the plugin's test suite.

The plugin under test is a *directory plugin*: Hermes requires an ``__init__.py`` at the repository
root, and the plugin's modules import each other with relative imports. That makes the repository root
a package — which collides with pytest's default ``prepend`` import mode, since collecting ``__init__.py``
as a test module is wrong.

So the suite runs in ``importlib`` mode (set in ``pytest.ini``) and this file installs the plugin as a
package named ``jev_effort_router_under_test`` whose ``__path__`` is the repository root — the same arrangement
Hermes' loader creates (it imports the plugin directory as ``hermes_plugins.<slug>``). Registering the
bare submodule names as well keeps ``from config import load_settings`` working in the tests.

``__init__.py`` is deliberately NOT executed here; ``tests/test_registration.py`` does that explicitly,
since that module carries the registration side effects under test.

The test doubles live in ``tests/stubs.py``, which the test modules import directly.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
PACKAGE = "jev_effort_router_under_test"

#: The repository root must contain ``__init__.py`` (Hermes' plugin requirement), so pytest sees a
#: module beside the tests and tries to collect it. It is the plugin entry point, not a test module.
collect_ignore = ["../__init__.py"]

for entry in (str(ROOT), str(TESTS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def _install_package() -> None:
    if PACKAGE in sys.modules:
        return
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(ROOT)]  # type: ignore[attr-defined]
    package.__package__ = PACKAGE
    sys.modules[PACKAGE] = package
    for child in sorted(ROOT.glob("*.py")):
        stem = child.stem
        if stem == "__init__":  # imported explicitly by the registration tests
            continue
        module = importlib.import_module(f"{PACKAGE}.{stem}")
        sys.modules.setdefault(stem, module)
        setattr(package, stem, module)


_install_package()


@pytest.fixture(autouse=True)
def _router_api_key(monkeypatch):
    """Every test runs as an operator with a configured key.

    The client short-circuits before the request when the backend's key (``TYPESAFE_API_KEY`` by
    default, ``OPENROUTER_API_KEY`` for the OpenRouter backend) is absent, so without
    this every routing test would take the "no key" path and silently pass for the wrong reason.
    Tests that assert the unconfigured behaviour delete it explicitly.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")

"""Each package stands on its own.

The layers import in one direction - `model.expr` under `syntax` under the
engine, the display and the session - and a package `__init__` that reaches
back up the stack breaks that silently: the import works or fails depending on
which package the program happened to import first. A fresh interpreter per
package is the only way to catch it.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

PACKAGES = [
    "rederive.model",
    "rederive.model.session",
    "rederive.syntax",
    "rederive.display",
    "rederive.engine",
]


@pytest.mark.parametrize("package", PACKAGES)
def test_a_package_imports_before_anything_else(package: str) -> None:
    subprocess.run([sys.executable, "-c", f"import {package}"], check=True)

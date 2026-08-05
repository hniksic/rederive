"""Each package stands on its own, and the app side holds no mathematics.

The layers import in one direction - `model.expr` under `syntax` under the
engine, the display and the session - and a package `__init__` that reaches
back up the stack breaks that silently: the import works or fails depending on
which package the program happened to import first. A fresh interpreter per
package is the only way to catch it.

The engine's own two halves are held apart the same way. Sympy belongs to the
worker, and an import that quietly puts it in the app process instead is invisible
from the inside - everything still works, a third of a second and some thirty
megabytes later. So that one is tested from a fresh interpreter too, by asking what
ended up in `sys.modules`.
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

#: What the app process imports, from the entry point down. None of it may reach
#: sympy: the mathematics lives in the worker, which imports it in `worker.serve`,
#: in the process where the computing happens. `rederive.engine` is on the list
#: because a package `__init__` is run by every import of anything below it, so an
#: engine that greeted its own submodules with sympy would defeat all the rest.
CLIENT_SIDE = [
    "rederive.engine",
    "rederive.engine.boundary",
    "rederive.engine.client",
    "rederive.model.session",
    "rederive.ui.app",
    "rederive.__main__",
]


@pytest.mark.parametrize("package", PACKAGES)
def test_a_package_imports_before_anything_else(package: str) -> None:
    subprocess.run([sys.executable, "-c", f"import {package}"], check=True)


@pytest.mark.parametrize("module", CLIENT_SIDE)
def test_the_app_side_imports_no_sympy(module: str) -> None:
    """No client-half import may pull sympy in, however far down the chain it is.

    The failure this catches is never a broken program, which is what makes it worth
    a test: a client-half module that imports a worker-half one for a constant or a
    type still works perfectly, and costs the app process a computer algebra system
    it will never call.
    """
    program = f"import sys, {module}; print('sympy' in sys.modules)"
    imported = subprocess.run(
        [sys.executable, "-c", program], check=True, capture_output=True, text=True
    )
    assert imported.stdout.strip() == "False", f"{module} imports sympy"


#: What plotting costs, and what the app process must not pay. The windows live
#: in a child process for the same reason the mathematics does, so the app side
#: may name the plot vocabulary and the availability check and nothing under
#: them: a Qt toolkit imported here is thirty megabytes for a picture drawn
#: elsewhere, and one that a session which never plots pays for regardless.
HEAVY = ("numpy", "PySide6", "pyqtgraph", "OpenGL")


@pytest.mark.parametrize("module", CLIENT_SIDE)
def test_the_app_side_imports_no_toolkit(module: str) -> None:
    program = (
        f"import sys, {module}; "
        f"print([name for name in {HEAVY!r} if name in sys.modules])"
    )
    imported = subprocess.run(
        [sys.executable, "-c", program], check=True, capture_output=True, text=True
    )
    assert imported.stdout.strip() == "[]", f"{module} imports {imported.stdout}"


def test_the_entry_point_imports_no_screen() -> None:
    """`rederive.__main__`'s body may name the command line and nothing more.

    The mirror of the rule above, and it costs the same to break. Spawning re-runs
    the parent's `__main__` in the child, and the `rederive` script imports this
    module at its top, so whatever the body names is loaded again in the worker -
    which paints nothing. Textual there is thirty-odd megabytes and a tenth of a
    second the computing half spends on a screen it does not have.
    """
    program = "import sys, rederive.__main__; print('textual' in sys.modules)"
    imported = subprocess.run(
        [sys.executable, "-c", program], check=True, capture_output=True, text=True
    )
    assert imported.stdout.strip() == "False", "rederive.__main__ imports textual"


def test_the_computing_half_offers_everything_the_client_half_does() -> None:
    """One import for whoever may compute, which is what makes the pair a pair.

    A caller entitled to the mathematics is entitled to the vocabulary as well, and
    having to import both halves to get a `Context` alongside a `simplify` would
    teach exactly the habit the split is for.
    """
    from rederive.engine import client, computing

    assert set(client.__all__) <= set(computing.__all__)

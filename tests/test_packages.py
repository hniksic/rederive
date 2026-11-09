"""Import-structure rules: packages import independently, and the app side has no math.

Layers import in one direction (`model.expr` under `syntax` under the engine, display
and session). A package `__init__` that imports back up the stack works or fails
depending on import order, so each package is imported in a fresh interpreter.

Sympy belongs to the worker. Importing it in the app process breaks nothing visibly but
costs a third of a second and some thirty megabytes, so this is checked in a fresh
interpreter by inspecting `sys.modules`.

The plot toolkit, psutil and pyperclip are opt-in extras, so the client half must
import without them. That also lets the package run where those wheels don't exist at
all, such as a browser.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGES = [
    "rederive.model",
    "rederive.model.session",
    "rederive.syntax",
    "rederive.display",
    "rederive.engine",
]

#: Modules the app process imports, which must not pull in sympy; the worker imports it
#: in `worker.serve`. `rederive.engine` is listed because its `__init__` runs on every
#: import of a submodule. All of `rederive.web` is app side, since the browser runs the
#: UI in the page's Pyodide and the math in a worker's, except `rederive.web.worker`,
#: the worker's entry point. The plot session, backend protocol, model, resampling,
#: view arithmetic and browser backend run on the browser's main thread, so they must
#: not import numpy either; `rederive.plot.web.sampler` is the exception, running in
#: the worker. The plot controls, actions and dialog forms are used by the page's
#: painting thread to show menus, handle keys and open overlays.
CLIENT_SIDE = [
    "rederive.engine",
    "rederive.engine.boundary",
    "rederive.engine.client",
    "rederive.engine.policy",
    "rederive.memory",
    "rederive.model.session",
    "rederive.platform",
    "rederive.platform.web",
    "rederive.plot.actions",
    "rederive.plot.backend",
    "rederive.plot.controls",
    "rederive.plot.forms",
    "rederive.plot.model",
    "rederive.plot.proxy",
    "rederive.plot.resample",
    "rederive.plot.session",
    "rederive.plot.view",
    "rederive.plot.web",
    "rederive.plot.web.backend",
    "rederive.plot.web.protocol",
    "rederive.ui.app",
    "rederive.web",
    "rederive.web.boot",
    "rederive.web.driver",
    "rederive.web.engine",
    "rederive.web.files",
    "rederive.web.plots",
    "rederive.__main__",
]


@pytest.mark.parametrize("package", PACKAGES)
def test_a_package_imports_before_anything_else(package: str) -> None:
    subprocess.run([sys.executable, "-c", f"import {package}"], check=True)


#: Heavy plotting modules the app process must not import. Plot windows run in a child
#: process, so the app side may import only the plot vocabulary and availability check:
#: a Qt toolkit here would cost thirty megabytes even in sessions that never plot.
HEAVY = ("numpy", "PySide6", "pyqtgraph", "OpenGL")


@pytest.mark.parametrize("module", CLIENT_SIDE)
def test_the_app_side_imports_no_sympy_and_no_toolkit(module: str) -> None:
    """No client-half module may transitively import sympy or the plot toolkit.

    Such an import, say of a worker-half module for a constant or type, breaks nothing
    visibly, which is why it needs a test. Both checks share one interpreter, since
    starting it is the expensive part.
    """
    program = (
        f"import sys, {module}; "
        f"print([name for name in ('sympy', *{HEAVY!r}) if name in sys.modules])"
    )
    imported = subprocess.run(
        [sys.executable, "-c", program], check=True, capture_output=True, text=True
    )
    assert imported.stdout.strip() == "[]", f"{module} imports {imported.stdout}"


#: Packages from the optional `plot` and `desktop` extras. No client-half module may
#: import them at module level, even transitively; code that needs the clipboard,
#: memory usage or toolkit must import it when called, so a slim install still starts.
OPTIONAL = ("PySide6", "pyqtgraph", "OpenGL", "psutil", "pyperclip")

#: Simulates an install without the extras: a meta-path finder makes importing them fail
#: as if they were missing, and the loop checks that the blocking actually works.
ABSENT = """
import sys


class Absent:
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in {names}:
            raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)
        return None


sys.meta_path.insert(0, Absent())

for name in {names}:
    try:
        __import__(name)
    except ModuleNotFoundError:
        continue
    raise AssertionError(f"{{name}} was imported; this machine is not the slim one")
"""


@pytest.mark.parametrize("module", CLIENT_SIDE)
def test_the_app_side_imports_with_no_extras_installed(module: str) -> None:
    """A bare `pip install rederive` must work.

    The extras are opt-in, so this is the ordinary install, and breaking it is invisible
    on a development machine: a top-level import that belongs inside a function works
    here but fails on every slim install.
    """
    program = ABSENT.format(names=set(OPTIONAL)) + f"\nimport {module}\n"
    subprocess.run([sys.executable, "-c", program], check=True)


#: Top-level module names of the plot window toolkit. `OpenGL` is PyOpenGL, used by
#: `pyqtgraph.opengl`.
TOOLKIT = ("PySide6", "pyqtgraph", "OpenGL")

#: The only package allowed to import the toolkit. The rest of `plot/` (session, model,
#: sampling, surface geometry, view arithmetic) is shared with the browser backend,
#: which can't use the toolkit.
TOOLKIT_PACKAGE = "rederive/plot/qt"


def test_only_the_qt_backend_names_a_toolkit() -> None:
    """The plot layer's central rule, checked by reading the source.

    Importing wouldn't catch imports inside functions: the window classes are
    deliberately imported where a window is made, and a module outside `plot/qt` doing
    the same would pass an import test yet still need the toolkit in the browser.
    """
    root = Path(__file__).resolve().parent.parent / "src"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if TOOLKIT_PACKAGE in path.as_posix():
            continue
        # The sources contain display glyphs, which Windows would otherwise decode
        # with its current code page.
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                named = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                named = [node.module or ""]
            else:
                continue
            for name in named:
                if name.partition(".")[0] in TOOLKIT:
                    offenders.append(f"{path.relative_to(root)}:{node.lineno} {name}")
    assert offenders == []


def test_the_entry_point_imports_no_screen() -> None:
    """`rederive.__main__` may import only what the command line needs.

    Spawning re-runs the parent's `__main__` in the child, and the `rederive` script
    imports this module at the top, so its imports are loaded again in the worker.
    Textual there would cost thirty-odd megabytes and a tenth of a second for a screen
    the worker doesn't have.
    """
    program = "import sys, rederive.__main__; print('textual' in sys.modules)"
    imported = subprocess.run(
        [sys.executable, "-c", program], check=True, capture_output=True, text=True
    )
    assert imported.stdout.strip() == "False", "rederive.__main__ imports textual"


def test_the_computing_half_offers_everything_the_client_half_does() -> None:
    """`computing` re-exports everything in `client`, so callers need one import.

    Needing both halves to get a `Context` alongside `simplify` would encourage mixing
    imports, the habit the split exists to prevent.
    """
    from rederive.engine import client, computing

    assert set(client.__all__) <= set(computing.__all__)

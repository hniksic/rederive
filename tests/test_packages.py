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

The third rule is about what the install has. The plot toolkit, psutil and pyperclip
are opt-in extras, so `pip install rederive` is a machine where none of them exist,
and the client half has to import there - a program that refuses to start because
nobody asked for a plot window is not a slim install, it is a broken one. That is
also what makes the package installable somewhere the wheels do not exist at all,
which is what a browser is.
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

#: What the app process imports, from the entry point down. None of it may reach
#: sympy: the mathematics lives in the worker, which imports it in `worker.serve`,
#: in the process where the computing happens. `rederive.engine` is on the list
#: because a package `__init__` is run by every import of anything below it, so an
#: engine that greeted its own submodules with sympy would defeat all the rest.
#: `rederive.web` is on it in full, and every module of it is the app side by
#: definition: the browser runs the UI in the page's own Pyodide and the
#: mathematics in a worker's, so a sympy import there would cost a second
#: interpreter's worth of it in the instance that draws the screen. The one
#: exception is `rederive.web.worker`, which is that other interpreter's entry
#: and is meant to import the whole engine. The plot session, its backend
#: protocol, its model, its re-sampling policy, its view arithmetic and the
#: whole of the browser backend are on the list for the same reason: in the
#: browser the plot list and the routing live on the main thread while the
#: arrays go from the worker to what draws them, so a numpy import in any of
#: them would put a numerical library in the instance that paints the screen.
#: `rederive.plot.web.sampler` is the exception and the mirror of
#: `rederive.web.worker`: it is the half that computes, and it runs where sympy
#: and numpy already are.
CLIENT_SIDE = [
    "rederive.engine",
    "rederive.engine.boundary",
    "rederive.engine.client",
    "rederive.engine.policy",
    "rederive.memory",
    "rederive.model.session",
    "rederive.platform",
    "rederive.platform.web",
    "rederive.plot.backend",
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


#: The wheels an install need not have: the `plot` extra and the `desktop` one.
#: No client-half module may import one of them at its top, however far down the
#: chain it is - what reaches for a clipboard, a resident set or a toolkit has to
#: do it where it is asked for, so that an install without them still starts.
OPTIONAL = ("PySide6", "pyqtgraph", "OpenGL", "psutil", "pyperclip")

#: A machine where none of them exist, made out of one that has them all. A
#: finder in front of every other refuses those names the way an import of a
#: wheel that was never fetched is refused, and the check under it is what stops
#: this from passing because the refusing quietly stopped working.
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
    """A bare `pip install rederive` is a program, and this is what says so.

    The extras are opt-in, so this is the ordinary install rather than a corner of
    one, and every way of breaking it is invisible on a development machine: an
    import written at the top of a module instead of inside the one function that
    needs it works perfectly here and fails on every install that took the package
    at its word.
    """
    program = ABSENT.format(names=set(OPTIONAL)) + f"\nimport {module}\n"
    subprocess.run([sys.executable, "-c", program], check=True)


#: The toolkit a plot window is made of, by the top-level name each is imported
#: under. `OpenGL` is PyOpenGL, which `pyqtgraph.opengl` rides.
TOOLKIT = ("PySide6", "pyqtgraph", "OpenGL")

#: The one package allowed to name it. Everything else in `plot/` - the plot
#: session, the model, the sampling policy, the surface geometry, the view
#: arithmetic - is what a second backend shares, and a toolkit import anywhere
#: in it is a line the browser cannot cross.
TOOLKIT_PACKAGE = "rederive/plot/qt"


def test_only_the_qt_backend_names_a_toolkit() -> None:
    """The rule the whole plot layer is arranged around, read off the source.

    Read rather than imported, because the imports that would break this are the
    ones written inside a function: the window classes are imported where a
    window is made, deliberately, and a module that did the same thing outside
    `plot/qt` would pass an import test and still be a toolkit in the browser.
    """
    root = Path(__file__).resolve().parent.parent / "src"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if TOOLKIT_PACKAGE in path.as_posix():
            continue
        # Named, because the sources hold the display's glyphs and Windows would
        # otherwise read them through whatever code page it is set to.
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

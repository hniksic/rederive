"""The Qt backend: the only place in the program that names a widget toolkit.

Everything a plot window is - the canvas, the legend, the status line, the
menus, the dialogs, the GL view - is Qt, and all of it is here, along with the
process it is all served in. What is above this package is the plot session's
bookkeeping, the sampling policy and the mathematics of the picture, none of
which knows what draws it, which is what lets a second backend draw the same
plots somewhere Qt cannot go.

Nothing outside this package may import `pyqtgraph`, `PySide6` or `OpenGL`, and
`tests/test_packages.py` is what says so.
"""

from __future__ import annotations

__all__ = ["toolkit"]


def toolkit() -> str:
    """The Qt and pyqtgraph a plot is drawn with, or the reason there is none.

    What `--version` reports about plotting, answered here because this is the
    package that knows the names. The import is the answer either way: a machine
    whose Qt will not load is one where the Plot command cannot work, and saying
    so costs nothing, where refusing to print a version at all would cost a user
    the rest of the lines.
    """
    try:
        import pyqtgraph
        from pyqtgraph.Qt import QtCore
    except ImportError as missing:
        return f"Qt unusable ({missing})"

    return f"Qt {QtCore.qVersion()}\npyqtgraph {pyqtgraph.__version__}"

"""The Qt backend, the only code in the program that uses a widget toolkit.

The plot windows (canvas, legend, status line, menus, dialogs, GL view) and the process
that hosts them live here. The code above this package - session bookkeeping, sampling
and the math of the picture - is toolkit-independent, so another backend could draw the
same plots.

Nothing outside this package may import `pyqtgraph`, `PySide6` or `OpenGL`; this is
enforced by `tests/test_packages.py`.
"""

from __future__ import annotations

__all__ = ["toolkit"]


def toolkit() -> str:
    """The Qt and pyqtgraph versions, or why they can't be loaded.

    Used by `--version`. A failed import is reported rather than raised, so that the
    rest of the version output still prints.
    """
    try:
        import pyqtgraph
        from pyqtgraph.Qt import QtCore
    except ImportError as missing:
        return f"Qt unusable ({missing})"

    return f"Qt {QtCore.qVersion()}\npyqtgraph {pyqtgraph.__version__}"

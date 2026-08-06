"""The Qt backend: the only place in the program that names a widget toolkit.

Everything a plot window is - the canvas, the legend, the status line, the
menus, the dialogs, the GL view - is Qt, and all of it is here. What is above
this package is the plot session's bookkeeping, the sampling policy and the
mathematics of the picture, none of which knows what draws it, which is what
lets a second backend draw the same plots somewhere Qt cannot go.

Nothing outside this package may import `pyqtgraph`, `PySide6` or `OpenGL`, and
`tests/test_packages.py` is what says so.
"""

from __future__ import annotations

__all__: list[str] = []

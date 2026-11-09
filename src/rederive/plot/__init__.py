"""Check whether this machine can draw a plot, without loading Qt.

The Plot command is always on the menu; this module tells the app whether it can
actually draw. Importing it must stay cheap: no numpy, sympy, pyqtgraph or Qt, since the
app process is sympy-free by design.

Only the display is checked, not the installed wheels. A missing wheel is a broken
install, reported by the host failing to start (with the same `Plot: ` prefix). On Linux
a display means `DISPLAY` or `WAYLAND_DISPLAY` is set; macOS and Windows always have one.
"""

from __future__ import annotations

import os
import sys

__all__ = ["PREFIX", "UNAVAILABLE", "available"]

#: Prefix of every plot message on the message line, so that errors from deep in the
#: host read as coming from the command.
PREFIX = "Plot: "

#: Message shown when there is no display to draw on.
UNAVAILABLE = "needs a graphical display"

#: Environment variables that indicate a Linux display server.
DISPLAY_VARIABLES = ("DISPLAY", "WAYLAND_DISPLAY")


def available() -> bool:
    """Whether a plot window could be opened on this machine."""
    if not sys.platform.startswith("linux"):
        return True
    return any(os.environ.get(name) for name in DISPLAY_VARIABLES)

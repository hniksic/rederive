"""Whether this machine can draw a plot at all, answered without loading Qt.

The Plot command is on the menu of every build, because a command that
disappears is a command nobody learns; what varies is whether it draws
anything. This module is what the app asks before it tries, and it is
deliberately the cheapest question in the package: importing it must not pull
in numpy, sympy, pyqtgraph or Qt, since the app process is sympy-free by
design and pays for Textual alone.

The test is about a display and not about the wheels. A missing wheel is a
broken install rather than a headless machine, and the two want different
words, so the wheel says so by failing to start the host - which the proxy
reports with the same `Plot: ` prefix. Here the only question is whether there
is a screen to put a window on: on Linux that is `DISPLAY` or `WAYLAND_DISPLAY`
being set, and on macOS and Windows there always is one.
"""

from __future__ import annotations

import os
import sys

__all__ = ["PREFIX", "UNAVAILABLE", "available"]

#: What every plot message on the message line starts with, so that a refusal
#: reads as the command's own however deep in the host it came from.
PREFIX = "Plot: "

#: What the command says where there is no screen to draw on.
UNAVAILABLE = "needs a graphical display"

#: The environment variables a Linux desktop session sets. Either one is a
#: display server this process could open a window on.
DISPLAY_VARIABLES = ("DISPLAY", "WAYLAND_DISPLAY")


def available() -> bool:
    """Whether a plot window could be opened on this machine."""
    if not sys.platform.startswith("linux"):
        return True
    return any(os.environ.get(name) for name in DISPLAY_VARIABLES)

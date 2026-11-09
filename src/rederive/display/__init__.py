"""Rendering expression trees as two-dimensional terminal text.

Code outside this package imports only the names re-exported here. The package doesn't
import the UI or the settings; callers pass options as plain data. It only renders: it
neither parses nor computes, so `2/4` typesets as 2 over 4, never as `1/2`.

There is no width limit or wrapping, because reflowing would break the guarantee the UI
relies on: each subexpression occupies one rectangle, which `Layout.at` returns for any
path into the selection tree. The pane clips and scrolls instead.

The selection tree lists the subexpressions the cursor can reach, in order. It follows
how the expression is drawn rather than how it was parsed: `a + b + c` offers three
terms although the parse nests two, and `SIN(x + 1)` offers its argument but not its
name.
"""

from __future__ import annotations

from rederive.display.boxes import Layout, Region
from rederive.display.layout import render
from rederive.display.options import ASTERISK, DOT, IMPLICIT, DisplayOptions

__all__ = [
    "ASTERISK",
    "DOT",
    "IMPLICIT",
    "DisplayOptions",
    "Layout",
    "Region",
    "render",
]

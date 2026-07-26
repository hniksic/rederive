"""Expression trees in, built-up terminal renders out.

Nothing outside this package may import from inside it except through the
names re-exported here, and nothing inside it imports the UI or the settings:
it takes its options as plain data the caller builds.

It renders and nothing else. It does not parse, and it computes no
mathematics: `2/4` typesets as a bar with 2 over 4, never as `1/2`.

A render is as wide as it is. There is no width parameter and no wrapping,
because a reflow would break the one guarantee the UI needs: a subexpression
occupies one rectangle, which `Layout.at` gives back for any route into the
selection tree. The pane clips and scrolls instead.

That selection tree is the other half of a render. It says which
subexpressions the cursor can reach and in what order, which is a fact about
how the expression was drawn rather than about how it was parsed: `a + b + c`
offers three terms though the parse nests two, and `SIN(x + 1)` offers its
argument and never its name.
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

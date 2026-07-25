"""The one place colors are chosen.

Everything on screen takes its color from `PALETTE` below, either through the
CSS variables the app exports (`$rd-...`) or through the Rich style strings in
`STYLES`. Adding the monochrome and green-phosphor presets of R-COL1/R-COL2
later means adding another palette here, not touching widget code.
"""

from __future__ import annotations

# Approximates the 16-color EGA palette as seen in the archived screenshots.
PALETTE = {
    "background": "#000000",
    "expression": "#cccccc",
    "menu": "#ffff55",
    "menu-highlight-bg": "#ffaa00",
    "menu-highlight-fg": "#000000",
    "message": "#ff5555",
    "annotation": "#ff5555",
    "status": "#55ff55",
    "rule": "#ff55ff",
    "selection-bg": "#cccccc",
    "selection-fg": "#000000",
}

# Rich style strings for the runs of text that widgets build themselves.
STYLES = {
    "expression": PALETTE["expression"],
    "selection": f"{PALETTE['selection-fg']} on {PALETTE['selection-bg']}",
    "menu": PALETTE["menu"],
    "menu-highlight": f"{PALETTE['menu-highlight-fg']} on {PALETTE['menu-highlight-bg']}",
    "annotation": PALETTE["annotation"],
    "status": PALETTE["status"],
}


def css_variables() -> dict[str, str]:
    """Palette entries as Textual CSS variables, e.g. `$rd-menu`."""
    return {f"rd-{name}": value for name, value in PALETTE.items()}

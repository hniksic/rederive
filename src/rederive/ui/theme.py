"""The one place colors are chosen.

The slot names below are Derive's own. `Options Color Work` sets a foreground
and a background for the work area; `Options Color Menu` sets Frame, Option,
Prompt, Status, Background, and Border. Everything on screen draws from those
eight slots, so a theme preset is just a table of color numbers - which is
literally what `SCHEME` is.

Rederive imitates the original in everything but color. Color is the exception
because the original ships monochrome - Work 15 on 0, Menu 7, 7, 15, 7 on
0, 0 - so a colorless screenshot is not evidence that this palette is wrong
and must not be "corrected" to match. Those numbers are the monochrome preset
of R-COL3, for whoever adds it.

The colors below are not invented either. They are the original's factory
settings from the last version whose defaults were colorful, as set in its
`Options Color` screens: Work 15 on 0, and Menu Frame 4, Option 14, Prompt 4,
Status 2 on Background 0, Border 0.
"""

from __future__ import annotations

#: The 16 EGA text colors, indexed by the numbers Options Color accepts.
EGA = (
    "#000000",  # 0 black
    "#0000aa",  # 1 blue
    "#00aa00",  # 2 green
    "#00aaaa",  # 3 cyan
    "#aa0000",  # 4 red
    "#aa00aa",  # 5 magenta
    "#aa5500",  # 6 brown
    "#aaaaaa",  # 7 light gray
    "#555555",  # 8 dark gray
    "#5555ff",  # 9 light blue
    "#55ff55",  # 10 light green
    "#55ffff",  # 11 light cyan
    "#ff5555",  # 12 light red
    "#ff55ff",  # 13 light magenta
    "#ffff55",  # 14 yellow
    "#ffffff",  # 15 white
)

#: The original's factory scheme, slot by slot.
SCHEME = {
    # Options Color Work
    "work-foreground": 15,  # expressions in the work area
    "work-background": 0,
    # Options Color Menu
    "frame": 4,  # the rule above the command bands
    "option": 14,  # the COMMAND: label and the menu words
    "prompt": 4,  # the message line
    "status": 2,  # the whole status line
    "menu-background": 0,
    "border": 0,  # pane borders, unused until there are panes
}

PALETTE = {slot: EGA[number] for slot, number in SCHEME.items()}


def _dimmed(slot: str) -> str:
    """The color of `slot` as DOS text mode can render it in a background.

    Only the eight low-intensity colors are available there, so an inverse
    video block loses the intensity bit: yellow 14 comes out brown 6, white 15
    comes out light gray 7. This is why the highlighted menu option reads as an
    orange block rather than a yellow one.
    """
    return EGA[SCHEME[slot] & 7]


def _inverse(slot: str, over: str) -> str:
    """A solid inverse-video block of `slot`, lettered in `over` (R-COL4)."""
    return f"{PALETTE[over]} on {_dimmed(slot)}"


# Rich style strings for the runs of text that widgets build themselves.
STYLES = {
    "work": PALETTE["work-foreground"],
    "selection": _inverse("work-foreground", over="work-background"),
    "frame": PALETTE["frame"],
    "option": PALETTE["option"],
    "option-highlight": _inverse("option", over="menu-background"),
    "prompt": PALETTE["prompt"],
    "status": PALETTE["status"],
}


def css_variables() -> dict[str, str]:
    """Palette slots as Textual CSS variables, e.g. `$rd-option`."""
    variables = {f"rd-{slot}": value for slot, value in PALETTE.items()}
    variables["rd-selection-background"] = _dimmed("work-foreground")
    return variables

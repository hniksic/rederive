"""The one place colors are chosen.

The slot names are Derive's. `Options Color Work` sets the work area's foreground and
background; `Options Color Menu` sets Frame, Option, Prompt, Status, Background, and
Border. Everything on screen uses these eight slots, so a color scheme is just those
eight settings.

The scheme lives in `Settings`, where `Options Color` can change it, and `Palette`
resolves it to the values Rich and the stylesheet need. The default scheme and its
origin are documented next to the color dialogs in `rederive.model.settings`.
"""

from __future__ import annotations

from rederive.model.settings import COLORS, Settings

#: The 16 EGA text colors, numbered as in Derive's Options Color commands (the order of
#: `COLORS`).
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

NUMBER: dict[str, int] = {color: number for number, color in enumerate(COLORS)}

#: Each color slot on screen, and the setting that fills it.
SLOTS: dict[str, str] = {
    "work-foreground": "WorkColor",  # expressions in the work area
    "work-background": "WorkBackgroundColor",
    "frame": "FrameColor",  # window borders, including the rule between panes
    "option": "OptionColor",  # the COMMAND: label and the menu words
    "prompt": "PromptColor",  # the message line
    "status": "StatusColor",  # the whole status line
    "menu-background": "MenuBackgroundColor",
    "border": "BorderColor",  # the background behind those borders, unused
}

#: The settings a repaint has to watch for.
COLOR_SETTINGS = frozenset(SLOTS.values())


class Palette:
    """The eight color slots, in the forms the widgets and stylesheet use.

    Slots are read on every call rather than cached, so `Options Color` changes need no
    invalidation.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def color(self, slot: str) -> str:
        """The hex color filling `slot`."""
        return EGA[self.number(slot)]

    def number(self, slot: str) -> int:
        value = self._settings[SLOTS[slot]]
        assert isinstance(value, str)
        return NUMBER[value]

    def dimmed(self, slot: str) -> str:
        """The color of `slot` as DOS text mode can render it in a background.

        Only the eight low-intensity colors are available as backgrounds, so inverse
        video drops the intensity bit: yellow 14 becomes brown 6, white 15 becomes light
        gray 7. That's why the highlighted menu option is orange rather than yellow.
        """
        return EGA[self.number(slot) & 7]

    def inverse(self, slot: str, over: str) -> str:
        """A solid inverse-video block of `slot`, lettered in `over` (R-COL4)."""
        return f"{self.color(over)} on {self.dimmed(slot)}"

    @property
    def styles(self) -> dict[str, str]:
        """Rich style strings for the runs of text that widgets build up."""
        return {
            "work": self.color("work-foreground"),
            "selection": self.inverse("work-foreground", over="work-background"),
            "frame": self.color("frame"),
            "option": self.color("option"),
            "option-highlight": self.inverse("option", over="menu-background"),
            # The word under the mouse pointer. An attribute rather than a color, so it
            # doesn't compete with the highlight or need a ninth slot, and it combines
            # with the highlight when the pointer rests on the highlighted word.
            "option-pointed": "underline",
            "prompt": self.color("prompt"),
            "status": self.color("status"),
        }

    def css_variables(self) -> dict[str, str]:
        """Palette slots as Textual CSS variables, e.g. `$rd-option`."""
        variables = {f"rd-{slot}": self.color(slot) for slot in SLOTS}
        variables["rd-selection-background"] = self.dimmed("work-foreground")
        return variables

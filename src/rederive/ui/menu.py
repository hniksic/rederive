"""The menus of the Algebra pane: the words, their mnemonic letters, and what
each one opens.

`Unremove` is listed unconditionally. R-MENU1 says it appears only while there
is something to restore, but the original shows it in the menu of a session
where nothing has ever been removed, so the requirement is wrong about this.
"""

from __future__ import annotations

from dataclasses import dataclass

from rederive.model import settings

MENU_TITLE = "COMMAND:"

#: How many options go on a menu's first line; the rest go on the second.
FIRST_LINE_OPTIONS = 10

ENTER_OPTION = "Enter option"


def mnemonic(word: str) -> str:
    """The lower-cased capital letter that invokes `word`, e.g. `l` for soLve."""
    for character in word:
        if character.isupper():
            return character.lower()
    return word[:1].lower()


@dataclass(frozen=True)
class Menu:
    """A band of words, one of which is highlighted."""

    title: str
    words: tuple[str, ...]
    message: str = ENTER_OPTION
    first_line: int = FIRST_LINE_OPTIONS

    @property
    def mnemonics(self) -> dict[str, int]:
        return {mnemonic(word): index for index, word in enumerate(self.words)}


@dataclass
class MenuCursor:
    """A menu on screen, and which of its words is highlighted."""

    menu: Menu
    index: int = 0

    @property
    def word(self) -> str:
        return self.menu.words[self.index]

    def move(self, step: int) -> None:
        self.index = (self.index + step) % len(self.menu.words)


ALGEBRA = Menu(
    MENU_TITLE,
    (
        "Author",
        "Build",
        "Calculus",
        "Declare",
        "Expand",
        "Factor",
        "Help",
        "Jump",
        "soLve",
        "Manage",
        "Options",
        "Plot",
        "Quit",
        "Remove",
        "Simplify",
        "Transfer",
        "Unremove",
        "moVe",
        "Window",
        "approX",
    ),
)

# Kept for the tests and callers that only want the Algebra words.
ALGEBRA_MENU = ALGEBRA.words
MNEMONICS = ALGEBRA.mnemonics

# `Display` and `Execute` are the two of Derive's nine Options commands that
# are not here: one chose between text and graphics modes on adapters that no
# longer exist, the other shelled out to DOS.
OPTIONS = Menu(
    "OPTIONS:",
    ("Color", "Input", "Mute", "Notation", "Output", "Precision", "Radix"),
)

COLOR = Menu("OPTIONS COLOR:", ("Menu", "Work"))

#: How each color is spelled on the color menu. Derive asked for a number, so
#: these mnemonics are the remake's own; every one of the sixteen needs a letter
#: of its own, which is what pushes `Blue`, `Brown`, `Gray` and `Aqua` off their
#: initials.
COLOR_WORDS: dict[str, str] = {
    "Black": "Black",
    "Blue": "blUe",
    "Green": "Green",
    "Cyan": "Cyan",
    "Red": "Red",
    "Magenta": "Magenta",
    "Brown": "brOwn",
    "Gray": "grAy",
    "Darkgray": "Darkgray",
    "Skyblue": "Skyblue",
    "Lime": "Lime",
    "Aqua": "aQua",
    "Pink": "Pink",
    "Violet": "Violet",
    "Yellow": "Yellow",
    "White": "White",
}

#: The color menu is titled by neither of the two commands that reach it: the
#: full path, `OPTIONS COLOR MENU BACKGROUND:`, would leave no room for the
#: colors. The message line says which slot is being colored instead.
COLORS = Menu(
    "COLOR:",
    tuple(COLOR_WORDS.values()),
    first_line=len(settings.COLORS) // 2,
)

#: What each word of the Options menu opens.
OPTIONS_TARGETS: dict[str, Menu | settings.Dialog] = {
    "Color": COLOR,
    "Input": settings.INPUT,
    "Mute": settings.MUTE,
    "Notation": settings.NOTATION,
    "Output": settings.OUTPUT,
    "Precision": settings.PRECISION,
    "Radix": settings.RADIX,
}

COLOR_TARGETS: dict[str, settings.Dialog] = {
    "Menu": settings.COLOR_MENU,
    "Work": settings.COLOR_WORK,
}


def color_at(index: int) -> str:
    """The color the color menu's `index`-th word stands for."""
    return settings.COLORS[index]


def choice_word(value: str) -> str:
    """How a field's value is spelled where its mnemonic letter matters."""
    return COLOR_WORDS.get(value, value)


def pick(choices: tuple[str, ...], letter: str) -> str | None:
    """The choice `letter` selects, or None if it selects none of them.

    First match wins, as in the original: on the Radix field `o` is Octal, and
    `Other` can only be reached by stepping to it.
    """
    for choice in choices:
        if mnemonic(choice_word(choice)) == letter:
            return choice
    return None

"""The Algebra pane's command menu: the words and their mnemonic letters.

`Unremove` is listed unconditionally. R-MENU1 says it appears only while there
is something to restore, but the original shows it in the menu of a session
where nothing has ever been removed, so the requirement is wrong about this.
"""

from __future__ import annotations

ALGEBRA_MENU: tuple[str, ...] = (
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
)

MENU_TITLE = "COMMAND:"

#: How many options go on the menu's first line; the rest go on the second.
FIRST_LINE_OPTIONS = 10


def mnemonic(word: str) -> str:
    """The lower-cased capital letter that invokes `word`, e.g. `l` for soLve."""
    for character in word:
        if character.isupper():
            return character.lower()
    return word[:1].lower()


MNEMONICS: dict[str, int] = {
    mnemonic(word): index for index, word in enumerate(ALGEBRA_MENU)
}

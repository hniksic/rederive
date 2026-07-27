"""The menus of the Algebra pane: the words, their mnemonic letters, and what
each one opens.

`Unremove` is listed unconditionally. R-MENU1 says it appears only while there
is something to restore, but the original shows it in the menu of a session
where nothing has ever been removed, so the requirement is wrong about this.
"""

from __future__ import annotations

from dataclasses import dataclass

from rederive.model import settings
from rederive.model.session import Bounds

MENU_TITLE = "COMMAND:"

#: How many options go on a menu's first line; the rest go on the second.
FIRST_LINE_OPTIONS = 10

ENTER_OPTION = "Enter option"

#: What the message line asks for on a field that names an expression.
ENTER_LABEL = "Enter label number"

#: What it asks for on the one field that takes a word instead as well.
ENTER_LABEL_OR_END = 'Enter label number or type "end"'

#: The word that sends the unremoved expressions past the last entry.
END = "end"

#: What the message line asks for on each field of the Declare dialogs.
ENTER_ROWS = "Enter number of rows"
ENTER_COLUMNS = "Enter number of columns"
ENTER_ELEMENTS = "Enter number of elements"
ENTER_LEFT_BOUND = "Enter left bound"
ENTER_RIGHT_BOUND = "Enter right bound"

#: The two symbols a bound is written with: `<` for a bound the variable can
#: never equal, `≤` for one it can.
OPEN = "<"
CLOSED = "≤"
STRICTNESS = (OPEN, CLOSED)

#: Columns a bound takes, so that the fields beside it do not shift as it is
#: typed. Wide enough for the widest bound the original shows, `-∞`, and then
#: some, as the original's own field is.
_BOUND_WIDTH = 6


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
    asks: str | None = None
    """What the message line says, when it is not the menu's own prompt.

    The two Declare Variable menus name the variable they are asking about, so
    their prompt is written when they are put up rather than when they are
    declared.
    """

    @property
    def word(self) -> str:
        return self.menu.words[self.index]

    @property
    def message(self) -> str:
        return self.menu.message if self.asks is None else self.asks

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

#: How much factoring Factor does. `raDical` carries its capital because `R`
#: belongs to Rational, and the amounts are in increasing order: each does
#: everything the one before it does, and costs more time doing it.
AMOUNT = Menu(
    "FACTOR: Amount:",
    ("Trivial", "Squarefree", "Rational", "raDical", "Complex"),
    message="Select amount of factoring",
)

#: How much factoring Expand does to the denominator it makes partial
#: fractions over. The same amounts less `Complex`, which the original leaves
#: off this menu: a complex factor would put `#i` in an answer nobody asked to
#: leave the reals for.
EXPAND_AMOUNT = Menu(
    "EXPAND: Amount:",
    tuple(word for word in AMOUNT.words if word != "Complex"),
    message=AMOUNT.message,
)

DECLARE = Menu("DECLARE:", ("Function", "Variable", "Matrix", "vectoR"))

#: What Declare Variable asks once it has a name, and then once it has a
#: domain. Both name the variable, which is what the `{name}` is for.
SELECT_DOMAIN = "Select value or domain of {name}"
SELECT_INTERVAL = "Select interval of {name}"

#: The two menus Declare Variable puts up. They carry the same title, which is
#: the original's: the message line is where it says which question is up.
DECLARE_VARIABLE = Menu(
    "DECLARE VARIABLE:",
    ("Value", "Integer", "Real", "Complex", "Nonscalar"),
    message=SELECT_DOMAIN,
)

#: The intervals, which only Integer and Real are asked about. `nonpoSitive`
#: and `nonneGative` carry their capitals late because `N` belongs to Negative.
DECLARE_INTERVAL = Menu(
    "DECLARE VARIABLE:",
    ("All", "Positive", "Negative", "nonpoSitive", "nonneGative", "Interval"),
    message=SELECT_INTERVAL,
)

#: The domains that have an interval to ask about. Complex and Nonscalar do
#: not, so choosing one of those is the last answer the command needs.
BOUNDED_DOMAINS = ("Integer", "Real")

# Four of these eight commands set something and are dialogs; the other four
# act on the session and ask for what they need on a line of their own.
# `Trigonometry` goes on a line of its own because all eight will not fit
# across eighty columns.
MANAGE = Menu(
    "MANAGE:",
    (
        "Annotate",
        "Branch",
        "Exponential",
        "Logarithm",
        "Ordering",
        "Renumber",
        "Substitute",
        "Trigonometry",
    ),
    first_line=7,
)

# `Display` and `Execute` are the two of Derive's nine Options commands that
# are not here: one chose between text and graphics modes on adapters that no
# longer exist, the other shelled out to DOS.
OPTIONS = Menu(
    "OPTIONS:",
    ("Color", "Input", "Mute", "Notation", "Output", "Precision", "Radix"),
)

COLOR = Menu("OPTIONS COLOR:", ("Menu", "Work"))

# The Transfer menus. `Print` is the one word of the original's that is gone:
# its four commands are a printer driver - printer type, paper size, PCL font
# strings - and paper is a non-goal, the way `Options Display` and `Options
# Execute` are.
TRANSFER = Menu("TRANSFER:", ("Load", "Save", "Merge", "Clear", "Demo"))

TRANSFER_LOAD = Menu("TRANSFER LOAD:", ("Derive", "State", "daTa", "Utility"))

# The original's four targets were Basic, C, Fortran and Pascal, the languages
# a 1990s reader would paste an expression into. The command is the same
# command; only the list is dated, so it is dated forward. Every word still
# carries a mnemonic of its own, which is what the menu needs of it.
TRANSFER_SAVE = Menu(
    "TRANSFER SAVE:",
    ("Derive", "C", "Python", "Rust", "Julia", "Options", "State"),
)

# Alphabetical, and so with the widest command first, exactly as the original
# lists it.
TRANSFER_CLEAR = Menu(
    "TRANSFER CLEAR:", ("All", "Expressions", "Functions", "Variables")
)


def save_block(first: int, last: int) -> settings.Dialog:
    """The dialog that asks which block of expressions a save writes.

    Put up by `Transfer Save Derive` when the Range option says Some. It offers
    the whole history and stores nothing: the block is answered for the one
    save that asked, and the next save asks again.
    """
    return settings.Dialog(
        "TRANSFER SAVE DERIVE:",
        (
            (
                settings.NumberField(
                    "Start", ENTER_LABEL, "SaveFirst", first, minimum=1, recorded=False
                ),
                settings.NumberField(
                    "End", ENTER_LABEL, "SaveLast", last, minimum=1, recorded=False
                ),
            ),
        ),
        stored=False,
    )


def remove_block(number: int) -> settings.Dialog:
    """The dialog that asks which block of expressions Remove takes out.

    Both fields offer the highlighted entry, so removing the expression you are
    looking at is two keystrokes. Naming the same label twice removes it alone,
    which is what the manual tells you to do for one expression.
    """
    return settings.Dialog(
        "REMOVE:",
        (
            (
                settings.NumberField(
                    "Start", ENTER_LABEL, "RemoveFirst", number, minimum=1, recorded=False
                ),
                settings.NumberField(
                    "End", ENTER_LABEL, "RemoveLast", number, minimum=1, recorded=False
                ),
            ),
        ),
        stored=False,
        tracks_selection=True,
    )


def unremove_before(number: int) -> settings.Dialog:
    """The dialog that asks where the removed expressions go back.

    The one field takes the label of the entry to put them in front of, or
    `end` to put them after the last one.
    """
    return settings.Dialog(
        "UNREMOVE:",
        (
            (
                settings.NumberField(
                    "Before",
                    ENTER_LABEL_OR_END,
                    "UnremoveBefore",
                    number,
                    minimum=1,
                    recorded=False,
                    word=END,
                ),
            ),
        ),
        stored=False,
        tracks_selection=True,
    )


def move_block(number: int) -> settings.Dialog:
    """The dialog that asks which block of expressions moVe rearranges, and where.

    All three fields offer the highlighted entry, so the command opens on a
    move of that one expression to where it already is. Only the destination
    takes `end`, which sends the block past the last entry; a block has to be
    delimited by labels that name something.
    """
    return settings.Dialog(
        "MOVE:",
        (
            (
                settings.NumberField(
                    "Before",
                    ENTER_LABEL_OR_END,
                    "MoveBefore",
                    number,
                    minimum=1,
                    recorded=False,
                    word=END,
                ),
                settings.NumberField(
                    "Start", ENTER_LABEL, "MoveFirst", number, minimum=1, recorded=False
                ),
                settings.NumberField(
                    "End", ENTER_LABEL, "MoveLast", number, minimum=1, recorded=False
                ),
            ),
        ),
        stored=False,
        tracks_selection=True,
    )


def annotate_entry(number: int) -> settings.Dialog:
    """The dialog that asks which expression `Manage Annotate` is annotating.

    The one field offers the highlighted entry, and takes the arrow keys the
    way `Remove` and `Unremove` do, so the expression can be picked rather than
    numbered. What to say about it is the next question, and that one needs a
    line rather than a field.
    """
    return settings.Dialog(
        "MANAGE ANNOTATE:",
        (
            (
                settings.NumberField(
                    "Expression",
                    ENTER_LABEL,
                    "AnnotateExpression",
                    number,
                    minimum=1,
                    recorded=False,
                ),
            ),
        ),
        stored=False,
        tracks_selection=True,
    )


def matrix_size(rows: int, columns: int) -> settings.Dialog:
    """The dialog that asks how big the matrix Declare Matrix is entering is.

    It opens on the last size entered, three by three until one has been, so
    that a session full of two-by-twos asks about the size once.
    """
    return settings.Dialog(
        "DECLARE MATRIX:",
        (
            (
                settings.NumberField(
                    "Rows", ENTER_ROWS, "MatrixRows", rows, minimum=1, recorded=False
                ),
                settings.NumberField(
                    "Columns",
                    ENTER_COLUMNS,
                    "MatrixColumns",
                    columns,
                    minimum=1,
                    recorded=False,
                ),
            ),
        ),
        stored=False,
    )


def vector_dimension(dimension: int | str) -> settings.Dialog:
    """The dialog that asks how many elements Declare vectoR is collecting.

    It opens on the last dimension entered, and on nothing at all until one
    has been: unlike a matrix, the original offers no size to start from.
    """
    return settings.Dialog(
        "DECLARE VECTOR:",
        (
            (
                settings.NumberField(
                    "Dimension",
                    ENTER_ELEMENTS,
                    "VectorDimension",
                    dimension,
                    minimum=1,
                    recorded=False,
                ),
            ),
        ),
        stored=False,
    )


def variable_bounds(name: str, bounds: Bounds) -> settings.Dialog:
    """The dialog that asks for the bounds of a variable's domain.

    One line reading `lower (<)≤ variable (<)≤ upper`, where each strictness
    field shows both symbols and parenthesizes the one in force. The variable
    is a word on the line rather than a field: it is what is being bounded,
    and there is nothing to answer about it.
    """
    return settings.Dialog(
        "DECLARE VARIABLE:",
        (
            (
                settings.TextField(
                    "Bounds", ENTER_LEFT_BOUND, "BoundLow", bounds.low, _BOUND_WIDTH
                ),
                _strictness("StrictLow", ENTER_LEFT_BOUND, bounds.closed_low),
                name,
                _strictness("StrictHigh", ENTER_RIGHT_BOUND, bounds.closed_high),
                settings.TextField(
                    "", ENTER_RIGHT_BOUND, "BoundHigh", bounds.high, _BOUND_WIDTH
                ),
            ),
        ),
        stored=False,
    )


def _strictness(setting: str, message: str, closed: bool) -> settings.ChoiceField:
    """Whether one bound is strict: `<` for a bound the variable cannot reach."""
    return settings.ChoiceField(
        "", message, setting, STRICTNESS, CLOSED if closed else OPEN, recorded=False
    )


def closed(value: str | int) -> bool:
    """Whether a strictness field is set to the nonstrict of the two symbols."""
    return value == CLOSED


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

#: What each word of the Manage menu opens. The four that are missing -
#: Annotate, Ordering, Renumber and Substitute - are commands rather than
#: screens, and the app is where they live.
MANAGE_TARGETS: dict[str, Menu | settings.Dialog] = {
    "Branch": settings.BRANCH,
    "Exponential": settings.EXPONENTIAL,
    "Logarithm": settings.LOGARITHM,
    "Trigonometry": settings.TRIGONOMETRY,
}

#: Every word that opens another screen rather than doing something, by the
#: menu it is listed on. A word on none of these lists is a command, and the
#: app decides whether it has one.
TARGETS: dict[Menu, dict[str, Menu | settings.Dialog]] = {
    ALGEBRA: {
        "Declare": DECLARE,
        "Manage": MANAGE,
        "Options": OPTIONS,
        "Transfer": TRANSFER,
    },
    MANAGE: MANAGE_TARGETS,
    OPTIONS: OPTIONS_TARGETS,
    COLOR: COLOR_TARGETS,
    TRANSFER: {
        "Load": TRANSFER_LOAD,
        "Save": TRANSFER_SAVE,
        "Clear": TRANSFER_CLEAR,
    },
    TRANSFER_SAVE: {"Options": settings.SAVE},
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

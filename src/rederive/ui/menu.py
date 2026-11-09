"""The Algebra pane's menus: their words, mnemonic letters, and what each word opens.

`Unremove` is always listed. R-MENU1 says it appears only when there is something to
restore, but Derive shows it even in a session where nothing was removed, so the
requirement is wrong here.
"""

from __future__ import annotations

from dataclasses import dataclass

from rederive.model import building, demos, settings
from rederive.model import help as helps
from rederive.model.session import Bounds

MENU_TITLE = "COMMAND:"

#: Number of options on a menu's first line; the rest go on the second.
FIRST_LINE_OPTIONS = 10

ENTER_OPTION = "Enter option"

#: Message line prompt for a field that takes a label.
ENTER_LABEL = "Enter label number"

#: Prompt for the one label field that also accepts a word.
ENTER_LABEL_OR_END = 'Enter label number or type "end"'

#: The word that places expressions after the last entry.
END = "end"

#: Prompt for a field that takes an expression. All Calculus fields do; even a
#: derivative's order is an expression, as in `DIF(u, x, n + 1)`.
ENTER_ANY = "Enter expression"

#: Prompt for a pair of limits that may be left blank, which asks for an indefinite
#: integral, antidifference, or antiquotient.
ENTER_OR_SKIP = "Enter expression or press Enter"

#: Prompts for the two fields of `Calculus Limit`.
ENTER_POINT = "Enter limit point"
SELECT_DIRECTION = "Select approach direction"

#: Prompts for the two fields of `Calculus Taylor`, the only command with a different
#: prompt for each field.
ENTER_DEGREE = "Enter maximum degree"
ENTER_EXPANSION_POINT = "Enter expansion point"

#: Prompt for both fields of soLve's interval (asked only in Approximate precision).
ENTER_BOUND = "Enter bound on solution"

#: Prompts for the fields of the Declare dialogs.
ENTER_ROWS = "Enter number of rows"
ENTER_COLUMNS = "Enter number of columns"
ENTER_ELEMENTS = "Enter number of elements"
ENTER_LEFT_BOUND = "Enter left bound"
ENTER_RIGHT_BOUND = "Enter right bound"

#: Prompt for the Window commands that take a window number.
ENTER_WINDOW = "Enter window number"

#: Bound symbols: `<` for a strict bound, `≤` for a bound the variable can equal.
OPEN = "<"
CLOSED = "≤"
STRICTNESS = (OPEN, CLOSED)

#: Fixed width of a bound field, so neighboring fields don't shift while typing. Wider
#: than the widest bound Derive shows, `-∞`, like Derive's own field.
_BOUND_WIDTH = 6


def mnemonic(word: str) -> str:
    """The lower-cased capital letter that invokes `word`, e.g. `l` for soLve.

    A digit counts as a capital, which distinguishes `2D-plot` from `3D-plot`.
    """
    for character in word:
        if character.isupper() or character.isdigit():
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
    """A message line that overrides the menu's own prompt.

    The Declare Variable menus use it to name the variable being declared.
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

#: How much factoring Factor does, in increasing order: each amount does everything the
#: previous one does, and takes longer. `raDical` uses `D` because `R` is Rational.
AMOUNT = Menu(
    "FACTOR: Amount:",
    ("Trivial", "Squarefree", "Rational", "raDical", "Complex"),
    message="Select amount of factoring",
)

#: How much Expand factors the denominator for partial fractions. Same as `AMOUNT` minus
#: `Complex`, which Derive omits here because it would put `#i` into real answers.
EXPAND_AMOUNT = Menu(
    "EXPAND: Amount:",
    tuple(word for word in AMOUNT.words if word != "Complex"),
    message=AMOUNT.message,
)

#: The seven Calculus commands. Each uses its initial as mnemonic, and each asks for an
#: expression and a variable before its own dialog. Derive 6 added `Table`, but the DOS
#: versions, and this one, don't have it.
CALCULUS = Menu(
    "CALCULUS:",
    ("Differentiate", "Integrate", "Limit", "Product", "Sum", "Taylor", "Vector"),
)

#: The operators `Build` offers, on one line, with `Done` last. Symbols are their own
#: mnemonics; words use their initials.
BUILD_OPERATOR = Menu(
    "BUILD: Operator:",
    building.WORDS,
    message="Select operator",
    first_line=len(building.WORDS),
)

#: Width of a Calculus field, enough for typical limits and points without shifting the
#: neighboring field, as in Derive. `Vector` has three fields per line instead of two,
#: so they are narrower.
_FIELD_WIDTH = 17
_NARROW_WIDTH = 14

#: The limit directions and their values as `LIM`'s fourth argument. Like Derive, the
#: argument is always written, so a two-sided limit is `LIM(u, x, a, 0)`, never the
#: three-argument form.
DIRECTIONS: dict[str, str] = {"Both": "0", "Left": "-1", "Right": "1"}


def _expression_field(
    label: str, setting: str, default: str, message: str, width: int = _FIELD_WIDTH
) -> settings.TextField:
    """A Calculus field holding an expression, which it doesn't validate."""
    return settings.TextField(label, message, setting, default, width)


def _limits(title: str, setting: str, low: str, high: str) -> settings.Dialog:
    """The `Lower limit`/`Upper limit` line shared by three commands.

    Leaving both blank asks for the indefinite form, so validation is left to the
    command, which rejects only a pair with one side blank.
    """
    return settings.Dialog(
        title,
        (
            (
                _expression_field(
                    "Lower limit", f"{setting}Lower", low, ENTER_OR_SKIP
                ),
                _expression_field(
                    "Upper limit", f"{setting}Upper", high, ENTER_OR_SKIP
                ),
            ),
        ),
        stored=False,
        enters=True,
    )


#: The order for `Calculus Differentiate`, defaulting to one. A negative order asks for
#: an antiderivative (manual 7.4).
DIFFERENTIATE = settings.Dialog(
    "CALCULUS DIFFERENTIATE:",
    ((_expression_field("Order", "DifferentiateOrder", "1", ENTER_ANY),),),
    stored=False,
    enters=True,
)

#: The bounds of an integral, blank for an indefinite integral.
INTEGRATE = _limits("CALCULUS INTEGRATE:", "Integrate", "", "")

#: The bounds of a sum and a product. Unlike an integral's, they default to `1` and `n`,
#: as in Derive.
SUM = _limits("CALCULUS SUM:", "Sum", "1", "n")
PRODUCT = _limits("CALCULUS PRODUCT:", "Product", "1", "n")

#: The limit point and the direction of approach.
LIMIT = settings.Dialog(
    "CALCULUS LIMIT:",
    (
        (
            _expression_field("Point", "LimitPoint", "0", ENTER_POINT),
            settings.ChoiceField(
                "From",
                SELECT_DIRECTION,
                "LimitDirection",
                tuple(DIRECTIONS),
                "Both",
                recorded=False,
            ),
        ),
    ),
    stored=False,
    enters=True,
)

#: The degree and expansion point of a Taylor series. The dialog asks for the degree
#: first but `TAYLOR` takes the point first; the command reorders them.
TAYLOR = settings.Dialog(
    "CALCULUS TAYLOR:",
    (
        (
            _expression_field("Degree", "TaylorDegree", "5", ENTER_DEGREE),
            _expression_field("Point", "TaylorPoint", "0", ENTER_EXPANSION_POINT),
        ),
    ),
    stored=False,
    enters=True,
)

#: The range of a `Calculus Vector` index. `End` has no default and is required, so, as in
#: Derive, the dialog opens on it rather than on the leftmost field.
VECTOR = settings.Dialog(
    "CALCULUS VECTOR:",
    (
        (
            _expression_field("Start", "VectorStart", "1", ENTER_ANY, _NARROW_WIDTH),
            _expression_field("End", "VectorEnd", "", ENTER_ANY, _NARROW_WIDTH),
            _expression_field("Step", "VectorStep", "1", ENTER_ANY, _NARROW_WIDTH),
        ),
    ),
    stored=False,
    opens_on="VectorEnd",
    enters=True,
)

#: The interval for soLve's numeric search, asked only in Approximate precision (the
#: other modes solve exactly). Defaults to `[-10, 10]` as in Derive. The command builds
#: its own expression, so Ctrl-Enter does nothing on this line.
SOLVE_BOUNDS = settings.Dialog(
    "SOLVE:",
    (
        (
            _expression_field("Lower", "SolveLower", "-10", ENTER_BOUND),
            _expression_field("Upper", "SolveUpper", "10", ENTER_BOUND),
        ),
    ),
    stored=False,
)

#: The two Plot commands: plot into the receiving window, or into a new one. `Plot` is
#: first and so highlighted, so `P` `P` plots the highlighted expression immediately,
#: matching Derive's muscle memory (one keystroke shorter on the first plot, since the
#: window opens itself). All other plot controls live in the plot window.
PLOT = Menu("PLOT:", ("Plot", "New"))

DECLARE = Menu("DECLARE:", ("Function", "Variable", "Matrix", "vectoR"))

#: Prompts Declare Variable shows after getting a name, then after getting a domain.
#: `{name}` is replaced by the variable.
SELECT_DOMAIN = "Select value or domain of {name}"
SELECT_INTERVAL = "Select interval of {name}"

#: The two menus Declare Variable shows. As in Derive they share a title, and the
#: message line tells them apart.
DECLARE_VARIABLE = Menu(
    "DECLARE VARIABLE:",
    ("Value", "Integer", "Real", "Complex", "Nonscalar"),
    message=SELECT_DOMAIN,
)

#: The intervals, asked only for Integer and Real. `nonpoSitive` and `nonneGative` use
#: later letters because `N` is Negative.
DECLARE_INTERVAL = Menu(
    "DECLARE VARIABLE:",
    ("All", "Positive", "Negative", "nonpoSitive", "nonneGative", "Interval"),
    message=SELECT_INTERVAL,
)

#: Domains that take an interval. Choosing Complex or Nonscalar ends the command.
BOUNDED_DOMAINS = ("Integer", "Real")

# Four of these commands are settings dialogs; the other four act on the session and
# prompt on their own line. `Trigonometry` is on the second line because all eight
# don't fit in eighty columns.
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

# Derive's `Display` (text/graphics mode for obsolete adapters) and `Execute` (DOS
# shell) are omitted. There is no plot options screen either: plot preferences are
# sticky (a new plot starts as the last was left) and are controlled from the plot
# windows.
OPTIONS = Menu(
    "OPTIONS:",
    ("Color", "Input", "Mute", "Notation", "Output", "Precision", "Radix"),
)

COLOR = Menu("OPTIONS COLOR:", ("Menu", "Work"))

# The Transfer menus. Derive's `Print` is omitted: its commands configure a printer
# driver (printer type, paper size, PCL font strings), and printing is a non-goal.
TRANSFER = Menu("TRANSFER:", ("Load", "Save", "Merge", "Clear", "Demo"))

TRANSFER_LOAD = Menu("TRANSFER LOAD:", ("Derive", "State", "daTa", "Utility"))

#: The word for Derive's Demo file prompt, now that the nine built-in demos have their
#: own words.
DEMO_OTHER = "Other"

#: The demos, as menu options rather than a file name prompt.
#:
#: This is the one departure from Derive's Transfer menu. Derive's Demo asked for a
#: file name because the demo files shipped on the program's diskette. Here the demos
#: are listed on the menu, and `Other` is Derive's original prompt for a demo file of
#: your own.
#:
#: The first line holds the six simplification scripts, the second the three plot
#: galleries. Each word has a unique mnemonic; the two plot galleries are told apart by
#: their digit, like `2D-plot` and `3D-plot`.
TRANSFER_DEMO = Menu(
    "TRANSFER DEMO:",
    tuple(demo.word for demo in demos.DEMOS) + (DEMO_OTHER,),
    first_line=sum(1 for demo in demos.DEMOS if not demo.plotting),
)

# Derive's targets were Basic, C, Fortran and Pascal. The command is unchanged, but the
# list of languages is updated. Each word has a unique mnemonic.
TRANSFER_SAVE = Menu(
    "TRANSFER SAVE:",
    ("Derive", "C", "Python", "Rust", "Julia", "Options", "State"),
)

# Alphabetical, so the broadest command comes first, as in Derive.
TRANSFER_CLEAR = Menu(
    "TRANSFER CLEAR:", ("All", "Expressions", "Functions", "Variables")
)

# The seven window commands, each with a unique initial.
#
# `Designate` is omitted. It set a window's kind, but plot windows are top-level
# desktop windows rather than leaves of this tree, so Algebra is the only kind. Its
# remaining use, resetting a window to a fresh worksheet, is covered by `Transfer
# Clear`. It would return if another kind of tree window were added.
WINDOW = Menu(
    "WINDOW:",
    ("Close", "Flip", "Goto", "Next", "Open", "Previous", "Split"),
)

WINDOW_SPLIT = Menu("WINDOW SPLIT:", ("Horizontal", "Vertical"))

#: Titles of the Window commands that prompt for something. `Open` doesn't prompt,
#: since there is only one kind of window.
WINDOW_CLOSE = "WINDOW CLOSE:"
WINDOW_GOTO = "WINDOW GOTO:"


def window_number(title: str, number: int, count: int) -> settings.Dialog:
    """The window number dialog for `Window Close` and `Window Goto`.

    Close defaults to the active window and Goto to the next one, the most likely
    choice for each.
    """
    return settings.Dialog(
        title,
        (
            (
                settings.NumberField(
                    "Window",
                    ENTER_WINDOW,
                    "WindowNumber",
                    number,
                    minimum=1,
                    maximum=count,
                    recorded=False,
                ),
            ),
        ),
        stored=False,
    )


def window_split(vertical: bool, at: int, low: int, high: int) -> settings.Dialog:
    """The dialog asking where to split a window.

    The line or column is counted from the window's corner. The bounds come from the
    window's current size, so a window too small to split has no valid answer.
    """
    word = "column" if vertical else "line"
    return settings.Dialog(
        f"WINDOW SPLIT {'VERTICAL' if vertical else 'HORIZONTAL'}:",
        (
            (
                settings.NumberField(
                    f"At {word}",
                    f"Enter {word} number",
                    "SplitAt",
                    at,
                    minimum=low,
                    maximum=high,
                    recorded=False,
                ),
            ),
        ),
        stored=False,
    )


def save_block(first: int, last: int) -> settings.Dialog:
    """The dialog asking which block of expressions to save.

    Shown by `Transfer Save Derive` when the Range option is Some. It defaults to the
    whole history and isn't stored, so each save asks again.
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
    """The dialog asking which block of expressions to Remove.

    Both fields default to the highlighted entry, so removing it takes two keystrokes.
    Giving the same label twice removes one expression, as the manual describes.
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
    """The dialog asking where to restore removed expressions.

    The field takes the label of the entry to insert before, or `end` to append.
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
    """The dialog asking which block of expressions to moVe, and where.

    All three fields default to the highlighted entry, i.e. a no-op move of that
    expression. Only the destination accepts `end`, meaning after the last entry; the
    block itself must be given by actual labels.
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
    """The dialog asking which expression `Manage Annotate` annotates.

    The field defaults to the highlighted entry and, like `Remove` and `Unremove`,
    follows the arrow keys, so the expression can be picked rather than numbered. The
    annotation text is asked for next, on a line of its own.
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
    """The dialog asking for the size of the matrix Declare Matrix will enter.

    It defaults to the last size entered, or three by three initially.
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
    """The dialog asking how many elements Declare vectoR will collect.

    It defaults to the last dimension entered. Unlike for matrices, Derive offers no
    initial default.
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

    One line reading `lower (<)≤ variable (<)≤ upper`, where each strictness field shows
    both symbols and parenthesizes the active one. The variable is plain text, not a
    field.
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
    """The field choosing whether a bound is strict (`<`) or not (`≤`)."""
    return settings.ChoiceField(
        "", message, setting, STRICTNESS, CLOSED if closed else OPEN, recorded=False
    )


def closed(value: str | int) -> bool:
    """Whether a strictness field is set to the nonstrict symbol."""
    return value == CLOSED


#: The help subjects, plus the word that exits help. Help replaces the expression view
#: rather than overlaying it, so its menus are ordinary: this one picks a subject, and
#: the ones below page through it.
HELP = Menu("HELP:", tuple(topic.word for topic in helps.TOPICS) + (helps.RESUME,))

#: The menu for reading a subject chosen from the subject menu, one per subject since
#: each is titled with its name. As in Derive, `Next` past the last page returns to
#: where the subject was opened rather than wrapping around.
HELP_PAGES: dict[str, Menu] = {
    topic.word: Menu(
        f"HELP {topic.word.upper()}:", (helps.NEXT, helps.PREVIOUS, helps.RESUME)
    )
    for topic in helps.TOPICS
}

#: The menu for a subject opened by F1 from a line. `Resume` returns to the line, and
#: `Subjects` leads to the rest of the help.
HELP_PAGES_ALONE: dict[str, Menu] = {
    topic.word: Menu(
        f"HELP {topic.word.upper()}:",
        (helps.NEXT, helps.PREVIOUS, helps.SUBJECTS, helps.RESUME),
    )
    for topic in helps.TOPICS
}

#: The subject each help page menu shows, for lookup by the app.
HELP_TOPICS: dict[Menu, str] = {
    menu: word
    for pages in (HELP_PAGES, HELP_PAGES_ALONE)
    for word, menu in pages.items()
}


#: Color names as spelled on the color menu. Derive asked for a number, so these
#: mnemonics are our own. All sixteen need unique letters, so `Blue`, `Brown`, `Gray`
#: and `Aqua` don't use their initials.
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

#: The color menu has a short title because a full path like `OPTIONS COLOR MENU
#: BACKGROUND:` would leave no room for the colors. The message line names the slot.
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

#: What each word of the Manage menu opens. Annotate, Ordering, Renumber and Substitute
#: are commands rather than screens, implemented in the app.
MANAGE_TARGETS: dict[str, Menu | settings.Dialog] = {
    "Branch": settings.BRANCH,
    "Exponential": settings.EXPONENTIAL,
    "Logarithm": settings.LOGARITHM,
    "Trigonometry": settings.TRIGONOMETRY,
}

#: Words that open another screen rather than act, grouped by menu. Any other word is a
#: command handled (or not) by the app.
TARGETS: dict[Menu, dict[str, Menu | settings.Dialog]] = {
    ALGEBRA: {
        "Calculus": CALCULUS,
        "Declare": DECLARE,
        "Manage": MANAGE,
        "Options": OPTIONS,
        "Plot": PLOT,
        "Transfer": TRANSFER,
        "Window": WINDOW,
    },
    WINDOW: {"Split": WINDOW_SPLIT},
    MANAGE: MANAGE_TARGETS,
    OPTIONS: OPTIONS_TARGETS,
    COLOR: COLOR_TARGETS,
    TRANSFER: {
        "Load": TRANSFER_LOAD,
        "Save": TRANSFER_SAVE,
        "Clear": TRANSFER_CLEAR,
        "Demo": TRANSFER_DEMO,
    },
    TRANSFER_SAVE: {"Options": settings.SAVE},
}


def color_at(index: int) -> str:
    """The color named by the color menu's `index`-th word."""
    return settings.COLORS[index]


def choice_word(value: str) -> str:
    """A field value's spelling with its mnemonic capital."""
    return COLOR_WORDS.get(value, value)


def pick(choices: tuple[str, ...], letter: str) -> str | None:
    """The choice `letter` selects, or None.

    The first match wins, as in Derive: on the Radix field `o` is Octal, and `Other` can
    only be reached with the arrow keys.
    """
    for choice in choices:
        if mnemonic(choice_word(choice)) == letter:
            return choice
    return None

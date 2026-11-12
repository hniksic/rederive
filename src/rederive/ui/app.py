"""The Rederive Algebra pane as a Textual application.

Key handling bypasses Textual's focus model. In DOS Derive the command menu and the
expression highlight are two simultaneous cursors driven by disjoint keys:
Tab/Space/Backspace/Enter and the mnemonic letters drive the menu, and the arrow keys
drive the selection. So the menu keys are app-level priority bindings, disabled by
`check_action` while a prompt line is up and its Input should get the keys. The keys
that move the highlight between expressions keep working under a prompt line, since a
line has no vertical movement; Left, Right, Home and End belong to the line while it
is up.

The mouse drives the same two cursors. Clicking a menu word does what its letter does,
clicking an expression moves the highlight (one level deeper per click, so a term of a
long sum is reachable without hitting its exact character), and the wheel scrolls the
pane. Each gesture applies only where its key would, so the mouse can't do anything
the keyboard can't. Textual's text selection is off, leaving Shift-drag (Option-drag on
a Mac) to the terminal; a drag over the input line selects text in the line, for
Ctrl-C and Ctrl-X.

Submenus and Options dialogs stack on the command menu. Esc pops one level, and
committing a dialog returns to the command menu, as in Derive. Commands with their own
question (the block `Remove` deletes, where `Unremove` restores) also use a dialog,
whose answer is passed to the command rather than stored in settings.

Commands that read a line (Author, Simplify, approX, Factor, Expand, Jump, the
Transfer commands) show the prompt band, an Input with a label. The mode says which
command the line belongs to and what Enter does. A completed command returns to the
command menu; Esc abandons the line and leaves the submenu up.

Ctrl-Enter is Enter followed by simplifying the entered expression, on every line that
enters an expression. Terminals without the extended keyboard protocol send Ctrl-J for
it.

Ctrl-C copies the highlighted expression to the system clipboard (through the terminal
and through desktop tools), or the selected text if part of the line is selected;
Ctrl-X also cuts it. Ctrl-V pastes from the system clipboard onto the line, falling back
on an internal copy where there are no desktop tools (as over ssh). The terminal's own
paste key works everywhere, and in a browser Ctrl-V goes to the browser. Derive's F3
and F4, which copy the highlight onto the line, still work, but help documents copy
and paste instead.

File name lines complete as you type. The first matching name appears dimmed after the
cursor and Right accepts it. Tab inserts the common prefix of all matches and opens a
list above the line: Up and Down move through it, updating the line, Enter enters a
directory or accepts a file, Esc closes the list, and typing narrows it. Unlike
Derive's read-only F1 listing, the list can be navigated.

Commands that reach the math engine are awaited on a worker task. Where the engine can
be aborted, the screen enters a modal state until the answer arrives: the worksheet
isn't read or repainted, and only Esc (abort) is active. The message line shows the
running command and elapsed time, and the status line shows memory use, so the user can
decide whether to wait. All dispatches go through one helper and finish in a completion
handler.

Two things use the band for something other than a menu. Yes/no questions (Quit, and
the Clear commands that discard expressions) leave the menu up without a highlight,
with `confirm` holding the action for Y. A demo takes over the band, showing the comment
for each step, and any key runs the next step.

Multi-question commands keep their partial answers in a state object, since each prompt
is gone when the next appears, and since the questions asked depend on the answers, the
sequence lives in the handlers rather than in a table:

- Factor and Expand share one flow, `Asking`: an expression, then variables one at a
  time on the prompt band, then an amount from a menu. A `Command` holds what differs
  (the prompt word, the amounts, when to ask for one, and what runs at the end). A
  number is factored without asking anything further.
- Manage Renumber runs immediately, Annotate asks for a label in a dialog then for the
  text on a line, and Ordering edits the order list on a line. None of them appends to
  the history.
- Manage Substitute (`Substituting`) asks for an expression, then a value for each of
  its variables, or one replacement for a highlighted subexpression. It appends the
  result unsimplified, so Ctrl-Enter on the last value is useful.
- The Declare commands (`Declaring`, `Defining`, `Entering`) replace each question with
  the next rather than stacking them, because Derive abandons the whole command from any
  of them; one Esc always returns to the Declare menu.
- Build (`Building`) loops until `Done`: an operand on a line, an operator from a menu,
  and another operand if the operator is binary. Each operator folds the tree so far
  into a new node. The menu is taken down while an operand is asked for, so Esc
  abandons the whole command as in Derive.
- soLve (`Solving`) asks for an expression, then as many variables as it leaves
  undetermined (computed from the target's variables and equation count), then, only in
  Approximate precision, a search interval in a dialog. It may append any number of
  expressions; with no solutions it appends nothing and says so on the message line,
  as Derive does.
- The seven Calculus commands ask for an expression and a variable (defaulting to the
  expression's primary variable), then show a command-specific dialog. A `Calculus`
  holds the head to write, the annotation word, the dialog, and how its answers become
  arguments. They append the head unevaluated, for Simplify.

Build and Calculus handle Ctrl-Enter on their last question themselves. Derive enters
just one expression, the simplified one, so the command removes the Ctrl-Enter mark and
simplifies what it built instead of appending it, as `Simp(#1+#1)` and
`Simp(Dif(#1,x))` show.

The `Window` commands work across worksheets. `self.session` is the active window's, so
other commands deal with a single session; window commands change which one is active
or create another. There is one work area per window over a frame that draws the
borders. With one window the frame is empty, matching Derive's unsplit borderless
screen.
"""

from __future__ import annotations

import asyncio
import re
import sys
import threading
import time
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from os.path import commonprefix
from pathlib import Path
from typing import Any, Callable

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.errors import NoWidget
from textual.geometry import Offset
from textual.suggester import Suggester
from textual.widget import Widget
from textual.widgets import Input, Static
from textual.widgets.input import Selection

from rederive import __version__, platform
from rederive.engine.client import Amount, EngineAborted
from rederive.model import building, state, windows, worksheet
from rederive.model import demos as catalogue
from rederive.model import help as helps
from rederive.model import session as sessions
from rederive.model.expr import Node
from rederive.model.plotting import Unplottable, classify, learned, preferences
from rederive.model.session import Session
from rederive.model.settings import (
    PLOT_PREFERENCES,
    ChoiceField,
    Dialog,
    DialogEditor,
    Settings,
    TextField,
)
from rederive.model.windows import Window, Windows
from rederive.plot import PREFIX, UNAVAILABLE, available
from rederive.plot import protocol as plots
from rederive.plot.proxy import PlotError, PlotProxy
from rederive.syntax import LANGUAGES, DeriveSyntaxError, Language, write_expression
from rederive.ui import menu as menus
from rederive.ui.menu import ALGEBRA, COLORS, ENTER_OPTION, Menu, MenuCursor
from rederive.ui.theme import COLOR_SETTINGS, Palette
from rederive.ui.widgets import (
    CompletionList,
    FieldBand,
    Frame,
    MenuBand,
    MenuRule,
    MessageLine,
    Panes,
    PromptLine,
    StatusLine,
    WorkArea,
)

MODE_MENU = "menu"
MODE_AUTHOR = "author"
MODE_SIMPLIFY = "simplify"
MODE_APPROX = "approx"
#: Prompt band modes while Factor or Expand is asking.
MODE_ASKING = "asking"
MODE_ASKING_VARIABLE = "asking_variable"
#: Build's two operand lines; the second returns after every binary operator.
MODE_BUILD = "build"
MODE_BUILD_NEXT = "build_next"
#: The two lines every Calculus command reads before its own dialog.
MODE_CALCULUS = "calculus"
MODE_CALCULUS_VARIABLE = "calculus_variable"
#: The two lines soLve reads before the interval (asked only in Approximate precision).
MODE_SOLVE = "solve"
MODE_SOLVE_VARIABLE = "solve_variable"
MODE_JUMP = "jump"
MODE_FILE = "file"
MODE_CONFIRM = "confirm"
MODE_VARIABLE_NAME = "variable_name"
MODE_VARIABLE_VALUE = "variable_value"
MODE_FUNCTION_NAME = "function_name"
MODE_FUNCTION_VALUE = "function_value"
MODE_FUNCTION_VARIABLE = "function_variable"
MODE_ELEMENT = "element"
MODE_ANNOTATION = "annotation"
MODE_ORDER = "order"
#: Prompt band modes while Manage Substitute is asking.
MODE_SUBSTITUTE = "substitute"
MODE_SUBSTITUTE_VALUE = "substitute_value"
MODE_DEMO = "demo"
#: The mode while a computation runs. It is modal because the command owns the session
#: until done: nothing may read the worksheet, and only Esc is active.
MODE_COMPUTE = "compute"
#: The mode while help is shown. A menu mode like the command menu, but the work area
#: shows a document, so the history navigation keys turn pages.
MODE_HELP = "help"

#: Modes in which the menu band has the keys.
MENU_MODES = (MODE_MENU, MODE_HELP)

#: Help paging direction per key: the keys that move through the history elsewhere.
HELP_PAGING = {"down": 1, "page_down": 1, "up": -1, "page_up": -1}

#: Modes in which the prompt band is shown and has the keys.
PROMPT_MODES = (
    MODE_AUTHOR,
    MODE_SIMPLIFY,
    MODE_APPROX,
    MODE_ASKING,
    MODE_ASKING_VARIABLE,
    MODE_BUILD,
    MODE_BUILD_NEXT,
    MODE_CALCULUS,
    MODE_CALCULUS_VARIABLE,
    MODE_SOLVE,
    MODE_SOLVE_VARIABLE,
    MODE_JUMP,
    MODE_FILE,
    MODE_VARIABLE_NAME,
    MODE_VARIABLE_VALUE,
    MODE_FUNCTION_NAME,
    MODE_FUNCTION_VALUE,
    MODE_FUNCTION_VARIABLE,
    MODE_ELEMENT,
    MODE_ANNOTATION,
    MODE_ORDER,
    MODE_SUBSTITUTE,
    MODE_SUBSTITUTE_VALUE,
)

#: Prompt lines that ignore history navigation keys: only the Factor and Expand
#: variable line, where Derive keeps the highlight still. Under every other line the
#: highlight can still move, since it is also used to find values to copy onto the
#: line.
SETTLED_MODES = (MODE_ASKING_VARIABLE,)

#: Prompt lines under which the highlight can be moved: all the others.
WALKED_MODES = tuple(mode for mode in PROMPT_MODES if mode not in SETTLED_MODES)

#: Prompt lines that start out with the highlighted expression's label, and follow the
#: highlight as it moves, as long as the offered label is still selected (typing takes
#: the line over). The Declare lines also take expressions but start empty, so they
#: have nothing to keep in sync.
LABELLED_MODES = (
    MODE_SIMPLIFY,
    MODE_APPROX,
    MODE_ASKING,
    MODE_BUILD,
    MODE_BUILD_NEXT,
    MODE_CALCULUS,
    MODE_SOLVE,
    MODE_SUBSTITUTE,
)

#: Prompt lines that enter an expression, where Ctrl-Enter enters and then simplifies.
#: Lines that derive an expression (Simplify, approX, Factor and Expand's expression)
#: already simplify it, and Jump enters nothing, so there Ctrl-Enter is plain Enter.
#: Substitute's value line is included because it leaves its result unsimplified.
ENTERING_MODES = (
    MODE_AUTHOR,
    MODE_VARIABLE_VALUE,
    MODE_FUNCTION_VALUE,
    MODE_ELEMENT,
    MODE_FILE,
    MODE_SUBSTITUTE_VALUE,
)

#: Prompt lines that take or name an expression, where expression keys apply: F3 and F4
#: copy onto them, and F6 gives them the left/right keys. The manual documents F3 for
#: the author line and says the Declare and Substitute value lines take "all the normal
#: line editing features provided by the Author command"; Derive shows the `Lin` flag
#: on exactly this set.
#:
#: Lines collecting something else (a variable, file name, label number, annotation)
#: always keep the left/right keys, regardless of the arrow key setting.
EXPRESSION_LINES = (
    tuple(mode for mode in ENTERING_MODES if mode != MODE_FILE) + LABELLED_MODES
)

#: What each Alt key inserts, a shortcut for typing the name out (as in Derive, whose
#: table used the IBM PC character set).
#:
#: Where Derive's case differs (its `Θ`, `Φ` and `Ω` were the only forms its character
#: set had), the key inserts what this program prints, so input and output match. `ê`
#: and `î` read as `#e` and `#i`, which can still be typed out.
#:
#: Alt-minus needs a terminal with the extended keyboard protocol; otherwise it arrives
#: as Escape plus hyphen and isn't seen. The letters work either way. Without Alt-minus,
#: type `±` as `+-`.
ALT_GLYPHS = {
    "a": "α",
    "b": "β",
    "d": "δ",
    "n": "ε",
    "h": "θ",
    "m": "μ",
    "p": "π",
    "s": "σ",
    "t": "τ",
    "f": "φ",
    "o": "ω",
    "g": "Γ",
    "q": "√",
    "e": "ê",
    "i": "î",
    "0": "∞",
    "minus": "±",
    # The subscript operator, a glyph in Derive and a word here, spaced as printed.
    "v": " SUB ",
}

# As in Derive, only the Author line mentions F1 help, although every line supports it.
ENTER_EXPRESSION = "Enter expression (press F1 for help)"
#: Prompt for commands that derive an expression, without the Author line's help hint.
ENTER_TO_DERIVE = "Enter expression"
#: Asked by Quit and the two Clear commands that discard expressions.
ABANDON_PROMPT = "Abandon expressions (Y/N)?"
AUTHOR_PROMPT = " AUTHOR expression: "
SIMPLIFY_PROMPT = " SIMPLIFY expression: "
APPROX_PROMPT = " APPROX expression: "
JUMP_PROMPT = " JUMP to: "
EXPRESSION_PROMPT = " {word} expression: "
VARIABLE_PROMPT = " {word} variable {number}: "
#: Build's two operand prompts; the second returns after every binary operator.
BUILD_PROMPT = " BUILD first expression: "
BUILD_NEXT_PROMPT = " BUILD next expression: "
#: The first and second prompts of every Calculus command.
CALCULUS_PROMPT = " CALCULUS {word} expression: "
CALCULUS_VARIABLE_PROMPT = " CALCULUS {word} variable: "
ENTER_VARIABLE = "Enter variable"
#: soLve's two prompts. As in Derive, the variable prompt isn't numbered, though it is
#: asked once per variable.
SOLVE_PROMPT = " SOLVE expression: "
SOLVE_VARIABLE_PROMPT = " SOLVE variable: "
#: Shown when solving finds nothing. Nothing is appended and the highlight stays put.
NO_SOLUTIONS = "No solutions found"
#: Message line text while collecting variables. Return on the first prompt selects
#: all of them; on later prompts it ends the list.
FIRST_VARIABLE = "Return for all or select 1: {variables}"
NEXT_VARIABLE = "Return for no more or select next: {variables}"
#: Shown once an answer arrives.
COMPUTE_TIME = "Compute time: {elapsed}"
#: Shown during a computation: the command, the elapsed time, and how to abort.
COMPUTING = "{command}: {elapsed}   ESC aborts"
#: Shown on abort. Like Derive, this reports the time even for unfinished commands.
ABORTED = "Aborted after {elapsed}"
#: How computing commands name themselves while running.
SIMPLIFYING = "Simplifying"
APPROXIMATING = "Approximating"
SOLVING = "Solving"
#: The name for Build and Calculus commands confirmed with plain Enter. They don't
#: reach the engine, so this is never shown, but the shared dispatch needs a name.
BUILDING = "Building"
#: Shown while a command asks the engine which variables to offer, which is usually
#: but not always fast.
READING = "Reading expression"
#: The Plot command's name while running. It queries the engine for variables and then
#: the plot host, which must start up on the first plot, so it can take a while.
PLOTTING = "Plotting"
#: Shown when a plot is done. No window is named, since windows are titled by content
#: and the target window has just been raised. The second wording tells the user that
#: the plot replaced an existing curve.
PLOTTED = "Plotting {label}"
REPLOTTED = "Replotting {label}"
#: Shown by Plot when nothing is highlighted.
NOTHING_TO_PLOT = "no expression to plot"
#: Shown when a plot window sends back a traced point, naming the new entry's label so
#: the next command can refer to it.
POINT_ENTERED = "Entered {text} as {label}"
#: Retry delay for a point sent back from a plot while a command owns the worksheets.
POINT_RETRY = 0.25
#: Seconds between elapsed time updates, fast enough to look like a running clock.
TICK = 0.1
#: How long a computation runs before the clock appears. Most commands finish sooner,
#: and a clock that appears only briefly would just flicker.
CLOCK_AFTER = 1.0

#: Jump's input: a label number, optionally surrounded by spaces. A sign is accepted,
#: as in Derive.
LABEL_NUMBER = re.compile(r"[-+]?[0-9]+")

#: Shown by Unremove when there is nothing to restore. Other refusals in Remove and
#: Unremove only beep, but this one has no dialog to leave up.
BUFFER_EMPTY = "Unremove buffer empty"

#: Confirmation of Ctrl-C, since copying leaves no visible trace. The copied expression
#: isn't quoted, since it may not fit on the line.
COPIED = "Copied the highlighted expression"
#: Confirmation of Ctrl-C copying selected text from the input line.
COPIED_TEXT = "Copied the selected text"
#: Confirmation of Ctrl-X.
CUT_TEXT = "Cut the selected text"
#: Shown by Ctrl-X with no selection on the line. Cutting only applies to the line, not
#: to a highlighted expression, which the message makes clear.
NOTHING_TO_CUT = "Nothing is selected on the line to cut"

# Prompts and messages for the Declare commands. `default` stands for all undeclared
# variables, hence the name field's hint.
VARIABLE_NAME_PROMPT = " DECLARE VARIABLE name: "
VARIABLE_VALUE_PROMPT = " DECLARE VARIABLE value: "
FUNCTION_NAME_PROMPT = " DECLARE FUNCTION name: "
FUNCTION_VALUE_PROMPT = " DECLARE FUNCTION value: "
FUNCTION_VARIABLE_PROMPT = " DECLARE FUNCTION variable: "
VECTOR_ELEMENT_PROMPT = " VECTOR element: "
MATRIX_ELEMENT_PROMPT = " MATRIX element: "
ENTER_VARIABLE_NAME = 'Enter name or type "default"'
ENTER_FUNCTION_NAME = "Enter name"
ENTER_DEFINITION = "Enter expression"
ENTER_FUNCTION_VARIABLE = "Enter variable or press ENTER"
ENTER_VECTOR_ELEMENT = "Enter vector element {number}"
ENTER_MATRIX_ELEMENT = "Enter matrix element ({row},{column})"

#: Default for a matrix element, so entering a sparse matrix is mostly pressing Enter.
#: Vector elements have no default, as in Derive.
MATRIX_ZERO = "0"

# The two Manage commands that read a line. Annotate starts with the entry's current
# annotation (`User`, or a command's `Simp(#3)`), and Ordering with the current order
# list, both selected so typing replaces them. Derive overwrites instead (typing `w`
# over `x y z` gives `w y z`), but every prefilled line here starts selected.
ANNOTATION_PROMPT = " ANNOTATION: "
ORDER_PROMPT = " MANAGE ORDER variables: "
ENTER_ANNOTATION = "Enter annotation"
ENTER_ORDER = "Enter variables in desired order"

# Manage Substitute reads the expression, then one line per value. A variable's line
# starts with its own name, so Enter leaves it unchanged; a subexpression's line starts
# empty, with the message line saying what is being replaced.
SUBSTITUTE_PROMPT = " MANAGE SUBSTITUTE expression: "
SUBSTITUTE_VALUE_PROMPT = " MANAGE SUBSTITUTE value: "
ENTER_REPLACEMENT = "Enter replacement for {name}"
SUBEXPRESSION = "subexpression"

# Prompts for the commands that take a file name. Instead of Derive's F1 directory
# listing, Tab completes on the line.
LOAD_PROMPT = " TRANSFER LOAD DERIVE file: "
LOAD_STATE_PROMPT = " TRANSFER LOAD STATE file: "
LOAD_DATA_PROMPT = " TRANSFER LOAD DATA file: "
LOAD_UTILITY_PROMPT = " TRANSFER LOAD UTILITY file: "
MERGE_PROMPT = " TRANSFER MERGE file: "
SAVE_PROMPT = " TRANSFER SAVE DERIVE file: "
SAVE_SOURCE_PROMPT = " TRANSFER SAVE {word} file: "
SAVE_STATE_PROMPT = " TRANSFER SAVE STATE file: "
DEMO_PROMPT = " TRANSFER DEMO OTHER file: "
ENTER_FILE = "Enter filename (TAB completes, opens the list)"
#: Message line text while the completion list is open, listing its keys.
BROWSING_FILES = "↑↓ choose   ENTER open   ESC close"
FILE_NOT_FOUND = "File not found"
CANNOT_READ = "Cannot read file"
CANNOT_WRITE = "Cannot write file"
#: Shown when lines in a loaded file fail to parse. Derive drops them silently, but a
#: missing expression is worth reporting.
UNREADABLE = "{count} expression{s} could not be read"
#: The same for a state file, whose lines are settings.
UNSET = "{count} setting{s} could not be read"
#: Default for the state file prompts before any state file has been used.
STATE_FILE = "rederive.ini"
#: Shown between demo steps. Esc is mentioned because a demo is the only thing that
#: keeps the screen on its own, and users should know how to leave it early.
PRESS_ANY_KEY = "Press any key to continue, ESC to stop"

#: Shown while downloading one of Derive's demos. It uses the file name rather than
#: the menu word, since that's what the demo is called afterwards.
FETCHING = "Fetching {name} from the Internet Archive..."
#: Shown when the download fails after all sources are tried, with the error from the
#: primary source.
CANNOT_FETCH = "{name} could not be downloaded: {trouble}"
#: Shown when a demo is requested while another is still downloading. Two at once would
#: race for the same worksheet.
STILL_FETCHING = "{name} is still on its way; the next one can be asked for after"

#: The opening notice, shown until something else is drawn. Derive's included an
#: address, fax number and a copying warning; this keeps the name, description and
#: version, with the help hint at the bottom, spaced like Derive's title page.
GREETING = (
    "R E D E R I V E",
    "A Mathematical Assistant",
    "",
    f"Version {__version__}",
)
#: The notice's footer line. It mentions the demos, which nothing else on screen
#: would.
GREETING_FOOTER = "Press H for help, T D for demos"
#: The footer when the demos are already offered elsewhere, such as a button on a web
#: page.
GREETING_HELP = "Press H for help"


class FileNames(Suggester):
    """Suggests the first matching file name as a file prompt is typed.

    Textual shows the rest of the name dimmed after the cursor and accepts it on Right
    or End. This complements the list Tab opens. Only names that extend the line are
    suggested.

    Nothing is cached, since files come and go while the program runs.
    """

    def __init__(self, suffix: str) -> None:
        # Case-sensitive, since the filesystem distinguishes `Work.mth` from `work.mth`.
        super().__init__(use_cache=False, case_sensitive=True)
        self.suffix = suffix

    async def get_suggestion(self, value: str) -> str | None:
        found = worksheet.completions(value, self.suffix)
        return found[0] if found else None


@dataclass
class Demonstration:
    """A demo in progress.

    A DMO file is a math file whose comments form the script: each expression is
    authored and simplified, with its preceding comment shown on the menu band, and
    the demo waits for a key between steps. Esc pauses it; `at` lets naming the same
    file again resume rather than restart it.

    A plot gallery (`plotting`) is the same, except each step is plotted.
    """

    path: Path
    #: The comment and the expression of each step, in file order.
    steps: tuple[tuple[str, str], ...]
    at: int = 0
    #: Whether a step is drawn rather than simplified.
    plotting: bool = False

    @property
    def done(self) -> bool:
        return self.at >= len(self.steps)


@dataclass(frozen=True)
class Command:
    """What differs between Factor and Expand, which ask the same questions.

    `needs_amount` says whether to ask for the amount: Factor asks unless the
    expression is a number, and Expand asks only if there is a denominator to factor.
    """

    #: The command word in prompts and in the amount menu title.
    word: str
    #: The command's name on the message line while running.
    running: str
    amounts: Menu
    needs_amount: Callable[[Session, str], bool]
    run: Callable[[Session, str, Amount, tuple[str, ...]], Awaitable[object]]


FACTOR = Command(
    "FACTOR",
    "Factoring",
    menus.AMOUNT,
    lambda session, request: not session.decomposes(request),
    lambda session, request, amount, variables: session.factor(
        request, amount, variables
    ),
)

EXPAND = Command(
    "EXPAND",
    "Expanding",
    menus.EXPAND_AMOUNT,
    lambda session, request: session.written_as_ratio(request),
    lambda session, request, amount, variables: session.expand(
        request, amount, variables
    ),
)


#: Menus that answer a command's question rather than listing commands.
AMOUNT_MENUS = (FACTOR.amounts, EXPAND.amounts)


def _text(values: Mapping[str, str | int], setting: str) -> str:
    """One answer from a Calculus dialog: an expression, or blank."""
    return str(values[setting]).strip()


def _filled(values: Mapping[str, str | int], *settings: str) -> str | None:
    """The first of `settings` left blank while another is filled in, or None.

    Paired limits must be both blank (indefinite) or both filled (definite). As in
    Derive, a half-filled pair is refused with the highlight on the empty field.
    """
    blank = [setting for setting in settings if not _text(values, setting)]
    return blank[0] if blank and len(blank) < len(settings) else None


def _needed(values: Mapping[str, str | int], *settings: str) -> str | None:
    """The first of the required `settings` left blank, or None."""
    return next((setting for setting in settings if not _text(values, setting)), None)


def _limits(values: Mapping[str, str | int], setting: str) -> tuple[str, ...]:
    """A pair of limits as arguments: both, or neither."""
    low, high = _text(values, f"{setting}Lower"), _text(values, f"{setting}Upper")
    return (low, high) if low and high else ()


@dataclass(frozen=True)
class Calculus:
    """What differs among the seven Calculus commands.

    All ask for an expression and a variable, then show their own `dialog`. Its answers
    become the arguments after the variable, in the head's order rather than the
    dialog's: `Taylor` asks for the degree first but writes the point first.

    None of them computes. Each appends `head(u, x, ...)` for a later Simplify, and
    `prefix` names it in the annotation, as in `Dif(#1,x)`.
    """

    word: str
    head: str
    prefix: str
    dialog: Dialog
    arguments: Callable[[Mapping[str, str | int]], tuple[str, ...]]
    refuses: Callable[[Mapping[str, str | int]], str | None] = lambda values: None


def _order(values: Mapping[str, str | int]) -> tuple[str, ...]:
    """The order of a derivative, omitted for a first derivative.

    Derive writes `DIF(u, x)` for a first derivative and `DIF(u, x, n)` otherwise. The
    text is compared rather than its value, so `2 - 1` is still written out.
    """
    order = _text(values, "DifferentiateOrder")
    return () if order == "1" else (order,)


def _vector_values(values: Mapping[str, str | int]) -> tuple[str, ...]:
    """A `Vector` index's start, end, and step, omitting a step of 1."""
    step = _text(values, "VectorStep")
    arguments = (_text(values, "VectorStart"), _text(values, "VectorEnd"))
    return arguments if step == "1" else (*arguments, step)


CALCULUS_COMMANDS: dict[str, Calculus] = {
    "Differentiate": Calculus(
        "DIFFERENTIATE",
        "DIF",
        "Dif",
        menus.DIFFERENTIATE,
        _order,
        lambda values: _needed(values, "DifferentiateOrder"),
    ),
    "Integrate": Calculus(
        "INTEGRATE",
        "INT",
        "Int",
        menus.INTEGRATE,
        partial(_limits, setting="Integrate"),
        lambda values: _filled(values, "IntegrateLower", "IntegrateUpper"),
    ),
    "Limit": Calculus(
        "LIMIT",
        "LIM",
        "Lim",
        menus.LIMIT,
        lambda values: (
            _text(values, "LimitPoint"),
            menus.DIRECTIONS[str(values["LimitDirection"])],
        ),
        lambda values: _needed(values, "LimitPoint"),
    ),
    "Product": Calculus(
        "PRODUCT",
        "PRODUCT",
        "Product",
        menus.PRODUCT,
        partial(_limits, setting="Product"),
        lambda values: _filled(values, "ProductLower", "ProductUpper"),
    ),
    "Sum": Calculus(
        "SUM",
        "SUM",
        "Sum",
        menus.SUM,
        partial(_limits, setting="Sum"),
        lambda values: _filled(values, "SumLower", "SumUpper"),
    ),
    "Taylor": Calculus(
        "TAYLOR",
        "TAYLOR",
        "Taylor",
        menus.TAYLOR,
        lambda values: (_text(values, "TaylorPoint"), _text(values, "TaylorDegree")),
        lambda values: _needed(values, "TaylorDegree", "TaylorPoint"),
    ),
    "Vector": Calculus(
        "VECTOR",
        "VECTOR",
        "Vector",
        menus.VECTOR,
        _vector_values,
        lambda values: _needed(values, "VectorEnd", "VectorStart", "VectorStep"),
    ),
}


@dataclass
class Helping:
    """The help on screen, and what leaving it returns to.

    `topic` is None on the subject menu; from a subject, Esc and `Resume` return there,
    and from the subject menu they leave help. `pages` is the subject's page count at
    the last layout; it depends on the pane size, so it is updated on every paint.

    `resume` and `message` record the screen's mode and message when help was opened.
    Help can be opened from a partly typed line, which must come back intact along with
    its message.
    """

    resume: str = MODE_MENU
    message: str = ENTER_OPTION
    topic: helps.Topic | None = None
    page: int = 0
    pages: int = 1


@dataclass
class Asking:
    """A Factor or Expand command in progress.

    It asks for an expression, then variables, then an amount, keeping the answers.
    """

    command: Command
    request: str = ""
    #: Variables not yet chosen, in the order offered.
    remaining: tuple[str, ...] = ()
    #: Variables chosen so far, in order; the first is the primary variable.
    chosen: tuple[str, ...] = ()


@dataclass
class Building:
    """A Build command in progress: the expression so far and its annotation.

    Each operator folds the expression so far into a new tree: `#1 + #2` becomes the
    sum, and a following `*` multiplies that sum by the next operand. `annotation` is
    built the same way from the operands' names, written flat as in Derive, so
    `#1+#2*#3` names `(#1 + #2)·#3`.

    `operator` is a binary operator awaiting its right operand, or None while the
    operator menu is shown.
    """

    node: Node | None = None
    annotation: str = ""
    operator: building.Operator | None = None


@dataclass
class Calculating:
    """A Calculus command in progress.

    Keeps the expression and variable from their lines; the rest is answered in the
    final dialog.
    """

    command: Calculus
    request: str = ""
    variable: str = ""


@dataclass
class Solving:
    """A soLve command in progress.

    It asks for an expression, then as many variables as it leaves undetermined, then,
    only in Approximate precision, a search interval. `wanted` is the number of
    variables still to ask for: one for an ambiguous scalar equation, and one per
    equation for an underdetermined system.
    """

    request: str = ""
    wanted: int = 0
    #: Variables not yet chosen, most significant first, the order defaults are
    #: offered in.
    remaining: tuple[str, ...] = ()
    #: Variables chosen so far, in order.
    chosen: tuple[str, ...] = ()


@dataclass
class Substituting:
    """A Manage Substitute command in progress.

    It asks for an expression, then for one value per variable, or for a single
    replacement of a highlighted subexpression. `part` says which.
    """

    request: str = ""
    part: bool = False
    #: Variables not yet answered, most significant first, as Derive asks them.
    remaining: tuple[str, ...] = ()
    #: Answers so far, one value per variable.
    values: tuple[tuple[str, str], ...] = ()


@dataclass
class Declaring:
    """A Declare Variable command in progress.

    It asks for a name, then a value or domain, then, for Integer and Real, an
    interval.
    """

    name: str
    #: The domain, once chosen from the menu.
    kind: str = ""


@dataclass
class Defining:
    """A Declare Function command in progress.

    It asks for a name and a definition. A blank definition declares an arbitrary
    function, whose variables are then asked for until a blank answer.
    """

    name: str
    variables: tuple[str, ...] = ()


@dataclass
class Entering:
    """A Declare Matrix or Declare vectoR command collecting its elements.

    A vector has `rows` set to None.
    """

    columns: int
    rows: int | None = None
    elements: tuple[str, ...] = ()

    @property
    def wanted(self) -> int:
        return self.columns * (self.rows or 1)

    @property
    def place(self) -> tuple[int, int]:
        """The row and column of the element being asked for, counting from 1."""
        entered = len(self.elements)
        return entered // self.columns + 1, entered % self.columns + 1

    @property
    def grid(self) -> list[tuple[str, ...]]:
        """The elements grouped into rows."""
        return [
            self.elements[start : start + self.columns]
            for start in range(0, len(self.elements), self.columns)
        ]


#: What navigation keys do in an Options dialog's number field.
CURSOR_MOVES = {
    "left": "back",
    "right": "forward",
    "first_sibling": "start",
    "last_sibling": "end",
}

#: Movements that navigate the history, even on a prompt line or a dialog whose fields
#: name expressions. Other movements move the cursor of whatever is shown.
ENTRY_MOVES = ("up", "down", "page_up", "page_down", "first_entry", "last_entry")

#: Movements that depend on the pane height.
PAGE_MOVES = ("page_up", "page_down")


def _takes_anything(values: dict[str, str | int]) -> str | None:
    """A refusal check that accepts everything, for dialogs without validation."""
    return None


def _pane_of(widget: Widget | None) -> WorkArea | None:
    """The work area that is or contains `widget`, or None.

    A pane paints into a child widget, so clicks land on the child and must be mapped
    to the pane.
    """
    if widget is None:
        return None
    for node in widget.ancestors_with_self:
        if isinstance(node, WorkArea):
            return node
    return None


def _variables_wanted(equations: int, variables: int) -> int:
    """How many variables soLve asks for.

    A scalar expression (`equations` is 0) asks once, unless it has exactly one
    variable. A system asks once per equation, but only if there are more variables
    than equations; otherwise there is nothing to choose.
    """
    if not equations:
        return 0 if variables == 1 else 1
    return equations if variables > equations else 0


@dataclass(frozen=True)
class Outcome:
    """The result of a session command, whichever task it ran on.

    Every outcome is captured here rather than raised, because the task that ran the
    command isn't the one that reports it, and a computation killed mid-way has no
    return path of its own.
    """

    value: object = None
    error: Exception | None = None
    seconds: float = 0.0


async def _ran(run: Callable[[], Awaitable[object]]) -> Outcome:
    """Await a session command and time it, however it ends.

    All exceptions are caught: a refusal, a dead engine worker, or a bug should be
    reported on the message line rather than crash the app.
    """
    started = time.monotonic()
    try:
        return Outcome(await run(), None, time.monotonic() - started)
    except Exception as error:
        return Outcome(None, error, time.monotonic() - started)


def _elapsed(seconds: float) -> str:
    """A duration, with precision appropriate to its size.

    `3.1s` under ten seconds, then `42s`, `10m 8s` or `2h 10m 8s`.
    """
    tenths = f"{seconds:.1f}"
    return f"{tenths}s" if float(tenths) < 10 else _whole(seconds)


def _whole(seconds: float) -> str:
    """A duration in whole seconds: `3s`, `42s`, `10m 8s` or `2h 10m 8s`.

    Used by the running clock, where tenths would be stale as soon as shown.
    """
    minutes, second = divmod(round(seconds), 60)
    if not minutes:
        return f"{second}s"
    hours, minute = divmod(minutes, 60)
    return f"{minute}m {second}s" if not hours else f"{hours}h {minute}m {second}s"


def _plot_reason(error: Exception) -> str:
    """The message for a failed Plot command, after the `Plot: ` prefix.

    Expected failures carry a readable message. Anything else is a bug, for which the
    exception type name is better than nothing.
    """
    return str(error) or type(error).__name__


def _appending(kept: list[tuple[str, bytes]]) -> Callable[[str, bytes], None]:
    """A callback for a download to store extra files it received.

    It runs on the download thread, so it only appends to a list that the event loop
    processes later.
    """
    return lambda file, data: kept.append((file, data))


def _reported(outcome: Outcome) -> str:
    """The message line text for a finished command."""
    error = outcome.error
    if error is None:
        return COMPUTE_TIME.format(elapsed=_elapsed(outcome.seconds))
    if isinstance(error, EngineAborted):
        return ABORTED.format(elapsed=_elapsed(outcome.seconds))
    return str(error)


class RederiveApp(App[None]):
    """A single full-screen Algebra pane."""

    CSS_PATH = "rederive.tcss"
    TITLE = "Rederive"
    AUTO_FOCUS = None
    #: The last line of the opening notice. Subclasses whose environment already
    #: offers the demos use `GREETING_HELP` instead.
    GREETING_LINE = GREETING_FOOTER
    # The screen belongs entirely to the pane.
    ENABLE_COMMAND_PALETTE = False
    # Leave screen text selection to the terminal; see "mouse routing".
    ALLOW_SELECT = False
    BINDINGS = [
        # Tab completes file names on a file prompt and steps the menu elsewhere;
        # `check_action` disables the first binding when it doesn't apply. Shift-Tab and
        # the arrows likewise navigate an open completion list, and act normally
        # otherwise.
        Binding("tab", "complete_file", "Complete filename", priority=True, show=False),
        Binding("tab", "menu_next", "Next option", priority=True, show=False),
        Binding("space", "menu_space", "Next option", priority=True, show=False),
        Binding(
            "shift+tab", "browse_previous", "Previous name", priority=True, show=False
        ),
        Binding(
            "shift+tab", "menu_previous", "Previous option", priority=True, show=False
        ),
        Binding("down", "browse_next", "Next name", priority=True, show=False),
        Binding("up", "browse_previous", "Previous name", priority=True, show=False),
        Binding("pagedown", "browse_page(1)", "Next page", priority=True, show=False),
        Binding("pageup", "browse_page(-1)", "Previous page", priority=True, show=False),
        Binding("backspace", "menu_erase", "Previous option", priority=True, show=False),
        Binding("delete", "menu_delete", "Delete", priority=True, show=False),
        # Enter on an open list takes the highlighted name rather than running the
        # command, so browsing never loads a file by accident.
        Binding("enter", "browse_take", "Take name", priority=True, show=False),
        Binding("enter", "menu_invoke", "Invoke option", priority=True, show=False),
        # Terminals without the extended keyboard protocol send Ctrl-Enter as Ctrl-J.
        Binding(
            "ctrl+j",
            "enter_and_simplify",
            "Enter and simplify",
            priority=True,
            show=False,
        ),
        Binding(
            "ctrl+enter",
            "enter_and_simplify",
            "Enter and simplify",
            priority=True,
            show=False,
        ),
        # Ctrl-C copies the highlighted expression and Ctrl-V pastes the clipboard onto
        # the line, to build new expressions from existing ones. The arrows still move
        # the highlight while a line is up, so you can move, copy, and paste.
        Binding("ctrl+c", "copy_highlighted", "Copy", priority=True, show=False),
        Binding("ctrl+v", "paste_clipboard", "Paste", priority=True, show=False),
        # Ctrl-X cuts the selection on the line. It is bound at the app level because
        # the line's own binding would also cut a label offered by a command.
        Binding("ctrl+x", "cut_selected", "Cut", priority=True, show=False),
        # Derive's equivalents, kept for users following the manual: F3 inserts the
        # highlighted expression onto the line, and F4 inserts it in parentheses. Help
        # documents copy and paste instead.
        Binding("f3", "insert_highlighted(0)", "Insert", priority=True, show=False),
        Binding("f4", "insert_highlighted(1)", "Insert ()", priority=True, show=False),
        # F1 and F2 step to the next window and the next overlay. They only apply at the
        # command menu, since on input lines Derive uses F1 for help.
        Binding("f1", "window_step(1)", "Next window", priority=True, show=False),
        Binding(
            "shift+f1", "window_step(-1)", "Previous window", priority=True, show=False
        ),
        # On an input line F1 is help instead; `check_action` picks the binding.
        Binding("f1", "help", "Help", priority=True, show=False),
        Binding("f2", "window_flip(1)", "Next overlay", priority=True, show=False),
        Binding(
            "shift+f2", "window_flip(-1)", "Previous overlay", priority=True, show=False
        ),
        # F6 toggles whether left/right keys go to the line or the highlight; Ins
        # toggles insert/overwrite. Derive also used Ctrl-V for Ins, but here it pastes.
        Binding("f6", "toggle_arrow_mode", "Arrow keys", priority=True, show=False),
        Binding("insert", "toggle_inserting", "Insert mode", priority=True, show=False),
        *[
            Binding(
                f"alt+{key}",
                f"insert_glyph({key!r})",
                "Glyph",
                priority=True,
                show=False,
            )
            for key in ALT_GLYPHS
        ],
        Binding("escape", "menu_escape", "Go back", priority=True, show=False),
        Binding("up", "nav('up')", "Up", priority=True, show=False),
        Binding("down", "nav('down')", "Down", priority=True, show=False),
        Binding("left", "nav('left')", "Left", priority=True, show=False),
        Binding("right", "nav('right')", "Right", priority=True, show=False),
        Binding("home", "nav('first_sibling')", "Home", priority=True, show=False),
        Binding("end", "nav('last_sibling')", "End", priority=True, show=False),
        Binding("pageup", "nav('page_up')", "Page up", priority=True, show=False),
        Binding("pagedown", "nav('page_down')", "Page down", priority=True, show=False),
        Binding("ctrl+home", "nav('first_entry')", "Top", priority=True, show=False),
        Binding("ctrl+end", "nav('last_entry')", "Bottom", priority=True, show=False),
        # Derive's alternative keys for the same two commands.
        Binding("ctrl+pageup", "nav('first_entry')", "Top", priority=True, show=False),
        Binding(
            "ctrl+pagedown", "nav('last_entry')", "Bottom", priority=True, show=False
        ),
        # Ctrl-Right and Ctrl-Left scroll a render that is too wide for the pane. They
        # don't move the selection, so they aren't `nav`.
        Binding("ctrl+right", "scroll_work(1)", "Right", priority=True, show=False),
        Binding("ctrl+left", "scroll_work(-1)", "Left", priority=True, show=False),
        # Alternatives for terminals where those Ctrl keys don't arrive. Being
        # characters, they only scroll when no line is being typed.
        Binding(">", "scroll_work_typed(1)", "Right", priority=True, show=False),
        Binding("<", "scroll_work_typed(-1)", "Left", priority=True, show=False),
    ]

    def __init__(
        self,
        session: Session | None = None,
        settings: Settings | None = None,
        demo: Path | None = None,
        opening: str = "",
        plotter: Any = None,
    ) -> None:
        """`demo` and `opening` come from the command line.

        The demo, if any, starts once the screen is up. `opening` is a message from
        loading the other files (such as a count of unparsable lines), shown instead
        of the usual prompt.

        `plotter` receives plot requests, defaulting to the standard plot host. Callers
        that draw plot windows differently (a browser, a test) pass their own.
        """
        # Set up before the base class, which reads the CSS variables on init. A session
        # brings its own settings, since there is only one store.
        if settings is None:
            settings = session.settings if session is not None else Settings()
        self.settings = settings
        self.palette = Palette(self.settings)
        super().__init__()
        #: All windows. `session` is the active one, which all commands here act on.
        self.windows = Windows(session if session is not None else Session(self.settings))
        #: Each window's work area, created with the window and kept for reuse after it
        #: closes. The mapping must be stable, since a pane's scroll belongs to its
        #: window.
        self.panes: dict[Window, WorkArea] = {}
        self.spare: list[WorkArea] = []
        #: The initial worksheet, and whether the opening notice is still shown. As in
        #: Derive, the notice is replaced by whatever draws the work area next (an
        #: expression, help, a window command, a Clear) and never returns, while menus
        #: and dialogs leave it in place. See `_opening_screen`.
        self.opening_session = self.windows.session
        self.greeting = True
        #: The plot windows, in a child process started on the first plot. Constructing
        #: this is cheap and doesn't import Qt or numpy, keeping them (like sympy) out
        #: of the app process. Callers may pass another implementation; the calls
        #: below are its whole interface.
        self.plots = PlotProxy() if plotter is None else plotter
        #: The event loop's thread (the app is run on the thread that creates it), used
        #: to tell whether an event must be passed to the loop thread.
        self._on_loop = threading.get_ident()
        self.plots.events = self._plot_reported
        self.plots.prefer(preferences(self.settings))
        self.settings.watch(self._settings_changed)
        self.mode = MODE_MENU
        #: The command menu, plus any submenus or dialogs stacked on it.
        self.stack: list[MenuCursor | DialogEditor] = [MenuCursor(ALGEBRA)]
        self.message = opening or ENTER_OPTION
        #: The demo named on the command line, until started.
        self.opening_demo = demo
        #: The action for the file named on the prompt line.
        self.file_command: Callable[[str], None] | None = None
        #: That command's file extension, used for completion.
        self.file_suffix = worksheet.SUFFIX
        #: The names in the completion list, and the index of the one copied onto the
        #: line (None if the line holds the user's typing). The list is closed when
        #: `completions` is empty. Kept together so the line and list never disagree.
        self.completions: list[str] = []
        self.completed: int | None = None
        #: The handler for the values of a dialog that doesn't store them.
        self.answer: Callable[[dict[str, str | int]], None] | None = None
        #: Returns the first field of that dialog the command refuses, if any.
        self.refuses: Callable[[dict[str, str | int]], str | None] = _takes_anything
        #: The label range the next save writes, if one was asked for.
        self.block: tuple[int | None, int | None] = (None, None)
        #: State of the Factor or Expand command in progress, if any.
        self.asking: Asking | None = None
        #: State of the Build in progress, if any.
        self.building: Building | None = None
        #: State of the Calculus command in progress, if any.
        self.calculating: Calculating | None = None
        #: State of the soLve command in progress, if any.
        self.solving: Solving | None = None
        #: State of the Manage Substitute in progress, if any.
        self.substituting: Substituting | None = None
        #: State of the multi-question Declare commands in progress, if any.
        self.declaring: Declaring | None = None
        self.defining: Defining | None = None
        self.entering: Entering | None = None
        #: The entry `Manage Annotate` is annotating, once chosen.
        self.annotating: int | None = None
        #: The defaults for the next Declare Matrix and Declare vectoR: the last size
        #: entered. As in Derive, matrices start at three by three and vectors with no
        #: default.
        self.matrix_size = (3, 3)
        self.dimension: int | str = ""
        #: Whether the left/right keys go to the line rather than the highlight. Reset
        #: from `ArrowKeyMode` whenever a line is shown, and toggled by F6.
        self.line_edit = True
        #: Insert mode, as opposed to Derive's overwrite mode. Unlike the arrow key mode,
        #: it isn't reset per command.
        self.inserting = True
        #: The line text and selection a command prefilled. Once the line differs, it
        #: belongs to the user, and copy and cut act on its selection.
        self.offering: tuple[str, Selection] | None = None
        #: The action for Y while a command asks a yes/no question.
        self.confirm: Callable[[], None] | None = None
        #: Whether that question was asked over a dialog, which the answer must close.
        self.confirmed_over = False
        #: The dialog `_ask` last showed, so a command can redisplay it with its answer.
        self.asked: Dialog | None = None
        #: The history when Ctrl-Enter was pressed, or None. Entries the command adds
        #: after it are simplified when the command is done.
        self.simplifying: tuple[sessions.Entry, ...] | None = None
        #: The help on screen, or None.
        self.helping: Helping | None = None
        #: The current demo, running or paused.
        self.demo: Demonstration | None = None
        #: The demo being downloaded, or None. Only one at a time, since two would race
        #: for the same worksheet.
        self.fetching: str | None = None
        #: Fetches one of Derive's demos. Uses sockets by default; in a browser tab,
        #: `rederive.web.demos` replaces it with a browser-based fetch.
        self.fetcher: Callable[[catalogue.Demo], Awaitable[bytes]] = self._downloaded
        #: Demos downloaded in this run, by Derive's file name. They are kept in memory
        #: and not written to disk, so a repeat viewing is instant but the next run
        #: downloads again.
        self.downloads: dict[str, str] = {}
        #: Whether a plot window closed while a demo was waiting. Closing a window
        #: returns focus here, and the closing click arrives along with it.
        self.closed_window = False
        #: The name of the running computation for the message line, or None.
        self.computing: str | None = None
        #: When it started, and the timer that updates the elapsed time.
        self.started = 0.0
        self.ticker: Any = None
        #: The mode to return to when it finishes: the one it was dispatched from, so a
        #: refused line finds its prompt still up.
        self.resumed = MODE_MENU
        #: The state file last read or written, offered by both state commands.
        self.state_file = STATE_FILE
        #: Each amount menu's cursor, kept across invocations since Derive opens it on
        #: the last choice.
        self.amounts = {
            command.amounts: MenuCursor(
                command.amounts, command.amounts.words.index("Rational")
            )
            for command in (FACTOR, EXPAND)
        }
        # Commands not listed here are present and navigable but inert.
        self.commands: dict[tuple[Menu, str], Callable[[], None]] = {
            (ALGEBRA, "Author"): self._command_author,
            (ALGEBRA, "Build"): self._command_build,
            (ALGEBRA, "Expand"): lambda: self._command_asking(EXPAND),
            (ALGEBRA, "Help"): self._command_help,
            (ALGEBRA, "Factor"): lambda: self._command_asking(FACTOR),
            (ALGEBRA, "Jump"): self._command_jump,
            (ALGEBRA, "Quit"): self._command_quit,
            (ALGEBRA, "Remove"): self._command_remove,
            (ALGEBRA, "Simplify"): self._command_simplify,
            (ALGEBRA, "soLve"): self._command_solve,
            (ALGEBRA, "Unremove"): self._command_unremove,
            (ALGEBRA, "moVe"): self._command_move,
            (ALGEBRA, "approX"): self._command_approx,
            (menus.MANAGE, "Annotate"): self._command_annotate,
            (menus.MANAGE, "Ordering"): self._command_ordering,
            (menus.MANAGE, "Renumber"): self._command_renumber,
            (menus.MANAGE, "Substitute"): self._command_substitute,
            (menus.DECLARE, "Function"): self._command_declare_function,
            (menus.DECLARE, "Variable"): self._command_declare_variable,
            (menus.DECLARE, "Matrix"): self._command_declare_matrix,
            (menus.DECLARE, "vectoR"): self._command_declare_vector,
            (menus.TRANSFER_DEMO, menus.DEMO_OTHER): self._command_demo,
            (menus.TRANSFER, "Merge"): self._command_merge,
            (menus.TRANSFER_CLEAR, "All"): self._command_clear_all,
            (menus.TRANSFER_CLEAR, "Expressions"): self._command_clear_expressions,
            (menus.TRANSFER_CLEAR, "Functions"): self._command_clear_functions,
            (menus.TRANSFER_CLEAR, "Variables"): self._command_clear_variables,
            (menus.TRANSFER_LOAD, "Derive"): self._command_load,
            (menus.TRANSFER_LOAD, "State"): self._command_load_state,
            (menus.TRANSFER_LOAD, "daTa"): self._command_load_data,
            (menus.TRANSFER_LOAD, "Utility"): self._command_load_utility,
            (menus.TRANSFER_SAVE, "Derive"): self._command_save,
            (menus.TRANSFER_SAVE, "State"): self._command_save_state,
            (menus.PLOT, "Plot"): partial(self._command_plot, None),
            (menus.PLOT, "New"): partial(self._command_plot, plots.Where.NEW),
            (menus.WINDOW, "Close"): self._command_window_close,
            (menus.WINDOW, "Flip"): self._command_window_flip,
            (menus.WINDOW, "Goto"): self._command_window_goto,
            (menus.WINDOW, "Next"): partial(self._command_window_step, 1),
            (menus.WINDOW, "Open"): self._command_window_open,
            (menus.WINDOW, "Previous"): partial(self._command_window_step, -1),
            (menus.WINDOW_SPLIT, "Horizontal"): partial(
                self._command_window_split, False
            ),
            (menus.WINDOW_SPLIT, "Vertical"): partial(self._command_window_split, True),
        }
        # The Calculus commands share one implementation, parametrized by `Calculus`.
        for word, calculus in CALCULUS_COMMANDS.items():
            self.commands[(menus.CALCULUS, word)] = partial(
                self._command_calculus, calculus
            )
        # Likewise the language saves, parametrized by language.
        for language in LANGUAGES:
            self.commands[(menus.TRANSFER_SAVE, language.word)] = partial(
                self._command_save_source, language
            )
        # Likewise the demos, parametrized by the catalogue entry.
        for demonstration in catalogue.DEMOS:
            self.commands[(menus.TRANSFER_DEMO, demonstration.word)] = partial(
                self._command_demonstration, demonstration
            )

    # -- composition -------------------------------------------------------

    def get_css_variables(self) -> dict[str, str]:
        return {**super().get_css_variables(), **self.palette.css_variables()}

    def get_driver_class(self) -> Any:
        """Textual's driver, replaced by one that handles terminal hangup where possible.

        Only Textual's Linux/macOS driver is replaced, by its subclass; a driver named in
        the environment, and the Windows driver, are left alone. `driver` is imported
        here because it imports the Linux driver, which fails on Windows.
        """
        chosen = super().get_driver_class()
        if sys.platform == "win32":
            return chosen
        from textual.drivers.linux_driver import LinuxDriver

        from rederive.ui.driver import TerminalDriver

        return TerminalDriver if chosen is LinuxDriver else chosen

    @property
    def session(self) -> Session:
        """The active window's worksheet, which commands act on."""
        return self.windows.session

    @property
    def work_area(self) -> WorkArea:
        """The active window's pane."""
        return self.panes[self.windows.active]

    def compose(self) -> ComposeResult:
        yield Panes(Frame(id="frame"), id="panes")
        # Between the work area, whose rows it borrows, and the input line it belongs to.
        yield CompletionList(id="completions")
        yield MenuRule(id="rule")
        yield MenuBand(id="menu")
        yield FieldBand(id="fields")
        yield Horizontal(
            Static(AUTHOR_PROMPT, id="prompt-label"),
            # The prompt chooses its own selection: the label number without the `#`.
            PromptLine(id="prompt-input", select_on_focus=False),
            id="prompt-band",
        )
        yield MessageLine(id="message")
        yield StatusLine(id="status")

    def on_unmount(self) -> None:
        """Shut down the plot host on exit.

        The host is a daemon child and would exit anyway, but this closes its windows
        cleanly.
        """
        self.plots.shutdown()

    def on_mount(self) -> None:
        self.query_one("#prompt-band").display = False
        self.query_one("#fields").display = False
        self.query_one("#completions").display = False
        self.refresh_screen()
        # Start the demo after the first render, so its first step has a screen.
        if self.opening_demo is not None:
            demo, self.opening_demo = self.opening_demo, None
            self._demo(str(demo))

    # -- top of the stack --------------------------------------------------

    @property
    def top(self) -> MenuCursor | DialogEditor:
        return self.stack[-1]

    @property
    def editor(self) -> DialogEditor | None:
        """The Options dialog on screen, if any."""
        return self.top if isinstance(self.top, DialogEditor) else None

    # -- rendering ---------------------------------------------------------

    def place_windows(self) -> None:
        """Give every window a pane, position it, and paint it.

        Panes are created on demand and kept for reuse when a window closes.

        As in Derive, every window shows its own highlight, active or not; the active
        window is marked by its number in the frame.

        A resize can arrive while a command is computing and owns the worksheet. Then
        the panes are only repositioned, not repainted.
        """
        panes = self.query_one(Panes)
        height, width = panes.size.height, panes.size.width
        areas = self.windows.areas(height, width)
        framed = self.windows.framed
        painting = self.mode != MODE_COMPUTE
        if painting and not self._opening_screen():
            self.greeting = False
        for window in self.windows.windows:
            pane = self.panes.get(window)
            if pane is None:
                pane = self.spare.pop() if self.spare else WorkArea(classes="work")
                pane.reset()
                self.panes[window] = pane
                if pane.parent is None:
                    panes.mount(pane)
            pane.display = True
            rect = windows.interior(areas[window], framed)
            pane.styles.offset = (rect.left, rect.top)
            pane.styles.width = rect.width
            pane.styles.height = rect.height
            if not painting:
                continue
            if self.helping is not None and window is self.windows.active:
                # Help appears only in the window it was opened from.
                page, titled = self._help_page(rect.height, rect.width)
                pane.show_help(page, rect.height, rect.width, titled)
            elif self.greeting:
                # There is only one window while the notice is shown.
                pane.show_greeting(
                    GREETING, self.GREETING_LINE, rect.height, rect.width
                )
            else:
                session = window.session
                pane.show(session.entries, session.selected, session.selection_rect())
        for window in [window for window in self.panes if window not in areas]:
            spare = self.panes.pop(window)
            spare.display = False
            self.spare.append(spare)
        self.query_one(Frame).refresh()
        self.query_one(MenuRule).refresh()

    def _opening_screen(self) -> bool:
        """Whether the screen is still as the program started, for the opening notice.

        True if there is one session, the initial one, it is empty, and no help is
        shown. Everything in Derive that replaces the notice (an expression, help, a
        split or overlay) breaks one of these, except the Clear commands, which clear
        the notice themselves.
        """
        sessions = self.windows.sessions()
        return (
            self.helping is None
            and len(sessions) == 1
            and sessions[0] is self.opening_session
            and not sessions[0].entries
        )

    def refresh_screen(self) -> None:
        """Push the whole model state at the widgets."""
        self.place_windows()
        editor = self.editor
        self.query_one("#menu").display = (
            editor is None and self.mode not in PROMPT_MODES
        )
        self.query_one("#fields").display = editor is not None
        if editor is not None:
            self.query_one(FieldBand).show(editor)
            # Unless a yes/no question is asked over the dialog, which takes the message
            # line.
            if self.mode != MODE_CONFIRM:
                self.message = editor.message
        elif self.mode == MODE_DEMO:
            # A demo shows the comment for the step just run in place of the menu.
            self.query_one(MenuBand).say(self._demo_comment())
        else:
            cursor = self.top
            assert isinstance(cursor, MenuCursor)
            # A yes/no question removes the menu highlight.
            highlighted = None if self.mode == MODE_CONFIRM else cursor.index
            # Update the pointer mark too, since whether words respond to clicks depends
            # on the mode, which may have just changed. The app tracks the pointer
            # position, since a stationary mouse sends no events.
            self._point(self.mouse_position)
            self.query_one(MenuBand).show(cursor.menu, highlighted)
        self.query_one(MessageLine).show(self.message)
        # The highlighted entry's annotation, and the name (not path) of the file the
        # session last read or wrote; the file prompt offers the full path.
        entry = self.session.selected_entry
        file = self.session.file
        self.query_one(StatusLine).show(
            "" if entry is None else entry.annotation,
            "" if file is None else file.name,
            self._flags(),
            f"{self.TITLE} {self.windows.kind}",
        )

    def _flags(self) -> str:
        """The mode indicators for the status line, where Derive shows them.

        Both only apply while a line is being typed. `Ovr` marks overwrite mode (Derive
        marked insert instead, but insert is the default here). `Lin` marks that the
        left/right keys go to the line, shown only on expression lines where they could
        go to the highlight.
        """
        if self.mode not in PROMPT_MODES:
            return ""
        words = [] if self.inserting else ["Ovr"]
        if self.line_edit and self.mode in EXPRESSION_LINES:
            words.append("Lin")
        return " ".join(words)

    def _show_flags(self) -> None:
        """Update the mode indicators without refreshing the rest of the screen."""
        self.query_one(StatusLine).indicate(self._flags())

    def _set_message(self, message: str) -> None:
        self.message = message
        self.query_one(MessageLine).show(message)

    def _settings_changed(self, changed: frozenset[str]) -> None:
        """React to settings changes that affect the screen or the plot host.

        A color change repaints everything. Display settings don't, since as in Derive
        an entry keeps the render it was authored with.

        Plot preferences go to the proxy, which forwards them with the next request,
        since the host may not be running. This covers both a loaded state file and a
        change reported by the host.
        """
        if changed & COLOR_SETTINGS:
            self.refresh_css()
            self.refresh_screen()
        if changed & PLOT_PREFERENCES:
            self.plots.prefer(preferences(self.settings))

    def _beep(self) -> None:
        """Beep for an invalid key, unless muted."""
        if self.settings["Mute"] == "No":
            self.bell()

    # -- computing ---------------------------------------------------------
    #
    # Every command that reaches the math engine goes through here. Where the engine
    # can be aborted, the screen is modal while it runs: the command owns the session,
    # so the worksheet isn't read or repainted until the completion handler runs. Only
    # the elapsed time and the memory gauge keep updating.

    @property
    def _abortable(self) -> bool:
        """Whether the session's engine can be aborted."""
        return hasattr(self.session.runner, "abort")

    def _compute(
        self,
        label: str,
        run: Callable[[], Awaitable[object]],
        done: Callable[[Outcome], None],
    ) -> None:
        """Await a session command on its own task, then pass the outcome to `done`.

        This works for every engine; how the call waits (on the desktop, a thread
        blocking on a pipe) is the engine's business. The task leaves the event loop
        free to repaint and handle Esc.

        The modal state is only used for abortable engines. An in-process engine can't
        be aborted or watched, so it gets the task without the mode, clock or Esc.
        """
        if self._abortable:
            self.resumed = self.mode
            self.mode = MODE_COMPUTE
            self.computing = label
            self.started = time.monotonic()
            # The line stays visible but loses focus, so all keys reach the app, which
            # ignores all but Esc.
            self.set_focus(None)
            # The clock appears only if the command is still running after a second.
            self.ticker = self.set_interval(TICK, self._tick)
        self.run_worker(self._computing(run, done))

    async def _computing(
        self, run: Callable[[], Awaitable[object]], done: Callable[[Outcome], None]
    ) -> None:
        """Wait for the answer on the worker task, then finish the command."""
        outcome = await _ran(run)
        self._computed(done, outcome)

    def _computed(self, done: Callable[[Outcome], None], outcome: Outcome) -> None:
        """Stop the clock and let the command finish.

        The mode is restored before the handler runs, since a handler that leaves its
        prompt up (for an unparsable line) expects that prompt's mode. `computing` says
        whether the command was modal. Focus isn't restored here: Textual applies focus
        on the next tick, and if the handler hides the prompt, a hidden line would keep
        the keyboard.
        """
        if self.ticker is not None:
            self.ticker.stop()
            self.ticker = None
        if self.computing is not None:
            self.computing = None
            self.mode = self.resumed
        done(outcome)

    def _tick(self) -> None:
        """Update the elapsed time on the message line.

        Nothing is shown before `CLOCK_AFTER`, and the line is only repainted when the
        whole-second figure changes.
        """
        if self.computing is None:
            return
        running = time.monotonic() - self.started
        if running < CLOCK_AFTER:
            return
        said = COMPUTING.format(command=self.computing, elapsed=_whole(running))
        if said != self.message:
            self._set_message(said)

    def _abort(self) -> None:
        """Esc during a computation: abort the engine."""
        abort = getattr(self.session.runner, "abort", None)
        if abort is not None:
            abort()

    # -- key routing -------------------------------------------------------

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Enable menu and navigation keys only while a command band is up.

        While a prompt line is up, the keys belong to the Input, including Space and
        Backspace. Exceptions:

        - History movements keep moving the highlight, since a line has no vertical
          movement, except on the Factor and Expand variable line. On expression lines,
          subexpression arrow mode sends left/right to the highlight as well, leaving
          the line no cursor movement.
        - Ctrl-Enter applies wherever Enter does: menus, dialogs and prompt lines.
        - Tab completes on file prompts and steps the menu elsewhere.
        - F3, F4 and F6 apply only on expression lines. The Alt keys, Ctrl-V and the
          insert toggle apply on every line.
        - Ctrl-C copies the highlighted expression when nothing is selected, so it also
          works at the menu, but only when the worksheet is shown (not in help or a
          demo). Ctrl-X applies wherever Ctrl-C does, so it can report that there's
          nothing to cut.
        - Completion list keys apply only while the list is open; otherwise Up and Down
          move through the history as on other lines.

        During a computation no bindings apply; Esc is handled with the unbound keys.
        """
        if self.mode == MODE_COMPUTE:
            return False
        if action == "complete_file":
            return self.mode == MODE_FILE
        if action.startswith("browse_"):
            return self.browsing
        if action.startswith("menu_"):
            return self.mode in MENU_MODES
        if action == "scroll_work":
            return self.mode == MODE_MENU
        if action == "scroll_work_typed":
            # Like the Ctrl keys, except that a text field takes the characters.
            return self.mode == MODE_MENU and self.editor is None
        if action == "help":
            # As in Derive, F1 is help on a line and the next window elsewhere. Help
            # can't be opened from within help.
            return self.mode in PROMPT_MODES
        if action.startswith("window_"):
            # Only at the command menu, not even in a Window submenu, so windows never
            # switch under a half-answered question.
            return self.mode == MODE_MENU and len(self.stack) == 1
        if action == "nav":
            if self.mode == MODE_MENU:
                return True
            if self.mode == MODE_HELP:
                # History keys turn the pages of a subject; the subject menu has none.
                return (
                    self.helping is not None
                    and self.helping.topic is not None
                    and parameters[0] in HELP_PAGING
                )
            if self.mode not in WALKED_MODES:
                return False
            if parameters[0] in ENTRY_MOVES:
                return True
            return self.mode in EXPRESSION_LINES and not self.line_edit
        if action == "enter_and_simplify":
            return self.mode == MODE_MENU or self.mode in PROMPT_MODES
        if action == "insert_highlighted":
            return self.mode in EXPRESSION_LINES
        if action == "toggle_arrow_mode":
            return self.mode in EXPRESSION_LINES
        if action in ("copy_highlighted", "cut_selected"):
            return self.mode == MODE_MENU or self.mode in PROMPT_MODES
        if action in ("insert_glyph", "toggle_inserting", "paste_clipboard"):
            return self.mode in PROMPT_MODES
        return True

    def on_key(self, event: Any) -> None:
        """Handle unbound keys: mnemonics, digits, and yes/no answers.

        A text field also takes other printable characters, since interval bounds use
        `-`, `.` and `/`.

        During a computation only Esc (abort) works. Other keys are discarded rather
        than queued, so keys pressed at a seemingly frozen screen don't all fire when
        the answer arrives.
        """
        if self.mode == MODE_COMPUTE:
            event.stop()
            event.prevent_default()
            if event.key == "escape":
                self._abort()
        elif self.mode in MENU_MODES:
            character = event.character
            if character and (
                character.isalnum()
                or self._typing_text(character)
                or self._menu_takes(character)
            ):
                event.stop()
                event.prevent_default()
                self._typed(character)
        elif self.mode == MODE_CONFIRM:
            event.stop()
            event.prevent_default()
            self._answer_confirm(bool(event.character) and event.character.lower() == "y")
        elif self.mode == MODE_DEMO:
            # Any key steps the demo; Esc pauses it.
            event.stop()
            event.prevent_default()
            if event.key == "escape":
                self._suspend_demo()
            else:
                self._demo_step()
        elif self.mode in PROMPT_MODES and event.key == "escape":
            event.stop()
            event.prevent_default()
            # Esc first closes an open completion list, leaving the chosen name on the
            # line; only the next Esc abandons the command.
            if self.browsing:
                self._close_list()
                self._set_message(ENTER_FILE)
            else:
                self._end_prompt(done=False)

    def _typing_text(self, character: str) -> bool:
        """Whether the active field takes `character` as text."""
        editor = self.editor
        return (
            character.isprintable()
            and editor is not None
            and isinstance(editor.field, TextField)
        )

    def _menu_takes(self, character: str) -> bool:
        """Whether the top menu has a word that `character` invokes.

        Letters and digits are always passed on; this is for other characters, which
        only the Build operator menu uses as mnemonics, such as `+` and `!`.
        """
        cursor = self.top
        if not isinstance(cursor, MenuCursor):
            return False
        return character.lower() in cursor.menu.mnemonics

    def _typed(self, character: str) -> None:
        """A letter or digit while a menu or a dialog is up."""
        editor = self.editor
        if editor is None:
            cursor = self.top
            assert isinstance(cursor, MenuCursor)
            index = cursor.menu.mnemonics.get(character.lower())
            if index is None:
                self._beep()
            else:
                self.invoke_command(index)
            return
        if editor.type_character(character):
            self.refresh_screen()
            return
        field = editor.field
        choice = (
            menus.pick(field.choices, character.lower())
            if isinstance(field, ChoiceField)
            else None
        )
        if choice is None:
            self._beep()
            return
        editor.choose(choice)
        if editor.settles_on_choice:
            self._commit()
        else:
            self.refresh_screen()

    # -- mouse routing -----------------------------------------------------
    #
    # Each mouse gesture ends in the same handler as the equivalent key: clicking a
    # menu word invokes it as its letter would, clicking an expression moves the
    # highlight, and clicking a dialog value sets it as typing its letter would. The
    # mouse can't reach anything the keyboard can't, and gestures apply only where
    # their keys do; for example, a click can't switch windows under a half-answered
    # question any more than F1 can.
    #
    # Only the pointer position is used. The command bands lay out for the actual
    # width, so each band is asked what it drew at a given cell.
    #
    # `ALLOW_SELECT` is off so that drag-selecting screen text is left to the terminal
    # instead of Textual.

    def on_click(self, event: events.Click) -> None:
        """Route a click to whatever it points at.

        Clicks are ignored during a computation, and in a demo they step it like any
        key.

        When a demo's plot window is closed, most desktops pass the closing click
        through to this window. That click is dropped so that closing a plot doesn't
        step the demo and open another one.
        """
        if self.mode == MODE_COMPUTE:
            return
        if self.mode == MODE_DEMO:
            if self.closed_window:
                self.closed_window = False
                return
            self._demo_step()
            return
        widget = self._under(event)
        if isinstance(widget, CompletionList):
            self._clicked_name(widget, event)
        elif isinstance(widget, MenuBand):
            self._clicked_word(widget, event)
        elif isinstance(widget, FieldBand):
            self._clicked_field(widget, event)
        else:
            pane = _pane_of(widget)
            if pane is not None:
                self._clicked_entry(pane, event)

    def on_mouse_move(self, event: events.MouseMove) -> None:
        """Mark the menu word under the pointer.

        The only gesture with no key equivalent; it just shows that words can be
        clicked, and only marks words that a click would invoke.
        """
        self._point(event.screen_offset)

    def _point(self, at: Offset | None) -> None:
        """Tell the menu band where the pointer is, if words are clickable."""
        self.query_one(MenuBand).point_at(at if self.picking else None)

    @property
    def picking(self) -> bool:
        """Whether a click on a menu word would invoke it.

        Not while a dialog or yes/no question occupies the band, during a computation,
        or in a demo.
        """
        return self.mode in MENU_MODES and self.editor is None

    @property
    def wants_typing(self) -> bool:
        """Whether the screen is waiting for typed input.

        True for a prompt line, a yes/no question, or a typed dialog field. Menus and
        dialog choices can be clicked, so they don't count. A touch screen uses this to
        decide whether to show its on-screen keyboard.
        """
        if isinstance(self.focused, PromptLine) or self.mode == MODE_CONFIRM:
            return True
        editor = self.editor
        return self.mode == MODE_MENU and editor is not None and editor.text is not None

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        self._wheel(1, event)

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        self._wheel(-1, event)

    def on_mouse_scroll_right(self, event: events.MouseScrollRight) -> None:
        self._wheel_across(1, event)

    def on_mouse_scroll_left(self, event: events.MouseScrollLeft) -> None:
        self._wheel_across(-1, event)

    def _under(self, event: events.MouseEvent) -> Widget | None:
        """The widget under the pointer, whatever the event was delivered to.

        Uses the pointer position, since wheel events don't say what they landed on.
        """
        try:
            widget, _ = self.get_widget_at(*event.screen_offset)
        except NoWidget:
            return None
        return widget

    def _clicked_word(self, band: MenuBand, event: events.Click) -> None:
        """Invoke the clicked menu word, as its letter would.

        Like a mnemonic, it leaves the highlight in place. Only applies when `picking`.
        """
        if not self.picking:
            return
        index = band.word_at(event.screen_offset)
        if index is not None:
            self.invoke_command(index)

    def _clicked_field(self, band: FieldBand, event: events.Click) -> None:
        """Handle a click on an Options dialog field or value.

        Clicking a value sets it as typing its letter would, which commits a dialog that
        settles on a choice. Clicking elsewhere on a field activates it, as Tab would,
        with the cursor at its start.
        """
        editor = self.editor
        if editor is None or self.mode != MODE_MENU:
            return
        found = band.field_at(event.screen_offset)
        if found is None:
            return
        index, choice = found
        if not editor.activate(index):
            self._beep()
        elif choice is not None and editor.choose(choice) and editor.settles_on_choice:
            self._commit()
            return
        self.refresh_screen()

    def _clicked_name(self, listing: CompletionList, event: events.Click) -> None:
        """Highlight and take the clicked name, as Enter on it would.

        A directory is entered with the list kept open, and a file is put on the line
        for the next Enter to open, so a click never opens a file.
        """
        if not self.browsing:
            return
        at = listing.name_at(event.screen_offset)
        if at is None or at >= len(self.completions):
            return
        self._highlight(at)
        self.action_browse_take()

    def _clicked_entry(self, pane: WorkArea, event: events.Click) -> None:
        """Select what was clicked in a window's history.

        A click in an inactive window activates it, like Window Goto, so it only works
        at the command menu with nothing half-answered.

        In the active window, the click moves the highlight wherever history keys apply:
        under every prompt line except the Factor and Expand variable line, and under
        dialogs whose fields name expressions. The line or field then takes the new
        label, as with the arrow keys.

        Unlike the arrows, a click reaches subexpressions regardless of the arrow key
        mode, which only resolves a key conflict the mouse doesn't have.
        """
        window = next((each for each in self.panes if self.panes[each] is pane), None)
        if window is None or self.mode == MODE_HELP:
            return
        editor = self.editor
        if window is not self.windows.active:
            if self.mode != MODE_MENU or len(self.stack) > 1:
                return
            self.windows.goto(self.windows.windows.index(window) + 1)
        elif self.mode == MODE_MENU:
            if editor is not None and not editor.dialog.tracks_selection:
                return
        elif self.mode not in WALKED_MODES:
            return
        pointed = pane.pointed(event.screen_offset)
        if pointed is not None:
            self.session.select_at(*pointed)
        if editor is not None:
            entry = self.session.selected_entry
            if entry is not None:
                editor.retype(str(entry.number))
        elif self.mode in LABELLED_MODES:
            self._relabel_prompt()
        self.refresh_screen()

    def _wheel(self, direction: int, event: events.MouseEvent) -> None:
        """Handle the wheel over whatever it points at.

        Textual already scrolls a history pane, leaving the highlight in place. This
        handles what doesn't scroll itself: the completion list, help pages, and
        sideways scrolling, which repaints one entry rather than scrolling the pane.
        """
        if self.mode == MODE_COMPUTE:
            return
        if event.shift or event.ctrl:
            self._wheel_across(direction, event)
            return
        widget = self._under(event)
        if isinstance(widget, CompletionList):
            if self.browsing:
                # Like Down and Up.
                if direction > 0:
                    self.action_browse_next()
                else:
                    self.action_browse_previous()
            return
        if _pane_of(widget) is not self.work_area or self.helping is None:
            return
        if self.helping.topic is not None:
            # Turn the subject's pages, like the history keys.
            self._turn_help(direction)

    def _wheel_across(self, direction: int, event: events.MouseEvent) -> None:
        """Scroll the pane under the pointer sideways.

        As with Ctrl-Right and Ctrl-Left, the selection and active window don't change.
        """
        if self.mode == MODE_COMPUTE:
            return
        pane = _pane_of(self._under(event))
        if pane is None:
            return
        pane.scroll_across(direction)
        self.refresh_screen()

    # -- menu actions ------------------------------------------------------

    def action_menu_next(self) -> None:
        editor = self.editor
        if editor is None:
            self._move_highlight(1)
        elif not editor.next_field():
            self._beep()
        self.refresh_screen()

    def action_menu_previous(self) -> None:
        editor = self.editor
        if editor is None:
            self._move_highlight(-1)
        elif not editor.previous_field():
            self._beep()
        self.refresh_screen()

    def action_menu_space(self) -> None:
        """Space steps the highlight, or the active field's value.

        On a text field it types a space, which is how a bound is blanked out.
        """
        editor = self.editor
        if editor is None:
            self._move_highlight(1)
        elif editor.lists_choices():
            self.stack.append(MenuCursor(COLORS, _color_index(editor)))
        elif isinstance(editor.field, TextField):
            editor.type_character(" ")
        elif not editor.cycle():
            self._beep()
        self.refresh_screen()

    def action_menu_erase(self) -> None:
        """Backspace reverses Space, or steps back.

        On a text field it erases the character before the cursor, and on a choice field
        it steps back through the choices rather than leaving the field.
        """
        editor = self.editor
        if editor is not None and (editor.erase() or editor.cycle(-1)):
            self.refresh_screen()
        else:
            self.action_menu_previous()

    def action_menu_delete(self) -> None:
        editor = self.editor
        if editor is not None and editor.delete():
            self.refresh_screen()
        else:
            self._beep()

    def action_menu_invoke(self) -> None:
        if self.editor is None:
            cursor = self.top
            assert isinstance(cursor, MenuCursor)
            self.invoke_command(cursor.index)
        else:
            self._commit()

    def action_enter_and_simplify(self) -> None:
        """Ctrl-Enter: Enter, then simplify the entered expression.

        The manual documents it only for vectors and matrices, but Derive accepts it on
        every line that enters an expression. Elsewhere it acts as Enter.
        """
        if self.mode in ENTERING_MODES or self._enters_expression():
            self.simplifying = tuple(self.session.entries)
        if self.mode == MODE_MENU:
            self.action_menu_invoke()
        else:
            self._submitted(self.query_one("#prompt-input", Input).value)

    def _enters_expression(self) -> bool:
        """Whether Enter on the current menu or dialog appends an unsimplified expression.

        True for `Done` on Build's operator menu and for each Calculus command's final
        dialog, where Ctrl-Enter is useful outside a prompt line.
        """
        if self.mode != MODE_MENU:
            return False
        top = self.top
        if isinstance(top, DialogEditor):
            return top.dialog.enters
        return top.menu is menus.BUILD_OPERATOR and top.word == building.DONE

    def action_menu_escape(self) -> None:
        """Pop the submenu or dialog on top, discarding its changes.

        Popping a menu that asks a question also abandons its command, since there is
        no earlier question to return to: the amount is Factor and Expand's last
        question, and the Declare Variable questions replace each other, so Esc returns
        to the Declare menu as in Derive.

        In help, Esc is `Resume`: from a subject to the subject menu, and from there
        out of help. On the command menu, with nothing to pop, it moves the highlight
        back to the first word.
        """
        if self.mode == MODE_HELP:
            self._help_back()
            return
        if len(self.stack) == 1:
            self._restart_menu()
            self.refresh_screen()
        else:
            left = self.stack.pop()
            if isinstance(left, MenuCursor) and left.menu in AMOUNT_MENUS:
                self.asking = None
            self.building = None
            self.calculating = None
            self.solving = None
            self.declaring = None
            self._restart_menu()
            self._ask_again()
            self.refresh_screen()

    def _ask_again(self) -> None:
        """Restore the uncovered menu's prompt on the message line.

        Dialogs do this themselves, since their prompt depends on the active field.
        """
        if isinstance(self.top, MenuCursor):
            self.message = self.top.message

    def _restart_menu(self) -> None:
        """Highlight the first word of the menu just uncovered.

        Derive redraws a menu each time it appears, so returning from a command or
        submenu always highlights the first word (Author on the command menu).
        """
        cursor = self.top
        if isinstance(cursor, MenuCursor):
            cursor.index = 0

    def action_nav(self, movement: str) -> None:
        """Arrow keys move the highlight, or a number field's cursor.

        In help they turn pages. With a prompt line up, only history movements reach
        here; `check_action` leaves the rest to the line. Choice fields have no cursor,
        so as in Derive they ignore arrows.

        In dialogs whose fields name expressions, history keys move the highlight and
        the field takes its label, which the manual recommends over typing the number.
        """
        if self.mode == MODE_HELP:
            self._turn_help(HELP_PAGING[movement])
            return
        if self.mode in WALKED_MODES:
            self._walk_prompt(movement)
            return
        editor = self.editor
        if editor is None:
            self._move_selection(movement)
        elif editor.dialog.tracks_selection and movement in ENTRY_MOVES:
            self._walk_history(editor, movement)
        elif not editor.move_cursor(CURSOR_MOVES.get(movement)):
            self._beep()
        self.refresh_screen()

    def _walk_prompt(self, movement: str) -> None:
        """Move the highlight while a prompt line is up.

        The line keeps its contents, so the history can be browsed mid-command; the
        manual recommends naming an expression by highlighting it.
        """
        self._move_selection(movement)
        if self.mode in LABELLED_MODES:
            self._relabel_prompt()
        self.refresh_screen()

    def _relabel_prompt(self) -> None:
        """Replace the label on the prompt line with the highlighted entry's.

        Only while the line is as the command prefilled it. Moving the cursor, typing,
        or selecting text makes the line the user's, and it is then left alone.
        """
        entry = self.session.selected_entry
        line = self.query_one("#prompt-input", Input)
        if entry is None or self.offering != (line.value, line.selection):
            return
        start, end = sorted(line.selection)
        number = str(entry.number)
        line.value = f"{line.value[:start]}{number}{line.value[end:]}"
        line.selection = Selection(start, start + len(number))
        # The relabeled line still counts as the command's offering.
        self.offering = (line.value, line.selection)

    def _selected_on_line(self) -> str | None:
        """The text the user selected on the prompt line, or None.

        None if no line is up, nothing is selected, or the selection is the command's
        untouched prefilled offering.
        """
        if self.mode not in PROMPT_MODES:
            return None
        line = self.query_one("#prompt-input", PromptLine)
        if line.selection.is_empty or self.offering == (line.value, line.selection):
            return None
        return line.selected_text

    def copy_to_clipboard(self, text: str) -> None:
        """Copy text to the system clipboard, also keeping the copy Ctrl-V falls back on.

        Two mechanisms are used. The base class writes an escape sequence asking the
        terminal to copy the text; that's the only option over ssh, but it reports
        nothing back, and VTE-based terminals ignore it. The platform layer copies via
        the desktop clipboard where there is one. Headless runs (tests) skip the
        platform layer so as not to touch the real clipboard.

        `_copy_refused` reports when neither mechanism works. That happens only in a
        browser, where the escape sequence goes nowhere and clipboard access needs the
        user's permission.
        """
        super().copy_to_clipboard(text)
        if self.is_headless:
            return
        platform.current().copy(text, self._copy_refused)

    def _copy_refused(self, why: str) -> None:
        """Report that the copy failed.

        This arrives asynchronously (a browser clipboard write returns a promise), so it
        overwrites the earlier confirmation on the message line, as it should.
        """
        self._set_message(why)

    def action_copy_highlighted(self) -> None:
        """Ctrl-C: copy the selected text on the line, or the highlighted expression.

        Without a user selection, it copies the highlighted expression or subexpression
        as linear text, which can be typed back in. A label prefilled on the line
        doesn't count as a selection, so under a line one can move the highlight and
        copy.

        The text goes to the system clipboard (see `copy_to_clipboard`). Since that
        gives no feedback, the message line confirms the copy.
        """
        marked = self._selected_on_line()
        if marked is not None:
            self.copy_to_clipboard(marked)
            self._set_message(COPIED_TEXT)
            return
        text = self.session.highlighted_text
        if text is None:
            self._beep()
            return
        self.copy_to_clipboard(text)
        self._set_message(COPIED)

    def action_cut_selected(self) -> None:
        """Ctrl-X: cut the user's selection on the line.

        It applies only to the line. Without a user selection (a highlighted expression
        or a prefilled label don't count), the message line says there is nothing to
        cut.
        """
        text = self._selected_on_line()
        if text is None:
            self._beep()
            if self.mode in PROMPT_MODES:
                self._set_message(NOTHING_TO_CUT)
            return
        line = self.query_one("#prompt-input", PromptLine)
        self.copy_to_clipboard(text)
        line.delete_selection()
        self._set_message(CUT_TEXT)

    def action_paste_clipboard(self) -> None:
        """Ctrl-V: paste the clipboard onto the line.

        It inserts at the cursor, replacing any selection, including the `#` of a
        prefilled label. It reads the system clipboard where possible, and otherwise
        (and in headless runs) the internal copy.
        """
        text = None if self.is_headless else platform.current().paste()
        if not text:
            text = self.clipboard
        if not text:
            self._beep()
            return
        self._paste_on_line(text)

    def on_prompt_line_pasted(self, event: PromptLine.Pasted) -> None:
        """Handle a paste from the terminal's key or the browser's Ctrl-V, like Ctrl-V."""
        if self.mode in PROMPT_MODES and event.text:
            self._paste_on_line(event.text)

    def _paste_on_line(self, text: str) -> None:
        """Write the first line of `text` onto the line, replacing a prefilled label."""
        lines = text.splitlines()
        if not lines or not lines[0]:
            return
        self._take_offered_label()
        self._write_on_line(lines[0])

    def action_insert_highlighted(self, fenced: int) -> None:
        """F3 and F4: insert the highlighted expression onto the line.

        F4 wraps it in parentheses, so that, say, `x + 1` next to a factor stays one
        term. It inserts at the cursor, replacing any selection, including the `#` of a
        prefilled label.
        """
        text = self.session.highlighted_text
        if text is None:
            self._beep()
            return
        if fenced:
            text = f"({text})"
        self._take_offered_label()
        self._write_on_line(text)

    def _take_offered_label(self) -> None:
        """Extend the selection over the `#` of a prefilled label, if still selected."""
        if self.mode not in LABELLED_MODES:
            return
        line = self.query_one("#prompt-input", PromptLine)
        start, end = sorted(line.selection)
        if start == 1 and end == len(line.value) and line.value.startswith("#"):
            line.selection = Selection(0, end)

    def action_insert_glyph(self, key: str) -> None:
        """An Alt key: insert its glyph onto the line."""
        self._write_on_line(ALT_GLYPHS[key])

    def _write_on_line(self, text: str) -> None:
        """Insert `text` at the cursor, replacing any selection."""
        line = self.query_one("#prompt-input", PromptLine)
        line.replace(text, *sorted(line.selection))

    def action_toggle_arrow_mode(self) -> None:
        """F6: toggle whether left/right keys go to the line or the highlight.

        Each line starts in the mode from `Options Input`; this changes it only for the
        current line.
        """
        self.line_edit = not self.line_edit
        self._show_flags()

    def action_toggle_inserting(self) -> None:
        """Ins: toggle insert and overwrite modes."""
        self.inserting = not self.inserting
        self.query_one("#prompt-input", PromptLine).overwrite = not self.inserting
        self._show_flags()

    def _move_selection(self, movement: str) -> bool:
        """Move the highlight, returning whether it moved.

        Page movements are sized by the pane height.
        """
        if movement in PAGE_MOVES:
            rows = max(1, self.work_area.size.height)
            return getattr(self.session, f"move_{movement}")(rows)
        return getattr(self.session, f"move_{movement}")()

    def _walk_history(self, editor: DialogEditor, movement: str) -> None:
        """Move the selection, and set the active field to the new entry's label."""
        if not self._move_selection(movement):
            self._beep()
        entry = self.session.selected_entry
        if entry is not None:
            editor.retype(str(entry.number))

    def action_scroll_work(self, direction: int) -> None:
        """Ctrl-Right and Ctrl-Left: scroll a selected render wider than the pane."""
        self.work_area.scroll_across(direction)
        self.refresh_screen()

    def action_scroll_work_typed(self, direction: int) -> None:
        """`>` and `<`, equivalent to the Ctrl keys."""
        self.action_scroll_work(direction)

    def _move_highlight(self, step: int) -> None:
        cursor = self.top
        assert isinstance(cursor, MenuCursor)
        cursor.move(step)

    def invoke_command(self, index: int) -> None:
        """Run the command at `index`, leaving the menu highlight alone."""
        cursor = self.top
        assert isinstance(cursor, MenuCursor)
        word = cursor.menu.words[index]
        # The color and amount menus answer a question, so they pick values, not
        # commands.
        if cursor.menu is COLORS:
            self._chose_color(index)
            return
        if cursor.menu in AMOUNT_MENUS:
            self._chose_amount(index)
            return
        if cursor.menu is menus.BUILD_OPERATOR:
            self._chose_operator(word)
            return
        # The help menus choose a subject and turn its pages.
        if cursor.menu is menus.HELP:
            self._chose_subject(word)
            return
        if cursor.menu in menus.HELP_TOPICS:
            self._read_help(word)
            return
        if cursor.menu is menus.DECLARE_VARIABLE:
            self._chose_domain(word)
            return
        if cursor.menu is menus.DECLARE_INTERVAL:
            self._chose_interval(word)
            return
        target = menus.TARGETS.get(cursor.menu, {}).get(word)
        if target is not None:
            self._open(target)
            return
        command = self.commands.get((cursor.menu, word))
        if command is None:
            self._set_message(f"{word}: not implemented yet")
        else:
            command()

    # -- Options -----------------------------------------------------------

    def _open(self, target: Menu | Dialog) -> None:
        """Push a submenu or an Options dialog."""
        if isinstance(target, Menu):
            self.stack.append(MenuCursor(target))
        else:
            self.stack.append(DialogEditor(target, self.settings))
        self._ask_again()
        self.refresh_screen()

    def _ask(
        self,
        dialog: Dialog,
        answer: Callable[[dict[str, str | int]], None],
        refuses: Callable[[dict[str, str | int]], str | None] = _takes_anything,
    ) -> None:
        """Push a dialog whose values go to `answer` instead of the settings.

        `refuses` returns the first field the command can't accept, for checks a field
        can't make itself, such as whether a label number names an existing entry.
        """
        self.answer = answer
        self.refuses = refuses
        self.asked = dialog
        self._open(dialog)

    def _answered(self, values: dict[str, str | int]) -> None:
        """Pass an unstored dialog's values to its command."""
        refused = self.refuses(values)
        if refused is not None:
            # As in Derive, a refused answer just beeps: the dialog stays up with its
            # contents, and the offending field becomes active.
            editor = self.editor
            assert editor is not None
            editor.focus(refused)
            self._beep()
            self.refresh_screen()
            return
        self.stack.pop()
        answer, self.answer = self.answer, None
        self.refuses = _takes_anything
        assert answer is not None
        answer(values)

    def _names_nothing(self, values: dict[str, str | int], *settings: str) -> str | None:
        """The first of `settings` whose label names no entry, or None.

        `end` is accepted, since it means the position after the last entry.
        """
        for setting in settings:
            value = values[setting]
            if value != menus.END and self.session.numbered(int(value)) is None:
                return setting
        return None

    def _chose_color(self, index: int) -> None:
        """Apply a color chosen from the color menu to the dialog beneath it."""
        self.stack.pop()
        editor = self.editor
        assert editor is not None
        editor.choose(menus.color_at(index))
        self.refresh_screen()

    def _commit(self) -> None:
        """Apply the dialog on top, and record what it changed."""
        editor = self.editor
        assert editor is not None
        asking = editor.prompt is not None
        values = editor.commit()
        if values is None:
            # Don't beep if Enter just opened the `Other` prompt.
            if asking or editor.prompt is None:
                self._beep()
            self.refresh_screen()
            return
        if not editor.dialog.stored:
            self._answered(values)
            return
        changed = editor.changes(self.settings)
        if editor.dialog.keeps_menu:
            self.stack.pop()
        else:
            del self.stack[1:]
        self._restart_menu()
        self.settings.apply(values)
        # Record after applying, so the record uses the new settings.
        for field in changed:
            if field.recorded:
                self.session.record(field.setting)
        self._return_to_menu(ENTER_OPTION)

    # -- commands ----------------------------------------------------------

    def _command_author(self) -> None:
        self._prompt(MODE_AUTHOR, AUTHOR_PROMPT, "", ENTER_EXPRESSION)

    def _command_simplify(self) -> None:
        """Ask which expression to simplify, offering the highlighted one."""
        self._ask_expression(MODE_SIMPLIFY, SIMPLIFY_PROMPT)

    def _command_approx(self) -> None:
        """Ask which expression to approximate, offering the highlighted one."""
        self._ask_expression(MODE_APPROX, APPROX_PROMPT)

    def _ask_expression(self, mode: str, label: str) -> None:
        """Show the expression line shared by Simplify and approX.

        It offers the highlighted expression's label. approX takes its precision from
        the settings, so the two commands differ only in what they run.
        """
        entry = self.session.selected_entry
        offered = "" if entry is None else f"#{entry.number}"
        self._prompt(mode, label, offered, ENTER_TO_DERIVE, keep=1)

    def _prompt(
        self, mode: str, label: str, offered: str, message: str, keep: int = 0
    ) -> None:
        """Show the prompt band for a command that reads a line.

        `offered` starts selected, so typing replaces it and Enter accepts it. `keep` is
        how many leading characters stay unselected, such as the `#` of a label. The
        selection is recorded as the offering, to tell it apart from a user selection.

        The arrow key mode is reset from the settings for each command; F6 changes it
        only for the current line.
        """
        self.mode = mode
        self.line_edit = self.settings["ArrowKeyMode"] == "LineEdit"
        self.query_one("#menu").display = False
        # Hide any dialog that asked the previous question (such as `Manage Annotate`'s
        # label or `Declare Matrix`'s size).
        self.query_one("#fields").display = False
        self.query_one("#prompt-band").display = True
        self.query_one("#prompt-label", Static).update(label)
        line = self.query_one("#prompt-input", PromptLine)
        # File prompts set their own suggester afterwards.
        line.suggester = None
        line.overwrite = not self.inserting
        self._close_list()
        line.value = offered
        line.selection = Selection(min(keep, len(offered)), len(offered))
        self.offering = (line.value, line.selection)
        line.focus()
        self._set_message(message)
        self._show_flags()

    # -- Help --------------------------------------------------------------
    #
    # Help replaces the expressions in the window it was opened from, with no border,
    # overlay or dimming. As in Derive it has two levels, a subject menu and a subject
    # read page by page, navigated with Next, Previous and Resume; there is no search
    # and no links.
    #
    # Help menus go on the regular menu stack, so leaving help uncovers whatever
    # opened it. The prompt line isn't on the stack, so its state is kept in `Helping`.
    #
    # F1 on a line opens the line editing subject directly, one level deep. Resume
    # returns to the line, and a `Subjects` word replaces the subject with the subject
    # menu.

    def _command_help(self) -> None:
        """The Help command: show the subject menu."""
        self._open_help(MODE_MENU, ENTER_OPTION)

    def action_help(self) -> None:
        """F1 on an input line: open the line editing subject, keeping the line.

        Resume returns straight to the line, and `Subjects` opens the subject menu.
        """
        self._open_help(self.mode, self.message, helps.EDITING)

    def _open_help(self, resume: str, message: str, word: str | None = None) -> None:
        """Show help, remembering what the screen was doing.

        A `word` opens that subject directly instead of the subject menu.
        """
        self.helping = Helping(resume, message)
        self.mode = MODE_HELP
        if word is None:
            self.stack.append(MenuCursor(menus.HELP))
        else:
            self.helping.topic = helps.BY_WORD[word]
            self.stack.append(MenuCursor(menus.HELP_PAGES_ALONE[word]))
        # The line is hidden with its text intact. The completion list is closed, since
        # help takes its rows and its current name is already on the line.
        self._close_list()
        self.query_one("#prompt-band").display = False
        self.set_focus(None)
        self.message = ENTER_OPTION
        self.refresh_screen()

    def _chose_subject(self, word: str) -> None:
        """Handle a choice from the subject menu: a subject, or Resume."""
        assert self.helping is not None
        if word == helps.RESUME:
            self._help_back()
            return
        self.helping.topic = helps.BY_WORD[word]
        self.helping.page = 0
        self.stack.append(MenuCursor(menus.HELP_PAGES[word]))
        self.refresh_screen()

    def _read_help(self, word: str) -> None:
        """Handle a word from a subject's page menu."""
        if word == helps.RESUME:
            self._help_back()
        elif word == helps.SUBJECTS:
            self._show_subjects()
        else:
            self._turn_help(1 if word == helps.NEXT else -1)

    def _show_subjects(self) -> None:
        """Subjects: replace the current subject with the subject menu.

        Replacing rather than pushing means one Resume still returns to the line that F1
        was pressed on. Subjects chosen from that menu are pushed on top of it as usual.
        """
        assert self.helping is not None
        self.helping.topic = None
        self.helping.page = 0
        self.stack[-1] = MenuCursor(menus.HELP)
        self.refresh_screen()

    def _turn_help(self, step: int) -> None:
        """Turn a page. Past the last page, leave the subject, as in Derive.

        Before the first page it stops rather than wrapping to the last.
        """
        assert self.helping is not None
        page = self.helping.page + step
        if page >= self.helping.pages:
            self._help_back()
            return
        self.helping.page = max(0, page)
        self.refresh_screen()

    def _help_back(self) -> None:
        """Resume: go up one level, or leave help from the top level.

        A subject opened from the subject menu returns to it; one opened directly by F1
        leaves help.
        """
        assert self.helping is not None
        self.stack.pop()
        if isinstance(self.top, MenuCursor) and self.top.menu is menus.HELP:
            self.helping.topic = None
            self.helping.page = 0
            self._restart_menu()
            self.refresh_screen()
            return
        resume, message = self.helping.resume, self.helping.message
        self.helping = None
        self.mode = resume
        if resume in PROMPT_MODES:
            # The line comes back exactly as it was left, cursor included.
            self.query_one("#prompt-band").display = True
            self.query_one("#prompt-input", Input).focus()
        else:
            self._restart_menu()
        self.message = message
        self.refresh_screen()

    def _help_page(self, rows: int, width: int) -> tuple[list[str], bool]:
        """The current page laid out for the pane size, and whether it has a title row.

        Pagination depends on the pane size, so it is redone on every paint, and the
        page number is clamped after a split or resize. The subject menu has no title
        row, since its heading is part of its centered layout.
        """
        assert self.helping is not None
        topic = self.helping.topic
        if topic is None:
            self.helping.pages = 1
            return helps.menu_page(rows, width), False
        pages = helps.pages(topic, self.settings, rows, width)
        self.helping.pages = len(pages)
        self.helping.page = min(self.helping.page, len(pages) - 1)
        return pages[self.helping.page], True

    # -- Jump --------------------------------------------------------------

    def _command_jump(self) -> None:
        """Ask which expression to highlight.

        Nothing is offered on the line, since the current label would be useless. With
        an empty history it just beeps, as in Derive.
        """
        if not self.session.entries:
            self._beep()
            return
        self._prompt(MODE_JUMP, JUMP_PROMPT, "", menus.ENTER_LABEL)

    def _jumped(self, answer: str) -> None:
        """Enter on the Jump line.

        An empty line leaves the highlight in place. Anything other than an existing
        label number beeps and leaves the line up for correction.
        """
        answer = answer.strip()
        if not answer:
            self._end_prompt()
            return
        if not LABEL_NUMBER.fullmatch(answer) or not self.session.jump(int(answer)):
            self._beep()
            return
        self._end_prompt()

    # -- Remove and Unremove -----------------------------------------------

    def _command_remove(self) -> None:
        """Ask which block of expressions to remove, offering the highlighted one.

        With an empty history it just beeps, as in Derive.
        """
        entry = self.session.selected_entry
        if entry is None:
            self._beep()
            return
        self._ask(menus.remove_block(entry.number), self._remove, self._both_labels)

    def _both_labels(self, values: dict[str, str | int]) -> str | None:
        return self._names_nothing(values, "RemoveFirst", "RemoveLast")

    def _remove(self, values: dict[str, str | int]) -> None:
        self.session.remove(int(values["RemoveFirst"]), int(values["RemoveLast"]))
        self._return_to_menu(ENTER_OPTION)

    def _command_unremove(self) -> None:
        """Restore the last removal, asking where if there is a choice.

        An empty buffer is reported on the message line. With an empty history there's
        no choice, so the expressions are restored directly.
        """
        if not self.session.removed:
            self._set_message(BUFFER_EMPTY)
            return
        entry = self.session.selected_entry
        if entry is None:
            self.session.unremove()
            self._return_to_menu(ENTER_OPTION)
            return
        self._ask(menus.unremove_before(entry.number), self._unremove, self._one_label)

    def _one_label(self, values: dict[str, str | int]) -> str | None:
        return self._names_nothing(values, "UnremoveBefore")

    def _unremove(self, values: dict[str, str | int]) -> None:
        before = values["UnremoveBefore"]
        self.session.unremove(None if before == menus.END else int(before))
        self._return_to_menu(ENTER_OPTION)

    # -- moVe --------------------------------------------------------------

    def _command_move(self) -> None:
        """Ask which block to move and where.

        With an empty history it just beeps, as in Derive.
        """
        entry = self.session.selected_entry
        if entry is None:
            self._beep()
            return
        self._ask(menus.move_block(entry.number), self._move, self._move_labels)

    def _move_labels(self, values: dict[str, str | int]) -> str | None:
        return self._names_nothing(values, "MoveBefore", "MoveFirst", "MoveLast")

    def _move(self, values: dict[str, str | int]) -> None:
        before = values["MoveBefore"]
        self.session.move_block(
            None if before == menus.END else int(before),
            int(values["MoveFirst"]),
            int(values["MoveLast"]),
        )
        self._return_to_menu(ENTER_OPTION)

    # -- Factor and Expand -------------------------------------------------

    def _command_asking(self, command: Command) -> None:
        """Ask which expression to work on, offering the highlighted one."""
        self.asking = Asking(command)
        entry = self.session.selected_entry
        offered = "" if entry is None else f"#{entry.number}"
        self._prompt(
            MODE_ASKING,
            EXPRESSION_PROMPT.format(word=command.word),
            offered,
            ENTER_TO_DERIVE,
            keep=1,
        )

    def _asked(self, request: str) -> None:
        """The expression is known; determine what else to ask.

        Variables are asked for only if there are at least two. The amount is asked for
        only where the command says it matters, so a number is factored and a sum
        expanded with no questions.

        The variables come from the engine, so this goes through the compute path, since
        converting a pathological expression can blow up here as in the command itself.
        Normally it takes well under a millisecond.
        """
        pending = self.asking
        assert pending is not None
        if not request.strip():
            self._end_prompt()
            return
        try:
            needs_amount = pending.command.needs_amount(self.session, request)
        except DeriveSyntaxError as error:
            self._refused(error)
            return
        pending.request = request
        self._compute(
            READING,
            partial(self.session.variables, request),
            partial(self._asked_variables, needs_amount),
        )

    def _asked_variables(self, needs_amount: bool, outcome: Outcome) -> None:
        """The variables are known: ask the next question, or run the command."""
        pending = self.asking
        assert pending is not None
        if isinstance(outcome.error, DeriveSyntaxError):
            self._refused(outcome.error)
            return
        if outcome.error is not None:
            self._end_prompt(_reported(outcome))
            return
        pending.remaining = outcome.value  # type: ignore[assignment]
        if len(pending.remaining) >= 2:
            self._ask_variable()
        elif needs_amount:
            self._ask_amount()
        else:
            self._answer()

    def _ask_variable(self) -> None:
        """Show the next variable question."""
        pending = self.asking
        assert pending is not None
        number = len(pending.chosen) + 1
        offer = FIRST_VARIABLE if number == 1 else NEXT_VARIABLE
        self._prompt(
            MODE_ASKING_VARIABLE,
            VARIABLE_PROMPT.format(word=pending.command.word, number=number),
            "",
            offer.format(variables=",".join(pending.remaining)),
        )

    def _asked_variable(self, answer: str) -> None:
        """One answer to a variable question.

        An empty line, or choosing the last remaining variable, ends the list. As in
        Derive, a name not on offer beeps and asks again.
        """
        pending = self.asking
        assert pending is not None
        answer = answer.strip()
        if not answer:
            self._variables_settled()
            return
        chosen = next(
            (name for name in pending.remaining if name.lower() == answer.lower()), None
        )
        if chosen is None:
            self._beep()
            self._ask_variable()
            return
        pending.chosen += (chosen,)
        pending.remaining = tuple(n for n in pending.remaining if n != chosen)
        if pending.remaining:
            self._ask_variable()
        else:
            self._variables_settled()

    def _variables_settled(self) -> None:
        """The variables are chosen: ask for the amount, or run the command."""
        pending = self.asking
        assert pending is not None
        if pending.command.needs_amount(self.session, pending.request):
            self._ask_amount()
        else:
            self._answer()

    def _ask_amount(self) -> None:
        """Hide the prompt band and push the amount menu."""
        pending = self.asking
        assert pending is not None
        self._hide_prompt()
        self.mode = MODE_MENU
        self.stack.append(self.amounts[pending.command.amounts])
        self.message = pending.command.amounts.message
        self.refresh_screen()

    def _chose_amount(self, index: int) -> None:
        """Choosing the amount, the last question, runs the command."""
        pending = self.asking
        assert pending is not None
        self.amounts[pending.command.amounts].index = index
        self.stack.pop()
        self._answer(pending.command.amounts.words[index])

    def _answer(self, amount: str | None = None) -> None:
        """Run the command and report how long it took."""
        pending = self.asking
        assert pending is not None
        self.asking = None
        self._compute(
            pending.command.running,
            partial(
                pending.command.run,
                self.session,
                pending.request,
                Amount(amount or "Rational"),
                pending.chosen,
            ),
            self._appended,
        )

    def _appended(self, outcome: Outcome) -> None:
        """Report the answer, or the reason there is none.

        Errors are reported rather than corrected, since the input was already parsed
        and the prompt may be gone.
        """
        self._end_prompt(_reported(outcome))

    # -- Build -------------------------------------------------------------

    def _command_build(self) -> None:
        """Ask for the first operand, offering the highlighted expression."""
        self.building = Building()
        self._ask_operand(MODE_BUILD, BUILD_PROMPT)

    def _ask_operand(self, mode: str, label: str) -> None:
        """Show an operand line, offering the highlighted expression's label.

        The arrow keys change the label to another expression, and F6 allows selecting
        a subexpression, so operands usually need no typing.
        """
        entry = self.session.selected_entry
        offered = "" if entry is None else f"#{entry.number}"
        self._prompt(mode, label, offered, ENTER_TO_DERIVE, keep=1)

    def _build_operand(self, text: str) -> None:
        """Resolve the first operand and ask for an operator."""
        pending = self.building
        assert pending is not None
        resolved = self._resolved(text)
        if resolved is None:
            return
        pending.node, pending.annotation = resolved
        self._ask_operator(0)

    def _build_next(self, text: str) -> None:
        """Resolve the right operand of the pending binary operator."""
        pending = self.building
        assert pending is not None
        operator = pending.operator
        assert operator is not None and pending.node is not None
        resolved = self._resolved(text)
        if resolved is None:
            return
        node, name = resolved
        pending.node = operator.build(pending.node, node)
        pending.annotation = operator.annotate(pending.annotation, name)
        pending.operator = None
        self._ask_operator(len(building.WORDS) - 1)

    def _resolved(self, text: str) -> tuple[Node, str] | None:
        """The expression an operand line names, and its name for the annotation.

        Returns None if an empty line abandoned the command, as on the author line, or
        if the line failed to parse and was left up for correction.
        """
        if not text.strip():
            self._end_prompt(done=False)
            return None
        try:
            return self.session.named_target(text)
        except DeriveSyntaxError as error:
            self._refused(error)
            return None

    def _ask_operator(self, index: int) -> None:
        """Hide the prompt band and show the operator menu.

        As in Derive, it opens on `+` the first time and on `Done` afterwards, the most
        likely next choice.
        """
        self._hide_prompt()
        self.mode = MODE_MENU
        cursor = self.top
        if isinstance(cursor, MenuCursor) and cursor.menu is menus.BUILD_OPERATOR:
            cursor.index = index
        else:
            self.stack.append(MenuCursor(menus.BUILD_OPERATOR, index))
        self.message = menus.BUILD_OPERATOR.message
        self.refresh_screen()

    def _chose_operator(self, word: str) -> None:
        """Apply an operator from the menu, or finish the expression.

        A unary operator is applied immediately, leaving the menu up. A binary one pops
        the menu and asks for its right operand, so Esc on that line abandons the
        command, as in Derive.
        """
        pending = self.building
        assert pending is not None and pending.node is not None
        operator = building.operator(word)
        if operator is None:
            self._built(pending)
            return
        if operator.arity == 1:
            pending.node = operator.build(pending.node)
            pending.annotation = operator.annotate(pending.annotation)
            self._ask_operator(len(building.WORDS) - 1)
            return
        pending.operator = operator
        self.stack.pop()
        self._ask_operand(MODE_BUILD_NEXT, BUILD_NEXT_PROMPT)

    def _built(self, pending: Building) -> None:
        """Done: append the built expression and return to the menu.

        Ctrl-Enter simplifies it, which needs the engine; plain Enter appends it as is.
        Both go through the compute path so the answer arrives in the usual handler.
        Errors are left to that handler, since every operand was already parsed.
        """
        assert pending.node is not None
        node, annotation = pending.node, pending.annotation
        self.building = None
        simplified = self._wanted_simplified()
        self._compute(
            SIMPLIFYING if simplified else BUILDING,
            partial(self.session.build, node, annotation, simplified),
            self._appended,
        )

    def _wanted_simplified(self) -> bool:
        """Whether the command was finished with Ctrl-Enter, clearing the mark.

        Build and Calculus don't use the general path, which appends and then
        simplifies. Derive enters only the simplified expression, with an annotation
        recording both steps, so the Ctrl-Enter mark is cleared here.
        """
        if self.simplifying is None:
            return False
        self.simplifying = None
        return True

    # -- Calculus ----------------------------------------------------------

    def _command_calculus(self, command: Calculus) -> None:
        """Ask which expression to work on, offering the highlighted one."""
        self.calculating = Calculating(command)
        entry = self.session.selected_entry
        offered = "" if entry is None else f"#{entry.number}"
        self._prompt(
            MODE_CALCULUS,
            CALCULUS_PROMPT.format(word=command.word),
            offered,
            ENTER_TO_DERIVE,
            keep=1,
        )

    def _calculus_expression(self, request: str) -> None:
        """The expression is known; find the variable to offer.

        As in Derive, this is the expression's primary variable, obtained from the
        engine through the compute path as in Factor. Without variables the line starts
        empty.
        """
        pending = self.calculating
        assert pending is not None
        if not request.strip():
            self._end_prompt(done=False)
            return
        try:
            self.session.reads(request)
        except DeriveSyntaxError as error:
            self._refused(error)
            return
        pending.request = request
        self._compute(
            READING,
            partial(self.session.variables, request),
            self._calculus_variables,
        )

    def _calculus_variables(self, outcome: Outcome) -> None:
        """The variables are known: ask which one, offering the primary one."""
        pending = self.calculating
        assert pending is not None
        if isinstance(outcome.error, DeriveSyntaxError):
            self._refused(outcome.error)
            return
        if outcome.error is not None:
            self._end_prompt(_reported(outcome))
            return
        variables: tuple[str, ...] = outcome.value  # type: ignore[assignment]
        self._prompt(
            MODE_CALCULUS_VARIABLE,
            CALCULUS_VARIABLE_PROMPT.format(word=pending.command.word),
            variables[0] if variables else "",
            ENTER_VARIABLE,
        )

    def _calculus_variable(self, text: str) -> None:
        """Accept the variable answer, which must be a variable.

        As in Derive, anything else is refused silently, with no message or beep, and
        the cursor moves to the start so the name can be typed over.
        """
        pending = self.calculating
        assert pending is not None
        if not self.session.is_variable(text):
            line = self.query_one("#prompt-input", Input)
            line.focus()
            line.cursor_position = 0
            return
        pending.variable = text.strip()
        self._hide_prompt()
        self.mode = MODE_MENU
        self._ask(pending.command.dialog, self._calculus_answered, self._calculus_refuses)

    def _calculus_refuses(self, values: dict[str, str | int]) -> str | None:
        """The field of the final dialog the current command refuses, if any."""
        pending = self.calculating
        assert pending is not None
        return pending.command.refuses(values)

    def _calculus_answered(self, values: dict[str, str | int]) -> None:
        """The final dialog is answered: build the expression and append it.

        Plain Enter computes nothing, only wrapping the head around the expression.
        It still goes through the compute path, which Ctrl-Enter's Simplify needs. An
        unparsable field is reported by the completion handler, since the dialog is
        gone by then.
        """
        pending = self.calculating
        assert pending is not None
        self.calculating = None
        command = pending.command
        arguments = (pending.variable, *command.arguments(values))
        simplified = self._wanted_simplified()
        run = partial(
            self.session.calculus,
            command.head,
            command.prefix,
            pending.request,
            arguments,
            simplified,
        )
        self._compute(SIMPLIFYING if simplified else BUILDING, run, self._appended)

    # -- soLve -------------------------------------------------------------
    #
    # Up to three questions, depending on the answers: an expression, then variables
    # one at a time, then, only in Approximate precision (the one mode that searches),
    # the interval. See `_variables_wanted` for how many variables are asked for.

    def _command_solve(self) -> None:
        """Ask which expression to solve, offering the highlighted one."""
        self.solving = Solving()
        entry = self.session.selected_entry
        offered = "" if entry is None else f"#{entry.number}"
        self._prompt(MODE_SOLVE, SOLVE_PROMPT, offered, ENTER_TO_DERIVE, keep=1)

    def _solve_expression(self, request: str) -> None:
        """The expression is known; determine what else to ask.

        The equation count comes cheaply from the tree; the variables come from the
        engine, through the compute path as in Factor.
        """
        pending = self.solving
        assert pending is not None
        if not request.strip():
            self._end_prompt()
            return
        try:
            equations = self.session.equations(request)
        except DeriveSyntaxError as error:
            self._refused(error)
            return
        pending.request = request
        self._compute(
            READING,
            partial(self.session.solve_variables, request),
            partial(self._solve_planned, equations),
        )

    def _solve_planned(self, equations: int, outcome: Outcome) -> None:
        """The variables are known: ask the next question, or run the command."""
        pending = self.solving
        assert pending is not None
        if isinstance(outcome.error, DeriveSyntaxError):
            self._refused(outcome.error)
            return
        if outcome.error is not None:
            self._end_prompt(_reported(outcome))
            return
        variables: tuple[str, ...] = outcome.value  # type: ignore[assignment]
        pending.remaining = variables
        pending.wanted = _variables_wanted(equations, len(variables))
        if pending.wanted:
            self._ask_solve_variable()
        else:
            self._solve_bounds()

    def _ask_solve_variable(self) -> None:
        """Show the variable line, offering the most significant remaining variable."""
        pending = self.solving
        assert pending is not None
        offered = pending.remaining[0] if pending.remaining else ""
        self._prompt(
            MODE_SOLVE_VARIABLE, SOLVE_VARIABLE_PROMPT, offered, ENTER_VARIABLE
        )

    def _solve_variable(self, text: str) -> None:
        """One answer to a variable question.

        Anything but a variable is refused silently, with the cursor moved to the start,
        as on other variable lines.
        """
        pending = self.solving
        assert pending is not None
        if not self.session.is_variable(text):
            line = self.query_one("#prompt-input", Input)
            line.focus()
            line.cursor_position = 0
            return
        chosen = text.strip()
        pending.chosen += (chosen,)
        pending.remaining = tuple(n for n in pending.remaining if n != chosen)
        if len(pending.chosen) < pending.wanted:
            self._ask_solve_variable()
        else:
            self._solve_bounds()

    def _solve_bounds(self) -> None:
        """Ask for the interval, only in Approximate precision."""
        if str(self.settings["Precision"]) != "Approximate":
            self._run_solve()
            return
        self._hide_prompt()
        self.mode = MODE_MENU
        self._ask(menus.SOLVE_BOUNDS, self._solve_bounded, self._solve_refuses)

    def _solve_refuses(self, values: dict[str, str | int]) -> str | None:
        """Refuse a blank bound."""
        return _needed(values, "SolveLower", "SolveUpper")

    def _solve_bounded(self, values: dict[str, str | int]) -> None:
        self._run_solve((_text(values, "SolveLower"), _text(values, "SolveUpper")))

    def _run_solve(self, bounds: tuple[str, str] | None = None) -> None:
        pending = self.solving
        assert pending is not None
        self.solving = None
        self._compute(
            SOLVING,
            partial(self.session.solve, pending.request, pending.chosen, bounds),
            self._solved,
        )

    def _solved(self, outcome: Outcome) -> None:
        """Append the solutions, or report that there were none."""
        if outcome.error is None and not outcome.value:
            self._end_prompt(NO_SOLUTIONS)
            return
        self._appended(outcome)

    # -- Declare -----------------------------------------------------------
    #
    # All four commands pass what they collected to the session, which writes and
    # authors the expression, and then return to the command menu. They differ in
    # their questions.

    def _command_declare_variable(self) -> None:
        self._prompt(MODE_VARIABLE_NAME, VARIABLE_NAME_PROMPT, "", ENTER_VARIABLE_NAME)

    def _variable_name(self, text: str) -> None:
        """The name is known; ask what to declare it as.

        A blank line abandons the command. A name that can't be declared (a predefined
        function or constant) is refused and asked again, as in Derive.
        """
        name = text.strip()
        if not name:
            self._end_prompt(done=False)
            return
        if not self.session.declarable(name):
            self._refuse_name()
            return
        self.declaring = Declaring(name)
        self._ask_domain()

    def _ask_domain(self) -> None:
        """Show the domain menu, opened on the variable's current domain."""
        pending = self.declaring
        assert pending is not None
        self._open_question(
            menus.DECLARE_VARIABLE,
            self.session.declared_as(pending.name),
            menus.SELECT_DOMAIN.format(name=pending.name),
        )

    def _chose_domain(self, word: str) -> None:
        """Handle Value or one of the four domains.

        Complex and Nonscalar have no interval, so they complete the command.
        """
        pending = self.declaring
        assert pending is not None
        self.stack.pop()
        if word == sessions.VALUE:
            self._prompt(MODE_VARIABLE_VALUE, VARIABLE_VALUE_PROMPT, "", ENTER_DEFINITION)
            return
        pending.kind = word
        if word not in menus.BOUNDED_DOMAINS:
            self._declared(lambda: self.session.declare_domain(pending.name, word))
            return
        self._open_question(
            menus.DECLARE_INTERVAL,
            self.session.declared_interval(pending.name),
            menus.SELECT_INTERVAL.format(name=pending.name),
        )

    def _chose_interval(self, word: str) -> None:
        """Handle a named interval, All, or Interval (bounds to be entered)."""
        pending = self.declaring
        assert pending is not None
        self.stack.pop()
        if word == sessions.INTERVAL:
            self._ask(
                menus.variable_bounds(pending.name, self.session.bounds_of(pending.name)),
                self._chose_bounds,
                self._both_bounds,
            )
            return
        bounds = sessions.NAMED_INTERVALS.get(word)
        self._declared(
            lambda: self.session.declare_domain(pending.name, pending.kind, bounds)
        )

    def _both_bounds(self, values: dict[str, str | int]) -> str | None:
        return next(
            (
                setting
                for setting in ("BoundLow", "BoundHigh")
                if not self.session.is_bound(str(values[setting]))
            ),
            None,
        )

    def _chose_bounds(self, values: dict[str, str | int]) -> None:
        pending = self.declaring
        assert pending is not None
        bounds = sessions.Bounds(
            str(values["BoundLow"]),
            str(values["BoundHigh"]),
            menus.closed(values["StrictLow"]),
            menus.closed(values["StrictHigh"]),
        )
        self._declared(
            lambda: self.session.declare_domain(pending.name, pending.kind, bounds)
        )

    def _variable_value(self, text: str) -> None:
        """Assign the value; a blank line leaves the variable unassigned."""
        pending = self.declaring
        assert pending is not None
        self._declared(lambda: self.session.declare_value(pending.name, text))

    def _command_declare_function(self) -> None:
        self._prompt(MODE_FUNCTION_NAME, FUNCTION_NAME_PROMPT, "", ENTER_FUNCTION_NAME)

    def _function_name(self, text: str) -> None:
        name = text.strip()
        if not name:
            self._end_prompt(done=False)
            return
        if not self.session.declarable(name):
            self._refuse_name()
            return
        self.defining = Defining(name)
        self._prompt(MODE_FUNCTION_VALUE, FUNCTION_VALUE_PROMPT, "", ENTER_DEFINITION)

    def _function_value(self, text: str) -> None:
        """Define the function; a blank line declares an arbitrary function."""
        pending = self.defining
        assert pending is not None
        if not text.strip():
            self._ask_function_variable()
            return
        self._declared(lambda: self.session.declare_function(pending.name, text))

    def _ask_function_variable(self) -> None:
        self._prompt(
            MODE_FUNCTION_VARIABLE,
            FUNCTION_VARIABLE_PROMPT,
            "",
            ENTER_FUNCTION_VARIABLE,
        )

    def _function_variable(self, text: str) -> None:
        """Add an arbitrary function's variable; a blank line ends the list.

        With no variables, Derive writes `g :=`, leaving the name an unassigned
        variable.
        """
        pending = self.defining
        assert pending is not None
        name = text.strip()
        if not name:
            self._declared(
                lambda: self.session.declare_arbitrary(pending.name, pending.variables)
            )
            return
        if not self.session.declarable(name):
            self._beep()
            self._ask_function_variable()
            return
        pending.variables += (name,)
        self._ask_function_variable()

    def _command_declare_matrix(self) -> None:
        self._ask(menus.matrix_size(*self.matrix_size), self._chose_size)

    def _chose_size(self, values: dict[str, str | int]) -> None:
        rows, columns = int(values["MatrixRows"]), int(values["MatrixColumns"])
        self.matrix_size = (rows, columns)
        self.entering = Entering(columns, rows)
        self._ask_element()

    def _command_declare_vector(self) -> None:
        self._ask(menus.vector_dimension(self.dimension), self._chose_dimension)

    def _chose_dimension(self, values: dict[str, str | int]) -> None:
        dimension = int(values["VectorDimension"])
        self.dimension = dimension
        self.entering = Entering(dimension)
        self._ask_element()

    def _ask_element(self) -> None:
        """Ask for the next element."""
        pending = self.entering
        assert pending is not None
        row, column = pending.place
        if pending.rows is None:
            message = ENTER_VECTOR_ELEMENT.format(number=column)
            self._prompt(MODE_ELEMENT, VECTOR_ELEMENT_PROMPT, "", message)
            return
        message = ENTER_MATRIX_ELEMENT.format(row=row, column=column)
        self._prompt(MODE_ELEMENT, MATRIX_ELEMENT_PROMPT, MATRIX_ZERO, message)

    def _element(self, text: str) -> None:
        """One element of the vector or matrix being entered.

        A blank line abandons the whole command, as in Derive, since elements can't be
        missing.
        """
        pending = self.entering
        assert pending is not None
        if not text.strip():
            self._end_prompt(done=False)
            return
        try:
            self.session.reads(text)
        except DeriveSyntaxError as error:
            self._refused(error)
            return
        pending.elements += (text.strip(),)
        if len(pending.elements) < pending.wanted:
            self._ask_element()
            return
        if pending.rows is None:
            self._declared(lambda: self.session.declare_vector(pending.elements))
        else:
            self._declared(lambda: self.session.declare_matrix(pending.grid))

    def _refuse_name(self) -> None:
        """Beep and clear the line to ask for the name again."""
        self._beep()
        self.query_one("#prompt-input", Input).value = ""

    def _declared(self, declare: Callable[[], object]) -> None:
        """Run a Declare command and return to the command menu.

        An unparsable line leaves the prompt up for correction, as with Author. Only
        commands that read a line can raise, since names and bounds were validated when
        entered.
        """
        try:
            declare()
        except DeriveSyntaxError as error:
            self._refused(error)
            return
        self._end_prompt()

    def _open_question(self, menu: Menu, word: str, asks: str) -> None:
        """Show a question menu, opened on the current answer `word`.

        The previous question was already popped, so Esc always abandons the whole
        command.
        """
        self._hide_prompt()
        self.mode = MODE_MENU
        index = menu.words.index(word) if word in menu.words else 0
        self.stack.append(MenuCursor(menu, index, asks))
        self.message = asks
        self.refresh_screen()

    # -- Manage ------------------------------------------------------------
    #
    # The four Manage commands that aren't settings screens. Renumber, Annotate and
    # Ordering modify the session without appending to the history. Substitute derives
    # an expression and asks like Factor and Expand.

    def _command_renumber(self) -> None:
        """Renumber the labels in sequence, immediately and without a message."""
        self.session.renumber()
        self._done_with_menu()

    def _command_annotate(self) -> None:
        """Ask which expression to annotate, offering the highlighted one.

        With an empty history it does nothing.
        """
        entry = self.session.selected_entry
        if entry is None:
            self._done_with_menu()
            return
        self._ask(menus.annotate_entry(entry.number), self._annotating, self._annotated)

    def _annotated(self, values: dict[str, str | int]) -> str | None:
        return self._names_nothing(values, "AnnotateExpression")

    def _annotating(self, values: dict[str, str | int]) -> None:
        """The expression is chosen; ask for the annotation, offering the current one."""
        number = int(values["AnnotateExpression"])
        entry = self.session.numbered(number)
        assert entry is not None
        self.annotating = number
        self._prompt(
            MODE_ANNOTATION, ANNOTATION_PROMPT, entry.annotation, ENTER_ANNOTATION
        )

    def _annotate(self, text: str) -> None:
        """Set the entry's annotation from the line.

        A blank line sets a blank annotation, the only way to clear one. Esc abandons
        the command, as on any prompt line.
        """
        number, self.annotating = self.annotating, None
        assert number is not None
        self.session.annotate(number, text)
        self._end_prompt()

    def _command_ordering(self) -> None:
        """Ask for the variable order list, offering the current one."""
        self._prompt(
            MODE_ORDER, ORDER_PROMPT, " ".join(self.session.order), ENTER_ORDER
        )

    def _ordered(self, text: str) -> None:
        """Set the variable order from the line.

        Anything other than a list of variable names beeps and stays up for
        correction. Unlike the Manage settings screens, nothing is recorded in the
        history.
        """
        names = self.session.order_list(text)
        if names is None:
            self._beep()
            return
        self.session.order = names
        self._end_prompt()

    def _command_substitute(self) -> None:
        """Ask which expression to substitute into, offering the highlighted one."""
        self.substituting = Substituting()
        entry = self.session.selected_entry
        offered = "" if entry is None else f"#{entry.number}"
        self._prompt(
            MODE_SUBSTITUTE, SUBSTITUTE_PROMPT, offered, ENTER_TO_DERIVE, keep=1
        )

    def _substitute_expression(self, request: str) -> None:
        """The expression is known; determine what else to ask.

        A highlighted subexpression needs one replacement. A whole expression needs a
        value per variable; without variables the command ends with the Manage menu up.
        """
        pending = self.substituting
        assert pending is not None
        if not request.strip():
            self._end_prompt()
            return
        try:
            pending.part = self.session.substitutes_part(request)
        except DeriveSyntaxError as error:
            self._refused(error)
            return
        pending.request = request
        if pending.part:
            self._ask_value()
            return
        self._compute(
            READING,
            partial(self.session.variables, request),
            self._substitute_variables,
        )

    def _substitute_variables(self, outcome: Outcome) -> None:
        """The variables are known: ask about the first, if any."""
        pending = self.substituting
        assert pending is not None
        if isinstance(outcome.error, DeriveSyntaxError):
            self._refused(outcome.error)
            return
        if outcome.error is not None:
            self._end_prompt(_reported(outcome))
            return
        pending.remaining = outcome.value  # type: ignore[assignment]
        if pending.remaining:
            self._ask_value()
        else:
            self._end_prompt(done=False)

    def _ask_value(self) -> None:
        """Ask for the next value.

        A variable's line offers its own name, so Enter leaves it unchanged. A
        subexpression's line starts empty, with the message line naming it.
        """
        pending = self.substituting
        assert pending is not None
        name = SUBEXPRESSION if pending.part else pending.remaining[0]
        self._prompt(
            MODE_SUBSTITUTE_VALUE,
            SUBSTITUTE_VALUE_PROMPT,
            "" if pending.part else name,
            ENTER_REPLACEMENT.format(name=name),
        )

    def _substitute_value(self, text: str) -> None:
        """One answer to a value question.

        An unparsable line stays up for correction, as with Author. A blank line leaves
        the variable unchanged and moves on; for a subexpression it abandons the
        command rather than appending a copy.
        """
        pending = self.substituting
        assert pending is not None
        blank = not text.strip()
        if not blank:
            try:
                self.session.reads(text)
            except DeriveSyntaxError as error:
                self._refused(error)
                return
        if pending.part:
            if blank:
                self._end_prompt(done=False)
            else:
                self._substituted(
                    lambda: self.session.substitute_part(pending.request, text)
                )
            return
        name, *rest = pending.remaining
        pending.values += ((name, text),)
        pending.remaining = tuple(rest)
        if pending.remaining:
            self._ask_value()
        else:
            self._substituted(
                lambda: self.session.substitute(pending.request, dict(pending.values))
            )

    def _substituted(self, run: Callable[[], object]) -> None:
        """Perform the substitution and report how long it took."""
        self.substituting = None
        started = time.monotonic()
        try:
            run()
        except DeriveSyntaxError as error:
            # Nearly unreachable, since every line was parsed when entered; the prompt
            # may be gone, so just report it.
            self._end_prompt(str(error))
            return
        took = _elapsed(time.monotonic() - started)
        self._end_prompt(COMPUTE_TIME.format(elapsed=took))

    # -- Transfer ----------------------------------------------------------

    def _command_save(self) -> None:
        """Save the worksheet as a math file, asking for the block if Range is Some."""
        self._begin_save(SAVE_PROMPT, self._save, worksheet.SUFFIX)

    def _chose_block(
        self,
        label: str,
        command: Callable[[str], None],
        suffix: str,
        values: dict[str, str | int],
    ) -> None:
        self.block = (int(values["SaveFirst"]), int(values["SaveLast"]))
        self._ask_file(label, command, suffix)

    def _command_save_source(self, language: Language) -> None:
        """Save the worksheet as source code, asking for the block if Range is Some.

        Same as `Transfer Save Derive` except for the notation and extension.
        """
        self._begin_save(
            SAVE_SOURCE_PROMPT.format(word=language.word.upper()),
            partial(self._save_source, language),
            language.suffix,
        )

    def _begin_save(
        self, label: str, command: Callable[[str], None], suffix: str
    ) -> None:
        """Ask for the block to save if Range is Some, then the file name.

        With an empty history it just beeps, as in Derive.
        """
        if not self.session.entries:
            self._beep()
            return
        if self.settings["SaveRange"] != "Some":
            self.block = (None, None)
            self._ask_file(label, command, suffix)
            return
        numbers = [entry.number for entry in self.session.entries]
        self._ask(
            menus.save_block(numbers[0], numbers[-1]),
            partial(self._chose_block, label, command, suffix),
        )

    def _command_load(self) -> None:
        self._ask_file(LOAD_PROMPT, self._load)

    def _command_merge(self) -> None:
        self._ask_file(MERGE_PROMPT, self._merge)

    def _command_load_data(self) -> None:
        self._ask_file(LOAD_DATA_PROMPT, self._load_data, worksheet.DATA_SUFFIX)

    def _command_load_utility(self) -> None:
        self._ask_file(LOAD_UTILITY_PROMPT, self._load_utility)

    def _command_save_state(self) -> None:
        self._ask_file(
            SAVE_STATE_PROMPT, self._save_state, state.SUFFIX, offered=self.state_file
        )

    def _command_load_state(self) -> None:
        self._ask_file(
            LOAD_STATE_PROMPT, self._load_state, state.SUFFIX, offered=self.state_file
        )

    def _ask_file(
        self,
        label: str,
        command: Callable[[str], None],
        suffix: str = worksheet.SUFFIX,
        offered: str | None = None,
    ) -> None:
        """Show the prompt band for a command that takes a file name.

        As in Derive, the session's last file is offered, so saving again is one
        keystroke. The State commands offer the last state file instead.

        The line completes files ending in `suffix`, and directories.
        """
        self.file_command = command
        self.file_suffix = suffix
        self._close_list()
        if offered is None:
            offered = "" if self.session.file is None else str(self.session.file)
        self._prompt(MODE_FILE, label, offered, ENTER_FILE)
        self.query_one("#prompt-input", Input).suggester = FileNames(suffix)

    # -- file name completion and browsing ------------------------------------

    @property
    def browsing(self) -> bool:
        """Whether the completion list is open."""
        return bool(self.completions)

    def action_complete_file(self) -> None:
        """Tab on a file prompt: complete the common prefix, then browse.

        Tab inserts the common prefix of the matching names and opens a list of them;
        a single match is accepted directly. With the list open and nothing more to
        complete, it steps to the next name. With no matches it beeps.
        """
        line = self.query_one("#prompt-input", Input)
        if self.browsing:
            # Typing may have narrowed the list to names sharing a longer prefix; complete
            # that before stepping.
            shared = commonprefix(self.completions)
            if self.completed is None and len(shared) > len(line.value):
                self._put(line, shared)
            else:
                self.action_browse_next()
            return
        found = worksheet.matches(line.value, self.file_suffix)
        if not found or found == [line.value]:
            self._beep()
            return
        if len(found) == 1:
            # A single match: take it, and list a directory's contents right away.
            self._take(line, found[0])
            return
        shared = commonprefix(found)
        if len(shared) > len(line.value):
            self._put(line, shared)
        self._open_list(found, self._at_in(found, line.value))

    def action_browse_next(self) -> None:
        """Down, or Tab with the list open: the next name, wrapping around."""
        self._highlight(0 if self.completed is None else self.completed + 1)

    def action_browse_previous(self) -> None:
        """Up or Shift-Tab: the previous name, wrapping around."""
        self._highlight(-1 if self.completed is None else self.completed - 1)

    def action_browse_take(self) -> None:
        """Enter on an open list: take the highlighted name.

        A directory is entered, with the list showing its contents. A file closes the
        list and stays on the line for the next Enter to open, so choosing a file never
        opens it.
        """
        at = 0 if self.completed is None else self.completed
        self._take(self.query_one("#prompt-input", Input), self.completions[at])

    def action_browse_page(self, by: int) -> None:
        """Page Up and Page Down: move by a screenful, stopping at the ends."""
        at = 0 if self.completed is None else self.completed
        rows = self.query_one(CompletionList).visible_rows(len(self.completions))
        self._highlight(at + by * rows, wrap=False)

    def _highlight(self, at: int, wrap: bool = True) -> None:
        """Highlight a name in the list and put it on the line.

        Every name starts with what was typed, so this only extends the line.
        """
        count = len(self.completions)
        self.completed = at % count if wrap else max(0, min(at, count - 1))
        self._put(
            self.query_one("#prompt-input", Input), self.completions[self.completed]
        )
        self._draw_list()

    def _take(self, line: Input, name: str) -> None:
        """Put `name` on the line, and if it is a directory, list its contents."""
        self._put(line, name)
        self._close_list()
        self._set_message(ENTER_FILE)
        if not name.endswith("/"):
            return
        found = worksheet.matches(name, self.file_suffix)
        if found:
            self._open_list(found, None)
        else:
            # Beep for a directory with nothing usable, rather than silently closing.
            self._beep()

    def _at_in(self, found: list[str], value: str) -> int | None:
        """The initially highlighted name in a new list: the one matching the line.

        Otherwise nothing is highlighted and the line stays the user's typing.
        """
        return found.index(value) if value in found else None

    def _open_list(self, found: list[str], at: int | None) -> None:
        self.completions, self.completed = found, at
        self._draw_list()
        self._set_message(BROWSING_FILES)

    def _draw_list(self) -> None:
        listing = self.query_one(CompletionList)
        listing.display = True
        listing.styles.height = listing.rows_for(len(self.completions))
        listing.show(
            [_last_name(name) for name in self.completions],
            self.completed,
            # All names are in the same directory.
            _directory_of(self.completions[0]),
        )

    def _close_list(self) -> None:
        """Close the completion list."""
        self.completions, self.completed = [], None
        listing = self.query(CompletionList)
        if listing:
            listing.first().display = False

    def _put(self, line: Input, name: str) -> None:
        """Put a name on the line, with the cursor at the end."""
        line.value = name
        line.cursor_position = len(name)

    def _save(self, name: str) -> None:
        self._written(lambda: self.session.save(worksheet.path_of(name), *self.block))

    def _save_source(self, language: Language, name: str) -> None:
        path = worksheet.path_of(name, language.suffix)
        self._written(lambda: self.session.save_source(path, language, *self.block))

    def _save_state(self, name: str) -> None:
        path = worksheet.path_of(name, state.SUFFIX)
        if self._written(lambda: self.session.save_state(path)):
            self.state_file = str(path)

    def _written(self, write: Callable[[], object]) -> bool:
        """Run a save, leaving the name up for correction if writing fails."""
        try:
            write()
        except OSError:
            self._refuse_file(CANNOT_WRITE)
            return False
        self._end_prompt()
        return True

    def _load(self, name: str) -> None:
        self._read(name, self.session.load)

    def _merge(self, name: str) -> None:
        self._read(name, self.session.merge)

    def _load_data(self, name: str) -> None:
        self._read(name, self.session.load_data, worksheet.DATA_SUFFIX)

    def _load_utility(self, name: str) -> None:
        self._read(name, self.session.load_utility)

    def _load_state(self, name: str) -> None:
        if self._read(name, self.session.load_state, state.SUFFIX, UNSET):
            self.state_file = str(worksheet.path_of(name, state.SUFFIX))

    def _read(
        self,
        name: str,
        command: Callable[[Path], int],
        suffix: str = worksheet.SUFFIX,
        refused: str = UNREADABLE,
    ) -> bool:
        """Read a file into the session, leaving the line up if it can't be read."""
        path = worksheet.path_of(name, suffix)
        try:
            skipped = command(path)
        except FileNotFoundError:
            self._refuse_file(FILE_NOT_FOUND)
            return False
        except OSError:
            self._refuse_file(CANNOT_READ)
            return False
        self._end_prompt(
            ENTER_OPTION
            if not skipped
            else refused.format(count=skipped, s="" if skipped == 1 else "s")
        )
        return True

    def _refuse_file(self, message: str) -> None:
        """Report the error and leave the name up for correction."""
        self._beep()
        self._set_message(message)

    # -- Transfer Clear ----------------------------------------------------

    def _command_clear_expressions(self) -> None:
        self._clearing(self.session.clear_expressions)

    def _command_clear_all(self) -> None:
        self._clearing(self.session.clear_all)

    def _command_clear_variables(self) -> None:
        """Clear all variable values, without confirmation since no expressions go."""
        self.session.clear_variables()
        self._done_with_menu()

    def _command_clear_functions(self) -> None:
        self.session.clear_functions()
        self._done_with_menu()

    def _clearing(self, clear: Callable[[], None]) -> None:
        """Clear the history, after confirming as Quit does.

        An empty history needs no confirmation. Either way the pane is drawn empty,
        which also removes the opening notice, as Derive's Clear did.
        """
        self.greeting = False
        if not self.session.entries:
            clear()
            self._done_with_menu()
            return
        self._ask_confirm(lambda: (clear(), self._done_with_menu()))

    # -- Transfer Demo -----------------------------------------------------

    def _command_demo(self) -> None:
        self._ask_file(DEMO_PROMPT, self._demo, worksheet.DEMO_SUFFIX)

    def _command_demonstration(self, demo: catalogue.Demo) -> None:
        """Run one of Derive's demos, downloading it if needed.

        Sources, in order: a local file of that name (from Derive's diskette or saved
        with `Transfer Save`), a download from earlier in this run, then the network.

        A plot gallery is refused before downloading if plotting is unavailable.
        """
        if self.fetching is not None:
            self._beep()
            self._set_message(STILL_FETCHING.format(name=self.fetching))
            return
        if demo.plotting and not available():
            self._plot_refused(UNAVAILABLE)
            return
        path = worksheet.path_of(demo.file)
        if platform.current().storage().exists(path):
            self.demonstrate(demo.file, demo.plotting)
            return
        if demo.file in self.downloads:
            self.demonstrate(demo.file, demo.plotting, self.downloads[demo.file])
            return
        self.fetching = demo.file
        self._set_message(FETCHING.format(name=demo.file))
        self.run_worker(self._fetching(demo))

    async def _fetching(self, demo: catalogue.Demo) -> None:
        """Wait for a demo download, then run it.

        Runs on its own task, since downloads can take a while, so the screen stays
        responsive. `fetcher` handles where the bytes come from and how they're kept;
        this handles waiting, errors, and running the demo from the received text.
        """
        try:
            raw = await self.fetcher(demo)
        except OSError as trouble:
            self._beep()
            self._set_message(CANNOT_FETCH.format(name=demo.file, trouble=trouble))
            return
        finally:
            self.fetching = None
        self.demonstrate(demo.file, demo.plotting, worksheet.decoded(raw))

    async def _downloaded(self, demo: catalogue.Demo) -> bytes:
        """Download one of the nine demos, keeping it in memory for this run.

        The default `fetcher`; a browser environment substitutes its own.

        Downloads are kept in memory rather than written to the current directory.
        The whole diskette arrives at once, so the other eight demos are kept too,
        to avoid downloading them again. `Transfer Save` can save a demo's worksheet.
        """
        # Imported lazily because urllib pulls in http, email and ssl, which a browser
        # page (using its own `fetch`) would otherwise load for nothing.
        from rederive import fetch

        kept: list[tuple[str, bytes]] = []
        raw = await asyncio.to_thread(fetch.fetched, demo, _appending(kept))
        for file, data in [*kept, (demo.file, raw)]:
            self.downloads[file] = worksheet.decoded(data)
        return raw

    def demonstrate(
        self, name: str, plotting: bool = False, text: str | None = None
    ) -> None:
        """Run `name` as a demo, from whatever state the program is in.

        Used by the web page's demo menu, which can be clicked at any time. Whatever was
        on screen (a prompt, help, a paused demo) is abandoned first.

        A plot gallery is refused with the same message as Plot when plotting is
        unavailable.
        """
        self.greeting = False
        self.helping = None
        self.mode = MODE_MENU
        # Abandon any half-answered command.
        self._end_prompt()
        if plotting and not available():
            self._plot_refused(UNAVAILABLE)
            return
        self._demo(name, plotting, text)

    def _demo(
        self, name: str, plotting: bool = False, text: str | None = None
    ) -> None:
        """Start the demo in `name`, or resume the paused one.

        Naming the paused demo's file resumes it, as the manual describes for issuing
        another Demo command. Any other name starts from the first step.

        `plotting` marks a plot gallery, whose steps are plotted rather than
        simplified.

        `text` is the file's contents, passed for a just-downloaded demo, which isn't
        stored on disk.
        """
        suffix = worksheet.SUFFIX if plotting else worksheet.DEMO_SUFFIX
        path = worksheet.path_of(name, suffix)
        if self.demo is None or self.demo.path != path or self.demo.done:
            try:
                steps = (
                    worksheet.demonstration(path)
                    if text is None
                    else worksheet.steps_of(text)
                )
            except FileNotFoundError:
                self._refuse_file(FILE_NOT_FOUND)
                return
            except OSError:
                self._refuse_file(CANNOT_READ)
                return
            self.demo = Demonstration(path, steps, plotting=plotting)
        # A demo simplifies each step anyway, so ignore Ctrl-Enter.
        self.simplifying = None
        self._hide_prompt()
        del self.stack[1:]
        self._restart_menu()
        self._demo_step()

    def _demo_step(self) -> None:
        """Author the next expression and start computing it.

        Unparsable steps are skipped rather than stopping the demo, since there's no
        line to correct. The computation is dispatched, so it can be aborted like any
        other; `_demo_ready` handles the result.

        A step is read in Word input mode, like every line the original reads from a
        file.
        """
        demo = self.demo
        assert demo is not None
        # The closed-window flag applies only within one wait between steps.
        self.closed_window = False
        while not demo.done:
            _, text = demo.steps[demo.at]
            demo.at += 1
            try:
                with self.session.reading_file():
                    self.session.author(text)
            except DeriveSyntaxError:
                continue
            request = f"#{self.session.entries[-1].number}"
            if demo.plotting:
                self._compute(
                    PLOTTING, partial(self._demo_plot, request), self._demo_ready
                )
            else:
                self._compute(
                    SIMPLIFYING,
                    partial(self.session.simplify, request),
                    self._demo_ready,
                )
            return
        # Out of steps: the only ending that also closes the plot.
        self._end_demo(finished=True)

    async def _demo_plot(self, request: str) -> object:
        """Plot a gallery step.

        Classifies and draws in one computation, unlike the Plot command, since a demo
        has nothing to show in between. Each picture replaces the previous one, since
        gallery curves are unrelated.
        """
        drawing = await self._classified(request, None)
        return await self._plot(replace(drawing, demonstrating=True))

    def _demo_ready(self, outcome: Outcome) -> None:
        """Show the step's result and wait for a key.

        A refused step (unparsable or unplottable) is skipped. Any other error,
        notably an abort, pauses the demo at that point.
        """
        if isinstance(outcome.error, (DeriveSyntaxError, PlotError, Unplottable)):
            self._demo_step()
            return
        if outcome.error is not None:
            self._end_demo(_reported(outcome))
            return
        self.mode = MODE_DEMO
        self.message = PRESS_ANY_KEY
        self.refresh_screen()

    def _suspend_demo(self) -> None:
        """Esc pauses the demo, to be resumed later."""
        self._end_demo()

    def _end_demo(self, message: str = ENTER_OPTION, finished: bool = False) -> None:
        """Stop the demo, however it ended.

        A gallery's plot window is released here. If the demo `finished`, the window
        closes with it, since it was the demo's screen; if stopped early, it stays as an
        ordinary window, which is how a picture can be kept.

        The release runs on a worker because the host is another process and this loop
        mustn't wait for it: the key that ended the demo may have come from the plot
        window, and the thread that delivered it is blocked until this call returns.
        """
        if self.demo is not None and self.demo.plotting:
            self.run_worker(self.plots.release(finished))
        self.closed_window = False
        self.mode = MODE_MENU
        self.query_one("#menu").display = True
        self._return_to_menu(message)

    def _demo_comment(self) -> str:
        """The comment of the current step, shown on the menu band."""
        demo = self.demo
        if demo is None or not demo.at:
            return ""
        return demo.steps[demo.at - 1][0]

    # -- Plot --------------------------------------------------------------
    #
    # Two commands for the highlighted expression: `Plot` classifies it and sends it to
    # the receiver (the plot window last used), and `New` sends it to a new window.
    # Everything else is done in the plot window: right-click a curve or its legend
    # entry to remove it, and use the toolbar to clear.
    #
    # Everything goes through the compute path, even the non-computations. Getting the
    # variables is an engine call, and the first plot starts a process that imports Qt
    # and sympy, so the message line's clock keeps the wait from looking like a freeze.
    #
    # The app holds no plot state. The host owns which window is the receiver and what
    # it contains, since only it sees the user interact with windows.

    def _command_plot(self, where: plots.Where | None) -> None:
        """Plot the highlighted expression, into the receiver or a new window.

        As with Simplify, a highlighted subexpression is plotted, otherwise the whole
        entry. It deliberately asks no questions.
        """
        if not available():
            self._plot_refused(UNAVAILABLE)
            return
        entry = self.session.selected_entry
        if entry is None:
            self._plot_refused(NOTHING_TO_PLOT)
            return
        request = f"#{entry.number}"
        self._compute(
            PLOTTING, partial(self._classified, request, where), self._plot_classified
        )

    async def _classified(self, request: str, where: plots.Where | None) -> plots.Add:
        """Classify `request` and build the plot request for it, without sending it.

        Runs on the compute path because getting the variables is an engine call. It
        doesn't depend on window state (polar is the window's own view mode).
        """
        target, label = self.session.named_target(request)
        variables = await self.session.variables(request)
        relation = await self.session.relation(request)
        text = write_expression(target, spaced=True)
        plotted = classify(target, variables, text, relation)
        kind = plotted.kind
        options = plotted.options
        if kind in (plots.PlotKind.FAMILY, plots.PlotKind.SURFACES):
            # A vector is one expression but several plots, so each element carries
            # its own text.
            options = replace(
                options,
                texts=tuple(
                    write_expression(element, spaced=True)
                    for element in target.children
                ),
            )
        if kind not in plots.DRAWN:
            raise PlotError(plots.UNDRAWN.format(kind=kind.value))
        return plots.Add(
            # Identify the worksheet by identity, since two overlays can each have a
            # `#3`; plots are keyed by the pair.
            worksheet=id(self.session),
            node=target,
            context=self.session.context,
            kind=kind,
            window=where,
            label=label,
            text=text,
            options=options,
            # The parse state lets the window's fields (parameter range, surface
            # domain) parse typed bounds like the worksheet does.
            state=self.session.state,
        )

    def _plot_classified(self, outcome: Outcome) -> None:
        """Send the classified plot, without asking anything.

        Parametric and polar plots are drawn over one turn; the range is adjusted from
        the plot window's toolbar.
        """
        if outcome.error is not None:
            self._plot_refused(_plot_reason(outcome.error))
            return
        request = outcome.value
        assert isinstance(request, plots.Add)
        self._send_plot(request)

    def _send_plot(self, request: plots.Add) -> None:
        self._compute(PLOTTING, partial(self._plot, request), self._plot_done)

    async def _plot(self, request: plots.Add) -> str:
        """Send a plot and return the confirmation message.

        Awaited like an engine command, so the loop keeps painting however the plotter
        waits. Only the host's answer says whether an existing curve was replaced, so
        the message (`Plotting` or `Replotting`) is made here.
        """
        placed = await self.plots.add(request)
        worded = REPLOTTED if placed.replaced else PLOTTED
        return worded.format(label=request.label)

    def _plot_done(self, outcome: Outcome) -> None:
        """Report the plot, or why it failed."""
        if outcome.error is not None:
            self._plot_refused(_plot_reason(outcome.error))
            return
        self._plot_finished(str(outcome.value))

    def _plot_finished(self, message: str) -> None:
        """Return to the command menu after a Plot command."""
        del self.stack[1:]
        self._restart_menu()
        self._return_to_menu(message)

    def _plot_refused(self, reason: str) -> None:
        """Report a plot refusal, with the common prefix."""
        self._plot_finished(f"{PREFIX}{reason}")

    def _plot_reported(self, event: plots.Event) -> None:
        """Pass an event from the plot windows to the event loop.

        Only the event loop may touch the screen. A host process reports on the proxy's
        reader thread and must cross to the loop; in-page panes already report on the
        loop, where `call_from_thread` would fail.
        """
        if threading.get_ident() == self._on_loop:
            self.call_next(self._plot_event, event)
            return
        self.call_from_thread(self._plot_event, event)

    def _plot_event(self, event: plots.Event) -> None:
        """Handle an event from the plot host, on the event loop.

        A closed window isn't reported, but during a demo it is noted, since the closing
        click is about to arrive and isn't a step cue. Evaluation trouble is reported
        rather than ignored. A traced point is authored into a worksheet.

        A changed sticky control is stored silently in the settings, where it outlives
        the host and is saved in state files; the settings watcher passes it back to the
        proxy for future hosts.
        """
        if isinstance(event, plots.Closed):
            # See `on_click`.
            self.closed_window = self.mode == MODE_DEMO
        elif isinstance(event, plots.Stepped):
            self._plot_stepped(event)
        elif isinstance(event, plots.Trouble):
            self._set_message(f"{PREFIX}{event.label}: {event.message}")
        elif isinstance(event, plots.Traced):
            self._plot_traced(event)
        elif isinstance(event, plots.Preferred):
            self.settings.apply(learned(event.preferences))

    def _plot_stepped(self, event: plots.Stepped) -> None:
        """Treat a key pressed in a demo's plot window as pressed here.

        The plot window likely has focus, so "any key to continue" must work there too.
        Only keys the window doesn't use arrive; Esc there closes the picture, while
        stopping the demo is this screen's Esc.

        Keys arriving after the demo has ended are dropped.
        """
        if self.mode == MODE_DEMO:
            self._demo_step()

    def _plot_traced(self, event: plots.Traced) -> None:
        """Author a point sent back from a plot window as a new entry.

        It goes to the originating worksheet if it is still open, otherwise to the
        active one, rather than dropping a point the user asked for.

        While a command computes, it owns the worksheets, so the point is retried
        later. During a dialog or edit, appending is safe: the panes repaint and the
        selection moves to the new entry, so `F3` can copy it onto the line.
        """
        if self.mode == MODE_COMPUTE:
            self.set_timer(POINT_RETRY, partial(self._plot_traced, event))
            return
        session = next(
            (one for one in self.windows.sessions() if id(one) == event.worksheet),
            self.session,
        )
        try:
            entry = session.author(event.text)
        except DeriveSyntaxError as error:
            # A point unparsable under the worksheet's settings (such as a digit invalid
            # in the input base) reports the parser's error, as Author would.
            self._set_message(f"{PREFIX}{error}")
            return
        self.place_windows()
        self._set_message(POINT_ENTERED.format(text=event.text, label=f"#{entry.number}"))

    # -- Window ------------------------------------------------------------
    #
    # Seven commands over the window tree. The app handles only the screen side (each
    # window's pane and size); the tree, numbering and geometry are in `model.windows`.
    #
    # Close can discard a worksheet, so it asks first, with the dialog still shown.
    #
    # All tree windows are Algebra windows (plot windows are desktop windows), so
    # `Open` doesn't ask for a type. Splitting copies the worksheet, and the copies
    # evolve independently.

    def _command_window_split(self, vertical: bool) -> None:
        """Ask where to cut the active window in two."""
        size = self._active_size(vertical)
        low, high = windows.split_range(vertical, size)
        if high < low:
            # No room for a divider. Derive just beeps.
            self._beep()
            return
        self._ask(
            menus.window_split(vertical, windows.split_default(size), low, high),
            partial(self._split_window, vertical),
        )

    def _split_window(self, vertical: bool, values: dict[str, str | int]) -> None:
        self.windows.split(vertical, int(values["SplitAt"]), self.session.copy())
        self._done_with_menu()

    def _command_window_close(self) -> None:
        """Ask which window to close, offering the active one."""
        count = len(self.windows.windows)
        if count == 1 and not self.windows.active.stacked:
            # The last window can't be closed. Derive just beeps.
            self._beep()
            return
        self._ask(
            menus.window_number(menus.WINDOW_CLOSE, self.windows.number, count),
            self._close_window,
        )

    def _close_window(self, values: dict[str, str | int]) -> None:
        window = self.windows.numbered(int(values["WindowNumber"]))
        assert window is not None
        closing = partial(self._drop_window, window)
        if window.session.entries:
            self._confirm_over(values, closing)
            return
        closing()

    def _drop_window(self, window: Window) -> None:
        for dropped in self.windows.close(window):
            dropped.discard()
        self._done_with_menu()

    def _command_window_goto(self) -> None:
        """Ask which window to make active, offering the next one."""
        count = len(self.windows.windows)
        self._ask(
            menus.window_number(
                menus.WINDOW_GOTO, self.windows.number % count + 1, count
            ),
            self._goto_window,
        )

    def _goto_window(self, values: dict[str, str | int]) -> None:
        self.windows.goto(int(values["WindowNumber"]))
        self._done_with_menu()

    def _command_window_step(self, direction: int) -> None:
        """Next and Previous, which run immediately."""
        self.windows.step(direction)
        self._done_with_menu()

    def _command_window_flip(self, direction: int = 1) -> None:
        """Flip: show the next overlay of the active window."""
        self.windows.active.flip(direction)
        self._done_with_menu()

    def _command_window_open(self) -> None:
        """Overlay a new worksheet on the active window.

        Derive asked for a window type, but plot windows are outside the tree here, so
        there's only one type and the command runs immediately.
        """
        self.windows.open(windows.ALGEBRA, self._new_session())
        self._done_with_menu()

    def _new_session(self) -> Session:
        """An empty worksheet for a new window, sharing the settings and the engine."""
        return Session(self.settings, self.session.runner)

    def _active_size(self, vertical: bool) -> int:
        """The current width or height of the active window's interior."""
        panes = self.query_one(Panes)
        rect = self.windows.interior(
            self.windows.active, panes.size.height, panes.size.width
        )
        return rect.width if vertical else rect.height

    def action_window_step(self, direction: int) -> None:
        """F1 and Shift-F1: go to the next or previous window."""
        self.windows.step(direction)
        self.refresh_screen()

    def action_window_flip(self, direction: int) -> None:
        """F2 and Shift-F2: show the next or previous overlay of this window."""
        self.windows.active.flip(direction)
        self.refresh_screen()

    # -- confirmations -----------------------------------------------------

    def _ask_confirm(self, confirmed: Callable[[], None]) -> None:
        """Ask a yes/no question, running `confirmed` on Y."""
        self.confirm = confirmed
        self.mode = MODE_CONFIRM
        self.message = ABANDON_PROMPT
        self.refresh_screen()

    def _confirm_over(
        self, values: dict[str, str | int], confirmed: Callable[[], None]
    ) -> None:
        """Ask a yes/no question with the just-answered dialog still shown.

        As in Derive, the band shows the entered value in parentheses (settled, not
        being edited) while the message line asks whether to discard the expressions.
        """
        assert self.asked is not None
        editor = DialogEditor(self.asked, self.settings)
        editor.values.update(values)
        # No active field, so every field prints as inactive.
        editor.active = -1
        self.stack.append(editor)
        self.confirmed_over = True
        self._ask_confirm(confirmed)

    def _answer_confirm(self, yes: bool) -> None:
        """Y runs the command; anything else returns to the menu it came from."""
        confirmed, self.confirm = self.confirm, None
        self.mode = MODE_MENU
        if self.confirmed_over:
            self.confirmed_over = False
            self.stack.pop()
        if yes and confirmed is not None:
            confirmed()
            return
        self._return_to_menu(ENTER_OPTION)

    def _done_with_menu(self) -> None:
        """Return to the command menu after a completed command."""
        del self.stack[1:]
        self._restart_menu()
        self._return_to_menu(ENTER_OPTION)

    def _command_quit(self) -> None:
        """Quit, confirming once for all windows if any have expressions."""
        if not any(session.entries for session in self.windows.sessions()):
            self.exit()
            return
        self._ask_confirm(self.exit)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Update an open completion list as the file prompt is edited.

        Typing narrows the list and Backspace widens it; deleting a separator goes back
        to the parent directory. Changes from highlighting a name are ignored.
        """
        if not self.browsing:
            return
        if self.completed is not None and event.value == self.completions[self.completed]:
            return
        found = worksheet.matches(event.value, self.file_suffix)
        if found:
            self._open_list(found, self._at_in(found, event.value))
        else:
            self._close_list()
            self._set_message(ENTER_FILE)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Enter on a prompt line submits the whole line."""
        event.stop()
        self._submitted(event.value)

    def _submitted(self, value: str) -> None:
        """Dispatch the submitted line to the command for the current mode."""
        if self.mode == MODE_SIMPLIFY:
            self._simplify(value)
        elif self.mode == MODE_APPROX:
            self._approx(value)
        elif self.mode == MODE_ASKING:
            self._asked(value)
        elif self.mode == MODE_ASKING_VARIABLE:
            self._asked_variable(value)
        elif self.mode == MODE_BUILD:
            self._build_operand(value)
        elif self.mode == MODE_BUILD_NEXT:
            self._build_next(value)
        elif self.mode == MODE_CALCULUS:
            self._calculus_expression(value)
        elif self.mode == MODE_CALCULUS_VARIABLE:
            self._calculus_variable(value)
        elif self.mode == MODE_SOLVE:
            self._solve_expression(value)
        elif self.mode == MODE_SOLVE_VARIABLE:
            self._solve_variable(value)
        elif self.mode == MODE_JUMP:
            self._jumped(value)
        elif self.mode == MODE_FILE:
            self._named(value)
        elif self.mode == MODE_VARIABLE_NAME:
            self._variable_name(value)
        elif self.mode == MODE_VARIABLE_VALUE:
            self._variable_value(value)
        elif self.mode == MODE_FUNCTION_NAME:
            self._function_name(value)
        elif self.mode == MODE_FUNCTION_VALUE:
            self._function_value(value)
        elif self.mode == MODE_FUNCTION_VARIABLE:
            self._function_variable(value)
        elif self.mode == MODE_ELEMENT:
            self._element(value)
        elif self.mode == MODE_ANNOTATION:
            self._annotate(value)
        elif self.mode == MODE_ORDER:
            self._ordered(value)
        elif self.mode == MODE_SUBSTITUTE:
            self._substitute_expression(value)
        elif self.mode == MODE_SUBSTITUTE_VALUE:
            self._substitute_value(value)
        else:
            self._author(value)

    def _named(self, name: str) -> None:
        """Enter on a file prompt; an empty line abandons the command."""
        if not name.strip():
            self._end_prompt(done=False)
            return
        assert self.file_command is not None
        self.file_command(name)

    def _author(self, text: str) -> None:
        """Enter the line as a new expression.

        An empty line returns to the menu, like Esc. An unparsable line is not entered:
        as in Derive, it beeps, reports the error, and leaves the cursor where parsing
        stopped, which may be past the actual mistake.

        A line with a trailing `=` also enters its value, which must be computed, so it
        runs as a computation with the clock and Esc. Other lines are appended directly.
        """
        if not text.strip():
            self._end_prompt(done=False)
            return
        try:
            if self.session.shows_value(text):
                # Ignore Ctrl-Enter: the value is already computed, and simplifying the
                # resulting equation would just yield true.
                self.simplifying = None
                self._compute(
                    SIMPLIFYING, partial(self.session.show_value, text), self._derived
                )
                return
            self.session.author(text)
        except DeriveSyntaxError as error:
            self._refused(error)
            return
        self._end_prompt()

    def _simplify(self, request: str) -> None:
        """Simplify what the line asks for."""
        self._derive(request, self.session.simplify, SIMPLIFYING)

    def _approx(self, request: str) -> None:
        """Approximate what the line asks for."""
        self._derive(request, self.session.approx, APPROXIMATING)

    def _derive(
        self, request: str, run: Callable[[str], Awaitable[object]], label: str
    ) -> None:
        """Run `run` on the line's expression and report how long it took.

        An empty line does nothing. An unparsable line stays up for correction, as with
        Author; this is detected in `_derived`, since the line is only parsed when the
        command runs on its task.
        """
        if not request.strip():
            self._end_prompt()
            return
        self._compute(label, partial(run, request), self._derived)

    def _derived(self, outcome: Outcome) -> None:
        """Report the answer, or refuse an unparsable line."""
        if isinstance(outcome.error, DeriveSyntaxError):
            self._refused(outcome.error)
            return
        self._end_prompt(_reported(outcome))

    def _refused(self, error: DeriveSyntaxError) -> None:
        """Report a parse error and leave the line up with the cursor at the error.

        The line is refocused, since a running command took focus away.
        """
        self._beep()
        self._set_message(str(error))
        line = self.query_one("#prompt-input", Input)
        line.focus()
        line.cursor_position = error.offset

    def _hide_prompt(self) -> None:
        """Hide the prompt band and show the menu band."""
        self._close_list()
        self.query_one("#prompt-input", Input).value = ""
        self.query_one("#prompt-band").display = False
        self.query_one("#menu").display = True
        self.set_focus(None)

    def _end_prompt(self, message: str = ENTER_OPTION, done: bool = True) -> None:
        """End the command, completed or abandoned, and show a menu.

        A completed command (`done`) returns to the command menu: `Transfer Save Derive`
        doesn't return to the Transfer Save menu. Abandoning, like Esc, leaves the
        submenu up.
        """
        self.asking = None
        self.building = None
        self.calculating = None
        self.solving = None
        self.substituting = None
        self.declaring = None
        self.defining = None
        self.entering = None
        self.annotating = None
        self._hide_prompt()
        if done:
            del self.stack[1:]
            self._restart_menu()
        self._return_to_menu(message)

    def _entered_to_simplify(self) -> list[sessions.Entry]:
        """The entries the command added, if it was invoked with Ctrl-Enter.

        Empty if Enter was used or nothing was entered. A command that enters several,
        such as a Transfer that reads a file, has all of them simplified in order.

        New entries are found by identity rather than label, since `Transfer Load`
        restarts the numbering.
        """
        before, self.simplifying = self.simplifying, None
        if before is None:
            return []
        known = {id(entry) for entry in before}
        return [entry for entry in self.session.entries if id(entry) not in known]

    async def _simplify_entered(self, entered: list[sessions.Entry]) -> None:
        """Simplify each entry in turn.

        An abort keeps what has been simplified so far and skips the rest.
        """
        for entry in entered:
            await self.session.simplify(f"#{entry.number}")

    def _return_to_menu(self, message: str) -> None:
        entered = self._entered_to_simplify()
        if not entered:
            self._menu_again(message)
            return
        self._compute(
            SIMPLIFYING,
            partial(self._simplify_entered, entered),
            partial(self._simplified_entered, message),
        )

    def _simplified_entered(self, message: str, outcome: Outcome) -> None:
        # Report the simplification's compute time, unless the command has its own
        # message (such as unreadable lines in a file). Errors are always reported.
        if message == ENTER_OPTION or outcome.error is not None:
            message = _reported(outcome)
        self._menu_again(message)

    def _menu_again(self, message: str) -> None:
        """Return to the command menu with `message`."""
        self.mode = MODE_MENU
        self.message = message
        self.refresh_screen()


def _last_name(name: str) -> str:
    """A completion's last component, keeping a trailing separator."""
    trimmed = name.rstrip("/")
    return trimmed.rpartition("/")[2] + name[len(trimmed) :]


def _directory_of(name: str) -> str:
    """The directory of a completion, for the list's title.

    A name without a directory is in the current directory, `./`.
    """
    head = name.rstrip("/").rpartition("/")[0]
    return f"{head}/" if head else "./"


def _color_index(editor: DialogEditor) -> int:
    """The color menu index of the field's current color, for opening the menu."""
    current = editor.value(editor.field)
    return next(
        (index for index in range(len(COLORS.words)) if menus.color_at(index) == current),
        0,
    )

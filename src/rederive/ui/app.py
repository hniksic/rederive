"""The Rederive Algebra pane as a Textual application.

Key handling deliberately bypasses Textual's focus model. In DOS Derive the
command menu and the expression highlight are two cursors that exist at the
same time, driven by disjoint key sets: Tab/Space/Backspace/Enter and the
mnemonic letters drive the menu, while the arrow keys drive the selection in
the work area. So the menu keys are app-level priority bindings, switched off
by `check_action` whenever the author line is up and the Input should get the
keystrokes instead.

The keys that move the highlight from one expression to another are the
exception: they keep working while a prompt line has the screen, since a line
of text has nothing vertical for them to do. The ones that move a cursor along
the line - Left, Right, Home, End - belong to the line while it is up, and to
the highlight when it is not.

Submenus and Options dialogs stack on top of the command menu: Esc pops one
level, and committing a dialog returns all the way to the command menu, as the
original does. A command with a question of its own puts up a dialog too - the
block `Remove` takes out, the place `Unremove` puts it back - and gets its
answer handed to it rather than stored, since it is about the command in hand
and not about the system.

A command that needs a line of its own - Author, Simplify, approX, Factor and
Expand an expression, Jump a label number, the Transfer commands a file name -
takes the screen with the prompt band instead, which is one Input with a label
in front of it. The mode says which command the line belongs to, and so what
Enter does with it. A command that runs is finished with, so it leaves the
command menu up rather than the submenu it was picked from; Esc, which abandons
the line instead, leaves that submenu where it was.

Ctrl-Enter is Enter with the expression the command enters simplified after it,
on every line an expression is entered from. It is one key in two spellings:
Ctrl-J is what a terminal sends for it unless it speaks the keyboard protocol
that tells the two apart.

An expression moves by way of the clipboard: Ctrl-C hands what is highlighted to
the terminal, which passes it to the system clipboard where it can, and Ctrl-V
writes the copy kept here onto the line being typed. Text copied in another
program comes the other way by the terminal's own paste key, arriving as if it
had been typed: a terminal hands over the clipboard when its user asks and not
when a program does. The original's F3 and F4, which write the highlight onto
the line directly, still work; the help offers copy and paste instead.

A line that names a file completes what is typed on it, and opens a list of the
names on offer to look through. The first name the letters so far could grow
into stands past the cursor, dimmed, and Right takes it - that is the fast path,
for a name half known. Tab is the other one: it writes out as much of the names
as they all share and opens the list above the line, where Up and Down walk the
names with the one in hand highlighted, Enter goes into a directory or takes a
file, Esc puts the list away, and typing narrows it. Tab has nothing else to do
on such a line, the menu it steps being off the screen.

The list is what makes a file worth looking for rather than only worth typing:
every key that moves the highlight moves the line with it, so there is never a
hidden position to keep track of, and every one of them has a key that undoes
it. That is the whole difference from the original's F1 file listing, which
could only be read and never walked.

A command that reaches the math engine does not run on the event loop. It is
handed to a thread, and the screen goes into a mode of its own until the answer
comes back: the worksheet is neither read nor repainted, since the thread owns
the session until it is done, and the only key that means anything is Esc,
which aborts. What does keep moving is the message line, saying which command
is running and how long it has been, and the status line's memory gauge - which
together are how a user decides whether this one is worth waiting for. Every
dispatch site goes through one helper, and every one of them finishes in a
completion handler rather than in the line that started it.

Two commands take the band for something other than a menu. A question with a
Y or N for an answer - Quit, and the two Clear commands that throw expressions
away - leaves the menu up with its highlight off, and `confirm` holds what a Y
does. A demonstration takes the band outright: the comment above each step
stands where the menu words go, and any key runs the next step.

Factor and Expand ask more than one question, and ask them in both forms: an
expression, then a variable at a time on the prompt band, then an amount off a
menu stacked on the command menu. The two are one flow here, `Asking`, because
the original puts the same questions in the same order for both; a `Command`
holds what differs, which is the word on the prompts, the amounts on offer,
when the amount is worth asking about, and what runs at the end. What has been
answered so far lives in the `Asking` until there is enough to run the command,
since each prompt is gone by the time the next one is up. How many questions
get asked depends on the answers - a number is decomposed without being asked
about at all - so the sequence is in the handlers rather than in a table.

The Manage commands that are not settings screens ask in both forms too, and
one of them asks nothing at all: Renumber runs on the keystroke, Annotate puts
up a dialog for the label and then a line for the text, and Ordering puts up a
line carrying the order list. None of the three appends anything to the
history, so none of them has a compute time to report.

Substitute is the fourth and the only one that derives an expression. It asks
the way Factor and Expand do - an expression, then a line per answer - and
what it asks for depends on what is highlighted: a value for each variable of
a whole expression, or one replacement for a highlighted subexpression. It
appends what it was given without simplifying it, so Ctrl-Enter on the last
value is worth pressing, and the lines it collects are `Substituting`.

The `Window` commands are the one group that works on more than one worksheet.
`self.session` is the active window's, so every command in this file goes on
reading and writing one session and knows nothing about the rest; what the
window commands do is change which one that is, or make another. The screen
follows: one work area per window, placed over a frame that draws what is
between them, and a frame with nothing to draw while there is one window -
which is the original's unsplit screen, borderless and eighty columns wide.

The Declare commands are the same shape and each keeps its own answers -
`Declaring`, `Defining`, `Entering` - but their questions replace each other
rather than stacking, because the original abandons the whole command from
whichever of them is up. A question answered off a menu therefore pops it
before putting the next one, and one Esc always lands back on the Declare menu.

Build is the one command with no end to its questions: an operand on a line, an
operator off a menu, another operand where the operator takes one, and round
again until `Done`. What it has put together lives in a `Building` and is one
tree however many operators have gone into it, since each of them folds the
last into a new node. The menu comes down rather than aside whenever an operand
is asked for, so that Esc on that line abandons the whole command as the
original does, and goes back up on `Done` and after every unary operator.

soLve asks in both forms as well, and is the one command that appends any
number of expressions rather than exactly one: an expression on a line, then a
variable at a time for as many as the expression leaves undecided, then - in
Approximate precision alone - the interval to search, off a dialog. What has
been answered so far lives in a `Solving`. How many variable questions get
asked is worked out from two numbers the session supplies before anything is
computed, the variables of the target and how many equations it is, so the
sequence is again in the handlers. An answer of no solutions at all appends
nothing and says so on the message line, that being the whole of what the
original does with one.

The seven Calculus commands are one flow the other way about: the same two
lines - an expression, then the variable, offered as the primary variable of
what was named - and then a dialog that differs. A `Calculus` holds what
differs, which is the head to write, the word the annotation is spelled with,
the dialog, and how what comes back off it becomes the arguments after the
variable. None of them computes: each appends the head unevaluated for a
Simplify after it to take.

Both of them take Ctrl-Enter on their last question and neither uses the
general path for it. The original enters one expression and not two - the built
form never reaches the history - so the mark Ctrl-Enter left is taken back off
and the command simplifies what it built instead of appending it, which is what
`Simp(#1+#1)` and `Simp(Dif(#1,x))` say.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from os.path import commonprefix
from pathlib import Path
from typing import Any, Callable

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.suggester import Suggester
from textual.widgets import Input, Static
from textual.widgets.input import Selection

from rederive import __version__
from rederive.engine import Amount, EngineAborted
from rederive.model import building, state, windows, worksheet
from rederive.model import help as helps
from rederive.model import session as sessions
from rederive.model.expr import Node
from rederive.model.session import Session
from rederive.model.settings import (
    ChoiceField,
    Dialog,
    DialogEditor,
    Settings,
    TextField,
)
from rederive.model.windows import Window, Windows
from rederive.syntax import LANGUAGES, DeriveSyntaxError, Language
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
#: The two modes the prompt band is in while Factor or Expand is asking.
MODE_ASKING = "asking"
MODE_ASKING_VARIABLE = "asking_variable"
#: The two lines Build reads an operand on, the second of which comes back
#: after every binary operator.
MODE_BUILD = "build"
MODE_BUILD_NEXT = "build_next"
#: The two lines every Calculus command reads before the one that is its own.
MODE_CALCULUS = "calculus"
MODE_CALCULUS_VARIABLE = "calculus_variable"
#: The two lines soLve reads before the interval Approximate precision asks for.
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
#: The two modes the prompt band is in while Manage Substitute is asking.
MODE_SUBSTITUTE = "substitute"
MODE_SUBSTITUTE_VALUE = "substitute_value"
MODE_DEMO = "demo"
#: The mode a computation holds the screen in. Modal on purpose: the thread
#: running the command owns the session until it is done, so nothing may read
#: the worksheet and no key but Esc means anything.
MODE_COMPUTE = "compute"
#: The mode help holds the screen in. A menu mode like the command menu - the
#: same keys move the same highlight - but the work area is showing a document
#: rather than a worksheet, so the keys that walk the history turn pages here.
MODE_HELP = "help"

#: The two modes a menu band is being driven in.
MENU_MODES = (MODE_MENU, MODE_HELP)

#: Which way each movement turns a page of help. The keys are the ones that
#: walk the history everywhere else, which is what makes them the ones a reader
#: reaches for.
HELP_PAGING = {"down": 1, "page_down": 1, "up": -1, "page_up": -1}

#: The modes in which the prompt band has the screen, and the keys with it.
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

#: The one prompt line that takes no notice of the keys that walk the history:
#: the variable line Factor and Expand collect on, where the original holds the
#: highlight still. Under every other line it goes on walking, whatever that
#: line is collecting. Which expression is being worked on may be settled by
#: then, but naming it is not all the highlight is for: it is also where a
#: value is found, and copying is how it is taken onto the line.
SETTLED_MODES = (MODE_ASKING_VARIABLE,)

#: The prompt lines the highlight can still be walked under, which is every
#: other one of them.
WALKED_MODES = tuple(mode for mode in PROMPT_MODES if mode not in SETTLED_MODES)

#: The prompt lines that come up naming an expression, and so take the label of
#: wherever the walk lands - for as long as the label they offered is still
#: selected on them. A line typed on is a line the user has taken over. The
#: Declare lines take an expression too but are offered nothing to begin with,
#: so there is nothing on them to keep in step with the highlight.
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

#: The prompt lines an expression is entered on, and so the ones Ctrl-Enter
#: says something on: it enters the line and simplifies what the command
#: entered. The lines that derive an expression instead - Simplify, approX, and
#: the expression Factor and Expand ask for - have simplified it already, and
#: Jump enters nothing at all, so on those Ctrl-Enter is Enter and nothing more.
#: Substitute's value line is one of these: it derives an expression and
#: deliberately leaves it unsimplified, so there is something left to ask for.
ENTERING_MODES = (
    MODE_AUTHOR,
    MODE_VARIABLE_VALUE,
    MODE_FUNCTION_VALUE,
    MODE_ELEMENT,
    MODE_FILE,
    MODE_SUBSTITUTE_VALUE,
)

#: The prompt lines an expression is written on or named on, which is where the
#: keys that work on an expression apply: F3 and F4 write onto them, and F6
#: hands them their sideways keys. The manual gives F3 for the author line and
#: then says of the Declare value lines and of Substitute's that they take "all
#: the normal line editing features provided by the Author command"; the lines
#: that name an expression take them too, and the original marks exactly this
#: set with the `Lin` flag on the status line.
#:
#: The lines left out are the ones collecting something that is not an
#: expression - a variable, a file name, a label number, an annotation - and
#: there the sideways keys stay the line's whatever the arrow key setting says.
EXPRESSION_LINES = (
    tuple(mode for mode in ENTERING_MODES if mode != MODE_FILE) + LABELLED_MODES
)

#: What each Alt key writes. The original held these in the IBM PC character
#: set and offered the Alt keys as a second way in beside spelling the name
#: out; the names still work, so this is a shorthand and never the only way to
#: write any of them.
#:
#: Where the original's table and ours disagree on a letter's case - it has
#: `Θ`, `Φ` and `Ω` because those were the only forms its character set held -
#: the key writes the spelling this program prints, so that what is typed and
#: what comes back read alike. `#e` and `#i` are written as the names rather
#: than as their glyphs for the same reason.
#:
#: Alt-minus needs a terminal that speaks the keyboard protocol telling a key
#: and its modifiers apart; without one the key arrives as an Escape and a
#: hyphen and nothing here sees it. The letters do not: those come through
#: either way. `±` is written `+-` where the key does not land.
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
    "e": "#e",
    "i": "#i",
    "0": "∞",
    "minus": "±",
    # The subscript operator, which is a glyph in the original and a word here.
    # It is spaced, as the printer spaces it.
    "v": " SUB ",
}

# The Author line is the one line the original advertises help on, and F1 is
# the key it advertises. Every other line takes the key too; only this one says
# so, there being nowhere to say it that is not the message line.
ENTER_EXPRESSION = "Enter expression (press F1 for help)"
#: What every command that derives an expression from another asks for, the
#: help the Author line offers being left off theirs.
ENTER_TO_DERIVE = "Enter expression"
#: What Quit and the two Clear commands that throw expressions away both ask.
ABANDON_PROMPT = "Abandon expressions (Y/N)?"
AUTHOR_PROMPT = " AUTHOR expression: "
SIMPLIFY_PROMPT = " SIMPLIFY expression: "
APPROX_PROMPT = " APPROX expression: "
JUMP_PROMPT = " JUMP to: "
EXPRESSION_PROMPT = " {word} expression: "
VARIABLE_PROMPT = " {word} variable {number}: "
#: What Build asks on each of its two operand lines. The first names what is
#: being built from; the second comes back after every binary operator.
BUILD_PROMPT = " BUILD first expression: "
BUILD_NEXT_PROMPT = " BUILD next expression: "
#: What every Calculus command asks first, and then second.
CALCULUS_PROMPT = " CALCULUS {word} expression: "
CALCULUS_VARIABLE_PROMPT = " CALCULUS {word} variable: "
ENTER_VARIABLE = "Enter variable"
#: What soLve asks on its two lines. The variable line carries no number: the
#: original asks it once per variable it wants and never says which one this is.
SOLVE_PROMPT = " SOLVE expression: "
SOLVE_VARIABLE_PROMPT = " SOLVE variable: "
#: What the message line says when a solve found nothing. Nothing is appended
#: and the highlight stays where it was, so the message is the whole of the
#: answer.
NO_SOLUTIONS = "No solutions found"
#: What the message line offers while the variables are being collected. The
#: first question may be answered for all of them at once; the ones after it
#: end the list instead, since something has been chosen by then.
FIRST_VARIABLE = "Return for all or select 1: {variables}"
NEXT_VARIABLE = "Return for no more or select next: {variables}"
#: What the message line says once an answer is in.
COMPUTE_TIME = "Compute time: {elapsed}"
#: What it says while one is being worked out: which command is running, how
#: long it has been running, and the one key that will stop it.
COMPUTING = "{command}: {elapsed}   ESC aborts"
#: And what it says when that key was pressed. The original reported the time a
#: command took whether or not it finished, and so does this.
ABORTED = "Aborted after {elapsed}"
#: How each command that computes names itself while it runs.
SIMPLIFYING = "Simplifying"
APPROXIMATING = "Approximating"
SOLVING = "Solving"
#: What a command is doing while it works out which variables to offer, which
#: is a question for the engine and so can cost something, though it hardly
#: ever does.
READING = "Reading expression"
#: Seconds between two readings of the elapsed time. Fast enough that the
#: figure looks like a clock rather than a series of guesses.
TICK = 0.1
#: How long a computation has to run before it says anything at all. Nearly
#: every command answers within this, and a clock that appears and vanishes
#: inside a tenth of a second is a flicker rather than a reading - so the
#: message line is left saying what it said until there is a wait worth
#: reporting.
CLOCK_AFTER = 1.0

#: All Jump takes on its line: a label number, spaces around it allowed. A sign
#: is not refused, since the original does not refuse one either.
LABEL_NUMBER = re.compile(r"[-+]?[0-9]+")

#: What Unremove says when there is nothing to put back. Every other refusal
#: in either command is the beep alone, but this one has no dialog to leave up.
BUFFER_EMPTY = "Unremove buffer empty"

#: What the message line says when Ctrl-C has taken a copy. A clipboard leaves
#: no mark on the screen, so this is the whole of the receipt. What was copied
#: is not quoted back: the message line is one line, and an expression is as
#: long as it likes.
COPIED = "Copied the highlighted expression"

# The lines the four Declare commands read, and what the message line asks for
# on each. `default` is the variable that stands for all the unnamed ones,
# which is why the name field says so.
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

#: What a matrix element comes up offering, so that a sparse matrix is mostly
#: Enter presses. A vector element is offered nothing, as in the original.
MATRIX_ZERO = "0"

# The two Manage commands that read a line. Annotate comes up carrying what the
# entry says now - `User`, or the `Simp(#3)` a command wrote - and Ordering
# carries the order list as it stands, both selected so that typing replaces
# them. The original's line editor overwrites instead, so typing `w` over
# `x y z` there gives `w y z`; every prefilled line in Rederive comes up
# selected, and this one is not worth an exception.
ANNOTATION_PROMPT = " ANNOTATION: "
ORDER_PROMPT = " MANAGE ORDER variables: "
ENTER_ANNOTATION = "Enter annotation"
ENTER_ORDER = "Enter variables in desired order"

# Manage Substitute, which reads a line for the expression and one for each
# value. A variable's line comes up carrying the variable's own name, so that
# Enter alone leaves it alone; a subexpression has no name to offer and its
# line comes up empty, the message line saying what is being replaced.
SUBSTITUTE_PROMPT = " MANAGE SUBSTITUTE expression: "
SUBSTITUTE_VALUE_PROMPT = " MANAGE SUBSTITUTE value: "
ENTER_REPLACEMENT = "Enter replacement for {name}"
SUBEXPRESSION = "subexpression"

# Every command that names a file, and what it asks for. The original offered
# a directory listing on F1; the line completes what is typed on Tab instead,
# which answers the same question without taking the screen.
LOAD_PROMPT = " TRANSFER LOAD DERIVE file: "
LOAD_STATE_PROMPT = " TRANSFER LOAD STATE file: "
LOAD_DATA_PROMPT = " TRANSFER LOAD DATA file: "
LOAD_UTILITY_PROMPT = " TRANSFER LOAD UTILITY file: "
MERGE_PROMPT = " TRANSFER MERGE file: "
SAVE_PROMPT = " TRANSFER SAVE DERIVE file: "
SAVE_SOURCE_PROMPT = " TRANSFER SAVE {word} file: "
SAVE_STATE_PROMPT = " TRANSFER SAVE STATE file: "
DEMO_PROMPT = " TRANSFER DEMO file: "
ENTER_FILE = "Enter filename (TAB completes, opens the list)"
#: What the message line says while that list is open, being the keys that are
#: worth knowing there and are not the ones the line itself already answers.
BROWSING_FILES = "↑↓ choose   ENTER open   ESC close"
FILE_NOT_FOUND = "File not found"
CANNOT_READ = "Cannot read file"
CANNOT_WRITE = "Cannot write file"
#: What the message line says when a file held a line that would not parse.
#: The original drops such a line without a word; saying so is cheap, and a
#: worksheet quietly missing an expression is worth knowing about.
UNREADABLE = "{count} expression{s} could not be read"
#: The same for a state file, whose lines are settings rather than expressions.
UNSET = "{count} setting{s} could not be read"
#: What the state prompts offer when nothing has been read or written yet.
STATE_FILE = "rederive.ini"
#: What the message line says between two steps of a demonstration.
PRESS_ANY_KEY = "Press any key to continue"

#: The opening notice, standing across the pane until something else is drawn
#: there. The original filled its screen with a version, an address, a fax
#: number and a plea not to copy the diskettes; what is worth keeping of that
#: is the name, what the program is, which version it is, and where the help
#: is. The name block sits high and the help line low, as the original spaced
#: them, so the pane reads as a title page rather than a huddle in the middle.
GREETING = (
    "R E D E R I V E",
    "A Mathematical Assistant",
    "",
    f"Version {__version__}",
)
#: The line the notice stands at the foot of the pane, away from the name.
GREETING_FOOTER = "Press H for help"


class FileNames(Suggester):
    """What a file prompt offers as it is typed: the first name on offer.

    Textual prints the rest of that name past the cursor, dimmed, and takes it
    on Right or End. That is the shortest way to a name already half known, and
    it stands alongside the list Tab opens rather than instead of it: the offer
    costs no keys and no screen, and the list is for when one name is not
    enough to go on. Only what would grow the line is offered, so the dimmed
    text is never empty and never repeats what has been typed.

    Nothing is cached: files come and go while the program runs, and a name
    that is offered but no longer there is worse than no offer at all.
    """

    def __init__(self, suffix: str) -> None:
        # Names are matched as they are typed, this being a filesystem that
        # tells `Work.mth` from `work.mth`.
        super().__init__(use_cache=False, case_sensitive=True)
        self.suffix = suffix

    async def get_suggestion(self, value: str) -> str | None:
        found = worksheet.completions(value, self.suffix)
        return found[0] if found else None


@dataclass
class Demonstration:
    """A demonstration file part way through.

    A DMO file is a math file whose comments are the script: each expression is
    authored and then simplified, with the comment above it on the band where
    the menu goes, and the demonstration waits there for a key. Esc leaves it
    where it is, which is what `at` is for - naming the same file again picks
    the demonstration up rather than starting it over.
    """

    path: Path
    #: The comment and the expression of each step, in file order.
    steps: tuple[tuple[str, str], ...]
    at: int = 0

    @property
    def done(self) -> bool:
        return self.at >= len(self.steps)


@dataclass(frozen=True)
class Command:
    """What tells Factor and Expand apart, the questions being the same.

    `needs_amount` says whether the amount is worth asking about for the
    expression in hand, which is a different question for each: Factor asks
    unless the expression is a number it can only decompose, and Expand asks
    only where there is a denominator to factor.
    """

    #: The word the prompts are written with, and the menu title's first word.
    word: str
    #: How the message line names the command while it is running.
    running: str
    amounts: Menu
    needs_amount: Callable[[Session, str], bool]
    run: Callable[[Session, str, Amount, tuple[str, ...]], object]


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


#: The menus that answer a question a command asked rather than list commands.
AMOUNT_MENUS = (FACTOR.amounts, EXPAND.amounts)


def _text(values: Mapping[str, str | int], setting: str) -> str:
    """One answer off a Calculus dialog, which is always an expression or blank."""
    return str(values[setting]).strip()


def _filled(values: Mapping[str, str | int], *settings: str) -> str | None:
    """The first of `settings` left blank while another was filled in.

    What the paired limits refuse: both blank asks for the indefinite form and
    both filled for the definite one, so a half-answered pair is the one thing
    that is neither, and the original puts the highlight on the empty half
    rather than taking it.
    """
    blank = [setting for setting in settings if not _text(values, setting)]
    return blank[0] if blank and len(blank) < len(settings) else None


def _needed(values: Mapping[str, str | int], *settings: str) -> str | None:
    """The first of `settings` left blank, for the fields that must be answered."""
    return next((setting for setting in settings if not _text(values, setting)), None)


def _limits(values: Mapping[str, str | int], setting: str) -> tuple[str, ...]:
    """A pair of limits as arguments: both of them, or neither."""
    low, high = _text(values, f"{setting}Lower"), _text(values, f"{setting}Upper")
    return (low, high) if low and high else ()


@dataclass(frozen=True)
class Calculus:
    """What tells the seven Calculus commands apart.

    All of them ask the same two questions - which expression, and which
    variable - and then one line of their own, which is `dialog`. What comes
    back off that line becomes the arguments after the variable, in the order
    the head takes them rather than the order the line asks them in: `Taylor`
    asks for the degree first and writes the point first.

    None of them computes. Each appends `head(u, x, ...)` for a later Simplify
    to take, and `prefix` is how the status line names that: `Dif(#1,x)`.
    """

    word: str
    head: str
    prefix: str
    dialog: Dialog
    arguments: Callable[[Mapping[str, str | int]], tuple[str, ...]]
    refuses: Callable[[Mapping[str, str | int]], str | None] = lambda values: None


def _order(values: Mapping[str, str | int]) -> tuple[str, ...]:
    """The order of a derivative, left off when it is the first.

    The original writes `DIF(u, x)` for a first derivative and `DIF(u, x, n)`
    for any other, so a field still reading `1` writes no argument at all. It
    is the text that is compared and not what it is worth: the field takes an
    expression, and `2 - 1` is not the answer `1` is.
    """
    order = _text(values, "DifferentiateOrder")
    return () if order == "1" else (order,)


def _vector_values(values: Mapping[str, str | int]) -> tuple[str, ...]:
    """Where a `Vector` index starts, ends, and steps - the step if it is not one."""
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
    """The help now on screen, and what leaving it goes back to.

    `topic` is None while the subject menu is showing, which is the level Esc
    and `Resume` fall back to from a subject and out of altogether from there.
    `pages` is how many pages the subject came to when it was last laid out -
    a number that belongs to the pane's size and not to the document, so it is
    written each time the page is painted and read by the keys that turn one.

    `resume` and `message` are what the screen was doing when help was asked
    for. Help can be asked for from a line half typed, and that line has to
    come back with its text and its question intact, so what the message line
    said goes with it.
    """

    resume: str = MODE_MENU
    message: str = ENTER_OPTION
    topic: helps.Topic | None = None
    page: int = 0
    pages: int = 1


@dataclass
class Asking:
    """A Factor or Expand command part way through its questions.

    The command asks for an expression, then for variables, then for an
    amount, and each answer has to outlive the prompt that collected it.
    """

    command: Command
    request: str = ""
    #: The variables not chosen yet, in the order they are offered.
    remaining: tuple[str, ...] = ()
    #: The ones chosen so far, in the order they were, which is what makes the
    #: first of them the primary variable.
    chosen: tuple[str, ...] = ()


@dataclass
class Building:
    """A Build command part way through, which is one expression and its name.

    Every operator folds what has been built so far into a new tree, so there
    is never more than the one expression to carry: `#1 + #2` becomes the sum,
    and a `*` after it multiplies that sum by whatever comes next. `annotation`
    is built the same way, out of what each operand was called - and it is
    written flat, as the original writes it, so `#1+#2*#3` names an expression
    that is `(#1 + #2)·#3`.

    `operator` is the binary operator waiting on its right operand, and None
    whenever the operator menu is what is up.
    """

    node: Node | None = None
    annotation: str = ""
    operator: building.Operator | None = None


@dataclass
class Calculating:
    """A Calculus command part way through its questions.

    The expression and the variable are collected on lines of their own and
    have to outlive them; everything else is answered at once, on the dialog
    the command finishes on.
    """

    command: Calculus
    request: str = ""
    variable: str = ""


@dataclass
class Solving:
    """A soLve command part way through its questions.

    It asks for an expression, then for as many variables as the expression
    leaves undecided, then - in Approximate precision alone - for the interval
    to search. `wanted` is how many variable answers are still owed, which is
    one for a scalar the expression does not settle and one per equation for an
    underdetermined system.
    """

    request: str = ""
    #: How many variables are still to be answered for.
    wanted: int = 0
    #: The ones not chosen yet, most main first, which is the order the
    #: defaults walk in.
    remaining: tuple[str, ...] = ()
    #: The ones chosen so far, in the order they were.
    chosen: tuple[str, ...] = ()


@dataclass
class Substituting:
    """A Manage Substitute command part way through its questions.

    It asks for an expression and then for what to write into it: one value
    per variable, or the one replacement a highlighted subexpression takes.
    `part` says which of the two, since a subexpression has no variables to
    count and no name to ask under.
    """

    request: str = ""
    part: bool = False
    #: The variables not answered yet, most main first, which is the order the
    #: original asks about them in.
    remaining: tuple[str, ...] = ()
    #: What has been answered so far: one value per variable answered.
    values: tuple[tuple[str, str], ...] = ()


@dataclass
class Declaring:
    """A Declare Variable command part way through its questions.

    The command asks for a name, then for a value or a domain, then - for the
    two domains that have one - for an interval.
    """

    name: str
    #: The domain, once one has been chosen off the menu.
    kind: str = ""


@dataclass
class Defining:
    """A Declare Function command part way through its questions.

    It asks for a name and a definition; a definition left blank turns it into
    an arbitrary function, and it goes on asking for that function's variables
    until a blank answer ends the list.
    """

    name: str
    variables: tuple[str, ...] = ()


@dataclass
class Entering:
    """A Declare Matrix or Declare vectoR command collecting its elements.

    A vector is one row of elements and says so by carrying no row count.
    """

    columns: int
    rows: int | None = None
    elements: tuple[str, ...] = ()

    @property
    def wanted(self) -> int:
        return self.columns * (self.rows or 1)

    @property
    def place(self) -> tuple[int, int]:
        """Which element is being asked for, counting rows and columns from 1."""
        entered = len(self.elements)
        return entered // self.columns + 1, entered % self.columns + 1

    @property
    def grid(self) -> list[tuple[str, ...]]:
        """The elements as rows, which is what a matrix is written from."""
        return [
            self.elements[start : start + self.columns]
            for start in range(0, len(self.elements), self.columns)
        ]


#: What the navigation keys mean inside an Options dialog's number field.
CURSOR_MOVES = {
    "left": "back",
    "right": "forward",
    "first_sibling": "start",
    "last_sibling": "end",
}

#: The movements that walk the history rather than the line or the field they
#: are pressed on: on a prompt line, and on the dialogs whose fields name
#: expressions. The rest move the cursor of whatever is up.
ENTRY_MOVES = ("up", "down", "page_up", "page_down", "first_entry", "last_entry")

#: The movements that have to be told how tall the pane is, being about how
#: much of the history is on screen rather than about how it is put together.
PAGE_MOVES = ("page_up", "page_down")


def _takes_anything(values: dict[str, str | int]) -> str | None:
    """What a dialog whose command judges none of its fields refuses: nothing."""
    return None


def _variables_wanted(equations: int, variables: int) -> int:
    """How many variables soLve asks about, given what it was handed.

    A scalar - no equations counted, the expression being no vector of them -
    asks once unless it holds exactly one variable, which settles it. A system
    asks once per equation, and only where there are more variables than
    equations: with as many or fewer there is nothing to choose.
    """
    if not equations:
        return 0 if variables == 1 else 1
    return equations if variables > equations else 0


@dataclass(frozen=True)
class Outcome:
    """What became of one session command, whichever thread it ran on.

    Every way a command can end arrives here rather than as an exception up
    the stack, because the stack the command ran on is not the one that has to
    report it: a computation that is killed mid-bignum has no return path of
    its own.
    """

    value: object = None
    error: Exception | None = None
    seconds: float = 0.0


def _ran(run: Callable[[], object]) -> Outcome:
    """Run one session command and time it, however it ends.

    Everything is caught. A refusal, a death of the engine worker and a bug
    that got past the engine's promises are all things the message line can
    say, and none of them is worth taking the app down for.
    """
    started = time.monotonic()
    try:
        return Outcome(run(), None, time.monotonic() - started)
    except Exception as error:
        return Outcome(None, error, time.monotonic() - started)


def _elapsed(seconds: float) -> str:
    """How long something took, said as coarsely as it can be said truthfully.

    A tenth of a second is worth knowing about a command that takes a moment
    and is noise about one that takes an hour, so the unit follows the figure:
    `3.1s` while a decimal still says something, `42s` past ten of them, and
    `10m 8s` or `2h 10m 8s` once there are minutes and hours to count.
    """
    tenths = f"{seconds:.1f}"
    return f"{tenths}s" if float(tenths) < 10 else _whole(seconds)


def _whole(seconds: float) -> str:
    """How long something has been running, counted in whole seconds.

    A figure that is still climbing has no use for a tenth that is out of date
    the moment it is read, so the running clock counts `3s`, `42s`, `10m 8s`
    and `2h 10m 8s` and leaves the decimal to the time finally reported.
    """
    minutes, second = divmod(round(seconds), 60)
    if not minutes:
        return f"{second}s"
    hours, minute = divmod(minutes, 60)
    return f"{minute}m {second}s" if not hours else f"{hours}h {minute}m {second}s"


def _reported(outcome: Outcome) -> str:
    """What the message line says about a command that has finished."""
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
    # Nothing on screen belongs to anything but the pane itself.
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        # Tab completes the name where a file is being named, and steps the
        # menu everywhere else; `check_action` picks between the two, which
        # leaves the second binding to take the key when the first declines it.
        # Shift-Tab and the arrows pair off the same way: on a file prompt with
        # the list open they walk it, and everywhere else they do what they
        # always did.
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
        # Enter on an open list takes the name it points at rather than the
        # command: the file is not read until the list is out of the way, so
        # looking around can never load something by accident.
        Binding("enter", "browse_take", "Take name", priority=True, show=False),
        Binding("enter", "menu_invoke", "Invoke option", priority=True, show=False),
        # Ctrl-Enter reaches a terminal as Ctrl-J, unless it speaks the keyboard
        # protocol that tells the two apart; both spell the same command.
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
        # Ctrl-C takes a copy of the highlighted expression and Ctrl-V puts it
        # back on the line being typed, which is how a new expression is built
        # out of ones already on the worksheet without any of it being typed
        # again. The arrows still walk the highlight with a line up, which is
        # what makes the pair worth having: move onto what is wanted, copy it,
        # and paste it where it is going.
        Binding("ctrl+c", "copy_highlighted", "Copy", priority=True, show=False),
        Binding("ctrl+v", "paste_clipboard", "Paste", priority=True, show=False),
        # The original's keys for the same thing, kept because a session driven
        # from the manual still reaches for them: F3 writes the highlighted
        # expression straight onto the line, and F4 writes it fenced. Neither is
        # advertised in the help, copy and paste being what a reader now expects.
        Binding("f3", "insert_highlighted(0)", "Insert", priority=True, show=False),
        Binding("f4", "insert_highlighted(1)", "Insert ()", priority=True, show=False),
        # F1 and F2 are the Window commands worth a key of their own: the next
        # window, and the next of a window's overlays. They apply at the
        # command menu and nowhere else, the original giving F1 to Help on
        # every line that is being typed.
        Binding("f1", "window_step(1)", "Next window", priority=True, show=False),
        Binding(
            "shift+f1", "window_step(-1)", "Previous window", priority=True, show=False
        ),
        # The other half of F1: on a line being typed there is no window to
        # step to that would not take the line with it, and the key is help
        # instead. The two never apply at once, so which of the bindings takes
        # the key is `check_action`'s to say.
        Binding("f1", "help", "Help", priority=True, show=False),
        Binding("f2", "window_flip(1)", "Next overlay", priority=True, show=False),
        Binding(
            "shift+f2", "window_flip(-1)", "Previous overlay", priority=True, show=False
        ),
        # F6 hands the sideways keys back and forth between the line and the
        # highlight; Ins does the same for making room and standing on what is
        # there. The original spelled Ins as Ctrl-V too, which that key can no
        # longer be: pasting is what it means everywhere else.
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
        # The original's other spelling of the same two commands.
        Binding("ctrl+pageup", "nav('first_entry')", "Top", priority=True, show=False),
        Binding(
            "ctrl+pagedown", "nav('last_entry')", "Bottom", priority=True, show=False
        ),
        # Ctrl-Right and Ctrl-Left scroll the work area sideways over a render
        # too wide for the pane. They move no selection, so they are not nav.
        Binding("ctrl+right", "scroll_work(1)", "Right", priority=True, show=False),
        Binding("ctrl+left", "scroll_work(-1)", "Left", priority=True, show=False),
    ]

    def __init__(
        self,
        session: Session | None = None,
        settings: Settings | None = None,
        demo: Path | None = None,
        opening: str = "",
    ) -> None:
        """`demo` and `opening` are what the command line asked for.

        A demonstration named there is started as soon as there is a screen to
        run it on; anything the reading of the other files had to say stands on
        the message line in place of the usual invitation, since the user has
        not been asked anything yet and a count of lines that would not parse
        is the more useful thing to be told.
        """
        # Before the base class, which asks for the CSS variables as it starts.
        # A session brings its own settings, there being only one store.
        if settings is None:
            settings = session.settings if session is not None else Settings()
        self.settings = settings
        self.palette = Palette(self.settings)
        super().__init__()
        #: Every window there is. `session` is whichever one is active, which
        #: is what every command in this file works on.
        self.windows = Windows(session if session is not None else Session(self.settings))
        #: The work area each window is drawn in, made as windows are and kept
        #: for reuse once they close: which window a widget belongs to has to
        #: hold still, since the sideways scroll of a pane is the window's.
        self.panes: dict[Window, WorkArea] = {}
        self.spare: list[WorkArea] = []
        #: The worksheet the program opened on, and whether the opening notice
        #: still stands over it. The original painted its notice once and gave
        #: it up to whatever drew the work area next - an expression, a help
        #: page, a window command, a Clear - while a menu opened and escaped or
        #: a dialog committed left it standing, and nothing ever brought it
        #: back. `_opening_screen` is that rule.
        self.opening_session = self.windows.session
        self.greeting = True
        self.settings.watch(self._settings_changed)
        self.mode = MODE_MENU
        #: The command menu, plus whatever submenu or dialog is stacked on it.
        self.stack: list[MenuCursor | DialogEditor] = [MenuCursor(ALGEBRA)]
        self.message = opening or ENTER_OPTION
        #: The demonstration the command line named, until it has been started.
        self.opening_demo = demo
        #: What to do with the file the prompt line is naming.
        self.file_command: Callable[[str], None] | None = None
        #: The extension that command supplies, and so what Tab completes to.
        self.file_suffix = worksheet.SUFFIX
        #: The names the open list is showing, and which of them has been taken
        #: onto the line - None while the list is open but the line is still
        #: the user's own typing. The list is closed when `completions` is
        #: empty. Keeping the two together is what stops the line and the list
        #: from ever disagreeing about which name is being talked about.
        self.completions: list[str] = []
        self.completed: int | None = None
        #: What to do with the values of the dialog that stores none.
        self.answer: Callable[[dict[str, str | int]], None] | None = None
        #: Which of that dialog's fields its command will not take, when it is
        #: choosy about them.
        self.refuses: Callable[[dict[str, str | int]], str | None] = _takes_anything
        #: The block of labels the next save writes, when one was asked for.
        self.block: tuple[int | None, int | None] = (None, None)
        #: A Factor or Expand command's answers so far, while it is asking.
        self.asking: Asking | None = None
        #: What a Build has put together so far, while it is building.
        self.building: Building | None = None
        #: A Calculus command's answers so far, while it is asking.
        self.calculating: Calculating | None = None
        #: A soLve command's answers so far, while it is asking.
        self.solving: Solving | None = None
        #: A Manage Substitute's answers so far, while it is asking.
        self.substituting: Substituting | None = None
        #: The same for the three Declare commands that ask more than one
        #: question, each of which is asking whenever it is not None.
        self.declaring: Declaring | None = None
        self.defining: Defining | None = None
        self.entering: Entering | None = None
        #: The entry `Manage Annotate` is asking about, once it has been named.
        self.annotating: int | None = None
        #: The shape the next Declare Matrix and Declare vectoR offer, which is
        #: the last one entered. The original starts a matrix at three by three
        #: and offers no dimension at all until a vector has been entered.
        self.matrix_size = (3, 3)
        self.dimension: int | str = ""
        #: Whether the arrow keys belong to the line being edited rather than
        #: to the highlight. Set from `ArrowKeyMode` each time a line goes up
        #: and toggled under it by F6, so that the setting says where every
        #: command starts and the key says where this one goes on.
        self.line_edit = True
        #: Whether a character typed on a line makes room for itself. Off is
        #: the original's overwrite mode, and unlike the arrow key mode it is
        #: not reset per command: it is how the user is typing, not what the
        #: command asked.
        self.inserting = True
        #: What a Y answers, while a command is asking for one.
        self.confirm: Callable[[], None] | None = None
        #: Whether that question has the dialog it came from standing under it,
        #: which is the one thing an answer has to put away either way.
        self.confirmed_over = False
        #: The dialog `_ask` last put up, so that a command can show it again
        #: with its answer settled on it.
        self.asked: Dialog | None = None
        #: The history as it stood when Ctrl-Enter was pressed, or None while no
        #: line has been taken that way. Whatever the command adds to it is what
        #: gets simplified once the command is done.
        self.simplifying: tuple[sessions.Entry, ...] | None = None
        #: The help on screen, or None while the worksheet has the work area.
        self.helping: Helping | None = None
        #: The demonstration under way, running or suspended.
        self.demo: Demonstration | None = None
        #: How the message line names the computation now running, or None
        #: while none is.
        self.computing: str | None = None
        #: When it started, and the timer that writes the elapsed time out.
        self.started = 0.0
        self.ticker: Any = None
        #: The mode to go back to once it has finished, which is the one the
        #: command was dispatched from: a refused line has to find its prompt
        #: still up and still its own.
        self.resumed = MODE_MENU
        #: The state file last read or written, offered back by both commands.
        self.state_file = STATE_FILE
        #: Each amount menu's own cursor, kept across invocations rather than
        #: made fresh: the original opens it on whatever was chosen last.
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
            (menus.TRANSFER, "Demo"): self._command_demo,
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
            (menus.WINDOW, "Close"): self._command_window_close,
            (menus.WINDOW, "Designate"): self._command_window_designate,
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
        # The seven Calculus commands differ only in what they write and what
        # their last line asks, so they are one command told which off the word.
        for word, calculus in CALCULUS_COMMANDS.items():
            self.commands[(menus.CALCULUS, word)] = partial(
                self._command_calculus, calculus
            )
        # The four language saves differ only in which language they write, so
        # they are one command told which one off the menu word.
        for language in LANGUAGES:
            self.commands[(menus.TRANSFER_SAVE, language.word)] = partial(
                self._command_save_source, language
            )

    # -- composition -------------------------------------------------------

    def get_css_variables(self) -> dict[str, str]:
        return {**super().get_css_variables(), **self.palette.css_variables()}

    @property
    def session(self) -> Session:
        """The active window's worksheet, which is the one commands act on."""
        return self.windows.session

    @property
    def work_area(self) -> WorkArea:
        """The active window's pane."""
        return self.panes[self.windows.active]

    def compose(self) -> ComposeResult:
        yield Panes(Frame(id="frame"), id="panes")
        # Above the line it belongs to and below the work area it borrows its
        # rows from, so that the names and the name being typed read together.
        yield CompletionList(id="completions")
        yield MenuRule(id="rule")
        yield MenuBand(id="menu")
        yield FieldBand(id="fields")
        yield Horizontal(
            Static(AUTHOR_PROMPT, id="prompt-label"),
            # The prompt decides for itself what comes up selected, which is
            # the label number alone and never the `#` in front of it.
            PromptLine(id="prompt-input", select_on_focus=False),
            id="prompt-band",
        )
        yield MessageLine(id="message")
        yield StatusLine(id="status")

    def on_mount(self) -> None:
        self.query_one("#prompt-band").display = False
        self.query_one("#fields").display = False
        self.query_one("#completions").display = False
        self.refresh_screen()
        # After the first render, so that the demonstration's first step has a
        # screen to happen on rather than one still being put together.
        if self.opening_demo is not None:
            demo, self.opening_demo = self.opening_demo, None
            self._demo(str(demo))

    # -- what is on top ----------------------------------------------------

    @property
    def top(self) -> MenuCursor | DialogEditor:
        return self.stack[-1]

    @property
    def editor(self) -> DialogEditor | None:
        """The Options dialog on screen, if one is."""
        return self.top if isinstance(self.top, DialogEditor) else None

    # -- rendering ---------------------------------------------------------

    def place_windows(self) -> None:
        """Give every window a pane, put it where the window is, and paint it.

        Panes are made as they are needed and set aside rather than destroyed
        when a window closes, so that a widget once made can be handed to the
        next window that wants one.

        Every window paints its own selection, active or not: the original
        shows the highlight in every pane at once, and what says which window
        a command would act on is its number in the frame.

        A computation under way is why the painting is conditional. A resize
        reaches here at any moment, and while a command is running the thread
        running it owns the worksheet - so the panes are moved to where the
        windows now are and left showing what they showed.
        """
        panes = self.query_one(Panes)
        height, width = panes.size.height, panes.size.width
        areas = self.windows.areas(height, width)
        framed = self.windows.framed
        painting = self.mode != MODE_COMPUTE
        # The sessions are read only while nothing is computing: while a command
        # runs the thread running it owns them.
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
                # Help stands in the window it was asked from and in no other,
                # so every other pane goes on showing its own worksheet.
                page, titled = self._help_page(rect.height, rect.width)
                pane.show_help(page, rect.height, rect.width, titled)
            elif self.greeting:
                # Which is the one window there is: the notice stands only
                # while the screen is the one the program opened with.
                pane.show_greeting(GREETING, GREETING_FOOTER, rect.height, rect.width)
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
        """Whether the work area is still the one the program started with.

        Which is what the opening notice stands on: the window the program
        opened, that window's own worksheet, nothing in it, and no help page
        over it. Everything the original gave the notice up to shows here as
        one of those - an expression to draw, a help page, a second session a
        split or an overlay made, another worksheet Designate put in the window
        - bar the two Clear commands, which draw over it without changing any
        of them and so put it away themselves.
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
            # Unless the dialog is standing there answered while its command
            # asks whether it may throw expressions away: then the question is
            # what the message line has to say, not the field.
            if self.mode != MODE_CONFIRM:
                self.message = editor.message
        elif self.mode == MODE_DEMO:
            # A demonstration takes the band for its own script, the comment
            # above the expression it has just run standing where the menu was.
            self.query_one(MenuBand).say(self._demo_comment())
        else:
            cursor = self.top
            assert isinstance(cursor, MenuCursor)
            # A confirmation takes the highlight off the menu entirely.
            highlighted = None if self.mode == MODE_CONFIRM else cursor.index
            self.query_one(MenuBand).show(cursor.menu, highlighted)
        self.query_one(MessageLine).show(self.message)
        # The annotation belongs to the entry, so it follows the selection: it
        # says where the expression now highlighted came from. Beside it is the
        # file the session last read or wrote, named rather than pathed: the
        # status line is a glance, and the prompt is where the whole path is
        # offered back for editing.
        entry = self.session.selected_entry
        file = self.session.file
        self.query_one(StatusLine).show(
            "" if entry is None else entry.annotation,
            "" if file is None else file.name,
            self._flags(),
            f"{self.TITLE} {self.windows.kind}",
        )

    def _flags(self) -> str:
        """The mode words for the status line, in the order the original has.

        `Ins` stands whenever a character typed would make room for itself,
        which is a way of typing rather than something a command asks for, so
        it stands at the menu too. `Lin` says the arrow keys are the line's,
        which is only worth saying where they could have been the highlight's:
        on a line an expression is written on, and on no other.
        """
        words = ["Ins"] if self.inserting else []
        if self.line_edit and self.mode in EXPRESSION_LINES:
            words.append("Lin")
        return " ".join(words)

    def _show_flags(self) -> None:
        """Put the mode words up without disturbing the rest of the screen."""
        self.query_one(StatusLine).indicate(self._flags())

    def _set_message(self, message: str) -> None:
        self.message = message
        self.query_one(MessageLine).show(message)

    def _settings_changed(self, changed: frozenset[str]) -> None:
        """React to settings that other parts of the screen are built from.

        Color is a property of painting, so a color change repaints everything.
        The settings that decide how an expression is drawn are not: an entry
        keeps the render it was authored with, as it does in the original.
        """
        if changed & COLOR_SETTINGS:
            self.refresh_css()
            self.refresh_screen()

    def _beep(self) -> None:
        """The error beep for a key with no command, unless Mute silenced it."""
        if self.settings["Mute"] == "No":
            self.bell()

    # -- computing ---------------------------------------------------------
    #
    # Every command that reaches the math engine goes through here, and the
    # screen is modal while one does. The thread running the command owns the
    # session, so the worksheet is neither read nor repainted until the answer
    # is in and the completion handler puts it back; what does keep moving is
    # the message line's elapsed time and the status line's memory gauge, which
    # together are how a user decides whether this one is worth waiting for.

    @property
    def _abortable(self) -> bool:
        """Whether the engine behind the session can be stopped once started."""
        return hasattr(self.session.runner, "abort")

    def _compute(
        self,
        label: str,
        run: Callable[[], object],
        done: Callable[[Outcome], None],
    ) -> None:
        """Run one session command, and hand what became of it to `done`.

        The command goes on a thread wherever the engine can be aborted: the
        call blocks that thread on a pipe and leaves the event loop free to
        repaint and to hear Esc. An engine computing in this process can be
        neither aborted nor watched, so putting the screen into a mode with no
        way out of it would be a lie; that one runs inline, and `done` is
        called before this returns.
        """
        if not self._abortable:
            done(_ran(run))
            return
        self.resumed = self.mode
        self.mode = MODE_COMPUTE
        self.computing = label
        self.started = time.monotonic()
        # The line being computed from keeps the screen, and loses the keys:
        # with nothing focused every key reaches the app, where all but Esc are
        # dropped.
        self.set_focus(None)
        # Nothing is written now: the clock starts only if the command is still
        # running a second from here, which almost none of them are.
        self.ticker = self.set_interval(TICK, self._tick)
        self.run_worker(partial(self._computing_thread, run, done), thread=True)

    def _computing_thread(
        self, run: Callable[[], object], done: Callable[[Outcome], None]
    ) -> None:
        """The thread side: block on the engine, then hand back to the loop."""
        outcome = _ran(run)
        self.call_from_thread(self._computed, done, outcome)

    def _computed(self, done: Callable[[Outcome], None], outcome: Outcome) -> None:
        """Back on the event loop: stop the clock and let the command finish.

        The mode goes back to the one the command was dispatched from before
        the handler runs, since a handler that leaves its prompt up - a line
        that would not read - is leaving the mode that prompt belongs to. The
        focus is not put back here: Textual applies a focus on the next beat,
        and a handler that takes the prompt down in this one would leave a
        hidden line holding the keyboard.
        """
        if self.ticker is not None:
            self.ticker.stop()
            self.ticker = None
        self.computing = None
        self.mode = self.resumed
        done(outcome)

    def _tick(self) -> None:
        """Write the elapsed time out, which is the whole of the repainting.

        Not before there is a wait worth reporting, and not where the figure
        has not moved: a command that answers at once leaves the message line
        as it found it, and a clock counted in whole seconds changes once a
        second however often it is read.
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
        """Esc while a computation runs: take the engine away from under it."""
        abort = getattr(self.session.runner, "abort", None)
        if abort is not None:
            abort()

    # -- key routing -------------------------------------------------------

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Menu and navigation keys only apply while a command band is up.

        While a prompt line has the screen they belong to the Input, down to
        Space and Backspace. The movements that walk the history are the
        exception: a line of text has nothing vertical for them to do, so they
        go on moving the highlight with the question still up - on every line
        but the one Factor and Expand collect a variable on. On a line an
        expression is written on, subexpression arrow mode sends the sideways
        ones the same way, leaving the line no movements at all - which is the
        trade the mode is: the cursor for the run of the expression tree.

        Ctrl-Enter is the other exception: it is a form of Enter, so it applies
        wherever Enter does - on a menu, on a dialog, and on a prompt line.

        Tab is a third: it completes the name on a file prompt, and steps the
        menu everywhere else.

        F3 and F4 are a fourth: they write onto the lines an expression is
        written on, and nowhere else. F6 goes with them, there being nothing
        for it to hand the sideways keys to on a line collecting a name; the
        Alt keys, Ctrl-V and the key that toggles insert mode apply to every
        line, since they are about the text.

        Ctrl-C is not about a line at all: what it copies is the highlighted
        expression, so it applies at the menu as well as under every line, and
        only where the worksheet is the thing on screen - not in help, and not
        over a demonstration.

        The keys that walk the list of names are the fifth, and they only
        apply while that list is open. Closed, Up and Down go on walking the
        history from a file prompt as they do from any other, so opening the
        list is what decides which of the two they mean - and the list is on
        screen saying so.

        A computation under way answers none of them. It is modal down to the
        last key: only Esc means anything, and it is handled where the keys
        with no binding are.
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
        if action == "help":
            # F1 is help on a line and the next window everywhere else, which
            # is the original's split. Asking for help from inside help would
            # have nowhere to go back to.
            return self.mode in PROMPT_MODES
        if action.startswith("window_"):
            # At the command menu and nowhere else, not even under a Window
            # submenu: switching windows out from under a half-answered
            # question is not something the original offers.
            return self.mode == MODE_MENU and len(self.stack) == 1
        if action == "nav":
            if self.mode == MODE_MENU:
                return True
            if self.mode == MODE_HELP:
                # The keys that walk the history turn the pages of a subject,
                # and there are no pages to turn on the subject menu.
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
        if action == "copy_highlighted":
            return self.mode == MODE_MENU or self.mode in PROMPT_MODES
        if action in ("insert_glyph", "toggle_inserting", "paste_clipboard"):
            return self.mode in PROMPT_MODES
        return True

    def on_key(self, event: Any) -> None:
        """Keys with no binding: mnemonics, digits, and the answers to a question.

        A field that holds text takes the rest of the printable characters as
        well, since an interval bound is written with `-`, `.` and `/`.

        While a computation runs there is one key: Esc, which aborts it. Every
        other key is swallowed rather than queued, so that a handful pressed at
        a frozen-looking screen does not all happen at once when the answer
        arrives.
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
            # Any key steps the demonstration; Esc suspends it where it is.
            event.stop()
            event.prevent_default()
            if event.key == "escape":
                self._suspend_demo()
            else:
                self._demo_step()
        elif self.mode in PROMPT_MODES and event.key == "escape":
            event.stop()
            event.prevent_default()
            # An open list of names is what Esc puts away first, the name it
            # left on the line and all. Only once the line is on its own does
            # the next Esc abandon the command, so backing out of looking
            # around is not the same key press as backing out of the command.
            if self.browsing:
                self._close_list()
                self._set_message(ENTER_FILE)
            else:
                self._end_prompt(done=False)

    def _typing_text(self, character: str) -> bool:
        """Whether `character` is one the active field takes as text."""
        editor = self.editor
        return (
            character.isprintable()
            and editor is not None
            and isinstance(editor.field, TextField)
        )

    def _menu_takes(self, character: str) -> bool:
        """Whether the menu on top has a word `character` invokes.

        Letters and digits are offered to whatever is up whatever it is; this
        is for the ones that are neither, which only the Build operator menu
        answers to - there `+` and `!` are mnemonics like any other.
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

        On a field that holds text it is a character like any other, which is
        how a bound is blanked out before another is typed over it.
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
        """Backspace undoes what Space does, and otherwise steps back.

        On a field that holds text it erases the character before the cursor,
        and on one whose choices Space steps forward through it steps back
        through them rather than leaving the field.
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
        """Ctrl-Enter: what Enter does, and the expression it enters simplified.

        The manual spells the key out for a vector and a matrix, but the
        original takes it on every line an expression is entered from, and it
        is the same thing on each of them: the command runs, and what it left
        in the history is simplified after it. On the lines that enter nothing
        of their own, and everywhere a menu or a dialog is up, it is Enter.
        """
        if self.mode in ENTERING_MODES or self._enters_expression():
            self.simplifying = tuple(self.session.entries)
        if self.mode == MODE_MENU:
            self.action_menu_invoke()
        else:
            self._submitted(self.query_one("#prompt-input", Input).value)

    def _enters_expression(self) -> bool:
        """Whether Enter on what is up would append an expression to simplify.

        Which is what makes Ctrl-Enter worth pressing somewhere other than on a
        prompt line: the `Done` of Build's operator menu, and the dialog each
        Calculus command finishes on. Both append an expression nobody has
        asked to have taken any further, so both leave something for a
        Simplify after them to do.
        """
        if self.mode != MODE_MENU:
            return False
        top = self.top
        if isinstance(top, DialogEditor):
            return top.dialog.enters
        return top.menu is menus.BUILD_OPERATOR and top.word == building.DONE

    def action_menu_escape(self) -> None:
        """Leave the submenu or dialog on top, abandoning what it was set to.

        A menu that asks a question rather than listing commands is abandoned
        along with the command that put it up. There is nothing to go back to:
        the amount is the last thing Factor and Expand ask for, and the Declare
        Variable questions are put up one in place of the last, so that Esc
        leaves any of them for the Declare menu as the original does.

        In help Esc is `Resume`: up one level from a subject, and out of help
        from the subject menu.
        """
        if self.mode == MODE_HELP:
            self._help_back()
            return
        if len(self.stack) > 1:
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
        """Put the uncovered menu's own prompt back on the message line.

        A dialog does this for itself, since its prompt follows the field the
        highlight is on.
        """
        if isinstance(self.top, MenuCursor):
            self.message = self.top.message

    def _restart_menu(self) -> None:
        """Highlight the first word of the menu that has just come back up.

        The original draws a menu afresh every time it appears, so where Tab
        left the highlight is forgotten as soon as an option is entered: what
        a command or a submenu comes back to is Author.
        """
        cursor = self.top
        if isinstance(cursor, MenuCursor):
            cursor.index = 0

    def action_nav(self, movement: str) -> None:
        """The arrows walk the history, or a number field's cursor.

        In help they turn pages instead: there is no history on the screen to
        walk, and paging is what a reader wants those keys for.

        With a prompt line up, only the movements that walk the history get
        here at all: the rest are the line's own, and `check_action` leaves
        them to it.

        A dialog's selection fields take no arrow keys at all in the original,
        which falls out of this: there is no cursor in them to move.

        A dialog whose fields name expressions is the exception. There the keys
        that walk the history still walk it, and the field takes the label of
        whatever they land on - which the manual recommends over typing the
        number, as being harder to get wrong.
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
        """Move the highlight while a prompt line has the screen.

        The question stays up and the line keeps what is on it, so this is how
        the history is looked around with a command part way through asking:
        the manual's own advice for naming an expression is to move the
        highlight onto it rather than to type its number.
        """
        self._move_selection(movement)
        if self.mode in LABELLED_MODES:
            self._relabel_prompt()
        self.refresh_screen()

    def _relabel_prompt(self) -> None:
        """Put the label of where the highlight landed on the prompt line.

        Only while the label the line was offered is still selected on it. A
        cursor key collapses that selection and typing replaces it, and either
        way the line is the user's from then on and is left as it stands.
        """
        entry = self.session.selected_entry
        line = self.query_one("#prompt-input", Input)
        start, end = sorted(line.selection)
        if entry is None or start == end:
            return
        number = str(entry.number)
        line.value = f"{line.value[:start]}{number}{line.value[end:]}"
        line.selection = Selection(start, start + len(number))

    def action_copy_highlighted(self) -> None:
        """Ctrl-C: take a copy of the highlighted expression.

        What is copied is what is highlighted: the whole of an expression
        selected whole, and just the part when the highlight is inside one. It
        is the entry's own text rather than the two-dimensional render, so what
        comes back is something a line will take again.

        The copy is offered to the terminal as well as kept here, which is what
        carries an expression out to another program. Whether the terminal
        passes it on to the system clipboard is the terminal's to say, and
        nothing comes back either way, so the receipt on the message line is
        what says the key landed.
        """
        text = self.session.highlighted_text
        if text is None:
            self._beep()
            return
        self.copy_to_clipboard(text)
        self._set_message(COPIED)

    def action_paste_clipboard(self) -> None:
        """Ctrl-V: put the copy back on the line being typed.

        In at the cursor, over whatever is selected, with the cursor left after
        it, as everything else that writes on the line does. A line offering a
        label gives up its `#` as well: what is pasted stands where an
        expression goes, and a `#` in front of one says nothing.

        What comes back is the copy this program holds. Text copied in another
        program arrives instead by the terminal's own paste key, as if it had
        been typed: a terminal hands over the clipboard when its user asks and
        not when a program does.
        """
        text = self.clipboard
        if not text:
            self._beep()
            return
        self._take_offered_label()
        self._write_on_line(text)

    def action_insert_highlighted(self, fenced: int) -> None:
        """F3 and F4: write the highlighted expression onto the author line.

        What goes on the line is what is highlighted, which is the whole of an
        expression selected whole and just the part when the highlight is
        inside one. F4 fences what it writes, since the line it lands on is
        being built out of other things: `x + 1` dropped beside a factor would
        otherwise read as two terms, and the fence is what keeps it one.

        It goes in at the cursor, over whatever is selected on the line, and
        the cursor is left after it so that another can follow.

        On a line offering a label the `#` goes too, though it stands outside
        the selection a typed digit replaces: what F3 writes is an expression,
        and a `#` in front of one says nothing.
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
        """Widen the selection over the `#` of a label the line still offers.

        Only while the whole of the offering is still selected, which is what
        says the line is as the command put it up rather than as the user has
        left it.
        """
        if self.mode not in LABELLED_MODES:
            return
        line = self.query_one("#prompt-input", PromptLine)
        start, end = sorted(line.selection)
        if start == 1 and end == len(line.value) and line.value.startswith("#"):
            line.selection = Selection(0, end)

    def action_insert_glyph(self, key: str) -> None:
        """An Alt key: write the glyph it stands for onto the line."""
        self._write_on_line(ALT_GLYPHS[key])

    def _write_on_line(self, text: str) -> None:
        """Put `text` in at the cursor, over whatever is selected.

        The cursor is left after it, so that what follows goes on after it
        rather than back where the line was.
        """
        line = self.query_one("#prompt-input", PromptLine)
        line.replace(text, *sorted(line.selection))

    def action_toggle_arrow_mode(self) -> None:
        """F6: hand the sideways keys to the line, or to the highlight.

        Which way it starts is the `Options Input` setting, read afresh each
        time a line goes up; this is how it is changed for the line in hand
        without changing what the next command starts in.
        """
        self.line_edit = not self.line_edit
        self._show_flags()

    def action_toggle_inserting(self) -> None:
        """Ins: make room for what is typed, or stand on it."""
        self.inserting = not self.inserting
        self.query_one("#prompt-input", PromptLine).overwrite = not self.inserting
        self._show_flags()

    def _move_selection(self, movement: str) -> bool:
        """One movement of the highlight, and whether it moved anything.

        A page is the one movement that has to know how tall the pane is: how
        far it goes is however many expressions are on screen.
        """
        if movement in PAGE_MOVES:
            rows = max(1, self.work_area.size.height)
            return getattr(self.session, f"move_{movement}")(rows)
        return getattr(self.session, f"move_{movement}")()

    def _walk_history(self, editor: DialogEditor, movement: str) -> None:
        """Move the selection, and label the active field with where it landed."""
        if not self._move_selection(movement):
            self._beep()
        entry = self.session.selected_entry
        if entry is not None:
            editor.retype(str(entry.number))

    def action_scroll_work(self, direction: int) -> None:
        """Ctrl-Right and Ctrl-Left, over a selected render wider than the pane."""
        self.work_area.scroll_across(direction)
        self.refresh_screen()

    def _move_highlight(self, step: int) -> None:
        cursor = self.top
        assert isinstance(cursor, MenuCursor)
        cursor.move(step)

    def invoke_command(self, index: int) -> None:
        """Run the command at `index`, leaving the menu highlight alone."""
        cursor = self.top
        assert isinstance(cursor, MenuCursor)
        word = cursor.menu.words[index]
        # The color and amount menus answer a question the command beneath
        # them asked, so what is picked off them is a value and not a command.
        if cursor.menu is COLORS:
            self._chose_color(index)
            return
        if cursor.menu in AMOUNT_MENUS:
            self._chose_amount(index)
            return
        if cursor.menu is menus.BUILD_OPERATOR:
            self._chose_operator(word)
            return
        # The two help menus choose a subject and turn its pages; neither
        # picks a command, and both are answered where help is kept.
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
        """Stack a submenu or an Options dialog on what is showing."""
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
        """Stack a dialog that stores nothing, and say who wants its values.

        `refuses` names the first field the command cannot use, for the answers
        a field cannot rule out on its own: a label number is a number whatever
        the history holds, so only the command knows whether it names anything.
        """
        self.answer = answer
        self.refuses = refuses
        self.asked = dialog
        self._open(dialog)

    def _answered(self, values: dict[str, str | int]) -> None:
        """Give such a dialog's values to the command that put it up."""
        refused = self.refuses(values)
        if refused is not None:
            # The original says nothing about an answer it will not take: the
            # question stays up, with what was typed still on it to correct,
            # and the highlight back on the field that was wrong.
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
        """The first of `settings` whose label names no entry, if any does not.

        The one answer that names no entry and is still an answer is `end`,
        which stands for the place past the last one rather than for an entry.
        """
        for setting in settings:
            value = values[setting]
            if value != menus.END and self.session.numbered(int(value)) is None:
                return setting
        return None

    def _chose_color(self, index: int) -> None:
        """A color picked off the color menu belongs to the dialog beneath it."""
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
            # Unless Enter has just put up the `Other` prompt, which is not a
            # refusal but the next question.
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
        # Recorded after the change, so that the record itself is written the
        # way the new settings say to write it.
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
        """Put up the line Simplify and approX both read, which is one line.

        The highlighted expression's label is offered, and there is nothing
        else to ask: approX takes its precision from the settings rather than
        from a question, so the two commands differ only in what they run.
        """
        entry = self.session.selected_entry
        offered = "" if entry is None else f"#{entry.number}"
        self._prompt(mode, label, offered, ENTER_TO_DERIVE, keep=1)

    def _prompt(
        self, mode: str, label: str, offered: str, message: str, keep: int = 0
    ) -> None:
        """Put the prompt band up for a command that reads a line.

        What is offered comes up selected, so that typing replaces it and Enter
        alone accepts it. `keep` is how much of it stands outside the
        selection: the `#` of a label number, which typing a digit should not
        take away, and nothing at all of a file name.

        The arrow key mode is read from the settings here rather than kept from
        the last command, because that setting says what a command starts in
        and F6 says where it goes on from there.
        """
        self.mode = mode
        self.line_edit = self.settings["ArrowKeyMode"] == "LineEdit"
        self.query_one("#menu").display = False
        # A dialog may be what asked the question before this one - the label
        # `Manage Annotate` reads, the size `Declare Matrix` reads - and the
        # band it was on goes with it.
        self.query_one("#fields").display = False
        self.query_one("#prompt-band").display = True
        self.query_one("#prompt-label", Static).update(label)
        line = self.query_one("#prompt-input", PromptLine)
        # A file prompt puts its own back; nothing else on a line completes.
        line.suggester = None
        line.overwrite = not self.inserting
        self._close_list()
        line.value = offered
        line.selection = Selection(min(keep, len(offered)), len(offered))
        line.focus()
        self._set_message(message)
        self._show_flags()

    # -- Help --------------------------------------------------------------
    #
    # Help stands where the expressions stand, in the window it was asked from
    # and in no other: no border of its own, nothing overlaid and nothing
    # dimmed underneath. It has two levels and no more - a menu of subjects,
    # and a subject read a page at a time - so the whole of the navigation is
    # Next, Previous and Resume. There is nothing to search and no link to
    # follow, which is the original's bargain: a reference to read, not a web
    # to walk.
    #
    # Both menus go on the stack every other menu goes on, so leaving one
    # uncovers whatever asked for help - the command menu, or the submenu a
    # half-answered command was picked from. The prompt line is not on that
    # stack, and what it was asking is remembered in the `Helping` instead.

    def _command_help(self) -> None:
        """The Help command: the subject menu, over the worksheet."""
        self._open_help(MODE_MENU, ENTER_OPTION)

    def action_help(self) -> None:
        """F1 on a line being typed, which keeps the line for the way back."""
        self._open_help(self.mode, self.message)

    def _open_help(self, resume: str, message: str) -> None:
        """Put help up, over whatever the screen was doing."""
        self.helping = Helping(resume, message)
        self.mode = MODE_HELP
        self.stack.append(MenuCursor(menus.HELP))
        # A line goes off the screen with its text intact: nothing is read off
        # it until it comes back, and nothing writes to it while it is gone. An
        # open list of names goes for good, since it stands in the rows help is
        # about to take and the name it was offering is already on the line.
        self._close_list()
        self.query_one("#prompt-band").display = False
        self.set_focus(None)
        self.message = ENTER_OPTION
        self.refresh_screen()

    def _chose_subject(self, word: str) -> None:
        """A subject off the subject menu, or the Resume at the end of it."""
        assert self.helping is not None
        if word == helps.RESUME:
            self._help_back()
            return
        self.helping.topic = helps.BY_WORD[word]
        self.helping.page = 0
        self.stack.append(MenuCursor(menus.HELP_PAGES[word]))
        self.refresh_screen()

    def _read_help(self, word: str) -> None:
        """Next, Previous or Resume, off the menu a subject is read under."""
        if word == helps.RESUME:
            self._help_back()
        else:
            self._turn_help(1 if word == helps.NEXT else -1)

    def _turn_help(self, step: int) -> None:
        """Turn a page; past the last one is out to the subject menu.

        Which is the original's way out and worth keeping: a subject read
        through ends where it was started from, with no key pressed to leave
        it. The other end stops instead - there is nothing before the first
        page, and coming round to the last would be a jump and not a turn.
        """
        assert self.helping is not None
        page = self.helping.page + step
        if page >= self.helping.pages:
            self._help_back()
            return
        self.helping.page = max(0, page)
        self.refresh_screen()

    def _help_back(self) -> None:
        """Resume: up one level, and out of help from the top one."""
        assert self.helping is not None
        self.stack.pop()
        if self.helping.topic is not None:
            self.helping.topic = None
            self.helping.page = 0
            self._restart_menu()
            self.refresh_screen()
            return
        resume, message = self.helping.resume, self.helping.message
        self.helping = None
        self.mode = resume
        if resume in PROMPT_MODES:
            # The line comes back as it was left, down to where the cursor
            # stood in it: only the band was taken away.
            self.query_one("#prompt-band").display = True
            self.query_one("#prompt-input", Input).focus()
        else:
            self._restart_menu()
        self.message = message
        self.refresh_screen()

    def _help_page(self, rows: int, width: int) -> tuple[list[str], bool]:
        """The page now showing, laid out for a pane this size, and its title.

        How a subject paginates is a question about the pane and not about the
        document, so it is answered here, as the page is painted: a window
        that has just been split or resized re-wraps and paginates again, and
        the page in hand is held inside whatever the new size came to. The
        flag says whether the first row is a title, the subject menu carrying
        its own heading inside a centred document rather than above one.
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

        Nothing is offered on the line: the command is for going somewhere
        else, so the label of where the highlight already is would be no use.

        A history with nothing in it has nothing to jump to, and the original
        asks nothing at all: no line, no message, just the beep.
        """
        if not self.session.entries:
            self._beep()
            return
        self._prompt(MODE_JUMP, JUMP_PROMPT, "", menus.ENTER_LABEL)

    def _jumped(self, answer: str) -> None:
        """Enter on the jump line: a label number and nothing else.

        A line with nothing on it asks for nothing, and leaves the highlight
        where it was. Anything that is not a number, and any number past the
        last label, is refused: the beep, and the line left up to be corrected.
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
        """Ask which block of expressions to take out, offering the highlighted one.

        A history with nothing in it has no block to name, and the original
        asks nothing at all: no dialog, no message, just the beep.
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
        """Put the last removal back, asking where when there is a choice.

        An empty buffer is the one thing either command says anything about. An
        empty history is no choice of where, so the expressions simply go back.
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
        """Ask which block to rearrange and where to put it.

        A history with nothing in it has no block to name, and the original
        asks nothing at all: no dialog, no message, just the beep.
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
        """The expression is settled: work out what is left to ask.

        Fewer than two variables is no choice, so none is asked for. The
        amount is asked for only where it would make a difference, which each
        command decides for itself - so a number is decomposed and a sum is
        expanded on the spot, with no question put at all.

        Which variables the expression holds is the engine's answer and not a
        walk over the tree, so even this question goes through the computing
        thread: converting an expression is where a hostile one detonates, and
        it would detonate here as readily as in the command itself. It is a
        round trip of well under a millisecond on anything sane.
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
        """The variables are known: put the next question, or run the command."""
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
        """Put up the next variable question."""
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

        An empty line ends the list, and so does choosing the last variable
        there was: either way what is left to ask is the amount. A name that is
        not one of the variables on offer is refused and the question put
        again, which is what the original does with it.
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
        """The variables are in: put the amount question, or run the command."""
        pending = self.asking
        assert pending is not None
        if pending.command.needs_amount(self.session, pending.request):
            self._ask_amount()
        else:
            self._answer()

    def _ask_amount(self) -> None:
        """Stack the amount menu on the command menu, the prompt band done."""
        pending = self.asking
        assert pending is not None
        self._hide_prompt()
        self.mode = MODE_MENU
        self.stack.append(self.amounts[pending.command.amounts])
        self.message = pending.command.amounts.message
        self.refresh_screen()

    def _chose_amount(self, index: int) -> None:
        """The amount is the last question, so choosing one runs the command."""
        pending = self.asking
        assert pending is not None
        self.amounts[pending.command.amounts].index = index
        self.stack.pop()
        self._answer(pending.command.amounts.words[index])

    def _answer(self, amount: str | None = None) -> None:
        """Run the command, and say how long the answer took."""
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
        """The answer is in, or the reason there is none.

        A refusal is reported rather than corrected: the line parsed when it
        was collected, so one is all but unreachable, and there may be no
        prompt left to put the cursor in.
        """
        self._end_prompt(_reported(outcome))

    # -- Build -------------------------------------------------------------

    def _command_build(self) -> None:
        """Ask for the first operand, offering the highlighted expression."""
        self.building = Building()
        self._ask_operand(MODE_BUILD, BUILD_PROMPT)

    def _ask_operand(self, mode: str, label: str) -> None:
        """Put up one of the two operand lines, offering what is highlighted.

        Which is the whole of what makes Build worth having: the label comes up
        already on the line, the arrow keys move it to another expression, and
        F6 walks into one - so an operand buried in an expression costs no
        typing at all.
        """
        entry = self.session.selected_entry
        offered = "" if entry is None else f"#{entry.number}"
        self._prompt(mode, label, offered, ENTER_TO_DERIVE, keep=1)

    def _build_operand(self, text: str) -> None:
        """The first operand: resolve what it names, and ask for an operator."""
        pending = self.building
        assert pending is not None
        resolved = self._resolved(text)
        if resolved is None:
            return
        pending.node, pending.annotation = resolved
        self._ask_operator(0)

    def _build_next(self, text: str) -> None:
        """The right operand of the binary operator that asked for it."""
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
        """What an operand line names, and what an annotation calls it.

        None when there is nothing to go on with: a line with nothing on it
        abandons the whole command, as it does on the author line, and one that
        does not read is left up to be corrected.
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
        """Stack the operator menu, the prompt band done with for now.

        It opens on `+` the first time and on `Done` from then on, which is
        where the original leaves it: an operand has just been settled, so the
        likeliest next word is the one that finishes.
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
        """One word off the operator menu: apply it, or finish the expression.

        A unary operator has everything it needs and folds the expression on
        the spot, leaving the menu up for another. A binary one takes the menu
        down and asks for its right operand - down rather than aside, so that
        Esc on that line abandons the command as the original does, instead of
        stepping back to the menu.
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
        """Done: append what was built, unsimplified, and give the menu back."""
        assert pending.node is not None
        node, annotation = pending.node, pending.annotation
        self.building = None
        if self._wanted_simplified():
            self._compute(
                SIMPLIFYING,
                partial(self.session.build, node, annotation, True),
                self._appended,
            )
            return
        started = time.monotonic()
        try:
            self.session.build(node, annotation)
        except DeriveSyntaxError as error:
            # All but unreachable: every operand was read as it was collected,
            # and what is written here is what they were read from.
            self._end_prompt(str(error))
            return
        took = _elapsed(time.monotonic() - started)
        self._end_prompt(COMPUTE_TIME.format(elapsed=took))

    def _wanted_simplified(self) -> bool:
        """Whether the key that finished the command was Ctrl-Enter.

        Answered here rather than left to the general path, because these two
        commands do not append an expression and then simplify it. The
        original enters one expression and not two: what Build put together is
        simplified instead of being appended, and the annotation records both
        steps at once. So the mark Ctrl-Enter left is taken back off, and there
        is nothing for the return to the menu to find.
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
        """The expression is settled: work out which variable to offer.

        The one the original offers is the primary variable of what was named,
        which is the engine's answer rather than a walk over the tree - and so
        is asked for the way Factor asks for the same list, on the computing
        thread. An expression with no variables in it is offered none, and the
        line comes up empty.
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
        """The variables are known: ask which one, offering the most main."""
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
        """The variable, which has to be one: anything else is not an answer.

        The original refuses such a line without saying anything at all - no
        message, no beep - and leaves it up with the cursor back at the start,
        so that the name can be typed over.
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
        """Which field of the last line the command in hand will not take."""
        pending = self.calculating
        assert pending is not None
        return pending.command.refuses(values)

    def _calculus_answered(self, values: dict[str, str | int]) -> None:
        """The last line is answered: write the expression and append it.

        Nothing is computed, so nothing goes to the computing thread: the
        command writes a head around what was named and stops, which is what
        leaves a Simplify after it something to do.
        """
        pending = self.calculating
        assert pending is not None
        self.calculating = None
        command = pending.command
        arguments = (pending.variable, *command.arguments(values))
        run = partial(
            self.session.calculus,
            command.head,
            command.prefix,
            pending.request,
            arguments,
        )
        if self._wanted_simplified():
            self._compute(SIMPLIFYING, partial(run, simplified=True), self._appended)
            return
        started = time.monotonic()
        try:
            run()
        except DeriveSyntaxError as error:
            # What a field holding something that is not an expression comes
            # to. The line is gone by now, so this is said and not corrected.
            self._end_prompt(str(error))
            return
        took = _elapsed(time.monotonic() - started)
        self._end_prompt(COMPUTE_TIME.format(elapsed=took))

    # -- soLve -------------------------------------------------------------
    #
    # Up to three questions, and how many of them get asked depends on the
    # answers rather than on a table: an expression, then a variable at a time
    # for as many as the expression leaves undecided, then the interval - in
    # Approximate precision alone, that being the one mode that searches.
    #
    # The count of variable questions is the one thing here that is not Factor
    # under another name. A scalar equation in one variable settles it and asks
    # nothing; in none or in several it asks once. A system asks once per
    # equation, and only where there are more variables than equations to go
    # round - a square or overdetermined system has no choice to offer.

    def _command_solve(self) -> None:
        """Ask which expression to solve, offering the highlighted one."""
        self.solving = Solving()
        entry = self.session.selected_entry
        offered = "" if entry is None else f"#{entry.number}"
        self._prompt(MODE_SOLVE, SOLVE_PROMPT, offered, ENTER_TO_DERIVE, keep=1)

    def _solve_expression(self, request: str) -> None:
        """The expression is settled: work out what is left to ask.

        How many equations it is comes off the tree and costs nothing; which
        variables it holds is the engine's answer, and goes to the computing
        thread for the reason Factor's does - converting an expression is where
        a hostile one detonates.
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
        """The variables are known: put the next question, or run the command."""
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
        """Put up the variable line, offering the most main one still free."""
        pending = self.solving
        assert pending is not None
        offered = pending.remaining[0] if pending.remaining else ""
        self._prompt(
            MODE_SOLVE_VARIABLE, SOLVE_VARIABLE_PROMPT, offered, ENTER_VARIABLE
        )

    def _solve_variable(self, text: str) -> None:
        """One answer to a variable question.

        A line that is not a variable is refused the way every other variable
        line refuses one: in silence, with the cursor back at the start so the
        name can be typed over.
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
        """The interval, which only Approximate precision has anything to do with."""
        if str(self.settings["Precision"]) != "Approximate":
            self._run_solve()
            return
        self._hide_prompt()
        self.mode = MODE_MENU
        self._ask(menus.SOLVE_BOUNDS, self._solve_bounded, self._solve_refuses)

    def _solve_refuses(self, values: dict[str, str | int]) -> str | None:
        """Neither bound may be left blank: an interval needs two ends."""
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
        """The solutions are in, or there were none.

        No solutions appends nothing and leaves the highlight where it was, so
        the message line is what says so - there being no beep machinery here
        and, in the original, nothing else said either.
        """
        if outcome.error is None and not outcome.value:
            self._end_prompt(NO_SOLUTIONS)
            return
        self._appended(outcome)

    # -- Declare -----------------------------------------------------------
    #
    # All four commands end the same way: they hand the session what they
    # collected, it writes the expression and authors it, and the command menu
    # comes back up. What differs is the questions, and how many of them the
    # answers call for.

    def _command_declare_variable(self) -> None:
        self._prompt(MODE_VARIABLE_NAME, VARIABLE_NAME_PROMPT, "", ENTER_VARIABLE_NAME)

    def _variable_name(self, text: str) -> None:
        """The name is in: ask what to declare it as.

        A blank line names nothing and abandons the command. A name that
        cannot be declared - a pre-defined function or constant - is refused
        and the question put again, as the original puts it again.
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
        """Stack the domain menu, opened on what the variable is already."""
        pending = self.declaring
        assert pending is not None
        self._open_question(
            menus.DECLARE_VARIABLE,
            self.session.declared_as(pending.name),
            menus.SELECT_DOMAIN.format(name=pending.name),
        )

    def _chose_domain(self, word: str) -> None:
        """A value, or one of the four domains.

        Complex and Nonscalar have no interval to ask about, so choosing one
        of them is the last answer the command needs.
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
        """One of the named intervals, the whole line, or bounds to be entered."""
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
        """The value, or a blank line to leave the variable unassigned."""
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
        """The definition, or a blank line to declare an arbitrary function."""
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
        """One of an arbitrary function's variables; a blank line ends the list.

        A function with no variables is no function: the original writes
        `g :=`, which leaves the name an unassigned variable.
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
        """Put up the next element question, one element at a time."""
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

        A blank line abandons the whole command, as it does in the original:
        there is no such thing as a vector with a hole in it.
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
        """Put the name question again, with nothing on the line."""
        self._beep()
        self.query_one("#prompt-input", Input).value = ""

    def _declared(self, declare: Callable[[], object]) -> None:
        """Run one Declare command, and leave the command menu up.

        A line that does not read leaves the prompt up to be corrected, as an
        authored one does. Only the commands that read a line can raise; the
        ones answered off a menu cannot, since a name and a bound were judged
        before they were taken.
        """
        try:
            declare()
        except DeriveSyntaxError as error:
            self._refused(error)
            return
        self._end_prompt()

    def _open_question(self, menu: Menu, word: str, asks: str) -> None:
        """Stack a menu that asks a question, opened on the word already true.

        It replaces whichever question was up rather than covering it, so that
        Esc leaves the whole command however far through it is.
        """
        self._hide_prompt()
        self.mode = MODE_MENU
        index = menu.words.index(word) if word in menu.words else 0
        self.stack.append(MenuCursor(menu, index, asks))
        self.message = asks
        self.refresh_screen()

    # -- Manage ------------------------------------------------------------
    #
    # The four commands here that are not settings screens. Three of them append
    # nothing to the history: renumbering, annotating and ordering are things
    # done to the session rather than expressions derived from it. Substitute is
    # the one that derives one, and it asks the way Factor and Expand do.

    def _command_renumber(self) -> None:
        """Put the labels back in sequence. No question, no message, no record.

        It runs on the keystroke and leaves the command menu up. A history
        already in sequence, and one with nothing in it, come to the same
        thing: the command runs and there is nothing to see.
        """
        self.session.renumber()
        self._done_with_menu()

    def _command_annotate(self) -> None:
        """Ask which expression to annotate, offering the highlighted one.

        A history with nothing in it has nothing to annotate, so the command is
        over before it has asked anything.
        """
        entry = self.session.selected_entry
        if entry is None:
            self._done_with_menu()
            return
        self._ask(menus.annotate_entry(entry.number), self._annotating, self._annotated)

    def _annotated(self, values: dict[str, str | int]) -> str | None:
        return self._names_nothing(values, "AnnotateExpression")

    def _annotating(self, values: dict[str, str | int]) -> None:
        """The expression is named: ask what to say about it.

        The line comes up carrying what the entry says now, so that Enter alone
        leaves it as it was and typing replaces it.
        """
        number = int(values["AnnotateExpression"])
        entry = self.session.numbered(number)
        assert entry is not None
        self.annotating = number
        self._prompt(
            MODE_ANNOTATION, ANNOTATION_PROMPT, entry.annotation, ENTER_ANNOTATION
        )

    def _annotate(self, text: str) -> None:
        """Enter on the annotation line: what the entry now says it came from.

        A blank line is a blank annotation and not an abandoned command, there
        being no other way to take one back; Esc is what abandons, and it
        leaves the Manage menu up as it does on any prompt line.
        """
        number, self.annotating = self.annotating, None
        assert number is not None
        self.session.annotate(number, text)
        self._end_prompt()

    def _command_ordering(self) -> None:
        """Ask for the variable order list, offering the one in force."""
        self._prompt(
            MODE_ORDER, ORDER_PROMPT, " ".join(self.session.order), ENTER_ORDER
        )

    def _ordered(self, text: str) -> None:
        """Enter on the ordering line: the variables in the order wanted.

        A line that is not a list of variable names is refused with the beep
        and left up to be corrected, which is what every refused answer gets.
        Nothing is recorded: unlike the four Manage screens that set something,
        this one appends no expression to the history.
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
        """The expression is settled: work out what is left to ask.

        A highlighted subexpression is one question, whatever it holds. A whole
        expression is one question per variable - and an expression with no
        variables in it leaves nothing to ask and nothing to append, so the
        command is over where it stands, the Manage menu still up.
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
        """The variables are known: ask about the first, or find none to ask about."""
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
        """Put up the next value question.

        A variable's own name is what its line offers, so that Enter alone
        leaves that variable standing. A subexpression is offered nothing:
        there is no name for it, and the message line says what is being
        replaced instead.
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

        A line that does not read is refused and left up to be corrected, as an
        authored one is. A blank line leaves that variable alone and asks about
        the next; on the subexpression question, where there is nothing else to
        ask, it abandons the command rather than appending a copy.
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
        """Write the values in, and say how long the answer took."""
        self.substituting = None
        started = time.monotonic()
        try:
            run()
        except DeriveSyntaxError as error:
            # Every line was read as it was collected, so this is all but
            # unreachable; there may be no prompt left to put the cursor in.
            self._end_prompt(str(error))
            return
        took = _elapsed(time.monotonic() - started)
        self._end_prompt(COMPUTE_TIME.format(elapsed=took))

    # -- Transfer ----------------------------------------------------------

    def _command_save(self) -> None:
        """Write the worksheet as a math file, asking the block first when Some."""
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
        """Write the worksheet as source code, asking the block first when Some.

        The same command as `Transfer Save Derive` down to the block it writes;
        only the notation and the extension differ.
        """
        self._begin_save(
            SAVE_SOURCE_PROMPT.format(word=language.word.upper()),
            partial(self._save_source, language),
            language.suffix,
        )

    def _begin_save(
        self, label: str, command: Callable[[str], None], suffix: str
    ) -> None:
        """Ask for a save's block if the Range option says Some, then its name.

        A history with nothing in it is nothing to write, and the original
        simply declines: no prompt, no message, just the beep.
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
        """Put the prompt band up for a command that names a file.

        The file the session last used comes up on the line, as it does in the
        original, so that saving twice over the same name is one keystroke. The
        two State commands offer their own file instead: a settings file is not
        the worksheet, and neither is a name for the other.

        The line completes what is typed on it, offering the files the command
        can use - the ones ending in `suffix` - and the directories to look in.
        """
        self.file_command = command
        self.file_suffix = suffix
        self._close_list()
        if offered is None:
            offered = "" if self.session.file is None else str(self.session.file)
        self._prompt(MODE_FILE, label, offered, ENTER_FILE)
        self.query_one("#prompt-input", Input).suggester = FileNames(suffix)

    # -- naming a file: completing it, and looking around for it -------------

    @property
    def browsing(self) -> bool:
        """Whether the list of names is open over the line being typed."""
        return bool(self.completions)

    def action_complete_file(self) -> None:
        """Tab on a file prompt: fill the name in as far as it goes, then look.

        Tab does one thing at a time and leaves the screen saying which. It
        writes out as much of the matching names as they all share and opens
        the list of them; where one name matches and nothing else does, it
        takes that name outright and there is nothing to open. With the list
        already up and nothing left to share, it steps to the next name in it.

        So Tab only ever moves forward, and what it will do next is on the
        screen rather than in a mode nobody can see. Nothing on offer is the
        beep, as a key with no command is.
        """
        line = self.query_one("#prompt-input", Input)
        if self.browsing:
            # Typing can narrow an open list to names that share more than the
            # line does; writing that out comes before stepping through them.
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
            # One name and one only: take it, and if it is a directory, look
            # straight inside it rather than asking for another Tab first.
            self._take(line, found[0])
            return
        shared = commonprefix(found)
        if len(shared) > len(line.value):
            self._put(line, shared)
        self._open_list(found, self._at_in(found, line.value))

    def action_browse_next(self) -> None:
        """Down, or Tab with the list up: the next name, wrapping round."""
        self._highlight(0 if self.completed is None else self.completed + 1)

    def action_browse_previous(self) -> None:
        """Up or Shift-Tab: the previous name, wrapping round.

        The key the old completion never had. Overshooting the name wanted no
        longer means going the whole way round the list to reach it again.
        """
        self._highlight(-1 if self.completed is None else self.completed - 1)

    def action_browse_take(self) -> None:
        """Enter on an open list: take the name it points at.

        A directory is gone into and the list stays up on what is inside it,
        which is how the tree is walked down. A file closes the list and stands
        on the line, where the next Enter is the one that reads it - so no file
        is ever opened by the same keystroke that chose it.
        """
        at = 0 if self.completed is None else self.completed
        self._take(self.query_one("#prompt-input", Input), self.completions[at])

    def action_browse_page(self, by: int) -> None:
        """Page Up and Page Down: a screenful of names at a time.

        They stop at the ends rather than wrapping, so that a long directory
        can be paged through without falling off either end of it.
        """
        at = 0 if self.completed is None else self.completed
        rows = self.query_one(CompletionList).visible_rows(len(self.completions))
        self._highlight(at + by * rows, wrap=False)

    def _highlight(self, at: int, wrap: bool = True) -> None:
        """Point the list at one of its names, and put that name on the line.

        Every name in the list starts with what was typed - that is what put it
        there - so writing one onto the line only ever carries the line
        further, and never takes back a letter the user chose.
        """
        count = len(self.completions)
        self.completed = at % count if wrap else max(0, min(at, count - 1))
        self._put(
            self.query_one("#prompt-input", Input), self.completions[self.completed]
        )
        self._draw_list()

    def _take(self, line: Input, name: str) -> None:
        """Accept `name` onto the line, going on inside it if it is a directory.

        Taking a directory reopens the list on what is in it, which is what
        walking down a tree is: one key per level, each level on the screen.
        """
        self._put(line, name)
        self._close_list()
        self._set_message(ENTER_FILE)
        if not name.endswith("/"):
            return
        found = worksheet.matches(name, self.file_suffix)
        if found:
            self._open_list(found, None)
        else:
            # A directory holding nothing this command can read is a dead end,
            # and the beep is what says so: the list would otherwise just go.
            self._beep()

    def _at_in(self, found: list[str], value: str) -> int | None:
        """Which name a freshly opened list points at, if it points at one.

        A line typed out into one of the names exactly points at that name, so
        that the list says where the line already is. Anything shorter points
        at nothing: the list is open to be looked at, and the line stays the
        user's until a key takes a name from it.
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
            # Every name in the list is in the same directory, so any of them
            # says which one the list is showing.
            _directory_of(self.completions[0]),
        )

    def _close_list(self) -> None:
        """Put the list away, whatever became of the name it was offering."""
        self.completions, self.completed = [], None
        listing = self.query(CompletionList)
        if listing:
            listing.first().display = False

    def _put(self, line: Input, name: str) -> None:
        """Write a name onto the line, the cursor after it, ready to go on."""
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
        """Run a save, leaving the name up to be corrected if it will not write."""
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
        """Read a file into the session, leaving the line up if it cannot be."""
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
        """Say what went wrong and leave the name up to be corrected."""
        self._beep()
        self._set_message(message)

    # -- Transfer Clear ----------------------------------------------------

    def _command_clear_expressions(self) -> None:
        self._clearing(self.session.clear_expressions)

    def _command_clear_all(self) -> None:
        self._clearing(self.session.clear_all)

    def _command_clear_variables(self) -> None:
        """Forget every assigned value. Nothing on screen goes, so nothing asks."""
        self.session.clear_variables()
        self._done_with_menu()

    def _command_clear_functions(self) -> None:
        self.session.clear_functions()
        self._done_with_menu()

    def _clearing(self, clear: Callable[[], None]) -> None:
        """Throw the history away, having asked first.

        The two commands that take expressions out ask before they do, in the
        same words Quit asks in. A history with nothing in it is nothing to
        lose, so there is nothing to ask about.

        Either way the pane is drawn empty afterwards, which is what takes the
        opening notice off a worksheet that had nothing to clear: the original's
        Clear drew over its notice too.
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

    def _demo(self, name: str) -> None:
        """Start the demonstration in `name`, or pick up the suspended one.

        Naming the file a suspended demonstration came from resumes it where it
        stopped, which is what the manual means by issuing another Demo command
        to carry on. Any other name starts that file from its first step.
        """
        path = worksheet.path_of(name, worksheet.DEMO_SUFFIX)
        if self.demo is None or self.demo.path != path or self.demo.done:
            try:
                steps = worksheet.demonstration(path)
            except FileNotFoundError:
                self._refuse_file(FILE_NOT_FOUND)
                return
            except OSError:
                self._refuse_file(CANNOT_READ)
                return
            self.demo = Demonstration(path, steps)
        # A demonstration simplifies each step as it takes it, so a file named
        # with Ctrl-Enter has nothing left to ask for.
        self.simplifying = None
        self._hide_prompt()
        del self.stack[1:]
        self._restart_menu()
        self._demo_step()

    def _demo_step(self) -> None:
        """Author the next expression and set its Simplify going.

        A step that does not parse is passed over rather than stopping the
        demonstration: a script is not a worksheet, and there is nothing on the
        line to correct. The Simplify is dispatched rather than run, so the
        step's own computation is as abortable as any other; what happens when
        it lands is `_demo_simplified`.
        """
        demo = self.demo
        assert demo is not None
        while not demo.done:
            _, text = demo.steps[demo.at]
            demo.at += 1
            try:
                self.session.author(text)
            except DeriveSyntaxError:
                continue
            request = f"#{self.session.entries[-1].number}"
            self._compute(
                SIMPLIFYING,
                partial(self.session.simplify, request),
                self._demo_simplified,
            )
            return
        self._end_demo()

    def _demo_simplified(self, outcome: Outcome) -> None:
        """The step's answer is in: show it and wait on a key.

        A refusal passes the step over as an unparsable one does. Anything
        else - an abort above all - ends the demonstration where it stands,
        which is where a suspended one is picked up from.
        """
        if isinstance(outcome.error, DeriveSyntaxError):
            self._demo_step()
            return
        if outcome.error is not None:
            self._end_demo(_reported(outcome))
            return
        self.mode = MODE_DEMO
        self.message = PRESS_ANY_KEY
        self.refresh_screen()

    def _suspend_demo(self) -> None:
        """Esc leaves the demonstration where it is, to be picked up later."""
        self._end_demo()

    def _end_demo(self, message: str = ENTER_OPTION) -> None:
        self.mode = MODE_MENU
        self.query_one("#menu").display = True
        self._return_to_menu(message)

    def _demo_comment(self) -> str:
        """The comment of the step now showing, which is the band's whole line."""
        demo = self.demo
        if demo is None or not demo.at:
            return ""
        return demo.steps[demo.at - 1][0]

    # -- Window ------------------------------------------------------------
    #
    # Eight commands over a tree of windows, of which the app owns only the
    # screen half: which pane a window is drawn in, and how big. The tree
    # itself, the numbering and the geometry are `model.windows`.
    #
    # Two of them can throw a worksheet away - Close, and Designate, which
    # makes a window over rather than converting it - so both ask before they
    # do, with the answered field still on the band.
    #
    # A window of a plot type is offered because the original offers it on the
    # same field, and refused because this program draws no plots. Everything
    # else works for every window: splitting copies the worksheet, and the two
    # copies are two derivations from there on.

    def _command_window_split(self, vertical: bool) -> None:
        """Ask where to cut the active window in two."""
        size = self._active_size(vertical)
        low, high = windows.split_range(vertical, size)
        if high < low:
            # Nowhere left to put a divider. The original refuses in silence.
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
            # There has to be a window. The original refuses outright: no
            # prompt, no message, the Window menu still up.
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
        """Next and Previous, which ask nothing and run on the keystroke."""
        self.windows.step(direction)
        self._done_with_menu()

    def _command_window_flip(self, direction: int = 1) -> None:
        """Flip, which brings the next overlay of the active window up."""
        self.windows.active.flip(direction)
        self._done_with_menu()

    def _command_window_designate(self) -> None:
        """Ask what type to make the active window, opening on what it is."""
        self._ask(
            menus.window_type(menus.WINDOW_DESIGNATE, self.windows.kind),
            self._designate_window,
        )

    def _designate_window(self, values: dict[str, str | int]) -> None:
        kind = str(values["WindowType"])
        if kind != windows.ALGEBRA:
            self._unplotted(kind)
            return
        if self.session.entries:
            self._confirm_over(values, partial(self._redesignate, kind))
            return
        self._redesignate(kind)

    def _redesignate(self, kind: str) -> None:
        """Make the active window over as a fresh window of `kind`.

        Which empties it: a type is not something a window is converted to,
        it is what a new window in the same rectangle would be.
        """
        for dropped in self.windows.designate(kind, self._new_session()):
            dropped.discard()
        self._done_with_menu()

    def _command_window_open(self) -> None:
        """Ask what type to overlay on the active window, offering Algebra."""
        self._ask(
            menus.window_type(menus.WINDOW_OPEN, windows.ALGEBRA), self._open_window
        )

    def _open_window(self, values: dict[str, str | int]) -> None:
        kind = str(values["WindowType"])
        if kind != windows.ALGEBRA:
            self._unplotted(kind)
            return
        self.windows.open(kind, self._new_session())
        self._done_with_menu()

    def _unplotted(self, kind: str) -> None:
        """Refuse a plot window, which is a window with nothing to put in it."""
        self.message = f"{kind}: not implemented yet"
        self.refresh_screen()

    def _new_session(self) -> Session:
        """An empty worksheet for a window that has just been made.

        It shares the settings, which belong to the system and not to a window,
        and the engine, which is the one child process every window computes in.
        """
        return Session(self.settings, self.session.runner)

    def _active_size(self, vertical: bool) -> int:
        """How wide or tall the active window's interior is on screen now."""
        panes = self.query_one(Panes)
        rect = self.windows.interior(
            self.windows.active, panes.size.height, panes.size.width
        )
        return rect.width if vertical else rect.height

    def action_window_step(self, direction: int) -> None:
        """F1 and Shift-F1: the next and the previous window, without the menu."""
        self.windows.step(direction)
        self.refresh_screen()

    def action_window_flip(self, direction: int) -> None:
        """F2 and Shift-F2: the next and the previous overlay of this window."""
        self.windows.active.flip(direction)
        self.refresh_screen()

    # -- confirmations -----------------------------------------------------

    def _ask_confirm(self, confirmed: Callable[[], None]) -> None:
        """Put a Y/N question up, and say what a Y does."""
        self.confirm = confirmed
        self.mode = MODE_CONFIRM
        self.message = ABANDON_PROMPT
        self.refresh_screen()

    def _confirm_over(
        self, values: dict[str, str | int], confirmed: Callable[[], None]
    ) -> None:
        """Ask the question with the dialog that was just answered still up.

        Which is where the original leaves it: the window number or the type
        entered stands on the band, in the parentheses that say it is settled
        rather than being typed, while the message line asks whether the
        expressions may go.
        """
        assert self.asked is not None
        editor = DialogEditor(self.asked, self.settings)
        editor.values.update(values)
        # No field is live, so every one of them prints the way a field that is
        # not the active one does.
        editor.active = -1
        self.stack.append(editor)
        self.confirmed_over = True
        self._ask_confirm(confirmed)

    def _answer_confirm(self, yes: bool) -> None:
        """Y runs the command; anything else leaves the menu it was picked from."""
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
        """A command that ran is finished with, and so is the path to it."""
        del self.stack[1:]
        self._restart_menu()
        self._return_to_menu(ENTER_OPTION)

    def _command_quit(self) -> None:
        """Leave, asking once for all the windows rather than once for each."""
        if not any(session.entries for session in self.windows.sessions()):
            self.exit()
            return
        self._ask_confirm(self.exit)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Typing on a file prompt with the list open: narrow it to what fits.

        The list follows the line rather than going stale on it, so a letter
        cuts it down and a Backspace opens it back out. Backspacing over a
        separator is what walks back up the tree, the list showing the parent
        directory again as soon as the separator is gone.

        The names this widget writes come back through here too; they are the
        ones the list is already pointing at, and nothing needs doing for them.
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
        """Enter on a prompt line: the whole line, wherever the cursor is."""
        event.stop()
        self._submitted(event.value)

    def _submitted(self, value: str) -> None:
        """Take the line, whichever command's question it is answering."""
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
        """Enter on a file prompt: a line with nothing on it names nothing."""
        if not name.strip():
            self._end_prompt(done=False)
            return
        assert self.file_command is not None
        self.file_command(name)

    def _author(self, text: str) -> None:
        """Enter the line as a new expression.

        A line with nothing on it enters nothing and gives the menu back, the
        way Esc does. A line that does not parse is not entered. Derive says
        so, beeps, and leaves the line up with the cursor where it stopped
        reading - which may be anywhere to the right of the mistake.
        """
        if not text.strip():
            self._end_prompt(done=False)
            return
        try:
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
        self, request: str, run: Callable[[str], object], label: str
    ) -> None:
        """Run `run` on what the line asks for, and say how long it took.

        An empty line asks for nothing, so it leaves the history alone. A line
        that does not read stays up to be corrected, as an authored one does -
        which is why the refusal is judged where the answer is, rather than
        here: the line is not read until the command runs, and by then the
        command is on a thread of its own.
        """
        if not request.strip():
            self._end_prompt()
            return
        self._compute(label, partial(run, request), self._derived)

    def _derived(self, outcome: Outcome) -> None:
        """The answer is in, or the reason there is none."""
        if isinstance(outcome.error, DeriveSyntaxError):
            self._refused(outcome.error)
            return
        self._end_prompt(_reported(outcome))

    def _refused(self, error: DeriveSyntaxError) -> None:
        """Say where the line stopped reading, and leave it up.

        The line is focused as well as pointed at, because a command that ran
        on a thread took the keyboard off it while it ran, and a refused line
        is the user's again.
        """
        self._beep()
        self._set_message(str(error))
        line = self.query_one("#prompt-input", Input)
        line.focus()
        line.cursor_position = error.offset

    def _hide_prompt(self) -> None:
        """Give the screen back to the menu band, whatever comes next."""
        self._close_list()
        self.query_one("#prompt-input", Input).value = ""
        self.query_one("#prompt-band").display = False
        self.query_one("#menu").display = True
        self.set_focus(None)

    def _end_prompt(self, message: str = ENTER_OPTION, done: bool = True) -> None:
        """The command is over, answered or abandoned: put a menu back up.

        A command that ran is finished with, so the whole path it was reached
        by goes: `Transfer Save Derive` leaves the command menu up, not the
        Transfer Save menu it was picked from. Abandoning the line instead
        leaves that menu where it was, which is what Esc does.
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
        """What the command entered, for a line taken with Ctrl-Enter.

        Empty when there is nothing to simplify: the line was taken with Enter,
        or the command entered no expression after all. A command that enters
        more than one - a Transfer that reads a file - has every one of them
        simplified, in the order they came in.

        What is new is told by identity rather than by label, since `Transfer
        Load` starts the numbering again and a number proves nothing.
        """
        before, self.simplifying = self.simplifying, None
        if before is None:
            return []
        known = {id(entry) for entry in before}
        return [entry for entry in self.session.entries if id(entry) not in known]

    def _simplify_entered(self, entered: list[sessions.Entry]) -> None:
        """Simplify each of them in turn, on the computing thread.

        An abort lands in the middle of the list and is left there: what has
        already been simplified stands, and the rest is abandoned, which is the
        same bargain every other abort makes.
        """
        for entry in entered:
            self.session.simplify(f"#{entry.number}")

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
        # Simplifying is the last thing that happened, so its compute time is
        # what the message line says - unless the command has something of its
        # own to say, as a file that would not all read has. Going wrong is
        # always worth saying, whatever the command had in mind.
        if message == ENTER_OPTION or outcome.error is not None:
            message = _reported(outcome)
        self._menu_again(message)

    def _menu_again(self, message: str) -> None:
        """The command menu has the screen again, with something to say."""
        self.mode = MODE_MENU
        self.message = message
        self.refresh_screen()


def _last_name(name: str) -> str:
    """A completion's last component, the directory separator kept if it has one."""
    trimmed = name.rstrip("/")
    return trimmed.rpartition("/")[2] + name[len(trimmed) :]


def _directory_of(name: str) -> str:
    """The directory a completion is in, as the list's title names it.

    What the names in the list have in common, which is the part of the path
    the list does not repeat down its rows. A name with no directory in front
    of it is in the one the program was started in.
    """
    head = name.rstrip("/").rpartition("/")[0]
    return f"{head}/" if head else "./"


def _color_index(editor: DialogEditor) -> int:
    """Where the color menu opens: on the color the field is already set to."""
    current = editor.value(editor.field)
    return next(
        (index for index in range(len(COLORS.words)) if menus.color_at(index) == current),
        0,
    )

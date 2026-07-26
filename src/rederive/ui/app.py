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

A command that needs a line of its own - Author, Simplify and Factor an
expression, Jump a label number, the Transfer commands a file name - takes the
screen with the prompt band instead, which is one Input with a label in front
of it. The mode says which command the line belongs to, and so what Enter does
with it. A command that runs is finished with, so it leaves the command menu up
rather than the submenu it was picked from; Esc, which abandons the line
instead, leaves that submenu where it was.

Two commands take the band for something other than a menu. A question with a
Y or N for an answer - Quit, and the two Clear commands that throw expressions
away - leaves the menu up with its highlight off, and `confirm` holds what a Y
does. A demonstration takes the band outright: the comment above each step
stands where the menu words go, and any key runs the next step.

Factor asks more than one question, and asks them in both forms: an expression,
then a factorization variable at a time on the prompt band, then an amount off
a menu stacked on the command menu. What has been answered so far lives in
`Factoring` until there is enough to run the command, since each prompt is gone
by the time the next one is up. How many questions get asked depends on the
answers - a number is decomposed without being asked about at all - so the
sequence is in the handlers rather than in a table.

The Declare commands are the same shape and each keeps its own answers -
`Declaring`, `Defining`, `Entering` - but their questions replace each other
rather than stacking, because the original abandons the whole command from
whichever of them is up. A question answered off a menu therefore pops it
before putting the next one, and one Esc always lands back on the Declare menu.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Input, Static
from textual.widgets.input import Selection

from rederive.engine import Amount
from rederive.model import session as sessions
from rederive.model import state, worksheet
from rederive.model.session import Session
from rederive.model.settings import (
    ChoiceField,
    Dialog,
    DialogEditor,
    Settings,
    TextField,
)
from rederive.syntax import LANGUAGES, DeriveSyntaxError, Language
from rederive.ui import menu as menus
from rederive.ui.menu import ALGEBRA, COLORS, ENTER_OPTION, Menu, MenuCursor
from rederive.ui.theme import COLOR_SETTINGS, Palette
from rederive.ui.widgets import (
    FieldBand,
    MenuBand,
    MenuRule,
    MessageLine,
    StatusLine,
    WorkArea,
)

MODE_MENU = "menu"
MODE_AUTHOR = "author"
MODE_SIMPLIFY = "simplify"
MODE_FACTOR = "factor"
MODE_FACTOR_VARIABLE = "factor_variable"
MODE_JUMP = "jump"
MODE_FILE = "file"
MODE_CONFIRM = "confirm"
MODE_VARIABLE_NAME = "variable_name"
MODE_VARIABLE_VALUE = "variable_value"
MODE_FUNCTION_NAME = "function_name"
MODE_FUNCTION_VALUE = "function_value"
MODE_FUNCTION_VARIABLE = "function_variable"
MODE_ELEMENT = "element"
MODE_DEMO = "demo"

#: The modes in which the prompt band has the screen, and the keys with it.
PROMPT_MODES = (
    MODE_AUTHOR,
    MODE_SIMPLIFY,
    MODE_FACTOR,
    MODE_FACTOR_VARIABLE,
    MODE_JUMP,
    MODE_FILE,
    MODE_VARIABLE_NAME,
    MODE_VARIABLE_VALUE,
    MODE_FUNCTION_NAME,
    MODE_FUNCTION_VALUE,
    MODE_FUNCTION_VARIABLE,
    MODE_ELEMENT,
)

#: The prompt lines the highlight can still be walked under: every one of them
#: but Factor's variable line, which takes no notice of those keys. What to
#: factor has been settled by then, so moving the highlight would say nothing
#: about what the command is about to do.
WALKED_MODES = tuple(mode for mode in PROMPT_MODES if mode != MODE_FACTOR_VARIABLE)

#: The prompt lines that come up naming an expression, and so take the label of
#: wherever the walk lands - for as long as the label they offered is still
#: selected on them. A line typed on is a line the user has taken over. The
#: Declare lines take an expression too but are offered nothing to begin with,
#: so there is nothing on them to keep in step with the highlight.
LABELLED_MODES = (MODE_SIMPLIFY, MODE_FACTOR)

# F1 does nothing yet: Help is not part of this milestone. The wording is the
# original's, kept so the screen reads right.
ENTER_EXPRESSION = "Enter expression (press F1 for help)"
ENTER_TO_SIMPLIFY = "Enter expression"
ENTER_TO_FACTOR = "Enter expression"
#: What Quit and the two Clear commands that throw expressions away both ask.
ABANDON_PROMPT = "Abandon expressions (Y/N)?"
AUTHOR_PROMPT = " AUTHOR expression: "
SIMPLIFY_PROMPT = " SIMPLIFY expression: "
FACTOR_PROMPT = " FACTOR expression: "
JUMP_PROMPT = " JUMP to: "
FACTOR_VARIABLE_PROMPT = " FACTOR variable {number}: "
#: What the message line offers while Factor is collecting variables. The
#: first question may be answered for all of them at once; the ones after it
#: end the list instead, since something has been chosen by then.
FIRST_VARIABLE = "Return for all or select 1: {variables}"
NEXT_VARIABLE = "Return for no more or select next: {variables}"
#: What the message line says once an answer is in.
COMPUTE_TIME = "Compute time: {seconds:.1f} seconds"

#: All Jump takes on its line: a label number, spaces around it allowed. A sign
#: is not refused, since the original does not refuse one either.
LABEL_NUMBER = re.compile(r"[-+]?[0-9]+")

#: What Unremove says when there is nothing to put back. Every other refusal
#: in either command is the beep alone, but this one has no dialog to leave up.
BUFFER_EMPTY = "Unremove buffer empty"

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

# Every command that names a file, and what it asks for. F1 does not list a
# directory yet, as it does not offer help yet; the wording is the original's,
# kept so the screen reads right.
LOAD_PROMPT = " TRANSFER LOAD DERIVE file: "
LOAD_STATE_PROMPT = " TRANSFER LOAD STATE file: "
LOAD_DATA_PROMPT = " TRANSFER LOAD DATA file: "
LOAD_UTILITY_PROMPT = " TRANSFER LOAD UTILITY file: "
MERGE_PROMPT = " TRANSFER MERGE file: "
SAVE_PROMPT = " TRANSFER SAVE DERIVE file: "
SAVE_SOURCE_PROMPT = " TRANSFER SAVE {word} file: "
SAVE_STATE_PROMPT = " TRANSFER SAVE STATE file: "
DEMO_PROMPT = " TRANSFER DEMO file: "
ENTER_FILE = "Enter filename"
ENTER_FILE_TO_READ = "Enter filename (press F1 for list)"
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


@dataclass
class Factoring:
    """A Factor command part way through its questions.

    The command asks for an expression, then for factorization variables, then
    for an amount, and each answer has to outlive the prompt that collected it.
    """

    request: str
    #: The variables not chosen yet, in the order they are offered.
    remaining: tuple[str, ...]
    #: The ones chosen so far, in the order they were, which is what makes the
    #: first of them the primary factorization variable.
    chosen: tuple[str, ...] = ()


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


class RederiveApp(App[None]):
    """A single full-screen Algebra pane."""

    CSS_PATH = "rederive.tcss"
    TITLE = "Rederive"
    AUTO_FOCUS = None
    # Nothing on screen belongs to anything but the pane itself.
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("tab", "menu_next", "Next option", priority=True, show=False),
        Binding("space", "menu_space", "Next option", priority=True, show=False),
        Binding(
            "shift+tab", "menu_previous", "Previous option", priority=True, show=False
        ),
        Binding("backspace", "menu_erase", "Previous option", priority=True, show=False),
        Binding("delete", "menu_delete", "Delete", priority=True, show=False),
        Binding("enter", "menu_invoke", "Invoke option", priority=True, show=False),
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
        self, session: Session | None = None, settings: Settings | None = None
    ) -> None:
        # Before the base class, which asks for the CSS variables as it starts.
        # A session brings its own settings, there being only one store.
        if settings is None:
            settings = session.settings if session is not None else Settings()
        self.settings = settings
        self.palette = Palette(self.settings)
        super().__init__()
        self.session = session if session is not None else Session(self.settings)
        self.settings.watch(self._settings_changed)
        self.mode = MODE_MENU
        #: The command menu, plus whatever submenu or dialog is stacked on it.
        self.stack: list[MenuCursor | DialogEditor] = [MenuCursor(ALGEBRA)]
        self.message = ENTER_OPTION
        #: What to do with the file the prompt line is naming.
        self.file_command: Callable[[str], None] | None = None
        #: What to do with the values of the dialog that stores none.
        self.answer: Callable[[dict[str, str | int]], None] | None = None
        #: What that dialog's command will take, when it is choosy about it.
        self.accepts: Callable[[dict[str, str | int]], bool] | None = None
        #: The block of labels the next save writes, when one was asked for.
        self.block: tuple[int | None, int | None] = (None, None)
        #: The Factor command's answers so far, while it is asking for them.
        self.factoring: Factoring | None = None
        #: The same for the three Declare commands that ask more than one
        #: question, each of which is asking whenever it is not None.
        self.declaring: Declaring | None = None
        self.defining: Defining | None = None
        self.entering: Entering | None = None
        #: The shape the next Declare Matrix and Declare vectoR offer, which is
        #: the last one entered. The original starts a matrix at three by three
        #: and offers no dimension at all until a vector has been entered.
        self.matrix_size = (3, 3)
        self.dimension: int | str = ""
        #: What a Y answers, while a command is asking for one.
        self.confirm: Callable[[], None] | None = None
        #: The demonstration under way, running or suspended.
        self.demo: Demonstration | None = None
        #: The state file last read or written, offered back by both commands.
        self.state_file = STATE_FILE
        #: The amount menu's own cursor, kept across invocations rather than
        #: made fresh: the original opens it on whatever was chosen last.
        self.amount = MenuCursor(menus.AMOUNT, menus.AMOUNT.words.index("Rational"))
        # Commands not listed here are present and navigable but inert.
        self.commands: dict[tuple[Menu, str], Callable[[], None]] = {
            (ALGEBRA, "Author"): self._command_author,
            (ALGEBRA, "Factor"): self._command_factor,
            (ALGEBRA, "Jump"): self._command_jump,
            (ALGEBRA, "Quit"): self._command_quit,
            (ALGEBRA, "Remove"): self._command_remove,
            (ALGEBRA, "Simplify"): self._command_simplify,
            (ALGEBRA, "Unremove"): self._command_unremove,
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
        }
        # The four language saves differ only in which language they write, so
        # they are one command told which one off the menu word.
        for language in LANGUAGES:
            self.commands[(menus.TRANSFER_SAVE, language.word)] = partial(
                self._command_save_source, language
            )

    # -- composition -------------------------------------------------------

    def get_css_variables(self) -> dict[str, str]:
        return {**super().get_css_variables(), **self.palette.css_variables()}

    def compose(self) -> ComposeResult:
        yield WorkArea(id="work")
        yield MenuRule(id="rule")
        yield MenuBand(id="menu")
        yield FieldBand(id="fields")
        yield Horizontal(
            Static(AUTHOR_PROMPT, id="prompt-label"),
            # The prompt decides for itself what comes up selected, which is
            # the label number alone and never the `#` in front of it.
            Input(id="prompt-input", select_on_focus=False),
            id="prompt-band",
        )
        yield MessageLine(id="message")
        yield StatusLine(id="status")

    def on_mount(self) -> None:
        self.query_one("#prompt-band").display = False
        self.query_one("#fields").display = False
        self.refresh_screen()

    # -- what is on top ----------------------------------------------------

    @property
    def top(self) -> MenuCursor | DialogEditor:
        return self.stack[-1]

    @property
    def editor(self) -> DialogEditor | None:
        """The Options dialog on screen, if one is."""
        return self.top if isinstance(self.top, DialogEditor) else None

    # -- rendering ---------------------------------------------------------

    def refresh_screen(self) -> None:
        """Push the whole model state at the widgets."""
        self.query_one(WorkArea).show(
            self.session.entries, self.session.selected, self.session.selection_rect()
        )
        editor = self.editor
        self.query_one("#menu").display = (
            editor is None and self.mode not in PROMPT_MODES
        )
        self.query_one("#fields").display = editor is not None
        if editor is not None:
            self.query_one(FieldBand).show(editor)
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
            "" if entry is None else entry.annotation, "" if file is None else file.name
        )

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

    # -- key routing -------------------------------------------------------

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Menu and navigation keys only apply while a command band is up.

        While a prompt line has the screen they belong to the Input, down to
        Space and Backspace. The movements that walk the history are the
        exception: a line of text has nothing vertical for them to do, so they
        go on moving the highlight with the question still up.
        """
        if action.startswith("menu_") or action == "scroll_work":
            return self.mode == MODE_MENU
        if action == "nav":
            return self.mode == MODE_MENU or (
                self.mode in WALKED_MODES and parameters[0] in ENTRY_MOVES
            )
        return True

    def on_key(self, event: Any) -> None:
        """Keys with no binding: mnemonics, digits, and the answers to a question.

        A field that holds text takes the rest of the printable characters as
        well, since an interval bound is written with `-`, `.` and `/`.
        """
        if self.mode == MODE_MENU:
            character = event.character
            if character and (character.isalnum() or self._typing_text(character)):
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
            self._end_prompt(done=False)

    def _typing_text(self, character: str) -> bool:
        """Whether `character` is one the active field takes as text."""
        editor = self.editor
        return (
            character.isprintable()
            and editor is not None
            and isinstance(editor.field, TextField)
        )

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
        """Backspace edits a number field, and otherwise steps back."""
        editor = self.editor
        if editor is not None and editor.erase():
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

    def action_menu_escape(self) -> None:
        """Leave the submenu or dialog on top, abandoning what it was set to.

        A menu that asks a question rather than listing commands is abandoned
        along with the command that put it up. There is nothing to go back to:
        the Factor amount is the last thing that command asks for, and the
        Declare Variable questions are put up one in place of the last, so that
        Esc leaves any of them for the Declare menu as the original does.
        """
        if len(self.stack) > 1:
            left = self.stack.pop()
            if isinstance(left, MenuCursor) and left.menu is menus.AMOUNT:
                self.factoring = None
            self.declaring = None
            self._ask_again()
            self.refresh_screen()

    def _ask_again(self) -> None:
        """Put the uncovered menu's own prompt back on the message line.

        A dialog does this for itself, since its prompt follows the field the
        highlight is on.
        """
        if isinstance(self.top, MenuCursor):
            self.message = self.top.message

    def action_nav(self, movement: str) -> None:
        """The arrows walk the history, or a number field's cursor.

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

    def _move_selection(self, movement: str) -> bool:
        """One movement of the highlight, and whether it moved anything.

        A page is the one movement that has to know how tall the pane is: how
        far it goes is however many expressions are on screen.
        """
        if movement in PAGE_MOVES:
            rows = max(1, self.query_one(WorkArea).size.height)
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
        """Ctrl-Right and Ctrl-Left, over a render wider than the pane."""
        self.query_one(WorkArea).scroll_half_pane(direction)

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
        if cursor.menu is menus.AMOUNT:
            self._chose_amount(index)
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
        accepts: Callable[[dict[str, str | int]], bool] | None = None,
    ) -> None:
        """Stack a dialog that stores nothing, and say who wants its values.

        `accepts` is what the command will not take, for the answers a field
        cannot rule out on its own: a label number is a number whatever the
        history holds, so only the command knows whether it names anything.
        """
        self.answer = answer
        self.accepts = accepts
        self._open(dialog)

    def _answered(self, values: dict[str, str | int]) -> None:
        """Give such a dialog's values to the command that put it up."""
        if self.accepts is not None and not self.accepts(values):
            # The original says nothing about an answer it will not take: the
            # question stays up, with what was typed still on it to correct.
            self._beep()
            self.refresh_screen()
            return
        self.stack.pop()
        answer, self.answer, self.accepts = self.answer, None, None
        assert answer is not None
        answer(values)

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
        entry = self.session.selected_entry
        offered = "" if entry is None else f"#{entry.number}"
        self._prompt(MODE_SIMPLIFY, SIMPLIFY_PROMPT, offered, ENTER_TO_SIMPLIFY, keep=1)

    def _prompt(
        self, mode: str, label: str, offered: str, message: str, keep: int = 0
    ) -> None:
        """Put the prompt band up for a command that reads a line.

        What is offered comes up selected, so that typing replaces it and Enter
        alone accepts it. `keep` is how much of it stands outside the
        selection: the `#` of a label number, which typing a digit should not
        take away, and nothing at all of a file name.
        """
        self.mode = mode
        self.query_one("#menu").display = False
        self.query_one("#prompt-band").display = True
        self.query_one("#prompt-label", Static).update(label)
        line = self.query_one("#prompt-input", Input)
        line.value = offered
        line.selection = Selection(min(keep, len(offered)), len(offered))
        line.focus()
        self._set_message(message)

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

    def _both_labels(self, values: dict[str, str | int]) -> bool:
        return all(
            self.session.numbered(int(values[setting])) is not None
            for setting in ("RemoveFirst", "RemoveLast")
        )

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

    def _one_label(self, values: dict[str, str | int]) -> bool:
        before = values["UnremoveBefore"]
        return before == menus.END or self.session.numbered(int(before)) is not None

    def _unremove(self, values: dict[str, str | int]) -> None:
        before = values["UnremoveBefore"]
        self.session.unremove(None if before == menus.END else int(before))
        self._return_to_menu(ENTER_OPTION)

    # -- Factor ------------------------------------------------------------

    def _command_factor(self) -> None:
        """Ask which expression to factor, offering the highlighted one."""
        entry = self.session.selected_entry
        offered = "" if entry is None else f"#{entry.number}"
        self._prompt(MODE_FACTOR, FACTOR_PROMPT, offered, ENTER_TO_FACTOR, keep=1)

    def _factor(self, request: str) -> None:
        """The expression is settled: work out what is left to ask.

        A number has nothing to ask about and is decomposed on the spot. One
        variable is no choice, so only the amount is asked for. Two or more and
        the variables are asked for first, one question at a time.
        """
        if not request.strip():
            self._end_prompt()
            return
        try:
            decomposes = self.session.decomposes(request)
            variables = self.session.factor_variables(request)
        except DeriveSyntaxError as error:
            self._refused(error)
            return
        if decomposes:
            self._factored(request, ())
            return
        self.factoring = Factoring(request, variables)
        if len(variables) < 2:
            self._ask_amount()
        else:
            self._ask_variable()

    def _ask_variable(self) -> None:
        """Put up the next factorization variable question."""
        pending = self.factoring
        assert pending is not None
        number = len(pending.chosen) + 1
        offer = FIRST_VARIABLE if number == 1 else NEXT_VARIABLE
        self._prompt(
            MODE_FACTOR_VARIABLE,
            FACTOR_VARIABLE_PROMPT.format(number=number),
            "",
            offer.format(variables=",".join(pending.remaining)),
        )

    def _factor_variable(self, answer: str) -> None:
        """One answer to a factorization variable question.

        An empty line ends the list, and so does choosing the last variable
        there was: either way what is left to ask is the amount. A name that is
        not one of the variables on offer is refused and the question put
        again, which is what the original does with it.
        """
        pending = self.factoring
        assert pending is not None
        answer = answer.strip()
        if not answer:
            self._ask_amount()
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
            self._ask_amount()

    def _ask_amount(self) -> None:
        """Stack the amount menu on the command menu, the prompt band done."""
        self._hide_prompt()
        self.mode = MODE_MENU
        self.stack.append(self.amount)
        self.message = menus.AMOUNT.message
        self.refresh_screen()

    def _chose_amount(self, index: int) -> None:
        """The amount is the last question, so choosing one runs the command."""
        self.amount.index = index
        self.stack.pop()
        pending = self.factoring
        self.factoring = None
        assert pending is not None
        self._factored(pending.request, pending.chosen, menus.AMOUNT.words[index])

    def _factored(
        self, request: str, variables: tuple[str, ...], amount: str | None = None
    ) -> None:
        """Run Factor, and say how long the answer took."""
        started = time.monotonic()
        try:
            self.session.factor(request, Amount(amount or "Rational"), variables)
        except DeriveSyntaxError as error:
            # The line parsed when it was collected, so this is all but
            # unreachable; there may be no prompt left to put the cursor in.
            self._end_prompt(str(error))
            return
        self._end_prompt(COMPUTE_TIME.format(seconds=time.monotonic() - started))

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

    def _both_bounds(self, values: dict[str, str | int]) -> bool:
        return all(
            self.session.is_bound(str(values[setting]))
            for setting in ("BoundLow", "BoundHigh")
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

    # -- Transfer ----------------------------------------------------------

    def _command_save(self) -> None:
        """Write the worksheet as a math file, asking the block first when Some."""
        self._begin_save(SAVE_PROMPT, self._save)

    def _chose_block(
        self, label: str, command: Callable[[str], None], values: dict[str, str | int]
    ) -> None:
        self.block = (int(values["SaveFirst"]), int(values["SaveLast"]))
        self._ask_file(label, ENTER_FILE, command)

    def _command_save_source(self, language: Language) -> None:
        """Write the worksheet as source code, asking the block first when Some.

        The same command as `Transfer Save Derive` down to the block it writes;
        only the notation and the extension differ.
        """
        self._begin_save(
            SAVE_SOURCE_PROMPT.format(word=language.word.upper()),
            partial(self._save_source, language),
        )

    def _begin_save(self, label: str, command: Callable[[str], None]) -> None:
        """Ask for a save's block if the Range option says Some, then its name.

        A history with nothing in it is nothing to write, and the original
        simply declines: no prompt, no message, just the beep.
        """
        if not self.session.entries:
            self._beep()
            return
        if self.settings["SaveRange"] != "Some":
            self.block = (None, None)
            self._ask_file(label, ENTER_FILE, command)
            return
        numbers = [entry.number for entry in self.session.entries]
        self._ask(
            menus.save_block(numbers[0], numbers[-1]),
            partial(self._chose_block, label, command),
        )

    def _command_load(self) -> None:
        self._ask_file(LOAD_PROMPT, ENTER_FILE_TO_READ, self._load)

    def _command_merge(self) -> None:
        self._ask_file(MERGE_PROMPT, ENTER_FILE_TO_READ, self._merge)

    def _command_load_data(self) -> None:
        self._ask_file(LOAD_DATA_PROMPT, ENTER_FILE_TO_READ, self._load_data)

    def _command_load_utility(self) -> None:
        self._ask_file(LOAD_UTILITY_PROMPT, ENTER_FILE_TO_READ, self._load_utility)

    def _command_save_state(self) -> None:
        self._ask_file(
            SAVE_STATE_PROMPT, ENTER_FILE, self._save_state, offered=self.state_file
        )

    def _command_load_state(self) -> None:
        self._ask_file(
            LOAD_STATE_PROMPT,
            ENTER_FILE_TO_READ,
            self._load_state,
            offered=self.state_file,
        )

    def _ask_file(
        self,
        label: str,
        message: str,
        command: Callable[[str], None],
        offered: str | None = None,
    ) -> None:
        """Put the prompt band up for a command that names a file.

        The file the session last used comes up on the line, as it does in the
        original, so that saving twice over the same name is one keystroke. The
        two State commands offer their own file instead: a settings file is not
        the worksheet, and neither is a name for the other.
        """
        self.file_command = command
        if offered is None:
            offered = "" if self.session.file is None else str(self.session.file)
        self._prompt(MODE_FILE, label, offered, message)

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
        """
        if not self.session.entries:
            clear()
            self._done_with_menu()
            return
        self._ask_confirm(lambda: (clear(), self._done_with_menu()))

    # -- Transfer Demo -----------------------------------------------------

    def _command_demo(self) -> None:
        self._ask_file(DEMO_PROMPT, ENTER_FILE_TO_READ, self._demo)

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
        self._hide_prompt()
        del self.stack[1:]
        self._demo_step()

    def _demo_step(self) -> None:
        """Author and simplify the next expression, and wait on it.

        A step that does not parse is passed over rather than stopping the
        demonstration: a script is not a worksheet, and there is nothing on the
        line to correct.
        """
        demo = self.demo
        assert demo is not None
        while not demo.done:
            _, text = demo.steps[demo.at]
            demo.at += 1
            try:
                self.session.author(text)
                self.session.simplify(f"#{self.session.entries[-1].number}")
            except DeriveSyntaxError:
                continue
            self.mode = MODE_DEMO
            self.message = PRESS_ANY_KEY
            self.refresh_screen()
            return
        self._end_demo()

    def _suspend_demo(self) -> None:
        """Esc leaves the demonstration where it is, to be picked up later."""
        self._end_demo()

    def _end_demo(self) -> None:
        self.mode = MODE_MENU
        self.query_one("#menu").display = True
        self._return_to_menu(ENTER_OPTION)

    def _demo_comment(self) -> str:
        """The comment of the step now showing, which is the band's whole line."""
        demo = self.demo
        if demo is None or not demo.at:
            return ""
        return demo.steps[demo.at - 1][0]

    # -- confirmations -----------------------------------------------------

    def _ask_confirm(self, confirmed: Callable[[], None]) -> None:
        """Put a Y/N question up, and say what a Y does."""
        self.confirm = confirmed
        self.mode = MODE_CONFIRM
        self.message = ABANDON_PROMPT
        self.refresh_screen()

    def _answer_confirm(self, yes: bool) -> None:
        """Y runs the command; anything else leaves the menu it was picked from."""
        confirmed, self.confirm = self.confirm, None
        self.mode = MODE_MENU
        if yes and confirmed is not None:
            confirmed()
            return
        self._return_to_menu(ENTER_OPTION)

    def _done_with_menu(self) -> None:
        """A command that ran is finished with, and so is the path to it."""
        del self.stack[1:]
        self._return_to_menu(ENTER_OPTION)

    def _command_quit(self) -> None:
        if not self.session.entries:
            self.exit()
            return
        self._ask_confirm(self.exit)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Enter on a prompt line: the whole line, wherever the cursor is."""
        event.stop()
        if self.mode == MODE_SIMPLIFY:
            self._simplify(event.value)
        elif self.mode == MODE_FACTOR:
            self._factor(event.value)
        elif self.mode == MODE_FACTOR_VARIABLE:
            self._factor_variable(event.value)
        elif self.mode == MODE_JUMP:
            self._jumped(event.value)
        elif self.mode == MODE_FILE:
            self._named(event.value)
        elif self.mode == MODE_VARIABLE_NAME:
            self._variable_name(event.value)
        elif self.mode == MODE_VARIABLE_VALUE:
            self._variable_value(event.value)
        elif self.mode == MODE_FUNCTION_NAME:
            self._function_name(event.value)
        elif self.mode == MODE_FUNCTION_VALUE:
            self._function_value(event.value)
        elif self.mode == MODE_FUNCTION_VARIABLE:
            self._function_variable(event.value)
        elif self.mode == MODE_ELEMENT:
            self._element(event.value)
        else:
            self._author(event.value)

    def _named(self, name: str) -> None:
        """Enter on a file prompt: a line with nothing on it names nothing."""
        if not name.strip():
            self._end_prompt(done=False)
            return
        assert self.file_command is not None
        self.file_command(name)

    def _author(self, text: str) -> None:
        """Enter the line as a new expression.

        A line that does not parse is not entered. Derive says so, beeps, and
        leaves the line up with the cursor where it stopped reading - which may
        be anywhere to the right of the mistake.
        """
        if not text.strip():
            return
        try:
            self.session.author(text)
        except DeriveSyntaxError as error:
            self._refused(error)
            return
        self._end_prompt()

    def _simplify(self, request: str) -> None:
        """Simplify what the line asks for, and say how long the answer took.

        An empty line asks for nothing, so it leaves the history alone. A line
        that does not read stays up to be corrected, as an authored one does.
        """
        if not request.strip():
            self._end_prompt()
            return
        started = time.monotonic()
        try:
            self.session.simplify(request)
        except DeriveSyntaxError as error:
            self._refused(error)
            return
        self._end_prompt(COMPUTE_TIME.format(seconds=time.monotonic() - started))

    def _refused(self, error: DeriveSyntaxError) -> None:
        """Say where the line stopped reading, and leave it up."""
        self._beep()
        self._set_message(str(error))
        self.query_one("#prompt-input", Input).cursor_position = error.offset

    def _hide_prompt(self) -> None:
        """Give the screen back to the menu band, whatever comes next."""
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
        self.factoring = None
        self.declaring = None
        self.defining = None
        self.entering = None
        self._hide_prompt()
        if done:
            del self.stack[1:]
        self._return_to_menu(message)

    def _return_to_menu(self, message: str) -> None:
        self.mode = MODE_MENU
        self.message = message
        self.refresh_screen()


def _color_index(editor: DialogEditor) -> int:
    """Where the color menu opens: on the color the field is already set to."""
    current = editor.value(editor.field)
    return next(
        (index for index in range(len(COLORS.words)) if menus.color_at(index) == current),
        0,
    )

"""The Rederive Algebra pane as a Textual application.

Key handling deliberately bypasses Textual's focus model. In DOS Derive the
command menu and the expression highlight are two cursors that exist at the
same time, driven by disjoint key sets: Tab/Space/Backspace/Enter and the
mnemonic letters drive the menu, while the arrow keys drive the selection in
the work area. So the menu keys are app-level priority bindings, switched off
by `check_action` whenever the author line is up and the Input should get the
keystrokes instead.

Submenus and Options dialogs stack on top of the command menu: Esc pops one
level, and committing a dialog returns all the way to the command menu, as the
original does.

A command that needs a line of its own - Author and Simplify an expression,
the Transfer commands a file name - takes the screen with the prompt band
instead, which is one Input with a label in front of it. The mode says which
command the line belongs to, and so what Enter does with it. A command that
runs is finished with, so it leaves the command menu up rather than the
submenu it was picked from; Esc, which abandons the line instead, leaves that
submenu where it was.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Input, Static
from textual.widgets.input import Selection

from rederive.model import worksheet
from rederive.model.session import Session
from rederive.model.settings import ChoiceField, Dialog, DialogEditor, Settings
from rederive.syntax import DeriveSyntaxError
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
MODE_FILE = "file"
MODE_CONFIRM_QUIT = "confirm_quit"

#: The modes in which the prompt band has the screen, and the keys with it.
PROMPT_MODES = (MODE_AUTHOR, MODE_SIMPLIFY, MODE_FILE)

# F1 does nothing yet: Help is not part of this milestone. The wording is the
# original's, kept so the screen reads right.
ENTER_EXPRESSION = "Enter expression (press F1 for help)"
ENTER_TO_SIMPLIFY = "Enter expression"
ABANDON_PROMPT = "Abandon expressions (Y/N)?"
AUTHOR_PROMPT = " AUTHOR expression: "
SIMPLIFY_PROMPT = " SIMPLIFY expression: "
#: What the message line says once an answer is in.
COMPUTE_TIME = "Compute time: {seconds:.1f} seconds"

# The three commands that name a file, and what they ask for. F1 does not list
# a directory yet, as it does not offer help yet; the wording is the
# original's, kept so the screen reads right.
LOAD_PROMPT = " TRANSFER LOAD DERIVE file: "
MERGE_PROMPT = " TRANSFER MERGE file: "
SAVE_PROMPT = " TRANSFER SAVE DERIVE file: "
ENTER_FILE = "Enter filename"
ENTER_FILE_TO_READ = "Enter filename (press F1 for list)"
FILE_NOT_FOUND = "File not found"
CANNOT_READ = "Cannot read file"
CANNOT_WRITE = "Cannot write file"
#: What the message line says when a file held a line that would not parse.
#: The original drops such a line without a word; saying so is cheap, and a
#: worksheet quietly missing an expression is worth knowing about.
UNREADABLE = "{count} expression{s} could not be read"

#: What the navigation keys mean inside an Options dialog's number field.
CURSOR_MOVES = {
    "left": "back",
    "right": "forward",
    "first_sibling": "start",
    "last_sibling": "end",
}


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
        Binding("ctrl+home", "nav('first_entry')", "Top", priority=True, show=False),
        Binding("ctrl+end", "nav('last_entry')", "Bottom", priority=True, show=False),
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
        #: The block of labels the next save writes, when one was asked for.
        self.block: tuple[int | None, int | None] = (None, None)
        # Commands not listed here are present and navigable but inert.
        self.commands: dict[tuple[Menu, str], Callable[[], None]] = {
            (ALGEBRA, "Author"): self._command_author,
            (ALGEBRA, "Quit"): self._command_quit,
            (ALGEBRA, "Simplify"): self._command_simplify,
            (menus.TRANSFER, "Merge"): self._command_merge,
            (menus.TRANSFER_LOAD, "Derive"): self._command_load,
            (menus.TRANSFER_SAVE, "Derive"): self._command_save,
        }

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
        else:
            cursor = self.top
            assert isinstance(cursor, MenuCursor)
            # Quit's confirmation takes the highlight off the menu entirely.
            highlighted = None if self.mode == MODE_CONFIRM_QUIT else cursor.index
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

        While a prompt line has the screen they all belong to the Input, down
        to Space and Backspace.
        """
        if action.startswith("menu_") or action in ("nav", "scroll_work"):
            return self.mode == MODE_MENU
        return True

    def on_key(self, event: Any) -> None:
        """Keys with no binding: mnemonic letters, digits, and Quit's answer."""
        if self.mode == MODE_MENU:
            character = event.character
            if character and character.isalnum():
                event.stop()
                event.prevent_default()
                self._typed(character)
        elif self.mode == MODE_CONFIRM_QUIT:
            event.stop()
            event.prevent_default()
            if event.character and event.character.lower() == "y":
                self.exit()
            else:
                self._return_to_menu(ENTER_OPTION)
        elif self.mode in PROMPT_MODES and event.key == "escape":
            event.stop()
            event.prevent_default()
            self._end_prompt(done=False)

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
        if editor.type_digit(character):
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
        """Space steps the highlight, or the active field's value."""
        editor = self.editor
        if editor is None:
            self._move_highlight(1)
        elif editor.lists_choices():
            self.stack.append(MenuCursor(COLORS, _color_index(editor)))
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
        """Leave the submenu or dialog on top, abandoning what it was set to."""
        if len(self.stack) > 1:
            self.stack.pop()
            self._ask_again()
            self.refresh_screen()

    def _ask_again(self) -> None:
        """Put the uncovered menu's own prompt back on the message line.

        A dialog does this for itself, since its prompt follows the field the
        highlight is on.
        """
        if isinstance(self.top, MenuCursor):
            self.message = self.top.menu.message

    def action_nav(self, movement: str) -> None:
        """The arrows walk the history, or a number field's cursor.

        A dialog's selection fields take no arrow keys at all in the original,
        which falls out of this: there is no cursor in them to move.
        """
        editor = self.editor
        if editor is None:
            getattr(self.session, f"move_{movement}")()
        elif not editor.move_cursor(CURSOR_MOVES.get(movement)):
            self._beep()
        self.refresh_screen()

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
        if cursor.menu is COLORS:
            self._chose_color(index)
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

    def _ask(self, dialog: Dialog, answer: Callable[[dict[str, str | int]], None]) -> None:
        """Stack a dialog that stores nothing, and say who wants its values."""
        self.answer = answer
        self._open(dialog)

    def _answered(self, values: dict[str, str | int]) -> None:
        """Give such a dialog's values to the command that put it up."""
        self.stack.pop()
        answer, self.answer = self.answer, None
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

    # -- Transfer ----------------------------------------------------------

    def _command_save(self) -> None:
        """Write the worksheet, asking first which block of it when Some.

        A history with nothing in it is nothing to write, and the original
        simply declines: no prompt, no message, just the beep.
        """
        if not self.session.entries:
            self._beep()
            return
        if self.settings["SaveRange"] != "Some":
            self.block = (None, None)
            self._ask_file(SAVE_PROMPT, ENTER_FILE, self._save)
            return
        numbers = [entry.number for entry in self.session.entries]
        self._ask(menus.save_block(numbers[0], numbers[-1]), self._chose_block)

    def _chose_block(self, values: dict[str, str | int]) -> None:
        self.block = (int(values["SaveFirst"]), int(values["SaveLast"]))
        self._ask_file(SAVE_PROMPT, ENTER_FILE, self._save)

    def _command_load(self) -> None:
        self._ask_file(LOAD_PROMPT, ENTER_FILE_TO_READ, self._load)

    def _command_merge(self) -> None:
        self._ask_file(MERGE_PROMPT, ENTER_FILE_TO_READ, self._merge)

    def _ask_file(self, label: str, message: str, command: Callable[[str], None]) -> None:
        """Put the prompt band up for a command that names a file.

        The file the session last used comes up on the line, as it does in the
        original, so that saving twice over the same name is one keystroke.
        """
        self.file_command = command
        offered = "" if self.session.file is None else str(self.session.file)
        self._prompt(MODE_FILE, label, offered, message)

    def _save(self, name: str) -> None:
        try:
            self.session.save(worksheet.path_of(name), *self.block)
        except OSError:
            self._refuse_file(CANNOT_WRITE)
            return
        self._end_prompt()

    def _load(self, name: str) -> None:
        self._read(name, self.session.load)

    def _merge(self, name: str) -> None:
        self._read(name, self.session.merge)

    def _read(self, name: str, command: Callable[[Path], int]) -> None:
        """Read a file into the history, leaving the line up if it cannot be."""
        try:
            skipped = command(worksheet.path_of(name))
        except FileNotFoundError:
            self._refuse_file(FILE_NOT_FOUND)
            return
        except OSError:
            self._refuse_file(CANNOT_READ)
            return
        self._end_prompt(
            ENTER_OPTION
            if not skipped
            else UNREADABLE.format(count=skipped, s="" if skipped == 1 else "s")
        )

    def _refuse_file(self, message: str) -> None:
        """Say what went wrong and leave the name up to be corrected."""
        self._beep()
        self._set_message(message)

    def _command_quit(self) -> None:
        if not self.session.entries:
            self.exit()
            return
        self.mode = MODE_CONFIRM_QUIT
        self.message = ABANDON_PROMPT
        self.refresh_screen()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Enter on a prompt line: the whole line, wherever the cursor is."""
        event.stop()
        if self.mode == MODE_SIMPLIFY:
            self._simplify(event.value)
        elif self.mode == MODE_FILE:
            self._named(event.value)
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

    def _end_prompt(self, message: str = ENTER_OPTION, done: bool = True) -> None:
        """Take the prompt line down, and put a menu back where it was.

        A command that ran is finished with, so the whole path it was reached
        by goes: `Transfer Save Derive` leaves the command menu up, not the
        Transfer Save menu it was picked from. Abandoning the line instead
        leaves that menu where it was, which is what Esc does.
        """
        self.query_one("#prompt-input", Input).value = ""
        self.query_one("#prompt-band").display = False
        self.query_one("#menu").display = True
        self.set_focus(None)
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

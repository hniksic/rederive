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
"""

from __future__ import annotations

from typing import Any, Callable

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Input, Static

from rederive.model import Session
from rederive.model.settings import ChoiceField, Dialog, DialogEditor, Settings
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
MODE_CONFIRM_QUIT = "confirm_quit"

# F1 does nothing yet: Help is not part of this milestone. The wording is the
# original's, kept so the screen reads right.
ENTER_EXPRESSION = "Enter expression (press F1 for help)"
ABANDON_PROMPT = "Abandon expressions (Y/N)?"
AUTHOR_PROMPT = " AUTHOR expression: "

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
    # TODO(display): Ctrl-Right and Ctrl-Left scroll the work area
    # horizontally by half a pane. They move no selection, so they are not nav
    # actions.
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
    ]

    def __init__(
        self, session: Session | None = None, settings: Settings | None = None
    ) -> None:
        # Before the base class, which asks for the CSS variables as it starts.
        self.settings = settings if settings is not None else Settings()
        self.palette = Palette(self.settings)
        super().__init__()
        self.session = session if session is not None else Session()
        self.settings.watch(self._settings_changed)
        self.mode = MODE_MENU
        #: The command menu, plus whatever submenu or dialog is stacked on it.
        self.stack: list[MenuCursor | DialogEditor] = [MenuCursor(ALGEBRA)]
        self.message = ENTER_OPTION
        # Commands not listed here are present and navigable but inert.
        self.commands: dict[str, Callable[[], None]] = {
            "Author": self._command_author,
            "Options": self._command_options,
            "Quit": self._command_quit,
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
            Static(AUTHOR_PROMPT, id="author-prompt"),
            Input(id="author-input"),
            id="author-band",
        )
        yield MessageLine(id="message")
        yield StatusLine(id="status")

    def on_mount(self) -> None:
        self.query_one("#author-band").display = False
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
        # TODO(display): pass the rendered layouts and the selection rectangle.
        self.query_one(WorkArea).show(
            self.session.entries, self.session.selected, self.session.selection_span()
        )
        editor = self.editor
        self.query_one("#menu").display = editor is None and self.mode != MODE_AUTHOR
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
        annotation = "User" if self.session.selected_entry is not None else ""
        self.query_one(StatusLine).show(annotation)

    def _set_message(self, message: str) -> None:
        self.message = message
        self.query_one(MessageLine).show(message)

    def _settings_changed(self, changed: frozenset[str]) -> None:
        """React to settings that other parts of the screen are built from."""
        # TODO(display): DisplayFormat, TimesOperator and OutputBase decide how
        # expressions render, so a change to any of them has to re-render the
        # work area the way COLOR_SETTINGS repaints it.
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

        While the author line has the screen they all belong to the Input, down
        to Space and Backspace.
        """
        if action.startswith("menu_") or action == "nav":
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
        elif self.mode == MODE_AUTHOR and event.key == "escape":
            event.stop()
            event.prevent_default()
            self._end_author()

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

    def _move_highlight(self, step: int) -> None:
        cursor = self.top
        assert isinstance(cursor, MenuCursor)
        cursor.move(step)

    def invoke_command(self, index: int) -> None:
        """Run the command at `index`, leaving the menu highlight alone."""
        cursor = self.top
        assert isinstance(cursor, MenuCursor)
        word = cursor.menu.words[index]
        if cursor.menu is ALGEBRA:
            command = self.commands.get(word)
            if command is None:
                self._set_message(f"{word}: not implemented yet")
            else:
                command()
        elif cursor.menu is menus.OPTIONS:
            self._open(menus.OPTIONS_TARGETS[word])
        elif cursor.menu is menus.COLOR:
            self._open(menus.COLOR_TARGETS[word])
        elif cursor.menu is COLORS:
            self._chose_color(index)

    # -- Options -----------------------------------------------------------

    def _open(self, target: Menu | Dialog) -> None:
        """Stack a submenu or an Options dialog on what is showing."""
        if isinstance(target, Menu):
            self.stack.append(MenuCursor(target))
        else:
            self.stack.append(DialogEditor(target, self.settings))
        self._ask_again()
        self.refresh_screen()

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
        changed = editor.changes(self.settings)
        del self.stack[1:]
        self.settings.apply(values)
        # Recorded after the change, so that the record itself is written the
        # way the new settings say to write it.
        for field in changed:
            if field.recorded:
                self.session.author(self.settings.assignment(field.setting))
        self._return_to_menu(ENTER_OPTION)

    # -- commands ----------------------------------------------------------

    def _command_author(self) -> None:
        self.mode = MODE_AUTHOR
        self.query_one("#menu").display = False
        self.query_one("#author-band").display = True
        author_input = self.query_one("#author-input", Input)
        author_input.value = ""
        author_input.focus()
        self._set_message(ENTER_EXPRESSION)

    def _command_options(self) -> None:
        self._open(menus.OPTIONS)

    def _command_quit(self) -> None:
        if not self.session.entries:
            self.exit()
            return
        self.mode = MODE_CONFIRM_QUIT
        self.message = ABANDON_PROMPT
        self.refresh_screen()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        text = event.value.strip()
        if not text:
            return
        self.session.author(text)
        self._end_author()

    def _end_author(self) -> None:
        self.query_one("#author-input", Input).value = ""
        self.query_one("#author-band").display = False
        self.query_one("#menu").display = True
        self.set_focus(None)
        self._return_to_menu(ENTER_OPTION)

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

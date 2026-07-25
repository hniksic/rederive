"""The Rederive Algebra pane as a Textual application.

Key handling deliberately bypasses Textual's focus model. In DOS Derive the
command menu and the expression highlight are two cursors that exist at the
same time, driven by disjoint key sets: Tab/Space/Backspace/Enter and the
mnemonic letters drive the menu, while the arrow keys drive the selection in
the work area. So the menu keys are app-level priority bindings, switched off
by `check_action` whenever the author line is up and the Input should get the
keystrokes instead.
"""

from __future__ import annotations

from typing import Any, Callable

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Input, Static

from rederive.model import Session
from rederive.ui.menu import ALGEBRA_MENU, MNEMONICS
from rederive.ui.theme import css_variables
from rederive.ui.widgets import MenuBand, MenuRule, MessageLine, StatusLine, WorkArea

MODE_MENU = "menu"
MODE_AUTHOR = "author"
MODE_CONFIRM_QUIT = "confirm_quit"

ENTER_OPTION = "Enter option"
ENTER_EXPRESSION = "Enter expression"
ABANDON_PROMPT = "Abandon expressions (Y/N)?"
AUTHOR_PROMPT = " AUTHOR expression: "


class RederiveApp(App[None]):
    """A single full-screen Algebra pane."""

    CSS_PATH = "rederive.tcss"
    TITLE = "Rederive"
    AUTO_FOCUS = None
    # Nothing on screen belongs to anything but the pane itself.
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("tab,space", "menu_next", "Next option", priority=True, show=False),
        Binding(
            "shift+tab,backspace",
            "menu_previous",
            "Previous option",
            priority=True,
            show=False,
        ),
        Binding("enter", "menu_invoke", "Invoke option", priority=True, show=False),
        Binding("up", "nav('up')", "Up", priority=True, show=False),
        Binding("down", "nav('down')", "Down", priority=True, show=False),
        Binding("left", "nav('left')", "Left", priority=True, show=False),
        Binding("right", "nav('right')", "Right", priority=True, show=False),
        Binding("home", "nav('first_sibling')", "Home", priority=True, show=False),
        Binding("end", "nav('last_sibling')", "End", priority=True, show=False),
        Binding("ctrl+home", "nav('first_entry')", "Top", priority=True, show=False),
        Binding("ctrl+end", "nav('last_entry')", "Bottom", priority=True, show=False),
    ]

    def __init__(self, session: Session | None = None) -> None:
        super().__init__()
        self.session = session if session is not None else Session()
        self.mode = MODE_MENU
        self.menu_index = 0
        self.message = ENTER_OPTION
        # Commands not listed here are present and navigable but inert.
        self.commands: dict[str, Callable[[], None]] = {
            "Author": self._command_author,
            "Quit": self._command_quit,
        }

    # -- composition -------------------------------------------------------

    def get_css_variables(self) -> dict[str, str]:
        return {**super().get_css_variables(), **css_variables()}

    def compose(self) -> ComposeResult:
        yield WorkArea(id="work")
        yield MenuRule(id="rule")
        yield MenuBand(id="menu")
        yield Horizontal(
            Static(AUTHOR_PROMPT, id="author-prompt"),
            Input(id="author-input"),
            id="author-band",
        )
        yield MessageLine(id="message")
        yield StatusLine(id="status")

    def on_mount(self) -> None:
        self.query_one("#author-band").display = False
        self.refresh_screen()

    # -- rendering ---------------------------------------------------------

    def refresh_screen(self) -> None:
        """Push the whole model state at the widgets."""
        self.query_one(WorkArea).show(
            self.session.entries, self.session.selected, self.session.selection_span()
        )
        self.query_one(MenuBand).show(self.menu_index)
        self.query_one(MessageLine).show(self.message)
        annotation = "User" if self.session.selected_entry is not None else ""
        self.query_one(StatusLine).show(annotation)

    def _set_message(self, message: str) -> None:
        self.message = message
        self.query_one(MessageLine).show(message)

    # -- key routing -------------------------------------------------------

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Menu and navigation keys only apply while the menu band is shown."""
        if action in ("menu_next", "menu_previous", "menu_invoke", "nav"):
            return self.mode == MODE_MENU
        return True

    def on_key(self, event: Any) -> None:
        """Keys with no binding: mnemonic letters, Esc, and the Quit answer."""
        if self.mode == MODE_MENU:
            character = event.character
            if character and character.lower() in MNEMONICS:
                event.stop()
                event.prevent_default()
                self.invoke_command(MNEMONICS[character.lower()])
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

    # -- menu actions ------------------------------------------------------

    def action_menu_next(self) -> None:
        self.menu_index = (self.menu_index + 1) % len(ALGEBRA_MENU)
        self.query_one(MenuBand).show(self.menu_index)

    def action_menu_previous(self) -> None:
        self.menu_index = (self.menu_index - 1) % len(ALGEBRA_MENU)
        self.query_one(MenuBand).show(self.menu_index)

    def action_menu_invoke(self) -> None:
        self.invoke_command(self.menu_index)

    def action_nav(self, movement: str) -> None:
        getattr(self.session, f"move_{movement}")()
        self.refresh_screen()

    def invoke_command(self, index: int) -> None:
        """Run the command at `index`, leaving the menu highlight alone."""
        word = ALGEBRA_MENU[index]
        command = self.commands.get(word)
        if command is None:
            self._set_message(f"{word}: not implemented yet")
        else:
            command()

    # -- commands ----------------------------------------------------------

    def _command_author(self) -> None:
        self.mode = MODE_AUTHOR
        self.query_one("#menu").display = False
        self.query_one("#author-band").display = True
        author_input = self.query_one("#author-input", Input)
        author_input.value = ""
        author_input.focus()
        self._set_message(ENTER_EXPRESSION)

    def _command_quit(self) -> None:
        if not self.session.entries:
            self.exit()
            return
        self.mode = MODE_CONFIRM_QUIT
        self._set_message(ABANDON_PROMPT)

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

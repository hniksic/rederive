"""Widgets for the four bands of an Algebra pane.

These render model state and nothing else: no navigation rules, no parsing.
"""

from __future__ import annotations

from rich.text import Text
from textual.containers import VerticalScroll
from textual.geometry import Region
from textual.widgets import Static

from rederive.model import Entry
from rederive.ui.menu import ALGEBRA_MENU, FIRST_LINE_OPTIONS, MENU_TITLE
from rederive.ui.theme import STYLES

#: Lines a single history entry occupies, including the blank line after it.
_LINES_PER_ENTRY = 2


def _label(entry: Entry) -> str:
    return f" #{entry.number}:  "


class WorkArea(VerticalScroll):
    """The numbered expression history, scrolled to keep the selection visible."""

    can_focus = False

    def compose(self):
        yield Static(id="work-content")

    def show(
        self,
        entries: list[Entry],
        selected: int | None,
        span: tuple[int, int] | None,
    ) -> None:
        text = Text(style=STYLES["expression"], no_wrap=True)
        for index, entry in enumerate(entries):
            if index:
                text.append("\n\n")
            label = _label(entry)
            text.append(label)
            start = len(text.plain)
            text.append(entry.text)
            if index == selected and span is not None:
                text.stylize(STYLES["selection"], start + span[0], start + span[1])
        self.query_one("#work-content", Static).update(text)
        if selected is not None:
            # After the refresh, so the new content has been laid out and the
            # scrollable region knows how tall it is.
            line = selected * _LINES_PER_ENTRY
            self.call_after_refresh(
                self.scroll_to_region, Region(0, line, 1, 1), animate=False
            )


class MessageLine(Static):
    """One line saying what is happening or what input is expected."""

    def show(self, message: str) -> None:
        self.update(Text(f" {message}", style=STYLES["annotation"]))


class MenuBand(Static):
    """The two-line command menu, one option highlighted in inverse video."""

    def show(self, highlighted: int) -> None:
        indent = " " + " " * len(MENU_TITLE) + " "
        first = self._line(f" {MENU_TITLE} ", 0, FIRST_LINE_OPTIONS, highlighted)
        second = self._line(indent, FIRST_LINE_OPTIONS, len(ALGEBRA_MENU), highlighted)
        self.update(Text("\n").join([first, second]))

    @staticmethod
    def _line(prefix: str, start: int, stop: int, highlighted: int) -> Text:
        text = Text(prefix, style=STYLES["menu"], no_wrap=True)
        for index in range(start, stop):
            if index > start:
                text.append(" ")
            style = STYLES["menu-highlight"] if index == highlighted else STYLES["menu"]
            text.append(ALGEBRA_MENU[index], style=style)
        return text


class MenuRule(Static):
    """The horizontal rule separating the work area from the command bands."""

    def render(self) -> Text:
        return Text("─" * self.size.width, no_wrap=True)


class StatusLine(Static):
    """Bottom line: selection annotation, memory field, pane type."""

    annotation = ""
    center = "Free:100%"
    pane = "Derive Algebra"

    def show(self, annotation: str) -> None:
        self.annotation = annotation
        self.refresh()

    def render(self) -> Text:
        width = max(self.size.width, len(self.pane) + 2)
        text = Text(" ", no_wrap=True)
        text.append(self.annotation, style=STYLES["annotation"])
        used = 1 + len(self.annotation)
        center_at = max(used + 1, (width - len(self.center)) // 2)
        text.append(" " * (center_at - used))
        text.append(self.center, style=STYLES["status"])
        used = center_at + len(self.center)
        pane_at = max(used + 1, width - 1 - len(self.pane))
        text.append(" " * (pane_at - used))
        text.append(self.pane, style=STYLES["status"])
        return text

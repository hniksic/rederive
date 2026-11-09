"""The dark styling of the plot windows' chrome.

The toolbar, fields, menus, status line and dialogs are styled dark to match the
near-black canvas, rather than following the desktop theme. `dress` applies the style
sheet and palette to the application, covering every window the host opens.

This covers only the chrome; the curve palette is in `plot.model`.

The module doesn't import Qt at import time, since the host imports it before knowing
whether Qt is available; functions import Qt when called.
"""

from __future__ import annotations

from typing import Any

__all__ = ["dangerous", "dress", "divider", "field", "monospace"]

#: The toolbar background and its bottom border.
CHROME = "#131318"
CHROME_EDGE = "#23232b"

#: Input fields: background, border and text.
FIELD = "#191920"
FIELD_EDGE = "#2b2b34"
FIELD_TEXT = "#d5d9e0"

#: Text colors: control text, its hover color, labels, and the dimmest, for disabled
#: items.
TEXT = "#b6bcc7"
TEXT_OVER = "#dde3ec"
MUTED = "#7d8595"
DIM = "#6d7482"

#: The single accent color, and the fill, border and text of controls that are on, so
#: "on" looks the same everywhere.
ACCENT = "#65b2f1"
ACCENT_FILL = "rgba(101, 178, 241, 0.14)"
ACCENT_EDGE = "rgba(101, 178, 241, 0.55)"
ACCENT_TEXT = "#cfe6fb"
ACCENT_FILL_OVER = "rgba(101, 178, 241, 0.22)"
ACCENT_EDGE_OVER = "rgba(101, 178, 241, 0.75)"
FIELD_EDGE_OVER = "#3b414e"

#: The status line: background, top border, text, and the brighter pointer readout.
STATUS = "#101015"
STATUS_EDGE = "#1f1f27"
STATUS_TEXT = "#8d93a0"
READOUT = "#d7dbe3"

#: Hover colors of destructive controls, the only warm color in the window, shown only
#: under the pointer.
DANGER_EDGE = "#7c3f42"
DANGER_TEXT = "#e2a1a1"
DANGER_FILL = "rgba(224, 110, 110, 0.07)"

#: The legend card and its row hover color. The hover color is an RGBA tuple, since a
#: painter rather than the style sheet uses it.
CARD = "rgba(17, 17, 23, 0.88)"
CARD_EDGE = "rgba(255, 255, 255, 0.07)"
CARD_ROW_OVER = (255, 255, 255, 18)
CARD_RADIUS = 9

#: The legend card for image exports, which use a white background.
CARD_PAPER = "rgba(255, 255, 255, 0.90)"
CARD_PAPER_EDGE = "rgba(0, 0, 0, 0.12)"
CARD_PAPER_ROW_OVER = (0, 0, 0, 18)

#: Object names the style sheet selects on: destructive controls (`clear`), a dialog's
#: primary button (`Apply`), and dividers between groups of toolbar controls.
DANGER = "danger"
PRIMARY = "primary"
DIVIDER = "divider"

#: Divider height, before the style sheet's margin trims both ends.
DIVIDER_LENGTH = 22

#: Toolbar button corner radius, which makes a pill shape at the button's height. (The
#: usual 999 trick doesn't work, since Qt draws the radius literally.)
PILL_RADIUS = 11

STYLESHEET = f"""
QMainWindow, QDialog {{ background: {CHROME}; }}
QWidget {{ color: {TEXT}; }}

QToolBar {{
    background: {CHROME};
    border: 0px;
    border-bottom: 1px solid {CHROME_EDGE};
    padding: 4px 6px;
    spacing: 3px;
}}
QToolBar QLabel {{ color: {MUTED}; background: transparent; padding: 0px 1px; }}
QFrame#{DIVIDER} {{ background: {CHROME_EDGE}; border: 0px; margin: 4px 7px; }}

QToolButton {{
    color: {TEXT};
    background: transparent;
    border: 1px solid {FIELD_EDGE};
    border-radius: {PILL_RADIUS}px;
    padding: 3px 13px;
}}
QToolButton:hover {{ color: {TEXT_OVER}; border-color: {FIELD_EDGE_OVER}; }}
QToolButton:checked {{
    color: {ACCENT_TEXT};
    background: {ACCENT_FILL};
    border-color: {ACCENT_EDGE};
}}
QToolButton:checked:hover {{
    background: {ACCENT_FILL_OVER};
    border-color: {ACCENT_EDGE_OVER};
}}
QToolButton#{DANGER}:hover {{
    color: {DANGER_TEXT};
    background: {DANGER_FILL};
    border-color: {DANGER_EDGE};
}}

QLineEdit {{
    color: {FIELD_TEXT};
    background: {FIELD};
    border: 1px solid {FIELD_EDGE};
    border-radius: 6px;
    padding: 2px 6px;
    selection-background-color: {ACCENT};
    selection-color: {CHROME};
}}
QLineEdit:hover {{ border-color: {FIELD_EDGE_OVER}; }}
QLineEdit:focus {{ border-color: {ACCENT}; }}

QMenu {{
    background: {CHROME};
    border: 1px solid {CHROME_EDGE};
    color: {FIELD_TEXT};
    padding: 4px;
}}
QMenu::item {{ padding: 4px 22px 4px 14px; border-radius: 4px; }}
QMenu::item:selected {{ background: rgba(255, 255, 255, 0.07); color: {TEXT_OVER}; }}
QMenu::item:disabled {{ color: {DIM}; }}
QMenu::separator {{ height: 1px; background: {CHROME_EDGE}; margin: 4px 8px; }}
QMenu QWidget {{ background: {CHROME}; color: {TEXT}; }}

QGroupBox {{
    border: 1px solid {CHROME_EDGE};
    border-radius: 7px;
    margin-top: 9px;
    padding: 10px 10px 8px 10px;
    color: {MUTED};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0px 4px;
    color: {MUTED};
}}

QPushButton {{
    color: {FIELD_TEXT};
    background: {FIELD};
    border: 1px solid {FIELD_EDGE};
    border-radius: 6px;
    padding: 5px 16px;
}}
QPushButton:hover {{ color: {TEXT_OVER}; border-color: {FIELD_EDGE_OVER}; }}
QPushButton#{PRIMARY} {{
    color: {CHROME};
    background: {ACCENT};
    border-color: {ACCENT};
}}
QPushButton#{PRIMARY}:hover {{ background: {ACCENT_TEXT}; border-color: {ACCENT_TEXT}; }}

QCheckBox, QRadioButton {{ color: {TEXT}; background: transparent; }}
QSpinBox, QDoubleSpinBox, QComboBox {{
    color: {FIELD_TEXT};
    background: {FIELD};
    border: 1px solid {FIELD_EDGE};
    border-radius: 4px;
    padding: 1px 4px;
}}

QToolTip {{
    color: {FIELD_TEXT};
    background: {FIELD};
    border: 1px solid {FIELD_EDGE};
    padding: 3px 6px;
}}
"""


def dress(application: Any) -> None:
    """Apply the dark styling to the application, before any window opens.

    The style sheet does most of the work; the palette covers elements it doesn't name,
    such as scroll bars, spin box arrows and menu ticks. Fusion is selected because it
    respects the palette everywhere, unlike native themes.
    """
    from pyqtgraph.Qt import QtGui

    try:
        application.setStyle("Fusion")
    except Exception:  # pragma: no cover - a Qt built without it
        pass
    role = QtGui.QPalette.ColorRole
    group = QtGui.QPalette.ColorGroup
    palette = QtGui.QPalette()
    for name, color in (
        ("Window", CHROME),
        ("WindowText", TEXT),
        ("Base", FIELD),
        ("AlternateBase", CHROME),
        ("Text", FIELD_TEXT),
        ("Button", FIELD),
        ("ButtonText", TEXT),
        ("ToolTipBase", FIELD),
        ("ToolTipText", FIELD_TEXT),
        ("Highlight", ACCENT),
        ("HighlightedText", CHROME),
        ("Link", ACCENT),
        ("PlaceholderText", MUTED),
    ):
        palette.setColor(getattr(role, name), QtGui.QColor(color))
    for name in ("WindowText", "Text", "ButtonText"):
        palette.setColor(group.Disabled, getattr(role, name), QtGui.QColor(DIM))
    application.setPalette(palette)
    application.setStyleSheet(STYLESHEET)


def divider(vertical: bool = True) -> Any:
    """A thin divider between groups of toolbar controls.

    A widget rather than a `QToolBar` separator, so the style sheet can draw it. It is
    given an explicit size, since an empty frame would otherwise have none.
    """
    from pyqtgraph.Qt import QtWidgets

    line = QtWidgets.QFrame()
    line.setObjectName(DIVIDER)
    if vertical:
        line.setFixedSize(1, DIVIDER_LENGTH)
    else:
        line.setFixedHeight(1)
    return line


#: The field class, created on the first call to `field`, since defining it at module
#: scope would import Qt.
_FITTED: Any = None


def field(width: int) -> Any:
    """A right-aligned numeric input field, `width` pixels wide.

    Qt scrolls text too long for the field to its end, so `-3.14159` might show as
    `.14159`, which reads as a different number. Instead, the text is elided at the end,
    and the full text returns when the field gets focus, so the value itself is always
    what is edited.
    """
    from pyqtgraph.Qt import QtCore, QtWidgets

    global _FITTED
    if _FITTED is None:

        class Fitted(QtWidgets.QLineEdit):
            """A line edit that elides text it can't show, but keeps the full text."""

            def __init__(self) -> None:
                super().__init__()
                #: The full text, and whether the shown text is elided.
                self._full = ""
                self._short = False

            def setText(self, text: str) -> None:
                self._full = text
                self._short = False
                super().setText(text)
                self._fit()

            def text(self) -> str:
                return self._full if self._short else super().text()

            def focusInEvent(self, event: Any) -> None:
                self._whole()
                super().focusInEvent(event)

            def keyPressEvent(self, event: Any) -> None:
                # Never type into elided text. Focus normally restores the full
                # text, but a key can arrive without focus.
                self._whole()
                super().keyPressEvent(event)

            def focusOutEvent(self, event: Any) -> None:
                # Store the typed text before passing the event on, since
                # `editingFinished` is emitted from inside it and its handler must
                # read the full text, not the elision.
                self._full = super().text()
                self._short = False
                super().focusOutEvent(event)
                self._fit()

            def resizeEvent(self, event: Any) -> None:
                super().resizeEvent(event)
                self._fit()

            def _whole(self) -> None:
                """Restore the full text, before typing."""
                if self._short:
                    self._short = False
                    super().setText(self._full)

            def _fit(self) -> None:
                """Elide the text to fit, unless the field has focus."""
                if self.hasFocus():
                    return
                shown = self.fontMetrics().elidedText(
                    self._full, QtCore.Qt.TextElideMode.ElideRight, self._room()
                )
                self._short = shown != self._full
                super().setText(shown)

            def _room(self) -> int:
                """The width available for text, after the style sheet's padding."""
                option = QtWidgets.QStyleOptionFrame()
                self.initStyleOption(option)
                return (
                    self.style()
                    .subElementRect(
                        QtWidgets.QStyle.SubElement.SE_LineEditContents, option, self
                    )
                    .width()
                )

        _FITTED = Fitted

    edit = _FITTED()
    edit.setFixedWidth(width)
    edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
    return edit


def dangerous(bar: Any, action: Any) -> None:
    """Mark an action's toolbar button as destructive, for its warning hover style."""
    button = bar.widgetForAction(action)
    if button is not None:
        button.setObjectName(DANGER)


def monospace(size: int = 9) -> Any:
    """A fixed-pitch font for numbers that change as the pointer moves.

    Uses tabular figures where supported, so digits stay in their columns.
    """
    from pyqtgraph.Qt import QtGui

    font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
    font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
    font.setPointSize(size)
    try:  # aligned digits even if the fallback font is proportional
        font.setFeature("tnum", 1)
    except Exception:  # pragma: no cover - an older toolkit
        pass
    return font

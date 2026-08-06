"""The dark chrome the plot windows wear, in one place.

The canvas has always been near-black. Everything around it - the toolbar, the
fields, the menus, the status line, the dialogs - came from whatever the
desktop's own theme happened to be, which put a bright frame around a dark
picture and made the window look like two programs. These are the colors that
frame it instead, and `dress` is the one call that puts them on an application:
one style sheet, one palette, and every window the host opens is inside it.

What is here is furniture. The curve palette is not: what a curve is drawn in
is the picture's own business and lives with the picture, in `window2d`.

The module is deliberately free of the toolkit at import time, since the host
imports it before it knows whether there is a toolkit to import at all; every
function here reaches for Qt when it is called.
"""

from __future__ import annotations

from typing import Any

__all__ = ["dangerous", "dress", "divider", "field", "monospace"]

#: The bar the controls stand on, and the hairline that ends it.
CHROME = "#131318"
CHROME_EDGE = "#23232b"

#: A field the keyboard can reach: its inset ground, its edge, and its ink.
FIELD = "#191920"
FIELD_EDGE = "#2b2b34"
FIELD_TEXT = "#d5d9e0"

#: The three weights the chrome writes in: a control's own text, the label
#: beside one, and the dimmest, which names a number rather than saying
#: anything.
TEXT = "#b6bcc7"
TEXT_OVER = "#dde3ec"
MUTED = "#7d8595"
DIM = "#6d7482"

#: The one accent, and what a control that is *on* is filled, edged and
#: written in. One color for the whole program, so that "this is on" reads the
#: same wherever it is said.
ACCENT = "#65b2f1"
ACCENT_FILL = "rgba(101, 178, 241, 0.14)"
ACCENT_EDGE = "rgba(101, 178, 241, 0.55)"
ACCENT_TEXT = "#cfe6fb"
ACCENT_FILL_OVER = "rgba(101, 178, 241, 0.22)"
ACCENT_EDGE_OVER = "rgba(101, 178, 241, 0.75)"
FIELD_EDGE_OVER = "#3b414e"

#: The status line: its ground, the hairline above it, the voice it speaks in,
#: and the brighter ink the pointer readout puts its numbers in.
STATUS = "#101015"
STATUS_EDGE = "#1f1f27"
STATUS_TEXT = "#8d93a0"
READOUT = "#d7dbe3"

#: What a control that throws something away turns into under the pointer. It
#: is the only warm color in the window, and it is never shown until the
#: pointer is actually on the control.
DANGER_EDGE = "#7c3f42"
DANGER_TEXT = "#e2a1a1"
DANGER_FILL = "rgba(224, 110, 110, 0.07)"

#: The floating card a legend is drawn as, and the wash a row takes under the
#: pointer. The wash is written twice because it is read twice: once by the
#: style sheet, which wants a string, and once by a painter, which wants four
#: numbers.
CARD = "rgba(17, 17, 23, 0.88)"
CARD_EDGE = "rgba(255, 255, 255, 0.07)"
CARD_ROW_OVER = (255, 255, 255, 18)
CARD_RADIUS = 9

#: The same card for the white background every image export is taken on, so
#: that a legend of paper colors is read on paper rather than on the dark.
CARD_PAPER = "rgba(255, 255, 255, 0.90)"
CARD_PAPER_EDGE = "rgba(0, 0, 0, 0.12)"
CARD_PAPER_ROW_OVER = (0, 0, 0, 18)

#: The object names the sheet knows a control by. A `clear` is the destructive
#: one, a dialog's `Apply` is the accented one, and a divider is the hairline
#: between two groups of controls: three selectors rather than three widgets
#: carrying styles of their own.
DANGER = "danger"
PRIMARY = "primary"
DIVIDER = "divider"

#: How tall the hairline between two groups of controls stands, before the
#: sheet's own margin takes a few pixels off either end of it.
DIVIDER_LENGTH = 22

#: How round a chip is drawn. Large enough to be a pill at the height a
#: toolbar button comes out at, and a plain number rather than the 999 the
#: trick is usually written with, because Qt draws the corner it is given.
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
    """Put the dark chrome on an application, once, before it opens a window.

    The style sheet is the whole of the design and the palette is the safety
    net under it: a scroll bar, a spin box arrow or a menu tick that no
    selector here names still has to be dark, because it is drawn beside things
    that are. Fusion is asked for by name for the same reason - it is the one
    style that takes a palette everywhere, where a native theme paints its own
    colors over half of it.
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
    """The hairline between two groups of controls on a toolbar.

    A separator that is a widget rather than a `QToolBar` separator, because
    the stock one is drawn by the style and this one is drawn by the sheet:
    one pixel of the chrome's edge color, standing clear of both groups. It is
    given a size of its own, since an empty frame in a bar's layout is a frame
    of no width and no height at all.
    """
    from pyqtgraph.Qt import QtWidgets

    line = QtWidgets.QFrame()
    line.setObjectName(DIVIDER)
    if vertical:
        line.setFixedSize(1, DIVIDER_LENGTH)
    else:
        line.setFixedHeight(1)
    return line


#: The class `field` hands out, made on its first call and kept. A widget class
#: cannot be written at module scope here - that would want the toolkit at
#: import time - and one class per field would be one class per window.
_FITTED: Any = None


def field(width: int) -> Any:
    """One number a person types, `width` pixels across and right-aligned.

    A field too narrow for what is in it has to say so. Qt scrolls such a text
    to its end, which on a right-aligned field leaves `.14159` of `-3.14159`:
    the sign and the whole number are gone and what is left reads as a
    different number altogether. So the text is elided at its end instead, and
    the whole of it comes back the moment the field takes the keyboard - what
    is edited is always the value and never the elision of it.
    """
    from pyqtgraph.Qt import QtCore, QtWidgets

    global _FITTED
    if _FITTED is None:

        class Fitted(QtWidgets.QLineEdit):
            """A field that elides what it cannot show, and knows what it holds."""

            def __init__(self) -> None:
                super().__init__()
                #: The text in full, which is what the field is worth however
                #: much of it is on show, and whether it is showing less.
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
                # Nothing is ever typed into an elision. Taking the keyboard is
                # what normally puts the whole text back, and this is the net
                # under a key that reaches the field some other way.
                self._whole()
                super().keyPressEvent(event)

            def focusOutEvent(self, event: Any) -> None:
                # Whatever was typed is the value now, and it is taken before
                # the event goes any further: `editingFinished` is emitted from
                # inside it, and what the handler reads has to be that text and
                # not the elision this ends with.
                self._full = super().text()
                self._short = False
                super().focusOutEvent(event)
                self._fit()

            def resizeEvent(self, event: Any) -> None:
                super().resizeEvent(event)
                self._fit()

            def _whole(self) -> None:
                """Put the text back in full, for a field about to be typed in."""
                if self._short:
                    self._short = False
                    super().setText(self._full)

            def _fit(self) -> None:
                """Show as much of the text as there is room for.

                A field being typed in is left alone: what is on show there is
                the text itself, and the value is whatever the keyboard has
                made of it rather than the last thing put in.
                """
                if self.hasFocus():
                    return
                shown = self.fontMetrics().elidedText(
                    self._full, QtCore.Qt.TextElideMode.ElideRight, self._room()
                )
                self._short = shown != self._full
                super().setText(shown)

            def _room(self) -> int:
                """How wide the text may be, which the sheet's padding decides."""
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
    """Name the button an action is drawn as, so the sheet can warn on it.

    A control that throws the picture away is a control that should say so
    where the pointer is on it and nowhere else, and the one thing the sheet
    cannot work out for itself is which control that is.
    """
    button = bar.widgetForAction(action)
    if button is not None:
        button.setObjectName(DANGER)


def monospace(size: int = 9) -> Any:
    """The face numbers that change under the eye are written in.

    A reading whose digits move as the pointer does has to keep its columns
    still, so this is the fixed-pitch face the platform offers, asked for
    tabular figures where the toolkit is new enough to ask.
    """
    from pyqtgraph.Qt import QtGui

    font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
    font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
    font.setPointSize(size)
    try:  # a proportional fallback still owes us aligned columns
        font.setFeature("tnum", 1)
    except Exception:  # pragma: no cover - an older toolkit
        pass
    return font

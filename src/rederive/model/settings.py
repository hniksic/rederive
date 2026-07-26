"""Derive's system control settings, and the Options screens that change them.

Every screen described here follows the program itself rather than the manual,
so the words, the defaults and the message-line prompts are the original's. Two
of the nine Options commands are gone: `Display` chose between text and graphics
modes on adapters that no longer exist, and `Execute` shelled out to DOS.

Three rules of the original that the descriptions below encode:

* A setting change is *recorded*: committing a dialog appends a `Name := Value`
  expression to the history for every field that changed, which is also how the
  same setting can be changed from the author line. `Options Color` and
  `Options Mute` record nothing, which is why they carry `recorded=False`.
* The record is written in the notation the settings themselves select, so
  `Options Output` reaches the spacing of its own record and `Options Radix`
  reaches its digits. `Settings.assignment` is where that happens.
* A choice field can prompt for a number instead, which is how `Radix` accepts
  a base of 2 through 36 that has no name. That is `ChoiceField.prompt`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

#: The colors a color slot can take, in the order Derive's Options Color
#: commands numbered them: EGA text attributes 0 through 15. Derive asked for
#: the number; these names are the remake's own. The intensified half is named
#: for what it looks like rather than prefixed with "light", which keeps the
#: menu narrow and gives every color a distinct mnemonic letter. Note that 7 is
#: the lighter gray and 8 the darker one - that inversion is the EGA palette's,
#: not a mistake here.
COLORS: tuple[str, ...] = (
    "Black",  # 0
    "Blue",  # 1
    "Green",  # 2
    "Cyan",  # 3
    "Red",  # 4
    "Magenta",  # 5
    "Brown",  # 6
    "Gray",  # 7, EGA light gray
    "Darkgray",  # 8
    "Skyblue",  # 9, EGA light blue
    "Lime",  # 10, EGA light green
    "Aqua",  # 11, EGA light cyan
    "Pink",  # 12, EGA light red
    "Violet",  # 13, EGA light magenta
    "Yellow",  # 14
    "White",  # 15
)

#: The named radix bases, and what they are worth.
BASES: dict[str, int] = {"Binary": 2, "Octal": 8, "Decimal": 10, "Hexadecimal": 16}

#: The choice that prompts for a number rather than standing for one.
OTHER = "Other"

_DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class NumberField:
    """A data entry field holding an integer, such as `Digits: 6`."""

    label: str
    message: str
    setting: str
    default: int
    minimum: int
    maximum: int | None = None
    recorded: bool = True

    def accepts(self, number: int) -> bool:
        return number >= self.minimum and (self.maximum is None or number <= self.maximum)


@dataclass(frozen=True)
class ChoiceField:
    """A selection field: one word out of a fixed list."""

    label: str
    message: str
    setting: str
    choices: tuple[str, ...]
    default: str
    recorded: bool = True
    inline: bool = True
    """False when the choices are too many to print on the band, as colors are.

    Such a field shows only its current value, and offers the rest as a submenu.
    """
    prompt: NumberField | None = None
    """What the `Other` choice asks for instead of a word."""


Field = ChoiceField | NumberField


@dataclass(frozen=True)
class Dialog:
    """One Options screen: a title and fields, grouped into the band's lines."""

    title: str
    lines: tuple[tuple[Field, ...], ...]

    @property
    def fields(self) -> tuple[Field, ...]:
        return tuple(field for line in self.lines for field in line)


PRECISION = Dialog(
    "OPTIONS PRECISION:",
    (
        (
            ChoiceField(
                "Mode",
                "Select arithmetic mode",
                "Precision",
                ("Approximate", "Exact", "Mixed"),
                "Exact",
            ),
            NumberField(
                "Digits", "Enter significant digits", "PrecisionDigits", 6, minimum=1
            ),
        ),
    ),
)

NOTATION = Dialog(
    "OPTIONS NOTATION:",
    (
        (
            ChoiceField(
                "Style",
                "Enter numerical output style",
                "Notation",
                ("Decimal", "Mixed", "Rational", "Scientific"),
                "Rational",
            ),
            NumberField(
                "Digits",
                "Enter significant digits to display",
                "NotationDigits",
                6,
                minimum=1,
            ),
        ),
    ),
)

INPUT = Dialog(
    "OPTIONS INPUT:",
    (
        (
            ChoiceField(
                "Mode",
                "Select input mode",
                "InputMode",
                ("Character", "Word"),
                "Character",
            ),
            ChoiceField(
                "Case",
                "Select case mode",
                "CaseMode",
                ("Insensitive", "Sensitive"),
                "Insensitive",
            ),
        ),
        (
            ChoiceField(
                "Use",
                "Select arrow key mode",
                "ArrowKeyMode",
                ("LineEdit", "Subexpression"),
                "LineEdit",
            ),
        ),
    ),
)

OUTPUT = Dialog(
    "OPTIONS OUTPUT:",
    (
        (
            ChoiceField(
                "Format",
                "Select display format",
                "DisplayFormat",
                ("Normal", "Compressed"),
                "Normal",
            ),
            ChoiceField(
                "Operator",
                "Select times operator",
                "TimesOperator",
                ("Asterisk", "Dot", "Implicit"),
                "Dot",
            ),
        ),
    ),
)

MUTE = Dialog(
    "OPTIONS MUTE:",
    (
        (
            ChoiceField(
                "Active",
                "Mute warning messages",
                "Mute",
                ("Yes", "No"),
                "No",
                recorded=False,
            ),
        ),
    ),
)

_RADIX_CHOICES = ("Binary", "Octal", "Decimal", "Hexadecimal", OTHER)

RADIX = Dialog(
    "OPTIONS RADIX:",
    (
        (
            ChoiceField(
                "Input",
                "Select number system",
                "InputBase",
                _RADIX_CHOICES,
                "Decimal",
                prompt=NumberField(
                    "Input", "Enter radix base", "InputBase", 10, minimum=2, maximum=36
                ),
            ),
        ),
        (
            ChoiceField(
                "Output",
                "Select number system",
                "OutputBase",
                _RADIX_CHOICES,
                "Decimal",
                prompt=NumberField(
                    "Output", "Enter radix base", "OutputBase", 10, minimum=2, maximum=36
                ),
            ),
        ),
    ),
)


def _color_field(label: str, setting: str, default: str) -> ChoiceField:
    return ChoiceField(
        label,
        f"Select {label.lower()} color (press Space for the list)",
        setting,
        COLORS,
        default,
        recorded=False,
        inline=False,
    )


# The color defaults are the factory settings of the last Derive version whose
# defaults were colorful, taken from its own Options Color screens. Rederive
# imitates the original in everything but color, and the original ships
# monochrome - Gray frame and options, White prompt, Gray status - so a
# colorless screenshot of it is not evidence that this scheme is wrong and must
# not be "corrected" to match.
COLOR_MENU = Dialog(
    "OPTIONS COLOR MENU:",
    (
        (
            _color_field("Frame", "FrameColor", "Red"),
            _color_field("Option", "OptionColor", "Yellow"),
            _color_field("Prompt", "PromptColor", "Red"),
            _color_field("Status", "StatusColor", "Green"),
        ),
        (
            _color_field("Background", "MenuBackgroundColor", "Black"),
            _color_field("Border", "BorderColor", "Black"),
        ),
    ),
)

COLOR_WORK = Dialog(
    "OPTIONS COLOR WORK:",
    (
        (
            _color_field("Foreground", "WorkColor", "White"),
            _color_field("Background", "WorkBackgroundColor", "Black"),
        ),
    ),
)

DIALOGS: tuple[Dialog, ...] = (
    PRECISION,
    NOTATION,
    INPUT,
    OUTPUT,
    MUTE,
    RADIX,
    COLOR_MENU,
    COLOR_WORK,
)

DEFAULTS: dict[str, str | int] = {
    field.setting: field.default for dialog in DIALOGS for field in dialog.fields
}


class Settings:
    """Every system control setting, and whom to tell when one changes.

    Storing the new value is not the whole job: colors have to reach the screen
    and the input modes have to reach the parser, so anything that depends on a
    setting registers with `watch` and is told which settings changed.
    """

    def __init__(self) -> None:
        self._values: dict[str, str | int] = dict(DEFAULTS)
        self._watchers: list[Callable[[frozenset[str]], None]] = []

    def __getitem__(self, setting: str) -> str | int:
        return self._values[setting]

    def watch(self, callback: Callable[[frozenset[str]], None]) -> None:
        """Call `callback` with the names of the settings that just changed."""
        self._watchers.append(callback)

    def apply(self, values: Mapping[str, str | int]) -> tuple[str, ...]:
        """Store `values`, and return the names of those that actually changed."""
        changed = tuple(
            setting for setting, value in values.items() if self._values[setting] != value
        )
        if not changed:
            return ()
        self._values.update(values)
        names = frozenset(changed)
        for watcher in self._watchers:
            watcher(names)
        return changed

    # -- writing values out ------------------------------------------------

    def assignment(self, setting: str) -> str:
        """The `Name := Value` expression Derive records for a settings change.

        Written the way the current settings say to write it: `Options Output`
        decides the spacing around `:=` and `Options Radix` the digits, so the
        record of a change is already in the notation that change selected.
        """
        space = "" if self._values["DisplayFormat"] == "Compressed" else " "
        return f"{setting}{space}:={space}{self.written(self._values[setting])}"

    def written(self, value: str | int) -> str:
        """`value` as it appears in an expression."""
        return self.numeral(value) if isinstance(value, int) else value

    def numeral(self, number: int) -> str:
        """`number` in the output radix.

        Above base ten the digits run A to Z, and a numeral that would begin
        with a letter takes a leading zero so that it cannot be read as a
        variable name: fourteen is `0E` in hexadecimal, not `E`.

        The display package formats numerals for the screen by the same rule.
        The two are kept apart so that the model does not depend on the display
        layer.
        """
        base = self.base("OutputBase")
        magnitude = abs(number)
        text = ""
        while True:
            text = _DIGITS[magnitude % base] + text
            magnitude //= base
            if not magnitude:
                break
        if text[0].isalpha():
            text = "0" + text
        return ("-" if number < 0 else "") + text

    def base(self, setting: str) -> int:
        """What a radix setting is worth, named or numbered."""
        value = self._values[setting]
        return BASES[value] if isinstance(value, str) else value


class DialogEditor:
    """One Options dialog while it is on screen.

    Nothing here touches the live settings. `commit` hands the values back for
    the caller to apply, which is what lets Esc walk away having changed
    nothing.
    """

    def __init__(self, dialog: Dialog, settings: Settings) -> None:
        self.dialog = dialog
        self.values = {field.setting: settings[field.setting] for field in dialog.fields}
        self.active = 0
        self.prompt: NumberField | None = None
        self.text: str | None = None
        """The digits of a number field being typed into, if that is the field."""
        self.cursor = 0
        self.numbers = {
            field.setting: _as_number(self.values[field.setting])
            for field in dialog.fields
            if isinstance(field, ChoiceField) and field.prompt is not None
        }
        """What a promptable field was worth when the dialog opened.

        Stepping the highlight onto `Other` does not change it: the original
        offers the radix that is in effect, not the one the highlight passed
        over on its way.
        """
        self._enter_field()

    # -- what is on the band -----------------------------------------------

    @property
    def lines(self) -> tuple[tuple[Field, ...], ...]:
        """The dialog's lines, or the sole `Other` prompt once it is up."""
        if self.prompt is not None:
            return ((self.prompt,),)
        return self.dialog.lines

    @property
    def fields(self) -> tuple[Field, ...]:
        return tuple(field for line in self.lines for field in line)

    @property
    def field(self) -> Field:
        return self.fields[self.active]

    @property
    def message(self) -> str:
        return self.field.message

    def value(self, field: Field) -> str | int:
        return self.values[field.setting]

    # -- moving between fields ---------------------------------------------

    def next_field(self) -> bool:
        return self._move(1)

    def previous_field(self) -> bool:
        return self._move(-1)

    def _move(self, step: int) -> bool:
        if self.open_prompt():
            return True
        if not self._leave_field():
            return False
        self.active = (self.active + step) % len(self.fields)
        self._enter_field()
        return True

    def open_prompt(self) -> bool:
        """Ask for a number, if the active field has been left on `Other`.

        The original does not prompt the moment `Other` is stepped onto; it
        prompts when the field is done with, which is Tab or Enter.
        """
        field = self.field
        if (
            self.prompt is not None
            or not isinstance(field, ChoiceField)
            or field.prompt is None
            or self.value(field) != OTHER
        ):
            return False
        # The prompt replaces the choice, so from here on the field holds a
        # number and `Other` is gone: Esc abandons the dialog anyway.
        self.prompt = field.prompt
        self.values[field.setting] = self.numbers[field.setting]
        self.active = 0
        self.text = str(self.numbers[field.setting])
        self.cursor = 0
        return True

    def _enter_field(self) -> None:
        """Start editing the active field, if it is one that is typed into."""
        field = self.field
        self.text = str(self.value(field)) if isinstance(field, NumberField) else None
        self.cursor = 0

    def _leave_field(self) -> bool:
        """Store what was typed, if the field takes it.

        A number the field will not have leaves the field as it is rather than
        putting the old value back, so that a mistyped digit can be corrected
        instead of retyped.
        """
        field = self.field
        if self.text is None or not isinstance(field, NumberField):
            return True
        number = int(self.text) if self.text.isdigit() else None
        if number is None or not field.accepts(number):
            return False
        self.values[field.setting] = number
        return True

    # -- changing a value --------------------------------------------------

    def choose(self, value: str) -> bool:
        """Select `value` in the active field, and move on to the next one.

        False means the active field has no such choice.
        """
        field = self.field
        if not isinstance(field, ChoiceField) or value not in field.choices:
            return False
        self.values[field.setting] = value
        self.next_field()
        return True

    def cycle(self) -> bool:
        """Step the active field to its next choice, as Space does."""
        field = self.field
        if not isinstance(field, ChoiceField) or not field.inline:
            return False
        current = self.value(field)
        index = field.choices.index(current) if current in field.choices else -1
        self.values[field.setting] = field.choices[(index + 1) % len(field.choices)]
        return True

    def lists_choices(self) -> bool:
        """Whether the active field offers its choices as a submenu."""
        field = self.field
        return isinstance(field, ChoiceField) and not field.inline

    def type_digit(self, character: str) -> bool:
        """Type a digit into the active number field, overwriting at the cursor."""
        if self.text is None or not character.isdigit():
            return False
        self.text = self.text[: self.cursor] + character + self.text[self.cursor + 1 :]
        self.cursor += 1
        return True

    def erase(self) -> bool:
        """Delete the digit before the cursor, as Backspace does."""
        if self.text is None or self.cursor == 0:
            return False
        self.cursor -= 1
        self.text = self.text[: self.cursor] + self.text[self.cursor + 1 :]
        return True

    def delete(self) -> bool:
        """Delete the digit at the cursor, as Del does."""
        if self.text is None or self.cursor >= len(self.text):
            return False
        self.text = self.text[: self.cursor] + self.text[self.cursor + 1 :]
        return True

    def move_cursor(self, where: str | None) -> bool:
        """Walk the cursor along a number field, as the arrows and Home/End do."""
        if self.text is None:
            return False
        place = {
            "back": self.cursor - 1,
            "forward": self.cursor + 1,
            "start": 0,
            "end": len(self.text),
        }.get(where)
        if place is None or not 0 <= place <= len(self.text):
            return False
        self.cursor = place
        return True

    # -- finishing ---------------------------------------------------------

    @property
    def settles_on_choice(self) -> bool:
        """Whether choosing a value finishes the dialog outright.

        It does when there is only one field to fill in, which is how
        `Options Mute` answers in a single keystroke.
        """
        return len(self.fields) == 1

    def commit(self) -> dict[str, str | int] | None:
        """The values to apply, or None if the dialog is not finished with.

        A field left on `Other` is not finished with: `open_prompt` puts up its
        number prompt, and the next Enter commits.
        """
        if self.open_prompt() or not self._leave_field():
            return None
        return dict(self.values)

    def changes(self, settings: Settings) -> tuple[Field, ...]:
        """The fields this dialog would change, in the order they are shown."""
        return tuple(
            field
            for field in self.dialog.fields
            if self.values[field.setting] != settings[field.setting]
        )


def _as_number(value: str | int) -> int:
    return BASES[value] if isinstance(value, str) else value

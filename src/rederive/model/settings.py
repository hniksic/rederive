"""Derive's system control settings, and the screens that change them.

Most screens are Options commands; four are Manage commands, which control
simplification. Both are treated the same here.

The screens follow the program rather than the manual: wording, defaults and
message-line prompts are the original's. Two Options commands are omitted:
`Display` (obsolete graphics adapters) and `Execute` (shell to DOS).

Rules of the original encoded below:

* Committing a dialog records a `Name := Value` expression in the history for each
  changed field; the same assignment can be authored directly. `Options Color` and
  `Options Mute` record nothing (`recorded=False`).
* The record is written in the notation the new settings select, so `Options
  Output` affects its spacing and `Options Radix` its digits. See
  `Settings.assignment`.
* A choice field can prompt for a number instead, so `Radix` accepts any base from
  2 to 36. See `ChoiceField.prompt`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from rederive.model import worksheet
from rederive.model.expr import Kind, Node

#: Color choices, indexed as EGA text attributes 0-15 as Derive numbered them. Derive
#: asked for the number; the names are ours. Bright colors get their own names rather
#: than a "light" prefix, which keeps the menu narrow and gives each a distinct
#: mnemonic letter. 7 being lighter than 8 is the EGA palette's quirk, not a bug.
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

#: The named radix bases and their values.
BASES: dict[str, int] = {"Binary": 2, "Octal": 8, "Decimal": 10, "Hexadecimal": 16}

#: The choice that prompts for a number.
OTHER = "Other"

#: The notation style each arithmetic mode switches to. Changing Options Precision
#: also sets Options Notation (style and digits), as the manual and the original do.
#: Only the field the user changed is recorded. `engine.context.NOTATION_FOR` applies
#: the same rule to a temporary context.
NOTATION_FOR: dict[str, str] = {
    "Exact": "Rational",
    "Approximate": "Scientific",
    "Mixed": "Mixed",
}

_DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class NumberField:
    """A data entry field holding an integer, such as `Digits: 6`."""

    label: str
    message: str
    setting: str
    default: int | str
    """The initial value, or the empty string to open blank.

    Only a dialog that stores nothing can open blank, since a setting always has a
    value. `Declare vectoR` does this: the original offers no default dimension.
    """
    minimum: int
    maximum: int | None = None
    recorded: bool = True
    word: str | None = None
    """A word accepted in place of a number, as `Unremove` takes `end`.

    Such a field accepts letters as well as digits, and holds the word once entered.
    """

    def accepts(self, number: int) -> bool:
        return number >= self.minimum and (self.maximum is None or number <= self.maximum)


@dataclass(frozen=True)
class TextField:
    """A data entry field holding text, such as an interval bound.

    Unlike a number field, it accepts anything typed; validation is left to the
    command that opened the dialog, since it involves parsing expressions.
    """

    label: str
    message: str
    setting: str
    default: str
    width: int
    recorded: bool = False


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
    """False when there are too many choices to show on the band, as with colors.

    Such a field shows only its current value and offers the choices as a submenu.
    """
    prompt: NumberField | None = None
    """The number prompt shown when `Other` is chosen."""


Field = ChoiceField | NumberField | TextField

#: A dialog line item: a field or a plain word printed between fields, such as the
#: variable name shown between its domain bounds.
Item = Field | str


@dataclass(frozen=True)
class Dialog:
    """One Options screen: a title and fields, grouped into the band's lines."""

    title: str
    lines: tuple[tuple[Item, ...], ...]
    stored: bool = True
    """Whether committing writes the fields to the settings.

    False for a dialog about the current command rather than the system, such as
    which expressions `Transfer Save` writes. Such a dialog opens on its fields'
    defaults and returns its values to the command that opened it.
    """
    keeps_menu: bool = False
    """Whether committing returns to the menu the dialog was opened from.

    Most dialogs complete the command and return to the command menu. `Transfer Save
    Options` keeps its menu up, since the save itself comes next.
    """
    tracks_selection: bool = False
    """Whether moving through the history writes the selected expression's label
    into the active field.

    True for `Remove` and `Unremove`, whose fields name expressions; the manual
    recommends picking with arrow keys as less error prone than typing labels.
    """
    opens_on: str = ""
    """The setting of the field initially highlighted, if not the first.

    `Calculus Vector` (Start, End, Step) opens on End, the only field without a
    default.
    """
    enters: bool = False
    """Whether committing enters an expression, which makes Ctrl-Enter meaningful.

    True for the Calculus dialogs, which append an unevaluated expression that
    Ctrl-Enter then simplifies.
    """

    @property
    def fields(self) -> tuple[Field, ...]:
        return tuple(
            item for line in self.lines for item in line if not isinstance(item, str)
        )


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


#: Upper limits for a surface's grid and a data point's size. The grid limit matches
#: the 3D window's own clamp, so a state file value that would be silently reduced is
#: refused instead; points larger than 20 pixels are just blobs.
PLOT_GRID_MAXIMUM = 256
PLOT_POINT_MAXIMUM = 20

# Sticky plot preferences: a new surface's grid and a new data plot's points start
# out as the last ones were left. They have no dialog (they're set from the 3D
# window's grid fields and a data plot's context menu, reported back by the plot
# host), but are settings so that the state file preserves them. They are never
# recorded, since toggling a plot window must not write to the worksheet.
#
# Equal scales is intentionally not sticky: a new window always opens with equal
# scales, and the `1:1` toggle handles exceptions. A one-off choice silently becoming
# the default would distort the next circle.
PLOT_FIELDS: tuple[Field, ...] = (
    NumberField(
        "Grid",
        "Grid points along each axis of a new surface",
        "PlotGrid",
        64,
        minimum=2,
        maximum=PLOT_GRID_MAXIMUM,
        recorded=False,
    ),
    ChoiceField(
        "Points",
        "How a new data plot draws its points",
        "PlotPoints",
        ("Discrete", "Connected"),
        "Discrete",
        recorded=False,
    ),
    NumberField(
        "Size",
        "Data point size in pixels",
        "PlotPointSize",
        5,
        minimum=1,
        maximum=PLOT_POINT_MAXIMUM,
        recorded=False,
    ),
    ChoiceField(
        "Mesh",
        "How a new surface is drawn",
        "PlotMesh",
        ("Solid", "Wire"),
        "Wire",
        recorded=False,
    ),
)


# -- the Manage screens ------------------------------------------------------
#
# The four Manage commands that set something. Each chooses the direction in which
# the simplifier applies a family of reversible transformations.

#: Directions for Exponential, Logarithm and Trigonometry. Collect applies the
#: transformations rightward and Expand leftward; Auto picks a direction per
#: transformation.
_DIRECTIONS = ("Auto", "Collect", "Expand")

BRANCH = Dialog(
    "MANAGE BRANCH:",
    (
        (
            # The only selection field with no caption, as in the original.
            ChoiceField(
                "",
                "Select preferred branch for roots",
                "Branch",
                ("Principal", "Real", "Any"),
                "Principal",
            ),
        ),
    ),
)

EXPONENTIAL = Dialog(
    "MANAGE EXPONENTIAL:",
    (
        (
            # The manual (section 6.1) says the default is Collect, but that is a
            # leftover from 1.x (`*EXP-EXPD* |Collect|` in the older DERIVE.INI).
            # The original's DERIVE.INI and screen both say Auto.
            ChoiceField(
                "Direction",
                "Exponential transformations",
                "Exponential",
                _DIRECTIONS,
                "Auto",
            ),
        ),
    ),
)

LOGARITHM = Dialog(
    "MANAGE LOGARITHM:",
    (
        (
            ChoiceField(
                "Direction",
                "Logarithm transformations",
                "Logarithm",
                _DIRECTIONS,
                "Auto",
            ),
        ),
    ),
)

# The program says MANAGE TRIGONOMETRY; the manual's "MANAGE TRIG" is wrong.
TRIGONOMETRY = Dialog(
    "MANAGE TRIGONOMETRY:",
    (
        (
            ChoiceField(
                "Direction",
                "Angle sums & multiple angles",
                "Trigonometry",
                _DIRECTIONS,
                "Auto",
            ),
            ChoiceField(
                "Toward",
                "Trig power transformations",
                "Trigpower",
                ("Auto", "Sines", "Cosines"),
                "Auto",
            ),
        ),
        (
            ChoiceField(
                "Angle",
                "Select angle mode",
                "Angle",
                ("Degree", "Radian"),
                "Radian",
            ),
        ),
    ),
)


# Options for the next `Transfer Save`. Nothing is recorded, and committing leaves
# the Transfer Save menu up.
SAVE = Dialog(
    "TRANSFER SAVE OPTIONS:",
    (
        (
            ChoiceField(
                "Range",
                "Select save range",
                "SaveRange",
                ("All", "Some"),
                "All",
                recorded=False,
            ),
            # The original's message is just `Annotation`; we phrase it as an
            # instruction like the other fields.
            ChoiceField(
                "Annotation",
                "Select annotation mode",
                "SaveAnnotation",
                ("Save", "Omit"),
                "Save",
                recorded=False,
            ),
            NumberField(
                "Length",
                "Enter line length",
                "SaveLength",
                worksheet.LINE_LENGTH,
                minimum=worksheet.MINIMUM_LENGTH,
                recorded=False,
            ),
        ),
    ),
    keeps_menu=True,
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


# The color defaults come from the last Derive version with colorful defaults. The
# version we otherwise imitate ships monochrome, so don't "correct" these to match
# its screenshots.
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
    BRANCH,
    EXPONENTIAL,
    LOGARITHM,
    TRIGONOMETRY,
    SAVE,
    COLOR_MENU,
    COLOR_WORK,
)

#: All setting fields: those in dialogs plus the sticky plot preferences.
_ALL_FIELDS: tuple[Field, ...] = (
    *(field for dialog in DIALOGS for field in dialog.fields),
    *PLOT_FIELDS,
)

DEFAULTS: dict[str, str | int] = {
    field.setting: field.default for field in _ALL_FIELDS
}

#: The sticky plot settings. The app watches them for changes to pass on to the plot
#: process.
PLOT_PREFERENCES: frozenset[str] = frozenset(field.setting for field in PLOT_FIELDS)

#: The field that owns each setting, for reading an authored `Name := Value`.
FIELDS: dict[str, Field] = {field.setting: field for field in _ALL_FIELDS}


def _read(field: Field, value: str) -> str | int | None:
    """The value an authored `Name := Value` sets `field` to, or None if invalid.

    Choices match case-insensitively, and numbers are accepted where the field
    prompts for one, so `InputBase := 7` works.
    """
    if isinstance(field, ChoiceField):
        for choice in field.choices:
            if choice != OTHER and choice.lower() == value.lower():
                return choice
        if field.prompt is None:
            return None
        field = field.prompt
    try:
        number = int(value)
    except ValueError:
        return None
    return number if field.accepts(number) else None


class Settings:
    """All system control settings, with change notification.

    Code that depends on a setting (the screen on colors, the parser on input
    modes) registers with `watch` to be told which settings changed.
    """

    def __init__(self) -> None:
        self._values: dict[str, str | int] = dict(DEFAULTS)
        self._watchers: list[Callable[[frozenset[str]], None]] = []

    def __getitem__(self, setting: str) -> str | int:
        return self._values[setting]

    def watch(self, callback: Callable[[frozenset[str]], None]) -> None:
        """Call `callback` with the names of the settings that just changed."""
        self._watchers.append(callback)

    def unwatch(self, callback: Callable[[frozenset[str]], None]) -> None:
        """Stop calling `callback`, as when a window's session closes."""
        if callback in self._watchers:
            self._watchers.remove(callback)

    def apply(self, values: Mapping[str, str | int]) -> tuple[str, ...]:
        """Store `values`, and return the names of those that actually changed.

        A precision setting also sets notation, unless the same call sets it too.
        See `NOTATION_FOR`.
        """
        values = _with_notation(values)
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

    def assign(self, setting: str, value: str) -> bool:
        """Apply an authored `Name := Value`, and return whether it took effect.

        Authoring `InputMode := Word` is equivalent to setting it in the Options
        Input dialog. Unknown settings and invalid values change nothing.
        """
        field = FIELDS.get(setting)
        if field is None:
            return False
        wanted = _read(field, value)
        if wanted is None:
            return False
        self.apply({setting: wanted})
        return True

    # -- writing values out ------------------------------------------------

    def assignment(self, setting: str) -> str:
        """The `Name := Value` expression Derive records for a settings change.

        It uses the current settings: `Options Output` decides the spacing around
        `:=` and `Options Radix` the digits.
        """
        space = "" if self._values["DisplayFormat"] == "Compressed" else " "
        return f"{setting}{space}:={space}{self.written(self._values[setting])}"

    def assignment_node(self, setting: str) -> Node:
        """The same record as an expression, built directly rather than parsed.

        Parsing wouldn't always work: `InputBase := 7` doesn't lex in base seven.

        Spans index into `assignment(setting)`. A number carries its decimal value
        and its written digits, as the lexer would report them.
        """
        value = self._values[setting]
        written = self.written(value)
        text = self.assignment(setting)
        start = len(text) - len(written)
        name = Node(Kind.NAME, 0, len(setting), (), setting)
        if isinstance(value, int):
            surface = None if written == str(value) else written
            right = Node(Kind.NUMBER, start, len(text), (), str(value), surface)
        else:
            right = Node(Kind.NAME, start, len(text), (), written)
        return Node(Kind.ASSIGN, 0, len(text), (name, right), ":=")

    def written(self, value: str | int) -> str:
        """`value` as it appears in an expression."""
        return self.numeral(value) if isinstance(value, int) else value

    def numeral(self, number: int) -> str:
        """`number` in the output radix.

        Digits above 9 are A to Z, and a numeral starting with a letter gets a
        leading zero so it isn't read as a name: fourteen is `0E` in hexadecimal.

        The display package duplicates this rule so the model doesn't depend on it.
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
        """The numeric value of a radix setting, named or numbered."""
        value = self._values[setting]
        return BASES[value] if isinstance(value, str) else value


class DialogEditor:
    """One Options dialog while it is on screen.

    The live settings are never touched; `commit` returns the values for the caller
    to apply, so Esc leaves everything unchanged.
    """

    def __init__(self, dialog: Dialog, settings: Settings) -> None:
        self.dialog = dialog
        # A dialog that stores nothing opens on its fields' defaults, which a caller
        # can compute (as the save range does from the history).
        self.values = {
            field.setting: settings[field.setting] if dialog.stored else field.default
            for field in dialog.fields
        }
        self.active = next(
            (
                index
                for index, field in enumerate(dialog.fields)
                if field.setting == dialog.opens_on
            ),
            0,
        )
        self.prompt: NumberField | None = None
        self.text: str | None = None
        """The text being edited, if the active field is typed into."""
        self.cursor = 0
        self.numbers = {
            field.setting: _as_number(self.values[field.setting])
            for field in dialog.fields
            if isinstance(field, ChoiceField) and field.prompt is not None
        }
        """The numeric value of each promptable field when the dialog opened.

        The `Other` prompt offers this value, not whatever choice the highlight
        passed over, as in the original.
        """
        self._enter_field()

    # -- what is on the band -----------------------------------------------

    @property
    def lines(self) -> tuple[tuple[Item, ...], ...]:
        """The dialog's lines, or the sole `Other` prompt once it is up."""
        if self.prompt is not None:
            return ((self.prompt,),)
        return self.dialog.lines

    @property
    def fields(self) -> tuple[Field, ...]:
        return tuple(
            item for line in self.lines for item in line if not isinstance(item, str)
        )

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

    def activate(self, index: int) -> bool:
        """Move the highlight to field `index`, as a mouse click does.

        This behaves like Tab: the field being left is stored, and an invalid number
        keeps the highlight in place for correction. A field left on `Other` opens
        its number prompt instead of moving. Clicking the active field does nothing.
        """
        if not 0 <= index < len(self.fields):
            return False
        if index == self.active:
            return True
        if self.open_prompt():
            return True
        if not self._leave_field():
            return False
        self.active = index
        self._enter_field()
        return True

    def focus(self, setting: str) -> None:
        """Move the highlight to the field holding `setting`, ready for editing.

        Used when Enter is refused, to put the highlight on the first invalid field
        with its typed text intact, as the original does.
        """
        for index, field in enumerate(self.fields):
            if field.setting == setting:
                self.active = index
                self._enter_field()
                return

    def open_prompt(self) -> bool:
        """Prompt for a number if the active field is on `Other`.

        As in the original, the prompt appears on Tab or Enter, not as soon as
        `Other` is selected.
        """
        field = self.field
        if (
            self.prompt is not None
            or not isinstance(field, ChoiceField)
            or field.prompt is None
            or self.value(field) != OTHER
        ):
            return False
        # The field now holds a number instead of `Other`; Esc abandons the whole
        # dialog anyway.
        self.prompt = field.prompt
        self.values[field.setting] = self.numbers[field.setting]
        self.active = 0
        self.text = str(self.numbers[field.setting])
        self.cursor = 0
        return True

    def _enter_field(self) -> None:
        """Start editing the active field, if it is one that is typed into."""
        field = self.field
        typed = isinstance(field, NumberField | TextField)
        self.text = str(self.value(field)) if typed else None
        self.cursor = 0

    def _leave_field(self) -> bool:
        """Store what was typed, if valid, and return whether it was.

        An invalid number is left as typed, so it can be corrected rather than
        retyped. A text field accepts anything; the calling command validates it.
        """
        field = self.field
        if self.text is None:
            return True
        if isinstance(field, TextField):
            self.values[field.setting] = self.text.strip()
            return True
        if not isinstance(field, NumberField):
            return True
        typed = self.text.strip()
        if field.word is not None and typed.lower() == field.word.lower():
            self.values[field.setting] = field.word
            return True
        number = int(typed) if typed.isdigit() else None
        if number is None or not field.accepts(number):
            return False
        self.values[field.setting] = number
        return True

    # -- changing a value --------------------------------------------------

    def choose(self, value: str) -> bool:
        """Select `value` in the active field and move to the next field.

        Return False if the active field has no such choice.
        """
        field = self.field
        if not isinstance(field, ChoiceField) or value not in field.choices:
            return False
        self.values[field.setting] = value
        self.next_field()
        return True

    def cycle(self, step: int = 1) -> bool:
        """Step the active field to its next choice, as Space does.

        Backspace steps the other way, with `step` of -1.
        """
        field = self.field
        if not isinstance(field, ChoiceField) or not field.inline:
            return False
        current = self.value(field)
        if current in field.choices:
            index = (field.choices.index(current) + step) % len(field.choices)
        else:
            # The field holds a prompted number, so enter the list from the end
            # nearest the step direction.
            index = 0 if step > 0 else len(field.choices) - 1
        self.values[field.setting] = field.choices[index]
        return True

    def lists_choices(self) -> bool:
        """Whether the active field offers its choices as a submenu."""
        field = self.field
        return isinstance(field, ChoiceField) and not field.inline

    def type_character(self, character: str) -> bool:
        """Type into the active data entry field, overwriting at the cursor.

        Number fields accept digits, plus letters if they take a word (validated
        when the field is left). Text fields accept any character.
        """
        field = self.field
        takes_letters = isinstance(field, TextField) or (
            isinstance(field, NumberField) and field.word is not None
        )
        if self.text is None or not (character.isdigit() or takes_letters):
            return False
        self.text = self.text[: self.cursor] + character + self.text[self.cursor + 1 :]
        self.cursor += 1
        return True

    def retype(self, text: str) -> None:
        """Replace the active field's text, as moving through the history does."""
        self.text = text
        self.cursor = 0

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
        """Move the cursor within the field, as the arrows and Home/End do."""
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
        """Whether choosing a value finishes the dialog.

        True for single-field dialogs, so `Options Mute` takes a single keystroke.
        """
        return len(self.fields) == 1

    def commit(self) -> dict[str, str | int] | None:
        """The values to apply, or None if the dialog isn't finished.

        A field left on `Other` opens its number prompt instead, and the next Enter
        commits.
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


def _with_notation(values: Mapping[str, str | int]) -> dict[str, str | int]:
    """`values` plus the notation settings implied by precision settings."""
    carried = dict(values)
    style = NOTATION_FOR.get(str(values.get("Precision", "")))
    if style is not None and "Notation" not in carried:
        carried["Notation"] = style
    if "PrecisionDigits" in values and "NotationDigits" not in carried:
        carried["NotationDigits"] = values["PrecisionDigits"]
    return carried


def _as_number(value: str | int) -> int:
    return BASES[value] if isinstance(value, str) else value

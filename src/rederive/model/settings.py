"""Derive's system control settings, and the screens that change them.

Most of those screens are Options commands; four of them are Manage commands,
which change how expressions are simplified rather than how the system behaves.
Both kinds are the same thing here, since a setting is a setting whichever menu
reaches it.

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

from rederive.model import worksheet
from rederive.model.expr import Kind, Node

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

#: What each arithmetic mode sets the notation style to, and the digit count
#: that follows the precision's. Options Precision carries Options Notation
#: with it - the manual says so, and the original does it - so that
#: approximate arithmetic is read in the style and the length that suit it.
#: The move is silent: only the field the user changed is recorded.
#: `engine.context.NOTATION_FOR` is the same rule for a temporary context.
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
    """What the field opens on; the empty string for one that opens blank.

    Only a dialog that stores nothing can open blank, since a setting always
    has a value. `Declare vectoR` is the one that does: the original offers no
    dimension until one has been entered.
    """
    minimum: int
    maximum: int | None = None
    recorded: bool = True
    word: str | None = None
    """A word the field takes in place of a number, as `Unremove` takes `end`.

    Such a field is typed into with letters as well as digits, and holds the
    word itself once one is entered.
    """

    def accepts(self, number: int) -> bool:
        return number >= self.minimum and (self.maximum is None or number <= self.maximum)


@dataclass(frozen=True)
class TextField:
    """A data entry field holding text, such as an interval bound.

    A number field takes digits and judges them for itself. This one takes
    whatever is typed and judges nothing: what a bound may say is a question
    about expressions, so the command that put the dialog up answers it.
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
    """False when the choices are too many to print on the band, as colors are.

    Such a field shows only its current value, and offers the rest as a submenu.
    """
    prompt: NumberField | None = None
    """What the `Other` choice asks for instead of a word."""


Field = ChoiceField | NumberField | TextField

#: What a dialog line may hold: fields, and plain words printed between them.
#: The bounds of a variable's domain are shown around the variable's own name,
#: which is a word on the line and not a field of it.
Item = Field | str


@dataclass(frozen=True)
class Dialog:
    """One Options screen: a title and fields, grouped into the band's lines."""

    title: str
    lines: tuple[tuple[Item, ...], ...]
    stored: bool = True
    """Whether committing writes the fields to the settings.

    False for a dialog that asks about the command in hand rather than about
    the system: the block of expressions `Transfer Save` writes is answered
    once and forgotten. Such a dialog opens on its fields' defaults and hands
    its values back to the command that put it up.
    """
    keeps_menu: bool = False
    """Whether committing goes back to the menu the dialog was opened from.

    Most dialogs are the whole command, so committing returns to the command
    menu. `Transfer Save Options` is not: you set the options and then issue
    the save they apply to, so the original leaves its menu up.
    """
    tracks_selection: bool = False
    """Whether walking the history writes what it lands on into the active field.

    True for the dialogs whose fields name expressions: `Remove` and
    `Unremove` let the arrow keys pick an expression instead of its label
    number, which the manual recommends as the less error prone of the two.
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


# -- the Manage screens ------------------------------------------------------
#
# The four Manage commands that set something. They say which way the
# simplifier applies a family of transformations it could apply either way, so
# the answer is a direction rather than a yes or no.

#: The three directions Exponential, Logarithm and Trigonometry each take.
#: Collect applies the transformations rightward and Expand leftward; Auto
#: mixes the two, one way for each transformation of the pair.
_DIRECTIONS = ("Auto", "Collect", "Expand")

BRANCH = Dialog(
    "MANAGE BRANCH:",
    (
        (
            # The one selection field in the program with no caption of its
            # own: there is nothing to say about it that the title has not
            # already said.
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
            # Section 6.1 of the manual says this field starts on Collect,
            # where the identically worded section 6.2 says Auto of Logarithm.
            # Collect was the 1.x default - the older DERIVE.INI carries
            # `*EXP-EXPD* |Collect|` - and 6.1 is a leftover of it: the
            # original's own DERIVE.INI says `*EXP-EXPD* |Auto|`, and so does
            # its screen.
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

# The band says MANAGE TRIGONOMETRY, though the manual's screen reproduction
# abbreviates it to MANAGE TRIG. The program is right and the manual wrong.
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


# The one dialog here that sets nothing lasting. It says how the next
# `Transfer Save` writes its file, which is why committing it leaves the
# Transfer Save menu up rather than returning to the command menu.
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
            # The original's message line for this field says only `Annotation`,
            # which tells a first-time user nothing; every other field on the
            # screen is phrased as an instruction, and so is this one.
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
    BRANCH,
    EXPONENTIAL,
    LOGARITHM,
    TRIGONOMETRY,
    SAVE,
    COLOR_MENU,
    COLOR_WORK,
)

DEFAULTS: dict[str, str | int] = {
    field.setting: field.default for dialog in DIALOGS for field in dialog.fields
}

#: The field that owns each setting, for reading an authored `Name := Value`.
FIELDS: dict[str, Field] = {
    field.setting: field for dialog in DIALOGS for field in dialog.fields
}


def _read(field: Field, value: str) -> str | int | None:
    """What an authored `Name := Value` sets `field` to, or None if nothing.

    A choice is matched however it was capitalised, and a number is read where
    the field takes one: `Options Radix` accepts `InputBase := 7` for the same
    reason its `Other` choice prompts for a base.
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
        """Store `values`, and return the names of those that actually changed.

        A precision setting brings its notation setting along, unless the same
        call sets that one too. See `NOTATION_FOR`.
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
        """Apply an authored `Name := Value`, and say whether it took.

        Authoring `InputMode := Word` changes the mode exactly as the Options
        Input dialog does, and the dialog then shows Word: the original keeps
        one store, reached from either side. A setting no dialog owns, or a
        value no field takes, changes nothing.
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

        Written the way the current settings say to write it: `Options Output`
        decides the spacing around `:=` and `Options Radix` the digits, so the
        record of a change is already in the notation that change selected.
        """
        space = "" if self._values["DisplayFormat"] == "Compressed" else " "
        return f"{setting}{space}:={space}{self.written(self._values[setting])}"

    def assignment_node(self, setting: str) -> Node:
        """The same record as an expression, built rather than parsed.

        Derive constructs the expression it records; it does not write the
        record out and read it back. `InputBase := 7` is why that matters: its
        text no longer lexes once the base it selects is in force, `7` not
        being a digit in base seven.

        The spans index `assignment(setting)`, so the two agree character for
        character. A number carries its decimal value and the digits it was
        written in, exactly as the lexer would have reported them.
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
        # A dialog that stores nothing has no settings behind its fields, so it
        # opens on their defaults - which is where a caller that computes them,
        # as the save block does from the history, puts the answer it offers.
        self.values = {
            field.setting: settings[field.setting] if dialog.stored else field.default
            for field in dialog.fields
        }
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

    def focus(self, setting: str) -> None:
        """Put the highlight on the field holding `setting`, ready to be retyped.

        Which is where the original leaves it when Enter is refused: on the
        first field the answer was wrong in, with what was typed still in it.
        """
        for index, field in enumerate(self.fields):
            if field.setting == setting:
                self.active = index
                self._enter_field()
                return

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
        typed = isinstance(field, NumberField | TextField)
        self.text = str(self.value(field)) if typed else None
        self.cursor = 0

    def _leave_field(self) -> bool:
        """Store what was typed, if the field takes it.

        A number the field will not have leaves the field as it is rather than
        putting the old value back, so that a mistyped digit can be corrected
        instead of retyped. A text field takes whatever is in it; only the
        command that put the dialog up knows what it can use.
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
        """Select `value` in the active field, and move on to the next one.

        False means the active field has no such choice.
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
            # A field holding a number typed into its prompt is on no choice
            # at all, so a step onto the list enters it from the near end.
            index = 0 if step > 0 else len(field.choices) - 1
        self.values[field.setting] = field.choices[index]
        return True

    def lists_choices(self) -> bool:
        """Whether the active field offers its choices as a submenu."""
        field = self.field
        return isinstance(field, ChoiceField) and not field.inline

    def type_character(self, character: str) -> bool:
        """Type into the active data entry field, overwriting at the cursor.

        Digits only, unless the field takes a word as well - then the letters
        that spell it are typed the same way, and what they spell is judged
        when the field is left. A text field takes any character at all.
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
        """Put `text` in the active field, as walking the history onto one does."""
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


def _with_notation(values: Mapping[str, str | int]) -> dict[str, str | int]:
    """`values` plus the notation settings a precision setting carries along."""
    carried = dict(values)
    style = NOTATION_FOR.get(str(values.get("Precision", "")))
    if style is not None and "Notation" not in carried:
        carried["Notation"] = style
    if "PrecisionDigits" in values and "NotationDigits" not in carried:
        carried["NotationDigits"] = values["PrecisionDigits"]
    return carried


def _as_number(value: str | int) -> int:
    return BASES[value] if isinstance(value, str) else value

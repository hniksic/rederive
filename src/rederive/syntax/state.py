"""Parse state: input modes and the live symbol table.

The lexer consults this table for every identifier: in Character mode `xyz` is `x*y*z`,
but `tera` is one name once it has been declared.
"""

from __future__ import annotations

import string
from collections.abc import Iterable
import unicodedata
from dataclasses import dataclass, field
from enum import StrEnum

from rederive import library
from rederive.syntax import names

_TO_LOWER = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)


def ascii_lower(text: str) -> str:
    """Case-fold ASCII letters only.

    Derive folds only `A`-`Z`, so `Θ` and `θ` are different names.
    """
    return text.translate(_TO_LOWER)


def fold(spelling: str) -> str:
    """The lookup key for a spelling, case-insensitive."""
    return ascii_lower(unicodedata.normalize("NFC", spelling))


def normalize(spelling: str) -> str:
    """The lookup key for a spelling, case-sensitive.

    Only lookup keys are normalized, not the source text, because NFC can change a
    string's length and invalidate reported offsets.
    """
    return unicodedata.normalize("NFC", spelling)


class InputMode(StrEnum):
    CHARACTER = "Character"
    WORD = "Word"


class CaseMode(StrEnum):
    INSENSITIVE = "Insensitive"
    SENSITIVE = "Sensitive"


class ClearWhat(StrEnum):
    VARIABLES = "Variables"
    FUNCTIONS = "Functions"
    ALL = "All"


@dataclass(frozen=True)
class VariableDeclaration:
    name: str
    has_value: bool = False


@dataclass(frozen=True)
class FunctionDeclaration:
    """A function `name` of parameters `params`.

    `variadic` marks the head `F v :=`, whose one parameter receives all the arguments
    of a call as a vector.
    """

    name: str
    params: tuple[str, ...] = ()
    has_body: bool = False
    variadic: bool = False


@dataclass(frozen=True)
class DomainDeclaration:
    name: str
    domain: str


@dataclass(frozen=True)
class SettingDeclaration:
    setting: str
    value: str


Declaration = (
    VariableDeclaration | FunctionDeclaration | DomainDeclaration | SettingDeclaration
)


@dataclass(frozen=True)
class FunctionInfo:
    name: str
    params: tuple[str, ...] = ()
    has_body: bool = False
    variadic: bool = False


@dataclass(frozen=True)
class VariableInfo:
    name: str
    has_value: bool = False
    domain: str | None = None


@dataclass(frozen=True)
class NameBinding:
    """A snapshot of what a name was bound to, for restoring it later.

    A name may be a variable, a function, or neither; all three cases are recorded.
    """

    name: str
    variable: VariableInfo | None = None
    function: FunctionInfo | None = None


@dataclass(frozen=True)
class NameInfo:
    """What a spelling resolves to."""

    canonical: str
    is_function: bool = False
    is_keyword_operator: bool = False
    is_constant: bool = False


def _table(entries: list[tuple[str, NameInfo]], key) -> dict[str, NameInfo]:
    """Index entries by `key` of their spelling; later entries win collisions."""
    return {key(spelling): info for spelling, info in entries}


_FUNCTION_ENTRIES = [
    (name, NameInfo(names.REWRITTEN_ALIASES.get(name, name), is_function=True))
    for name in names.KNOWN_FUNCTIONS
]
_KEYWORD_ENTRIES = [
    (word, NameInfo(word, is_keyword_operator=True))
    for word in names.DISPLACEABLE_OPERATORS
    | names.KEYWORD_OPERATORS
    | names.INFIX_WORDS
] + [(spelling, NameInfo(names.EXIT)) for spelling in (names.EXIT, names.EXIT.upper())]
_CONSTANT_ENTRIES = [(name, NameInfo(name, is_constant=True)) for name in names.CONSTANTS]
_WORD_ENTRIES = [(word, NameInfo(word)) for word in names.SETTING_VALUES | names.SETTINGS]

#: The names the session starts with, by exact spelling: Derive's functions, keyword
#: operators, constants, setting names and values, the Greek letters by name, and the
#: glyphs that spell a function or a constant.
_EXACT = _table(
    _WORD_ENTRIES
    + [(name, NameInfo(letter)) for name, letter in names.GREEK_LETTERS.items()]
    + [(glyph, NameInfo(letter)) for glyph, letter in names.DOS_GREEK.items()]
    + _KEYWORD_ENTRIES
    + _CONSTANT_ENTRIES
    + [
        (glyph, NameInfo(name, is_constant=True))
        for glyph, name in names.CONSTANT_GLYPHS.items()
    ]
    + _FUNCTION_ENTRIES
    + [
        (glyph, NameInfo(name, is_function=True))
        for glyph, name in names.FUNCTION_GLYPHS.items()
    ],
    normalize,
)

#: The same names by their ASCII-folded spelling, for `CaseMode := Insensitive`. A
#: function wins over a setting value of the same letters (`NORMAL` over `Normal`). The
#: Greek names are absent: their case is respected in both modes.
_FOLDED = _table(
    _WORD_ENTRIES + _KEYWORD_ENTRIES + _CONSTANT_ENTRIES + _FUNCTION_ENTRIES, fold
)

#: The constants, which match in any case spelling in both case modes.
_CONSTANTS = _table(_CONSTANT_ENTRIES, fold)

#: The functions of the pre-defined library by exact spelling, and by folded spelling for
#: `CaseMode := Insensitive`; see `ParseState.library_function`.
_LIBRARY = {normalize(name): name for name in library.FUNCTIONS}
_LIBRARY_FOLDED = {fold(name): name for name in library.FUNCTIONS}

#: The system variables of Derive's plot windows, the position of the cross. Plotting
#: doesn't set them, but they are names, also in Character mode, without a value, so
#: that a worksheet state assigning them reads.
PLOT_CROSS = ("hCross", "vCross", "xCross", "yCross", "zCross")

#: The variables of the pre-defined library, and the names of `PLOT_CROSS`, which read as
#: if the session had declared them.
_PREDEFINED = (*library.VARIABLES, *PLOT_CROSS)
_LIBRARY_VARIABLES = {name: NameInfo(name) for name in _PREDEFINED}
_LIBRARY_VARIABLES_FOLDED = {fold(name): NameInfo(name) for name in _PREDEFINED}


def reserved(spelling: str, function: bool = False) -> bool:
    """Whether no name can be bound with exactly this spelling.

    A binding of a setting's name assigns the setting, and a setting may be a parameter,
    but it can't name a function or be declared a domain, which `function` asks about.
    Reserved are the constants in every spelling, the keyword operators and keywords, the
    values of settings, and the built-in functions except those that yield to a
    definition. Another spelling of a reserved name is free: `perm` beside `PERM`.
    """
    exact = normalize(spelling)
    return (
        fold(exact) in _CONSTANTS
        or exact in names.KEYWORD_OPERATORS
        or exact in names.RESERVED_WORDS
        or exact in names.UNDEFINABLE_FUNCTIONS
        or exact in names.SETTING_VALUES
        or (function and exact in names.SETTINGS)
    )


@dataclass
class ParseState:
    """Input modes and the live symbol table.

    Defaults are Character mode, case-insensitive, base 10.

    Names are case-distinct in both case modes: `perm` and `PERM`, `h` and `H` are
    different names, and the tables are keyed by exact spelling. The case mode decides
    only what a spelling that matches no name exactly refers to; see `resolve`.

    The names of the pre-defined library are known from the start, except those in
    `withdrawn`, which the session cleared (see `rederive.library`). A library variable
    reads as a declared one; a library function is a name only before `(`, which the
    lexer asks about (`library_function`).

    `reading_file` is set while the text of a file is read, where an unknown name
    directly before `(` is a function, never a factor. `restoring` is set while the state
    block of a worksheet is read, whose text Derive wrote as it displays it: there a
    user's function name that an operand follows without parentheses, `k x`, is a
    factor, as it was when the line was saved, whatever the state defines later.
    """

    input_mode: InputMode = InputMode.CHARACTER
    case_mode: CaseMode = CaseMode.INSENSITIVE
    input_base: int = 10
    functions: dict[str, FunctionInfo] = field(default_factory=dict)
    variables: dict[str, VariableInfo] = field(default_factory=dict)
    revision: int = 0
    reading_file: bool = False
    restoring: bool = False
    withdrawn: frozenset[str] = frozenset()

    _index: dict[str, NameInfo] = field(default_factory=dict, repr=False)
    _folded: dict[str, NameInfo] = field(default_factory=dict, repr=False)
    _index_stamp: tuple[int, ...] = field(default=(-1,), repr=False)

    @property
    def stamp(self) -> tuple:
        """Everything a lexed token depends on.

        The lexer's per-position token cache is valid only while this is unchanged.
        """
        return (
            self.revision,
            len(self.functions),
            len(self.variables),
            self.input_mode,
            self.case_mode,
            self.input_base,
            self.reading_file,
            self.restoring,
            self.withdrawn,
        )

    def resolve(self, spelling: str) -> NameInfo | None:
        """What `spelling` refers to, or None if it's unknown.

        First the name with exactly this spelling: a name the session declared, else one
        it starts with. A Greek name refers to its letter, which may itself be declared.
        Then, under `CaseMode := Insensitive` only, a name equal when ASCII letters are
        folded, where the names the session starts with win over declared ones, so `Perm`
        is `PERM` even beside a user's `perm`. Constants match in any case spelling in
        both modes. A library variable comes after the other names in each step.
        """
        self._reindex()
        exact = normalize(spelling)
        found = self._index.get(exact)
        if found is not None:
            return found
        insensitive = self.case_mode == CaseMode.INSENSITIVE
        folded = fold(spelling)
        found = _EXACT.get(exact)
        if found is not None:
            if found.canonical != exact:
                declared = self._index.get(found.canonical)
                if declared is not None and not found.is_function:
                    return declared
            loose = _FOLDED.get(folded) if insensitive else None
            if loose is not None and loose.is_function and exact in names.SETTING_VALUES:
                # `Normal` and `Expand` are setting values and, in this mode, functions;
                # the parser takes the function's name for the value where it stands
                # alone.
                return loose
            return found
        found = _CONSTANTS.get(folded)
        if found is not None:
            return found
        found = self._library_variable(exact)
        if found is not None or not insensitive:
            return found
        return (
            _FOLDED.get(folded)
            or self._folded.get(folded)
            or self._library_variable(folded, folded=True)
        )

    def _library_variable(self, key: str, folded: bool = False) -> NameInfo | None:
        """The library variable spelled `key`, or folded to it, unless withdrawn."""
        found = (_LIBRARY_VARIABLES_FOLDED if folded else _LIBRARY_VARIABLES).get(key)
        return None if found is None or found.canonical in self.withdrawn else found

    def library_function(self, spelling: str) -> str | None:
        """The library function `spelling` names where a call of it is written, or None.

        That is the function spelled exactly so, unless the session knows this exact
        spelling as a name of its own or a built-in; under `CaseMode := Insensitive`,
        else the one equal when ASCII letters are folded, unless a built-in is. A
        function the session withdrew is no longer one.
        """
        exact = normalize(spelling)
        if self.knows_exactly(exact):
            return None
        name = _LIBRARY.get(exact)
        if name is None and self.case_mode == CaseMode.INSENSITIVE:
            folded = fold(exact)
            if folded not in _FOLDED:
                name = _LIBRARY_FOLDED.get(folded)
        return None if name is None or name in self.withdrawn else name

    def withdraw(self, names: Iterable[str]) -> None:
        """Take the library's definitions of `names` away for the rest of the
        session, as clearing them does."""
        self.withdrawn = self.withdrawn | (frozenset(names) & library.NAMES)

    def lookup(self, spelling: str) -> str | None:
        """Canonical name for a spelling, or None if unknown."""
        found = self.resolve(spelling)
        return found.canonical if found is not None else None

    def is_function(self, name: str) -> bool:
        """Whether `name` is a function: one `resolve` finds, or a library function of
        exactly this spelling."""
        found = self.resolve(name)
        if found is not None:
            return found.is_function
        exact = normalize(name)
        return exact in _LIBRARY and exact not in self.withdrawn

    def is_declared(self, name: str) -> bool:
        """Whether the session declared `name`, spelled exactly so."""
        return name in self.functions or name in self.variables

    def knows_exactly(self, spelling: str) -> bool:
        """Whether `spelling` names something as spelled, without folding case: a name
        the session declared, one it starts with, or a constant (in any spelling)."""
        exact = normalize(spelling)
        return self.is_declared(exact) or exact in _EXACT or fold(exact) in _CONSTANTS

    def bound_name(self, spelling: str) -> str:
        """The name a binding of `spelling` denotes, as in `spelling := ...`.

        A binding never folds case: it denotes the name with exactly that spelling,
        creating it if there is none. A Greek name binds its letter, so `gamma(k) :=`
        defines a function named `γ`, and so does a Greek letter of the DOS character
        set.
        """
        exact = normalize(spelling)
        if self.is_declared(exact):
            return exact
        return names.GREEK_LETTERS.get(exact) or names.DOS_GREEK.get(exact) or exact

    def canonical_variable(self, spelling: str) -> str:
        """How a variable spelling is recorded.

        A known name keeps its spelling; a new one is lowercased unless case is
        significant, so `R` and `r` are the same undeclared variable.
        """
        known = self.resolve(spelling)
        if known is not None and not _is_word_operator(known):
            return known.canonical
        if self.case_mode == CaseMode.SENSITIVE:
            return normalize(spelling)
        lowered = ascii_lower(normalize(spelling))
        found = self.resolve(lowered)
        if found is not None and not found.is_function and not _is_word_operator(found):
            # `BETA` lowered is `beta`, the Greek name of `β`, which a file would spell
            # the same way.
            return found.canonical
        return lowered

    def declare(self, decl: Declaration) -> None:
        """Apply one declaration to this state.

        An empty definition of a built-in that yields to definitions gives the name
        back to the built-in.
        """
        match decl:
            case VariableDeclaration(name=name, has_value=has_value):
                # An assignment overrides an earlier domain declaration.
                self.functions.pop(name, None)
                self.variables[name] = VariableInfo(name, has_value, None)
            case FunctionDeclaration(
                name=name, params=params, has_body=has_body, variadic=variadic
            ):
                self.variables.pop(name, None)
                if not has_body and name in names.YIELDING_FUNCTIONS:
                    self.functions.pop(name, None)
                else:
                    self.functions[name] = FunctionInfo(
                        name, params, has_body, variadic
                    )
            case DomainDeclaration(name=name, domain=domain):
                previous = self.variables.get(name)
                has_value = previous.has_value if previous is not None else False
                self.functions.pop(name, None)
                self.variables[name] = VariableInfo(name, has_value, domain)
            case SettingDeclaration(setting=setting, value=value):
                self._apply_setting(setting, value)
        self.revision += 1

    def binding(self, name: str) -> NameBinding:
        """The current binding of `name`, for `rebind` to restore later."""
        return NameBinding(name, self.variables.get(name), self.functions.get(name))

    def rebind(self, binding: NameBinding) -> None:
        """Restore a name to the state recorded in `binding`.

        Used for temporary declarations, such as a function parameter that must be known
        while the body is lexed. This bumps the revision like `declare` does, because
        the table size alone doesn't change when a name comes and goes, and the lexer
        cache depends on both.
        """
        if binding.variable is None:
            self.variables.pop(binding.name, None)
        else:
            self.variables[binding.name] = binding.variable
        if binding.function is None:
            self.functions.pop(binding.name, None)
        else:
            self.functions[binding.name] = binding.function
        self.revision += 1

    def tables(self) -> tuple[dict[str, FunctionInfo], dict[str, VariableInfo]]:
        """Copies of the function and variable tables, for `restore` to put back."""
        return dict(self.functions), dict(self.variables)

    def restore(
        self, tables: tuple[dict[str, FunctionInfo], dict[str, VariableInfo]]
    ) -> None:
        """Put back the tables `tables` copied, undoing every declaration since."""
        functions, variables = tables
        self.functions.clear()
        self.functions.update(functions)
        self.variables.clear()
        self.variables.update(variables)
        self.revision += 1

    def clear(self, what: ClearWhat) -> None:
        """Transfer Clear, which takes away the library's variables or functions too."""
        if what in (ClearWhat.VARIABLES, ClearWhat.ALL):
            self.variables.clear()
            self.withdraw(library.VARIABLES)
        if what in (ClearWhat.FUNCTIONS, ClearWhat.ALL):
            self.functions.clear()
            self.withdraw(library.FUNCTIONS)
        self.revision += 1

    def _apply_setting(self, setting: str, value: str) -> None:
        match setting:
            case "InputMode":
                self.input_mode = _INPUT_MODES.get(fold(value), self.input_mode)
            case "CaseMode":
                self.case_mode = _CASE_MODES.get(fold(value), self.case_mode)
            case "InputBase":
                self.input_base = _input_base(value, self.input_base)

    def _reindex(self) -> None:
        """Index the declared names exactly and folded.

        Among declared names that fold together, a function wins over a variable, and
        otherwise the one declared first.
        """
        stamp = self.stamp
        if stamp == self._index_stamp:
            return
        index: dict[str, NameInfo] = {}
        folded: dict[str, NameInfo] = {}
        for name in self.functions:
            info = NameInfo(name, is_function=True)
            index[normalize(name)] = info
            folded.setdefault(fold(name), info)
        for name in self.variables:
            info = NameInfo(name)
            index[normalize(name)] = info
            folded.setdefault(fold(name), info)
        self._index = index
        self._folded = folded
        self._index_stamp = stamp


def _is_word_operator(info: NameInfo) -> bool:
    """Whether `info` is a word operator of Derive 6, which a spelling other than its
    own names only where an operator can stand."""
    return info.is_keyword_operator and info.canonical in names.INFIX_WORDS


_INPUT_MODES = {"character": InputMode.CHARACTER, "word": InputMode.WORD}
_CASE_MODES = {
    "insensitive": CaseMode.INSENSITIVE,
    "sensitive": CaseMode.SENSITIVE,
}
_INPUT_BASES = {"binary": 2, "octal": 8, "decimal": 10, "hexadecimal": 16}


def _input_base(value: str, fallback: int) -> int:
    named = _INPUT_BASES.get(fold(value))
    if named is not None:
        return named
    try:
        return int(value)
    except ValueError:
        return fallback

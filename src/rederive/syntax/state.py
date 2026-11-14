"""Parse state: input modes and the live symbol table.

The lexer consults this table for every identifier: in Character mode `xyz` is `x*y*z`,
but `tera` is one name once it has been declared.
"""

from __future__ import annotations

import string
import unicodedata
from dataclasses import dataclass, field
from enum import StrEnum

from rederive.syntax import names

_TO_LOWER = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)
_TO_UPPER = str.maketrans(string.ascii_lowercase, string.ascii_uppercase)


def ascii_lower(text: str) -> str:
    """Case-fold ASCII letters only.

    Derive folds only `A`-`Z`, so `Θ` and `θ` are different names.
    """
    return text.translate(_TO_LOWER)


def ascii_upper(text: str) -> str:
    return text.translate(_TO_UPPER)


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
    name: str
    params: tuple[str, ...] = ()
    has_body: bool = False


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
    is_displaceable: bool = False
    """Set on a name a definition may override, such as a setting value."""


def _inventory(entries: list[tuple[str, NameInfo]]) -> tuple[dict[str, NameInfo], ...]:
    """Index an inventory both exactly and case-folded.

    Entries are listed lowest priority first, so later ones win collisions. For
    example, a function beats a setting value when `NORMAL` and `Normal` fold together.
    """
    exact: dict[str, NameInfo] = {}
    folded: dict[str, NameInfo] = {}
    for spelling, info in entries:
        exact[normalize(spelling)] = info
        folded[fold(spelling)] = info
    return exact, folded


# Names that a declaration cannot displace.
_RESERVED_EXACT, _RESERVED_FOLDED = _inventory(
    [(word, NameInfo(word, is_keyword_operator=True)) for word in names.KEYWORD_OPERATORS]
    + [(name, NameInfo(name, is_constant=True)) for name in names.CONSTANTS]
    + [
        (glyph, NameInfo(name, is_constant=True))
        for glyph, name in names.CONSTANT_GLYPHS.items()
    ]
    + [(name, NameInfo(name, is_function=True)) for name in names.BUILTIN_FUNCTIONS]
    + [
        (glyph, NameInfo(name, is_function=True))
        for glyph, name in names.FUNCTION_GLYPHS.items()
    ]
)

# Known names that a declaration displaces. Derive's utility files define `EXACT(...)`
# and `BETA(z,w)` and use `DEGREE_KELVIN`, so setting values and Greek letters must not
# outrank declared names.
_PREDECLARED_EXACT, _PREDECLARED_FOLDED = _inventory(
    [(value, NameInfo(value, is_displaceable=True)) for value in names.SETTING_VALUES]
    + [(setting, NameInfo(setting, is_displaceable=True)) for setting in names.SETTINGS]
    + [
        (name, NameInfo(names.GREEK_CANONICAL.get(name, name), is_displaceable=True))
        for name in names.GREEK_VARIABLES
    ]
    + [
        (glyph, NameInfo(names.GREEK_CANONICAL.get(name, name), is_displaceable=True))
        for glyph, name in names.GREEK_GLYPHS.items()
    ]
)


@dataclass
class ParseState:
    """Input modes and the live symbol table.

    Defaults are Character mode, case-insensitive, base 10.
    """

    input_mode: InputMode = InputMode.CHARACTER
    case_mode: CaseMode = CaseMode.INSENSITIVE
    input_base: int = 10
    functions: dict[str, FunctionInfo] = field(default_factory=dict)
    variables: dict[str, VariableInfo] = field(default_factory=dict)
    revision: int = 0

    _index: dict[str, NameInfo] = field(default_factory=dict, repr=False)
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
        )

    def resolve(self, spelling: str) -> NameInfo | None:
        """What `spelling` refers to, or None if it's unknown."""
        self._reindex()
        if self.case_mode == CaseMode.SENSITIVE:
            exact = normalize(spelling)
            found = (
                _RESERVED_EXACT.get(exact)
                or self._index.get(exact)
                or _PREDECLARED_EXACT.get(exact)
            )
            if found is not None:
                return found
            # Constants match case-insensitively even under `CaseMode := Sensitive`, but
            # functions don't.
            loose = _RESERVED_FOLDED.get(fold(spelling))
            return loose if loose is not None and loose.is_constant else None
        folded = fold(spelling)
        return (
            _RESERVED_FOLDED.get(folded)
            or self._index.get(folded)
            or _PREDECLARED_FOLDED.get(folded)
        )

    def same_name(self, one: str, other: str) -> bool:
        """Whether two spellings name the same thing under the current case mode."""
        if self.case_mode == CaseMode.SENSITIVE:
            return normalize(one) == normalize(other)
        return fold(one) == fold(other)

    def lookup(self, spelling: str) -> str | None:
        """Canonical name for a spelling, or None if unknown."""
        found = self.resolve(spelling)
        return found.canonical if found is not None else None

    def is_function(self, name: str) -> bool:
        found = self.resolve(name)
        return found is not None and found.is_function

    def canonical_variable(self, spelling: str) -> str:
        """How a variable spelling is recorded.

        A known name keeps its original spelling; a new one is lowercased unless case
        is significant.
        """
        known = self.lookup(spelling)
        if known is not None:
            return known
        if self.case_mode == CaseMode.SENSITIVE:
            return normalize(spelling)
        return ascii_lower(normalize(spelling))

    def canonical_function(self, spelling: str) -> str:
        """How a function name is recorded: uppercase, unless case is significant."""
        known = self.lookup(spelling)
        if known is not None and self.is_function(known):
            return known
        if self.case_mode == CaseMode.SENSITIVE:
            return normalize(spelling)
        return ascii_upper(normalize(spelling))

    def declare(self, decl: Declaration) -> None:
        """Apply one declaration to this state."""
        match decl:
            case VariableDeclaration(name=name, has_value=has_value):
                self.functions.pop(name, None)
                previous = self.variables.get(name)
                domain = previous.domain if previous is not None else None
                self.variables[name] = VariableInfo(name, has_value, domain)
            case FunctionDeclaration(name=name, params=params, has_body=has_body):
                self.variables.pop(name, None)
                self.functions[name] = FunctionInfo(name, params, has_body)
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
        """Transfer Clear."""
        if what in (ClearWhat.VARIABLES, ClearWhat.ALL):
            self.variables.clear()
        if what in (ClearWhat.FUNCTIONS, ClearWhat.ALL):
            self.functions.clear()
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
        stamp = self.stamp
        if stamp == self._index_stamp:
            return
        index: dict[str, NameInfo] = {}
        sensitive = self.case_mode == CaseMode.SENSITIVE
        key = normalize if sensitive else fold
        for name in self.variables:
            index[key(name)] = NameInfo(name)
        for name in self.functions:
            index[key(name)] = NameInfo(name, is_function=True)
        self._index = index
        self._index_stamp = stamp


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

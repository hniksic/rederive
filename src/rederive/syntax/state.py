"""Parse state: the modes plus the live symbol table.

Name lexing is not a self-contained lexical rule. In Character mode `xyz` is
`x*y*z` but `tera` is one name once `tera` has been declared, so the lexer has
to consult this table on every identifier.
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

    Derive folds `A`-`Z` and nothing else, so `Θ` is a name in its own right
    rather than a spelling of `θ`.
    """
    return text.translate(_TO_LOWER)


def ascii_upper(text: str) -> str:
    return text.translate(_TO_UPPER)


def fold(spelling: str) -> str:
    """The key a spelling is matched under, case-insensitively."""
    return ascii_lower(unicodedata.normalize("NFC", spelling))


def normalize(spelling: str) -> str:
    """The key a spelling is matched under, case-sensitively.

    Normalising the lookup key rather than the source text matters: NFC can
    change a string's length, which would invalidate every reported offset.
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
    """What one name stood for at some moment, enough to put it back.

    A name is a variable, or a function, or neither, so all three states have
    to be recorded for a scoped declaration to be undone exactly.
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
    """Set on a name a definition may take over, such as a setting value."""


def _inventory(entries: list[tuple[str, NameInfo]]) -> tuple[dict[str, NameInfo], ...]:
    """Index an inventory both exactly and case-folded.

    Listed lowest priority first, so that a function wins a collision against a
    setting value: `NORMAL` and `Normal` fold to the same key.
    """
    exact: dict[str, NameInfo] = {}
    folded: dict[str, NameInfo] = {}
    for spelling, info in entries:
        exact[normalize(spelling)] = info
        folded[fold(spelling)] = info
    return exact, folded


# Names that cannot be redefined. A declaration of one of these does not
# displace it.
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

# Names that are known but that a declaration displaces. Derive's own utility
# files define `EXACT(...)` and `BETA(z,w)` and use `DEGREE_KELVIN`, so a
# setting value or a Greek letter cannot outrank a declared name.
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
    """Modes plus the live symbol table.

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
        """Everything a lexed token can depend on.

        The lexer caches by position, and a cached token is only good while
        this is unchanged.
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
        """What `spelling` names here, or None if it names nothing yet."""
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
            # Constants are matched case-insensitively even under
            # `CaseMode := Sensitive`, but functions are not.
            loose = _RESERVED_FOLDED.get(fold(spelling))
            return loose if loose is not None and loose.is_constant else None
        folded = fold(spelling)
        return (
            _RESERVED_FOLDED.get(folded)
            or self._index.get(folded)
            or _PREDECLARED_FOLDED.get(folded)
        )

    def lookup(self, spelling: str) -> str | None:
        """Canonical name for a spelling, or None if unknown."""
        found = self.resolve(spelling)
        return found.canonical if found is not None else None

    def is_function(self, name: str) -> bool:
        found = self.resolve(name)
        return found is not None and found.is_function

    def canonical_variable(self, spelling: str) -> str:
        """How a variable spelling is recorded.

        A name already known keeps the spelling it was first written with; a
        new one is folded to lower case, which is the canonical spelling for
        variables.
        """
        known = self.lookup(spelling)
        if known is not None:
            return known
        if self.case_mode == CaseMode.SENSITIVE:
            return normalize(spelling)
        return ascii_lower(normalize(spelling))

    def canonical_function(self, spelling: str) -> str:
        """How a function name is recorded: upper case, unless case matters."""
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
        """What `name` stands for right now, for `rebind` to restore later."""
        return NameBinding(name, self.variables.get(name), self.functions.get(name))

    def rebind(self, binding: NameBinding) -> None:
        """Put a name back the way `binding` found it.

        For a declaration that is only good for part of a parse, such as the
        formal parameter of a definition, which has to be a known name while
        the body is lexed and must not outlive it. Taking a name away changes
        as much as adding one did, so this bumps the revision as `declare`
        does: a name that came and went again leaves the table the size it
        found it, and the lexer caches tokens against both.
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

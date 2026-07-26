"""The evaluation state every engine command reads.

`Context` is plain data - settings, variable domains, assignments, function
definitions and labels - and knows nothing about the session, so a test or a
command can build one directly. `Context.from_settings` is the bridge from the
session's `Settings` store and the parser's declarations.

A variable nobody declared is a real number, not a complex one;
`Context.domain` is where that default lives.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TypeVar

from rederive.engine.ordering import ORDER_LIST
from rederive.model.expr import Kind, Node
from rederive.model.settings import Settings
from rederive.syntax.state import Declaration, DomainDeclaration

_Choice = TypeVar("_Choice", bound=StrEnum)


class Precision(StrEnum):
    EXACT = "Exact"
    APPROXIMATE = "Approximate"
    MIXED = "Mixed"


class Notation(StrEnum):
    """How a number the engine works out is written. See `engine.notation`."""

    DECIMAL = "Decimal"
    MIXED = "Mixed"
    RATIONAL = "Rational"
    SCIENTIFIC = "Scientific"


class Branch(StrEnum):
    PRINCIPAL = "Principal"
    REAL = "Real"
    ANY = "Any"


class Direction(StrEnum):
    """What Exponential, Logarithm and Trigonometry each select."""

    AUTO = "Auto"
    COLLECT = "Collect"
    EXPAND = "Expand"


class TrigPower(StrEnum):
    AUTO = "Auto"
    SINES = "Sines"
    COSINES = "Cosines"


class Angle(StrEnum):
    RADIAN = "Radian"
    DEGREE = "Degree"


class DomainKind(StrEnum):
    """The domains a `:epsilon` declaration names.

    `UNKNOWN` is the parser's fallback for a domain it does not recognise; it
    is treated as Real, which is what an undeclared variable is anyway.
    """

    INTEGER = "Integer"
    REAL = "Real"
    COMPLEX = "Complex"
    NONSCALAR = "Nonscalar"
    UNKNOWN = "?"


@dataclass(frozen=True)
class Domain:
    """What `x :epsilon Real [0, inf)` says about `x`.

    The bounds are expression trees rather than numbers: a declaration may
    name a bound symbolically, and the conversion layer alone decides what
    facts a bound is worth.
    """

    kind: DomainKind = DomainKind.REAL
    low: Node | None = None
    high: Node | None = None
    closed_low: bool = False
    closed_high: bool = False

    @property
    def has_interval(self) -> bool:
        return self.low is not None or self.high is not None


#: The notation each precision mode selects. Options Precision moves Options
#: Notation with it, so that approximate arithmetic is read in the style that
#: suits it; the original does this silently, recording only the precision.
NOTATION_FOR: dict[Precision, Notation] = {
    Precision.EXACT: Notation.RATIONAL,
    Precision.APPROXIMATE: Notation.SCIENTIFIC,
    Precision.MIXED: Notation.MIXED,
}

#: What an undeclared variable is, where nothing says otherwise.
DEFAULT_DOMAIN = Domain()

#: The name that stands for every variable no declaration names. `Declare
#: Variable` takes it where it asks for a name, and `default :epsilon Complex`
#: is how the manual has you widen the domain of everything at once.
DEFAULT_NAME = "default"


def domain_of_node(node: Node) -> tuple[str, Domain] | None:
    """Read a `DOMAIN` node as a name and the domain it declares.

    The parser reports a `DomainDeclaration` for the same node, but that
    carries only the kind; the interval survives in the tree alone, so this is
    how a caller keeps it.
    """
    if node.kind is not Kind.DOMAIN or not node.children:
        return None
    name = str(node.children[0].value)
    kind = _choice(DomainKind, str(node.value), DomainKind.UNKNOWN)
    interval = next((c for c in node.children[1:] if c.kind is Kind.INTERVAL), None)
    if interval is None or len(interval.children) != 2:
        return name, Domain(kind)
    brackets = str(interval.value)
    return name, Domain(
        kind,
        low=interval.children[0],
        high=interval.children[1],
        closed_low=brackets.startswith("["),
        closed_high=brackets.endswith("]"),
    )


#: A user function: its parameter names and the body they stand in.
Definition = tuple[tuple[str, ...], Node]


@dataclass(frozen=True)
class Context:
    """Everything an engine command's answer may depend on.

    The factory values are exact arithmetic to six digits, principal branch,
    every transformation mode on Auto, radians, base ten, and the variable
    order list Derive starts with.
    """

    precision: Precision = Precision.EXACT
    precision_digits: int = 6
    notation: Notation = Notation.RATIONAL
    notation_digits: int = 6
    branch: Branch = Branch.PRINCIPAL
    exponential: Direction = Direction.AUTO
    logarithm: Direction = Direction.AUTO
    trigonometry: Direction = Direction.AUTO
    trigpower: TrigPower = TrigPower.AUTO
    angle: Angle = Angle.RADIAN
    input_base: int = 10
    order: tuple[str, ...] = ORDER_LIST
    """The variable order list, most main first, as `Manage Ordering` left it.

    Session state rather than a setting: it is a list of variable names and not
    a value a dialog field takes, nothing records it as a `Name := Value`, and
    `Settings` holds exactly what the Options and Manage screens own. What it
    shares with the settings is that a command's answer depends on it, which is
    what this class is, so it travels here.
    """
    domains: Mapping[str, Domain] = field(default_factory=dict)
    assignments: Mapping[str, Node] = field(default_factory=dict)
    functions: Mapping[str, Definition] = field(default_factory=dict)
    labels: Mapping[int, Node] = field(default_factory=dict)

    def domain(self, name: str) -> Domain:
        """What is known about the variable `name`.

        A variable nobody named falls back on whatever `default` was declared,
        and on Real where that was not declared either.
        """
        declared = self.domains.get(name)
        if declared is not None:
            return declared
        return self.domains.get(DEFAULT_NAME, DEFAULT_DOMAIN)

    def with_precision(
        self, precision: Precision, digits: int | None = None
    ) -> Context:
        """The same context evaluated at another precision, as approX asks for.

        The notation goes with it, style and digits both. Changing the
        precision changes the style numbers are written in, and asking for a
        number of digits asks for that many to be shown - the manual says both,
        and the original does both. That is what makes approX answer in
        scientific notation however the notation is set (`approX 10^7·π` is
        `3.14159·10⁷` under Rational too), and what makes approX to twelve
        digits show twelve. The style an answer was written in is the style it
        keeps, since a render is never remade.
        """
        return replace(
            self,
            precision=precision,
            precision_digits=self.precision_digits if digits is None else digits,
            notation=NOTATION_FOR[precision],
            notation_digits=self.notation_digits if digits is None else digits,
        )

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        declarations: Iterable[Declaration] = (),
        *,
        order: Sequence[str] | None = None,
        domains: Mapping[str, Domain] | None = None,
        assignments: Mapping[str, Node] | None = None,
        functions: Mapping[str, Definition] | None = None,
        labels: Mapping[int, Node] | None = None,
    ) -> Context:
        """Build a context from the session's settings and declarations.

        `declarations` supplies the domain *kinds* the session has recorded;
        `domains` overrides them where the caller kept an interval too, which
        only the `DOMAIN` node has.
        """
        declared = {
            declaration.name: Domain(
                _choice(DomainKind, declaration.domain, DomainKind.UNKNOWN)
            )
            for declaration in declarations
            if isinstance(declaration, DomainDeclaration)
        }
        declared.update(domains or {})
        return cls(
            precision=_setting(settings, "Precision", Precision, Precision.EXACT),
            precision_digits=_number(settings, "PrecisionDigits", 6),
            notation=_setting(settings, "Notation", Notation, Notation.RATIONAL),
            notation_digits=_number(settings, "NotationDigits", 6),
            branch=_setting(settings, "Branch", Branch, Branch.PRINCIPAL),
            exponential=_setting(settings, "Exponential", Direction, Direction.AUTO),
            logarithm=_setting(settings, "Logarithm", Direction, Direction.AUTO),
            trigonometry=_setting(
                settings, "Trigonometry", Direction, Direction.AUTO
            ),
            trigpower=_setting(settings, "Trigpower", TrigPower, TrigPower.AUTO),
            angle=_setting(settings, "Angle", Angle, Angle.RADIAN),
            input_base=settings.base("InputBase"),
            order=ORDER_LIST if order is None else tuple(order),
            domains=declared,
            assignments=dict(assignments or {}),
            functions=dict(functions or {}),
            labels=dict(labels or {}),
        )


def _choice(enum: type[_Choice], value: str, fallback: _Choice) -> _Choice:
    """The member `value` names, however it was capitalised."""
    for member in enum:
        if member.value.lower() == value.lower():
            return member
    return fallback


def _setting(
    settings: Settings, name: str, enum: type[_Choice], fallback: _Choice
) -> _Choice:
    """Read one setting as the member of `enum` that spells it.

    Every setting read here is owned by a dialog field, so the store carries
    it; the fallback is for a value that field's choices do not name.
    """
    return _choice(enum, str(settings[name]), fallback)


def _number(settings: Settings, name: str, fallback: int) -> int:
    value = settings[name]
    return value if isinstance(value, int) else fallback

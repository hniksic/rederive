"""The evaluation state every engine command reads.

`Context` is plain data (settings, variable domains, assignments, function definitions
and labels) independent of the session, so tests and commands can build one directly.
`Context.from_settings` builds one from the session's `Settings` and the parser's
declarations.

Undeclared variables are real, not complex; see `Context.domain`.
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
    """How computed numbers are written. See `engine.notation`."""

    DECIMAL = "Decimal"
    MIXED = "Mixed"
    RATIONAL = "Rational"
    SCIENTIFIC = "Scientific"


class Branch(StrEnum):
    PRINCIPAL = "Principal"
    REAL = "Real"
    ANY = "Any"


class Direction(StrEnum):
    """The choices for the Exponential, Logarithm and Trigonometry settings."""

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

    `UNKNOWN` is the parser's fallback for an unrecognized domain; it is treated as
    Real, like an undeclared variable.
    """

    INTEGER = "Integer"
    REAL = "Real"
    COMPLEX = "Complex"
    NONSCALAR = "Nonscalar"
    UNKNOWN = "?"


@dataclass(frozen=True)
class Domain:
    """What `x :epsilon Real [0, inf)` says about `x`.

    The bounds are expression trees, since they may be symbolic; the conversion layer
    decides what facts they imply.
    """

    kind: DomainKind = DomainKind.REAL
    low: Node | None = None
    high: Node | None = None
    closed_low: bool = False
    closed_high: bool = False

    @property
    def has_interval(self) -> bool:
        return self.low is not None or self.high is not None


#: The notation each precision mode selects. Setting Options Precision also changes the
#: notation, as Derive does silently (recording only the precision).
NOTATION_FOR: dict[Precision, Notation] = {
    Precision.EXACT: Notation.RATIONAL,
    Precision.APPROXIMATE: Notation.SCIENTIFIC,
    Precision.MIXED: Notation.MIXED,
}

#: The domain of an undeclared variable, unless `default` is declared.
DEFAULT_DOMAIN = Domain()

#: Name standing for every undeclared variable. `Declare Variable` accepts it, and the
#: manual uses `default :epsilon Complex` to widen the domain of all variables.
DEFAULT_NAME = "default"


def domain_of_node(node: Node) -> tuple[str, Domain] | None:
    """Read a `DOMAIN` node as a name and the domain it declares.

    The parser's `DomainDeclaration` for the node carries only the kind; the interval
    is available only from the tree.
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


#: A user function: its parameter names and its body.
Definition = tuple[tuple[str, ...], Node]


@dataclass(eq=False)
class Performed:
    """The assignments and definitions a command performed while computing, in order.

    A function body's `x := u` or `F(x) := u` is performed when the function is called,
    as in Derive, and takes effect at once: FIT_.MTH's `SUBST(u, x, x0) :=
    [SUBST_AUX1(u, x), SUBST_AUX2(x0)] * [0, 1]` defines `SUBST_AUX2` in the first
    element and calls it in the second. `nodes` are the `ASSIGN` and `FUNDEF` trees,
    which the answer carries back for the session to perform too (see
    `Result.performed`). `functions` and `assignments` are the meanings they gave, None
    for a name whose body or value was taken away.
    """

    nodes: list[Node] = field(default_factory=list)
    functions: dict[str, Definition | None] = field(default_factory=dict)
    assignments: dict[str, Node | None] = field(default_factory=dict)

    def perform(self, node: Node) -> None:
        """Record `node`, an `ASSIGN` of a name or a `FUNDEF`, and give its meaning."""
        if node.kind is Kind.FUNDEF:
            parameters = tuple(str(child.value) for child in node.children[0].children)
            body = node.children[1] if len(node.children) > 1 else None
            self.functions[str(node.value)] = None if body is None else (parameters, body)
        else:
            value = node.children[1] if len(node.children) > 1 else None
            self.assignments[str(node.children[0].value)] = value
        self.nodes.append(node)


@dataclass(frozen=True)
class Context:
    """Everything an engine command's answer may depend on.

    Defaults are exact arithmetic to six digits, principal branch, every
    transformation mode on Auto, radians, base ten, and Derive's initial order list.
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
    """The variable order list, most main first, as set by `Manage Ordering`.

    This is session state rather than a setting (it isn't a dialog field value or a
    `Name := Value`), but command answers depend on it, so it belongs here.
    """
    arbitrary_index: int = 1
    """The next free `@n` for soLve's arbitrary values.

    Session state, like the order list: `x = x` solves to `x = @1` and the counter
    only increases, so no two solves in a worksheet produce the same constant. It
    lives here so that answers depend only on the tree and the context.
    """
    domains: Mapping[str, Domain] = field(default_factory=dict)
    assignments: Mapping[str, Node] = field(default_factory=dict)
    functions: Mapping[str, Definition] = field(default_factory=dict)
    labels: Mapping[int, Node] = field(default_factory=dict)
    performed: Performed = field(default_factory=Performed, compare=False)

    def function(self, name: str) -> Definition | None:
        """The definition of function `name`, including one performed by this command."""
        if name in self.performed.functions:
            return self.performed.functions[name]
        return self.functions.get(name)

    def assigned(self, name: str) -> Node | None:
        """The value of variable `name`, including one performed by this command."""
        if name in self.performed.assignments:
            return self.performed.assignments[name]
        return self.assignments.get(name)

    def domain(self, name: str) -> Domain:
        """The domain of variable `name`.

        Undeclared variables get the domain declared for `default`, or Real.
        """
        declared = self.domains.get(name)
        if declared is not None:
            return declared
        return self.domains.get(DEFAULT_NAME, DEFAULT_DOMAIN)

    def with_precision(
        self, precision: Precision, digits: int | None = None
    ) -> Context:
        """The same context at another precision, as approX uses.

        The notation style and digits change with it, as in Derive. So approX answers
        in scientific notation regardless of the notation setting (`approX 10^7·π` is
        `3.14159·10⁷` under Rational too), and approX to twelve digits shows twelve.
        An answer keeps the style it was written in, since renders are never redone.
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
        arbitrary_index: int = 1,
        domains: Mapping[str, Domain] | None = None,
        assignments: Mapping[str, Node] | None = None,
        functions: Mapping[str, Definition] | None = None,
        labels: Mapping[int, Node] | None = None,
    ) -> Context:
        """Build a context from the session's settings and declarations.

        `declarations` supplies the recorded domain *kinds*; `domains` overrides them
        with full domains, including intervals, which only the `DOMAIN` node has.
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
            arbitrary_index=arbitrary_index,
            domains=declared,
            assignments=dict(assignments or {}),
            functions=dict(functions or {}),
            labels=dict(labels or {}),
        )


def _choice(enum: type[_Choice], value: str, fallback: _Choice) -> _Choice:
    """The member named by `value`, case-insensitively."""
    for member in enum:
        if member.value.lower() == value.lower():
            return member
    return fallback


def _setting(
    settings: Settings, name: str, enum: type[_Choice], fallback: _Choice
) -> _Choice:
    """Read a setting as a member of `enum`.

    Every setting read here belongs to a dialog field, so it is always present; the
    fallback covers values outside the field's choices.
    """
    return _choice(enum, str(settings[name]), fallback)


def _number(settings: Settings, name: str, fallback: int) -> int:
    value = settings[name]
    return value if isinstance(value, int) else fallback

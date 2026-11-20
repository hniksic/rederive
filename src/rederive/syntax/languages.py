"""Writing expression trees as source code in a programming language.

Like `writer.py`, this turns a tree into one line of text, parenthesizing an operand
that binds more loosely than its position requires.

The original offered Basic, C, Fortran and Pascal; Rederive offers C, Python and Julia
instead, as more useful targets today.

As in the original, a function the target has no equivalent for is passed through with
the target's spelling: `INT(SIN(x), x)` becomes `int(sin(x), x)`, which no compiler
accepts but any reader understands.

A numeral is written by its decimal value, whatever radix it was typed in and however
many zeros it had, since `0ff` or `010` would be wrong or invalid in every target. Derive
numbers are exact, so `1/3` is one third. Python's and Julia's `/` divide exactly that
way, but C's truncates when both operands are integers, so in C a division between two
integer-valued operands has the whole numerals in them written with a decimal point:
`1/3` is `1.0 / 3.0` and `(1 + 2)/4` is `(1.0 + 2.0) / 4.0`. An operand is
integer-valued when it is built of whole numerals by sums, products, quotients and signs
(or is a conditional choosing between two such); anything else, a name or a call,
is taken to be a `double`, so `x/3` stays as it is. The original wrote numerals as
typed, and so wrote `1/3` into C as an integer division.

Where a target does have an equivalent, it's used, which improves on the original:
`pi` becomes `M_PI`, `x=y` becomes `==`, and Julia's `log` gets its base first.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from rederive.model.expr import Kind, Node, parameter_names
from rederive.syntax.writer import write_expression

# Precedence levels shared by the three targets, loosest first. In all three, power binds
# tighter than a sign, and `a - b - c` associates to the left.
(
    ASSIGN,
    OR,
    AND,
    NOT,
    RELATION,
    SUM,
    NEG,
    PRODUCT,
    POWER,
    POSTFIX,
    ATOM,
) = range(11)

#: Pseudo-name under which the postfix factorial is looked up in `forms`.
FACTORIAL = "!"

#: How `a^b` is written: as a call or an infix operator.
POWER_CALL = "call"
POWER_OPERATOR = "operator"


@dataclass(frozen=True)
class Language:
    """A target language and how it spells Derive's functions and constants.

    `names` renames a Derive function regardless of arity. `forms` handles cases a
    rename can't express, and takes precedence over `names`. A form is a template that
    must parenthesize itself: `{0}` is an argument as a call argument.

    A name in neither table is passed through, spelled according to `spelling`.
    """

    word: str
    """The word on the Transfer Save menu; its capital letter is the mnemonic."""

    suffix: str
    comment: str
    spelling: str
    """`lower` or `keep`: how an unmapped name is spelled."""

    names: Mapping[str, str]
    forms: Mapping[tuple[str, int], str]
    constants: Mapping[str, str]
    relations: Mapping[str, str]
    logic: Mapping[Kind, str]
    power: str
    brackets: tuple[str, str] = ("[", "]")
    power_operator: str = "**"
    truncating: bool = False
    """Whether `/` between two integers truncates, so that such a division must be
    written with floating numerals; see the module docstring."""

    lambdas: bool = False
    """Whether `F(x) := u` is written as a lambda rather than an assignment.

    Used for Python, where a lambda is the only one-line way to define a function.
    """

    def spell(self, name: str) -> str:
        return name.lower() if self.spelling == "lower" else name


#: Trigonometric, hyperbolic and elementary functions as spelled in C's math.h, shared
#: by targets that spell them the same way.
_C_NAMES: dict[str, str] = {
    name: name.lower()
    for name in """
    SIN COS TAN ASIN ACOS ATAN SINH COSH TANH ASINH ACOSH ATANH
    SQRT EXP ERF ERFC FLOOR
    """.split()
}

_PYTHON_NAMES = {name: f"math.{spelling}" for name, spelling in _C_NAMES.items()}

C = Language(
    word="C",
    suffix=".c",
    comment="//",
    spelling="lower",
    names=_C_NAMES
    | {
        "LN": "log",
        "LOG": "log",
        "ABS": "fabs",
        "MOD": "fmod",
        "MAX": "fmax",
        "MIN": "fmin",
        "GAMMA": "tgamma",
    },
    forms={
        ("ATAN", 2): "atan2({0}, {1})",
        # C has no two-argument log, so use the change of base.
        ("LOG", 2): "(log({0}) / log({1}))",
        (FACTORIAL, 1): "tgamma({0} + 1)",
        ("IF", 3): "({0} ? {1} : {2})",
    },
    constants={
        "pi": "M_PI",
        "#e": "M_E",
        "inf": "INFINITY",
        "deg": "(M_PI / 180)",
    },
    relations={"=": "==", "/=": "!="},
    logic={Kind.AND: "&&", Kind.OR: "||", Kind.NOT: "!", Kind.XOR: "^"},
    power=POWER_CALL,
    truncating=True,
    # An initializer list, the closest C has to a vector literal.
    brackets=("{", "}"),
)

PYTHON = Language(
    word="Python",
    suffix=".py",
    comment="#",
    spelling="lower",
    names=_PYTHON_NAMES
    | {
        "LN": "math.log",
        "LOG": "math.log",
        "MOD": "math.fmod",
        "GAMMA": "math.gamma",
        # Builtins rather than math functions.
        "ABS": "abs",
        "MAX": "max",
        "MIN": "min",
    },
    forms={
        ("ATAN", 2): "math.atan2({0}, {1})",
        ("LOG", 2): "math.log({0}, {1})",
        (FACTORIAL, 1): "math.factorial({0})",
        ("IF", 3): "({1} if {0} else {2})",
    },
    constants={
        "pi": "math.pi",
        "#e": "math.e",
        "#i": "1j",
        "inf": "math.inf",
        "true": "True",
        "false": "False",
        "deg": "(math.pi / 180)",
    },
    relations={"=": "==", "/=": "!="},
    logic={Kind.AND: "and", Kind.OR: "or", Kind.NOT: "not ", Kind.XOR: "^"},
    power=POWER_OPERATOR,
    lambdas=True,
)

JULIA = Language(
    word="Julia",
    suffix=".jl",
    comment="#",
    spelling="lower",
    # Julia is the closest to Derive: `^` is exponentiation and Base has the elementary
    # and reciprocal trigonometric functions under Derive's names, so little needs
    # mapping. Julia's `log` is the natural log, so LN maps to it.
    names={"LN": "log"},
    forms={
        # Julia's two-argument log takes the base first, Derive's last.
        ("LOG", 2): "log({1}, {0})",
        (FACTORIAL, 1): "factorial({0})",
        ("IF", 3): "ifelse({0}, {1}, {2})",
    },
    constants={"#e": "ℯ", "#i": "im", "inf": "Inf", "deg": "(pi / 180)"},
    relations={"=": "==", "/=": "!="},
    logic={Kind.AND: "&&", Kind.OR: "||", Kind.NOT: "!", Kind.XOR: "⊻"},
    power=POWER_OPERATOR,
    power_operator="^",
)

#: The targets, in Transfer Save menu order.
LANGUAGES: tuple[Language, ...] = (C, PYTHON, JULIA)


def write_source(node: Node, language: Language) -> str:
    """Write `node` as one line of `language`."""
    return _Writer(language).operand(node, ASSIGN)


@dataclass
class _Writer:
    """Applies one language's spelling rules to a tree."""

    language: Language
    floating: set[int] = field(default_factory=set)
    """The `id` of each whole numeral to be written with a decimal point."""

    def operand(self, node: Node, required: int) -> str:
        """`node` as an operand that must bind at least as tightly as `required`."""
        text, binding = self.spell(node)
        return f"({text})" if binding < required else text

    def spell(self, node: Node) -> tuple[str, int]:
        """`node` without enclosing parentheses, and how tightly the result binds."""
        language = self.language
        match node.kind:
            case Kind.NUMBER:
                return self._numeral(node)
            case Kind.NAME:
                name = str(node.value)
                return language.constants.get(name, name), ATOM
            case Kind.STRING:
                return f'"{node.value}"', ATOM
            case Kind.SUM:
                return self._sum(node), SUM
            case Kind.PRODUCT:
                return self._product(node), PRODUCT
            case Kind.BINOP:
                return self._binop(node)
            case Kind.UNOP:
                return str(node.value) + self.operand(node.children[0], NEG), NEG
            case Kind.POSTOP:
                return self._postop(node)
            case Kind.ABS:
                return self.call("ABS", node.children), ATOM
            case Kind.SUB:
                # Not translated to indexing, since Julia indexes from one like Derive
                # and the others from zero. Written as a call, as in the original.
                return self.call("SUB", node.children), ATOM
            case Kind.CALL | Kind.APPLY:
                return self.call(str(node.children[0].value), node.children[1:]), ATOM
            case Kind.FUNCPOW:
                # `SIN^2(x)` is the call raised to a power, not iterated composition.
                name, exponent, operand = node.children
                call = self.call(str(name.value), (operand,))
                return self.power(call, exponent)
            case Kind.VECTOR:
                opening, closing = language.brackets
                return f"{opening}{self.arguments(node.children)}{closing}", ATOM
            case Kind.REL:
                return self._relation(node), RELATION
            case Kind.NOT:
                word = language.logic[Kind.NOT]
                return word + self.operand(node.children[0], NOT), NOT
            case Kind.AND | Kind.OR | Kind.XOR:
                return self._logical(node)
            case Kind.ASSIGN:
                return self._assignment(node), ASSIGN
            case Kind.FUNDEF:
                return self._definition(node), ASSIGN
        # Anything with no equivalent in the targets (a domain, an interval, an entry
        # label, the unknown value) is written in Derive notation, as the original did.
        # A leaf binds tightest; anything else is assumed to bind loosest, since extra
        # parentheses are harmless.
        return write_expression(node), ATOM if node.is_atom else ASSIGN

    def _numeral(self, node: Node) -> tuple[str, int]:
        """A numeral by its value; see the module docstring.

        A value with no finite decimal form, as `0.1` typed in base three, is a ratio,
        written as the quotient it is.
        """
        value = str(node.value)
        numerator, ratio, denominator = value.partition("/")
        if ratio:
            if self.language.truncating:
                numerator, denominator = f"{numerator}.0", f"{denominator}.0"
            return f"{numerator} / {denominator}", PRODUCT
        if id(node) in self.floating:
            return f"{value}.0", ATOM
        return value, ATOM

    # -- calls -------------------------------------------------------------

    def call(self, name: str, arguments: tuple[Node, ...]) -> str:
        """A function call, using the target's form or name for it if there is one."""
        language = self.language
        form = language.forms.get((name, len(arguments)))
        if form is not None:
            return self._formatted(form, arguments)
        spelling = language.names.get(name)
        if spelling is None:
            return f"{language.spell(name)}({self.arguments(arguments)})"
        return f"{spelling}({self.arguments(arguments)})"

    def _formatted(self, form: str, arguments: tuple[Node, ...]) -> str:
        """Fill a form's `{n}` slots with its arguments."""
        return form.format(*(self.operand(argument, ASSIGN) for argument in arguments))

    def arguments(self, children: tuple[Node, ...]) -> str:
        return ", ".join(self.operand(child, ASSIGN) for child in children)

    # -- operators ---------------------------------------------------------

    def _sum(self, node: Node) -> str:
        parts = [self.operand(node.children[0], NEG)]
        for index, child in enumerate(node.children[1:]):
            parts += [f" {str(node.value)[index]} ", self.operand(child, PRODUCT)]
        return "".join(parts)

    def _product(self, node: Node) -> str:
        """Factors joined by `*`.

        All three targets associate `*` and `/` to the left, so a factor that is itself a
        division can go unparenthesized only in first position: `a*b/c*d` needs no
        parentheses, but `a*(b/c)` does. The Derive writer follows the same rule.
        """
        return " * ".join(
            self.operand(child, PRODUCT if _leads(child, index) else POWER)
            for index, child in enumerate(node.children)
        )

    def _binop(self, node: Node) -> tuple[str, int]:
        operator = str(node.value)
        if operator == "^":
            base, exponent = node.children
            # Write `#e^u` as an exponential call, as the original did.
            if base.kind is Kind.NAME and base.value == "#e":
                return self.call("EXP", (exponent,)), ATOM
            return self.power(base, exponent)
        if operator == "/":
            if self.language.truncating and all(map(_integral, node.children)):
                for child in node.children:
                    self.floating.update(map(id, _whole_numerals(child)))
            left = self.operand(node.children[0], PRODUCT)
            return f"{left} / {self.operand(node.children[1], POWER)}", PRODUCT
        # The dot product, for which no target has an operator.
        left = self.operand(node.children[0], PRODUCT)
        return f"{left} . {self.operand(node.children[1], POWER)}", PRODUCT

    def power(self, base: Node | str, exponent: Node) -> tuple[str, int]:
        """`base` raised to `exponent`, in the target's notation.

        `base` may be already-written text, as when raising the call in `SIN^2(x)`.
        """
        language = self.language
        if language.power == POWER_CALL:
            # Call arguments need no parentheses.
            high = self.operand(exponent, ASSIGN)
            return f"pow({self._as_text(base, ASSIGN)}, {high})", ATOM
        # `^` and `**` associate to the right, so the exponent needs no parentheses.
        left = self._as_text(base, POSTFIX)
        return f"{left} {language.power_operator} {self.operand(exponent, POWER)}", POWER

    def _as_text(self, node: Node | str, required: int) -> str:
        return node if isinstance(node, str) else self.operand(node, required)

    def _postop(self, node: Node) -> tuple[str, int]:
        """The factorial, using the target's form for it if it has one."""
        operator = str(node.value)
        form = self.language.forms.get((operator, 1))
        if form is not None:
            return self._formatted(form, node.children), ATOM
        return self.operand(node.children[0], POSTFIX) + operator, POSTFIX

    def _relation(self, node: Node) -> str:
        operator = str(node.value)
        operator = self.language.relations.get(operator, operator)
        left = self.operand(node.children[0], SUM)
        return f"{left} {operator} {self.operand(node.children[1], SUM)}"

    def _logical(self, node: Node) -> tuple[str, int]:
        level = {Kind.OR: OR, Kind.XOR: OR, Kind.AND: AND}[node.kind]
        word = self.language.logic[node.kind]
        left = self.operand(node.children[0], level)
        return f"{left} {word} {self.operand(node.children[1], level + 1)}", level

    # -- definitions -------------------------------------------------------

    def _assignment(self, node: Node) -> str:
        """`x := u`, which every target spells `x = u`."""
        left = self.operand(node.children[0], POSTFIX)
        if len(node.children) < 2:
            # `x :=` clears a value and has no right side.
            return f"{left} ="
        return f"{left} = {self.operand(node.children[1], ASSIGN)}"

    def _definition(self, node: Node) -> str:
        """`F(x) := u`, as an assignment or a lambda.

        C gets an assignment, as in the original: a line the reader turns into a
        function, not one a compiler accepts.
        """
        parameters = ", ".join(parameter_names(node.children[0]))
        # `F(x) :=` declares an arbitrary function and has no body.
        body = self.operand(node.children[1], ASSIGN) if len(node.children) > 1 else ""
        if self.language.lambdas:
            head = f"{node.value} = lambda {parameters}:"
        else:
            head = f"{node.value}({parameters}) ="
        return f"{head} {body}" if body else head


def _leads(node: Node, index: int) -> bool:
    """Whether a factor needs no parentheses at `index`; see `_Writer._product`."""
    return index == 0 or not (node.kind is Kind.BINOP and node.value in ("/", "."))


def _integral(node: Node) -> bool:
    """Whether `node` is integer-valued in C; see the module docstring."""
    match node.kind:
        case Kind.NUMBER:
            return str(node.value).isdigit()
        case Kind.SUM | Kind.PRODUCT | Kind.UNOP:
            return all(map(_integral, node.children))
        case Kind.BINOP:
            return node.value == "/" and all(map(_integral, node.children))
        case Kind.CALL:
            return _is_conditional(node) and all(map(_integral, node.children[2:]))
    return False


def _whole_numerals(node: Node) -> list[Node]:
    """The numerals that make `node` integer-valued, for an `_integral` node: all of
    them but those in a conditional's test."""
    if node.kind is Kind.NUMBER:
        return [node]
    children = node.children[2:] if _is_conditional(node) else node.children
    return [numeral for child in children for numeral in _whole_numerals(child)]


def _is_conditional(node: Node) -> bool:
    """Whether `node` is `IF(test, then, else)`."""
    return (
        node.kind is Kind.CALL
        and node.children[0].value == "IF"
        and len(node.children) == 4
    )

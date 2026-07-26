"""Expression trees as source code in a programming language.

`writer.py` run against another notation. The job is the same - a tree in, one
line of text out - so the two live together, and the precedence rule is the
same one: an operand looser than its position gets parentheses.

The original offered Basic, C, Fortran and Pascal, the languages a 1990s reader
would paste an expression into. Rederive offers C, Python, Rust and Julia
instead. The command is the same command; only the list of targets is dated,
and dating it to 2026 is worth more than reproducing a Fortran writer nobody
will run.

What the original did with an expression the target has no form for, this does
too: the name is spelled the target's way and passed through. `INT(SIN(x), x)`
goes out as `int(sin(x), x)`, which no compiler accepts and every reader
understands. Numerals go out as they were written, so Derive's exact `1/3` is
`1/3` in C - integer division, and wrong - exactly as the original wrote it.
Widening it to `1.0/3.0` would be a guess about which of the two the reader
meant.

Where a target does have a form, it gets used, and that is the whole
improvement over the original: `pi` is `M_PI` rather than an undeclared
variable, `x=y` is `==` rather than an assignment, and Julia's `log` takes its
base first. Julia needs the fewest such rules and C the most, which is what
picking four languages a generation apart from each other buys.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from rederive.model.expr import Kind, Node
from rederive.syntax.writer import write_expression

# The precedence ladder the four targets agree on, loosest first. It is Derive's
# ladder with the power step renamed: all four bind `^`-or-equivalent tighter
# than a sign, and all four read `a - b - c` to the left.
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

#: The pseudo-name a postfix factorial is looked up under, `!` not being one.
FACTORIAL = "!"

#: How `a^b` is written: as a call, as an infix operator, or as a method on the
#: base. Rust is the odd one out, and the reason this is a field at all.
POWER_CALL = "call"
POWER_OPERATOR = "operator"
POWER_METHOD = "method"


@dataclass(frozen=True)
class Language:
    """One target language: how it spells the things Derive has names for.

    `names` renames a Derive function whatever its arity; `forms` is for the
    ones a rename cannot express, and wins over `names`. A form is a template
    over its arguments, and must fence itself: `{0}` is an argument as a call
    would take it, `{t0}` the same argument tight enough to be a method
    receiver or the base of a power.

    A name in neither table is passed through under `spelling`, which is what
    the original does with everything it has no opinion about.
    """

    word: str
    """The word on the Transfer Save menu, whose capital is its mnemonic."""

    suffix: str
    comment: str
    spelling: str
    """`lower` or `keep`: what an unmapped name is spelled as."""

    names: Mapping[str, str]
    forms: Mapping[tuple[str, int], str]
    constants: Mapping[str, str]
    relations: Mapping[str, str]
    logic: Mapping[Kind, str]
    power: str
    brackets: tuple[str, str] = ("[", "]")
    power_operator: str = "**"
    methods: bool = False
    """Whether a mapped name is written as a method on its first argument."""

    float_receivers: bool = False
    """Whether a whole numeral receiving a method is written with a point.

    Rust's `2.sqrt()` does not even tokenize - the lexer takes `2.` as the
    numeral - and `2.0.sqrt()` is the same number written so that it does.
    """

    lambdas: bool = False
    """Whether `F(x) := u` is written as a lambda rather than as an assignment.

    Python is the one target where a function definition is neither an
    assignment nor spellable in one line any other way.
    """

    def spell(self, name: str) -> str:
        return name.lower() if self.spelling == "lower" else name


#: The trigonometric, hyperbolic and elementary names, where the target spells
#: them the way C's math.h does. Shared so that a target only has to say how it
#: differs.
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
        # C has no two-argument log, and the change of base is the definition.
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
    # An initializer list, which is as close as C gets to a vector literal.
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
        # The two that are builtins rather than library functions.
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

RUST = Language(
    word="Rust",
    suffix=".rs",
    comment="//",
    spelling="lower",
    # Written out rather than shared with C: the two agree on the trigonometry
    # and part company after it, `f64` having no erf and no gamma.
    names={
        name: name.lower()
        for name in """
        SIN COS TAN ASIN ACOS ATAN SINH COSH TANH ASINH ACOSH ATANH
        SQRT EXP FLOOR ABS MAX MIN
        """.split()
    }
    | {"LN": "ln", "LOG": "ln", "SIGN": "signum"},
    forms={
        ("ATAN", 2): "{t0}.atan2({1})",
        ("LOG", 2): "{t0}.log({1})",
        # Derive's MOD takes the sign of its second argument, as this does and
        # as Rust's own `%` does not.
        ("MOD", 2): "{t0}.rem_euclid({1})",
        # Neither is a method, so both are written the way the original wrote
        # what it had no name for: as a call the reader has to supply.
        (FACTORIAL, 1): "factorial({0})",
        ("IF", 3): "(if {0} {{ {1} }} else {{ {2} }})",
    },
    constants={
        "pi": "std::f64::consts::PI",
        "#e": "std::f64::consts::E",
        "inf": "f64::INFINITY",
        "deg": "(std::f64::consts::PI / 180.0)",
    },
    relations={"=": "==", "/=": "!="},
    logic={Kind.AND: "&&", Kind.OR: "||", Kind.NOT: "!", Kind.XOR: "^"},
    power=POWER_METHOD,
    methods=True,
    float_receivers=True,
)

JULIA = Language(
    word="Julia",
    suffix=".jl",
    comment="#",
    spelling="lower",
    # Julia is the closest of the four to Derive: `^` is exponentiation, the
    # elementary functions are all in Base under the names Derive uses, and the
    # reciprocal trigonometric ones are there too, which is why the table is
    # short and holds no trigonometry at all. `log` is the natural one, so
    # Derive's two names for it meet there.
    names={"LN": "log"},
    forms={
        # Julia's two-argument log takes the base first, Derive's takes it last.
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

#: The targets, in the order the Transfer Save menu lists them.
LANGUAGES: tuple[Language, ...] = (C, PYTHON, RUST, JULIA)


def write_source(node: Node, language: Language) -> str:
    """`node` as one line of `language`."""
    return _Writer(language).operand(node, ASSIGN)


@dataclass
class _Writer:
    """One language's spelling rules, applied down a tree."""

    language: Language

    def operand(self, node: Node, required: int) -> str:
        """`node` where an operand binding at least `required` is expected."""
        text, binding = self.spell(node)
        return f"({text})" if binding < required else text

    def spell(self, node: Node) -> tuple[str, int]:
        """`node` unfenced, and how tightly what came out binds."""
        language = self.language
        match node.kind:
            case Kind.NUMBER:
                # As written, not as valued: see the module docstring.
                return str(node.surface or node.value), ATOM
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
                # Indexing, which no target can be given: Julia counts from one
                # as Derive does, the other three from zero. It passes through
                # as the call the original wrote for it.
                return self.call("SUB", node.children), ATOM
            case Kind.CALL | Kind.APPLY:
                return self.call(str(node.children[0].value), node.children[1:]), ATOM
            case Kind.FUNCPOW:
                # `SIN^2(x)` is the call raised to the power, as the original
                # writes it, and not an iterated composition.
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
        # Everything with no form in any of the four - a domain, an interval,
        # an entry label, the unknown value - passes through in Derive's own
        # notation, which is what the original does with what it cannot spell.
        # A leaf binds as tightly as anything; anything else is taken to bind
        # as loosely, since nothing here knows what it would bind like, and a
        # pair of parentheses too many is the harmless way to be wrong.
        return write_expression(node), ATOM if node.is_atom else ASSIGN

    # -- calls -------------------------------------------------------------

    def call(self, name: str, arguments: tuple[Node, ...]) -> str:
        """A function call, under whichever of the target's names fits."""
        language = self.language
        form = language.forms.get((name, len(arguments)))
        if form is not None:
            return self._formatted(form, arguments)
        spelling = language.names.get(name)
        if spelling is None:
            return f"{language.spell(name)}({self.arguments(arguments)})"
        if language.methods and arguments:
            receiver = self._receiver(arguments[0])
            return f"{receiver}.{spelling}({self.arguments(arguments[1:])})"
        return f"{spelling}({self.arguments(arguments)})"

    def _formatted(self, form: str, arguments: tuple[Node, ...]) -> str:
        """Fill a form's `{n}` and `{tn}` slots with its arguments."""
        loose = [self.operand(argument, ASSIGN) for argument in arguments]
        tight = {
            f"t{at}": self._receiver(argument)
            for at, argument in enumerate(arguments)
        }
        return form.format(*loose, **tight)

    def arguments(self, children: tuple[Node, ...]) -> str:
        return ", ".join(self.operand(child, ASSIGN) for child in children)

    # -- operators ---------------------------------------------------------

    def _sum(self, node: Node) -> str:
        parts = [self.operand(node.children[0], NEG)]
        for index, child in enumerate(node.children[1:]):
            parts += [f" {str(node.value)[index]} ", self.operand(child, PRODUCT)]
        return "".join(parts)

    def _product(self, node: Node) -> str:
        """A run of factors, `*` between them.

        All four targets read a run of `*` and `/` from the left, so a factor
        that is itself a division may stand bare only at the head, where there
        is nothing to its left for the `/` to take: `a*b/c*d` needs no fences
        and `a*(b/c)` needs them. This is the rule the Derive writer follows
        for the same reason.
        """
        return " * ".join(
            self.operand(child, PRODUCT if _leads(child, index) else POWER)
            for index, child in enumerate(node.children)
        )

    def _binop(self, node: Node) -> tuple[str, int]:
        operator = str(node.value)
        if operator == "^":
            base, exponent = node.children
            # `#e^u` is the exponential, which every target has a name for and
            # none of them has the constant to raise. The original writes it
            # that way too.
            if base.kind is Kind.NAME and base.value == "#e":
                return self.call("EXP", (exponent,)), ATOM
            return self.power(base, exponent)
        if operator == "/":
            left = self.operand(node.children[0], PRODUCT)
            return f"{left} / {self.operand(node.children[1], POWER)}", PRODUCT
        # The dot product, which none of the four has an operator for.
        left = self.operand(node.children[0], PRODUCT)
        return f"{left} . {self.operand(node.children[1], POWER)}", PRODUCT

    def power(self, base: Node | str, exponent: Node) -> tuple[str, int]:
        """`base` raised to `exponent`, however the target spells that.

        Rust asks which of its two methods applies, `powi` taking an integer
        exponent and `powf` a real one; the other three do not care.

        `base` is a node, except where the caller has already spelled it: a
        `SIN^2(x)` raises a call it had to build first.
        """
        language = self.language
        high = self.operand(exponent, ASSIGN)
        if language.power == POWER_CALL:
            # A call fences its own arguments, so the base needs nothing.
            return f"pow({self._as_text(base, ASSIGN)}, {high})", ATOM
        if language.power == POWER_METHOD:
            method = "powi" if _is_whole(exponent) else "powf"
            return f"{self._receiver(base)}.{method}({high})", ATOM
        # `^` and `**` both fold to the right, so the exponent nests bare.
        left = self._as_text(base, POSTFIX)
        return f"{left} {language.power_operator} {self.operand(exponent, POWER)}", POWER

    def _receiver(self, node: Node | str) -> str:
        """`node` as something a method can be called on."""
        text = self._as_text(node, POSTFIX)
        if self.language.float_receivers and text.isdigit():
            return f"{text}.0"
        return text

    def _as_text(self, node: Node | str, required: int) -> str:
        return node if isinstance(node, str) else self.operand(node, required)

    def _postop(self, node: Node) -> tuple[str, int]:
        """The factorial, which only two of the four have a name for."""
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
            # `x :=` takes a value away, and has no right side to write.
            return f"{left} ="
        return f"{left} = {self.operand(node.children[1], ASSIGN)}"

    def _definition(self, node: Node) -> str:
        """`F(x) := u`, as an assignment or as a lambda.

        C and Rust have neither, and take the assignment, which is what the
        original wrote for them - a line a reader turns into a function rather
        than one a compiler accepts.
        """
        parameters = ", ".join(str(child.value) for child in node.children[0].children)
        # `F(x) :=` leaves F an arbitrary function, and has no body to write.
        body = self.operand(node.children[1], ASSIGN) if len(node.children) > 1 else ""
        if self.language.lambdas:
            head = f"{node.value} = lambda {parameters}:"
        else:
            head = f"{node.value}({parameters}) ="
        return f"{head} {body}" if body else head


def _leads(node: Node, index: int) -> bool:
    """Whether a factor may stand bare where it is: see `_Writer._product`."""
    return index == 0 or not (node.kind is Kind.BINOP and node.value in ("/", "."))


def _is_whole(node: Node) -> bool:
    """Whether `node` is a whole-number literal, possibly signed."""
    if node.kind is Kind.UNOP and node.value == "-":
        return _is_whole(node.children[0])
    return node.kind is Kind.NUMBER and str(node.surface or node.value).isdigit()

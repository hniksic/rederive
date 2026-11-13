"""sympy -> author notation.

Converts every engine result to text: `^` for powers, `SQRT(u)`, `#e` and `#i`,
upper-case function names, `?` for the unknown value. The output must reparse to the
same expression, and must use the session's notation (input base and the `Options
Notation` style, handled by `engine.notation`). Decimal styles bend the first rule, as
in the original; the exact value is kept separately in `Result.value`.

`AuthorPrinter` subclasses sympy's `StrPrinter` and overrides each construct spelled
differently. For sums and products it overrides only operand order and minus-sign
placement, leaving sign extraction and denominators to sympy, so sympy's conventions
show through (`x*k^-2` prints as `x/k^2`) without changing meaning. A range is printed
as a chain `-2 < x < 2`, and a matrix product nested to the right as the original does:
`a . (b . (c . d))`.

Operand order follows the context's order list. A sum puts calls first, then terms by
variable and descending degree, with constants last: `2*x^2 - 3*a*x + 5*y^3`. A product
puts the number first, then constants, variables, powers of sums and calls: `(x + 2)*(x
- 2)*(x + 1)^2`. A two-term sum is swapped to avoid a leading minus (`y - x`), a longer
one isn't.

Inert heads are printed by class name (`_print_PlusMinus`), so this module needn't
import them.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from decimal import Decimal
from fractions import Fraction
from typing import ParamSpec, TypeVar

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.core.relational import Relational
from sympy.printing.precedence import (
    PRECEDENCE,
    PRECEDENCE_FUNCTIONS,
    PRECEDENCE_VALUES,
    precedence,
)

from rederive.engine import notation
from rederive.engine.approximation import is_approximation, reapproximated
from rederive.engine.context import Angle, Context, Notation, Precision
from rederive.engine.ordering import main_order
from rederive.syntax.names import (
    BUILTIN_FUNCTIONS,
    EXTENDED_FUNCTIONS,
    GREEK_GLYPHS,
    PRIME_MARK,
)

# Binding strength of the inert heads, at the precedence of the operator each prints
# as. Without an entry sympy treats a `Function` as tight as `SIN(x)`, printing
# `#e^(±inf*z)` as `#e^±inf*z`.
#
# `Dot` sits below `Add` rather than at `Mul`: `x*(a . b)` needs its parentheses, and
# sympy rates a negated product at the `Add` level, so `Mul` would lose them in
# `-x*(a . b)`. `MatMul` sits at `Add`, needing parentheses inside a power or
# transpose but not a sum.
PRECEDENCE_VALUES.setdefault("PlusMinus", PRECEDENCE["Add"])
PRECEDENCE_VALUES.setdefault("ComplexInfinity", PRECEDENCE["Add"])
PRECEDENCE_VALUES.setdefault("Dot", PRECEDENCE["Add"] - 1)
PRECEDENCE_VALUES.setdefault("MatMul", PRECEDENCE["Add"])
PRECEDENCE_VALUES.setdefault("Subscript", PRECEDENCE["Atom"] - 1)
PRECEDENCE_VALUES.setdefault("Power", PRECEDENCE["Pow"])
PRECEDENCE_VALUES.setdefault("Added", PRECEDENCE["Add"])
PRECEDENCE_VALUES.setdefault("Quotient", PRECEDENCE["Mul"])
# A truth value used as an arithmetic operand binds as tightly as its content.
PRECEDENCE_FUNCTIONS.setdefault("Truth", lambda expr: precedence(expr.args[0]))

_DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

#: The separator between the parts of a subscripted symbol's name.
_SUBSCRIPT = " SUB "

#: The plus-or-minus operator, as the glyph rather than its alternative input spelling
#: `"+-"`, since a bare `+-u` would read back as a plus over a minus, i.e. `-u`.
_PLUS_MINUS = "±"

#: Relations spelled differently from sympy; others are written as is.
_RELATIONS = {"==": "=", "!=": "/="}

#: Operators with the bound below the subject, mapped to their flipped form for a
#: chain: `x > -2` becomes `-2 < x`.
_FROM_BELOW = {">": "<", ">=": "<="}

#: The operators a chain can be built from.
_ORDERINGS = ("<", "<=", ">", ">=")

#: sympy heads whose author-notation name is not just their name upper-cased.
_FUNCTION_NAMES = {
    "Antidifference": "SUM",
    "Antiquotient": "PRODUCT",
    "MovingLimit": "LIM",
    "Substitution": "SUBS",
    "log": "LN",
    "arg": "PHASE",
    "atan2": "ATAN",
    "conjugate": "CONJ",
    "binomial": "COMB",
    "FallingFactorial": "PERM",
    "Heaviside": "STEP",
    # Upper-casing would give `CHI`, Derive's chi-square distribution; `Shi` is
    # renamed to match.
    "Chi": "COSH_INT",
    "Shi": "SINH_INT",
    "Determinant": "DET",
    "Trace": "TRACE",
}

#: Logical operator precedence, tightest first, to decide when an operand needs
#: parentheses. Relations and anything tighter never do.
_LOGICAL_PRECEDENCE = {"NOT": 5, "AND": 4, "OR": 3, "XOR": 2, "IMP": 1}

_LOGICAL_HEADS = {
    "And": "AND",
    "Or": "OR",
    "Xor": "XOR",
    "Implies": "IMP",
    "Not": "NOT",
}


def _logical_order(operands, order: Sequence[str]) -> list[sp.Basic]:
    """The operands of a conjunction or disjunction, in order-list order.

    `c OR b AND a` is written `a AND b OR c`. `NOT` doesn't move a term, but a negated
    literal comes before the same variable unnegated.
    """
    return sorted(operands, key=lambda operand: _logical_key(operand, order))


def _logical_key(operand: sp.Basic, order: Sequence[str]) -> tuple:
    """The sort key of one operand: its literals, in order, each with its sign.

    Always a tuple of literal keys, even for a single literal, so any two operands
    compare.
    """
    if isinstance(operand, (sp.And, sp.Or, sp.Xor)):
        return tuple(
            sorted(
                literal
                for argument in operand.args
                for literal in _logical_key(argument, order)
            )
        )
    negated = isinstance(operand, sp.Not)
    inner = operand.args[0] if negated else operand
    return ((_variable_key(inner, order), 0 if negated else 1),)


def _variable_key(operand: sp.Basic, order: Sequence[str]) -> tuple:
    """A sort key for the main variable of one literal: by order-list position, then
    alphabetically for variables off the list, then by string for literals with none.
    """
    names = main_order((symbol.name for symbol in operand.free_symbols), order)
    if not names:
        return (2, str(operand))
    name = names[0]
    if name in order:
        return (0, order.index(name))
    return (1, name)


#: Sort positions for a term's kernels, front of the sum first: calls, anything built
#: on a variable, sums in a denominator (`x^2 + y + 1/(x + 1)`), and constant kernels
#: (`SQRT(3) - 1`). `_NOTHING` ends every key, so purely numeric terms come after all
#: others, and `#i` after even those, so a complex number's real part comes first.
_CALLED, _BUILT, _UNDER, _CONSTANT, _NOTHING, _IMAGINARY = range(6)

#: Order of the two `_BUILT` kernels of one variable: a power of a compound base comes
#: before the bare variable, `SQRT(x + 1) + x`.
_OVER_A_SUM, _OVER_THE_VARIABLE = range(2)

#: The suffix of every term's key.
_AFTER_EVERY_KERNEL = (_NOTHING,)


def _term_order(terms: Sequence[sp.Basic], order: Sequence[str]) -> list[sp.Basic]:
    """The terms of a sum, in the original's order. The sort is stable, so terms the order
    list doesn't distinguish keep sympy's order, which makes printing a fixed point.
    """
    return sorted(terms, key=lambda term: _term_key(term, order))


def _term_key(term: sp.Basic, order: Sequence[str]) -> tuple:
    """The sort key of one term of a sum: its kernels, most main first.

    Comparing kernels in order makes `2*x^2 - 3*a*x + 5*y^3` put the x terms first and
    higher powers before lower. A sum in a denominator sorts first among a term's
    kernels, so the proper part of a rational function goes last; `#i` likewise, so a
    complex number's real part comes first. All-constant terms sort by `_constant_key`.
    """
    kernels = sorted(
        (
            _kernel_key(factor, order)
            for factor in sp.Mul.make_args(term)
            if not factor.is_Number
        ),
        key=lambda kernel: (kernel[0] != _UNDER, kernel[0] != _IMAGINARY, kernel),
    )
    if kernels and all(kernel[0] == _CONSTANT for kernel in kernels):
        return (_constant_key(term), _AFTER_EVERY_KERNEL)
    return (*kernels, _AFTER_EVERY_KERNEL)


def _constant_key(term: sp.Basic) -> tuple:
    """The sort key of a term with no variable: by value, largest first.

    This matches the manual, where the whole term's value, coefficient and sign
    included, decides: `SQRT(3) + SQRT(2)` (3.8 p.42) but `-SQRT(2)/8 - SQRT(6)/8 +
    1/2` (6.3 p.104). Non-real terms go last, by string.
    """
    try:
        return (_CONSTANT, 0, -float(term), "")
    except (TypeError, ValueError):
        return (_CONSTANT, 1, 0.0, str(term))


def _kernel_key(factor: sp.Basic, order: Sequence[str]) -> tuple:
    """The sort key of one kernel of a term: its kind, its variable, then a tiebreaker.

    A call comes first regardless of its variable (`SIN(y) + x`); other kernels compare
    by variable, then a power of a sum precedes the bare variable (`SQRT(x + 1) + x`),
    and powers run by descending degree. A sum in a denominator contributes no
    variable.
    """
    if factor is sp.I:
        return (_IMAGINARY,)
    kind, base, exponent, head = _kernel_parts(factor)
    if kind == _A_CONSTANT_KERNEL:
        return (_CONSTANT, str(factor))
    place, within = _variable_key(base, order)
    if kind == _A_CALLED_KERNEL:
        return (_CALLED, (place, within), _degree(exponent), _head_key(head))
    if kind == _A_BRACKET_KERNEL:
        if _under_the_bar(exponent):
            return (_UNDER, _degree(exponent))
        argument = _Reversed(_shape_key(base, exponent, order))
        return (_BUILT, (place, within), _OVER_A_SUM, argument)
    return (_BUILT, (place, within), _OVER_THE_VARIABLE, _degree(exponent))


#: Sort positions for a product's factors, leftmost first: the numeric coefficient,
#: other constants, plain variables, powers of sums, and calls, as in `2*pi*r^2` and
#: `x*SQRT(x + 1)`. The kind decides before the variable does.
_A_NUMBER, _A_CONSTANT, _A_VARIABLE, _A_POWER, _A_CALL = range(5)


def _is_integer_root(factor: sp.Basic) -> bool:
    """Whether `factor` is a positive integer raised to a fraction, like `3^(1/100)`."""
    return (
        isinstance(factor, sp.Pow)
        and factor.base.is_Integer
        and factor.base > 1
        and factor.exp.is_Rational
        and not factor.exp.is_Integer
    )


def _merged_roots(factors: Sequence[sp.Basic]) -> list[sp.Basic]:
    """The factors of a product, with roots of integers merged into one root.

    Sympy splits `12^(1/100)` into `2^(1/50)*3^(1/100)`, and the original recombines
    them. The merged power is built unevaluated so it stays merged. Unlike the
    original, roots aren't merged into a higher order (`2^(2/3)*SQRT(3)` stays) or when
    the result is longer.
    """
    merged = list(factors)
    for positive in (True, False):
        roots = [
            factor
            for factor in merged
            if _is_integer_root(factor) and bool(factor.exp.is_positive) == positive
        ]
        if len(roots) < 2:
            continue
        denominator = sp.ilcm(*(root.exp.q for root in roots))
        if denominator != max(root.exp.q for root in roots):
            continue
        powers = [abs(root.exp.p) * denominator // root.exp.q for root in roots]
        common = sp.igcd(*powers)
        radicand = sp.Integer(1)
        for root, power in zip(roots, powers):
            radicand *= root.base ** (power // common)
        exponent = sp.Rational(common, denominator) * (1 if positive else -1)
        root = sp.Pow(radicand, exponent, evaluate=False)
        if len(sp.sstr(root)) >= len(sp.sstr(sp.Mul(*roots, evaluate=False))):
            continue
        first = merged.index(roots[0])
        merged[first] = root
        merged = [factor for factor in merged if factor not in roots[1:]]
    return merged


def _raised(base: sp.Basic, power: sp.Basic) -> sp.Basic:
    """`base^power`, unevaluated where evaluation would split an integer root that
    `_merged_roots` combined."""
    if base.is_Integer and power.is_Rational and not power.is_Integer:
        return sp.Pow(base, power, evaluate=False)
    return base**power


def _factor_order(factors: Sequence[sp.Basic], order: Sequence[str]) -> list[sp.Basic]:
    """The factors of a product, in the original's order.

    Variables compare as in a sum, but kinds of factor run the other way (calls last),
    and calls of one variable are reversed: `SIN(x)*COS(x)` versus `COS(x) + SIN(x)`.
    """
    return sorted(factors, key=lambda factor: _factor_key(factor, order))


def _factor_key(factor: sp.Basic, order: Sequence[str]) -> tuple:
    """The sort key of one factor of a product: its kind, its variable, then a tiebreaker.

    Exponents of plain variables are ignored (`x^3*y^5*z^2`), while powers of sums sort
    by `_shape_key`. Constants keep sympy's order.
    """
    if factor.is_Number:
        return (_A_NUMBER, str(factor))
    kind, base, exponent, head = _kernel_parts(factor)
    if kind == _A_CONSTANT_KERNEL:
        return (_A_CONSTANT,)
    place, within = _factor_variable_key(base, order)
    if kind == _A_CALLED_KERNEL:
        return (_A_CALL, (place, within), _Reversed(_head_key(head)))
    if kind == _A_BRACKET_KERNEL:
        return (_A_POWER, (place, within), _shape_key(base, exponent, order))
    return (_A_VARIABLE, (place, within))


def _factor_variable_key(operand: sp.Basic, order: Sequence[str]) -> tuple:
    """A sort key for a factor's main variable: in a product, variables off the order list
    come first, alphabetically, as in the manual's `2*a*b^2*c*x^5*z^3`.
    """
    place, within = _variable_key(operand, order)
    return ((1, within), (0, within), (2, within))[place]


#: The order of calls of one variable in a sum (reversed in a product), determined
#: pairwise from the original; it isn't alphabetical. Only these eight are verified;
#: other heads sort after them alphabetically as a placeholder.
_HEAD_ORDER = ("#e^", "ABS", "ASIN", "ATAN", "LN", "COS", "TAN", "SIN")

#: The name `_kernel_parts` gives the exponential, which sympy holds as a power of
#: `#e` but the original treats as an ordinary call.
_EXPONENTIAL = "#e^"


#: The four kinds of kernel. Sums and products order them differently, so the names
#: imply no sort position.
(
    _A_CONSTANT_KERNEL,
    _A_NAMED_KERNEL,
    _A_BRACKET_KERNEL,
    _A_CALLED_KERNEL,
) = ("constant", "name", "bracket", "call")


def _kernel_parts(factor: sp.Basic) -> tuple[str, sp.Basic, sp.Basic, str]:
    """`factor` as its kind, base, exponent, and head name.

    The exponential counts as a call, not a power of `#e`, and a power of a number like
    `2^x` is a kernel of its exponent's variable, not a constant.
    """
    base, exponent = (
        (factor, sp.S.One) if isinstance(factor, sp.exp) else factor.as_base_exp()
    )
    if not base.free_symbols:
        base, exponent = factor, sp.S.One
    if not base.free_symbols:
        return _A_CONSTANT_KERNEL, base, exponent, ""
    if isinstance(base, sp.exp):
        return _A_CALLED_KERNEL, base, exponent, _EXPONENTIAL
    if isinstance(base, (sp.Add, sp.Mul)):
        return _A_BRACKET_KERNEL, base, exponent, ""
    if isinstance(base, (sp.Symbol, sp.MatrixSymbol)):
        return _A_NAMED_KERNEL, base, exponent, ""
    return _A_CALLED_KERNEL, base, exponent, _head_name(base)


def _head_name(base: sp.Basic) -> str:
    """The head name of a kernel that is neither a name nor a bracket: a call's own head,
    or else the sympy class name as a placeholder.
    """
    if isinstance(base, sp.Function):
        return function_name(base)
    return type(base).__name__.upper()


def _head_key(name: str) -> tuple:
    """The sort key of a function head among the heads of its variable."""
    if name in _HEAD_ORDER:
        return (0, _HEAD_ORDER.index(name), "")
    return (1, 0, name)


class _Reversed:
    """A key that sorts in reverse, since kernels of one variable sort in a sum in the
    reverse of their order in a product.
    """

    __slots__ = ("key",)

    def __init__(self, key: tuple) -> None:
        self.key = key

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Reversed) and self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    def __lt__(self, other: "_Reversed") -> bool:
        return other.key < self.key


def _shape_key(base: sp.Basic, exponent: sp.Basic, order: Sequence[str]) -> tuple:
    """The sort key of a power of a sum, ascending as in the original's factorizations: by
    degree of the base, then multiplicity, then coefficients from the leading one down,
    as in `(x + 3)*(x + 1)^2` and `(x + 2)*(x - 2)`.
    """
    degree, coefficients = _polynomial_key(base, order)
    return (degree, _size(exponent), coefficients)


def _polynomial_key(base: sp.Basic, order: Sequence[str]) -> tuple:
    """A compound base as its degree in its main variable, then its coefficients. Anything
    that isn't a polynomial shares one position, ahead of every polynomial.
    """
    names = main_order((symbol.name for symbol in base.free_symbols), order)
    variable = next(
        (symbol for symbol in base.free_symbols if symbol.name == names[0]), None
    )
    try:
        polynomial = sp.Poly(base, variable)
        coefficients = tuple(map(_coefficient_key, polynomial.all_coeffs()))
    except Exception:
        return ((0,), ())
    return ((1, polynomial.degree()), coefficients)


def _coefficient_key(value: sp.Basic) -> tuple:
    """The sort key of one coefficient: by magnitude, positive before negative, so `(x +
    2)*(x - 2)` and `x*(x + 2)`.
    """
    try:
        number, rest = value.as_coeff_Mul()
        return (_magnitude(number), 1 if number.is_negative else 0, str(rest))
    except Exception:
        return (0.0, 0, str(value))


def _magnitude(number: sp.Basic) -> Fraction | float:
    """The magnitude of a numeric coefficient of any kind."""
    if number.is_Rational:
        return abs(Fraction(int(number.p), int(number.q)))
    try:
        return abs(float(number))
    except (TypeError, ValueError):
        return 0.0


P = ParamSpec("P")
R = TypeVar("R")

#: The added built-ins (`EXTENDED_FUNCTIONS`) whose names the user has taken for a
#: function or variable of their own, during the command that is running.
_shadowed: ContextVar[frozenset[str]] = ContextVar("shadowed", default=frozenset())


def shadowed(context: Context) -> frozenset[str]:
    """The added built-ins that a user function or variable in `context` shadows.

    Derive has no such built-ins, so its utility files define functions of the same
    names, often differently: EXP_INT.MTH's `EI(x, m)` is a series of `m` terms. The
    user's definition then owns the name, on input and in every answer. An answer can't
    spell the built-in, since reading it back would call the user's function, so it is
    left standing as if the engine had no name for it: with EXP_INT.MTH loaded,
    `INT(#e^(-t)/t, t, 1, inf)` stays an integral rather than becoming `-EI(-1)`.
    Derive's own built-ins can't be redefined, so they are never shadowed.
    """
    taken = frozenset(context.functions) | frozenset(context.assignments)
    return EXTENDED_FUNCTIONS & taken


def shadowed_names() -> frozenset[str]:
    """The added built-ins shadowed during the running command; see `shadowed`."""
    return _shadowed.get()


@contextmanager
def shadowing(context: Context) -> Iterator[None]:
    """Spell no built-in that `context` shadows, while the block runs."""
    token = _shadowed.set(shadowed(context))
    try:
        yield
    finally:
        _shadowed.reset(token)


def shadowing_built_ins(command: Callable[P, R]) -> Callable[P, R]:
    """Decorate a command taking `(node, context, ...)` to run under `shadowing` of its
    context, so neither its answer nor anything it computes on the way spells a built-in
    the user has redefined."""

    @functools.wraps(command)
    def running(*args: P.args, **kwargs: P.kwargs) -> R:
        context = args[1] if len(args) > 1 else kwargs.get("context")
        if context is None:
            return command(*args, **kwargs)
        with shadowing(context):
            return command(*args, **kwargs)

    return running


def spelling(expression: sp.Basic) -> str | None:
    """The author-notation name of a function head, or None if it has none.

    Inert heads keep their authored name, and rederive heads use their upper-cased
    class name. Sympy heads use `_FUNCTION_NAMES`, or their upper-cased class name only
    if it is a built-in function, since the notation only has names an author can type
    back. A built-in the user has redefined has no name (see `shadowed`).
    """
    name = _built_in_spelling(expression)
    if isinstance(expression, AppliedUndef):
        return name
    return None if name in _shadowed.get() else name


def _built_in_spelling(expression: sp.Basic) -> str | None:
    """The spelling of a function head, whatever the user has redefined."""
    name = type(expression).__name__
    if _is_digamma(expression):
        return "DIGAMMA"
    if name in _FUNCTION_NAMES:
        return _FUNCTION_NAMES[name]
    if isinstance(expression, AppliedUndef):
        return name
    if type(expression).__module__.startswith("rederive."):
        return name.upper()
    return name.upper() if name.upper() in BUILTIN_FUNCTIONS else None


def function_name(expression: sp.Basic) -> str:
    """The name a function head sorts under: its spelling, or else its upper-cased class
    name, as for a head with its own print method such as `n!`."""
    return spelling(expression) or type(expression).__name__.upper()


def is_written(expression: sp.Basic) -> bool:
    """Whether the notation can express every function head in `expression`, i.e. each has
    a spelling or its own print method in `AuthorPrinter`. Calculus heads whose answer
    fails this are left unevaluated, as in Derive.
    """
    try:
        calls = expression.atoms(sp.Function)
    except Exception:
        return True
    return all(_has_word(call) for call in calls)


def _has_word(call: sp.Basic) -> bool:
    if spelling(call) is not None:
        return True
    if not isinstance(call, AppliedUndef) and _built_in_spelling(call) in _shadowed.get():
        return False
    return any(f"_print_{kind.__name__}" in vars(AuthorPrinter) for kind in _own(call))


def _own(call: sp.Basic) -> list[type]:
    """The classes of `call` up to (excluding) the generic function head, whose print
    method is the one that looks up a spelling."""
    kinds = []
    for kind in type(call).__mro__:
        if kind in (sp.Function, AppliedUndef):
            break
        kinds.append(kind)
    return kinds


def _is_suspended(expression: sp.Basic) -> bool:
    """Whether a call is the evaluator's suspended conditional, by class name, as for
    the other inert heads."""
    return any(kind.__name__ == "Suspended" for kind in type(expression).__mro__)


def _is_digamma(expression: sp.Basic) -> bool:
    """Whether a call is the digamma function, which sympy represents as polygamma of
    order zero but the notation names separately."""
    return isinstance(expression, sp.polygamma) and expression.args[0] == 0


def _degree(exponent: sp.Basic) -> tuple:
    """A sort key for a power, descending. Symbolic exponents share one position after
    every numeric one.
    """
    if exponent.is_Rational:
        return (0, -Fraction(int(exponent.p), int(exponent.q)))
    return (1, Fraction(0))


def _size(exponent: sp.Basic) -> tuple:
    """The same key as `_degree`, but ascending, as a product sorts."""
    place, degree = _degree(exponent)
    return (place, -degree)


def author_text(
    expression: sp.Basic,
    context: Context | None = None,
    authored: dict[sp.Basic, str] | None = None,
) -> str:
    """Write `expression` the way the author line would.

    `authored` maps subexpressions to text to print instead; `pipeline.simplify` fills
    it with author-written conditionals, whose arms are shown as written.
    """
    if context is not None and context.angle is Angle.DEGREE:
        expression = in_degrees(expression, context)
        authored = {
            in_degrees(key, context): text for key, text in (authored or {}).items()
        }
    return AuthorPrinter(context, authored).doprint(named(expression))


#: Heads that take an angle and heads that return one, mapped to their printed names.
#: `PHASE` returns one, being `ATAN(y, x)` of the parts.
_TAKING_ANGLE = {
    sp.sin: "SIN",
    sp.cos: "COS",
    sp.tan: "TAN",
    sp.cot: "COT",
    sp.sec: "SEC",
    sp.csc: "CSC",
}
_GIVING_ANGLE = {
    sp.asin: "ASIN",
    sp.acos: "ACOS",
    sp.atan: "ATAN",
    sp.acot: "ACOT",
    sp.asec: "ASEC",
    sp.acsc: "ACSC",
    sp.atan2: "ATAN",
    sp.arg: "PHASE",
}


def in_degrees(expression: sp.Basic, context: Context) -> sp.Basic:
    """`expression`, with angles in radians, rewritten to use degrees.

    Under `Angle := Degree` the sine of `pi*x/180` radians prints as `SIN(x)`. Each
    head is replaced by an inert one of the same name, and `pi/180` factors cancel. In
    an approximate answer the leftover `pi` coefficient is re-approximated so that
    `SIN(0.0174532*x)` prints as `SIN(x)`.
    """
    if not expression.has(*_TAKING_ANGLE, *_GIVING_ANGLE):
        return expression
    for head, name in _TAKING_ANGLE.items():
        inert = sp.Function(name)
        expression = expression.replace(
            head, lambda angle, inert=inert: inert(angle * 180 / sp.pi)
        )
    for head, name in _GIVING_ANGLE.items():
        inert = sp.Function(name)
        expression = expression.replace(
            head, lambda *args, inert=inert: inert(*args) * sp.pi / 180
        )
    expression = expression.replace(lambda e: e.is_Mul, _cancelling_pi)
    if context.precision is Precision.EXACT:
        return expression
    return expression.replace(
        lambda e: e.is_Mul, lambda product: _approximate_pi(product, context)
    )


def _approximate_pi(product: sp.Basic, context: Context) -> sp.Basic:
    """`product` with its numeric coefficient approximated, if it contains `pi`."""
    number, rest = product.as_independent(*product.free_symbols, as_Add=False)
    if not number.has(sp.pi) or number == product:
        return product
    return reapproximated(number, context) * rest


def _pi_power(term: sp.Basic) -> sp.Basic:
    return term.as_powers_dict().get(sp.pi, sp.Integer(0))


def _cancelling_pi(product: sp.Basic) -> sp.Basic:
    """`product` with a `pi` common to every term of a sum cancelled against a `pi`
    divisor, as in `(180*pi - pi*ASIN(a))/pi` from converting an inverse function.
    """
    below = -_pi_power(product)
    if not below.is_Integer or below <= 0:
        return product
    factors = list(product.args)
    for place, factor in enumerate(factors):
        if not factor.is_Add:
            continue
        powers = [_pi_power(term) for term in factor.args]
        if not all(power.is_Integer and power > 0 for power in powers):
            continue
        cancelled = min(below, *powers)
        factors[place] = sp.Add(*(term / sp.pi**cancelled for term in factor.args))
        return sp.Mul(*factors) * sp.pi**cancelled
    return product


def named(expression: sp.Basic) -> sp.Basic:
    """The expression with sympy's `Dummy` variables renamed to ordinary symbols.

    Two dummies can share a name, and a `Dummy` sorts by identity rather than name, so
    both must be renamed apart from all names in use for the text to read back the
    same: otherwise `SUM(SUM(f(k), k, 0, n), k, 0, n)` would read back with one
    variable. Matrix dimensions are excluded, since naming them would put a
    substitutable variable on the worksheet.
    """
    shapes = _shapes(expression)
    dummies = sorted(expression.atoms(sp.Dummy) - shapes, key=sp.default_sort_key)
    if not dummies:
        return expression
    taken = {
        symbol.name.lower()
        for symbol in expression.atoms(sp.Symbol)
        if not isinstance(symbol, sp.Dummy)
    }
    renamed = {}
    for dummy in dummies:
        name = _plain(dummy.name)
        while name in taken:
            name += "_"
        taken.add(name)
        renamed[dummy] = sp.Symbol(name, **dummy.assumptions0)
    if expression.has(sp.Subs):
        expression = expression.replace(sp.Subs, Substitution)
    try:
        return expression.xreplace(renamed)
    except Exception:
        return expression


class Substitution(sp.Function):
    """A stand-in for sympy's `Subs`, equal only to itself as written.

    A `Subs` equals itself with its bound variable renamed, so sympy's constructor
    cache would hand `named` back the old product, dummy and all. Its bound variables
    aren't free, so products order by what is substituted in.
    """

    @property
    def free_symbols(self) -> set[sp.Basic]:
        inner, variables, points = self.args
        return (inner.free_symbols - set(variables)) | points.free_symbols


def _shapes(expression: sp.Basic) -> set[sp.Dummy]:
    """The dummy variables used only for a symbolic matrix's dimensions."""
    if not expression.has(sp.MatrixSymbol):
        return set()
    return {
        symbol
        for matrix in expression.atoms(sp.MatrixSymbol)
        for dimension in matrix.shape
        if isinstance(dimension, sp.Basic)
        for symbol in dimension.atoms(sp.Dummy)
    }


#: One reading of a relation as a bound: the variable, the operator, and the bound.
_Bound = tuple[sp.Basic, str, sp.Basic]


def _bounds(relation: sp.Basic) -> list[_Bound]:
    """Every reading of a relation as a bound on one of its variables. `-3 < x` reads as
    `x > -3`; `x < y` bounds either, decided when paired with another relation.
    """
    if not isinstance(relation, Relational) or relation.rel_op not in _ORDERINGS:
        return []
    left, right = relation.lhs, relation.rhs
    readings = []
    if isinstance(left, sp.Symbol) and not right.has(left):
        readings.append((left, relation.rel_op, right))
    if isinstance(right, sp.Symbol) and not left.has(right):
        readings.append((right, relation.reversed.rel_op, left))
    return readings


def _bracket(
    first: list[_Bound], second: list[_Bound]
) -> tuple[sp.Basic, tuple[str, sp.Basic], tuple[str, sp.Basic]] | None:
    """The variable two relations bound from both sides, with each bound, or None."""
    for subject, one, value in first:
        for other, two, bound in second:
            if other != subject or (one in _FROM_BELOW) == (two in _FROM_BELOW):
                continue
            if one in _FROM_BELOW:
                return subject, (one, value), (two, bound)
            return subject, (two, bound), (one, value)
    return None


def _plain(name: str) -> str:
    """A dummy's name reduced to one letter, which reads back as one name in any mode.

    Character mode reads an undeclared `xi` as `x*i`, so a bound variable would turn
    into a product. A letter plus the underscores `named` adds is a single name.
    """
    letter = next((char for char in name if "a" <= char.lower() <= "z"), "v")
    return letter.lower()


def numeral(number: int, base: int = 10) -> str:
    """`number` written in `base`. Above base ten the digits run A to Z, and a numeral
    starting with a letter gets a leading zero: fourteen is `0E` in hexadecimal.
    """
    text = _digits_of(abs(number), base)
    if text[0].isalpha():
        text = "0" + text
    return ("-" if number < 0 else "") + text


#: The number of digits above which `_digits_of` splits a number in two.
_DIGIT_CHUNK = 64


def _digits_of(magnitude: int, base: int, width: int = 0) -> str:
    """The digits of a nonnegative `magnitude` in `base`, zero-padded to `width`. Long
    numbers are split in half recursively, since digit-by-digit division is quadratic.
    """
    if magnitude < base**_DIGIT_CHUNK:
        text = ""
        while magnitude:
            text = _DIGITS[magnitude % base] + text
            magnitude //= base
        return text.rjust(width, "0") if width else text or "0"
    half = int(magnitude.bit_length() / math.log2(base)) // 2
    high, low = divmod(magnitude, base**half)
    return _digits_of(high, base, max(width - half, 0)) + _digits_of(low, base, half)


def _under_the_bar(power: sp.Basic) -> bool:
    """Whether a factor raised to `power` is printed as a denominator: true for a negative
    numeric exponent (`1/k^2`), not for a symbolic one (`2^(-m)`).
    """
    return bool(power.is_Number and power.is_negative)


def _all_negated(factor: sp.Basic) -> bool:
    """Whether every term of a bracket has a minus sign, so the sign moves out to the
    product: `z*(-b - 2)` is written `- z*(b + 2)`.
    """
    return isinstance(factor, sp.Add) and all(
        term.could_extract_minus_sign() for term in factor.args
    )


def _unevaluated(expr: sp.Basic) -> bool:
    """Whether `expr` is a product sympy was told not to evaluate, by sympy's own test: a
    leading one, or a number or integer power of one after the front.
    """
    arguments = expr.args
    return bool(arguments) and (
        arguments[0] is sp.S.One
        or any(
            isinstance(argument, sp.Number)
            or (argument.is_Pow and all(part.is_Integer for part in argument.args))
            for argument in arguments[1:]
        )
    )


class AuthorPrinter(sp.StrPrinter):
    """A `StrPrinter` that writes author notation."""

    def __init__(
        self,
        context: Context | None = None,
        authored: dict[sp.Basic, str] | None = None,
    ) -> None:
        # `grlex` is descending total degree, the original's polynomial order:
        # `x^2 + c`, not `c + x^2`.
        super().__init__({"full_prec": False, "order": "grlex"})
        self.context = context or Context()
        self.authored = authored or {}
        self._unspelling = False

    def _print(self, expr, **kwargs):
        """Print one subexpression, using the author's text if it is recorded.

        The text replaces the whole subtree at the precedence the caller already
        applied, so undecided conditionals show exactly as written. Sets are
        intercepted here and sent to `_unspelled`, since the notation has no name for
        any of sympy's set classes.
        """
        if self.authored and isinstance(expr, (sp.Piecewise, AppliedUndef)):
            written = self.authored.get(expr)
            if written is not None:
                return written
        if isinstance(expr, sp.Set) and not self._unspelling:
            return self._unspelled(expr)
        return super()._print(expr, **kwargs)

    # -- sums ---------------------------------------------------------------

    def _as_ordered_terms(self, expr, order=None):
        """The terms of a sum in order-list order, with a two-term sum swapped to avoid a
        leading minus.

        `SQRT(3) - 1`, not `-1 + SQRT(3)`, but `-x^2 + 2*x - 1` keeps its order, as in
        the original. A complex number like `-2 + 2*#i` isn't swapped.
        """
        terms = _term_order(super()._as_ordered_terms(expr, order), self.context.order)
        if len(terms) == 2 and self._minus_first(terms[0]):
            if not self._minus_first(terms[1]) and not terms[1].has(sp.I):
                return [terms[1], terms[0]]
        return terms

    def _minus_first(self, term: sp.Basic) -> bool:
        """Whether `term` is printed with a leading minus. Quotients are checked by
        printing them, since the printer moves the numerator's sign in front of the
        bar.
        """
        if term.could_extract_minus_sign():
            return True
        if not isinstance(term, sp.Mul):
            return False
        if not all(isinstance(factor, sp.Expr) for factor in term.args):
            return False
        if sp.fraction(term)[1] == 1:
            return False
        return self._print(term).startswith("-")

    # -- names --------------------------------------------------------------

    def _print_Symbol(self, expr):
        """A variable, written in ASCII, with Greek letters spelled out (`alpha`) so the
        text can be typed on the author line.
        """
        return _SUBSCRIPT.join(
            GREEK_GLYPHS.get(part, part) for part in expr.name.split(_SUBSCRIPT)
        )

    def _print_Dummy(self, expr):
        """A sympy `Dummy`, written like any other name, since `_k1` doesn't lex. `named`
        normally renames these first.
        """
        return _plain(self._print_Symbol(expr))

    def _print_MatrixSymbol(self, expr):
        """A declared nonscalar, written like the variable it is."""
        return self._print_Symbol(expr)

    # -- numbers ------------------------------------------------------------

    def _print_Integer(self, expr):
        return self._number(Fraction(int(expr)))

    def _print_Rational(self, expr):
        return self._number(Fraction(expr.p, expr.q))

    def _print_Float(self, expr):
        """A float at the context's precision, in the selected notation.

        Rational notation prints plain digits. The grammar has no exponent notation, so
        large or small floats are written in full, and a whole float keeps its point.
        """
        text = super()._print_Float(sp.Float(expr, self.context.precision_digits))
        if "e" in text or "E" in text:
            text = format(Decimal(text), "f")
        if self.context.notation is not Notation.RATIONAL:
            return self._number(Fraction(Decimal(text)))
        return text if "." in text else text + "."

    def _print_Mul(self, expr):
        """A product, in order-list order.

        In a decimal style, a rational coefficient goes in front as a number (`x/3` is
        `0.333333·x`). An unevaluated product, like `Factor`'s `2*3^2*5*3607*3803`, is
        printed by sympy as is.
        """
        if _unevaluated(expr):
            return super()._print_Mul(expr)
        coefficient, rest = expr.as_coeff_Mul()
        if isinstance(coefficient, sp.Rational) and coefficient.q != 1:
            written = self._number(Fraction(coefficient.p, coefficient.q))
            if "/" not in written:
                return self._beside(written, rest)
        return self._divided(expr)

    def _divided(self, expr) -> str:
        """A product with its denominator determined as `_print_Pow` determines one.

        Sympy's product printer puts every negatively signed exponent below the bar,
        but `_print_Pow` does so only for numeric exponents (`1/k^2`, but `2^(-m)`), so
        this makes a power print the same everywhere. `#e^(-z)` is left alone.

        Two additions to sympy's sign extraction: a bracket of only negated terms gives
        its sign to the product (`- z*(b + 2)`), and a numerator that would lead with a
        minus gives its sign to the quotient (`-(x^2 - y^2 - 2)/2`).
        """
        sign = ""
        level = precedence(expr)
        if expr.could_extract_minus_sign():
            expr, sign = -expr, "-"
        above, below = [], []
        factors = _factor_order(
            _merged_roots(expr.as_ordered_factors()), self.context.order
        )
        for index, factor in enumerate(factors):
            if _all_negated(factor):
                factors[index] = -factor
                sign = "" if sign else "-"
                break
        for factor in factors:
            if isinstance(factor, sp.Rational):
                if factor.p != 1:
                    above.append(sp.Integer(factor.p))
                if factor.q != 1:
                    below.append(sp.Integer(factor.q))
            elif isinstance(factor, sp.Pow) and _under_the_bar(factor.exp):
                power = -factor.exp
                below.append(factor.base if power == 1 else _raised(factor.base, power))
            else:
                above.append(factor)
        if below:
            for index, factor in enumerate(above):
                if self._leads_negative(factor):
                    above[index] = -factor
                    sign = "" if sign else "-"
                    break

        def written(parts):
            return [self.parenthesize(part, level, strict=False) for part in parts]

        text = "*".join(written(above)) or "1"
        if len(below) == 1:
            text += "/" + written(below)[0]
        elif below:
            # One denominator, as sympy writes it: `a/(x*y)`, not `a/x/y`, which
            # would display as a nested fraction.
            text += "/(" + "*".join(written(below)) + ")"
        return sign + text

    def _leads_negative(self, factor: sp.Basic) -> bool:
        """Whether a bracket would print with a leading minus, after pair swapping."""
        if not isinstance(factor, sp.Add):
            return False
        terms = self._as_ordered_terms(factor)
        return bool(terms) and terms[0].could_extract_minus_sign()

    def _beside(self, written: str, rest) -> str:
        """An already printed coefficient times the rest of its product, which keeps its
        denominator, so `1/(x - 1)` gets the coefficient above the bar.
        """
        above, below = [], []
        factors = _merged_roots(rest.as_ordered_factors())
        for factor in _factor_order(factors, self.context.order):
            base, power = factor.as_base_exp()
            if power.is_Rational and power.is_negative:
                below.append(base if power == -1 else _raised(base, -power))
            else:
                above.append(factor)
        level = PRECEDENCE["Mul"]
        text = written
        for factor in above:
            text += "*" + self.parenthesize(factor, level, strict=False)
        if below:
            # One denominator, as sympy writes it: `a/(x*y)`, not `a/x/y`.
            under = [self.parenthesize(factor, level, strict=False) for factor in below]
            text += "/" + (under[0] if len(under) == 1 else "(" + "*".join(under) + ")")
        return text

    def _number(self, value: Fraction) -> str:
        """`value` in the notation `Options Notation` selects.

        Ratios use the session's base; decimal styles are always base ten. Decimal
        notation writes an exact number in full but an approximate one in scientific
        style, so `APPROX(100!)` is `9.33262*10^157` rather than 158 digits.
        """
        style, digits = self.context.notation, self.context.notation_digits
        if style is Notation.RATIONAL or (
            style is Notation.MIXED and notation.simple(value)
        ):
            base = self.context.input_base
            if value.denominator == 1:
                return numeral(value.numerator, base)
            return f"{numeral(value.numerator, base)}/{numeral(value.denominator, base)}"
        if style is Notation.DECIMAL and not is_approximation(value):
            return notation.decimal(value, digits)
        return notation.scientific(value, digits)

    def parenthesize(self, item, level, strict=False):
        """Parenthesize a number printed with a power of ten where it binds tighter than a
        product, as in an exponent: sympy treats numbers as atoms.
        """
        text = super().parenthesize(item, level, strict)
        if (
            isinstance(item, sp.Number)
            and level > PRECEDENCE["Mul"]
            and not text.startswith("(")
            and ("*" in text or "^" in text)
        ):
            return f"({text})"
        return text

    # -- constants and special values ---------------------------------------

    def _print_Exp1(self, expr):
        return "#e"

    def _print_ImaginaryUnit(self, expr):
        return "#i"

    def _print_EulerGamma(self, expr):
        return "euler_gamma"

    def _print_Infinity(self, expr):
        return "inf"

    def _print_NegativeInfinity(self, expr):
        return "-inf"

    def _print_ComplexInfinity(self, expr):
        """Unsigned infinity, the value of `1/0` and `TAN(pi/2)`."""
        return f"{_PLUS_MINUS}inf"

    def _print_NaN(self, expr):
        return "?"

    def _print_BooleanTrue(self, expr):
        return "true"

    def _print_BooleanFalse(self, expr):
        return "false"

    # -- operators ----------------------------------------------------------

    def _print_Pow(self, expr, rational=False):
        level = PRECEDENCE["Pow"]
        if expr.exp is sp.S.Half and not rational:
            return f"SQRT({self._print(expr.base)})"
        if expr.is_commutative and not rational:
            if -expr.exp is sp.S.Half:
                return f"1/SQRT({self._print(expr.base)})"
            if expr.exp is sp.S.NegativeOne:
                return f"1/{self.parenthesize(expr.base, level)}"
            if expr.exp.is_Number and expr.exp.is_negative:
                # A denominator, as the inherited `_print_Mul` writes one in a
                # product: `1/k^2` rather than `k^(-2)`.
                return f"1/{self._print(sp.Pow(expr.base, -expr.exp))}"
        base = self.parenthesize(expr.base, level)
        exponent = self.parenthesize(expr.exp, level)
        return f"{base}^{exponent}"

    def _print_MatPow(self, expr):
        """A power of a matrix, written like any other power.

        This avoids sympy's `m**(1/2)`. A negative exponent stays unparenthesized in
        the exponent, `a^-1`, as in the original, since there is no `1/u` for a matrix.
        """
        if expr.exp.is_Number and expr.exp.is_negative:
            base = self.parenthesize(expr.base, PRECEDENCE["Pow"])
            return f"{base}^{self._print(expr.exp)}"
        return self._print_Pow(expr)

    def _print_Inverse(self, expr):
        """`a^-1`, the same power sympy keeps under a separate head."""
        return self._print_MatPow(expr)

    def _print_MatMul(self, expr):
        """A matrix product, using the dot operator of section 8.4.

        Derive nests a chain to the right with parentheses, `a . (b . (c . d))`, since
        the grammar is left-associative. A scalar coefficient goes in front, and a
        negated product needs no parentheses, so a sum reads `a . b - b . a`.
        """
        coefficient, matrices = expr.as_coeff_matrices()
        if not matrices:
            return self._print(coefficient)
        product = self._nested(matrices)
        if coefficient == 1:
            return product
        if coefficient == -1:
            return f"-{product}"
        if len(matrices) > 1:
            product = f"({product})"
        return f"{self.parenthesize(coefficient, PRECEDENCE['Mul'])}*{product}"

    def _nested(self, matrices) -> str:
        """A sequence of matrices as the original's right-nested chain."""
        level = PRECEDENCE["Mul"]
        text = self.parenthesize(matrices[-1], level)
        for index in range(len(matrices) - 2, -1, -1):
            if index < len(matrices) - 2:
                text = f"({text})"
            text = f"{self.parenthesize(matrices[index], level)} . {text}"
        return text

    def _print_Transpose(self, expr):
        """``u` ``, the same operator the inert head uses."""
        return f"{self.parenthesize(expr.arg, PRECEDENCE['Func'])}`"

    def _print_exp(self, expr):
        """`#e^u`, never `EXP(u)`, as the notation spells it."""
        return f"#e^{self.parenthesize(expr.args[0], PRECEDENCE['Pow'])}"

    def _print_factorial(self, expr):
        return f"{self.parenthesize(expr.args[0], PRECEDENCE['Func'])}!"

    def _print_Relational(self, expr):
        level = PRECEDENCE["Relational"]
        operator = _RELATIONS.get(expr.rel_op, expr.rel_op)
        left = self.parenthesize(expr.lhs, level)
        right = self.parenthesize(expr.rhs, level)
        return f"{left} {operator} {right}"

    # -- functions ----------------------------------------------------------

    def _print_Function(self, expr):
        """`NAME(u, ...)`, or the call quoted under its sympy name if the notation has no
        name for it.
        """
        if _is_suspended(expr):
            return self._print_Suspended(expr)
        name = spelling(expr)
        if name is None:
            if not self._unspelling:
                return self._unspelled(expr)
            name = type(expr).__name__
        return f"{name}({self.stringify(expr.args, ', ')})"

    def _print_Suspended(self, expr):
        """A suspended conditional as its display, the conditional as written with its
        variables' values substituted. `released` normally replaces it before printing;
        its class name (`IF_2`) is internal and never printed. Sympy prints a subclass
        of an undefined function as a function, so `_print_Function` calls this.
        """
        kind = type(expr)
        shown = kind.display
        if expr.args != kind.variables:
            values = dict(zip(kind.variables, expr.args, strict=True))
            try:
                shown = shown.subs(values, simultaneous=True)
            except Exception:
                with sp.evaluate(False):
                    shown = shown.xreplace(values)
        return self._print(shown)

    def _print_polygamma(self, expr):
        """`DIGAMMA(x)` for the order-zero polygamma function, since sympy has no
        separate digamma function."""
        if _is_digamma(expr):
            return f"DIGAMMA({self._print(expr.args[1])})"
        return self._print_Function(expr)

    def _print_Gcd(self, expr):
        """`GCD(n, m)`, with arguments from least main up, as Derive writes them: `GCD(m,
        2)` is `GCD(2, m)`.
        """
        order = self.context.order
        operands = sorted(
            expr.args, key=lambda term: _term_key(term, order), reverse=True
        )
        return f"GCD({self.stringify(operands, ', ')})"

    def _print_Min(self, expr):
        return f"MIN({self.stringify(expr.args, ', ')})"

    def _print_Max(self, expr):
        return f"MAX({self.stringify(expr.args, ', ')})"

    def _print_floor(self, expr):
        """`FLOOR(m, n)` if the argument is a quotient, otherwise `FLOOR(u)`.

        The notation's two-argument floor (section 6.7: `MOD(m, n)` is `m - n*FLOOR(m,
        n)`) is used only if the denominator contains a variable, with a numeric factor
        moved to the numerator: `FLOOR(m/n + 1/2)` is `FLOOR(m + n/2, n)`.
        """
        numerator, denominator = sp.fraction(sp.together(expr.args[0]))
        if denominator.is_number:
            return f"FLOOR({self._print(expr.args[0])})"
        coefficient, denominator = denominator.as_coeff_Mul()
        over = self._print(denominator)
        return f"FLOOR({self._print(numerator / coefficient)}, {over})"

    def _print_Heaviside(self, expr):
        """`STEP(u)`, omitting the value at zero, which is ours, not the author's."""
        return f"STEP({self._print(expr.args[0])})"

    def _print_Determinant(self, expr):
        """`DET(u)` for a matrix sympy keeps unevaluated. It isn't a `Function`, so
        `spelling` doesn't cover it.
        """
        return f"DET({self._print(expr.arg)})"

    def _print_Trace(self, expr):
        return f"TRACE({self._print(expr.arg)})"

    def _print_Piecewise(self, expr):
        """A case split, as nested `IF(test, then, else)`. Without a final `true`
        condition the last case has no else, which is correct, since the value is then
        `?`.
        """
        pairs = list(expr.args)
        text = None
        if pairs and pairs[-1][1] is sp.true:
            text = self._print(pairs.pop()[0])
        for value, condition in reversed(pairs):
            parts = [self._print(condition), self._print(value)]
            if text is not None:
                parts.append(text)
            text = f"IF({', '.join(parts)})"
        return "?" if text is None else text


    # -- calculus heads that reached the output unevaluated -----------------

    def _print_Derivative(self, expr):
        primed = self._primed(expr, expr.variable_count[0][0])
        if primed is not None:
            return primed
        text = self._print(expr.expr)
        for variable, count in expr.variable_count:
            parts = [text, self._print(variable)]
            if count != 1:
                parts.append(self._print(count))
            text = f"DIF({', '.join(parts)})"
        return text

    def _primed(self, expr, point) -> str | None:
        """`BB'(r)`: `expr` written with prime notation, or None.

        The original uses primes only for an undefined function of one argument
        differentiated by it, and they make results like Ricci components readable.
        `point` is the argument, or the substituted point of a chain-rule `Subs`
        (`2*r*BB'(r^2)`).
        """
        if len(expr.variable_count) != 1:
            return None
        variable, count = expr.variable_count[0]
        call = expr.expr
        if not isinstance(call, AppliedUndef) or len(call.args) != 1:
            return None
        if call.args[0] != variable or not count.is_Integer or count < 1:
            return None
        marks = PRIME_MARK * int(count)
        return f"{function_name(call)}{marks}({self._print(point)})"

    def _print_Integral(self, expr):
        return self._limited("INT", expr)

    def _print_Sum(self, expr):
        return self._limited("SUM", expr)

    def _print_Product(self, expr):
        return self._limited("PRODUCT", expr)

    def _limited(self, head: str, expr) -> str:
        """`INT(u, x)` and `INT(u, x, a, b)`, one head per limit."""
        text = self._print(expr.function)
        for limit in expr.limits:
            parts = [text, *(self._print(part) for part in limit)]
            text = f"{head}({', '.join(parts)})"
        return text

    def _print_Limit(self, expr):
        """`LIM(u, x, a)`, with a fourth argument for a one-sided limit. The direction
        sympy records at infinity is omitted, since the author didn't ask for a
        one-sided limit.
        """
        expression, variable, point, direction = expr.args
        parts = [self._print(expression), self._print(variable), self._print(point)]
        side = {"+": "1", "-": "-1"}.get(str(direction))
        if side is not None and not point.is_infinite:
            parts.append(side)
        return f"LIM({', '.join(parts)})"

    # -- bounded limits that don't converge ------------------------------------

    def _print_AccumulationBounds(self, expr):
        """`INTERVAL(a, b)`: a value known only to lie between `a` and `b`.

        This is the result of a bounded limit that doesn't converge: `LIM(SIN(x), x,
        inf)` is `INTERVAL(-1, 1)`. The name is Mathematica's and Maple's, and it is an
        atom, so it works inside larger expressions.
        """
        return f"INTERVAL({self.stringify(expr.args, ', ')})"

    # -- roots of polynomials that don't factor --------------------------------

    def _print_RootSum(self, expr):
        """`ROOT_SUM(p, t, u)`: the sum of `u` over every root `t` of `p`. The summand, a
        `Lambda` in sympy, is written in the polynomial's generator.
        """
        variable = expr.poly.gen
        parts = (expr.poly.as_expr(), variable, expr.fun(variable))
        return f"ROOT_SUM({self.stringify(parts, ', ')})"

    def _print_ComplexRootOf(self, expr):
        """`ROOT_OF(p, t, n)`: the `n`-th root of `p` in `t`, counted from zero. The
        generator is taken from the polynomial.
        """
        parts = (expr.poly.as_expr(), expr.poly.gen, sp.Integer(expr.index))
        return f"ROOT_OF({self.stringify(parts, ', ')})"

    # -- vectors and matrices ------------------------------------------------

    def _print_Tuple(self, expr):
        """A tuple, written as a vector, since the grammar can't read a parenthesized
        list.
        """
        return self._vector(expr.args)

    def _print_Subs(self, expr):
        """`SUBS(u, [x], [a])`, upper-cased to be a fixed point, or a prime over an
        undefined function: `DIF(BB(r^2), r)` is `2*r*BB'(r^2)`.
        """
        inner, variables, points = expr.args
        if isinstance(inner, sp.Derivative) and len(variables) == len(points) == 1:
            primed = self._primed(inner, points[0])
            if primed is not None and inner.variable_count[0][0] == variables[0]:
                return primed
        return f"SUBS({self.stringify(expr.args, ', ')})"

    _print_Substitution = _print_Subs

    def _print_MatrixBase(self, expr):
        """A single row is written flat; taller matrices as a vector of rows."""
        if expr.rows == 1:
            return self._vector(expr)
        return self._vector([expr[row, :] for row in range(expr.rows)])

    def _vector(self, elements) -> str:
        return f"[{', '.join(self._print(element) for element in elements)}]"

    # -- logic ---------------------------------------------------------------

    def _print_And(self, expr):
        chained = self._chain(expr)
        return chained if chained is not None else self._infix("AND", expr.args)

    def _chain(self, expr) -> str | None:
        """Two bounds on one variable as the chain `-2 < x < 2`, as the original writes a
        range. The two sides' strictness is independent.
        """
        if len(expr.args) != 2:
            return None
        found = _bracket(*(_bounds(operand) for operand in expr.args))
        if found is None:
            return None
        subject, (lower, low), (upper, high) = found
        level = PRECEDENCE["Relational"]
        return " ".join(
            (
                self.parenthesize(low, level),
                _FROM_BELOW[lower],
                self.parenthesize(subject, level),
                upper,
                self.parenthesize(high, level),
            )
        )

    def _print_Or(self, expr):
        return self._infix("OR", expr.args)

    def _print_Xor(self, expr):
        return self._infix("XOR", expr.args)

    def _print_Implies(self, expr):
        # `IMP` isn't commutative, so its operands keep their order.
        return self._infix("IMP", expr.args, order=False)

    def _print_Not(self, expr):
        return f"NOT {self._logical_operand(expr.args[0], 'NOT')}"

    def _infix(self, word: str, operands, order: bool = True) -> str:
        if order:
            operands = _logical_order(operands, self.context.order)
        return f" {word} ".join(
            self._logical_operand(operand, word) for operand in operands
        )

    def _logical_operand(self, operand: sp.Basic, word: str) -> str:
        """The operand, parenthesized if it binds more loosely than `word`."""
        inner = _LOGICAL_HEADS.get(type(operand).__name__)
        if isinstance(operand, sp.Function) and type(operand).__name__ == "Logical":
            inner = str(operand.args[0])
        text = self._print(operand)
        if inner is None:
            return text
        if _LOGICAL_PRECEDENCE[inner] > _LOGICAL_PRECEDENCE[word]:
            return text
        return f"({text})"

    # -- the inert heads ------------------------------------------------------

    def _print_PlusMinus(self, expr):
        operand = self.parenthesize(expr.args[0], PRECEDENCE["Mul"], strict=True)
        return f"{_PLUS_MINUS}{operand}"

    def _print_StringLiteral(self, expr):
        return f'"{expr.name}"'

    def _unspelled(self, expr):
        """A head the notation can't express, written as a quoted string.

        Printed bare, sympy's `EmptySet` would read back as `e*m*p*t*y*s*e*t` and
        `Interval(0, 1)` as the built-in `INTERVAL`, both looking like computed
        answers. The quoted text is what would otherwise print; nothing inside is
        quoted again.
        """
        self._unspelling = True
        try:
            return f'"{super()._print(expr)}"'
        finally:
            self._unspelling = False

    def _print_Power(self, expr):
        base, exponent = expr.args
        level = PRECEDENCE["Pow"]
        # The exponent isn't parenthesized at its own level, since `^` is
        # right-associative: `10^10^10` reads back as the same tower.
        written = self.parenthesize(base, level, strict=True)
        return f"{written}^{self.parenthesize(exponent, level)}"

    def _print_Transposed(self, expr):
        return f"{self.parenthesize(expr.args[0], PRECEDENCE['Func'])}`"

    def _print_Subscript(self, expr):
        level = PRECEDENCE["Func"]
        base, index = expr.args
        return f"{self.parenthesize(base, level)} SUB {self.parenthesize(index, level)}"

    def _print_Added(self, expr):
        text = ""
        for term in expr.args:
            written = self.parenthesize(term, PRECEDENCE["Add"], strict=True)
            if not text:
                text = written
            elif written.startswith("-"):
                text += f" - {written[1:]}"
            else:
                text += f" + {written}"
        return text

    def _print_Quotient(self, expr):
        level = PRECEDENCE["Mul"]
        left, right = expr.args
        over = self.parenthesize(right, level, strict=True)
        return f"{self.parenthesize(left, level)}/{over}"

    def _print_Dot(self, expr):
        level = PRECEDENCE["Mul"]
        left, right = expr.args
        return f"{self.parenthesize(left, level)} . {self.parenthesize(right, level)}"

    def _print_Truth(self, expr):
        return self._print(expr.args[0])

    def _print_InertVector(self, expr):
        return self._vector(expr.args)

    def _print_Logical(self, expr):
        """A logical operator sympy rejected, with operands unsimplified in their written
        order: `3 AND p` is not `p AND 3`.
        """
        word = str(expr.args[0])
        operands = expr.args[1:]
        if len(operands) == 1:
            return f"{word} {self._logical_operand(operands[0], word)}"
        return self._infix(word, operands, order=False)

    def _print_Assign(self, expr):
        target, operator, *value = expr.args
        parts = [self._print(target), str(operator), *(self._print(v) for v in value)]
        return " ".join(parts)

    def _print_FunDef(self, expr):
        name, parameters, *body = expr.args
        head = f"{name}({', '.join(self._print(p) for p in parameters)})"
        return " ".join([head, ":=", *(self._print(part) for part in body)])

    def _print_Declare(self, expr):
        """`x :epsilon Real`, with the interval when one was declared."""
        target, kind, *interval = expr.args
        text = f"{self._print(target)} :epsilon {kind}"
        if len(interval) == 3:
            brackets, low, high = interval
            opening, closing = str(brackets)
            text += f" {opening}{self._print(low)}, {self._print(high)}{closing}"
        return text

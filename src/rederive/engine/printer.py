"""sympy -> author notation, as the tree of the author line that writes a value.

Every engine result becomes a tree here: `^` for powers, `SQRT(u)`, `#e` and `#i`,
upper-case function names, `?` for the unknown value. The tree must mean the same
expression, and must use the session's notation (input base and the `Options Notation`
style, handled by `engine.notation`). Decimal styles bend the first rule, as in the
original; the exact value is kept separately in `Result.value`. Its text is written from
the tree (`syntax.writer`).

`TreePrinter` is a sympy printer whose methods build tree nodes, one per construct. For
sums and products it decides only operand order and minus-sign placement, leaving sign
extraction and denominators to sympy, so sympy's conventions show through (`x*k^-2` is
written `x/k^2`) without changing meaning. A range is written as a chain `-2 < x < 2`,
and a matrix product nested to the right as the original does: `a . (b . (c . d))`.

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
from dataclasses import replace
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
from sympy.printing.printer import Printer

from rederive.engine import notation
from rederive.engine.approximation import is_approximation
from rederive.engine.context import Context, Notation
from rederive.engine.ordering import main_order
from rederive.model.expr import VARIADIC, Kind, Node
from rederive.syntax.lexer import numeral_value
from rederive.syntax.names import (
    BUILTIN_FUNCTIONS,
    ELLIPSIS,
    PRIME_MARK,
    YIELDING_FUNCTIONS,
)
from rederive.syntax.writer import write_expression

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
# A vector is bracketed already, and a set braced.
PRECEDENCE_VALUES.setdefault("Vector", PRECEDENCE["Atom"])
PRECEDENCE_VALUES.setdefault("SetOf", PRECEDENCE["Atom"])
# The set operators bind more loosely than a sum and more tightly than a relation.
PRECEDENCE_VALUES.setdefault("SetUnion", PRECEDENCE["Relational"] + 1)
PRECEDENCE_VALUES.setdefault("SetIntersection", PRECEDENCE["Relational"] + 2)
PRECEDENCE_VALUES.setdefault("SetDifference", PRECEDENCE["Relational"] + 3)
# `RETURN(u)` reads what follows into its operand, so it is parenthesized as an operand.
PRECEDENCE_VALUES.setdefault("RETURN", PRECEDENCE["Lambda"])
# A truth value used as an arithmetic operand binds as tightly as its content.
PRECEDENCE_FUNCTIONS.setdefault("Truth", lambda expr: precedence(expr.args[0]))
# A logical operator sympy declined (`Logical`, such as `3 AND p`) binds as its word
# does, every one of them more loosely than a relation: `(5 AND d) = 0` keeps its
# parentheses.
_LOGICAL_LEVELS = {
    "NOT": PRECEDENCE["And"] + 1,
    "AND": PRECEDENCE["And"],
    "OR": PRECEDENCE["Or"],
    "XOR": PRECEDENCE["Xor"],
    "IMP": PRECEDENCE["Xor"],
    "IFF": PRECEDENCE["Xor"],
}
PRECEDENCE_FUNCTIONS.setdefault(
    "Logical", lambda expr: _LOGICAL_LEVELS.get(str(expr.args[0]), PRECEDENCE["Xor"])
)

_DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

#: The separator between the parts of a subscripted symbol's name.
_SUBSCRIPT = " SUB "

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
    "AnyMovingLimit": "LIM",
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
    "Modulo": "MOD",
    "SymmetricModulo": "MODS",
}


def logical_order(operands, order: Sequence[str]) -> list[sp.Basic]:
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


def term_order(terms: Sequence[sp.Basic], order: Sequence[str]) -> list[sp.Basic]:
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
    1/2` (6.3 p.104). Non-real terms go last, by string, leaving out the numeric
    coefficient, so a sum of truth values is `false + 2*true`.
    """
    try:
        return (_CONSTANT, 0, -float(term), "")
    except (TypeError, ValueError):
        return (_CONSTANT, 1, 0.0, str(term.as_coeff_Mul()[1]), str(term))


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


def factor_order(factors: Sequence[sp.Basic], order: Sequence[str]) -> list[sp.Basic]:
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

#: The built-ins that yield to a definition (`YIELDING_FUNCTIONS`) whose names the user
#: has taken for a function or variable of their own, during the command that is running.
_shadowed: ContextVar[frozenset[str]] = ContextVar("shadowed", default=frozenset())


def shadowed(context: Context) -> frozenset[str]:
    """The built-ins that a user function or variable in `context` shadows.

    Old utility files define functions that later became built-ins, such as `ZETA` and
    `FLOOR`, and those names yield to a definition. The user's definition then owns the
    name, on input and in every answer. An answer can't spell the built-in, since
    reading it back would call the user's function, so it is left standing as if the
    engine had no name for it. The other built-ins can't be redefined, so they are
    never shadowed.
    """
    taken = frozenset(context.functions) | frozenset(context.assignments)
    return YIELDING_FUNCTIONS & taken


def shadowed_names() -> frozenset[str]:
    """The built-ins shadowed during the running command; see `shadowed`."""
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


class WrittenBase(AppliedUndef):
    """An expression kept as the user wrote it, which nothing evaluates: the body of a
    definition that holds a program (see `_Evaluator._fundef`). Each is its own class
    with the tree in `node`, applied to nothing, which is its tree in an answer."""

    node: Node


class HeldBase(AppliedUndef):
    """A held value, the value of a quote `'u`: the expression `u` as written, with the
    values of the locals in scope filled in, and nothing evaluated. Each is its own
    class with the tree in `node`, applied to nothing, so that sympy builds sums and
    products of it without looking inside: it has no free symbols and is no number.
    Equal trees are one class (see `to_sympy.held`), so two equal held values are the
    same term. Its tree in an answer is the held tree; `precedence` is how tightly
    that binds, on sympy's scale.
    """

    node: Node
    precedence: int


def has_held(expression: object) -> bool:
    """Whether `expression` has a held value in it."""
    try:
        return isinstance(expression, sp.Basic) and bool(expression.atoms(HeldBase))
    except Exception:
        return False


def is_written(expression: sp.Basic) -> bool:
    """Whether the notation can express every function head in `expression`, i.e. each has
    a spelling or its own print method in `TreePrinter`. Calculus heads whose answer
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
    return any(f"_print_{kind.__name__}" in vars(TreePrinter) for kind in _own(call))


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
    """A sort key for a power, descending. A real constant exponent sorts by its value
    (`x^(2*#e)` before `x^pi` before `x^SQRT(5)`); symbolic exponents share one position
    after every numeric one.
    """
    if exponent.is_Rational:
        return (0, -Fraction(int(exponent.p), int(exponent.q)))
    if exponent.is_number and exponent.is_extended_real:
        value = sp.N(exponent, 30)
        if value.is_Float:
            return (0, -Fraction(str(value)))
    return (1, Fraction(0))


def _size(exponent: sp.Basic) -> tuple:
    """The same key as `_degree`, but ascending, as a product sorts."""
    place, degree = _degree(exponent)
    return (place, -degree)


def named(expression: sp.Basic) -> sp.Basic:
    """The expression with sympy's `Dummy` variables renamed to ordinary symbols.

    Two dummies can share a name, and a `Dummy` sorts by identity rather than name, so
    both must be renamed apart from all names in use for the answer to mean the same
    when it is computed with again: otherwise `SUM(SUM(f(k), k, 0, n), k, 0, n)` would
    come back with one variable. Matrix dimensions are excluded, since naming them would
    put a substitutable variable on the worksheet.
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


def _contents_in_front(
    above: list[sp.Basic], below: list[sp.Basic]
) -> tuple[list[sp.Basic], list[sp.Basic]]:
    """A quotient's factors, with the integer content of a lone sum above or below a
    denominator with a variable written in front, as the normal form writes it.

    The reader multiplies a number into a sum it stands alone beside, so `2*(x + 1)/y`
    reads back as `(2*x + 2)/y`, and `1/(2*(x + 1))` as `1/(2*x + 2)`. Writing such a
    sum with its content in front again keeps the text a fixed point. The numbers
    above and below are combined into one ratio.
    """
    if all(factor.is_Number for factor in below):
        return above, below
    tops = [factor for factor in above if not factor.is_Number]
    bottoms = [factor for factor in below if not factor.is_Number]
    number = sp.Mul(*(f for f in above if f.is_Number)) / sp.Mul(
        *(f for f in below if f.is_Number)
    )
    taken = False
    for factors, power in ((tops, 1), (bottoms, -1)):
        if len(factors) == 1 and factors[0].is_Add:
            content, primitive = factors[0].as_content_primitive()
            if content.is_Integer and content > 1 and _integral(primitive):
                number *= content**power
                factors[0] = primitive
                taken = True
    if not taken:
        return above, below
    if number.p != 1:
        tops.insert(0, sp.Integer(number.p))
    if number.q != 1:
        bottoms.insert(0, sp.Integer(number.q))
    return tops, bottoms


def _integral(total: sp.Add) -> bool:
    return all(term.as_coeff_Mul()[0].is_Integer for term in total.args)


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


def _node(
    kind: Kind,
    children: Sequence[Node] = (),
    value: str | None = None,
    surface: str | None = None,
) -> Node:
    """A node of an answer's tree. Its span is given when the tree is written (see
    `writer.written`)."""
    return Node(kind, 0, 0, tuple(children), value, surface)


def _name(name: str) -> Node:
    return _node(Kind.NAME, value=name)


def _call(name: str, arguments: Sequence[Node]) -> Node:
    return _node(Kind.CALL, (_name(name), *arguments))


def _binop(operator: str, left: Node, right: Node) -> Node:
    return _node(Kind.BINOP, (left, right), operator)


def _signed(sign: str, operand: Node) -> Node:
    return _node(Kind.UNOP, (operand,), sign)


def _relation(operator: str, left: Node, right: Node) -> Node:
    return _node(Kind.REL, (left, right), operator)


def _numeral(written: str, base: int = 10) -> Node:
    """The numeral `written` in `base`, with a leading minus as a sign."""
    if written.startswith("-"):
        return _signed("-", _numeral(written[1:], base))
    value = numeral_value(written, base)
    return _node(Kind.NUMBER, value=value, surface=None if written == value else written)


def _verbatim(text: str) -> Node:
    """A leaf written as `text`, which only the text of an unspelled head holds (see
    `TreePrinter._unspelled`)."""
    return _node(Kind.NAME, value=text)


#: The operators that join the factors of a product run: `*`, and `/` and the dot
#: product, which take the run before them as their left operand.
_JOINING = ("*", "/", ".")


def _run(pieces: Sequence[tuple[str, Node, bool]]) -> Node:
    """The product run of `pieces`, each an operator joining it to what is before (`""`
    for the first), a node, and whether the node stands in parentheses.

    A piece that doesn't is joined into the run as the notation reads it: the factors of
    a product become factors of the run, `/` and the dot product take the run before
    them as their left operand, and a sign in front applies to the whole run. So `-0.25`
    times `x` is `-(0.25*x)`, and `x` times `a/b` is `(x*a)/b`.
    """
    sign = None
    joined: list[tuple[str, Node]] = []
    for index, (operator, node, fenced) in enumerate(pieces):
        if not fenced and not index and node.kind is Kind.UNOP:
            sign, node = str(node.value), node.children[0]
        stream = [(operator, node)] if fenced else _stream(node, operator)
        joined.extend(stream)
    run: list[Node] = []
    for operator, node in joined:
        if operator in ("/", "."):
            run = [_binop(operator, _product(run), node)]
        else:
            run.append(node)
    product = _product(run)
    return product if sign is None else _signed(sign, product)


def _stream(node: Node, operator: str) -> list[tuple[str, Node]]:
    """`node` as the factors and operators of a product run, the first joined by
    `operator`."""
    if node.kind is Kind.PRODUCT:
        first, *rest = node.children
        return [*_stream(first, operator), *(("*", child) for child in rest)]
    if node.kind is Kind.BINOP and node.value in ("/", "."):
        left, right = node.children
        return [*_stream(left, operator), (str(node.value), right)]
    return [(operator, node)]


def _product(factors: Sequence[Node]) -> Node:
    if len(factors) == 1:
        return factors[0]
    return _node(Kind.PRODUCT, factors, "*" * (len(factors) - 1))


def _sum_of(terms: Sequence[tuple[Node, bool]]) -> Node:
    """The sum of `terms`, each a node and whether it stands in parentheses.

    A later term written with a minus sign is subtracted, its sign the run's, and a
    term that is a sum and stands bare joins the run, as the notation reads it.
    """
    children: list[Node] = []
    signs = ""
    for node, fenced in terms:
        if children:
            unsigned = None if fenced else _without_minus(node)
            signs += "+" if unsigned is None else "-"
            node = node if unsigned is None else unsigned
        if node.kind is Kind.SUM and not fenced:
            if children:
                signs += str(node.value)
            else:
                signs = str(node.value)
            children.extend(node.children)
        else:
            children.append(node)
    if len(children) == 1:
        return children[0]
    return _node(Kind.SUM, children, signs)


def _leads_with_minus(node: Node) -> bool:
    """Whether `node` is written starting with a minus sign."""
    return _without_minus(node) is not None


def _without_minus(node: Node) -> Node | None:
    """`node` with the minus sign it is written starting with taken off, or None if it
    starts with none. The sign is the node's own, or that of its first operand where the
    operator lets an operand stand signed in first place, as a sum and a vector product
    do."""
    if node.kind is Kind.UNOP:
        return node.children[0] if node.value == "-" else None
    if node.kind is Kind.SUM or (
        node.kind is Kind.INFIX and str(node.value) in ("DOTPRODUCT", "CROSSPRODUCT")
    ):
        first = _without_minus(node.children[0])
        if first is not None:
            return replace(node, children=(first, *node.children[1:]))
    return None


def tree(
    expression: sp.Basic,
    context: Context | None = None,
    authored: dict[sp.Basic, Node] | None = None,
) -> Node:
    """`expression` as the tree of the author line that writes it.

    `authored` maps subexpressions to trees to stand for them; `pipeline.simplify` fills
    it with author-written conditionals, whose arms are shown as written.
    """
    return TreePrinter(context, authored).doprint(named(expression))


def author_text(
    expression: sp.Basic,
    context: Context | None = None,
    authored: dict[sp.Basic, Node] | None = None,
) -> str:
    """Write `expression` the way the author line would: its `tree`, written."""
    return write_expression(tree(expression, context, authored))


class TreePrinter(Printer):
    """Builds the tree of the author notation for a sympy expression.

    Each `_print_<Class>` method returns the tree of its expression. Sums and products
    take their operand order from the order list and their sign placement from
    `_divided`, leaving sign extraction and denominators to sympy.

    Some layout decisions depend on how the operands they arrange are written: whether
    one starts with a minus sign (`_leads_with_minus`), and whether sympy's precedence
    sets it apart (`_fenced`), since a part that isn't joins the run around it, as `x`
    times `a/b` is `x*a/b` (see `_run`).
    """

    #: No sympy class has a method of this name, so every one is built here.
    printmethod = "_tree"
    # `grlex` is descending total degree, the original's polynomial order: `x^2 + c`,
    # not `c + x^2`.
    _default_settings = {"order": "grlex"}

    def __init__(
        self,
        context: Context | None = None,
        authored: dict[sp.Basic, Node] | None = None,
    ) -> None:
        super().__init__()
        self.context = context or Context()
        self.authored = authored or {}
        self._unspelling = False

    def doprint(self, expr) -> Node:
        return self._print(expr)

    def _print(self, expr, **kwargs) -> Node:
        """The tree of one subexpression, the author's own if it is recorded.

        Sets are intercepted here and sent to `_unspelled`, since the notation has no
        name for any of sympy's set classes.
        """
        if self.authored and isinstance(expr, (sp.Piecewise, AppliedUndef)):
            written = self.authored.get(expr)
            if written is not None:
                return written
        if isinstance(expr, sp.Set):
            return self._unspelled(expr)
        return super()._print(expr, **kwargs)

    def emptyPrinter(self, expr) -> Node:
        """An expression the notation has no form for, as for a head without a name."""
        return self._unspelled(expr)

    def _print_Basic(self, expr) -> Node:
        return self._unspelled(expr)

    def _print_str(self, expr) -> Node:
        return _name(expr)

    def _arguments(self, arguments) -> list[Node]:
        return [self._print(argument) for argument in arguments]

    def _fenced(self, item, level: int, strict: bool = False) -> tuple[Node, bool]:
        """`item`'s tree, and whether sympy's precedence sets it apart in a place that
        binds at `level`, as parentheses would: one that binds more loosely, or as
        loosely unless `strict`. A part that isn't is read as part of the run around
        it. A number written with a power of ten is set apart where it binds tighter
        than a product, since sympy treats numbers as atoms."""
        node = self._print(item)
        own = precedence(item)
        fenced = own < level or (not strict and own <= level)
        if (
            not fenced
            and isinstance(item, sp.Number)
            and level > PRECEDENCE["Mul"]
            and node.kind in (Kind.PRODUCT, Kind.BINOP)
        ):
            fenced = True
        return node, fenced

    # -- sums ---------------------------------------------------------------

    def _as_ordered_terms(self, expr, order=None):
        """The terms of a sum in order-list order, with a two-term sum swapped to avoid a
        leading minus.

        `SQRT(3) - 1`, not `-1 + SQRT(3)`, but `-x^2 + 2*x - 1` keeps its order, as in
        the original. A complex number like `-2 + 2*#i` isn't swapped.
        """
        terms = term_order(super()._as_ordered_terms(expr, order), self.context.order)
        if len(terms) == 2 and self._minus_first(terms[0]):
            if not self._minus_first(terms[1]) and not terms[1].has(sp.I):
                return [terms[1], terms[0]]
        return terms

    def _minus_first(self, term: sp.Basic) -> bool:
        """Whether `term` is written with a leading minus. Quotients are checked by
        building them, since the numerator's sign moves in front of the bar."""
        if term.could_extract_minus_sign():
            return True
        if not isinstance(term, sp.Mul):
            return False
        if not all(isinstance(factor, sp.Expr) for factor in term.args):
            return False
        if sp.fraction(term)[1] == 1:
            return False
        return _leads_with_minus(self._print(term))

    def _print_Add(self, expr) -> Node:
        """A sum, its terms in order, each later one's minus sign its sign in the run."""
        level = precedence(expr)
        terms = [
            (*self._fenced(term, level, strict=True), term.is_Add)
            for term in self._as_ordered_terms(expr)
        ]
        return _sum_of([(node, fenced or nested) for node, fenced, nested in terms])

    # -- names --------------------------------------------------------------

    def _print_Symbol(self, expr) -> Node:
        """A variable, its subscripts, if it is named with any, as links: `x SUB 1`."""
        return self._subscripted(expr.name)

    def _subscripted(self, name: str) -> Node:
        base, *indices = _subscript_parts(name)
        if base.startswith("#") and base[1:].isdigit():
            # A label standing for the expression it numbers, `#3`.
            node = _node(Kind.LABEL, value=base[1:])
        else:
            node = _name(base)
        for index in indices:
            if index.startswith("("):
                part = self._subscripted(index[1:-1])
            elif index.isdigit():
                part = _numeral(numeral(int(index), self.context.input_base))
                part = replace(part, value=index) if part.kind is Kind.NUMBER else part
            else:
                part = _name(index)
            node = _node(Kind.SUB, (node, part))
        return node

    def _print_Dummy(self, expr) -> Node:
        """A sympy `Dummy`, written like any other name. `named` normally renames these
        first."""
        return _name(_plain(expr.name))

    def _print_MatrixSymbol(self, expr) -> Node:
        """A declared nonscalar, written like the variable it is."""
        return self._subscripted(expr.name)

    # -- numbers ------------------------------------------------------------

    def _print_Integer(self, expr) -> Node:
        return self._number(Fraction(int(expr)))

    def _print_Rational(self, expr) -> Node:
        return self._number(Fraction(expr.p, expr.q))

    def _print_Float(self, expr) -> Node:
        """A float at the context's precision, in the selected notation.

        Rational notation writes plain digits. The grammar has no exponent notation, so
        large or small floats are written in full, and a whole float keeps its point.
        """
        text = sp.StrPrinter({"full_prec": False})._print_Float(
            sp.Float(expr, self.context.precision_digits)
        )
        if "e" in text or "E" in text:
            text = format(Decimal(text), "f")
        if self.context.notation is not Notation.RATIONAL:
            return self._number(Fraction(Decimal(text)))
        return _numeral(text if "." in text else text + ".")

    def _print_Mul(self, expr) -> Node:
        """A product, in order-list order.

        In a decimal style, a rational coefficient goes in front as a number (`x/4` is
        `0.25·x`). Under Decimal, an exact coefficient that the decimal would cut short
        keeps its ratio where the ratio is no longer than the decimal (`x/3`, but
        `0.948699·x` for 10541/11111 at six digits), so a short exact factor is shown as
        computed. An unevaluated product, like `Factor`'s `2*3^2*5*3607*3803`, is
        written as it is.
        """
        if _unevaluated(expr):
            return self._as_written(expr)
        coefficient, rest = expr.as_coeff_Mul()
        if getattr(rest, "degree_unit", False) and coefficient.is_Rational:
            # An angle in degrees, written with the postfix operator: `30 DEG`.
            number = self._print(coefficient)
            if coefficient.q == 1 and number.kind is Kind.UNOP:
                inner = _node(Kind.POSTOP, number.children, "DEG")
                return _signed(str(number.value), inner)
            return _node(Kind.POSTOP, (number,), "DEG")
        if isinstance(coefficient, sp.Rational) and coefficient.q != 1:
            value = Fraction(coefficient.p, coefficient.q)
            if not self._is_ratio(value) and not self._keeps_ratio(value):
                return self._beside(self._number(value), rest)
        return self._divided(expr)

    def _as_written(self, expr) -> Node:
        """An unevaluated product, its factors in the order given, as sympy writes one:
        negative powers below the bar."""
        level = precedence(expr)
        above, below = [], []
        for factor in expr.args:
            if isinstance(factor, sp.Pow) and bool(factor.exp.as_coeff_Mul()[0] < 0):
                if factor.exp.is_Number:
                    power = -factor.exp
                else:
                    parts = list(factor.exp.args)
                    parts[0] = -parts[0]
                    power = sp.Mul._from_args(parts)
                below.append(
                    sp.Pow(factor.base, power, evaluate=False)
                    if power - 1
                    else factor.base
                )
            else:
                above.append(factor)
        pieces = []
        if above and not above[0].is_Add and above[0].could_extract_minus_sign():
            pieces.append(("", self._print(above.pop(0)), False))
        for factor in above:
            node, fenced = self._fenced(factor, level)
            pieces.append(("*" if pieces else "", node, fenced))
        if not pieces:
            pieces.append(("", _numeral("1"), False))
        if not below:
            return _run(pieces)
        under = []
        if len(below) > 1 and below[0].could_extract_minus_sign():
            under.append(("", self._print(below.pop(0)), False))
        for factor in below:
            node, fenced = self._fenced(factor, level)
            under.append(("*" if under else "", node, fenced))
        if len(under) > 1:
            return _run([*pieces, ("/", _run(under), True)])
        return _run([*pieces, ("/", under[0][1], under[0][2])])

    def _is_ratio(self, value: Fraction) -> bool:
        """Whether the notation writes `value`, not whole, as a ratio."""
        style = self.context.notation
        return style is Notation.RATIONAL or (
            style is Notation.MIXED and notation.simple(value)
        )

    def _keeps_ratio(self, value: Fraction) -> bool:
        """Whether an exact coefficient written as a cut-short decimal is better written
        as its ratio, which is not longer. Only Decimal writes whole numbers in full, as
        a ratio needs.
        """
        if self.context.notation is not Notation.DECIMAL or is_approximation(value):
            return False
        written = notation.decimal(value, self.context.notation_digits)
        if notation.value_of(written) == value:
            return False
        ratio = len(str(value.numerator)) + 1 + len(str(value.denominator))
        return ratio <= len(written)

    def _divided(self, expr) -> Node:
        """A product with its denominator determined as `_print_Pow` determines one.

        Sympy's product printer puts every negatively signed exponent below the bar,
        but `_print_Pow` does so only for numeric exponents (`1/k^2`, but `2^(-m)`), so
        this makes a power print the same everywhere. `#e^(-z)` is left alone.

        Two additions to sympy's sign extraction: a bracket of only negated terms gives
        its sign to the product (`- z*(b + 2)`), and a numerator that would lead with a
        minus gives its sign to the quotient (`-(x^2 - y^2 - 2)/2`).
        """
        negative = False
        level = precedence(expr)
        if expr.could_extract_minus_sign():
            expr, negative = -expr, True
        above, below = [], []
        factors = factor_order(
            _merged_roots(expr.as_ordered_factors()), self.context.order
        )
        for index, factor in enumerate(factors):
            if _all_negated(factor):
                factors[index] = -factor
                negative = not negative
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
        above, below = _contents_in_front(above, below)
        if below:
            for index, factor in enumerate(above):
                if self._leads_negative(factor):
                    above[index] = -factor
                    negative = not negative
                    break
        pieces = [
            ("*" if index else "", *self._fenced(part, level))
            for index, part in enumerate(above)
        ] or [("", _numeral("1"), False)]
        if len(below) == 1:
            pieces.append(("/", *self._fenced(below[0], level)))
        elif below:
            # One denominator, as sympy writes it: `a/(x*y)`, not `a/x/y`, which
            # would display as a nested fraction.
            under = [
                ("*" if index else "", *self._fenced(part, level))
                for index, part in enumerate(below)
            ]
            pieces.append(("/", _run(under), True))
        if negative:
            first, *rest = pieces
            pieces = [("", _signed("-", _run([first])), False), *rest]
        return _run(pieces)

    def _leads_negative(self, factor: sp.Basic) -> bool:
        """Whether a bracket would print with a leading minus, after pair swapping."""
        if not isinstance(factor, sp.Add):
            return False
        terms = self._as_ordered_terms(factor)
        return bool(terms) and terms[0].could_extract_minus_sign()

    def _beside(self, written: Node, rest) -> Node:
        """A coefficient times the rest of its product, which keeps its denominator, so
        `1/(x - 1)` gets the coefficient above the bar.
        """
        above, below = [], []
        factors = _merged_roots(rest.as_ordered_factors())
        for factor in factor_order(factors, self.context.order):
            base, power = factor.as_base_exp()
            if power.is_Rational and power.is_negative:
                below.append(base if power == -1 else _raised(base, -power))
            else:
                above.append(factor)
        level = PRECEDENCE["Mul"]
        pieces = [("", written, False)]
        pieces += [("*", *self._fenced(factor, level)) for factor in above]
        if len(below) == 1:
            pieces.append(("/", *self._fenced(below[0], level)))
        elif below:
            # One denominator, as sympy writes it: `a/(x*y)`, not `a/x/y`.
            under = [
                ("*" if index else "", *self._fenced(factor, level))
                for index, factor in enumerate(below)
            ]
            pieces.append(("/", _run(under), True))
        return _run(pieces)

    def _number(self, value: Fraction) -> Node:
        """`value` in the notation `Options Notation` selects.

        Ratios use the session's base; decimal styles are always base ten. Decimal
        notation writes an exact number in full but an approximate one in scientific
        style, so `APPROX(100!)` is `9.33262*10^157` rather than 158 digits.
        """
        digits = self.context.notation_digits
        if self._is_ratio(value):
            base = self.context.input_base
            if value.denominator == 1:
                return _numeral(numeral(value.numerator, base), base)
            ratio = _binop(
                "/",
                _numeral(numeral(abs(value.numerator), base), base),
                _numeral(numeral(value.denominator, base), base),
            )
            return _signed("-", ratio) if value < 0 else ratio
        style = self.context.notation
        if style is Notation.DECIMAL and not is_approximation(value):
            return _numeral(notation.decimal(value, digits))
        mantissa, power = notation.scientific(value, digits)
        if power is None:
            return _numeral(mantissa)
        negative = mantissa.startswith("-")
        mantissa = mantissa.removeprefix("-")
        ten = _binop("^", _numeral("10"), _numeral(str(power)))
        number = _product([_numeral(mantissa), ten]) if mantissa else ten
        return _signed("-", number) if negative else number

    # -- constants and special values ---------------------------------------

    def _print_Exp1(self, expr) -> Node:
        return _name("#e")

    def _print_ImaginaryUnit(self, expr) -> Node:
        return _name("#i")

    def _print_Pi(self, expr) -> Node:
        return _name("pi")

    def _print_ZeroMatrix(self, expr) -> Node:
        return _numeral("0")

    def _print_Degree(self, expr) -> Node:
        return _name("deg")

    def _print_EulerGamma(self, expr) -> Node:
        return _name("euler_gamma")

    def _print_Infinity(self, expr) -> Node:
        return _name("inf")

    def _print_NegativeInfinity(self, expr) -> Node:
        return _signed("-", _name("inf"))

    def _print_ComplexInfinity(self, expr) -> Node:
        """Unsigned infinity, the value of `1/0` and `TAN(pi/2)`."""
        return _signed("+-", _name("inf"))

    def _print_NaN(self, expr) -> Node:
        return _node(Kind.UNKNOWN, value="?")

    def _print_BooleanTrue(self, expr) -> Node:
        return _name("true")

    def _print_BooleanFalse(self, expr) -> Node:
        return _name("false")

    # -- operators ----------------------------------------------------------

    def _print_Pow(self, expr, rational=False) -> Node:
        level = PRECEDENCE["Pow"]
        if expr.exp is sp.S.Half and not rational:
            return _call("SQRT", [self._print(expr.base)])
        if expr.is_commutative and not rational:
            if -expr.exp is sp.S.Half:
                root = _call("SQRT", [self._print(expr.base)])
                return _binop("/", _numeral("1"), root)
            if expr.exp is sp.S.NegativeOne:
                if expr.base.is_Add:
                    content = expr.base.as_content_primitive()[0]
                    if content.is_Integer and content > 1:
                        return self._divided(expr)
                below = self._fenced(expr.base, level)
                return _run([("", _numeral("1"), False), ("/", *below)])
            if expr.exp.is_Number and expr.exp.is_negative:
                # A denominator, as `_divided` writes one in a product: `1/k^2` rather
                # than `k^(-2)`.
                denominator = self._print(sp.Pow(expr.base, -expr.exp))
                return _binop("/", _numeral("1"), denominator)
        return self._power(expr.base, expr.exp)

    def _power(self, base, exponent) -> Node:
        return _binop("^", self._print(base), self._print(exponent))

    def _print_MatPow(self, expr) -> Node:
        """A power of a matrix, written like any other power, `a^-1`; there is no `1/u`
        for a matrix."""
        if expr.exp.is_Number and expr.exp.is_negative:
            return self._power(expr.base, expr.exp)
        return self._print_Pow(expr)

    def _print_Inverse(self, expr) -> Node:
        """`a^-1`, the same power sympy keeps under a separate head."""
        return self._print_MatPow(expr)

    def _print_MatMul(self, expr) -> Node:
        """A matrix product, using the dot operator of section 8.4.

        Derive nests a chain to the right, `a . (b . (c . d))`, since the grammar is
        left-associative. A scalar coefficient goes in front, and a negated product is
        signed, so a sum reads `a . b - b . a`.
        """
        coefficient, matrices = expr.as_coeff_matrices()
        if not matrices:
            return self._print(coefficient)
        product = self._print(matrices[-1])
        for matrix in reversed(matrices[:-1]):
            product = _binop(".", self._print(matrix), product)
        if coefficient == 1:
            return product
        if coefficient == -1:
            return _signed("-", product)
        number, fenced = self._fenced(coefficient, PRECEDENCE["Mul"])
        return _run([("", number, fenced), ("*", product, len(matrices) > 1)])

    def _print_Transpose(self, expr) -> Node:
        """``u` ``, the same operator the inert head uses."""
        return _node(Kind.POSTOP, (self._print(expr.arg),), "`")

    def _print_exp(self, expr) -> Node:
        """`#e^u`, never `EXP(u)`, as the notation spells it."""
        return _binop("^", _name("#e"), self._print(expr.args[0]))

    def _print_factorial(self, expr) -> Node:
        return _node(Kind.POSTOP, (self._print(expr.args[0]),), "!")

    _print_Factorial = _print_factorial

    def _print_Relational(self, expr) -> Node:
        operator = _RELATIONS.get(expr.rel_op, expr.rel_op)
        return _relation(operator, self._print(expr.lhs), self._print(expr.rhs))

    # -- functions ----------------------------------------------------------

    def _print_Function(self, expr) -> Node:
        """`NAME(u, ...)`, or the call quoted under its sympy name if the notation has no
        name for it.
        """
        if _is_suspended(expr):
            return self._print_Suspended(expr)
        if isinstance(expr, (WrittenBase, HeldBase)):
            return type(expr).node
        name = spelling(expr)
        if name is None:
            return self._unspelled(expr)
        return _call(name, self._arguments(expr.args))

    def _print_Suspended(self, expr) -> Node:
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

    def _print_polygamma(self, expr) -> Node:
        """`DIGAMMA(x)` for the order-zero polygamma function, since sympy has no
        separate digamma function."""
        if _is_digamma(expr):
            return _call("DIGAMMA", [self._print(expr.args[1])])
        return self._print_Function(expr)

    def _print_Gcd(self, expr) -> Node:
        """`GCD(n, m)`, with arguments from least main up, as Derive writes them: `GCD(m,
        2)` is `GCD(2, m)`.
        """
        order = self.context.order
        operands = sorted(
            expr.args, key=lambda term: _term_key(term, order), reverse=True
        )
        return _call("GCD", self._arguments(operands))

    def _print_Modulo(self, expr) -> Node:
        """`MOD(m, n)` and `MODS(m, n)`, written `MOD(m)` for the unit modulus."""
        name = spelling(expr)
        value, modulus = expr.args
        if name is None or modulus != 1:
            return self._print_Function(expr)
        return _call(name, [self._print(value)])

    def _print_Min(self, expr) -> Node:
        return _call("MIN", self._arguments(expr.args))

    def _print_Max(self, expr) -> Node:
        return _call("MAX", self._arguments(expr.args))

    def _print_floor(self, expr) -> Node:
        """`FLOOR(m, n)` if the argument is a quotient, otherwise `FLOOR(u)`.

        The notation's two-argument floor (section 6.7: `MOD(m, n)` is `m - n*FLOOR(m,
        n)`) is used only if the denominator contains a variable, with a numeric factor
        moved to the numerator: `FLOOR(m/n + 1/2)` is `FLOOR(m + n/2, n)`.
        """
        numerator, denominator = sp.fraction(sp.together(expr.args[0]))
        if denominator.is_number:
            return _call("FLOOR", [self._print(expr.args[0])])
        coefficient, denominator = denominator.as_coeff_Mul()
        return _call(
            "FLOOR", [self._print(numerator / coefficient), self._print(denominator)]
        )

    def _print_Heaviside(self, expr) -> Node:
        """`STEP(u)`, omitting the value at zero, which is ours, not the author's."""
        return _call("STEP", [self._print(expr.args[0])])

    def _print_Determinant(self, expr) -> Node:
        """`DET(u)` for a matrix sympy keeps unevaluated. It isn't a `Function`, so
        `spelling` doesn't cover it.
        """
        return _call("DET", [self._print(expr.arg)])

    def _print_Trace(self, expr) -> Node:
        return _call("TRACE", [self._print(expr.arg)])

    def _print_Piecewise(self, expr) -> Node:
        """A case split, as nested `IF(test, then, else)`. Without a final `true`
        condition the last case has no else, which is correct, since the value is then
        `?`.
        """
        pairs = list(expr.args)
        node = None
        if pairs and pairs[-1][1] is sp.true:
            node = self._print(pairs.pop()[0])
        for value, condition in reversed(pairs):
            parts = [self._print(condition), self._print(value)]
            if node is not None:
                parts.append(node)
            node = _call("IF", parts)
        return _node(Kind.UNKNOWN, value="?") if node is None else node

    def _print_ITE(self, expr) -> Node:
        """A case split between truth values, as `IF(test, then, else)`. Sympy makes
        one of a comparison with a case split: `DIMENSION(IF(k = 1, [f], [])) < 1`.
        """
        return _call("IF", self._arguments(expr.args))

    # -- calculus heads that reached the output unevaluated -----------------

    def _print_Derivative(self, expr) -> Node:
        primed = self._primed(expr, expr.variable_count[0][0])
        if primed is not None:
            return primed
        node = self._print(expr.expr)
        for variable, count in expr.variable_count:
            parts = [node, self._print(variable)]
            if count != 1:
                parts.append(self._print(count))
            node = _call("DIF", parts)
        return node

    def _primed(self, expr, point) -> Node | None:
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
        return _call(f"{function_name(call)}{marks}", [self._print(point)])

    def _print_Integral(self, expr) -> Node:
        return self._limited("INT", expr)

    def _print_Sum(self, expr) -> Node:
        return self._limited("SUM", expr)

    def _print_Product(self, expr) -> Node:
        return self._limited("PRODUCT", expr)

    def _limited(self, head: str, expr) -> Node:
        """`INT(u, x)` and `INT(u, x, a, b)`, one head per limit."""
        node = self._print(expr.function)
        for limit in expr.limits:
            node = _call(head, [node, *self._arguments(limit)])
        return node

    def _print_Limit(self, expr) -> Node:
        """`LIM(u, x, a)`, with a fourth argument for a one-sided limit. The direction
        sympy records at infinity is omitted, since the author didn't ask for a
        one-sided limit.
        """
        expression, variable, point, direction = expr.args
        parts = self._arguments((expression, variable, point))
        side = {"+": "1", "-": "-1"}.get(str(direction))
        if side is not None and not point.is_infinite:
            parts.append(_numeral(side))
        return _call("LIM", parts)

    # -- bounded limits that don't converge ------------------------------------

    def _print_AccumulationBounds(self, expr) -> Node:
        """`INTERVAL(a, b)`: a value known only to lie between `a` and `b`.

        This is the result of a bounded limit that doesn't converge: `LIM(SIN(x), x,
        inf)` is `INTERVAL(-1, 1)`. The name is Mathematica's and Maple's, and it is an
        atom, so it works inside larger expressions.
        """
        return _call("INTERVAL", self._arguments(expr.args))

    # -- roots of polynomials that don't factor --------------------------------

    def _print_RootSum(self, expr) -> Node:
        """`ROOT_SUM(p, t, u)`: the sum of `u` over every root `t` of `p`. The summand, a
        `Lambda` in sympy, is written in the polynomial's generator.
        """
        variable = expr.poly.gen
        parts = (expr.poly.as_expr(), variable, expr.fun(variable))
        return _call("ROOT_SUM", self._arguments(parts))

    def _print_ComplexRootOf(self, expr) -> Node:
        """`ROOT_OF(p, t, n)`: the `n`-th root of `p` in `t`, counted from zero. The
        generator is taken from the polynomial.
        """
        parts = (expr.poly.as_expr(), expr.poly.gen, sp.Integer(expr.index))
        return _call("ROOT_OF", self._arguments(parts))

    # -- vectors and matrices ------------------------------------------------

    def _print_Tuple(self, expr) -> Node:
        """A tuple, written as a vector, since the grammar can't read a parenthesized
        list.
        """
        return self._vector(expr.args)

    def _print_tuple(self, expr) -> Node:
        return self._vector(expr)

    def _print_list(self, expr) -> Node:
        return self._vector(expr)

    def _print_Subs(self, expr) -> Node:
        """`SUBS(u, [x], [a])`, upper-cased to be a fixed point, or a prime over an
        undefined function: `DIF(BB(r^2), r)` is `2*r*BB'(r^2)`.
        """
        inner, variables, points = expr.args
        if isinstance(inner, sp.Derivative) and len(variables) == len(points) == 1:
            primed = self._primed(inner, points[0])
            if primed is not None and inner.variable_count[0][0] == variables[0]:
                return primed
        return _call("SUBS", self._arguments(expr.args))

    _print_Substitution = _print_Subs

    def _print_MatrixBase(self, expr) -> Node:
        """A sympy matrix, met only in a product with a symbolic one, as a vector of
        rows like any explicit matrix (a `Vector`)."""
        rows = [self._vector(expr.row(row)) for row in range(expr.rows)]
        return _node(Kind.VECTOR, rows)

    def _vector(self, elements) -> Node:
        return _node(Kind.VECTOR, self._arguments(elements))

    # -- logic ---------------------------------------------------------------

    def _print_And(self, expr) -> Node:
        chained = self._chain(expr)
        if chained is not None:
            return chained
        order = not getattr(expr, "in_order", False)
        return self._infix("AND", expr.args, order=order)

    def _chain(self, expr) -> Node | None:
        """Two bounds on one variable as the chain `-2 < x < 2`, as the original writes a
        range. The two sides' strictness is independent.
        """
        if len(expr.args) != 2:
            return None
        found = _bracket(*(_bounds(operand) for operand in expr.args))
        if found is None:
            return None
        subject, (lower, low), (upper, high) = found
        inner = _relation(_FROM_BELOW[lower], self._print(low), self._print(subject))
        return _relation(upper, inner, self._print(high))

    def _print_Or(self, expr) -> Node:
        order = not getattr(expr, "in_order", False)
        return self._infix("OR", expr.args, order=order)

    def _print_Xor(self, expr) -> Node:
        return self._infix("XOR", expr.args)

    def _print_Implies(self, expr) -> Node:
        # `IMP` isn't commutative, so its operands keep their order.
        return self._infix("IMP", expr.args, order=False)

    def _print_Equivalent(self, expr) -> Node:
        if len(expr.args) != 2:
            # All operands have one truth value: each equivalent to the next.
            links = [
                sp.Equivalent(first, second, evaluate=False)
                for first, second in zip(expr.args, expr.args[1:], strict=False)
            ]
            return self._infix("AND", links, order=False)
        return self._infix("IFF", expr.args, order=False)

    def _print_Not(self, expr) -> Node:
        return _node(Kind.NOT, (self._print(expr.args[0]),))

    def _infix(self, word: str, operands, order: bool = True) -> Node:
        """The operands joined by the logical operator `word`, which associates as it is
        read: `IMP` and `IFF` to the left, the others to the right."""
        if order:
            operands = logical_order(operands, self.context.order)
        kind = _LOGICAL_KINDS[word]
        nodes = self._arguments(operands)
        if word in ("IMP", "IFF"):
            node = nodes[0]
            for other in nodes[1:]:
                node = _node(kind, (node, other))
            return node
        node = nodes[-1]
        for other in reversed(nodes[:-1]):
            node = _node(kind, (other, node))
        return node

    # -- the inert heads ------------------------------------------------------

    def _print_PlusMinus(self, expr) -> Node:
        return _signed("+-", self._print(expr.args[0]))

    def _print_StringLiteral(self, expr) -> Node:
        return _node(Kind.STRING, value=expr.name)

    def _unspelled(self, expr) -> Node:
        """A head the notation can't express, written as a quoted string.

        Written bare, sympy's `EmptySet` would read back as `e*m*p*t*y*s*e*t` and
        `Interval(0, 1)` as the built-in `INTERVAL`, both looking like computed
        answers. The string holds what sympy writes for it, its operands written as
        usual; nothing inside is quoted again.
        """
        if self._unspelling:
            return _verbatim(_Quoting(self).doprint(expr))
        self._unspelling = True
        try:
            return _node(Kind.STRING, value=_Quoting(self).doprint(expr))
        finally:
            self._unspelling = False

    def _print_Power(self, expr) -> Node:
        return self._power(*expr.args)

    def _print_Transposed(self, expr) -> Node:
        return _node(Kind.POSTOP, (self._print(expr.args[0]),), "`")

    def _print_Subscript(self, expr) -> Node:
        base, index = expr.args
        return _node(Kind.SUB, (self._print(base), self._print(index)))

    def _print_InfixBase(self, expr) -> Node:
        """A word operator that stands."""
        left, right = expr.args[:2]
        children = (self._print(left), self._print(right))
        return _node(Kind.INFIX, children, expr.operator)

    def _print_Added(self, expr) -> Node:
        """A sum kept as written: a term that is a sum itself joins the run, as
        `(x + 1) + [1, 2]` reads as `x + 1 + [1, 2]`."""
        level = PRECEDENCE["Add"]
        return _sum_of([self._fenced(term, level, strict=True) for term in expr.args])

    def _print_Quotient(self, expr) -> Node:
        level = PRECEDENCE["Mul"]
        left, right = expr.args
        return _run(
            [("", *self._fenced(left, level)), ("/", self._print(right), True)]
        )

    def _print_Dot(self, expr) -> Node:
        """`u . v`; a chain nested to the left, `P . L . D`, is left-associative."""
        left, right = expr.args
        return _binop(".", self._print(left), self._print(right))

    def _print_UnevaluatedExpr(self, expr) -> Node:
        return self._print(expr.args[0])

    def _print_Truth(self, expr) -> Node:
        return self._print(expr.args[0])

    def _print_Vector(self, expr) -> Node:
        return self._vector(expr.args)

    def _print_SetOf(self, expr) -> Node:
        return _node(Kind.SET, self._arguments(expr.args))

    def _print_ElidedVector(self, expr) -> Node:
        return _node(Kind.VECTOR, self._elided(expr.args))

    def _print_ElidedSet(self, expr) -> Node:
        return _node(Kind.SET, self._elided(expr.args))

    def _elided(self, terms) -> list[Node]:
        """The terms of an arithmetic sequence, the ellipsis before the last."""
        written = self._arguments(terms)
        ellipsis = _node(Kind.ELLIPSIS, value=ELLIPSIS)
        return [*written[:-1], ellipsis, written[-1]]

    def _print_SetUnion(self, expr) -> Node:
        return self._set_operation(expr, "UNION")

    def _print_SetIntersection(self, expr) -> Node:
        return self._set_operation(expr, "INTERSECTION")

    def _print_SetDifference(self, expr) -> Node:
        return self._set_operation(expr, "\\")

    def _set_operation(self, expr, operator: str) -> Node:
        """Operands left to right, the same operator nested on the left, as the
        grammar's left-associative set operators read."""
        first, *rest = self._arguments(expr.args)
        for operand in rest:
            first = _node(Kind.SETOP, (first, operand), operator)
        return first

    def _print_SetComplement(self, expr) -> Node:
        return _node(Kind.POSTOP, (self._print(expr.args[0]),), "`")

    def _print_Logical(self, expr) -> Node:
        """A logical operator sympy rejected, with operands unsimplified in their written
        order: `3 AND p` is not `p AND 3`.
        """
        word = str(expr.args[0])
        operands = expr.args[1:]
        if len(operands) == 1:
            return _node(Kind.NOT, (self._print(operands[0]),))
        return self._infix(word, operands, order=False)

    def _print_Assign(self, expr) -> Node:
        target, operator, *value = expr.args
        children = [self._print(target), *self._arguments(value)]
        return _node(Kind.ASSIGN, children, str(operator))

    def _print_FunDef(self, expr) -> Node:
        name, parameters, *body = expr.args
        if isinstance(parameters, sp.Tuple):
            params = _node(Kind.PARAMS, self._arguments(parameters))
        else:
            # The variadic head `F v`, whose one parameter takes all the arguments.
            params = _node(Kind.PARAMS, [self._print(parameters)], VARIADIC)
        return _node(Kind.FUNDEF, [params, *self._arguments(body)], str(name))

    def _print_Declare(self, expr) -> Node:
        """`x :epsilon Real`, with the interval when one was declared."""
        target, kind, *interval = expr.args
        children = [self._print(target)]
        if len(interval) == 3:
            brackets, low, high = interval
            children.append(
                _node(Kind.INTERVAL, self._arguments((low, high)), str(brackets))
            )
        return _node(Kind.DOMAIN, children, str(kind))


#: The node kind of each logical operator.
_LOGICAL_KINDS = {
    "AND": Kind.AND,
    "OR": Kind.OR,
    "XOR": Kind.XOR,
    "IMP": Kind.IMP,
    "IFF": Kind.IFF,
}


def _subscript_parts(name: str) -> list[str]:
    """The parts of a subscripted symbol's name, `x SUB 1 SUB (i SUB k)`: the base and
    each index, an index that is itself subscripted in parentheses."""
    parts, depth, start = [], 0, 0
    for at, char in enumerate(name):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif not depth and name.startswith(_SUBSCRIPT, at) and at >= start:
            parts.append(name[start:at])
            start = at + len(_SUBSCRIPT)
    parts.append(name[start:])
    return parts


class _Quoting(sp.StrPrinter):
    """The text of an expression the notation has no form for (see
    `TreePrinter._unspelled`): sympy's, its operands written in author notation."""

    def __init__(self, trees: TreePrinter) -> None:
        super().__init__({"full_prec": False, "order": "grlex"})
        self.trees = trees
        self.depth = 0

    def _print(self, expr, **kwargs) -> str:
        """The expression itself as sympy writes it, a nameless head inside it so too,
        and any other operand in author notation."""
        if self.depth and isinstance(expr, sp.Basic) and not _nameless(expr):
            return write_expression(self.trees._print(expr))
        self.depth += 1
        try:
            return super()._print(expr, **kwargs)
        finally:
            self.depth -= 1

    def _print_Function(self, expr) -> str:
        return f"{type(expr).__name__}({self.stringify(expr.args, ', ')})"


def _nameless(expr: sp.Basic) -> bool:
    """Whether the notation has no form for `expr`: a set, or a head with no name."""
    if isinstance(expr, sp.Set):
        return True
    return isinstance(expr, sp.Function) and not _has_word(expr)

"""sympy -> author notation.

The engine's results leave sympy as text, and this is where that text is
written: `^` for powers, `SQRT(u)` for square roots, `#e` and `#i` for the
constants, upper-case function names, `?` for the unknown value. Every result
of every command goes through here, so nothing in it may assume which command
produced the expression.

Two rules govern the output. It must reparse to the same expression - the
printer never leans on juxtaposition and parenthesises wherever the grammar
needs it - and it must be written in the notation the session is using, which
is why numerals come out in the context's input base and in the style
`Options Notation` selects. `engine.notation` holds that second part.

A decimal style bends the first rule, and the original bends it the same way:
one third written `0.333333` reads back as a different number. Derive shows
that text on the screen and writes it to an MTH file alike, so what the
notation cuts is cut from the worksheet too - but not from the value. The
exact writing is kept beside the shown one, which is `Result.value`, and it is
what the next command is given; `3·#n` is 1 whatever `#n` is shown as.

`AuthorPrinter` subclasses sympy's `StrPrinter` and overrides one method per
construct spelled differently: powers, roots, the constants, function names,
vectors, the unevaluated calculus heads, the inert heads, and numerals. Of the
sum and product spine it overrides only the order the operands come in and
where a bracket's minus sign goes; sign extraction, collecting negative powers
into a denominator, and a product sympy was told not to evaluate are all still
sympy's, because reimplementing them would mean carrying a few hundred lines of
subtle logic for no gain in what the output means. The cost is that sympy's own
conventions show through: `x*k^-2` comes out as `x/k^2` because that is how a
denominator is split. That is a form difference, never a meaning difference,
and both forms reparse to the same tree. `_print_Pow` follows the same
convention for a power standing alone, so that one expression does not print
two ways.

One construct is written shorter than sympy holds it. A conjunction of two
relations that bound one variable is a range, and a range is written as the
chain `-2 < x < 2` - which is how the original writes one, how the author line
reads one back, and how the layout draws one. Everything else keeps its `AND`.

One is written longer. Sympy holds a matrix product flat and the original
nests it to the right, parentheses and all: `((a . b) . c) . d` comes back
`a . (b . (c . d))`, which is where the original's own canonical form puts
them, so `_print_MatMul` writes the nesting rather than the chain. An inverse
goes with it - there is no `1/u` for a matrix, and `u^-1` is the whole of the
notation for one.

The order a sum and a product are written in is ours, and it is the order
list's - the same list the rest of the engine takes its variables in. A sum
runs by descending kernel: a call leads whatever variable it holds,
`SIN(y) + x`, then everything built on a variable, by that variable and then
by descending degree, so `5*y^3 + 2*x^2 - 3*a*x` is written
`2*x^2 - 3*a*x + 5*y^3` and `SQRT(x + 1) + x` stands as it is. A term holding
no variable goes last, `x^2 + c + 1`.

A product is not that order backwards. The number leads, then the constants,
then the plain variables, then the powers of a sum, then the calls, so that
`y*SIN(x)` and `x*SQRT(x + 1)` are written as they stand and a factorization
comes out `(x + 2)*(x - 2)*(x + 1)^2`. The variable is compared the same way
round in both while the kinds of factor run the other way, which is why there
are two comparisons here and not one.

A sum of *two* is turned round rather than begun with a minus sign - `y - x`,
`SQRT(3) - 1`, `2*x - x^2` - and a sum of three or more is not: `-x^2 + 2*x - 1`
keeps the minus, as the original does.

Where the list says nothing the operands keep whatever order sympy already had
them in, which is at least the same order every time. The operands of `AND` and
`OR` are ours by the same list: `c OR b AND a` is written `a AND b OR c`.

The inert heads of `to_sympy` are printed by name rather than by import: a
sympy printer dispatches on the class name, so `_print_PlusMinus` finds
`PlusMinus` without this module depending on the module that defines it.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from fractions import Fraction

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.core.relational import Relational
from sympy.printing.precedence import PRECEDENCE, PRECEDENCE_VALUES, precedence

from rederive.engine import notation
from rederive.engine.context import Context, Notation
from rederive.engine.ordering import main_order
from rederive.syntax.names import GREEK_GLYPHS

# How tightly the inert heads bind. Sympy's `parenthesize` reads these by class
# name, and without an entry it takes any `Function` for a call as tight as
# `SIN(x)` - which would write `#e^(±inf*z)` as `#e^±inf*z`, a different
# expression. Each is registered as the operator it is written as.
#
# `Dot` sits below `Add` rather than at `Mul`, where it belongs: `.` and `*` are
# one precedence level in the grammar and associate to the left, so `x*(a . b)`
# needs its parentheses to keep from reading as `(x*a) . b` - and sympy prices a
# negated product at the `Add` level, so an entry at `Mul` would lose them again
# in `-x*(a . b)`. Complex infinity is written with the plus-or-minus operator
# and binds as loosely as one.
#
# `MatMul` is the same operator over the matrices sympy does know how to
# multiply, and sits at `Add` rather than below it: a matrix product is written
# without parentheses inside a sum - `a . b + a . c` - and with them inside a
# power or a transpose.
PRECEDENCE_VALUES.setdefault("PlusMinus", PRECEDENCE["Add"])
PRECEDENCE_VALUES.setdefault("ComplexInfinity", PRECEDENCE["Add"])
PRECEDENCE_VALUES.setdefault("Dot", PRECEDENCE["Add"] - 1)
PRECEDENCE_VALUES.setdefault("MatMul", PRECEDENCE["Add"])
PRECEDENCE_VALUES.setdefault("Subscript", PRECEDENCE["Atom"] - 1)
PRECEDENCE_VALUES.setdefault("Power", PRECEDENCE["Pow"])

_DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

#: What joins the parts of a subscripted symbol's name.
_SUBSCRIPT = " SUB "

#: The plus-or-minus operator, as the glyph rather than its other input
#: spelling `"+-"`. A bare `+-u` is neither: it is a plus over a minus, and
#: reads back as `-u`.
_PLUS_MINUS = "±"

#: How each relation is written. Everything not here is written as it is.
_RELATIONS = {"==": "=", "!=": "/="}

#: The two operators that put a bound below their subject, and how each is
#: written once the chain has turned it round: `x > -2` is `-2 < x`.
_FROM_BELOW = {">": "<", ">=": "<="}

#: The operators a chain can be built out of at all.
_ORDERINGS = ("<", "<=", ">", ">=")

#: sympy heads whose author-notation name is not just their name upper-cased.
_FUNCTION_NAMES = {
    "Antidifference": "SUM",
    "Antiquotient": "PRODUCT",
    "log": "LN",
    "arg": "PHASE",
    "atan2": "ATAN",
    "conjugate": "CONJ",
    "binomial": "COMB",
    "FallingFactorial": "PERM",
    "Heaviside": "STEP",
    # Upper-casing would write the hyperbolic cosine integral as `CHI`, which
    # is Derive's chi-square distribution and reads back as one; its partner is
    # named alongside it so that the pair is written the same way.
    "Chi": "COSH_INT",
    "Shi": "SINH_INT",
    "Determinant": "DET",
    "Trace": "TRACE",
}

#: The logical operators, tightest first, so that an operand knows when it
#: needs parentheses. A relation and anything below it never does.
_LOGICAL_PRECEDENCE = {"NOT": 5, "AND": 4, "OR": 3, "XOR": 2, "IMP": 1}

_LOGICAL_HEADS = {
    "And": "AND",
    "Or": "OR",
    "Xor": "XOR",
    "Implies": "IMP",
    "Not": "NOT",
}


def _logical_order(operands, order: Sequence[str]) -> list[sp.Basic]:
    """The operands of a conjunction or a disjunction, in the original's order.

    A commutative operator has no order of its own, so one is chosen, and the
    one chosen is the order list's: `r OR q OR p` is written `p OR q OR r`,
    and `c OR b AND a` is written `a AND b OR c`. A term sorts by the variable
    it leads with rather than by any later one, which is why the conjuncts are
    put in order first and the disjuncts after. `NOT` does not move a term -
    `q OR NOT p` is `NOT p OR q` - but it does decide a tie, a negated literal
    coming before the same variable unnegated.
    """
    return sorted(operands, key=lambda operand: _logical_key(operand, order))


def _logical_key(operand: sp.Basic, order: Sequence[str]) -> tuple:
    """Where one operand sorts: its literals, in order, each with its sign.

    Always a tuple of literal keys, whether the operand is one literal or a
    whole subexpression, so that any two operands compare.
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
    """The most main variable of one literal, as something sortable.

    A variable on the order list is more main than one after it and than one
    off the list altogether; the rest go alphabetically. A literal that names
    no variable at all - a truth value, a numeral - sorts by how it is
    written, which is arbitrary but the same every time.
    """
    names = main_order((symbol.name for symbol in operand.free_symbols), order)
    if not names:
        return (2, str(operand))
    name = names[0]
    if name in order:
        return (0, order.index(name))
    return (1, name)


#: The places a kernel of a term can sort in, nearest the front of the sum
#: first. A function call leads whatever variable it holds - `SIN(y) + x` -
#: then everything built on a variable; then a sum under the bar, which does
#: not lend the term the variable it holds and so closes the sum whatever that
#: variable is - `y - 1/(x + 1) + 1`, `x^2 + y + 1/(x + 1)`. Then a kernel
#: holding no variable at all, such as `SQRT(3)`, so that a sum is written
#: `SQRT(3) - 1`.
#:
#: The last two are no kernels. `_NOTHING` closes every term, so that a term
#: made of nothing but a number sorts behind every term that is not; and `#i`
#: sorts behind even that, so that the real part of a complex number is written
#: first - `1 + 7*#i`, `4/3 + #i/2`.
_CALLED, _BUILT, _UNDER, _CONSTANT, _NOTHING, _IMAGINARY = range(6)

#: Which of the two `_BUILT` kernels of one variable leads: a power of a
#: compound base beats the bare variable, `SQRT(x + 1) + x`.
_OVER_A_SUM, _OVER_THE_VARIABLE = range(2)

#: What stands at the end of every term's key.
_AFTER_EVERY_KERNEL = (_NOTHING,)


def _term_order(terms: Sequence[sp.Basic], order: Sequence[str]) -> list[sp.Basic]:
    """The terms of a sum, in the original's order.

    A stable sort over the order sympy already put them in, so that terms the
    order list says nothing about keep an arrangement that is at least the same
    every time - which is what makes printing a fixed point.
    """
    return sorted(terms, key=lambda term: _term_key(term, order))


def _term_key(term: sp.Basic, order: Sequence[str]) -> tuple:
    """Where one term of a sum sorts: the kernels it is made of, most main
    first.

    A term sorts by the kernel it leads with rather than by any later one, so
    `2*x^2 - 3*a*x + 5*y^3` runs the two x terms first however high the power of
    `y` beside them is, and among terms of one variable the higher power leads.
    Both fall out of comparing the kernels in order: the leading kernel decides
    unless the two terms lead with the same one at the same power, and then the
    kernel after it does.

    A term with a sum under the bar leads with that sum whatever else it holds,
    so that the proper part of a rational function closes the sum it belongs
    to: `y + (2*x + 3)/((x + 1)*(x + 2))`, where the numerator is an x and the
    term still goes behind the plain `y`.

    An imaginary term leads with the `#i` in the same way, whatever it is
    multiplied by, so that the real part of a complex number is written first
    however that part is spelled: `1 + SQRT(3)*#i` and not `SQRT(3)*#i + 1`.
    """
    kernels = sorted(
        (
            _kernel_key(factor, order)
            for factor in sp.Mul.make_args(term)
            if not factor.is_Number
        ),
        key=lambda kernel: (kernel[0] != _UNDER, kernel[0] != _IMAGINARY, kernel),
    )
    return (*kernels, _AFTER_EVERY_KERNEL)


def _kernel_key(factor: sp.Basic, order: Sequence[str]) -> tuple:
    """Where one kernel of a term sorts: its kind, its variable, then what
    tells two kernels of that kind and that variable apart.

    A call leads whatever variable it holds, so `SIN(y) + x` is written as it
    stands and `COS(u) + SIN(t)` is written `SIN(t) + COS(u)`. Everything else
    compares by its variable first - `SQRT(y + 1) + x` is written
    `x + SQRT(y + 1)` - and only then does a power of a sum beat the bare
    variable, `SQRT(x + 1) + x`. Powers of the bare variable run by descending
    degree, `x^2 + SQRT(x)`.

    A sum or a product raised to a power is one kernel and not the variables
    inside it: `(x + 1)^3` is one bracket, and it lends the term the variable
    it is a polynomial in. A sum under the bar lends it nothing: the proper
    part of a rational function closes the sum it belongs to.
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


#: The places a factor of a product can sort in, leftmost first. The numeric
#: coefficient leads, then what stands beside it as a constant - `2*pi*r^2`,
#: `SQRT(2)*z/2`, `x - #i*y` - then the plain variables, then the powers of a
#: sum, then the calls: `2*x*y`, `x*SQRT(x + 1)`, `y*SIN(x)`. A plain variable
#: beats a kernel even of an earlier variable, and a kernel beats a call.
_A_NUMBER, _A_CONSTANT, _A_VARIABLE, _A_POWER, _A_CALL = range(5)


def _factor_order(factors: Sequence[sp.Basic], order: Sequence[str]) -> list[sp.Basic]:
    """The factors of a product, in the original's order.

    A product is not a sum written backwards. The variable is compared the same
    way round in both - `x*y` beside `x + y` - while the kinds of factor run
    the other way, a call last in a product where it leads a sum, and the calls
    of one variable run backwards among themselves: `SIN(x)*COS(x)` where the
    sum is `COS(x) + SIN(x)`. So the two need two comparisons and not one.
    """
    return sorted(factors, key=lambda factor: _factor_key(factor, order))


def _factor_key(factor: sp.Basic, order: Sequence[str]) -> tuple:
    """Where one factor of a product sorts: its kind, its variable, then what
    tells two factors of that kind and that variable apart.

    The exponents of plain variables have no say - `z^2*x^3*y^5` is written
    `x^3*y^5*z^2` - while the powers of a sum sort by the degree of the base,
    then by the power it is raised to, then by its coefficients from the
    leading one down: `(x + 2)*(x - 2)*(x + 1)^2`.

    Two constants keep whatever order sympy had them in, there being no
    variable in either to compare.
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
    """The most main variable of one factor, as a product sorts by it.

    A product is the one place the order list runs the other way at the top.
    The manual's own example is `z^3*b^2*a*2*x^5*c`, which is written
    `2*a*b^2*c*x^5*z^3`: the names the list does not know go alphabetically
    ahead of the names it does, and the names it does keep the list's order.
    """
    place, within = _variable_key(operand, order)
    return ((1, within), (0, within), (2, within))[place]


#: The order the calls of one variable run in in a sum, and the reverse of the
#: order they run in in a product. Checked against the original a pair at a
#: time, each pair tried both ways round and a fixed point one way only: `#e^x`
#: leads, and `LN` before `COS` and `TAN` before `SIN` are not alphabetical, so
#: this is an order of the original's own rather than a spelling of one.
#:
#: These eight are the whole of what has been read. Every other head sorts
#: behind them and among those alphabetically, which is a place to put them and
#: not a reading of where the original puts them.
_HEAD_ORDER = ("#e^", "ABS", "ASIN", "ATAN", "LN", "COS", "TAN", "SIN")

#: What `_kernel_parts` calls the exponential, which sympy holds as a power of
#: `#e` and the original holds as a call like any other.
_EXPONENTIAL = "#e^"


#: The four kinds of kernel a factor can be, which is what decides where both
#: a sum and a product put it. The two put the kinds in a different order, so
#: what a kind is called says nothing about where it sorts.
(
    _A_CONSTANT_KERNEL,
    _A_NAMED_KERNEL,
    _A_BRACKET_KERNEL,
    _A_CALLED_KERNEL,
) = ("constant", "name", "bracket", "call")


def _kernel_parts(factor: sp.Basic) -> tuple[str, sp.Basic, sp.Basic, str]:
    """`factor` as its kind, what it is a power of, the power, and the name of
    the head it is written with.

    The exponential is a call and not a power of `#e`: `#e^x` is a kernel of
    `x`, and it leads a sum of calls of `x` the way the original leads one.
    A power of a number - `SQRT(3)`, `2^(-m)` - is a kernel of whatever its
    exponent holds, so that `2^x + y` is not written as though `2^x` were a
    constant.
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
    """What a kernel that is neither a name nor a bracket is written with.

    A call is written with its own head; anything else sympy holds as a class
    of its own - a derivative, a sum, an integral - is written with the name of
    that class, which is a place to put it and not a reading of where the
    original puts it.
    """
    if isinstance(base, sp.Function):
        return function_name(base)
    return type(base).__name__.upper()


def _head_key(name: str) -> tuple:
    """Where one function head sorts among the heads of its variable."""
    if name in _HEAD_ORDER:
        return (0, _HEAD_ORDER.index(name), "")
    return (1, 0, name)


class _Reversed:
    """A key that sorts the other way round.

    The kernels of one variable compare by the mirror in a sum of what they
    compare by in a product - `SQRT(x + 2) + SQRT(x + 1)` beside
    `(x + 1)*(x + 2)` - so the one key serves both.
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
    """Where a power of a sum sorts among the powers of sums of its variable.

    Ascending is the order a product's factors are written in, which is the
    order the original writes a factorization in: the degree of the base
    first, `(x - 2)*(x^2 + 1)`; then the multiplicity, so that a repeated
    factor closes a run of same-degree ones, `(x + 3)*(x + 1)^2`; then the
    coefficients from the leading one down, `(x + 3)*(2*x + 1)` and
    `(x + 2)*(x - 2)`.

    A sum's kernels are written in the mirror of it: `SQRT(x + 2) +
    SQRT(x + 1)` where the product is `(x + 1)*(x + 2)`, and `SQRT(x - 2) +
    SQRT(x + 2)` where the product is `(x + 2)*(x - 2)`.
    """
    degree, coefficients = _polynomial_key(base, order)
    return (degree, _size(exponent), coefficients)


def _polynomial_key(base: sp.Basic, order: Sequence[str]) -> tuple:
    """A compound base as its degree in its own main variable, and then its
    coefficients from the leading one down.

    Anything sympy cannot read as a polynomial - a sum holding a call, a
    product of two brackets - is one place, ahead of every polynomial and in
    whatever order it was already in.
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
    """Where one coefficient sorts: by size, and a positive one ahead of the
    negative of the same size.

    `(x + 2)*(x - 2)` and `(x + y)*(x - y)`, and `x` counting as `x + 0`, zero
    being the smallest size there is: `x*(x + 2)`.
    """
    try:
        number, rest = value.as_coeff_Mul()
        return (_magnitude(number), 1 if number.is_negative else 0, str(rest))
    except Exception:
        return (0.0, 0, str(value))


def _magnitude(number: sp.Basic) -> Fraction | float:
    """How big a numeric coefficient is, whatever kind of number it is."""
    if number.is_Rational:
        return abs(Fraction(int(number.p), int(number.q)))
    try:
        return abs(float(number))
    except (TypeError, ValueError):
        return 0.0


def function_name(expression: sp.Basic) -> str:
    """What a function head is called in author notation.

    An inert head keeps the name it was authored with; a sympy head is
    upper-cased, which is the spelling of a function name.
    """
    name = type(expression).__name__
    if name in _FUNCTION_NAMES:
        return _FUNCTION_NAMES[name]
    if isinstance(expression, AppliedUndef):
        return name
    return name.upper()


def _degree(exponent: sp.Basic) -> tuple:
    """A power as something sortable, descending.

    A symbolic exponent has no size to compare with a numeric one, and none to
    compare with another symbolic one either, so all of those are one place -
    behind every numeric power, and among themselves in whatever order they
    were already in.
    """
    if exponent.is_Rational:
        return (0, -Fraction(int(exponent.p), int(exponent.q)))
    return (1, Fraction(0))


def _size(exponent: sp.Basic) -> tuple:
    """The same power sortable ascending, which is how a product takes one."""
    place, degree = _degree(exponent)
    return (place, -degree)


def author_text(
    expression: sp.Basic,
    context: Context | None = None,
    authored: dict[sp.Basic, str] | None = None,
) -> str:
    """Write `expression` the way the author line would.

    `authored` names subexpressions that are to be written as the text beside
    them rather than as the form they were converted to. `pipeline.simplify`
    fills it with the conditionals an author wrote, whose arms are the one part
    of an answer that is shown as written and not as computed.
    """
    return AuthorPrinter(context, authored).doprint(named(expression))


def named(expression: sp.Basic) -> sp.Basic:
    """The same expression with sympy's invented variables named as ours.

    A `Dummy` is an ordinary variable by the time it reaches the worksheet -
    the bound variable the Leibniz rule introduces is written `k1` and read
    back as the variable `k1`. Two of them can carry one name, though, which
    sympy tells apart by an identity no text can hold, so they are renamed
    apart from each other and from every name already in use before anything
    is written. Without that, `SUM(SUM(f(k1), k1, 0, n), k1, 0, n)` is written
    from an expression with two variables and read back as one with one.

    Order needs it too: a `Dummy` sorts by its identity where a symbol sorts by
    its name, so a sum holding one is written in one order and reads back in
    another - the answer would settle only on a second pass.

    A matrix's dimensions are the exception. They are never written, so they
    need no name, and giving one to a declared nonscalar's shape would put a
    variable on the worksheet that the user could then substitute for.
    """
    shapes = _shapes(expression)
    dummies = sorted(expression.atoms(sp.Dummy) - shapes, key=sp.default_sort_key)
    if not dummies:
        return expression
    taken = {
        symbol.name
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
    try:
        return expression.xreplace(renamed)
    except Exception:
        return expression


def _shapes(expression: sp.Basic) -> set[sp.Dummy]:
    """The invented variables that only give a matrix its dimensions.

    A matrix written out has its dimensions as plain counts; only a symbolic
    one - a declared nonscalar - carries a variable there.
    """
    if not expression.has(sp.MatrixSymbol):
        return set()
    return {
        symbol
        for matrix in expression.atoms(sp.MatrixSymbol)
        for dimension in matrix.shape
        if isinstance(dimension, sp.Basic)
        for symbol in dimension.atoms(sp.Dummy)
    }


#: One reading of a relation: the variable it is about, which way it goes, and
#: what it bounds that variable by.
_Bound = tuple[sp.Basic, str, sp.Basic]


def _bounds(relation: sp.Basic) -> list[_Bound]:
    """Every way of reading a relation as a bound on one of its variables.

    `-3 < x` and `x > -3` are the same bound written two ways, so both come
    back as the second; `x < y` is a bound on either of them, and which one it
    is about is not decided until it is put beside the relation it is chained
    with.
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
    """The variable two relations bracket, with its bound from each side.

    None where they bracket nothing: two variables bounded apart, or one
    bounded twice from the same side. Where more than one reading brackets
    something the first is taken, the readings being the same statement.
    """
    for subject, one, value in first:
        for other, two, bound in second:
            if other != subject or (one in _FROM_BELOW) == (two in _FROM_BELOW):
                continue
            if one in _FROM_BELOW:
                return subject, (one, value), (two, bound)
            return subject, (two, bound), (one, value)
    return None


def _plain(name: str) -> str:
    """A dummy's name as the lexer would read one back."""
    stripped = name.lstrip("_")
    return stripped if stripped[:1].isalpha() else f"v{stripped}"


def numeral(number: int, base: int = 10) -> str:
    """`number` written in `base`.

    Above base ten the digits run A to Z, and a numeral that would begin with
    a letter takes a leading zero so that it cannot be read as a variable
    name: fourteen is `0E` in hexadecimal, not `E`.
    """
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


def _under_the_bar(power: sp.Basic) -> bool:
    """Whether a factor raised to `power` is a denominator in the notation.

    A numeric exponent is: `k^(-2)` is written `1/k^2`. A symbolic one is not,
    `2^(-m)` being how the notation writes that.
    """
    return bool(power.is_Number and power.is_negative)


def _all_negated(factor: sp.Basic) -> bool:
    """Whether every term of a bracket carries a minus sign.

    Such a bracket cannot be written without a leading one, so the sign goes
    out to the product instead: `z*(-b - 2)` is written `- z*(b + 2)`. A
    bracket with one positive term in it can be turned round rather than
    negated, and is left alone - `x*(2 - w)`.
    """
    return isinstance(factor, sp.Add) and all(
        term.could_extract_minus_sign() for term in factor.args
    )


def _unevaluated(expr: sp.Basic) -> bool:
    """Whether `expr` is a product sympy was told not to evaluate.

    Such a product is written by sympy's own spine, which is the only one that
    keeps every factor and every identity exactly where it was put. The test is
    sympy's: a leading one, or a number or a whole power of one anywhere but at
    the front, is something an evaluated product would have folded away.
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
        # `grlex` is descending total degree, which is the order the original
        # writes a polynomial in: `x^2 + c`, not `c + x^2`.
        super().__init__({"full_prec": False, "order": "grlex"})
        self.context = context or Context()
        self.authored = authored or {}
        self._unspelling = False

    def _print(self, expr, **kwargs):
        """One subexpression, as the author wrote it where that is on record.

        The text stands in for the whole subtree and nothing under it is
        printed, which is the point: the arms of a conditional whose test could
        not be decided are shown as they were written, arithmetic and operand
        order and all. Only the expression is replaced, never its position, so
        what surrounds it is spaced and parenthesized exactly as it would have
        been - the caller has already priced this subtree by its precedence,
        and the text is written at the same one.

        A conditional is the only thing recorded, and looking one up here costs
        a dictionary miss per printed node while the record is empty, which is
        every command but Simplify.

        A set is the one head caught on the way past rather than by a
        `_print_*` method of its own: sympy has a dozen classes for them and
        the notation has a word for none of them, so the family is answered
        once, by `_unspelled`.
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
        """The terms of a sum, in the order list's order, and a sum of two
        turned round rather than begun with a minus sign.

        `SQRT(3) - 1` and `SIN(x) - x*COS(x)`, not `-1 + SQRT(3)` and
        `-x*COS(x) + SIN(x)`: the original rearranges a sum to spare it a
        leading minus. It does that to a *pair* only. `-x^2 + 2*x` is written
        `2*x - x^2`, while `-x^2 + 2*x - 1` is written as it stands and keeps
        the minus - three terms are left in the order they were put in, and a
        pair of negated terms has nowhere to turn to.

        A complex number in rectangular form is the pair that does not turn:
        `-2 + 2*#i` keeps its minus, the real part leading whatever its sign.
        """
        terms = _term_order(super()._as_ordered_terms(expr, order), self.context.order)
        if len(terms) == 2 and terms[0].could_extract_minus_sign():
            if not terms[1].could_extract_minus_sign() and not terms[1].has(sp.I):
                return [terms[1], terms[0]]
        return terms

    # -- names --------------------------------------------------------------

    def _print_Symbol(self, expr):
        """A variable, written in ASCII.

        A Greek variable is held under its letter and printed as its name,
        `alpha` for `α`: drawing the letter is the display layer's business,
        and the text has to be something the author line could have been
        typed with.
        """
        return _SUBSCRIPT.join(
            GREEK_GLYPHS.get(part, part) for part in expr.name.split(_SUBSCRIPT)
        )

    def _print_Dummy(self, expr):
        """A sympy-invented variable, written like any other name.

        Sympy marks a `Dummy` apart by writing it with a leading underscore,
        and the author line has no such name: `_k1` does not lex, so a result
        carrying one would come back unreadable. `named` has normally renamed
        these away before printing starts; one printed on its own is written
        the same way.
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
        """A float at the context's precision, written as the notation says.

        Rational notation leaves it as plain digits: an approximation is a
        float here where the original holds a ratio, so there is no ratio to
        write it as. There is no exponent notation to reparse either, so a
        very large or very small float is written out in full, and a whole one
        keeps its point so that it does not read back as an integer.
        """
        text = super()._print_Float(sp.Float(expr, self.context.precision_digits))
        if "e" in text or "E" in text:
            text = format(Decimal(text), "f")
        if self.context.notation is not Notation.RATIONAL:
            return self._number(Fraction(Decimal(text)))
        return text if "." in text else text + "."

    def _print_Mul(self, expr):
        """A product, in the order list's order.

        A ratio the notation does not write as a ratio leads the product it is
        the coefficient of: sympy prices `x/3` as an x over a 3 and prints the
        denominator, which is right while a ratio is written as one, but under
        a decimal style the coefficient is a number like any other and goes in
        front - `x/3` is `0.333333·x`.

        A product sympy was told not to evaluate is written by sympy, which
        keeps every factor exactly where it was put: that is the whole of what
        `Factor` over an integer builds, and `2*3^2*5*3607*3803` is a
        decomposition and not a product to be reordered.
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
        """A product whose denominator is read the way `_print_Pow` reads one.

        Sympy's product spine sends every negatively signed exponent below the
        bar, so `2^(-m)` inside a product comes out `1/2^m`. Alone it does not:
        `_print_Pow` makes a denominator only of a numeric exponent, which is
        the notation's own habit - `1/k^2`, but `2^(-m)`. This is the same
        reading over a whole product, so that one power is written the same way
        wherever it stands.

        A power of `#e` is left alone. Sympy holds `#e^(-z)` as an exponential
        rather than a power, and `#e^(-z)/2` is what the notation writes it as.

        Sign extraction is sympy's, with one addition: a bracket that is
        nothing but negated terms hands its sign to the product, so that a
        coefficient of `-b - 2` is written `- z*(b + 2)` and not
        `z*(-b - 2)`. The order of the factors is the order list's, here and
        in `_beside`.
        """
        sign = ""
        level = precedence(expr)
        if expr.could_extract_minus_sign():
            expr, sign = -expr, "-"
        above, below = [], []
        factors = _factor_order(expr.as_ordered_factors(), self.context.order)
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
                below.append(factor.base if power == 1 else factor.base**power)
            else:
                above.append(factor)

        def written(parts):
            return [self.parenthesize(part, level, strict=False) for part in parts]

        text = "*".join(written(above)) or "1"
        if len(below) == 1:
            text += "/" + written(below)[0]
        elif below:
            # One denominator, as sympy writes it: `a/(x*y)` and not `a/x/y`,
            # which is the same number drawn as a fraction inside a fraction.
            text += "/(" + "*".join(written(below)) + ")"
        return sign + text

    def _beside(self, written: str, rest) -> str:
        """A coefficient already written, times the rest of its product.

        The rest keeps whatever denominator it had, so the coefficient of
        `1/(x - 1)` goes over that rather than beside it. Which factors those
        are is the one thing this takes over from sympy's product spine: a
        factor raised to a negative power is a denominator, exactly as
        `_print_Mul` reads one. Sign extraction is still sympy's.
        """
        above, below = [], []
        for factor in _factor_order(rest.as_ordered_factors(), self.context.order):
            base, power = factor.as_base_exp()
            if power.is_Rational and power.is_negative:
                below.append(base if power == -1 else base**-power)
            else:
                above.append(factor)
        level = PRECEDENCE["Mul"]
        text = written
        for factor in above:
            text += "*" + self.parenthesize(factor, level, strict=False)
        if below:
            # One denominator, as sympy writes it: `a/(x*y)` and not `a/x/y`,
            # which is the same number drawn as a fraction inside a fraction.
            under = [self.parenthesize(factor, level, strict=False) for factor in below]
            text += "/" + (under[0] if len(under) == 1 else "(" + "*".join(under) + ")")
        return text

    def _number(self, value: Fraction) -> str:
        """`value` in the notation `Options Notation` selects.

        Rational notation, and Mixed over a number simple enough for it, write
        the ratio; the numerals of a ratio are written in the session's base,
        as every whole number is. The decimal styles are base ten, the base
        being the one thing a decimal point numeral cannot carry.
        """
        style, digits = self.context.notation, self.context.notation_digits
        if style is Notation.RATIONAL or (
            style is Notation.MIXED and notation.simple(value)
        ):
            base = self.context.input_base
            if value.denominator == 1:
                return numeral(value.numerator, base)
            return f"{numeral(value.numerator, base)}/{numeral(value.denominator, base)}"
        if style is Notation.DECIMAL:
            return notation.decimal(value, digits)
        return notation.scientific(value, digits)

    def parenthesize(self, item, level, strict=False):
        """Fence a number the notation wrote as a power of ten.

        `1.23456*10^8` is a product where sympy prices a number as an atom, so
        nothing else would fence it: `x^123456789` has to come out with the
        exponent in parentheses to read back as itself. A product does not
        need them, `1.23456*10^8*x` associating the way it is meant to.
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
        """Unsigned infinity, which is what `1/0` and `TAN(pi/2)` are worth."""
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
                # A denominator, the way the inherited `_print_Mul` writes one
                # inside a product: `1/k^2` rather than `k^(-2)`.
                return f"1/{self._print(sp.Pow(expr.base, -expr.exp))}"
        base = self.parenthesize(expr.base, level)
        exponent = self.parenthesize(expr.exp, level)
        return f"{base}^{exponent}"

    def _print_MatPow(self, expr):
        """A power of a matrix, written like any other power.

        `SQRT` of a matrix is one of these, and sympy's own spelling for it is
        `m**(1/2)` - which the grammar reads as a product with an empty factor
        in it, so the answer would come back as a different expression.

        A negative exponent stays on the line. There is no `1/u` for a matrix,
        `u^-1` being the notation's whole spelling for an inverse, and it is
        written without parentheses round the exponent: `a^-1`, as the original
        writes one.
        """
        if expr.exp.is_Number and expr.exp.is_negative:
            base = self.parenthesize(expr.base, PRECEDENCE["Pow"])
            return f"{base}^{self._print(expr.exp)}"
        return self._print_Pow(expr)

    def _print_Inverse(self, expr):
        """`a^-1`, which is the same power sympy keeps under its own head."""
        return self._print_MatPow(expr)

    def _print_MatMul(self, expr):
        """A matrix product, as the dot operator section 8.4 writes one with.

        Derive nests a run of them to the right and prints the parentheses:
        `((a . b) . c) . d` comes back `a . (b . (c . d))`, and `a . (b . c)`
        is a fixed point. So a flat product is written that way rather than as
        the chain `a . b . c` that the left-associating grammar would read
        differently.

        A scalar coefficient multiplies the whole run, `-2*(a . b)`, and stands
        in front of it the way it does in any other product. A negated product
        needs no parentheses: a scalar commutes with everything, so `-a . b`
        and `-(a . b)` are the one expression, and the shorter spelling is what
        lets a sum be written `a . b - b . a`.
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
        """A run of matrices as the right-nested chain the original writes."""
        level = PRECEDENCE["Mul"]
        text = self.parenthesize(matrices[-1], level)
        for index in range(len(matrices) - 2, -1, -1):
            if index < len(matrices) - 2:
                text = f"({text})"
            text = f"{self.parenthesize(matrices[index], level)} . {text}"
        return text

    def _print_Transpose(self, expr):
        """``u` ``, the same operator the inert head is written with."""
        return f"{self.parenthesize(expr.arg, PRECEDENCE['Func'])}`"

    def _print_exp(self, expr):
        """`#e^u`, never `EXP(u)`: the constant is how the notation spells it."""
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
        return f"{function_name(expr)}({self.stringify(expr.args, ', ')})"

    def _print_Min(self, expr):
        return f"MIN({self.stringify(expr.args, ', ')})"

    def _print_Max(self, expr):
        return f"MAX({self.stringify(expr.args, ', ')})"

    def _print_floor(self, expr):
        """`FLOOR(m, n)` where the argument is a quotient, `FLOOR(u)` where not.

        The notation's floor takes the numerator and the denominator apart, and
        that is the spelling section 6.7 prints: `MOD(m, n)` is
        `m - n*FLOOR(m, n)`. So a quotient is written back as the pair it was
        authored as, which is what makes `FLOOR(m, n)` a fixed point of
        printing rather than a form the reader is never shown again.

        A number underneath is no such pair. `FLOOR(x/2)` divides by two the
        way any other expression does, and `FLOOR(x, 2)` would be the longer
        way to say it - so the pair is written only where the denominator holds
        a variable, and a numeric factor of that denominator goes back into the
        numerator, `FLOOR(m/n + 1/2)` being `FLOOR(m + n/2, n)` and not
        `FLOOR(2*m + n, 2*n)`.
        """
        numerator, denominator = sp.fraction(sp.together(expr.args[0]))
        if denominator.is_number:
            return f"FLOOR({self._print(expr.args[0])})"
        coefficient, denominator = denominator.as_coeff_Mul()
        over = self._print(denominator)
        return f"FLOOR({self._print(numerator / coefficient)}, {over})"

    def _print_Heaviside(self, expr):
        """`STEP(u)`. The value at zero is ours, not the author's."""
        return f"STEP({self._print(expr.args[0])})"

    def _print_Determinant(self, expr):
        """`DET(u)` over a matrix sympy holds rather than works out.

        A head of its own rather than a `Function`, so the name it is written
        under is this and not `function_name`'s upper-casing.
        """
        return f"DET({self._print(expr.arg)})"

    def _print_Trace(self, expr):
        return f"TRACE({self._print(expr.arg)})"

    def _print_Piecewise(self, expr):
        """A case split, as the nested `IF` the notation has for one.

        Sympy answers a conditional integral or a conditional limit with a
        `Piecewise`, and `IF(test, then, else)` is what Derive calls that. A
        final `true` condition is the else clause; without one the last case
        has none, which is right - outside every condition the value is `?`.
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
        text = self._print(expr.expr)
        for variable, count in expr.variable_count:
            parts = [text, self._print(variable)]
            if count != 1:
                parts.append(self._print(count))
            text = f"DIF({', '.join(parts)})"
        return text

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
        expression, variable, point, direction = expr.args
        parts = [self._print(expression), self._print(variable), self._print(point)]
        side = {"+": "1", "-": "-1"}.get(str(direction))
        if side is not None:
            parts.append(side)
        return f"LIM({', '.join(parts)})"

    # -- a limit that stayed bounded without settling ------------------------

    def _print_AccumulationBounds(self, expr):
        """`INTERVAL(a, b)`: a value known only to lie between `a` and `b`.

        What a bounded limit that does not settle comes to - `LIM(SIN(x), x,
        inf)` is every value in `INTERVAL(-1, 1)` and no one of them. Sympy
        calls it the accumulation bounds; `INTERVAL` is what Mathematica and
        Maple call the same thing, and it was free here because the notation
        writes a solution set as a chained relation rather than as a range.

        A whole answer is the smaller half of what this is for. The value turns
        up inside larger expressions as readily as on its own, `LIM(x*SIN(x),
        x, inf)` being `inf*SIGN(INTERVAL(-1, 1))`, and a call is an atom, so
        the precedence rules bracket it exactly as they do any other head.
        """
        return f"INTERVAL({self.stringify(expr.args, ', ')})"

    # -- the roots of a polynomial that will not factor ----------------------

    def _print_RootSum(self, expr):
        """`ROOT_SUM(p, t, u)`: the sum of `u` over every root `t` of `p`.

        Sympy answers a rational integral this way where the denominator has no
        factors to take it apart into, and the summand it carries is a `Lambda`,
        which the notation has no spelling for. What it does have is the binding
        heads' own shape, the bound variable second - so the summand is written
        in the polynomial's generator and the `Lambda` disappears.
        """
        variable = expr.poly.gen
        parts = (expr.poly.as_expr(), variable, expr.fun(variable))
        return f"ROOT_SUM({self.stringify(parts, ', ')})"

    def _print_ComplexRootOf(self, expr):
        """`ROOT_OF(p, t, n)`: the `n`-th root of `p` in `t`, counted from zero.

        A single root of the kind a `ROOT_SUM` sums over. The generator is not
        among the arguments sympy holds - those are the polynomial and the index
        alone - so it is read off the polynomial, which is where the class
        itself keeps it.
        """
        parts = (expr.poly.as_expr(), expr.poly.gen, sp.Integer(expr.index))
        return f"ROOT_OF({self.stringify(parts, ', ')})"

    # -- vectors and matrices ------------------------------------------------

    def _print_Tuple(self, expr):
        """A tuple, as the vector the notation has for one.

        Sympy heads such as `hyper`, `meijerg` and `Subs` carry tuples of
        arguments, and `(1, m + 1)` is not something the grammar reads: a
        parenthesised list is not an expression. A vector is, so that is what
        it is written as, and the head reads back as an inert call over
        vectors.
        """
        return self._vector(expr.args)

    def _print_Subs(self, expr):
        """`SUBS(u, [x], [a])`, upper-cased like any other head.

        Sympy prints this one itself, in mixed case, and a name that changed
        case on the way out would not be a fixed point.
        """
        return f"SUBS({self.stringify(expr.args, ', ')})"

    def _print_MatrixBase(self, expr):
        """A row is written flat; anything taller is a vector of its rows."""
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
        """Two bounds on one variable, written as the chain `-2 < x < 2`.

        Which is how the original writes a range, and how a range is written on
        the author line: `AND` is for statements that are not one range, and a
        conjunction that is one has a shorter spelling the grammar reads back.
        The two strictnesses are independent, so `-2 < x <= 2` is a chain too.
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
        # `IMP` is the one that is not commutative: which operand implies which
        # is the whole statement, so this one keeps the order it is held in.
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
        """Parenthesised when the operand binds more loosely than `word` does."""
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
        """A head the notation has no word for, written as the string it is.

        Sympy's sets reach here as `Interval(0, 1)` or as the bare word
        `Reals`, and author notation says neither. A bare word is a product of
        its letters to a Character-mode reader, so `EmptySet` comes back as
        `e*m*p*t*y*s*e*t`; `Interval(0, 1)` reads as the built-in `INTERVAL`,
        which is the range a limit keeps to and not a set of reals. Both look
        computed and neither is, which is worse than an answer that says
        plainly it could not be written - and a quoted string is what the
        notation has for text it cannot read.

        Between the quotes is the text the answer would have shown without
        them, author notation wherever a part of it has one. Nothing inside is
        quoted again: the whole head is the one thing that could not be said.
        """
        self._unspelling = True
        try:
            return f'"{super()._print(expr)}"'
        finally:
            self._unspelling = False

    def _print_Power(self, expr):
        base, exponent = expr.args
        level = PRECEDENCE["Pow"]
        # The exponent is not parenthesized at its own level, `^` associating
        # to the right: `10^10^10` reads back as the same tower it prints.
        written = self.parenthesize(base, level, strict=True)
        return f"{written}^{self.parenthesize(exponent, level)}"

    def _print_Transposed(self, expr):
        return f"{self.parenthesize(expr.args[0], PRECEDENCE['Func'])}`"

    def _print_Subscript(self, expr):
        level = PRECEDENCE["Func"]
        base, index = expr.args
        return f"{self.parenthesize(base, level)} SUB {self.parenthesize(index, level)}"

    def _print_Dot(self, expr):
        level = PRECEDENCE["Mul"]
        left, right = expr.args
        return f"{self.parenthesize(left, level)} . {self.parenthesize(right, level)}"

    def _print_InertVector(self, expr):
        return self._vector(expr.args)

    def _print_Logical(self, expr):
        """The operator sympy declined, with its operands as they were written.

        Nothing here has been simplified - that is what the head is for - so
        nothing here is in a normal form either, and the order the operands
        were written in is the only one they have: `3 AND p` is not `p AND 3`.
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

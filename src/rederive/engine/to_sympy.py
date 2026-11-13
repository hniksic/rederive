"""Node -> sympy: a faithful translation, not a simplification.

Every node kind maps to the equivalent sympy object, and nothing happens beyond sympy's
automatic evaluation (`2 + 3` becomes `5` because `Add` does that, not because Simplify
asked). Command-independent semantics live here: constant and function tables,
assumptions from declared domains, angle mode, and special-value overrides such as
`SIGN(0)`. Which transformations to try is up to the calling pipeline.

The mapping is total. Constructs sympy has no head for become inert heads that carry
their operands and print back as written, so anything the engine doesn't understand
survives a round trip.

Labels, assignments and function definitions are handled by `evaluate`, which is this
conversion with an environment threaded through. On its own, `to_sympy` converts labels
and leaves every other name a variable.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from fractions import Fraction
from itertools import product

import sympy as sp
from sympy.concrete.gosper import gosper_term
from sympy.core.function import AppliedUndef
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean, BooleanFunction

from rederive.engine.approximation import GUARD, approximated, simplest
from rederive.engine.context import (
    Angle,
    Context,
    Domain,
    DomainKind,
    Precision,
    domain_of_node,
)
from rederive.engine.integration import integral
from rederive.engine.ordering import main_order
from rederive.model.expr import Kind, Node
from rederive.syntax.names import BINDING_FUNCTIONS, PRIME_MARK
from rederive.syntax.writer import write_expression

__all__ = [
    "CALCULUS_HEADS",
    "COMMAND_HEADS",
    "DIMENSION",
    "Antidifference",
    "Antiquotient",
    "Approx",
    "Assign",
    "Converter",
    "Declare",
    "Added",
    "Dot",
    "FunDef",
    "InertVector",
    "Logical",
    "MovingLimit",
    "PlusMinus",
    "Power",
    "Quotient",
    "StringLiteral",
    "Subscript",
    "Taylor",
    "Transposed",
    "as_condition",
    "authored_conditionals",
    "chosen_from",
    "is_conditional",
    "iterated",
    "steps",
    "to_sympy",
    "vector_of",
]

#: Heads left for a command to evaluate: the in-expression forms of Factor, Expand and
#: soLve, and `RATE`, a bounded soLve of section 6.12's annuity. They need a simplified
#: operand, so evaluation runs them once the operand is a value; an enclosing call then
#: works normally, e.g. `RHS(SOLVE(x^2 - 5*x + 6 = 0, x))` is a vector of roots.
COMMAND_HEADS = ("EXPAND", "FACTOR", "SOLVE", "RATE")


#: Largest number, in bits, a power may build: 16 megabits, about five million decimal
#: digits. Generous so it never fires on legitimate work (`2^1000000` is under it), but
#: far below what makes a machine swap.
POWER_BITS = 1 << 24

#: Largest power, in bits of numerator or denominator, that is approximated when the
#: exact power is too big. Approximate numbers are rationals, so approximating
#: `(1/3)^10^7` would need a denominator of millions of digits, taking seconds.
APPROXIMATE_POWER_BITS = 1 << 22


#: The dimension of every declared nonscalar. A `MatrixSymbol` needs a shape, and a
#: shared square shape is the only way `a . b` can always be built, since declarations
#: say nothing about dimensions. It is a `Dummy` so it can't be confused with the
#: user's `n` (assigning `n := 3` would reshape every matrix) and stays out of
#: `free_symbols`.
DIMENSION = sp.Dummy("n", positive=True, integer=True)


# -- inert heads ------------------------------------------------------------
#
# Constructs sympy has no object for. Sympy carries these heads around without
# rewriting them, and the printer turns them back into the original notation.


class PlusMinus(sp.Function):
    """`+-u`, kept ambiguous.

    Its argument simplifies, but arithmetic doesn't see through it, so `+-1 + 1` stays
    a sum rather than becoming `0` or `2`.
    """

    nargs = 1


class Gcd(sp.Function):
    """`GCD(m, n)` with non-numeric arguments.

    Derive's GCD is a function of integers, so with symbolic arguments it stands rather
    than becoming a polynomial gcd (sympy would give `1` for two different variables,
    while section 6.13 writes `LCM(m, n)` as `|m*n|/GCD(n, m)`). It evaluates once the
    arguments are numbers.
    """

    nargs = 2

    @classmethod
    def eval(cls, *args: sp.Basic) -> sp.Basic | None:
        if all(argument.is_number for argument in args):
            return _gcd_of(list(args))
        return None


class StringLiteral(sp.Symbol):
    """A string literal: data, not mathematics.

    A `Symbol` subclass so sympy treats it as atomic, but a distinct class so `"x"` and
    the variable `x` differ.
    """


class Dot(sp.Function):
    """`u . v` where the shapes will not multiply."""

    nargs = 2


class Added(sp.Function):
    """`u + v` where the shapes can't be added.

    Vectors of different lengths, or a matrix and a vector, have no sum, and sympy
    refuses to build one. Terms of the same shape are added together, and the scalars
    are combined into one leading term.
    """


class Quotient(sp.Function):
    """`u/v` where `v` is a vector or non-square matrix, which has no inverse."""

    nargs = 2


class Subscript(sp.Function):
    """`u SUB v` where `u` isn't a vector and the two aren't both names.

    The common `x SUB 1` becomes a symbol instead, so it can be differentiated with
    respect to and solved for. This head covers everything else.
    """

    nargs = 2


class Transposed(sp.Function):
    """``u` `` where `u` is not a matrix."""

    nargs = 1


class Power(sp.Function):
    """`u^v` whose value is a number too big to build.

    A separate head rather than an unevaluated `Pow`, because `together` and
    `as_numer_denom` rebuild powers from their parts, which would trigger the
    evaluation. See `_explodes` for when one is made.

    It can still be approximated if the result is small enough
    (`APPROXIMATE_POWER_BITS`). This is done in floating point, with extra precision
    equal to the exponent's bit length, since the exponent magnifies the base's error:
    `(1 + 10^-20)^10^20` is `#e` to twenty decimal places.
    """

    nargs = 2

    def _eval_evalf(self, prec: int) -> sp.Float | None:
        base, exponent = self.args
        if not (base.is_Rational and exponent.is_Integer) or base.is_zero:
            return None
        size = abs(int(exponent)) * abs(float(sp.log(abs(base), 2).evalf(15)))
        if size > APPROXIMATE_POWER_BITS:
            return None
        working = prec + int(exponent).bit_length() + 16
        value = sp.Float(base, precision=working) ** exponent
        return sp.Float(value, precision=prec)


class Truth(sp.Function):
    """A logical value or relation used in arithmetic, such as `false` in `x*PRIME(4)`.

    Sympy arithmetic accepts only expressions; a boolean operand is either refused or
    produces a sum or product nothing can take apart. Wrapped, it behaves like an
    unknown (`true + true` is `2*true`) and prints as its value.
    """

    nargs = 1


class InertVector(sp.Function):
    """A vector sympy can't hold as a matrix, typically a ragged one."""


class Logical(sp.Function):
    """A boolean operator sympy declines, such as `3 AND p`.

    The first argument names the operator; the rest are its operands.
    """


class Assign(sp.Function):
    """`name := value`. The value converts; the assignment stays put."""


class FunDef(sp.Function):
    """`F(x, y) := body`, as `FunDef(name, params, body)`."""


class Declare(sp.Function):
    """`x :epsilon Real [0, inf)`, which simplifies to itself."""


# -- a computation held back until a pipeline asks for it --------------------


class Taylor(sp.Function):
    """`TAYLOR(u, x, a, n)`, unevaluated until `.doit()`.

    Sympy has no head for an uncomputed series, so this provides one. It behaves like
    the other calculus heads and prints back as written if it doesn't evaluate.
    """

    nargs = 4

    def doit(self, deep: bool = False, **hints) -> sp.Basic:
        """The Taylor polynomial, or this head if there is none.

        Derive's order is the maximum degree, while sympy's is the exponent the
        expansion stops before, hence the `+ 1`. The order term is dropped, and the
        result must be a polynomial: a series in fractional powers or with logarithms
        isn't a Taylor polynomial.

        Otherwise the definition is used, matching the original where the derivatives
        don't exist: `TAYLOR(SQRT(x), x, 0, 0)` is 0 (just the value at the point), and
        `TAYLOR(SQRT(x), x, 0, 2)` is `?`, since the first derivative at zero is
        infinite (7.3, p.176).

        At a kink or jump, sympy expands from the right only (`ABS(x)` at zero would
        give `x`). Such functions get a Taylor polynomial only if the expansions from
        both sides agree; otherwise the call stands.
        """
        expression, variable, point, order = self.args
        if not (order.is_Integer and order >= 0):
            return self
        try:
            series = expression.series(variable, point, int(order) + 1).removeO()
        except Exception:
            return self
        if expression.has(*_KINKED) and not _from_both_sides(
            expression, variable, point, int(order), series
        ):
            return self
        if series.is_polynomial(variable):
            return series
        summed = _summed_from_derivatives(expression, variable, point, int(order))
        return self if summed is None else summed


#: Functions that are not analytic where their argument crosses a jump or kink.
_KINKED = (
    sp.Abs,
    sp.sign,
    sp.floor,
    sp.ceiling,
    sp.frac,
    sp.Heaviside,
    sp.Max,
    sp.Min,
    sp.Piecewise,
)


def _from_both_sides(
    expression: sp.Basic,
    variable: sp.Basic,
    point: sp.Basic,
    order: int,
    series: sp.Basic,
) -> bool:
    """Whether `series`, the expansion from the right, matches the one from the left.

    `ABS(x)` at zero is `x` from the right and `-x` from the left, so it has no Taylor
    polynomial there. `x*ABS(x)` is 0 from both sides to first order, which is all a
    once-differentiable function promises.
    """
    try:
        left = expression.series(variable, point, order + 1, dir="-").removeO()
        return sp.expand(left - series) == 0
    except Exception:
        return False


def _summed_from_derivatives(
    expression: sp.Basic, variable: sp.Basic, point: sp.Basic, order: int
) -> sp.Basic | None:
    """The Taylor polynomial built term by term from its definition.

    Each term is a derivative at the point over a factorial. An infinite derivative
    makes the whole polynomial undefined (`?`, i.e. `nan`). Returns None if a derivative
    can't be evaluated at the point.
    """
    terms: list[sp.Basic] = []
    derivative = expression
    for degree in range(order + 1):
        try:
            if degree:
                derivative = sp.diff(derivative, variable)
            coefficient = _at_the_point(derivative, variable, point)
        except Exception:
            return None
        if coefficient is None:
            return None
        if coefficient.has(sp.nan, sp.zoo) or coefficient.is_infinite:
            return sp.nan
        terms.append(coefficient * (variable - point) ** degree / sp.factorial(degree))
    return sp.Add(*terms)


def _at_the_point(
    expression: sp.Basic, variable: sp.Basic, point: sp.Basic
) -> sp.Basic | None:
    """The value of `expression` at `point`, or its limit there.

    The limit handles derivatives that are `0/0` at the point, as every derivative of
    `EXP(-1/x^2)` is at zero.
    """
    value = expression.subs(variable, point)
    if not (value.has(sp.nan, sp.zoo) or value.is_infinite):
        return value
    limit = sp.limit(expression, variable, point)
    return None if isinstance(limit, sp.Limit) else limit


class MovingLimit(sp.Function):
    """`LIM(u, x, a)` where `a` mentions `x`: a deferred substitution.

    A limit to a moving point isn't a limit, and the original simply substitutes:
    `LIM((x*v - y)^2, y, x*v - y)` is `y^2`, with all of `y` replaced at once. Sympy
    refuses to build such a limit, so the meaning is supplied here.

    `ODE1.MTH` relies on this: `CLAIRAUT` substitutes `x*v - y` for `y`,
    differentiates, and substitutes back, both via this call; `HOMOGENEOUS` and
    `GEN_HOM` change variables the same way.

    It is a head because the substitution must happen after its operand is evaluated:
    in `LIM(DIF(LIM(p, y, x*v - y), y), y, x*v - y)`, substituting into an unevaluated
    derivative would produce a `SUBS` nothing expands. So it waits like the calculus
    heads, and is evaluated after the derivative beneath it.
    """

    nargs = 3

    def doit(self, deep: bool = False, **hints) -> sp.Basic:
        """The substitution, or this head if it fails."""
        expression, variable, point = self.args
        try:
            return expression.subs(variable, point)
        except Exception:
            return self


class Approx(sp.Function):
    """`APPROX(u, n)`: `u` rounded to `n` digits, deferred until evaluation.

    Rounding must apply to the answer, not the question: `APPROX(INT(u, x))` must see
    the integral's value. So it waits like the calculus heads and is evaluated after
    them, which conversion can't do without a pipeline.

    The digit count is filled in even if the call omitted it, because the caller's
    precision is known only here.
    """

    nargs = 2


class Antidifference(sp.Function):
    """`SUM(u, n)` with no limits, unevaluated until `.doit()`.

    `F` is the antidifference of `f` when `f(n) = F(n + 1) - F(n)`, the discrete
    analogue of an antiderivative. The constant of summation is dropped, as for
    integrals. Sympy has no head for this, as with `Taylor`.
    """

    nargs = 2

    def doit(self, deep: bool = False, **hints) -> sp.Basic:
        """The antidifference, or this head if there is none.

        Gosper's algorithm gives a rational certificate `t` with `t(n)*f(n)` the
        antidifference; no certificate means no hypergeometric antidifference exists
        (e.g. `SUM(1/n^2, n)`). The certificate carries no constant.

        Powers are then combined, so `SUM(1/2^n, n)` comes out as one power rather than
        `-2/2^n`.
        """
        expression, index = self.args
        if type(index) is not sp.Symbol:
            return self
        if expression == 0:
            return sp.S.Zero
        try:
            certificate = gosper_term(expression, index)
        except Exception:
            return self
        if certificate is None:
            return self
        try:
            return sp.powsimp(certificate * expression)
        except Exception:
            return self


class Antiquotient(sp.Function):
    """`PRODUCT(u, n)` with no limits, unevaluated until `.doit()`.

    `F` is the antiquotient of `f` when `f(n) = F(n + 1)/F(n)`, the multiplicative
    analogue of section 7.5's antidifference. The constant factor is dropped.
    """

    nargs = 2

    def doit(self, deep: bool = False, **hints) -> sp.Basic:
        """The antiquotient, or this head if there is none.

        Built from the definition as the product of the body up to `n - 1`, whose
        ratio `F(n + 1)/F(n)` is the body.

        Sympy may answer with a `RisingFactorial` (which Derive can't write) or the
        unevaluated product. `combsimp` converts some of these to factorials;
        otherwise the head stands, as for the manual's `PRODUCT(k^2 + 1, k)`.
        """
        expression, index = self.args
        if type(index) is not sp.Symbol:
            return self
        running = sp.Dummy(index.name)
        try:
            value = sp.Product(
                expression.subs(index, running), (running, 1, index - 1)
            ).doit()
        except Exception:
            return self
        if value.has(sp.RisingFactorial, sp.gamma):
            try:
                value = sp.combsimp(value).rewrite(sp.factorial)
            except Exception:
                return self
        if value.has(sp.Product, sp.Sum, sp.RisingFactorial, sp.gamma):
            return self
        return value


#: Heads representing deferred computations: the calculus heads, and `APPROX`, which
#: must round the value of an inner integral rather than the integral itself.
#: Evaluation works out each one as it is built, innermost first.
CALCULUS_HEADS = (
    sp.Derivative,
    sp.Integral,
    sp.Sum,
    sp.Product,
    sp.Limit,
    Antidifference,
    Antiquotient,
    MovingLimit,
    Taylor,
    Approx,
)


# -- the constant table -----------------------------------------------------

#: `e` and `i` are ordinary variables; only `#e` and `#i` are constants (the parser
#: has already distinguished them).
CONSTANTS: dict[str, sp.Basic] = {
    "pi": sp.pi,
    "#e": sp.E,
    "#i": sp.I,
    "inf": sp.oo,
    "deg": sp.pi / 180,
    "true": sp.true,
    "false": sp.false,
    "euler_gamma": sp.EulerGamma,
    # An arbitrary point on the complex unit circle, as in Derive's answer to
    # `|z| = 2`: `z = 2*unit_circle`. It is a bare symbol without the assumptions of an
    # undeclared variable; in particular it isn't real, so `SQRT(x^2) -> ABS(x)` must
    # not fire on it. Derive's special arithmetic for it (such as `unit_circle*inf`
    # for complex infinity) is not implemented.
    "unit_circle": sp.Symbol("unit_circle"),
}

_RELATIONS: dict[str, Callable[..., sp.Basic]] = {
    "=": sp.Eq,
    "/=": sp.Ne,
    "<": sp.Lt,
    "<=": sp.Le,
    ">": sp.Gt,
    ">=": sp.Ge,
}

#: Bitwise meaning of each boolean operator on integers, in two's complement: `NOT 5`
#: is -6 and `3 OR 5` is 7. Keyed by the operator name a held `Logical` head stores,
#: so it can be retried with both readings once its operands are known.
_BITWISE: dict[str, Callable[..., int]] = {
    "NOT": lambda a: ~a,
    "AND": lambda a, b: a & b,
    "OR": lambda a, b: a | b,
    "XOR": lambda a, b: a ^ b,
    "IMP": lambda a, b: ~a | b,
}

_LOGICAL_NAMES: dict[Kind, str] = {
    Kind.NOT: "NOT",
    Kind.AND: "AND",
    Kind.OR: "OR",
    Kind.XOR: "XOR",
    Kind.IMP: "IMP",
}


def to_sympy(node: Node, context: Context | None = None) -> sp.Basic:
    """Translate `node` into the sympy object that means the same thing."""
    return Converter(context or Context()).convert(node)


class Converter:
    """State for one conversion: the context and the labels being expanded.

    `parameters` are those of the definition whose body is being converted.
    """

    def __init__(self, context: Context, open_labels: frozenset[int] = frozenset()):
        self.context = context
        self.open_labels = open_labels
        self.parameters: frozenset[str] = frozenset()

    # -- the walk -----------------------------------------------------------

    def convert(self, node: Node) -> sp.Basic:
        match node.kind:
            case Kind.NUMBER:
                return self._number(node)
            case Kind.NAME:
                return self._name(node)
            case Kind.STRING:
                return StringLiteral(str(node.value))
            case Kind.LABEL:
                return self._label(node)
            case Kind.UNKNOWN:
                return sp.nan
            case Kind.SUM:
                return self._sum(node)
            case Kind.PRODUCT:
                return self._product(node)
            case Kind.BINOP:
                return self._binop(node)
            case Kind.UNOP:
                return self._unop(node)
            case Kind.POSTOP:
                return self._postop(node)
            case Kind.SUB:
                return self._sub(node)
            case Kind.ABS:
                # Bars and the call share one head, so they treat matrices alike.
                return self.call("ABS", [self.convert(node.children[0])])
            case Kind.CALL | Kind.APPLY:
                return self.call(str(node.children[0].value), self._children(node)[1:])
            case Kind.FUNCPOW:
                return self._funcpow(node)
            case Kind.VECTOR:
                return self._vector(node)
            case Kind.REL:
                return self._relation(node)
            case Kind.NOT | Kind.AND | Kind.OR | Kind.XOR | Kind.IMP:
                return self._logical(node)
            case Kind.ASSIGN:
                return self._assign(node)
            case Kind.FUNDEF:
                return self._fundef(node)
            case Kind.DOMAIN:
                return self._declaration(node)
            case Kind.SHOWVALUE:
                return self.convert(node.children[0])
        # PARAMS and INTERVAL only appear inside their owning nodes; anything else
        # becomes an inert head.
        return self.opaque(str(node.kind).upper(), self._children(node))

    def _children(self, node: Node) -> list[sp.Basic]:
        return [self.convert(child) for child in node.children]

    # -- leaves -------------------------------------------------------------

    def _number(self, node: Node) -> sp.Basic:
        """A numeral, written in `value` as `2.5`, `25` or `1/3`.

        Exact and Mixed read it as an exact rational, so `0.1` is one tenth, not the
        nearest binary float. Approximate rounds that rational to the current
        precision; that input rounding is the difference between the modes, since the
        arithmetic afterwards is exact either way.
        """
        value = sp.Rational(str(node.value))
        if self.context.precision is Precision.APPROXIMATE:
            digits = self.context.precision_digits
            return sp.Rational(simplest(Fraction(value.p, value.q), digits))
        return value

    def _name(self, node: Node) -> sp.Basic:
        name = str(node.value)
        if name == "deg":
            # One degree, expressed in the current angle unit.
            return self.angle_out(CONSTANTS[name])
        constant = CONSTANTS.get(name)
        if constant is not None:
            return constant
        return self.symbol(name)

    def symbol(self, name: str) -> sp.Basic:
        """The symbol for `name`, with its declared assumptions.

        A degenerate interval declares a value: `x :epsilon Real [7, 7]` makes `x` the
        number 7.

        A name declared nonscalar becomes a `MatrixSymbol`, since sympy knows section
        8.8's matrix algebra (transposes and inverses reverse products, the determinant
        of an inverse is a reciprocal) for those but not for merely noncommutative
        symbols.

        This applies only to names declared explicitly. `default :epsilon Nonscalar`
        would make every variable a matrix, including arguments of `SIN`, so the
        default domain contributes its assumptions but not its shape.
        """
        domain = self.context.domain(name)
        value = self._degenerate(domain)
        if value is not None:
            return value
        if domain.kind is DomainKind.NONSCALAR and name in self.context.domains:
            return sp.MatrixSymbol(name, DIMENSION, DIMENSION)
        return sp.Symbol(name, **self.assumptions(domain))

    def assumptions(self, domain: Domain) -> dict[str, bool]:
        """The sympy assumptions implied by a domain declaration.

        Complex means "not known to be real": sympy leaves `real` undecided, so rewrites
        needing a real variable, like `SQRT(x^2)` -> `ABS(x)`, don't fire.
        """
        if domain.kind is DomainKind.NONSCALAR:
            return {"commutative": False}
        if domain.kind is DomainKind.COMPLEX:
            return {"complex": True}
        facts = {"integer": True} if domain.kind is DomainKind.INTEGER else {"real": True}
        low, high = self._bound(domain.low), self._bound(domain.high)
        # An interval implies a sign; the bound is strict if open or away from zero.
        if low is not None and low.is_nonnegative:
            strict = low.is_positive or not domain.closed_low
            facts["positive" if strict else "nonnegative"] = True
        elif high is not None and high.is_nonpositive:
            strict = high.is_negative or not domain.closed_high
            facts["negative" if strict else "nonpositive"] = True
        return facts

    def _degenerate(self, domain: Domain) -> sp.Basic | None:
        """The value of a one-point interval such as `[7, 7]`, or None."""
        if not (domain.closed_low and domain.closed_high):
            return None
        low, high = self._bound(domain.low), self._bound(domain.high)
        if low is None or high is None or low != high:
            return None
        return low

    def _bound(self, node: Node | None) -> sp.Basic | None:
        """The value of an interval bound, or None if it isn't a number.

        Bounds are converted without domains, since domains are being built from them
        and a bound naming a declared variable would recurse.
        """
        if node is None:
            return None
        try:
            value = Converter(replace(self.context, domains={})).convert(node)
        except Exception:
            return None
        return value if getattr(value, "is_number", False) else None

    def _label(self, node: Node) -> sp.Basic:
        """`#3`: the expression labeled 3, or an inert `#3` if unknown.

        A label that refers back to itself, directly or indirectly, stays inert.
        """
        try:
            number = int(str(node.value))
        except ValueError:
            number = -1
        target = self.context.labels.get(number)
        if target is None or number in self.open_labels:
            return sp.Symbol(f"#{node.value}")
        deeper = Converter(self.context, self.open_labels | {number})
        return deeper.convert(target)

    # -- operators ----------------------------------------------------------

    def over_equations(
        self, build: Callable[[list[sp.Basic]], sp.Basic], operands: list[sp.Basic]
    ) -> sp.Basic | None:
        """Apply `build` to both sides of whichever `operands` are equations.

        Per section 4.13 (pp.108-111), arithmetic on an equation applies to both sides:
        `(x^2 + 5*x + 6 = 0) - 6` is `x^2 + 5*x = -6`, `4*(x^2 + 5*x + 6 = 0)` is
        `4*x^2 + 20*x + 24 = 0`, and `EXP(LN(x) = 5)` is `x = #e^5`. Non-equation
        operands appear on both sides, and multiple equations are combined side by side.

        Returns None if no operand is an equation, the usual case.

        Only equalities are mapped, since any operation on both sides preserves an
        equation. Inequalities don't survive everything (multiplying by a negative
        reverses them), so they stand as written.
        """
        if not any(isinstance(operand, sp.Equality) for operand in operands):
            return None
        sides = [
            [
                operand.args[index] if isinstance(operand, sp.Equality) else operand
                for operand in operands
            ]
            for index in (0, 1)
        ]
        return self._related(sp.Eq, build(sides[0]), build(sides[1]))

    def _operands(self, node: Node) -> list[sp.Basic]:
        """An arithmetic operator's operands, with logical ones wrapped in `Truth`."""
        return [
            Truth(operand)
            if _truth_value(operand)
            else operand
            for operand in self._children(node)
        ]

    def _sum(self, node: Node) -> sp.Basic:
        """A sequence of terms, with one operator between each pair."""
        operators = str(node.value)

        def added(terms: list[sp.Basic]) -> sp.Basic:
            signed = [terms[0]]
            for index, term in enumerate(terms[1:]):
                signed.append(-term if operators[index] == "-" else term)
            if len(_added_terms(signed)) > len(signed):
                return _unconformable_sum(signed)
            try:
                return sp.Add(*signed)
            except sp.ShapeError:
                return _unconformable_sum(signed)

        terms = self._operands(node)
        mapped = self.over_equations(added, terms)
        return added(terms) if mapped is None else mapped

    def _product(self, node: Node) -> sp.Basic:
        """A sequence of factors.

        Multiplying two vectors means the dot product when the shapes allow nothing
        else, so a product sympy refuses is folded factor by factor.

        A product containing a declared nonscalar is folded that way from the start,
        since section 8.4 makes `a*b` the matrix product for matrices, built in `_dot`.
        """
        factors = self._operands(node)
        mapped = self.over_equations(self._multiplied, factors)
        return self._multiplied(factors) if mapped is None else mapped

    def _multiplied(self, factors: list[sp.Basic]) -> sp.Basic:
        if not any(_symbolic_matrix(factor) for factor in factors):
            try:
                return sp.Mul(*factors)
            except Exception:
                pass
        result = factors[0]
        for factor in factors[1:]:
            result = self._times(result, factor)
        return result

    def _times(self, left: sp.Basic, right: sp.Basic) -> sp.Basic:
        if _symbolic_matrix(left) or _symbolic_matrix(right):
            return self._dot(left, right)
        try:
            return left * right
        except Exception:
            return self._dot(left, right)

    def _binop(self, node: Node) -> sp.Basic:
        operands = self._operands(node)
        operator = str(node.value)

        def applied(sides: list[sp.Basic]) -> sp.Basic:
            return self._operated(operator, sides[0], sides[1])

        mapped = self.over_equations(applied, operands)
        return applied(operands) if mapped is None else mapped

    def _operated(self, operator: str, left: sp.Basic, right: sp.Basic) -> sp.Basic:
        match operator:
            case "/":
                # Dividing by a matrix means multiplying by its inverse; sympy refuses
                # the division itself.
                if _symbolic_matrix(right):
                    return self._times(left, _matrix_power(right, sp.Integer(-1)))
                if isinstance(right, sp.MatrixBase):
                    if not right.is_square:
                        return Quotient(left, right)
                    return self._times(left, self._operated("^", right, sp.Integer(-1)))
                try:
                    return left / right
                except Exception:
                    return sp.Mul(left, sp.Pow(right, -1, evaluate=False))
            case "^":
                if _explodes(left, right):
                    return Power(left, right)
                if _symbolic_matrix(left):
                    return _matrix_power(left, right)
                if isinstance(left, sp.MatrixBase) and not left.is_square:
                    # Only square matrices have powers; a vector's power isn't
                    # elementwise either, so it stands.
                    return Power(left, right)
                try:
                    return left**right
                except Exception:
                    # An explicit matrix power sympy refuses is a singular matrix's
                    # inverse. Use the same unevaluated head as for symbolic matrices,
                    # which prints as `u^-1`.
                    if isinstance(left, sp.MatrixBase):
                        return sp.MatPow(left, right)
                    return sp.Pow(left, right, evaluate=False)
        return self._dot(left, right)

    def _dot(self, left: sp.Basic, right: sp.Basic) -> sp.Basic:
        """`u . v`: the matrix product, with the dot product as a special case.

        Two flat vectors give a number (`[2, 3] . [4, 5]`), not a 1x1 matrix. Otherwise
        `n` by `m` times `m` by `p` is the matrix product, and nonconforming shapes keep
        the operator unevaluated.

        A flat vector right of a matrix is treated as a column, and the result is flat
        again: each row dotted with the vector. An explicit column like `[[2], [3]]` is
        a matrix and keeps its shape.

        Declared nonscalars are matrices with unknown elements, so their product is
        symbolic, which sympy rewrites by the rules of section 8.8.
        """
        if _symbolic_matrix(left) or _symbolic_matrix(right):
            return _symbolic_product(left, right)
        if isinstance(left, sp.MatrixBase) and isinstance(right, sp.MatrixBase):
            try:
                if left.rows == 1 and right.rows == 1:
                    return left.dot(right)
                if left.cols == right.rows:
                    return left * right
                if right.rows == 1 and left.cols == right.cols:
                    return (left * right.T).T
            except Exception:
                pass
        return Dot(left, right)

    def _unop(self, node: Node) -> sp.Basic:
        operator = str(node.value)

        def applied(sides: list[sp.Basic]) -> sp.Basic:
            return self._prefixed(operator, sides[0])

        operands = self._operands(node)
        mapped = self.over_equations(applied, operands)
        return applied(operands) if mapped is None else mapped

    def _prefixed(self, operator: str, operand: sp.Basic) -> sp.Basic:
        match operator:
            case "-":
                return -operand
            case "+-":
                # `±inf` is unsigned infinity, which is how the printer writes `zoo`.
                if operand in (sp.oo, -sp.oo):
                    return sp.zoo
                return PlusMinus(operand)
        return operand

    def _postop(self, node: Node) -> sp.Basic:
        """`u!`, `u%` and ``u` ``.

        Per section 8.5, the transpose of a scalar is the scalar, so ``x` `` is `x`
        unless `x` is declared nonscalar. Matrices are transposed by sympy. Anything
        else, such as an inert head or a relation, keeps the operator.
        """
        operand = self.convert(node.children[0])
        match str(node.value):
            case "!":
                return sp.factorial(operand)
            case "%":
                return operand / 100
        if isinstance(operand, sp.MatrixBase):
            return operand.T
        if _symbolic_matrix(operand):
            return sp.Transpose(operand).doit()
        collapsed = _transposed_scalar(operand)
        return Transposed(operand) if collapsed is None else collapsed

    def _sub(self, node: Node) -> sp.Basic:
        """Element access on a vector, or a subscripted variable otherwise.

        A vector is either a matrix or an inert vector (for anything deeper). Both are
        indexable, so `SUB` chains across dimensions: `q SUB 2 SUB 1 SUB 2` reads a
        number from a vector of matrices. A non-vector base gives a subscripted name
        like `x SUB 1`.

        A vector index reaches through several dimensions: `m SUB [2, 3]` is
        `m SUB 2 SUB 3`.
        """
        base, index = (self.operand(child) for child in self._children(node))
        if not isinstance(base, (sp.MatrixBase, InertVector)):
            return self._subscript(node.children[0], base, index)
        element = _at(base, index)
        if element is None:
            # Not a usable index yet, so keep the access unevaluated.
            return self.opaque("ELEMENT", [base, index])
        return element

    def _subscript(self, base_node: Node, base: sp.Basic, index: sp.Basic) -> sp.Basic:
        """`x SUB 1` as a single symbol named as written.

        A single symbol can be differentiated with respect to and solved for like any
        variable, and its name prints and reparses as the subscript.

        This requires both parts to be names or plain numerals. Otherwise a
        subexpression would be hidden inside a name, out of reach of substitution, so a
        `Subscript` head is used instead.
        """
        if not (type(base) is sp.Symbol and _nameable(index)):
            return Subscript(base, index)
        domain = (
            self.context.domain(str(base_node.value))
            if base_node.kind is Kind.NAME
            else Domain()
        )
        name = f"{base.name} SUB {index}"
        return sp.Symbol(name, **self.assumptions(domain))

    def _funcpow(self, node: Node) -> sp.Basic:
        """`(SIN^n)(x)`: apply, then raise. The parser has ruled out arcsine."""
        name, exponent, operand = node.children
        applied = self.call(str(name.value), [self.convert(operand)])
        return applied ** self.convert(exponent)

    def _vector(self, node: Node) -> sp.Basic:
        """A vector of rows is a matrix; a flat vector is a row.

        Matrices have only two dimensions, so a vector of matrices stays a vector.
        Stacking `[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]` would merge it into a 2x4 and
        lose a dimension.

        Ragged or mixed vectors stay inert.
        """
        if not node.children:
            return sp.Matrix(0, 0, [])
        return self.vector_over(node.children, self._children(node))

    def vector_over(self, rows: Sequence[Node], elements: list[sp.Basic]) -> sp.Basic:
        """The vector for `rows`, built from their converted values."""
        if all(row.kind is Kind.VECTOR for row in rows):
            widths = {len(row.children) for row in rows}
            if (
                len(widths) == 1
                and widths != {0}
                and _all_matrices(elements)
                and all(element.rows == 1 for element in elements)
            ):
                try:
                    return sp.Matrix([list(row) for row in elements])
                except Exception:
                    return InertVector(*elements)
        if any(row.kind is Kind.VECTOR for row in rows):
            return InertVector(*elements)
        # A vector of relations or booleans is just a container, not a matrix.
        if not all(isinstance(element, sp.Expr) for element in elements):
            return InertVector(*elements)
        try:
            return sp.Matrix(1, len(elements), elements)
        except Exception:
            return InertVector(*elements)

    def _relation(self, node: Node) -> sp.Basic:
        """A relation, built unevaluated; conversion never decides one.

        A chain (left-nested in the grammar) means the conjunction of its links:
        `1 <= a <= b - 1` is `1 <= a AND a <= b - 1`, as the shipped library uses with
        `c <= a <= 0`. Read literally, it would compare a truth value with a number.
        """
        links, _ = self._chain(node)
        if len(links) == 1:
            return links[0]
        try:
            return sp.And(*(_settled(link) for link in links))
        except Exception:
            return Logical(sp.Symbol("AND"), *links)

    def _chain(self, node: Node) -> tuple[list[sp.Basic], sp.Basic]:
        """The links of a relation chain, and the operand it ends on."""
        left_node, right_node = node.children
        relation = _RELATIONS[str(node.value)]
        right = self.convert(right_node)
        if left_node.kind is Kind.REL:
            links, previous = self._chain(left_node)
            return [*links, self._related(relation, previous, right)], right
        return [self._related(relation, self.convert(left_node), right)], right

    def _related(self, relation, left: sp.Basic, right: sp.Basic) -> sp.Basic:
        try:
            return relation(left, right, evaluate=False)
        except Exception:
            return relation(left, right)

    def _logical(self, node: Node) -> sp.Basic:
        """Boolean on booleans, bitwise on integers."""
        return _connected(_LOGICAL_NAMES[node.kind], self._children(node))

    # -- definitions, which convert their value and keep their shape --------

    def _assign(self, node: Node) -> sp.Basic:
        lhs = self.convert(node.children[0])
        rest = [self.convert(child) for child in node.children[1:]]
        return Assign(lhs, sp.Symbol(str(node.value)), *rest)

    def _fundef(self, node: Node) -> sp.Basic:
        params = sp.Tuple(*self._children(node.children[0]))
        outer = self.parameters
        self.parameters = frozenset(str(param) for param in params.args)
        try:
            body = [self.convert(child) for child in node.children[1:]]
        finally:
            self.parameters = outer
        return FunDef(sp.Symbol(str(node.value)), params, *body)

    def binds_parameter(self, variable: sp.Basic) -> bool:
        """Whether a binding call's variable is a parameter of the definition at hand.

        Such a call stays a call in the definition: its variable is whatever a call of
        the function supplies, which the other arguments may mention. `SB(u, v, a) :=
        ITERATE(u, v, a, 1)` isn't `u`, and `G(u, v) := DIF(u, v)` isn't 0.
        """
        if not self.parameters:
            return False
        if isinstance(variable, (sp.MatrixBase, InertVector)):
            variables = list(variable)
        else:
            variables = [variable]
        return any(
            getattr(each, "name", "").split(" SUB ")[0] in self.parameters
            for each in variables
        )

    def _declaration(self, node: Node) -> sp.Basic:
        """A declaration, converted to a `Declare` head describing itself.

        The named symbol carries the declared facts even if the context doesn't have
        them yet, and stays a symbol even if the declaration gives it a value.
        """
        name = str(node.children[0].value)
        declared = domain_of_node(node)
        domain = declared[1] if declared is not None else self.context.domain(name)
        parts: list[sp.Basic] = [
            sp.Symbol(name, **self.assumptions(domain)),
            sp.Symbol(str(node.value)),
        ]
        for child in node.children[1:]:
            if child.kind is Kind.INTERVAL:
                parts.append(sp.Symbol(str(child.value)))
                parts.extend(self._children(child))
        return Declare(*parts)

    # -- calls --------------------------------------------------------------

    def call(self, name: str, args: Sequence[sp.Basic]) -> sp.Basic:
        """A function call via the function tables, or an inert head.

        A call the tables can't handle (wrong argument count, a matrix where a number
        belongs) becomes an inert head rather than a guess.

        Functions in `MAPPED_OVER_EQUATIONS` applied to an equation apply to both sides.
        """
        primed = self._primed(name, args)
        if primed is not None:
            return primed
        if name in MAPPED_OVER_EQUATIONS:
            mapped = self.over_equations(
                lambda sides: self.call(name, sides), list(args)
            )
            if mapped is not None:
                return mapped
        handler = FUNCTIONS.get(name)
        if handler is None:
            return self.opaque(name, args)
        if name in BINDING_FUNCTIONS and len(args) > 1 and self.binds_parameter(args[1]):
            return self.opaque(name, args)
        try:
            result = handler(self, list(args))
        except Exception:
            result = None
        return self.opaque(name, args) if result is None else result

    def _primed(self, name: str, args: Sequence[sp.Basic]) -> sp.Basic | None:
        """`BB'(r)`: a function differentiated once per prime mark.

        The derivative is with respect to the first argument, as `CC'(u, v)` means in
        the original, and is taken before substituting the point: `BB'(r^2)` is `BB'`
        at `r^2`, not the derivative of `BB(r^2)`. If the point is a variable not in
        the other arguments, the plain `DIF(BB(r), r)` suffices and prints back as the
        same prime; otherwise a substitution is needed, since `CC'(u, u)` differs from
        `DIF(CC(u, u), u)`.

        The head is looked up normally, so `SIN'(x)` is the derivative of the sine.
        Only an undefined head stays a derivative of itself, which the printer writes
        with a prime.

        Returns None if there are no marks or the derivative fails; the call then
        becomes an inert head that prints back with the same marks.
        """
        base = name.rstrip(PRIME_MARK)
        order = len(name) - len(base)
        if not order or not base or not args:
            return None
        try:
            return self._derivative_at(base, order, list(args))
        except Exception:
            return None

    def _derivative_at(
        self, base: str, order: int, args: list[sp.Basic]
    ) -> sp.Basic | None:
        point = args[0]
        if isinstance(point, sp.Symbol) and not any(
            argument.has(point) for argument in args[1:]
        ):
            return sp.Derivative(self.call(base, [point, *args[1:]]), (point, order))
        variable = sp.Dummy("xi")
        head = self.call(base, [variable, *args[1:]])
        derivative = sp.Derivative(head, (variable, order))
        if not isinstance(head, AppliedUndef):
            return derivative.doit().subs(variable, point)
        return sp.Subs(derivative, (variable,), (point,))

    def operand(self, value: sp.Basic) -> sp.Basic:
        """`value` as an operand to index into. Overridden by the evaluator."""
        return value

    def opaque(self, name: str, args: Sequence[sp.Basic]) -> sp.Basic:
        """An inert application that prints back as itself."""
        return sp.Function(name)(*args)

    # -- the angle mode, which is a conversion rule and not a printing one ---

    def angle_in(self, value: sp.Basic) -> sp.Basic:
        """A trig argument converted to radians from the user's angle unit."""
        if self.context.angle is Angle.DEGREE:
            return value * sp.pi / 180
        return value

    def angle_out(self, value: sp.Basic) -> sp.Basic:
        """An inverse trig result converted to the user's angle unit."""
        if self.context.angle is Angle.DEGREE:
            return value * 180 / sp.pi
        return value


def _symbolic_product(left: sp.Basic, right: sp.Basic) -> sp.Basic:
    """`u . v` where at least one side is a symbolic matrix.

    Sympy's product handles most of section 8.8: it flattens nested products, so
    `(a . b) . c` and `a . (b . c)` are the same, and reverses products under
    transposition and inversion.

    Unlike Derive, sympy cancels a matrix against its inverse. The original returns
    `a . a^-1` unchanged, since an identity of unknown dimension can't be written. So
    where factors would cancel, the product is built unevaluated.
    """
    factors = [*_matrix_factors(left), *_matrix_factors(right)]
    try:
        if _cancelling(factors):
            return sp.MatMul(*factors)
        return left * right
    except Exception:
        return Dot(left, right)


def _transposed_scalar(value: sp.Basic) -> sp.Basic | None:
    """`value` if it is its own transpose, or None if that isn't settled.

    Per section 8.5, a scalar's transpose is the scalar, and sympy decides what is
    scalar. If sympy returns an unevaluated transpose, as for an inert head like
    `VECTOR(u, i, 1, n)``, the operator is kept.
    """
    if not isinstance(value, sp.Expr):
        return None
    try:
        transposed = sp.Transpose(value).doit()
    except Exception:
        return None
    return None if isinstance(transposed, sp.Transpose) else transposed


def _matrix_power(base: sp.Basic, exponent: sp.Basic) -> sp.Basic:
    """`u^n` where `u` is a symbolic matrix.

    A zero exponent would give an identity of unknown dimension, which can't be
    written, so the power is kept as authored. Everything else is left to sympy, which
    turns `(a . b)^-1` into `b^-1 . a^-1`.
    """
    if exponent == 0:
        return sp.MatPow(base, exponent)
    try:
        return base**exponent
    except Exception:
        return sp.MatPow(base, exponent)


def _truth_value(operand: sp.Basic) -> bool:
    """Whether `operand` is a logical value or relation rather than an expression.

    Sympy symbols are both, and count as expressions here. Equations are excluded, since
    arithmetic on them is handled by `over_equations`.
    """
    return (
        isinstance(operand, Boolean)
        and not isinstance(operand, (sp.Expr, sp.Equality))
    )


def _unconformable_sum(terms: list[sp.Basic]) -> sp.Basic:
    """The sum of `terms` where some are matrices of different shapes.

    Matrices of each shape are added together, as are the scalars, and the resulting
    terms are kept in order of first appearance.
    """
    scalars: list[sp.Basic] = []
    shapes: dict[tuple[int, int], list[sp.Basic]] = {}
    for term in _added_terms(terms):
        if isinstance(term, sp.MatrixBase):
            shapes.setdefault(term.shape, []).append(term)
        else:
            scalars.append(term)
    added = [sp.Add(*group) for group in shapes.values()]
    if len(added) < 2:
        return sp.Add(*scalars, *added)
    scalar = sp.Add(*scalars)
    return Added(*added) if scalar == 0 else Added(scalar, *added)


def _added_terms(terms: list[sp.Basic]) -> list[sp.Basic]:
    """`terms` with the terms of any nested unconformable sum spliced in."""
    spliced = []
    for term in terms:
        if isinstance(term, Added):
            spliced.extend(term.args)
        elif (
            isinstance(term, sp.Mul)
            and len(term.args) == 2
            and term.args[0].is_Number
            and isinstance(term.args[1], Added)
        ):
            spliced.extend(term.args[0] * part for part in term.args[1].args)
        else:
            spliced.append(term)
    return spliced


def _matrix_factors(value: sp.Basic) -> list[sp.Basic]:
    """The factors of a matrix product, for flattening."""
    return list(value.args) if isinstance(value, sp.MatMul) else [value]


def _symbolic_matrix(value: sp.Basic) -> bool:
    """Whether `value` is a symbolic matrix rather than one with explicit elements.

    An explicit matrix is a `MatrixBase` (and, when immutable, also a `MatrixExpr`), and
    is computed with directly. Symbolic ones are declared nonscalars and expressions
    built from them.
    """
    return isinstance(value, sp.MatrixExpr) and not isinstance(value, sp.MatrixBase)


def _cancelling(factors: Sequence[sp.Basic]) -> bool:
    """Whether these factors include a matrix and its inverse.

    Checked by summing exponents rather than adjacency, since sympy also cancels
    `a . b . a^-1` after reordering what commutes.
    """
    exponents: dict[sp.Basic, sp.Basic] = {}
    for factor in factors:
        base, exponent = (
            factor.args if isinstance(factor, sp.MatPow) else (factor, sp.S.One)
        )
        if not _symbolic_matrix(base):
            continue
        exponents[base] = exponents.get(base, sp.S.Zero) + exponent
    return any(total == 0 for total in exponents.values())


def _explodes(base: sp.Basic, exponent: sp.Basic) -> bool:
    """Whether `base^exponent` is a number too big to build.

    Sympy evaluates `Integer**Integer` on construction, so this must be checked first:
    `10^10^10` would be ten billion digits, and an OOM kill would take the worksheet
    with it. As with anything Derive can't do, the expression comes back as written,
    an inert power that prints and saves as `10^10^10`.

    Only exact rationals are checked; float powers cost only the working precision, and
    symbolic ones aren't built as numbers.
    """
    # Check by class rather than assumptions, since the base may be a matrix.
    if not (isinstance(exponent, sp.Integer) and isinstance(base, sp.Rational)):
        return False
    if base.is_zero:
        return False
    bits = max(abs(base.p).bit_length(), abs(base.q).bit_length())
    return bits * abs(int(exponent)) > POWER_BITS


def _nameable(value: sp.Basic) -> bool:
    """Whether a subscript index can be part of a symbol's name.

    Names and nonnegative numerals can; a negative numeral can't, since the grammar
    doesn't read back `x SUB -1`.
    """
    if type(value) is sp.Symbol:
        return True
    return isinstance(value, sp.Integer) and value >= 0


def _all_matrices(elements: list) -> bool:
    return all(isinstance(element, sp.MatrixBase) for element in elements)


def _element(matrix: sp.MatrixBase, index: sp.Basic) -> sp.Basic | None:
    """`v SUB n`, counting from 1, or None if `n` isn't a valid index."""
    if not isinstance(index, sp.Integer) or index < 1:
        return None
    number = int(index)
    if matrix.rows == 1:
        return matrix[0, number - 1] if number <= matrix.cols else None
    return matrix[number - 1, :] if number <= matrix.rows else None


# -- the function table -----------------------------------------------------

Handler = Callable[[Converter, list], sp.Basic]


def _one(args: list) -> sp.Basic:
    (value,) = args
    return value


def _direct(func: Callable[..., sp.Basic]) -> Handler:
    """A head that maps directly to a sympy function.

    Symbolic matrix arguments are refused: sympy would expand `ABS(a)` and `RE(a)` into
    their definitions in terms of `a . CONJ(a)` and the conjugate, which is no
    simplification and uses heads Derive can't write. Such calls come back as written.
    """

    def handler(conv: Converter, args: list) -> sp.Basic:
        if any(_symbolic_matrix(argument) for argument in args):
            raise TypeError("not a scalar")
        return func(*args)

    return handler


def _trig(func: Callable[..., sp.Basic]) -> Handler:
    """A trigonometric head, with its argument in the current angle unit.

    Special angles that sympy writes differently from the original are computed here;
    everything else is left to sympy.
    """

    def handler(conv: Converter, args: list) -> sp.Basic:
        argument = conv.angle_in(_one(args))
        halved = _halved_angle(func, argument)
        return _within_an_octant(func(argument)) if halved is None else halved

    return handler


def _within_an_octant(value: sp.Basic) -> sp.Basic:
    """A sine or cosine of a rational multiple of pi, at an angle of at most `pi/4`.

    Sympy brings the angle into the first quadrant; past `pi/4` it is the cofunction
    of the complement, as Derive writes it: `COS(204285811*pi/536870912)` is
    `SIN(64149645*pi/536870912)`, and `SIN(3*pi/7)` is `COS(pi/14)`.
    """
    sign, call = (-1, -value) if value.could_extract_minus_sign() else (1, value)
    cofunction = {sp.sin: sp.cos, sp.cos: sp.sin}.get(type(call))
    if cofunction is None:
        return value
    turns = call.args[0] / sp.pi
    if not turns.is_Rational or not sp.Rational(1, 4) < turns < sp.Rational(1, 2):
        return value
    return sign * cofunction(sp.pi * (sp.Rational(1, 2) - turns))


#: Odd parts of `n` for which `pi/n` has trigonometric values in radicals (section
#: 6.3 p.104): `n` must be a power of two times 1, 3, 5 or 15. Others, like `pi/7` or
#: `pi/56`, stay as written.
_ALGEBRAIC_PARTS = frozenset({1, 3, 5, 15})

#: The largest power of two in a denominator whose sine and cosine are written in
#: radicals by `_halved_angle`.
_MOST_HALVED = 64


def _halved_angle(func: Callable[..., sp.Basic], argument: sp.Basic) -> sp.Basic | None:
    """The sine or cosine of `argument` computed by the half-angle formula, or None.

    The original uses the half-angle formula for denominators with more than two factors
    of two, writing nested roots: `SIN(pi/24)` is `SQRT(-SQRT(2)/8 - SQRT(6)/8 + 1/2)`.
    Sympy uses an angle difference instead (a sum of two radicals), and has no value for
    `SIN(pi/16)` at all.

    Only applies with at least three factors of two. For denominators `m`, `2*m` or
    `4*m`, sympy already has flat values like `SQRT(2)/2` or `SQRT(6)/4 - SQRT(2)/4`,
    and halving would introduce nesting neither writes.

    Each further factor of two nests one more root, so only denominators with at most
    `_MOST_HALVED` as their power of two are written in radicals; beyond it the value
    stays a call. RANDOM_NORMAL in MISC.MTH takes the cosine of `2*pi*RANDOM(1)`,
    a multiple of `pi/2^29`, which written out would be 27 roots deep.

    The sign of the root comes from where the angle falls in the period: sine is
    positive between 0 and pi (mod 2*pi), and cosine is the sine a quarter turn on. The
    value is never zero, which would need a denominator of 1 or 2.
    """
    if func not in (sp.sin, sp.cos):
        return None
    turns = argument / sp.pi
    if not turns.is_Rational or turns.q % 8:
        return None
    twos = turns.q & -turns.q
    if turns.q // twos not in _ALGEBRAIC_PARTS or twos > _MOST_HALVED:
        return None
    doubled = 2 * argument
    cosine = _halved_angle(sp.cos, doubled)
    if cosine is None:
        cosine = sp.cos(doubled)
    whole = (1 - cosine) if func is sp.sin else (1 + cosine)
    root = sp.sqrt(whole / 2)
    place = (turns if func is sp.sin else turns + sp.Rational(1, 2)) % 2
    return -root if place > 1 else root


def _arctrig(func: Callable[..., sp.Basic]) -> Handler:
    return lambda conv, args: conv.angle_out(func(_one(args)))


def _gcd_of(values: list[sp.Basic]) -> sp.Basic:
    """The greatest common divisor of `values`, unevaluated for non-numbers.

    The numbers are combined into one divisor, and zero contributes nothing:
    `GCD(6, 4, m)` is `GCD(2, m)` and `GCD(m, 0)` is `ABS(m)`. The rest are nested
    pairwise from the left.
    """
    numbers = [value for value in values if value.is_number]
    others = [value for value in values if not value.is_number]
    divisor = sp.S.Zero
    for number in numbers:
        divisor = sp.gcd(divisor, number)
    if not others:
        return abs(divisor)
    standing = ([divisor] if divisor else []) + others
    if len(standing) == 1:
        return sp.Abs(standing[0])
    result = standing[0]
    for value in standing[1:]:
        result = Gcd(result, value)
    return result


def _lcm_of(values: list[sp.Basic]) -> sp.Basic:
    """The least common multiple of `values`, combined pairwise from the left.

    For non-numbers, `LCM(m, n)` is `|m*n|/GCD(n, m)`.
    """
    result = values[0] if len(values) > 1 else sp.Abs(values[0])
    for value in values[1:]:
        if result.is_number and value.is_number:
            result = sp.lcm(result, value)
        else:
            result = sp.Abs(result * value) / _gcd_of([result, value])
    return result


_TRIGONOMETRIC = {
    "SIN": sp.sin,
    "COS": sp.cos,
    "TAN": sp.tan,
    "COT": sp.cot,
    "SEC": sp.sec,
    "CSC": sp.csc,
}

_INVERSE_TRIGONOMETRIC = {
    "ASIN": sp.asin,
    "ACOS": sp.acos,
    "ATAN": sp.atan,
    "ACOT": sp.acot,
    "ASEC": sp.asec,
    "ACSC": sp.acsc,
}

def _ln(value: sp.Basic) -> sp.Basic:
    """`LN(z)`, which is `-inf` at zero.

    That is the limit from the positive side, and the real part `LN(ABS(z))` tends to it
    from every direction. Sympy gives complex infinity, which would allow `+inf` too.
    """
    if getattr(value, "is_zero", None) is True:
        return -sp.oo
    return sp.log(value)


#: Functions that map directly to a sympy function.
_DIRECT: dict[str, Callable[..., sp.Basic]] = {
    "SQRT": sp.sqrt,
    "EXP": sp.exp,
    "LN": _ln,
    "ABS": sp.Abs,
    "MAX": sp.Max,
    "MIN": sp.Min,
    "GAMMA": sp.gamma,
    "COMB": sp.binomial,
    "PERM": sp.ff,
    "RE": sp.re,
    "IM": sp.im,
    "CONJ": sp.conjugate,
    "ERFC": sp.erfc,
    "ZETA": sp.zeta,
    "SINH": sp.sinh,
    "COSH": sp.cosh,
    "TANH": sp.tanh,
    "COTH": sp.coth,
    "SECH": sp.sech,
    "CSCH": sp.csch,
    "ASINH": sp.asinh,
    "ACOSH": sp.acosh,
    "ATANH": sp.atanh,
    "ACOTH": sp.acoth,
    "ASECH": sp.asech,
    "ACSCH": sp.acsch,
    # The hyperbolic cosine and sine integrals, which Derive has no names for.
    # `CHI` (upper-cased `Chi`) is already taken, so these are `COSH_INT` and the
    # matching `SINH_INT`.
    "COSH_INT": sp.Chi,
    "SINH_INT": sp.Shi,
    "EI": sp.Ei,
    "SI": sp.Si,
    "CI": sp.Ci,
    "ERFI": sp.erfi,
    # Sympy represents digamma as polygamma of order zero; the printer writes it back
    # as `DIGAMMA`.
    "DIGAMMA": sp.digamma,
    "POLYGAMMA": sp.polygamma,
    "POLYLOG": sp.polylog,
    "LAMBERTW": sp.LambertW,
}


def _sign(conv: Converter, args: list) -> sp.Basic:
    """`SIGN(0)` is `+-1`, overriding sympy's `0`.

    Section 6.8 defines the sign of a complex number as `z/|z|`, so `SIGN(3 + 4*#i)` is
    `3/5 + 4*#i/5`. Sympy only does this for pure imaginaries, so other complex numbers
    are divided by their magnitude here. Symbolic arguments stay unevaluated.
    """
    value = _one(args)
    if value.is_zero:
        return PlusMinus(sp.Integer(1))
    if value.is_number and value.is_extended_real is False:
        magnitude = sp.Abs(value)
        # An unevaluated magnitude is no improvement over `SIGN`.
        if not magnitude.has(sp.Abs):
            return value / magnitude
    return sp.sign(value)


def _step(conv: Converter, args: list) -> sp.Basic:
    """`STEP(x)` is `SIGN(x)/2 + 1/2`.

    The 6.1 help gives this when the sign of `x` is unknown, and the sign of zero is
    indeterminate: `STEP(0)` is `SIGN(0)/2 + 1/2`, undecided at the jump, just as `CHI`
    is at the edge of its interval.
    """
    value = _one(args)
    if value.is_zero:
        return (_sign(conv, [value]) + 1) / 2
    return sp.Heaviside(value, sp.Rational(1, 2))


def _chi(conv: Converter, args: list) -> sp.Basic:
    """`CHI(a, x, b)`: the indicator of the interval between `a` and `b`.

    Computed as `SIGN(x - a)/2 - SIGN(x - b)/2`, which the manual gives for when the
    comparisons can't be made, and which the original uses in every case. The signs do
    the work: 1 inside, 0 outside, and -1 for a reversed interval, matching the manual's
    `CHI(a, x, b) = -CHI(b, x, a)`.

    At the endpoints, `SIGN(0)` is `±1`, so `CHI(0, 0, 1)` is `±1/2 + 1/2`, undecided
    exactly on the edge.

    `a` defaults to 0 and `b` to 1, so `CHI(x)` indicates the unit interval.
    """
    if len(args) == 1:
        low, value, high = sp.Integer(0), args[0], sp.Integer(1)
    elif len(args) == 2:
        low, value, high = args[0], args[1], sp.Integer(1)
    else:
        low, value, high = args
    return (_sign(conv, [value - low]) - _sign(conv, [value - high])) / 2


def _log(conv: Converter, args: list) -> sp.Basic:
    """`LOG(z)` is the natural logarithm; `LOG(z, w)` is `LN(z)/LN(w)`."""
    if len(args) == 2 and getattr(args[0], "is_zero", None) is True:
        return _ln(args[0]) / _ln(args[1])
    if len(args) == 2:
        return sp.log(*args)
    return _ln(*args)


def _normal(conv: Converter, args: list) -> sp.Basic:
    """`NORMAL(z, m, s)`: the normal distribution with mean `m` and deviation `s`.

    The statistical function, unrelated to normal form. The manual defines it as
    `(ERF(SQRT(2)*z/(2*s) - SQRT(2)*m/(2*s)) + 1)/2`. With the defaults `m = 0` and
    `s = 1`, `NORMAL(z)` is the standard normal cumulative distribution.
    """
    value, *rest = args
    mean = rest[0] if rest else sp.Integer(0)
    deviation = rest[1] if len(rest) > 1 else sp.Integer(1)
    return (sp.erf((value - mean) / (deviation * sp.sqrt(2))) + 1) / 2


# -- the annuity of section 6.12 ---------------------------------------------
#
# The financial functions describe the contract
# `v*(1 + i)^n + p*(1 + i*t)*((1 + i)^n - 1)/i + f = 0`: a present value `v` earning a
# periodic rate `i` over `n` periods with a fixed payment `p`, ending at a future
# value `f`, where `t` says when in the period payments fall. Each function solves
# the equation for one variable.
#
# All of them divide by the rate, so they are undefined at a zero rate:
# `PMT(0, 10, 1000)` is `?`, and the original takes no limit.
#
# Four of the five are solved here. The rate can't be isolated, so the original
# searches for it; `RATE` therefore only fills in its arguments here and is answered
# by the pipeline, which has a solver.


def annuity(
    rate: sp.Basic,
    periods: sp.Basic,
    payment: sp.Basic,
    present: sp.Basic,
    future: sp.Basic,
    when: sp.Basic,
) -> sp.Basic:
    """Section 6.12's contract, which is zero when its six terms are consistent."""
    return (
        present * (1 + rate) ** periods
        + payment * (1 + rate * when) * ((1 + rate) ** periods - 1) / rate
        + future
    )


def _financial(args: list, count: int) -> list:
    """The arguments of a financial function, with optional ones filled in.

    The two optional values default to zero, as does the payment time (end of period).
    Too few or too many arguments raise, leaving the call inert.
    """
    if not 2 <= len(args) <= count:
        raise TypeError("not a financial contract")
    return args + [sp.Integer(0)] * (count - len(args))


def _present_value(conv: Converter, args: list) -> sp.Basic:
    """`PVAL(i, n, p, f, t)`: the contract's present value."""
    rate, periods, payment, future, when = _financial(args, 5)
    due = payment * (rate * when + 1)
    return ((due - future * rate) / (1 + rate) ** periods - due) / rate


def _future_value(conv: Converter, args: list) -> sp.Basic:
    """`FVAL(i, n, p, v, t)`: the contract's future value."""
    rate, periods, payment, present, when = _financial(args, 5)
    due = payment * (rate * when + 1)
    grown = (1 + rate) ** periods * (rate * (payment * when + present) + payment)
    return (due - grown) / rate


def _payment(conv: Converter, args: list) -> sp.Basic:
    """`PMT(i, n, v, f, t)`: the payment per period."""
    rate, periods, present, future, when = _financial(args, 5)
    growth = (1 + rate) ** periods
    return rate * (present * growth + future) / ((1 - growth) * (rate * when + 1))


def _periods(conv: Converter, args: list) -> sp.Basic:
    """`NPER(i, p, v, f, t)`: the number of periods."""
    rate, payment, present, future, when = _financial(args, 5)
    due = payment * (rate * when + 1)
    return sp.log(
        (due - future * rate) / (rate * (payment * when + present) + payment)
    ) / sp.log(1 + rate)


#: The generator `RANDOM` draws from, separate from the module's so seeding it doesn't
#: affect anything else in the process.
_GENERATOR = random.Random()

#: The program start time, from which `RANDOM(0)` measures, and the number of outcomes
#: a number in [0, 1) is drawn from. The original draws a rational (`RANDOM(1)` gives
#: 242469/1540430), and a fraction over a power of two is likewise a rational.
_STARTED = time.monotonic()
_OUTCOMES = 2**32


def _random(conv: Converter, args: list) -> sp.Basic:
    """`RANDOM(n)`: a pseudo-random number, or a new seed.

    Section 6.13 gives four cases by the integer `n`: above 1, an integer in [0, n); at
    1, a number in [0, 1); below 0, seed with `-n` and return it; at 0, seed from the
    clock and return the reading. A non-integer like `RANDOM(6.5)` stays as written.

    Unlike the original, there is one draw per written call: a generated vector
    substitutes into an already converted body, so `VECTOR(RANDOM(6), n, 1, 100)`
    repeats one throw a hundred times. Drawing later instead would break the other
    case, since two dice in one sum would be one identical inert call to sympy.
    """
    number = _one(args)
    if not number.is_Integer:
        raise TypeError("not a whole number of outcomes")
    outcomes = int(number)
    if outcomes > 1:
        return sp.Integer(_GENERATOR.randrange(outcomes))
    if outcomes == 1:
        return sp.Rational(_GENERATOR.randrange(_OUTCOMES), _OUTCOMES)
    if outcomes < 0:
        _GENERATOR.seed(-outcomes)
        return sp.Integer(-outcomes)
    centiseconds = int((time.monotonic() - _STARTED) * 100)
    _GENERATOR.seed(centiseconds)
    return sp.Integer(centiseconds)


def _rate(conv: Converter, args: list) -> sp.Basic:
    """`RATE(n, p, v, f, t)`: the contract's rate, with search bounds filled in.

    The original makes the search visible: this simplifies to
    `RATE(n, p, v, f, t, 0, 1)`, where the last two arguments bound the search, and a
    negative rate or one above 100% needs explicit bounds. Only the filling in happens
    here, and it is idempotent.
    """
    if len(args) == 6 or len(args) > 7:
        raise TypeError("not a financial contract")
    bounds = args[5:7] or [sp.Integer(0), sp.Integer(1)]
    return conv.opaque("RATE", _financial(args[:5], 5) + bounds)


def _atan(conv: Converter, args: list) -> sp.Basic:
    if len(args) == 2:
        return conv.angle_out(sp.atan2(args[0], args[1]))
    return conv.angle_out(sp.atan(_one(args)))


def _acot(conv: Converter, args: list) -> sp.Basic:
    """`ACOT(x, y)` is `ATAN(y, x)`, per section 6.4.

    The two arguments are the legs of the triangle rather than their ratio, which
    preserves the quadrant.
    """
    if len(args) == 2:
        return conv.angle_out(sp.atan2(args[1], args[0]))
    return conv.angle_out(sp.acot(_one(args)))


def _phase(conv: Converter, args: list) -> sp.Basic:
    """`PHASE(z)`, an angle in the user's angle unit."""
    return conv.angle_out(_direct(sp.arg)(conv, args))


def _erf(conv: Converter, args: list) -> sp.Basic:
    """`ERF(z, w)` is `ERF(w) - ERF(z)`; with one argument, the ordinary error
    function."""
    if len(args) == 2:
        return sp.erf(args[1]) - sp.erf(args[0])
    return sp.erf(_one(args))


def _absolute(conv: Converter, args: list) -> sp.Basic:
    """`ABS(z)`, and the magnitude `ABS(v)` of a vector.

    Section 8.2 defines a vector's absolute value as the square root of the sum of the
    squares of its elements. Applied to a matrix as a vector of rows, this squares
    every element once.
    """
    value = _one(args)
    if _symbolic_matrix(value):
        raise TypeError("not a scalar")
    if not isinstance(value, sp.MatrixBase):
        return sp.Abs(value)
    squares = [_absolute(conv, [element]) ** 2 for element in _elements_of(value)]
    return sp.sqrt(sp.Add(*squares))


def _floor(conv: Converter, args: list) -> sp.Basic:
    """`FLOOR(m, n)` is the floor of `m/n`; `n` defaults to 1.

    Vectors are floored element-wise, as NUMBER.MTH's
    `CONTINUED_FRACTION(u, n) := FLOOR(ITERATES(1/MOD(x_), x_, u, n))` requires.
    """
    numerator, denominator = _with_unit_default(args)
    if isinstance(numerator, sp.MatrixBase):
        return numerator.applyfunc(lambda element: sp.floor(element / denominator))
    return sp.floor(numerator / denominator)


def _mod(conv: Converter, args: list) -> sp.Basic:
    """`MOD(m, n)` is `m - n FLOOR(m, n)`; `MOD(m, 0)` is `m`."""
    value, modulus = _with_unit_default(args)
    if modulus.is_zero:
        return value
    return value - modulus * sp.floor(value / modulus)


def _mods(conv: Converter, args: list) -> sp.Basic:
    """`MODS(m, n)` lands in [-n/2, n/2): `m - n FLOOR(m + n/2, n)`."""
    value, modulus = _with_unit_default(args)
    if modulus.is_zero:
        return value
    return value - modulus * sp.floor((value + modulus / 2) / modulus)


def _with_unit_default(args: list) -> tuple[sp.Basic, sp.Basic]:
    if len(args) == 1:
        return args[0], sp.Integer(1)
    first, second = args
    return first, second


def _statistic(over: Callable[[list], sp.Basic]) -> Handler:
    """A statistic of a sample, given in any of three forms.

    The sample is the arguments, the elements of a single vector, or, for a single
    matrix, each of its rows (giving a vector of answers).

    A single argument that is still an unevaluated call isn't treated as a sample yet:
    the utility files write `RMS(VECTOR(...))`, where the vector is generated only when
    the definition is applied, and treating it as a sample of one would compute `ABS`
    of an unbuilt vector. Any other single argument is a sample of one.
    """

    def handler(conv: Converter, args: list) -> sp.Basic:
        if len(args) == 1:
            (value,) = args
            if isinstance(value, AppliedUndef):
                raise TypeError("no sample yet")
            if isinstance(value, sp.MatrixBase):
                if value.rows > 1:
                    rows = [list(value[row, :]) for row in range(value.rows)]
                    return vector_of([over(row) for row in rows])
                values = list(value)
            else:
                values = args
        else:
            values = args
        if not values:
            raise ValueError("no sample")
        return over(values)

    return handler


def _average(values: list) -> sp.Basic:
    """The arithmetic mean."""
    return sp.Add(*values) / len(values)


def _root_mean_square(values: list) -> sp.Basic:
    return sp.sqrt(sp.Add(*(value**2 for value in values)) / len(values))


def _variance(values: list) -> sp.Basic:
    """The unbiased sample variance, as the manual specifies.

    Divided by `n - 1` rather than `n`, since deviations from the sample mean are
    smaller than from the true mean by one degree of freedom. For a sample of one,
    Derive gives 0 rather than `0/0`.

    The result is factored to recombine the `1/n` factors from the mean: `VAR(x, y)` is
    `(x - y)^2/2`, as in Derive, not three expanded terms.
    """
    if len(values) == 1:
        return sp.Integer(0)
    mean = _average(values)
    deviations = sp.Add(*((value - mean) ** 2 for value in values))
    return sp.factor(deviations / (len(values) - 1))


def _standard_deviation(values: list) -> sp.Basic:
    """The square root of the sample variance."""
    return sp.sqrt(_variance(values))


def _fit(conv: Converter, args: list) -> sp.Basic:
    """`FIT(v, A)`: the least squares fit of a parameterized expression to data.

    The label vector `v` lists the data variables followed by the expression. Each row
    of the data matrix `A` gives values for the data variables and then the target
    value. The expression's remaining variables are the parameters solved for.

    The expression must be linear in the parameters, but not necessarily in the data
    variables, so `q*ATAN(t) + r*SIN(t)` works. Substituting a row's data gives an
    expression in the parameters whose derivatives are their coefficients; a coefficient
    that still contains a parameter means nonlinearity, and the call stands. Each row
    becomes one equation of a linear system solved by least squares, exactly when rows
    equal parameters.

    In Approximate mode the data and the system are evaluated numerically first, as
    for `ROW_REDUCE`: data written as `TAN(0.1)` would otherwise be solved for exactly,
    in a field of sixteen tangents.

    Degenerate cases raise, leaving the call inert: no parameters, rows of the wrong
    width, too few rows, or a rank-deficient design matrix.
    """
    labels, data = args
    labels = _matrix(labels)
    if labels.rows != 1 or labels.cols < 2:
        raise TypeError("not a label vector")
    *variables, expression = list(labels)
    if any(type(variable) is not sp.Symbol for variable in variables):
        raise TypeError("not data variables")
    if len(set(variables)) != len(variables):
        raise ValueError("a data variable twice")
    data = _worked_out(_matrix(data), conv.context)
    if data.cols != len(variables) + 1:
        raise ValueError("data of the wrong width")
    parameters = sorted(
        expression.free_symbols - set(variables), key=sp.default_sort_key
    )
    if not parameters:
        raise ValueError("no parametric variables")
    design, targets = [], []
    for index in range(data.rows):
        row = list(data[index, :])
        here = sp.expand(expression.subs(dict(zip(variables, row))))
        coefficients = [sp.diff(here, parameter) for parameter in parameters]
        if any(coefficient.has(*parameters) for coefficient in coefficients):
            raise ValueError("not linear in the parameters")
        design.append(coefficients)
        targets.append(row[-1] - here.subs(dict.fromkeys(parameters, 0)))
    design = _worked_out(sp.Matrix(design), conv.context)
    targets = _worked_out(sp.Matrix(targets), conv.context)
    solution = design.solve_least_squares(targets)
    return expression.subs(dict(zip(parameters, solution)))


def _is_number(conv: Converter, args: list) -> sp.Basic:
    """`NUMBER(u)`: whether `u` simplified to a number.

    Always answers true or false, since an inert test would leave its `IF` or `SELECT`
    undecided. The utility files rely on this: `NUMBER(d)` is false while `d` is still an
    unbound parameter, which tells a supplied argument from an omitted one.

    A number is an integer, a rational, or a float from Approximate mode. Radicals,
    `#i` and `inf` don't count.
    """
    return sp.true if isinstance(_one(args), (sp.Rational, sp.Float)) else sp.false


def _is_prime(conv: Converter, args: list) -> sp.Basic:
    """`PRIME(m)`: whether `m` is a prime number.

    A predicate like `NUMBER`, so that e.g. `SELECT(PRIME(k), k, 1, 100)` works. A
    non-integer isn't simply false, since `PRIME(k)` for a variable `k` hasn't been
    asked yet, so such a call stays as written.

    The manual's second argument is the number of rounds of a probabilistic test. The
    test here is deterministic, so it is accepted and ignored.
    """
    value, *rounds = args
    if not isinstance(value, sp.Integer):
        raise TypeError("not an integer")
    return sp.true if sp.isprime(int(value)) else sp.false


def _next_prime(conv: Converter, args: list) -> sp.Basic:
    """`NEXT_PRIME(m)`: the first prime strictly above `m`, which needn't be an
    integer.

    `NEXT_PRIME(7)` is 11. A repeat count is accepted and ignored, as for `PRIME`.
    """
    value, *rounds = args
    if not isinstance(value, sp.Rational):
        raise TypeError("not a number")
    return sp.Integer(sp.nextprime(int(sp.floor(value))))


def _numerator(conv: Converter, args: list) -> sp.Basic:
    return sp.fraction(sp.together(_one(args)))[0]


def _denominator(conv: Converter, args: list) -> sp.Basic:
    return sp.fraction(sp.together(_one(args)))[1]


def _terms(conv: Converter, args: list) -> sp.Basic:
    """`TERMS(u)`: the terms of `u` as a vector.

    Terms are syntactic, so nothing is expanded: `TERMS(x*(a + b)^2 + c)` has two, and
    the manual says to compose with `EXPAND` otherwise. A non-sum is a single term.

    Terms are ordered by descending total degree, as the printer writes sums.

    A vector distributes, each element giving a row, so `TERMS` of a vector of sums is a
    matrix if they have equal numbers of terms.
    """
    value = _one(args)
    if isinstance(value, sp.MatrixBase):
        return vector_of([_terms(conv, [element]) for element in _elements_of(value)])
    if not isinstance(value, sp.Expr):
        raise TypeError("no terms")
    return vector_of(list(value.as_ordered_terms(order="grlex")))


def _factors(conv: Converter, args: list) -> sp.Basic:
    """`FACTORS(u)`: the factors of `u` as a vector.

    The counterpart of `TERMS`, and likewise syntactic: `FACTORS(x^2 - 1)` is one factor
    (compose with `FACTOR` otherwise), and `FACTORS(12)` is `[12]`, not a prime
    factorization.

    Factors are ordered as in Derive, by `_by_mainness`: `FACTORS(2*x*y)` is
    `[x, y, 2]`, and `[(x + 1)^2, x]` and `[SIN(x), x]` put compound factors first.

    A vector distributes, each element giving a row, as with `TERMS`.
    """
    value = _one(args)
    if isinstance(value, sp.MatrixBase):
        return vector_of([_factors(conv, [element]) for element in _elements_of(value)])
    if not isinstance(value, sp.Expr):
        raise TypeError("no factors")
    factors = value.as_ordered_factors() if value.is_Mul else [value]
    return vector_of(_by_mainness(factors))


def _by_mainness(pieces: list) -> list:
    """`pieces` sorted from most main to least, as Derive orders them.

    Mainness follows `main_order` for the piece's variables. Among pieces with the same
    main variable, compound ones come first (`SIN(x)` and `(x + 1)^2` before `x`), which
    is the reverse of sympy's order. Pieces without variables, like a numeric
    coefficient, go last.
    """
    order = main_order(name for piece in pieces for name in _named_in(piece))

    def rank(numbered: tuple[int, sp.Basic]) -> tuple[int, int]:
        position, piece = numbered
        names = _named_in(piece)
        return min((order.index(name) for name in names), default=len(order)), -position

    return [piece for _, piece in sorted(enumerate(pieces), key=rank)]


def _named_in(value: sp.Basic) -> set[str]:
    """The names of variables in `value`, excluding string literals."""
    return {symbol.name for symbol in value.free_symbols if type(symbol) is sp.Symbol}


def _side(index: int) -> Handler:
    """`LHS(u)` and `RHS(u)`: one side of a relation.

    Vectors distribute, so `RHS(SOLVE(x^2 - 5*x + 6 = 0, x))` is the vector of roots. A
    non-relation is returned as is for both, as Derive gives `LHS(2*x + 3)` as
    `2*x + 3`.
    """

    def handler(conv: Converter, args: list) -> sp.Basic:
        value = _one(args)
        if isinstance(value, sp.MatrixBase):
            return vector_of([handler(conv, [e]) for e in _elements_of(value)])
        if isinstance(value, InertVector):
            # A vector of relations, which sympy can't hold as a matrix.
            return vector_of([handler(conv, [e]) for e in value.args])
        if value.is_Relational:
            return value.args[index]
        return value

    return handler


def _quotient(conv: Converter, args: list) -> sp.Basic:
    """`QUOTIENT(u, v)`: the polynomial quotient of `u` by `v`."""
    return _divided(args)[0]


def _remainder(conv: Converter, args: list) -> sp.Basic:
    """`REMAINDER(u, v)`: the polynomial remainder of `u` by `v`."""
    return _divided(args)[1]


def _divided(args: list) -> tuple[sp.Basic, sp.Basic]:
    """Polynomial division in the main variable, with other variables as coefficients.

    Other variables act as field coefficients: `QUOTIENT(x*y + 1, y)` is `x + 1/y` with
    no remainder, since in `x` the divisor is a constant.

    Two numbers divide exactly: `QUOTIENT(7, 2)` is `7/2`, not 3, as in Derive (the
    manual says `u` and `v` should be rationals or polynomials).

    The main variable is the most main (by `_by_mainness`) of sympy's generators, which
    may be a kernel: `QUOTIENT(SIN(x), SIN(x)^2)` is 0 with remainder `SIN(x)`, and
    `QUOTIENT(SIN(x), x)` is `SIN(x)/x`, the divisor being a coefficient. Both match
    Derive.
    """
    numerator, denominator = args
    try:
        generators = sp.parallel_poly_from_expr((numerator, denominator))[1].gens
    except sp.PolificationFailed:
        # Two numbers: divide exactly.
        return numerator / denominator, sp.Integer(0)
    return sp.div(numerator, denominator, _by_mainness(list(generators))[0])


def _polynomial_gcd(conv: Converter, args: list) -> sp.Basic:
    """`POLY_GCD(u, v)`: the greatest common divisor of two polynomials.

    Numbers are constant polynomials, so `POLY_GCD(12, 18)` is 6.
    """
    left, right = args
    return sp.gcd(left, right)


def _variables(conv: Converter, args: list) -> sp.Basic:
    """`VARIABLES(u)`: the free variables of `u`, from most main to least.

    The order is `ordering`'s, as in Derive: `x`, `y`, `z` first, then alphabetical. A
    fixed order is needed since sympy returns a set and the answer must print the same
    every time.

    The original symbols are returned, so they keep their declared domains. String
    literals are excluded.
    """
    found = _one(args).free_symbols
    by_name = {symbol.name: symbol for symbol in found if type(symbol) is sp.Symbol}
    return vector_of([by_name[name] for name in main_order(by_name)])


def _dif(conv: Converter, args: list) -> sp.Basic:
    """`DIF(u, x)` and `DIF(u, x, n)`, unevaluated.

    A negative order means repeated antiderivatives (7.4, p.177): `DIF(u, x, -2)` is
    `INT(INT(u, x), x)`, since sympy has no head for a negative-order derivative.

    Vectors and matrices are differentiated element-wise and keep their shape:
    `DIF([1, 1], x)` is `[0, 0]`, where sympy would give the scalar 0.
    """
    _variable(args[1])
    if isinstance(args[0], sp.MatrixBase):
        return args[0].applyfunc(lambda element: _dif(conv, [element, *args[1:]]))
    if len(args) == 2:
        return sp.Derivative(args[0], args[1], evaluate=False)
    expression, variable, order = args
    if order.is_Integer and order.is_negative:
        if isinstance(expression, Relational):
            raise TypeError("no antiderivative of an equation")
        return sp.Integral(expression, *[variable] * int(-order))
    return sp.Derivative(expression, (variable, order), evaluate=False)


def _integral(conv: Converter, args: list) -> sp.Basic:
    """`INT(u, x)` and `INT(u, x, a, b)`, both unevaluated.

    Vectors and matrices are integrated element-wise, keeping their shape. A vector
    limit gives one integral per element: `INT(x^2, x, 0, [1, 2])` is `[1/3, 8/3]`. Two
    vector limits are refused, and the call stands.

    A definite integral of an equation integrates both sides. An antiderivative of one
    doesn't, since each side has its own constant, so `INT(y = x, x)` stands.
    """
    _variable(args[1])
    if isinstance(args[0], sp.MatrixBase):
        return args[0].applyfunc(lambda element: _integral(conv, [element, *args[1:]]))
    if len(args) == 2:
        if isinstance(args[0], Relational):
            raise TypeError("no antiderivative of an equation")
        return sp.Integral(args[0], args[1])
    distributed = _over_vector_limits(conv, _integral, args)
    if distributed is not None:
        return distributed
    expression, variable, low, high = args
    if isinstance(expression, sp.Equality):
        return sp.Eq(
            sp.Integral(expression.lhs, (variable, low, high)),
            sp.Integral(expression.rhs, (variable, low, high)),
            evaluate=False,
        )
    if isinstance(expression, Relational):
        raise TypeError("not an equation")
    return sp.Integral(expression, (variable, low, high))


def _variable(variable: sp.Basic) -> None:
    """Refuse a vector, or an element not yet known, as a calculus function's variable.

    `DIF(x1*x2, [x1, x2] SUB k)` is a derivative in `x1` or `x2` once `k` is known, and
    stays a call until then, as in Derive. Taken in the unevaluated element it would be
    0.
    """
    if isinstance(variable, (sp.MatrixBase, InertVector)) or _is_element(variable):
        raise TypeError("not a variable")


def _is_element(value: sp.Basic) -> bool:
    """Whether `value` is an `ELEMENT` access kept unevaluated."""
    return isinstance(value, AppliedUndef) and type(value).__name__ == "ELEMENT"


def _over_vector_limits(
    conv: Converter, build: Callable[[Converter, list], sp.Basic], args: list
) -> sp.Basic | None:
    """Apply `build` for each element of whichever limit in `args` is a vector.

    `args` is `u, k, a, b`. Returns the vector of results, or None if neither limit is
    a vector. Two vector limits are refused.
    """
    expression, variable, low, high = args
    if isinstance(low, sp.MatrixBase) and isinstance(high, sp.MatrixBase):
        raise TypeError("two vectors of limits")
    if isinstance(low, sp.MatrixBase):
        return low.applyfunc(lambda end: build(conv, [expression, variable, end, high]))
    if isinstance(high, sp.MatrixBase):
        return high.applyfunc(lambda end: build(conv, [expression, variable, low, end]))
    return None


def _summation(conv: Converter, args: list) -> sp.Basic:
    """`SUM(u, k, a, b)`, and `SUM(v)` over the elements of a vector.

    A vector third argument lists the values of `k`; otherwise it is the upper end of a
    range from 1, so `SUM(u, k, m)` is `SUM(u, k, 1, m)`. A vector limit gives one sum
    per element, as for `INT`: `SUM(k, k, [1, 2], 3)` is `[6, 5]`.

    An indefinite `SUM(u, k)` is the antidifference of `u`, deferred like the calculus
    heads, and of each element of a vector `u`: `SUM([1, 2], x)` is `[x, 2*x]`.
    """
    if len(args) == 1 and isinstance(args[0], sp.MatrixBase):
        return sp.Add(*args[0])
    _variable(args[1])
    if len(args) == 2 and isinstance(args[0], sp.MatrixBase):
        return args[0].applyfunc(lambda element: Antidifference(element, args[1]))
    if len(args) == 2:
        return Antidifference(*args)
    if len(args) == 3 and isinstance(args[2], sp.MatrixBase):
        return sp.Add(*_over_values(conv, args))
    args = _from_one(args)
    distributed = _over_vector_limits(conv, _summation, args)
    if distributed is not None:
        return distributed
    expression, index, low, high = args
    return sp.Sum(expression, (index, low, high))


def _product(conv: Converter, args: list) -> sp.Basic:
    """`PRODUCT(u, k, a, b)`, and `PRODUCT(v)` over a vector's elements.

    Arguments work as for `SUM`. An indefinite `PRODUCT(u, k)` is the antiquotient of
    `u`.
    """
    if len(args) == 1 and isinstance(args[0], sp.MatrixBase):
        return sp.Mul(*args[0])
    _variable(args[1])
    if len(args) == 2:
        return Antiquotient(*args)
    if len(args) == 3 and isinstance(args[2], sp.MatrixBase):
        return sp.Mul(*_over_values(conv, args))
    args = _from_one(args)
    distributed = _over_vector_limits(conv, _product, args)
    if distributed is not None:
        return distributed
    expression, index, low, high = args
    return sp.Product(expression, (index, low, high))


def _from_one(args: list) -> list:
    """Sum or product arguments with a lower limit of 1 filled in if omitted."""
    if len(args) == 3:
        return [*args[:2], sp.Integer(1), args[2]]
    return args


def _over_values(conv: Converter, args: list) -> list[sp.Basic]:
    """The body of a sum or product evaluated at each listed index value."""
    expression, index, values = args
    if type(index) is not sp.Symbol:
        raise TypeError("not a variable")
    return [expression.subs(index, value) for value in _elements_of(values)]


def _taylor(conv: Converter, args: list) -> sp.Basic:
    """`TAYLOR(u, x, a, n)`, unevaluated; the pipeline decides when to compute it."""
    return Taylor(*args)


def _approximation(conv: Converter, args: list) -> sp.Basic:
    """`APPROX(u)` at the session's precision, `APPROX(u, n)` at `n` digits.

    Left unevaluated as explained in `Approx`. The digit count is filled in here, the
    last place that knows the caller's precision.

    The result has the requested digits, but inside a larger expression the printer
    uses the session's precision, so extra digits are computed but not shown. A call
    that is the whole expression goes through `pipeline.approx` and shows them.

    A numeric count must be usable; a symbolic one is kept as is. NUMBER.MTH gives
    `PARTS(i_)` a count in terms of `i_`, known only once the enclosing sum is
    expanded. `digit_count` is checked again when rounding.
    """
    value, *rest = args
    digits = _one(rest) if rest else sp.Integer(conv.context.precision_digits)
    if digits.is_number and digit_count(digits) is None:
        raise ValueError("not a precision")
    return Approx(value, digits)


def digit_count(digits: sp.Basic) -> int | None:
    """The integer number of digits `digits` asks for, or None if not a usable count.

    The count needn't be an integer: NUMBER.MTH asks `PARTS` for
    `LOG(1/(4*n*SQRT(3))*EXP(pi*SQRT(2*n/3)), 10) + 5` digits, which is 5.78 for
    `n = 4`. The floor is used, and less than one digit is invalid.
    """
    try:
        whole = sp.floor(digits)
    except Exception:
        return None
    return int(whole) if whole.is_Integer and whole > 0 else None


def _conditional(conv: Converter, args: list) -> sp.Basic:
    """`IF(c)`, `IF(c, u)` and `IF(c, u, v)` as a sympy `Piecewise`.

    Choosing a branch is up to the evaluator, which decides the test before converting
    either branch; this converts a conditional as written, which is how an undecidable
    one is shown. Derive's four-argument form (with a value for an undecidable test) has
    no `Piecewise` equivalent and stays an inert head.

    `IF(c)` alone is the test as 1 or 0, so `SUM(IF(PRIME(n)), n, 1, 100)` counts
    primes.
    """
    if len(args) == 1:
        return sp.Piecewise(
            (sp.Integer(1), as_condition(args[0])), (sp.Integer(0), sp.true)
        )
    if len(args) == 2:
        test, then = args
        return sp.Piecewise((then, as_condition(test)))
    test, then, otherwise = args
    return sp.Piecewise((then, as_condition(test)), (otherwise, sp.true))


def authored_conditionals(node: Node, context: Context) -> dict[sp.Basic, str]:
    """Every conditional in `node`, as text keyed by its converted expression.

    Derive shows an undecidable conditional exactly as typed: the test isn't rearranged
    and the arms aren't evaluated, so `IF(x > 0, 2 + 3, 4 + 5)` keeps its arithmetic.
    The converted expression has already lost that (`2 + 3` is 5), so the text is taken
    from the tree and given to the printer, which uses it wherever that expression
    appears in an answer.

    The text always converts back to its key, so this only changes spelling, never
    value. The one imprecision is that two conditionals converting to the same
    expression share one spelling.

    Keying by expression rather than splicing means commands run on the same input as
    before. A conditional the evaluator decided has no key in the answer (only the
    chosen arm remains), and one it reshaped (typically an argument substituted into a
    function's `IF`) no longer matches and prints as computed. Both match Derive.

    The authored subtree is converted before any definitions are substituted, since
    the goal is to show what the author wrote; if that differs from what the command
    converted, there is no match and the answer prints as computed.
    """
    authored: dict[sp.Basic, str] = {}
    for found in _conditional_nodes(node):
        try:
            converted = to_sympy(found, context)
        except Exception:
            continue
        if is_conditional(converted):
            authored.setdefault(converted, write_expression(found, spaced=True))
    return authored


def _conditional_nodes(node: Node) -> list[Node]:
    """Every `IF` call in the tree, outermost first, including nested ones."""
    found = []
    if node.kind in (Kind.CALL, Kind.APPLY) and len(node.children) > 1:
        if str(node.children[0].value).upper() == "IF":
            found.append(node)
    for child in node.children:
        found += _conditional_nodes(child)
    return found


def is_conditional(expression: sp.Basic) -> bool:
    """Whether this is a conditional: a `Piecewise`, or an inert `IF`.

    The inert `IF` covers Derive's four-argument form, which has no `Piecewise`
    equivalent; both are treated alike when resolved.
    """
    if isinstance(expression, sp.Piecewise):
        return True
    return (
        isinstance(expression, AppliedUndef)
        and type(expression).__name__ == "IF"
        and 1 <= len(expression.args) <= 4
    )


def as_condition(test: sp.Basic) -> sp.Basic:
    """A conditional's test, read as Derive reads it.

    Relations and boolean combinations of them (including `Logical` heads) are tests as
    written. Any other expression is shorthand for comparing with zero: `IF(0, a, b)` is
    `a` and `IF(5, a, b)` is `b`, because Derive reads the test as `test = 0`. Sympy's
    truthiness would read it the opposite way.
    """
    if isinstance(test, (Boolean, Logical)):
        return _test(test)
    return sp.Eq(test, 0)


def _test(test: sp.Basic) -> sp.Basic:
    """A condition with its relations evaluated rather than held.

    Elsewhere relations are built unevaluated, so the pipeline decides them using
    declared domains and simplified sides. Conditional tests are the exception:
    `Piecewise` decides its own conditions and mishandles unevaluated relations.

    This matters most for conjunctions, where an unevaluated link can't be decided and
    `Piecewise` refuses to build. `NUMBER.MTH` guards `FIBONACCI` with
    `n >= 0 AND FLOOR(n) = n`, and both links must be settled once `n` is a number.
    """
    if test.is_Relational:
        return test.func(test.lhs, test.rhs)
    if isinstance(test, BooleanFunction) and test.args:
        return test.func(*(_test(operand) for operand in test.args))
    return test


def _truth_table(conv: Converter, args: list) -> sp.Basic:
    """`TRUTH_TABLE(p, q, ..., b1, b2, ...)`: the truth table, as a matrix.

    The leading bare variables are the truth variables and the rest are expressions to
    evaluate; with only variables, as in `TRUTH_TABLE(p, q)`, the table lists just the
    assignments.

    The first row is the column headings as written. The remaining rows cover every
    assignment, last variable changing fastest and `true` before `false`, as the manual
    prints it.

    Every expression must be true or false at every assignment; otherwise the call
    stays as written.
    """
    variables: list[sp.Basic] = []
    for value in args:
        if type(value) is not sp.Symbol:
            break
        variables.append(value)
    expressions = args[len(variables) :]
    if not variables:
        raise ValueError("no truth variables")
    rows = [InertVector(*(_as_written(value) for value in args))]
    for assignment in product([sp.true, sp.false], repeat=len(variables)):
        written = dict(zip(variables, assignment, strict=True))
        values = [_decided(expression, written) for expression in expressions]
        rows.append(InertVector(*assignment, *values))
    return vector_of(rows)


#: Each sympy logical head and its operator name, for building inert `Logical`
#: headings that Simplify leaves alone.
_HEADINGS = {sp.And: "AND", sp.Or: "OR", sp.Xor: "XOR", sp.Implies: "IMP", sp.Not: "NOT"}


def _as_written(expression: sp.Basic) -> sp.Basic:
    """A column heading: the expression as written, made inert.

    A live `p XOR q` heading would be simplified into
    `NOT p AND q OR p AND NOT q`, which is correct but a worse label. The inert head
    keeps the operator as written.
    """
    word = _HEADINGS.get(type(expression))
    if word is None:
        return expression
    operands = [_as_written(operand) for operand in expression.args]
    return Logical(sp.Symbol(word), *operands)


def _decided(expression: sp.Basic, assignment: dict) -> sp.Basic:
    """One cell: the expression's value under one assignment."""
    value = _test(expression.subs(assignment))
    if value not in (sp.true, sp.false):
        raise TypeError("not a boolean")
    return value


def _substitution(conv: Converter, args: list) -> sp.Basic:
    """`SUBS(u, [x, ...], [a, ...])`, as the printer writes sympy's `Subs`.

    This must read back as a `Subs`: it holds a derivative that only makes sense
    unevaluated (`d/dv f(v)` at `v = y` isn't the derivative of a constant), and as an
    inert head that derivative would be evaluated to zero.
    """
    expression, variables, points = args
    return sp.Subs(expression, tuple(_matrix(variables)), tuple(_matrix(points)))


def _limit(conv: Converter, args: list) -> sp.Basic:
    """`LIM(u, x, a)` is two-sided; a fourth argument picks a side.

    A zero side means two-sided, which is what the `Calculus Limit` command writes for
    Both (it always writes the fourth argument).

    Vectors of variables and points take the limits one after another, in order, which
    the manual suggests where plain substitution would divide by zero. These are
    iterated, not multivariate, limits: `LIM(u, [x, y], [a, b])` may differ from
    `LIM(u, [y, x], [b, a])`, as the manual notes.

    A point mentioning the variable is a substitution (`MovingLimit`), and any side
    argument is ignored.

    An omitted point is zero, for each variable of a vector: PSERIES.MTH writes
    `LIM(P, x)` for the value of `P` at zero.
    """
    if len(args) == 2:
        variables = args[1]
        point = (
            sp.zeros(*variables.shape)
            if isinstance(variables, sp.MatrixBase)
            else sp.Integer(0)
        )
        args = [*args, point]
    if isinstance(args[1], sp.MatrixBase):
        return _iterated_limit(args)
    if len(args) == 3:
        return _approached(*args, "+-")
    expression, variable, point, side = args
    if not side:
        return _approached(expression, variable, point, "+-")
    return _approached(expression, variable, point, "+" if side > 0 else "-")


def _approached(
    expression: sp.Basic, variable: sp.Basic, point: sp.Basic, direction: str
) -> sp.Basic:
    """One limit as a sympy `Limit`, or a `MovingLimit` substitution.

    Matrices are handled element-wise, since sympy has no matrix limits. SOLVE.MTH's
    `NEWTON_AUX` relies on this, substituting the current iterate into a whole augmented
    matrix with `LIM(a, x, xk)`.
    """
    if isinstance(expression, sp.MatrixBase):
        return expression.applyfunc(
            lambda element: _approached(element, variable, point, direction)
        )
    if point.has(variable):
        return MovingLimit(expression, variable, point)
    return sp.Limit(expression, variable, point, dir=direction)


def _iterated_limit(args: list) -> sp.Basic:
    """`LIM(u, [x, y], [a, b])`: the limit in `x`, then in `y`.

    The first variable is innermost, so the outer limit applies to the inner result.
    All limits are left for evaluation.
    """
    expression, variables, points = args
    names = _elements_of(variables)
    values = _elements_of(points)
    if len(names) != len(values):
        raise ValueError("not a point for every variable")
    for name, value in zip(names, values, strict=True):
        expression = _approached(expression, name, value, "+-")
    return expression


def _interval(conv: Converter, args: list) -> sp.Basic:
    """`INTERVAL(a, b)`: a value known only to lie between `a` and `b`.

    This is the result of a limit that stays bounded without converging: near zero,
    `SIN(1/x)` takes every value in `INTERVAL(-1, 1)`. Sympy's accumulation bounds
    support interval arithmetic, so `INTERVAL(-1, 1) + 1` is `INTERVAL(0, 2)`, and the
    value can be differentiated, integrated and approximated.

    The lower bound comes first. Reversed bounds, or bounds whose order is unknown like
    `INTERVAL(x, 2)`, raise and leave the call inert.
    """
    low, high = args
    if (low <= high) is not sp.true:
        raise ValueError("not a lower bound below an upper one")
    return sp.AccumBounds(low, high)


def _root_sum(conv: Converter, args: list) -> sp.Basic:
    """`ROOT_SUM(p, t, u)`: the sum of `u` over every root `t` of `p`.

    This is the logarithmic part of a rational integral whose denominator doesn't
    factor: `INT(1/(x^5 - x - 1), x)` is such a sum over the quintic's roots, which have
    no radical form. Sympy holds the summand as a `Lambda`; the notation names the bound
    variable second, as `SUM` does.

    Requires exactly three arguments, a variable, and a polynomial in that variable
    alone; otherwise it raises, so e.g. `ROOT_SUM(u, k)` stays inert.
    """
    polynomial, variable, summand = args
    if type(variable) is not sp.Symbol:
        raise TypeError("not a variable")
    if polynomial.free_symbols - {variable}:
        raise ValueError("not a polynomial in that variable alone")
    # `Poly` rejects non-polynomials like `SIN(t)`, and constants have no roots.
    if sp.Poly(polynomial, variable).degree() < 1:
        raise ValueError("not a polynomial with roots")
    return sp.RootSum(polynomial, sp.Lambda(variable, summand), variable)


def _root_of(conv: Converter, args: list) -> sp.Basic:
    """`ROOT_OF(p, t, n)`: the `n`-th root of `p` in `t`, counting from zero.

    A single root of the kind `ROOT_SUM` sums over, which sympy returns when solving
    quintics and higher. The index uses sympy's order (real roots ascending, then
    complex) and is the only thing distinguishing roots of the same polynomial.
    """
    polynomial, variable, index = args
    if type(variable) is not sp.Symbol:
        raise TypeError("not a variable")
    if not isinstance(index, sp.Integer):
        raise TypeError("not an index")
    return sp.CRootOf(polynomial, variable, int(index))


def _determinant(conv: Converter, args: list) -> sp.Basic:
    """`DET(u)`: the determinant, or section 8.8's identities for symbolic matrices.

    For symbolic matrices sympy applies `DET(a^-1)` = `1/DET(a)` and
    `DET(a . b)` = `DET(a)*DET(b)`.
    """
    value = _one(args)
    if _symbolic_matrix(value):
        return sp.Determinant(value).doit()
    return _matrix(value).det()


def _trace(conv: Converter, args: list) -> sp.Basic:
    value = _one(args)
    if _symbolic_matrix(value):
        return sp.Trace(value).doit()
    return _matrix(value).trace()


def _dimension(conv: Converter, args: list) -> sp.Basic:
    """The number of elements in a vector, or rows in a matrix.

    Vectors of relations count too: the shipped libraries branch on the number of
    `SOLVE` solutions, and no solutions is an empty vector rather than an error.
    """
    value = _one(args)
    if isinstance(value, InertVector):
        return sp.Integer(len(value.args))
    matrix = _matrix(value)
    return sp.Integer(matrix.cols if matrix.rows == 1 else matrix.rows)


def _element_of(conv: Converter, args: list) -> sp.Basic:
    """`ELEMENT(v, i)`, `ELEMENT(m, i, j)`, and so on for any number of dimensions,
    counting from 1.

    Each index selects from the previous result: `ELEMENT(q, 2, 1, 2)` reads a number
    from a vector of matrices. Vectors of relations work too, as the shipped ODE library
    uses in `RHS((SOLVE(z, y)) SUB 1)`. A vector index applies its elements in turn, so
    `m SUB [2, 3]` is `m SUB 2 SUB 3`.
    """
    value, *indices = args
    if not indices:
        raise ValueError("no index")
    for index in indices:
        value = _at(value, index)
        if value is None:
            raise ValueError("not an index")
    return value


def _at(value: sp.Basic, index: sp.Basic) -> sp.Basic | None:
    """One element of a vector, whichever way it is represented.

    Returns None if `index` is invalid or `value` isn't a vector, which can happen
    midway through an index vector that reaches a scalar with indices left over.
    """
    if isinstance(index, sp.MatrixBase):
        for step in _elements_of(index):
            value = _at(value, step)
            if value is None:
                return None
        return value
    if isinstance(value, sp.MatrixBase):
        return _element(value, index)
    if not isinstance(value, InertVector):
        return None
    if not isinstance(index, sp.Integer):
        return None
    number = int(index)
    return value.args[number - 1] if 1 <= number <= len(value.args) else None


def _delete_element(conv: Converter, args: list) -> sp.Basic:
    """`DELETE_ELEMENT(v, n)`: `v` without its `n`th element, counting from 1.

    A matrix's elements are its rows, so this deletes a row. VECTOR.MTH's `MINOR`
    relies on this: delete a row, transpose, delete a row again, transpose back.

    `n` defaults to 1, as for the other element functions.
    """
    vector, *rest = args
    elements = _elements_of(_matrix(vector))
    place = _place(_one(rest) if rest else sp.Integer(1), len(elements))
    return vector_of(elements[:place] + elements[place + 1 :])


def _replace_element(conv: Converter, args: list) -> sp.Basic:
    """`REPLACE_ELEMENT(u, v, n)`: `v` with its `n`th element replaced by `u`.

    New value first, then the vector, as in the manual and `INSERT_ELEMENT`. `n`
    defaults to 1.
    """
    value, vector, *rest = args
    elements = _elements_of(_matrix(vector))
    place = _place(_one(rest) if rest else sp.Integer(1), len(elements))
    return vector_of(elements[:place] + [value] + elements[place + 1 :])


def _insert_element(conv: Converter, args: list) -> sp.Basic:
    """`INSERT_ELEMENT(u, v, n)`: `v` with `u` inserted before its `n`th element.

    Arguments are ordered as for `REPLACE_ELEMENT`, and `n` defaults to 1. One past the
    end is allowed here, to append: `INSERT_ELEMENT(d, [a, b, c], 4)` is `[a, b, c, d]`.
    """
    value, vector, *rest = args
    elements = _elements_of(_matrix(vector))
    place = _place(_one(rest) if rest else sp.Integer(1), len(elements) + 1)
    return vector_of(elements[:place] + [value] + elements[place:])


def _reversed_vector(conv: Converter, args: list) -> sp.Basic:
    """`REVERSE_VECTOR(v)`: `v` with its elements reversed (a matrix's rows, for a
    matrix)."""
    return vector_of(_elements_of(_matrix(_one(args)))[::-1])


def _appended(conv: Converter, args: list) -> sp.Basic:
    """`APPEND(v, w, ...)`: the elements of each, concatenated into one vector.

    A matrix's elements are its rows, so appending matrices stacks them, as the
    manual's `APPEND_COLUMNS` exercise uses (transpose, append, transpose back). A
    single matrix is the documented exception and flattens: `APPEND([[a, b], [c, d]])`
    is `[a, b, c, d]`.

    A plain vector appended to a matrix of its width becomes one row, as SOLVE.MTH's
    `NEWTONS` uses to put its system under its gradients. A one-row matrix is the same
    as a vector here, so the `[u]` in `APPEND(GRAD(u, x), [u])` arrives as `u`. Treating
    it as a row of matching width gives Derive's matrix instead of a ragged vector.
    """
    if len(args) == 1 and isinstance(args[0], sp.MatrixBase):
        return vector_of(list(args[0]))
    width = _stacked_width(args)
    elements: list[sp.Basic] = []
    for vector in args:
        matrix = _matrix(vector)
        if width is not None and matrix.rows == 1 and matrix.cols == width:
            elements.append(matrix)
        else:
            elements.extend(_elements_of(matrix))
    return vector_of(elements)


def _stacked_width(args: list) -> int | None:
    """The common row width of the multi-row matrices in `args`, if any.

    None if there are no multi-row matrices (so `APPEND([a, b], [c, d])` is
    `[a, b, c, d]`, not a matrix), or if their widths differ.
    """
    widths = {
        argument.cols
        for argument in args
        if isinstance(argument, sp.MatrixBase) and argument.rows > 1
    }
    return widths.pop() if len(widths) == 1 else None


def _place(index: sp.Basic, count: int) -> int:
    """The zero-based position of the 1-based `index` among `count` elements."""
    if not isinstance(index, sp.Integer) or not 1 <= index <= count:
        raise ValueError("not an index")
    return int(index) - 1


def _generated_vector(conv: Converter, args: list) -> sp.Basic:
    """`VECTOR(u, k, ...)`: a vector of `u` at each value of `k`.

    The values are given as a count, a range, a range with a step, or a vector of
    values. Nested calls make a matrix.

    Non-numeric bounds leave the call inert until they are numbers, which keeps
    `VECTOR([v SUB i], i, DIMENSION(v))` intact in a definition until it is applied.
    """
    body, index, *rest = args
    if type(index) is not sp.Symbol:
        raise TypeError("not a variable")
    return vector_of([body.subs(index, value) for value in steps(rest)])


def steps(rest: list) -> list[sp.Basic]:
    """The values a generated vector's variable takes, in order.

    There are `(n - m)/s + 1` of them, rounded down: one if the step overshoots the
    range, none if the range runs the wrong way.
    """
    if len(rest) == 1 and isinstance(rest[0], sp.MatrixBase):
        return _elements_of(rest[0])
    if not rest or len(rest) > 3:
        raise ValueError("not a sequence")
    low = sp.Integer(1) if len(rest) == 1 else rest[0]
    high = rest[0] if len(rest) == 1 else rest[1]
    step = rest[2] if len(rest) == 3 else sp.Integer(1)
    count = int(sp.floor((high - low) / step)) + 1
    return [low + step * offset for offset in range(max(count, 0))]


def _selected(conv: Converter, args: list) -> sp.Basic:
    """`SELECT(u, k, ...)`: the values of `k` for which `u(k)` holds.

    Values are specified as for `VECTOR`. The result is the values themselves:
    `SELECT(PRIME(k), k, 1, 100)` is the primes under a hundred.

    Every test must be true or false. An undecidable one would silently drop its
    element, so the call stays as written instead, which also keeps a `SELECT` in a
    definition intact until it is applied.
    """
    body, index, *rest = args
    if type(index) is not sp.Symbol:
        raise TypeError("not a variable")
    return chosen_from(steps(rest), lambda value: body.subs(index, value))


def chosen_from(values: list, test: Callable[[sp.Basic], sp.Basic]) -> sp.Basic:
    """The vector of `values` passing `test`; an undecided test raises."""
    chosen = []
    for value in values:
        held = _test(test(value))
        if held is sp.true:
            chosen.append(value)
        elif held is not sp.false:
            raise ValueError("undecided")
    return vector_of(chosen)


def _iterates(conv: Converter, args: list) -> sp.Basic:
    """`ITERATES(u, x, x0)` and `ITERATES(u, x, x0, n)`: the sequence as a vector.

    `x0`, `u(x0)`, `u(u(x0))` and so on: `n` updates give `n + 1` elements, and
    without a count it runs until a value repeats.
    """
    body, variable, start, *rest = args
    return iterated("ITERATES", conv, variable, start, rest, body=body)


def _iterate(conv: Converter, args: list) -> sp.Basic:
    """`ITERATE(...)`: the last element of the `ITERATES` sequence.

    With a count, that is the `n`th update. Without one, the sequence ended on a
    repeat, and only a fixed point (a value repeating *itself*) counts as converged; a
    longer cycle gives `?`, as in the manual. The repeat is checked at displayed
    precision, as in `_until_repeated`.
    """
    body, variable, start, *rest = args
    return iterated("ITERATE", conv, variable, start, rest, body=body)


#: Evaluates an iteration's body for given variables and their current values.
Step = Callable[[list, list], sp.Basic]


def iterated(
    name: str,
    conv: Converter,
    variable: sp.Basic,
    start: sp.Basic,
    rest: list,
    step: Step | None = None,
    body: sp.Basic | None = None,
) -> sp.Basic:
    """`ITERATE` or `ITERATES` from `start`, with the optional count in `rest`.

    `step` computes the next iterate. Conversion substitutes values into the converted
    `body`; the evaluator passes its own step, which re-evaluates the body each time.
    """
    sequence = _sequence(conv, variable, start, rest, step, body)
    if name == "ITERATES":
        return vector_of(sequence)
    if rest:
        return sequence[-1]
    came_round = _as_shown(sequence[-1], conv.context) == _as_shown(
        sequence[-2], conv.context
    )
    return sequence[-1] if came_round else sp.nan


def _substituted(body: sp.Basic) -> Step:
    return lambda names, values: body.subs(
        _bindings(body, names, values), simultaneous=True
    )


def _sequence(
    conv: Converter,
    variable: sp.Basic,
    start: sp.Basic,
    rest: list,
    step: Step | None,
    body: sp.Basic | None,
) -> list[sp.Basic]:
    """The iterates, as many as requested.

    A negative count iterates the inverse of `u` `|n|` times, which `MISC.MTH` uses to
    define `INVERSE(u, x) := ITERATE(u, x, x, -1)`. This needs the converted body.
    """
    names, values = _iterated_over(variable, start)
    if step is None:
        if body is None:
            raise ValueError("no body to iterate")
        step = _substituted(body)
    if not rest:
        return _until_repeated(conv, step, names, values)
    count = _one(rest)
    if not isinstance(count, sp.Integer):
        raise TypeError("not a count")
    if count < 0:
        if body is None:
            raise ValueError("no body to invert")
        step, count = _substituted(_inverted(body, names)), -count
    iterates = [_state(names, values)]
    for _ in range(int(count)):
        values = _updated(conv, step, names, values)
        iterates.append(_state(names, values))
    return iterates


def _until_repeated(
    conv: Converter, step: Step, names: list, values: list
) -> list[sp.Basic]:
    """The sequence, run until a value repeats.

    The repeated value is kept as the last element, so `ITERATES(1/x, x, 2)` is
    `[2, 1/2, 2]`, which tells `ITERATE` whether the cycle has length one.

    Repeats are detected on displayed values, not carried ones. Approximate iterates
    carry guard digits, so a converging sequence would keep differing in invisible
    digits until it hit the bound; the manual's output stops when the display stops
    changing. That is why `ITERATES(#e^(-x/20), x, 1)` ends in three copies of
    `0.953446`: the first two differ in the seventh digit. In exact modes this is plain
    equality.

    Derive iterates until memory runs out if nothing repeats. Here, an iteration that
    doesn't repeat within the bounds below raises, leaving the call as written.
    """
    iterates = [_state(names, values)]
    shown = [_as_shown(iterates[-1], conv.context)]
    for _ in range(_ITERATIONS):
        values = _updated(conv, step, names, values)
        iterates.append(_state(names, values))
        shown.append(_as_shown(iterates[-1], conv.context))
        if shown[-1] in shown[:-1]:
            return iterates
        if outsized(iterates[-1]):
            break
    raise ValueError("comes round to nothing")


#: Limits for an uncounted iteration: the number of steps and the size of an iterate.
#: Both are needed, since a non-repeating sequence usually blows up faster than a step
#: limit catches (repeated squaring passes 30000 digits in 17 steps). Converging
#: iterations come nowhere near either.
_ITERATIONS = 100
_ITERATE_BITS = 100_000
_ITERATE_OPERATIONS = 1000


def outsized(value: sp.Basic) -> bool:
    """Whether a value has grown too large to keep iterating on.

    This is the size limit for open-ended computations like uncounted iteration.
    """
    if sp.count_ops(value) > _ITERATE_OPERATIONS:
        return True
    return any(
        int(number).bit_length() > _ITERATE_BITS for number in value.atoms(sp.Integer)
    )


def _iterated_over(variable: sp.Basic, start: sp.Basic) -> tuple[list, list]:
    """The variables an iteration updates, and their starting values.

    Either one variable and one value, or a vector of variables with a vector of
    values, one of the manual's two forms of Fibonacci:
    `ITERATE([k, j + k], [j, k], [0, 1], n)`.

    The other form is one variable holding the vector, accessed by subscripts:
    `ITERATE([v SUB 2, v SUB 1 + v SUB 2], v, [0, 1], n)`. That is one name and one
    value here; `_bindings` fills in the subscripts.
    """
    if isinstance(variable, sp.MatrixBase):
        names = _elements_of(variable)
        values = _elements_of(_matrix(start))
        if len(names) != len(values):
            raise ValueError("not that many values")
    else:
        names, values = [variable], [start]
    if any(type(name) is not sp.Symbol for name in names):
        raise TypeError("not a variable")
    return names, values


def _state(names: list, values: list) -> sp.Basic:
    """One iterate: the single variable's value, or the vector of all values."""
    return values[0] if len(names) == 1 else vector_of(values)


def _updated(conv: Converter, step: Step, names: list, values: list) -> list:
    """The variables' next values: the body evaluated at the current ones.

    All are substituted at once, since a system's update reads only the previous
    iterate, and it must return as many values as it took.

    In Approximate mode each iterate is evaluated numerically, which is what iteration
    means there and not just faster: left exact, each step nests the previous one into
    every transcendental head, and SOLVE.MTH's `NEWTONS` over `#e^(x*y)` and
    `ATAN(x*y)` builds a six-deep tower that never reduces to the printed six digits.
    """
    written = _worked_out(step(names, values), conv.context)
    if len(names) == 1:
        return [written]
    elements = _elements_of(_matrix(written))
    if len(elements) != len(names):
        raise ValueError("not that many values")
    return elements


def _worked_out(value: sp.Basic, context: Context) -> sp.Basic:
    """`value` evaluated numerically, in Approximate mode only.

    This is an intermediate value, so it carries guard digits to avoid compounding
    rounding errors. The manual's `ITERATES(#e^(-x/20), x, 1, 5)` shows `0.953441` at
    the fourth iterate, not the `0.953440` that rounding each step to six digits gives.

    Exact and Mixed modes stay exact throughout, however large the values grow.
    """
    if context.precision is not Precision.APPROXIMATE:
        return value
    digits = context.precision_digits + GUARD
    return approximated(value, context.with_precision(Precision.APPROXIMATE, digits))


def _as_shown(value: sp.Basic, context: Context) -> sp.Basic:
    """`value` at displayed precision, used to compare iterates.

    The counterpart of `_worked_out`, which adds guard digits; this removes them. Exact
    iterates are returned unchanged.
    """
    if context.precision is not Precision.APPROXIMATE:
        return value
    return approximated(value, context)


def _bindings(body: sp.Basic, names: list, values: list) -> dict:
    """Substitutions for the body: each variable, and each of its subscripts.

    `v SUB 1` is a separate symbol, so substituting for `v` doesn't reach it; the
    elements of a vector value are bound to those subscript symbols as well.
    """
    bindings = dict(zip(names, values, strict=True))
    for name, value in zip(names, values, strict=True):
        if not isinstance(value, (sp.MatrixBase, InertVector)):
            continue
        try:
            elements = _elements_of(_matrix(value))
        except Exception:
            continue
        for symbol in body.free_symbols:
            place = _subscripted(symbol, name)
            if place is not None and 1 <= place <= len(elements):
                bindings[symbol] = elements[place - 1]
    return bindings


def _subscripted(symbol: sp.Symbol, name: sp.Symbol) -> int | None:
    """The index if `symbol` is a subscript of `name`, else None."""
    prefix = f"{name.name} SUB "
    if not symbol.name.startswith(prefix):
        return None
    index = symbol.name[len(prefix) :]
    return int(index) if index.isdigit() else None


def _inverted(body: sp.Basic, names: list) -> sp.Basic:
    """The inverse of `body`, for an iteration with a negative count.

    Solves `u(x) = t` for `x`: `ITERATES(TAN(x), x, x, -1)` is `[x, ATAN(x)]`. Only a
    unique (or principal, see `_undone`) inverse is accepted, and systems have none.
    """
    if len(names) != 1:
        raise ValueError("no inverse of a system")
    (name,) = names
    point = sp.Dummy("t")
    undone = _undone(body, name, point)
    if undone is None:
        raise ValueError("not invertible")
    return undone.subs(point, name)


def _undone(body: sp.Basic, name: sp.Symbol, value: sp.Basic) -> sp.Basic | None:
    """`body = value` solved for `name`, on the principal branch if there are several.

    A periodic function has many solutions, but Derive gives the principal one:
    `INVERSE(SIN(x/b), x)` is `b*ASIN(x)` (9.21, p.285). So with multiple solutions the
    outermost call is undone by its principal inverse, then its argument recursively.

    Powers work the same way: an even power has two roots, and the principal one is
    the reciprocal power, as in `INT_SUBST(t*SIN(t^2), t, t^2)`, which substitutes the
    manual's `SQRT(t)` (9.21, p.285).

    Returns None if there is no known inverse or no solution.
    """
    solutions = sp.solve(sp.Eq(body, value), name)
    if len(solutions) == 1:
        return solutions[0]
    if isinstance(body, sp.Pow) and body.base.has(name) and not body.exp.has(name):
        return _undone(body.base, name, sp.Pow(value, 1 / body.exp))
    if not (isinstance(body, sp.Function) and len(body.args) == 1):
        return None
    inverse = _inverse_function(body)
    if inverse is None:
        return None
    return _undone(body.args[0], name, inverse(value))


#: Principal inverses of functions that aren't one-to-one, for which sympy gives no
#: inverse: `ASIN` for `SIN`, and so on.
_PRINCIPAL_INVERSES = {
    sp.sin: sp.asin,
    sp.cos: sp.acos,
    sp.sec: sp.asec,
    sp.csc: sp.acsc,
    sp.cosh: sp.acosh,
    sp.sech: sp.asech,
}


def _inverse_function(body: sp.Basic) -> Callable | None:
    """The inverse of the call `body`, or None."""
    principal = _PRINCIPAL_INVERSES.get(type(body))
    if principal is not None:
        return principal
    try:
        return body.inverse()
    except (AttributeError, ValueError, NotImplementedError):
        return None


def _elements_of(matrix: sp.MatrixBase) -> list[sp.Basic]:
    """A vector's elements, or a matrix's rows."""
    if matrix.rows == 1:
        return list(matrix)
    return [matrix[row, :] for row in range(matrix.rows)]


def vector_of(elements: list) -> sp.Basic:
    """A vector of computed values, shaped by the values alone.

    Unlike `_vector`, there is no notation to go by: rows of equal width stack into a
    matrix, plain expressions make one row, and anything else is an `InertVector`.
    """
    if not elements:
        return sp.Matrix(0, 0, [])
    if _all_matrices(elements):
        shapes = {element.shape for element in elements}
        if len(shapes) == 1 and elements[0].rows == 1 and elements[0].cols:
            return sp.Matrix([list(element) for element in elements])
        return InertVector(*elements)
    if not all(isinstance(element, sp.Expr) for element in elements):
        return InertVector(*elements)
    return sp.Matrix(1, len(elements), elements)


def _connected(word: str, operands: list[sp.Basic]) -> sp.Basic:
    """`AND` and the other logical operators, interpreted by their operands.

    Bitwise on integers, boolean on anything sympy accepts as a statement, and an inert
    `Logical` otherwise (e.g. an operand that is still a variable). Evaluation
    reconverts it once values are known: `LUCAS`'s `(n AND d_) = 0` tests a bit of `n`
    once iteration makes `d_` a number.
    """
    bitwise = _BITWISE.get(word)
    numbers = operands and all(isinstance(operand, sp.Integer) for operand in operands)
    if bitwise is not None and numbers:
        try:
            return sp.Integer(bitwise(*(int(operand) for operand in operands)))
        except TypeError:
            pass
    head = _BOOLEAN_HEADS.get(word)
    if head is not None:
        try:
            return head(*(_settled(operand) for operand in operands))
        except Exception:
            pass
    return Logical(sp.Symbol(word), *operands)


#: The inverse of `_HEADINGS`: operator name to sympy head.
_BOOLEAN_HEADS = {word: head for head, word in _HEADINGS.items()}


def _settled(operand: sp.Basic) -> sp.Basic:
    """A boolean operand with relations put in canonical form.

    A conjunction stores its operands as a set, ordered by spelling, and `a >= b` and
    `b <= a` spell the same statement differently. Ranges print as chains, which
    rewrites one into the other, so without canonicalization the order would change on
    every round trip.

    Nothing is decided or simplified; only the orientation of the relation changes.
    """
    return operand.canonical if isinstance(operand, Relational) else operand


def _identity_matrix(conv: Converter, args: list) -> sp.Basic:
    size = _one(args)
    if not isinstance(size, sp.Integer) or size < 1:
        raise ValueError("not a size")
    return sp.eye(int(size))


def _cross(conv: Converter, args: list) -> sp.Basic:
    """`CROSS(u, v)` for three-element or two-element vectors.

    In the plane, the result is the scalar third component of the spatial cross
    product (the other two being zero).
    """
    left, right = (_matrix(argument) for argument in args)
    if left.shape == right.shape == (1, 2):
        return left[0, 0] * right[0, 1] - left[0, 1] * right[0, 0]
    return left.cross(right)


def _row_reduce(conv: Converter, args: list) -> sp.Basic:
    """`ROW_REDUCE(A)`, and `ROW_REDUCE(A, B)` on the augmented matrix.

    The manual's echelon form is the reduced one (leading 1s with zeros above), as
    computed by `rref`.

    A second matrix is adjoined to the right, solving `A . X = B`: `B`'s columns end up
    holding `X` if `A` is nonsingular. A vector second argument is a single column, as
    in `A . X = b`, which VECTOR.MTH's `APPROX_EIGENVECTOR` uses.

    In Approximate mode the entries are evaluated numerically first. Pivoting asks
    whether entries are zero, and sympy simplifies undecided ones: for a Newton step
    with nested exponentials and arctangents, that means factoring a polynomial in
    several transcendental generators, which never finishes. Approximate numbers are
    rationals, so zero-testing is trivial.
    """
    matrix, *rest = args
    matrix = _worked_out(_matrix(matrix), conv.context)
    if not rest:
        return matrix.rref()[0]
    adjoined = _worked_out(_matrix(_one(rest)), conv.context)
    if adjoined.rows == 1 and matrix.rows != 1:
        adjoined = adjoined.T
    return sp.Matrix.hstack(matrix, adjoined).rref()[0]


def _characteristic_polynomial(conv: Converter, args: list) -> sp.Basic:
    """`CHARPOLY(A, v)`: `DET(A - v*IDENTITY_MATRIX(DIMENSION(A)))`.

    The manual defines it as this determinant, not the usual monic polynomial
    `DET(v*I - A)`; they differ by `(-1)^n`. Sympy computes the monic one efficiently,
    and the sign is adjusted for odd dimensions.

    The polynomial is computed over a dummy and `v` substituted afterwards, because
    `Poly` strips its generator's assumptions, and a `v` without its declared domain
    wouldn't cancel against the user's `v`.
    """
    matrix, variable = _matrix_and_variable(conv, args)
    polynomial = matrix.charpoly(sp.Dummy()).as_expr(variable)
    return polynomial if matrix.rows % 2 == 0 else -polynomial


def _eigenvalues(conv: Converter, args: list) -> sp.Basic:
    """`EIGENVALUES(A, v)`: the zeros of `A`'s characteristic polynomial in `v`.

    Returned as equations, `[z = 2, z = b]`, as Derive writes solutions elsewhere. Each
    zero appears once regardless of multiplicity.

    Sympy returns an unordered set, so sympy's canonical order is used for stable
    output; it matches the manual's examples (numbers before names, `a - b` before
    `a + b`).

    Zeros expressible only as `CRootOf` are rejected and the call stays as written: the
    manual uses the quadratic, cubic and quartic formulas and notes exact eigenvalues
    are rarely available beyond that.

    `trig=True` makes `roots` use Viete's cubic rather than Cardano's for the casus
    irreducibilis, so symmetric matrices, whose eigenvalues are real, don't get answers
    in `#i`.
    """
    matrix, variable = _matrix_and_variable(conv, args)
    zeros = sorted(matrix.eigenvals(trig=True), key=sp.default_sort_key)
    if any(zero.has(sp.CRootOf) for zero in zeros):
        raise ValueError("no closed form")
    return vector_of([sp.Eq(variable, zero, evaluate=False) for zero in zeros])


def _matrix_and_variable(conv: Converter, args: list) -> tuple[sp.MatrixBase, sp.Basic]:
    """A matrix and the variable for the answer.

    The variable defaults to `w`, as in the manual, with its declared domain. Sympy
    checks squareness; if it fails, the call stays as written.
    """
    matrix, *rest = args
    variable = _one(rest) if rest else conv.symbol("w")
    if type(variable) is not sp.Symbol:
        raise TypeError("not a variable")
    return _matrix(matrix), variable


def _gradient(conv: Converter, args: list) -> sp.Basic:
    """`GRAD(u)`, and `GRAD(u, alpha)` in the coordinate system `alpha`.

    The gradient of a scalar field: element `i` is the rate of growth of `u` per unit
    *length* along coordinate `i`, so each derivative is divided by its scale factor.
    In spherical coordinates a radian of colongitude is `r*SIN(phi)` of arc.

    For a vector field, the gradients of the elements are the columns of the result:
    element `[i][j]` is the rate of element `j` along coordinate `i`. That is the
    transpose of `VECTOR.MTH`'s `JACOBIAN(u, alpha)`. SOLVE.MTH's `NEWTONS` fixes this
    orientation: ``APPEND(GRAD(u, x), [u])` `` appends the field under the gradients and
    transposes, giving the augmented matrix `[J | u]` whose row reduction is a Newton
    step. The other orientation would solve against the transposed Jacobian and fail to
    converge.

    A vector hidden *inside* a head is refused. Converting the body of
    `VECTOR(GRAD(u SUB m_, alpha), m_, DIMENSION(u))` with `m_` still symbolic gives
    GRAD the inert `ELEMENT(u, m_)`, which carries the whole vector `u`; sympy's matrix
    chain rule would produce nonsense. The call stays as written, and evaluation calls
    it again once the index has a value.
    """
    expression, *rest = args
    if isinstance(expression, sp.MatrixBase):
        variables, scales = _coordinates(conv, rest)
        columns = [
            _gradient_of(element, variables, scales) for element in _field(expression)
        ]
        return vector_of([vector_of(list(row)) for row in zip(*columns, strict=True)])
    if _holds_vector(expression):
        raise TypeError("not an expression")
    variables, scales = _coordinates(conv, rest)
    return vector_of(_gradient_of(expression, variables, scales))


def _divergence(conv: Converter, args: list) -> sp.Basic:
    """`DIV(v)`, and `DIV(v, alpha)`: the divergence of the field `v`.

    The flux out of an infinitesimal coordinate box per unit volume. The volume is
    `H = h1*h2*...*hn` times the box's coordinate extent, and the faces crossed along
    coordinate `i` have area `H/hi` times theirs, which places the scale factors.
    """
    elements = _field(args[0])
    variables, scales = _coordinates(conv, args[1:], len(elements))
    return _divergence_of(elements, variables, scales)


def _laplacian(conv: Converter, args: list) -> sp.Basic:
    """`LAPLACIAN(u)`, and `LAPLACIAN(u, alpha)`: `DIV(GRAD(u))`.

    Computed from the manual's definition rather than a separate formula, so the scale
    factors are handled in one place.
    """
    expression, *rest = args
    if isinstance(expression, sp.MatrixBase):
        raise TypeError("not an expression")
    variables, scales = _coordinates(conv, rest)
    gradient = _gradient_of(expression, variables, scales)
    return _divergence_of(gradient, variables, scales)


def _curl(conv: Converter, args: list) -> sp.Basic:
    """`CURL(v)`, and `CURL(v, alpha)`, for a vector of two or three elements.

    The circulation around an infinitesimal loop per unit area, one loop per pair of
    coordinates. In space, the three circulations form a vector, each component
    labeled by the axis its loop turns about.

    In the plane there is a single loop, and Derive returns that scalar rather than
    `[0, 0, w]`, which the manual calls the more common convention.
    """
    elements = _field(args[0])
    if len(elements) not in (2, 3):
        raise ValueError("not a plane or space vector")
    variables, scales = _coordinates(conv, args[1:], len(elements))
    if len(elements) == 2:
        return _circulation(elements, variables, scales, 0, 1)
    turns = [_circulation(elements, variables, scales, *pair) for pair in _LOOPS]
    return vector_of(turns)


#: The coordinate pair for each component of a spatial curl: the next two coordinates
#: cyclically. This order (rather than sorted) gives the middle component its correct
#: sign, making the curl a right-handed cross product.
_LOOPS = ((1, 2), (2, 0), (0, 1))


def _potential(conv: Converter, args: list) -> sp.Basic:
    """`POTENTIAL(v)`, `POTENTIAL(v, a)` and `POTENTIAL(v, a, alpha)`.

    The line integral of `v` from `a` to the coordinates themselves, along a staircase
    path moving one coordinate at a time: leg `i` moves coordinate `i` from `a SUB i`,
    with earlier coordinates already at their final values and later ones still at
    `a`. For a conservative `v`, the result is a scalar whose gradient is `v`.

    For a non-conservative field, Derive still returns the integral along this path.
    The manual says POTENTIAL "merely computes a certain line integral", leaving it to
    the caller to check `GRAD` of the answer (or `CURL(v)` against zero).

    `a` defaults to the origin. The manual notes this can be a poor choice (starting
    where the field is infinite gives an infinite potential), which is what the second
    argument is for.
    """
    elements = _field(args[0])
    start, rest = _start(args[1:], len(elements))
    variables, scales = _coordinates(conv, rest, len(elements))
    walked = sp.Integer(0)
    for index, element in enumerate(elements):
        arrived = zip(variables[index + 1 :], start[index + 1 :])
        along = (scales[index] * element).subs(list(arrived), simultaneous=True)
        walked += _leg(along, variables[index], start[index])
    if walked is sp.nan:
        # Legs diverging in opposite directions have no sum.
        raise ValueError("no value")
    return walked


def _vector_potential(conv: Converter, args: list) -> sp.Basic:
    """`VECTOR_POTENTIAL(v)`, with the same optional `a` and `alpha`.

    A vector whose curl is the three-element `v`. Vector potentials differ by a
    gradient, so the manual's choice is used: the third element is zero, and the other
    two are the line integrals this forces.

    With `A SUB 3` zero, two curl equations become integrations along the third
    coordinate, fixing `h1*A SUB 1` and `h2*A SUB 2` up to a function of the first two
    coordinates. The third equation then only needs to hold on the slice
    `x3 = a SUB 3`, where those integrals vanish; that gives the second integral in the
    first element.

    As with POTENTIAL, a field with nonzero divergence still gets this vector; checking
    `CURL` of the answer is left to the caller.
    """
    elements = _field(args[0])
    if len(elements) != 3:
        raise ValueError("not a space vector")
    start, rest = _start(args[1:], 3)
    variables, scales = _coordinates(conv, rest, 3)
    (x1, x2, x3), (h1, h2, h3) = variables, scales
    first, second, third = elements
    lifted = _leg(h1 * h3 * second, x3, start[2])
    sliced = _leg((h1 * h2 * third).subs(x3, start[2]), x2, start[1])
    return vector_of(
        [
            (lifted - sliced) / h1,
            -_leg(h2 * h3 * first, x3, start[2]) / h2,
            sp.Integer(0),
        ]
    )


def _holds_vector(value: sp.Basic) -> bool:
    """Whether `value` contains a vector anywhere."""
    if isinstance(value, (sp.MatrixBase, InertVector)):
        return True
    return any(
        isinstance(found, (sp.MatrixBase, InertVector))
        for found in sp.preorder_traversal(value)
    )


def _gradient_of(expression: sp.Basic, variables: list, scales: list) -> list[sp.Basic]:
    return [sp.diff(expression, x) / h for x, h in zip(variables, scales)]


def _divergence_of(elements: list, variables: list, scales: list) -> sp.Basic:
    volume = sp.Mul(*scales)
    faces = zip(elements, variables, scales)
    flux = sum((sp.diff(volume / h * u, x) for u, x, h in faces), sp.Integer(0))
    return flux / volume


def _circulation(field: list, variables: list, scales: list, j: int, k: int) -> sp.Basic:
    """The circulation in the `j`, `k` coordinate plane, per unit area.

    The scale factors inside the derivatives turn components into side lengths; the
    one outside divides by the enclosed area.
    """
    along_k = sp.diff(scales[k] * field[k], variables[j])
    along_j = sp.diff(scales[j] * field[j], variables[k])
    return (along_k - along_j) / (scales[j] * scales[k])


def _leg(integrand: sp.Basic, variable: sp.Basic, low: sp.Basic) -> sp.Basic:
    """`INT(integrand, variable, low, variable)`, evaluated.

    The integration variable is a dummy since the upper limit is the coordinate
    itself. If sympy can't do the integral, the dummy would remain under an unreadable
    name, so this raises and the calling function stays as written.

    Singularities are handled as `INT` does: a leg starting where the field is infinite
    is `inf` or `-inf` as a whole, like `INT(1/t, t, 0, x)`, not an antiderivative plus
    infinity.
    """
    dummy = sp.Dummy()
    walked = integral(
        sp.Integral(integrand.subs(variable, dummy), (dummy, low, variable))
    )
    if walked.has(dummy):
        raise ValueError("no antiderivative")
    if walked.has(sp.oo, sp.zoo, sp.nan) and walked not in (sp.oo, -sp.oo):
        raise ValueError("no value")
    return walked


def _start(rest: list, count: int) -> tuple[list[sp.Basic], list]:
    """The starting point of the line integrals, and the remaining arguments.

    Defaults to the origin, the manual's "vector of zeros", with one coordinate per
    field element.
    """
    if not rest:
        return [sp.Integer(0)] * count, []
    given, *remaining = rest
    start = _field(given)
    if len(start) != count:
        raise ValueError("not a starting point")
    return start, remaining


def _coordinates(
    conv: Converter, rest: list, count: int | None = None
) -> tuple[list[sp.Basic], list[sp.Basic]]:
    """The coordinate variables and scale factors for a vector differential operator.

    Defaults to Cartesian `x`, `y`, `z`, with their declared domains.

    A vector of variables gives Cartesian coordinates of any dimension:
    `GRAD(u, [w, x, y, z])` is a four-element gradient. A two-row matrix gives
    variables above and scale factors `hi` below, with
    `ds^2 = (h1*dx1)^2 + ... + (hn*dxn)^2`; this describes any orthogonal curvilinear
    system, so the same formulas serve Cartesian, cylindrical and spherical coordinates.
    The names `cylindrical` and `spherical` are assignments in VECTOR.MTH.

    `count`, the number of field elements, must match the number of coordinates: Derive
    returns `DIV([x, y])` unchanged rather than guessing a two-dimensional system. The
    same goes for a two-element `CURL`, which needs an explicit planar system.
    """
    if rest:
        given = _matrix(_one(rest))
        if given.rows == 1:
            variables, scales = list(given), [sp.Integer(1)] * given.cols
        elif given.rows == 2:
            variables, scales = list(given[0, :]), list(given[1, :])
        else:
            raise ValueError("not a coordinate system")
    else:
        variables = [conv.symbol(name) for name in ("x", "y", "z")]
        scales = [sp.Integer(1)] * 3
    if not variables or any(type(name) is not sp.Symbol for name in variables):
        raise TypeError("not coordinate variables")
    if len(set(variables)) != len(variables):
        raise ValueError("a coordinate twice")
    if count is not None and len(variables) != count:
        raise ValueError("not that many coordinates")
    return variables, scales


def _field(value: sp.Basic) -> list[sp.Basic]:
    """A vector field's components; matrices are refused."""
    matrix = _matrix(value)
    if matrix.rows != 1 or not matrix.cols:
        raise TypeError("not a vector")
    return list(matrix)


def _matrix(value: sp.Basic) -> sp.MatrixBase:
    if not isinstance(value, sp.MatrixBase):
        raise TypeError("not a matrix")
    return value


#: Converters for each function name. Other names become inert heads, which keeps user
#: functions intact through a round trip and makes `DIF(F(x)^3, x)` differentiable.
#: `SOLVE` is absent because it is in `COMMAND_HEADS`, run by the evaluator.
FUNCTIONS: dict[str, Handler] = {
    **{name: _trig(func) for name, func in _TRIGONOMETRIC.items()},
    **{name: _arctrig(func) for name, func in _INVERSE_TRIGONOMETRIC.items()},
    **{name: _direct(func) for name, func in _DIRECT.items()},
    "ATAN": _atan,
    "ACOT": _acot,
    "PHASE": _phase,
    "ERF": _erf,
    "ABS": _absolute,
    "SIGN": _sign,
    "STEP": _step,
    "LOG": _log,
    "FLOOR": _floor,
    "MOD": _mod,
    "MODS": _mods,
    "CHI": _chi,
    "NORMAL": _normal,
    "PVAL": _present_value,
    "FVAL": _future_value,
    "PMT": _payment,
    "NPER": _periods,
    "RATE": _rate,
    "RANDOM": _random,
    # Per section 6.13 these take arguments like a statistic: any number of values, a
    # vector, or a matrix whose rows give a vector of answers.
    "GCD": _statistic(_gcd_of),
    "LCM": _statistic(_lcm_of),
    "AVERAGE": _statistic(_average),
    "RMS": _statistic(_root_mean_square),
    "VAR": _statistic(_variance),
    "STDEV": _statistic(_standard_deviation),
    "FIT": _fit,
    "NUMBER": _is_number,
    "PRIME": _is_prime,
    "NEXT_PRIME": _next_prime,
    "NUMERATOR": _numerator,
    "DENOMINATOR": _denominator,
    "TERMS": _terms,
    "FACTORS": _factors,
    "LHS": _side(0),
    "RHS": _side(1),
    "QUOTIENT": _quotient,
    "REMAINDER": _remainder,
    "POLY_GCD": _polynomial_gcd,
    "VARIABLES": _variables,
    "DIF": _dif,
    "INT": _integral,
    "SUM": _summation,
    "PRODUCT": _product,
    "LIM": _limit,
    "INTERVAL": _interval,
    "ROOT_SUM": _root_sum,
    "ROOT_OF": _root_of,
    "TAYLOR": _taylor,
    "APPROX": _approximation,
    "IF": _conditional,
    "TRUTH_TABLE": _truth_table,
    "SUBS": _substitution,
    "DET": _determinant,
    "TRACE": _trace,
    "DIMENSION": _dimension,
    "ELEMENT": _element_of,
    "DELETE_ELEMENT": _delete_element,
    "REPLACE_ELEMENT": _replace_element,
    "INSERT_ELEMENT": _insert_element,
    "REVERSE_VECTOR": _reversed_vector,
    "APPEND": _appended,
    "IDENTITY_MATRIX": _identity_matrix,
    "CROSS": _cross,
    "ROW_REDUCE": _row_reduce,
    "CHARPOLY": _characteristic_polynomial,
    "EIGENVALUES": _eigenvalues,
    "VECTOR": _generated_vector,
    "SELECT": _selected,
    "ITERATE": _iterate,
    "ITERATES": _iterates,
    "GRAD": _gradient,
    "DIV": _divergence,
    "LAPLACIAN": _laplacian,
    "CURL": _curl,
    "POTENTIAL": _potential,
    "VECTOR_POTENTIAL": _vector_potential,
}


#: Functions that map over an equation argument (section 4.13 p.111):
#: `EXP(LN(x) = 5)` is `x = #e^5`. These are all functions of numbers, for which
#: applying to both sides is the only sensible meaning.
#:
#: Functions that take an equation as such are excluded: `LHS` and `RHS` take a side,
#: `SOLVE` solves it, and `IF` tests it. Unknown names are excluded too, since a user
#: function isn't known to be a function of numbers.
MAPPED_OVER_EQUATIONS: frozenset[str] = (
    frozenset(_TRIGONOMETRIC)
    | frozenset(_INVERSE_TRIGONOMETRIC)
    | frozenset(_DIRECT)
    | {"ATAN", "ACOT", "ERF", "LOG", "SIGN", "STEP", "FLOOR", "MOD", "MODS"}
)

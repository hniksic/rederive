"""Proofs that an answer to an integral is correct.

No sympy answer is trusted. An antiderivative is kept only if its derivative is proved
equal to the integrand, and its endpoint difference is taken as a definite integral
only if it is proved continuous between the endpoints.
"""

from __future__ import annotations

from collections.abc import Callable

import sympy as sp

from rederive.engine.rewriting import Rewrite, attempt

__all__ = [
    "Stretch",
    "agrees",
    "agrees_at",
    "differentiates_back",
    "holds",
    "proved_zero",
    "unbroken",
]


def unbroken(
    expression: sp.Basic, variable: sp.Basic, low: sp.Basic, high: sp.Basic
) -> bool:
    """Whether the expression is proved continuous in the variable on the open interval
    from `low` to `high`, off every branch cut, for every real parameter value at which
    it is defined.

    This is what makes an antiderivative's endpoint difference equal the integral. A
    jump (a logarithm whose argument crosses zero) makes the difference miss it, and on
    a branch cut sympy differentiates from one side: `POLYLOG(2, z)` past 1
    differentiates to `-LN(1 - z)/z` with the wrong sign of `#i*pi`. A logarithm or
    power of an argument negative throughout stays on one side of its cut, so it counts
    as continuous.

    The proof follows the expression's structure, as described in `Stretch`.
    """
    span = attempt(high, lambda h: h - low)
    if span is None or span.is_extended_positive is not True:
        return False
    return Stretch(variable, low, high).continuous(expression)


class Stretch:
    """An open interval of a real variable, and proofs about expressions over it.

    An expression is continuous if it is a constant, the variable, or a sum, product or
    function of continuous expressions such that:

    - a power with a constant exponent has a nonzero base, a nonnegative integer
      exponent, or a positive exponent and a real base;
    - a power with the variable in its exponent has a positive base, or a nonzero
      constant one;
    - entire functions (`ENTIRE`, and the absolute value) take any argument; `REAL`
      functions a real one; the quotients in `QUOTIENTS` a real one off the poles;
    - `NONZERO` functions take a real nonzero argument; the cosine integrals and the
      logarithmic integral a positive one (the latter not 1);
    - the arcsine and arccosine take an argument in [-1, 1], the inverse hyperbolic
      cosine one at least 1, the inverse hyperbolic tangent and cotangent one inside
      and outside (-1, 1), and `POLYLOG(s, z)` a real `z` below 1.

    An expression is real if built from real constants, the variable, and these
    functions at real arguments, with roots of nonnegative and logarithms of positive
    arguments.

    Signs are decided by sympy's assumptions, with the variable written as an endpoint
    plus or minus a positive offset `p`, or as the point `p/(1 + p)` of the way along.
    Failing that, a parameter-free argument is decided by `solveset` on the interval
    and its sign at the middle.
    """

    #: The functions continuous for any continuous argument.
    ENTIRE = (
        sp.exp, sp.sin, sp.cos, sp.sinh, sp.cosh, sp.erf, sp.erfc, sp.erfi, sp.Si,
        sp.Shi, sp.fresnels, sp.fresnelc,
    )
    #: The functions continuous and real for a continuous real argument.
    REAL = (sp.atan, sp.asinh, sp.tanh)
    #: The quotients, each mapped to the function whose zeros are its poles.
    QUOTIENTS = {sp.tan: sp.cos, sp.sec: sp.cos, sp.cot: sp.sin, sp.csc: sp.sin}
    #: The functions continuous for a continuous real argument that is not zero.
    NONZERO = (sp.log, sp.Ei, sp.sign, sp.acot, sp.coth)

    def __init__(self, variable: sp.Basic, low: sp.Basic, high: sp.Basic):
        self.variable, self.low, self.high = variable, low, high
        offset = sp.Dummy("p", positive=True)
        self.covers = []
        if low.is_finite and high.is_finite:
            self.covers.append(low + (high - low) * offset / (1 + offset))
        if low.is_finite:
            self.covers.append(low + offset)
        if high.is_finite:
            self.covers.append(high - offset)
        if not self.covers:
            self.covers.append(sp.Dummy("r", real=True))
        if low.is_finite and high.is_finite:
            self.middle = (low + high) / 2
        elif low.is_finite:
            self.middle = low + 1
        elif high.is_finite:
            self.middle = high - 1
        else:
            self.middle = sp.S.Zero
        self.known: dict[tuple[str, sp.Basic], bool] = {}

    def _remembered(self, kind: str, expression: sp.Basic, decide) -> bool:
        key = (kind, expression)
        if key not in self.known:
            self.known[key] = False
            self.known[key] = bool(decide(expression))
        return self.known[key]

    def continuous(self, expression: sp.Basic) -> bool:
        return self._remembered("continuous", expression, self._continuous)

    def real(self, expression: sp.Basic) -> bool:
        return self._remembered("real", expression, self._real)

    def positive(self, expression: sp.Basic) -> bool:
        return self._remembered("positive", expression, self._positive)

    def nonnegative(self, expression: sp.Basic) -> bool:
        return self._remembered("nonnegative", expression, self._nonnegative)

    def nonzero(self, expression: sp.Basic) -> bool:
        return self._remembered("nonzero", expression, self._nonzero)

    def _continuous(self, expression: sp.Basic) -> bool:
        if not expression.has(self.variable) or expression == self.variable:
            return True
        if isinstance(expression, (sp.Add, sp.Mul)):
            return all(self.continuous(term) for term in expression.args)
        if isinstance(expression, sp.Pow):
            base, exponent = expression.args
            if not self.continuous(base):
                return False
            if exponent.has(self.variable):
                if not self.continuous(exponent):
                    return False
                if base.has(self.variable):
                    return self.positive(base)
                return base.is_zero is False
            if exponent.is_integer and exponent.is_nonnegative:
                return True
            if exponent.is_integer:
                return self.nonzero(base)
            if not (self.real(base) and self.real(exponent)):
                return False
            return exponent.is_positive is True or self.nonzero(base)
        if not isinstance(expression, sp.Function):
            return False
        arguments = expression.args
        if isinstance(expression, sp.polylog):
            order, argument = arguments
            return (
                not order.has(self.variable)
                and self.continuous(argument)
                and self.real(argument)
                and self.positive(1 - argument)
            )
        if len(arguments) != 1:
            return False
        (argument,) = arguments
        if not self.continuous(argument):
            return False
        if isinstance(expression, self.ENTIRE + (sp.Abs,)):
            return True
        if not self.real(argument):
            return False
        if isinstance(expression, self.REAL):
            return True
        for quotient, denominator in self.QUOTIENTS.items():
            if isinstance(expression, quotient):
                return self.nonzero(denominator(argument))
        if isinstance(expression, self.NONZERO):
            return self.nonzero(argument)
        return self._real_kernel(expression, argument)

    def _real_kernel(self, expression: sp.Basic, argument: sp.Basic) -> bool:
        """Whether a function that is real on only part of the real line is continuous and
        real at this argument.
        """
        if isinstance(expression, (sp.asin, sp.acos)):
            return self.nonnegative(1 - argument**2)
        if isinstance(expression, sp.acosh):
            return self.nonnegative(argument - 1)
        if isinstance(expression, sp.atanh):
            return self.positive(1 - argument**2)
        if isinstance(expression, sp.acoth):
            return self.positive(argument**2 - 1)
        if isinstance(expression, (sp.Ci, sp.Chi)):
            return self.positive(argument)
        if isinstance(expression, sp.li):
            return self.positive(argument) and self.nonzero(argument - 1)
        return False

    def _real(self, expression: sp.Basic) -> bool:
        if expression == self.variable:
            return True
        if not isinstance(expression, (sp.Add, sp.Mul, sp.Pow, sp.Function)):
            return expression.is_extended_real is True
        if isinstance(expression, (sp.Add, sp.Mul)):
            return all(self.real(term) for term in expression.args)
        if isinstance(expression, sp.Pow):
            base, exponent = expression.args
            if not (self.real(base) and self.real(exponent)):
                return False
            return exponent.is_integer is True or self.nonnegative(base)
        if isinstance(expression, sp.polylog):
            order, argument = expression.args
            return self.real(order) and self.real(argument) and self.positive(
                1 - argument
            )
        if len(expression.args) != 1 or not self.real(expression.args[0]):
            return False
        argument = expression.args[0]
        if isinstance(expression, sp.log):
            return self.positive(argument)
        if isinstance(
            expression,
            self.ENTIRE + self.REAL + tuple(self.QUOTIENTS) + (sp.Abs, sp.sign),
        ) or isinstance(expression, (sp.Ei, sp.acot, sp.coth)):
            return True
        return self._real_kernel(expression, argument)

    def _assumed(self, expression: sp.Basic, ask: Callable[[sp.Basic], bool]) -> bool:
        """Whether sympy's assumptions answer yes over one of the interval's covers."""
        if not expression.has(self.variable):
            return bool(attempt(expression, lambda e: sp.S(ask(e) is True)))
        for cover in self.covers:
            written = attempt(expression, lambda e: e.subs(self.variable, cover))
            if written is None:
                continue
            for form in (written, attempt(written, sp.together)):
                if form is not None and attempt(form, lambda e: sp.S(ask(e) is True)):
                    return True
        return False

    def _nowhere(self, relation: sp.Basic) -> bool:
        """Whether `solveset` finds that a relation in the variable alone holds nowhere
        on the interval."""
        if relation.free_symbols != {self.variable}:
            return False
        inside = sp.Interval.open(self.low, self.high)
        found = attempt(relation, lambda r: sp.solveset(r, self.variable, inside))
        return found is not None and found.is_empty is True

    def _plain(self, expression: sp.Basic) -> bool:
        """Whether the expression is a continuous real function of the variable alone."""
        return (
            expression.free_symbols == {self.variable}
            and self.continuous(expression)
            and self.real(expression)
        )

    def _positive(self, expression: sp.Basic) -> bool:
        if self._assumed(expression, lambda e: e.is_positive):
            return True
        if not self._plain(expression) or not self._nowhere(sp.Eq(expression, 0)):
            return False
        at_middle = attempt(expression, lambda e: e.subs(self.variable, self.middle))
        return at_middle is not None and at_middle.is_positive is True

    def _nonnegative(self, expression: sp.Basic) -> bool:
        if self._assumed(expression, lambda e: e.is_nonnegative):
            return True
        return self._plain(expression) and self._nowhere(sp.Lt(expression, 0))

    def _nonzero(self, expression: sp.Basic) -> bool:
        if self._assumed(expression, lambda e: e.is_zero is False):
            return True
        if isinstance(expression, sp.Mul):
            return all(self.nonzero(factor) for factor in expression.args)
        if isinstance(expression, sp.Pow) and not expression.exp.has(self.variable):
            return self.nonzero(expression.base)
        if isinstance(expression, sp.exp):
            return True
        conjugate = self._conjugate(expression)
        if conjugate is not None and self.continuous(expression):
            product = attempt(expression * conjugate, sp.expand)
            if product is not None and self.nonzero(product):
                return True
        return self._plain(expression) and self._nowhere(sp.Eq(expression, 0))

    def _conjugate(self, expression: sp.Basic) -> sp.Basic | None:
        """The sum with the sign of its single square-root term flipped, or None.

        Since `(p + c*SQRT(R))*(p - c*SQRT(R)) = p^2 - c^2*R`, the sum is nonzero
        wherever that product is and the sum is continuous.
        """
        if not isinstance(expression, sp.Add):
            return None
        rooted = [
            term
            for term in expression.args
            if any(
                power.exp == sp.S.Half and power.base.has(self.variable)
                for power in term.atoms(sp.Pow)
            )
        ]
        if len(rooted) != 1:
            return None
        return expression - 2 * rooted[0]


def agrees(candidate: sp.Basic, answer: sp.Basic) -> bool:
    """Whether a candidate is the answer, decided rather than merely not refuted."""
    if candidate.has(sp.nan, sp.zoo):
        return False
    if answer is sp.nan or answer.is_infinite or candidate.is_infinite:
        return candidate == answer
    difference = attempt(candidate, lambda c: sp.simplify(c - answer))
    return difference == 0


def agrees_at(
    candidate: sp.Basic, parameter: sp.Basic, point: sp.Basic, answer: sp.Basic
) -> bool:
    """Whether a candidate is the answer at a point, or tends to it there from both
    sides where the answer is infinite and the candidate has a pole there."""
    value = attempt(candidate, lambda c: c.subs(parameter, point))
    if value is not None and agrees(value, answer):
        return True
    if answer is sp.nan or not answer.is_infinite:
        return False
    return all(
        attempt(candidate, lambda c: sp.limit(c, parameter, point, side)) == answer
        for side in ("+", "-")
    )


def differentiates_back(
    value: sp.Basic, integrand: sp.Basic, variable: sp.Basic
) -> bool:
    """Whether the answer is proved an antiderivative of the integrand, checking each
    piecewise case where it applies.

    Sympy's Risch answers `INT(1/(x^2 + a), x)` with zero for real `a`, and its Meijer
    G route gets `INT(SIN(x)/x^2, x)` wrong for negative `x`, so nothing is trusted.

    A case applies where its condition holds and earlier ones don't. If that pins a
    parameter (`a = 0` after a case for `a /= 0`), the case is proved at that value. If
    it pins the variable to isolated points, or the integrand is undefined at the
    pinned values (`COT(a*x)` at `a = 0`), nothing needs proving. Every other case must
    hold unconditionally.
    """
    if not integrand.has(variable) and value == integrand * variable:
        # A constant integrand, which may be an interval that sympy's derivative
        # doesn't map back to itself.
        return True
    folded = attempt(value, sp.piecewise_fold)
    if folded is None:
        return False
    if not isinstance(folded, sp.Piecewise):
        return holds(folded, integrand, variable)
    earlier: list[sp.Basic] = []
    for case, condition in folded.args:
        where = attempt(
            sp.And(condition, *(sp.Not(before) for before in earlier)), sp.to_dnf
        )
        earlier.append(condition)
        if where is None:
            return False
        for region in sp.Or.make_args(where) if where is not sp.false else ():
            if not _holds_in(region, case, integrand, variable):
                return False
    return True


def _holds_in(
    region: sp.Basic, case: sp.Basic, integrand: sp.Basic, variable: sp.Basic
) -> bool:
    """Whether a case is proved an antiderivative where a conjunction of conditions holds;
    see `differentiates_back`.

    An equation pins a parameter if it involves only that parameter (`LN(a) = 0` pins
    `a` to 1) or equates it to something free of it. With several solutions, each is
    proved. Equations that pin nothing are ignored.
    """
    conditions = sp.And.make_args(region)
    if any(isinstance(c, sp.Eq) and c.has(variable) for c in conditions):
        return True
    choices: list[dict[sp.Basic, sp.Basic]] = [{}]
    for condition in conditions:
        if not isinstance(condition, sp.Eq):
            continue
        widened = []
        for pinned in choices:
            values = _pins(attempt(condition, lambda c: c.subs(pinned)), variable)
            if values is None:
                widened.append(pinned)
                continue
            widened.extend({**pinned, **value} for value in values)
        choices = widened
    for pinned in choices:
        if not pinned:
            if not holds(case, integrand, variable):
                return False
            continue
        if any(
            attempt(condition, lambda c: c.subs(pinned)) is sp.false
            for condition in conditions
        ):
            continue
        there = attempt(integrand, lambda f: f.subs(pinned))
        written = attempt(case, lambda c: c.subs(pinned))
        if there is None or written is None:
            return False
        if there.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
            continue
        if not holds(written, there, variable):
            return False
    return True


def _pins(
    equation: sp.Basic | None, variable: sp.Basic
) -> list[dict[sp.Basic, sp.Basic]] | None:
    """All values an equation pins one of the parameters to, or None if it pins none;
    see `_holds_in`."""
    if equation is sp.true or not isinstance(equation, sp.Eq):
        return [] if equation is sp.false else None
    left, right = equation.args
    for symbol, other in ((left, right), (right, left)):
        if isinstance(symbol, sp.Symbol) and not other.has(symbol):
            return [{symbol: other}]
    symbols = equation.free_symbols - {variable}
    if len(symbols) != 1:
        return None
    (symbol,) = symbols
    try:
        found = sp.solve(left - right, symbol, dict=True)
    except Exception:
        return None
    return found or None


def holds(case: sp.Basic, integrand: sp.Basic, variable: sp.Basic) -> bool:
    """Whether this unconditional case's derivative is proved equal to the integrand."""
    if case.has(sp.Piecewise):
        return False
    difference = attempt(case, lambda c: sp.diff(c, variable) - integrand)
    return difference is not None and proved_zero(difference)


#: Size limit, in operations, for a difference passed to `sympy.simplify` after the
#: cheap steps fail.
_SIMPLIFIED_OPERATIONS = 400

#: Cheap exact steps tried before `sympy.simplify`: put over a common denominator,
#: plain and after expanding special functions, rewriting trigonometric and hyperbolic
#: functions as exponentials, or rationalizing denominators.
_ZERO_STEPS: tuple[Rewrite, ...] = (
    lambda d: sp.cancel(sp.together(d)),
    lambda d: sp.cancel(sp.together(sp.expand_func(d))),
    lambda d: sp.cancel(sp.together(d.rewrite(sp.exp))),
    lambda d: sp.cancel(sp.together(sp.radsimp(sp.together(d)))),
)


def proved_zero(difference: sp.Basic) -> bool:
    """Whether the difference is proved zero by exact rewriting alone.

    The cheap `_ZERO_STEPS` settle most cases. `sympy.simplify` is the last resort,
    used only on small differences because its cost grows fast. Numeric evaluation is
    never used as evidence.
    """
    if difference == 0:
        return True
    if any(attempt(difference, step) == 0 for step in _ZERO_STEPS):
        return True
    try:
        if sp.count_ops(difference) > _SIMPLIFIED_OPERATIONS:
            return False
    except Exception:
        return False
    simplified = attempt(difference, sp.simplify)
    if simplified is None:
        return False
    return simplified == 0 or any(
        attempt(simplified, step) == 0 for step in _ZERO_STEPS
    )

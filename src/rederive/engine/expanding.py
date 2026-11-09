"""Writing an expression as a sum: the rewrites, and nothing else.

The mathematics of Derive's Expand, sympy expression in and out. It sits below
`pipeline`, which uses it to evaluate an authored `EXPAND`; the Expand command is
`expand.py` on top of both.

Simplify's normal form (in `normal`) does the same about one generator at a time, so
both commands treat everything free of the variables as one opaque quantity.

The manual defines the goal as maximizing the number of terms algebraically
independent with respect to the *expansion variables*, and this module takes that
literally. Everything free of those variables is opaque and not expanded; the rest is
fully expanded, and terms with the same powers of the expansion variables are
collected. So `(x + 2*y + 1)^3` about `x` gives powers of `2*y + 1`, not a polynomial in
`y`, and `3*a^2*x^2*y^3 + 7*(b + 1)^2*x^2*y^3` about `x` and `y` collects into
`x^2*y^3*(3*a^2 + 7*(b + 1)^2)`, keeping `(b + 1)^2`.

A ratio whose denominator contains an expansion variable gets partial fraction
expansion instead: long division if the numerator's degree allows, then factoring the
denominator by the *amount* (the only reason Expand asks for one), then splitting the
proper part over the factors by undetermined coefficients. Sympy's `apart` isn't used
because the factors may be radical: under raDical, `1/(x^2 - 2)` splits over `x -
SQRT(2)` and `x + SQRT(2)`, which no decomposition over the rationals reaches.

There is no `Complex` amount, since Derive's Expand menu lists only four.

Exponents are never expanded: `2^(x + 1)` is not `2*2^x` to Derive. Trigonometric,
exponential and logarithmic expansion belong to the Trigonometry and Exponential
options, per the manual.

Every transformation runs inside a `try` and a failure keeps the previous form, so
like Simplify this never turns a valid entry into an error.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import sympy as sp

from rederive.engine.boundary import DEFAULT_AMOUNT, Amount
from rederive.engine.factoring import Finish, factored_expression
from rederive.engine.ordering import ORDER_LIST, main_order
from rederive.engine.shape import attempt, distributed

__all__ = ["EXPAND_AMOUNTS", "expanded_expression"]

#: The amounts Expand offers, which are Factor's without `Complex`.
EXPAND_AMOUNTS = (Amount.TRIVIAL, Amount.SQUAREFREE, Amount.RATIONAL, Amount.RADICAL)


def expanded_expression(
    expression: sp.Basic,
    amount: Amount = DEFAULT_AMOUNT,
    variables: Sequence[str] = (),
    finish: Finish | None = None,
    order: Sequence[str] = ORDER_LIST,
) -> sp.Basic:
    """`expression` written as a sum, about `variables`.

    `variables` names the expansion variables in the order chosen. Empty means all of
    them, per the manual, unlike in `factoring`, where empty lets sympy pick. `order`
    is the session's order list, used to order "all of them".

    Each scalar in the expression is expanded and then passed through `finish`, so
    rounding applies to the terms rather than to the original.

    Never raises on output of the conversion layer; a failed rewrite is skipped.
    """
    request = _Request(amount, tuple(variables), finish, tuple(order))
    return distributed(expression, lambda scalar: _leaf(scalar, request))


@dataclass(frozen=True)
class _Request:
    """The expansion parameters, passed down to each scalar."""

    amount: Amount
    variables: tuple[str, ...]
    finish: Finish | None
    order: tuple[str, ...]


def _leaf(expression: sp.Basic, request: _Request) -> sp.Basic:
    """One scalar, expanded about the chosen variables it contains."""
    if not isinstance(expression, sp.Expr):
        return expression
    variables = _generators(expression, request.variables, request.order)
    if variables:
        expression = attempt(
            expression, lambda e: _expanded(e, variables, request.amount)
        )
    return request.finish(expression) if request.finish else expression


def _generators(
    expression: sp.Expr, variables: Sequence[str], order: Sequence[str]
) -> tuple[sp.Symbol, ...]:
    """The expansion variables, as the symbols in this expression.

    The caller names them as text (from a prompt); resolving them against the
    expression gets symbols carrying their declared domain assumptions. Names not in
    the expression are dropped, so `(x + 1)^2` is left alone when expanding about `y`.

    Empty means all variables, most main first, so the primary variable is the one
    Derive would choose.
    """
    # `type is` rather than `isinstance`: string literals are a `Symbol` subclass, and
    # aren't variables.
    free = {
        symbol.name: symbol
        for symbol in expression.free_symbols
        if type(symbol) is sp.Symbol
    }
    chosen = variables if variables else main_order(free, order)
    return tuple(free[name] for name in chosen if name in free)


def _expanded(
    expression: sp.Expr, variables: tuple[sp.Symbol, ...], amount: Amount
) -> sp.Expr:
    """One scalar, by whichever of the two expansions applies.

    Partial fractions are used only if an expansion variable is in the denominator
    (`x/(a^2 - 1)` about `x` has nothing to split over) and the split succeeds.
    Otherwise it is a polynomial expansion.
    """
    _, denominator = sp.fraction(expression)
    if denominator.free_symbols & set(variables):
        split = _partial_fractions(expression, variables, amount)
        if split is not None:
            return split
    return _polynomial(expression, variables)


# -- polynomial expansion ----------------------------------------------------


def held_about(
    expression: sp.Expr, variable: sp.Symbol
) -> tuple[sp.Expr, dict[sp.Dummy, sp.Expr]]:
    """`expression` with everything free of `variable` replaced by placeholders, and
    the map back.

    Public so the reversibility guarantee can be tested. `xreplace` with the map
    restores the placeholders.

    Restoring must give back the original expression, and this is checked rather
    than trusted. `_frozen` rebuilds unknown heads from rewritten arguments, and a
    head that doesn't survive that loses content: a `RootSum` once came back without
    its sum, yielding an antiderivative missing half of itself. `_BOUND` lists the
    known cases, but the check makes an unknown one leave the expression unchanged
    rather than produce a wrong answer. On failure the whole expression becomes one
    placeholder, leaving the normal form nothing to do.
    """
    held: dict[sp.Dummy, sp.Expr] = {}
    frozen = _frozen(expression, {variable}, held)
    try:
        reversible = frozen.xreplace(held) == expression
    except Exception:
        reversible = False
    if reversible:
        return frozen, held
    whole: dict[sp.Dummy, sp.Expr] = {}
    return _held(expression, whole), whole


def _polynomial(expression: sp.Expr, variables: tuple[sp.Symbol, ...]) -> sp.Expr:
    """Multiplied out about `variables`, then collected by their powers."""
    held: dict[sp.Dummy, sp.Expr] = {}
    frozen = _frozen(expression, set(variables), held)
    # Expand doesn't split powers: `x^(a + b)` stays, as does `(x*y)^n`.
    multiplied = sp.expand(frozen, power_exp=False, power_base=False)
    return _collected(multiplied, variables).xreplace(held)


def _frozen(
    expression: sp.Expr, variables: set[sp.Symbol], held: dict[sp.Dummy, sp.Expr]
) -> sp.Expr:
    """`expression` with everything free of `variables` replaced by placeholders.

    Placeholders make coefficients opaque so multiplying out can't reach inside. In a
    sum or product, all parts free of the variables share *one* placeholder, so `x +
    2*y + 1` becomes `x + d` and `(x + 2*y + 1)^3` about `x` is written in powers of
    `2*y + 1`.

    A product's numeric factor stays outside the placeholder, but a sum's numeric term
    goes inside. In a product the number is a coefficient for collecting to find, and
    with the `2` inside, `2*x*(a + b)` would become `x*(2*a + 2*b)`. In a sum it is
    part of the opaque quantity; otherwise `3*x*(2*y + 1)^2` would become `3*x*(4*y^2
    + 4*y + 1)`.
    """
    if not (expression.free_symbols & variables):
        return _held(expression, held)
    if isinstance(expression, (sp.Add, sp.Mul)):
        involved, free = [], []
        for argument in expression.args:
            (involved if argument.free_symbols & variables else free).append(argument)
        parts = [_frozen(argument, variables, held) for argument in involved]
        if isinstance(expression, sp.Mul):
            parts.extend(argument for argument in free if argument.is_Number)
            free = [argument for argument in free if not argument.is_Number]
        if free:
            parts.append(_held(expression.func(*free), held))
        return expression.func(*parts)
    if isinstance(expression, sp.Pow):
        # Only the base is expanded. A non-numeric exponent gets a placeholder too,
        # since sympy would otherwise turn `2^(x + 1)` into `2*2^x`, which Derive's
        # Expand doesn't do.
        base, exponent = expression.args
        if not exponent.is_Number:
            exponent = _held(exponent, held)
        return sp.Pow(_frozen(base, variables, held), exponent)
    if isinstance(expression, _BOUND):
        return _held(expression, held)
    if expression.args:
        return expression.func(
            *(_frozen(argument, variables, held) for argument in expression.args)
        )
    return expression


#: Heads that bind their own variable. Each is replaced whole by a placeholder,
#: expansion variables and all, for two reasons.
#:
#: Not all their arguments are operands: a derivative carries its variable and
#: order, a substitution its bound variables and points, a root sum its polynomial
#: and shared generator. These are tuples, lambdas and dummies, and rebuilding from
#: rewritten arguments misreads them: `DIF(u, xi)` becomes `DIF(u, [xi, 1])`,
#: `SUBS(u, [xi], [y])` becomes `SUBS(u, [[xi]], [y])`, and a root sum vanishes.
#:
#: Also, their contents shouldn't be expanded anyway. `sp.expand` is deep, so it
#: would multiply out an integrand, splitting the integral into one per term.
_BOUND = (
    sp.Derivative,
    sp.Integral,
    sp.Sum,
    sp.Product,
    sp.Limit,
    sp.Subs,
    sp.RootSum,
)


def _held(expression: sp.Expr, held: dict[sp.Dummy, sp.Expr]) -> sp.Expr:
    """A placeholder for `expression`, or the expression itself if it is a number.

    Numbers aren't held, so arithmetic like `(2 + 3)^2` still evaluates to 25.
    """
    if expression.is_number:
        return expression
    placeholder = sp.Dummy()
    held[placeholder] = expression
    return placeholder


def _collected(
    expression: sp.Expr,
    variables: tuple[sp.Symbol, ...],
) -> sp.Expr:
    """`expression` with terms of equal powers of the expansion variables collected.

    Per the manual, `a*(x + 1)^2 + b*(x + 1)^2` about `x` is `x^2*(a + b) + 2*x*(a +
    b) + a + b`, not six terms.

    A collected coefficient keeps its common factor outside: `2*x*(a + b)` rather than
    `x*(2*a + 2*b)`, as in the manual's expansion about `y` giving `12*y^2*(x + 1)`.
    The normal form does the opposite, keeping numeric content inside.
    """
    if not isinstance(expression, sp.Add):
        return expression
    groups: dict[tuple[sp.Expr, ...], list[sp.Expr]] = {}
    for term in expression.args:
        degrees, rest = _degrees(term, variables)
        groups.setdefault(degrees, []).append(rest)
    return sp.Add(
        *(
            sp.Mul(*(base**power for base, power in zip(variables, degrees)))
            * _coefficient(rest, degrees)
            for degrees, rest in groups.items()
        )
    )


def _coefficient(terms: list[sp.Expr], degrees: tuple[sp.Expr, ...]) -> sp.Expr:
    """The sum of the coefficients of one degree, with any common factor outside.

    Not done at degree zero, which holds everything free of the expansion variables
    (possibly the whole expression); pulling out a factor there would be factoring.
    So `2*SIN(x) + 2*COS(x)` is left as is, and `LN(x^2 - x) - LN(x)` isn't turned
    into `LN(x*(x - 1)) - LN(x)` by `factor_terms` reaching inside the logarithm.
    """
    if len(terms) == 1 or not any(degrees):
        return sp.Add(*terms)
    return sp.factor_terms(sp.Add(*terms))


def _degrees(
    term: sp.Expr, variables: tuple[sp.Symbol, ...]
) -> tuple[tuple[sp.Expr, ...], sp.Expr]:
    """One term split into its powers of the expansion variables and the rest."""
    powers: dict[sp.Symbol, sp.Expr] = {}
    rest: list[sp.Expr] = []
    for factor in sp.Mul.make_args(term):
        base, exponent = factor.as_base_exp()
        if base in variables and exponent.is_Number and base not in powers:
            powers[base] = exponent
        else:
            rest.append(factor)
    degrees = tuple(powers.get(variable, sp.S.Zero) for variable in variables)
    return degrees, sp.Mul(*rest)


# -- partial fraction expansion ----------------------------------------------


def _partial_fractions(
    expression: sp.Expr, variables: tuple[sp.Symbol, ...], amount: Amount
) -> sp.Expr | None:
    """`expression` as a polynomial plus a sum of super-proper ratios.

    None if the decomposition fails (e.g. the numerator isn't a polynomial in the
    primary variable, or the system has no solution); the caller then expands it as a
    polynomial.

    The primary variable is the most main expansion variable in the denominator, not
    necessarily the first chosen: `x/(a^2 - 1)` about `x` and `a` splits over `a`.
    """
    numerator, denominator = sp.fraction(expression)
    primary = next(
        (variable for variable in variables if variable in denominator.free_symbols),
        None,
    )
    if primary is None:
        return None
    try:
        quotient, remainder = sp.div(numerator, denominator, primary)
        split = _denominator_factors(denominator, primary, variables, amount)
        if split is None:
            return None
        parts = _undetermined(remainder, split, primary, variables)
    except Exception:
        return None
    if parts is None:
        return None
    return sp.Add(_polynomial(quotient, variables), *parts)


def _denominator_factors(
    denominator: sp.Expr,
    primary: sp.Symbol,
    variables: tuple[sp.Symbol, ...],
    amount: Amount,
) -> tuple[sp.Expr, list[tuple[sp.Expr, int]]] | None:
    """The denominator as content times factors containing the primary variable.

    This is all the amount controls: Trivial leaves `x^2 - 1` whole and hence
    unexpanded, while raDical splits `x^2 - 2` over `SQRT(2)`.

    None if no factor contains the primary variable or the factors don't multiply
    back to the denominator.
    """
    factored = factored_expression(denominator, amount, [v.name for v in variables])
    content = sp.S.One
    factors: list[tuple[sp.Expr, int]] = []
    for factor in sp.Mul.make_args(factored):
        base, exponent = factor.as_base_exp()
        if primary in base.free_symbols and exponent.is_Integer and exponent > 0:
            factors.append((base, int(exponent)))
        else:
            content *= factor
    if not factors:
        return None
    product = content * sp.Mul(*(base**power for base, power in factors))
    if sp.expand(product - denominator) != 0:
        return None
    return content, factors


def _undetermined(
    remainder: sp.Expr,
    split: tuple[sp.Expr, list[tuple[sp.Expr, int]]],
    primary: sp.Symbol,
    variables: tuple[sp.Symbol, ...],
) -> list[sp.Expr] | None:
    """The proper part split over the denominator's factors.

    Each factor, at each power up to its multiplicity, gets a numerator with unknown
    coefficients and lower degree in the primary variable than the factor's base
    (what the manual calls super-proper). Multiplying by the denominator gives a
    polynomial identity whose coefficients form a linear system in the unknowns.

    Each term is multiplied by the product of the *other* factors rather than the
    denominator divided by its own, because sympy cancels only over the rationals:
    `(x^2 - 2)/(x - SQRT(2))` stays uncancelled, and the identity wouldn't be a
    polynomial.

    None if the system has no solution, leaving the ratio unsplit.
    """
    content, factors = split
    unknowns: list[sp.Dummy] = []
    terms: list[tuple[sp.Expr, sp.Expr]] = []
    identity = -remainder
    for index, (base, power) in enumerate(factors):
        degree = sp.Poly(base, primary).degree()
        others = sp.Mul(
            *(other**exponent for other, exponent in _without(factors, index))
        )
        for exponent in range(1, power + 1):
            coefficients = [sp.Dummy() for _ in range(degree)]
            unknowns.extend(coefficients)
            top = sp.Add(*(c * primary**i for i, c in enumerate(coefficients)))
            terms.append((top, base**exponent))
            identity += top * content * base ** (power - exponent) * others
    if not unknowns:
        return None
    system = sp.Poly(sp.expand(identity), primary).all_coeffs()
    solutions = sp.solve(system, unknowns, dict=True)
    if not solutions or any(unknown not in solutions[0] for unknown in unknowns):
        return None
    parts = (
        _over(sp.expand(top.subs(solutions[0])), base, variables)
        for top, base in terms
    )
    return [part for part in parts if part != 0]


def _without(
    factors: list[tuple[sp.Expr, int]], index: int
) -> list[tuple[sp.Expr, int]]:
    """`factors` with the one at `index` left out."""
    return [factor for position, factor in enumerate(factors) if position != index]


def _over(
    top: sp.Expr, base: sp.Expr, variables: tuple[sp.Symbol, ...]
) -> sp.Expr:
    """One super-proper term, with its denominator distributed where appropriate.

    Per the manual, a numerator that is a sum containing expansion variables has the
    denominator distributed over it, so `(x + 1)/(x^2 + 1)` becomes two terms. A
    numerator free of them stays one term (the manual's `81*(a + 1)/(x - 3)` isn't
    split) and is collected rather than multiplied out.
    """
    if isinstance(top, sp.Add) and top.free_symbols & set(variables):
        return sp.Add(*(term / base for term in top.args))
    return sp.factor_terms(top) / base

"""Writing an expression as a sum: the rewrites, and nothing else.

This is the mathematics of Derive's Expand with no command around it. It takes
a sympy expression and gives one back, so it sits below `pipeline` and can be
used by it: the `EXPAND` head an authored line may carry is evaluated during
Simplify, and the Expand command is `expand.py` on top of both.

Simplify's normal form is built on it too. Writing a sum about the one variable
that is most main in it is this same operation about one expansion variable,
and `held_about` and `collected_by` are here for `normal` to use rather than
reimplement - which is why the rule that everything free of the variables is
one opaque quantity shows up in both commands' answers.

The manual states the goal as maximizing the number of terms that are
algebraically independent with respect to the *expansion variables*, and the
whole of this file follows from taking that literally. Everything free of
those variables is one opaque quantity, so nothing about it is expanded and
nothing about it is multiplied out; what is left is expanded as far as it
goes, and terms carrying the same powers of the expansion variables are
collected back together. That is why `(x + 2*y + 1)^3` about `x` alone comes
out as powers of `2*y + 1` rather than as a polynomial in `y`, and why
`3*a^2*x^2*y^3 + 7*(b + 1)^2*x^2*y^3` about `x` and `y` collects into
`x^2*y^3*(3*a^2 + 7*(b + 1)^2)` with the `(b + 1)^2` still standing.

A ratio whose denominator holds an expansion variable is a different
operation under the same name: partial fraction expansion. Long division
first, if the numerator has the degree for it; then the denominator is
factored - by an *amount*, which is `factoring`'s dial and the reason Expand
asks for one at all - and the proper part is split over those factors by
undetermined coefficients. Undetermined coefficients rather than sympy's
`apart` because the factors may be radical: `1/(x^2 - 2)` splits over
`x - SQRT(2)` and `x + SQRT(2)` when raDical was asked for, which no
decomposition over the rationals reaches.

`Complex` is missing from the amounts here, and that is the original's doing:
Expand's amount menu lists four where Factor's lists five.

Exponents are left alone throughout. `2^(x + 1)` is not `2*2^x` to Derive, and
neither trigonometric nor exponential nor logarithmic expansion is Expand's
business - the manual sends those to the Trigonometry and Exponential options
instead.

Every transformation is offered inside a `try` and a failure leaves the
previous form standing, so this is as total as Simplify is: nothing sympy does
can turn a valid entry into an error.
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

    `variables` names the expansion variables in the order they were chosen,
    empty meaning all of them - which is the opposite of what empty means to
    `factoring`, where sympy is left to pick its own generators. Expand has
    nowhere to leave the choice: the manual says expansion is with respect to
    all the variables in the expression when none are named. `order` is the
    session's variable order list, which is what "all of them" is ordered by.

    Every scalar the expression is built out of is expanded and then passed
    through `finish`, so a caller that rounds can round the terms rather than
    what they were expanded from.

    Total: a rewrite that fails is a rewrite not taken, and nothing here
    raises on anything the conversion layer produced.
    """
    request = _Request(amount, tuple(variables), finish, tuple(order))
    return distributed(expression, lambda scalar: _leaf(scalar, request))


@dataclass(frozen=True)
class _Request:
    """What was asked for, carried down to every scalar it reaches."""

    amount: Amount
    variables: tuple[str, ...]
    finish: Finish | None
    order: tuple[str, ...]


def _leaf(expression: sp.Basic, request: _Request) -> sp.Basic:
    """One scalar, expanded about whichever of its variables were chosen."""
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
    """The expansion variables, as the symbols this expression holds.

    Named by the caller as text, because that is what a prompt collects, and
    resolved against the expression so that the symbol carries whatever a
    declared domain told the conversion layer about it. A name the expression
    does not hold drops out, which is what leaves `(x + 1)^2` alone when `y`
    was the variable asked for.

    Empty means all of them, most main first, so that the primary expansion
    variable is the one Derive would have made primary anyway.
    """
    # `type is` rather than `isinstance`: a string literal is a `Symbol`
    # subclass, and a string is data rather than a variable to expand about.
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
    """One scalar, by whichever of the two expansions applies to it.

    A ratio is a partial fraction expansion only where an expansion variable
    is in its denominator - `x/(a^2 - 1)` about `x` alone has nothing to split
    over - and only where the split succeeds. Everything else, and anything
    that did not, is a polynomial expansion.
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
    """`expression` with everything free of `variable` stood in for, and the map back.

    Public because Simplify's normal form is this same operation about one
    variable: what is opaque to an expansion is opaque to the normal form too,
    and both want to work on the frozen expression rather than only on the
    answer. `xreplace` with the map puts the placeholders back.
    """
    held: dict[sp.Dummy, sp.Expr] = {}
    return _frozen(expression, {variable}, held), held


def collected_by(expression: sp.Expr, variable: sp.Symbol) -> sp.Expr:
    """A multiplied-out sum gathered back up by the powers of `variable`.

    The normal form's collecting rather than Expand's, and the two differ over
    a coefficient's numeric content: this one leaves it where it stands, so
    that `6*x*y^2*z + 2*x*z^3` is `x*(6*y^2*z + 2*z^3)` - which is what the
    manual's own LAPLACIAN answers.
    """
    return _collected(expression, (variable,), content_outside=False)


def _polynomial(expression: sp.Expr, variables: tuple[sp.Symbol, ...]) -> sp.Expr:
    """Multiplied out about `variables`, and collected back by their powers."""
    held: dict[sp.Dummy, sp.Expr] = {}
    frozen = _frozen(expression, set(variables), held)
    # The two rewrites that take a power apart are off, neither being anything
    # Expand does: `x^(a + b)` is not `x^a*x^b` and `(x*y)^n` is not `x^n*y^n`.
    multiplied = sp.expand(frozen, power_exp=False, power_base=False)
    return _collected(multiplied, variables).xreplace(held)


def _frozen(
    expression: sp.Expr, variables: set[sp.Symbol], held: dict[sp.Dummy, sp.Expr]
) -> sp.Expr:
    """`expression` with everything free of `variables` stood in for.

    A placeholder is what makes an expansion variable's coefficient opaque, so
    that multiplying out cannot reach inside it. The sum and the product take
    all their parts that are free of the variables *together*, as one
    placeholder rather than one each: that is what leaves `x + 2*y + 1` as
    `x + d`, and so leaves `(x + 2*y + 1)^3` about `x` written in powers of
    `2*y + 1`.

    A product's numeric factor stays outside the placeholder, and a sum's
    numeric term does not. In a product the number is the coefficient the
    collecting is meant to find in common, and a `2*x*(a + b)` frozen again
    with the `2` inside comes back `x*(2*a + 2*b)`; in a sum it is part of the
    one opaque quantity, and without it `2*y` and `1` would be multiplied out
    separately and `3*x*(2*y + 1)^2` would come out `3*x*(4*y^2 + 4*y + 1)`.
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
        # What is expanded is the base. An exponent that is not a number is
        # stood in for as well, because sympy reads `2^(x + 1)` as a power of
        # a sum and would multiply it out into `2*2^x`, which is not something
        # Derive's Expand does.
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


#: The heads that bind a variable of their own. Each is stood in for whole,
#: expansion variables and all, for two reasons.
#:
#: Not every argument of one is an operand: a derivative carries the variable it
#: is taken over and how many times, a substitution the variables it binds and
#: the points they take, a root sum the polynomial whose roots it runs over and
#: the generator that polynomial and its summand share. Those travel as tuples,
#: lambdas and dummies, and rebuilding the head from rewritten arguments reads
#: them as something else - `DIF(u, xi)` comes back `DIF(u, [xi, 1])`,
#: `SUBS(u, [xi], [y])` comes back `SUBS(u, [[xi]], [y])`, and a root sum does
#: not come back at all: it rebuilds into nothing and the term is lost.
#:
#: And what is inside one is not being expanded anyway. `sp.expand` goes deep,
#: so an integrand would be multiplied out where nobody asked - which splits the
#: integral into one per term, and an integral that had no answer can end up
#: beside one that does.
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
    """A placeholder standing in for `expression`, or the number itself.

    A number is not stood in for: `(2 + 3)^2` is 25 whatever was asked for,
    and holding it back would leave the arithmetic undone.
    """
    if expression.is_number:
        return expression
    placeholder = sp.Dummy()
    held[placeholder] = expression
    return placeholder


def _collected(
    expression: sp.Expr,
    variables: tuple[sp.Symbol, ...],
    content_outside: bool = True,
) -> sp.Expr:
    """Terms carrying the same powers of every expansion variable, put together.

    Which is the manual's rule that terms of equal degree in the expansion
    variables are collected: `a*(x + 1)^2 + b*(x + 1)^2` about `x` is
    `x^2*(a + b) + 2*x*(a + b) + a + b`, not six terms.

    A coefficient that several terms went into keeps whatever it has in
    common outside: `2*x*(a + b)` rather than `x*(2*a + 2*b)`, which is the
    same rule that leaves a non-expansion variable unexpanded, and which the
    manual's own expansion about `y` alone shows in `12*y^2*(x + 1)`. The
    normal form collects the other way round - see `collected_by`.
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
            * _coefficient(rest, degrees, content_outside)
            for degrees, rest in groups.items()
        )
    )


def _coefficient(
    terms: list[sp.Expr], degrees: tuple[sp.Expr, ...], content_outside: bool = True
) -> sp.Expr:
    """What the terms of one degree carry, with any common factor outside.

    Only where there is a degree for them to be the coefficient of. The group
    at degree zero is everything the expansion variables did not reach, and
    that is the whole expression when they reach nothing - taking a common
    factor out of it is not collecting a coefficient, it is factoring, which is
    the opposite of what this command does. `2*SIN(x) + 2*COS(x)` is left as it
    stands rather than collected, and `LN(x^2 - x) - LN(x)` is left alone
    rather than written `LN(x*(x - 1)) - LN(x)`, which is `factor_terms`
    reaching inside a logarithm to factor its argument.

    Without `content_outside` what the terms have in common numerically stays
    inside, which is the normal form's rule rather than Expand's.
    """
    if len(terms) == 1 or not any(degrees):
        return sp.Add(*terms)
    gathered = sp.factor_terms(sp.Add(*terms))
    return gathered if content_outside else _with_the_number_back(gathered)


def _with_the_number_back(coefficient: sp.Expr) -> sp.Expr:
    """`coefficient` with any numeric factor multiplied back into its sum."""
    number, rest = coefficient.as_coeff_Mul()
    if number is sp.S.One or not isinstance(rest, sp.Add):
        return coefficient
    return sp.Add(*(number * term for term in rest.args))


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

    None where the decomposition does not come out - a numerator that is no
    polynomial in the primary variable, a system with no solution - and the
    caller expands the expression as a polynomial instead.

    The primary variable is the most main expansion variable that the
    denominator actually holds, which need not be the first one chosen:
    `x/(a^2 - 1)` about `x` and `a` splits over `a`, there being no `x` in the
    denominator to split over.
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
    """The denominator as a content times factors that hold the primary variable.

    Factored by the amount asked for, which is the whole of what the amount
    decides: Trivial leaves `x^2 - 1` whole and the expansion with it, while
    raDical splits `x^2 - 2` and lets the expansion reach `SQRT(2)`.

    None where nothing came apart into a factor holding the primary variable,
    or where the factors do not multiply back to what they were factored from
    - there is nothing to decompose over in either case.
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

    Each factor to each power up to its own gets a numerator of lower degree
    in the primary variable than the factor's base has - which is what the
    manual calls super-proper - written with unknown coefficients. Multiplying
    the whole sum back up by the denominator gives one polynomial identity in
    the primary variable, and its coefficients are a linear system in the
    unknowns.

    Each term is multiplied by the product of the *other* factors rather than
    by the denominator divided by its own, because dividing means cancelling
    and sympy cancels over the rationals: `(x^2 - 2)/(x - SQRT(2))` comes back
    uncancelled, and the identity would not be a polynomial at all.

    None if the system will not solve, which leaves the ratio unsplit rather
    than half-split.
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
    """One super-proper term, with its denominator distributed where it should be.

    The manual's last rule: a numerator that is a sum containing expansion
    variables has the denominator distributed over it, so `(x + 1)/(x^2 + 1)`
    becomes two terms. A numerator free of them is left as one - the manual's
    own `81*(a + 1)/(x - 3)` is not split into two terms in `a` - and is
    collected rather than left multiplied out, since nothing about a
    non-expansion variable is expanded unnecessarily.
    """
    if isinstance(top, sp.Add) and top.free_symbols & set(variables):
        return sp.Add(*(term / base for term in top.args))
    return sp.factor_terms(top) / base

"""Writing an expression as a product: the rewrites, and nothing else.

This is the mathematics of Derive's Factor with no command around it. It takes
a sympy expression and gives one back, so it sits below `pipeline` and can be
used by it: the `FACTOR` head an authored line may carry is evaluated during
Simplify, and the Factor command is `factor.py` on top of both.

Derive's Factor has two dials and no others. The *amount* says how hard to
try, in five steps that each do everything the one before it does and then
more; the *factorization variables* say which variables the factors are
allowed to be about, and everything else is left alone. There is no "factor
out this" hint, so there is nothing here that takes one.

The five amounts:

* Trivial puts the expression over a common denominator and pulls out the gcd
  of the numeric coefficients and the least power of each variable. It never
  factors a sum.
* Squarefree also splits off perfect powers, and deliberately stops there:
  `x^4 + 2*x^3 - 3*x^2 - 8*x - 4` becomes `(x + 1)^2*(x^2 - 4)` and the
  `x^2 - 4` is left whole, because its two factors are the same power.
* Rational also splits products of sums, as far as that goes without new
  fractional powers or complex numbers.
* raDical also allows fractional powers of numbers, so `x^2 - 2` splits.
* Complex also allows complex numbers, so `x^2 + 2` splits.

The first three are sympy calls. The last two are not: sympy factors over the
rationals and over an extension it must be handed, and neither is what "as far
as the reals allow" means. So those two work from the roots - the irreducible
factors of the rational factorization, split by `roots` where it succeeds -
and rebuild the product. raDical keeps a conjugate pair together as the real
quadratic it multiplies out to, which is what leaves `x^3 - 2` as
`(x - 2^(1/3))*(x^2 + 2^(1/3)*x + 2^(2/3))` rather than refusing to split it
at all.

A root that cannot be shown to be real is treated as real, not rejected. That
is what factors `a*x^2 + b*x + c` into the quadratic formula, where nothing is
known about `a`, `b` or `c` and Derive still splits it.

Every transformation is offered inside a `try` and a failure leaves the
previous form standing, so this is as total as Simplify is: nothing sympy does
can turn a valid entry into an error.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import sympy as sp
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean

from rederive.engine.to_sympy import Assign, Declare, FunDef, InertVector, Logical

__all__ = ["Amount", "amount_named", "factored_expression"]

#: What to put a factored scalar through afterwards, if anything. The precision
#: mode is the only caller so far, and it is a callback rather than a `Context`
#: because rounding belongs to `pipeline`, which sits above this file.
Finish = Callable[[sp.Expr], sp.Expr]


class Amount(StrEnum):
    """How hard Factor tries, in the words the menu spells them with.

    `raDical` carries its capital because that is where its mnemonic letter is
    - `R` belongs to Rational - and the menu reads the letter off the word.
    """

    TRIVIAL = "Trivial"
    SQUAREFREE = "Squarefree"
    RATIONAL = "Rational"
    RADICAL = "raDical"
    COMPLEX = "Complex"


#: What the amount is when nobody says. The manual's default, and the menu's.
DEFAULT_AMOUNT = Amount.RATIONAL


def amount_named(word: str) -> Amount | None:
    """The amount `word` names, however it was capitalised, or None.

    However it was capitalised because the word may have come from an authored
    `FACTOR(u, Rational, x)`, and the parser's default case mode does not
    preserve the capital that the menu spelling carries.
    """
    for member in Amount:
        if member.value.lower() == word.lower():
            return member
    return None


def factored_expression(
    expression: sp.Basic,
    amount: Amount = DEFAULT_AMOUNT,
    variables: Sequence[str] = (),
    finish: Finish | None = None,
) -> sp.Basic:
    """`expression` written as a product, as far as `amount` allows.

    `variables` names the factorization variables in the order they were
    chosen, empty meaning all of them. Every scalar the expression is built
    out of is factored and then passed through `finish`, so a caller that
    rounds can round the factors rather than what they were factored from.

    Total: a rewrite that fails is a rewrite not taken, and nothing here
    raises on anything the conversion layer produced.
    """
    return _distributed(expression, _Request(amount, tuple(variables), finish))


@dataclass(frozen=True)
class _Request:
    """What was asked for, carried down to every scalar it reaches."""

    amount: Amount
    variables: tuple[str, ...]
    finish: Finish | None


# -- by shape ----------------------------------------------------------------


def _distributed(expression: sp.Basic, request: _Request) -> sp.Basic:
    """Factor every scalar this expression is built out of.

    A relation is factored side by side, a vector or matrix element by
    element, a definition on its value alone - the same distribution Simplify
    makes, and for the same reason: what makes an entry a definition or a
    relation must survive for the line to still read as one.

    A boolean beyond a single relation comes back untouched. Derive factors
    those into conjunctive normal form, which is a different operation
    sharing a name, and not this milestone's.
    """
    def again(part: sp.Basic) -> sp.Basic:
        return _distributed(part, request)

    if isinstance(expression, Relational):
        left, right = again(expression.lhs), again(expression.rhs)
        try:
            return expression.func(left, right, evaluate=False)
        except Exception:
            return expression.func(left, right)
    if isinstance(expression, Declare):
        return expression
    if isinstance(expression, (Assign, FunDef)):
        head, operator, *value = expression.args
        return expression.func(head, operator, *(again(part) for part in value))
    if isinstance(expression, sp.MatrixBase):
        return expression.applyfunc(again)
    if isinstance(expression, (InertVector, Logical)):
        return expression.func(*(again(argument) for argument in expression.args))
    if isinstance(expression, Boolean):
        return expression
    return _leaf(expression, request)


def _leaf(expression: sp.Basic, request: _Request) -> sp.Basic:
    """One scalar, factored by the amount asked for.

    A number is its prime decomposition whatever the amount, which is why the
    original does not even ask for one when the whole expression is a number.
    """
    if not isinstance(expression, sp.Expr):
        return expression
    if expression.is_number:
        product = _decomposed(expression)
    elif not expression.free_symbols:
        return expression
    else:
        generators = _generators(expression, request.variables)
        product = _attempt(
            expression, lambda e: _AMOUNTS[request.amount](e, generators)
        )
    return request.finish(product) if request.finish else product


def _generators(
    expression: sp.Expr, variables: Sequence[str]
) -> tuple[sp.Symbol, ...]:
    """The factorization variables, as the symbols this expression holds.

    Named by the caller as text, because that is what a prompt collects, and
    resolved against the expression so that the symbol carries whatever a
    declared domain told the conversion layer about it.

    Empty when nobody chose, and that is passed on empty rather than filled in
    with every symbol: sympy picks its own generators then, which is the case
    where the expression is a polynomial in something that is not a symbol at
    all - `SIN(x)^2 - 1` factors in `SIN(x)`, and naming `x` would say it
    factors in nothing.
    """
    free = {symbol.name: symbol for symbol in expression.free_symbols}
    return tuple(free[name] for name in variables if name in free)


# -- the amounts -------------------------------------------------------------


def _trivial(expression: sp.Expr, generators: tuple[sp.Symbol, ...]) -> sp.Expr:
    """Over a common denominator, with the content pulled out.

    The generators are not consulted. Trivial content is the gcd of the
    coefficients and the least power of every variable that occurs, which is
    the same expression whichever variables were named.
    """
    return sp.factor_terms(sp.together(expression))


def _squarefree(expression: sp.Expr, generators: tuple[sp.Symbol, ...]) -> sp.Expr:
    """Trivial content, plus the perfect powers split off.

    Numerator and denominator separately, because `sqf` wants a polynomial and
    a ratio of two factored polynomials is the factored ratio.
    """
    numerator, denominator = sp.fraction(sp.together(expression))
    top = sp.sqf(numerator, *generators)
    if denominator == 1:
        return top
    return top / sp.sqf(denominator, *generators)


def _rational(expression: sp.Expr, generators: tuple[sp.Symbol, ...]) -> sp.Expr:
    """As far as it goes over the rationals.

    `factor` restricted to the chosen generators is exactly Derive's rule that
    a subexpression in none of the factorization variables is left alone:
    factoring `x^2*y^2 - x^2 - y^4 + y^2` about `x` alone keeps `y^2 - 1`
    whole, where naming both variables splits it.
    """
    return sp.factor(expression, *generators)


def _radical(expression: sp.Expr, generators: tuple[sp.Symbol, ...]) -> sp.Expr:
    return _from_roots(expression, generators, complex_allowed=False)


def _complex(expression: sp.Expr, generators: tuple[sp.Symbol, ...]) -> sp.Expr:
    return _from_roots(expression, generators, complex_allowed=True)


_AMOUNTS: dict[Amount, Callable[[sp.Expr, tuple[sp.Symbol, ...]], sp.Expr]] = {
    Amount.TRIVIAL: _trivial,
    Amount.SQUAREFREE: _squarefree,
    Amount.RATIONAL: _rational,
    Amount.RADICAL: _radical,
    Amount.COMPLEX: _complex,
}


# -- splitting what the rationals cannot -------------------------------------


def _from_roots(
    expression: sp.Expr,
    generators: tuple[sp.Symbol, ...],
    *,
    complex_allowed: bool,
) -> sp.Expr:
    """The rational factorization, with each factor split by its roots.

    Splitting happens in the primary factorization variable - the first one
    chosen - which is what makes the choice an ordering and not just a set.
    With nothing chosen it is the first variable by name, since one of them
    has to go first and the expression itself offers no order.
    """
    variable = generators[0] if generators else _primary(expression)
    content, factors = sp.factor_list(expression, *generators)
    product: list[sp.Expr] = [content]
    for base, multiplicity in factors:
        if variable in base.free_symbols:
            base = _split(base, variable, complex_allowed)
        product.append(base**multiplicity)
    return sp.Mul(*product)


def _primary(expression: sp.Expr) -> sp.Symbol:
    return min(expression.free_symbols, key=lambda symbol: symbol.name)


def _split(base: sp.Expr, variable: sp.Symbol, complex_allowed: bool) -> sp.Expr:
    """One factor irreducible over the rationals, split by its roots.

    Comes back as it went in unless every root is known: `roots` answers with
    what it could solve, and a partial answer is no factorization. A quintic
    it cannot crack is a factor that stays whole, which is the honest result
    rather than a numeric approximation nobody asked for.
    """
    polynomial = sp.Poly(base, variable)
    if polynomial.degree() < 2:
        return base
    roots = sp.roots(polynomial)
    if sum(roots.values()) != polynomial.degree():
        return base
    leading = polynomial.LC()
    product: list[sp.Expr] = [leading] if leading != 1 else []
    conjugates: list[sp.Expr] = []
    for root, multiplicity in roots.items():
        if complex_allowed or root.is_real is not False:
            # Written `x + (-r)` rather than `x - r` so that a root already
            # carrying a minus sign has it folded into the numerator
            # `together` builds, which is where the original puts it:
            # `x + (SQRT(b^2 - 4*a*c) + b)/(2*a)`, not `x - (-b - SQRT(...))/(2*a)`.
            product.append((variable + sp.together(-root)) ** multiplicity)
        else:
            conjugates.extend([root] * multiplicity)
    paired = _paired(conjugates, variable)
    if paired is None:
        return base
    return sp.Mul(*product, *paired)


def _paired(roots: list[sp.Expr], variable: sp.Symbol) -> list[sp.Expr] | None:
    """The complex roots, two by two, as the real quadratics they multiply to.

    This is what "as far as the reals allow" means: a conjugate pair has no
    real linear factors but their product has real coefficients, so the pair
    becomes one quadratic rather than nothing. None when a root has no partner
    here, which leaves the whole factor alone rather than half-split.
    """
    remaining = list(roots)
    quadratics = []
    while remaining:
        root = remaining.pop()
        mate = next(
            (
                other
                for other in remaining
                if sp.simplify(other - sp.conjugate(root)) == 0
            ),
            None,
        )
        if mate is None:
            return None
        remaining.remove(mate)
        quadratics.append(_quadratic(variable, root, mate))
    return quadratics


def _quadratic(variable: sp.Symbol, root: sp.Expr, mate: sp.Expr) -> sp.Expr:
    """`(x - r)*(x - conj(r))` multiplied out and collected about `x`.

    Collected rather than left expanded so that the coefficient of `x` is one
    number: `x^2 + x*(1 - SQRT(5))/2 + 1` rather than the same thing with the
    `x` term split in two.
    """
    product = sp.collect(sp.expand((variable - root) * (variable - mate)), variable)
    if not product.is_Add:
        return product
    return product.func(*(sp.together(term) for term in product.args))


# -- numbers -----------------------------------------------------------------


def _decomposed(expression: sp.Expr) -> sp.Expr:
    """A rational number as a product of prime powers.

    `1234567890/49` is `2*3^2*5*3607*3803/7^2`: the denominator's primes come
    back with negative exponents, which is the same product written where the
    notation can hold it. A negative number keeps its sign outside the product,
    as `-12` -> `-2^2*3`, rather than carrying a factor of -1.

    Anything that is not a rational number - a root, a constant, a float - has
    no prime decomposition and comes back as itself, and so does a number with
    nothing to decompose: 0, 1, -1 and the primes are already written as a
    product of primes.
    """
    if not isinstance(expression, sp.Rational) or expression in (0, 1, -1):
        return expression
    return _attempt(expression, _primes)


def _primes(number: sp.Rational) -> sp.Expr:
    """`number` as a product of prime powers, built rather than evaluated.

    Every piece has to be assembled unevaluated: sympy would otherwise
    multiply the product straight back into the integer it came from, which is
    the one thing this must not return. A negative number is negated as a
    whole, rather than given a factor of -1 to carry, so that the sign lands
    before the product where the original puts it.

    The whole is held unevaluated for a second reason as well. A decomposition
    says what divides what, so no precision mode may round it - Derive prints
    `2*3^2*5*3607*3803` in Approximate mode too - and rounding it would lose
    the number and not just the factors, six significant digits of
    `1234567890` being a different integer. Sealing it here means every caller
    is exempt without having to know it needs to be.
    """
    powers = sp.factorrat(abs(number))
    pieces = [
        sp.Integer(prime) if power == 1 else sp.Pow(prime, power, evaluate=False)
        for prime, power in sorted(powers.items())
    ]
    product = sp.UnevaluatedExpr(sp.Mul(*pieces, evaluate=False))
    return -product if number < 0 else product


def _attempt(
    expression: sp.Expr, rewrite: Callable[[sp.Expr], sp.Basic]
) -> sp.Expr:
    """`rewrite`, or the expression unchanged if it will not run.

    Unlike Simplify's gate this does not ask whether the answer got shorter.
    Factoring was asked for by name, and `(x + 2)*(x - 2)` is the answer to
    that question even though it counts more operations than `x^2 - 4`.
    """
    try:
        candidate = rewrite(expression)
    except Exception:
        return expression
    return candidate if isinstance(candidate, sp.Expr) else expression

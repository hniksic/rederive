"""Writing an expression as a product: the rewrites, and nothing else.

The mathematics of Derive's Factor, sympy expression in and out. It sits below
`pipeline`, which uses it to evaluate an authored `FACTOR`; the Factor command is
`factor.py` on top of both.

Factor has two parameters. The *amount* says how hard to try, in five cumulative
steps. The *factorization variables* say which variables factors may involve;
everything else is left alone.

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

The first three are sympy calls. Sympy factors only over the rationals or a given
extension, not "as far as the reals allow", so the last two split each irreducible
rational factor by its `roots` (where that succeeds) and rebuild the product. raDical
keeps each conjugate pair together as a real quadratic, so `x^3 - 2` becomes
`(x - 2^(1/3))*(x^2 + 2^(1/3)*x + 2^(2/3))`.

A root not provably non-real is treated as real. That is how `a*x^2 + b*x + c` factors
by the quadratic formula with nothing known about `a`, `b` or `c`, as in Derive.

Every transformation runs inside a `try`, and a failure keeps the previous form, so
like Simplify this never turns a valid entry into an error.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import sympy as sp

from rederive.engine.boundary import DEFAULT_AMOUNT, Amount
from rederive.engine.shape import attempt as _attempt
from rederive.engine.shape import distributed

__all__ = ["amount_named", "factored_expression"]

#: Optional post-processing of a factored scalar, currently only rounding for the
#: precision mode. It is a callback rather than a `Context` because rounding belongs
#: to `pipeline`, which sits above this module.
Finish = Callable[[sp.Expr], sp.Expr]


def amount_named(word: str) -> Amount | None:
    """The amount named by `word`, case-insensitively, or None.

    Case is ignored because the word may come from an authored `FACTOR(u, Rational,
    x)`, and the parser's default case mode doesn't preserve the menu's capitals.
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

    `variables` names the factorization variables in the order chosen, empty meaning
    all. Each scalar in the expression is factored and then passed through `finish`, so
    rounding applies to the factors rather than to the original.

    Never raises on output of the conversion layer; a failed rewrite is skipped.
    """
    request = _Request(amount, tuple(variables), finish)
    return distributed(expression, lambda scalar: _leaf(scalar, request))


@dataclass(frozen=True)
class _Request:
    """The factoring parameters, passed down to each scalar."""

    amount: Amount
    variables: tuple[str, ...]
    finish: Finish | None


# -- one scalar --------------------------------------------------------------


def _leaf(expression: sp.Basic, request: _Request) -> sp.Basic:
    """One scalar, factored by the requested amount.

    A number is decomposed into primes regardless of amount, which is why Derive
    doesn't ask for an amount when the whole expression is a number.
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
    """The factorization variables, as the symbols in this expression.

    The caller names them as text (from a prompt); resolving them against the
    expression gets symbols carrying their declared domain assumptions.

    If none were chosen this stays empty rather than listing every symbol, so sympy
    picks its own generators. That matters for polynomials in non-symbols: `SIN(x)^2 -
    1` factors in `SIN(x)`, but not in `x`.
    """
    free = {symbol.name: symbol for symbol in expression.free_symbols}
    return tuple(free[name] for name in variables if name in free)


# -- the amounts -------------------------------------------------------------


def _trivial(expression: sp.Expr, generators: tuple[sp.Symbol, ...]) -> sp.Expr:
    """Over a common denominator, with the content pulled out.

    The generators are ignored: the content (gcd of the coefficients and the least
    power of each variable) is the same whichever variables were named.
    """
    return sp.factor_terms(sp.together(expression))


def _squarefree(expression: sp.Expr, generators: tuple[sp.Symbol, ...]) -> sp.Expr:
    """Trivial content, plus the perfect powers split off.

    Numerator and denominator are handled separately, because `sqf` needs a
    polynomial.
    """
    numerator, denominator = sp.fraction(sp.together(expression))
    top = sp.sqf(numerator, *generators)
    if denominator == 1:
        return top
    return top / sp.sqf(denominator, *generators)


def _rational(expression: sp.Expr, generators: tuple[sp.Symbol, ...]) -> sp.Expr:
    """Factored over the rationals.

    Restricting `factor` to the chosen generators matches Derive's rule that
    subexpressions free of the factorization variables are left alone: factoring
    `x^2*y^2 - x^2 - y^4 + y^2` about `x` keeps `y^2 - 1` whole, but about both splits
    it.
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

    Splitting happens in the primary (first chosen) factorization variable, so the
    choice is ordered. With none chosen, the alphabetically first variable is used.
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
    """A factor irreducible over the rationals, split by its roots.

    Returned unchanged unless `roots` finds all roots; a partial answer is no
    factorization. An unsolvable quintic thus stays whole rather than being
    approximated numerically.

    `trig` requests Viete's solution of the cubic, which only affects the casus
    irreducibilis: an irreducible cubic with three real roots that Cardano's formula
    can only write using `#i`. The trigonometric form is visibly real, so `x^3 - 3*x +
    1` splits under raDical.
    """
    polynomial = sp.Poly(base, variable)
    if polynomial.degree() < 2:
        return base
    roots = sp.roots(polynomial, trig=True)
    if sum(roots.values()) != polynomial.degree():
        return base
    leading = polynomial.LC()
    product: list[sp.Expr] = [leading] if leading != 1 else []
    conjugates: list[sp.Expr] = []
    for root, multiplicity in roots.items():
        if complex_allowed or root.is_real is not False:
            product.append(_linear(variable, root) ** multiplicity)
        else:
            conjugates.extend([root] * multiplicity)
    paired = _paired(conjugates, variable)
    if paired is None:
        return base
    return sp.Mul(*product, *paired)


def _linear(variable: sp.Symbol, root: sp.Expr) -> sp.Expr:
    """The factor `x - r`, with the minus sign placed as Derive does.

    The root is put over a common denominator, as Derive writes it. A leading minus
    goes into the operator rather than the numerator: the quadratic formula's factors
    are `x + (SQRT(b^2 - 4*a*c) + b)/(2*a)` and `x - (SQRT(b^2 - 4*a*c) - b)/(2*a)`
    (manual 4.6, p.79).
    """
    root = sp.together(root)
    if root.could_extract_minus_sign():
        return variable + sp.together(-root)
    return variable - root


def _paired(roots: list[sp.Expr], variable: sp.Symbol) -> list[sp.Expr] | None:
    """The complex roots, paired into conjugates, as the real quadratics they give.

    A conjugate pair has no real linear factors, but its product is a real
    quadratic. Returns None if a root has no conjugate, so the whole factor is left
    alone rather than half-split.
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

    Collected so the coefficient of `x` is a single term: `x^2 + x*(1 - SQRT(5))/2 +
    1` rather than having the `x` term split in two.
    """
    product = sp.collect(sp.expand((variable - root) * (variable - mate)), variable)
    if not product.is_Add:
        return product
    return product.func(*(sp.together(term) for term in product.args))


# -- numbers -----------------------------------------------------------------


def _decomposed(expression: sp.Expr) -> sp.Expr:
    """A rational number as a product of prime powers.

    `1234567890/49` is `2*3^2*5*3607*3803/7^2`, the denominator's primes having
    negative exponents. A negative number keeps its sign outside, as `-12` ->
    `-2^2*3`, rather than carrying a factor of -1.

    Non-rationals (roots, constants, floats) are returned as is, as are 0, 1 and -1.
    """
    if not isinstance(expression, sp.Rational) or expression in (0, 1, -1):
        return expression
    return _attempt(expression, _primes)


def _primes(number: sp.Rational) -> sp.Expr:
    """`number` as a product of prime powers, built rather than evaluated.

    The pieces are assembled unevaluated, since sympy would otherwise multiply them
    back into the original integer. A negative number is negated as a whole, putting
    the sign before the product as Derive does.

    The `UnevaluatedExpr` wrapper also keeps any precision mode from rounding it:
    Derive prints `2*3^2*5*3607*3803` in Approximate mode too, and six significant
    digits of `1234567890` would be a different integer. Sealing it here exempts every
    caller automatically.

    The search is unbounded, so a sixty-digit semiprime can occupy a core
    indefinitely. Bounding it with `limit=` is worse: sympy then disables
    elliptic-curve factoring and may return an untested composite among the primes,
    so it is slower *and* wrong. Unbounded, every factorization is complete, and a
    hopeless number can be aborted with Esc.
    """
    powers = sp.factorrat(abs(number))
    pieces = [
        sp.Integer(prime) if power == 1 else sp.Pow(prime, power, evaluate=False)
        for prime, power in sorted(powers.items())
    ]
    product = sp.UnevaluatedExpr(sp.Mul(*pieces, evaluate=False))
    return -product if number < 0 else product

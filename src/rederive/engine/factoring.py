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
from sympy.core.mul import _keep_coeff

from rederive.engine.boundary import DEFAULT_AMOUNT, Amount
from rederive.engine.matrix_factoring import MATRIX_AMOUNTS, matrix_product
from rederive.engine.shape import attempt as _attempt
from rederive.engine.shape import distributed
from rederive.engine.vectors import Vector

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
    content: bool = False,
) -> sp.Basic:
    """`expression` written as a product, as far as `amount` allows.

    `variables` names the factorization variables in the order chosen, empty meaning
    all. Each scalar in the expression is factored and then passed through `finish`, so
    rounding applies to the factors rather than to the original. A logical formula is
    put in conjunctive normal form.

    `content` asks for a vector's content to be pulled out instead: `[10, 100, 1000]`
    is `10*[1, 10, 100]` (see `vector_content`); Trivial pulls out no sum. A vector
    without content is factored element by element.

    The matrix amounts write a matrix as a product of matrices (see
    `matrix_factoring`); anything else they factor as Rational does.

    Never raises on output of the conversion layer; a failed rewrite is skipped.
    """
    if amount in MATRIX_AMOUNTS:
        try:
            product = matrix_product(expression, amount, finish)
        except Exception:
            product = None
        if product is not None:
            return product
        amount = DEFAULT_AMOUNT
    if content and isinstance(expression, Vector):
        whole = amount is not Amount.TRIVIAL
        pulled = _attempt(expression, lambda vector: vector_content(vector, whole))
        if pulled is not expression:
            return pulled
    request = _Request(amount, tuple(variables), finish)
    return distributed(
        expression, lambda scalar: _leaf(scalar, request), _conjunctive
    )


def _conjunctive(formula: sp.Basic) -> sp.Basic:
    """A logical formula in conjunctive normal form: `p OR (q AND r)` is `(p OR q) AND
    (p OR r)`."""
    try:
        return sp.to_cnf(formula)
    except Exception:
        return formula


def vector_content(vector: Vector, whole: bool = True) -> sp.Basic:
    """`vector` as its content times the vector of cofactors, or `vector` if the
    content is 1. Without `whole`, as for Trivial, the content has no sum in it.

    The content is the gcd of the elements, at every depth of a matrix: the gcd of
    their numeric coefficients times the common factors of the rest. Multiple angles
    are written out where that finds more in common, so `[2*r*COS(t)/3 -
    2*r*COS(t)^2/3, 2*r*SIN(t)/3 - r*SIN(2*t)/3]` is `2*r*(COS(t) - 1)/3*[-COS(t),
    -SIN(t)]`. The product is held unevaluated, so that it is the answer as written;
    used in arithmetic it is the vector again.
    """
    elements = list(_elements(vector))
    if len(elements) < 2 or not all(isinstance(e, sp.Expr) for e in elements):
        return vector
    if all(element == 0 for element in elements):
        return vector
    common = _common_factor(elements, whole)
    if whole and any(element.has(sp.sin, sp.cos) for element in elements):
        expanded = [sp.expand_trig(element) for element in elements]
        wider = _common_factor(expanded, whole)
        if sp.count_ops(wider) > sp.count_ops(common):
            common, elements = wider, expanded
            vector = _scaled(vector, sp.expand_trig)
    if common == 1:
        return vector
    cofactors = _scaled(vector, lambda element: sp.cancel(element / common))
    return sp.Mul(*sp.Mul.make_args(common), cofactors, evaluate=False)


def _common_factor(elements: list[sp.Expr], whole: bool) -> sp.Expr:
    """The gcd of `elements`, not all zero: of their numeric coefficients, times the
    gcd of the rest as polynomials, or without `whole` only the factors of that which
    are no sums."""
    nonzero = [element for element in elements if element != 0]
    numbers, primitives = zip(
        *(element.as_content_primitive() for element in nonzero), strict=True
    )
    numeric = sp.Rational(
        sp.igcd(*(number.p for number in numbers)),
        sp.ilcm(*(number.q for number in numbers)),
    )
    symbolic = primitives[0]
    for primitive in primitives[1:]:
        symbolic = sp.gcd(symbolic, primitive)
    symbolic = sp.factor_terms(symbolic.as_content_primitive()[1])
    if not whole:
        symbolic = sp.Mul(
            *(
                factor
                for factor in sp.Mul.make_args(symbolic)
                if not factor.as_base_exp()[0].is_Add
            )
        )
    return numeric * symbolic


def _elements(vector: sp.Basic):
    """The scalars of a vector or matrix."""
    for element in vector.args:
        if isinstance(element, Vector):
            yield from _elements(element)
        else:
            yield element


def _scaled(vector: Vector, rewrite: Callable[[sp.Expr], sp.Expr]) -> Vector:
    return Vector(
        *(
            _scaled(element, rewrite) if isinstance(element, Vector) else rewrite(element)
            for element in vector.args
        )
    )


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

    Exponentials are content too. Each is split into a product, one factor per term of
    its exponent, so that `#e^(2*t - b)` shares `#e^(2*t)` with `a*#e^(2*t)`; the
    factors left in each product of the result are joined again. So
    `a*#e^(2*t) - #e^(2*t - b)` is `#e^(2*t - b)*(a*#e^b - 1)`, as in Derive, which
    ODE.MTH's `BARE` relies on to strip such factors from a differential equation.
    """
    split = expression.replace(
        sp.exp,
        lambda exponent: sp.Mul(
            *(sp.exp(term) for term in sp.Add.make_args(sp.expand(exponent)))
        ),
    )
    return _joined_exponentials(_over_common_denominator(split))


def _over_common_denominator(expression: sp.Expr) -> sp.Expr:
    """`expression` as one quotient, with the content of each side pulled out.

    A numeric denominator stays apart from the sum it divides, which sympy would
    otherwise distribute back over the terms: `x^2/4 + 3` is `(x^2 + 12)/4`.
    """
    numerator, denominator = sp.fraction(sp.together(expression))
    top, bottom = sp.factor_terms(numerator), sp.factor_terms(denominator)
    coefficient, rest = bottom.as_coeff_Mul()
    return _keep_coeff(1 / coefficient, top / rest)


def _is_exponential(factor: sp.Basic) -> bool:
    return isinstance(factor, sp.exp) or factor is sp.E


def _joined_exponentials(expression: sp.Expr) -> sp.Expr:
    """`expression` with the exponentials in each product multiplied into one."""

    def joined(product: sp.Mul) -> sp.Expr:
        exponents = [
            sp.S.One if factor is sp.E else factor.args[0]
            for factor in product.args
            if _is_exponential(factor)
        ]
        rest = [factor for factor in product.args if not _is_exponential(factor)]
        return sp.Mul(*rest) * sp.exp(sp.Add(*exponents))

    return expression.replace(
        lambda e: e.is_Mul and sum(map(_is_exponential, e.args)) > 1, joined
    )


def _squarefree(expression: sp.Expr, generators: tuple[sp.Symbol, ...]) -> sp.Expr:
    """Trivial content, plus the perfect powers split off.

    Numerator and denominator are handled separately, because `sqf` needs a
    polynomial.
    """
    numerator, denominator = sp.fraction(sp.together(expression))
    top = sp.sqf(numerator, *generators)
    if denominator == 1:
        return top
    coefficient, rest = sp.sqf(denominator, *generators).as_coeff_Mul()
    return _keep_coeff(1 / coefficient, top / rest)


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


def integer_powers(number: int) -> dict[int, int]:
    """The prime factorization of a positive integer, as primes and their powers.

    The primes below a million are divided out first, and only what remains goes to
    sympy's full search: on a number whose factors are all small (`p - 1` for a prime
    built as a product of small primes plus one) the full search can run for minutes
    before trying division. Every factor the bounded pass leaves that is not prime is
    factored completely, so the result is complete either way.
    """
    powers: dict[int, int] = {}
    for factor, power in sp.factorint(number, limit=10**6).items():
        if factor < 10**12 or sp.isprime(factor):
            powers[factor] = powers.get(factor, 0) + power
            continue
        for prime, inner in sp.factorint(factor).items():
            powers[prime] = powers.get(prime, 0) + inner * power
    return powers


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
    number that takes long to factor takes as long as the user lets it. The small
    primes are divided out first, though (see `integer_powers`).
    """
    number = sp.Rational(number)
    powers = integer_powers(abs(number.p))
    for prime, power in integer_powers(number.q).items():
        powers[prime] = powers.get(prime, 0) - power
    pieces = [
        sp.Integer(prime) if power == 1 else sp.Pow(prime, power, evaluate=False)
        for prime, power in sorted(powers.items())
    ]
    product = sp.UnevaluatedExpr(sp.Mul(*pieces, evaluate=False))
    return -product if number < 0 else product

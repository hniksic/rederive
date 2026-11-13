"""Derive's normal form: every sum and quotient written partially factored, recursively.

Derive writes every sum as a rational function of its most main generator. The terms
containing it go over one denominator, the gcd is cancelled, long division extracts
the polynomial part, and that polynomial is collected by powers of the generator.
Meanwhile everything free of the generator is an opaque coefficient, and each
coefficient is then normalized the same way about its own generator. Terms free of
the generator are untouched; they join the constant term, and that sum is written
about the next generator.

So `(x + 1)^9 + y` is a ninth-degree polynomial in `x` plus `y`, while `(y + 1)^9 + x`
is `x + (y + 1)^9`, the power being free of `x`. This is the manual's section 4.3
demonstration, and why `Manage Ordering` changes Simplify's answers. It also keeps
`3*x*(2*y + 1)^2` in powers of `2*y + 1` rather than multiplied out in `y`.

Products are never distributed, only cancelled factor by factor: `2*x*(x - 3)^2`, `(x
+ 1)*(y + 1)` and `(x + 1)^2/y` are unchanged, and `(x^2 + 2*x*y + y^2)/(x^2 - y^2)`
becomes `(x + y)/(x - y)`, not `1 + 2*y/(x - y)`. This is what makes it a normal form
rather than a full expansion.

Two further rules shape a sum. If some term has a sum as a factor, a factor common to
all terms is taken outside: `2*(x^2 - y^2)^6 - (x^2 - y^2)^5*(2*x^2 - 3)` becomes
`(x^2 - y^2)^5*(3 - 2*y^2)`, while a sum of monomials keeps its terms. And a
denominator free of the generator (a coefficient's denominator) shared by two or more
terms puts the whole sum over it: `(x + 1)^2/y + z` is `(x^2 + 2*x + y*z + 1)/y`,
while `x/y + z` stays.

The generator is a kernel, not just a variable: a maximal subexpression that isn't a
sum, product, or power with numeric exponent, ranked by its most main variable, plain
variables first. `LN(x)*(y + 1)^2 + LN(x)*z` is collected on `LN(x)` into `(y^2 + 2*y
+ z + 1)*LN(x)`, since `LN(x)` is the most main thing the sum is a polynomial in.

The walk is a single bottom-up pass. Every argument is normalized before its parent,
and writing a node only builds sums from already written parts, so the result is a
fixed point without repeating the walk.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import reduce

import sympy as sp
from sympy.core.mul import _keep_coeff
from sympy.core.intfunc import integer_log

from rederive.engine.expanding import _BOUND
from rederive.engine.ordering import ORDER_LIST
from rederive.engine.vectors import Vector, mapped

__all__ = ["normal_form"]


def normal_form(
    expression: sp.Basic, order: Sequence[str] = ORDER_LIST, over_numbers: bool = True
) -> sp.Basic:
    """`expression` with every sum and quotient in it written in Derive's normal form.

    `order` is the session's variable order list, which decides each sum's generator.

    `over_numbers` says whether a sum is put over a *numeric* denominator shared by
    several terms. Derive keeps a quotient produced by a computation, as when solving
    divides by a coefficient: `x = (1 - z)/2`, not `x = 1/2 - z/2`. But an authored sum
    keeps its terms apart: `x^2/6 + x/2` and `1/2 + SIN(x)/2` stay as written. So the
    pipeline passes `False` and solving uses the default. Denominators containing a
    variable are combined either way.

    Never raises; a failed rewrite is skipped.
    """
    return _Writer(tuple(order), over_numbers).written(expression)


class _Writer:
    """One walk of the normal form, with a memo of what it has written.

    The memo is purely a cache, so a coefficient appearing in several places is
    written once.
    """

    def __init__(self, order: tuple[str, ...], over_numbers: bool) -> None:
        self.order = order
        self.over_numbers = over_numbers
        self.memo: dict[sp.Basic, sp.Basic] = {}

    def written(self, expression: sp.Basic) -> sp.Basic:
        try:
            return self.memo[expression]
        except (KeyError, TypeError):
            pass
        answer = self._written(expression)
        try:
            self.memo[expression] = answer
        except TypeError:
            pass
        return answer

    def _written(self, expression: sp.Basic) -> sp.Basic:
        if isinstance(expression, Vector):
            entries = _attempt(expression, lambda v: mapped(v, self.written))
            return expression if entries is None else entries
        if not isinstance(expression, sp.Basic) or expression.is_Atom:
            return expression
        if not expression.args or isinstance(expression, _BOUND):
            # Don't reach into heads that bind their own variable: not all their
            # arguments are operands.
            return expression
        algebraic = isinstance(expression, sp.Expr) and bool(expression.is_commutative)
        if algebraic and not expression.free_symbols and not expression.has(*_BOUND):
            rebuilt = _attempt(expression, _rebuilt)
            return self._number(expression if rebuilt is None else rebuilt)
        arguments = [self.written(argument) for argument in expression.args]
        if algebraic and (
            isinstance(expression, (sp.Add, sp.Mul)) or _reciprocal_sum(expression)
        ):
            return self._step(expression.func(*arguments))
        if all(new is old for new, old in zip(arguments, expression.args)):
            return expression
        rebuilt = _attempt(arguments, lambda parts: expression.func(*parts))
        return expression if rebuilt is None else rebuilt

    def _step(self, built: sp.Basic) -> sp.Basic:
        """A sum or product of written parts, written in turn.

        Dispatches on the built result, which may differ from the input: sympy turns a
        number times a sum into a sum.
        """
        if not (isinstance(built, sp.Expr) and built.is_commutative):
            return built
        if built.is_Add:
            rewrite = self._sum
        elif built.is_Mul or _reciprocal_sum(built):
            rewrite = self._product
        else:
            return built
        answer = _attempt(built, rewrite)
        return built if answer is None else answer

    def _number(self, number: sp.Expr) -> sp.Basic:
        """A number, simplified by plain arithmetic.

        Multiplied out, rationalized and denested term by term, so `(1 + 2*#i)*(3 +
        #i)` is `1 + 7*#i` and `(2 + #i)/(#i + 1)` is `3/2 - #i/2`.

        The result is kept only if shorter: `SQRT(5)*(2 + #i)/5` multiplied out is two
        terms each with the root, and `(#e^pi/2 - #e^(-pi)/2)/pi` is better left as
        Simplify writes `SINH(pi)/pi`.
        """
        value = _attempt(number, _arithmetic)
        if value is None:
            return number
        terms = []
        for term in sp.Add.make_args(value):
            denested = _attempt(term, sp.sqrtdenest)
            terms.append(term if denested is None else denested)
        value = sp.Add(*terms)
        if _power_evaluated(number, value):
            return value
        value = min(value, _constants_divided(number), key=sp.count_ops)
        try:
            shorter = sp.count_ops(value) < sp.count_ops(number)
        except Exception:
            shorter = False
        return value if shorter else number

    # -- products -----------------------------------------------------------

    def _product(self, product: sp.Basic) -> sp.Basic:
        """A product, cancelled but never multiplied out.

        Common factors of numerator and denominator are cancelled, so `(x^2 + 2*x*y +
        y^2)/(x^2 - y^2)` becomes `(x + y)/(x - y)`. The denominator keeps its original
        factors minus what cancelled: `1/((x^2 + y^2 + 6*y + 18)*(x^2 + y^2 - 6*y +
        18))` is not expanded into a quartic. A constant other than a number is a
        factor like any other: `(pi - pi*ERF(z))/pi` is `1 - ERF(z)`.
        """
        if not (product.is_Mul or _reciprocal_sum(product)):
            return product
        # An undecidable four-argument `IF` can put a truth value in a product; only
        # expressions have numerators.
        if not all(isinstance(factor, sp.Expr) for factor in product.args):
            return product
        tops, bottoms = _fraction(product)
        tops, bottoms = _constants_cancelled(tops, bottoms)
        tops, bottoms = _shared_powers_cancelled(tops, bottoms)
        tops, bottoms = _symbolic_powers_multiplied_in(tops, bottoms)
        if not all(bottom.is_Number for bottom in bottoms):
            cancelled = _cancelled(tops, bottoms)
            if cancelled is not None:
                # Factors left after cancelling a gcd are new and must be written.
                tops = [self.written(factor) for factor in cancelled[0]]
                bottoms = [self.written(factor) for factor in cancelled[1]]
        answer = _quotient(tops, bottoms)
        # Cancelling the whole denominator can leave a number times a sum, which sympy
        # turns into a sum, so write it as one.
        return self._step(answer) if answer.is_Add else _folded(answer)

    # -- sums ---------------------------------------------------------------

    def _sum(self, total: sp.Basic) -> sp.Basic:
        """One sum written about its generator, and its free part about the next.

        A common factor is sought twice, before and after writing, since collecting
        can create one (`phi*x*y - y*(phi + omega)`), and the result must be stable
        under the rule.
        """
        if not total.is_Add:
            return total
        unified = _unified_kernels(total)
        if unified is not total:
            # Unified kernels may cancel, leaving something other than a sum.
            return self._step(unified)
        generator = self._main(total)
        if generator is None:
            return total
        common = self._common_factor(total, generator)
        if common is not None:
            return common
        answer = self._about(total, generator)
        if not answer.is_Add:
            return answer
        generator = self._main(answer)
        common = None if generator is None else self._common_factor(answer, generator)
        return answer if common is None else common

    def _about(self, total: sp.Add, generator: sp.Basic) -> sp.Basic:
        """`total` as a rational function of `generator`, the rest about the next."""
        holding = [term for term in total.args if _holds(term, generator)]
        free = [term for term in total.args if not _holds(term, generator)]
        holder = _Holder()
        # Replace a compound generator by a symbol during the algebra so nothing
        # reaches inside it; multiplying out would e.g. split an integral in two.
        symbol = holder.standing_for(generator)
        frozen = sp.Add(*(_frozen(term, generator, symbol, holder) for term in holding))

        def thaw(part: sp.Basic) -> sp.Basic:
            return self.written(part.xreplace(holder.held))

        written = None
        if _thaws_back(frozen, holder, holding) and _expanded_terms(frozen) <= _EXPANSION:
            try:
                written = _rational_in(frozen, symbol, thaw)
            except Exception:
                written = None
        if written is None:
            # Not a rational function of the generator (`1/SQRT(x) + 1/x`, `(x + 1)^n
            # + y`), or too large to multiply out: keep the terms holding it as is.
            terms = list(holding)
            rest = self.written(sp.Add(*free))
            return _over_one_denominator(self._shared(sp.Add(*terms, rest)))
        terms, constant = written
        # The constant term joins the terms free of the generator, and that sum is
        # written about its own most main generator.
        terms.append(self.written(constant.xreplace(holder.held) + sp.Add(*free)))
        return self._shared(sp.Add(*terms))

    def _common_factor(self, total: sp.Add, generator: sp.Basic) -> sp.Basic | None:
        """A sum of compound products with a factor in common, written with it outside.

        `2*(x^2 - y^2)^6 - (x^2 - y^2)^5*(2*x^2 - 3)` becomes `(x^2 - y^2)^5*(3 -
        2*y^2)`, as in Derive, rather than a degree-twelve polynomial in `x`. A factor
        is common if every term has it, with the lowest power shared.

        Only if some term has a sum as a factor. Extracting a common factor from a sum
        of monomials is factoring, Factor's job: `x^2 + 2*x`, `2*x + 2*y` and `y*x^2 +
        2*y*x + 2*y` are left alone.

        Never a power of the generator itself, since the sum is a polynomial in it: `x^2
        + 2*x*(a + 1)^10` is already in normal form, and taking `x` out would give a
        different one.
        """
        if not any(_is_compound(term) for term in total.args):
            return None
        shares = [_positive_powers(term) for term in total.args]
        common = {
            base: min(share[base] for share in shares)
            for base in shares[0]
            if base != generator and all(base in share for share in shares[1:])
        }
        if not common:
            return None
        factor = sp.Mul(*(base**exponent for base, exponent in common.items()))
        quotients = [term / factor for term in total.args]
        if not _divided_out(quotients, shares, common):
            quotients = [_without(term, common) for term in total.args]
        if not _divided_out(quotients, shares, common):
            # The factor didn't divide out, as when the exponent of an exponential is
            # written in a form sympy doesn't cancel against its negation: taking it
            # out again and again would never end.
            return None
        rest = self.written(sp.Add(*quotients))
        return self._step(sp.Mul(factor, rest))

    def _shared(self, total: sp.Basic) -> sp.Basic:
        """A sum whose terms share a coefficient's denominator, written over it.

        So `(x + 1)^2/y + z` becomes `(x^2 + 2*x + y*z + 1)/y` (three terms over `y`),
        while `x/y + z` stays (only one). Two terms suffice: `(u - v)/h*(x - a) + w`
        becomes `(x*(u - v) - a*(u - v) + h*w)/h`.

        Denominators containing the generator are excluded, having been handled by
        the rational step. Without `over_numbers`, numeric denominators are excluded
        too, being just rational coefficients.
        """
        if not total.is_Add:
            return total
        generator = self._main(total)
        seen: dict[sp.Basic, int] = {}
        for term in total.args:
            if not isinstance(term, sp.Expr):
                return total
            denominator = _carried(term)[1]
            if denominator == 1:
                continue
            if denominator.free_symbols:
                if generator is not None and _holds(denominator, generator):
                    continue
            elif not self.over_numbers:
                continue
            seen[denominator] = seen.get(denominator, 0) + 1
        if not any(count > 1 for count in seen.values()):
            return total
        holder = _Holder()
        # Freeze the numerators but not the denominators, which are combined over.
        parts = []
        symbol = None if generator is None else holder.standing_for(generator)
        for numerator, denominator in map(_carried, total.args):
            if generator is not None:
                numerator = _frozen(numerator, generator, symbol, holder)
            parts.append(numerator / denominator)
        numerator, denominator = sp.fraction(sp.together(sp.Add(*parts)))
        if denominator == 1:
            return total
        numerator = self.written(numerator.xreplace(holder.held))
        if not denominator.free_symbols and isinstance(numerator, sp.Add):
            # Keep the number outside, or sympy would multiply it into the sum.
            return _keep_coeff(1 / denominator, numerator)
        return _quotient(
            list(sp.Mul.make_args(numerator)), list(sp.Mul.make_args(denominator))
        )

    def _main(self, total: sp.Basic) -> sp.Basic | None:
        """The most main generator of `total`, by the order list.

        A kernel ranks by its most main variable, plain variables before compound
        kernels with the same one: `x` before `LN(x)` before `y`.
        """
        best = None
        for kernel in _kernels(total):
            names = [s.name for s in kernel.free_symbols if type(s) is sp.Symbol]
            if not names:
                continue
            key = (min(map(self._rank, names)), not kernel.is_Symbol, str(kernel))
            if best is None or key < best[0]:
                best = (key, kernel)
        return None if best is None else best[1]

    def _rank(self, name: str) -> tuple[int, str]:
        if name in self.order:
            return self.order.index(name), ""
        return len(self.order), name


def _rational_in(
    frozen: sp.Expr, generator: sp.Basic, thaw: Callable[[sp.Basic], sp.Basic]
) -> tuple[list[sp.Basic], sp.Basic]:
    """The terms holding the generator, as a polynomial in it plus a proper part.

    Everything free of the generator is frozen, so denominators free of it belong to
    coefficients and are handled when those are written. That leaves `x/y + z`
    alone while `1/(x + 1) + 1/(x + 2) + y` becomes `(2*x + 3)/((x + 1)*(x + 2)) + y`.

    Returns the written terms and the constant term, still frozen, to which the terms
    free of the generator are added. Raises if there is no ratio of polynomials in
    the generator, as in `SIN(x)/(x + 1)`.
    """
    # Hide roots of quotients behind symbols during the algebra, since their bases
    # are written separately: the root of `(x + 1)/(x - 1)` would otherwise become
    # the root of `2/(x - 1) + 1`.
    hidden: dict[sp.Basic, sp.Basic] = {}
    frozen = _roots_of_quotients_hidden(frozen, hidden)
    shown = {symbol: root for root, symbol in hidden.items()}
    numerator, denominator = sp.fraction(sp.together(frozen))
    numerator = _expanded(numerator)
    cancelled = _cancelled([numerator], list(sp.Mul.make_args(denominator)))
    if cancelled is not None:
        numerator = _expanded(sp.Mul(*cancelled[0]))
        denominator = sp.Mul(*cancelled[1])
    if denominator.has(generator):
        quotient, remainder = sp.div(numerator, denominator, generator)
    else:
        # A denominator free of the generator is a coefficient and divides each term:
        # `(x + y*z)/y` is `x/y + z`, a polynomial in `x`.
        quotient, remainder = _expanded(numerator / denominator), sp.S.Zero
    terms, constant = _collected(quotient.xreplace(shown), generator, thaw)
    if remainder != 0:
        rest, rest_constant = _collected(
            _expanded(remainder).xreplace(shown), generator, thaw
        )
        # Cancel again unfrozen, since the frozen gcd saw only the generator.
        tops = [sp.Add(*rest, thaw(rest_constant))]
        bottoms = [
            thaw(factor.xreplace(shown)) for factor in sp.Mul.make_args(denominator)
        ]
        cancelled = _cancelled(tops, bottoms)
        if cancelled is not None:
            tops, bottoms = ([thaw(factor) for factor in side] for side in cancelled)
        terms.append(_quotient(tops, bottoms))
    return terms, constant


def _collected(
    polynomial: sp.Basic, generator: sp.Basic, thaw: Callable[[sp.Basic], sp.Basic]
) -> tuple[list[sp.Basic], sp.Basic]:
    """A multiplied-out polynomial collected by each term's generator-dependent part.

    Usually that is a power of the generator, and each coefficient is thawed and
    written with numeric content inside: `6*x*y^2*z + 2*x*z^3` is `x*(6*y^2*z +
    2*z^3)`, as the manual's LAPLACIAN answers. The generator may also appear inside
    a kernel like `(1 + i)^n` or `LN(x)`, collected likewise: the annuity equation's
    `v*(1 + i)^n + p*t*(1 + i)^n - (p*t + v)*(1 + i)^n` collects to zero. Terms free
    of the generator are returned separately as the constant term.
    """
    groups: dict[sp.Basic, list[sp.Basic]] = {}
    constant = []
    for term in sp.Add.make_args(polynomial):
        if not term.has(generator):
            constant.append(term)
            continue
        holding, rest = [], []
        for factor in sp.Mul.make_args(term):
            (holding if factor.has(generator) else rest).append(factor)
        groups.setdefault(sp.Mul(*holding), []).append(sp.Mul(*rest))
    terms = [
        thaw(sp.Mul(sp.Add(*coefficients), key)) for key, coefficients in groups.items()
    ]
    return terms, sp.Add(*constant)


def _fraction(product: sp.Expr) -> tuple[list[sp.Expr], list[sp.Expr]]:
    """A product's numerator and denominator factors, as lists.

    Kept as lists because sympy distributes a number over a sum when multiplying: the
    numerator of `2*(a + b)/(n + 1)` built alone would be `2*a + 2*b`.
    """
    tops, bottoms = [], []
    for factor in sp.Mul.make_args(product):
        top, bottom = sp.fraction(factor)
        if top != 1:
            tops.append(top)
        if bottom != 1:
            bottoms.append(bottom)
    return tops, bottoms


def _cancelled(
    tops: list[sp.Expr], bottoms: list[sp.Expr]
) -> tuple[list[sp.Expr], list[sp.Expr]] | None:
    """The numerator and denominator factors with common factors cancelled, or None.

    Cancelling factor by factor keeps the rest in shape: `(x + 1)^3*(x + 2)/(x + 1)` is
    `(x + 1)^2*(x + 2)`, not a cubic. A partly cancelled factor is replaced by what
    remains. None if nothing cancels.

    Factors too large to expand are skipped, since gcd works on the expanded
    polynomial: `(v + w + x + y + z)^60/q` would be six hundred thousand terms.
    """
    highs = [list(factor.as_base_exp()) for factor in tops]
    lows = [list(factor.as_base_exp()) for factor in bottoms]
    changed = False
    for low in lows:
        if not _cancellable(low):
            continue
        for high in highs:
            while low[1] > 0 and high[1] > 0 and _cancellable(high):
                try:
                    common = sp.gcd(high[0], low[0])
                except Exception:
                    break
                if common.is_Number:
                    break
                for side, sides in ((high, highs), (low, lows)):
                    rest = sp.quo(side[0], common)
                    side[1] -= 1
                    if rest != 1:
                        sides.append([rest, sp.S.One])
                changed = True
    if not changed:
        return None
    return _powers(highs), _powers(lows)


def _shared_powers_cancelled(
    tops: list[sp.Expr], bottoms: list[sp.Expr]
) -> tuple[list[sp.Expr], list[sp.Expr]]:
    """The factors with the power of a variable that every term of a numerator sum has
    cancelled against its power in the denominator, where an exponent is not an integer.

    The gcd takes `x^(18/7)` for a generator apart from `x`, and knows nothing of
    symbolic exponents, so it finds nothing common to `4*x^(18/7) - 3*x^2` and `x^2`,
    where the quotient is `x^(4/7) - 3/4`, or to `x^(2*k + 3) + x^(4*k + 3)` and
    `x^(4*k + 3)`, where for positive `k` it is `x^(-2*k) + 1`. A number of the power
    is cancelled only in whole: `(x + SQRT(x))/x^2` stays.
    """
    lows = {}
    for index, bottom in enumerate(bottoms):
        base, exponent = bottom.as_base_exp()
        if base.is_Symbol and base not in lows:
            lows[base] = index
    if not lows:
        return tops, bottoms
    tops, bottoms = list(tops), list(bottoms)
    for place, top in enumerate(tops):
        if not top.is_Add:
            continue
        for variable, index in lows.items():
            low = bottoms[index].as_base_exp()[1]
            exponents = [term.as_powers_dict().get(variable, sp.S.Zero) for term in top.args]
            if all(exponent.is_Integer for exponent in exponents) and low.is_Integer:
                continue
            least = _least(exponents)
            if least is None:
                continue
            if least.is_Number:
                least = sp.floor(least)
            if low.is_Number:
                low = sp.floor(low)
            shared = _least([least, low])
            if shared is None or not shared.is_positive:
                continue
            top = sp.Add(*(_lowered(term, variable, shared) for term in top.args))
            bottoms[index] = variable ** sp.expand(
                bottoms[index].as_base_exp()[1] - shared
            )
        tops[place] = top
    return tops, [bottom for bottom in bottoms if bottom != 1]


def _symbolic_powers_multiplied_in(
    tops: list[sp.Expr], bottoms: list[sp.Expr]
) -> tuple[list[sp.Expr], list[sp.Expr]]:
    """The factors with a reciprocal power of a variable to a symbolic exponent
    multiplied into a numerator sum with a term in that very power, as Derive writes
    it: `x^(-2*k)*(x^(2*k)*(1 - 4*v^2)/4 + z^2)` is `z^2*x^(-2*k) - v^2 + 1/4`. A
    numeric power stays a denominator: `(x^2 + z)/x^2`.
    """
    tops, bottoms = list(tops), list(bottoms)
    for bottom in list(bottoms):
        base, exponent = bottom.as_base_exp()
        if not base.is_Symbol or exponent.is_Number:
            continue
        for place, top in enumerate(tops):
            # Only where a term is left free of the power, so that the sum does not
            # share it again for the common factor to take back out.
            if top.is_Add and any(
                sp.expand(term.as_powers_dict().get(base, sp.S.Zero) - exponent) == 0
                for term in top.args
            ):
                tops[place] = sp.Add(*(term / bottom for term in top.args))
                bottoms.remove(bottom)
                break
    return tops, bottoms


def _lowered(term: sp.Expr, variable: sp.Symbol, shared: sp.Expr) -> sp.Expr:
    """`term` with the power of `variable` in it lowered by `shared`."""
    rest = [f for f in sp.Mul.make_args(term) if f.as_base_exp()[0] != variable]
    total = term.as_powers_dict().get(variable, sp.S.Zero)
    return sp.Mul(*rest, variable ** sp.expand(total - shared))


def _least(exponents: list[sp.Expr]) -> sp.Expr | None:
    """The exponent no greater than any other, if that can be told."""
    for candidate in exponents:
        if all((exponent - candidate).is_nonnegative for exponent in exponents):
            return candidate
    return None


def _constants_cancelled(
    tops: list[sp.Expr], bottoms: list[sp.Expr]
) -> tuple[list[sp.Expr], list[sp.Expr]]:
    """The factors with a constant common to a numerator sum's terms merged into the
    constant factors, if that leaves a rational number.

    `(SQRT(2)*x + SQRT(2))/SQRT(2)` is `SQRT(2)*(SQRT(2)*x + SQRT(2))/2` to sympy, and
    is `x + 1`.
    """
    constants = [top for top in tops if not top.free_symbols]
    numbers = [bottom for bottom in bottoms if not bottom.free_symbols]
    if not constants or all(top.is_Rational for top in constants):
        return tops, bottoms
    constant = sp.Mul(*constants) / sp.Mul(*numbers)
    rest = [top for top in tops if top.free_symbols]
    for index, top in enumerate(rest):
        if not top.is_Add:
            continue
        factored = _attempt(top, sp.factor_terms)
        if factored is None:
            continue
        common, primitive = factored.as_independent(
            *top.free_symbols, as_Add=False
        )
        merged = constant * common
        if common.is_Rational or not merged.is_Rational:
            continue
        rest[index] = primitive
        return [merged, *rest], [b for b in bottoms if b.free_symbols]
    return tops, bottoms


def _cancellable(factor: list[sp.Basic]) -> bool:
    base, exponent = factor
    return (
        bool(exponent.is_Integer)
        and exponent > 0
        and not base.is_Number
        and _expanded_terms(base) <= _EXPANSION
    )


def _powers(factors: list[list[sp.Basic]]) -> list[sp.Expr]:
    return [base**exponent for base, exponent in factors if exponent != 0 and base != 1]


def _quotient(tops: list[sp.Expr], bottoms: list[sp.Expr]) -> sp.Expr:
    """Numerator factors over denominator factors, written as Derive writes a quotient.

    Over a denominator with a variable, a sum in the numerator has its common factor
    taken out, as a gcd leaves it: `(n^2 + n)/(x + 1)` is `n*(n + 1)/(x + 1)`, and
    `(4*x + 6)/(x^2 + 1)` is `2*(2*x + 3)/(x^2 + 1)`. This is not full factoring:
    `(x^2 - 1)/(y + 1)` stays.

    Over a number, a number alone beside a sum in the numerator is multiplied in, as
    sympy multiplies it in anyway, and the integer content of a numerator sum cancels
    against the number: `(x - 1)*(6*x^2 + 12*x)/6` is `(x - 1)*(x^2 + 2*x)`.
    """
    tops, bottoms = _cleared(tops, bottoms)
    if bottoms and all(bottom.is_Integer for bottom in bottoms):
        tops, bottoms = _contents_cancelled(tops, bottoms)
    if any(bottom.free_symbols for bottom in bottoms):
        tops, bottoms = _numbers_taken_out(tops, bottoms)
        tops = [part for top in tops for part in sp.Mul.make_args(_content(top))]
        numbers = [top for top in tops if top.is_Number]
        sums = [top for top in tops if top.is_Add]
        if len(numbers) == len(sums) == 1 == len(tops) - 1 and numbers[0] < 0:
            # The sign of a number beside a lone sum goes into the sum: `(a^2 -
            # 2*x^2)/x`, not `-(2*x^2 - a^2)/x`. An all-negative sum gives it back
            # below.
            tops = [-numbers[0], -sums[0]]
    elif bottoms and len(tops) == 2 and sum(top.is_Number for top in tops) == 1:
        tops = [sp.Add(*map(_folded, sp.Add.make_args(sp.Mul(*tops))))]
    # Pull the sign out of an all-negative sum, `-z*(b + 2)` not `z*(-b - 2)`, which
    # is also how the text reads back.
    negated = [_all_negated(top) for top in tops]
    if any(negated):
        tops = [-top if flip else top for top, flip in zip(tops, negated)]
        tops.append(sp.Integer(-1) ** sum(negated))
    quotient = sp.Mul(*tops, *(sp.Pow(bottom, -1) for bottom in bottoms))
    return _coefficient_spelled(quotient)


def _contents_cancelled(
    tops: list[sp.Expr], bottoms: list[sp.Expr]
) -> tuple[list[sp.Expr], list[sp.Expr]]:
    """The factors over an integer denominator with the part of each numerator sum's
    integer content that divides the denominator cancelled.

    Only what cancels is taken out, so a sum keeps its coefficients otherwise:
    `(x - 1)*(2*x + 4)/6` is `(x - 1)*(x + 2)/3`.
    """
    denominator = int(sp.Mul(*bottoms))
    tops = list(tops)
    for index, top in enumerate(tops):
        if not top.is_Add:
            continue
        content, primitive = top.as_content_primitive()
        if not content.is_Integer:
            continue
        shared = sp.igcd(int(content), denominator)
        if shared == 1:
            continue
        tops[index] = sp.Add(*(term / shared for term in top.args))
        denominator //= shared
    if denominator == int(sp.Mul(*bottoms)):
        return tops, bottoms
    return tops, [] if denominator == 1 else [sp.Integer(denominator)]


def _folded(product: sp.Expr) -> sp.Expr:
    """An integer coefficient that is a power of a power's base, folded into the power.

    Sympy folds `2*2^n` into `2^(n + 1)` but leaves `-2*2^n`, yet `- 2*2^n*u` reads
    back as the former. So the coefficient is folded regardless of sign: `-4*2^n`
    becomes `-2^(n + 2)`, as the next Simplify would read it.
    """
    if not product.is_Mul:
        return product
    coefficient, rest = product.as_coeff_Mul()
    if not (coefficient.is_Integer and abs(coefficient) > 1):
        return product
    factors = list(sp.Mul.make_args(rest))
    for index, factor in enumerate(factors):
        base, exponent = factor.as_base_exp()
        if base.is_Integer and base > 1 and not exponent.is_Number:
            power, exact = integer_log(abs(int(coefficient)), int(base))
            if exact:
                sign = 1 if coefficient > 0 else -1
                others = factors[:index] + factors[index + 1 :]
                return sp.Mul(sign, *others, sp.Pow(base, exponent + power))
    return product


def _cleared(
    tops: list[sp.Expr], bottoms: list[sp.Expr]
) -> tuple[list[sp.Expr], list[sp.Expr]]:
    """The factors with fractional coefficients cleared from sums in the denominator.

    A denominator has no denominators of its own, even numeric: `2/(u/2 + w/2)` is
    `4/(u + w)`, the one canonical form among all scalings. Otherwise the two could
    appear side by side as terms that cancel without looking like it.
    """
    cleared, scale = [], sp.S.One
    for bottom in bottoms:
        base, exponent = bottom.as_base_exp()
        if base.is_Add and exponent.is_Integer and exponent > 0:
            numbers = [term.as_coeff_Mul()[0] for term in base.args]
            if all(number.is_Rational for number in numbers):
                shared = reduce(sp.ilcm, (number.q for number in numbers), 1)
                if shared != 1:
                    base = sp.Add(*(shared * term for term in base.args))
                    scale *= sp.Integer(shared) ** exponent
                    bottom = base**exponent
        cleared.append(bottom)
    if scale == 1:
        return tops, bottoms
    number = sp.Mul(scale, *(top for top in tops if top.is_Number))
    return [number, *(top for top in tops if not top.is_Number)], cleared


def _numbers_taken_out(
    tops: list[sp.Expr], bottoms: list[sp.Expr]
) -> tuple[list[sp.Expr], list[sp.Expr]]:
    """The factors with the numeric content of each sum taken out of it, above and below
    the line, and combined into one number in front.

    Derive writes a sum in a quotient with coprime integer coefficients and its content
    in front: `(3 - 4*x)/(2*x^2 + 2)` is `(3 - 4*x)/(2*(x^2 + 1))`, and `(4*x + 6)/(x^2
    + 1)` is `2*(2*x + 3)/(x^2 + 1)`. The contents then cancel: `(2*x + 2)/(4*x^2 + 8)`
    is `(x + 1)/(2*(x^2 + 2))`. A denominator sum gives up its sign with its content,
    while a numerator's sign is settled by `_quotient`.
    """
    number = sp.S.One
    taken_above = []
    for top in tops:
        if top.is_Number:
            number *= top
            continue
        content, rest = _numeric_content(top, False)
        number *= content
        taken_above.append(rest)
    taken_below = []
    for bottom in bottoms:
        if bottom.is_Number:
            number /= bottom
            continue
        content, rest = _numeric_content(bottom, True)
        base, exponent = rest.as_base_exp()
        if _all_negated(base) and exponent.is_Integer:
            content, rest = content * (-1) ** exponent, (-base) ** exponent
        number /= content
        taken_below.append(rest)
    return [number, *taken_above], taken_below


def _numeric_content(factor: sp.Expr, below: bool) -> tuple[sp.Expr, sp.Expr]:
    """The numeric content of a sum, or of a positive integer power of one, and the rest.

    `(4*x + 6)^2` is `4` and `(2*x + 3)^2`. Above the line the content may be a
    fraction, `(x/2 + 1/3)` giving `1/6` and `3*x + 2`; below it only an integer is
    taken, as `_cleared` has already cleared what fractions it can.
    """
    base, exponent = factor.as_base_exp()
    if not (base.is_Add and exponent.is_Integer and exponent > 0):
        return sp.S.One, factor
    content, primitive = base.as_content_primitive()
    if not (content.is_Integer if below else content.is_Rational) or content == 1:
        return sp.S.One, factor
    return content**exponent, primitive**exponent


def _reciprocal_sum(expression: sp.Basic) -> bool:
    """Whether `expression` is a sum to a negative integer power, a quotient alone."""
    return (
        isinstance(expression, sp.Pow)
        and expression.base.is_Add
        and expression.exp.is_Integer
        and expression.exp < 0
    )


def _all_negated(factor: sp.Expr) -> bool:
    return isinstance(factor, sp.Add) and all(
        term.could_extract_minus_sign() for term in factor.args
    )


def _content(top: sp.Expr) -> sp.Expr:
    if not isinstance(top, sp.Add):
        return top
    content = _attempt(top, sp.factor_terms)
    if isinstance(content, sp.Mul) and not content.as_coeff_Mul()[1].is_Add:
        return content
    return top


def _coefficient_spelled(product: sp.Expr) -> sp.Expr:
    """A product containing one sum, with its rational coefficient inside or in front.

    Inside if that gives some term an integer coefficient, as in Derive's answer to
    the manual's difference equation, `(m^2/2 + m/2 + 1)*(m - 1)!`. In front if the
    fraction would go under every term: `x*(x - 1)/2`. Also in front if the product
    has a non-numeric denominator, so the sum keeps integer coefficients:
    `(4*x + 3)/(2*(x + 1)^2)`, not `(2*x + 3/2)/(x + 1)^2`.

    The terms' common denominator is taken out first, so `x*(x/2 - 1/2)` also becomes
    `x*(x - 1)/2`, and the product has one form regardless of input.
    """
    if not product.is_Mul:
        return product
    coefficient, rest = product.as_coeff_Mul()
    factors = sp.Mul.make_args(rest)
    sums = [factor for factor in factors if factor.is_Add]
    if not coefficient.is_Rational or len(factors) < 2 or len(sums) != 1:
        return product
    others = [factor for factor in factors if not factor.is_Add]
    # The sign stays in front rather than negating every term of the sum.
    sign = -1 if coefficient < 0 else 1
    coefficient = abs(coefficient)
    terms = list(sums[0].args)
    numbers = [term.as_coeff_Mul()[0] for term in terms]
    if not all(number.is_Rational for number in numbers):
        return product
    shared = reduce(sp.ilcm, (number.q for number in numbers), 1)
    if shared != 1:
        terms = [shared * term for term in terms]
        coefficient = coefficient / shared
    under = any(map(_under_the_line, others))
    if under:
        # The number common to the sum's terms goes in front, where it cancels with
        # the denominator: `(4*x + 6)/y` is `2*(2*x + 3)/y` and `(4*x + 6)/(4*y)` is
        # `(2*x + 3)/(2*y)`. A term's number includes content inside brackets or
        # denominators: `(2*b - 2*c)*u` has a 2, `4/(2*v + 2)` is `2/(v + 1)`.
        contents = [term.as_content_primitive() for term in terms]
        if all(content.is_Integer for content, _ in contents):
            common = reduce(sp.igcd, (content for content, _ in contents), 0)
            if common != 1 or shared != 1:
                terms = [content / common * rest for content, rest in contents]
                coefficient = coefficient * common
    rewritten = terms != list(sums[0].args)
    if coefficient.q == 1:
        if not under and all(sp.fraction(other)[0] == 1 for other in others):
            # An integer beside only denominators with no variable goes into the
            # sum, matching `_quotient` and how it reads back.
            if not rewritten and coefficient == 1:
                return product
            spread = sp.Add(*(_folded(coefficient * term) for term in terms))
            return sp.Mul(sign, *others, spread)
        if not rewritten:
            return product
        return sp.Mul(sign * coefficient, *others, sp.Add(*terms))
    spread = [coefficient * term for term in terms]
    if not under and any(term.as_coeff_Mul()[0].is_integer for term in spread):
        return sp.Mul(sign, *others, sp.Add(*spread))
    return sp.Mul(sign * coefficient, *others, sp.Add(*terms))


def _under_the_line(factor: sp.Expr) -> bool:
    """Whether the factor is a power with a negative numeric exponent, written in the
    denominator. `y^(-n)` and `#e^(-x)` don't qualify."""
    exponent = factor.as_base_exp()[1]
    return bool(exponent.is_Number and exponent.is_negative)


def _carried(term: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """A term's numerator and denominator, including denominators inside coefficients.

    `b*(1 - 1/n)` has denominator `n` just like `a/n`, so together they become `(a +
    b*(n - 1))/n`.
    """
    numerator, denominator = sp.fraction(term)
    inner = [
        factor
        for factor in sp.Mul.make_args(numerator)
        if factor.is_Add
        and any(sp.fraction(part)[1].free_symbols for part in factor.args)
    ]
    if not inner:
        return numerator, denominator
    combined = _attempt(term, sp.together)
    return (numerator, denominator) if combined is None else sp.fraction(combined)


def _over_one_denominator(total: sp.Basic) -> sp.Basic:
    """A sum whose terms are all over powers of the same base, combined over it.

    For a sum that isn't a rational function of its generator, a repeated denominator
    is still worth combining, since the lower power divides the higher: `x^2/(1 -
    x^2)^(3/2) + 1/SQRT(1 - x^2)` is `1/(1 - x^2)^(3/2)`, Derive's answer to
    `DIF(x/SQRT(1 - x^2), x)`. (Derive leaves that sum alone when authored; of the two
    fixed points, this picks the one it gives to the derivative.)

    Every term must have the denominator, and combining must shorten the sum. The
    first rules out `b*SQRT(b^2 - 4*a*c)/(4*a) - c*LN(b + SQRT(b^2 - 4*a*c)) -
    b^2/(4*a)`, Derive's answer to an integral. The second rules out `1/SQRT(x) + 1/x`
    over `x^(3/2)`, which Derive also leaves alone.
    """
    if not total.is_Add:
        return total
    bases = set()
    for term in total.args:
        if not isinstance(term, sp.Expr):
            return total
        denominator = sp.fraction(term)[1]
        if denominator == 1:
            return total
        bases.add(denominator.as_base_exp()[0])
    if len(bases) != 1:
        return total
    combined = _attempt(total, sp.together)
    if combined is None or sp.count_ops(combined) >= sp.count_ops(total):
        return total
    return combined


# -- freezing what is free of the generator --------------------------------------


class _Holder:
    """Placeholders for one sum's coefficients, and the map back.

    Each distinct coefficient gets one placeholder, so the same quantity in two terms
    is the same symbol to gcd and collecting: `x*(a + b) + (a + b)` has one
    coefficient, not two that happen to be equal.
    """

    def __init__(self) -> None:
        self.held: dict[sp.Dummy, sp.Basic] = {}
        self.placeholders: dict[sp.Basic, sp.Dummy] = {}

    def hold(self, expression: sp.Basic) -> sp.Basic:
        # Numbers aren't held, so `(2 + 3)^2` still evaluates to 25.
        if expression.is_number:
            return expression
        placeholder = self.placeholders.get(expression)
        if placeholder is None:
            placeholder = self.placeholders[expression] = sp.Dummy()
            self.held[placeholder] = expression
        return placeholder

    def standing_for(self, generator: sp.Basic) -> sp.Symbol:
        """The symbol to do the algebra in: the generator itself, or a stand-in."""
        if isinstance(generator, sp.Symbol):
            return generator
        symbol = sp.Dummy()
        self.held[symbol] = generator
        return symbol


def _frozen(
    expression: sp.Basic, generator: sp.Basic, symbol: sp.Symbol, holder: _Holder
) -> sp.Basic:
    """`expression` with every maximal part free of the generator replaced by a
    placeholder.

    In a sum or product, all parts free of the generator share one placeholder, so `x
    + 2*y + 1` becomes `x + d` and `(x + 2*y + 1)^3` about `x` is in powers of `2*y +
    1`. A product's numeric factor stays outside, being the coefficient collecting
    looks for, as does a power's numeric exponent. Heads binding their own variable
    are replaced whole, generator and all; see `expanding._BOUND`.
    """
    if expression == generator:
        return symbol
    if not _holds(expression, generator) or isinstance(expression, _BOUND):
        return holder.hold(expression)

    def frozen(part: sp.Basic) -> sp.Basic:
        return _frozen(part, generator, symbol, holder)

    if isinstance(expression, (sp.Add, sp.Mul)):
        involved, free = [], []
        for argument in expression.args:
            (involved if _holds(argument, generator) else free).append(argument)
        parts = [frozen(argument) for argument in involved]
        if isinstance(expression, sp.Mul):
            parts.extend(argument for argument in free if argument.is_Number)
            free = [argument for argument in free if not argument.is_Number]
        if free:
            parts.append(holder.hold(expression.func(*free)))
        return expression.func(*parts)
    if isinstance(expression, sp.Pow):
        base, exponent = expression.args
        if _holds(exponent, generator):
            exponent = frozen(exponent)
        elif not exponent.is_Number:
            exponent = holder.hold(exponent)
        return sp.Pow(frozen(base), exponent)
    if not expression.args:
        return holder.hold(expression)
    return expression.func(*(frozen(argument) for argument in expression.args))


def _thaws_back(frozen: sp.Basic, holder: _Holder, terms: list[sp.Basic]) -> bool:
    """Whether restoring the placeholders gives back the frozen terms.

    `_frozen` rebuilds unknown heads from rewritten arguments, and a head that doesn't
    survive that loses content (a `RootSum` once lost its sum entirely). If the check
    fails, the terms are left as is.
    """
    try:
        return frozen.xreplace(holder.held) == sp.Add(*terms)
    except Exception:
        return False


def _holds(expression: sp.Basic, generator: sp.Basic) -> bool:
    if isinstance(generator, sp.Symbol):
        return generator in expression.free_symbols
    return expression.has(generator)


def _kernels(expression: sp.Basic) -> set[sp.Basic]:
    """The maximal parts of `expression` that are not sums, products or powers.

    Powers with rational exponents are looked through, so `(x + 1)^(3/2)` and `SQRT(x
    + 1)` both have kernel `x` and can share a denominator. A power with a non-numeric
    exponent, like `(1 + i)^n`, is a kernel itself.
    """
    if isinstance(expression, (sp.Add, sp.Mul)):
        return {kernel for argument in expression.args for kernel in _kernels(argument)}
    if isinstance(expression, sp.Pow) and expression.exp.is_Rational:
        return _kernels(expression.base)
    if not expression.free_symbols or isinstance(expression, sp.Dummy):
        return set()
    return {expression}


def _unified_kernels(total: sp.Add) -> sp.Basic:
    """`total` with kernels differing only in how their arguments are written unified.

    Products aren't multiplied out, so two normal-form arguments can be the same
    polynomial written differently: `2*x*(y + 1) + y` and `x*(2*y + 2) + y`. Their
    calls then look distinct, and `#e^(2*x*(y + 1) + y) - #e^(2*x*y + 2*x + y)` would
    not cancel. Each such group is replaced by its shortest member. Returns `total`
    itself if nothing is unified.
    """
    groups: dict[tuple, list[sp.Basic]] = {}
    for kernel in _calls(total):
        if not all(_expanded_terms(a) <= _EXPANSION for a in kernel.args):
            continue
        try:
            key = (kernel.func, tuple(_expanded(a) for a in kernel.args))
        except Exception:
            continue
        group = groups.setdefault(key, [])
        if kernel not in group:
            group.append(kernel)
    replacements = {}
    for group in groups.values():
        if len(group) < 2:
            continue
        chosen = min(group, key=lambda k: (sp.count_ops(k), sp.default_sort_key(k)))
        replacements.update({k: chosen for k in group if k != chosen})
    if not replacements:
        return total
    return total.xreplace(replacements)


def _calls(expression: sp.Basic) -> list[sp.Basic]:
    """The function calls and symbolic powers in `expression` containing a variable.

    Like the walk, this doesn't look inside heads that bind a variable.
    """
    found = []
    if isinstance(expression, _BOUND) or expression.is_Atom:
        return found
    if isinstance(expression, (sp.Function, sp.exp)) or (
        isinstance(expression, sp.Pow) and not expression.exp.is_Number
    ):
        if expression.free_symbols and isinstance(expression, sp.Expr):
            found.append(expression)
    for argument in expression.args:
        found.extend(_calls(argument))
    return found


def _is_compound(term: sp.Basic) -> bool:
    """Whether a term has a sum, or a positive integer power of one, as a factor."""
    for factor in sp.Mul.make_args(term):
        base, exponent = factor.as_base_exp()
        if base.is_Add and exponent.is_Integer and exponent > 0:
            return True
    return False


def _positive_powers(term: sp.Basic) -> dict[sp.Basic, sp.Basic]:
    """A term's numerator factors containing a variable, as a base-to-exponent map."""
    powers: dict[sp.Basic, sp.Basic] = {}
    for factor in sp.Mul.make_args(term):
        base, exponent = factor.as_base_exp()
        if not exponent.is_Number:
            # A power with a non-numeric exponent is shared whole: `(1 + i)^n`, not
            # some power of `1 + i`.
            base, exponent = factor, sp.S.One
        if not base.free_symbols:
            continue
        if exponent.is_Rational and exponent > 0:
            powers[base] = powers.get(base, sp.S.Zero) + exponent
    return powers


def _divided_out(
    quotients: list[sp.Basic],
    shares: list[dict[sp.Basic, sp.Basic]],
    common: dict[sp.Basic, sp.Basic],
) -> bool:
    """Whether each quotient has less of each common factor than its term had."""
    for quotient, share in zip(quotients, shares, strict=True):
        left = _positive_powers(quotient)
        if any(left.get(base, sp.S.Zero) >= share[base] for base in common):
            return False
    return True


def _without(term: sp.Basic, common: dict[sp.Basic, sp.Basic]) -> sp.Basic:
    """`term` divided by the product of `common`'s powers, factor by factor as
    `_positive_powers` takes them apart, so that a factor goes even where its written
    form isn't the one sympy would cancel: `-2*t` in an exponent may be written
    `(-1)*2*t`, so that `#e^(-2*t)/#e^(-2*t)` stays."""
    left = dict(common)
    factors = []
    for factor in sp.Mul.make_args(term):
        base, exponent = factor.as_base_exp()
        if not exponent.is_Number:
            base, exponent = factor, sp.S.One
        taken = left.pop(base, None)
        if taken is None:
            factors.append(factor)
        elif exponent != taken:
            factors.append(base ** (exponent - taken))
    if left:
        return term / sp.Mul(*(base**exponent for base, exponent in left.items()))
    return sp.Mul(*factors)


# -- sizes ---------------------------------------------------------------------


def _expanded(expression: sp.Basic) -> sp.Basic:
    """Multiplied out, without splitting powers.

    Derive keeps `x^(a + b)` and `(x*y)^n` as they are. A logarithm sheds a positive
    factor only where a variable multiplies it: `x*LN(2*x)` is `x*LN(x) + x*LN(2)`, whose
    second term collects with others in `x`, while `LN(2*x) + x` stays.
    """
    expanded = sp.expand(expression, power_exp=False, power_base=False, log=False)
    return sp.Add(
        *(
            term
            if _lone_logarithm(term)
            else sp.expand(term, power_exp=False, power_base=False)
            for term in sp.Add.make_args(expanded)
        )
    )


def _lone_logarithm(term: sp.Basic) -> bool:
    """Whether `term` is a constant multiple of one logarithm of a variable argument."""
    variable = [factor for factor in sp.Mul.make_args(term) if factor.free_symbols]
    return len(variable) == 1 and isinstance(variable[0], sp.log)


def _roots_of_quotients_hidden(
    expression: sp.Basic, hidden: dict[sp.Basic, sp.Basic]
) -> sp.Basic:
    """`expression` with each non-integer power of a base with a variable in its
    denominator replaced by a symbol, recorded in `hidden`."""
    if not expression.free_symbols or expression.is_Atom:
        return expression
    if isinstance(expression, sp.Pow) and not expression.exp.is_Integer:
        if sp.denom(sp.together(expression.base)).free_symbols:
            return hidden.setdefault(expression, sp.Dummy())
    return expression.func(
        *(_roots_of_quotients_hidden(argument, hidden) for argument in expression.args)
    )


#: Maximum number of terms in an expansion the normal form will build. Derive writes
#: `(v + w + x + y + z)^60 + 1` about `x` instantly, as sixty-one powers of `x`, and
#: freezing makes it equally cheap here. This guards against expansions freezing
#: doesn't shrink (a dozen squared binomials in the generator give half a million
#: terms) and gcds of numerators nobody asked to expand. The largest expansion in the
#: corpus is 120 terms, two orders of magnitude below this.
_EXPANSION = 20_000


def _expanded_terms(expression: sp.Basic) -> int:
    """An upper bound on the number of terms in `expression` multiplied out.

    Terms that would collect are counted separately, which errs on the safe side. The
    count is capped at `_CEILING` so that nested powers can't make estimating
    expensive.
    """
    if isinstance(expression, sp.Add):
        return min(sum(_expanded_terms(term) for term in expression.args), _CEILING)
    if isinstance(expression, sp.Mul):
        total = 1
        for factor in expression.args:
            total = min(total * _expanded_terms(factor), _CEILING)
        return total
    if isinstance(expression, sp.Pow):
        return _power_terms(expression)
    return 1


#: The cap on `_expanded_terms`, one past `_EXPANSION`.
_CEILING = _EXPANSION + 1


def _power_terms(power: sp.Pow) -> int:
    """The number of terms in a power of a sum, or one if it isn't one.

    `(u_1 + ... + u_k)^n` has `C(n + k - 1, k - 1)` terms, computed as a running
    product so it can stop as soon as it passes the limit.
    """
    terms = _expanded_terms(power.base)
    if terms <= 1 or not power.exp.is_Integer or power.exp <= 0:
        return 1
    exponent = int(power.exp)
    count = 1.0
    for index in range(1, terms):
        count *= (exponent + index) / index
        if count > _EXPANSION:
            return _CEILING
    return round(count)


#: Maximum total digits in an evaluated power's integers before the power is kept as
#: written. `(1 + SQRT(2))^100` has two forty-digit numbers, `(1 + SQRT(2))^1000` two of
#: nearly four hundred, which is unreadable compared to the power.
_POWER_DIGITS = 100


def _power_evaluated(number: sp.Expr, value: sp.Expr) -> bool:
    """Whether `value` is `number` with its powers of sums of surds evaluated.

    An integer power of a sum of rationals, `#i` and roots of rationals lies in the
    field the roots span, and Simplify answers with its value there, shorter or not:
    `(1 + SQRT(2))^5` is `29*SQRT(2) + 41` and `(2 + #i)^3` is `2 + 11*#i`. Only for
    fully algebraic numbers (so `pi*(1 + SQRT(2))^2` keeps its power), when no such
    power remains and the digits stay readable.
    """
    if not (_algebraic(number) and _algebraic(value)):
        return False
    if not any(_power_of_surds(power) for power in number.atoms(sp.Pow)):
        return False
    if any(_power_of_surds(power) for power in value.atoms(sp.Pow)):
        return False
    digits = sum(
        len(str(abs(atom.p))) + len(str(atom.q)) - 1 for atom in value.atoms(sp.Rational)
    )
    return digits <= _POWER_DIGITS


def _power_of_surds(power: sp.Pow) -> bool:
    return power.exp.is_Integer and abs(power.exp) > 1 and power.base.is_Add


def _algebraic(number: sp.Expr) -> bool:
    """Whether `number` is arithmetic on rationals, `#i` and roots of rationals."""
    return not number.atoms(sp.Function, sp.NumberSymbol) and all(
        power.exp.is_Rational and (power.exp.is_Integer or power.base.is_Rational)
        for power in number.atoms(sp.Pow)
    )


def _arithmetic(number: sp.Expr) -> sp.Expr | None:
    """A number multiplied out and rationalized, or None if that gets too large.

    Rationalizing multiplies the numerator by the conjugate, which for a least-squares
    fit's coefficients (quotients of sums of `SIN(4)` and `ATAN(4)`) yields far too
    many terms.

    Nor if multiplying out inside function arguments more than doubles the number,
    whose result would then be kept only if shorter: the third Newton step for `COS(x)
    - x` from 1, a tower of 150 operations, multiplies out to 2,700 and rationalizes
    for minutes.
    """
    if _expanded_terms(number) > _EXPANSION:
        return None
    expanded = sp.expand(number, log=False)
    if sp.count_ops(expanded) > 2 * sp.count_ops(number) + _GROWTH:
        return None
    rationalized = sp.radsimp(expanded)
    if _expanded_terms(rationalized) > _EXPANSION:
        return None
    return _expanded(rationalized)


def _constants_divided(number: sp.Expr) -> sp.Expr:
    """A number with symbolic constants, such as `pi` and `#e`, as a quotient of
    polynomials in them with the common factors cancelled, or the number itself.

    `(4*#e*pi^2 + #e)/(4*pi^2 + 1)` is `#e`. The cancelled factors are factors of the
    denominator, so nonzero wherever the number is defined. Only for numbers built of
    such constants and rationals by arithmetic, which the polynomial gcd handles.
    """
    if not number.atoms(sp.NumberSymbol) or number.atoms(sp.Function):
        return number
    if any(not power.exp.is_Integer for power in number.atoms(sp.Pow)):
        return number
    if _expanded_terms(number) > _EXPANSION:
        return number

    def divided(value: sp.Expr) -> sp.Expr:
        top, bottom = value.as_numer_denom()
        if not bottom.atoms(sp.NumberSymbol):
            return value
        # Only a factor other than a rational: otherwise `pi/2 + 1/2` would merely be
        # put over its denominator.
        if sp.gcd(top, bottom).is_Rational:
            return value
        return sp.cancel(value)

    cancelled = _attempt(number, divided)
    return number if cancelled is None else cancelled


#: Operations a number may gain by `_arithmetic` multiplying it out, beyond doubling.
_GROWTH = 20


def _rebuilt(expression: sp.Basic) -> sp.Basic:
    """Every sum and product in `expression` rebuilt with evaluation.

    A rewrite may return an unevaluated product such as `(SQRT(2) + SQRT(2)*#i)/2`, a
    number in front of a sum, which the notation can't represent: it reads back
    distributed.
    """
    if not expression.args or isinstance(expression, _BOUND):
        return expression
    arguments = [_rebuilt(argument) for argument in expression.args]
    if all(new is old for new, old in zip(arguments, expression.args)):
        if not isinstance(expression, (sp.Add, sp.Mul)):
            return expression
    return expression.func(*arguments)


def _attempt(expression, rewrite):
    try:
        candidate = rewrite(expression)
    except Exception:
        return None
    return candidate if isinstance(candidate, sp.Basic) else None

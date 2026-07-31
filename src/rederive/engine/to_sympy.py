"""Node -> sympy: a faithful translation, not a simplification.

This is one of the engine's two doors, and it belongs to no single command.
Every node kind maps to the sympy object that means the same thing, and nothing
more happens here than sympy's own automatic evaluation: `2 + 3` becomes `5`
because `Add` says so, not because Simplify asked. Command-agnostic semantics -
the constant and function tables, assumptions from declared domains, the angle
mode, the special-value overrides such as `SIGN(0)` - live here; which
transformations to attempt is the calling pipeline's business.

The mapping is total. A construct sympy has no head for converts to an inert
head that carries its operands and prints back to the author notation it came
from, so that anything the engine does not understand survives a round trip
untouched.

Substitution of labels, assignments and function definitions is a separate
pre-pass: `to_sympy` works on an unsubstituted tree just as well.
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

from rederive.engine.approximation import simplest
from rederive.engine.context import (
    Angle,
    Context,
    Domain,
    DomainKind,
    Precision,
    domain_of_node,
)
from rederive.engine.ordering import main_order
from rederive.model.expr import Kind, Node
from rederive.syntax.names import BUILTIN_FUNCTIONS

__all__ = [
    "COMMAND_HEADS",
    "Antidifference",
    "Antiquotient",
    "Approx",
    "Assign",
    "Declare",
    "Dot",
    "FunDef",
    "InertVector",
    "Logical",
    "PlusMinus",
    "Power",
    "StringLiteral",
    "Subscript",
    "Taylor",
    "Transposed",
    "reread",
    "to_sympy",
]

#: The heads a conversion deliberately leaves standing for the pipeline to
#: evaluate: the author-line spellings of Factor, Expand and soLve, and `RATE`,
#: which is a bounded soLve of section 6.12's annuity under another name. Their
#: answers depend on a simplified operand, so they cannot be worked out here,
#: and a call over one of them cannot be worked out either - `DIMENSION` of a
#: `SOLVE` is a number nobody knows yet. Such a call stays inert too, and
#: `reread` is what offers it again once the head inside it is gone.
COMMAND_HEADS = ("EXPAND", "FACTOR", "SOLVE", "RATE")


#: How big a number raising to a power may build before the engine declines to
#: build it: sixteen megabits, about five million decimal digits and two
#: megabytes of them. Generous by design, since it must never fire on
#: legitimate work - `2^1000000` is under it, and `10000!` is a power of
#: nothing - and far below what makes a machine swap.
POWER_BITS = 1 << 24


# -- inert heads ------------------------------------------------------------
#
# What sympy has no object for. Each is a head sympy will carry around but
# never rewrite, and the printer turns each back into the notation it came
# from.


class PlusMinus(sp.Function):
    """`+-u`, with the ambiguity sealed in.

    Its argument simplifies; arithmetic does not penetrate it, so `+-1 + 1`
    stays a sum of two things rather than becoming `0` or `2`.
    """

    nargs = 1


class StringLiteral(sp.Symbol):
    """A string literal: data, not mathematics.

    A `Symbol` rather than a function so that it is atomic everywhere sympy
    looks at it, and its own class so that `"x"` and the variable `x` are
    different objects.
    """


class Dot(sp.Function):
    """`u . v` where the shapes will not multiply."""

    nargs = 2


class Subscript(sp.Function):
    """`u SUB v` where `u` is no vector and the two are not both names.

    The common `x SUB 1` is a symbol in its own right rather than this, so
    that a subscripted variable can be differentiated with respect to and
    solved for. This head is for everything else, where a name would have to
    swallow a whole subexpression to be written at all.
    """

    nargs = 2


class Transposed(sp.Function):
    """``u` `` where `u` is not a matrix."""

    nargs = 1


class Power(sp.Function):
    """`u^v` whose value is a number too big to be worth building.

    A head of its own rather than a `Pow` built with `evaluate=False`, because
    that is not durable: `together` and `as_numer_denom` rebuild a power from
    its parts, and rebuilding is exactly the evaluation being refused. This
    carries the same two operands and no sympy routine can talk it into
    multiplying them out. See `_explodes` for what makes one.
    """

    nargs = 2


class InertVector(sp.Function):
    """A vector sympy will not hold as a matrix, a ragged one above all."""


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

    Sympy computes a series but has no head to hold an uncomputed one, so this
    is that head: it converts like the other calculus heads, waits where they
    wait, and prints back as the call it was authored as when it will not
    evaluate.
    """

    nargs = 4

    def doit(self, deep: bool = False, **hints) -> sp.Basic:
        """The Taylor polynomial, or this head where there is none.

        Derive's order is the maximum degree, and sympy's is the exponent the
        expansion is cut off before, so the order asked for is one less than
        the one passed on. The order term goes: a Taylor polynomial is a
        polynomial, which is also the test of whether one was found - a series
        in fractional powers, or one with a logarithm in it, is not one, and
        the head stays as it was written rather than answer with something that
        is no polynomial.
        """
        expression, variable, point, order = self.args
        if not (order.is_Integer and order >= 0):
            return self
        try:
            series = expression.series(variable, point, int(order) + 1).removeO()
        except Exception:
            return self
        return series if series.is_polynomial(variable) else self


class Approx(sp.Function):
    """`APPROX(u, n)`: `u` rounded to `n` digits, held until it can be.

    Rounding is the last thing a command does, and what it has to round is the
    answer and not the question: `APPROX(INT(u, x))` must see the value that
    integral came out as. So this waits where the calculus heads wait, and the
    pipeline that evaluates those evaluates this one after them - which the
    conversion cannot do itself, having no pipeline to call.

    The head carries its digit count even where the call left it out, because
    what precision the caller was working at is known here and nowhere later.
    """

    nargs = 2


class Antidifference(sp.Function):
    """`SUM(u, n)` with no limits, unevaluated until `.doit()`.

    `F` is the antidifference of `f` when `f(n) = F(n + 1) - F(n)`, which is
    the discrete analogue of an antiderivative and is what a definite sum
    telescopes down to. The constant of summation is dropped the way an
    integral's is.

    Sympy has no head for one, so this is that head, and it waits where the
    calculus heads wait for the reason `Taylor` gives.
    """

    nargs = 2

    def doit(self, deep: bool = False, **hints) -> sp.Basic:
        """The antidifference, or this head where there is none.

        Gosper's algorithm decides it: its certificate `t` is the rational
        function with `t(n)*f(n)` the antidifference, and no certificate means
        no hypergeometric antidifference exists - `SUM(1/n^2, n)` among them,
        which is a documented refusal and not an error. The certificate is
        canonical and carries no constant, so nothing has to be taken off it.

        Only the powers are gathered afterwards, so that `SUM(1/2^n, n)` comes
        out as one power rather than as `-2/2^n`.
        """
        expression, index = self.args
        if type(index) is not sp.Symbol:
            return self
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

    `F` is the antiquotient of `f` when `f(n) = F(n + 1)/F(n)`: the
    antidifference of section 7.5 with multiplication in place of addition, and
    what a definite product telescopes down to. The constant factor is dropped
    the way the constant of summation is.
    """

    nargs = 2

    def doit(self, deep: bool = False, **hints) -> sp.Basic:
        """The antiquotient, or this head where there is none.

        There is no routine for one, so it is built from its definition: the
        running product of the body up to `n - 1`, which telescopes to exactly
        an `F` with `F(n + 1)/F(n)` the body again.

        What sympy answers with is not always a closed form. A body Derive has
        no answer for comes back holding a `RisingFactorial`, which the
        notation has no spelling for, or the product itself, which is no answer
        at all; `combsimp` turns some of those into factorials, and where it
        does not the head stands, as it does for the manual's own
        `PRODUCT(k^2 + 1, k)`.
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


# -- the constant table -----------------------------------------------------

#: `e` and `i` are ordinary variables; only `#e` and `#i` are the constants.
#: The parser has already told the two apart, so nothing here needs to.
CONSTANTS: dict[str, sp.Basic] = {
    "pi": sp.pi,
    "#e": sp.E,
    "#i": sp.I,
    "inf": sp.oo,
    "deg": sp.pi / 180,
    "true": sp.true,
    "false": sp.false,
    "euler_gamma": sp.EulerGamma,
    # An arbitrary point on the complex unit circle, which is what Derive
    # answers `|z| = 2` with: `z = 2*unit_circle`. Sympy has no such object, so
    # it is a symbol - but a bare one, carrying none of the assumptions an
    # undeclared variable gets. Real is exactly what it is not, and `SQRT(x^2)
    # -> ABS(x)` firing on it would be a guess. The arithmetic Derive gives it,
    # `unit_circle*inf` for complex infinity above all, is out of scope.
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

_BOOLEAN: dict[Kind, Callable[..., sp.Basic]] = {
    Kind.NOT: sp.Not,
    Kind.AND: sp.And,
    Kind.OR: sp.Or,
    Kind.XOR: sp.Xor,
    Kind.IMP: sp.Implies,
}

#: How each boolean operator reads as a bitwise one on integers, two's
#: complement throughout: `NOT 5` is -6 and `3 OR 5` is 7.
_BITWISE: dict[Kind, Callable[..., int]] = {
    Kind.NOT: lambda a: ~a,
    Kind.AND: lambda a, b: a & b,
    Kind.OR: lambda a, b: a | b,
    Kind.XOR: lambda a, b: a ^ b,
    Kind.IMP: lambda a, b: ~a | b,
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
    return _Converter(context or Context()).convert(node)


class _Converter:
    """One conversion's worth of state: the context, and the labels in hand."""

    def __init__(self, context: Context, open_labels: frozenset[int] = frozenset()):
        self.context = context
        self.open_labels = open_labels

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
                return sp.Abs(self.convert(node.children[0]))
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
        # PARAMS and INTERVAL reach the engine only inside the node that owns
        # them; anything else is an inert head over whatever it holds.
        return self.opaque(str(node.kind).upper(), self._children(node))

    def _children(self, node: Node) -> list[sp.Basic]:
        return [self.convert(child) for child in node.children]

    # -- leaves -------------------------------------------------------------

    def _number(self, node: Node) -> sp.Basic:
        """A numeral, which `value` spells as `2.5`, as `25` or as `1/3`.

        Exact and Mixed read it as the rational it is, so `0.1` is one tenth
        and not the binary float nearest to it. Approximate rounds that
        rational to the current precision, which is where the difference
        between the two modes comes from: the numbers going in are rounded,
        and what is done with them afterwards is exact either way.
        """
        value = sp.Rational(str(node.value))
        if self.context.precision is Precision.APPROXIMATE:
            digits = self.context.precision_digits
            return sp.Rational(simplest(Fraction(value.p, value.q), digits))
        return value

    def _name(self, node: Node) -> sp.Basic:
        name = str(node.value)
        constant = CONSTANTS.get(name)
        if constant is not None:
            return constant
        return self.symbol(name)

    def symbol(self, name: str) -> sp.Basic:
        """The symbol `name` stands for, carrying its declared assumptions.

        A degenerate interval declares a value rather than a range, so `x
        :epsilon Real [7, 7]` makes `x` the number 7.
        """
        domain = self.context.domain(name)
        value = self._degenerate(domain)
        if value is not None:
            return value
        return sp.Symbol(name, **self.assumptions(domain))

    def assumptions(self, domain: Domain) -> dict[str, bool]:
        """The sympy assumptions a domain declaration is worth.

        Complex means "not known to be real", which is the assumption that
        stops `SQRT(x^2)` from becoming `ABS(x)`: sympy leaves `real`
        undecided under it, so no rewrite that needs a real variable fires.
        """
        if domain.kind is DomainKind.NONSCALAR:
            return {"commutative": False}
        if domain.kind is DomainKind.COMPLEX:
            return {"complex": True}
        facts = {"integer": True} if domain.kind is DomainKind.INTEGER else {"real": True}
        low, high = self._bound(domain.low), self._bound(domain.high)
        # An interval is worth the sign it implies, and a bound is strict
        # either because it is written open or because it is away from zero.
        if low is not None and low.is_nonnegative:
            strict = low.is_positive or not domain.closed_low
            facts["positive" if strict else "nonnegative"] = True
        elif high is not None and high.is_nonpositive:
            strict = high.is_negative or not domain.closed_high
            facts["negative" if strict else "nonpositive"] = True
        return facts

    def _degenerate(self, domain: Domain) -> sp.Basic | None:
        """The value of a one-point interval such as `[7, 7]`, if that is one."""
        if not (domain.closed_low and domain.closed_high):
            return None
        low, high = self._bound(domain.low), self._bound(domain.high)
        if low is None or high is None or low != high:
            return None
        return low

    def _bound(self, node: Node | None) -> sp.Basic | None:
        """What an interval bound is worth, or None if it is not a number.

        Bounds convert without domains in hand: they are what the domains are
        being read from, and a bound that named a declared variable would send
        the reading round in a circle.
        """
        if node is None:
            return None
        try:
            value = _Converter(replace(self.context, domains={})).convert(node)
        except Exception:
            return None
        return value if getattr(value, "is_number", False) else None

    def _label(self, node: Node) -> sp.Basic:
        """`#3`: whatever expression 3 is, or an inert `#3` if nobody knows.

        A label that reaches itself, directly or through another, stops being
        followed and stays inert.
        """
        try:
            number = int(str(node.value))
        except ValueError:
            number = -1
        target = self.context.labels.get(number)
        if target is None or number in self.open_labels:
            return sp.Symbol(f"#{node.value}")
        deeper = _Converter(self.context, self.open_labels | {number})
        return deeper.convert(target)

    # -- operators ----------------------------------------------------------

    def _sum(self, node: Node) -> sp.Basic:
        """A run of terms, with one operator per gap."""
        operators = str(node.value)
        terms = [self.convert(node.children[0])]
        for index, child in enumerate(node.children[1:]):
            term = self.convert(child)
            terms.append(-term if operators[index] == "-" else term)
        return sp.Add(*terms)

    def _product(self, node: Node) -> sp.Basic:
        """A run of factors.

        Multiplication of two vectors is the dot product where the shapes
        leave no other reading, which is why a run that sympy will not
        multiply is folded factor by factor rather than abandoned.
        """
        factors = self._children(node)
        try:
            return sp.Mul(*factors)
        except Exception:
            pass
        result = factors[0]
        for factor in factors[1:]:
            result = self._times(result, factor)
        return result

    def _times(self, left: sp.Basic, right: sp.Basic) -> sp.Basic:
        try:
            return left * right
        except Exception:
            return self._dot(left, right)

    def _binop(self, node: Node) -> sp.Basic:
        left, right = self._children(node)
        match str(node.value):
            case "/":
                try:
                    return left / right
                except Exception:
                    return sp.Mul(left, sp.Pow(right, -1, evaluate=False))
            case "^":
                if _explodes(left, right):
                    return Power(left, right)
                try:
                    return left**right
                except Exception:
                    return sp.Pow(left, right, evaluate=False)
        return self._dot(left, right)

    def _dot(self, left: sp.Basic, right: sp.Basic) -> sp.Basic:
        """`u . v`: the matrix product, of which the dot product is one case.

        Two flat vectors are the case the notation is named after, and their
        product is the number `[2, 3] . [4, 5]` is worth rather than the one by
        one matrix holding it. Everything else conforms or it does not: `n` by
        `m` times `m` by `p` is the matrix product, and shapes that will not
        multiply keep the operator itself, unevaluated.

        A flat vector to the right of a matrix stands for the column it would
        have to be, and what comes back is flat again: the matrix's rows dotted
        with the vector, one number each. A column written as a column keeps
        its shape instead, `[[2], [3]]` being a matrix and not a vector.
        """
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
        operand = self.convert(node.children[0])
        match str(node.value):
            case "-":
                return -operand
            case "+-":
                # `±inf` is the notation for unsigned infinity, and the way
                # back from what the printer writes `zoo` as.
                if operand in (sp.oo, -sp.oo):
                    return sp.zoo
                return PlusMinus(operand)
        return operand

    def _postop(self, node: Node) -> sp.Basic:
        operand = self.convert(node.children[0])
        match str(node.value):
            case "!":
                return sp.factorial(operand)
            case "%":
                return operand / 100
        if isinstance(operand, sp.MatrixBase):
            return operand.T
        return Transposed(operand)

    def _sub(self, node: Node) -> sp.Basic:
        """Element access on a vector, a subscripted variable otherwise.

        An index that is a vector reaches through as many dimensions as it
        holds, `m SUB [2, 3]` being `m SUB 2 SUB 3`.
        """
        base, index = self._children(node)
        if not isinstance(base, sp.MatrixBase):
            return self._subscript(node.children[0], base, index)
        element = _at(base, index)
        if element is None:
            # An index that is no index yet: the access itself survives.
            return self.opaque("ELEMENT", [base, index])
        return element

    def _subscript(self, base_node: Node, base: sp.Basic, index: sp.Basic) -> sp.Basic:
        """`x SUB 1` as one symbol, named the way it is written.

        One symbol rather than a head over two, so that a subscripted variable
        can be differentiated with respect to and solved for like any other.
        The name is the author notation, which is what makes it print and
        reparse as the subscript it is.

        That only works while both parts are names or plain numerals. Anything
        else would have to hide a whole subexpression inside a name, where no
        substitution could reach it, so it becomes a `Subscript` instead.
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
        """A vector of vectors is a matrix; a flat one is a row.

        Ragged or mixed vectors are no matrix at all, and stay inert.
        """
        rows = node.children
        if not rows:
            return sp.Matrix(0, 0, [])
        elements = self._children(node)
        if all(row.kind is Kind.VECTOR for row in rows):
            widths = {len(row.children) for row in rows}
            if len(widths) == 1 and widths != {0} and _all_matrices(elements):
                try:
                    return sp.Matrix([list(row) for row in elements])
                except Exception:
                    return InertVector(*elements)
        if any(row.kind is Kind.VECTOR for row in rows):
            return InertVector(*elements)
        # A matrix holds mathematics; a vector of relations or of booleans is
        # a container, and stays one.
        if not all(isinstance(element, sp.Expr) for element in elements):
            return InertVector(*elements)
        try:
            return sp.Matrix(1, len(elements), elements)
        except Exception:
            return InertVector(*elements)

    def _relation(self, node: Node) -> sp.Basic:
        """Assembled unevaluated: the engine never decides a bare relation.

        A chain nests to the left in the grammar and means the conjunction of
        its links: `1 <= a <= b - 1` is `1 <= a AND a <= b - 1`, which is how
        the shipped library writes `c <= a <= 0`. Reading it as written would
        be comparing the truth value of the first link with a number, which is
        not a statement anybody made.
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
        operands = self._children(node)
        if operands and all(isinstance(operand, sp.Integer) for operand in operands):
            try:
                return sp.Integer(_BITWISE[node.kind](*(int(o) for o in operands)))
            except TypeError:
                pass
        try:
            return _BOOLEAN[node.kind](*(_settled(o) for o in operands))
        except Exception:
            return Logical(sp.Symbol(_LOGICAL_NAMES[node.kind]), *operands)

    # -- definitions, which convert their value and keep their shape --------

    def _assign(self, node: Node) -> sp.Basic:
        lhs = self.convert(node.children[0])
        rest = [self.convert(child) for child in node.children[1:]]
        return Assign(lhs, sp.Symbol(str(node.value)), *rest)

    def _fundef(self, node: Node) -> sp.Basic:
        params = sp.Tuple(*self._children(node.children[0]))
        body = [self.convert(child) for child in node.children[1:]]
        return FunDef(sp.Symbol(str(node.value)), params, *body)

    def _declaration(self, node: Node) -> sp.Basic:
        """A declaration describes itself.

        The symbol it names carries the facts the declaration states, whether
        or not the context has caught up with it, and it stays a symbol even
        where the declaration gives it a value.
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
        """A function call, by the function tables, opaque where it is not.

        A call the tables cannot make sense of - the wrong number of arguments,
        a matrix where a number belongs - falls back to the inert head, which
        is the "return it unchanged rather than guess" rule in miniature.

        A call over an unevaluated command head is such a call, and it is one
        the pipeline will come back to: `RHS(SOLVE(u, x))` has no relation to
        take a side of until the `SOLVE` has become the vector it stands for.
        The head may stand anywhere inside the argument, `RHS(SOLVE(z, y) SUB
        1)` being how the shipped ODE library reads a solution out.
        """
        if any(_holds_command(argument) for argument in args):
            return self.opaque(name, args)
        handler = FUNCTIONS.get(name) or SYMPY_HEADS.get(name)
        if handler is None:
            return self.opaque(name, args)
        try:
            result = handler(self, list(args))
        except Exception:
            result = None
        return self.opaque(name, args) if result is None else result

    def opaque(self, name: str, args: Sequence[sp.Basic]) -> sp.Basic:
        """An inert application: it survives, and it prints back as itself."""
        return sp.Function(name)(*args)

    # -- the angle mode, which is a conversion rule and not a printing one ---

    def angle_in(self, value: sp.Basic) -> sp.Basic:
        """A trig argument, in radians whatever the user measured it in."""
        if self.context.angle is Angle.DEGREE:
            return value * sp.pi / 180
        return value

    def angle_out(self, value: sp.Basic) -> sp.Basic:
        """An inverse trig result, in the unit the user is working in."""
        if self.context.angle is Angle.DEGREE:
            return value * 180 / sp.pi
        return value


def _explodes(base: sp.Basic, exponent: sp.Basic) -> bool:
    """Whether `base^exponent` is a number too big to be worth building.

    Sympy evaluates `Integer**Integer` as the power is constructed, so this is
    the only place the question can be asked: by the time `10^10^10` exists it
    is ten billion digits of it, four gigabytes, and an OOM kill rather than an
    answer - and the kill takes the worksheet with it. Refusing beforehand is
    what Derive does with what it cannot do: the expression comes back as it
    was written, an inert power that prints and saves as `10^10^10`.

    Only exact rationals are judged. A float power costs the working precision
    whatever the exponent, and a symbolic one is not a number to build.
    """
    # By class rather than by assumption: not everything raised to a power is
    # a number with assumptions to ask about, a matrix least of all.
    if not (isinstance(exponent, sp.Integer) and isinstance(base, sp.Rational)):
        return False
    if base.is_zero:
        return False
    bits = max(abs(base.p).bit_length(), abs(base.q).bit_length())
    return bits * abs(int(exponent)) > POWER_BITS


def _nameable(value: sp.Basic) -> bool:
    """Whether a subscript index can be part of a symbol's name.

    A name and a nonnegative numeral can; a negative one cannot, since
    `x SUB -1` is not something the grammar reads back.
    """
    if type(value) is sp.Symbol:
        return True
    return isinstance(value, sp.Integer) and value >= 0


def _all_matrices(elements: list) -> bool:
    return all(isinstance(element, sp.MatrixBase) for element in elements)


def _element(matrix: sp.MatrixBase, index: sp.Basic) -> sp.Basic | None:
    """`v SUB n`, counting from 1. None if that is not an index."""
    if not isinstance(index, sp.Integer) or index < 1:
        return None
    number = int(index)
    if matrix.rows == 1:
        return matrix[0, number - 1] if number <= matrix.cols else None
    return matrix[number - 1, :] if number <= matrix.rows else None


# -- the function table -----------------------------------------------------

Handler = Callable[[_Converter, list], sp.Basic]


def _one(args: list) -> sp.Basic:
    (value,) = args
    return value


def _direct(func: Callable[..., sp.Basic]) -> Handler:
    return lambda conv, args: func(*args)


def _trig(func: Callable[..., sp.Basic]) -> Handler:
    return lambda conv, args: func(conv.angle_in(_one(args)))


def _arctrig(func: Callable[..., sp.Basic]) -> Handler:
    return lambda conv, args: conv.angle_out(func(_one(args)))


def _fold(func: Callable[[sp.Basic, sp.Basic], sp.Basic]) -> Handler:
    def handler(conv: _Converter, args: list) -> sp.Basic:
        if not args:
            raise ValueError("no arguments")
        result = args[0]
        for argument in args[1:]:
            result = func(result, argument)
        return result

    return handler


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

#: Functions that are their sympy counterpart applied to its arguments.
_DIRECT: dict[str, Callable[..., sp.Basic]] = {
    "SQRT": sp.sqrt,
    "EXP": sp.exp,
    "LN": sp.log,
    "ABS": sp.Abs,
    "MAX": sp.Max,
    "MIN": sp.Min,
    "GAMMA": sp.gamma,
    "COMB": sp.binomial,
    "PERM": sp.ff,
    "RE": sp.re,
    "IM": sp.im,
    "CONJ": sp.conjugate,
    "PHASE": sp.arg,
    "ERF": sp.erf,
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
}


def _sign(conv: _Converter, args: list) -> sp.Basic:
    """`SIGN(0)` is `+-1`, overriding sympy's `0`."""
    value = _one(args)
    if value.is_zero:
        return PlusMinus(sp.Integer(1))
    return sp.sign(value)


def _step(conv: _Converter, args: list) -> sp.Basic:
    """`STEP(x)` is `SIGN(x)/2 + 1/2`, so `STEP(0)` is a half."""
    return sp.Heaviside(_one(args), sp.Rational(1, 2))


def _chi(conv: _Converter, args: list) -> sp.Basic:
    """`CHI(a, x, b)`: the indicator of the interval between `a` and `b`.

    One difference of signs, `SIGN(x - a)/2 - SIGN(x - b)/2`, which is the
    equivalent the manual gives for the case where the comparisons cannot be
    made - and, as the original shows, the answer in every other case too.
    Nothing here has to rule on where `x` lies, because the signs do: between
    the two the difference is 1, outside it is 0, and beyond a reversed pair it
    is -1, which is the manual's `CHI(a, x, b) = -CHI(b, x, a)`.

    The endpoints follow from the same formula rather than from a rule of their
    own. `SIGN(0)` is `±1`, so `CHI(0, 0, 1)` is `±1/2 + 1/2`: the indicator is
    undecided exactly on the edge of the interval it indicates.

    `a` defaults to 0 and `b` to 1, so `CHI(x)` indicates the unit interval.
    """
    if len(args) == 1:
        low, value, high = sp.Integer(0), args[0], sp.Integer(1)
    elif len(args) == 2:
        low, value, high = args[0], args[1], sp.Integer(1)
    else:
        low, value, high = args
    return (_sign(conv, [value - low]) - _sign(conv, [value - high])) / 2


def _log(conv: _Converter, args: list) -> sp.Basic:
    """`LOG(z)` is the natural logarithm; `LOG(z, w)` is `LN(z)/LN(w)`."""
    return sp.log(*args)


def _normal(conv: _Converter, args: list) -> sp.Basic:
    """`NORMAL(z, m, s)`: the normal distribution of mean `m` and deviation `s`.

    The statistical function, which is what Derive means by the name - nothing
    to do with normal form, and nothing a pipeline is needed for. The manual
    writes it over the error function as `(ERF(√2·z/(2·s) - √2·m/(2·s)) + 1)/2`,
    and with `m` defaulting to 0 and `s` to 1 that leaves `NORMAL(z)` the
    cumulative distribution `Φ(z)` of the standard Gaussian.
    """
    value, *rest = args
    mean = rest[0] if rest else sp.Integer(0)
    deviation = rest[1] if len(rest) > 1 else sp.Integer(1)
    return (sp.erf((value - mean) / (deviation * sp.sqrt(2))) + 1) / 2


# -- the annuity of section 6.12 ---------------------------------------------
#
# `v*(1 + i)^n + p*(1 + i*t)*((1 + i)^n - 1)/i + f = 0` is the contract the
# financial functions describe: a present value `v` earning a periodic rate `i`
# over `n` periods against a fixed payment `p`, ending at a future value `f`,
# where `t` says where in the period the payment falls. Each function below is
# that equation solved for one of its own.
#
# Every one of them divides by the rate, so every one is undefined where there
# is no rate: `PMT(0, 10, 1000)` is `?`, and the original takes no limit to
# rescue it.
#
# Four of the five are solved for here. The rate is not one of them: no
# rearrangement isolates `i`, and what the original does instead is search for
# it, which is why `RATE` fills its arguments in here and is answered by the
# pipeline, where a solver can be reached.


def annuity(
    rate: sp.Basic,
    periods: sp.Basic,
    payment: sp.Basic,
    present: sp.Basic,
    future: sp.Basic,
    when: sp.Basic,
) -> sp.Basic:
    """Section 6.12's contract, worth zero when its six terms agree."""
    return (
        present * (1 + rate) ** periods
        + payment * (1 + rate * when) * ((1 + rate) ** periods - 1) / rate
        + future
    )


def _financial(args: list, count: int) -> list:
    """The arguments of a financial function, with the ones it lets go filled in.

    The two values a contract can do without default to zero, and so does the
    time of payment, which is the end of the period. Fewer arguments than say
    what the contract is, or more than the function takes, and there is nothing
    to compute: the call stays inert, as any other unreadable one does.
    """
    if not 2 <= len(args) <= count:
        raise TypeError("not a financial contract")
    return args + [sp.Integer(0)] * (count - len(args))


def _present_value(conv: _Converter, args: list) -> sp.Basic:
    """`PVAL(i, n, p, f, t)`: what the contract is worth now."""
    rate, periods, payment, future, when = _financial(args, 5)
    due = payment * (rate * when + 1)
    return ((due - future * rate) / (1 + rate) ** periods - due) / rate


def _future_value(conv: _Converter, args: list) -> sp.Basic:
    """`FVAL(i, n, p, v, t)`: what the contract will be worth."""
    rate, periods, payment, present, when = _financial(args, 5)
    due = payment * (rate * when + 1)
    grown = (1 + rate) ** periods * (rate * (payment * when + present) + payment)
    return (due - grown) / rate


def _payment(conv: _Converter, args: list) -> sp.Basic:
    """`PMT(i, n, v, f, t)`: what is paid each period."""
    rate, periods, present, future, when = _financial(args, 5)
    growth = (1 + rate) ** periods
    return rate * (present * growth + future) / ((1 - growth) * (rate * when + 1))


def _periods(conv: _Converter, args: list) -> sp.Basic:
    """`NPER(i, p, v, f, t)`: how many periods it runs for."""
    rate, payment, present, future, when = _financial(args, 5)
    due = payment * (rate * when + 1)
    return sp.log(
        (due - future * rate) / (rate * (payment * when + present) + payment)
    ) / sp.log(1 + rate)


#: The generator `RANDOM` draws from. Its own rather than the module's, so
#: that seeding it says something about this program and not about whatever
#: else in the process wanted a random number.
_GENERATOR = random.Random()

#: When the program started, which is what `RANDOM(0)` reports the distance
#: from, and the outcomes a number in [0, 1) is drawn from. The original draws
#: a rational rather than a decimal - `RANDOM(1)` answers 242469/1540430 - and
#: a power of two divided out is the same kind of number, reduced the same way.
_STARTED = time.monotonic()
_OUTCOMES = 2**32


def _random(conv: _Converter, args: list) -> sp.Basic:
    """`RANDOM(n)`: a pseudo-random number, or the seed one is drawn from next.

    Section 6.13 gives four cases and lets the size of a whole `n` choose
    between them: above 1 an integer in [0, n), at 1 a number in [0, 1), below
    0 a seed of `-n` which it answers with, and at 0 a seed taken from the
    clock, answered with the reading it was taken from. `n` is a whole number
    of outcomes or it is nothing: `RANDOM(6.5)` chooses no case and stays as it
    stands.

    One draw per call written, which is where this parts company with the
    original: a generated vector substitutes into a body already converted, so
    `VECTOR(RANDOM(6), n, 1, 100)` is one throw shown a hundred times rather
    than the manual's hundred throws. Drawing later instead would cost the
    other half of it, two dice in one sum being one inert call to sympy however
    many times it is written.
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


def _rate(conv: _Converter, args: list) -> sp.Basic:
    """`RATE(n, p, v, f, t)`: the rate the contract implies, and where to look.

    The original shows the search rather than performing it silently: this
    simplifies to `RATE(n, p, v, f, t, 0, 1)`, the last two arguments being the
    bounds it looks between, and a rate that is negative or above 100% has to
    be given bounds of its own as the sixth and seventh. Filling those in is
    all that happens here, and doing it again to the filled-in form changes
    nothing.
    """
    if len(args) == 6 or len(args) > 7:
        raise TypeError("not a financial contract")
    bounds = args[5:7] or [sp.Integer(0), sp.Integer(1)]
    return conv.opaque("RATE", _financial(args[:5], 5) + bounds)


def _atan(conv: _Converter, args: list) -> sp.Basic:
    if len(args) == 2:
        return conv.angle_out(sp.atan2(args[0], args[1]))
    return conv.angle_out(sp.atan(_one(args)))


def _floor(conv: _Converter, args: list) -> sp.Basic:
    """`FLOOR(m, n)` is the floor of `m/n`, and `n` defaults to 1."""
    numerator, denominator = _with_unit_default(args)
    return sp.floor(numerator / denominator)


def _mod(conv: _Converter, args: list) -> sp.Basic:
    """`MOD(m, n)` is `m - n FLOOR(m, n)`; `MOD(m, 0)` is `m`."""
    value, modulus = _with_unit_default(args)
    if modulus.is_zero:
        return value
    return value - modulus * sp.floor(value / modulus)


def _mods(conv: _Converter, args: list) -> sp.Basic:
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
    """A statistic of a sample, in each of the three forms it can be written.

    The sample is the arguments themselves, or the elements of one vector, or -
    where the one argument is a matrix - the elements of each of its rows, in
    which case the answers make a vector.

    Which of the three a single argument is depends on what it turned out to be,
    and one that is still a call nobody could make is none of them yet: the
    utility files write `RMS(VECTOR(...))`, where nothing generates until the
    definition is applied, and reading that as a sample of one would answer
    `ABS` of a vector that has not been built. A sample of one is what anything
    else is, the formulas being written over `n` arguments for any `n`.
    """

    def handler(conv: _Converter, args: list) -> sp.Basic:
        if len(args) == 1:
            (value,) = args
            if isinstance(value, AppliedUndef):
                raise TypeError("no sample yet")
            if isinstance(value, sp.MatrixBase):
                if value.rows > 1:
                    rows = [list(value[row, :]) for row in range(value.rows)]
                    return _vector_of([over(row) for row in rows])
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
    """The arithmetic mean: the sum over how many were summed."""
    return sp.Add(*values) / len(values)


def _root_mean_square(values: list) -> sp.Basic:
    return sp.sqrt(sp.Add(*(value**2 for value in values)) / len(values))


def _variance(values: list) -> sp.Basic:
    """The sample variance, which the manual is explicit is the unbiased one.

    Divided by `n - 1` and not by `n`: the deviations are taken from the
    sample's own average rather than from the distribution's, which makes them
    smaller than the true ones by just the amount that one degree of freedom
    accounts for. A sample of one is the case that leaves, and Derive answers
    it with the zero its deviations sum to rather than with the `0/0` the
    formula reads as.

    Each deviation carries a factor of `1/n` from the average inside it, so the
    sum arrives as `n` separate squares over `n^2` and has to be put back over
    one denominator to be read: `VAR(x, y)` is `(x - y)^2/2`, which is Derive's
    answer and not the three terms multiplying it out gives.
    """
    if len(values) == 1:
        return sp.Integer(0)
    mean = _average(values)
    deviations = sp.Add(*((value - mean) ** 2 for value in values))
    return sp.factor(deviations / (len(values) - 1))


def _standard_deviation(values: list) -> sp.Basic:
    """The square root of the sample variance, on the same convention."""
    return sp.sqrt(_variance(values))


def _fit(conv: _Converter, args: list) -> sp.Basic:
    """`FIT(v, A)`: the least squares fit of a parameterized expression to data.

    The label vector `v` names the data variables and then the expression
    written over them, and the data matrix `A` carries one row per point: a
    value for each data variable, then the value the expression is meant to
    take there. What is left of the expression's variables once the data ones
    are taken out are the parametric ones, and those are what the fit solves
    for.

    The expression has to be linear in the parametric variables, but not in the
    data ones - which is what makes `q*ATAN(t) + r*SIN(t)` a fit anyone can
    ask for. Substituting a row's data values leaves an expression in the
    parameters alone, and its derivative with respect to each parameter is that
    parameter's coefficient there; a coefficient that still mentions a
    parameter says the dependence was never linear, and the call comes back the
    way it was written. What the parameters do not account for moves to the
    other side, so the row reads as one equation of the linear system whose
    least squares solution is the fit. As many rows as parameters solves it
    exactly; more rows minimize the squared discrepancies.

    Everything degenerate raises: no parametric variables, rows of the wrong
    width, fewer rows than the fit has freedom, or a design matrix of deficient
    rank. None of those name a fit, and an inert call says so better than an
    answer that was picked out of a family would.
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
    data = _matrix(data)
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
    solution = sp.Matrix(design).solve_least_squares(sp.Matrix(targets))
    return expression.subs(dict(zip(parameters, solution)))


def _is_number(conv: _Converter, args: list) -> sp.Basic:
    """`NUMBER(u)`: whether `u` simplified to a number.

    A predicate, and it answers one of the two truth-values however the question
    comes out, because a test that stayed inert would leave the `IF` or `SELECT`
    it was written for undecided. That is what the utility files use it for:
    `NUMBER(d)` is false while `d` is still the parameter it was written as,
    which is how a definition tells an argument that was supplied from one that
    was left out.

    A number is an integer or a rational, which is what Derive's exact
    arithmetic is over, together with the approximation of one that Approximate
    mode works in. A radical, `#i` and `inf` are not numbers by that reading.
    """
    return sp.true if isinstance(_one(args), (sp.Rational, sp.Float)) else sp.false


def _is_prime(conv: _Converter, args: list) -> sp.Basic:
    """`PRIME(m)`: whether `m` is a prime number.

    A predicate like `NUMBER`, and answering one of the two truth-values is the
    whole of what it is for: `SELECT(PRIME(k), k, 1, 100)` has nothing to select
    on from anything else. What is no integer is no prime, but neither is it
    false - `PRIME(k)` with `k` still a variable is a question nobody has asked
    yet - so that call comes back as it was written.

    The manual's second argument is how many rounds of a probabilistic test to
    run before believing the answer. The test here is not probabilistic, so the
    count is accepted and makes no difference.
    """
    value, *rounds = args
    if not isinstance(value, sp.Integer):
        raise TypeError("not an integer")
    return sp.true if sp.isprime(int(value)) else sp.false


def _next_prime(conv: _Converter, args: list) -> sp.Basic:
    """`NEXT_PRIME(m)`: the first prime above `m`, which need be no integer.

    Strictly above, so `NEXT_PRIME(7)` is 11. A repeat count is accepted and
    ignored, as it is for `PRIME`.
    """
    value, *rounds = args
    if not isinstance(value, sp.Rational):
        raise TypeError("not a number")
    return sp.Integer(sp.nextprime(int(sp.floor(value))))


def _numerator(conv: _Converter, args: list) -> sp.Basic:
    return sp.fraction(sp.together(_one(args)))[0]


def _denominator(conv: _Converter, args: list) -> sp.Basic:
    return sp.fraction(sp.together(_one(args)))[1]


def _terms(conv: _Converter, args: list) -> sp.Basic:
    """`TERMS(u)`: what `u` is written as the sum of, as a vector.

    Syntactic terms, so nothing is multiplied out first: `TERMS(x*(a + b)^2 + c)`
    has two of them, and the manual tells the caller to compose with `EXPAND`
    where that is not what was wanted. An expression that is no sum is one term.

    An order there had to be, and it is the one the terms were already in:
    descending total degree, which is what the printer writes a sum by.

    A vector distributes, each element contributing its own row - so `TERMS` of
    a vector of sums is the matrix of their terms wherever those come out the
    same length.
    """
    value = _one(args)
    if isinstance(value, sp.MatrixBase):
        return _vector_of([_terms(conv, [element]) for element in _elements_of(value)])
    if not isinstance(value, sp.Expr):
        raise TypeError("no terms")
    return _vector_of(list(value.as_ordered_terms(order="grlex")))


def _factors(conv: _Converter, args: list) -> sp.Basic:
    """`FACTORS(u)`: what `u` is written as the product of, as a vector.

    `TERMS`'s counterpart, and syntactic in the same way: nothing is factored
    first, so `FACTORS(x^2 - 1)` is one factor and the manual tells the caller
    to compose with `FACTOR` where that is not what was wanted. A number is one
    factor too - `FACTORS(12)` is `[12]`, not the prime factorisation, which is
    what `FACTOR` is for.

    The order is Derive's, which is by the factors' bases and not their
    exponents: most main first, `main_order`'s reading of most main, so
    `FACTORS(2*x*y)` is `[x, y, 2]` with the number - main in nothing - last.
    Among factors over the same main variable the compound comes before the
    simple, `[(x + 1)^2, x]` and `[SIN(x), x]`, which is descending in sympy's
    own ordering of them.

    A vector distributes, each element contributing its own row, exactly as
    `TERMS` does.
    """
    value = _one(args)
    if isinstance(value, sp.MatrixBase):
        return _vector_of([_factors(conv, [element]) for element in _elements_of(value)])
    if not isinstance(value, sp.Expr):
        raise TypeError("no factors")
    factors = value.as_ordered_factors() if value.is_Mul else [value]
    return _vector_of(_by_mainness(factors))


def _by_mainness(pieces: list) -> list:
    """`pieces` from most main to least, which is Derive's ordering of them.

    Most main is `main_order`'s reading of the variable a piece is mainly
    about, and the compound comes before the simple where two are about the
    same variable: `SIN(x)` ahead of `x`, `(x + 1)^2` ahead of `x`. That second
    part is descending in sympy's own ordering, which is the one that puts the
    plain symbol first. What mentions no variable at all is main in nothing and
    goes last, which is where a product's numeric coefficient belongs.
    """
    order = main_order(name for piece in pieces for name in _named_in(piece))

    def rank(numbered: tuple[int, sp.Basic]) -> tuple[int, int]:
        position, piece = numbered
        names = _named_in(piece)
        return min((order.index(name) for name in names), default=len(order)), -position

    return [piece for _, piece in sorted(enumerate(pieces), key=rank)]


def _named_in(value: sp.Basic) -> set[str]:
    """The variables `value` mentions. A string literal is data, not one."""
    return {symbol.name for symbol in value.free_symbols if type(symbol) is sp.Symbol}


def _side(index: int) -> Handler:
    """`LHS(u)` and `RHS(u)`: one side of a relation.

    A vector distributes, which is the manual's own reason for the pair:
    `RHS(SOLVE(x^2 - 5*x + 6 = 0, x))` is the vector of roots. What is no
    relation has no sides to take and answers with itself, both ways - Derive
    gives `LHS(2*x + 3)` back as `2*x + 3`.
    """

    def handler(conv: _Converter, args: list) -> sp.Basic:
        value = _one(args)
        if isinstance(value, sp.MatrixBase):
            return _vector_of([handler(conv, [e]) for e in _elements_of(value)])
        if isinstance(value, InertVector):
            # Which is what a vector of relations is: sympy holds no matrix of
            # them, and a vector of relations is exactly what has sides worth
            # taking.
            return _vector_of([handler(conv, [e]) for e in value.args])
        if value.is_Relational:
            return value.args[index]
        return value

    return handler


def _quotient(conv: _Converter, args: list) -> sp.Basic:
    """`QUOTIENT(u, v)`: how many times `v` goes into `u`."""
    return _divided(args)[0]


def _remainder(conv: _Converter, args: list) -> sp.Basic:
    """`REMAINDER(u, v)`: what is left of `u` when `v` has gone into it."""
    return _divided(args)[1]


def _divided(args: list) -> tuple[sp.Basic, sp.Basic]:
    """Polynomial division, in the main variable and over everything else.

    The division is with respect to one variable, and the other variables ride
    along in the coefficients as a field would: `QUOTIENT(x*y + 1, y)` is
    `x + 1/y` with nothing left over, because as polynomials in `x` the divisor
    is a constant and a constant divides exactly.

    Two numbers have no variable to divide in, and are a field on their own:
    the quotient is the fraction and the remainder is nothing, so
    `QUOTIENT(7, 2)` is `7/2` rather than 3. That is Derive's answer, and the
    manual's "`u` and `v` should be rational numbers or polynomials" is the
    same statement - the rationals are where its division is exact.

    What the main one is, sympy's own reading of what the two are polynomials
    in, ordered by `_by_mainness`: usually a variable, and a kernel where the
    expression is a polynomial in one. `QUOTIENT(SIN(x), SIN(x)^2)` is 0 with
    `SIN(x)` left over, which only makes sense in `SIN(x)`; `QUOTIENT(SIN(x),
    x)` is `SIN(x)/x` in the same reading, the divisor being a coefficient
    there. Both are Derive's answers.
    """
    numerator, denominator = args
    try:
        generators = sp.parallel_poly_from_expr((numerator, denominator))[1].gens
    except sp.PolificationFailed:
        # Two numbers, which are a field on their own.
        return numerator / denominator, sp.Integer(0)
    return sp.div(numerator, denominator, _by_mainness(list(generators))[0])


def _polynomial_gcd(conv: _Converter, args: list) -> sp.Basic:
    """`POLY_GCD(u, v)`: the greatest common divisor of two polynomials.

    `GCD` over a wider domain rather than a different function, and numbers are
    the polynomials of no variables: `POLY_GCD(12, 18)` is 6.
    """
    left, right = args
    return sp.gcd(left, right)


def _variables(conv: _Converter, args: list) -> sp.Basic:
    """`VARIABLES(u)`: the free variables of `u`, from most main to least.

    Most main first is `ordering`'s order and Derive's own: the order list `x`,
    `y`, `z` ahead of everything else, and everything else alphabetically. An
    order there had to be - sympy holds the free variables as a set, and an
    answer the worksheet stores as text has to print the same every time.

    The symbols found are the ones returned, rather than symbols built again
    from their names, so that each carries the domain it was declared with.

    A string literal is a `Symbol` to sympy and data to Derive, and is no more a
    free variable of the expression holding it than a numeral is.
    """
    found = _one(args).free_symbols
    by_name = {symbol.name: symbol for symbol in found if type(symbol) is sp.Symbol}
    return _vector_of([by_name[name] for name in main_order(by_name)])


def _dif(conv: _Converter, args: list) -> sp.Basic:
    """`DIF(u, x)` and `DIF(u, x, n)`, unevaluated. `.doit()` is a decision."""
    if len(args) == 2:
        return sp.Derivative(args[0], args[1], evaluate=False)
    expression, variable, order = args
    return sp.Derivative(expression, (variable, order), evaluate=False)


def _integral(conv: _Converter, args: list) -> sp.Basic:
    """`INT(u, x)` and `INT(u, x, a, b)`, both unevaluated."""
    if len(args) == 2:
        return sp.Integral(args[0], args[1])
    expression, variable, low, high = args
    return sp.Integral(expression, (variable, low, high))


def _summation(conv: _Converter, args: list) -> sp.Basic:
    """`SUM(u, k, a, b)`, and `SUM(v)` over the elements of a vector.

    A third argument that is a vector is the values `k` takes, rather than the
    ends of a range it runs through.

    An indefinite `SUM(u, k)` is the antidifference of `u`, which is a
    computation and so waits for the pipeline the way the calculus heads do.
    """
    if len(args) == 1 and isinstance(args[0], sp.MatrixBase):
        return sp.Add(*args[0])
    if len(args) == 2:
        return Antidifference(*args)
    if len(args) == 3 and isinstance(args[2], sp.MatrixBase):
        return sp.Add(*_over_values(conv, args))
    expression, index, low, high = args
    return sp.Sum(expression, (index, low, high))


def _product(conv: _Converter, args: list) -> sp.Basic:
    """`PRODUCT(u, k, a, b)`, and `PRODUCT(v)` over a vector's elements.

    A third argument that is a vector names the values, as it does for `SUM`,
    and an indefinite `PRODUCT(u, k)` is the antiquotient of `u`.
    """
    if len(args) == 1 and isinstance(args[0], sp.MatrixBase):
        return sp.Mul(*args[0])
    if len(args) == 2:
        return Antiquotient(*args)
    if len(args) == 3 and isinstance(args[2], sp.MatrixBase):
        return sp.Mul(*_over_values(conv, args))
    expression, index, low, high = args
    return sp.Product(expression, (index, low, high))


def _over_values(conv: _Converter, args: list) -> list[sp.Basic]:
    """The body of a sum or a product at each value its index is given."""
    expression, index, values = args
    if type(index) is not sp.Symbol:
        raise TypeError("not a variable")
    return [
        _retried(conv, expression.subs(index, value))
        for value in _elements_of(values)
    ]


def _taylor(conv: _Converter, args: list) -> sp.Basic:
    """`TAYLOR(u, x, a, n)`, unevaluated. Computing it is a pipeline's call."""
    return Taylor(*args)


def _approximation(conv: _Converter, args: list) -> sp.Basic:
    """`APPROX(u)` at the session's precision, `APPROX(u, n)` at `n` digits.

    Unevaluated, for the reason `Approx` gives: the rounding waits for the
    pipeline. The digit count is filled in here because this is the last place
    that knows what the caller was working at.

    The number that comes back carries the digits it was asked for, but the
    printer writes every float at the session's precision, so a count above
    that one is computed and not shown.
    """
    value, *rest = args
    digits = _one(rest) if rest else sp.Integer(conv.context.precision_digits)
    if not (isinstance(digits, sp.Integer) and digits > 0):
        raise ValueError("not a precision")
    return Approx(value, digits)


def _conditional(conv: _Converter, args: list) -> sp.Basic:
    """`IF(c, u)` and `IF(c, u, v)` as the case split sympy writes them as.

    Which branch a case split is worth is the pipeline's business, and it is
    the same question for a `Piecewise` that came back from an integral. What
    has no `Piecewise` is Derive's fourth argument, the value where the test
    cannot be decided at all; that form stays an inert head, and the pipeline
    resolves the two alike.
    """
    if len(args) == 2:
        test, then = args
        return sp.Piecewise((then, _test(test)))
    test, then, otherwise = args
    return sp.Piecewise((then, _test(test)), (otherwise, sp.true))


def _test(test: sp.Basic) -> sp.Basic:
    """A condition as sympy reads one: a relation evaluated, not held.

    Everywhere else a relation is assembled undecided, because whether one
    holds is no question of Simplify's. The test of a conditional is the one
    place where it is, and `Piecewise` is entitled to answer it - an
    unevaluated relation is also the one form of a condition it mishandles.
    """
    if test.is_Relational:
        return test.func(test.lhs, test.rhs)
    return test


def _truth_table(conv: _Converter, args: list) -> sp.Basic:
    """`TRUTH_TABLE(p, q, ..., b1, b2, ...)`: the table, as a matrix.

    The leading arguments that are bare variables are the truth variables and
    everything after them is an expression to evaluate; a call that is nothing
    but variables is the table of the assignments themselves, which is what
    `TRUTH_TABLE(p, q)` answers with.

    The first row names the columns exactly as they were written, and the rows
    under it run through every assignment with the last variable changing
    fastest and `true` before `false` - the order the manual prints its table
    in, which is the binary numbers counted down from all-ones.

    Every expression has to come out true or false at every assignment. One
    that does not is no Boolean expression, and the call comes back as written
    rather than a table with a hole in it.
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
    return _vector_of(rows)


#: The sympy head each logical operator lives under, and the word it is
#: written with. `Logical` is the inert one, so a heading held under it is a
#: heading Simplify walks past.
_HEADINGS = {sp.And: "AND", sp.Or: "OR", sp.Xor: "XOR", sp.Implies: "IMP", sp.Not: "NOT"}


def _as_written(expression: sp.Basic) -> sp.Basic:
    """A column heading: the expression as a name for itself rather than a claim.

    A heading is what the column is about, and `p XOR q` heads the column of
    what `p XOR q` comes to at each assignment. Left live, Simplify would
    answer it - and a column headed `NOT p AND q OR p AND NOT q` names its
    rows no better for being right. The inert head keeps the operator standing
    where it was written; the rows below it are where the operator is applied.
    """
    word = _HEADINGS.get(type(expression))
    if word is None:
        return expression
    operands = [_as_written(operand) for operand in expression.args]
    return Logical(sp.Symbol(word), *operands)


def _decided(expression: sp.Basic, assignment: dict) -> sp.Basic:
    """One cell: an expression at one assignment of its variables."""
    value = _test(expression.subs(assignment))
    if value not in (sp.true, sp.false):
        raise TypeError("not a boolean")
    return value


def _substitution(conv: _Converter, args: list) -> sp.Basic:
    """`SUBS(u, [x, ...], [a, ...])`, the way the printer writes sympy's `Subs`.

    The way back matters more here than legibility. A `Subs` holds a derivative
    that is only meaningful unevaluated - `d/dv f(v)` at `v = y` is not the
    derivative of a constant - and reading one back as an inert head would put
    that derivative where the pipeline evaluates it, to zero.
    """
    expression, variables, points = args
    return sp.Subs(expression, tuple(_matrix(variables)), tuple(_matrix(points)))


def _hyper(conv: _Converter, args: list) -> sp.Basic:
    """`HYPER([a, ...], [b, ...], z)`, the way the printer writes `hyper`.

    Sympy's head carries tuples, which the notation has no spelling for, so it
    is written over vectors - and this is the way back, without which a result
    holding one would not read back as the expression it was printed from.
    """
    top, bottom, argument = args
    return sp.hyper(tuple(_matrix(top)), tuple(_matrix(bottom)), argument)


def _limit(conv: _Converter, args: list) -> sp.Basic:
    """`LIM(u, x, a)` is two-sided; a fourth argument picks a side.

    A zero picks neither, and is the two-sided limit again: that is what
    `Calculus Limit` writes for the direction it calls Both, and it writes the
    argument whichever direction was chosen, so the four-argument form is the
    only one that command ever builds.

    A vector of variables against a vector of points is the limits taken one
    after another, in the order written, which is the form the manual offers
    for substituting where substitution alone would divide by zero. Iterated
    and not multivariate: `LIM(u, [x, y], [a, b])` need not agree with
    `LIM(u, [y, x], [b, a])`, and the manual says so.
    """
    if isinstance(args[1], sp.MatrixBase):
        return _iterated_limit(args)
    if len(args) == 3:
        return sp.Limit(*args, dir="+-")
    expression, variable, point, side = args
    if not side:
        return sp.Limit(expression, variable, point, dir="+-")
    return sp.Limit(expression, variable, point, dir="+" if side > 0 else "-")


def _iterated_limit(args: list) -> sp.Basic:
    """`LIM(u, [x, y], [a, b])`: the limit in `x`, and then in `y`.

    The first variable is approached innermost, so that the outer limit is
    taken of what the inner one came to.
    """
    expression, variables, points = args
    names = _elements_of(variables)
    values = _elements_of(points)
    if len(names) != len(values):
        raise ValueError("not a point for every variable")
    for name, value in zip(names, values, strict=True):
        expression = sp.Limit(expression, name, value, dir="+-")
    return expression


def _determinant(conv: _Converter, args: list) -> sp.Basic:
    return _matrix(_one(args)).det()


def _trace(conv: _Converter, args: list) -> sp.Basic:
    return _matrix(_one(args)).trace()


def _dimension(conv: _Converter, args: list) -> sp.Basic:
    """How many elements a vector has, or how many rows a matrix has.

    A vector of relations is a vector too, and counting one is what the shipped
    libraries do to a `SOLVE`: how many solutions there are is the question
    they branch on, and no solutions is an empty vector rather than an error.
    """
    value = _one(args)
    if isinstance(value, InertVector):
        return sp.Integer(len(value.args))
    matrix = _matrix(value)
    return sp.Integer(matrix.cols if matrix.rows == 1 else matrix.rows)


def _element_of(conv: _Converter, args: list) -> sp.Basic:
    """`ELEMENT(v, i)` and `ELEMENT(m, i, j)`, counting from 1.

    A vector of relations is a vector too, and taking one out of it is how the
    shipped ODE library reads a solution: `RHS((SOLVE(z, y)) SUB 1)`.

    An index that is itself a vector is the indices in turn, so that
    `m SUB [2, 3]` reaches what `m SUB 2 SUB 3` reaches.
    """
    if len(args) == 2:
        element = _at(args[0], args[1])
        if element is None:
            raise ValueError("not an index")
        return element
    matrix, row, column = args
    return _matrix(matrix)[int(row) - 1, int(column) - 1]


def _at(value: sp.Basic, index: sp.Basic) -> sp.Basic | None:
    """One element of a vector however the vector is held."""
    if isinstance(index, sp.MatrixBase):
        for step in _elements_of(index):
            value = _at(value, step)
            if value is None:
                return None
        return value
    if not isinstance(value, InertVector):
        return _element(_matrix(value), index)
    if not isinstance(index, sp.Integer):
        return None
    number = int(index)
    return value.args[number - 1] if 1 <= number <= len(value.args) else None


def _delete_element(conv: _Converter, args: list) -> sp.Basic:
    """`DELETE_ELEMENT(v, n)`: `v` with its `n`th element gone, counting from 1.

    A matrix's elements are its rows, as they are everywhere else, so what goes
    is the `n`th row. That is what VECTOR.MTH's `MINOR` is built on: delete a
    row, transpose, delete what was a column, transpose back.

    `n` defaults to 1, as it does for the other two element functions.
    """
    vector, *rest = args
    elements = _elements_of(_matrix(vector))
    place = _place(_one(rest) if rest else sp.Integer(1), len(elements))
    return _vector_of(elements[:place] + elements[place + 1 :])


def _replace_element(conv: _Converter, args: list) -> sp.Basic:
    """`REPLACE_ELEMENT(u, v, n)`: `v` with `u` where its `n`th element was.

    The new value comes first and the vector second, which is the manual's order
    and the one `INSERT_ELEMENT` shares. `n` defaults to 1.
    """
    value, vector, *rest = args
    elements = _elements_of(_matrix(vector))
    place = _place(_one(rest) if rest else sp.Integer(1), len(elements))
    return _vector_of(elements[:place] + [value] + elements[place + 1 :])


def _insert_element(conv: _Converter, args: list) -> sp.Basic:
    """`INSERT_ELEMENT(u, v, n)`: `v` with `u` written in before its `n`th.

    The new value first and the vector second, the order `REPLACE_ELEMENT`
    has, and `n` defaults to 1. One past the end is an index here where it is
    none anywhere else, because that is how an element is added to the end:
    `INSERT_ELEMENT(d, [a, b, c], 4)` is `[a, b, c, d]`.
    """
    value, vector, *rest = args
    elements = _elements_of(_matrix(vector))
    place = _place(_one(rest) if rest else sp.Integer(1), len(elements) + 1)
    return _vector_of(elements[:place] + [value] + elements[place:])


def _reversed_vector(conv: _Converter, args: list) -> sp.Basic:
    """`REVERSE_VECTOR(v)`: `v` with its elements back to front.

    A matrix's elements are its rows, so a matrix comes back with its rows in
    the opposite order and each row as it was.
    """
    return _vector_of(_elements_of(_matrix(_one(args)))[::-1])


def _appended(conv: _Converter, args: list) -> sp.Basic:
    """`APPEND(v, w, ...)`: the elements of each, run together into one vector.

    A matrix's elements are its rows, so appending matrices stacks them - which
    is what the manual's `APPEND_COLUMNS` exercise turns on: transpose the two,
    append, transpose back. A single matrix is the exception the manual names
    separately, and it flattens: `APPEND([[a, b], [c, d]])` is `[a, b, c, d]`,
    not the matrix it started as.
    """
    if len(args) == 1 and isinstance(args[0], sp.MatrixBase):
        return _vector_of(list(args[0]))
    elements: list[sp.Basic] = []
    for vector in args:
        elements.extend(_elements_of(_matrix(vector)))
    return _vector_of(elements)


def _place(index: sp.Basic, count: int) -> int:
    """Which of `count` elements an index picks out, counting from 1."""
    if not isinstance(index, sp.Integer) or not 1 <= index <= count:
        raise ValueError("not an index")
    return int(index) - 1


def _generated_vector(conv: _Converter, args: list) -> sp.Basic:
    """`VECTOR(u, k, ...)`: `u` at each value `k` takes, as a vector.

    Four ways of saying which values those are - a count, a range, a range with
    a step, and a vector of the values themselves - and a nested call makes a
    matrix, a vector of vectors being all a matrix is.

    Bounds that are not numbers describe no sequence, and the call stays inert
    until they are: that is what makes `VECTOR([v SUB i], i, DIMENSION(v))`
    keep its shape in a definition and generate when the definition is applied.
    """
    body, index, *rest = args
    if type(index) is not sp.Symbol:
        raise TypeError("not a variable")
    generated = [body.subs(index, value) for value in _steps(rest)]
    return _vector_of([_retried(conv, element) for element in generated])


def _steps(rest: list) -> list[sp.Basic]:
    """The values a generated vector's variable takes, in order.

    `(n - m)/s + 1` of them, rounded down, which is one element for a range that
    the step cannot cross and none at all for a range going the wrong way.
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


def _selected(conv: _Converter, args: list) -> sp.Basic:
    """`SELECT(u, k, ...)`: the values of `k` for which `u(k)` holds.

    Which values those are is said in the same four ways `VECTOR` says it, and
    `_steps` reads them the same. What comes back is the values themselves and
    not anything computed from them, which is the whole difference between the
    two: `SELECT(PRIME(k), k, 1, 100)` is the primes under a hundred.

    Every test has to come out true or false. One that stays a relation nobody
    can decide would silently drop the element it was asked about, so the call
    comes back as it was written instead - which is also what keeps a `SELECT`
    inside a definition intact until the definition is applied.
    """
    body, index, *rest = args
    if type(index) is not sp.Symbol:
        raise TypeError("not a variable")
    chosen = []
    for value in _steps(rest):
        held = _test(_retried(conv, body.subs(index, value)))
        if held is sp.true:
            chosen.append(value)
        elif held is not sp.false:
            raise ValueError("undecided")
    return _vector_of(chosen)


def _iterates(conv: _Converter, args: list) -> sp.Basic:
    """`ITERATES(u, x, x0)` and `ITERATES(u, x, x0, n)`: the sequence, as a vector.

    `x0`, `u(x0)`, `u(u(x0))` and so on: `n` updates make `n + 1` elements, and
    a count left out means "until a value comes round again".
    """
    return _vector_of(_sequence(conv, args))


def _iterate(conv: _Converter, args: list) -> sp.Basic:
    """`ITERATE(...)`: the same sequence's last element, and nothing else.

    Counted, that is the `n`th update and there is no more to ask. Uncounted,
    the sequence ended by repeating something, and only a value that repeated
    *itself* is a value the iteration arrived at: a longer cycle converges to
    nothing, and the manual's answer for that is `?`.
    """
    sequence = _sequence(conv, args)
    if len(args) > 3:
        return sequence[-1]
    return sequence[-1] if sequence[-1] == sequence[-2] else sp.nan


def _sequence(conv: _Converter, args: list) -> list[sp.Basic]:
    """The iterates, however many were asked for.

    A negative count iterates the inverse of `u` instead, `|n|` times, which is
    what `MISC.MTH` defines `INVERSE(u, x) := ITERATE(u, x, x, -1)` on.
    """
    body, variable, start, *rest = args
    names, values = _iterated_over(variable, start)
    if not rest:
        return _until_repeated(conv, body, names, values)
    count = _one(rest)
    if not isinstance(count, sp.Integer):
        raise TypeError("not a count")
    if count < 0:
        body, count = _inverted(body, names), -count
    iterates = [_state(names, values)]
    for _ in range(int(count)):
        values = _updated(conv, body, names, values)
        iterates.append(_state(names, values))
    return iterates


def _until_repeated(
    conv: _Converter, body: sp.Basic, names: list, values: list
) -> list[sp.Basic]:
    """The sequence run out to where it comes round.

    The repeated value is the last element rather than being dropped, so
    `ITERATES(1/x, x, 2)` is `[2, 1/2, 2]` - which is what tells `ITERATE`
    whether the cycle it found has length one.

    What Derive does when nothing ever repeats is iterate until memory is gone.
    That is no answer an engine can give, so an iteration that has not come
    round within the bounds below comes back the call it was written as.
    """
    iterates = [_state(names, values)]
    for _ in range(_ITERATIONS):
        values = _updated(conv, body, names, values)
        iterates.append(_state(names, values))
        if iterates[-1] in iterates[:-1]:
            return iterates
        if _outsized(iterates[-1]):
            break
    raise ValueError("comes round to nothing")


#: How far an uncounted iteration is run before it is given up on, and how big
#: an iterate may get on the way. Both bounds are needed: a sequence that does
#: not come round usually runs away instead, and it runs away faster than any
#: count can catch - repeated squaring passes thirty thousand digits in
#: seventeen steps. Neither bound is near anything a converging iteration
#: reaches.
_ITERATIONS = 100
_ITERATE_BITS = 100_000
_ITERATE_OPERATIONS = 1000


def _outsized(value: sp.Basic) -> bool:
    """Whether an iterate has grown past what carrying it any further is worth."""
    if sp.count_ops(value) > _ITERATE_OPERATIONS:
        return True
    return any(
        int(number).bit_length() > _ITERATE_BITS for number in value.atoms(sp.Integer)
    )


def _iterated_over(variable: sp.Basic, start: sp.Basic) -> tuple[list, list]:
    """The variables an iteration updates, and the values they start at.

    One variable and one value, or - the form the manual writes Fibonacci in -
    a vector of variables and a vector of their values, so that an iteration
    remembering more than one previous iterate needs no subscripts:
    `ITERATE([k, j + k], [j, k], [0, 1], n)`.
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
    """One iterate: the value the variable took, or the vector they all took."""
    return values[0] if len(names) == 1 else _vector_of(values)


def _updated(conv: _Converter, body: sp.Basic, names: list, values: list) -> list:
    """The variables' next values: the body where they stand now.

    Every one written in at once, since the update of a system reads all of the
    previous iterate and none of the one being built. The heads that could not
    be read while the variables were variables are read again, as they are for
    a generated vector's elements, and a system's update has to come back as
    many values as it consumed.
    """
    written = body.subs(dict(zip(names, values, strict=True)), simultaneous=True)
    written = _retried(conv, written)
    if len(names) == 1:
        return [written]
    elements = _elements_of(_matrix(written))
    if len(elements) != len(names):
        raise ValueError("not that many values")
    return elements


def _inverted(body: sp.Basic, names: list) -> sp.Basic:
    """The function that undoes `body`, for an iteration counted backwards.

    Solving `u(x) = t` for `x`: `ITERATES(TAN(x), x, x, -1)` is `[x, ATAN(x)]`.
    A function it is only where the solution is the one it has, so anything
    with a choice of inverses has none here, and a system of variables has none
    at all.
    """
    if len(names) != 1:
        raise ValueError("no inverse of a system")
    (name,) = names
    point = sp.Dummy("t")
    solutions = sp.solve(sp.Eq(body, point), name)
    if len(solutions) != 1:
        raise ValueError("not invertible")
    return solutions[0].subs(point, name)


def _elements_of(matrix: sp.MatrixBase) -> list[sp.Basic]:
    """What a vector holds: its elements, or its rows where it is a matrix."""
    if matrix.rows == 1:
        return list(matrix)
    return [matrix[row, :] for row in range(matrix.rows)]


def _vector_of(elements: list) -> sp.Basic:
    """A vector of values that were computed rather than written.

    Their shape is all there is to go on, `_vector` having had the notation to
    read: rows of one width stack into a matrix, plain expressions make one row,
    and anything else is a vector sympy will not hold.
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


def _retried(conv: _Converter, value: sp.Basic) -> sp.Basic:
    """`value` with every head that could not be read before read again.

    A generated element is a converted body with a number written in, and the
    body was converted while the variable was still a variable: `v SUB i` had no
    index to take and became the inert `ELEMENT(v, i)`. Now that it has one, the
    call is worth making. A head the tables still cannot make sense of comes
    back the head it was.
    """
    return value.replace(
        lambda found: isinstance(found, AppliedUndef),
        lambda found: conv.call(type(found).__name__, found.args),
    )


def _settled(operand: sp.Basic) -> sp.Basic:
    """One operand of a boolean, written the one way a boolean can hold it.

    A conjunction holds its operands in a set, so which of them comes first is
    decided by how each is spelled rather than by how it was written - and one
    statement has two spellings, `a >= b` and `b <= a`. That matters because a
    range is *printed* as a chain, which turns the first into the second on the
    way out: without a canonical form the same statement would come back in a
    different order every time it was read.

    Nothing is decided here and nothing is simplified. `x >= 1` is already
    canonical; what changes is only which side a relation nobody is comparing
    was written from.
    """
    return operand.canonical if isinstance(operand, Relational) else operand


def _is_command_head(value: sp.Basic) -> bool:
    """Whether this is a command the pipeline has yet to evaluate."""
    return isinstance(value, AppliedUndef) and type(value).__name__ in COMMAND_HEADS


def _holds_command(value: sp.Basic) -> bool:
    """Whether such a head stands anywhere inside `value`."""
    if _is_command_head(value):
        return True
    return any(_is_command_head(found) for found in value.atoms(AppliedUndef))


def reread(expression: sp.Basic, context: Context) -> sp.Basic:
    """Every head the tables know, offered to them again.

    The pipeline calls this once it has evaluated a `FACTOR`, `EXPAND` or
    `SOLVE`, because a call written *around* one of those was left inert by the
    conversion: there was nothing to apply it to yet. Now there is, so the same
    tables get a second look at it, and `RHS(SOLVE(x^2 - 5*x + 6 = 0, x))`
    becomes the vector of roots the manual says it is.

    A subscript is offered too where its base has turned into a vector, `u SUB
    i` being `ELEMENT(u, i)` under another spelling.

    Only the names the tables define. A user's own function is inert because it
    is a user's own function, and offering it again would say nothing.
    """
    conv = _Converter(context)
    return expression.replace(
        _is_rereadable, lambda found: _reoffered(conv, found), simultaneous=False
    )


def _is_rereadable(found: sp.Basic) -> bool:
    if isinstance(found, AppliedUndef):
        return type(found).__name__ in FUNCTIONS
    return isinstance(found, Subscript) and isinstance(
        found.args[0], (sp.MatrixBase, InertVector)
    )


def _reoffered(conv: _Converter, found: sp.Basic) -> sp.Basic:
    name = "ELEMENT" if isinstance(found, Subscript) else type(found).__name__
    return conv.call(name, found.args)


def _identity_matrix(conv: _Converter, args: list) -> sp.Basic:
    size = _one(args)
    if not isinstance(size, sp.Integer) or size < 1:
        raise ValueError("not a size")
    return sp.eye(int(size))


def _cross(conv: _Converter, args: list) -> sp.Basic:
    """`CROSS(u, v)` over three elements each, or over two.

    The plane's case is the third component of the space's, the one the other
    two come to zero in, and it is that number rather than a vector holding it.
    """
    left, right = (_matrix(argument) for argument in args)
    if left.shape == right.shape == (1, 2):
        return left[0, 0] * right[0, 1] - left[0, 1] * right[0, 0]
    return left.cross(right)


def _row_reduce(conv: _Converter, args: list) -> sp.Basic:
    """`ROW_REDUCE(A)`, and `ROW_REDUCE(A, B)` over the augmented matrix.

    The echelon form the manual describes is the reduced one - the first nonzero
    element of every row is 1 and everything above it is 0 - which is what
    `rref` computes, by the multiplications and additions of rows that leave a
    system's solution set alone.

    A second matrix is adjoined to the right of the first and the pair reduced
    together, which is how `A . X = B` is solved: the columns `B` occupied come
    out holding `X` wherever `A` was nonsingular. A second argument that is a
    vector is one such column, `A . X = b` being the everyday case and the one
    VECTOR.MTH's `APPROX_EIGENVECTOR` is written in terms of.
    """
    matrix, *rest = args
    matrix = _matrix(matrix)
    if not rest:
        return matrix.rref()[0]
    adjoined = _matrix(_one(rest))
    if adjoined.rows == 1 and matrix.rows != 1:
        adjoined = adjoined.T
    return sp.Matrix.hstack(matrix, adjoined).rref()[0]


def _characteristic_polynomial(conv: _Converter, args: list) -> sp.Basic:
    """`CHARPOLY(A, v)`: `DET(A - v*IDENTITY_MATRIX(DIMENSION(A)))`.

    That determinant, and not the monic polynomial textbooks usually mean by the
    name, because the manual defines the function as the determinant of the
    difference of the matrix and a variable times the identity matrix. Negating
    all `n` rows of `v*I - A` multiplies its determinant by `(-1)^n`, so the two
    agree in even dimensions and differ in sign in odd ones. Sympy computes the
    monic one - the cheap way, by recurrence rather than by expanding a
    determinant full of `v` - and the sign puts it back the way Derive reads it.

    The polynomial is computed over a dummy and the variable written in
    afterwards, because a polynomial canonicalizes its generator into a bare
    symbol: computed over `v` itself, the answer would hold a `v` stripped of
    whatever `v` was declared, which is a different variable of the same name and
    would not cancel against the one the user has.
    """
    matrix, variable = _matrix_and_variable(conv, args)
    polynomial = matrix.charpoly(sp.Dummy()).as_expr(variable)
    return polynomial if matrix.rows % 2 == 0 else -polynomial


def _eigenvalues(conv: _Converter, args: list) -> sp.Basic:
    """`EIGENVALUES(A, v)`: the zeros of `A`'s characteristic polynomial in `v`.

    Equations rather than bare values, `[z = 2, z = b]`, because what the
    function does is solve the characteristic equation, and that is how Derive
    writes a solved equation's answer everywhere else. Each zero is listed once
    however many times it is a root: a multiplicity says how many parameters the
    eigenvector carries, and is no second eigenvalue.

    Sympy hands back a set, which has no order, and the answer has to print the
    same every time. Sympy's own canonical order is the one taken, and it is the
    order the manual's examples print in: numbers before names, and `a - b`
    before `a + b`.

    A zero sympy can only name by its index in a polynomial's root list is no
    answer: the manual solves the characteristic equation by the quadratic,
    cubic and quartic formulas and says exact eigenvalues are rarely attainable
    beyond that, so a matrix whose eigenvalues will not come out in radicals
    comes back the call it was written as.
    """
    matrix, variable = _matrix_and_variable(conv, args)
    zeros = sorted(matrix.eigenvals(), key=sp.default_sort_key)
    if any(zero.has(sp.CRootOf) for zero in zeros):
        raise ValueError("no closed form")
    return _vector_of([sp.Eq(variable, zero, evaluate=False) for zero in zeros])


def _matrix_and_variable(conv: _Converter, args: list) -> tuple[sp.MatrixBase, sp.Basic]:
    """A matrix and the variable to write an answer about it in.

    The variable is optional and defaults to `w`, which is the manual's default
    and carries whatever `w` has been declared, as a written one would. Whether
    the matrix is square is sympy's to complain about, and a call it will not
    take is a call that comes back the way it was written.
    """
    matrix, *rest = args
    variable = _one(rest) if rest else conv.symbol("w")
    if type(variable) is not sp.Symbol:
        raise TypeError("not a variable")
    return _matrix(matrix), variable


def _gradient(conv: _Converter, args: list) -> sp.Basic:
    """`GRAD(u)`, and `GRAD(u, alpha)` in the coordinates `alpha` describes.

    The gradient of a scalar field: the vector whose element `i` says how fast
    `u` grows per unit of length along coordinate `i`. Per unit of *length*, not
    per unit of coordinate, which is what the scale factor divides out - in
    spherical coordinates a radian of colongitude is `r*SIN(phi)` of arc, so the
    derivative with respect to it has to be divided by that to be a rate the
    other elements can be compared against.

    A vector has no gradient here: the manual's GRAD takes an expression, and
    VECTOR.MTH builds the Jacobian of a vector out of one GRAD per element.
    """
    expression, *rest = args
    if isinstance(expression, sp.MatrixBase):
        raise TypeError("not an expression")
    variables, scales = _coordinates(conv, rest)
    return _vector_of(_gradient_of(expression, variables, scales))


def _divergence(conv: _Converter, args: list) -> sp.Basic:
    """`DIV(v)`, and `DIV(v, alpha)`: how much the field `v` spreads out.

    The flux out of an infinitesimal coordinate box per unit of the volume it
    encloses. That volume is `H = h1*h2*...*hn` times the box's extent in the
    coordinates, and the pair of faces the field crosses along coordinate `i`
    has area `H/hi` times theirs - which is the whole of why the scale factors
    sit where they do, and why they cancel wherever they are all 1.
    """
    elements = _field(args[0])
    variables, scales = _coordinates(conv, args[1:], len(elements))
    return _divergence_of(elements, variables, scales)


def _laplacian(conv: _Converter, args: list) -> sp.Basic:
    """`LAPLACIAN(u)`, and `LAPLACIAN(u, alpha)`: `DIV(GRAD(u))`.

    Which is what the manual defines it as, so it is computed that way rather
    than by a formula of its own - the two would have to agree anyway, and one
    of them would then be a second place to get the scale factors wrong.
    """
    expression, *rest = args
    if isinstance(expression, sp.MatrixBase):
        raise TypeError("not an expression")
    variables, scales = _coordinates(conv, rest)
    gradient = _gradient_of(expression, variables, scales)
    return _divergence_of(gradient, variables, scales)


def _curl(conv: _Converter, args: list) -> sp.Basic:
    """`CURL(v)`, and `CURL(v, alpha)`, for a vector of two or three elements.

    The circulation around an infinitesimal loop per unit of the area it
    encloses, one loop per pair of coordinates. In space the three of them are
    the elements of a vector, each named after the coordinate its loop turns
    about.

    In the plane there is one loop and so one number, and Derive returns that
    number rather than the space vector `[0, 0, w]` it is the last element of.
    The manual calls this the more common convention; it is also the only one
    that keeps the answer in the plane the question was asked in.
    """
    elements = _field(args[0])
    if len(elements) not in (2, 3):
        raise ValueError("not a plane or space vector")
    variables, scales = _coordinates(conv, args[1:], len(elements))
    if len(elements) == 2:
        return _circulation(elements, variables, scales, 0, 1)
    turns = [_circulation(elements, variables, scales, *pair) for pair in _LOOPS]
    return _vector_of(turns)


#: Which pair of coordinates each element of a space curl turns in: the one
#: after the element's own, cyclically, and the one after that. Taking them in
#: that order rather than in sorted order is what gives the middle element its
#: sign, and it is the order that makes the curl a right-handed cross product of
#: the derivative with the field.
_LOOPS = ((1, 2), (2, 0), (0, 1))


def _potential(conv: _Converter, args: list) -> sp.Basic:
    """`POTENTIAL(v)`, `POTENTIAL(v, a)` and `POTENTIAL(v, a, alpha)`.

    The line integral of `v` from `a` to the coordinates themselves, along the
    staircase path that moves one coordinate at a time: leg `i` runs coordinate
    `i` from `a SUB i` up to where it is, with the coordinates before it already
    arrived and the ones after it still at `a`. Where `v` is conservative the
    path does not matter and the result is a scalar whose gradient is `v`.

    Where it is not, the integral still has a value and Derive still returns it.
    The manual is explicit that POTENTIAL "merely computes a certain line
    integral" and that checking `GRAD` of the answer against `v` - or, in two
    and three dimensions, `CURL(v)` against zero - is the caller's job. So there
    is nothing to detect here and no `?` to answer: a field with no potential
    gets the integral along that one path, which is a real number and the wrong
    answer to a question that has none.

    `a` defaults to the origin. The manual warns that this is a choice and not
    always a good one - a start where the field is infinite gives an infinite
    potential - and that is what the second argument is for.
    """
    elements = _field(args[0])
    start, rest = _start(args[1:], len(elements))
    variables, scales = _coordinates(conv, rest, len(elements))
    walked = sp.Integer(0)
    for index, element in enumerate(elements):
        arrived = zip(variables[index + 1 :], start[index + 1 :])
        along = (scales[index] * element).subs(list(arrived), simultaneous=True)
        walked += _leg(along, variables[index], start[index])
    return walked


def _vector_potential(conv: _Converter, args: list) -> sp.Basic:
    """`VECTOR_POTENTIAL(v)`, and the same optional `a` and `alpha`.

    A vector whose curl is `v`, for a three-element `v`. Vector potentials
    differ by any gradient, so one has to be picked, and the one picked here is
    the manual's: the third element is zero, and the other two are the line
    integrals that forces.

    With `A SUB 3` gone, two of the three curl equations are ordinary
    integrations along the third coordinate, and they fix `h1*A SUB 1` and
    `h2*A SUB 2` up to a function of the first two coordinates. The third
    equation then has to hold everywhere, and it is enough to make it hold on
    the slice `x3 = a SUB 3`, where those integrals vanish - that is the second
    integral in the first element, and there is nothing left over for the second
    element to carry.

    As with POTENTIAL, a field with nonzero divergence has no vector potential
    and gets this vector anyway: the manual leaves comparing `CURL` of the
    answer against `v` to the caller.
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
    return _vector_of(
        [
            (lifted - sliced) / h1,
            -_leg(h2 * h3 * first, x3, start[2]) / h2,
            sp.Integer(0),
        ]
    )


def _gradient_of(expression: sp.Basic, variables: list, scales: list) -> list[sp.Basic]:
    return [sp.diff(expression, x) / h for x, h in zip(variables, scales)]


def _divergence_of(elements: list, variables: list, scales: list) -> sp.Basic:
    volume = sp.Mul(*scales)
    faces = zip(elements, variables, scales)
    flux = sum((sp.diff(volume / h * u, x) for u, x, h in faces), sp.Integer(0))
    return flux / volume


def _circulation(field: list, variables: list, scales: list, j: int, k: int) -> sp.Basic:
    """The loop in the `j`, `k` coordinate plane, per unit of the area it holds.

    What goes around the loop is the field's component along each side times the
    length of that side, which is where the scale factors inside the derivatives
    come from; the area they enclose is where the one outside comes from.
    """
    along_k = sp.diff(scales[k] * field[k], variables[j])
    along_j = sp.diff(scales[j] * field[j], variables[k])
    return (along_k - along_j) / (scales[j] * scales[k])


def _leg(integrand: sp.Basic, variable: sp.Basic, low: sp.Basic) -> sp.Basic:
    """`INT(integrand, variable, low, variable)`, done rather than held.

    The integration variable is a dummy because the upper limit is the
    coordinate itself, and an integral sympy cannot do would leave that dummy in
    the answer under a name no worksheet can read - so it is no answer, and the
    call it came from comes back the way it was written.
    """
    dummy = sp.Dummy()
    walked = sp.integrate(integrand.subs(variable, dummy), (dummy, low, variable))
    if walked.has(dummy):
        raise ValueError("no antiderivative")
    return walked


def _start(rest: list, count: int) -> tuple[list[sp.Basic], list]:
    """Where the line integrals start, and whatever arguments follow it.

    The origin unless a vector says otherwise, one coordinate per element of the
    field, which is the manual's default of "a vector of zeros".
    """
    if not rest:
        return [sp.Integer(0)] * count, []
    given, *remaining = rest
    start = _field(given)
    if len(start) != count:
        raise ValueError("not a starting point")
    return start, remaining


def _coordinates(
    conv: _Converter, rest: list, count: int | None = None
) -> tuple[list[sp.Basic], list[sp.Basic]]:
    """The coordinate system a differential vector operator works in.

    Three-dimensional Cartesian `x`, `y`, `z` when nothing says otherwise, each
    carrying whatever it has been declared, as a written one would.

    A vector of variables names Cartesian coordinates of the caller's choosing,
    and there may be any number of them - `GRAD(u, [w, x, y, z])` is a
    four-element gradient. A two-row matrix is a coordinate geometry matrix:
    variables above, and below them the scale factors `hi` for which
    `ds^2 = (h1*dx1)^2 + ... + (hn*dxn)^2`. That is all an orthogonal curvilinear
    system is, which is why the same three lines of formula serve Cartesian,
    cylindrical and spherical coordinates alike. Only the matrices themselves are
    built in; the names `cylindrical` and `spherical` are assignments in
    VECTOR.MTH.

    `count` is how many coordinates the field has elements for, and it has to be
    all of them: a field with fewer elements than the system has coordinates is
    not a field on that system, and Derive answers `DIV([x, y])` with itself
    rather than with the divergence of some two-dimensional field it guessed at.
    A two-element `CURL` is the same question and the same answer - what makes
    it a planar curl is being given a planar system to work in.
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
    """A vector field's components. A matrix holds no field; a vector does."""
    matrix = _matrix(value)
    if matrix.rows != 1 or not matrix.cols:
        raise TypeError("not a vector")
    return list(matrix)


def _matrix(value: sp.Basic) -> sp.MatrixBase:
    if not isinstance(value, sp.MatrixBase):
        raise TypeError("not a matrix")
    return value


#: What each function name converts to. A name absent from here converts to an
#: inert head over its converted arguments, which is what keeps every arbitrary
#: user function alive through a round trip - and what makes
#: `DIF(F(x)^3, x)` differentiable. `SOLVE` is absent for a different reason:
#: it is `COMMAND_HEADS`, and the pipeline evaluates it.
FUNCTIONS: dict[str, Handler] = {
    **{name: _trig(func) for name, func in _TRIGONOMETRIC.items()},
    **{name: _arctrig(func) for name, func in _INVERSE_TRIGONOMETRIC.items()},
    **{name: _direct(func) for name, func in _DIRECT.items()},
    "ATAN": _atan,
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
    "GCD": _fold(sp.gcd),
    "LCM": _fold(sp.lcm),
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
    "TAYLOR": _taylor,
    "APPROX": _approximation,
    "IF": _conditional,
    "TRUTH_TABLE": _truth_table,
    "SUBS": _substitution,
    "HYPER": _hyper,
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


def _sympy_heads() -> dict[str, Handler]:
    """The way back from a sympy head nobody in Derive has a name for.

    A result can carry a function the notation was never given a spelling for -
    `BESSELI` and `EXP_POLAR` come out of a Bessel series, `EI` out of an
    exponential integral - and the printer writes each as its sympy class name
    upper-cased, because that is a name the grammar can read. Without the way
    back such a name reads as an inert head: the same mathematics, a different
    object, which sorts by another name and makes the answer settle only on a
    second Simplify. So the printer's rule is inverted here, over sympy's own
    function classes.

    Only names Derive does not define itself. Everything in the inventory has a
    reading of its own or waits for the pipeline as `SOLVE` does, and nothing
    here may displace one. What is left is names no Derive worksheet can mean
    anything else by.
    """
    reserved = set(FUNCTIONS) | BUILTIN_FUNCTIONS
    heads: dict[str, Handler] = {}
    for name in dir(sp.functions):
        head = getattr(sp.functions, name)
        if isinstance(head, sp.FunctionClass) and name.upper() not in reserved:
            heads.setdefault(name.upper(), _direct(head))
    return heads


#: What each such name converts to. A call these cannot take - the wrong number
#: of arguments, a vector where a tuple belongs, as `MEIJERG` carries - raises
#: and falls back to the inert head, like any other entry.
SYMPY_HEADS: dict[str, Handler] = _sympy_heads()

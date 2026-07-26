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

from collections.abc import Callable, Sequence
from dataclasses import replace
from fractions import Fraction

import sympy as sp

from rederive.engine.approximation import simplest
from rederive.engine.context import (
    Angle,
    Context,
    Domain,
    DomainKind,
    Precision,
    domain_of_node,
)
from rederive.model.expr import Kind, Node
from rederive.syntax.names import BUILTIN_FUNCTIONS

__all__ = [
    "Assign",
    "Declare",
    "Dot",
    "FunDef",
    "InertVector",
    "Logical",
    "PlusMinus",
    "StringLiteral",
    "Subscript",
    "Taylor",
    "Transposed",
    "to_sympy",
]


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
        """
        if isinstance(left, sp.MatrixBase) and isinstance(right, sp.MatrixBase):
            try:
                if left.rows == 1 and right.rows == 1:
                    return left.dot(right)
                if left.cols == right.rows:
                    return left * right
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
        """Element access on a vector, a subscripted variable otherwise."""
        base, index = self._children(node)
        if not isinstance(base, sp.MatrixBase):
            return self._subscript(node.children[0], base, index)
        element = _element(base, index)
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
        """Assembled unevaluated: the engine never decides a bare relation."""
        left, right = self._children(node)
        relation = _RELATIONS[str(node.value)]
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
            return _BOOLEAN[node.kind](*operands)
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
        """
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


def _log(conv: _Converter, args: list) -> sp.Basic:
    """`LOG(z)` is the natural logarithm; `LOG(z, w)` is `LN(z)/LN(w)`."""
    return sp.log(*args)


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


def _numerator(conv: _Converter, args: list) -> sp.Basic:
    return sp.fraction(sp.together(_one(args)))[0]


def _denominator(conv: _Converter, args: list) -> sp.Basic:
    return sp.fraction(sp.together(_one(args)))[1]


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

    An indefinite `SUM(u, k)` has no sympy object to be, so it stays inert
    until a pipeline knows what to do with it.
    """
    if len(args) == 1 and isinstance(args[0], sp.MatrixBase):
        return sp.Add(*args[0])
    expression, index, low, high = args
    return sp.Sum(expression, (index, low, high))


def _product(conv: _Converter, args: list) -> sp.Basic:
    """`PRODUCT(u, k, a, b)`, and `PRODUCT(v)` over a vector's elements."""
    if len(args) == 1 and isinstance(args[0], sp.MatrixBase):
        return sp.Mul(*args[0])
    expression, index, low, high = args
    return sp.Product(expression, (index, low, high))


def _taylor(conv: _Converter, args: list) -> sp.Basic:
    """`TAYLOR(u, x, a, n)`, unevaluated. Computing it is a pipeline's call."""
    return Taylor(*args)


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
    """`LIM(u, x, a)` is two-sided; a fourth argument picks a side."""
    if len(args) == 3:
        return sp.Limit(*args, dir="+-")
    expression, variable, point, side = args
    return sp.Limit(expression, variable, point, dir="+" if side > 0 else "-")


def _determinant(conv: _Converter, args: list) -> sp.Basic:
    return _matrix(_one(args)).det()


def _trace(conv: _Converter, args: list) -> sp.Basic:
    return _matrix(_one(args)).trace()


def _dimension(conv: _Converter, args: list) -> sp.Basic:
    """How many elements a vector has, or how many rows a matrix has."""
    matrix = _matrix(_one(args))
    return sp.Integer(matrix.cols if matrix.rows == 1 else matrix.rows)


def _element_of(conv: _Converter, args: list) -> sp.Basic:
    """`ELEMENT(v, i)` and `ELEMENT(m, i, j)`, counting from 1."""
    if len(args) == 2:
        element = _element(_matrix(args[0]), args[1])
        if element is None:
            raise ValueError("not an index")
        return element
    matrix, row, column = args
    return _matrix(matrix)[int(row) - 1, int(column) - 1]


def _identity_matrix(conv: _Converter, args: list) -> sp.Basic:
    size = _one(args)
    if not isinstance(size, sp.Integer) or size < 1:
        raise ValueError("not a size")
    return sp.eye(int(size))


def _cross(conv: _Converter, args: list) -> sp.Basic:
    left, right = args
    return _matrix(left).cross(_matrix(right))


def _matrix(value: sp.Basic) -> sp.MatrixBase:
    if not isinstance(value, sp.MatrixBase):
        raise TypeError("not a matrix")
    return value


#: What each function name converts to. A name absent from here converts to an
#: inert head over its converted arguments, which is what keeps `SOLVE`,
#: `RANDOM`, the financial functions and every arbitrary user function alive
#: through a round trip - and what makes `DIF(F(x)^3, x)` differentiable.
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
    "GCD": _fold(sp.gcd),
    "LCM": _fold(sp.lcm),
    "NUMERATOR": _numerator,
    "DENOMINATOR": _denominator,
    "DIF": _dif,
    "INT": _integral,
    "SUM": _summation,
    "PRODUCT": _product,
    "LIM": _limit,
    "TAYLOR": _taylor,
    "IF": _conditional,
    "SUBS": _substitution,
    "HYPER": _hyper,
    "DET": _determinant,
    "TRACE": _trace,
    "DIMENSION": _dimension,
    "ELEMENT": _element_of,
    "IDENTITY_MATRIX": _identity_matrix,
    "CROSS": _cross,
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
    reading of its own, or is deliberately inert - `SOLVE` and `FIT` must stay
    that way - and nothing here may displace one. What is left is names no
    Derive worksheet can mean anything else by.
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

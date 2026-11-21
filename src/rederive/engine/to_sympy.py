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

import contextlib
import contextvars
import datetime
import functools
import math
import operator
import string
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import product
from typing import Any

import mpmath
import sympy as sp
import sympy.printing.precedence as sp_precedence
from sympy.concrete.gosper import gosper_term
from sympy.core import relational as sympy_relational
from sympy.core.function import AppliedUndef, UndefinedFunction
from sympy.core.relational import Relational
from sympy.functions.elementary.piecewise import ExprCondPair
from sympy.logic.boolalg import Boolean, BooleanAtom, BooleanFunction
from sympy.matrices.exceptions import MatrixError
from sympy.ntheory.factor_ import factorint, perfect_power

from rederive.engine.approximation import (
    GUARD,
    approximated,
    carried_over,
    drawn_fraction,
    rescaled,
)
from rederive.engine.boundary import DEFAULT_AMOUNT, Amount
from rederive.engine.context import (
    Angle,
    Branch,
    Context,
    Domain,
    DomainKind,
    Notation,
    Precision,
    domain_of_node,
)
from rederive.engine.fresh import forgotten
from rederive.engine.integration import integral
from rederive.engine.ordering import main_order
from rederive.engine.printer import (
    HeldBase,
    WrittenBase,
    author_text,
    factor_order,
    logical_order,
    spelling,
    term_order,
    tree,
)
from rederive.engine.sets import (
    MOST_TERMS,
    SetComplement,
    SetSymbol,
    SetDifference,
    SetIntersection,
    SetOf,
    SetUnion,
    cartesian,
    elided,
    is_set,
    power_set,
    size,
)
from rederive.engine.vectors import (
    Vector,
    column,
    from_matrix,
    is_matrix,
    mapped,
    matrix_of,
    transposed,
)
from rederive.model.dfw import DERIVE_CHARACTERS
from rederive.model.expr import VARIADIC, Kind, Node, parameter_names
from rederive.model.settings import BASES
from rederive.syntax.names import (
    BINDING_FUNCTIONS,
    EXIT,
    KEPT_ALIASES,
    PRIME_MARK,
    PROGRAM_FUNCTIONS,
)
from rederive.syntax import writer
from rederive.syntax.writer import precedence as writer_precedence
from rederive.syntax.writer import write_expression

__all__ = [
    "CALCULUS_HEADS",
    "SIMPLE",
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
    "InOrder",
    "InOrderOr",
    "Logical",
    "MovingLimit",
    "AnyMovingLimit",
    "ON_FORM",
    "DECOMPOSING",
    "PREDICATES",
    "PlusMinus",
    "Factorial",
    "Power",
    "Quotient",
    "StringLiteral",
    "Subscript",
    "Taylor",
    "Transposed",
    "Vector",
    "added",
    "as_condition",
    "is_simple",
    "simple_constants",
    "unwrapped",
    "Commanded",
    "settled_key",
    "settled_from",
    "mark_settled",
    "settled_since",
    "SETTLED_ADDED",
    "vector_from",
    "authored_conditionals",
    "chosen_from",
    "connected",
    "divided",
    "dotted",
    "integer_root",
    "is_conditional",
    "iterated",
    "multiplied",
    "negated",
    "over_values",
    "power",
    "settled",
    "sorted_by",
    "steps",
    "table_row",
    "terms_by_expansion",
    "to_sympy",
]

#: Heads left for a command to evaluate: the in-expression forms of Factor, Expand and
#: soLve, the other solving functions, and `RATE`, a bounded soLve of section 6.12's
#: annuity. They need a simplified operand, so evaluation runs them once the operand is
#: a value; an enclosing call then works normally, e.g. `RHS(SOLVE(x^2 = 4, x, 0, 5))`
#: is 2.
COMMAND_HEADS = (
    "EXPAND",
    "FACTOR",
    "SOLVE",
    "SOLUTIONS",
    "NSOLVE",
    "NSOLUTIONS",
    "RATE",
)


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

    Its argument simplifies, but sums don't see through it, so `+-1 + 1` stays a sum
    rather than becoming `0` or `2`. Section 4.9 makes `+-u` a value of known magnitude
    and unknown sign, `s*u` for `s` one of 1 and -1, so the arithmetic that only
    multiplies the sign is done: `+-(-u)` is `+-u`, an odd power `(+-u)^n` is
    `+-(u^n)` (so `1/(+-1)` is `+-1`) and an even one is `u^n`. Products are taken in
    by `ambiguous_products`.
    """

    nargs = 1

    @classmethod
    def eval(cls, operand: sp.Basic) -> sp.Basic | None:
        if operand.is_zero:
            return sp.S.Zero
        if isinstance(operand, PlusMinus):
            return operand
        if operand.could_extract_minus_sign():
            return cls(-operand)
        return None

    def _eval_power(self, exponent: sp.Basic) -> sp.Basic | None:
        if not exponent.is_integer:
            return None
        if exponent.is_even:
            return self.args[0] ** exponent
        if exponent.is_odd:
            return PlusMinus(self.args[0] ** exponent)
        return None


def ambiguous_products(expression: sp.Basic) -> sp.Basic:
    """Each product with a factor `+-u` written as `+-` the product, since the unknown
    sign stays unknown when multiplied: `-COS(phi)*(+-1)` is `+-COS(phi)`, and so is a
    product of several ambiguous factors."""

    def taken_in(product: sp.Basic) -> sp.Basic:
        rest = [
            factor.args[0] if isinstance(factor, PlusMinus) else factor
            for factor in product.args
        ]
        return PlusMinus(sp.Mul(*rest))

    if not expression.has(PlusMinus):
        return expression
    return expression.replace(
        lambda e: e.is_Mul and any(isinstance(f, PlusMinus) for f in e.args), taken_in
    )


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


class Modulo(sp.Function):
    """`MOD(m, n)`, which is `m - n*FLOOR(m/n)`.

    Numbers, Gaussian ones included, are reduced at once, a power of integers by
    modular powering without building the power. Otherwise the call stands, normalized
    by identities that hold for every value: a rational modulus is cleared, a common
    integer factor moves out, and terms that are integer multiples of the modulus drop.
    """

    nargs = 2
    symmetric = False

    @classmethod
    def eval(cls, value: sp.Basic, modulus: sp.Basic) -> sp.Basic | None:
        return _residue(cls, value, modulus)


    def fdiff(self, argindex: int = 1) -> sp.Basic:
        """The derivative between the jumps, where the floor is constant."""
        if argindex == 1:
            return sp.S.One
        return -self.whole()

    def whole(self) -> sp.Basic:
        """The multiple of the modulus taken away: `FLOOR(m/n)`."""
        value, modulus = self.args
        return sp.floor(value / modulus)

    def as_floors(self) -> sp.Basic:
        """`m - n*FLOOR(m/n)`, the definition, for calculus on it."""
        value, modulus = self.args
        return value - modulus * self.whole()

    def _numpycode(self, printer: object) -> str:
        """The definition's code, for evaluating at many points at once to plot."""
        return printer._print(self.as_floors())

    _mpmathcode = _numpycode
    _lambdacode = _numpycode
    _pythoncode = _numpycode


class SymmetricModulo(Modulo):
    """`MODS(m, n)`, which is `m - n*FLOOR(m/n + 1/2)`, in [-n/2, n/2) for n > 0."""

    symmetric = True

    def whole(self) -> sp.Basic:
        value, modulus = self.args
        return sp.floor(value / modulus + sp.S.Half)


def residues_as_floors(value: sp.Basic) -> sp.Basic:
    """`value` with each `MOD` and `MODS` written as its definition, a floor."""
    if not value.has(Modulo):
        return value
    return value.replace(lambda part: isinstance(part, Modulo), lambda m: m.as_floors())


def floors_as_residues(value: sp.Basic) -> sp.Basic:
    """`value` with each sum's terms `u - v*FLOOR(u/v)` collected into `MOD(u, v)` (see
    `_floors_as_residues`)."""
    if not (isinstance(value, sp.Basic) and value.has(sp.floor)):
        return value
    return value.replace(
        lambda part: isinstance(part, sp.Add) and part.has(sp.floor),
        _floors_as_residues,
    )


def _residue(cls: type[Modulo], value: sp.Basic, modulus: sp.Basic) -> sp.Basic | None:
    """`cls(value, modulus)` worked out, or None for the call to stand as it is."""
    if not (isinstance(value, sp.Expr) and isinstance(modulus, sp.Expr)):
        raise TypeError("not a number")
    if isinstance(value, Vector) or isinstance(modulus, Vector):
        raise TypeError("not a number")
    if modulus.is_zero:
        return value
    if isinstance(value, sp.Integer) and isinstance(modulus, sp.Integer):
        number, divisor = int(value), int(modulus)
        if cls.symmetric:
            nearest = (2 * number + divisor) // (2 * divisor)
            return sp.Integer(number - divisor * nearest)
        return sp.Integer(number % divisor)
    powered = _modular_power(value, modulus)
    if powered is not None:
        return cls(powered, modulus) if cls.symmetric else powered
    if value.is_number and modulus.is_number:
        quotient = value / modulus
        if cls.symmetric:
            real, imaginary = quotient.as_real_imag()
            whole = sp.floor(real + sp.S.Half) + sp.I * sp.floor(imaginary + sp.S.Half)
        else:
            whole = sp.floor(quotient)
        if whole.has(sp.floor):
            return None
        return sp.expand(value - modulus * whole)
    if modulus.is_Rational and not modulus.is_Integer:
        return cls(modulus.q * value, modulus.p) / modulus.q
    kept = [term for term in sp.Add.make_args(value) if not (term / modulus).is_integer]
    if len(kept) < len(sp.Add.make_args(value)):
        return cls(sp.Add(*kept), modulus)
    content, _ = value.as_content_primitive()
    divisor = (
        abs(modulus) if modulus.is_Rational else modulus.as_content_primitive()[0]
    )
    if content.is_Integer and divisor.is_Integer:
        common = sp.gcd(content, divisor)
        if common > 1:
            return common * cls(value / common, modulus / common)
    return None


def _modular_power(value: sp.Basic, modulus: sp.Basic) -> sp.Integer | None:
    """`MOD(value, modulus)` by modular powering, where `value` is a nonnegative integer
    power of an integer and the modulus an integer, or None."""
    if not isinstance(modulus, sp.Integer):
        return None
    if isinstance(value, Power):
        base, exponent = value.args
    elif isinstance(value, sp.Pow):
        base, exponent = value.base, value.exp
    else:
        return None
    if not (isinstance(base, sp.Integer) and isinstance(exponent, sp.Integer)):
        return None
    if exponent < 0:
        return None
    return sp.Integer(pow(int(base), int(exponent), int(modulus)))


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
    """`u^v` whose value is a number too big to build, or a power of a vector that stands.

    A separate head rather than an unevaluated `Pow`, because `together` and
    `as_numer_denom` rebuild powers from their parts, which would trigger the
    evaluation. See `_explodes` for when one is made. A vector's power that becomes an
    integer one by substitution is worked out (`power`).

    It can still be approximated if the result is small enough
    (`APPROXIMATE_POWER_BITS`). This is done in floating point, with extra precision
    equal to the exponent's bit length, since the exponent magnifies the base's error:
    `(1 + 10^-20)^10^20` is `#e` to twenty decimal places.
    """

    nargs = 2

    @classmethod
    def eval(cls, base: sp.Basic, exponent: sp.Basic) -> sp.Basic | None:
        if (
            isinstance(base, Vector)
            and isinstance(exponent, sp.Integer)
            and base.args
            and not any(isinstance(element, Vector) for element in base.args)
        ):
            return power(base, exponent)
        return None

    def _eval_evalf(self, prec: int) -> sp.Float | None:
        base, exponent = self.args
        if not (base.is_Rational and exponent.is_Rational) or base.is_zero:
            return None
        size = abs(float(exponent)) * abs(float(sp.log(abs(base), 2).evalf(15)))
        if size > APPROXIMATE_POWER_BITS:
            return None
        working = prec + int(abs(exponent) + 1).bit_length() + 16
        exact = sp.Float(exponent, precision=working)
        value = sp.Float(abs(base), precision=working) ** exact
        if base.is_negative and not exponent.is_Integer:
            # The principal value: (-b)^e is b^e*#e^(#i*pi*e). The fraction of a turn
            # is reduced exactly first, so a huge numerator loses no digits.
            turn = exponent - 2 * sp.floor(exponent / 2)
            with mpmath.workprec(working):
                angle = mpmath.pi * mpmath.mpf(turn.p) / turn.q
                cosine, sine = mpmath.cos(angle), mpmath.sin(angle)
            return sp.Float(value * sp.Float(cosine, precision=working), precision=prec) \
                + sp.Float(value * sp.Float(sine, precision=working), precision=prec) * sp.I
        if base.is_negative:
            value = value if exponent % 2 == 0 else -value
        return sp.Float(value, precision=prec)


class Factorial(sp.Function):
    """`n!` of an integer `n` whose value is too big to build (see `_explodes_factorial`).

    Like `Power`, a separate head, so nothing rebuilds and evaluates it. It can be
    approximated, which takes no longer than for a small `n`.
    """

    nargs = 1
    is_integer = True
    is_positive = True
    is_even = True

    def _eval_evalf(self, prec: int) -> sp.Float | None:
        (number,) = self.args
        if not number.is_Integer:
            return None
        with mpmath.workprec(prec):
            return sp.Float(mpmath.factorial(int(number)), precision=prec)


class Truth(sp.Function):
    """A logical value or relation used in arithmetic, such as `false` in `x*PRIME(4)`.

    Sympy arithmetic accepts only expressions; a boolean operand is either refused or
    produces a sum or product nothing can take apart. Wrapped, it behaves like an
    unknown (`true + true` is `2*true`) and prints as its value.
    """

    nargs = 1


class LogicalSymbol(sp.Symbol):
    """A variable declared `:epsilon Logical`: a truth value, not a number, which only
    the logical operators act on."""

    is_number = False


class InfixBase(sp.Function):
    """A word operator of Derive 6 that stands, such as `[1, 2] DOTPRODUCT [1, 2, 3]`.

    Each operator is a subclass; the first two arguments are the operands.
    """

    word = ""
    nargs = 2

    @property
    def operator(self) -> str:
        return self.word

    @property
    def precedence(self) -> int:
        """How tightly the operator binds as written, on sympy's scale: a subscript link
        like a subscript, the vector products between sums and products, `SUBSET` just
        below the relations, so that a relation's operand is parenthesized."""
        return _INFIX_PRECEDENCE.get(self.word, sp_precedence.PRECEDENCE["Atom"] - 1)


class RowsOf(InfixBase):
    """`A ROW i` where `A` isn't a matrix or `i` isn't an index of it."""

    word = "ROW"


class ColumnsOf(InfixBase):
    """`A COL j` where `A` isn't a matrix or `j` isn't an index of it."""

    word = "COL"


class Deeper(InfixBase):
    """`A SUB SUB j`, the third argument being how many levels down the index applies,
    one per `SUB` after the first, where some element on the way doesn't have it."""

    nargs = 3

    @property
    def operator(self) -> str:
        return " ".join(["SUB"] * (int(self.args[2]) + 1))


class DotProduct(InfixBase):
    """`u DOTPRODUCT v` other than of two flat vectors of one length."""

    word = "DOTPRODUCT"


class CrossProduct(InfixBase):
    """`u CROSSPRODUCT v` other than of two vectors of two or three elements."""

    word = "CROSSPRODUCT"


class SubsetOf(InfixBase):
    """`u SUBSET v` other than of two explicit sets."""

    word = "SUBSET"


_INFIX_PRECEDENCE = {
    "DOTPRODUCT": sp_precedence.PRECEDENCE["Add"] + 4,
    "CROSSPRODUCT": sp_precedence.PRECEDENCE["Add"] + 6,
    "SUBSET": sp_precedence.PRECEDENCE["Relational"] - 1,
}


class InOrder(sp.And):
    """A conjunction kept and written in the order given, as the equations of a system's
    solution follow the order of its variables: `SOLVE(..., [z, x, y])` gives `z = 1
    AND x = 2 AND y = 3`."""

    #: Tells the printer not to sort the operands.
    in_order = True

    def __new__(cls, *args: sp.Basic) -> sp.Basic:
        args = tuple(dict.fromkeys(sp.sympify(arg) for arg in args))
        if sp.false in args:
            return sp.false
        args = tuple(arg for arg in args if arg is not sp.true)
        if not args:
            return sp.true
        if len(args) == 1:
            return args[0]
        conjunction = sp.Basic.__new__(cls, *args)
        conjunction._argset = frozenset(args)
        return conjunction


class InOrderOr(sp.Or):
    """A disjunction kept and written in the order given, as `SOLVE` lists the
    solutions of an equation: `x = 2 - #i OR x = 2 + #i`, so that `RHS` of it is the
    second."""

    #: Tells the printer not to sort the operands.
    in_order = True

    def __new__(cls, *args: sp.Basic) -> sp.Basic:
        args = tuple(dict.fromkeys(sp.sympify(arg) for arg in args))
        if sp.true in args:
            return sp.true
        args = tuple(arg for arg in args if arg is not sp.false)
        if not args:
            return sp.false
        if len(args) == 1:
            return args[0]
        disjunction = sp.Basic.__new__(cls, *args)
        disjunction._argset = frozenset(args)
        return disjunction


class Logical(sp.Function):
    """A boolean operator sympy declines, such as `3 AND p`.

    The first argument names the operator; the rest are its operands.
    """


class Assign(sp.Function):
    """`name := value`. The value converts; the assignment stays put."""


class FunDef(sp.Function):
    """`F(x, y) := body`, as `FunDef(name, params, body)`; see `formals`."""


class Commanded(sp.Function):
    """The answer of a `FACTOR`, `EXPAND` or `SOLVE` head, in the command's form.

    A factorization is a form, not just a value: `FACTOR(12)` is `2^2*3`, which
    Simplify's normal form would multiply back out. A solution is written as soLve
    writes it, `x >= (3*y - 7)/2`, not as Simplify would. So the answer is wrapped
    through the rest of the pipeline and unwrapped only at the end. A call taking it
    as an argument gets the unwrapped value: `RHS(SOLVE(x^2 - 5*x + 6 = 0, x))` is
    `[2, 3]`.
    """

    nargs = 1

    def _eval_derivative(self, symbol: sp.Basic) -> sp.Basic:
        # The answer is its value: `DIF(EXPAND(u)/2, x)` is the derivative of `u/2`.
        return self.args[0].diff(symbol)


def formals(head: Node, converted: Sequence[sp.Basic]) -> sp.Basic:
    """A `FunDef`'s parameters from a definition's `PARAMS` and its converted children:
    a tuple of names, and of assignments for parameters with a default, or for the
    variadic head `F v`, the one name alone."""
    if head.value == VARIADIC:
        return converted[0]
    return sp.Tuple(*converted)


def programmed(node: Node) -> bool:
    """Whether `node` holds a program: an assignment, a definition, a program
    construct, or `exit`."""
    if node.kind in (Kind.ASSIGN, Kind.FUNDEF):
        return True
    if node.kind is Kind.NAME and node.value == EXIT:
        return True
    if node.kind is Kind.CALL and str(node.children[0].value) in _STATEMENT_CALLS:
        return True
    return any(programmed(child) for child in node.children)


#: Calls that are done for their effects: the program constructs, and the built-ins
#: that loop for effect, show or write text, or read a file.
_STATEMENT_CALLS = PROGRAM_FUNCTIONS | {"MAP", "DISPLAY", "WRITE", "LOAD"}


def as_written(node: Node) -> sp.Basic:
    """`node` as an atom that prints as written (see `WrittenBase`), one class per
    written text, so equal trees are the same atom."""
    text = write_expression(node)
    cls = _WRITTEN.get(text)
    if cls is None:
        cls = _WRITTEN[text] = UndefinedFunction(
            f"WRITTEN_{len(_WRITTEN)}", bases=(WrittenBase,), __dict__={"node": node}
        )
    return cls()


#: The classes `as_written` made, by written text.
_WRITTEN: dict[str, type] = {}


def held(node: Node) -> sp.Basic:
    """The held value of the tree `node` (see `HeldBase`), one class per written text,
    so equal trees are the same atom."""
    text = write_expression(node)
    cls = _HELD.get(text)
    if cls is None:
        level = _HELD_PRECEDENCE[writer_precedence(node)]
        cls = _HELD[text] = UndefinedFunction(
            f"HELD_{len(_HELD)}",
            bases=(HeldBase,),
            __dict__={"node": node, "precedence": level},
        )
    return cls()


#: The classes `held` made, by written text.
_HELD: dict[str, type] = {}

#: The sympy precedence a held value prints at, by the writer's precedence of its tree.
_HELD_PRECEDENCE = {
    writer.ASSIGN: sp_precedence.PRECEDENCE["Lambda"],
    writer.IFF: sp_precedence.PRECEDENCE["Lambda"],
    writer.IMP: sp_precedence.PRECEDENCE["Lambda"],
    writer.XOR: sp_precedence.PRECEDENCE["Xor"],
    writer.OR: sp_precedence.PRECEDENCE["Or"],
    writer.AND: sp_precedence.PRECEDENCE["And"],
    writer.NOT: sp_precedence.PRECEDENCE["And"] + 1,
    writer.RELATION: sp_precedence.PRECEDENCE["Relational"],
    writer.UNION: sp_precedence.PRECEDENCE["Relational"] + 1,
    writer.INTERSECTION: sp_precedence.PRECEDENCE["Relational"] + 2,
    writer.DIFFERENCE: sp_precedence.PRECEDENCE["Relational"] + 3,
    writer.SUM: sp_precedence.PRECEDENCE["Add"],
    writer.DOT: sp_precedence.PRECEDENCE["Add"] - 1,
    writer.CROSS: sp_precedence.PRECEDENCE["Add"] - 1,
    writer.NEG: sp_precedence.PRECEDENCE["Add"],
    writer.PRODUCT: sp_precedence.PRECEDENCE["Mul"],
    writer.POW: sp_precedence.PRECEDENCE["Pow"],
    writer.POSTFIX: sp_precedence.PRECEDENCE["Func"],
    writer.SUB: sp_precedence.PRECEDENCE["Atom"] - 1,
    writer.ATOM: sp_precedence.PRECEDENCE["Atom"],
}


def held_value(node: Node, plain: Converter) -> sp.Basic:
    """The value of a quote of the tree `node`, its names already filled in: held, but
    a vector or set of held elements, in which a name is itself, and a number or a
    string itself."""
    if node.kind in (Kind.VECTOR, Kind.SET) and not any(
        child.kind is Kind.ELLIPSIS for child in node.children
    ):
        elements = [
            plain.convert(child) if child.kind is Kind.NAME else held_value(child, plain)
            for child in node.children
        ]
        return Vector(*elements) if node.kind is Kind.VECTOR else SetOf(*elements)
    if node.kind in (Kind.NUMBER, Kind.STRING):
        return plain.convert(node)
    return held(node)


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
            series = None
        if series is None:
            # Sympy can't expand an integral with a variable limit, but its derivative
            # is known: the integrand. There are no derivatives at infinity.
            if expression.has(*_KINKED) or not point.is_finite:
                return self
            summed = _summed_from_derivatives(expression, variable, point, int(order))
            return self if summed is None else summed
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

    An unevaluated indefinite integral over the variable is an antiderivative `G(y)`
    of `F(y)`, and `G(c*y)` is an antiderivative of `c*F(c*y)`. So at a point that is
    the variable times a constant, the integral's variable is changed, as Derive does:
    `LIM(INT(F(y), y), y, y/x)` is `INT(F(y/x)/x, y)`. At any other point that
    mentions the variable the head stands, since `INT(F(y), y, y^2)` would mean
    nothing.
    """

    nargs = 3

    @property
    def free_symbols(self) -> set[sp.Basic]:
        """The variable is bound in the operand, which it is substituted out of."""
        expression, variable, point = self.args
        return (expression.free_symbols - {variable}) | point.free_symbols

    def _eval_subs(self, old: sp.Basic, new: sp.Basic) -> sp.Basic | None:
        """Substituting for the variable substitutes into the point only: with `y`
        bound in the operand, `LIM(F(y), y, x*y)` at `y = w` is `LIM(F(y), y, x*w)`."""
        expression, variable, point = self.args
        if old == variable:
            return self.func(expression, variable, point._subs(old, new))
        return None

    def doit(self, deep: bool = False, **hints) -> sp.Basic:
        """The substitution, or this head if it fails."""
        expression, variable, point = self.args
        try:
            indefinite = [
                found
                for found in expression.atoms(sp.Integral)
                if found.limits == ((variable,),)
            ]
            if not indefinite:
                return self._substituted(expression, variable, point)
            factor = sp.cancel(point / variable)
            # A symbolic factor is taken to be nonzero, as `SQRT(t)` for real `t`.
            if factor.has(variable) or factor.is_zero or factor == 0:
                return self
            held = {found: sp.Dummy() for found in sp.ordered(indefinite)}
            rest = expression.xreplace(held)
            if rest.has(sp.Integral):
                return self
            changed = {
                dummy: _scaled_integral(found.function, variable, factor)
                for found, dummy in held.items()
            }
            return self._substituted(rest, variable, point).xreplace(changed)
        except Exception:
            return self

    @staticmethod
    def _substituted(
        expression: sp.Basic, variable: sp.Basic, point: sp.Basic
    ) -> sp.Basic:
        return expression.subs(variable, point)


class AnyMovingLimit(MovingLimit):
    """A `MovingLimit` under Branch Any, whose substitution takes each root of a power
    whatever the sign of its base (see `any_root`): `LIM(SQRT(y), y, y*x^2)` is
    `x*SQRT(y)`, where sympy, for real `x`, writes `ABS(x)*SQRT(y)`."""

    @staticmethod
    def _substituted(
        expression: sp.Basic, variable: sp.Basic, point: sp.Basic
    ) -> sp.Basic:
        # Without assumptions, so that sympy keeps the root of a power for `any_root`.
        symbols = (expression.free_symbols | point.free_symbols) - {variable}
        plain = {
            symbol: sp.Dummy(symbol.name)
            for symbol in sp.ordered(symbols)
            if symbol.is_Symbol
        }
        moved = variable.xreplace(plain)
        if moved == variable:
            plain[variable] = moved = sp.Dummy(variable.name)
        substituted = expression.xreplace(plain).subs(moved, point.xreplace(plain))

        def rooted(power: sp.Pow) -> sp.Basic:
            root = any_root(*power.args)
            return power if root is None else root

        substituted = substituted.replace(
            lambda e: e.is_Pow and e.exp.is_Rational and not e.exp.is_Integer, rooted
        )
        return substituted.xreplace({dummy: symbol for symbol, dummy in plain.items()})


def _scaled_integral(
    integrand: sp.Basic, variable: sp.Basic, factor: sp.Basic
) -> sp.Basic:
    """`INT(integrand, variable)` with `variable` changed to `factor*variable`, the new
    integrand over one denominator and its constant factor taken out:
    `INT(1/(SIN(1/y) + y^3), y)` at `y/x` is `x^2*INT(1/(x^3*SIN(x/y) + y^3), y)`."""
    changed = sp.together(factor * integrand.subs(variable, factor * variable))
    constant, varying = changed.as_independent(variable, as_Add=False)
    return constant * sp.Integral(varying, variable)


class Approx(sp.Function):
    """`APPROX(u, n)`: `u` rounded to `n` digits, deferred until evaluation.

    Rounding must apply to the answer, not the question: `APPROX(INT(u, x))` must see
    the integral's value. So it waits like the calculus heads and is evaluated after
    them, which conversion can't do without a pipeline.

    The digit count is filled in even if the call omitted it, because the caller's
    precision is known only here.

    Rounding is no function of a variable to differentiate: the derivative of a rounded
    value is the derivative rounded, `DIF(APPROX(b*LN(x)), x)` is `APPROX(b/x)`.
    """

    nargs = 2

    def _eval_derivative(self, variable: sp.Basic) -> sp.Basic:
        value, digits = self.args
        return self.func(sp.diff(value, variable), digits)


class Antidifference(sp.Function):
    """`SUM(u, n)` with no limits, unevaluated until `.doit()`.

    `F` is the antidifference of `f` when `f(n) = F(n + 1) - F(n)`, the discrete
    analogue of an antiderivative. The constant of summation is dropped, as for
    integrals. Sympy has no head for this, as with `Taylor`.
    """

    nargs = 2

    def _eval_subs(self, old: sp.Basic, new: sp.Basic) -> sp.Basic | None:
        """A value for the index is a substitution into the unknown antidifference, kept as
        such, not a value for the bound index: RECUREQN.MTH's `LIN2_CCF_BV` takes
        `LIM` of an unsolved sum at a point, which isn't `SUM(u(0), 0)`."""
        if old == self.args[1] and not isinstance(new, sp.Symbol):
            return sp.Subs(self, old, new)
        return None

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

    def _eval_subs(self, old: sp.Basic, new: sp.Basic) -> sp.Basic | None:
        """A value for the index is a substitution into the unknown antiquotient, kept as
        such, not a value for the bound index: RECUREQN.MTH's `LIN2_CCF_BV` takes
        `LIM` of an unsolved sum at a point, which isn't `SUM(u(0), 0)`."""
        if old == self.args[1] and not isinstance(new, sp.Symbol):
            return sp.Subs(self, old, new)
        return None

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
    AnyMovingLimit,
    Taylor,
    Approx,
)


# -- the constant table -----------------------------------------------------


class UnitCircle(sp.Symbol):
    """`unit_circle`, an arbitrary point on the complex unit circle, as in Derive's
    answer to `ABS(z) = 2` for a complex `z`: `z = 2*unit_circle`.

    A symbol without the assumptions of an undeclared variable; in particular it isn't
    real, so `SQRT(x^2) -> ABS(x)` must not fire on it. What is known of it is its
    absolute value, 1. Derive's special arithmetic for it (such as `unit_circle*inf`
    for complex infinity) is not implemented.
    """

    def __new__(cls, name: str = "unit_circle", **assumptions: bool) -> UnitCircle:
        return super().__new__(cls, name, **assumptions)

    def _eval_Abs(self) -> sp.Basic:
        return sp.S.One


#: The constant `unit_circle`.
UNIT_CIRCLE = UnitCircle()


class Degree(sp.NumberSymbol):
    """`deg`, one degree, under `Angle := Degree`: the number `pi/180`, kept as a
    constant so that an angle in degrees reads as one, `45*deg`.

    The angle mode affects only how angles are shown. Trigonometric functions take
    radians in either mode (`Converter.angle_in` puts `pi/180` in its place), and an
    approximation is in radians. Under `Angle := Radian`, `deg` is `pi/180` itself.
    """

    is_real = True
    is_positive = True
    is_negative = False
    is_irrational = True
    is_number = True
    is_algebraic = False
    is_transcendental = True
    __slots__ = ()

    #: Marks the constant for the printer, which can't import this module.
    degree_unit = True

    def __new__(cls) -> Degree:
        return sp.AtomicExpr.__new__(cls)

    def _as_mpf_val(self, prec: int) -> Any:
        libmp = mpmath.libmp
        return libmp.mpf_div(libmp.mpf_pi(prec + 10), libmp.from_int(180), prec)

    def approximation_interval(self, number_cls: type) -> tuple | None:
        if issubclass(number_cls, sp.Rational):
            return (sp.Rational(1745, 100000), sp.Rational(1746, 100000))
        return None

    def _sympystr(self, printer: Any) -> str:
        return "deg"

    def __getnewargs__(self) -> tuple:
        return ()


#: The constant `deg` under `Angle := Degree`.
DEGREE = Degree()


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
    "unit_circle": UNIT_CIRCLE,
}

RELATIONS: dict[str, Callable[..., sp.Basic]] = {
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
    "IFF": lambda a, b: ~(a ^ b),
}

LOGICAL_WORDS: dict[Kind, str] = {
    Kind.NOT: "NOT",
    Kind.AND: "AND",
    Kind.OR: "OR",
    Kind.XOR: "XOR",
    Kind.IMP: "IMP",
    Kind.IFF: "IFF",
}

#: The nodes that are statements, true or false: relations and their connectives.
STATEMENT_KINDS = frozenset({Kind.REL, *LOGICAL_WORDS})

#: The relations that compare two statements, by the connective they then are.
COMPARING_TRUTHS = {"=": "IFF", "/=": "XOR"}


def is_statement(value: sp.Basic) -> bool:
    """Whether a value is a relation, a combination of relations, or a variable declared
    Logical (not a truth value alone, nor an undeclared variable, which may yet be a
    number, nor a logical operator on one, which is bitwise on integers)."""
    if isinstance(value, (Relational, LogicalSymbol)):
        return True
    if isinstance(value, BooleanFunction):
        return all(
            is_statement(arg) or isinstance(arg, BooleanAtom) for arg in value.args
        )
    if isinstance(value, Logical):
        return all(is_statement(arg) for arg in value.args[1:])
    return False


def numeral(node: Node) -> sp.Rational:
    """The numeral `node`, written in `value` as `2.5`, `25` or `1/3`.

    It is the exact rational it spells in every precision mode, so `0.1` is one tenth,
    not the nearest binary float, and a numeral typed with more digits than the
    precision keeps them: approximation applies to the irrational numbers of an answer,
    not to what the user wrote. A program reads its numerals at every pass, so their
    values are remembered.
    """
    text = str(node.value)
    value = _NUMERALS.get(text)
    if value is None:
        value = sp.Rational(text)
        if len(_NUMERALS) >= _NUMERALS_KEPT:
            _NUMERALS.clear()
        _NUMERALS[text] = value
    return value


#: The values of the numerals `numeral` read last, by their text, and how many it keeps.
_NUMERALS: dict[str, sp.Rational] = {}
_NUMERALS_KEPT = 4096
forgotten(_NUMERALS.clear)


def to_sympy(node: Node, context: Context | None = None) -> sp.Basic:
    """Translate `node` into the sympy object that means the same thing."""
    return Converter(context or Context()).convert(node)


#: The symbols `Converter.symbol` made for names whose domain has no interval, by name,
#: kind of domain and whether the name is declared; and how many it keeps.
_PLAIN_SYMBOLS: dict[tuple[str, DomainKind, bool], sp.Basic] = {}
_PLAIN_SYMBOLS_KEPT = 4096
forgotten(_PLAIN_SYMBOLS.clear)


class Converter:
    """State for one conversion: the context and the labels being expanded.

    `parameters` are those of the definition whose body is being converted.
    """

    def __init__(self, context: Context, open_labels: frozenset[int] = frozenset()):
        self.context = context
        self.open_labels = open_labels
        self.parameters: frozenset[str] = frozenset()
        # The variables of the binding calls whose body is being converted, and whether
        # every variable counts, for `mentions_parameters`.
        self.held: frozenset[str] = frozenset()
        self.holding_forms = False

    def sets_simplified(self, values: list[sp.Basic]) -> list[sp.Basic]:
        """`values` with each set among them simplified, so that its elements are
        compared as simplified expressions. Without an environment to simplify in, the
        values are returned as they are."""
        return values

    def simplified_values(self, values: list[sp.Basic]) -> list[sp.Basic]:
        """`values` simplified, for a built-in that reads its arguments' form, as
        Derive simplifies them first. Without an environment to simplify in, the values
        are returned as they are."""
        return values

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
                return self.call(str(node.children[0].value), self._arguments(node))
            case Kind.FUNCPOW:
                return self._funcpow(node)
            case Kind.VECTOR:
                return self._vector(node)
            case Kind.SET:
                return self._set(node)
            case Kind.SETOP:
                return self._set_operation(node)
            case Kind.REL:
                return self._relation(node)
            case Kind.NOT | Kind.AND | Kind.OR | Kind.XOR | Kind.IMP | Kind.IFF:
                return self._logical(node)
            case Kind.INFIX:
                return self._infix(node)
            case Kind.ASSIGN:
                return self._assign(node)
            case Kind.FUNDEF:
                return self._fundef(node)
            case Kind.DOMAIN:
                return self._declaration(node)
            case Kind.SHOWVALUE:
                return self.convert(node.children[0])
            case Kind.QUOTE:
                # Without an evaluator a quote stays as written.
                return as_written(node)
        # PARAMS, INTERVAL and ELLIPSIS only appear inside their owning nodes; anything
        # else becomes an inert head.
        return self.opaque(str(node.kind).upper(), self._children(node))

    def _children(self, node: Node) -> list[sp.Basic]:
        return [self.convert(child) for child in node.children]

    def _arguments(self, node: Node) -> list[sp.Basic]:
        """A call's arguments converted, a binding call's body with its variables held
        as parameters are (see `mentions_parameters`): `VECTOR(STRING(k), k, 1, n)`
        isn't `VECTOR("k", k, 1, n)`."""
        head, *arguments = node.children
        if str(head.value) not in BINDING_FUNCTIONS or len(arguments) < 2:
            return [self.convert(argument) for argument in arguments]
        index = arguments[1]
        names = [index] if index.kind is Kind.NAME else index.children
        bound = frozenset(str(n.value) for n in names if n.kind is Kind.NAME)
        outer = self.held
        self.held = outer | bound
        try:
            body = self.convert(arguments[0])
        finally:
            self.held = outer
        return [body, *(self.convert(argument) for argument in arguments[1:])]

    # -- leaves -------------------------------------------------------------

    def _number(self, node: Node) -> sp.Basic:
        return numeral(node)

    def _name(self, node: Node) -> sp.Basic:
        name = str(node.value)
        if name == "deg":
            return self.degree()
        constant = CONSTANTS.get(name)
        if constant is not None:
            return constant
        return self.symbol(name)

    def symbol(self, name: str) -> sp.Basic:
        """The symbol for `name`, with its declared assumptions.

        A degenerate interval declares a value: `x :epsilon Real [7, 7]` makes `x` the
        number 7.

        A name declared a vector becomes a `MatrixSymbol`, since sympy knows section
        8.8's matrix algebra (transposes and inverses reverse products, the determinant
        of an inverse is a reciprocal) for those but not for merely noncommutative
        symbols.

        A name declared a set becomes a `SetSymbol`, which the set operators act on.

        This applies only to names declared explicitly. `default :epsilon Vector`
        would make every variable a matrix, including arguments of `SIN`, so the
        default domain contributes its assumptions but not its shape.
        """
        domain = self.context.domain(name)
        if not domain.has_interval:
            # Without an interval the symbol depends on the name, the kind of domain
            # and whether the name is declared; a loop looks its variables up often.
            key = (name, domain.kind, name in self.context.domains)
            found = _PLAIN_SYMBOLS.get(key)
            if found is None:
                found = self._symbol_in(name, domain)
                if len(_PLAIN_SYMBOLS) >= _PLAIN_SYMBOLS_KEPT:
                    _PLAIN_SYMBOLS.clear()
                _PLAIN_SYMBOLS[key] = found
            return found
        return self._symbol_in(name, domain)

    def _symbol_in(self, name: str, domain: Domain) -> sp.Basic:
        """The symbol for `name`, of domain `domain` (see `symbol`)."""
        value = self._degenerate(domain)
        if value is not None:
            return value
        if domain.kind.is_matrix and name in self.context.domains:
            return sp.MatrixSymbol(name, DIMENSION, DIMENSION)
        if domain.kind is DomainKind.SET and name in self.context.domains:
            return SetSymbol(name)
        if domain.kind is DomainKind.LOGICAL and name in self.context.domains:
            return LogicalSymbol(name)
        return sp.Symbol(name, **self.assumptions(domain))

    def assumptions(self, domain: Domain) -> dict[str, bool]:
        """The sympy assumptions implied by a domain declaration.

        Complex means "not known to be real": sympy leaves `real` undecided, so rewrites
        needing a real variable, like `SQRT(x^2)` -> `ABS(x)`, don't fire.
        """
        if domain.kind.is_matrix:
            return {"commutative": False}
        if domain.kind is DomainKind.COMPLEX:
            return {"complex": True}
        if not domain.kind.is_scalar:
            # A set or a truth value is no number.
            return {}
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
        """An arithmetic operator's operands, with logical ones wrapped in `Truth`, and
        a number `FACTOR` wrote as a product of primes the number again: with `u :=
        FACTOR(6)` in a program, `u + 1` is 7. (`FACTOR(6) + 1` itself is the command's
        answer in a sum, which keeps its form.)"""
        return self.operands_of(self._children(node))

    def operands_of(self, values: list[sp.Basic]) -> list[sp.Basic]:
        """The converted children `values` as an arithmetic operator's operands (see
        `_operands`)."""
        return [
            Truth(operand)
            if _truth_value(operand)
            else _factored_number_released(operand)
            for operand in values
        ]

    def _sum(self, node: Node) -> sp.Basic:
        """A sequence of terms, with one operator between each pair."""
        return self.summed(str(node.value), self._operands(node))

    def summed(self, operators: str, terms: list[sp.Basic]) -> sp.Basic:
        """The sum of the operands `terms`, `operators` the signs between them."""

        def signed(terms: list[sp.Basic]) -> sp.Basic:
            signs = [terms[0]]
            for index, term in enumerate(terms[1:]):
                signs.append(negated(term) if operators[index] == "-" else term)
            return added(signs)

        mapped = self.over_equations(signed, terms)
        return signed(terms) if mapped is None else mapped

    def _product(self, node: Node) -> sp.Basic:
        """A sequence of factors; see `multiplied`."""
        return self.multiplied_out(self._operands(node))

    def multiplied_out(self, factors: list[sp.Basic]) -> sp.Basic:
        """The product of the operands `factors`."""
        mapped = self.over_equations(multiplied, factors)
        return multiplied(factors) if mapped is None else mapped

    def _binop(self, node: Node) -> sp.Basic:
        return self.binary(str(node.value), self._operands(node))

    def binary(self, operator: str, operands: list[sp.Basic]) -> sp.Basic:
        """`/`, `^` or `.` applied to the values of its operands, to both sides of an
        equation among them."""

        def applied(sides: list[sp.Basic]) -> sp.Basic:
            return self._operated(operator, sides[0], sides[1])

        mapped = self.over_equations(applied, operands)
        return applied(operands) if mapped is None else mapped

    def _operated(self, operator: str, left: sp.Basic, right: sp.Basic) -> sp.Basic:
        match operator:
            case "/":
                return divided(left, right)
            case "^":
                return self.raised(left, right)
        return dotted(left, right)

    def raised(self, base: sp.Basic, exponent: sp.Basic) -> sp.Basic:
        """`base^exponent`, a root of a power taken whatever its base's sign under
        Branch Any (see `any_root`)."""
        if self.context.branch is Branch.ANY:
            root = any_root(base, exponent)
            if root is not None:
                return root
        if self.context.precision is Precision.APPROXIMATE:
            root = approximate_root(base, exponent)
            if root is not None:
                return root
        return power(base, exponent)

    def _unop(self, node: Node) -> sp.Basic:
        return self.prefixed(str(node.value), self._operands(node))

    def prefixed(self, operator: str, operands: list[sp.Basic]) -> sp.Basic:
        """The prefix operator `operator` applied to its operand, one of `operands`."""

        def applied(sides: list[sp.Basic]) -> sp.Basic:
            return self._prefixed(operator, sides[0])

        mapped = self.over_equations(applied, operands)
        return applied(operands) if mapped is None else mapped

    def _prefixed(self, operator: str, operand: sp.Basic) -> sp.Basic:
        match operator:
            case "-":
                return negated(operand)
            case "+-":
                # `±inf` is unsigned infinity, which is how the printer writes `zoo`.
                if operand in (sp.oo, -sp.oo):
                    return sp.zoo
                return PlusMinus(operand)
        return operand

    def _postop(self, node: Node) -> sp.Basic:
        """`u!`, `u%` and ``u` ``.

        Per section 8.5, the transpose of a scalar is the scalar, so ``x` `` is `x`
        unless `x` is declared nonscalar. A vector is transposed as `transposed` says;
        one without a transpose keeps the operator, as does anything else that isn't a
        scalar, such as an inert head or a relation.
        """
        operand = self.convert(node.children[0])
        match str(node.value):
            case "!":
                return self._factorial(operand)
            case "%":
                return operand / 100
            case "DEG":
                # `u DEG` is `u*deg`.
                return multiplied([operand, self.degree()])
        return self._transpose(operand)

    def _factorial(self, operand: sp.Basic) -> sp.Basic:
        """`u!`, of each element of a vector or set as for the functions of
        `ELEMENTWISE`."""
        if isinstance(operand, Vector):
            return Vector(*(self._factorial(element) for element in operand.args))
        if isinstance(operand, SetOf):
            elements = [self._factorial(element) for element in operand.args]
            (joined,) = self.sets_simplified([SetOf(*elements)])
            return joined
        if _explodes_factorial(operand):
            return Factorial(operand)
        return sp.factorial(operand)

    def _transpose(self, operand: sp.Basic) -> sp.Basic:
        """``operand` ``, given its value; see `_postop`. A set's is its complement, and
        a logical value's its negation."""
        if is_set(operand):
            return SetComplement(operand)
        if is_logical(operand):
            return connected("NOT", [operand])
        if isinstance(operand, Vector):
            flipped = transposed(operand)
            return Transposed(operand) if flipped is None else flipped
        if _symbolic_matrix(operand):
            return sp.Transpose(operand).doit()
        collapsed = _transposed_scalar(operand)
        return Transposed(operand) if collapsed is None else collapsed

    def _sub(self, node: Node) -> sp.Basic:
        """Element access on a vector, string or integer, a lookup in a set of pairs,
        or a subscripted variable otherwise (see `_element_at`).

        Elements may be vectors in turn, so `SUB` chains across dimensions:
        `q SUB 2 SUB 1 SUB 2` reads a number from a vector of matrices. A vector index
        selects: `m SUB [2, 3]` is the vector of rows 2 and 3. A base that is no
        sequence gives a subscripted name like `x SUB 1`.

        The index is evaluated before the base, as it is in an element assignment: `l
        SUB DIM(NEWL(APPEND(l, [u])))`, where `NEWL` assigns `l`, reads the element it
        appended.
        """
        base_node, index_node = node.children
        index = self.operand(self.convert(index_node))
        return self.element(node, self.operand(self.convert(base_node)), index)

    def element(self, node: Node, base: sp.Basic, index: sp.Basic) -> sp.Basic:
        """`node`, `base SUB index`, of the operands `base` and `index` (see `_sub`)."""
        if is_set(base):
            (base,) = self.sets_simplified([base])
        if isinstance(base, sp.And):
            # A conjunction, such as a system's solution, indexes its conjuncts in
            # printed order: `(x = 6 AND y = 9/2) SUB 1` is `x = 6`.
            operands = printed_operands(base, self.context.order)
            place = place_of(index, len(operands))
            if place is not None:
                return operands[place]
        element = digits_in(self.context, lambda: _element_at(base, index))
        if element is not None:
            return element
        if isinstance(base, Vector):
            # Not a usable index yet, so keep the access unevaluated.
            return self.opaque("ELEMENT", [base, index])
        return self._subscript(node.children[0], base, index)

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
        if " " in str(index):
            # A subscripted index keeps its parentheses: `x SUB (i SUB k)`.
            index = f"({index})"
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
        """The vector of the elements' values, whatever they are (see `vectors`)."""
        return self._listed(node, Vector, braces=False)

    def _set(self, node: Node) -> sp.Basic:
        """The set of the elements' values (see `sets`)."""
        return self._listed(node, SetOf, braces=True)

    def _listed(self, node: Node, head: type[sp.Function], braces: bool) -> sp.Basic:
        """A vector or set, whose elements may be the ends of an arithmetic sequence:
        `[0, 2, ..., 10]` is `[0, 2, 4, 6, 8, 10]`."""
        if not any(child.kind is Kind.ELLIPSIS for child in node.children):
            return head(*self._children(node))
        terms = [self.convert(c) for c in node.children if c.kind is not Kind.ELLIPSIS]
        return elided(terms, braces)

    def _set_operation(self, node: Node) -> sp.Basic:
        """A union, intersection or difference, with the elements compared simplified.

        A union or intersection of two logical values is their disjunction or
        conjunction: `true UNION false` is `true`.

        An operand is a set, so the transpose of a variable there is its complement:
        in ``a` INTERSECTION b` `` the variables are sets, where ``a` `` alone would be
        the transpose of a scalar, `a`.
        """
        left, right = self.sets_simplified(
            [self._set_operand(child) for child in node.children]
        )
        word = _LOGICAL_SET_OPERATIONS.get(str(node.value))
        if word is not None and is_logical(left) and is_logical(right):
            return connected(word, [left, right])
        match str(node.value):
            case "UNION":
                return SetUnion(left, right)
            case "INTERSECTION":
                return SetIntersection(left, right)
        return SetDifference(left, right)

    def _set_operand(self, node: Node) -> sp.Basic:
        """An operand of a set operation; see `_set_operation`."""
        if node.kind is Kind.POSTOP and str(node.value) == "`":
            operand = self.convert(node.children[0])
            if type(operand) is sp.Symbol:
                return SetComplement(operand)
            return self._transpose(operand)
        return self.convert(node)

    def _relation(self, node: Node) -> sp.Basic:
        """A relation, built unevaluated; conversion never decides one.

        A chain (left-nested in the grammar) means the conjunction of its links:
        `1 <= a <= b - 1` is `1 <= a AND a <= b - 1`, as the shipped library uses with
        `c <= a <= 0`. Read literally, it would compare a truth value with a number.

        An equation between two statements compares truth values: `(y = 2*x) = (y = 3
        - x)` is their equivalence, and with `/=` their exclusive or. Its right side is
        a statement only where parentheses make it one (relations nest to the left), or
        where both sides are values that are statements, such as the parameters of
        `IP(l1, l2) := SOLUTIONS(l1 = l2, [x, y])`.
        """
        left_node, right_node = node.children
        if (
            str(node.value) in COMPARING_TRUTHS
            and left_node.kind in STATEMENT_KINDS
            and right_node.kind in STATEMENT_KINDS
        ):
            sides = [self.side(left_node), self.side(right_node)]
            return connected(COMPARING_TRUTHS[str(node.value)], sides)
        links, _ = self._chain(node)
        if len(links) == 1:
            return links[0]
        try:
            return sp.And(*(_settled(link) for link in links))
        except Exception:
            return Logical(sp.Symbol("AND"), *links)

    def _chain(self, node: Node) -> tuple[list[sp.Basic], sp.Basic]:
        """The links of a relation chain, and the operand it ends on. A relation in
        parentheses is an operand, not a link: `(x > 1) = true` is no chain."""
        left_node, right_node = node.children
        relation = RELATIONS[str(node.value)]
        if left_node.kind is Kind.REL and left_node.start == node.start:
            links, previous = self._chain(left_node)
            right = self.side(right_node)
            return [*links, self._related(relation, previous, right)], right
        left = self.side(left_node)
        right = self.side(right_node)
        return [self.related(str(node.value), left, right)], right

    def related(self, word: str, left: sp.Basic, right: sp.Basic) -> sp.Basic:
        """The relation `word` between the values `left` and `right`: between two
        statements, an equation is their equivalence (see `_relation`), and so is one
        between a statement and a truth value: `(x > 1) = false` is `x <= 1`."""
        connective = COMPARING_TRUTHS.get(word)
        if connective is not None and (
            is_statement(left)
            and (is_statement(right) or isinstance(right, BooleanAtom))
            or isinstance(left, BooleanAtom)
            and is_statement(right)
        ):
            return connected(connective, [left, right])
        return self._related(RELATIONS[word], left, right)

    def side(self, node: Node) -> sp.Basic:
        """A side of a relation."""
        return self.sided(self.convert(node))

    def sided(self, value: sp.Basic) -> sp.Basic:
        """The converted side `value` of a relation as the relation takes it.
        Overridden by the evaluator."""
        return value

    def _related(self, relation, left: sp.Basic, right: sp.Basic) -> sp.Basic:
        try:
            return relation(left, right, evaluate=False)
        except Exception:
            return relation(left, right)

    def _logical(self, node: Node) -> sp.Basic:
        """Boolean on booleans, bitwise on integers."""
        return connected(LOGICAL_WORDS[node.kind], self._children(node))

    def _infix(self, node: Node) -> sp.Basic:
        """A word operator of Derive 6: rows, columns or a deeper level of a vector, a
        vector product, or the subset relation. Where it doesn't apply it stands."""
        word = str(node.value)
        left, right = (self.operand(child) for child in self._children(node))
        match word:
            case "ROW" | "COL":
                found = _rows(left, right, columns=word == "COL")
                if found is not None:
                    return found
                return (ColumnsOf if word == "COL" else RowsOf)(left, right)
            case "DOTPRODUCT":
                found = _dot_product(left, right)
                return DotProduct(left, right) if found is None else found
            case "CROSSPRODUCT":
                found = _cross_product(left, right)
                return CrossProduct(left, right) if found is None else found
            case "SUBSET":
                left, right = self.sets_simplified([left, right])
                found = _subset(left, right)
                return SubsetOf(left, right) if found is None else found
        levels = len(word.split()) - 1
        found = _deeper(left, right, levels)
        return Deeper(left, right, sp.Integer(levels)) if found is None else found

    # -- definitions, which convert their value and keep their shape --------

    def _assign(self, node: Node) -> sp.Basic:
        lhs = self.convert(node.children[0])
        rest = [self.convert(child) for child in node.children[1:]]
        return Assign(lhs, sp.Symbol(str(node.value)), *rest)

    def _fundef(self, node: Node) -> sp.Basic:
        """A definition, its body converted with the parameters standing for themselves;
        one that holds a program is kept as written (see `programmed`)."""
        head = node.children[0]
        params = formals(head, self._children(head))
        if len(node.children) > 1 and programmed(node.children[1]):
            body = as_written(node.children[1])
            return FunDef(sp.Symbol(str(node.value)), params, body)
        outer = self.parameters
        self.parameters = frozenset(parameter_names(head))
        try:
            body = [self.convert(child) for child in node.children[1:]]
        finally:
            self.parameters = outer
        return FunDef(sp.Symbol(str(node.value)), params, *body)

    def binds_parameter(
        self, variable: sp.Basic, others: Sequence[sp.Basic] = ()
    ) -> bool:
        """Whether a binding call's variable is a parameter of the definition at hand
        that its other arguments, `others`, may capture.

        Such a call stays a call in the definition: its variable is whatever a call of
        the function supplies, which another parameter may mention. `SB(u, v, a) :=
        ITERATE(u, v, a, 1)` isn't `u`, and `G(u, v) := DIF(u, v)` isn't 0. Where the
        other arguments mention no parameter but the bound ones, the call is worked out:
        `F(x) := DIF(x^3, x)` is `F(x) := 3*x^2`. The variable of a binding around the
        call may capture it too, being whatever that binding supplies: in `P(u, m, x)
        := SUM(VECTOR(LIM(u_, x, 1), u_, TERMS(u)))` the limit stays a call.
        """
        if not self.parameters:
            return False
        variables = list(variable.args) if isinstance(variable, Vector) else [variable]
        bound = {
            getattr(each, "name", "").split(" SUB ")[0] for each in variables
        } & self.parameters
        if not bound:
            return False
        capturing = self.capturing() - bound
        return any(
            not isinstance(symbol, StringLiteral)
            and getattr(symbol, "name", "").split(" SUB ")[0] in capturing
            for value in others
            if isinstance(value, sp.Basic)
            for symbol in value.free_symbols
        )

    def capturing(self) -> frozenset[str]:
        """The names whose values a call or an enclosing binding supplies later: the
        parameters of the definition at hand and the variables bound around here."""
        return self.parameters | self.held

    def mentions_parameters(self, args: Sequence[sp.Basic]) -> bool:
        """Whether an argument mentions a parameter of the definition at hand, or a
        variable of a binding call around it.

        A built-in that answers from the form of its argument then stays a call, since
        the argument is whatever a call or the binding supplies: `F(x) :=
        IF(NUMBER?(x), 1, 2)` isn't `F(x) := 2`.
        """
        held = self.parameters | self.held
        if not (held or self.holding_forms):
            return False
        return any(
            self.holding_forms
            or getattr(symbol, "name", "").split(" SUB ")[0] in held
            for value in args
            if isinstance(value, sp.Basic)
            for symbol in value.free_symbols
            if not isinstance(symbol, StringLiteral)
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
        if name not in _FACTORED_FORM:
            # A number `FACTOR` wrote as a product of primes is an argument as a
            # number: `MOD(FACTOR(6), 4)` is 2.
            args = [
                _factored_number_released(arg) if isinstance(arg, sp.Basic) else arg
                for arg in args
            ]
        if name in MAPPED_OVER_EQUATIONS:
            mapped = self.over_equations(
                lambda sides: self.call(name, sides), list(args)
            )
            if mapped is not None:
                return mapped
        handler = FUNCTIONS.get(name)
        if handler is None:
            return self.opaque(name, args)
        if name in ELEMENTWISE:
            spread = self._elementwise(name, list(args))
            if spread is not None:
                return spread
        if (
            name in BINDING_FUNCTIONS
            and len(args) > 1
            and self.binds_parameter(args[1], [args[0], *args[2:]])
        ):
            return self.opaque(name, args)
        if name in ON_FORM and self.mentions_parameters(args):
            return self.opaque(name, args)
        try:
            if _DIGIT_BASE.get() == 10 and self.context.setting_values.get(
                "OutputBase", 10
            ) in (10, "Decimal"):
                result = handler(self, list(args))
            else:
                result = digits_in(self.context, lambda: handler(self, list(args)))
        except Exception:
            result = None
        return self.opaque(name, args) if result is None else result

    def _elementwise(self, name: str, args: list[sp.Basic]) -> sp.Basic | None:
        """`name` applied to each element of its vector or set arguments (see
        `ELEMENTWISE`), None if no argument is one or the function takes a square
        matrix as a whole. Collections that don't pair up leave the call standing."""
        places = [i for i, value in enumerate(args) if isinstance(value, (Vector, SetOf))]
        if not places:
            return None
        if name in ("EXP", "SQRT") and len(args) == 1 and _square_matrix(args[0]):
            return None

        def at(place: int, element: sp.Basic) -> list[sp.Basic]:
            return [*args[:place], element, *args[place + 1 :]]

        if len(places) == 1:
            (place,) = places
            collection = args[place]
            results = [self.call(name, at(place, each)) for each in collection.args]
            if isinstance(collection, Vector):
                return Vector(*results)
            (joined,) = self.sets_simplified([SetOf(*results)])
            return joined
        collections = [args[place] for place in places]
        if not all(isinstance(value, Vector) for value in collections) or (
            len({len(value.args) for value in collections}) != 1
        ):
            return self.opaque(name, args)
        rows = []
        for index in range(len(collections[0].args)):
            row = list(args)
            for place in places:
                row[place] = args[place].args[index]
            rows.append(self.call(name, row))
        return Vector(*rows)

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

    # -- the angle mode, which says how an angle is shown ----------------------

    def degree(self) -> sp.Basic:
        """`deg`, one degree: kept as a constant under `Angle := Degree`, so that
        `30*deg` stays an angle in degrees, and `pi/180` under `Angle := Radian`."""
        if self.context.angle is Angle.DEGREE:
            return DEGREE
        return sp.pi / 180

    def angle_in(self, value: sp.Basic) -> sp.Basic:
        """A trigonometric argument in radians, as the functions take it in either
        angle mode: a degree is `pi/180`."""
        if isinstance(value, sp.Basic) and value.has(DEGREE):
            return value.xreplace({DEGREE: sp.pi / 180})
        return value

    def angle_out(self, value: sp.Basic) -> sp.Basic:
        """An angle an inverse trigonometric function gives, shown in degrees under
        `Angle := Degree` where it is a rational multiple of pi: `ASIN(1/2)` is
        `30*deg`. Its value is the same; any other angle is left as it is."""
        if self.context.angle is not Angle.DEGREE:
            return value
        return in_degrees(value)


def in_degrees(value: sp.Basic) -> sp.Basic:
    """`value` as a number of degrees, `(180*q)*deg`, if it is `q*pi` with a rational
    `q`, and otherwise `value`."""
    if not isinstance(value, sp.Expr):
        return value
    turns = value / sp.pi
    if not turns.is_Rational or turns == 0:
        return value
    return sp.Mul(180 * turns, DEGREE)


def _symbolic_product(left: sp.Basic, right: sp.Basic) -> sp.Basic:
    """`u . v` where at least one side is a symbolic matrix.

    Sympy's product handles most of section 8.8: it flattens nested products, so
    `(a . b) . c` and `a . (b . c)` are the same, and reverses products under
    transposition and inversion.

    Unlike Derive, sympy cancels a matrix against its inverse. The original returns
    `a . a^-1` unchanged, since an identity of unknown dimension can't be written. So
    where factors would cancel, the product is built unevaluated.

    An explicit matrix takes part as a sympy matrix. Anything else that is a vector,
    such as a flat one, has no place in a product of matrices, which sympy would treat
    as a scalar, so the operator stays.
    """
    operands = [
        matrix_of(side) if isinstance(side, Vector) else side for side in (left, right)
    ]
    if any(operand is None for operand in operands):
        return Dot(left, right)
    factors = [*_matrix_factors(operands[0]), *_matrix_factors(operands[1])]
    try:
        if _cancelling(factors):
            return sp.MatMul(*factors)
        return operands[0] * operands[1]
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


# -- vector arithmetic --------------------------------------------------------
#
# To sympy a `Vector` is an opaque scalar, so arithmetic on vectors is done here (see
# `vectors`). Each function returns a settled value, in which no sum, product or power
# has a vector operand, except one that has no value: an unconformable sum such as
# `[1, 2] + 2`, or a product or power that stays as written.


def added(terms: list[sp.Basic]) -> sp.Basic:
    """The sum of `terms`, vectors of one length added element by element.

    As in Derive, only the lengths matter, so the elements' sums follow the same rule:
    `[[1, 2], [3, 4]] + [1, 2]` is `[[1, 2] + 1, [3, 4] + 2]`. The scalars are added
    together too. A square matrix plus a scalar `c` is the matrix plus `c` times the
    identity, so a polynomial in a matrix has its constant term: `[[1, 2], [3, 4]] + 7`
    is `[[8, 2], [3, 11]]`, and `A - lambda` is `A - lambda*IDENTITY_MATRIX(n)`. Any
    other vector plus a scalar adds the scalar to each element, as Derive 6 does:
    `[49, 48] - 48` is `[1, 0]`. Vectors of different lengths have no sum, so a term is
    kept for each length and one for the scalars: `[1, 2] + [1, 2, 3]` stays. With more
    than one vector term, they are kept in order of first appearance, after the scalars.
    """
    if terms and all(isinstance(term, sp.Rational) for term in terms):
        # A loop's counter: sympy's sum of numbers is their sum.
        return functools.reduce(operator.add, terms)
    terms = _added_terms(terms)
    if not any(isinstance(term, Vector) for term in terms):
        return _scalar_sum(terms)
    scalars: list[sp.Basic] = []
    lengths: dict[int, list[Vector]] = {}
    for term in terms:
        if isinstance(term, Vector):
            lengths.setdefault(len(term.args), []).append(term)
        else:
            scalars.append(term)
    sums = [_summed(vectors) for vectors in lengths.values()]
    scalar = _scalar_sum(scalars)
    if len(sums) == 1:
        if scalar == 0:
            return sums[0]
        if _square_matrix(sums[0]) is None and _plain_scalar(scalar):
            return Vector(*(added([element, scalar]) for element in sums[0].args))
        shifted = _shifted_diagonal(sums[0], scalar)
        return sp.Add(scalar, sums[0]) if shifted is None else shifted
    return Added(*sums) if scalar == 0 else Added(scalar, *sums)


def _plain_scalar(value: sp.Basic) -> bool:
    """Whether `value` is a scalar expression with no vector, set or symbolic matrix in
    it."""
    return isinstance(value, sp.Expr) and not value.has(Vector, SetOf, sp.MatrixExpr)


def _shifted_diagonal(matrix: Vector, scalar: sp.Basic) -> Vector | None:
    """`matrix + scalar*I` for a square matrix of expressions and a certain scalar, or
    None if they aren't."""
    sympy_matrix = matrix_of(matrix)
    if sympy_matrix is None or not sympy_matrix.is_square or not _scalar(scalar):
        return None
    return Vector(
        *(
            Vector(
                *(
                    _scalar_sum([element, scalar]) if column == row else element
                    for column, element in enumerate(elements.args)
                )
            )
            for row, elements in enumerate(matrix.args)
        )
    )


def _scalar_sum(terms: list[sp.Basic]) -> sp.Basic:
    """The sum of terms none of which is a vector, such as two vectors' elements."""
    on_sides = _on_sides(_scalar_sum, terms)
    if on_sides is not None:
        return on_sides
    parts = [
        part
        for term in terms
        if isinstance(term, sp.Expr)
        for part in sp.Add.make_args(term)
    ]
    ambiguous = [part for part in parts if part.has(PlusMinus)]
    if any(sp.zoo in sp.Mul.make_args(part) for part in parts) and (
        sum(1 for part in parts if part.has(sp.zoo, sp.oo, -sp.oo)) > 1
    ):
        # `+-inf*u + inf*v` has no value for any u and v: where both are nonzero the
        # infinities may cancel, and where one is zero its product is undefined.
        return sp.nan
    if len(ambiguous) > 1:
        # Each `+-` is a sign of its own, so two of them may add or cancel:
        # `SIGN(0) + SIGN(0)` is any of -2, 0 and 2.
        return sp.nan
    total = sp.Add(*(_arithmetic(term) for term in terms))
    absorbed = _ABSORBED.get()
    infinities = (sp.oo, -sp.oo, sp.zoo)
    if absorbed is not None and total.has(*infinities):
        finite = sp.Add(*(part for part in parts if not part.has(*infinities)))
        absorbed.update(finite.free_symbols - total.free_symbols)
    return _floors_as_residues(total) if total.has(sp.floor) else total


#: The variables of the terms an infinity has absorbed in a sum, collected while the
#: argument of `VARIABLES` is evaluated (see `absorbed_by_infinity`), or None.
_ABSORBED: contextvars.ContextVar[set[sp.Basic] | None] = contextvars.ContextVar(
    "absorbed", default=None
)


@contextlib.contextmanager
def absorbed_by_infinity(collect: bool) -> Iterator[None]:
    """Within it, collect the variables an infinity absorbs in a sum, into a fresh set,
    or, unless `collect`, collect none.

    `VARIABLES` lists them with those of its argument's value: `a + inf` is `inf` for
    every `a`, but `VARIABLES(n + a)` with `n := inf` is `[a]`, as in Derive, whose
    time-value-of-money program finds the one unknown among its arguments, of which
    the number of periods may be `inf`, in the variables of their sum. The terms that
    cancel or vanish otherwise take their variables with them: `VARIABLES(0*a)` and
    `VARIABLES(a - a)` are `[]`.
    """
    token = _ABSORBED.set(set() if collect else None)
    try:
        yield
    finally:
        _ABSORBED.reset(token)


def _floors_as_residues(total: sp.Basic) -> sp.Basic:
    """`total` with the terms `u - v*FLOOR(u/v)` of a sum collected into `MOD(u, v)`:
    `y - 5*FLOOR(y/5)` is `MOD(y, 5)` and `x - FLOOR(x)` is `MOD(x)`."""
    if not isinstance(total, sp.Add):
        return total
    for term in total.args:
        floors = [f for f in sp.Mul.make_args(term) if isinstance(f, sp.floor)]
        if len(floors) != 1:
            continue
        (floor,) = floors
        modulus = -term / floor
        if modulus.has(sp.floor) or modulus.is_zero is not False:
            continue
        value = sp.expand(modulus * floor.args[0])
        others = sp.Add.make_args(total - term)
        if not set(sp.Add.make_args(value)) <= set(others):
            continue
        return _floors_as_residues(total - term - value + Modulo(value, modulus))
    return total


def _summed(vectors: list[Vector]) -> Vector:
    """The element-wise sum of vectors of one length."""
    if len(vectors) == 1:
        return vectors[0]
    columns = zip(*(vector.args for vector in vectors), strict=True)
    return Vector(*(added(list(column)) for column in columns))


def _added_terms(terms: list[sp.Basic]) -> list[sp.Basic]:
    """`terms` with the terms of any nested sum of a vector spliced in.

    That is an unconformable sum, a number times one, and a vector plus a scalar.
    """
    spliced = []
    for term in terms:
        if isinstance(term, Added) or _vectorial(term):
            spliced.extend(_added_terms(list(term.args)))
        elif (
            isinstance(term, sp.Mul)
            and len(term.args) == 2
            and term.args[0].is_Number
            and isinstance(term.args[1], Added)
        ):
            spliced.extend(
                _times(term.args[0], part) for part in _added_terms([term.args[1]])
            )
        else:
            spliced.append(term)
    return spliced


def _vectorial(value: sp.Basic) -> bool:
    """Whether `value` is the sum of a vector and a scalar, such as `[1, 2] + 2`."""
    return isinstance(value, sp.Add) and any(
        isinstance(term, Vector) for term in value.args
    )


def negated(value: sp.Basic) -> sp.Basic:
    """`-value`, element by element for a vector."""
    return _times(sp.S.NegativeOne, value)


def multiplied(factors: list[sp.Basic]) -> sp.Basic:
    """The product of `factors`.

    A scalar times a vector multiplies each element. Two vectors multiply as `u . v`
    (`dotted`), which is their dot product or the matrix product, so `*` and `.` are
    the same between vectors. The factors are taken from the left, keeping a product
    of matrices in order.

    A product containing a declared nonscalar is taken the same way, since section 8.4
    makes `a*b` the matrix product for matrices.

    A scalar times a set multiplies each element, and sets multiply as their Cartesian
    product.
    """
    if factors and all(isinstance(factor, sp.Rational) for factor in factors):
        # Sympy's product of numbers is their product.
        return functools.reduce(operator.mul, factors)
    if not any(
        isinstance(factor, (Vector, SetOf))
        or _vectorial(factor)
        or _symbolic_matrix(factor)
        for factor in factors
    ):
        try:
            return _ambiguous_product([_arithmetic(factor) for factor in factors])
        except Exception:
            pass
    result = factors[0]
    # The sets whose Cartesian product `result` is, so that a run of sets makes vectors
    # as long as the run: `{a}*{b}*{c}` is `{[a, b, c]}`.
    run: list[SetOf] = [result] if isinstance(result, SetOf) else []
    for factor in factors[1:]:
        if run and isinstance(factor, SetOf):
            run.append(factor)
            result = cartesian(run)
            continue
        result = _times(result, factor)
        run = [result] if isinstance(result, SetOf) else []
    return result


def _ambiguous_product(factors: list[sp.Basic]) -> sp.Basic:
    """The product of scalar factors, two or more `+-u` taken as one.

    Each `+-` is a sign of its own, so `SIGN(0)*SIGN(0)` is `+-1`, not the square
    `SIGN(0)^2`, which is 1.
    """
    parts = [part for factor in factors for part in sp.Mul.make_args(factor)]
    signs = [part for part in parts if isinstance(part, PlusMinus)]
    if len(signs) < 2:
        return sp.Mul(*factors)
    rest = [part for part in parts if not isinstance(part, PlusMinus)]
    return sp.Mul(*rest) * PlusMinus(sp.Mul(*(sign.args[0] for sign in signs)))


def _times(left: sp.Basic, right: sp.Basic) -> sp.Basic:
    """`left*right`, as `multiplied` describes; of two sets, the Cartesian product."""
    if isinstance(left, sp.Rational) and isinstance(right, sp.Rational):
        return left * right
    if _symbolic_matrix(left) or _symbolic_matrix(right):
        return dotted(left, right)
    if isinstance(left, Vector) and isinstance(right, Vector):
        return dotted(left, right)
    if isinstance(left, Vector):
        return mapped(left, lambda element: _times(element, right))
    if isinstance(right, Vector):
        return mapped(right, lambda element: _times(left, element))
    if isinstance(left, SetOf) and isinstance(right, SetOf):
        return cartesian([left, right])
    if isinstance(left, SetOf):
        return SetOf(*(_times(element, right) for element in left.args))
    if isinstance(right, SetOf):
        return SetOf(*(_times(left, element) for element in right.args))
    if _vectorial(left) or _vectorial(right):
        # A number distributes over the sum, as sympy distributes it over any sum;
        # anything else multiplies it as a whole.
        number, total = (left, right) if _vectorial(right) else (right, left)
        if number.is_Number:
            return added([_times(number, term) for term in total.args])
        return sp.Mul(left, right)
    on_sides = _on_sides(lambda sides: _times(*sides), [left, right])
    if on_sides is not None:
        return on_sides
    left, right = _arithmetic(left), _arithmetic(right)
    try:
        return left * right
    except Exception:
        return dotted(left, right)


def _on_sides(
    build: Callable[[list[sp.Basic]], sp.Basic], operands: list[sp.Basic]
) -> sp.Basic | None:
    """`build` applied to both sides of whichever `operands` are equations.

    The arithmetic of vector elements, which `Converter.over_equations` does for
    operands: `x*[y = 1]` is `[x*y = x]`. None if no operand is an equation.
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
    return sp.Eq(build(sides[0]), build(sides[1]), evaluate=False)


def _arithmetic(operand: sp.Basic) -> sp.Basic:
    """`operand` as an arithmetic operand: a truth value is wrapped in `Truth`."""
    return Truth(operand) if _truth_value(operand) else operand


def dotted(left: sp.Basic, right: sp.Basic) -> sp.Basic:
    """`u . v`: the dot product of vectors, of which the matrix product is a case.

    A vector of non-vectors times a vector of the same length is the sum of the
    products of their elements, `u1*v1 + u2*v2 + ...`: `[2, 3] . [4, 5]` is 23. The
    elements of `v` may be vectors, so a vector times a matrix combines its rows:
    `[2, 3] . [[a, b], [c, d]]` is `[2*a + 3*c, 2*b + 3*d]`. A vector of vectors, a
    matrix, gives the vector of its rows' products with `v`, so a matrix times a vector
    takes the vector as a column, `[[a, b], [c, d]] . [2, 3]` being
    `[2*a + 3*b, 2*c + 3*d]`, and the product of two matrices is the matrix product.
    The shapes decide everything: `[[1], [2]] . [[3, 4]]` is the outer product
    `[[3, 4], [6, 8]]`, and the one-row `[[1, 2]] . [3, 4]` is `[11]`.

    Lengths that don't match keep the operator unevaluated, as does a vector with some
    elements vectors and others not.

    Declared nonscalars are matrices with unknown elements, so their product is
    symbolic, which sympy rewrites by the rules of section 8.8.

    Two scalars multiply: anything not declared nonscalar is a scalar (section 8.5), so
    `a . b - b . a` is 0. Only operands that are certainly scalars count, see `_scalar`.
    """
    if _symbolic_matrix(left) or _symbolic_matrix(right):
        return _symbolic_product(left, right)
    if isinstance(left, Vector) and isinstance(right, Vector):
        product = _vector_product(left, right)
        if product is not None:
            return product
    if _scalar(left) and _scalar(right):
        return left * right
    return Dot(left, right)


def _scalar(value: sp.Basic) -> bool:
    """Whether `value` is certainly a scalar.

    It must be built of numbers and variables that aren't nonscalar (a variable under
    `default :epsilon Nonscalar` is noncommutative) by arithmetic and sympy's
    functions. A user function, an inert head such as `VECTOR(u, i, 1, n)` or an
    unevaluated operator such as `Dot` might stand for a vector, so it doesn't count.
    """
    if not isinstance(value, sp.Expr) or not value.is_commutative:
        return False
    for part in sp.preorder_traversal(value):
        if isinstance(part, StringLiteral):
            return False
        if isinstance(part, sp.Symbol | sp.Add | sp.Mul | sp.Pow):
            continue
        if part.is_Atom and part.is_number:
            continue
        if isinstance(part, sp.Function) and not isinstance(part, AppliedUndef):
            if type(part).__module__.startswith("sympy."):
                continue
        return False
    return True


def _vector_product(left: Vector, right: Vector) -> sp.Basic | None:
    """`left . right` as described in `dotted`, or None if the shapes don't conform."""
    rows = [isinstance(element, Vector) for element in left.args]
    if left.args and all(rows):
        products = [_vector_product(row, right) for row in left.args]
        if any(product is None for product in products):
            return None
        return Vector(*products)
    if any(rows) or len(left.args) != len(right.args):
        return None
    total = added(
        [_times(u, v) for u, v in zip(left.args, right.args, strict=True)]
    )
    return None if isinstance(total, Added) or _vectorial(total) else total


def divided(left: sp.Basic, right: sp.Basic) -> sp.Basic:
    """`u/v`.

    Dividing by a matrix means multiplying by its inverse, which only a square matrix
    has; another matrix keeps the quotient. Dividing by a flat vector is multiplying
    by its inverse `v/(v . v)` (see `power`), so `[0, 2]/[1, 0]` is the dot product 0,
    which LaplaceTransforms' shift of a power of `t` relies on. A vector divided by a
    scalar is divided element by element.
    """
    if _symbolic_matrix(right):
        return _times(left, _matrix_power(right, sp.Integer(-1)))
    if isinstance(right, Vector):
        matrix = matrix_of(right)
        flat = right.args and not any(isinstance(e, Vector) for e in right.args)
        if flat and not (isinstance(left, Vector) and is_matrix(left)):
            return _times(left, power(right, sp.Integer(-1)))
        if matrix is None or not matrix.is_square:
            return Quotient(left, right)
        return _times(left, power(right, sp.Integer(-1)))
    if isinstance(left, Vector) or _vectorial(left):
        return _times(left, power(right, sp.Integer(-1)))
    try:
        return left / right
    except Exception:
        return sp.Mul(left, sp.Pow(right, -1, evaluate=False))


def power(base: sp.Basic, exponent: sp.Basic) -> sp.Basic:
    """`u^v`, where for a square matrix that is the matrix power.

    `[[1, 1], [0, 1]]^n` is `[[1, n], [0, 1]]`, and `^-1` is the inverse. A singular
    matrix's inverse is kept in the head used for symbolic matrices, which prints as
    `u^-1`. Only square matrices have matrix powers.

    A vector's integer power is a power of its dot product with itself, times the
    vector for an odd exponent: `[1, 2]^2` is 5, `[1, 2]^3` is `[5, 10]` and
    `[1, 2]^-1` is `[1/5, 2/5]`. Any other vector's power stands: `[a, b]^k` is not
    `[a^k, b^k]`, which would disagree with the integer power at k = 2.

    A set's power to a positive integer is the Cartesian product of that many copies:
    `{1, 2}^2` is `{[1, 1], [1, 2], [2, 1], [2, 2]}`.
    """
    if _explodes(base, exponent):
        return Power(base, exponent)
    if (
        isinstance(exponent, sp.Mul)
        and sp.zoo in exponent.args
        and isinstance(base, sp.Expr)
        and base.is_number
        and base.is_positive
        and base != 1
    ):
        # `#e^(+-inf*u)` is 0 or infinity where u is nonzero, and undefined where it is
        # zero: it has no value for any u.
        return sp.nan
    if isinstance(base, SetOf):
        return _set_power(base, exponent)
    if isinstance(exponent, Vector) and isinstance(base, sp.Expr):
        if not isinstance(base, (Vector, sp.MatrixExpr)):
            return _scalar_to_vector(base, exponent)
    if _symbolic_matrix(base):
        return _matrix_power(base, exponent)
    if isinstance(base, Vector):
        if (
            isinstance(exponent, sp.Integer)
            and base.args
            and not any(isinstance(element, Vector) for element in base.args)
        ):
            square = dotted(base, base)
            if exponent % 2 == 0:
                return power(square, exponent // 2)
            return multiplied([power(square, (exponent - 1) // 2), base])
        raised = _integer_matrix_power(base, exponent)
        if raised is not None:
            return raised
        matrix = matrix_of(base)
        if matrix is None or not matrix.is_square:
            return Power(base, exponent)
        try:
            raised = matrix**exponent
        except Exception:
            return sp.MatPow(matrix, exponent)
        return from_matrix(raised) if isinstance(raised, sp.MatrixBase) else raised
    try:
        return base**exponent
    except Exception:
        return sp.Pow(base, exponent, evaluate=False)


def _integer_matrix_power(base: Vector, exponent: sp.Basic) -> Vector | None:
    """The power of a square matrix of integers to a nonnegative integer, by repeated
    squaring over machine integers, or None for any other power: a program raising
    `[[0, 1], [1, 1]]` to the nth power for Fibonacci numbers does it in a loop, and a
    sympy matrix costs dozens of times as much."""
    if not isinstance(exponent, sp.Integer) or exponent < 0:
        return None
    size = len(base.args)
    rows = []
    for row in base.args:
        if not isinstance(row, Vector) or len(row.args) != size:
            return None
        if not all(isinstance(element, sp.Integer) for element in row.args):
            return None
        rows.append([int(element) for element in row.args])
    if not size:
        return None
    raised = [[int(i == j) for j in range(size)] for i in range(size)]
    count = int(exponent)
    while count:
        if count & 1:
            raised = _integer_product(raised, rows)
        count >>= 1
        if count:
            rows = _integer_product(rows, rows)
    return Vector(*(Vector(*(sp.Integer(element) for element in row)) for row in raised))


def _integer_product(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    """The product of two square matrices of integers."""
    columns = list(zip(*right, strict=True))
    return [[sum(map(operator.mul, row, column)) for column in columns] for row in left]


def _scalar_to_vector(base: sp.Expr, exponent: Vector) -> sp.Basic:
    """`c^v` for a scalar `c`: for a square matrix `v`, the matrix exponential of
    `LN(c)*v` (standing if it has no closed form), otherwise the power to each element:
    `x^[0, 1, 2]` is `[1, x, x^2]`."""
    matrix = _square_matrix(exponent)
    if matrix is None:
        return Vector(*(power(base, element) for element in exponent.args))
    try:
        exponential = (_ln(base) * matrix).exp()
    except Exception:
        return Power(base, exponent)
    if exponential.has(sp.I) and not matrix.has(sp.I):
        exponential = exponential.applyfunc(lambda entry: entry.rewrite(sp.cos))
    return from_matrix(exponential)


def _set_power(base: SetOf, exponent: sp.Basic) -> sp.Basic:
    if (
        isinstance(exponent, sp.Integer)
        and exponent > 0
        and len(base.args) ** int(exponent) <= MOST_TERMS
    ):
        return cartesian([base] * int(exponent))
    return Power(base, exponent)


def any_root(base: sp.Basic, exponent: sp.Basic) -> sp.Basic | None:
    """A fractional power of a product of powers, each power's exponent multiplied
    in, as Branch Any takes it: `SQRT(4/SIN(x)^2)` is `2/SIN(x)`, where sympy, for real
    `x`, writes `2/ABS(SIN(x))`. None where that does not apply."""
    if not (
        isinstance(base, sp.Expr)
        and not isinstance(base, Vector)
        and isinstance(exponent, sp.Rational)
        and not exponent.is_Integer
        and base.free_symbols
    ):
        return None
    factors = []
    changed = False
    for factor in sp.Mul.make_args(base):
        inner, power_of = factor.as_base_exp()
        if power_of.is_Integer and power_of != 1 and inner.free_symbols:
            factors.append(inner ** (power_of * exponent))
            changed = True
        else:
            factors.append(factor**exponent)
    if not changed:
        return None
    try:
        return sp.Mul(*factors)
    except Exception:
        return None


def settled(value: sp.Basic) -> sp.Basic:
    """`value` with any vector arithmetic left undone by substitution done.

    Substituting a vector for a variable leaves it an operand of sympy's sum, product
    or power, which treats it as a scalar: `(x*y).subs(x, [1, 2])` is a product of `y`
    and a vector rather than `[y, 2*y]`. This redoes such operations as the converter
    does, innermost first. A product of two vectors may have been reordered by sympy
    already, which matters for matrices; evaluation avoids the question by
    evaluating a body at each value instead of substituting into it.
    """
    if not isinstance(value, sp.Basic) or not value.has(Vector, SetOf):
        return value
    args = [settled(argument) for argument in value.args]
    if isinstance(value, sp.Add):
        return added(args)
    if isinstance(value, sp.Mul):
        return multiplied(args)
    if isinstance(value, sp.Pow):
        return power(*args)
    if args == list(value.args):
        return value
    try:
        if isinstance(value, Relational):
            return value.func(*args, evaluate=False)
        return value.func(*args)
    except Exception:
        return value


def _matrix_factors(value: sp.Basic) -> list[sp.Basic]:
    """The factors of a matrix product, for flattening."""
    return list(value.args) if isinstance(value, sp.MatMul) else [value]


def _symbolic_matrix(value: sp.Basic) -> bool:
    """Whether `value` is a symbolic matrix rather than one with explicit elements.

    Symbolic ones are declared nonscalars and expressions built from them. An explicit
    matrix is a `Vector`, but becomes a sympy `MatrixBase` (when immutable, also a
    `MatrixExpr`) inside a product with a symbolic one.
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
    if not (isinstance(exponent, sp.Rational) and isinstance(base, sp.Rational)):
        return False
    if base.is_zero:
        return False
    bits = max(abs(base.p).bit_length(), abs(base.q).bit_length())
    if isinstance(exponent, sp.Integer):
        return bits * abs(int(exponent)) > POWER_BITS
    # Sympy takes a root of a rational number by raising its prime factors to the
    # exponent's numerator modulo its denominator: with a denominator of 40 digits,
    # `2.13^((1/0.3856)^3)` would never be built.
    return bits * max(exponent.q, abs(exponent.p) // exponent.q) > POWER_BITS


def _explodes_factorial(operand: sp.Basic) -> bool:
    """Whether `operand!` is a number too big to build, by the measure of `_explodes`:
    `(10!)!` would be 22 million digits."""
    if not (isinstance(operand, sp.Integer) and operand > 1):
        return False
    return math.lgamma(int(operand) + 1) / math.log(2) > POWER_BITS


def _nameable(value: sp.Basic) -> bool:
    """Whether a subscript index can be part of a symbol's name.

    Names and nonnegative numerals can; a negative numeral can't, since the grammar
    doesn't read back `x SUB -1`.
    """
    if type(value) is sp.Symbol:
        return True
    return isinstance(value, sp.Integer) and value >= 0


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


def _extremum(func: Callable[..., sp.Basic]) -> Handler:
    """`MAX` or `MIN`, of its arguments or of the elements of a single vector or set.

    `MAX([3, 1, 2])` is 3, and a matrix gives the extremum of each row:
    `MAX([[1, 2], [3, 4]])` is `[2, 4]`. Of no elements there is none.
    """
    plain = _direct(func)

    def handler(conv: Converter, args: list) -> sp.Basic:
        if len(args) == 1 and isinstance(args[0], (Vector, SetOf)):
            elements = args[0].args
            if not elements:
                raise ValueError("no elements")
            if isinstance(args[0], Vector) and all(
                isinstance(element, Vector) for element in elements
            ):
                return Vector(*(handler(conv, [element]) for element in elements))
            if not any(isinstance(element, (Vector, SetOf)) for element in elements):
                return extremum(conv, list(elements))
        return extremum(conv, args)

    def extremum(conv: Converter, args: list) -> sp.Basic:
        # `+-inf` is either infinity, which one unknown, so an extremum with it has no
        # single value: `MIN(x, +-inf)` is `x` or `-inf`.
        if any(arg is sp.zoo or arg is sp.nan for arg in args):
            return sp.nan
        return plain(conv, _undominated(args, func is sp.Min))

    def rational(conv: Converter, args: list) -> sp.Basic:
        # Rational numbers are compared exactly as they are: sympy's own extremum of a
        # long list of them costs a thousand times more.
        whole = len(args) == 1 and isinstance(args[0], (Vector, SetOf))
        numbers = args[0].args if whole else args
        if numbers and all(isinstance(number, sp.Rational) for number in numbers):
            return pick(numbers, key=_exactly)
        return handler(conv, args)

    pick = max if func is sp.Max else min
    return rational


def _undominated(args: list, minimum: bool) -> list:
    """The arguments of `MIN` (`minimum`) or `MAX` without those that never decide it,
    where all of them are affine in the same real variables with rational coefficients.

    An argument equal to a convex combination of the others plus a constant `d` is at
    least their minimum where `d >= 0`, and at most their maximum where `d <= 0`, for
    every value of the variables: then it is dropped. `MIN(1 - x - y, (19 - x -
    7*y)/13, (5*x + y + 7)/3)` is `MIN(1 - x - y, (5*x + y + 7)/3)`, the second being
    17/26 of the first and 9/26 of the third.
    """
    if len(args) < 3 or len(args) > _MOST_PRUNED:
        return args
    symbols = set()
    for arg in args:
        if not isinstance(arg, sp.Expr) or arg.has(Vector, SetOf):
            return args
        symbols |= arg.free_symbols
    if not symbols or any(symbol.is_real is False for symbol in symbols):
        return args
    variables = sorted(symbols, key=sp.default_sort_key)
    rows = []
    for arg in args:
        try:
            polynomial = sp.Poly(arg, *variables)
        except Exception:
            return args
        if polynomial.total_degree() > 1 or not all(
            coefficient.is_Rational for coefficient in polynomial.coeffs()
        ):
            return args
        rows.append(
            [polynomial.coeff_monomial(variable) for variable in variables]
            + [polynomial.coeff_monomial(1)]
        )
    kept = list(range(len(args)))
    changed = True
    while changed and len(kept) > 2:
        changed = False
        for index in kept:
            others = [other for other in kept if other != index]
            weights = sp.symbols(f"w0:{len(others)}")
            equations = [
                sum(w * rows[o][place] for w, o in zip(weights, others, strict=True))
                - rows[index][place]
                for place in range(len(variables))
            ] + [sum(weights) - 1]
            solved = sp.linsolve(equations, weights)
            if not isinstance(solved, sp.FiniteSet) or len(solved) != 1:
                continue
            (values,) = solved.args
            if any(not value.is_Rational or value < 0 for value in values):
                continue
            offset = rows[index][-1] - sum(
                value * rows[o][-1] for value, o in zip(values, others, strict=True)
            )
            if (offset >= 0) if minimum else (offset <= 0):
                kept.remove(index)
                changed = True
                break
    return [args[index] for index in kept]


#: The most arguments of `MIN` or `MAX` that `_undominated` looks at.
_MOST_PRUNED = 8


def _exactly(number: sp.Rational) -> int | Fraction:
    """A rational number as Python compares it fastest, exactly."""
    if isinstance(number, sp.Integer):
        return int(number)
    return Fraction(int(number.p), int(number.q))

    return handler


def _exp(conv: Converter, args: list) -> sp.Basic:
    """`EXP(u)`, which is `#e^u`, so a square matrix's is its matrix exponential."""
    argument = _one(args)
    if isinstance(argument, Vector):
        return power(sp.E, argument)
    return _direct(sp.exp)(conv, args)


def _exponential_integral(conv: Converter, args: list) -> sp.Basic:
    """`EI(x)`, and `EI(x, m)`, Derive's partial sum of its series, `euler_gamma +
    LN(x) + SUM(x^n/(n*n!), n, 1, m)`.

    The series converges to `EI(x)` for a positive `x`, and through `LN(x)` to
    `EI(x) + pi*#i` for a negative one, which `m = inf` gives where the sign is known.
    """
    if len(args) == 1:
        return _direct(sp.Ei)(conv, args)
    value, count = args
    if count == sp.oo and value.is_positive:
        return sp.Ei(value)
    if count == sp.oo and value.is_negative:
        return sp.Ei(value) + sp.pi * sp.I
    index = _summation_index(args)
    summand = value**index / (index * sp.factorial(index))
    return sp.EulerGamma + _ln(value) + conv.call("SUM", [summand, index, 1, count])


def _polylogarithm(conv: Converter, args: list) -> sp.Basic:
    """`POLYLOG(n, z)`, and `POLYLOG(n, z, m)`, Derive's partial sum of its series,
    `SUM(z^k/k^n, k, 1, m)`, which for `m = inf` is `POLYLOG(n, z)` where it is known to
    converge: for `|z| < 1`, and for `|z| = 1` when `n > 1`."""
    if len(args) == 2:
        return _direct(sp.polylog)(conv, args)
    order, value, count = args
    if count == sp.oo and value.is_number:
        size = sp.Abs(value)
        if (size < 1) is sp.true or ((size - 1).is_zero and (order > 1) is sp.true):
            return sp.polylog(order, value)
    index = _summation_index(args)
    return conv.call("SUM", [value**index / index**order, index, 1, count])


def _polygamma(conv: Converter, args: list) -> sp.Basic:
    """`POLYGAMMA(n, z)`, the `n`th derivative of `DIGAMMA(z)`; of order -1 it is
    `LN(GAMMA(z))`, which `DIGAMMA` is the derivative of, as Derive defines it."""
    if len(args) == 2 and args[0] == -1:
        return sp.log(sp.gamma(args[1]))
    return _direct(sp.polygamma)(conv, args)


def _summation_index(args: list) -> sp.Symbol:
    """A summation variable that none of `args` mentions: `n`, else `k`, else `j`."""
    names = {symbol.name for arg in args for symbol in arg.free_symbols}
    for name in ("n", "k", "j", "i", "n_"):
        if name not in names:
            return sp.Symbol(name)
    return sp.Dummy("n")


def _sqrt(conv: Converter, args: list) -> sp.Basic:
    """`SQRT(u)`, taken whatever the sign of a power in `u` under Branch Any.

    A vector's root is its power to 1/2, so a square matrix has its matrix root.
    """
    argument = _one(args)
    if isinstance(argument, Vector):
        return power(argument, sp.S.Half)
    if conv.context.branch is Branch.ANY:
        root = any_root(argument, sp.S.Half)
        if root is not None:
            return root
    if conv.context.precision is Precision.APPROXIMATE:
        root = approximate_root(argument, sp.S.Half)
        if root is not None:
            return root
    if isinstance(argument, sp.Integer) and argument.p >= 0:
        return integer_root(argument.p)
    return _direct(sp.sqrt)(conv, args)


def integer_root(number: int) -> sp.Basic:
    """The square root of the integer `number`, at least zero, as sympy writes it: an
    integer, or the square factors it finds taken out of the root, `SQRT(12)` being
    `2*SQRT(3)`.

    Sympy finds them as this does, but asks the assumptions of each new integer it
    makes on the way, and a program taking the roots of many integers spends most of
    its time there. Here they are found on Python integers: the base of a perfect
    power, or else the factors that trial division up to 2^15 finds, a larger cofactor
    left as it is. A root that still has square factors is taken again.
    """
    outside = 1
    while True:
        root = math.isqrt(number)
        if root * root == number:
            return sp.Integer(outside * root)
        power_of = perfect_power(number)
        factors = (
            {int(power_of[0]): int(power_of[1])}
            if power_of is not False
            else factorint(number, limit=2**15, use_rho=False, use_pm1=False)
        )
        taken, inside = 1, 1
        for factor, exponent in factors.items():
            half, odd = divmod(exponent, 2)
            taken *= factor**half
            if odd:
                inside *= factor
        if taken == 1:
            break
        outside *= taken
        number = inside
    radical = sp.Pow(sp.Integer(number), sp.S.Half, evaluate=False)
    if outside == 1:
        return radical
    return sp.Mul(sp.Integer(outside), radical, evaluate=False)


def approximate_root(base: sp.Basic, exponent: sp.Basic) -> sp.Basic | None:
    """Under Approximate, a fractional power of a positive rational with long terms,
    left unevaluated where it is irrational, or None for any other power.

    Evaluated, sympy takes the square factors out of the root, which means factoring
    the numerator and denominator by trial division: a loop that takes `ABS` of a
    complex iterate, or a root of a carried value at 40 digits, spends most of its time
    there. The value is the same, and the step rounds it anyway, so it is kept as
    written; a rational root is still found, by integer roots.
    """
    if (
        not isinstance(base, sp.Rational)
        or not isinstance(exponent, sp.Rational)
        or exponent.q == 1
        or base <= 0
    ):
        return None
    numerator, denominator = int(base.p), int(base.q)
    if numerator.bit_length() + denominator.bit_length() <= _SHORT_BITS:
        return None
    top, exact_top = sp.integer_nthroot(numerator, int(exponent.q))
    bottom, exact_bottom = sp.integer_nthroot(denominator, int(exponent.q))
    if exact_top and exact_bottom:
        return sp.Rational(top, bottom) ** exponent.p
    return sp.Pow(base, exponent, evaluate=False)


#: The bits of a rational's numerator and denominator together up to which
#: `approximate_root` leaves its roots to sympy, which factors such numbers at once.
_SHORT_BITS = 64


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
    "ASEC": sp.asec,
    "ACSC": sp.acsc,
}

def _zeta(power: sp.Basic, shift: sp.Basic = sp.S.One) -> sp.Basic:
    """`ZETA(s)`, Riemann's zeta function, and `ZETA(s, z)`, Hurwitz's: the sum of
    `1/(k + z)^s` for k from 0.

    Besides sympy's closed forms, a shift by a half-integer is Riemann's function less
    finitely many terms: `ZETA(s, 1/2)` is `(2^s - 1)*ZETA(s)`, so `ZETA(2, 1/2)` is
    `pi^2/2`.
    """
    found = sp.zeta(power, shift)
    if not isinstance(found, sp.zeta) or len(found.args) != 2:
        return found
    half = shift - sp.S.Half
    if not (half.is_Integer and half >= 0) or power == 1:
        return found
    if half and not power.is_Integer:
        return found
    terms = sum(1 / (k + sp.S.Half) ** power for k in range(int(half)))
    return (2**power - 1) * sp.zeta(power) - terms


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
    "ZETA": lambda *args: _zeta(*args),
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

    A vector's sign is the unit vector in its direction, the vector divided by its
    magnitude (`ABS`): `SIGN([3, 4])` is `[3/5, 4/5]`. The zero vector has no
    direction, and its sign stands.
    """
    value = _one(args)
    if isinstance(value, Vector):
        magnitude = _absolute(conv, [value])
        if magnitude.is_zero:
            return conv.opaque("SIGN", [value])
        return divided(value, magnitude)
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

    `CHI(a, x, b, c)` takes the value `c` at `a` and `1 - c` at `b`, and
    `CHI(a, x, b, c, d)` the values `c` and `d` there. Elsewhere either is
    `CHI(a, x, b)`, but only where `x` is known to differ from both ends: otherwise the
    value at an end could be lost, so the call stands, with both end values.
    """
    if len(args) == 1:
        low, value, high = sp.Integer(0), args[0], sp.Integer(1)
    elif len(args) == 2:
        low, value, high = args[0], args[1], sp.Integer(1)
    elif len(args) == 3:
        low, value, high = args
    else:
        low, value, high, at_low, *rest = args
        if len(rest) > 1:
            raise TypeError("too many arguments")
        at_high = rest[0] if rest else 1 - at_low
        for end, at_end in ((low, at_low), (high, at_high)):
            meets = sp.expand(value - end).is_zero
            if meets is None:
                # The call stands with both end values written out, as Derive writes
                # it: `CHI(a, x, b, 1)` is `CHI(a, x, b, 1, 0)`.
                return conv.opaque("CHI", [low, value, high, at_low, at_high])
            if meets:
                return at_end
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


def _interest_free(rate: sp.Basic) -> bool:
    """Whether the rate is zero, where the annuity equation is its limit
    `v + n*p + f = 0`: the formulas for a nonzero rate divide by it."""
    return isinstance(rate, sp.Expr) and rate.is_zero is True


def _present_value(conv: Converter, args: list) -> sp.Basic:
    """`PVAL(i, n, p, f, t)`: the contract's present value."""
    rate, periods, payment, future, when = _financial(args, 5)
    if _interest_free(rate):
        return -(periods * payment + future)
    due = payment * (rate * when + 1)
    return ((due - future * rate) / (1 + rate) ** periods - due) / rate


def _future_value(conv: Converter, args: list) -> sp.Basic:
    """`FVAL(i, n, p, v, t)`: the contract's future value."""
    rate, periods, payment, present, when = _financial(args, 5)
    if _interest_free(rate):
        return -(present + periods * payment)
    due = payment * (rate * when + 1)
    grown = (1 + rate) ** periods * (rate * (payment * when + present) + payment)
    return (due - grown) / rate


def _payment(conv: Converter, args: list) -> sp.Basic:
    """`PMT(i, n, v, f, t)`: the payment per period."""
    rate, periods, present, future, when = _financial(args, 5)
    if _interest_free(rate):
        return -(present + future) / periods
    growth = (1 + rate) ** periods
    return rate * (present * growth + future) / ((1 - growth) * (rate * when + 1))


def _periods(conv: Converter, args: list) -> sp.Basic:
    """`NPER(i, p, v, f, t)`: the number of periods; with neither interest nor payments,
    no number of them is determined."""
    rate, payment, present, future, when = _financial(args, 5)
    if _interest_free(rate):
        if isinstance(payment, sp.Expr) and payment.is_zero is True:
            return sp.nan
        return -(present + future) / payment
    due = payment * (rate * when + 1)
    return sp.log(
        (due - future * rate) / (rate * (payment * when + present) + payment)
    ) / sp.log(1 + rate)


#: Derive's generator: a 32-bit linear congruential state, stepped as
#: `s <- (_MULTIPLIER*s + 1) mod _MODULUS`.
_MULTIPLIER = 2654435721
_MODULUS = 2**32


def _drawn(context: Context) -> int:
    """Step the session's random state once (see `Context.random_state`) and return
    the new state."""
    store = context.store
    state = context.random_state if store.random is None else store.random
    store.random = (_MULTIPLIER * state + 1) % _MODULUS
    store.version += 1
    return store.random


def _seeded(context: Context, seed: int) -> None:
    """Make `seed` the random state, reduced to 32 bits."""
    context.store.random = seed % _MODULUS
    context.store.version += 1


def _fraction_drawn(context: Context) -> sp.Rational:
    """A number in [0, 1): two steps of the state, as the high and low halves of a
    64-bit fraction."""
    high = _drawn(context)
    low = _drawn(context)
    drawn_fraction()
    return sp.Rational(high * _MODULUS + low, _MODULUS**2)


def _random(conv: Converter, args: list) -> sp.Basic:
    """`RANDOM(n)`: a pseudo-random number from Derive's generator, or a new seed.

    For an integer n above 1, `FLOOR(n*f)`, an integer in [0, n), where f is a fraction
    in [0, 1) whose 32-bit digits are successive states, the first the most significant:
    one step for n up to 2^32, and for a larger n as many steps as its outcomes need 32-bit
    digits (five for 10^40), so that a large n has all its digits drawn. For 1, a number
    in [0, 1) from two steps (see `_fraction_drawn`); for a positive non-integer rational
    n, n times that. A negative integer n makes `-n` the
    state and returns it; 0 seeds with the centiseconds since the start of the current
    year, and returns them. Anything else stays as written and draws nothing.

    The state is the session's (see `Context.random_state`) and the command carries its
    final value back (`Store.random`), so recorded draws after a seed reproduce. There
    is one draw per evaluation of the call: the evaluator evaluates the body of
    `VECTOR(RANDOM(6), n, 1, 100)` once per value, so it throws a hundred times. A
    draw counts as a change of the command's state (see `Store.version`), so that a
    variable whose value draws is not taken as having the value it had before.
    """
    number = _one(args)
    context = conv.context
    if number.is_Integer:
        outcomes = int(number)
        if outcomes > 1:
            steps = max(1, -(-(outcomes - 1).bit_length() // 32))
            digits = 0
            for _ in range(steps):
                digits = digits * _MODULUS + _drawn(context)
            return sp.Integer(digits * outcomes // _MODULUS**steps)
        if outcomes == 1:
            return _fraction_drawn(context)
        if outcomes < 0:
            _seeded(context, -outcomes)
            return sp.Integer(-outcomes)
        now = datetime.datetime.now()
        start = datetime.datetime(now.year, 1, 1)
        centiseconds = int((now - start).total_seconds() * 100)
        _seeded(context, centiseconds)
        return sp.Integer(centiseconds)
    if number.is_Rational and number > 0:
        return number * _fraction_drawn(context)
    raise TypeError("not a number of outcomes")


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
    """`ATAN(z)`, and `ATAN(y, x)`, the angle of the point (x, y).

    Where sympy keeps the arctangent of a real radical, the angle may still be a
    rational multiple of pi: see `_angle_of_tangent`.
    """
    if len(args) == 2:
        return conv.angle_out(sp.atan2(args[0], args[1]))
    value = _one(args)
    if value is sp.zoo:
        # `+-inf`, infinity of either sign, as a real division by zero gives it: its
        # angle is `pi/2` of that sign, not just some angle between.
        return conv.angle_out(PlusMinus(sp.pi / 2))
    return conv.angle_out(_arctangent(value))


#: The largest denominator of `r` that `_angle_of_tangent` tries in `r*pi`.
_ANGLE_DENOMINATOR = 60


def _angle_of_tangent(value: sp.Basic) -> sp.Basic | None:
    """The angle `r*pi` in (-pi/2, pi/2) whose tangent is `value`, for a real radical
    `value` and a rational `r` of small denominator; None if there is none.

    `ATAN(SQRT(6) - SQRT(3) + SQRT(2) - 2)` is `pi/24`. The candidate `r` is read off
    the numeric arctangent, and is the answer only if `TAN(r*pi) - value` is proved
    zero exactly: its minimal polynomial is `x`.
    """
    if not (
        isinstance(value, sp.Expr)
        and value.is_number
        and not value.is_Rational
        and value.is_real
        and value.has(sp.Pow)
        and all(power.exp.is_Rational for power in value.atoms(sp.Pow))
        and not value.atoms(sp.Function)
        and sp.count_ops(value) <= 40
    ):
        return None
    try:
        turns = sp.N(sp.atan(value) / sp.pi, 40)
        ratio = Fraction(str(turns)).limit_denominator(_ANGLE_DENOMINATOR)
        if abs(turns - sp.Rational(ratio.numerator, ratio.denominator)) > 1e-30:
            return None
        angle = sp.pi * sp.Rational(ratio.numerator, ratio.denominator)
        unknown = sp.Dummy()
        if sp.minimal_polynomial(sp.tan(angle) - value, unknown) != unknown:
            return None
    except Exception:
        return None
    return angle


def _acot(conv: Converter, args: list) -> sp.Basic:
    """`ACOT(z)` is `pi/2 - ATAN(z)`, and `ACOT(x, y)` is `ATAN(y, x)`, per section 6.4.

    That makes `ACOT(x)` run from `pi` down to 0 over the reals, continuous at zero.
    Sympy's `acot` is `ATAN(1/z)` instead, an odd function with values in
    `(-pi/2, pi/2]`, so it isn't used.

    The two arguments are the legs of the triangle rather than their ratio, which
    preserves the quadrant.
    """
    if len(args) == 2:
        return conv.angle_out(sp.atan2(args[1], args[0]))
    return conv.angle_out(sp.pi / 2 - sp.atan(_one(args)))


class DegreesBase(sp.Function):
    """An inverse trigonometric function valued in degrees: `ARCSIN(z)` is
    `180*ASIN(z)/pi`, whatever the Angle setting.

    Where the inverse in radians has a value free of inverse trigonometric functions
    (`ASIN(1/2)` is `pi/6`), the call is that value in degrees (30). Otherwise it
    stands as this head, `ARCCOS(-SQRT(6)/3)`, and approximates to degrees.
    """

    @classmethod
    def radians(cls, *args: sp.Basic) -> sp.Basic:
        """The angle in radians, as the radian inverse gives it."""
        raise NotImplementedError

    @classmethod
    def eval(cls, *args: sp.Basic) -> sp.Basic | None:
        try:
            angle = cls.radians(*args)
        except Exception:
            return None
        if angle.has(*_INVERSE_ANGLES):
            return None
        return 180 * angle / sp.pi

    def degrees(self) -> sp.Basic:
        """The definition, `180/pi` times the radian inverse."""
        return 180 * type(self).radians(*self.args) / sp.pi

    def _eval_evalf(self, prec: int) -> sp.Basic:
        return self.degrees()._eval_evalf(prec)

    def fdiff(self, argindex: int = 1) -> sp.Basic:
        """The derivative of the definition, by the chain rule."""
        point = sp.Dummy()
        args = list(self.args)
        args[argindex - 1] = point
        derivative = sp.diff(180 * type(self).radians(*args) / sp.pi, point)
        return derivative.subs(point, self.args[argindex - 1])


#: The radian inverses whose presence keeps a degree-valued call standing.
_INVERSE_ANGLES = (sp.asin, sp.acos, sp.atan, sp.acot, sp.asec, sp.acsc, sp.atan2)


def _arctangent(value: sp.Basic) -> sp.Basic:
    """`ATAN(z)` in radians, recognizing the tangent of a rational angle (see
    `_angle_of_tangent`)."""
    angle = sp.atan(value)
    if angle.has(sp.atan):
        known = _angle_of_tangent(value)
        if known is not None:
            return known
    return angle


class Arcsin(DegreesBase):
    nargs = 1

    @classmethod
    def radians(cls, value):
        return sp.asin(value)


class Arccos(DegreesBase):
    nargs = 1

    @classmethod
    def radians(cls, value):
        return sp.acos(value)


class Arcsec(DegreesBase):
    nargs = 1

    @classmethod
    def radians(cls, value):
        return sp.acos(1 / value)


class Arccsc(DegreesBase):
    nargs = 1

    @classmethod
    def radians(cls, value):
        return sp.asin(1 / value)


class Arctan(DegreesBase):
    """`ARCTAN(z)`, and `ARCTAN(y, x)`, the angle of the point (x, y) in degrees."""

    nargs = (1, 2)

    @classmethod
    def radians(cls, *args):
        if len(args) == 2:
            return sp.atan2(*args)
        return _arctangent(*args)


class Arccot(DegreesBase):
    """`ARCCOT(z)` is `180*ACOT(z)/pi`, and `ARCCOT(x, y)` is `ARCTAN(y, x)`, with the
    argument order of `ACOT(x, y)`."""

    nargs = (1, 2)

    @classmethod
    def radians(cls, *args):
        if len(args) == 2:
            return sp.atan2(args[1], args[0])
        return sp.pi / 2 - _arctangent(*args)


class Dilog(sp.Function):
    """`DILOG(z)`, the integral of `LN(t)/(1 - t)` from 1 to `z`, which is
    `POLYLOG(2, 1 - z)`, not `POLYLOG(2, z)`.

    It has the polylogarithm's exact values, `DILOG(0)` being `pi^2/6`; elsewhere the
    call stands as this head.
    """

    nargs = 1

    @classmethod
    def eval(cls, value: sp.Basic) -> sp.Basic | None:
        found = sp.polylog(2, 1 - value)
        if found.has(sp.polylog):
            return None
        return found

    def _eval_evalf(self, prec: int) -> sp.Basic:
        return sp.polylog(2, 1 - self.args[0])._eval_evalf(prec)

    def fdiff(self, argindex: int = 1) -> sp.Basic:
        (value,) = self.args
        return sp.log(value) / (1 - value)


#: The degree-valued inverse functions by name.
_IN_DEGREES: dict[str, type[DegreesBase]] = {
    "ARCSIN": Arcsin,
    "ARCCOS": Arccos,
    "ARCTAN": Arctan,
    "ARCCOT": Arccot,
    "ARCSEC": Arcsec,
    "ARCCSC": Arccsc,
}


def _in_degrees(name: str) -> Handler:
    head = _IN_DEGREES[name]

    def handler(conv: Converter, args: list) -> sp.Basic:
        return head(*args)

    return handler


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
    if not isinstance(value, Vector):
        if isinstance(value, sp.Mul):
            # An even power of a real factor is its own magnitude, below the line too:
            # `ABS(x/y^4)` is `ABS(x)/y^4` wherever it is defined.
            even, rest = sp.sift(
                value.args,
                lambda factor: factor.is_Pow
                and factor.exp.is_Integer
                and factor.exp % 2 == 0
                and factor.base.is_extended_real is True,
                binary=True,
            )
            if even:
                return sp.Mul(*even) * sp.Abs(sp.Mul(*rest))
        if conv.context.precision is Precision.APPROXIMATE:
            magnitude = _complex_magnitude(value)
            if magnitude is not None:
                return magnitude
        return sp.Abs(value)
    squares = [_absolute(conv, [element]) ** 2 for element in value.args]
    total = sp.Add(*squares)
    if conv.context.precision is Precision.APPROXIMATE:
        root = approximate_root(total, sp.S.Half)
        if root is not None:
            return root
    return sp.sqrt(total)


def _magnitude_of_power(cls: type[sp.Abs], arg: sp.Basic) -> sp.Basic | None:
    """`ABS(w^r)` as `ABS(w)^r`, for a base `w` not known real and a real exponent `r`.

    This holds for every complex `w`: `w^r` is `#e^(r*LN(w))`, whose magnitude is
    `#e^(r*RE(LN(w)))`, and both sides are 0 or undefined together at `w = 0`. Sympy
    takes the magnitude of such a power as the root of it times its conjugate, which
    keeps the power and the conjugate apart: `ABS((x + #i*y)^2)` would be `SQRT((x +
    #i*y)^2*(x - #i*y)^2)` rather than `x^2 + y^2`. Sympy's own rules take the
    magnitude of a product factor by factor, so this reaches the powers in a product
    too. A real base and a numeric one are left to sympy, which handles them.
    """
    if (
        isinstance(arg, sp.Pow)
        and arg.base.is_extended_real is not True
        and not arg.base.is_number
        and arg.exp.is_extended_real is True
    ):
        return cls(arg.base) ** arg.exp
    return _sympy_abs_eval(cls, arg)


# Sympy evaluates `Abs` also when `subs`, `xreplace` or a product rebuilds one, so only
# its own `eval` sees every magnitude of a power.
_sympy_abs_eval = sp.Abs.eval.__func__
sp.Abs.eval = classmethod(_magnitude_of_power)


def _complex_magnitude(value: sp.Basic) -> sp.Basic | None:
    """The magnitude of a complex number with rational parts as `approximate_root`
    gives it, or None for anything else."""
    if not isinstance(value, sp.Add | sp.Mul):
        return None
    real, rest = value.as_coeff_Add()
    imaginary, unit = rest.as_coeff_Mul()
    if unit is not sp.I or not isinstance(real, sp.Rational):
        return None
    return approximate_root(real**2 + imaginary**2, sp.S.Half)


def _floor(conv: Converter, args: list) -> sp.Basic:
    """`FLOOR(m, n)` is the floor of `m/n`; `n` defaults to 1, and `FLOOR(m, 0)` is `?`.

    A complex number's floor is taken in each coordinate.
    """
    numerator, denominator = _with_unit_default(args)
    if denominator.is_zero:
        return sp.nan
    return sp.floor(numerator / denominator)


def _ceiling(conv: Converter, args: list) -> sp.Basic:
    """`CEILING(m, n)`, the least integer at least `m/n`, which is `-FLOOR(-m/n)`;
    symbolic, it is written so."""
    numerator, denominator = _with_unit_default(args)
    if denominator.is_zero:
        return sp.nan
    return -sp.floor(-numerator / denominator)


def _round(conv: Converter, args: list) -> sp.Basic:
    """`ROUND(m, n)`, the integer nearest `m/n`, a tie rounding up: `FLOOR(m/n + 1/2)`.

    A complex number is rounded in each coordinate.
    """
    numerator, denominator = _with_unit_default(args)
    if denominator.is_zero:
        return sp.nan
    quotient = numerator / denominator
    if quotient.is_number and not quotient.is_extended_real:
        real, imaginary = quotient.as_real_imag()
        return sp.floor(real + sp.S.Half) + sp.I * sp.floor(imaginary + sp.S.Half)
    return sp.floor(quotient + sp.S.Half)


def _mod(conv: Converter, args: list) -> sp.Basic:
    """`MOD(m, n)`; `n` defaults to 1, giving the fractional part (see `Modulo`)."""
    return Modulo(*_with_unit_default(args))


def _mods(conv: Converter, args: list) -> sp.Basic:
    """`MODS(m, n)`; `n` defaults to 1 (see `SymmetricModulo`)."""
    return SymmetricModulo(*_with_unit_default(args))


def _integers(args: list) -> list[int]:
    """`args` as Python integers, for a function of integers; refused otherwise."""
    if not all(isinstance(argument, sp.Integer) for argument in args):
        raise TypeError("not integers")
    return [int(argument) for argument in args]


def _inverse_mod(conv: Converter, args: list) -> sp.Basic:
    """`INVERSE_MOD(a, m)`: the `x` in [0, |m|) with `a*x` congruent to 1 modulo `m`,
    `?` if `a` and `m` aren't coprime."""
    number, modulus = _integers(args)
    if modulus == 0:
        raise ValueError("no modulus")
    try:
        return sp.Integer(pow(number, -1, abs(modulus)))
    except ValueError:
        return sp.nan


def _power_mod(conv: Converter, args: list) -> sp.Basic:
    """`POWER_MOD(a, n, m)`: `MOD(a^n, m)` by modular powering. A negative `n` powers
    the inverse of `a`, `?` if there is none."""
    base, exponent, modulus = _integers(args)
    if modulus == 0:
        raise ValueError("no modulus")
    try:
        return sp.Integer(pow(base, exponent, modulus))
    except ValueError:
        return sp.nan


def _shift(conv: Converter, args: list) -> sp.Basic:
    """`SHIFT(n, c)`: the integer `n` shifted left `c` bits, right for a negative `c`,
    rounding toward -inf."""
    number, count = _integers(args)
    if count >= 0:
        if count > POWER_BITS:
            raise ValueError("too big")
        return sp.Integer(number << count)
    return sp.Integer(number >> -count)


def _poly_mod(symmetric: bool) -> Handler:
    """`POLY_MOD(u, n)` and `POLY_MODS(u, n)`: the polynomial `u` with each coefficient
    reduced modulo the integer `n`, by `MOD` or `MODS`.

    A rational coefficient is a residue: its denominator is inverted modulo `n`, so the
    call stands if it can't be. As `MOD(c, 0)` is `c`, modulus 0 leaves every
    coefficient as it is: the polynomial expanded.
    """
    reduce = SymmetricModulo if symmetric else Modulo

    def handler(conv: Converter, args: list) -> sp.Basic:
        value, modulus = args
        if not isinstance(modulus, sp.Integer):
            raise TypeError("not a modulus")
        if value.is_Relational:
            return value.func(*(handler(conv, [side, modulus]) for side in value.args))
        if not isinstance(value, sp.Expr) or isinstance(value, Vector):
            raise TypeError("not a polynomial")
        generators = tuple(sorted(value.free_symbols, key=str))
        if not generators:
            if not isinstance(value, sp.Rational):
                raise TypeError("not a rational number")
            terms = {(): value}
        else:
            polynomial = _rational_polynomial(value, generators)
            if polynomial is None:
                # A quotient whose denominator divides out, as `u*(p + 1)/(2*y)` with
                # `y` a factor of `u` does in an elliptic division polynomial.
                cancelled = sp.cancel(value)
                if sp.denom(cancelled).free_symbols:
                    raise TypeError("not a polynomial with numeric coefficients")
                generators = tuple(sorted(cancelled.free_symbols, key=str))
                polynomial = _rational_polynomial(cancelled, generators)
            if polynomial is None:
                raise TypeError("not a polynomial with numeric coefficients")
            terms = polynomial.as_dict()
        reduced = []
        for exponents, coefficient in terms.items():
            coefficient = sp.Rational(coefficient)
            if modulus.is_zero:
                residue = coefficient
            else:
                numerator = coefficient.p * pow(coefficient.q, -1, abs(int(modulus)))
                residue = reduce(sp.Integer(numerator), modulus)
            monomial = sp.Mul(
                *(g**e for g, e in zip(generators, exponents, strict=True))
            )
            reduced.append(residue * monomial)
        return sp.Add(*reduced)

    return handler


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
            if is_matrix(value):
                return mapped(value, lambda row: over(list(row.args)))
            if isinstance(value, Vector):
                values = list(value.args)
            else:
                values = args
        else:
            values = args
        if not values:
            raise ValueError("no sample")
        return over(values)

    return handler


def _pairwise(handler: Handler) -> Handler:
    """`handler` taking two or more vectors of one length element by element:
    `GCD([6, 15], [10, 4])` is `[GCD(6, 10), GCD(15, 4)]`."""

    def pairing(conv: Converter, args: list) -> sp.Basic:
        if len(args) > 1 and all(isinstance(value, Vector) for value in args):
            if len({len(value.args) for value in args}) != 1:
                raise ValueError("lengths differ")
            return Vector(
                *(pairing(conv, list(row)) for row in zip(*(v.args for v in args)))
            )
        return handler(conv, args)

    return pairing


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
    in a field of sixteen tangents. The solution is then rounded to working precision,
    as a value handed on to the next step (see `approximation.carried`): solved from
    approximate data it is exact only in name, a ratio of integers hundreds of digits
    long, and FIT_.MTH fits again with weights computed from it. Exact data still give
    their exact solution, which the rounding keeps, since the simplest rational within
    the tolerance of a ratio of small integers is that ratio.

    With one argument, the label vector is the matrix's first row and the data its
    other rows: `FIT([[x, a*x + b], [0, 1], [1, 3]])` is `2*x + 1`, as in Derive.
    CURV_FIT.MTH fits that way.

    Degenerate cases raise, leaving the call inert: no parameters, rows of the wrong
    width, too few rows, or a rank-deficient design matrix.
    """
    if len(args) == 1:
        labels, *rows = _elements(args[0])
        if not rows:
            raise TypeError("no data")
        data = Vector(*rows)
    else:
        labels, data = args
    labels = _elements(labels)
    if len(labels) < 2:
        raise TypeError("not a label vector")
    *variables, expression = labels
    if any(type(variable) is not sp.Symbol for variable in variables):
        raise TypeError("not data variables")
    if len(set(variables)) != len(variables):
        raise ValueError("a data variable twice")
    data = matrix_of(_worked_out(data, conv.context))
    if data is None:
        raise TypeError("not a data matrix")
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
    solution = _worked_out(design.solve_least_squares(targets), conv.context)
    return expression.subs(dict(zip(parameters, solution)))


# -- predicates -----------------------------------------------------------------
#
# The recognizers (`INTEGER?` ... `LOGICAL?`) test the form of their argument's
# simplified value and the type predicates (`INTEGER_TYPE?` ... `LOGICAL_TYPE?`) the type
# it is known to have. Both always answer true or false, since an inert test would leave
# its `IF` or `SELECT` undecided; programs call them in loops, so each is decided from
# the value's structure. The argument is simplified only where its form could still
# change (see `_formed`).


def _truth(holds: bool) -> sp.Basic:
    return sp.true if holds else sp.false


def is_plain(value: sp.Basic) -> bool:
    """Whether simplifying `value` leaves it as it is: a rational number, a name, a
    string, a truth value, `?`, or a vector or explicit set of such values."""
    if isinstance(value, (sp.Rational, sp.Symbol, BooleanAtom)) or value is sp.nan:
        return True
    if isinstance(value, (Vector, SetOf)):
        return all(is_plain(element) for element in value.args)
    return _plain_number(value)


def is_simple(value: sp.Basic) -> bool:
    """Whether `value` is as simple as it gets: a rational number, `pi`, `#e`, an
    infinity, a string, a truth value, `?`, a variable, or a vector or set of those.

    A vector or set remembers the answer (under the attribute `SIMPLE`), since a program
    asks it of the list it builds at every step, which holds the lists it built before;
    `vector_from` passes it on to the vectors list functions make.
    """
    return type(value) in _SIMPLE_TYPES or simple_constants(value) is not None


def simple_constants(value: sp.Basic) -> frozenset[sp.Basic] | None:
    """The constants (`pi`, `#e` and the infinities) the simple `value` holds, or None
    if it isn't simple (see `is_simple`). A test that holds `±inf` is decided for both
    its signs, so a program testing a long list asks this, remembered on the list,
    rather than looking through it for one."""
    kind = type(value)
    if kind in _SIMPLE_TYPES:
        return _NO_CONSTANTS
    if kind in _CONSTANT_SETS:
        return _CONSTANT_SETS[kind]
    if not isinstance(value, (Vector, SetOf)):
        return _NO_CONSTANTS if isinstance(value, _SIMPLE) else None
    known = value.__dict__.get(SIMPLE)
    if known is None:
        try:
            known = _constants_of(value.args)
        except RecursionError:
            known = _simple_walk(value)
        value.__dict__[SIMPLE] = known
    return None if known is False else known


def _constants_of(items: Iterable[sp.Basic]) -> frozenset[sp.Basic] | bool:
    """The constants `items` hold if each is simple, or False."""
    found = _NO_CONSTANTS
    for item in items:
        held = simple_constants(item)
        if held is None:
            return False
        if held:
            found = found | held
    return found


def _simple_walk(value: sp.Basic) -> frozenset[sp.Basic] | bool:
    """`_constants_of` the elements of `value` without recursion, for vectors nested
    too deep for it."""
    found = set()
    pending = [value]
    while pending:
        part = pending.pop()
        if type(part) in _SIMPLE_TYPES:
            continue
        if isinstance(part, (Vector, SetOf)):
            pending.extend(part.args)
        elif type(part) in _CONSTANT_SETS:
            found.add(part)
        elif not isinstance(part, _SIMPLE):
            return False
    return frozenset(found)


def vector_from(
    items: Sequence[sp.Basic], sources: Sequence[Vector] = (), added: Sequence = ()
) -> Vector:
    """The vector of `items`, which are elements of the vectors `sources` and the
    sympy objects `added`, built without looking at them again (see `Vector.of`).

    It is known simple (see `is_simple`) if every source is and every added item is:
    a loop that adjoins to a long list then needs no walk of the list to know it."""
    if not all(isinstance(item, sp.Basic) for item in added):
        return Vector(*items)
    return _known_simple(Vector.of(tuple(items)), sources, added)


def _shared(made: Vector, source: Vector, added: Sequence[sp.Basic]) -> Vector:
    """`made`, which a list function made of the elements of `source` with `added` put
    among them (see `Vector.sliced`), with what is known of `source` carried over: that
    it is simple, and that approximating leaves it as it is (see `carried_over`)."""
    return carried_over(_known_simple(made, [source], added), source, added)


def _known_simple(
    vector: Vector, sources: Sequence[Vector], added: Sequence[sp.Basic]
) -> Vector:
    """`vector`, made of elements of `sources` and `added`, marked simple (see
    `is_simple`) if every source and every added item is, with the constants they hold
    (see `simple_constants`), and not simple if an added item isn't. Otherwise, if
    every source is simple or settled under the same settings (see `SETTLED`), it is
    marked settled under them if every added item is simple or so settled too, and
    else marked with them and the other added items (see `SETTLED_FROM`)."""
    if sources or added:
        held = _constants_of(added)
        if held is False:
            # An added item is an element of `vector`.
            vector.__dict__[SIMPLE] = False
        elif not any(
            SETTLED in source.__dict__ and SIMPLE not in source.__dict__
            for source in sources
        ):
            # A settled source is looked through only where its simplicity is asked:
            # a loop extending a table of roots would look through it at every pass.
            from_sources = _constants_of(sources)
            if from_sources is not False:
                vector.__dict__[SIMPLE] = held | from_sources
                return vector
        if all(unwrapped(source) for source in sources) and not any(
            isinstance(item, _WRAPPED) for item in added
        ):
            vector.__dict__[UNWRAPPED] = True
        key = None
        rest: list[sp.Basic] = []
        for source in sources:
            found = _settled_part(source, key)
            if found is None:
                return vector
            key, pending = found
            rest.extend(pending)
        for item in added:
            found = _settled_part(item, key)
            if found is None:
                rest.append(item)
            else:
                key, pending = found
                rest.extend(pending)
        if key is not None and key is not True:
            settled_since(vector, key, rest)
    return vector


def settled_since(vector: Vector, key: object, added: Sequence[sp.Basic]) -> Vector:
    """`vector`, made of elements of vectors settled under the settings `key` and of
    `added`, marked settled under them if `added` holds only simple values (see
    `SETTLED`), and else with them and `added` (see `SETTLED_FROM`), unless it holds
    more than `SETTLED_ADDED` values."""
    rest = [item for item in added if not is_simple(item)]
    if not rest:
        vector.__dict__[SETTLED] = key
    elif len(rest) <= SETTLED_ADDED:
        vector.__dict__[SETTLED_FROM] = (key, tuple(rest))
    return vector


def unwrapped(value: Vector | SetOf) -> bool:
    """Whether no element of the vector or set `value` is an assignment, a definition
    or a command's answer, which its use as a value unwraps; remembered on the value,
    which a program may read element by element as often as it has elements, and
    passed on to the vectors list functions make (see `_known_simple`)."""
    found = value.__dict__.get(UNWRAPPED)
    if found is None:
        found = isinstance(value.__dict__.get(SIMPLE), frozenset) or not any(
            isinstance(element, _WRAPPED) for element in value.args
        )
        value.__dict__[UNWRAPPED] = found
    return found


#: What the use of a vector as a value unwraps (see `unwrapped`).
_WRAPPED = (Assign, FunDef, Commanded)


def _settled_part(
    value: sp.Basic, key: object
) -> tuple[object, tuple[sp.Basic, ...]] | None:
    """The settings `value` is settled under (see `SETTLED`), or True if it is simple,
    with the values `SETTLED_FROM` records on it besides, if they agree with `key`:
    None, True or other settings. None if it is neither or they disagree."""
    if isinstance(value, Vector):
        state = value.__dict__
        found = state.get(SETTLED)
        pending: tuple[sp.Basic, ...] = ()
        if found is None and SETTLED_FROM in state:
            found, pending = state[SETTLED_FROM]
        if found is not None:
            if key is None or key is True or found is key or found == key:
                return found, pending
            return None
    if not is_simple(value):
        return None
    return (True if key is None else key), ()


def settled_key(value: sp.Basic) -> object:
    """The settings under which a program stores the vector `value` as it is (see
    `SETTLED`), or None if it isn't known to."""
    return value.__dict__.get(SETTLED) if isinstance(value, Vector) else None


def settled_from(value: sp.Basic) -> tuple[object, tuple[sp.Basic, ...]] | None:
    """The settings and the elements of `SETTLED_FROM` the vector `value` records, or
    None."""
    return value.__dict__.get(SETTLED_FROM) if isinstance(value, Vector) else None


def mark_settled(value: Vector, key: object) -> Vector:
    """`value`, recorded as stored as it is under the settings `key` (see `SETTLED`)."""
    value.__dict__[SETTLED] = key
    return value


#: The attribute under which a vector or set records whether it `is_simple`: the
#: constants it holds (see `simple_constants`) if it is, and False if it isn't.
SIMPLE = "_simple"

#: The attribute `unwrapped` remembers its answer in.
UNWRAPPED = "_unwrapped"

#: The attribute under which a vector records the settings under which simplifying it
#: as a program stores a value leaves it as it is (see `_Evaluator._settled` in
#: evaluate.py): simplifying it so gave it back unchanged, or a list function made it
#: of the elements of such vectors and of values simplifying leaves as they are. A
#: loop that extends a long table of roots or symbolic expressions then simplifies
#: none of its old elements again.
SETTLED = "_settled"

#: The attribute under which a vector a list function made of the elements of vectors
#: settled under some settings (see `SETTLED`), and of other values, records those
#: settings and the other values: it is settled if simplifying leaves each of those as
#: it is.
SETTLED_FROM = "_settled_from"

#: The most values other than settled and simple ones that `SETTLED_FROM` records.
SETTLED_ADDED = 16

#: The constants `is_simple` takes as simple, which simplifying exactly leaves as they
#: are.
_SIMPLE_CONSTANTS = (sp.pi, sp.E, sp.oo, -sp.oo, sp.zoo)

#: Each of those as the set of constants it holds, by its type.
_CONSTANT_SETS = {
    type(constant): frozenset({constant}) for constant in _SIMPLE_CONSTANTS
}

#: What a simple value without any of those holds.
_NO_CONSTANTS: frozenset[sp.Basic] = frozenset()

#: The values `is_simple` takes as simple, besides variables.
_SIMPLE = (sp.Rational, type(sp.nan), BooleanAtom, StringLiteral)

#: The commonest of those, by exact type, to look up before testing the classes.
_SIMPLE_TYPES = frozenset(
    {
        sp.Symbol,
        sp.Integer,
        sp.Rational,
        type(sp.S.Zero),
        type(sp.S.One),
        type(sp.S.NegativeOne),
        type(sp.S.Half),
        type(sp.true),
        type(sp.false),
        StringLiteral,
    }
)


def _plain_number(value: sp.Basic) -> bool:
    """Whether `value` is a number sympy keeps in its simplest form as it builds it: a
    rational, `#i`, a root of a rational (`SQRT(8)` is built as `2*SQRT(2)`), or a sum
    or product of those."""
    if isinstance(value, sp.Rational) or value is sp.I:
        return True
    if isinstance(value, sp.Pow):
        return isinstance(value.base, sp.Rational) and isinstance(value.exp, sp.Rational)
    if isinstance(value, (sp.Add, sp.Mul)):
        return all(_plain_number(part) for part in value.args)
    return False


def _formed(conv: Converter, value: sp.Basic, deep: bool = True) -> sp.Basic:
    """`value` simplified, as a predicate reads it, unless it is plain already, or,
    unless `deep`, a vector or set, whose kind and length simplifying keeps."""
    if not deep and isinstance(value, (Vector, SetOf)):
        return value
    if is_plain(value):
        return value
    (simplified,) = conv.simplified_values([value])
    return simplified


def _numeric(conv: Converter, value: sp.Basic) -> sp.Basic:
    """`value` as the number predicates read it: under Approximate, a constant
    approximated, with an imaginary part too small to show dropped, since an
    approximated real number is a rational."""
    if (
        conv.context.precision is not Precision.APPROXIMATE
        or isinstance(value, sp.Rational)
        or not isinstance(value, sp.Expr)
        or not value.is_number
        or value.has(sp.nan, sp.oo, -sp.oo, sp.zoo)
    ):
        return value
    approximate = approximated(value, conv.context)
    real, imaginary = approximate.as_real_imag()
    if not (isinstance(real, sp.Rational) and isinstance(imaginary, sp.Rational)):
        return approximate
    if imaginary and abs(imaginary) < abs(real) * sp.Rational(
        1, 10**conv.context.precision_digits
    ):
        return real
    return approximate


def _gaussian(value: sp.Basic) -> bool:
    """Whether `value` is a rational number or `a + b*#i` with `a` and `b` rational."""
    if isinstance(value, sp.Rational):
        return True
    if not (isinstance(value, sp.Expr) and value.is_number and value.has(sp.I)):
        return False
    real, imaginary = value.as_real_imag()
    return isinstance(real, sp.Rational) and isinstance(imaginary, sp.Rational)


def _recognizer(test: Callable[[sp.Basic], bool], numeric: bool = False) -> Handler:
    """A recognizer of one argument, `test` deciding it on the simplified value (for a
    number predicate, read as `_numeric` says)."""

    def handler(conv: Converter, args: list) -> sp.Basic:
        value = _formed(conv, _one(args), deep=False)
        if numeric:
            value = _numeric(conv, value)
        return _truth(test(value))

    return handler


def _is_variable(value: sp.Basic) -> bool:
    """A variable's name, an arbitrary constant included, or a string, which Derive
    takes for a variable of that name (a Groebner basis program tests `VARIABLE?` of the
    variables "x1", "x2", ... it is given); not a constant, or a subscripted name, which
    stands for an element."""
    if isinstance(value, sp.MatrixSymbol):
        return True
    return (
        isinstance(value, sp.Symbol)
        and " SUB " not in value.name
        and value not in CONSTANTS.values()
    )


def _is_matrix(value: sp.Basic) -> bool:
    """A vector of one or more vectors, all of one length."""
    if not (isinstance(value, Vector) and value.args):
        return False
    rows = value.args
    return all(isinstance(row, Vector) for row in rows) and (
        len({len(row.args) for row in rows}) == 1
    )


def _headed(name: str, *heads: str) -> Handler:
    """A recognizer of an expression's top-level operator, as `STRING` names it: true
    for a value that isn't a number and whose head is one of `heads`. Only a sum, a
    product, a quotient or a power can have such a head, so nothing else is printed to
    find out."""

    def handler(conv: Converter, args: list) -> sp.Basic:
        value = _formed(conv, _one(args), deep=False)
        if not isinstance(value, (sp.Add, sp.Mul, sp.Pow, sp.exp)) or value.is_number:
            return sp.false
        try:
            head, operands = string_parts(conv, value)
        except Exception:
            return sp.false
        if head not in heads:
            return sp.false
        if name == "POWER?":
            exponent = operands[1]
            return _truth(isinstance(exponent, sp.Expr) and exponent.is_number)
        return sp.true

    return handler


def _is_integer_type(value: sp.Basic) -> bool:
    return isinstance(value, sp.Expr) and value.is_integer is True


def _is_real_type(value: sp.Basic) -> bool:
    return (
        isinstance(value, sp.Expr)
        and not isinstance(value, (Vector, StringLiteral))
        and value.is_real is True
    )


def _is_number_type(value: sp.Basic) -> bool:
    """A scalar numeric expression: not a vector, set, string, relation or truth value,
    and nothing in it a matrix or a string."""
    if not isinstance(value, sp.Expr) or isinstance(value, (Vector, SetOf)):
        return False
    if is_set(value) or isinstance(value, (sp.MatrixExpr, Logical, LogicalSymbol)):
        return False
    return not value.has(
        Vector, StringLiteral, sp.MatrixSymbol, SetOf, SetSymbol, LogicalSymbol
    )


def _is_vector_type(value: sp.Basic) -> bool:
    return isinstance(value, (Vector, sp.MatrixExpr))


def _is_logical_type(value: sp.Basic) -> bool:
    """`true`, `false`, a relation, a logical connective of anything, a variable
    declared Logical, or a predicate's call that stands."""
    if isinstance(value, LogicalSymbol):
        return True
    if isinstance(value, (Boolean, Logical)) and not isinstance(value, sp.Symbol):
        return True
    return isinstance(value, AppliedUndef) and type(value).__name__.endswith("?")


def _is_number(conv: Converter, args: list) -> sp.Basic:
    """`NUMBER?(u)`, also `NUMBER(u)`: whether `u` simplifies to a number, a rational or
    a Gaussian rational `a + b*#i`; not `pi`, `SQRT(2)`, `inf` or `?`.

    The utility files rely on its answering at once: `NUMBER(d)` is false while `d` is
    still an unbound parameter, which tells a supplied argument from an omitted one.
    """
    return _recognizer(_gaussian, numeric=True)(conv, args)


def _distributed(handler: Handler) -> Handler:
    """`handler` applied to each element of a vector (recursively, so a matrix) or of
    an explicit set given as the first argument."""

    def distributing(conv: Converter, args: list) -> sp.Basic:
        first, *rest = args
        if isinstance(first, Vector):
            return Vector(*(distributing(conv, [each, *rest]) for each in first.args))
        if isinstance(first, SetOf):
            elements = [distributing(conv, [each, *rest]) for each in first.args]
            (joined,) = conv.sets_simplified([SetOf(*elements)])
            return joined
        return handler(conv, args)

    return distributing


def _real_number(conv: Converter, value: sp.Basic) -> sp.Expr | None:
    """`value` as a real number for the prime functions, or None if it is none."""
    value = _numeric(conv, value)
    if isinstance(value, sp.Rational):
        return value
    if (
        isinstance(value, sp.Expr)
        and value.is_number
        and value.is_extended_real
        and value.is_finite
    ):
        return value
    return None


def _is_prime(conv: Converter, args: list) -> sp.Basic:
    """`PRIME?(m)`, also `PRIME(m)`: whether `m` is a prime number.

    Any number that isn't a prime is false: a negative or non-integer one, a real or
    complex constant. Anything else stays as written, since `PRIME?(k)` for a variable
    `k` hasn't been asked yet. The test is exact: `sp.isprime` runs a deterministic
    Miller-Rabin test below 3.3*10^24 and the BPSW test above.

    The manual's second argument is the number of rounds of a probabilistic test. The
    test here is deterministic, so it is accepted and ignored.
    """
    value, *rounds = args
    value = _numeric(conv, value)
    if isinstance(value, sp.Integer):
        return _truth(sp.isprime(int(value)))
    if isinstance(value, sp.Expr) and value.is_number and value is not sp.nan:
        return sp.false
    raise TypeError("not a number")


def _next_prime(conv: Converter, args: list) -> sp.Basic:
    """`NEXT_PRIME(m)`: the first prime strictly above the real number `m`, which needn't
    be an integer.

    `NEXT_PRIME(7)` is 11. A repeat count is accepted and ignored, as for `PRIME`.
    """
    value, *rounds = args
    number = _real_number(conv, value)
    if number is None:
        raise TypeError("not a number")
    floor = sp.floor(number)
    if not isinstance(floor, sp.Integer):
        raise TypeError("no integer part")
    return sp.Integer(sp.nextprime(int(floor)))


def _previous_prime(conv: Converter, args: list) -> sp.Basic:
    """`PREVIOUS_PRIME(m)`: the last prime strictly below the real number `m`, or `?` if
    there is none."""
    value, *rounds = args
    number = _real_number(conv, value)
    if number is None:
        raise TypeError("not a number")
    ceiling = sp.ceiling(number)
    if not isinstance(ceiling, sp.Integer):
        raise TypeError("no integer part")
    if ceiling <= 2:
        return sp.nan
    return sp.Integer(sp.prevprime(int(ceiling)))


def _identical(conv: Converter, args: list) -> sp.Basic:
    """`IDENTICAL?(u, v)`: whether `u` and `v` simplify to the same expression. Numbers
    are identical when equal, an approximate number being a rational, and sets when
    their elements are, in their order."""
    first, second = args
    return _truth(_formed(conv, first) == _formed(conv, second))


def _member(conv: Converter, args: list) -> sp.Basic:
    """`MEMBER?(u, v)`: whether some element of the vector or explicit set `v` is
    identical to `u` (see `_identical`). Any other `v` leaves the call standing."""
    value, collection = args
    collection = _formed(conv, collection)
    if not isinstance(collection, (Vector, SetOf)):
        raise TypeError("not a collection")
    return _truth(_formed(conv, value) in collection.args)


# -- strings ----------------------------------------------------------------------


def string_parts(conv: Converter, value: sp.Basic) -> tuple[str, list[sp.Basic]]:
    """The head and operands `STRING` takes `value` apart into: the operator or function
    at its top as Rederive writes it, and its operands, which are values (see
    `_string_head`). Raises for a value with no operator, such as a name."""
    # The answer writer reads trees; it is imported here, since it imports this module.
    from rederive.engine.from_sympy import written_tree

    exact = replace(conv.context, notation=Notation.RATIONAL)
    node = written_tree(value, exact)
    head, operands = _string_head(node)
    plain = Converter(exact)
    values = []
    for operand in operands:
        if isinstance(operand, Node):
            values.append(plain.convert(operand))
        else:
            negative, part = operand
            if isinstance(part, Node):
                part = plain.convert(part)
            values.append(negated(part) if negative else part)
    return head, values


#: The heads of `STRING` for operators, by node kind and value.
_STRING_OPERATORS: dict[tuple[Kind, str | None], str] = {
    (Kind.UNOP, "-"): "-",
    (Kind.UNOP, "+-"): "±",
    (Kind.BINOP, "/"): "/",
    (Kind.BINOP, "^"): "^",
    (Kind.BINOP, "."): ".",
    (Kind.REL, "="): "=",
    (Kind.REL, "/="): "≠",
    (Kind.REL, "<"): "<",
    (Kind.REL, "<="): "≤",
    (Kind.REL, ">"): ">",
    (Kind.REL, ">="): "≥",
    (Kind.SETOP, "UNION"): "∪",
    (Kind.SETOP, "INTERSECTION"): "∩",
    (Kind.SETOP, "\\"): "\\",
    (Kind.POSTOP, "!"): "!",
    (Kind.POSTOP, "`"): "`",
    (Kind.SUB, None): "↓",
}

#: The heads of `STRING` for the logical operators.
_STRING_LOGICAL = {
    Kind.NOT: "¬",
    Kind.AND: "∧",
    Kind.OR: "∨",
    Kind.XOR: "⊻",
    Kind.IMP: "→",
    Kind.IFF: "↔",
}

#: The heads of `STRING` for calls written with a sign of their own.
_STRING_CALLS = {"INT": "∫", "SUM": "∑", "PRODUCT": "∏", "DIF": "∂", "ELEMENT": "↓"}


def _string_head(node: Node) -> tuple[str, list]:
    """A written value's head for `STRING`, and its operands: nodes, or a pair of a
    flag, true for a term subtracted in a sum, and the term's node; or values made
    here, as the exponent 1/2 of a root."""
    kind = node.kind
    children = list(node.children)
    if kind is Kind.SUM:
        operators = str(node.value)
        terms: list = [children[0]]
        for index, term in enumerate(children[1:]):
            terms.append((operators[index] == "-", term))
        return "+", terms
    if kind is Kind.PRODUCT:
        return "·", children
    if kind in _STRING_LOGICAL:
        return _STRING_LOGICAL[kind], children
    if kind is Kind.ABS:
        return "ABS", children
    if kind is Kind.INFIX:
        return str(node.value), children
    if kind is Kind.CALL:
        name = str(children[0].value)
        arguments = children[1:]
        if name == "SQRT" and len(arguments) == 1:
            return "^", [arguments[0], (False, sp.Rational(1, 2))]
        if name == "EXP" and len(arguments) == 1:
            return "^", [(False, sp.E), arguments[0]]
        return _STRING_CALLS.get(name, name), arguments
    key = (kind, None if kind is Kind.SUB else str(node.value))
    head = _STRING_OPERATORS.get(key)
    if head is None:
        raise TypeError("no operator")
    return head, children


def _output_numeral(number: int, context: Context) -> str:
    """The digits of a nonnegative integer in the output base, as displayed: digits
    above 9 are upper-case letters, and a numeral starting with one gets a leading 0."""
    setting = context.setting_values.get("OutputBase", 10)
    base = BASES.get(setting, 10) if isinstance(setting, str) else int(setting)
    if base == 10:
        return str(number)
    text = ""
    while True:
        number, digit = divmod(number, base)
        text = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[digit] + text
        if not number:
            break
    return "0" + text if text[0].isalpha() else text


def _shown_number(value: sp.Rational, context: Context) -> str | None:
    """A positive non-integer as the current notation shows it, where that is a decimal
    (with an exponent part or not); None where it is a ratio."""
    text = author_text(value, context)
    return None if "/" in text else text


def _string(conv: Converter, args: list) -> sp.Basic:
    """`STRING(u)`: `u` simplified, as text where it is a name, a constant or an
    integer (in the output base) or a decimal as shown, a negative one after a minus
    sign, as `NAME_TO_CODES` reads it; and otherwise
    the vector of the name of its operator or function and its operands (see
    `string_parts`): `STRING(x + y - z)` is `["+", x, y, -z]`. It distributes over the
    elements of vectors and sets. An assignment or definition leaves it standing.
    Under Approximate precision a number is the decimal it is shown as."""
    return _string_of(conv, _numeric(conv, _formed(conv, _one(args))))


def _string_of(conv: Converter, value: sp.Basic) -> sp.Basic:
    if isinstance(value, StringLiteral):
        return value
    if isinstance(value, Vector):
        return Vector(*(_string_of(conv, element) for element in value.args))
    if isinstance(value, SetOf):
        (strings,) = conv.sets_simplified(
            [SetOf(*(_string_of(conv, element) for element in value.args))]
        )
        return strings
    if isinstance(value, (Assign, FunDef, Declare)):
        raise TypeError("not a value")
    if isinstance(value, sp.Integer):
        numeral = _output_numeral(abs(int(value)), conv.context)
        return StringLiteral(numeral if value >= 0 else "-" + numeral)
    if isinstance(value, sp.Rational):
        shown = _shown_number(abs(value), conv.context)
        if shown is not None:
            return StringLiteral(shown if value > 0 else "-" + shown)
    if isinstance(value, (sp.Symbol, sp.MatrixSymbol, BooleanAtom)) or value in (
        sp.nan,
        sp.oo,
        sp.pi,
        sp.E,
        sp.I,
        sp.EulerGamma,
    ):
        if isinstance(value, sp.Symbol) and " SUB " in value.name:
            base, index = value.name.split(" SUB ", 1)
            return Vector(StringLiteral("↓"), sp.Symbol(base), _subscript_index(index))
        return StringLiteral(author_text(value, conv.context))
    head, operands = string_parts(conv, value)
    return Vector(StringLiteral(head), *operands)


def _subscript_index(index: str) -> sp.Basic:
    """The index of a subscripted name, from the text after its `SUB`."""
    index = index.strip("()")
    return sp.Integer(index) if index.isdigit() else sp.Symbol(index)


#: The codes that are no character of a string: NUL, line feed, carriage return and the
#: double quote, which ends one.
_NO_CHARACTER = frozenset({0, 10, 13, 34})

#: Derive's codes of the symbols of its own font, by character: `NAME_TO_CODES("≠")` is
#: `[148]`, as Derive's statistics programs test it. Letters that Derive's font has at
#: their ISO 8859-1 codes are left out: Derive 6 writes a letter beyond ASCII as DEL and
#: its hexadecimal digits (see `_DEL`).
_DERIVE_CODES = {
    character: code
    for code, character in DERIVE_CHARACTERS.items()
    if not (code == ord(character) and character.isalpha())
}

#: The code that introduces any other character beyond ASCII, as four hexadecimal
#: digits: the Life program of the newsletter decodes `[127, 50, 53, 97, 48]` as "■".
_DEL = 127


def _name_to_codes(conv: Converter, args: list) -> sp.Basic:
    """`NAME_TO_CODES(s)`: the character codes of a string, of the digits of an integer
    (with its sign), of a number as the current notation shows it, or of a name; for a
    vector, of each element. A symbol of Derive's font has Derive's code (see
    `DERIVE_CHARACTERS`), another character up to 255 its own, except a letter: that,
    and any character beyond 255, is DEL, 127, followed by the codes of its four
    lowercase hexadecimal digits, as Derive writes it in a file (the Caesar cipher of
    the newsletter enciphers "ÄÖäö" so)."""
    value = _one(args)
    if isinstance(value, Vector):
        return Vector(*(_name_to_codes(conv, [element]) for element in value.args))
    text = _named_text(conv, value)
    codes: list[int] = []
    for character in text:
        code = _DERIVE_CODES.get(character, ord(character))
        if character in _DERIVE_CODES or code < 128 or (
            code < 256 and not character.isalpha()
        ):
            codes.append(code)
        else:
            codes.extend((_DEL, *(ord(digit) for digit in f"{code:04x}")))
    return Vector(*(sp.Integer(code) for code in codes))


def _named_text(conv: Converter, value: sp.Basic) -> str:
    """The text `NAME_TO_CODES` takes the codes of; raises where there is none."""
    if isinstance(value, StringLiteral):
        return value.name
    if isinstance(value, sp.Integer):
        sign = "-" if value < 0 else ""
        return sign + _output_numeral(abs(int(value)), conv.context)
    if isinstance(value, sp.Rational):
        return author_text(value, conv.context)
    if _is_variable(value) and not isinstance(value, sp.MatrixSymbol):
        return value.name
    raise TypeError("no text")


def _codes_to_name(conv: Converter, args: list) -> sp.Basic:
    """`CODES_TO_NAME(v)`: the text of the character codes in `v`, a vector or a single
    code: the integer it spells, if it is the digits of one with an optional sign, or
    else the string. The codes are read as `NAME_TO_CODES` writes them: Derive's code
    of a symbol of its font, DEL followed by four hexadecimal digits for any other
    character beyond ASCII; another code is that of the character in Unicode. A vector
    of vectors gives one for each.

    A code that is no character of a string (see `_NO_CHARACTER`, and the surrogates of
    UTF-16) leaves the call standing.
    """
    value = _one(args)
    if isinstance(value, Vector) and value.args and all(
        isinstance(element, Vector) for element in value.args
    ):
        return Vector(*(_codes_to_name(conv, [element]) for element in value.args))
    codes = list(value.args) if isinstance(value, Vector) else [value]
    if not all(isinstance(code, sp.Integer) for code in codes):
        raise TypeError("not codes")
    numbers = [int(code) for code in codes]
    characters = []
    place = 0
    while place < len(numbers):
        code = numbers[place]
        if code == _DEL:
            digits = "".join(
                chr(each) for each in numbers[place + 1 : place + 5] if 0 < each < 128
            )
            if len(digits) != 4 or not all(c in string.hexdigits for c in digits):
                raise ValueError("not a character")
            characters.append(chr(int(digits, 16)))
            place += 5
            continue
        if (
            not 1 <= code <= sys.maxunicode
            or code in _NO_CHARACTER
            or 0xD800 <= code <= 0xDFFF
        ):
            raise ValueError("not a character")
        characters.append(DERIVE_CHARACTERS.get(code) or chr(code))
        place += 1
    text = "".join(characters)
    digits = text[1:] if text.startswith("-") else text
    if digits and digits.isascii() and digits.isdigit():
        return sp.Integer(int(text))
    return StringLiteral(text)


# -- substitution ---------------------------------------------------------------


def _substituted_exactly(conv: Converter, args: list) -> sp.Basic:
    """`SUBST(u, v, w)`: `u` with each occurrence of `v` replaced by `w`, `v` being a
    subexpression of `u`'s simplified form: `x^2` is no part of `x^4`, nor `x*y` of
    `x*y*z`. `SUBST(u, v1, w1, v2, w2, ...)` substitutes one pair after another, each in
    the simplified result of the one before; `SUBST(u, [v1, ...], [w1, ...])` all at
    once, none of the replacements scanned again; a number replaces each of `v1, ...`,
    so `SUBST(u, v, 1)` sets every variable of the vector `v` to 1, as Groebner basis
    programs take a monomial's coefficient (a variable instead of the number may yet be
    a vector of values, so that call stands). A variable bound in `u`, as a sum's
    index, is not replaced. Any other argument list leaves the call standing."""
    if len(args) < 3 or len(args) % 2 == 0:
        raise TypeError("not pairs")
    value, *pairs = conv.simplified_values(list(args))
    for number in range(0, len(pairs), 2):
        target, replacement = pairs[number], pairs[number + 1]
        if (
            isinstance(target, Vector)
            and _plain_scalar(replacement)
            and replacement.is_number
        ):
            mapping = dict.fromkeys(target.args, replacement)
        elif isinstance(target, Vector):
            if len(replacement.args) != len(target.args):
                raise TypeError("not as many values as targets")
            mapping = dict(zip(target.args, replacement.args, strict=True))
        else:
            mapping = {target: replacement}
        value = settled(replaced_exactly(value, mapping))
        if number + 2 < len(pairs):
            (value,) = conv.simplified_values([value])
    return value


def replaced_exactly(value: sp.Basic, mapping: dict[sp.Basic, sp.Basic]) -> sp.Basic:
    """`value` with each subexpression that is a key of `mapping` replaced by its value,
    all at once; inside a sum, product, integral or limit, its bound variable is left
    alone."""
    if value in mapping:
        return mapping[value]
    if not value.args or isinstance(value, StringLiteral):
        return value
    if isinstance(value, sp.Derivative) and any(
        variable in mapping for variable in value.variables
    ):
        return value.subs(mapping)
    bound = getattr(value, "bound_symbols", ()) or ()
    if isinstance(value, (Antidifference, Antiquotient, MovingLimit)):
        bound = (value.args[1],)
    inner = (
        {key: new for key, new in mapping.items() if key not in bound}
        if bound
        else mapping
    )
    if not inner:
        return value
    if isinstance(value, (Vector, SetOf)):
        return value.func(*(replaced_exactly(part, inner) for part in value.args))
    parts = [replaced_exactly(part, inner) for part in value.args]
    if all(new is old for new, old in zip(parts, value.args, strict=True)):
        return value
    try:
        return value.func(*parts)
    except Exception:
        return value.xreplace(dict(zip(value.args, parts, strict=True)))


_PREDICATE_HANDLERS: dict[str, Handler] = {
    "INTEGER?": _recognizer(lambda v: isinstance(v, sp.Integer), numeric=True),
    "RATIONAL?": _recognizer(lambda v: isinstance(v, sp.Rational), numeric=True),
    "EVEN?": _recognizer(lambda v: isinstance(v, sp.Integer) and v % 2 == 0, True),
    "ODD?": _recognizer(lambda v: isinstance(v, sp.Integer) and v % 2 == 1, True),
    "NUMBER?": _is_number,
    "VARIABLE?": _recognizer(_is_variable),
    "STRING?": _recognizer(lambda v: isinstance(v, StringLiteral)),
    "SUM?": _headed("SUM?", "+"),
    "PRODUCT?": _headed("PRODUCT?", "·", "/", "-"),
    "POWER?": _headed("POWER?", "^"),
    "VECTOR?": _recognizer(lambda v: isinstance(v, Vector)),
    "MATRIX?": _recognizer(_is_matrix),
    "SET?": _recognizer(lambda v: isinstance(v, SetOf)),
    "LOGICAL?": _recognizer(lambda v: isinstance(v, BooleanAtom)),
    "INTEGER_TYPE?": _recognizer(_is_integer_type),
    "REAL_TYPE?": _recognizer(_is_real_type),
    "NUMBER_TYPE?": _recognizer(_is_number_type),
    "VECTOR_TYPE?": _recognizer(_is_vector_type),
    "SET_TYPE?": _recognizer(is_set),
    "LOGICAL_TYPE?": _recognizer(_is_logical_type),
    "IDENTICAL?": _identical,
    "MEMBER?": _member,
}

#: The recognizers, type predicates and comparisons of form, which answer true or false
#: at once.
PREDICATES = frozenset(_PREDICATE_HANDLERS)

#: Built-ins answering at once from the form or type of their arguments' values, true
#: or false where a variable may yet be anything: `NUMBER?(x)` is false. In a
#: definition they stand where an argument mentions a parameter (see
#: `Converter.mentions_parameters`).
ON_FORM = frozenset(
    {*PREDICATES, "NUMBER", "STRING", "NAME_TO_CODES", "CODES_TO_NAME"}
    | {"FACTORS", "TERMS"}
)

#: The built-ins of `ON_FORM` that take an expression apart rather than test what a
#: variable is. In the body of a function called in a definition being shown, they work
#: on its arguments with the definition's parameters standing for themselves, as the
#: rest of that body does: `F1(x) := POLYMODS(x^4 + 75*x^3, 79, x)` reduces the
#: coefficients of the polynomial in x. Written in the definition itself, on its
#: parameters, they stand like the others.
DECOMPOSING = frozenset({"FACTORS", "TERMS"})


def _numerator(conv: Converter, args: list) -> sp.Basic:
    return _fraction_of(conv, _one(args))[0]


def _denominator(conv: Converter, args: list) -> sp.Basic:
    return _fraction_of(conv, _one(args))[1]


def _fraction_of(conv: Converter, value: sp.Basic) -> tuple[sp.Basic, sp.Basic]:
    """The numerator and denominator of `value`, as Derive reads them off its
    simplified argument: the factors with negative exponents and the denominator of
    the numeric coefficient are the denominator, the rest the numerator.

    Nothing is put over a common denominator, so a sum is its own numerator:
    `DENOMINATOR(1/x + 1/y)` and `DENOMINATOR(x/4 + 1/3)` are 1, while
    `DENOMINATOR((x + 1)/(x*y))` is `x*y` and `DENOMINATOR(x^(-k))` is `x^k`.

    In Approximate mode a quotient of approximate numbers is first written at the
    scale of its value, as it is displayed (see `approximation.rescaled`), so
    `DENOMINATOR` of a fitted `(a*x^2 + b*x + c)/(x + d)` is `x + d` as shown, not
    that times a long integer.
    """
    if conv.context.precision is Precision.APPROXIMATE:
        value = rescaled(value, conv.context.precision_digits)
    return sp.fraction(value)


def _terms(conv: Converter, args: list) -> sp.Basic:
    """`TERMS(u)`: the terms of `u` as a vector.

    Terms are those of `u` as Simplify gives it (the evaluator simplifies it, see
    `evaluate._SIMPLIFIED_ARGUMENTS`), so nothing is expanded: `TERMS(x*(a + b)^2 + c)`
    has two. With an amount and variables, `u` is expanded first as `EXPAND` would:
    `TERMS(u, Trivial)` is `TERMS(EXPAND(u, Trivial))`.

    Terms are in the order the printer writes the sum: by the variables in their
    order, higher powers first. Variables to expand in come first in that order, so the
    first term of `TERMS(u, t)`, which is also `TERMS(EXPAND(u, t))`, has the highest
    power of `t`, as Groebner basis programs read a leading term. A logical
    formula's terms are the disjuncts of its disjunctive normal form, and a relation or
    any other formula without `OR` is a single term.

    A vector distributes, each element giving a row, so `TERMS` of a vector of sums is a
    matrix if they have equal numbers of terms; a set gives a set of vectors.

    Zero is the empty sum, so `TERMS(0)` is `[]`, as `FACTORS(1)` is.

    A quotient of polynomials that cancels is the sum it cancels to, as Derive
    simplifies the argument first: `CYCLOTOMIC(2, x)` builds `(x^2 - 1)/(x - 1)`, whose
    terms are those of `x + 1`.
    """
    # Expanding and factoring are built on the conversion's heads.
    from rederive.engine.expanding import expanded_expression

    value, *rest = args
    if isinstance(value, (Vector, SetOf)):
        return _over_elements(conv, value, lambda element: _terms(conv, [element, *rest]))
    value = _released_numbers(value)
    if (
        isinstance(value, sp.Mul)
        and sp.fraction(value)[1] != 1
        and value.is_rational_function()
    ):
        value = sp.cancel(value)
    order = conv.context.order
    if rest:
        amount, variables = _amount_and_variables(rest, expanding=True)
        value = expanded_expression(value, amount, variables, order=order)
        order = _expansion_order(variables, order)
    if is_logical(value):
        if isinstance(value, (sp.And, sp.Or, sp.Not, sp.Xor, sp.Implies, sp.Equivalent)):
            normal = sp.to_dnf(value)
            disjuncts = normal.args if isinstance(normal, sp.Or) else (normal,)
            return Vector(*logical_order(disjuncts, conv.context.order))
        return Vector(value)
    if not isinstance(value, sp.Expr):
        raise TypeError("no terms")
    if value == 0:
        return Vector()
    terms = term_order(sp.Add.make_args(value), order)
    if rest:
        terms = _by_powers(terms, variables)
    return Vector(*terms)


def _by_powers(terms: Sequence[sp.Basic], variables: Sequence[str]) -> list[sp.Basic]:
    """The terms of an expansion in `variables`, those free of them taken together as
    one, the coefficient of their zeroth power, as the expansion collects every other
    power: the terms of `1 - a + b*x` in `x` are `b*x` and `1 - a`, which a Pade
    approximation program solves for by matching coefficients."""
    names = set(variables)
    if not names:
        return list(terms)
    free = [
        term
        for term in terms
        if not any(getattr(symbol, "name", None) in names for symbol in term.free_symbols)
    ]
    if len(free) < 2:
        return list(terms)
    kept = [term for term in terms if term not in free]
    return [*kept, sp.Add(*free)]


def _expansion_order(variables: Sequence[str], order: Sequence[str]) -> tuple[str, ...]:
    """The variable order for the terms of an expansion in `variables`: those first."""
    return (*variables, *(name for name in order if name not in variables))


def terms_by_expansion(
    terms: sp.Basic, arguments: Sequence[sp.Basic], order: Sequence[str]
) -> sp.Basic:
    """`TERMS(EXPAND(u, ...))`'s answer `terms` in the order `TERMS(u, ...)` gives:
    by the powers of the variables the expansion's `arguments` name, so its first term
    has the highest power of the first variable. Anything else is left as it is."""
    if not isinstance(terms, Vector) or not all(
        isinstance(term, sp.Expr) and not isinstance(term, Vector) for term in terms.args
    ):
        return terms
    try:
        _, variables = _amount_and_variables(list(arguments), expanding=True)
    except TypeError:
        return terms
    ordered = term_order(terms.args, _expansion_order(variables, order))
    return Vector(*_by_powers(ordered, variables))


def _unknown_integer(value: sp.Expr) -> bool:
    """Whether `value` is an integer that is no number: an integer variable, or a
    `FLOOR` or `CEILING`, or arithmetic on them that is an integer. (A step or a sign
    is an integer too, but one of a few known values, and a factor like any other.)"""
    if value.is_integer is not True:
        return False
    return value.has(sp.floor, sp.ceiling) or any(
        symbol.is_integer for symbol in value.free_symbols
    )


def _factors(conv: Converter, args: list) -> sp.Basic:
    """`FACTORS(u)`: the factors of `u` as base and exponent pairs, `[[b, e], ...]`, so
    that `u` is the product of the `b^e`.

    A rational number is factored into primes, in increasing order, an exponent
    negative for a prime of the denominator: `FACTORS(360)` is `[[2, 3], [3, 2], [5,
    1]]`, and a negative number starts with `[-1, 1]`. Anything else is read as the
    product it prints as, without factoring: a numeric coefficient other than 1 is the
    first pair, `[c, 1]`, then each factor gives its base and exponent in printed order
    (function kernels first, see `_factor_rank`), sums last and in reverse, those of
    the denominator last with negative exponents. A sum or any other
    non-product is the one pair `[u, 1]`. One is the empty product, so `FACTORS(1)` is
    `[]`. With an amount and variables, `u` is factored first as `FACTOR` would.
    Without them, an integer that is no number, such as `FLOOR(x)` or an integer
    variable, stands: its prime factors are unknown.

    A logical formula's factors are the clauses of its conjunctive normal form, a plain
    vector; one without `AND` is a single clause. Vectors and sets distribute, except
    that a matrix amount gives the vector of a matrix's factors (`FACTORS(A, Turing)`),
    and stands for anything else.
    """
    from rederive.engine.factoring import amount_named, factored_expression
    from rederive.engine.matrix_factoring import MATRIX_AMOUNTS, matrix_factors

    value, *rest = args
    if rest and isinstance(rest[0], sp.Symbol):
        amount = amount_named(rest[0].name)
        if amount in MATRIX_AMOUNTS:
            factors = matrix_factors(value, amount) if len(rest) == 1 else None
            if factors is None:
                raise TypeError("not a matrix")
            return Vector(*factors)
    if isinstance(value, (Vector, SetOf)):
        return _over_elements(
            conv, value, lambda element: _factors(conv, [element, *rest])
        )
    value = _released_numbers(value)
    if rest:
        amount, variables = _amount_and_variables(rest, expanding=False)
        value = _released_numbers(factored_expression(value, amount, variables))
    if is_logical(value):
        if isinstance(value, (sp.And, sp.Or, sp.Not, sp.Xor, sp.Implies, sp.Equivalent)):
            normal = sp.to_cnf(value)
            clauses = normal.args if isinstance(normal, sp.And) else (normal,)
            return Vector(*logical_order(clauses, conv.context.order))
        return Vector(value)
    if not isinstance(value, sp.Expr) or isinstance(value, StringLiteral):
        raise TypeError("no factors")
    if isinstance(value, sp.Rational):
        return _prime_pairs(value)
    if not rest and _unknown_integer(value):
        # The factors of an integer are its prime factors, which are unknown until it
        # is a number: `FACTORS(FLOOR(x))` read as the product it prints as would make
        # `FLOOR(x)` a prime.
        raise TypeError("an integer of unknown value")
    coefficient, rest_of = value.as_coeff_Mul()
    pairs = [Vector(coefficient, sp.S.One)] if coefficient != 1 else []
    above, below = [], []
    printed = factor_order(sp.Mul.make_args(rest_of), conv.context.order)
    ranked = sorted(printed, key=_factor_rank)
    # Sums, and powers of sums, come last, in the reverse of their printed order, as
    # Derive 6 lists them: `FACTORS((x^2 - 1)*(x + 3)^2, Rational)` is `[[x + 3, 2], [x
    # - 1, 1], [x + 1, 1]]` (the help's example), and the factors of `4*x^2 + 12*x*y +
    # 5*y^2` are `2*x + 5*y`, then `2*x + y`.
    sums = [factor for factor in ranked if factor.as_base_exp()[0].is_Add]
    ranked = [factor for factor in ranked if factor not in sums] + sums[::-1]
    for factor in ranked:
        base, exponent = factor.as_base_exp()
        if exponent.is_Rational and exponent.is_negative:
            below.append(Vector(base, exponent))
        elif isinstance(factor, sp.Pow):
            above.append(Vector(base, exponent))
        else:
            above.append(Vector(factor, sp.S.One))
    return Vector(*pairs, *above, *below)


def _factor_rank(factor: sp.Basic) -> int:
    """Where a factor goes among the pairs of `FACTORS`: exponentials first, then
    other function kernels, then the rest (powers of variables, sums) in printed order.
    Derive 6 lists them so: LaplaceTransforms reads the factors of
    `4*t^2*#e^(3*t)*SIN(4*t)` as the exponential, the sine and the power of `t`, in
    that order."""
    base = factor.as_base_exp()[0]
    if isinstance(base, sp.exp) or base is sp.E:
        return 0
    if isinstance(base, sp.Function):
        return 1
    return 2


def _prime_pairs(number: sp.Rational) -> Vector:
    """The prime factorization of a rational number as `[p, e]` pairs (see
    `_factors`)."""
    if number == 0:
        return Vector(Vector(sp.S.Zero, sp.S.One))
    pairs = [(-1, 1)] if number < 0 else []
    primes = dict(sp.factorint(abs(number.p)))
    for prime, exponent in sp.factorint(number.q).items():
        primes[prime] = -exponent
    pairs.extend(sorted(primes.items()))
    return Vector(*(Vector(sp.Integer(p), sp.Integer(e)) for p, e in pairs))


def _over_elements(
    conv: Converter, value: Vector | SetOf, apply: Callable[[sp.Basic], sp.Basic]
) -> sp.Basic:
    """`apply` to each element of a vector or set, giving a vector or set."""
    if isinstance(value, Vector):
        return mapped(value, apply)
    (joined,) = conv.sets_simplified([SetOf(*(apply(each) for each in value.args))])
    return joined


#: Built-ins that read a number `FACTOR` wrote as a product of primes as that product.
_FACTORED_FORM = frozenset({"FACTOR", "FACTORS", "STRING"})


def _factored_number_released(value: sp.Basic) -> sp.Basic:
    """`value` evaluated again if it is a number `FACTOR` wrote as a product of primes,
    or its negation; anything else as it is. Only the top is looked at, since looking
    through every argument of every call is too slow."""
    if isinstance(value, sp.UnevaluatedExpr) or (
        isinstance(value, sp.Mul)
        and any(isinstance(factor, sp.UnevaluatedExpr) for factor in value.args)
    ):
        return _released_numbers(value)
    return value


def _released_numbers(value: sp.Basic) -> sp.Basic:
    """`value` with a number `FACTOR` wrote as an unevaluated product of prime powers
    evaluated again, for reading it as a value."""
    if not value.has(sp.UnevaluatedExpr):
        return value
    return value.replace(
        lambda part: isinstance(part, sp.UnevaluatedExpr),
        lambda part: part.args[0].doit() if part.args[0].is_number else part.args[0],
    )


def _amount_and_variables(rest: list, expanding: bool) -> tuple[Amount, list[str]]:
    """The amount word and variables after the target of `FACTOR` or `EXPAND` (or of
    `FACTORS` and `TERMS`, which apply them): a word that names no amount the command
    takes is a variable."""
    from rederive.engine.expanding import EXPAND_AMOUNTS
    from rederive.engine.factoring import amount_named

    amounts = EXPAND_AMOUNTS if expanding else tuple(Amount)
    amount = DEFAULT_AMOUNT
    variables: list[str] = []
    for argument in rest:
        if not isinstance(argument, sp.Symbol):
            raise TypeError("not a variable")
        named = amount_named(argument.name)
        if named is not None and named in amounts and not variables:
            amount = named
        else:
            variables.append(argument.name)
    return amount, variables


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
    """`LHS(u)` and `RHS(u)`: one side of a relation, or one operand of a logical or set
    operator.

    A conjunction or disjunction of more than two operands, in printed order, reads as
    the first and the rest: `RHS(a AND b AND c)` is `b AND c`. Vectors distribute, so
    `RHS([x = 2, x = 3])` is `[2, 3]`. Any other expression `u` reads as `u = 0`, as in
    Derive: `LHS(2*x + 3)` is `2*x + 3` and `RHS(x + y)` is 0. ODE.MTH's
    `SIDE(w) := LHS(w) - RHS(w)` takes an expression for an equation this way. A
    negation, a truth value, a set or a string has no sides, so the call stands.

    A command that stands, like a `SOLVE` of no form it accepts, has no known value to
    read, so both sides are the call itself. An element selection that stands, like an
    element of a disjunction of solutions, may be a relation, so the call stands.
    """

    def handler(conv: Converter, args: list) -> sp.Basic:
        value = _one(args)
        if isinstance(value, Vector):
            return mapped(value, lambda element: handler(conv, [element]))
        if value.is_Relational:
            return value.args[index]
        if isinstance(value, (sp.And, sp.Or, sp.Xor, SetUnion, SetIntersection)):
            if isinstance(value, (sp.And, sp.Or, sp.Xor)):
                operands = printed_operands(value, conv.context.order)
            else:
                operands = list(value.args)
            if index == 0:
                return operands[0]
            rest = operands[1:]
            return rest[0] if len(rest) == 1 else type(value)(*rest)
        if isinstance(value, (sp.Implies, sp.Equivalent, SetDifference)):
            return value.args[index]
        if is_logical(value) or is_set(value) or isinstance(value, StringLiteral):
            raise TypeError("no sides")
        if _is_element(value) or isinstance(value, Subscript):
            raise TypeError("an unknown element")
        standing = isinstance(value, AppliedUndef) and type(value).__name__ in (
            COMMAND_HEADS
        )
        if index == 0 or standing:
            return value
        return sp.Integer(0)

    return handler


def _quotient(conv: Converter, args: list) -> sp.Basic:
    """`QUOTIENT(u, v)` and `QUOTIENT(u, v, x)`: the polynomial quotient of `u` by
    `v`."""
    return _divided(args)[0]


def _remainder(conv: Converter, args: list) -> sp.Basic:
    """`REMAINDER(u, v)` and `REMAINDER(u, v, x)`: the polynomial remainder of `u` by
    `v`."""
    return _divided(args)[1]


def _divided(args: list) -> tuple[sp.Basic, sp.Basic]:
    """Polynomial division in the main variable, or in the variable given as a third
    argument, with other variables as coefficients.

    `QUOTIENT(x^2 + y, x + y, y)` is 1 with remainder `x^2 - x`, dividing in `y`.

    Other variables act as field coefficients: `QUOTIENT(x*y + 1, y)` is `x + 1/y` with
    no remainder, since in `x` the divisor is a constant.

    Two numbers divide exactly: `QUOTIENT(7, 2)` is `7/2`, not 3, as in Derive (the
    manual says `u` and `v` should be rationals or polynomials).

    The main variable is the most main (by `_by_mainness`) of sympy's generators, which
    may be a kernel: `QUOTIENT(SIN(x), SIN(x)^2)` is 0 with remainder `SIN(x)`, and
    `QUOTIENT(SIN(x), x)` is `SIN(x)/x`, the divisor being a coefficient. Both match
    Derive.

    A quotient that cancels to a polynomial is that polynomial: `REMAINDER(u, (z^17 +
    1)/(z + 1))` divides by `z^16 - z^15 + ... + 1`.
    """
    args = [_cancelled_to_polynomial(arg) for arg in args[:2]] + list(args[2:])
    if len(args) == 3:
        numerator, denominator, variable = args
        if isinstance(variable, Vector) or not isinstance(variable, sp.Expr):
            raise TypeError("not a variable")
        polynomials = [_rational_polynomial(side, (variable,)) for side in args[:2]]
        if all(polynomial is not None for polynomial in polynomials):
            quotient, remainder = polynomials[0].div(polynomials[1])
            return quotient.as_expr(), remainder.as_expr()
        try:
            return sp.div(numerator, denominator, variable)
        except sp.PolynomialError as error:
            raise TypeError("not polynomials in the variable") from error
    numerator, denominator = args
    try:
        generators = sp.parallel_poly_from_expr((numerator, denominator))[1].gens
    except sp.PolificationFailed:
        # Two numbers: divide exactly.
        return numerator / denominator, sp.Integer(0)
    return sp.div(numerator, denominator, _by_mainness(list(generators))[0])


def _cancelled_to_polynomial(value: sp.Basic) -> sp.Basic:
    """`value`, or the polynomial it cancels to where it is a quotient of polynomials
    whose denominator divides its numerator."""
    if not isinstance(value, sp.Expr) or isinstance(value, Vector):
        return value
    numerator, denominator = sp.fraction(value)
    if not denominator.free_symbols:
        return value
    try:
        cancelled = sp.cancel(value)
    except sp.PolynomialError:
        return value
    if sp.fraction(cancelled)[1].free_symbols:
        return value
    return sp.expand(cancelled)


def _rational_polynomial(
    value: sp.Basic, generators: tuple[sp.Basic, ...]
) -> sp.Poly | None:
    """`value` as a polynomial in `generators` with rational coefficients, or None.

    The polynomial is built as `value` is, its monomials read off directly and a product
    of sums made by multiplying their polynomials, which is much faster than expanding
    the product as an expression: a program working in `Z[x]/(x^r - 1)` multiplies
    polynomials of thousands of terms.
    """
    if not isinstance(value, sp.Expr) or isinstance(value, Vector):
        return None
    if not value.free_symbols <= set(generators):
        return None
    parts = sp.Add.make_args(value)
    terms: dict[tuple[int, ...], sp.Rational] = {}
    compound = []
    for part in parts:
        monomial = _monomial(part, generators)
        if monomial is None:
            compound.append(part)
            continue
        exponents, coefficient = monomial
        terms[exponents] = terms.get(exponents, sp.S.Zero) + coefficient
    zero = {(0,) * len(generators): 0}
    total = sp.Poly.from_dict(terms or zero, *generators, domain=sp.QQ)
    for part in compound:
        if isinstance(part, sp.Mul):
            factors = [_rational_polynomial(f, generators) for f in part.args]
            if any(factor is None for factor in factors):
                return None
            product = factors[0]
            for factor in factors[1:]:
                product = product * factor
        elif isinstance(part, sp.Pow) and part.exp.is_Integer and part.exp >= 0:
            base = _rational_polynomial(part.base, generators)
            if base is None:
                return None
            product = base ** int(part.exp)
        else:
            return None
        total = total + product
    return total


def _monomial(
    value: sp.Basic, generators: tuple[sp.Basic, ...]
) -> tuple[tuple[int, ...], sp.Rational] | None:
    """`value` as the exponents and coefficient of a monomial in `generators`, or
    None."""
    coefficient, rest = value.as_coeff_Mul()
    if not isinstance(coefficient, sp.Rational):
        return None
    exponents = [0] * len(generators)
    for factor in sp.Mul.make_args(rest):
        if factor == 1:
            continue
        base, exponent = factor.as_base_exp()
        if base not in generators or not exponent.is_Integer or exponent < 0:
            return None
        exponents[generators.index(base)] += int(exponent)
    return tuple(exponents), coefficient


def _polynomial_gcd(conv: Converter, args: list) -> sp.Basic:
    """`POLY_GCD(u, v)`: the greatest common divisor of two polynomials.

    Numbers are constant polynomials, so `POLY_GCD(12, 18)` is 6.

    Powers of one base whose exponents differ by integers are multiples of one
    generator: `POLY_GCD(x^(n + 1), x^n)` is `x^n`, since `x^(n + 1)` is `x*x^n`.
    """
    left, right = args
    written, generators = _over_common_powers((left, right))
    if not generators:
        return sp.gcd(left, right)
    return sp.gcd(*written).xreplace(generators)


def _over_common_powers(
    sides: tuple[sp.Basic, ...],
) -> tuple[list[sp.Basic], dict[sp.Symbol, sp.Basic]]:
    """`sides` with each power of a symbolic exponent written as a power of the base
    times a generator, and the generators' values.

    The powers of one base whose exponents differ by integers share a generator, the
    power of the least exponent: `x^(n + 1)` and `x^(n - 1)` become `x^2*g` and `g`,
    with `g` standing for `x^(n - 1)`. Empty if no two powers share one.
    """
    groups: dict[tuple[sp.Basic, sp.Basic], list[sp.Pow]] = {}
    for side in sides:
        for power in side.atoms(sp.Pow):
            if power.exp.is_number:
                continue
            shift, rest = power.exp.as_coeff_Add()
            if not shift.is_Integer:
                rest, shift = power.exp, sp.Integer(0)
            groups.setdefault((power.base, rest), []).append(power)
    replacements: dict[sp.Basic, sp.Basic] = {}
    generators: dict[sp.Symbol, sp.Basic] = {}
    for (base, rest), powers in groups.items():
        if len(set(powers)) < 2:
            continue
        shifts = {power: power.exp - rest for power in powers}
        least = min(shifts.values())
        generator = sp.Dummy("g")
        generators[generator] = base ** (rest + least)
        for power, shift in shifts.items():
            replacements[power] = base ** (shift - least) * generator
    if not generators:
        return list(sides), {}
    return [side.xreplace(replacements) for side in sides], generators


def _groebner_basis(conv: Converter, args: list) -> sp.Basic:
    """`GROEBNER_BASIS(polys, vars)`: the reduced Groebner basis of the ideal the
    polynomials generate, lexicographic with `vars SUB 1 > vars SUB 2 > ...`.

    An equation `a = b` stands for `a - b`; symbols not in `vars` are coefficients. Each
    element is the primitive polynomial with integer coefficients (denominators cleared,
    content removed, not monic) whose leading coefficient is positive, and the elements
    are listed by ascending leading monomial, the one in the last variables only first.
    The unit ideal gives `[1]`, the zero ideal `[]`; a set of polynomials gives a set.

    An element that is not a polynomial in `vars` with rational coefficients, or a
    `vars` element that is not a variable, leaves the call as written.
    """
    from rederive.engine.solving import lex_basis

    given, variables = args
    if not isinstance(given, (Vector, SetOf)) or not isinstance(variables, Vector):
        raise TypeError("not polynomials and variables")
    chosen = tuple(variables.args)
    if not chosen or any(type(name) is not sp.Symbol for name in chosen):
        raise TypeError("not variables")
    if len(set(chosen)) != len(chosen):
        raise ValueError("a variable twice")
    residuals = []
    for element in given.args:
        if isinstance(element, sp.Equality):
            element = element.lhs - element.rhs
        if not isinstance(element, sp.Expr) or isinstance(element, Vector):
            raise TypeError("not a polynomial")
        if sp.fraction(sp.together(element))[1].has(*chosen):
            raise TypeError("not a polynomial")
        residuals.append(element)
    basis = lex_basis(residuals, chosen, rational=True) if residuals else []
    normalized = sorted(
        (_primitive(sp.Poly(element, *chosen)) for element in basis if element != 0),
        key=lambda polynomial: polynomial.monoms(order="lex")[0],
    )
    elements = [polynomial.as_expr() for polynomial in normalized]
    return SetOf(*elements) if isinstance(given, SetOf) else Vector(*elements)


def _primitive(polynomial: sp.Poly) -> sp.Poly:
    """`polynomial` with denominators cleared and its content removed, the content
    being the gcd of its coefficients as polynomials in the other symbols, with a
    positive leading coefficient."""
    numerator = sp.fraction(sp.together(polynomial.as_expr()))[0]
    whole = sp.Poly(numerator, *polynomial.gens)
    content = sp.gcd_list(whole.coeffs()) if whole.length() > 1 else whole.LC()
    whole = sp.Poly(sp.cancel(whole.as_expr() / content), *polynomial.gens)
    leading = whole.LC(order="lex")
    if sp.Poly(leading).LC() < 0 if leading.free_symbols else leading < 0:
        whole = -whole
    return whole


def _variables(conv: Converter, args: list) -> sp.Basic:
    """`VARIABLES(u)`: the free variables of `u`, from most main to least.

    The order is that of the variable order list, as in Derive: the listed variables
    first (`x`, `y`, `z` by default), then the others alphabetically. A fixed order is
    needed since sympy returns a set and the answer must print the same
    every time.

    The original symbols are returned, so they keep their declared domains. Strings in
    `u` follow, in code-point order: a program finds the integral sign in the `STRING`
    tree of an answer with `MEMBER?("∫", VARIABLES(...))`.

    The variables of terms an infinity absorbed in a sum in `u` are listed too (see
    `absorbed_by_infinity`).
    """
    found = _one(args).free_symbols | (_ABSORBED.get() or set())
    by_name = {symbol.name: symbol for symbol in found if type(symbol) is sp.Symbol}
    strings = sorted(
        (symbol for symbol in found if isinstance(symbol, StringLiteral)),
        key=lambda string: string.name,
    )
    ordered = main_order(by_name, conv.context.order)
    return Vector(*(by_name[name] for name in ordered), *strings)


def _dif(conv: Converter, args: list) -> sp.Basic:
    """`DIF(u, x)` and `DIF(u, x, n)`, unevaluated.

    A negative order means repeated antiderivatives (7.4, p.177): `DIF(u, x, -2)` is
    `INT(INT(u, x), x)`, since sympy has no head for a negative-order derivative.

    Vectors and matrices are differentiated element-wise and keep their shape:
    `DIF([1, 1], x)` is `[0, 0]`, where sympy would give the scalar 0. Both sides of an
    equation are differentiated: `DIF(y = x^2, x)` is `0 = 2*x`. So are the elements
    of a set.

    A one-element vector is its element: `DIF(u, [x])` is `DIF(u, x)`, which a program
    differentiating in `v SUB [k]` relies on.
    """
    if len(args) not in (2, 3):
        raise TypeError("not a derivative")
    if isinstance(args[1], Vector) and len(args[1].args) == 1:
        args = [args[0], args[1].args[0], *args[2:]]
    _variable(args[1])
    if isinstance(args[0], Vector):
        return mapped(args[0], lambda element: _dif(conv, [element, *args[1:]]))
    if isinstance(args[0], SetOf):
        return SetOf(*(_dif(conv, [element, *args[1:]]) for element in args[0].args))
    if isinstance(args[0], sp.Equality):
        if len(args) == 3 and args[2].is_Integer and args[2].is_negative:
            raise TypeError("no antiderivative of an equation")
        sides = [_dif(conv, [side, *args[1:]]) for side in args[0].args]
        return sp.Eq(*sides, evaluate=False)
    if isinstance(args[0], Relational):
        raise TypeError("no derivative of an inequality")
    if len(args) == 2:
        return sp.Derivative(args[0], args[1], evaluate=False)
    expression, variable, order = args
    if order.is_Integer and order.is_negative:
        if isinstance(expression, Relational):
            raise TypeError("no antiderivative of an equation")
        return sp.Integral(expression, *[variable] * int(-order))
    return sp.Derivative(expression, (variable, order), evaluate=False)


def _integral(conv: Converter, args: list) -> sp.Basic:
    """`INT(u, x)` and `INT(u, x, a, b)`, both unevaluated, and `INT(u, x, c)`, the
    antiderivative with the constant `c` added.

    Vectors and matrices are integrated element-wise, keeping their shape, and so are
    sets. A vector limit gives one integral per element: `INT(x^2, x, 0, [1, 2])` is
    `[1/3, 8/3]`. Two vector limits are refused, and the call stands.

    A definite integral of an equation integrates both sides. An antiderivative of one
    doesn't, since each side has its own constant, so `INT(y = x, x)` stands; with a
    constant it does, the constant added to the right side, which is how a separation
    of variables is written: `INT(y = x, x, c)` is `x*y = x^2/2 + c`. The constant is
    added to each element of a vector or set.
    """
    if len(args) not in (2, 3, 4):
        raise TypeError("not an integral")
    _variable(args[1])
    if len(args) == 3:
        return _with_constant(conv, *args)
    if isinstance(args[0], Vector):
        return mapped(args[0], lambda element: _integral(conv, [element, *args[1:]]))
    if isinstance(args[0], SetOf):
        integrals = (_integral(conv, [element, *args[1:]]) for element in args[0].args)
        return SetOf(*integrals)
    if len(args) == 2:
        if isinstance(args[0], Relational):
            raise TypeError("no antiderivative of an equation")
        return sp.Integral(args[0], args[1])
    distributed = _over_vector_limits(conv, _integral, args)
    if distributed is not None:
        return distributed
    expression, variable, low, high = args
    if low == high and not low.has(sp.oo, -sp.oo, sp.zoo, sp.nan) and not isinstance(
        expression, Relational
    ):
        # Over an interval of no width the integral is 0, whatever the integrand is
        # there: a density whose own formula divides by the width is undefined on it.
        return sp.Integer(0)
    if isinstance(expression, sp.Equality):
        return sp.Eq(
            sp.Integral(expression.lhs, (variable, low, high)),
            sp.Integral(expression.rhs, (variable, low, high)),
            evaluate=False,
        )
    if isinstance(expression, Relational):
        raise TypeError("not an equation")
    return sp.Integral(expression, (variable, low, high))


def _with_constant(
    conv: Converter, expression: sp.Basic, variable: sp.Basic, constant: sp.Basic
) -> sp.Basic:
    """`INT(u, x, c)`: the antiderivative of `u` plus `c` (see `_integral`)."""
    if isinstance(constant, (Vector, SetOf)) or isinstance(constant, Relational):
        raise TypeError("not a constant")
    if isinstance(expression, Vector):
        return mapped(
            expression, lambda element: _with_constant(conv, element, variable, constant)
        )
    if isinstance(expression, SetOf):
        return SetOf(
            *(
                _with_constant(conv, element, variable, constant)
                for element in expression.args
            )
        )
    if isinstance(expression, sp.Equality):
        return sp.Eq(
            sp.Integral(expression.lhs, variable),
            sp.Integral(expression.rhs, variable) + constant,
            evaluate=False,
        )
    if isinstance(expression, Relational):
        raise TypeError("not an equation")
    return sp.Integral(expression, variable) + constant


def _variable(variable: sp.Basic) -> None:
    """Refuse a vector, or an element not yet known, as a calculus function's variable.

    `DIF(x1*x2, [x1, x2] SUB k)` is a derivative in `x1` or `x2` once `k` is known, and
    stays a call until then, as in Derive. Taken in the unevaluated element it would be
    0.
    """
    if isinstance(variable, Vector) or _is_element(variable):
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
    if isinstance(low, Vector) and isinstance(high, Vector):
        raise TypeError("two vectors of limits")
    if isinstance(low, Vector):
        return mapped(low, lambda end: build(conv, [expression, variable, end, high]))
    if isinstance(high, Vector):
        return mapped(high, lambda end: build(conv, [expression, variable, low, end]))
    return None


def _summation(conv: Converter, args: list) -> sp.Basic:
    """`SUM(u, k, a, b)`, and `SUM(v)` over the elements of a vector or set.

    A vector or set third argument lists the values of `k`; otherwise it is the upper
    end of a range from 1, so `SUM(u, k, m)` is `SUM(u, k, 1, m)`. A vector limit gives
    one sum per element, as for `INT`: `SUM(k, k, [1, 2], 3)` is `[6, 5]`. A fifth
    argument is a step, as for `VECTOR` (see `_stepped`).

    An indefinite `SUM(u, k)` is the antidifference of `u`, deferred like the calculus
    heads, and of each element of a vector `u`: `SUM([1, 2], x)` is `[x, 2*x]`.

    Vectors are added as vectors, so the sum of a matrix is the sum of its rows,
    `SUM([[1, 2], [3, 4]])` being `[4, 6]`, and a vector summed over a range is the
    vector of its elements' sums.
    """
    if len(args) == 1 and isinstance(args[0], (Vector, SetOf)):
        return added(list(args[0].args))
    if len(args) not in (2, 3, 4, 5):
        raise TypeError("not a sum")
    _variable(args[1])
    if len(args) == 2 and isinstance(args[0], Vector):
        return mapped(args[0], lambda element: _summation(conv, [element, args[1]]))
    if len(args) == 2:
        return Antidifference(*args)
    if len(args) == 3 and isinstance(args[2], (Vector, SetOf)):
        return added(_over_values(conv, args))
    if len(args) == 5:
        return _stepped(conv, _summation, args, added)
    args = _from_one(args)
    distributed = _over_vector_limits(conv, _summation, args)
    if distributed is not None:
        return distributed
    expression, index, low, high = args
    if isinstance(expression, Vector):
        return mapped(expression, lambda e: _summation(conv, [e, index, low, high]))
    return sp.Sum(_indexed(expression, index), (index, low, high))


def _product(conv: Converter, args: list) -> sp.Basic:
    """`PRODUCT(u, k, a, b)`, and `PRODUCT(v)` over a vector's elements.

    Arguments work as for `SUM`. An indefinite `PRODUCT(u, k)` is the antiquotient of
    `u`.
    """
    if len(args) == 1 and isinstance(args[0], (Vector, SetOf)):
        return _multiplied_out(list(args[0].args))
    if len(args) not in (2, 3, 4, 5):
        raise TypeError("not a product")
    _variable(args[1])
    if len(args) == 2:
        return Antiquotient(*args)
    if len(args) == 3 and isinstance(args[2], (Vector, SetOf)):
        return _multiplied_out(_over_values(conv, args))
    if len(args) == 5:
        return _stepped(conv, _product, args, _multiplied_out)
    args = _from_one(args)
    distributed = _over_vector_limits(conv, _product, args)
    if distributed is not None:
        return distributed
    expression, index, low, high = args
    return sp.Product(_indexed(expression, index), (index, low, high))


def _multiplied_out(factors: list[sp.Basic]) -> sp.Basic:
    """The product of `factors`, 1 if there are none."""
    return multiplied(factors) if factors else sp.Integer(1)


def _stepped(
    conv: Converter,
    build: Callable[[Converter, list], sp.Basic],
    args: list,
    combine: Callable[[list[sp.Basic]], sp.Basic],
) -> sp.Basic:
    """`SUM(u, k, m, n, s)` or `PRODUCT(u, k, m, n, s)`: `k` runs `m, m + s, ...` while
    not past `n`, the index range of `VECTOR(u, k, m, n, s)`.

    Numeric ends give the terms, combined by `combine`; there are none if the range runs
    the wrong way. A step of 1 is the four-argument form. Otherwise the index is
    counted by a fresh variable `j` from 0 to `FLOOR((n - m)/s)`, with `k` written as
    `m + s*j`. A zero or symbolic step leaves the call standing.
    """
    expression, index, low, high, step = args
    if type(index) is not sp.Symbol or not step.is_number or step.is_zero is not False:
        raise TypeError("not a step")
    if low.is_number and high.is_number:
        count = sp.floor((high - low) / step) + 1
        if not count.is_Integer:
            raise TypeError("not a range")
        if count <= _STEPPED_OUT:
            values = Vector(*steps(args[2:]))
            return combine(_over_values(conv, [expression, index, values]))
    if step == 1:
        return build(conv, [expression, index, low, high])
    count = sp.Dummy("j", integer=True, nonnegative=True)
    last = sp.floor((high - low) / step)
    return build(conv, [expression.subs(index, low + step * count), count, 0, last])


#: Number of terms up to which a sum or product with a step and numeric ends is written
#: out; a longer one with a step of 1 is the four-argument form.
_STEPPED_OUT = 20000


def _indexed(expression: sp.Basic, index: sp.Basic) -> sp.Basic:
    """`expression` with each subscript by `index` made a `Subscript` of it.

    `x SUB i` is otherwise a symbol of its own, which a sum over `i` would take for a
    constant: `SUM(x SUB i, i, 1, n)` is a sum of `n` different terms, not `n*x SUB i`.
    """
    if type(index) is not sp.Symbol or not isinstance(expression, sp.Basic):
        return expression
    suffix = f" SUB {index.name}"
    found = {
        symbol: Subscript(
            sp.Symbol(symbol.name[: -len(suffix)], **symbol.assumptions0), index
        )
        for symbol in expression.free_symbols
        if isinstance(symbol, sp.Symbol)
        and symbol.name.endswith(suffix)
        and " SUB " not in symbol.name[: -len(suffix)]
    }
    return expression.xreplace(found) if found else expression


def _from_one(args: list) -> list:
    """Sum or product arguments with a lower limit of 1 filled in if omitted.

    A third argument that is a standing call of a built-in giving a vector, as
    `FACTORS(n)` stands in a definition, is the vector of values to come, not an upper
    limit, so the call stands.
    """
    if len(args) == 3:
        last = args[2]
        if isinstance(last, AppliedUndef) and type(last).__name__ in _COLLECTING:
            raise TypeError("values still to come")
        return [*args[:2], sp.Integer(1), last]
    return args


#: Built-ins giving a vector that stand while their argument is still to come.
_COLLECTING = frozenset({"FACTORS", "TERMS", "NAME_TO_CODES"})


def _over_values(conv: Converter, args: list) -> list[sp.Basic]:
    """The body of a sum or product evaluated at each listed index value."""
    expression, index, values = args
    if type(index) is not sp.Symbol:
        raise TypeError("not a variable")
    return [settled(expression.subs(index, value)) for value in values.args]


def _subst_diff(conv: Converter, args: list) -> sp.Basic:
    """`SUBST_DIFF(u, x, a, b)`: `u` at `x = b` minus `u` at `x = a`, the step that
    finishes a definite integral or sum.

    Each end is substituted, and where `u` is undefined there, or the end is infinite,
    the value is the limit approached from the side of the other end (see `_at_end`).
    Vectors and sets are taken element by element, and an equation side by side.
    """
    if len(args) != 4:
        raise TypeError("not a difference")
    expression, variable, low, high = args
    if type(variable) is not sp.Symbol:
        raise TypeError("not a variable")
    if isinstance(expression, Vector):
        return mapped(
            expression, lambda element: _subst_diff(conv, [element, *args[1:]])
        )
    if isinstance(expression, SetOf):
        return SetOf(
            *(_subst_diff(conv, [element, *args[1:]]) for element in expression.args)
        )
    if isinstance(expression, sp.Equality):
        sides = [_subst_diff(conv, [side, *args[1:]]) for side in expression.args]
        return sp.Eq(*sides, evaluate=False)
    if not isinstance(expression, sp.Expr) or isinstance(low, Vector | SetOf):
        raise TypeError("not a value")
    if isinstance(high, Vector | SetOf):
        raise TypeError("not a value")
    return _at_end(expression, variable, high, low) - _at_end(
        expression, variable, low, high
    )


def _at_end(
    expression: sp.Expr, variable: sp.Symbol, end: sp.Basic, other: sp.Basic
) -> sp.Basic:
    """The value of `expression` at `variable = end`, for `SUBST_DIFF`: substituted, or
    where that is undefined or `end` infinite, the limit from the side of `other`.

    A limit at a finite end needs the side known: the call stands if the order of the
    ends is not.
    """
    if end.is_infinite or end.has(sp.oo, -sp.oo, sp.zoo):
        return sp.Limit(expression, variable, end)
    try:
        value = expression.subs(variable, end)
    except Exception:
        value = sp.nan
    if not value.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
        return value
    side = (other - end).is_positive
    if side is None:
        side = False if (other - end).is_negative else None
    if side is None:
        raise ValueError("the side is not known")
    return sp.Limit(expression, variable, end, dir="+" if side else "-")


def _taylor(conv: Converter, args: list) -> sp.Basic:
    """`TAYLOR(u, x, a, n)`, unevaluated; the pipeline decides when to compute it.

    Three arguments are `TAYLOR(u, x, n)`, about 0. Both sides of an equation are
    expanded, and the elements of a set. A vector of variables stands: there is no
    multivariate expansion.
    """
    if len(args) == 3:
        args = [args[0], args[1], sp.Integer(0), args[2]]
    if len(args) != 4:
        raise TypeError("not a Taylor polynomial")
    expression, variable, *rest = args
    _variable(variable)
    if isinstance(expression, sp.Equality):
        sides = [Taylor(side, variable, *rest) for side in expression.args]
        return sp.Eq(*sides, evaluate=False)
    if isinstance(expression, Relational):
        raise TypeError("not an equation")
    if isinstance(expression, SetOf):
        return SetOf(*(Taylor(element, variable, *rest) for element in expression.args))
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

    A test sympy doesn't take as a condition, like `PRIME(m)` for a variable `m`, also
    leaves an inert head, which shows the test as written. A conditional whose values
    are all the same is that value, whatever its test: `IF(? >= ?, ?, ?)` is `?`.
    """
    if len(args) >= 3 and all(arg == args[1] for arg in args[2:]):
        return args[1]
    if not isinstance(as_condition(args[0]), Boolean) or args[0].has(PlusMinus):
        # A `Piecewise` on a test with a plus-or-minus never finishes building.
        return conv.opaque("IF", args)
    if len(args) == 1:
        return sp.Piecewise(
            (sp.Integer(1), as_condition(args[0])), (sp.Integer(0), sp.true)
        )
    if len(args) == 2:
        test, then = args
        return sp.Piecewise((then, as_condition(test)))
    test, then, otherwise = args
    return sp.Piecewise((then, as_condition(test)), (otherwise, sp.true))


def authored_conditionals(
    node: Node,
    context: Context,
    simplify_test: Callable[[sp.Basic], sp.Basic] | None = None,
    retested: Mapping[sp.Basic, sp.Basic] | None = None,
) -> dict[sp.Basic, Node]:
    """Every conditional in `node`, as a tree keyed by its converted expression.

    Derive shows an undecidable conditional exactly as typed: the test isn't rearranged
    and the arms aren't evaluated, so `IF(x > 0, 2 + 3, 4 + 5)` keeps its arithmetic.
    The converted expression has already lost that (`2 + 3` is 5), so the tree is
    given to the printer, which puts it in wherever that expression appears in an
    answer.

    The tree always converts back to its key, so this only changes spelling, never
    value. The one imprecision is that two conditionals converting to the same
    expression share one spelling.

    Keying by expression rather than splicing means commands run on the same input as
    before. A conditional the evaluator decided has no key in the answer (only the
    chosen arm remains), and one it reshaped (typically an argument substituted into a
    function's `IF`) no longer matches and prints as computed. Both match Derive.

    The authored subtree is converted before any definitions are substituted, since
    the goal is to show what the author wrote; if that differs from what the command
    converted, there is no match and the answer prints as computed.

    A test with calculus in it is shown worked out (see `with_worked_test`), and
    the arms still as written: `IF(DIF(x^2, x) = 0, 1 + 1, 2 + 2)` is `IF(2*x = 0, 1 +
    1, 2 + 2)`. Given `simplify_test`, a test is shown simplified by it, the arms still
    as written: `IF(t < -SQRT(1/2), t, 1 + 1)` is `IF(t < -SQRT(2)/2, t, 1 + 1)`.

    `retested` maps a conditional as written to the one the evaluation shows with its
    test as computed (see `Store.retested`), whose test is shown so, simplified by
    `simplify_test`, with the arms as written: with `f(x) := x^2`, `IF(f(a) = 4, 1 + 1,
    2)` is `IF(a^2 = 4, 1 + 1, 2)`.
    """
    authored: dict[sp.Basic, Node] = {}
    converter = Converter(context)
    # A conditional may be in a definition, whose parameters the built-ins that answer
    # from the form of a value leave standing.
    converter.holding_forms = True
    for found in _conditional_nodes(node):
        try:
            converted = converter.convert(found)
        except Exception:
            continue
        if not is_conditional(converted):
            continue
        if converted not in authored:
            shown = None
            if simplify_test is not None:
                shown = _with_test_simplified(found, converter, simplify_test)
            authored[converted] = shown or found
        worked = _with_test_worked(found, converted, context)
        if worked is not None:
            authored.setdefault(*worked)
        if retested and converted in retested:
            shown = retested[converted]
            tested = _with_test_computed(found, shown, context, simplify_test)
            if tested is not None:
                authored.setdefault(shown, tested)
    return authored


def _with_test_computed(
    found: Node,
    shown: sp.Basic,
    context: Context,
    simplify_test: Callable[[sp.Basic], sp.Basic] | None,
) -> Node | None:
    """The authored `IF` call `found` with the test of the conditional `shown`,
    simplified by `simplify_test` if that leaves a relation, and its arms as written."""
    try:
        test = shown.args[0].cond if isinstance(shown, sp.Piecewise) else shown.args[0]
        if simplify_test is not None and isinstance(test, Relational):
            simplified = simplify_test(test)
            if isinstance(simplified, Relational):
                test = simplified
        return _with_test(found, tree(test, context))
    except Exception:
        return None


def _with_test(found: Node, test: Node) -> Node:
    """The authored `IF` call `found` with the test `test` and its arms as written."""
    head, _, *arms = found.children
    return Node(Kind.CALL, found.start, found.end, (head, test, *arms))


def _with_test_simplified(
    found: Node,
    converter: Converter,
    simplify_test: Callable[[sp.Basic], sp.Basic],
) -> Node | None:
    """The authored `IF` call `found` with its test simplified, or None if simplifying
    changes nothing or decides it."""
    if found.kind is not Kind.CALL or len(found.children) < 3:
        return None
    if found.children[1].kind is not Kind.REL:
        return None
    try:
        simplified = simplify_test(converter.convert(found.children[1]))
    except Exception:
        return None
    if not isinstance(simplified, Relational):
        return None
    test = tree(simplified, converter.context)
    shown = write_expression(test)
    if shown == write_expression(found.children[1]):
        return None
    return _with_test(found, test)


def _with_test_worked(
    found: Node, converted: sp.Basic, context: Context
) -> tuple[sp.Basic, Node] | None:
    """The authored conditional `found` with its test worked out, and its tree, or None
    if its test has no calculus to work out."""
    if not converted.has(*CALCULUS_HEADS) or len(found.children) < 3:
        return None
    try:
        worked = with_worked_test(converted, lambda head: head.doit())
        if worked == converted:
            return None
        test = worked.args[0].cond if isinstance(worked, sp.Piecewise) else worked.args[0]
        return worked, _with_test(found, tree(test, context))
    except Exception:
        return None


def with_worked_test(
    conditional: sp.Basic, work: Callable[[sp.Basic], sp.Basic]
) -> sp.Basic:
    """`conditional` with the outermost calculus heads in its test replaced by what
    `work` gives for them, and its arms as they are: `IF(DIF(y^2, y) = 0, ...)`
    becomes `IF(2*y = 0, ...)`, as Derive shows an undecided conditional.

    The test is rebuilt around the answers as usual, except that a relation and the
    conditional are rebuilt as they stand, since sympy would reduce `2*y = 0` to `y =
    0`. A head whose answer still contains it, such as a sum that converges only
    under a condition, is left as it is, rather than nest a copy of that conditional
    in itself.
    """

    def worked(test: sp.Basic) -> sp.Basic:
        heads = [
            found
            for found in sp.preorder_traversal(test)
            if isinstance(found, CALCULUS_HEADS)
        ]
        outermost = [h for h in heads if not any(h != o and o.has(h) for o in heads)]
        answers = {}
        for head in outermost:
            try:
                answer = work(head)
            except Exception:
                continue
            if answer != head and not answer.has(head):
                answers[head] = answer
        return _replaced_in(test, answers) if answers else test

    try:
        if isinstance(conditional, sp.Piecewise):
            pairs = [ExprCondPair(arm, worked(test)) for arm, test in conditional.args]
            with sp.evaluate(False):
                return sp.Piecewise(*pairs)
        if is_conditional(conditional) and conditional.args:
            test, *arms = conditional.args
            with sp.evaluate(False):
                return conditional.func(worked(test), *arms)
    except Exception:
        pass
    return conditional


def _replaced_in(value: sp.Basic, answers: dict[sp.Basic, sp.Basic]) -> sp.Basic:
    """`value` with `answers` substituted, rebuilt around them as usual except for
    relations and conditionals (see `with_worked_test`)."""
    if value in answers:
        return answers[value]
    if not value.args or not any(value.has(found) for found in answers):
        return value
    args = [_replaced_in(arg, answers) for arg in value.args]
    if isinstance(value, Relational | sp.Piecewise | ExprCondPair) or is_conditional(
        value
    ):
        with sp.evaluate(False):
            return value.func(*args)
    return value.func(*args)


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

    A vector is zero when each of its elements is, so `IF([0, 0], 1, 2, 3)` is 1. A
    predicate still waiting for a value, like `PRIME(m)`, is a test as written, and is
    returned as is although sympy doesn't see it as a condition (see `_conditional`).
    """
    if isinstance(test, (Boolean, Logical)):
        return _test(test)
    if isinstance(test, Vector):
        return sp.And(*(as_condition(element) for element in test.args))
    if isinstance(test, AppliedUndef) and type(test).__name__ in _PREDICATES:
        return test
    return sp.Eq(test, 0)


#: Built-ins answering true or false, which may stay a call until their argument is
#: known.
_PREDICATES = frozenset({"PRIME", "NUMBER", "PRIME?", *PREDICATES})


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
    rows = [Vector(*(_as_written(value) for value in args))]
    for assignment in product([sp.true, sp.false], repeat=len(variables)):
        written = dict(zip(variables, assignment, strict=True))
        values = [_decided(expression, written) for expression in expressions]
        rows.append(Vector(*assignment, *values))
    return Vector(*rows)


#: Each sympy logical head and its operator name, for building inert `Logical`
#: headings that Simplify leaves alone.
_HEADINGS = {
    sp.And: "AND",
    sp.Or: "OR",
    sp.Xor: "XOR",
    sp.Implies: "IMP",
    sp.Equivalent: "IFF",
    sp.Not: "NOT",
}


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
    return sp.Subs(expression, tuple(_elements(variables)), tuple(_elements(points)))


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
            mapped(variables, lambda _: sp.Integer(0))
            if isinstance(variables, Vector)
            else sp.Integer(0)
        )
        args = [*args, point]
    whatever = conv.context.branch is Branch.ANY
    if isinstance(args[1], Vector):
        return _iterated_limit(args, whatever)
    if len(args) == 3:
        return _approached(*args, "+-", whatever)
    expression, variable, point, side = args
    if not side:
        return _approached(expression, variable, point, "+-", whatever)
    direction = "+" if side > 0 else "-"
    return _approached(expression, variable, point, direction, whatever)


def _approached(
    expression: sp.Basic,
    variable: sp.Basic,
    point: sp.Basic,
    direction: str,
    whatever: bool = False,
) -> sp.Basic:
    """One limit as a sympy `Limit`, or a `MovingLimit` substitution.

    Vectors are handled element-wise, at any depth, since sympy has no limits of
    vectors. SOLVE.MTH's `NEWTON_AUX` relies on this, substituting the current iterate
    into a whole augmented matrix with `LIM(a, x, xk)`. A relation takes the limit of
    each side, as Derive 3.14 does: `LIM(SIN(x)/x = c, x, 0)` is `1 = c`.
    """
    if isinstance(expression, Vector):
        return mapped(
            expression,
            lambda element: _approached(element, variable, point, direction, whatever),
        )
    if isinstance(expression, Relational):
        return expression.func(
            _approached(expression.lhs, variable, point, direction, whatever),
            _approached(expression.rhs, variable, point, direction, whatever),
        )
    if isinstance(expression, (sp.And, sp.Or)) and all(
        isinstance(part, Relational) for part in expression.args
    ):
        # A chain `c = c = u` is the conjunction of its links, each taken alike.
        return expression.func(
            *(
                _approached(part, variable, point, direction, whatever)
                for part in expression.args
            )
        )
    if point.has(variable):
        moving = AnyMovingLimit if whatever else MovingLimit
        return moving(expression, variable, point)
    return sp.Limit(expression, variable, point, dir=direction)


def _iterated_limit(args: list, whatever: bool = False) -> sp.Basic:
    """`LIM(u, [x, y], [a, b])`: the limit in `x`, then in `y`.

    The first variable is innermost, so the outer limit applies to the inner result.
    All limits are left for evaluation.

    The variables are taken one after another, as in Derive, so a later point may
    mention an earlier variable, and an earlier point a later one, which then moves
    too: `LIM(x + 2*y, [x, y], [x*y, x])` is `x*(x + 2)`, and `LIM(x + y, [x, y],
    [y, 2])` is 4.
    """
    expression, variables, points = args
    names = _elements(variables)
    values = _elements(points)
    if len(names) != len(values):
        raise ValueError("not a point for every variable")
    for name, value in zip(names, values, strict=True):
        expression = _approached(expression, name, value, "+-", whatever)
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
    return _square(value).det()


def _trace(conv: Converter, args: list) -> sp.Basic:
    value = _one(args)
    if _symbolic_matrix(value):
        return sp.Trace(value).doit()
    return _square(value).trace()


def _square_matrix(value: sp.Basic) -> sp.ImmutableMatrix | None:
    """`value` as a square matrix of expressions, or None if it isn't one."""
    matrix = matrix_of(value)
    return matrix if matrix is not None and matrix.is_square else None


def _square(value: sp.Basic) -> sp.ImmutableMatrix:
    """`value` as a square matrix of expressions, for `DET` and `TRACE`."""
    matrix = _linear(value)
    if not matrix.is_square:
        raise TypeError("not a square matrix")
    return matrix


#: The base the digits of an integer taken as a sequence are in: the output base in
#: force, so that `DIM(74)` counts the seven digits `1001010` shows under `OutputBase :=
#: Binary`. Set around each built-in call (see `Converter.call` and `digits_in`).
_DIGIT_BASE: contextvars.ContextVar[int] = contextvars.ContextVar(
    "digit_base", default=10
)


def _digit_base(context: Context) -> int:
    """The output base `context` shows integers in."""
    setting = context.setting_values.get("OutputBase", 10)
    return BASES.get(setting, 10) if isinstance(setting, str) else int(setting)


def _digits(number: int) -> str:
    """The digits of a nonnegative integer in `_DIGIT_BASE`, upper-case letters above
    9."""
    base = _DIGIT_BASE.get()
    if base == 10:
        return str(number)
    text = ""
    while True:
        number, digit = divmod(number, base)
        text = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[digit] + text
        if not number:
            return text


def digits_in[T](context: Context, compute: Callable[[], T]) -> T:
    """`compute()` with integers taken as sequences of their digits in the output base
    of `context`."""
    base = _digit_base(context)
    if base == _DIGIT_BASE.get():
        return compute()
    token = _DIGIT_BASE.set(base)
    try:
        return compute()
    finally:
        _DIGIT_BASE.reset(token)


@dataclass(frozen=True)
class _Sequence:
    """A vector, string or integer taken as a sequence, as the vector functions take
    them: a vector's elements, a string's characters, or the digits of an integer's
    absolute value in the output base (see `_DIGIT_BASE`).

    `items` are the elements of a vector, and the characters of the string or of the
    digits otherwise. An integer's sign is not a digit: `negative` carries it over to
    each digit selected and to an integer rebuilt from digits.
    """

    kind: str
    items: tuple
    negative: bool = False
    #: The vector whose elements `items` are, for a vector.
    source: Vector | None = None

    def element(self, place: int) -> sp.Basic:
        """The element at a zero-based place: a one-character string, a digit with the
        integer's sign."""
        item = self.items[place]
        if self.kind == "vector":
            return item
        if self.kind == "string":
            return StringLiteral(item)
        digit = sp.Integer(int(item, _DIGIT_BASE.get()))
        return -digit if self.negative else digit

    def rebuilt(self, items: Sequence, added: Sequence = ()) -> sp.Basic:
        """A sequence of this kind with these items, which are its own items and those
        `added`. Rebuilt from digits, an integer drops leading zeros; raises if there
        are no digits, which no integer has."""
        if self.kind == "vector":
            if self.source is None:
                return Vector(*items)
            return vector_from(items, [self.source], added)
        if self.kind == "string":
            return StringLiteral("".join(items))
        if not items:
            raise ValueError("no digits")
        number = int("".join(items), _DIGIT_BASE.get())
        return sp.Integer(-number if self.negative else number)


def _as_sequence(value: sp.Basic) -> _Sequence | None:
    """`value` as a sequence, or None if it is none of a vector, string or integer."""
    if isinstance(value, Vector):
        return _Sequence("vector", value.args, source=value)
    if isinstance(value, StringLiteral):
        return _Sequence("string", tuple(value.name))
    if isinstance(value, sp.Integer):
        return _Sequence("integer", tuple(_digits(abs(int(value)))), bool(value < 0))
    return None


def _as_sequence_of(value: sp.Basic) -> _Sequence:
    """`value` as a sequence; raises if it isn't one."""
    found = _as_sequence(value)
    if found is None:
        raise TypeError("not a sequence")
    return found


def _text(value: sp.Basic) -> str | None:
    """The characters a string or integer contributes when combined with a string: its
    text, or its digits with the sign. None for anything else."""
    if isinstance(value, StringLiteral):
        return value.name
    if isinstance(value, sp.Integer):
        return ("-" if value < 0 else "") + _digits(abs(int(value)))
    return None


def _joined(values: Sequence[sp.Basic]) -> sp.Basic:
    """Strings and integers concatenated: a string if any of them is one, with each
    integer written in its digits, else the integer of all the digits, with the sign of
    the first. Beside a string, an empty vector is the empty string, as a program
    collecting text in a list that starts as `[]` relies on: `ADJOIN([], " a")` is
    `" a"`. Raises if anything else is among them."""
    strings = any(isinstance(value, StringLiteral) for value in values)
    texts = [
        "" if strings and isinstance(value, Vector) and not value.args else _text(value)
        for value in values
    ]
    if not values or any(text is None for text in texts):
        raise TypeError("not strings or integers")
    if strings:
        return StringLiteral("".join(texts))  # type: ignore[arg-type]
    digits = "".join(_digits(abs(int(value))) for value in values)
    number = int(digits, _DIGIT_BASE.get())
    return sp.Integer(-number if values[0] < 0 else number)


def _splicing(value: sp.Basic, target: sp.Basic) -> tuple[_Sequence, tuple]:
    """`target` as a sequence, and the items `value` puts into it.

    In a vector `value` is one element. In a string or integer it contributes all its
    characters or digits, and if either is a string, both are read as text (see
    `_joined`). Raises if `value` can't go into `target`."""
    sequence = _as_sequence_of(target)
    if sequence.kind == "vector":
        return sequence, (value,)
    if sequence.kind == "string" or isinstance(value, StringLiteral):
        inserted = _text(value)
        if inserted is None:
            raise TypeError("not a string")
        return _as_sequence_of(StringLiteral(_text(target))), tuple(inserted)
    if not isinstance(value, sp.Integer):
        raise TypeError("not an integer")
    return sequence, tuple(str(abs(int(value))))


def _dimension(conv: Converter, args: list) -> sp.Basic:
    """The number of elements in a vector, which for a matrix is its rows, in a set, of
    characters in a string or of digits in an integer.

    Vectors of anything count: the shipped libraries branch on the number of `SOLVE`
    solutions, and no solutions is an empty vector rather than an error.
    """
    value = _one(args)
    if is_set(value):
        (value,) = conv.sets_simplified([value])
    if isinstance(value, SetOf):
        return size(value)
    if isinstance(value, Vector):
        return sp.Integer(value.size())
    return sp.Integer(len(_as_sequence_of(value).items))


def _first(conv: Converter, args: list) -> sp.Basic:
    """`FIRST(v)`: the first element of a sequence, or of a set in its order. The first
    element of an empty vector is undefined, as `[] SUB 1` is."""
    value = _one(args)
    if is_set(value):
        (value,) = conv.sets_simplified([value])
    if isinstance(value, SetOf):
        return value.args[0]
    if isinstance(value, Vector):
        if not value.size():
            return sp.nan
        return value.item(0)
    sequence = _as_sequence_of(value)
    if not sequence.items:
        raise ValueError("empty")
    return sequence.element(0)


def _rest(conv: Converter, args: list) -> sp.Basic:
    """`REST(v)`, all but the first element, and `REST(v, n)`, all but the first `n`,
    or the last `-n` if `n` is negative; of the same kind as `v`.

    A set loses its first elements in its order. The rest of an empty vector, set or
    string is empty, as a program that walks a list with `v := REST(v)` until `v = []`
    relies on."""
    value, *rest = args
    if len(rest) > 1:
        raise TypeError("too many arguments")
    count = rest[0] if rest else sp.Integer(1)
    if not isinstance(count, sp.Integer):
        raise TypeError("not a count")
    if is_set(value):
        (value,) = conv.sets_simplified([value])
    if isinstance(value, Vector):
        length = value.size()
        if not length:
            return value
        number = int(count)
        if number >= 0:
            kept = value.sliced(min(number, length), length)
        else:
            kept = value.sliced(0, max(length + number, 0))
        return _shared(kept, value, [])
    items: tuple
    if isinstance(value, SetOf):
        items = value.args
    else:
        sequence = _as_sequence_of(value)
        items = sequence.items
    if not items:
        return value
    number = int(count)
    kept = items[number:] if number >= 0 else items[: max(len(items) + number, 0)]
    return SetOf(*kept) if isinstance(value, SetOf) else sequence.rebuilt(kept)


def _adjoined(conv: Converter, args: list) -> sp.Basic:
    """`ADJOIN(u, v)`: `u` put before the elements of the vector `v`, added to the set
    `v`, or put before the string or integer `v` as `APPEND` would."""
    value, target = args
    if isinstance(target, Vector):
        return _shared(target.adjoined(value), target, [value])
    if is_set(target):
        (target,) = conv.sets_simplified([target])
    if isinstance(target, SetOf):
        (joined,) = conv.sets_simplified([SetOf(value, *target.args)])
        return joined
    if _text(target) is None:
        raise TypeError("not a sequence")
    return _joined([value, target])


def _power_set(conv: Converter, args: list) -> sp.Basic | None:
    """`POWER_SET(s)`, all the subsets of `s`, and `POWER_SET(s, n)`, those of `n`
    elements (see `sets.power_set`)."""
    if len(args) not in (1, 2):
        return None
    return power_set(*conv.sets_simplified(args))


def _element_of(conv: Converter, args: list) -> sp.Basic:
    """`ELEMENT(v, i)`, `ELEMENT(m, i, j)`, and so on for any number of dimensions:
    `v SUB i`, `m SUB i SUB j` (see `_element_at`).

    Each index selects from the previous result: `ELEMENT(q, 2, 1, 2)` reads a number
    from a vector of matrices. Vectors of relations work too, as the shipped ODE library
    uses in `RHS((SOLVE(z, y)) SUB 1)`.
    """
    value, *indices = args
    if not indices:
        raise ValueError("no index")
    for index in indices:
        value = _element_at(value, index)
        if value is None:
            raise ValueError("not an index")
    return value


def _element_at(value: sp.Basic, index: sp.Basic) -> sp.Basic | None:
    """`value SUB index`: an element of a vector, string or integer, or a selection of
    them for a vector of indices (see `_picked`), or the value of a key in a set of
    pairs (see `_looked_up`). None if `value` is none of these or `index` isn't one of
    its positions.

    A string's element is a one-character string and a selection a string. An
    integer's element is a digit and a selection the integer of those digits, both
    with the integer's sign.
    """
    if isinstance(value, SetOf):
        return _looked_up(value, index)
    sequence = _as_sequence(value)
    if sequence is None:
        return None
    if sequence.kind == "vector":
        return _picked(value, index, beyond=True)
    count = len(sequence.items)
    if isinstance(index, Vector):
        places = [place_of(each, count) for each in index.args]
        if None in places:
            return None
        try:
            return sequence.rebuilt([sequence.items[p] for p in places if p is not None])
        except ValueError:
            return None
    place = place_of(index, count)
    return None if place is None else sequence.element(place)


def _looked_up(collection: SetOf, key: sp.Basic) -> sp.Basic | None:
    """`s SUB k` of a set of pairs, a function given by its graph: the second element
    of the one element of `s` that is a vector starting with `k`. None if there is no
    such element or more than one."""
    found = [
        element.args[1]
        for element in collection.args
        if isinstance(element, Vector) and len(element.args) >= 2
        if element.args[0] == key
    ]
    return found[0] if len(found) == 1 else None


def printed_operands(value: sp.Basic, order) -> list[sp.Basic]:
    """The operands of a conjunction, disjunction or exclusive or in the order they
    print: a conjunction kept in order as given, any other in the order list's order."""
    if getattr(value, "in_order", False):
        return list(value.args)
    return logical_order(value.args, order)


def place_of(index: sp.Basic, count: int) -> int | None:
    """The zero-based place of `index` among `count` elements, counting from the right
    if it is negative; None if it isn't an integer or out of range."""
    if not isinstance(index, sp.Integer):
        return None
    number = int(index)
    if 1 <= number <= count:
        return number - 1
    if -count <= number <= -1:
        return count + number
    return None


def _picked(value: sp.Basic, index: sp.Basic, beyond: bool = False) -> sp.Basic | None:
    """`value SUB index` as Derive 6 indexes: an integer counts from the left, or from
    the right if negative, and a vector of indices selects one element for each, so a
    matrix of indices gives a matrix. None if `value` isn't a vector or an index isn't
    one of its positions.

    With `beyond`, an element of the empty vector, at a single integer index other
    than 0, is `?`, as Derive gives `[] SUB 1`. Beyond the end of any other vector the
    access stands: Derive's Bezier programs multiply `p SUB (n + 1)` by 0 at `j = n`.
    """
    if isinstance(index, Vector):
        picked = [_picked(value, each) for each in index.args]
        return None if any(each is None for each in picked) else Vector(*picked)
    if value is sp.nan:
        return sp.nan
    if not isinstance(value, Vector):
        return None
    place = place_of(index, value.size())
    if beyond and not value.size() and isinstance(index, sp.Integer) and index != 0:
        return sp.nan
    return None if place is None else value.item(place)


def _deeper(value: sp.Basic, index: sp.Basic, levels: int) -> sp.Basic | None:
    """`value SUB SUB index` with `levels` more `SUB`s than one: the index applied that
    many levels down, so `A SUB SUB j` is column `j` of a matrix. None if an element on
    the way isn't a vector or hasn't the index."""
    if not levels:
        return _picked(value, index)
    if not isinstance(value, Vector):
        return None
    parts = [_deeper(element, index, levels - 1) for element in value.args]
    return None if any(part is None for part in parts) else Vector(*parts)


def _rows(value: sp.Basic, index: sp.Basic, columns: bool) -> sp.Basic | None:
    """`A ROW i` or `A COL j` of a matrix `A`, a row or column or, for a vector of
    indices, a matrix of them. None if `A` isn't a matrix."""
    if not is_matrix(value):
        return None
    return _deeper(value, index, 1) if columns else _picked(value, index)


def _dot_product(left: sp.Basic, right: sp.Basic) -> sp.Basic | None:
    """`u DOTPRODUCT v`, the sum of each element of `u` times the conjugate of `v`'s, for
    two vectors of one length whose elements aren't vectors."""
    if not (isinstance(left, Vector) and isinstance(right, Vector)):
        return None
    if len(left.args) != len(right.args):
        return None
    pairs = list(zip(left.args, right.args, strict=True))
    if any(isinstance(element, Vector) for pair in pairs for element in pair):
        return None
    return sp.Add(*(sp.expand(u * sp.conjugate(v)) for u, v in pairs))


def _cross_product(left: sp.Basic, right: sp.Basic) -> sp.Basic | None:
    """`u CROSSPRODUCT v`: the cross product of two vectors of three elements, or the
    scalar `u1*v2 - u2*v1` of two of two."""
    if not (isinstance(left, Vector) and isinstance(right, Vector)):
        return None
    try:
        return _cross(None, [left, right])  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _subset(left: sp.Basic, right: sp.Basic) -> sp.Basic | None:
    """`u SUBSET v` of two explicit sets: whether each element of `u` is one of `v`,
    compared as simplified expressions, as `MEMBER?` compares them: `{a, b} SUBSET {a,
    c}` and `{x} SUBSET {1}` are `false`."""
    if not (isinstance(left, SetOf) and isinstance(right, SetOf)):
        return None
    return _truth(set(left.args) <= set(right.args))


def _deleted(conv: Converter, args: list) -> sp.Basic:
    """`DELETE(v, n)`: the sequence `v` without its `n`th element, counting from the
    right if `n` is negative; `DELETE(v, w)` without the elements at each index of the
    vector or set `w`.

    A matrix's elements are its rows, so this deletes a row. VECTOR.MTH's `MINOR`
    relies on this: delete a row, transpose, delete a row again, transpose back.

    `n` defaults to 1, as for the other element functions. An index past either end
    designates no element, so nothing is deleted: the Josephus program of the newsletter
    deletes element `DIM(v)` of a vector one shorter, and Derive's recorded `jo(41, 3)`
    is 31, the survivor, only so. A zero, fractional or symbolic index leaves the call
    standing.
    """
    target, *rest = args
    if len(rest) > 1:
        raise TypeError("too many arguments")
    sequence = _as_sequence_of(target)
    index = rest[0] if rest else sp.Integer(1)
    count = len(sequence.items)
    indices = index.args if isinstance(index, (Vector, SetOf)) else (index,)
    indices = tuple(
        each
        for each in indices
        if not (isinstance(each, sp.Integer) and abs(int(each)) > count)
    )
    places = {place_of(each, count) for each in indices}
    if None in places:
        raise ValueError("not an index")
    kept = [item for place, item in enumerate(sequence.items) if place not in places]
    return sequence.rebuilt(kept)


def _replaced(conv: Converter, args: list) -> sp.Basic:
    """`REPLACE(u, v, n)`: the sequence `v` with its `n`th element replaced by `u`,
    counting from the right if `n` is negative.

    New value first, then the sequence, as in the manual and `INSERT`. `n` defaults to
    1. A string or integer `u` replaces one character or digit by all of its own (see
    `_splicing`). An integer `n` that designates no element (0, or past either end)
    replaces nothing, as `DELETE` deletes nothing there: a program walking a row index
    down past the first row leaves the matrix as it is. A fractional or symbolic `n`
    leaves the call standing.
    """
    value, target, *rest = args
    if len(rest) > 1:
        raise TypeError("too many arguments")
    sequence, items = _splicing(value, target)
    index = rest[0] if rest else sp.Integer(1)
    place = place_of(index, len(sequence.items))
    if place is None and isinstance(index, sp.Integer):
        return target
    if place is None:
        raise ValueError("not an index")
    whole = sequence.items
    return sequence.rebuilt(whole[:place] + items + whole[place + 1 :], items)


def _inserted(conv: Converter, args: list) -> sp.Basic:
    """`INSERT(u, v, n)`: the sequence `v` with `u` inserted before its `n`th element.

    Arguments are ordered as for `REPLACE`, and `n` defaults to 1. One past the end
    appends: `INSERT(d, [a, b, c], 4)` is `[a, b, c, d]`. So does 0, the end of the
    count from the right, where -1 inserts before the last element.
    """
    value, target, *rest = args
    if len(rest) > 1:
        raise TypeError("too many arguments")
    sequence, items = _splicing(value, target)
    index = rest[0] if rest else sp.Integer(1)
    count = len(sequence.items)
    if not isinstance(index, sp.Integer) or not -count <= index <= count + 1:
        raise ValueError("not an index")
    place = int(index) - 1 if index > 0 else count + int(index)
    whole = sequence.items
    return sequence.rebuilt(whole[:place] + items + whole[place:], items)


def _reversed(conv: Converter, args: list) -> sp.Basic:
    """`REVERSE(v)`: the sequence `v` with its elements reversed (a matrix's rows, for a
    matrix). An integer's digits reversed lose their leading zeros, and keep the
    integer's sign."""
    sequence = _as_sequence_of(_one(args))
    return sequence.rebuilt(sequence.items[::-1])


def _located(conv: Converter, args: list) -> sp.Basic:
    """`POSITION(u, v)`: the index of the first element of the vector `v` that is `u`,
    or `false` if there is none; `POSITION(u, v, n)` the first from the `n`th on,
    counting from the right if `n` is negative.

    Elements are compared as simplified expressions, so a matrix is searched for a row.
    In a string or integer, the characters or digits of the string or integer `u` are
    searched for as a run, and the index is that of the first.
    """
    value, target, *rest = args
    if len(rest) > 1:
        raise TypeError("too many arguments")
    start = rest[0] if rest else sp.Integer(1)
    sequence = _as_sequence_of(target)
    count = len(sequence.items)
    if not isinstance(start, sp.Integer) or start == 0 or start < -count:
        raise ValueError("not an index")
    first = int(start) - 1 if start > 0 else count + int(start)
    if sequence.kind == "vector":
        for place in range(first, count):
            if sequence.items[place] == value:
                return sp.Integer(place + 1)
        return sp.false
    wanted = _text(value)
    if wanted is None:
        raise TypeError("not a string or integer")
    if first > count:
        return sp.false
    found = "".join(sequence.items).find(wanted, first)
    return sp.false if found < 0 else sp.Integer(found + 1)


def _sort(conv: Converter, args: list) -> sp.Basic:
    """`SORT(v)`: the elements of the vector or set `v` in ascending order, as a vector.

    The 4.11 README documents it for a vector of different rational numbers, which is
    what NUMBER.MTH's `FAREY` and `DIVISORS` sort. Beyond that it follows the 6.1 help:
    numbers go by their real parts, ties broken on their imaginary parts, so `SQRT(2)`
    comes between 1 and 2; variables and strings follow the numbers, in the order of
    their names and texts; then any other expression, in the order of sympy's sort key;
    and last vectors, by their first elements, ties broken on the second, and so on.
    Equal elements are all kept.

    Numbers are compared exactly: rationals directly, other constants by the sign of
    their difference, decided symbolically. A comparison that can't be decided that way
    leaves the call as written, as does an unknown `?` among the elements.

    In Approximate mode, including inside `APPROX`, the elements are evaluated
    numerically first, so every comparison is of plain numbers and the order agrees
    with the values shown. Elements equal at working precision keep their order.

    `SORT(v, p, x, y)` orders by the test `p` instead (see `sorted_by`).
    """
    if len(args) == 4:
        collection, test, first, second = args
        if not (type(first) is sp.Symbol and type(second) is sp.Symbol):
            raise TypeError("not a variable")
        return sorted_by(
            _collected(conv, collection),
            lambda u, v: settled(test.subs({first: u, second: v}, simultaneous=True)),
        )
    elements = _collected(conv, _worked_out(_one(args), conv.context))
    for element in elements:
        _sorting_rank(element)
    return Vector(*sorted(elements, key=functools.cmp_to_key(_sorting_order)))


def _collected(conv: Converter, value: sp.Basic) -> list[sp.Basic]:
    """The elements of a vector, or of a set in its order; raises for anything else."""
    if is_set(value):
        (value,) = conv.sets_simplified([value])
    if isinstance(value, (Vector, SetOf)):
        return list(value.args)
    raise TypeError("not a collection")


def sorted_by(
    elements: list[sp.Basic], test: Callable[[sp.Basic, sp.Basic], sp.Basic]
) -> sp.Basic:
    """`SORT(v, p, x, y)`: the elements, sorted stably so that an element `u` moves
    ahead of an element `v` before it only where the test `p` at `x = u` and `y = v` is
    true. Raises if a test needed is neither true nor false.
    """

    def ahead(u: sp.Basic, v: sp.Basic) -> int:
        held = _test(test(u, v))
        if held is sp.true:
            return -1
        if held is sp.false:
            return 0
        raise ValueError("undecided")

    return Vector(*sorted(elements, key=functools.cmp_to_key(ahead)))


def _sorting_order(a: sp.Basic, b: sp.Basic) -> int:
    """-1, 0 or 1 as `a` goes before, with or after `b` in `SORT`'s order; raises if
    either has no place in it or the order of two numbers can't be decided exactly."""
    rank_a, rank_b = _sorting_rank(a), _sorting_rank(b)
    if rank_a != rank_b:
        return -1 if rank_a < rank_b else 1
    if rank_a == 0:
        real_a, imaginary_a = a.as_real_imag()
        real_b, imaginary_b = b.as_real_imag()
        return _real_order(real_a, real_b) or _real_order(imaginary_a, imaginary_b)
    if rank_a == 1:
        return (a.name > b.name) - (a.name < b.name)
    if rank_a == 2:
        key_a, key_b = sp.default_sort_key(a), sp.default_sort_key(b)
        return (key_a > key_b) - (key_a < key_b)
    for part_a, part_b in zip(a.args, b.args):
        if order := _sorting_order(part_a, part_b):
            return order
    return (len(a.args) > len(b.args)) - (len(a.args) < len(b.args))


def _sorting_rank(element: sp.Basic) -> int:
    """`SORT`'s group of `element`: numbers, then variables and strings, then other
    expressions, then vectors; raises if it has no place in the order."""
    if isinstance(element, Vector):
        for part in element.args:
            _sorting_rank(part)
        return 3
    if element.has(sp.nan):
        raise ValueError("unknown")
    if isinstance(element, sp.Symbol):
        return 1
    if isinstance(element, sp.Expr) and element.is_number:
        return 0
    return 2


def _real_order(a: sp.Expr, b: sp.Expr) -> int:
    """-1, 0 or 1 as the real constant `a` is less than, equal to or greater than `b`,
    decided exactly; raises if it can't be."""
    if a == b:
        return 0
    if isinstance(a, sp.Rational) and isinstance(b, sp.Rational):
        return bool(a > b) - bool(a < b)
    difference = a - b
    if difference.is_zero:
        return 0
    if difference.is_extended_real:
        if difference.is_extended_positive:
            return 1
        if difference.is_extended_negative:
            return -1
    raise TypeError("order undecided")


def _appended(conv: Converter, args: list) -> sp.Basic:
    """`APPEND(v, w, ...)`: the elements of each, concatenated into one vector.

    Elements are whatever the vectors hold, so appending matrices stacks their rows, as
    the manual's `APPEND_COLUMNS` exercise uses (transpose, append, transpose back),
    and `APPEND([[1, 2]], [[3, 4]])` is `[[1, 2], [3, 4]]`. SOLVE.MTH's `NEWTONS` puts
    its system under its gradients with `APPEND(GRAD(u, x), [u])`, where `[u]` is the
    one-row matrix holding the vector `u`.

    A single vector appends the elements of its vector elements, keeping the others in
    place, which flattens a matrix, `APPEND([[a, b], [c, d]])` being `[a, b, c, d]`, and
    removes one level of any other vector of vectors. A single vector of anything else
    is itself.

    Strings and integers concatenate instead (see `_joined`): `APPEND("a", 3)` is
    `"a3"`, `APPEND(12, 345)` is `12345`.
    """
    if not args:
        raise TypeError("nothing to append")
    if not all(isinstance(argument, Vector) for argument in args):
        return _joined(args)
    if len(args) == 1:
        (vector,) = args
        return Vector(
            *(
                part
                for element in vector.args
                for part in (
                    element.args if isinstance(element, Vector) else (element,)
                )
            )
        )
    first = args[0]
    if len(args) == 2:
        second = args[1]
        if first.size() >= _SHARED_LENGTH:
            return _shared(first.extended(second.args), first, second.args)
        if second.size() >= _SHARED_LENGTH:
            return _shared(second.prefixed(first.args), second, first.args)
    return vector_from([element for vector in args for element in vector.args], args)


#: The length from which `APPEND` of two vectors puts the other's elements in the run of
#: a vector this long (see `Vector.extended`) rather than copying both: building a list
#: by appending to it element by element then takes time in proportion to its length.
_SHARED_LENGTH = 16


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
    return Vector(*(settled(body.subs(index, value)) for value in steps(rest)))


def _table(conv: Converter, args: list) -> sp.Basic:
    """`TABLE(u, k, ...)`: the vector of the rows `[k, u]` at each value of `k`, given
    as for `VECTOR`; a vector `u` gives the row `[k, u1, ..., um]`."""
    body, index, *rest = args
    if type(index) is not sp.Symbol:
        raise TypeError("not a variable")
    return Vector(
        *(table_row(value, settled(body.subs(index, value))) for value in steps(rest))
    )


def table_row(value: sp.Basic, entry: sp.Basic, spread: bool = True) -> Vector:
    """A row of `TABLE`: the value of its variable, then the entry, or the entry's
    elements if it is a vector and `spread`.

    Derive spreads the entry where the expression tabulated is a vector before its
    variable takes values: `TABLE([x^2, x^3], x, 5)` has three columns, while a call
    `TABLE(f(n), n, 0, 5)` of a function returning a vector has the vector as the
    second column."""
    if spread and isinstance(entry, Vector):
        return Vector(value, *entry.args)
    return Vector(value, entry)


def steps(rest: list) -> list[sp.Basic]:
    """The values a generated vector's variable takes, in order (see `stepped`)."""
    return list(stepped(rest))


def stepped(rest: list) -> Iterable[sp.Basic]:
    """The values a generated vector's variable takes, in order, each made as it is
    reached: a test over a long range that an early value decides makes only those.

    There are `(n - m)/s + 1` of them, rounded down: one if the step overshoots the
    range, none if the range runs the wrong way. Over a vector or a set they are its
    elements, those of a set in its order.
    """
    if len(rest) == 1 and isinstance(rest[0], (Vector, SetOf)):
        return list(rest[0].args)
    if not rest or len(rest) > 3:
        raise ValueError("not a sequence")
    low = sp.Integer(1) if len(rest) == 1 else rest[0]
    high = rest[0] if len(rest) == 1 else rest[1]
    step = rest[2] if len(rest) == 3 else sp.Integer(1)
    if step.is_zero is not False:
        raise ValueError("no step")
    count = int(sp.floor((high - low) / step)) + 1
    return (low + step * offset for offset in range(max(count, 0)))


def _selected(conv: Converter, args: list) -> sp.Basic:
    """`SELECT(u, k, ...)`: the values of `k` for which `u(k)` holds.

    Values are specified as for `VECTOR`. The result is the values themselves:
    `SELECT(PRIME(k), k, 1, 100)` is the primes under a hundred. Over a set they are a
    set, otherwise a vector (see `over_values`).

    In Derive an element whose test is undecided is dropped: `SELECT(k = y, k, [x, y])`
    is `[y]`, which ODE.MTH's `FREE` relies on. The evaluator does that, deciding the
    test at each value (see `_Evaluator._counted`). Here the body is substituted into
    without being evaluated again, as when a definition is shown, so every test must be
    true or false; otherwise the call stays as written, which keeps a `SELECT` in a
    definition intact until it is applied.
    """
    body, index, *rest = args
    if type(index) is not sp.Symbol:
        raise TypeError("not a variable")
    chosen = chosen_from(steps(rest), lambda value: settled(body.subs(index, value)))
    return over_values(chosen, rest)


def collection_of(rest: list) -> tuple[Iterable[sp.Basic], bool]:
    """The values `MAP_LIST`, `MAP`, `EVERY` and `SOME` give their variable, from the
    arguments after it, and whether a `MAP_LIST` over them gives a set.

    One argument is a collection: a vector's elements in order, a set's in its order, a
    string's characters (see `_as_sequence`); only a set gives a set. An integer `n` is
    the count of values 1 to n, as for `VECTOR`: Derive 6's own utility functions map
    over `DIM(u)` so (`JACOBIAN`). Two or three are a range `m, n` with an optional
    step `s`, as for `VECTOR` (see `stepped`: the values of a range are made as they
    are reached). Raises for anything else.
    """
    if len(rest) == 1:
        (values,) = rest
        if isinstance(values, SetOf):
            return list(values.args), True
        if isinstance(values, sp.Integer):
            return stepped([sp.Integer(1), values]), False
        sequence = _as_sequence(values)
        if sequence is None:
            raise TypeError("not a collection")
        return [sequence.element(place) for place in range(len(sequence.items))], False
    if len(rest) not in (2, 3) or not all(
        isinstance(bound, sp.Expr) and bound.is_real for bound in rest
    ):
        raise TypeError("not a range")
    return stepped(rest), False


def _mapped_list(conv: Converter, args: list) -> sp.Basic:
    """`MAP_LIST(u, x, c)`, `MAP_LIST(u, k, m, n, s)`: `u` at each value of the variable
    (see `collection_of`), as a vector, or as a set over a set.

    Here the body is substituted into without being evaluated again, as when a
    definition is shown; the evaluator evaluates it at each value (see
    `_Evaluator._counted`).
    """
    body, index, *rest = args
    if type(index) is not sp.Symbol:
        raise TypeError("not a variable")
    values, as_set = collection_of(rest)
    results = [settled(body.subs(index, value)) for value in values]
    return SetOf(*results) if as_set else Vector(*results)


def _mapped(conv: Converter, args: list) -> sp.Basic:
    """`MAP(u, x, ...)`: `u` at each value as for `MAP_LIST`, for its effects; true."""
    _mapped_list(conv, args)
    return sp.true


def _every(conv: Converter, args: list) -> sp.Basic:
    """`EVERY(u, x, ...)`: whether `u` holds at every value (see `collection_of`)."""
    return _quantified(args, sp.false)


def _some(conv: Converter, args: list) -> sp.Basic:
    """`SOME(u, x, ...)`: whether `u` holds at some value (see `collection_of`)."""
    return _quantified(args, sp.true)


def _quantified(args: list, deciding: sp.Basic) -> sp.Basic:
    """`EVERY` (`deciding` false) or `SOME` (true): the first value whose test comes
    out `deciding` decides; with none, the other truth value. Raises where a test is
    undecided and none decides, leaving the call as written."""
    body, index, *rest = args
    if type(index) is not sp.Symbol:
        raise TypeError("not a variable")
    values, _ = collection_of(rest)
    undecided = False
    for value in values:
        held = _test(settled(body.subs(index, value)))
        if held is deciding:
            return deciding
        if held is not sp.true and held is not sp.false:
            undecided = True
    if undecided:
        raise ValueError("undecided")
    return sp.Not(deciding)


def over_values(chosen: Vector, rest: list) -> sp.Basic:
    """What `SELECT` chose among the values its last arguments give: a set if they are
    an explicit set, else the vector."""
    if len(rest) == 1 and isinstance(rest[0], SetOf):
        return SetOf(*chosen.args)
    return chosen


def chosen_from(
    values: list,
    test: Callable[[sp.Basic], sp.Basic],
    decide: Callable[[sp.Basic], bool | None] | None = None,
) -> sp.Basic:
    """The vector of `values` passing `test`.

    With `decide`, which answers whether a test holds or None if undecided, an
    undecided test drops its element, as in Derive. Without it, an undecided test
    raises.

    A test that isn't a truth value or relation, like `SELECT(k - 1, k, 1, 3)`, is not
    true, so its element is dropped.
    """
    chosen = []
    for value in values:
        held = _test(test(value))
        if held is sp.true:
            chosen.append(value)
            continue
        if held is sp.false:
            continue
        if decide is None:
            raise ValueError("undecided")
        if isinstance(held, (Boolean, Logical)) and decide(held):
            chosen.append(value)
    return Vector(*chosen)


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
        return Vector(*sequence)
    if rest:
        return sequence[-1]
    came_round = _as_shown(sequence[-1], conv.context) == _as_shown(
        sequence[-2], conv.context
    ) or _converging(sequence, conv.context)
    return sequence[-1] if came_round else sp.nan


def _converging(sequence: list[sp.Basic], context: Context) -> bool:
    """Whether approximate iterates that came round to an earlier one are converging,
    the cycle being one in the last digit shown only.

    A limit near a rounding boundary shows as two values in turn: with four digits,
    `ITERATES(-0.5*SIN(t) + pi/3, t, pi/3)` ends `0.7181, 0.7182, 0.7181`. Unlike a
    true cycle, the steps between the carried values keep shrinking.
    """
    if context.precision is not Precision.APPROXIMATE or len(sequence) < 3:
        return False
    values = [_parts_of(value) for value in sequence[-3:]]
    if any(part is None for part in values) or len({len(v) for v in values}) != 1:
        return False
    first, second, third = values
    before = max(abs(a - b) for a, b in zip(first, second))
    after = max(abs(a - b) for a, b in zip(second, third))
    size = max(abs(part) for part in third)
    tolerance = size * sp.Rational(10) ** (1 - context.precision_digits)
    return after <= tolerance and after < sp.Rational(9, 10) * before


def _parts_of(value: sp.Basic) -> list[sp.Basic] | None:
    """The real and imaginary parts of a number, or of each element of a vector of
    them, flattened; None if `value` isn't one."""
    elements = value.args if isinstance(value, Vector) else (value,)
    parts = []
    for element in elements:
        if isinstance(element, Vector) or not element.is_number:
            return None
        real, imaginary = element.as_real_imag()
        if not (real.is_Number and imaginary.is_Number):
            return None
        parts += [real, imaginary]
    return parts


def _substituted(body: sp.Basic) -> Step:
    return lambda names, values: settled(
        body.subs(_bindings(body, names, values), simultaneous=True)
    )


def _solved_at_each_value(body: sp.Basic, names: list) -> Step:
    """The step of an iteration by the inverse of `body` where it has no single inverse:
    `body = x0` solved for the variable at each numeric value `x0`, the first of its
    solutions in Derive's order taken, as Derive takes it: `ITERATE(pi*u*(8 -
    2*u^2/3), u, 0, -1)` is 0. Raises where the value isn't a number or the equation
    has no solution found."""
    if len(names) != 1:
        raise ValueError("no inverse of a system")
    (name,) = names
    from rederive.engine.solving import listed_in_order

    def step(_names: list, values: list) -> sp.Basic:
        (value,) = values
        if not (isinstance(value, sp.Expr) and value.is_number):
            raise ValueError("no inverse at a symbolic value")
        found = sp.solve(sp.Eq(body, value), name)
        if not found:
            raise ValueError("no solution")
        first = listed_in_order([(each,) for each in found])[0][0]
        if any(spelling(call) is None for call in first.atoms(sp.Function)):
            raise ValueError("no name for the inverse")
        return first

    return step


def _sequence(
    conv: Converter,
    variable: sp.Basic,
    start: sp.Basic,
    rest: list,
    step: Step | None,
    body: sp.Basic | None,
) -> list[sp.Basic]:
    """The iterates, as many as requested.

    A count that isn't an integer is truncated toward zero, as in Derive:
    `ITERATES(x + 1, x, 0, 5/2)` is `[0, 1, 2]` and `ITERATES(x + 1, x, 0, -5/2)` is
    `[0, -1, -2]`. ORTH_POL.MTH counts `n/2` steps for an odd `n`.

    A negative count iterates the inverse of `u` `|n|` times, which `MISC.MTH` uses to
    define `INVERSE(u, x) := ITERATE(u, x, x, -1)`. This needs the converted body.
    """
    names, values = _iterated_over(variable, start)
    if step is None:
        if body is None:
            raise ValueError("no body to iterate")
        step = _substituted(body)
    if not rest or _one(rest) is sp.oo:
        return _until_repeated(conv, step, names, values)
    count = _one(rest)
    if isinstance(count, (sp.Rational, sp.Float)):
        count = sp.floor(count) if count >= 0 else sp.ceiling(count)
    if not isinstance(count, sp.Integer):
        raise TypeError("not a count")
    if count < 0:
        if body is None:
            raise ValueError("no body to invert")
        try:
            step = _substituted(_inverted(body, names))
        except ValueError:
            step = _solved_at_each_value(body, names)
        count = -count
    iterates = [_state(names, values)]
    for left in reversed(range(int(count))):
        if left == 0 and len(names) > 1:
            # The last iterate of a system needn't be a vector of values, as when
            # an undecided `IF` chooses it, since nothing is substituted into it.
            iterates.append(_worked_out(step(names, values), conv.context))
            break
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

    As in Derive, there is no limit on the number of steps: an iteration that never
    repeats runs until the command is stopped (see `Context.step_limit`). Only an
    iterate grown too large to keep iterating on (see `outsized`) ends it, leaving the
    call as written: by then each step costs more than the last, and memory runs out
    long before anything repeats.
    """
    iterates = [_state(names, values)]
    shown = {_as_shown(iterates[-1], conv.context)}
    while True:
        values = _updated(conv, step, names, values)
        iterates.append(_state(names, values))
        latest = _as_shown(iterates[-1], conv.context)
        if latest in shown:
            return iterates
        shown.add(latest)
        if outsized(iterates[-1]):
            raise ValueError("comes round to nothing")


#: Limits on the size of an iterate of an uncounted iteration. A non-repeating sequence
#: usually blows up (repeated squaring passes 30000 digits in 17 steps), and a symbolic
#: iterate can grow by a level of nesting every step, making each step slower than the
#: last; a repeating one comes nowhere near either.
_ITERATE_BITS = 100_000
_ITERATE_OPERATIONS = 1000


def outsized(value: sp.Basic) -> bool:
    """Whether a value has grown too large to keep iterating on.

    This is the size limit for open-ended computations like uncounted iteration. A
    fraction's numerator and denominator count as numbers: Newton's method for
    `SQRT(3)` from 2 doubles their length at every step.
    """
    if sp.count_ops(value) > _ITERATE_OPERATIONS:
        return True
    return any(
        max(abs(number.p), number.q).bit_length() > _ITERATE_BITS
        for number in value.atoms(sp.Rational)
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
    if isinstance(variable, Vector):
        names = list(variable.args)
        values = _elements(start)
        if len(names) != len(values):
            raise ValueError("not that many values")
    else:
        names, values = [variable], [start]
    if not all(_is_variable(name) for name in names):
        raise TypeError("not a variable")
    return names, values


def _is_variable(value: sp.Basic) -> bool:
    """Whether `value` can be a bound or coordinate variable: a name, or a string, which
    Derive takes for a variable of that name, as a program naming its variables `"y0"`,
    `"y1"`, ... does."""
    return type(value) in (sp.Symbol, StringLiteral)


def _state(names: list, values: list) -> sp.Basic:
    """One iterate: the single variable's value, or the vector of all values."""
    return values[0] if len(names) == 1 else Vector(*values)


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
    elements = _elements(written)
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

    A number nearer a nonzero integer than a unit of the digit after the last is that
    integer:
    Newton's method settling on -1 carries values a little above and a little below it,
    shown as `-1.000000000` and `-0.9999999999`, which are -1 to the precision asked for.
    """
    if context.precision is not Precision.APPROXIMATE:
        return value
    digits = context.precision_digits
    return approximated(value, context).replace(
        lambda found: isinstance(found, sp.Rational) and not found.is_Integer,
        lambda found: _integer_to_digits(found, digits),
    )


def _integer_to_digits(number: sp.Rational, digits: int) -> sp.Rational:
    """`number`, or the nonzero integer it is to `digits` significant digits."""
    whole = sp.Integer(round(number))
    if whole and abs(number - whole) < sp.Rational(1, 10**digits) * abs(number):
        return whole
    return number


def _bindings(body: sp.Basic, names: list, values: list) -> dict:
    """Substitutions for the body: each variable, and each of its subscripts.

    `v SUB 1` is a separate symbol, so substituting for `v` doesn't reach it; the
    elements of a vector value are bound to those subscript symbols as well.
    """
    bindings = dict(zip(names, values, strict=True))
    for name, value in zip(names, values, strict=True):
        if not isinstance(value, Vector):
            continue
        elements = value.args
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
    unique (or principal, see `_undone`) inverse is accepted, and systems have none; a
    `u` free of `x` has none, and its inverse is undefined.
    An inverse written with a function the notation has no name for, such as sympy's
    inverse error function, is no answer: the iteration is left as written, which
    reads back, where the function's sympy name wouldn't.
    """
    if len(names) != 1:
        raise ValueError("no inverse of a system")
    (name,) = names
    if isinstance(body, sp.Expr) and not isinstance(body, Vector) and not body.has(name):
        # A constant takes every x to one value: it has no inverse, so the iterate is
        # undefined (`ITERATE(0, y, 0, -1)`, the y-intercept of the line x = 0).
        return sp.nan
    point = sp.Dummy("t")
    undone = _undone(body, name, point)
    if undone is None:
        raise ValueError("not invertible")
    if any(spelling(call) is None for call in undone.atoms(sp.Function)):
        raise ValueError("no name for the inverse")
    return undone.subs(point, name)


def _undone(body: sp.Basic, name: sp.Symbol, value: sp.Basic) -> sp.Basic | None:
    """`body = value` solved for `name`, on the principal branch if there are several.

    A periodic function has many solutions, but Derive gives the principal one:
    `INVERSE(SIN(x/b), x)` is `b*ASIN(x)` (9.21, p.285). So with multiple solutions the
    outermost call is undone by its principal inverse, then its argument recursively.

    Powers work the same way: an even power has two roots, and the principal one is
    the reciprocal power, as in `INT_SUBST(t*SIN(t^2), t, t^2)`, which substitutes the
    manual's `SQRT(t)` (9.21, p.285).

    Of several solutions, the only one written without a sign or an added constant is
    principal: `INVERSE(LN(1 - COS(x)) - LN(COS(x) + 1), x)` is `ACOS(-TANH(x/2))`
    rather than `3*pi/2 - ASIN(TANH(x/2))`.

    A sum or product with `name` in one term or factor only is undone by moving the
    rest to the other side, then undoing that term: `INVERSE(x^3 + 1, x)` is
    `(x - 1)^(1/3)`, the principal one of three cube roots.

    Returns None if there is no known inverse or no solution.
    """
    solutions = sp.solve(sp.Eq(body, value), name)
    if len(solutions) == 1:
        return solutions[0]
    plain = [solution for solution in solutions if _plain_solution(solution)]
    if len(solutions) > 1 and len(plain) == 1:
        return plain[0]
    if isinstance(body, sp.Add | sp.Mul):
        inner = [part for part in body.args if part.has(name)]
        if len(inner) != 1:
            return None
        (part,) = inner
        rest = body.func(*(other for other in body.args if other is not part))
        moved = value - rest if isinstance(body, sp.Add) else value / rest
        return _undone(part, name, moved)
    if isinstance(body, sp.Pow) and body.base.has(name) and not body.exp.has(name):
        return _undone(body.base, name, sp.Pow(value, 1 / body.exp))
    if not (isinstance(body, sp.Function) and len(body.args) == 1):
        return None
    inverse = _inverse_function(body)
    if inverse is None:
        return None
    return _undone(body.args[0], name, inverse(value))


def _plain_solution(solution: sp.Basic) -> bool:
    """Whether `solution` is written without a leading sign or an added constant."""
    if not isinstance(solution, sp.Expr) or solution.could_extract_minus_sign():
        return False
    return not (solution.is_Add and any(not term.free_symbols for term in solution.args))


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


def _elements(value: sp.Basic) -> list[sp.Basic]:
    """A vector's elements, which for a matrix are its rows; anything else raises."""
    if not isinstance(value, Vector):
        raise TypeError("not a vector")
    return list(value.args)


def connected(word: str, operands: list[sp.Basic]) -> sp.Basic:
    """`AND` and the other logical operators, interpreted by their operands.

    Bitwise on integers, boolean on anything sympy accepts as a statement, and an inert
    `Logical` otherwise (e.g. an operand that is still a variable). Evaluation
    reconverts it once values are known: `LUCAS`'s `(n AND d_) = 0` tests a bit of `n`
    once iteration makes `d_` a number.

    A `false` operand decides a conjunction and a `true` one a disjunction, whatever the
    others are: `1/2 AND false` is `false`, as in Derive. Beside a truth value or a
    statement, a number `u` is the test `u = 0`.
    """
    crossed = _SET_OPERATIONS.get(word)
    if crossed is not None and all(
        isinstance(each, SetOf) or isinstance(each, SetSymbol) for each in operands
    ):
        return crossed(*operands)
    deciding = DECIDING.get(word)
    if deciding is not None and any(operand is deciding for operand in operands):
        return deciding
    bitwise = _BITWISE.get(word)
    numbers = operands and all(isinstance(operand, sp.Integer) for operand in operands)
    if bitwise is not None and numbers:
        try:
            return sp.Integer(bitwise(*(int(operand) for operand in operands)))
        except TypeError:
            pass
    head = _BOOLEAN_HEADS.get(word)
    if head is not None and any(
        isinstance(each, BooleanAtom) or is_statement(each) for each in operands
    ):
        # A number beside a truth value is read as a test, as `IF` reads one: `u = 0`.
        # So `false OR 2` is `false` and `IF(false OR 0, 1, 2)` is 1, as in Derive.
        tested = [
            (sp.true if each == 0 else sp.false)
            if isinstance(each, sp.Number) and each.is_real
            else each
            for each in operands
        ]
        if tested != operands:
            return connected(word, tested)
    if head is not None and not any(isinstance(each, sp.Number) for each in operands):
        # A number beside a statement is no truth value: sympy would read `p IFF 3`
        # as `p`.
        try:
            return head(*(_settled(operand) for operand in operands))
        except Exception:
            pass
    return Logical(sp.Symbol(word), *operands)


def _undecided_kinds(lhs: sp.Basic, rhs: sp.Basic) -> bool:
    """Whether sympy would decide an equation of `lhs` and `rhs` by their kinds alone,
    although they may yet be equal: a logical connective that may be bitwise arithmetic
    beside anything but a truth value (`(n AND d) = 0` is false to sympy, since only a
    statement equals a statement, but `n AND d` is a number once `n` and `d` are
    integers), and a variable declared Vector beside a vector or another such variable,
    which sympy takes for unequal."""
    for one, other in ((lhs, rhs), (rhs, lhs)):
        if _bitwise_possible(one) and not is_logical(other):
            return True
        if (
            isinstance(one, sp.MatrixExpr)
            and isinstance(other, (Vector, sp.MatrixExpr))
            and one != other
        ):
            return True
    return False


def _bitwise_possible(value: sp.Basic) -> bool:
    """Whether `value` is a logical connective with an operand, at any depth through the
    connectives, that may hold an integer: a variable not declared Logical."""
    if not isinstance(value, BooleanFunction) or value.is_Relational:
        return False
    return any(
        type(operand) is sp.Symbol or _bitwise_possible(operand) for operand in value.args
    )


def _unknown_against_a_number(lhs: sp.Basic, rhs: sp.Basic) -> bool:
    """Whether one side is a variable sympy knows no more about than that it may be
    real, and the other a finite real number, which sympy can't relate. Asking it would
    cost a pass through its assumptions, which a program testing `x > k` with `k`
    counting pays at every step."""
    for side, other in ((lhs, rhs), (rhs, lhs)):
        if (
            type(side) is sp.Symbol
            and isinstance(other, sp.Number)
            and other.is_finite
            and side.assumptions0 in _ANY_REAL
        ):
            return True
    return False


#: What sympy knows of a variable that may be any real number (an undeclared one), or
#: may be anything at all.
_ANY_REAL = (sp.Symbol("x", real=True).assumptions0, sp.Symbol("x").assumptions0)


def _is_eq(lhs: sp.Basic, rhs: sp.Basic, assumptions=None) -> bool | None:
    """Sympy's `is_eq`, undecided where `_undecided_kinds` says so, and answered at once
    where it can't decide anyway (see `_unknown_against_a_number`)."""
    if assumptions is None and (
        _undecided_kinds(lhs, rhs) or _unknown_against_a_number(lhs, rhs)
    ):
        return None
    return _sympy_is_eq(lhs, rhs, assumptions)


def _is_ge(lhs: sp.Basic, rhs: sp.Basic, assumptions=None) -> bool | None:
    """Sympy's `is_ge`, answered at once where it can't decide anyway."""
    if assumptions is None and _unknown_against_a_number(lhs, rhs):
        return None
    return _sympy_is_ge(lhs, rhs, assumptions)


# Sympy decides every relation it builds evaluated through `is_eq` or `is_ge` of
# `sympy.core.relational` (`is_le`, `is_gt` and `is_lt` call `is_ge`), also when
# `subs`, `xreplace` or `Piecewise` rebuilds one, and its dispatch has no way to say
# "undecided" for a pair of kinds it rules unequal. The one hook that reaches every
# such relation is the module's own names.
_sympy_is_eq = sympy_relational.is_eq
_sympy_is_ge = sympy_relational.is_ge
sympy_relational.is_eq = _is_eq
sympy_relational.is_ge = _is_ge

#: The operand value that decides each operator alone.
DECIDING = {"AND": sp.false, "OR": sp.true}

#: The set operations the logical operators are on explicit sets: `{1, 2} AND {2, 3}` is
#: `{2}`.
_SET_OPERATIONS = {"AND": SetIntersection, "OR": SetUnion, "NOT": SetComplement}

#: The logical operators the set operations are on logical values.
_LOGICAL_SET_OPERATIONS = {"UNION": "OR", "INTERSECTION": "AND"}


def is_logical(value: sp.Basic) -> bool:
    """Whether `value` is a logical value: a truth value, a relation, or a logical
    operator on such values. A variable is none, unless declared Logical."""
    if isinstance(value, (Logical, LogicalSymbol)):
        return True
    return isinstance(value, Boolean) and not isinstance(value, (sp.Symbol, sp.Expr))

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
    return from_matrix(sp.eye(int(size)))


def _cross(conv: Converter, args: list) -> sp.Basic:
    """`CROSS(u, v)` for three-element or two-element vectors.

    In the plane, the result is the scalar third component of the spatial cross
    product (the other two being zero).
    """
    left, right = (_field(argument) for argument in args)
    if len(left) == len(right) == 2:
        return left[0] * right[1] - left[1] * right[0]
    if len(left) != 3 or len(right) != 3:
        raise ValueError("not plane or space vectors")
    return Vector(*sp.Matrix(left).cross(sp.Matrix(right)))


def _row_reduce(conv: Converter, args: list) -> sp.Basic:
    """`ROW_REDUCE(A)`, and `ROW_REDUCE(A, B)` on the augmented matrix.

    The manual's echelon form is the reduced one (leading 1s with zeros above), as
    computed by `rref`.

    A second matrix is adjoined to the right, solving `A . X = B`: `B`'s columns end up
    holding `X` if `A` is nonsingular. A flat vector second argument is a single column,
    as in `A . X = b`, which VECTOR.MTH's `APPROX_EIGENVECTOR` uses.

    In Approximate mode the entries are evaluated numerically first. Pivoting asks
    whether entries are zero, and sympy simplifies undecided ones: for a Newton step
    with nested exponentials and arctangents, that means factoring a polynomial in
    several transcendental generators, which never finishes. Approximate numbers are
    rationals, so zero-testing is trivial.
    """
    matrix, *rest = args
    matrix = _linear(_worked_out(matrix, conv.context))
    if not rest:
        return from_matrix(matrix.rref()[0])
    adjoined = _worked_out(_one(rest), conv.context)
    if isinstance(adjoined, Vector) and not is_matrix(adjoined):
        adjoined = column(adjoined)
    return from_matrix(sp.Matrix.hstack(matrix, _linear(adjoined)).rref()[0])


def _rank(conv: Converter, args: list) -> sp.Basic:
    """`RANK(A)`, the number of independent rows of the matrix `A`, by Gaussian
    elimination.

    Each pivot must be provably nonzero, and every entry left over at the end provably
    zero; an entry that is either for some values of its variables, such as `x - 2`,
    leaves the call as written. In Approximate mode the entries are evaluated
    numerically first, and the rank is that of the rationals they become.
    """
    value = _worked_out(_one(args), conv.context)
    matrix = _linear(value)
    rows = [list(matrix.row(row)) for row in range(matrix.rows)]
    rank = 0
    for col in range(matrix.cols):
        zeros = [_vanishes(rows[row][col]) for row in range(rank, len(rows))]
        if False not in zeros:
            if None in zeros:
                raise ValueError("undecided pivot")
            continue
        pivot = rank + zeros.index(False)
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        lead = rows[rank][col]
        for row in range(rank + 1, len(rows)):
            factor = rows[row][col]
            rows[row] = [
                sp.cancel(lead * entry - factor * above)
                for entry, above in zip(rows[row], rows[rank], strict=True)
            ]
        rank += 1
    return sp.Integer(rank)


def _vanishes(entry: sp.Expr) -> bool | None:
    """Whether `entry` is zero for every value of its variables, False if for none,
    None if that is undecided."""
    known = entry.is_zero
    if known is None:
        known = sp.simplify(entry).is_zero
    return known


def _linear(value: sp.Basic) -> sp.ImmutableMatrix:
    """`value` as a matrix for linear algebra; see `matrix_of`."""
    matrix = matrix_of(value)
    if matrix is None:
        raise TypeError("not a matrix")
    return matrix


def _characteristic_polynomial(conv: Converter, args: list) -> sp.Basic:
    """`CHARPOLY(A, v)`: `DET(A - v*IDENTITY_MATRIX(DIMENSION(A)))`.

    The manual defines it as this determinant, not the usual monic polynomial
    `DET(v*I - A)`; they differ by `(-1)^n`. Sympy computes the monic one efficiently,
    and the sign is adjusted for odd dimensions.

    The polynomial is computed over a dummy and `v` substituted afterwards, because
    `Poly` strips its generator's assumptions, and a `v` without its declared domain
    wouldn't cancel against the user's `v`.

    `v` may be any value, at which the polynomial is taken: `CHARPOLY(A, 0)` is
    `DET(A)`.
    """
    matrix, *rest = args
    value = _one(rest) if rest else conv.symbol("w")
    if isinstance(value, (Vector, SetOf)):
        raise TypeError("not a scalar")
    matrix = _linear(matrix)
    expression = matrix.charpoly(sp.Dummy()).as_expr(value)
    return expression if matrix.rows % 2 == 0 else -expression


def _eigenvalues(conv: Converter, args: list) -> sp.Basic:
    """`EIGENVALUES(A, v)`: the zeros of `A`'s characteristic polynomial in `v`, as
    equations `[z = 2, z = b]`, as Derive writes solutions elsewhere; `EIGENVALUES(A)`:
    the zeros themselves, `[2, b]`. Each zero appears once regardless of multiplicity.

    Numbers come in the order of `SOLUTIONS`, as Derive's recorded results list them
    (`[1, -2]`, `[0, SQRT(201)/10 - 19/10, -SQRT(201)/10 - 19/10]`); symbolic zeros in
    sympy's canonical order, which matches the manual's examples (numbers before
    names, `a - b` before `a + b`).

    Zeros expressible only as `CRootOf` are rejected: the manual uses the quadratic,
    cubic and quartic formulas and notes exact eigenvalues are rarely available beyond
    that. Then the characteristic polynomial is solved factor by factor, as `SOLVE`
    does, a factor without such zeros staying as its equation in the variable (`w`
    without one); if no factor has them, the call stays as written. Approximating, such
    a factor's zeros are all found, complex ones too.

    `trig=True` makes `roots` use Viete's cubic rather than Cardano's for the casus
    irreducibilis, so symmetric matrices, whose eigenvalues are real, don't get answers
    in `#i`.
    """
    matrix, variable = _matrix_and_variable(conv, args)
    as_values = len(args) == 1
    approximate = conv.context.precision is not Precision.EXACT
    try:
        zeros = sorted(matrix.eigenvals(trig=True), key=sp.default_sort_key)
    except MatrixError:
        zeros = None
    if zeros and all(zero.is_number for zero in zeros):
        from rederive.engine.solving import listed_in_order

        zeros = [row[0] for row in listed_in_order([(zero,) for zero in zeros])]
    if zeros is None or any(zero.has(sp.CRootOf) for zero in zeros):
        return _eigenvalues_by_factors(matrix, variable, as_values, approximate)
    if as_values:
        return Vector(*zeros)
    return Vector(*(sp.Eq(variable, zero, evaluate=False) for zero in zeros))


def _eigenvalues_by_factors(
    matrix: sp.MatrixBase, variable: sp.Symbol, as_values: bool, approximate: bool
) -> Vector:
    """The zeros of each factor of the characteristic polynomial that has them in
    radicals, every zero of the others if `approximate`, and otherwise the equation of
    each that hasn't; see `_eigenvalues`."""
    _, factors = sp.factor_list(matrix.charpoly(sp.Dummy()).as_expr(variable))
    zeros: set[sp.Basic] = set()
    unsolved: list[sp.Basic] = []
    for factor, _ in factors:
        if not factor.has(variable):
            continue
        found = sp.roots(factor, variable, trig=True)
        if sum(found.values()) == sp.degree(factor, variable):
            zeros.update(found)
        elif approximate and factor.is_polynomial(variable) and not (
            factor.free_symbols - {variable}
        ):
            zeros.update(sp.Poly(factor, variable).all_roots())
        else:
            unsolved.append(sp.Eq(factor, 0, evaluate=False))
    if not zeros:
        raise ValueError("no closed form")
    ordered = sorted(zeros, key=sp.default_sort_key)
    if as_values:
        return Vector(*ordered, *unsolved)
    return Vector(*(sp.Eq(variable, zero, evaluate=False) for zero in ordered), *unsolved)


def _matrix_and_variable(conv: Converter, args: list) -> tuple[sp.MatrixBase, sp.Basic]:
    """A matrix and the variable for the answer.

    The variable defaults to `w`, as in the manual, with its declared domain. Sympy
    checks squareness; if it fails, the call stays as written.
    """
    matrix, *rest = args
    variable = _one(rest) if rest else conv.symbol("w")
    if type(variable) is not sp.Symbol:
        raise TypeError("not a variable")
    return _linear(matrix), variable


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
    if isinstance(expression, Vector):
        variables, scales = _coordinates(conv, rest)
        columns = [
            _gradient_of(element, variables, scales) for element in _field(expression)
        ]
        return Vector(*(Vector(*row) for row in zip(*columns, strict=True)))
    if _holds_vector(expression):
        raise TypeError("not an expression")
    variables, scales = _coordinates(conv, rest)
    return Vector(*_gradient_of(expression, variables, scales))


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
    if isinstance(expression, Vector):
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
    return Vector(*turns)


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
    return Vector(
        (lifted - sliced) / h1,
        -_leg(h2 * h3 * first, x3, start[2]) / h2,
        sp.Integer(0),
    )


def _holds_vector(value: sp.Basic) -> bool:
    """Whether `value` contains a vector anywhere."""
    return value.has(Vector)


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
        system = _one(rest)
        given = _elements(system)
        if not is_matrix(system):
            variables, scales = given, [sp.Integer(1)] * len(given)
        elif len(given) == 2:
            variables, scales = list(given[0].args), list(given[1].args)
        else:
            raise ValueError("not a coordinate system")
    else:
        variables = [conv.symbol(name) for name in ("x", "y", "z")]
        scales = [sp.Integer(1)] * 3
    if not variables or not all(_is_variable(name) for name in variables):
        raise TypeError("not coordinate variables")
    if len(set(variables)) != len(variables):
        raise ValueError("a coordinate twice")
    if count is not None and len(variables) != count:
        raise ValueError("not that many coordinates")
    return variables, scales


def _field(value: sp.Basic) -> list[sp.Basic]:
    """A vector field's components; vectors of vectors are refused."""
    elements = _elements(value)
    if not elements or any(isinstance(element, Vector) for element in elements):
        raise TypeError("not a vector")
    return elements


#: Converters for each function name. Other names become inert heads, which keeps user
#: functions intact through a round trip and makes `DIF(F(x)^3, x)` differentiable.
#: `SOLVE` is absent because it is in `COMMAND_HEADS`, run by the evaluator.
FUNCTIONS: dict[str, Handler] = {
    **{name: _trig(func) for name, func in _TRIGONOMETRIC.items()},
    **{name: _arctrig(func) for name, func in _INVERSE_TRIGONOMETRIC.items()},
    **{name: _direct(func) for name, func in _DIRECT.items()},
    "SQRT": _sqrt,
    "EXP": _exp,
    "EI": _exponential_integral,
    "POLYLOG": _polylogarithm,
    "POLYGAMMA": _polygamma,
    "MAX": _extremum(sp.Max),
    "MIN": _extremum(sp.Min),
    "ATAN": _atan,
    "ACOT": _acot,
    "PHASE": _phase,
    "ERF": _erf,
    "ABS": _absolute,
    "SIGN": _sign,
    "STEP": _step,
    "LOG": _log,
    "FLOOR": _floor,
    "CEILING": _ceiling,
    "ROUND": _round,
    "MOD": _mod,
    "MODS": _mods,
    "INVERSE_MOD": _inverse_mod,
    "POWER_MOD": _power_mod,
    "SHIFT": _shift,
    "POLY_MOD": _poly_mod(symmetric=False),
    "POLY_MODS": _poly_mod(symmetric=True),
    "CHI": _chi,
    "NORMAL": _normal,
    "PVAL": _present_value,
    "FVAL": _future_value,
    "PMT": _payment,
    "NPER": _periods,
    "RATE": _rate,
    "RANDOM": _random,
    **{name: _in_degrees(name) for name in _IN_DEGREES},
    "DILOG": lambda conv, args: Dilog(*args),
    # Per section 6.13 these take arguments like a statistic: any number of values, a
    # vector, or a matrix whose rows give a vector of answers.
    "GCD": _pairwise(_statistic(_gcd_of)),
    "LCM": _pairwise(_statistic(_lcm_of)),
    "AVERAGE": _statistic(_average),
    "RMS": _statistic(_root_mean_square),
    "VARIANCE": _statistic(_variance),
    "STDEV": _statistic(_standard_deviation),
    "FIT": _fit,
    **_PREDICATE_HANDLERS,
    "PRIME?": _distributed(_is_prime),
    "NEXT_PRIME": _distributed(_next_prime),
    "PREVIOUS_PRIME": _distributed(_previous_prime),
    "STRING": _string,
    "NAME_TO_CODES": _name_to_codes,
    "CODES_TO_NAME": _codes_to_name,
    "SUBST": _substituted_exactly,
    "NUMERATOR": _numerator,
    "DENOMINATOR": _denominator,
    "TERMS": _terms,
    "FACTORS": _factors,
    "LHS": _side(0),
    "RHS": _side(1),
    "QUOTIENT": _quotient,
    "REMAINDER": _remainder,
    "POLY_GCD": _polynomial_gcd,
    "GROEBNER_BASIS": _groebner_basis,
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
    "SUBST_DIFF": _subst_diff,
    "APPROX": _approximation,
    "IF": _conditional,
    "TRUTH_TABLE": _truth_table,
    "SUBS": _substitution,
    "DET": _determinant,
    "TRACE": _trace,
    "DIM": _dimension,
    "POWER_SET": _power_set,
    "ELEMENT": _element_of,
    "FIRST": _first,
    "REST": _rest,
    "ADJOIN": _adjoined,
    "DELETE": _deleted,
    "REPLACE": _replaced,
    "INSERT": _inserted,
    "REVERSE": _reversed,
    "POSITION": _located,
    "SORT": _sort,
    "APPEND": _appended,
    "IDENTITY_MATRIX": _identity_matrix,
    "CROSS": _cross,
    "ROW_REDUCE": _row_reduce,
    "RANK": _rank,
    "CHARPOLY": _characteristic_polynomial,
    "EIGENVALUES": _eigenvalues,
    "VECTOR": _generated_vector,
    "TABLE": _table,
    "SELECT": _selected,
    "MAP_LIST": _mapped_list,
    "MAP": _mapped,
    "EVERY": _every,
    "SOME": _some,
    "ITERATE": _iterate,
    "ITERATES": _iterates,
    "GRAD": _gradient,
    "DIV": _divergence,
    "LAPLACIAN": _laplacian,
    "CURL": _curl,
    "POTENTIAL": _potential,
    "VECTOR_POTENTIAL": _vector_potential,
}

# An old name Derive 6 keeps computes as the function it stands for.
FUNCTIONS |= {old: FUNCTIONS[new] for old, new in KEPT_ALIASES.items()}


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
    | {"ATAN", "ACOT", "ERF", "LOG", "SIGN", "STEP", "FLOOR", "CEILING", "ROUND"}
    | {"MOD", "MODS"}
)

#: Functions applied to each element of a vector (so of a matrix, at every depth) or of
#: an explicit set, as Derive 6 applies its elementary, special and integer functions to
#: a nonscalar argument: `SIN([0, pi])` is `[0, 0]`. One collection among the arguments
#: is mapped over with the others fixed, several vectors of one length pairwise:
#: `MOD(10, [3, 4])` is `[1, 2]` and `LOG([8, 9], [2, 3])` is `[3, 2]`.
#:
#: Functions with a meaning of their own on a vector are not here: `ABS` and `SIGN`
#: (length and direction), the aggregates `MAX`, `MIN`, `GCD`, ..., and `EXP` and `SQRT`
#: of a square matrix, which are the matrix exponential and square root.
ELEMENTWISE: frozenset[str] = (
    frozenset(_TRIGONOMETRIC)
    | frozenset(_INVERSE_TRIGONOMETRIC)
    | (frozenset(_DIRECT) - {"ABS", "MAX", "MIN", "COMB", "PERM"})
    | {"ATAN", "ACOT", "PHASE", "ERF", "LOG", "STEP", "FLOOR", "CEILING", "ROUND"}
    | {"MOD", "MODS", "INVERSE_MOD", "POWER_MOD", "SHIFT", "POLY_MOD", "POLY_MODS"}
    | {"PRIME?", "PRIME", "NEXT_PRIME", "PREVIOUS_PRIME", "RANDOM"}
    | frozenset(_IN_DEGREES)
    | {"DILOG"}
)

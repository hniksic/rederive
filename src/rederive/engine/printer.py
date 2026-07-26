"""sympy -> author notation.

The engine's results leave sympy as text, and this is where that text is
written: `^` for powers, `SQRT(u)` for square roots, `#e` and `#i` for the
constants, upper-case function names, `?` for the unknown value. Every result
of every command goes through here, so nothing in it may assume which command
produced the expression.

Two rules govern the output. It must reparse to the same expression - the
printer never leans on juxtaposition and parenthesises wherever the grammar
needs it - and it must be written in the notation the session is using, which
is why numerals come out in the context's input base.

`AuthorPrinter` subclasses sympy's `StrPrinter` and overrides one method per
construct spelled differently: powers, roots, the constants, function names,
vectors, the unevaluated calculus heads, the inert heads, and numerals. What
it does *not* override is the sum and product spine. Term ordering, sign
extraction, collecting negative powers into a denominator, unevaluated and
noncommutative products - all of that is inherited, because reimplementing it
would mean carrying a few hundred lines of subtle logic for no gain in what
the output means. The cost is that sympy's own conventions show through:
`x*k^-2` comes out as `x/k^2` because that is how `_print_Mul` splits a
denominator. That is a form difference, never a meaning difference, and both
forms reparse to the same tree. `_print_Pow` follows the same convention for
a power standing alone, so that one expression does not print two ways.

The inert heads of `to_sympy` are printed by name rather than by import: a
sympy printer dispatches on the class name, so `_print_PlusMinus` finds
`PlusMinus` without this module depending on the module that defines it.
"""

from __future__ import annotations

from decimal import Decimal

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.printing.precedence import PRECEDENCE

from rederive.engine.context import Context
from rederive.syntax.names import GREEK_GLYPHS

_DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

#: What joins the parts of a subscripted symbol's name.
_SUBSCRIPT = " SUB "

#: The plus-or-minus operator, as the glyph rather than its other input
#: spelling `"+-"`. A bare `+-u` is neither: it is a plus over a minus, and
#: reads back as `-u`.
_PLUS_MINUS = "±"

#: How each relation is written. Everything not here is written as it is.
_RELATIONS = {"==": "=", "!=": "/="}

#: sympy heads whose author-notation name is not just their name upper-cased.
_FUNCTION_NAMES = {
    "log": "LN",
    "arg": "PHASE",
    "atan2": "ATAN",
    "conjugate": "CONJ",
    "binomial": "COMB",
    "FallingFactorial": "PERM",
    "Heaviside": "STEP",
    "Determinant": "DET",
    "Trace": "TRACE",
}

#: The logical operators, tightest first, so that an operand knows when it
#: needs parentheses. A relation and anything below it never does.
_LOGICAL_PRECEDENCE = {"NOT": 5, "AND": 4, "OR": 3, "XOR": 2, "IMP": 1}

_LOGICAL_HEADS = {
    "And": "AND",
    "Or": "OR",
    "Xor": "XOR",
    "Implies": "IMP",
    "Not": "NOT",
}


def author_text(expression: sp.Basic, context: Context | None = None) -> str:
    """Write `expression` the way the author line would."""
    return AuthorPrinter(context).doprint(expression)


def numeral(number: int, base: int = 10) -> str:
    """`number` written in `base`.

    Above base ten the digits run A to Z, and a numeral that would begin with
    a letter takes a leading zero so that it cannot be read as a variable
    name: fourteen is `0E` in hexadecimal, not `E`.
    """
    magnitude = abs(number)
    text = ""
    while True:
        text = _DIGITS[magnitude % base] + text
        magnitude //= base
        if not magnitude:
            break
    if text[0].isalpha():
        text = "0" + text
    return ("-" if number < 0 else "") + text


class AuthorPrinter(sp.StrPrinter):
    """A `StrPrinter` that writes author notation."""

    def __init__(self, context: Context | None = None) -> None:
        super().__init__({"full_prec": False})
        self.context = context or Context()

    # -- names --------------------------------------------------------------

    def _print_Symbol(self, expr):
        """A variable, written in ASCII.

        A Greek variable is held under its letter and printed as its name,
        `alpha` for `α`: drawing the letter is the display layer's business,
        and the text has to be something the author line could have been
        typed with.
        """
        return _SUBSCRIPT.join(
            GREEK_GLYPHS.get(part, part) for part in expr.name.split(_SUBSCRIPT)
        )

    # -- numbers ------------------------------------------------------------

    def _print_Integer(self, expr):
        return numeral(int(expr), self.context.input_base)

    def _print_Rational(self, expr):
        base = self.context.input_base
        if expr.q == 1:
            return numeral(expr.p, base)
        return f"{numeral(expr.p, base)}/{numeral(expr.q, base)}"

    def _print_Float(self, expr):
        """A float at the context's precision, and always as plain digits.

        There is no exponent notation to reparse, so a very large or very
        small float is written out in full, and a whole one keeps its point so
        that it does not read back as an integer.
        """
        text = super()._print_Float(sp.Float(expr, self.context.precision_digits))
        if "e" in text or "E" in text:
            text = format(Decimal(text), "f")
        return text if "." in text else text + "."

    # -- constants and special values ---------------------------------------

    def _print_Exp1(self, expr):
        return "#e"

    def _print_ImaginaryUnit(self, expr):
        return "#i"

    def _print_EulerGamma(self, expr):
        return "euler_gamma"

    def _print_Infinity(self, expr):
        return "inf"

    def _print_NegativeInfinity(self, expr):
        return "-inf"

    def _print_ComplexInfinity(self, expr):
        """Unsigned infinity, which is what `1/0` and `TAN(pi/2)` are worth."""
        return f"{_PLUS_MINUS}inf"

    def _print_NaN(self, expr):
        return "?"

    def _print_BooleanTrue(self, expr):
        return "true"

    def _print_BooleanFalse(self, expr):
        return "false"

    # -- operators ----------------------------------------------------------

    def _print_Pow(self, expr, rational=False):
        level = PRECEDENCE["Pow"]
        if expr.exp is sp.S.Half and not rational:
            return f"SQRT({self._print(expr.base)})"
        if expr.is_commutative and not rational:
            if -expr.exp is sp.S.Half:
                return f"1/SQRT({self._print(expr.base)})"
            if expr.exp is sp.S.NegativeOne:
                return f"1/{self.parenthesize(expr.base, level)}"
            if expr.exp.is_Number and expr.exp.is_negative:
                # A denominator, the way the inherited `_print_Mul` writes one
                # inside a product: `1/k^2` rather than `k^(-2)`.
                return f"1/{self._print(sp.Pow(expr.base, -expr.exp))}"
        base = self.parenthesize(expr.base, level)
        exponent = self.parenthesize(expr.exp, level)
        return f"{base}^{exponent}"

    def _print_exp(self, expr):
        """`#e^u`, never `EXP(u)`: the constant is how the notation spells it."""
        return f"#e^{self.parenthesize(expr.args[0], PRECEDENCE['Pow'])}"

    def _print_factorial(self, expr):
        return f"{self.parenthesize(expr.args[0], PRECEDENCE['Func'])}!"

    def _print_Relational(self, expr):
        level = PRECEDENCE["Relational"]
        operator = _RELATIONS.get(expr.rel_op, expr.rel_op)
        left = self.parenthesize(expr.lhs, level)
        right = self.parenthesize(expr.rhs, level)
        return f"{left} {operator} {right}"

    # -- functions ----------------------------------------------------------

    def _print_Function(self, expr):
        return f"{self.function_name(expr)}({self.stringify(expr.args, ', ')})"

    def _print_Min(self, expr):
        return f"MIN({self.stringify(expr.args, ', ')})"

    def _print_Max(self, expr):
        return f"MAX({self.stringify(expr.args, ', ')})"

    def _print_Heaviside(self, expr):
        """`STEP(u)`. The value at zero is ours, not the author's."""
        return f"STEP({self._print(expr.args[0])})"

    def function_name(self, expression: sp.Basic) -> str:
        """What a function head is called in author notation.

        An inert head keeps the name it was authored with; a sympy head is
        upper-cased, which is the spelling of a function name.
        """
        name = type(expression).__name__
        if name in _FUNCTION_NAMES:
            return _FUNCTION_NAMES[name]
        if isinstance(expression, AppliedUndef):
            return name
        return name.upper()

    # -- calculus heads that reached the output unevaluated -----------------

    def _print_Derivative(self, expr):
        text = self._print(expr.expr)
        for variable, count in expr.variable_count:
            parts = [text, self._print(variable)]
            if count != 1:
                parts.append(self._print(count))
            text = f"DIF({', '.join(parts)})"
        return text

    def _print_Integral(self, expr):
        return self._limited("INT", expr)

    def _print_Sum(self, expr):
        return self._limited("SUM", expr)

    def _print_Product(self, expr):
        return self._limited("PRODUCT", expr)

    def _limited(self, head: str, expr) -> str:
        """`INT(u, x)` and `INT(u, x, a, b)`, one head per limit."""
        text = self._print(expr.function)
        for limit in expr.limits:
            parts = [text, *(self._print(part) for part in limit)]
            text = f"{head}({', '.join(parts)})"
        return text

    def _print_Limit(self, expr):
        expression, variable, point, direction = expr.args
        parts = [self._print(expression), self._print(variable), self._print(point)]
        side = {"+": "1", "-": "-1"}.get(str(direction))
        if side is not None:
            parts.append(side)
        return f"LIM({', '.join(parts)})"

    # -- vectors and matrices ------------------------------------------------

    def _print_MatrixBase(self, expr):
        """A row is written flat; anything taller is a vector of its rows."""
        if expr.rows == 1:
            return self._vector(expr)
        return self._vector([expr[row, :] for row in range(expr.rows)])

    def _vector(self, elements) -> str:
        return f"[{', '.join(self._print(element) for element in elements)}]"

    # -- logic ---------------------------------------------------------------

    def _print_And(self, expr):
        return self._infix("AND", expr.args)

    def _print_Or(self, expr):
        return self._infix("OR", expr.args)

    def _print_Xor(self, expr):
        return self._infix("XOR", expr.args)

    def _print_Implies(self, expr):
        return self._infix("IMP", expr.args)

    def _print_Not(self, expr):
        return f"NOT {self._logical_operand(expr.args[0], 'NOT')}"

    def _infix(self, word: str, operands) -> str:
        return f" {word} ".join(
            self._logical_operand(operand, word) for operand in operands
        )

    def _logical_operand(self, operand: sp.Basic, word: str) -> str:
        """Parenthesised when the operand binds more loosely than `word` does."""
        inner = _LOGICAL_HEADS.get(type(operand).__name__)
        if isinstance(operand, sp.Function) and type(operand).__name__ == "Logical":
            inner = str(operand.args[0])
        text = self._print(operand)
        if inner is None:
            return text
        if _LOGICAL_PRECEDENCE[inner] > _LOGICAL_PRECEDENCE[word]:
            return text
        return f"({text})"

    # -- the inert heads ------------------------------------------------------

    def _print_PlusMinus(self, expr):
        operand = self.parenthesize(expr.args[0], PRECEDENCE["Mul"], strict=True)
        return f"{_PLUS_MINUS}{operand}"

    def _print_StringLiteral(self, expr):
        return f'"{expr.name}"'

    def _print_Transposed(self, expr):
        return f"{self.parenthesize(expr.args[0], PRECEDENCE['Func'])}`"

    def _print_Subscript(self, expr):
        level = PRECEDENCE["Func"]
        base, index = expr.args
        return f"{self.parenthesize(base, level)} SUB {self.parenthesize(index, level)}"

    def _print_Dot(self, expr):
        level = PRECEDENCE["Mul"]
        left, right = expr.args
        return f"{self.parenthesize(left, level)} . {self.parenthesize(right, level)}"

    def _print_InertVector(self, expr):
        return self._vector(expr.args)

    def _print_Logical(self, expr):
        word = str(expr.args[0])
        operands = expr.args[1:]
        if len(operands) == 1:
            return f"{word} {self._logical_operand(operands[0], word)}"
        return self._infix(word, operands)

    def _print_Assign(self, expr):
        target, operator, *value = expr.args
        parts = [self._print(target), str(operator), *(self._print(v) for v in value)]
        return " ".join(parts)

    def _print_FunDef(self, expr):
        name, parameters, *body = expr.args
        head = f"{name}({', '.join(self._print(p) for p in parameters)})"
        return " ".join([head, ":=", *(self._print(part) for part in body)])

    def _print_Declare(self, expr):
        """`x :epsilon Real`, with the interval when one was declared."""
        target, kind, *interval = expr.args
        text = f"{self._print(target)} :epsilon {kind}"
        if len(interval) == 3:
            brackets, low, high = interval
            opening, closing = str(brackets)
            text += f" {opening}{self._print(low)}, {self._print(high)}{closing}"
        return text

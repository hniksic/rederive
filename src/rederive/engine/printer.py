"""sympy -> author notation.

The engine's results leave sympy as text, and this is where that text is
written: `^` for powers, `SQRT(u)` for square roots, `#e` and `#i` for the
constants, upper-case function names, `?` for the unknown value. Every result
of every command goes through here, so nothing in it may assume which command
produced the expression.

Two rules govern the output. It must reparse to the same expression - the
printer never leans on juxtaposition and parenthesises wherever the grammar
needs it - and it must be written in the notation the session is using, which
is why numerals come out in the context's input base and in the style
`Options Notation` selects. `engine.notation` holds that second part.

A decimal style bends the first rule, and the original bends it the same way:
one third written `0.333333` reads back as a different number. Derive shows
that text on the screen and writes it to an MTH file alike, so what the
notation cuts is cut from the worksheet too - but not from the value. The
exact writing is kept beside the shown one, which is `Result.value`, and it is
what the next command is given; `3·#n` is 1 whatever `#n` is shown as.

`AuthorPrinter` subclasses sympy's `StrPrinter` and overrides one method per
construct spelled differently: powers, roots, the constants, function names,
vectors, the unevaluated calculus heads, the inert heads, and numerals. What
it does *not* override is the sum and product spine. Sign extraction,
collecting negative powers into a denominator, unevaluated and noncommutative
products - all of that is inherited, because reimplementing it would mean
carrying a few hundred lines of subtle logic for no gain in what the output
means. The cost is that sympy's own conventions show through: `x*k^-2` comes
out as `x/k^2` because that is how `_print_Mul` splits a denominator. That is a
form difference, never a meaning difference, and both forms reparse to the same
tree. `_print_Pow` follows the same convention for a power standing alone, so
that one expression does not print two ways.

Of the ordering only the two rules the original is recognisable by are ours:
terms run by descending degree, and a sum does not begin with a minus sign
unless every term is negated, so that a sum is written `x^2 + c` and
`SQRT(3) - 1`. Beyond those two, term order is sympy's.

The inert heads of `to_sympy` are printed by name rather than by import: a
sympy printer dispatches on the class name, so `_print_PlusMinus` finds
`PlusMinus` without this module depending on the module that defines it.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.printing.precedence import PRECEDENCE, PRECEDENCE_VALUES

from rederive.engine import notation
from rederive.engine.context import Context, Notation
from rederive.syntax.names import GREEK_GLYPHS

# How tightly the inert heads bind. Sympy's `parenthesize` reads these by class
# name, and without an entry it takes any `Function` for a call as tight as
# `SIN(x)` - which would write `#e^(±inf*z)` as `#e^±inf*z`, a different
# expression. Each is registered as the operator it is written as.
#
# `Dot` sits below `Add` rather than at `Mul`, where it belongs: `.` and `*` are
# one precedence level in the grammar and associate to the left, so `x*(a . b)`
# needs its parentheses to keep from reading as `(x*a) . b` - and sympy prices a
# negated product at the `Add` level, so an entry at `Mul` would lose them again
# in `-x*(a . b)`. Complex infinity is written with the plus-or-minus operator
# and binds as loosely as one.
PRECEDENCE_VALUES.setdefault("PlusMinus", PRECEDENCE["Add"])
PRECEDENCE_VALUES.setdefault("ComplexInfinity", PRECEDENCE["Add"])
PRECEDENCE_VALUES.setdefault("Dot", PRECEDENCE["Add"] - 1)
PRECEDENCE_VALUES.setdefault("Subscript", PRECEDENCE["Atom"] - 1)
PRECEDENCE_VALUES.setdefault("Power", PRECEDENCE["Pow"])

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
    return AuthorPrinter(context).doprint(named(expression))


def named(expression: sp.Basic) -> sp.Basic:
    """The same expression with sympy's invented variables named as ours.

    A `Dummy` is an ordinary variable by the time it reaches the worksheet -
    the bound variable the Leibniz rule introduces is written `k1` and read
    back as the variable `k1`. Two of them can carry one name, though, which
    sympy tells apart by an identity no text can hold, so they are renamed
    apart from each other and from every name already in use before anything
    is written. Without that, `SUM(SUM(f(k1), k1, 0, n), k1, 0, n)` is written
    from an expression with two variables and read back as one with one.

    Order needs it too: a `Dummy` sorts by its identity where a symbol sorts by
    its name, so a sum holding one is written in one order and reads back in
    another - the answer would settle only on a second pass.
    """
    dummies = sorted(expression.atoms(sp.Dummy), key=sp.default_sort_key)
    if not dummies:
        return expression
    taken = {
        symbol.name
        for symbol in expression.atoms(sp.Symbol)
        if not isinstance(symbol, sp.Dummy)
    }
    renamed = {}
    for dummy in dummies:
        name = _plain(dummy.name)
        while name in taken:
            name += "_"
        taken.add(name)
        renamed[dummy] = sp.Symbol(name, **dummy.assumptions0)
    try:
        return expression.xreplace(renamed)
    except Exception:
        return expression


def _plain(name: str) -> str:
    """A dummy's name as the lexer would read one back."""
    stripped = name.lstrip("_")
    return stripped if stripped[:1].isalpha() else f"v{stripped}"


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
        # `grlex` is descending total degree, which is the order the original
        # writes a polynomial in: `x^2 + c`, not `c + x^2`.
        super().__init__({"full_prec": False, "order": "grlex"})
        self.context = context or Context()

    # -- sums ---------------------------------------------------------------

    def _as_ordered_terms(self, expr, order=None):
        """The terms of a sum, led by one that is not negated.

        `SQRT(3) - 1` and `SIN(x) - x*COS(x)`, not `-1 + SQRT(3)` and
        `-x*COS(x) + SIN(x)`: a sum is written starting with a term it can be
        written starting with, and only a sum of nothing but negated terms
        begins with a minus sign. The first such term moves to the front and
        nothing else changes order.
        """
        terms = super()._as_ordered_terms(expr, order)
        leading = next(
            (
                index
                for index, term in enumerate(terms)
                if not term.could_extract_minus_sign()
            ),
            0,
        )
        return [terms[leading], *terms[:leading], *terms[leading + 1 :]]

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

    def _print_Dummy(self, expr):
        """A sympy-invented variable, written like any other name.

        Sympy marks a `Dummy` apart by writing it with a leading underscore,
        and the author line has no such name: `_k1` does not lex, so a result
        carrying one would come back unreadable. `named` has normally renamed
        these away before printing starts; one printed on its own is written
        the same way.
        """
        return _plain(self._print_Symbol(expr))

    # -- numbers ------------------------------------------------------------

    def _print_Integer(self, expr):
        return self._number(Fraction(int(expr)))

    def _print_Rational(self, expr):
        return self._number(Fraction(expr.p, expr.q))

    def _print_Float(self, expr):
        """A float at the context's precision, written as the notation says.

        Rational notation leaves it as plain digits: an approximation is a
        float here where the original holds a ratio, so there is no ratio to
        write it as. There is no exponent notation to reparse either, so a
        very large or very small float is written out in full, and a whole one
        keeps its point so that it does not read back as an integer.
        """
        text = super()._print_Float(sp.Float(expr, self.context.precision_digits))
        if "e" in text or "E" in text:
            text = format(Decimal(text), "f")
        if self.context.notation is not Notation.RATIONAL:
            return self._number(Fraction(Decimal(text)))
        return text if "." in text else text + "."

    def _print_Mul(self, expr):
        """A product led by a ratio the notation does not write as a ratio.

        sympy prices `x/3` as an x over a 3 and prints the denominator, which
        is right while a ratio is written as one. Under a decimal style the
        coefficient is a number like any other and goes in front, which is
        where the original puts it: `x/3` is `0.333333·x`.
        """
        coefficient, rest = expr.as_coeff_Mul()
        if isinstance(coefficient, sp.Rational) and coefficient.q != 1:
            written = self._number(Fraction(coefficient.p, coefficient.q))
            if "/" not in written:
                return self._beside(written, rest)
        return super()._print_Mul(expr)

    def _beside(self, written: str, rest) -> str:
        """A coefficient already written, times the rest of its product.

        The rest keeps whatever denominator it had, so the coefficient of
        `1/(x - 1)` goes over that rather than beside it. Which factors those
        are is the one thing this takes over from sympy's product spine: a
        factor raised to a negative power is a denominator, exactly as
        `_print_Mul` reads one. Sign extraction and the order of the factors
        are still sympy's, `as_ordered_factors` being what it prints from.
        """
        above, below = [], []
        for factor in rest.as_ordered_factors():
            base, power = factor.as_base_exp()
            if power.is_Rational and power.is_negative:
                below.append(base if power == -1 else base**-power)
            else:
                above.append(factor)
        text = written
        for factor in above:
            text += "*" + self.parenthesize(factor, PRECEDENCE["Mul"])
        if below:
            # One denominator, as sympy writes it: `a/(x*y)` and not `a/x/y`,
            # which is the same number drawn as a fraction inside a fraction.
            under = below[0] if len(below) == 1 else sp.Mul(*below)
            text += "/" + self.parenthesize(under, PRECEDENCE["Mul"])
        return text

    def _number(self, value: Fraction) -> str:
        """`value` in the notation `Options Notation` selects.

        Rational notation, and Mixed over a number simple enough for it, write
        the ratio; the numerals of a ratio are written in the session's base,
        as every whole number is. The decimal styles are base ten, the base
        being the one thing a decimal point numeral cannot carry.
        """
        style, digits = self.context.notation, self.context.notation_digits
        if style is Notation.RATIONAL or (
            style is Notation.MIXED and notation.simple(value)
        ):
            base = self.context.input_base
            if value.denominator == 1:
                return numeral(value.numerator, base)
            return f"{numeral(value.numerator, base)}/{numeral(value.denominator, base)}"
        if style is Notation.DECIMAL:
            return notation.decimal(value, digits)
        return notation.scientific(value, digits)

    def parenthesize(self, item, level, strict=False):
        """Fence a number the notation wrote as a power of ten.

        `1.23456*10^8` is a product where sympy prices a number as an atom, so
        nothing else would fence it: `x^123456789` has to come out with the
        exponent in parentheses to read back as itself. A product does not
        need them, `1.23456*10^8*x` associating the way it is meant to.
        """
        text = super().parenthesize(item, level, strict)
        if (
            isinstance(item, sp.Number)
            and level > PRECEDENCE["Mul"]
            and not text.startswith("(")
            and ("*" in text or "^" in text)
        ):
            return f"({text})"
        return text

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

    def _print_MatPow(self, expr):
        """A power of a matrix, written like any other power.

        `SQRT` of a matrix is one of these, and sympy's own spelling for it is
        `m**(1/2)` - which the grammar reads as a product with an empty factor
        in it, so the answer would come back as a different expression.
        """
        return self._print_Pow(expr)

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

    def _print_Piecewise(self, expr):
        """A case split, as the nested `IF` the notation has for one.

        Sympy answers a conditional integral or a conditional limit with a
        `Piecewise`, and `IF(test, then, else)` is what Derive calls that. A
        final `true` condition is the else clause; without one the last case
        has none, which is right - outside every condition the value is `?`.
        """
        pairs = list(expr.args)
        text = None
        if pairs and pairs[-1][1] is sp.true:
            text = self._print(pairs.pop()[0])
        for value, condition in reversed(pairs):
            parts = [self._print(condition), self._print(value)]
            if text is not None:
                parts.append(text)
            text = f"IF({', '.join(parts)})"
        return "?" if text is None else text

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

    def _print_Tuple(self, expr):
        """A tuple, as the vector the notation has for one.

        Sympy heads such as `hyper`, `meijerg` and `Subs` carry tuples of
        arguments, and `(1, m + 1)` is not something the grammar reads: a
        parenthesised list is not an expression. A vector is, so that is what
        it is written as, and the head reads back as an inert call over
        vectors.
        """
        return self._vector(expr.args)

    def _print_Subs(self, expr):
        """`SUBS(u, [x], [a])`, upper-cased like any other head.

        Sympy prints this one itself, in mixed case, and a name that changed
        case on the way out would not be a fixed point.
        """
        return f"SUBS({self.stringify(expr.args, ', ')})"

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

    def _print_Power(self, expr):
        base, exponent = expr.args
        level = PRECEDENCE["Pow"]
        # The exponent is not parenthesized at its own level, `^` associating
        # to the right: `10^10^10` reads back as the same tower it prints.
        written = self.parenthesize(base, level, strict=True)
        return f"{written}^{self.parenthesize(exponent, level)}"

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

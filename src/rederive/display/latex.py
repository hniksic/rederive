r"""Typesetting expression trees as LaTeX math, for a document read elsewhere.

The shape is the one the screen draws, decided by the same rules: the precedence ladder
and fencing of `forms.py`, and the layout's choices of where a quotient is built up
(`\frac`, at the top level and in matrices) and where it is linear (`a/b`, in exponents,
subscripts and the bounds of the big operators), how a run of terms is fenced, and how
the special forms take their bodies. Numbers and names are shown as the screen shows
them, in the output base and with the Greek letters as letters.

LaTeX does a few things better than a terminal grid, and there it does them: exponents
and subscripts are raised and lowered at any depth instead of turning into `^` and `↓`,
`√` takes an overbar instead of parentheses, a derivative of order `n` is
`\frac{d^n}{dx^n}` rather than `(d/dx)^n`, and the familiar functions get their usual
spellings, `\sin`, `\ln`, `\log_b`, `\Gamma`, `\lfloor x \rfloor`. A program is written
as the calls it is, since LaTeX math has no indented blocks: `IF`, `PROG` and `LOOP`
become `\operatorname{IF}(...)` and so on, like any other function without a notation of
its own.

Only what MathJax and KaTeX both render with no extra packages is used, and no backslash
is followed by ASCII punctuation, except in `\#`, `\$` and `\%`, which have no other
spelling both accept: GitHub drops the backslash from such a pair inside math, as
Markdown escaping, before MathJax sees it. So a matrix row ends in `\cr`, a thin space is
`\thinspace`, braces are `\lbrace` and `\rbrace`, an ampersand is `\And` and an
underscore is drawn with `\underline`.

`_Typesetter` walks the tree and decides the shape; `_Latex` spells each construct.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from rederive.display import forms, glyphs
from rederive.display.layout import ALONE, LINEAR_SCRIPTS
from rederive.display.options import ASTERISK, DOT, IMPLICIT, DisplayOptions
from rederive.model.expr import VARIADIC, Kind, Node, is_chain, is_deep
from rederive.syntax.names import GREEK_NAMES

_DEFAULTS = DisplayOptions()


def write_latex(node: Node, options: DisplayOptions = _DEFAULTS) -> str:
    """`node` as LaTeX math, without the delimiters of a math block."""
    return _Typesetter(options, _Latex()).box(node, 0)


# -- the shape ---------------------------------------------------------------


@dataclass
class _Typesetter:
    """Decides the shape of a tree as `layout.Engine` does, leaving the spelling to
    `latex`.

    The `level` passed down is the script level, as in the layout: a quotient or matrix
    is built up only at level 0.
    """

    options: DisplayOptions
    latex: _Latex
    shrinking: bool = False
    """Whether a quotient here would be set smaller than the screen draws it: inside
    another quotient or a matrix, where LaTeX shrinks a `\frac`."""

    def box(self, node: Node, level: int) -> str:
        """`node` at `level`, without fences."""
        latex = self.latex
        match node.kind:
            case Kind.NUMBER:
                options = self.options
                shown = glyphs.numeral(
                    str(node.value), options.output_base, options.notation_digits
                )
                return latex.numeral(shown)
            case Kind.NAME:
                return latex.name(str(node.value))
            case Kind.STRING:
                return latex.string(str(node.value))
            case Kind.LABEL:
                return latex.label(str(node.value))
            case Kind.UNKNOWN:
                return "?"
            case Kind.SUM | Kind.PRODUCT:
                return self._run(node, level)
            case Kind.BINOP:
                return self._binop(node, level)
            case Kind.UNOP:
                operand = self.operand(node.children[0], level, forms.MUL)
                return latex.sign(str(node.value), operand)
            case Kind.POSTOP:
                operand = self.operand(
                    node.children[0], level, forms.POSTFIX, forms.POSTFIX
                )
                return latex.postfix(operand, str(node.value), _raised(node.children[0]))
            case Kind.SUB:
                return self._subscript(node, level)
            case Kind.ABS:
                return latex.absolute(self.box(node.children[0], level))
            case Kind.CALL | Kind.APPLY:
                return self._application(node, level)
            case Kind.FUNCPOW:
                inner = self._application(node, level)
                return latex.power(inner, self.box(node.children[1], level + 1), False)
            case Kind.VECTOR:
                if forms.is_matrix(node) and level == 0:
                    with self._shrinking():
                        rows = [
                            [self.box(cell, level) for cell in line.children]
                            for line in node.children
                        ]
                    return latex.matrix(rows)
                return latex.vector(self._joined(node.children, level))
            case Kind.SET:
                return latex.set(self._joined(node.children, level))
            case Kind.ELLIPSIS:
                return latex.ellipsis()
            case Kind.SETOP:
                return self._binary(node, level, latex.set_operator(str(node.value)))
            case Kind.REL:
                left, right = node.children
                first = self.operand(left, level, forms.RELATION, forms.RELATION)
                second = self.operand(right, level, forms.ADD)
                return latex.infix(first, latex.relation(str(node.value)), second)
            case Kind.NOT:
                return latex.negation(self.operand(node.children[0], level, forms.NOT))
            case Kind.AND | Kind.OR | Kind.XOR | Kind.IMP | Kind.IFF:
                return self._binary(node, level, latex.logical(node.kind))
            case Kind.INFIX:
                return self._infix(node, level)
            case Kind.ASSIGN:
                target = self.box(node.children[0], level)
                value = (
                    self.box(node.children[1], level) if len(node.children) > 1 else None
                )
                return latex.assignment(target, str(node.value), value)
            case Kind.FUNDEF:
                head = self._head(node, level)
                body = (
                    self.box(node.children[1], level) if len(node.children) > 1 else None
                )
                return latex.assignment(head, ":=", body)
            case Kind.PARAMS:
                return self._joined(node.children, level)
            case Kind.DOMAIN:
                name = self.box(node.children[0], level)
                interval = None
                if len(node.children) > 1:
                    interval = self.box(node.children[1], level)
                return latex.domain(name, str(node.value), interval)
            case Kind.INTERVAL:
                # Always linear, as on screen.
                level = max(level, LINEAR_SCRIPTS)
                low, high = (self.box(child, level) for child in node.children)
                return latex.interval(str(node.value), low, high)
            case Kind.SHOWVALUE:
                inner = self.operand(
                    node.children[0], level, forms.ASSIGNMENT, forms.RELATION
                )
                return latex.shown_value(inner)
            case Kind.QUOTE:
                return latex.quoted(self.operand(node.children[0], level, forms.ATOM))
        raise AssertionError(f"untypesettable node kind: {node.kind}")

    def operand(self, node: Node, level: int, required: int, after: int = ALONE) -> str:
        """`node` where it needs at least `required` precedence, fenced as on screen;
        see `layout.Engine.operand`."""
        text = self.box(node, level)
        if forms.needs_fences(node, required, after):
            return self.latex.fenced(text)
        return text

    def _body(self, node: Node, level: int, required: int) -> str:
        """A special form's body, as `forms.body` decides: a built-up quotient is never
        fenced."""
        if forms.is_quotient(node) and level == 0:
            return self.box(node, level)
        return self.operand(node, level, required)

    @contextmanager
    def _shrinking(self) -> Iterator[None]:
        shrinking, self.shrinking = self.shrinking, True
        try:
            yield
        finally:
            self.shrinking = shrinking

    def _joined(self, children: tuple[Node, ...], level: int) -> str:
        return self.latex.joined([self.box(child, level) for child in children])

    # -- operators ----------------------------------------------------------

    def _run(self, node: Node, level: int) -> str:
        """A sum or product, its terms fenced as `layout.Engine._run` fences them."""
        own = forms.precedence(node)
        last = len(node.children) - 1
        terms = []
        for index, child in enumerate(node.children):
            term = self.operand(child, level, own, own if index < last else ALONE)
            if forms.fenced_in_run(node, child):
                term = self.latex.fenced(term)
            terms.append(term)
        if node.kind is Kind.PRODUCT:
            return self.latex.product(terms, self.options.times)
        return self.latex.sum(terms, str(node.value))

    def _binop(self, node: Node, level: int) -> str:
        operator = str(node.value)
        numerator, denominator = left, right = node.children
        if operator == "/":
            if level == 0:
                full = self.shrinking
                with self._shrinking():
                    top = self.box(numerator, level)
                    bottom = self.box(denominator, level)
                return self.latex.fraction(top, bottom, full)
            # Inside a script a quotient is never built up, so its operands are fenced
            # as ordinary ones.
            first = self.operand(numerator, level, forms.MUL, forms.MUL)
            second = self.operand(denominator, level, forms.NEG)
            return self.latex.infix(first, "/", second, spaced=False)
        if operator == "^":
            base = self.operand(left, level, forms.POSTFIX, forms.POW)
            return self.latex.power(base, self.box(right, level + 1), _raised(left))
        first = self.operand(left, level, forms.MUL, forms.MUL)
        second = self.operand(right, level, forms.MUL)
        return self.latex.infix(first, self.latex.DOT_PRODUCT, second)

    def _binary(self, node: Node, level: int, operator: str) -> str:
        """A left-associative operator: a set operation or a logical connective."""
        own = forms.precedence(node)
        left, right = node.children
        first = self.operand(left, level, own, own)
        second = self.operand(right, level, own + 1)
        return self.latex.infix(first, operator, second)

    def _infix(self, node: Node, level: int) -> str:
        """A word operator of Derive 6, as `layout.Engine._infix` draws it."""
        left, right = node.children
        word = str(node.value)
        if is_chain(node):
            first = self.operand(left, level, forms.POSTFIX, forms.POSTFIX)
            second = self.operand(right, level, forms.SCRIPT)
            if is_deep(node):
                return self.latex.deep_index(first, len(word.split()) - 1, second)
            return self.latex.infix(first, self.latex.word(word), second)
        own = forms.precedence(node)
        first = self.operand(left, level, own, own)
        second = self.operand(right, level, own + 1)
        return self.latex.infix(first, self.latex.infix_word(word), second)

    def _subscript(self, node: Node, level: int) -> str:
        """`v SUB i` lowered, nested subscripts merged into one, `v_{1,2}`, as on
        screen. A subscript is written linear, whatever it holds."""
        indices = []
        base = node
        while base.kind is Kind.SUB:
            base, index = base.children
            indices.append(self.box(index, LINEAR_SCRIPTS))
        under = self.operand(base, level, forms.POSTFIX, forms.POSTFIX)
        script = self.latex.joined(indices[::-1], tight=True)
        return self.latex.subscript(under, script, _lowered(base))

    # -- applications -------------------------------------------------------

    def _application(self, node: Node, level: int) -> str:
        """A call, a bare application, or the application inside a `FUNCPOW`."""
        name_node, arguments = forms.application(node)
        name = str(name_node.value)
        latex = self.latex
        if forms.special(node) is not None:
            return self._special(name, arguments, level)
        if (name, len(arguments)) == ("LOG", 2):
            base = self.box(arguments[1], level + 1)
            return latex.call(latex.logarithm(base), self.box(arguments[0], level))
        if len(arguments) == 1 and name in latex.BRACKETED:
            return latex.bracketed(name, self.box(arguments[0], level))
        joined = self._joined(arguments, level)
        argument = arguments[0] if len(arguments) == 1 else None
        if (
            argument is not None
            and forms.is_matrix(argument)
            and len(argument.children) > 1
            and level == 0
        ):
            # As on screen, a call drops its parentheses around a built-up matrix.
            return latex.bare_call(latex.function(name), joined)
        return latex.call(latex.function(name), joined)

    def _special(self, name: str, arguments: tuple[Node, ...], level: int) -> str:
        """The forms of `forms.FORMS`, their bodies fenced as there."""
        latex = self.latex
        if name == "SQRT":
            return latex.root(self.box(arguments[0], level))
        if name == "ABS":
            return latex.absolute(self.box(arguments[0], level))
        if name == "INT":
            integrand = self._body(arguments[0], level, forms.MUL)
            variable = self.box(arguments[1], level)
            bounds = None
            if len(arguments) == 4:
                bounds = tuple(self.box(bound, level + 1) for bound in arguments[2:])
            return latex.integral(integrand, variable, bounds)
        if name in ("SUM", "PRODUCT"):
            body = self._body(arguments[0], level, forms.MUL)
            index = self.box(arguments[1], level + 1)
            if len(arguments) == 2:
                return latex.big_operator(name, body, index, None, None)
            start, end = (self.box(bound, level + 1) for bound in arguments[2:])
            return latex.big_operator(name, body, index, start, end)
        if name == "LIM":
            body = self._body(arguments[0], level, forms.MUL)
            variable = self.box(arguments[1], level + 1)
            point = self.box(arguments[2], level + 1)
            side = "" if len(arguments) == 3 else forms.side_of(arguments[3])
            return latex.limit(body, variable, point, side or "")
        # The derivative, the remaining form.
        body = self._body(arguments[0], level, forms.POW)
        variable = self.box(arguments[1], level)
        order = self.box(arguments[2], level + 1) if len(arguments) == 3 else None
        return latex.derivative(body, variable, order)

    def _head(self, node: Node, level: int) -> str:
        """The head `F(x, y)` of a definition, or `F v` for a variadic one."""
        parameters = node.children[0]
        listed = self._joined(parameters.children, level)
        function = self.latex.function(str(node.value))
        if parameters.value == VARIADIC:
            return self.latex.bare_call(function, listed)
        return self.latex.call(function, listed)


def _raised(node: Node) -> bool:
    """Whether `node` ends in a superscript of its own, which a power or another
    raised mark must not follow directly."""
    if node.kind is Kind.NAME:
        return node.value == "deg"
    return node.kind is Kind.POSTOP and str(node.value) in _Latex.RAISED_POSTFIX


# -- the spelling ------------------------------------------------------------


class _Latex:
    """How LaTeX spells each construct; the shape has already been decided."""

    #: The dot product, a bullet as on screen, apart from the multiplication dot.
    DOT_PRODUCT = r"\bullet"

    #: The postfix operators that are written as a superscript.
    RAISED_POSTFIX = {"DEG": r"^{\circ}", "`": r"^{\mathrm{T}}"}

    #: Functions with a delimiter of their own instead of a name.
    BRACKETED = {"FLOOR": (r"\lfloor", r"\rfloor"), "CEILING": (r"\lceil", r"\rceil")}

    # -- leaves -------------------------------------------------------------

    @staticmethod
    def numeral(shown: str) -> str:
        """A numeral as the screen shows it. Digits above nine are letters, set
        upright so they don't read as variables."""
        if any(character.isalpha() for character in shown):
            return rf"\mathrm{{{shown}}}"
        return shown

    @staticmethod
    def name(value: str) -> str:
        constant = _CONSTANTS.get(value)
        if constant is not None:
            return constant
        return _variable(value)

    @staticmethod
    def function(value: str) -> str:
        """A function's name: its usual spelling if it has one, else as written."""
        known = _FUNCTIONS.get(value)
        if known is not None:
            return known
        stem, primes = _primed(value)
        if _is_letter(stem) or _indexed(stem) is not None:
            return _variable(value)
        return rf"\operatorname{{{_escaped(stem)}}}{primes}"

    @staticmethod
    def string(value: str) -> str:
        """A string in quotes, as text. The characters LaTeX treats specially are set
        as math between runs of text, since text mode in MathJax doesn't know their
        escapes and KaTeX rejects them bare."""
        parts = []
        for run in re.split(r"([\\#$%&_{}~^])", f"“{value}”"):
            if run in _SPECIAL:
                parts.append(_SPECIAL[run].strip())
            elif run:
                parts.append(rf"\text{{{run}}}")
        return "".join(parts)

    @staticmethod
    def label(value: str) -> str:
        return rf"\#{value}"

    @staticmethod
    def ellipsis() -> str:
        return r"\ldots"

    # -- grouping -----------------------------------------------------------

    @staticmethod
    def fenced(inner: str) -> str:
        return _delimited("(", inner, ")")

    @staticmethod
    def joined(items: list[str], tight: bool = False) -> str:
        return ("," if tight else ", ").join(items)

    @staticmethod
    def vector(joined: str) -> str:
        return _delimited("[", joined, "]")

    @staticmethod
    def set(joined: str) -> str:
        return _delimited(r"\lbrace ", joined, r" \rbrace")

    @staticmethod
    def matrix(rows: list[list[str]]) -> str:
        body = r" \cr ".join(" & ".join(row) for row in rows)
        return rf"\begin{{bmatrix}} {body} \end{{bmatrix}}"

    @staticmethod
    def interval(brackets: str, low: str, high: str) -> str:
        opening, closing = brackets
        return _delimited(opening, f"{low}, {high}", closing)

    @staticmethod
    def absolute(inner: str) -> str:
        if any(tall in inner for tall in _TALL):
            return rf"\left| {inner} \right|"
        # Plain `|` is an ordinary symbol, after which a sign would be spaced as a
        # binary operator.
        return rf"\lvert {inner} \rvert"

    @staticmethod
    def root(inner: str) -> str:
        return rf"\sqrt{{{inner}}}"

    # -- operators ----------------------------------------------------------

    @staticmethod
    def sum(terms: list[str], gaps: str) -> str:
        parts = [terms[0]]
        for gap, term in zip(gaps, terms[1:], strict=True):
            parts += [f" {gap} ", term]
        return "".join(parts)

    @staticmethod
    def product(factors: list[str], times: str) -> str:
        return _TIMES[times].join(factors)

    @staticmethod
    def fraction(numerator: str, denominator: str, full: bool) -> str:
        """A built-up quotient; `full` keeps it at the size of the text around it."""
        command = r"\dfrac" if full else r"\frac"
        return rf"{command}{{{numerator}}}{{{denominator}}}"

    @staticmethod
    def power(base: str, exponent: str, raised: bool) -> str:
        """`base` raised to `exponent`. A `raised` base already ends in a superscript,
        so it's braced: TeX rejects two superscripts in a row."""
        if raised:
            base = f"{{{base}}}"
        return f"{base}^{{{exponent}}}"

    @staticmethod
    def subscript(base: str, script: str, lowered: bool) -> str:
        """`base` with `script` lowered. A `lowered` base already has a subscript, and
        TeX rejects two in a row; braced, the second would sit beside the first and read
        as one, `x_{12}`, so the base is fenced instead."""
        if lowered:
            base = _delimited("(", base, ")")
        return f"{base}_{{{script}}}"

    @staticmethod
    def deep_index(base: str, depth: int, script: str) -> str:
        """`A SUB SUB j`, drawn with its arrows as on screen."""
        return base + r"{\downarrow}" * (depth + 1) + script

    @staticmethod
    def sign(sign: str, operand: str) -> str:
        return _SIGNS.get(sign, sign) + operand

    @classmethod
    def postfix(cls, operand: str, operator: str, raised: bool) -> str:
        script = cls.RAISED_POSTFIX.get(operator)
        if script is None:
            return operand + (r"\%" if operator == "%" else operator)
        if raised:
            operand = f"{{{operand}}}"
        return operand + script

    @staticmethod
    def infix(left: str, operator: str, right: str, spaced: bool = True) -> str:
        return f"{left} {operator} {right}" if spaced else f"{left}{operator}{right}"

    @staticmethod
    def relation(operator: str) -> str:
        return _RELATIONS.get(operator, operator)

    @staticmethod
    def logical(kind: Kind) -> str:
        return _LOGICAL[kind]

    @staticmethod
    def negation(operand: str) -> str:
        return rf"\lnot {operand}"

    @staticmethod
    def set_operator(operator: str) -> str:
        return _SET_OPERATORS[operator]

    @staticmethod
    def word(word: str) -> str:
        return rf"\mathbin{{\mathrm{{{word}}}}}"

    @staticmethod
    def infix_word(word: str) -> str:
        return _INFIX_WORDS[word]

    @staticmethod
    def assignment(target: str, operator: str, value: str | None) -> str:
        """`target := value`, or an update such as `:+`, whose `+` is braced so it
        isn't spaced as a binary operator."""
        if operator != ":=":
            operator = "".join(f"{{{character}}}" for character in operator)
        spelled = rf"\mathrel{{{operator}}}"
        return f"{target} {spelled}" if value is None else f"{target} {spelled} {value}"

    @staticmethod
    def shown_value(inner: str) -> str:
        return f"{inner} ="

    @staticmethod
    def domain(name: str, domain: str, interval: str | None) -> str:
        spelled = _DOMAINS.get(domain, rf"\mathrm{{{_escaped(domain)}}}")
        written = rf"{name} \in {spelled}"
        return written if interval is None else rf"{written}\thinspace {interval}"

    @staticmethod
    def quoted(operand: str) -> str:
        return r"\text{'}" + operand

    # -- calls and the special forms ----------------------------------------

    @staticmethod
    def call(function: str, joined: str) -> str:
        return function + _delimited("(", joined, ")")

    @staticmethod
    def bare_call(function: str, joined: str) -> str:
        return rf"{function}\thinspace {joined}"

    @classmethod
    def bracketed(cls, name: str, inner: str) -> str:
        opening, closing = cls.BRACKETED[name]
        return _delimited(opening + " ", inner, " " + closing)

    @staticmethod
    def logarithm(base: str) -> str:
        return rf"\log_{{{base}}}"

    @staticmethod
    def integral(
        integrand: str, variable: str, bounds: tuple[str, ...] | None
    ) -> str:
        sign = r"\int" if bounds is None else rf"\int_{{{bounds[0]}}}^{{{bounds[1]}}}"
        return rf"{sign} {integrand} \thinspace d{variable}"

    @staticmethod
    def big_operator(
        name: str, body: str, index: str, start: str | None, end: str | None
    ) -> str:
        sign = r"\sum" if name == "SUM" else r"\prod"
        if start is None:
            return rf"{sign}_{{{index}}} {body}"
        return rf"{sign}_{{{index}={start}}}^{{{end}}} {body}"

    @staticmethod
    def limit(body: str, variable: str, point: str, side: str) -> str:
        if side:
            point = f"{{{point}}}^{{{side}}}"
        return rf"\lim_{{{variable} \to {point}}} {body}"

    @staticmethod
    def derivative(body: str, variable: str, order: str | None) -> str:
        if order is None:
            return rf"\frac{{d}}{{d{variable}}} {body}"
        return rf"\frac{{d^{{{order}}}}}{{d{variable}^{{{order}}}}} {body}"


def _delimited(opening: str, inner: str, closing: str) -> str:
    r"""`inner` between delimiters, which grow with it if it's taller than a line, as the
    screen's fences do. Plain delimiters are used otherwise, since `\left` and `\right`
    space themselves off from a function name: `f\left(x\right)` reads `f (x)`."""
    if any(tall in inner for tall in _TALL):
        return rf"\left{opening.strip()} {inner} \right{closing.strip()}"
    return f"{opening}{inner}{closing}"


#: What makes a formula taller than a line of text: a built-up quotient, a matrix, a big
#: operator with its limits above and below, or delimiters already grown.
_TALL = (r"\frac", r"\dfrac", r"\begin", r"\sum", r"\prod", r"\int", r"\lim", r"\left")


# -- names -------------------------------------------------------------------

#: An underscore in a name. `\_` is the usual spelling, but loses its backslash on GitHub.
_UNDERSCORE = r"\underline{\hphantom{x}}"

#: The constants, as the screen shows them where LaTeX has the glyph, upright where
#: they are words. `#e` and `#i` are upright, as is usual for the constants, which keeps
#: them apart from variables `e` and `i`.
_CONSTANTS: dict[str, str] = {
    "pi": r"\pi",
    "inf": r"\infty",
    "deg": r"{}^{\circ}",
    "#e": r"\mathrm{e}",
    "#i": r"\mathrm{i}",
    "true": r"\mathrm{true}",
    "false": r"\mathrm{false}",
    "euler_gamma": rf"\mathrm{{euler{_UNDERSCORE}gamma}}",
    "unit_circle": rf"\mathrm{{unit{_UNDERSCORE}circle}}",
}

#: The Greek letters by name, as LaTeX spells them. The capitals that look like Latin
#: letters have no command of their own and are upright Latin letters; the small
#: epsilon and phi are the variants the screen's glyphs show.
_GREEK: dict[str, str] = {
    name: "\\" + name
    for name in """
    alpha beta gamma delta zeta eta theta iota kappa lambda mu nu xi rho sigma tau
    upsilon chi psi omega Gamma Delta Theta Lambda Xi Sigma Upsilon Phi Psi Omega
    """.split()
} | {
    "epsilon": r"\varepsilon",
    "phi": r"\varphi",
    "omicron": "o",
    "Alpha": r"\mathrm{A}",
    "Beta": r"\mathrm{B}",
    "Epsilon": r"\mathrm{E}",
    "Zeta": r"\mathrm{Z}",
    "Eta": r"\mathrm{H}",
    "Iota": r"\mathrm{I}",
    "Kappa": r"\mathrm{K}",
    "Mu": r"\mathrm{M}",
    "Nu": r"\mathrm{N}",
    "Omicron": r"\mathrm{O}",
    "Rho": r"\mathrm{P}",
    "Tau": r"\mathrm{T}",
    "Chi": r"\mathrm{X}",
}

#: Functions with a usual spelling: LaTeX's own operator names, and a few that are
#: always written so in mathematics.
_FUNCTIONS: dict[str, str] = {
    name: "\\" + name.lower()
    for name in """
    SIN COS TAN COT SEC CSC SINH COSH TANH COTH LN LOG EXP MAX MIN GCD DET
    """.split()
} | {
    "ASIN": r"\arcsin",
    "ACOS": r"\arccos",
    "ATAN": r"\arctan",
    "GAMMA": r"\Gamma",
} | {
    name: rf"\operatorname{{{spelling}}}"
    for name, spelling in {
        "ACOT": "arccot",
        "ASEC": "arcsec",
        "ACSC": "arccsc",
        "SECH": "sech",
        "CSCH": "csch",
        "ASINH": "arsinh",
        "ACOSH": "arcosh",
        "ATANH": "artanh",
        "ACOTH": "arcoth",
        "ERF": "erf",
        "ERFC": "erfc",
        "SIGN": "sign",
        "LCM": "lcm",
    }.items()
}

#: The characters LaTeX math gives a meaning of their own, and their literal spelling.
_SPECIAL = {
    "\\": r"\backslash ",
    "#": r"\#",
    "$": r"\$",
    "%": r"\%",
    "&": r"\And ",
    "_": _UNDERSCORE,
    "{": r"\lbrace ",
    "}": r"\rbrace ",
    "~": r"\sim ",
    "^": r"\hat{}",
}

def _variable(value: str) -> str:
    """A variable's name. One letter is set as it is, a Greek letter as the letter, and
    a longer name as one italic word, so that `ab` doesn't read as `a` times `b`. A
    letter with an index after an underscore, `x_1` or `a_max`, is subscripted. Primes
    stay outside, where TeX raises them."""
    stem, primes = _primed(value)
    indexed = _indexed(stem)
    if indexed is not None:
        letter, index = indexed
        index = _escaped(index) if index.isdigit() else _word(index)
        return f"{_letter(letter)}_{{{index}}}{primes}"
    if _is_letter(stem):
        return _letter(stem) + primes
    return _word(stem) + primes


def _indexed(stem: str) -> tuple[str, str] | None:
    """The letter and index of a name such as `x_1`, or None if it isn't one."""
    letter, underscore, index = stem.partition("_")
    if underscore and index and _is_letter(letter) and index.isalnum():
        return letter, index
    return None


def _lowered(node: Node) -> bool:
    """Whether `node` is a name written with a subscript of its own, which another
    subscript must not follow directly."""
    if node.kind is not Kind.NAME or str(node.value) in _CONSTANTS:
        return False
    return _indexed(_primed(str(node.value))[0]) is not None


def _is_letter(stem: str) -> bool:
    return len(stem) == 1 or stem in GREEK_NAMES


def _letter(stem: str) -> str:
    """One letter, Greek or not."""
    greek = GREEK_NAMES.get(stem)
    if greek is not None:
        return _GREEK[greek]
    return _escaped(stem)


def _word(text: str) -> str:
    """A longer name, or the index of one, as a single italic word."""
    if _is_letter(text):
        return _letter(text)
    return rf"\mathit{{{_escaped(text)}}}"


def _primed(value: str) -> tuple[str, str]:
    """A name split into its stem and the primes that end it."""
    stem = value.rstrip("'")
    return stem, value[len(stem) :]


def _escaped(text: str) -> str:
    """The characters of a name as math: LaTeX's special characters escaped, a Greek
    letter as its command, and any other letter beyond ASCII as text, which math mode
    has no glyph for."""
    spelled = []
    for character in text:
        greek = GREEK_NAMES.get(character)
        if character in _SPECIAL:
            spelled.append(_SPECIAL[character])
        elif greek is not None:
            spelled.append(f"{{{_GREEK[greek]}}}")
        elif not character.isascii():
            spelled.append(rf"\text{{{character}}}")
        else:
            spelled.append(character)
    return "".join(spelled)


#: Operators, by their canonical spelling.
_SIGNS = {"+-": r"\pm "}
_RELATIONS = {"/=": r"\ne", "<=": r"\le", ">=": r"\ge"}
_LOGICAL = {
    Kind.AND: r"\land",
    Kind.OR: r"\lor",
    Kind.XOR: r"\veebar",
    Kind.IMP: r"\Rightarrow",
    Kind.IFF: r"\Leftrightarrow",
}
_SET_OPERATORS = {"UNION": r"\cup", "INTERSECTION": r"\cap", "\\": r"\setminus"}
_INFIX_WORDS = {
    "SUBSET": r"\subseteq",
    "DOTPRODUCT": r"\bullet",
    "CROSSPRODUCT": r"\times",
}
_TIMES = {DOT: r" \cdot ", ASTERISK: r" \ast ", IMPLICIT: r" \thinspace "}
_DOMAINS = {"Integer": r"\mathbb{Z}", "Real": r"\mathbb{R}", "Complex": r"\mathbb{C}"}

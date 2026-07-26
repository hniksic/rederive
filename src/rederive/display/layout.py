"""Node in, Box out.

One pass, purely functional, driven by two pieces of context: the script level
and what is left of the caller's height budget.

The script level is how deep inside a superscript or a subscript we are.
Fractions and matrices are built up at level 0 and linear below it, and `^`
itself stops raising at level 2, so an exponent tower flattens to `y^z^w`
rather than climbing off the top of the pane.

The height budget degrades outermost-first. Every node's unconstrained box is
memoised, so asking "how tall would this be built up?" costs one layout per
node however often the question is put.

Each box also names its selectable operands, because what the cursor can
reach is decided by how a node is drawn: a chain of sums offers its terms
rather than its left-nested pairs, a call offers its arguments and never its
name, and a special form names its own. The original's rule is that you
select what you see.
"""

from __future__ import annotations

from rederive.display import forms, glyphs
from rederive.display.boxes import (
    Box,
    Layout,
    Placed,
    centered,
    fenced,
    paint,
    row,
    text,
)
from rederive.display.options import DisplayOptions
from rederive.model.expr import Kind, Node

#: Word operators keep their spaces in both formats: they would not lex
#: otherwise, `xANDy` being one name. `NOT` is the other one and prints as a
#: prefix; `SUB` never prints as a word at all, since a subscript too deep to
#: lower writes the arrow `a↓b` rather than being spelled out.
_WORD_OPERATORS = {
    Kind.AND: "AND",
    Kind.OR: "OR",
    Kind.XOR: "XOR",
    Kind.IMP: "IMP",
}

#: Beyond this level a script is written flat, `^` as `a^b` and `SUB` as `a↓b`.
_LINEAR_SCRIPTS = 2

#: `after` for an operand with nothing drawn to its right. Below every
#: precedence on the ladder, so it can absorb nothing.
ALONE = -1

#: The operators drawn as a run of terms rather than as nested pairs. `a + b
#: - c` offers three operands and the sign belongs to the chain, so the third
#: is `c` and not `-c`. `/` and the dot product are absent: each is drawn as a
#: group of its own, so `a·b/c` offers `a·b` and `c`.
_CHAINS = {"+": "sum", "-": "sum", "*": "product"}

#: The default options, held as one value so that the common call builds none.
_DEFAULTS = DisplayOptions()


def render(node: Node, options: DisplayOptions = _DEFAULTS) -> Layout:
    """Typeset `node`. The result is as wide as it is; the pane clips it."""
    engine = Engine(options)
    return paint(engine.box(node, 0, options.height))


class Engine:
    """One render's worth of layout.

    Public to `forms.py`, which is handed an instance rather than importing
    back into here.
    """

    def __init__(self, options: DisplayOptions) -> None:
        self.options = options
        self._free: dict[tuple[int, int], Box] = {}

    # -- what the forms use -------------------------------------------------

    def box(self, node: Node, level: int, budget: int | None) -> Box:
        """`node` as a box, degraded only as far as `budget` forces."""
        free = self.free(node, level)
        if budget is None or free.height <= budget:
            return free
        return self._build(node, level, budget)

    def free(self, node: Node, level: int) -> Box:
        """`node` laid out with no height limit, measured once and kept."""
        key = (id(node), level)
        box = self._free.get(key)
        if box is None:
            box = self._build(node, level, None)
            self._free[key] = box
        return box

    def operand(
        self,
        node: Node,
        level: int,
        budget: int | None,
        required: int,
        after: int = ALONE,
    ) -> Box:
        """`node` in a position that needs at least `required` precedence.

        `after` is how tightly the operator drawn to the right of this operand
        binds, in the box the caller is building; `ALONE` when nothing is
        drawn there. It decides a form that ends in its body, which no
        precedence of its own can: `Σ k` is fenced before a `+ 1` and bare
        after a `2·`, sitting beside the same operator either way. A comma, a
        closing bracket and the `dx` of an integral are terminators rather
        than operators, so they leave an operand `ALONE`.
        """
        box = self.box(node, level, budget)
        if self._needs_fences(node, required, after):
            return self.fence(box)
        return box

    @staticmethod
    def _needs_fences(node: Node, required: int, after: int) -> bool:
        body = forms.body_precedence(node)
        if body is None:
            return forms.precedence(node) < required
        # A form that ends in its body is fenced by what would be read into
        # that body: an operator to its right that binds at least as tightly,
        # or a prefix as tight as `√`, which takes a power's operand. A looser
        # prefix - a unary minus, a `2·` - reads correctly without them.
        return after >= body or required >= forms.POW

    def fence(self, box: Box) -> Box:
        return fenced(box, glyphs.PARENS)

    def bar(self, width: int) -> str:
        """A fraction bar. Compressed format ends it in a half-width cell."""
        if self.options.compressed:
            return glyphs.BAR * (width - 1) + glyphs.HALF_BAR
        return glyphs.BAR * width

    def raised(
        self,
        base: Box,
        script: Box,
        node: Node | None = None,
        operands: tuple[Box, ...] = (),
    ) -> Box:
        """`script` immediately right of `base`, one row clear of its baseline.

        The script's columns stay blank on the baseline row, which is why
        `x^2 + 2x` renders `x  + 2·x` with two spaces.
        """
        return Box(
            base.width + script.width,
            max(base.above, script.height),
            base.below,
            node,
            "",
            (Placed(base, 0, 0), Placed(script, base.width, -1 - script.below)),
            operands,
        )

    def built_up(self, node: Node, level: int, budget: int | None) -> bool:
        """Whether a division or a matrix takes its built-up form here."""
        if level > 0:
            return False
        return budget is None or self.free(node, level).height <= budget

    # -- dispatch -----------------------------------------------------------

    def _build(self, node: Node, level: int, budget: int | None) -> Box:
        match node.kind:
            case Kind.NUMBER:
                base = self.options.output_base
                return text(glyphs.numeral(str(node.value), base), node)
            case Kind.NAME:
                return text(glyphs.name(str(node.value)), node)
            case Kind.STRING:
                return text(f'"{node.value}"', node)
            case Kind.LABEL:
                return text(f"#{node.value}", node)
            case Kind.UNKNOWN:
                return text("?", node)
            case Kind.BINOP:
                return self._binop(node, level, budget)
            case Kind.UNOP:
                return self._sign(node, level, budget)
            case Kind.POSTOP:
                inner = self.operand(
                    node.children[0], level, budget, forms.POSTFIX, forms.POSTFIX
                )
                return row([inner, text(str(node.value))], node, (inner,))
            case Kind.SUB:
                return self._subscript(node, level, budget)
            case Kind.ABS:
                return forms.absolute_value(self, node, node.children, level, budget)
            case Kind.CALL | Kind.APPLY:
                return self._application(node, node, level, budget)
            case Kind.FUNCPOW:
                # The application and the exponent are two operands, as they
                # are in the original, which holds a power of an application
                # where this tree holds one node. Both regions name that node,
                # there being no other to name.
                inner = self._application(node, node, level, budget)
                exponent = self.box(node.children[1], level + 1, budget)
                return self.raised(inner, exponent, node, (inner, exponent))
            case Kind.VECTOR:
                return self._vector(node, level, budget)
            case Kind.REL:
                return self._relation(node, level, budget)
            case Kind.NOT:
                inner = self.operand(node.children[0], level, budget, forms.NOT)
                return row([text("NOT "), inner], node, (inner,))
            case Kind.AND | Kind.OR | Kind.XOR | Kind.IMP:
                return self._logical(node, level, budget)
            case Kind.ASSIGN:
                return self._assignment(node, level, budget)
            case Kind.FUNDEF:
                return self._definition(node, level, budget)
            case Kind.PARAMS:
                # A definition draws its own parameter list, name and all, so
                # this is the standalone case: the dispatch is total over the
                # kinds rather than only over the ones reached from a root.
                return self._joined(node.children, level, budget, node)
            case Kind.DOMAIN:
                return self._domain(node, level, budget)
            case Kind.INTERVAL:
                return self._interval(node, level, budget)
            case Kind.SHOWVALUE:
                inner = self.operand(
                    node.children[0], level, budget, forms.ASSIGNMENT, forms.RELATION
                )
                return row([inner, text(self._spaced("=").rstrip())], node, (inner,))
        raise AssertionError(f"unrenderable node kind: {node.kind}")

    # -- spacing ------------------------------------------------------------

    def _spaced(self, symbol: str) -> str:
        """A symbol operator, spaced unless the format is Compressed."""
        return symbol if self.options.compressed else f" {symbol} "

    # -- operators ----------------------------------------------------------

    def _binop(self, node: Node, level: int, budget: int | None) -> Box:
        operator = str(node.value)
        if operator == "/":
            return self._division(node, level, budget)
        if operator == "^":
            return self._power(node, level, budget)
        left, right = node.children
        if operator in ("*", "."):
            spelling = (
                glyphs.TIMES[self.options.times]
                if operator == "*"
                else glyphs.DOT_PRODUCT
            )
            first = self._factor(left, level, budget, forms.MUL, forms.MUL)
            second = self.operand(right, level, budget, forms.MUL)
            return row(
                [first, text(spelling), second],
                node,
                self._terms(node, left, first) + self._terms(node, right, second),
            )
        # `a - (b - c)` needs its fences where `a + (b + c)` does not.
        tighter = forms.ADD if operator == "+" else forms.MUL
        first = self.operand(left, level, budget, forms.ADD, forms.ADD)
        second = self.operand(right, level, budget, tighter)
        return row(
            [first, text(self._spaced(operator)), second],
            node,
            self._terms(node, left, first) + self._terms(node, right, second),
        )

    @staticmethod
    def _terms(node: Node, child: Node, box: Box) -> tuple[Box, ...]:
        """What `child` contributes to `node`'s operands.

        A chain splices in the terms of a same-kind operand instead of
        offering it whole, so `a + b - c` offers three. A fenced operand
        splices to itself, which is right: `a - (b - c)` is drawn as two
        terms and the parentheses are what say so.
        """
        chain = _CHAINS.get(str(node.value)) if node.kind is Kind.BINOP else None
        if chain is not None and _CHAINS.get(_binop_value(child)) == chain:
            return box.operands
        return (box,)

    def _sign(
        self, node: Node, level: int, budget: int | None, detached: bool = False
    ) -> Box:
        """`-u`, `+u` or `±u`.

        The sign abuts a leaf and stands off anything else: `-x`, `3 - -2`,
        but `- SIN(x)`, `- √x`, `- [1, 2]` and `- a/b`. What it governs is
        plain when the operand is one token and worth marking when it is not.
        """
        operand = self.operand(node.children[0], level, budget, forms.MUL)
        sign = glyphs.operator(str(node.value))
        if detached or not node.children[0].is_atom:
            sign += " "
        return row([text(sign), operand], node, (operand,))

    def _factor(
        self, node: Node, level: int, budget: int | None, required: int, after: int
    ) -> Box:
        """The left operand of a product, a linear quotient or a dot product.

        `rederive.syntax` binds a unary sign tighter than these operators,
        where the original binds it looser: `-x/y` arrives here as `(-x)/y`
        where the original holds `-(x/y)`. Standing the sign off renders what
        the original renders for the same authored text, `- x/y` and `- x·y`.

        A compensation, not a rule, and it can go once the parser agrees: with
        the sign outside, the operand of the sign is a quotient rather than a
        leaf and it stands off on its own. A built-up numerator is left alone
        either way, since `(-x)/y` really does put `-x` over the bar.
        """
        if _is_negation(node):
            return self._sign(node, level, budget, detached=True)
        return self.operand(node, level, budget, required, after)

    def _division(self, node: Node, level: int, budget: int | None) -> Box:
        numerator, denominator = node.children
        if self.built_up(node, level, budget):
            # Neither operand ever needs fences: the bar groups them.
            top = self.box(numerator, level, budget)
            bottom = self.box(denominator, level, budget)
            width = max(top.width, bottom.width) + 2
            return Box(
                width,
                top.height,
                bottom.height,
                node,
                "",
                (
                    Placed(text(self.bar(width)), 0, 0),
                    Placed(top, centered(width, top.width), -1 - top.below),
                    Placed(bottom, centered(width, bottom.width), 1 + bottom.above),
                ),
                (top, bottom),
            )
        left = self._quotient_operand(numerator, level, budget, forms.MUL, forms.MUL)
        right = self._quotient_operand(denominator, level, budget, forms.NEG, ALONE)
        return row([left, text("/"), right], node, (left, right))

    def _quotient_operand(
        self, node: Node, level: int, budget: int | None, required: int, after: int
    ) -> Box:
        """An operand of a linear `/`, where a built-up fraction needs no fences."""
        if _is_quotient(node) and self.built_up(node, level, budget):
            return self.box(node, level, budget)
        return self._factor(node, level, budget, required, after)

    def _power(self, node: Node, level: int, budget: int | None) -> Box:
        base, exponent = node.children
        under = self.operand(base, level, budget, forms.SCRIPT, forms.POW)
        if level >= _LINEAR_SCRIPTS:
            script = self.operand(exponent, level, budget, forms.NEG)
            return row([under, text("^"), script], node, (under, script))
        # Raised, so the exponent needs no fences however loose it is.
        script = self.box(exponent, level + 1, budget)
        return self.raised(under, script, node, (under, script))

    def _subscript(self, node: Node, level: int, budget: int | None) -> Box:
        base, script = node.children
        if level >= _LINEAR_SCRIPTS:
            # Flat: `a↓b`, the arrow standing in for the row there is no
            # longer any room to lower into, exactly as `a^b` stands in for
            # the row there is no room to raise into.
            flat = self.operand(base, level, budget, forms.SCRIPT, forms.SCRIPT)
            deeper = self.operand(script, level, budget, forms.POSTFIX)
            return row(
                [flat, text(glyphs.LOWERED), deeper], node, (flat, deeper)
            )
        lowered = self.box(script, level + 1, budget)
        under = self.operand(base, level, budget, forms.SCRIPT, forms.SCRIPT)
        dy = 1 + lowered.above
        parts = [Placed(under, 0, 0)]
        dx = under.width
        if base.kind is Kind.SUB:
            # A subscript on a subscript merges: `v SUB 1 SUB 2` is `v` over
            # `1,2` rather than a script hung off the first script's corner.
            parts.append(Placed(text(","), dx, dy))
            dx += 1
        parts.append(Placed(lowered, dx, dy))
        return Box(
            max(under.width, dx + lowered.width),
            under.above,
            max(under.below, dy + lowered.below),
            node,
            "",
            tuple(parts),
            (under, lowered),
        )

    def _relation(self, node: Node, level: int, budget: int | None) -> Box:
        """A chain of relations, which the parser keeps as one node."""
        operators = str(node.value).split()
        first = node.children[0]
        operands = [self.operand(first, level, budget, forms.ADD, forms.RELATION)]
        parts = [operands[0]]
        for index, operator in enumerate(operators, start=1):
            child = node.children[index]
            after = ALONE if index == len(operators) else forms.RELATION
            operands.append(self.operand(child, level, budget, forms.ADD, after))
            parts.append(text(self._spaced(glyphs.operator(operator))))
            parts.append(operands[-1])
        return row(parts, node, tuple(operands))

    def _logical(self, node: Node, level: int, budget: int | None) -> Box:
        word = _WORD_OPERATORS[node.kind]
        own = forms.precedence(node)
        left, right = node.children
        first = self.operand(left, level, budget, own, own)
        second = self.operand(right, level, budget, own + 1)
        return row([first, text(f" {word} "), second], node, (first, second))

    # -- applications -------------------------------------------------------

    def _application(
        self, node: Node, owner: Node | None, level: int, budget: int | None
    ) -> Box:
        """A call, a bare application, or the application inside a `FUNCPOW`.

        `owner` is the node the box belongs to, which is `None` for a
        `FUNCPOW`: there the exponent goes on the outside and the box that
        carries the node is the raised one.
        """
        form = forms.special(node)
        name, arguments = forms.application(node)
        if form is not None:
            return form(self, owner, arguments, level, budget)
        # Arguments never need fences; the call's own parentheses group them.
        # The name is drawn as the head of the application rather than as
        # something applied to, so it carries no node and is never selectable.
        listed = fenced(self._joined(arguments, level, budget), glyphs.PARENS)
        return row([text(str(name.value)), listed], owner, (listed,))

    def _joined(
        self,
        children: tuple[Node, ...],
        level: int,
        budget: int | None,
        node: Node | None = None,
    ) -> Box:
        """A comma-separated list, spaced the same way in both formats.

        An element needs no fences whatever it is: the brackets or the call's
        own parentheses group the list, and a comma terminates a form's body
        rather than being read into it.
        """
        parts: list[Box] = []
        elements: list[Box] = []
        for index, child in enumerate(children):
            if index:
                parts.append(text(", "))
            elements.append(self.box(child, level, budget))
            parts.append(elements[-1])
        return row(parts, node, tuple(elements))

    # -- vectors ------------------------------------------------------------

    def _vector(self, node: Node, level: int, budget: int | None) -> Box:
        if _is_matrix(node) and self.built_up(node, level, budget):
            return self._matrix(node, level, budget)
        # One row of elements, so the brackets need no padding, but they do
        # have to be built up as soon as an element is more than a row tall.
        return fenced(
            self._joined(node.children, level, budget),
            glyphs.BRACKETS,
            threshold=1,
            node=node,
        )

    def _matrix(self, node: Node, level: int, budget: int | None) -> Box:
        cells = [
            [self.box(cell, level, budget) for cell in line.children]
            for line in node.children
        ]
        widths = [
            max(line[column].width for line in cells) for column in range(len(cells[0]))
        ]
        width = sum(widths) + 2 * (len(widths) - 1)
        lines = []
        for source, line in zip(node.children, cells, strict=True):
            parts = []
            offset = 0
            for column, cell in enumerate(line):
                dx = offset + centered(widths[column], cell.width)
                parts.append(Placed(cell, dx, 0))
                offset += widths[column] + 2
            lines.append(
                Box(
                    width,
                    max(cell.above for cell in line),
                    max(cell.below for cell in line),
                    source,
                    "",
                    tuple(parts),
                    tuple(line),
                )
            )
        # One blank row between rows, and the baseline in the middle of the
        # block so that the fences sit symmetrically about it.
        height = sum(line.height for line in lines) + len(lines) - 1
        baseline = height // 2
        parts = []
        top = 0
        for line in lines:
            parts.append(Placed(line, 0, top + line.above - baseline))
            top += line.height + 1
        grid = Box(
            width, baseline, height - 1 - baseline, None, "", tuple(parts), tuple(lines)
        )
        return fenced(grid, glyphs.BRACKETS, pad=1, threshold=1, node=node)

    # -- declarations -------------------------------------------------------

    def _assignment(self, node: Node, level: int, budget: int | None) -> Box:
        operator = str(node.value)
        left = self.box(node.children[0], level, budget)
        if len(node.children) == 1:
            return row([left, text(self._spaced(operator).rstrip())], node, (left,))
        right = self.box(node.children[1], level, budget)
        return row([left, text(self._spaced(operator)), right], node, (left, right))

    def _definition(self, node: Node, level: int, budget: int | None) -> Box:
        # `F(x)` selects as one operand, name included: the head is drawn as a
        # unit, so the parameter list carries it and offers the parameters.
        parameters = node.children[0]
        listed = fenced(self._joined(parameters.children, level, budget), glyphs.PARENS)
        head = row([text(str(node.value)), listed], parameters, (listed,))
        if len(node.children) == 1:
            spelling = ":=" if self.options.compressed else " :="
            return row([head, text(spelling)], node, (head,))
        body = self.box(node.children[1], level, budget)
        return row([head, text(self._spaced(":=")), body], node, (head, body))

    def _domain(self, node: Node, level: int, budget: int | None) -> Box:
        name = self.box(node.children[0], level, budget)
        domain = text(str(node.value))
        if len(node.children) == 1:
            return row([name, text(glyphs.DOMAIN), domain], node, (name,))
        # The domain and its interval select together, as one operand drawn in
        # two pieces, so the interval's own box is widened to hold both.
        interval = self.box(node.children[1], level, budget)
        span = row(
            [domain, text(" "), interval], node.children[1], interval.operands
        )
        return row([name, text(glyphs.DOMAIN), span], node, (name, span))

    def _interval(self, node: Node, level: int, budget: int | None) -> Box:
        opening, closing = str(node.value)
        low = self.box(node.children[0], level, budget)
        high = self.box(node.children[1], level, budget)
        return row(
            [text(opening), low, text(", "), high, text(closing)], node, (low, high)
        )


def _binop_value(node: Node) -> str | None:
    return str(node.value) if node.kind is Kind.BINOP else None


def _is_quotient(node: Node) -> bool:
    return node.kind is Kind.BINOP and node.value == "/"


def _is_negation(node: Node) -> bool:
    return node.kind is Kind.UNOP and node.value == "-"


def _is_matrix(node: Node) -> bool:
    """A vector of vectors of equal, non-zero length."""
    lines = node.children
    if not lines or not all(line.kind is Kind.VECTOR for line in lines):
        return False
    return len({len(line.children) for line in lines}) == 1 and bool(lines[0].children)

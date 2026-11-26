"""Layout: turning a `Node` into a `Box`.

A single functional pass, driven by the script level and the remaining height budget.

The script level is the nesting depth inside superscripts and subscripts. Fractions and
matrices are built up at level 0 and linear below it, and `^` stops raising at level 2,
so an exponent tower flattens to `y^z^w` instead of climbing off the top of the pane.

When the height budget is exceeded, the outermost constructs are degraded first. Each
node's unconstrained box is memoized, so checking how tall a built-up form would be
costs one layout per node.

Each box lists its selectable operands, following Derive's rule that you select what you
see: a chain of sums offers its terms rather than nested pairs, a call offers its
arguments but not its name, and special forms choose their own.
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
from rederive.model.expr import VARIADIC, Kind, Node, is_chain
from rederive.syntax.names import EXIT

#: Word operators keep their spaces in both formats, since `xANDy` would lex as one
#: name. `NOT` is handled separately as a prefix, and `SUB` never prints as a word (a
#: subscript too deep to lower is written `a↓b`).
_WORD_OPERATORS = {
    Kind.AND: "AND",
    Kind.OR: "OR",
    Kind.XOR: "XOR",
    Kind.IMP: "IMP",
    Kind.IFF: "IFF",
}

#: Beyond this level a script is written flat, `^` as `a^b` and `SUB` as `a↓b`.
LINEAR_SCRIPTS = 2

#: The tallest box whose parentheses are still single characters, `fenced`'s default.
_SHORT_FENCE = 2

#: `after` for an operand with nothing drawn to its right. Lower than every precedence on
#: the ladder.
ALONE = -1

#: Default options, shared so the common call doesn't construct new ones.
_DEFAULTS = DisplayOptions()


def render(node: Node, options: DisplayOptions = _DEFAULTS) -> Layout:
    """Typeset `node`. The width is unlimited; the pane clips it.

    A program is drawn as an indented block (see `Engine.block`).
    """
    engine = Engine(options)
    block = engine.block(node, options.height, whole=True)
    return paint(block if block is not None else engine.box(node, 0, options.height))


class Engine:
    """Layout state for one render.

    Passed to the functions in `forms.py`, so they don't need to import this module.
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
        """`node` laid out with no height limit, memoized."""
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

        `after` is the precedence of the operator drawn to the right of the operand, or
        `ALONE` if there is none. It matters for forms ending in a body, which a single
        precedence can't describe: `Σ k` is fenced before `+ 1` but not after `2·`.
        Commas, closing brackets and an integral's `dx` are terminators, so an operand
        before them is `ALONE`.
        """
        box = self.box(node, level, budget)
        if forms.needs_fences(node, required, after):
            return self.fence(box)
        return box

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
        """`script` immediately right of `base`, raised one row above its baseline.

        The script's columns stay blank on the baseline row, so `x^2 + 2x` renders as
        `x  + 2·x` with two spaces.
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
        """Whether a division or matrix is drawn built-up here."""
        if level > 0:
            return False
        return budget is None or self.free(node, level).height <= budget

    # -- program blocks -----------------------------------------------------

    def block(self, node: Node, budget: int | None, whole: bool = False) -> Box | None:
        """`node` drawn as an indented block of lines, or None if it isn't one.

        As in Derive, a program is drawn one statement to a line: `PROG` as a line
        `Prog` with its statements indented under it, `LOOP` likewise as `Loop`, and an
        `IF` as a line `If test` with its clauses indented under the test (`IF(test,
        exit)` stays on its line, `If test exit`). A definition whose body is a block,
        and a `whole` expression assigning one, have the head on a line of its own and
        the block indented under it. A `PROG`, `LOOP` or `IF` anywhere else is drawn as
        the call it is.
        """
        if node.kind is Kind.FUNDEF and len(node.children) == 2:
            body = self.block(node.children[1], budget)
            if body is None:
                return None
            head = self._head(node, budget)
            lines = [(0, row([head, text(" :=")])), (2, body)]
            return _stacked(lines, node, (head, body))
        if whole and node.kind is Kind.ASSIGN and len(node.children) == 2:
            value = self.block(node.children[1], budget)
            if value is None:
                return None
            target = self.box(node.children[0], 0, budget)
            operator = text(f" {node.value}")
            return _stacked(
                [(0, row([target, operator])), (2, value)], node, (target, value)
            )
        if node.kind is not Kind.CALL:
            return None
        name, *arguments = node.children
        keyword = _BLOCKS.get(str(name.value))
        if keyword is None or not arguments:
            return None
        if keyword == "If":
            return self._if_block(node, arguments, budget)
        statements = [self.statement(argument, budget) for argument in arguments]
        lines = [(0, text(keyword)), *((2, line) for line in statements)]
        return _stacked(lines, node, tuple(statements))

    def statement(self, node: Node, budget: int | None) -> Box:
        """`node` as a line of a block: a block itself, `RETURN u`, or as drawn
        anywhere."""
        block = self.block(node, budget)
        if block is not None:
            return block
        if node.kind is Kind.CALL and len(node.children) == 2:
            name, operand = node.children
            if name.value == "RETURN":
                inner = self.box(operand, 0, budget)
                return row([text("RETURN "), inner], node, (inner,))
        return self.box(node, 0, budget)

    def _if_block(
        self, node: Node, arguments: list[Node], budget: int | None
    ) -> Box | None:
        test_node, *clauses = arguments
        if not clauses or len(clauses) > 3:
            return None
        test = self.box(test_node, 0, budget)
        if len(clauses) == 1 and clauses[0].kind is Kind.NAME and (
            clauses[0].value == EXIT
        ):
            leave = self.box(clauses[0], 0, budget)
            return row([text("If "), test, text(" "), leave], node, (test, leave))
        lines = [self.statement(clause, budget) for clause in clauses]
        header = row([text("If "), test])
        return _stacked(
            [(0, header), *((3, line) for line in lines)], node, (test, *lines)
        )

    def _head(self, node: Node, budget: int | None) -> Box:
        """The head `F(x, y)` of a definition, selectable as one operand, name
        included, offering the parameters inside it."""
        parameters = node.children[0]
        listed = self._joined(parameters.children, 0, budget)
        if parameters.value == VARIADIC:
            return row([text(f"{node.value} "), listed], parameters, (listed,))
        listed = fenced(listed, glyphs.PARENS)
        return row([text(str(node.value)), listed], parameters, (listed,))

    # -- dispatch -----------------------------------------------------------

    def _build(self, node: Node, level: int, budget: int | None) -> Box:
        match node.kind:
            case Kind.NUMBER:
                options = self.options
                shown = glyphs.numeral(
                    str(node.value), options.output_base, options.notation_digits
                )
                return text(shown, node)
            case Kind.NAME:
                return text(glyphs.name(str(node.value)), node)
            case Kind.STRING:
                return text(f'"{node.value}"', node)
            case Kind.LABEL:
                return text(f"#{node.value}", node)
            case Kind.UNKNOWN:
                return text("?", node)
            case Kind.SUM | Kind.PRODUCT:
                return self._run(node, level, budget)
            case Kind.BINOP:
                return self._binop(node, level, budget)
            case Kind.UNOP:
                return self._sign(node, level, budget)
            case Kind.POSTOP:
                inner = self.operand(
                    node.children[0], level, budget, forms.POSTFIX, forms.POSTFIX
                )
                operator = str(node.value)
                if operator == "DEG":
                    operator = glyphs.DEGREE
                return row([inner, text(operator)], node, (inner,))
            case Kind.SUB:
                return self._subscript(node, level, budget)
            case Kind.ABS:
                return forms.absolute_value(self, node, node.children, level, budget)
            case Kind.CALL | Kind.APPLY:
                return self._application(node, node, level, budget)
            case Kind.FUNCPOW:
                # The application and the exponent are two operands, as in Derive, which
                # represents this as a power of an application. Here it's a single node,
                # so both regions refer to it.
                inner = self._application(node, node, level, budget)
                exponent = self.box(node.children[1], level + 1, budget)
                return self.raised(inner, exponent, node, (inner, exponent))
            case Kind.VECTOR:
                return self._vector(node, level, budget)
            case Kind.SET:
                return fenced(
                    self._joined(node.children, level, budget),
                    glyphs.BRACES,
                    threshold=1,
                    node=node,
                )
            case Kind.ELLIPSIS:
                return text(str(node.value), node)
            case Kind.SETOP:
                return self._set_operation(node, level, budget)
            case Kind.REL:
                return self._relation(node, level, budget)
            case Kind.NOT:
                inner = self.operand(node.children[0], level, budget, forms.NOT)
                return row([text("NOT "), inner], node, (inner,))
            case Kind.AND | Kind.OR | Kind.XOR | Kind.IMP | Kind.IFF:
                return self._logical(node, level, budget)
            case Kind.INFIX:
                return self._infix(node, level, budget)
            case Kind.ASSIGN:
                return self._assignment(node, level, budget)
            case Kind.FUNDEF:
                return self._definition(node, level, budget)
            case Kind.PARAMS:
                # Definitions draw their own parameter list, so this only handles a
                # standalone `PARAMS`, keeping the dispatch total over all kinds.
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
            case Kind.QUOTE:
                inner = self.operand(node.children[0], level, budget, forms.ATOM)
                return row([text("'"), inner], node, (inner,))
        raise AssertionError(f"unrenderable node kind: {node.kind}")

    # -- spacing ------------------------------------------------------------

    def _spaced(self, symbol: str) -> str:
        """A symbol operator, spaced unless the format is Compressed."""
        return symbol if self.options.compressed else f" {symbol} "

    # -- operators ----------------------------------------------------------

    def _binop(self, node: Node, level: int, budget: int | None) -> Box:
        """`/`, `^` and the dot product, the only true binary operators."""
        operator = str(node.value)
        if operator == "/":
            return self._division(node, level, budget)
        if operator == "^":
            return self._power(node, level, budget)
        left, right = node.children
        first = self.operand(left, level, budget, forms.MUL, forms.MUL)
        second = self.operand(right, level, budget, forms.MUL)
        return row([first, text(glyphs.DOT_PRODUCT), second], node, (first, second))

    def _run(self, node: Node, level: int, budget: int | None) -> Box:
        """A sum or product, drawn as a flat run of terms.

        Each gap has its own operator, so `a + b - c` offers three operands rather than
        `a + b` and `c`.
        """
        own = forms.precedence(node)
        gaps = [self._gap(node, index) for index in range(len(node.children) - 1)]
        operands = []
        parts: list[Box] = []
        for index, child in enumerate(node.children):
            if index:
                parts.append(text(gaps[index - 1]))
            after = own if index < len(gaps) else ALONE
            box = self.operand(child, level, budget, own, after)
            if forms.fenced_in_run(node, child):
                box = self.fence(box)
            operands.append(box)
            parts.append(box)
        return row(parts, node, tuple(operands))

    def _gap(self, node: Node, index: int) -> str:
        """How the operator before term `index + 1` is drawn.

        All gaps in a product look the same, juxtaposition included: Derive writes `a b`
        as `a·b`.
        """
        if node.kind is Kind.PRODUCT:
            return glyphs.TIMES[self.options.times]
        return self._spaced(str(node.value)[index])

    def _sign(
        self, node: Node, level: int, budget: int | None, detached: bool = False
    ) -> Box:
        """`-u`, `+u` or `±u`.

        The sign touches a single-token operand and is separated by a space from anything
        else: `-x` and `3 - -2`, but `- SIN(x)`, `- √x`, `- [1, 2]` and `- a/b`.
        """
        operand = self.operand(node.children[0], level, budget, forms.MUL)
        sign = glyphs.operator(str(node.value))
        if detached or not node.children[0].is_atom:
            sign += " "
        return row([text(sign), operand], node, (operand,))

    def _division(self, node: Node, level: int, budget: int | None) -> Box:
        numerator, denominator = node.children
        if self.built_up(node, level, budget):
            # The bar groups the operands, so they never need fences.
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
        """An operand of a linear `/`. A built-up fraction needs no fences here."""
        if _is_quotient(node) and self.built_up(node, level, budget):
            return self.box(node, level, budget)
        return self.operand(node, level, budget, required, after)

    def _power(self, node: Node, level: int, budget: int | None) -> Box:
        base, exponent = node.children
        under = self.operand(base, level, budget, forms.POSTFIX, forms.POW)
        if level >= LINEAR_SCRIPTS:
            script = self.operand(exponent, level, budget, forms.NEG)
            return row([under, text("^"), script], node, (under, script))
        # A raised exponent never needs fences.
        script = self.box(exponent, level + 1, budget)
        return self.raised(under, script, node, (under, script))

    def _subscript(self, node: Node, level: int, budget: int | None) -> Box:
        base, script = node.children
        if level >= LINEAR_SCRIPTS:
            # Flat `a↓b`: the arrow replaces lowering, just as `a^b` replaces raising.
            flat = self.operand(base, level, budget, forms.POSTFIX, forms.POSTFIX)
            deeper = self.operand(script, level, budget, forms.SCRIPT)
            if script.kind is Kind.SUB:
                # `a↓b↓c` reads as the chain `a SUB b SUB c`.
                deeper = self.fence(deeper)
            return row(
                [flat, text(glyphs.LOWERED), deeper], node, (flat, deeper)
            )
        # A lowered script is written flat, whatever it holds: `x SUB (y^z)` lowers
        # `y^z`, and `x SUB (y SUB z)` lowers `y↓z`.
        lowered = self.box(script, LINEAR_SCRIPTS, budget)
        under = self.operand(base, level, budget, forms.POSTFIX, forms.POSTFIX)
        if self._bare_call(base, level, budget):
            # Without its parentheses the call would seem to subscript its argument.
            under = self.fence(under)
        parts = [Placed(under, 0, 0)]
        dx = under.width
        if base.kind is Kind.SUB:
            # Nested subscripts merge: `v SUB 1 SUB 2` draws `v` with subscript `1,2`,
            # on the row the first subscript already occupies.
            dy = under.below + lowered.above
            parts.append(Placed(text(","), dx, dy))
            dx += 1
        else:
            # The script goes on a row of its own below the base, however tall the base
            # is: a matrix's subscript sits under its bottom corner.
            dy = under.below + 1 + lowered.above
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
        """`a = b`. Chains nest to the left and draw flat.

        The left operand may itself be an unfenced relation, so `a=b<c` renders as
        `a = b < c`; a relation on the right is fenced, as in `a=(b<c)`.
        """
        left, right = node.children
        first = self.operand(left, level, budget, forms.RELATION, forms.RELATION)
        second = self.operand(right, level, budget, forms.ADD)
        operator = glyphs.operator(str(node.value))
        return row([first, text(self._spaced(operator)), second], node, (first, second))

    def _set_operation(self, node: Node, level: int, budget: int | None) -> Box:
        """A union, intersection or difference, left-associative like the grammar."""
        own = forms.precedence(node)
        left, right = node.children
        # A conjunction inside a disjunction is fenced although `AND` binds more
        # tightly, as the original draws a disjunction of solutions.
        fenced = forms.AND + 1 if node.kind is Kind.OR else 0
        first_needs = max(own, fenced if left.kind is Kind.AND else 0)
        second_needs = max(own + 1, fenced if right.kind is Kind.AND else 0)
        first = self.operand(left, level, budget, first_needs, own)
        second = self.operand(right, level, budget, second_needs)
        operator = str(node.value)
        symbol = operator if operator == "\\" else f" {operator} "
        return row([first, text(symbol), second], node, (first, second))

    def _infix(self, node: Node, level: int, budget: int | None) -> Box:
        """A word operator of Derive 6. A subscript link (`A ROW i`, `A↓↓j`) takes an
        index as a lowered subscript does; the others are left-associative."""
        left, right = node.children
        word = str(node.value)
        symbol = f" {glyphs.INFIX.get(word, word)} "
        if is_chain(node):
            first = self.operand(left, level, budget, forms.POSTFIX, forms.POSTFIX)
            second = self.operand(right, level, budget, forms.SCRIPT)
            if word in glyphs.INFIX:
                symbol = glyphs.INFIX[word]
            elif word.startswith("SUB"):
                symbol = glyphs.LOWERED * len(word.split())
        else:
            own = forms.precedence(node)
            first = self.operand(left, level, budget, own, own)
            second = self.operand(right, level, budget, own + 1)
        return row([first, text(symbol), second], node, (first, second))

    def _logical(self, node: Node, level: int, budget: int | None) -> Box:
        word = _WORD_OPERATORS[node.kind]
        left, right = node.children
        first_needs, second_needs = forms.logical_operands(node)
        first = self.operand(left, level, budget, first_needs, forms.precedence(node))
        second = self.operand(right, level, budget, second_needs)
        return row([first, text(f" {word} "), second], node, (first, second))

    # -- applications -------------------------------------------------------

    def _application(
        self, node: Node, owner: Node | None, level: int, budget: int | None
    ) -> Box:
        """A call, a bare application, or the application inside a `FUNCPOW`.

        `owner` is the node the box belongs to.
        """
        form = forms.special(node)
        name, arguments = forms.application(node)
        if form is not None:
            return form(self, owner, arguments, level, budget)
        # Arguments never need fences inside the call's parentheses. The name has no
        # node, so it is never selectable.
        joined = self._joined(arguments, level, budget)
        if self._bare_call(node, level, budget):
            return row([text(f"{name.value} "), joined], owner, (joined,))
        listed = fenced(joined, glyphs.PARENS)
        return row([text(str(name.value)), listed], owner, (listed,))

    def _bare_call(self, node: Node, level: int, budget: int | None) -> bool:
        """Whether `node` is a call drawn without parentheses, as `DET [...]`.

        Following the original, they are dropped when the only argument is a vector or
        matrix tall enough that the parentheses would be built up too: the brackets
        delimit the argument on their own, and a space follows the name instead. A
        two-row vector keeps its one-cell parentheses, as in `DIV([1, 2·y, 3·z^2])`.
        """
        if node.kind not in forms.APPLICATIONS or forms.special(node) is not None:
            return False
        _, arguments = forms.application(node)
        if len(arguments) != 1 or arguments[0].kind is not Kind.VECTOR:
            return False
        return self.box(arguments[0], level, budget).height > _SHORT_FENCE

    def _joined(
        self,
        children: tuple[Node, ...],
        level: int,
        budget: int | None,
        node: Node | None = None,
    ) -> Box:
        """A comma-separated list, spaced the same way in both formats.

        Elements never need fences: the enclosing brackets or parentheses group the list,
        and a comma terminates a form's body.
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
        if forms.is_matrix(node) and self.built_up(node, level, budget):
            return self._matrix(node, level, budget)
        # A single row needs no bracket padding, but the brackets are built up as soon as
        # an element is taller than one row.
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
                # A cell's middle column goes on the column's middle column, so an
                # even-width cell in an odd-width column leans left: `27` over `125`.
                dx = offset + widths[column] // 2 - cell.width // 2
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
        # One blank row between rows, with the baseline in the middle so the fences sit
        # symmetrically.
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
        head = self._head(node, budget)
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
        # The domain and its interval are selected together, so the interval's box is
        # widened to include the domain.
        interval = self.box(node.children[1], level, budget)
        span = row(
            [domain, text(" "), interval], node.children[1], interval.operands
        )
        return row([name, text(glyphs.DOMAIN), span], node, (name, span))

    def _interval(self, node: Node, level: int, budget: int | None) -> Box:
        """`[0, ∞)`, always drawn flat.

        Derive writes interval bounds linearly, in author notation: `x :ε Real (1/2, 5)`
        and `y :ε Real (2^10, 1/3)` are each one row. The bounds are laid out at the level
        where scripts go flat, which also makes fractions linear.
        """
        opening, closing = str(node.value)
        level = max(level, LINEAR_SCRIPTS)
        low = self.box(node.children[0], level, budget)
        high = self.box(node.children[1], level, budget)
        return row(
            [text(opening), low, text(", "), high, text(closing)], node, (low, high)
        )


#: The program constructs drawn as blocks, by name, with the keyword that heads them.
_BLOCKS = {"PROG": "Prog", "LOOP": "Loop", "IF": "If"}


def _stacked(
    lines: list[tuple[int, Box]], node: Node, operands: tuple[Box, ...]
) -> Box:
    """Boxes one under the other, each at its indentation, the baseline the first's."""
    first = lines[0][1]
    parts = []
    bottom = -1 - first.above
    for indent, box in lines:
        dy = bottom + 1 + box.above
        parts.append(Placed(box, indent, dy))
        bottom = dy + box.below
    width = max(indent + box.width for indent, box in lines)
    return Box(width, first.above, bottom, node, "", tuple(parts), operands)


def _is_quotient(node: Node) -> bool:
    return node.kind is Kind.BINOP and node.value == "/"

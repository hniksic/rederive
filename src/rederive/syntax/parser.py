"""Recursive descent with precedence climbing.

Some productions aren't ordinary precedence levels:

`operand` is what a bare application consumes: an optionally signed `SUB` chain over
applications. It binds looser than `!` and tighter than `^`, so `SIN x!` is `SIN(x)!`
and `SIN x^2` is `SIN(x)^2`.

`:=` is handled in `primary`: a name followed by `:=` becomes an assignment before any
surrounding operator is considered, so `"inapplicable"=a_:=F(...)` is a relation whose
right operand is an assignment.

`sum` and `product` build one node per run rather than nested pairs, as the original
does. This lets Transfer Save keep the parentheses of `(a-b)+c` and `(a*b)*c`, and makes
the selection highlight see three operands in `a-b+c`. A run stores one operator per
gap, so in `a-b+c` the operand is `b` and `-` is among the operators; the highlight
covers `b`, not `- b`. `/` and the dot product are binary and close a run (`(a*b)/c`
saves as `a*b/c`), so `a*b/c*d` is a product of `a·b/c` and `d`.
"""

from __future__ import annotations

from dataclasses import dataclass

from rederive.model.expr import Kind, Node
from rederive.syntax import names
from rederive.syntax.errors import DeriveSyntaxError
from rederive.syntax.lexer import Lexer
from rederive.syntax.state import (
    Declaration,
    DomainDeclaration,
    FunctionDeclaration,
    NameBinding,
    SettingDeclaration,
    VariableDeclaration,
    fold,
)
from rederive.syntax.tokens import Token, TokenKind

_FACTOR_KINDS = (
    TokenKind.NUMBER,
    TokenKind.NAME,
    TokenKind.STRING,
    TokenKind.LABEL,
)
_SIGNS = ("-", "+", "+-")
_POSTFIX = ("!", "%", "`")
_RELATIONS = ("=", "/=", "<", "<=", ">", ">=")
_CLOSERS = (",", "]", ")")
#: Setting values by folded spelling. `NORMAL` and `EXPAND` are also built-in functions,
#: so a function name with nothing to apply to is looked up here first.
_SETTING_VALUES = {fold(value): value for value in names.SETTING_VALUES}
_LOGICAL_KINDS = {
    "OR": Kind.OR,
    "XOR": Kind.XOR,
    "IMP": Kind.IMP,
    "AND": Kind.AND,
}


@dataclass(frozen=True)
class _Parsed:
    """A node plus the extent of the text that produced it.

    `node.start`/`node.end` is the highlight span, which excludes enclosing parentheses;
    `lo`/`hi` include them, so an enclosing node's span covers its operands' parentheses.
    """

    node: Node
    lo: int
    hi: int

    @classmethod
    def of(cls, node: Node) -> _Parsed:
        return cls(node, node.start, node.end)


class Parser:
    """Parser for a single expression."""

    def __init__(self, lexer: Lexer) -> None:
        self.lexer = lexer
        self.pos = lexer.start
        self.bars = 0
        self.declarations: list[Declaration] = []

    # -- token handling -----------------------------------------------------

    def peek(self) -> Token:
        return self.lexer.token_at(self.pos)

    def after(self, token: Token) -> Token:
        return self.lexer.token_at(token.end)

    def advance(self) -> Token:
        token = self.peek()
        self.pos = token.end
        return token

    def error(self, offset: int, expected: str) -> DeriveSyntaxError:
        return DeriveSyntaxError(offset, expected, self.lexer.source)

    def expect(self, text: str, expected: str) -> Token:
        token = self.peek()
        if not _is_op(token, text):
            raise self.error(token.start, expected)
        return self.advance()

    def starts_factor(self, token: Token) -> bool:
        """Whether `token` can begin a factor, so juxtaposition applies."""
        if token.kind is TokenKind.EOF:
            return False
        if token.kind in _FACTOR_KINDS:
            return True
        if token.kind is not TokenKind.OP:
            return False
        if token.value == "|":
            # Where an operator could continue the expression, `|` closes the innermost
            # open bar rather than opening a new one.
            return self.bars == 0
        return token.value in ("(", "[", "?")

    def starts_operand(self, token: Token) -> bool:
        return self.starts_factor(token) or _is_op(token, *_SIGNS)

    # -- the top of the grammar ---------------------------------------------

    def parse(self) -> Node:
        parsed = self.assignment()
        token = self.peek()
        if _is_op(token, "=") and self.after(token).kind is TokenKind.EOF:
            # The author line's trailing `=`: show the value of this.
            self.advance()
            parsed = _Parsed.of(
                Node(Kind.SHOWVALUE, parsed.lo, token.end, (parsed.node,))
            )
        token = self.peek()
        if token.kind is not TokenKind.EOF:
            raise self.error(token.start, "end of the expression")
        return parsed.node

    def assignment(self) -> _Parsed:
        return self.implication()

    def implication(self) -> _Parsed:
        """`IMP`, the only left-associative logical operator."""
        left = self.exclusive_or()
        while _is_op(self.peek(), "IMP"):
            self.advance()
            right = self.exclusive_or()
            left = _combine(Kind.IMP, left, right)
        return left

    def exclusive_or(self) -> _Parsed:
        return self._logical("XOR", self.disjunction)

    def disjunction(self) -> _Parsed:
        return self._logical("OR", self.conjunction)

    def conjunction(self) -> _Parsed:
        return self._logical("AND", self.negation)

    def _logical(self, word: str, tighter) -> _Parsed:
        """`AND`, `OR` and `XOR`, which are right-associative.

        `a AND b AND c` is `a AND (b AND c)`, as in the original and its saved form.
        """
        left = tighter()
        if not _is_op(self.peek(), word):
            return left
        self.advance()
        return _combine(_LOGICAL_KINDS[word], left, self._logical(word, tighter))

    def negation(self) -> _Parsed:
        token = self.peek()
        if _is_op(token, "NOT"):
            self.advance()
            operand = self.negation()
            return _Parsed(
                Node(Kind.NOT, token.start, operand.hi, (operand.node,)),
                token.start,
                operand.hi,
            )
        return self.relation()

    def relation(self) -> _Parsed:
        """Binary relations, left-associative: `a=b<c` is `(a=b)<c`."""
        left = self.sum()
        while True:
            token = self.peek()
            if not _is_op(token, *_RELATIONS):
                break
            if token.value == "=" and self.after(token).kind is TokenKind.EOF:
                break  # the author line's trailing `=`
            self.advance()
            right = self.sum()
            left = _combine(Kind.REL, left, right, token.value, token.surface)
        return left

    def sum(self) -> _Parsed:
        """A run of `+` and `-` over signed terms, as one node."""
        terms = [self.signed()]
        gaps: list[tuple[str, str]] = []
        while _is_op(self.peek(), "+", "-"):
            token = self.advance()
            gaps.append((token.value, token.surface or token.value))
            terms.append(self.signed())
        return _run(Kind.SUM, terms, gaps)

    def signed(self) -> _Parsed:
        """A sign applying to a whole product: `-x/y*z` is `-(x/y·z)`.

        From tightest: `^`, then `*` `/` `.` and juxtaposition, then unary sign, then
        binary `+` and `-`.
        """
        return self._sign(self.signed) or self.product()

    def product(self) -> _Parsed:
        """`*`, `/`, the dot product, and juxtaposition, all left-associative.

        A run of `*` and juxtaposition is one node; `/` and the dot product are binary
        and take the run as their left operand. So `a*b/c*d` is a product of `a·b/c` and
        `d`, matching how the original's cursor selects it.
        """
        factors = [self.power()]
        gaps: list[tuple[str, str]] = []
        while True:
            token = self.peek()
            if _is_op(token, "*"):
                self.advance()
                # A run of consecutive `*` behaves as a single `*`.
                while _is_op(self.peek(), "*"):
                    self.advance()
                gaps.append(("*", token.surface or "*"))
                factors.append(self.factor())
            elif self.starts_factor(token):
                gaps.append(("*", " "))
                factors.append(self.factor())
            elif _is_op(token, "/", "."):
                self.advance()
                left = _run(Kind.PRODUCT, factors, gaps)
                right = self.factor()
                factors = [
                    _combine(Kind.BINOP, left, right, token.value, token.surface)
                ]
                gaps = []
            else:
                break
        return _run(Kind.PRODUCT, factors, gaps)

    def factor(self) -> _Parsed:
        """A factor, optionally signed.

        The sign applies only to the factor, so `a*-b*c` is `(a·(-b))·c`.
        """
        return self._sign(self.factor) or self.power()

    def _sign(self, inner) -> _Parsed | None:
        """`-u`, `+u` or `±u`, with `inner` parsing the `u`; None if there's no sign."""
        token = self.peek()
        if not _is_op(token, *_SIGNS):
            return None
        self.advance()
        operand = inner()
        node = Node(
            Kind.UNOP,
            token.start,
            operand.hi,
            (operand.node,),
            token.value,
            token.surface,
        )
        return _Parsed(node, token.start, operand.hi)

    def power(self) -> _Parsed:
        base = self.subscripted()
        token = self.peek()
        if _is_op(token, "^"):
            self.advance()
            # Right-associative. The exponent may have its own sign, which applies only
            # to the exponent: `a^-b*c` is `(a^(-b))·c`.
            return _combine(Kind.BINOP, base, self.factor(), "^", token.surface)
        return base

    def subscripted(self) -> _Parsed:
        """`u SUB i`, where the index is a single application.

        A postfix operator after a subscript applies to the element, not the index:
        `ODE_APPR.MTH` writes `v_ SUB 1!` for the factorial of the first element, which
        is the reading that reproduces the manual's answer.
        """
        left = self.postfix()
        while _is_op(self.peek(), "SUB"):
            token = self.advance()
            right = self.application()
            left = _combine(Kind.SUB, left, right, None, token.surface)
            left = self.postfixed(left)
        return left

    def postfix(self) -> _Parsed:
        return self.postfixed(self.application())

    def postfixed(self, left: _Parsed) -> _Parsed:
        """`left` with any following postfix operators applied."""
        while _is_op(self.peek(), *_POSTFIX):
            token = self.advance()
            left = _Parsed(
                Node(Kind.POSTOP, left.lo, token.end, (left.node,), token.value),
                left.lo,
                token.end,
            )
        return left

    # -- application --------------------------------------------------------

    def application(self) -> _Parsed:
        token = self.peek()
        if token.kind is TokenKind.NAME and not token.is_defhead:
            following = self.after(token)
            if token.is_function:
                if _is_op(following, "("):
                    return self.call(token)
                if _is_op(following, "^"):
                    return self.function_power(token)
                keyword = self.setting_keyword(token, following)
                if keyword is not None:
                    return keyword
                return self.bare_application(token)
            if _is_op(following, "("):
                parsed = self.argument_list_on_a_variable(token, following)
                if parsed is not None:
                    return parsed
        return self.primary()

    def call(self, name: Token) -> _Parsed:
        self.advance()
        self.expect("(", "'('")
        arguments = self.arguments()
        close = self.expect(")", "')'")
        children = (_name_node(name), *(argument.node for argument in arguments))
        node = Node(Kind.CALL, name.start, close.end, children)
        return _Parsed(node, name.start, close.end)

    def arguments(self) -> list[_Parsed]:
        """A non-empty comma-separated argument list.

        Arity is checked by the engine: many built-ins are variadic, and calling a user
        function with fewer arguments than parameters is meaningful.
        """
        if _is_op(self.peek(), ")"):
            raise self.error(self.peek().start, "an argument")
        arguments = [self.assignment()]
        while _is_op(self.peek(), ","):
            self.advance()
            arguments.append(self.assignment())
        return arguments

    def argument_list_on_a_variable(self, name: Token, opening: Token) -> _Parsed | None:
        """`NAME(...)` where `NAME` is not a known function.

        With one argument this is a product, so return None and let juxtaposition handle
        it. Anything else is a syntax error, raised after consuming the whole list.
        """
        count = self.lexer.count_arguments(opening.start)
        if count is None or count == 1:
            return None
        self.advance()
        self.expect("(", "'('")
        if not _is_op(self.peek(), ")"):
            self.arguments()
        self.expect(")", "')'")
        raise self.error(self.pos, "a function name")

    def setting_keyword(self, token: Token, following: Token) -> _Parsed | None:
        """A setting value spelled like a function, where the function can't stand.

        `Normal` and `Expand` are both setting values and built-in functions. With
        something to apply to they are the function; otherwise they are the value. The
        original likewise accepts `DisplayFormat := Normal` and a bare `Normal`, but
        rejects a bare `SIN`.
        """
        if self.starts_operand(following):
            return None
        written = token.surface or token.value
        spelled = _SETTING_VALUES.get(fold(written))
        if spelled is None:
            return None
        self.advance()
        surface = None if written == spelled else written
        return _Parsed.of(Node(Kind.NAME, token.start, token.end, (), spelled, surface))

    def bare_application(self, name: Token) -> _Parsed:
        """`SIN x`: exactly one operand, with `^` and `!` binding outside it."""
        self.advance()
        token = self.peek()
        if not self.starts_operand(token):
            raise self.error(token.start, "an operand")
        operand = self.operand()
        children = (_name_node(name), operand.node)
        return _Parsed(
            Node(Kind.APPLY, name.start, operand.hi, children), name.start, operand.hi
        )

    def function_power(self, name: Token) -> _Parsed:
        """`SQRT^3 25`: apply, then raise to the power.

        So `SIN^-1 x` is `1/SIN(x)`, not `ASIN(x)`.
        """
        self.advance()
        self.expect("^", "'^'")
        exponent = self.factor()
        token = self.peek()
        if not self.starts_operand(token):
            raise self.error(token.start, "an operand")
        operand = self.operand()
        children = (_name_node(name), exponent.node, operand.node)
        return _Parsed(
            Node(Kind.FUNCPOW, name.start, operand.hi, children), name.start, operand.hi
        )

    def operand(self) -> _Parsed:
        signed = self._sign(self.operand)
        if signed is not None:
            return signed
        left = self.application()
        while _is_op(self.peek(), "SUB"):
            sub = self.advance()
            right = self.application()
            left = _combine(Kind.SUB, left, right, None, sub.surface)
        return left

    # -- primaries ----------------------------------------------------------

    def primary(self) -> _Parsed:
        token = self.peek()
        if token.kind is TokenKind.NUMBER:
            self.advance()
            surface = token.surface if token.surface != token.value else None
            node = Node(Kind.NUMBER, token.start, token.end, (), token.value, surface)
            return _Parsed.of(node)
        if token.kind is TokenKind.NAME:
            return self.name_primary(token)
        if token.kind is TokenKind.STRING:
            following = self.after(token)
            if _is_op(following, ":=", ":=="):
                return self.assignment_to(token, _string_node(token))
            self.advance()
            return _Parsed.of(_string_node(token))
        if token.kind is TokenKind.LABEL:
            self.advance()
            return _Parsed.of(Node(Kind.LABEL, token.start, token.end, (), token.value))
        if _is_op(token, "?"):
            self.advance()
            return _Parsed.of(Node(Kind.UNKNOWN, token.start, token.end, (), "?"))
        if _is_op(token, "("):
            return self.group(token)
        if _is_op(token, "["):
            return self.vector(token)
        if _is_op(token, "|"):
            return self.absolute_value(token)
        raise self.error(token.start, "an operand")

    def name_primary(self, token: Token) -> _Parsed:
        if token.is_defhead:
            return self.function_definition(token)
        following = self.after(token)
        if _is_op(following, ":=", ":=="):
            return self.assignment_to(token, _name_node(token))
        if _is_op(following, ":epsilon"):
            return self.domain_declaration(token)
        self.advance()
        return _Parsed.of(_name_node(token))

    def group(self, opening: Token) -> _Parsed:
        self.advance()
        inner = self.assignment()
        closing = self.expect(")", "')'")
        # The parentheses widen the extent but not the highlight span.
        return _Parsed(inner.node, opening.start, closing.end)

    def vector(self, opening: Token) -> _Parsed:
        """A vector literal; brackets are never used for grouping."""
        self.advance()
        elements: list[_Parsed] = []
        if not _is_op(self.peek(), "]"):
            elements.append(self.assignment())
            while _is_op(self.peek(), ","):
                self.advance()
                elements.append(self.assignment())
        closing = self.expect("]", "']'")
        node = Node(
            Kind.VECTOR,
            opening.start,
            closing.end,
            tuple(element.node for element in elements),
        )
        return _Parsed.of(node)

    def absolute_value(self, opening: Token) -> _Parsed:
        """`|u|` is `ABS(u)`, and bars nest."""
        if not self.starts_operand(self.after(opening)):
            raise self.error(opening.start, "an operand")
        self.advance()
        self.bars += 1
        inner = self.implication()
        closing = self.peek()
        if not _is_op(closing, "|"):
            raise self.error(closing.start, "'|'")
        self.advance()
        self.bars -= 1
        return _Parsed.of(Node(Kind.ABS, opening.start, closing.end, (inner.node,)))

    # -- declarations -------------------------------------------------------

    def assignment_to(self, name: Token, lhs: Node) -> _Parsed:
        """`name := expr`, with an empty right side allowed."""
        self.advance()
        operator = self.advance()
        right = None
        if not self._ends_assignment(self.peek()):
            right = self.assignment()
        children = (lhs,) if right is None else (lhs, right.node)
        end = operator.end if right is None else right.hi
        self._record_assignment(lhs, right)
        node = Node(Kind.ASSIGN, name.start, end, children, operator.value)
        return _Parsed(node, name.start, end)

    def function_definition(self, name: Token) -> _Parsed:
        """`F(x,y) := body`, or `F(x,y):=` for an arbitrary function.

        Parameters are registered as variables while the body is parsed, because in
        Character mode `F(mx,mf) := mx+mf` would otherwise lex `mx` as `m*x`. Afterwards,
        as in the original, a parameter that was not a known name stays one: it is
        reported as a variable declaration, so once the line is entered, a later `mx`
        reads whole. A name that was already known keeps what it was, so after `A(t):=`
        and `FF(a):=a^2`, `A(t)` is still a call, and a parameter named like a valued
        or declared variable leaves its value and domain alone.
        """
        self.advance()
        self.expect("(", "'('")
        if _is_op(self.peek(), ")"):
            raise self.error(self.peek().start, "a parameter")
        parameters: list[Node] = []
        shadowed: list[NameBinding] = []
        while True:
            parameters.append(self.parameter(shadowed))
            if not _is_op(self.peek(), ","):
                break
            self.advance()
        closing = self.expect(")", "')'")
        operator = self.peek()
        if not _is_op(operator, ":=", ":=="):
            raise self.error(operator.start, "':='")
        if name.value in names.UNDEFINABLE_FUNCTIONS:
            raise self.error(operator.start, "a function that can be defined")
        self.advance()
        params_node = Node(
            Kind.PARAMS,
            parameters[0].start if parameters else closing.start,
            closing.end,
            tuple(parameters),
        )
        spelled = tuple(str(parameter.value) for parameter in parameters)
        self._declare(FunctionDeclaration(str(name.value), spelled, has_body=False))
        body = None
        try:
            if not self._ends_assignment(self.peek()):
                body = self.assignment()
        finally:
            self._release(shadowed, str(name.value))
        children = (params_node,) if body is None else (params_node, body.node)
        end = operator.end if body is None else body.hi
        if body is not None:
            self._declare(FunctionDeclaration(str(name.value), spelled, has_body=True))
        node = Node(Kind.FUNDEF, name.start, end, children, name.value)
        return _Parsed(node, name.start, end)

    def parameter(self, shadowed: list[NameBinding]) -> Node:
        """One parameter, declared while the body is parsed.

        It's declared in the caller's state directly, so the body can use it; whether it
        is reported in `declarations` is up to `_release`. `shadowed` collects each name's
        previous binding, so `_release` can restore a session variable of the same name.
        """
        token = self.lexer.name_run_at(self.pos)
        self.pos = token.end
        canonical = self.lexer.new_variable(token.value)
        shadowed.append(self.lexer.state.binding(canonical))
        self.lexer.state.declare(VariableDeclaration(canonical))
        return Node(Kind.NAME, token.start, token.end, (), canonical, _surface(token))

    def _release(self, shadowed: list[NameBinding], defined: str) -> None:
        """Undo the parameter declarations, latest first, and report the new names.

        The order matters for a repeated parameter, whose second binding shadows its
        first. A parameter named like the function being defined is left alone, since the
        name now belongs to the function.

        A parameter that was not a known name is reported as a variable declaration, which
        the session applies when the line is entered: the parameter outlives the
        definition as a plain variable. Undoing it here first keeps a line that is only
        parsed, never entered, from changing the state.
        """
        state = self.lexer.state
        for binding in reversed(shadowed):
            if state.same_name(binding.name, defined):
                continue
            state.rebind(binding)
            if binding.variable is None and binding.function is None:
                self.declarations.append(VariableDeclaration(binding.name))

    def domain_declaration(self, name: Token) -> _Parsed:
        """`x :ε Real [0, inf)`.

        An unrecognized domain is not an error; the domain becomes `?`.
        """
        self.advance()
        self.advance()
        token = self.lexer.name_run_at(self.pos)
        self.pos = token.end
        domain = self.lexer.state.lookup(token.value) or token.value
        if domain not in names.DOMAINS:
            domain = "?"
        children: tuple[Node, ...] = (_name_node(name),)
        end = token.end
        if _is_op(self.peek(), "(", "["):
            interval = self.interval()
            children += (interval.node,)
            end = interval.hi
        self.declarations.append(DomainDeclaration(str(name.value), domain))
        node = Node(Kind.DOMAIN, name.start, end, children, domain)
        return _Parsed(node, name.start, end)

    def interval(self) -> _Parsed:
        """An interval such as `[0, inf)`, closed by either kind of bracket."""
        opening = self.advance()
        low = self.implication()
        self.expect(",", "','")
        high = self.implication()
        closing = self.peek()
        if not _is_op(closing, ")", "]"):
            raise self.error(closing.start, "')' or ']'")
        self.advance()
        node = Node(
            Kind.INTERVAL,
            opening.start,
            closing.end,
            (low.node, high.node),
            opening.value + closing.value,
        )
        return _Parsed.of(node)

    def _ends_assignment(self, token: Token) -> bool:
        return token.kind is TokenKind.EOF or _is_op(token, *_CLOSERS)

    def _record_assignment(self, lhs: Node, right: _Parsed | None) -> None:
        if lhs.kind is not Kind.NAME:
            return
        name = str(lhs.value)
        if name in names.SETTINGS:
            self.declarations.append(SettingDeclaration(name, _setting_value(right)))
            return
        self.declarations.append(VariableDeclaration(name, has_value=right is not None))

    def _declare(self, declaration: Declaration) -> None:
        """Register `declaration` in the state immediately, and report it.

        A function must be registered before its body is lexed. (`parameter` registers
        without reporting, since the caller doesn't gain parameter names.)
        """
        self.lexer.state.declare(declaration)
        self.declarations.append(declaration)


def _is_op(token: Token, *texts: str) -> bool:
    return token.kind is TokenKind.OP and token.value in texts


def _combine(
    kind: Kind,
    left: _Parsed,
    right: _Parsed,
    value: str | None = None,
    surface: str | None = None,
) -> _Parsed:
    node = Node(kind, left.lo, right.hi, (left.node, right.node), value, surface)
    return _Parsed(node, node.start, node.end)


def _run(kind: Kind, parts: list[_Parsed], gaps: list[tuple[str, str]]) -> _Parsed:
    """A `SUM` or `PRODUCT` over `parts`, with one `gap` between each pair.

    A single part is returned as is.
    """
    if not gaps:
        return parts[0]
    value = "".join(canonical for canonical, _ in gaps)
    spelled = "".join(surface for _, surface in gaps)
    node = Node(
        kind,
        parts[0].lo,
        parts[-1].hi,
        tuple(part.node for part in parts),
        value,
        None if spelled == value else spelled,
    )
    return _Parsed(node, node.start, node.end)


def _surface(token: Token) -> str | None:
    return token.surface if token.surface != token.value else None


def _name_node(token: Token) -> Node:
    return Node(Kind.NAME, token.start, token.end, (), token.value, _surface(token))


def _string_node(token: Token) -> Node:
    return Node(Kind.STRING, token.start, token.end, (), token.value)


def _setting_value(right: _Parsed | None) -> str:
    if right is None:
        return ""
    return str(right.node.value) if right.node.value is not None else ""

"""Recursive descent with precedence climbing.

Two of the productions are not ordinary precedence levels and are worth
naming here:

`operand` is what a bare application consumes - a `SUB` chain over
applications, optionally signed. It is looser than `!` and tighter than `^`, so
`SIN x!` is `SIN(x)!` and `SIN x^2` is `SIN(x)^2`.

`:=` is settled in `primary`: a name immediately followed by `:=` becomes an
assignment before any surrounding operator is considered, which is what makes
`"inapplicable"=a_:=F(...)` a relation whose right operand is an assignment.

`sum` and `product` build one node per run rather than a spine of pairs,
because that is what the original builds: Transfer Save keeps the parentheses
of `(a-b)+c` and `(a*b)*c`, which it would drop if they were the same tree as
`a-b+c` and `a*b*c`, and the selection highlight gives each of those runs
three operands. A run carries one operator per gap, so `a-b+c` holds `b` bare
and its sign among the operators; the highlight covers `b`, not `- b`. `/` and
the dot product are genuinely binary and close a run - `(a*b)/c` saves as
`a*b/c` - which is why `a*b/c*d` is a product of `a·b/c` and `d`.
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
#: The values a setting takes, indexed by their folded spelling. `NORMAL` and
#: `EXPAND` are built-in function names as well, so a name met with nothing to
#: apply to is looked up here before it is read as an application.
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

    `node.start`/`node.end` is the highlight span, which excludes enclosing
    parentheses; `lo`/`hi` include them, so an enclosing node's span reaches
    over the parentheses of its operands.
    """

    node: Node
    lo: int
    hi: int

    @classmethod
    def of(cls, node: Node) -> _Parsed:
        return cls(node, node.start, node.end)


class Parser:
    """One expression's worth of parsing."""

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
            # A `|` met where an operator could continue the expression closes
            # the innermost open bar rather than opening a new one.
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
        """`IMP` is the one logical operator that folds left."""
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
        """`AND`, `OR` and `XOR`, each nesting to the right.

        `a AND b AND c` is `a AND (b AND c)`, which is what the original
        builds and what its own saved form spells out.
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
        """A relation is binary, and a chain of them nests to the left.

        `a=b<c` is `(a=b)<c`: one operator per node, two operands each.
        """
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
        """A run of `+` and `-` over signed terms, held as one node."""
        terms = [self.signed()]
        gaps: list[tuple[str, str]] = []
        while _is_op(self.peek(), "+", "-"):
            token = self.advance()
            gaps.append((token.value, token.surface or token.value))
            terms.append(self.signed())
        return _run(Kind.SUM, terms, gaps)

    def signed(self) -> _Parsed:
        """A sign over a whole product: `-x/y*z` is `-(x/y·z)`.

        The ladder is `^` over `*` `/` `.` and juxtaposition, over a unary
        sign, over binary `+` and `-`.
        """
        return self._sign(self.signed) or self.product()

    def product(self) -> _Parsed:
        """`*`, `/`, the dot product, and juxtaposition, all left to right.

        A run of `*` and juxtaposition is one node; `/` and the dot product
        are binary and close the run, becoming its left operand. `a*b/c*d` is
        therefore a product of `a·b/c` and `d`, which is how the original
        offers it to the cursor.
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
        """A sign is still accepted where a factor is expected.

        There it governs that factor alone and no more of the product, so
        `a*-b*c` is `(a·(-b))·c`.
        """
        return self._sign(self.factor) or self.power()

    def _sign(self, inner) -> _Parsed | None:
        """`-u`, `+u` or `±u`, with `inner` parsing the `u`.

        `None` when no sign is there, leaving the caller to parse whatever
        binds tighter.
        """
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
            # Right-associative, and the exponent may carry its own sign,
            # which reaches no further than the exponent: `a^-b*c` is
            # `(a^(-b))·c`.
            return _combine(Kind.BINOP, base, self.factor(), "^", token.surface)
        return base

    def subscripted(self) -> _Parsed:
        """`u SUB i`, whose index reaches no further than one application.

        A postfix operator written after a subscript applies to the element and
        not to the index: `ODE_APPR.MTH` writes a Taylor coefficient's
        denominator `v_ SUB 1!`, meaning the factorial of the first element, and
        the answer the manual prints for that file is the answer that reading
        gives. So the index is one application and the postfix run belongs to
        the whole subscript.
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
        """`left` under the run of postfix operators written after it."""
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
        """A comma-separated argument list, which may not be empty.

        Arity is the engine's business: many built-ins are variadic, and
        calling a user function with fewer arguments than it has parameters is
        meaningful.
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

        With exactly one argument this is a product, which the ordinary
        juxtaposition rule already builds, so we decline it here and let the
        product level have it. Anything else is a syntax error, reported once
        the whole list has been consumed.
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
        """A setting's value where the function of that name could not stand.

        `Normal` and `Expand` are values Options takes and built-in functions
        as well. Given an argument list they are the function; given nothing to
        apply to they are the value, which is an operand in its own right. The
        original accepts `DisplayFormat := Normal` and a bare `Normal`, and
        rejects a bare `SIN`, which is exactly this rule.
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
        """`SIN x`: exactly one operand, and `^` and `!` bind outside it."""
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
        """`SQRT^3 25`: apply, then raise.

        The exponent attaches to the name, so `SIN^-1 x` is the reciprocal
        `1/SIN(x)` and not `ASIN(x)`.
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
        """A vector literal. Brackets never group, so this is the only `[`."""
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

        The parameters are registered as known variables for the length of the
        body and no longer. Registering them is unavoidable, since `F(mx,mf) :=
        mx+mf` cannot be lexed otherwise: in Character mode `mx` is `m*x` until
        something declares it. Keeping them afterwards is a leak - the session
        would gain a variable the user never asked for, and after `FF(a):=a^2`
        a later `A(t)` would read as the product `a*t` rather than as a call.
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
        """One formal parameter, declared for as long as the body is read.

        Declared in the caller's state but not reported to it: a parameter is a
        name the body uses, not one the user declared, so what `_release` takes
        away again must not come back through the caller. `shadowed` collects
        what each name stood for beforehand, which is what puts a genuine
        session variable of the same name back.
        """
        token = self.lexer.name_run_at(self.pos)
        self.pos = token.end
        canonical = self.lexer.new_variable(token.value)
        shadowed.append(self.lexer.state.binding(canonical))
        self.lexer.state.declare(VariableDeclaration(canonical))
        return Node(Kind.NAME, token.start, token.end, (), canonical, _surface(token))

    def _release(self, shadowed: list[NameBinding], defined: str) -> None:
        """Undo the parameter declarations, latest first.

        Latest first because a name written twice was shadowed by its own first
        declaration the second time. A parameter spelled like the function
        being defined is left alone: the definition head has taken that name
        over since, and it is the head's now.
        """
        for binding in reversed(shadowed):
            if binding.name != defined:
                self.lexer.state.rebind(binding)

    def domain_declaration(self, name: Token) -> _Parsed:
        """`x :ε Real [0, inf)`.

        An unrecognised domain is not an error; the domain becomes `?`.
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
        """`[0, inf)`: standard notation, and it may close with either bracket."""
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
        """Register now, and report it too.

        A definition head must be in the table before its own body is lexed,
        which is why the parser touches the caller's state at all. The other
        place it does, `parameter`, registers without reporting, because a
        parameter is not a name the caller gains.
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
    """A `SUM` or a `PRODUCT` over `parts`, one `gap` between each pair.

    A run of one is just its term: nothing was written to make it a run.
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

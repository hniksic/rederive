"""Recursive descent with precedence climbing.

Some productions aren't ordinary precedence levels:

`chain` is a primary followed by subscript links (`SUB`, `SUB SUB`, `ROW`, `COL`, each
with one index) and postfix operators (`` ` ``, `!`, `%`, `DEG`), applied left to right:
`k SUB i!` is `(k SUB i)!`. It binds tighter than `^`.

`operand` is what a bare application consumes: an optionally signed application with
subscript links but no postfix operators, so `SIN x!` is `SIN(x)!` and `SIN x^2` is
`SIN(x)^2`.

`:=` is handled in `primary`: a name followed by `:=` becomes an assignment before any
surrounding operator is considered, so `"inapplicable"=a_:=F(...)` is a relation whose
right operand is an assignment. A name with subscripts followed by `:=` is a target too,
which `chain` recognizes once it has read the links. `RETURN u` takes everything up to
the next comma or closing bracket, as an assignment's right side does.

Some forms are statements, recognized only where an expression starts: the variadic
definition head `F v := u` is one at the top, in a vector and as an argument of `PROG`.

`sum` and `product` build one node per run rather than nested pairs, as the original
does. This lets Transfer Save keep the parentheses of `(a-b)+c` and `(a*b)*c`, and makes
the selection highlight see three operands in `a-b+c`. A run stores one operator per
gap, so in `a-b+c` the operand is `b` and `-` is among the operators; the highlight
covers `b`, not `- b`. `/` and the dot product are binary and close a run (`(a*b)/c`
saves as `a*b/c`), so `a*b/c*d` is a product of `a·b/c` and `d`.
"""

from __future__ import annotations

from dataclasses import dataclass

from rederive.model.expr import (
    CHAIN_WORDS,
    VARIADIC,
    Kind,
    Node,
    is_deep,
    parameter_names,
)
from rederive.syntax import names
from rederive.syntax.errors import DeriveSyntaxError
from rederive.syntax.lexer import QUOTE, Lexer, is_letter
from rederive.syntax.state import (
    CaseMode,
    Declaration,
    DomainDeclaration,
    FunctionDeclaration,
    NameBinding,
    SettingDeclaration,
    VariableDeclaration,
    ascii_lower,
    fold,
    normalize,
    reserved,
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
_CLOSERS = (",", "]", ")", "}", ";")
#: The update operators, which change a variable's value by an operation.
_UPDATES = (":+", ":-", ":*", ":/")
#: The operators after an assignment's target.
_ASSIGNING = (":=", *_UPDATES)
#: Domain names by folded spelling.
_DOMAINS = {fold(domain): domain for domain in names.DOMAINS}
#: Setting values by folded spelling. `NORMAL` and `EXPAND` are also built-in functions,
#: so a function name with nothing to apply to is looked up here first.
_SETTING_VALUES = {fold(value): value for value in names.SETTING_VALUES}
#: Setting names by folded spelling.
_SETTINGS = {fold(setting): setting for setting in names.SETTINGS}
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
        #: Parameters spelled like a name known before they were declared, which stay
        #: what they were once the definition is read.
        self.known: set[str] = set()
        #: The bindings of variables the expression assigns, as they were before the
        #: assignment declared them for the rest of it (see `_record_assignment`).
        self.assigned: list[NameBinding] = []
        #: The parameters of the definitions whose bodies are being read: an
        #: assignment to one assigns a local, which declares nothing in the session.
        self.formals: frozenset[str] = frozenset()

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
        return token.value in ("(", "[", "{", "?")

    def starts_operand(self, token: Token) -> bool:
        """Whether `token` can begin an operand: a factor, a sign, or a quote, which
        is no factor, so `f '(x)` is no product."""
        return self.starts_factor(token) or _is_op(token, *_SIGNS, QUOTE)

    # -- the top of the grammar ---------------------------------------------

    def parse(self) -> Node:
        """The expression, read whole; the state is left as it was found, what the
        expression declares being reported in `declarations`."""
        try:
            return self._parsed()
        finally:
            state = self.lexer.state
            for binding in reversed(self.assigned):
                state.rebind(binding)

    def _parsed(self) -> Node:
        parsed = self.statement()
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

    def statement(self) -> _Parsed:
        """An expression where a statement may stand, which may be a variadic head."""
        head = self._variadic_head()
        if head is not None:
            return self.variadic_definition(*head)
        return self.assignment()

    def assignment(self) -> _Parsed:
        return self.equivalence()

    def equivalence(self) -> _Parsed:
        """`IFF`, the loosest operator, left-associative like `IMP`."""
        left = self.implication()
        while self._word(self.peek(), "IFF"):
            self._take()
            right = self.implication()
            left = _combine(Kind.IFF, left, right)
        return left

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
        """Binary relations, left-associative: `a=b<c` is `(a=b)<c`.

        `SUBSET` is among them: `a UNION b SUBSET c` compares the union.
        """
        left = self.set_operation()
        while True:
            token = self.peek()
            if self._word(token, "SUBSET"):
                self._take()
                right = self.set_operation()
                left = _combine(Kind.INFIX, left, right, "SUBSET", _glyph(token))
                continue
            if not _is_op(token, *_RELATIONS):
                break
            if token.value == "=" and self.after(token).kind is TokenKind.EOF:
                break  # the author line's trailing `=`
            self.advance()
            right = self.set_operation()
            left = _combine(Kind.REL, left, right, token.value, token.surface)
        return left

    def set_operation(self, level: int = 0) -> _Parsed:
        """The set operators, between relations and sums.

        From loosest: `UNION`, `INTERSECTION`, then the difference `\\`, each
        left-associative, so `s UNION t INTERSECTION u` is `s UNION (t INTERSECTION u)`
        and `a \\ b \\ c` is `(a \\ b) \\ c`.
        """
        if level == len(names.SET_OPERATORS):
            return self.sum()
        operator = names.SET_OPERATORS[level]
        left = self.set_operation(level + 1)
        while _is_op(self.peek(), operator):
            token = self.advance()
            right = self.set_operation(level + 1)
            left = _combine(Kind.SETOP, left, right, operator, token.surface)
        return left

    def sum(self) -> _Parsed:
        """A run of `+` and `-` over signed terms, as one node."""
        terms = [self.dot_product()]
        gaps: list[tuple[str, str]] = []
        while _is_op(self.peek(), "+", "-", "+-"):
            token = self.peek()
            if token.value == "+-":
                # `n "+-" 1` adds the plus-or-minus of the term: `n + ±1`.
                gaps.append(("+", "+"))
            else:
                self.advance()
                gaps.append((token.value, token.surface or token.value))
            terms.append(self.dot_product())
        return _run(Kind.SUM, terms, gaps)

    def dot_product(self) -> _Parsed:
        """`DOTPRODUCT`, looser than `CROSSPRODUCT` and tighter than `+` and `-`."""
        return self._vector_product("DOTPRODUCT", self.cross_product)

    def cross_product(self) -> _Parsed:
        """`CROSSPRODUCT`, looser than a sign: `-a CROSSPRODUCT b` is `(-a)
        CROSSPRODUCT b`."""
        return self._vector_product("CROSSPRODUCT", self.signed)

    def _vector_product(self, word: str, tighter) -> _Parsed:
        left = tighter()
        while self._word(self.peek(), word):
            token = self._take()
            right = tighter()
            left = _combine(Kind.INFIX, left, right, word, _glyph(token))
        return left

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
            elif self.starts_factor(token) and not self._word(token, *names.INFIX_WORDS):
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
        base = self.chain()
        token = self.peek()
        if _is_op(token, "^"):
            self.advance()
            # Right-associative. The exponent may have its own sign, which applies only
            # to the exponent: `a^-b*c` is `(a^(-b))·c`.
            return _combine(Kind.BINOP, base, self.factor(), "^", token.surface)
        return base

    def chain(self) -> _Parsed:
        """An application with its subscript links and postfix operators, left to right.

        Each applies to everything before it: `k SUB i!` is `(k SUB i)!` and ``a` SUB 1``
        is ``(a`) SUB 1``. `ODE_APPR.MTH` writes `v_ SUB 1!` for the factorial of the
        first element, which is the reading that reproduces the manual's answer.

        A name with subscript links followed by an assignment operator is the target of
        the assignment: `a SUB i := 3` assigns an element.
        """
        left = self._linked(self.application(), postfix=True)
        if _is_op(self.peek(), *_ASSIGNING) and _is_target(left.node):
            return self._assigned(left)
        return left

    def _linked(self, left: _Parsed, postfix: bool) -> _Parsed:
        """`left` with the subscript links that follow, and with `postfix` the postfix
        operators too."""
        while True:
            token = self.peek()
            link = self._link(token)
            if link is not None:
                word, end = link
                self.pos = end
                right = self.index()
                kind = Kind.SUB if word == "SUB" else Kind.INFIX
                value = None if word == "SUB" else word
                left = _combine(kind, left, right, value, _glyph(token))
                continue
            if not postfix:
                return left
            if _is_op(token, *_POSTFIX):
                operator = token.value
            elif self._word(token, "DEG"):
                operator = "DEG"
            else:
                return left
            taken = self._take()
            end = taken.word_end if taken.kind is TokenKind.NAME else taken.end
            left = _Parsed(
                Node(Kind.POSTOP, left.lo, end, (left.node,), operator),
                left.lo,
                end,
            )

    def _link(self, token: Token) -> tuple[str, int] | None:
        """The subscript link starting at `token`, and where it ends, if it is one.

        `SUB SUB` is one link however many blanks stand between the words, and so is a
        longer run of `SUB`: `B SUB SUB SUB k` indexes two levels down.
        """
        if _is_op(token, "SUB"):
            words = ["SUB"]
            end = token.end
            following = self.lexer.token_at(end)
            while _is_op(following, "SUB"):
                words.append("SUB")
                end = following.end
                following = self.lexer.token_at(end)
            return " ".join(words), end
        if self._word(token, *CHAIN_WORDS):
            word = token.value if token.kind is TokenKind.OP else str(token.word)
            return word, token.end if token.kind is TokenKind.OP else token.word_end
        return None

    def index(self) -> _Parsed:
        """The index of a subscript link: one application, which may be signed, as in `v
        SUB -1`. A name in it followed by `:=` is not assigned: in `a SUB i := 3` the
        target is the element."""
        signed = self._sign(self.index)
        if signed is not None:
            return signed
        return self.application(indexing=True)

    # -- word operators -----------------------------------------------------

    def _word(self, token: Token, *words: str) -> bool:
        """Whether `token`, where an operator can stand, is one of the operator `words`.

        That is the word itself, or another spelling of it that is no user's name (see
        `Token.word`); an infix word must then also be followed by an operand.
        """
        if token.kind is TokenKind.OP:
            return token.value in words
        if token.kind is not TokenKind.NAME or token.word not in words:
            return False
        if token.word not in names.INFIX_WORDS:
            return True
        return self.starts_operand(self.lexer.token_at(token.word_end))

    def _take(self) -> Token:
        """Advance past the operator at the current token, which `_word` accepted."""
        token = self.peek()
        self.pos = token.word_end if token.kind is TokenKind.NAME else token.end
        return token

    # -- application --------------------------------------------------------

    def application(self, indexing: bool = False) -> _Parsed:
        """A call, a bare application, or a primary.

        With `indexing`, this is the index of a subscript link, where a name followed by
        `:=` is not the target of an assignment.
        """
        token = self.peek()
        if _is_op(token, QUOTE):
            return self.quote(token)
        if token.kind is TokenKind.NAME and not token.is_defhead:
            following = self.after(token)
            if token.is_function and token.value == "RETURN":
                return self.return_statement(token)
            if token.is_function and self._folded_variable(token, following):
                self.advance()
                written = token.surface or token.value
                lowered = ascii_lower(normalize(written))
                surface = written if written != lowered else None
                return _Parsed.of(
                    Node(Kind.NAME, token.start, token.end, (), lowered, surface)
                )
            if token.is_function:
                if _is_op(following, "("):
                    return self.call(token)
                if self._juxtaposed(token):
                    self.advance()
                    written = token.surface or token.value
                    factor = Node(Kind.NAME, token.start, token.end, (), written)
                    return _Parsed.of(factor)
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
        return self.primary(indexing)

    def quote(self, token: Token) -> _Parsed:
        """`'u`, the quote of the application `u` that follows: a call, a bare
        application or a primary, itself possibly quoted. The subscript links, postfix
        operators and `^` after it apply to the quote, so `'p4^2` is `('p4)^2`."""
        self.advance()
        if not self.starts_operand(self.peek()) or _is_op(self.peek(), *_SIGNS):
            raise self.error(self.peek().start, "an operand")
        operand = self.application()
        node = Node(Kind.QUOTE, token.start, operand.hi, (operand.node,))
        return _Parsed(node, token.start, operand.hi)

    def _juxtaposed(self, token: Token) -> bool:
        """Whether the function name `token`, which no `(` follows, is a variable: a
        user's function in the state block of a worksheet (see `ParseState.restoring`),
        where only a built-in applies to an operand without parentheses."""
        state = self.lexer.state
        return state.restoring and token.value in state.functions

    def _folded_variable(self, token: Token, following: Token) -> bool:
        """Whether the function name `token` is a variable all the same: it names the
        function only by folding case, and nothing it could apply to follows.

        Recorded Derive worksheets read `ITERATES(N(n), n, 1, 25)` with `n` a variable
        beside the function `N`, and `fre*liste` beside a function `FRE`: where the
        function couldn't stand, the spelling is an undeclared variable, its ASCII
        letters in lower case. A setting value (`Expand`), and a spelling whose lower
        case is itself a function, keep their meaning.
        """
        state = self.lexer.state
        written = token.surface or token.value
        if (
            state.case_mode != CaseMode.INSENSITIVE
            or state.knows_exactly(written)
            or _is_op(following, "(", "^")
            or self.starts_operand(following)
            or fold(written) in _SETTING_VALUES
        ):
            return False
        lowered = ascii_lower(normalize(written))
        return not state.knows_exactly(lowered) or lowered in state.variables

    def return_statement(self, name: Token) -> _Parsed:
        """`RETURN u`, whose operand extends to the next comma or closing bracket, so
        `RETURN(a) + 1` returns `a + 1`. It is a call of `RETURN` with one argument."""
        self.advance()
        if not self.starts_operand(self.peek()):
            raise self.error(self.peek().start, "an operand")
        operand = self.assignment()
        node = Node(Kind.CALL, name.start, operand.hi, (_name_node(name), operand.node))
        return _Parsed(node, name.start, operand.hi)

    def call(self, name: Token) -> _Parsed:
        """`F(u, v)`, or `F()`, a call with no arguments.

        The arguments of `PROG` are statements.
        """
        self.advance()
        self.expect("(", "'('")
        statements = name.value == "PROG"
        arguments = [] if _is_op(self.peek(), ")") else self.arguments(statements)
        close = self.expect(")", "')'")
        children = (_name_node(name), *(argument.node for argument in arguments))
        node = Node(Kind.CALL, name.start, close.end, children)
        return _Parsed(node, name.start, close.end)

    def arguments(self, statements: bool = False) -> list[_Parsed]:
        """A non-empty comma-separated argument list, each a statement if `statements`.

        Arity is checked by the engine: many built-ins are variadic, and calling a user
        function with fewer arguments than parameters is meaningful.
        """
        if _is_op(self.peek(), ")"):
            raise self.error(self.peek().start, "an argument")
        argument = self.statement if statements else self.assignment
        arguments = [argument()]
        while _is_op(self.peek(), ","):
            self.advance()
            arguments.append(argument())
        return arguments

    def argument_list_on_a_variable(self, name: Token, opening: Token) -> _Parsed | None:
        """`NAME(...)` where `NAME` is not a known function.

        With one argument, a name known as anything else, such as a variable or a
        parameter, multiplies, so return None and let juxtaposition handle it. An
        unknown name is a function called before it is defined, or one that never is,
        when the call can't be a product: with no argument or several, which goes for a
        known name too. In a file it is a call whenever the `(`
        follows the name directly, since Derive writes every product with `*`; the name
        is then declared an arbitrary function, so later lines read it the same way.
        """
        count = self.lexer.count_arguments(opening.start)
        if count is None:
            return None
        written = name.surface or name.value
        info = self.lexer.state.resolve(written)
        # Another spelling of a word operator, such as `row`, is no known name here.
        known = info is not None and not (
            info.is_keyword_operator and info.canonical in names.INFIX_WORDS
        )
        if known and not info.is_function and not info.is_constant:
            # A Greek letter is known by its name only as a letter: until the session
            # declares it, it is as unknown as any other name.
            letter = names.GREEK_LETTERS.get(written) or names.DOS_GREEK.get(written)
            if letter is not None and not self.lexer.state.is_declared(letter):
                known = False
        if known and count == 1:
            return None
        in_file = self.lexer.state.reading_file and opening.start == name.end
        if count == 1 and not in_file:
            return None
        if in_file and not known:
            parameters = tuple(f"_{index}" for index in range(count))
            self._declare(FunctionDeclaration(written, parameters, has_body=False))
        self.advance()
        self.expect("(", "'('")
        arguments = [] if _is_op(self.peek(), ")") else self.arguments()
        close = self.expect(")", "')'")
        head = Node(Kind.NAME, name.start, name.end, (), written)
        children = (head, *(argument.node for argument in arguments))
        node = Node(Kind.CALL, name.start, close.end, children)
        return _Parsed(node, name.start, close.end)

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
        return self._linked(self.application(), postfix=False)

    # -- primaries ----------------------------------------------------------

    def primary(self, indexing: bool = False) -> _Parsed:
        token = self.peek()
        if token.kind is TokenKind.NUMBER:
            self.advance()
            surface = token.surface if token.surface != token.value else None
            node = Node(Kind.NUMBER, token.start, token.end, (), token.value, surface)
            return _Parsed.of(node)
        if token.kind is TokenKind.NAME:
            return self.name_primary(token, indexing)
        if token.kind is TokenKind.STRING:
            following = self.after(token)
            if _is_op(following, *_ASSIGNING) and not indexing:
                return self.assignment_to(token, _string_node(token))
            if _is_op(following, ":epsilon"):
                return self.domain_declaration(token)
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
        if _is_op(token, "{"):
            return self.set_literal(token)
        if _is_op(token, "|"):
            return self.absolute_value(token)
        raise self.error(token.start, "an operand")

    def name_primary(self, token: Token, indexing: bool = False) -> _Parsed:
        if token.is_defhead:
            return self.function_definition(token)
        following = self.after(token)
        if _is_op(following, *_ASSIGNING) and not indexing:
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
        return self._listed(Kind.VECTOR, opening, "]")

    def set_literal(self, opening: Token) -> _Parsed:
        """A set literal such as `{1, 2, 3}`; `{}` is the empty set."""
        return self._listed(Kind.SET, opening, "}")

    def _listed(self, kind: Kind, opening: Token, closer: str) -> _Parsed:
        """The elements of a vector or set, up to `closer`.

        One element may be the ellipsis of an arithmetic sequence, written between its
        last two terms: `[1, ..., n]` counts by one, `[0, 2, ..., 10]` by the difference
        of the first two. It stands nowhere else.

        In a vector, `;` separates the rows of a matrix: `[a, b; c, d]` is `[[a, b], [c,
        d]]`. Each row needs an element. Elements of a vector are statements.
        """
        self.advance()
        rows: list[list[_Parsed]] = []
        elements: list[_Parsed] = []
        if not _is_op(self.peek(), closer):
            elements.append(self._element(kind))
            while True:
                token = self.peek()
                if _is_op(token, ","):
                    self.advance()
                    elements.append(self._element(kind))
                elif kind is Kind.VECTOR and _is_op(token, ";"):
                    self.advance()
                    rows.append(elements)
                    elements = [self._element(kind)]
                else:
                    break
        closing = self.expect(closer, f"'{closer}'")
        if rows:
            children = tuple(
                Node(Kind.VECTOR, row[0].lo, row[-1].hi, self._sequence(row))
                for row in [*rows, elements]
            )
        else:
            children = self._sequence(elements)
        return _Parsed.of(Node(kind, opening.start, closing.end, children))

    def _sequence(self, elements: list[_Parsed]) -> tuple[Node, ...]:
        """The elements' nodes, with an ellipsis only where one can stand."""
        children = tuple(element.node for element in elements)
        for index, child in enumerate(children):
            if child.kind is Kind.ELLIPSIS and (
                len(children) not in (3, 4) or index != len(children) - 2
            ):
                raise self.error(child.start, "an element")
        return children

    def _element(self, kind: Kind) -> _Parsed:
        token = self.peek()
        if _is_op(token, names.ELLIPSIS):
            self.advance()
            node = Node(Kind.ELLIPSIS, token.start, token.end, (), names.ELLIPSIS)
            return _Parsed.of(node)
        return self.statement() if kind is Kind.VECTOR else self.assignment()

    def absolute_value(self, opening: Token) -> _Parsed:
        """`|u|` is `ABS(u)`, and bars nest."""
        if not self.starts_operand(self.after(opening)):
            raise self.error(opening.start, "an operand")
        self.advance()
        self.bars += 1
        inner = self.equivalence()
        closing = self.peek()
        if not _is_op(closing, "|"):
            raise self.error(closing.start, "'|'")
        self.advance()
        self.bars -= 1
        return _Parsed.of(Node(Kind.ABS, opening.start, closing.end, (inner.node,)))

    # -- declarations -------------------------------------------------------

    def assignment_to(self, name: Token, lhs: Node) -> _Parsed:
        """`name := expr`, with an empty right side allowed, or an update `name :+ expr`.

        A reserved spelling can't be assigned (`reserved`), except a setting's name,
        whose assignment changes the setting. Its value may be a word that is also a
        function's name: `Angle := Grad` is a setting line with an unknown value.
        """
        self.advance()
        if lhs.kind is Kind.NAME and reserved(name.surface or name.value):
            raise self.error(self.peek().start, "a name that can be assigned")
        return self._assigned(_Parsed(lhs, name.start, name.end))

    def _assigned(self, target: _Parsed) -> _Parsed:
        """The assignment to `target` whose operator is the current token.

        An update (`:+`, `:-`, `:*`, `:/`) needs a right side, and declares nothing: it
        changes a variable that has a value already.
        """
        lhs = target.node
        operator = self.advance()
        update = operator.value in _UPDATES
        right = None
        token = self.peek()
        if (
            not update
            and lhs.kind is Kind.NAME
            and _setting_of(lhs) is not None
            and token.kind is TokenKind.NAME
            and token.is_function
            and self._ends_assignment(self.after(token))
        ):
            self.advance()
            written = token.surface or token.value
            written = _SETTING_VALUES.get(fold(written), written)
            right = _Parsed.of(Node(Kind.NAME, token.start, token.end, (), written))
        elif not self._ends_assignment(token):
            right = self.assignment()
        elif update:
            raise self.error(token.start, "a value")
        if lhs.kind is Kind.NAME and not update:
            lhs = self._setting_line(lhs, right)
        children = (lhs,) if right is None else (lhs, right.node)
        end = operator.end if right is None else right.hi
        if not update:
            self._record_assignment(lhs, right)
        node = Node(Kind.ASSIGN, target.lo, end, children, operator.value)
        return _Parsed(node, target.lo, end)

    def function_definition(self, name: Token) -> _Parsed:
        """`F(x,y) := body`, or `F(x,y):=` for an arbitrary function.

        A parameter may have a default, `F(x, y := 0)`, any expression up to the next
        comma or the closing parenthesis; and there may be no parameters, `F() := u`.

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
        parameters: list[Node] = []
        shadowed: list[NameBinding] = []
        try:
            if not _is_op(self.peek(), ")"):
                while True:
                    parameters.append(self._defaulted(self.parameter(shadowed)))
                    if not _is_op(self.peek(), ","):
                        break
                    self.advance()
            closing = self.expect(")", "')'")
            params_node = Node(
                Kind.PARAMS,
                parameters[0].start if parameters else closing.start,
                closing.end,
                tuple(parameters),
            )
            return self._defined(name, params_node, shadowed)
        finally:
            self._release(shadowed, str(name.value))

    def _defaulted(self, parameter: Node) -> Node:
        """`parameter`, or the assignment of its default if one follows."""
        if not _is_op(self.peek(), ":="):
            return parameter
        operator = self.advance()
        if self._ends_assignment(self.peek()):
            raise self.error(self.peek().start, "a default value")
        default = self.assignment()
        return Node(
            Kind.ASSIGN,
            parameter.start,
            default.hi,
            (parameter, default.node),
            operator.value,
        )

    def variadic_definition(self, name: Token, parameter: Token) -> _Parsed:
        """`F v := body`, whose one parameter receives a call's arguments as a vector."""
        shadowed: list[NameBinding] = []
        self.pos = parameter.start
        try:
            declared = self.parameter(shadowed)
            params_node = Node(
                Kind.PARAMS, declared.start, declared.end, (declared,), VARIADIC
            )
            return self._defined(name, params_node, shadowed)
        finally:
            self._release(shadowed, str(name.value))

    def _defined(
        self, name: Token, params_node: Node, shadowed: list[NameBinding]
    ) -> _Parsed:
        """The definition of `name` from its `:=` on, the parameters being declared."""
        operator = self.peek()
        if not _is_op(operator, ":="):
            raise self.error(operator.start, "':='")
        if reserved(name.surface or name.value, function=True):
            raise self.error(operator.start, "a function that can be defined")
        self.advance()
        spelled = parameter_names(params_node)
        variadic = params_node.value == VARIADIC
        self._declare(
            FunctionDeclaration(str(name.value), spelled, False, variadic)
        )
        body = None
        outer = self.formals
        self.formals = outer | frozenset(spelled)
        try:
            if not self._ends_assignment(self.peek()):
                body = self.assignment()
        finally:
            self.formals = outer
            self._release(shadowed, str(name.value))
            shadowed.clear()
        children = (params_node,) if body is None else (params_node, body.node)
        end = operator.end if body is None else body.hi
        if body is not None:
            self._declare(FunctionDeclaration(str(name.value), spelled, True, variadic))
        node = Node(Kind.FUNDEF, name.start, end, children, name.value)
        return _Parsed(node, name.start, end)

    def _variadic_head(self) -> tuple[Token, Token] | None:
        """The name and parameter of a variadic head `F v :=` at the current position.

        Both are taken whole in either input mode. The name must not be a built-in
        function, a constant, a keyword or a word operator, so `SIN x := 3` and `NOT p :=
        q` read as before.
        """
        lexer = self.lexer
        text = lexer.text
        start = lexer.skip_space(self.pos)
        if start >= lexer.limit or not is_letter(text[start]):
            return None
        first = lexer.name_end(start)
        second = lexer.skip_space(first)
        if second == first or second >= lexer.limit or not is_letter(text[second]):
            return None
        end = lexer.name_end(second)
        if not lexer.assignment_at(lexer.skip_space(end)):
            return None
        written, parameter = text[start:first], text[second:end]
        if reserved(written, function=True) or reserved(parameter):
            return None
        state = lexer.state
        known = state.resolve(written)
        if known is not None and (
            known.is_constant
            or known.is_keyword_operator
            or (known.is_function and known.canonical not in state.functions)
        ):
            return None
        name = Token(
            TokenKind.NAME,
            start,
            first,
            state.bound_name(written),
            written,
            is_function=True,
        )
        return name, Token(TokenKind.NAME, second, end, parameter)

    def parameter(self, shadowed: list[NameBinding]) -> Node:
        """One parameter, declared while the body is parsed.

        It's declared in the caller's state directly, so the body can use it; whether it
        is reported in `declarations` is up to `_release`. `shadowed` collects each name's
        previous binding, so `_release` can restore a session variable of the same name.

        A reserved spelling is no parameter (`reserved`), though a setting's name is.
        Any other spelling is, even one that differs from a built-in's only in case: in
        `FF(int) := int + 1`, `int` is the parameter, and `INT` the function. A parameter
        named like a known name hides it in the body.

        A string is a parameter too, as generated code writes `"$start1"`; it names a
        string variable, so nothing is declared.
        """
        token = self.peek()
        if token.kind is TokenKind.STRING:
            self.advance()
            return _string_node(token)
        token = self.lexer.name_run_at(self.pos)
        if reserved(token.value):
            raise self.error(token.start, "a parameter")
        state = self.lexer.state
        canonical = state.bound_name(token.value)
        self.pos = token.end
        if state.resolve(token.value) is not None:
            self.known.add(canonical)
        shadowed.append(state.binding(canonical))
        state.declare(VariableDeclaration(canonical))
        surface = token.value if token.value != canonical else None
        return Node(Kind.NAME, token.start, token.end, (), canonical, surface)

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
            if binding.name == defined:
                continue
            state.rebind(binding)
            if (
                binding.variable is None
                and binding.function is None
                and binding.name not in self.known
            ):
                self.declarations.append(VariableDeclaration(binding.name))

    def domain_declaration(self, name: Token) -> _Parsed:
        """`x :ε Real [0, inf)`, also of a string, which is a variable in arithmetic:
        `"IRR" :ε Vector`.

        An unrecognized domain is not an error; the domain becomes `?`.
        """
        self.advance()
        operator = self.advance()
        if reserved(name.surface or name.value, function=True):
            raise self.error(operator.start, "a name that can be declared")
        token = self.lexer.name_run_at(self.pos)
        self.pos = token.end
        domain = _DOMAINS.get(fold(token.value), "?")
        string = name.kind is TokenKind.STRING
        declared = _string_node(name) if string else _name_node(name)
        children: tuple[Node, ...] = (declared,)
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
        low = self.equivalence()
        self.expect(",", "','")
        high = self.equivalence()
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
        if _setting_of(lhs) is not None:
            value = _setting_value(right, self.lexer.text)
            self.declarations.append(SettingDeclaration(name, value))
            return
        if name in self.formals:
            return
        declaration = VariableDeclaration(name, has_value=right is not None)
        self.declarations.append(declaration)
        state = self.lexer.state
        if not state.is_declared(name):
            # Declared at once, so the rest of the line reads it as this variable: in
            # `S := 2299160, s := S`, the `S` is the one just assigned, and in Character
            # mode `[w1 := 2, w1^2]` reads `w1` whole.
            self.assigned.append(state.binding(name))
            state.declare(declaration)

    def _setting_line(self, lhs: Node, right: _Parsed | None) -> Node:
        """`lhs`, renamed to the setting a line in another case spelling assigns.

        Under `CaseMode := Insensitive`, `inputmode := word` sets `InputMode`, as old
        files write it, when no name is spelled `inputmode` and the value is one the
        setting takes; `angle := 30` and `angle :=` are variables.
        """
        name = str(lhs.value)
        state = self.lexer.state
        if (
            name in names.SETTINGS
            or state.case_mode != CaseMode.INSENSITIVE
            or state.is_declared(name)
        ):
            return lhs
        setting = _SETTINGS.get(fold(name))
        if setting is None or right is None:
            return lhs
        choices = names.SETTING_CHOICES.get(setting, frozenset())
        value = right.node
        if value.kind is Kind.NUMBER:
            legal = names.NUMBER in choices
        elif value.kind is Kind.VECTOR:
            legal = names.VECTOR in choices
        elif value.kind is Kind.NAME:
            legal = fold(str(value.value)) in {fold(choice) for choice in choices}
        else:
            legal = False
        if not legal:
            return lhs
        surface = lhs.surface or name
        return Node(Kind.NAME, lhs.start, lhs.end, (), setting, surface)

    def _declare(self, declaration: Declaration) -> None:
        """Register `declaration` in the state immediately, and report it.

        A function must be registered before its body is lexed. (`parameter` registers
        without reporting, since the caller doesn't gain parameter names.)
        """
        self.lexer.state.declare(declaration)
        self.declarations.append(declaration)


def _setting_of(lhs: Node) -> str | None:
    """The setting an assignment to `lhs` changes, or None if it assigns a variable."""
    name = str(lhs.value)
    return name if lhs.kind is Kind.NAME and name in names.SETTINGS else None


def _is_op(token: Token, *texts: str) -> bool:
    return token.kind is TokenKind.OP and token.value in texts


def _glyph(token: Token) -> str | None:
    """The glyph an operator was written as, such as `⊆` for `SUBSET`, or None."""
    return token.surface if token.kind is TokenKind.OP and token.surface else None


def _is_target(node: Node) -> bool:
    """Whether `node` is a name with subscript links, the target of an element
    assignment: `a SUB i`, `a SUB i SUB j`, `a SUB SUB j`."""
    linked = False
    while node.kind is Kind.SUB or is_deep(node):
        node = node.children[0]
        linked = True
    return linked and node.kind is Kind.NAME


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


def _setting_value(right: _Parsed | None, text: str) -> str:
    """The value a setting is assigned: a word or number, or else its text, such as the
    vector `[x, y, z]` that `VariableOrder` takes."""
    if right is None:
        return ""
    if right.node.value is not None:
        return str(right.node.value)
    return text[right.node.start : right.node.end]

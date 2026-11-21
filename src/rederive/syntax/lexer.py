"""The pull lexer.

This is the only module that consults `ParseState` to resolve names. Lexing a name
depends on the live symbol table, and in Character mode also on lookahead:
`tera:=10^12` defines `tera`, while a bare `tera` splits into four letters.

Every entry point is a pure function of position and state, so the parser can look
ahead or back up just by lexing at a different offset.
"""

from __future__ import annotations

import unicodedata
from dataclasses import replace
from fractions import Fraction

from rederive.syntax import names
from rederive.syntax.errors import DeriveSyntaxError
from rederive.syntax.source import Source
from rederive.syntax.state import InputMode, ParseState
from rederive.syntax.tokens import Token, TokenKind

#: Blanks between tokens. A line end is one too, since an expression may be entered over
#: several lines; a file is read one line at a time and never shows the lexer one.
_SPACE = " \t\n\r"
_ASSIGN_OPS = (":==", ":=", "≔")
_UPDATE_OPS = (":+", ":-", ":*", ":/")
#: Spellings of the subscript operator other than the word: the arrow, and the byte DOS
#: files store for it.
_SUBSCRIPT_GLYPHS = tuple(
    glyph for glyph, word in names.OPERATOR_GLYPHS.items() if word == "SUB"
)
_OPENERS = "([{"
_CLOSERS = ")]}"
_MARK_CATEGORIES = frozenset(("Mn", "Mc"))
_OPERATORS_BY_LENGTH = tuple(sorted(names.OPERATORS, key=len, reverse=True))
_PLUS_MINUS = '"+-"'
#: The token value of the quote operator.
QUOTE = names.PRIME_MARK
_QUESTION = "?"


def is_letter(char: str) -> bool:
    """Whether `char` may begin an identifier.

    That is an ASCII letter or any other character from U+0080 on that is not reserved
    for something else (`names.NON_LETTERS`), save combining marks, which only continue
    a name, and control characters.
    """
    if char.isascii():
        return char.isalpha()
    return char not in names.NON_LETTERS and unicodedata.category(char) not in (
        "Mn",
        "Mc",
        "Cc",
    )


def _is_continuation(char: str) -> bool:
    if char.isascii():
        return char.isalnum() or char == "_"
    return char not in names.NON_LETTERS and unicodedata.category(char) != "Cc"


def is_name(text: str) -> bool:
    """Whether `text` is a single name, as accepted by the Declare command.

    This is a lexical check, independent of input mode: a declared name may have
    several characters even in Character mode, where `area` would otherwise read as
    `a*r*e*a`. A name may end in one `?`, as `Natural?` does.
    """
    normalized = unicodedata.normalize("NFC", text).removesuffix(_QUESTION)
    return bool(
        normalized
        and is_letter(normalized[0])
        and all(_is_continuation(char) for char in normalized[1:])
    )


def _is_mark(char: str) -> bool:
    return unicodedata.category(char) in _MARK_CATEGORIES


def _digit_value(char: str) -> int | None:
    if "0" <= char <= "9":
        return ord(char) - ord("0")
    if "A" <= char <= "Z":
        return ord(char) - ord("A") + 10
    if "a" <= char <= "z":
        return ord(char) - ord("a") + 10
    return None


class Lexer:
    """Produces tokens at given positions in a `Source`.

    `limit` ends the token stream, so a file can be lexed one expression at a time
    with offsets still indexing the whole `Source.text`.
    """

    def __init__(
        self,
        source: Source,
        state: ParseState,
        start: int = 0,
        limit: int | None = None,
    ) -> None:
        self.source = source
        self.text = source.text
        self.state = state
        self.start = start
        self.limit = len(source.text) if limit is None else limit
        self._cache: dict[int, Token] = {}
        self._stamp: tuple | None = None

    # -- positions ----------------------------------------------------------

    def skip_space(self, pos: int) -> int:
        while pos < self.limit and self.text[pos] in _SPACE:
            pos += 1
        return pos

    def at(self, pos: int, literal: str) -> bool:
        return pos + len(literal) <= self.limit and self.text.startswith(literal, pos)

    def error(self, offset: int, expected: str) -> DeriveSyntaxError:
        return DeriveSyntaxError(offset, expected, self.source)

    # -- tokens -------------------------------------------------------------

    def token_at(self, pos: int) -> Token:
        """The token starting at or after `pos`, skipping whitespace."""
        stamp = self.state.stamp
        if stamp != self._stamp:
            self._cache.clear()
            self._stamp = stamp
        cached = self._cache.get(pos)
        if cached is None:
            cached = self._lex(pos)
            self._cache[pos] = cached
        return cached

    def _lex(self, pos: int) -> Token:
        pos = self.skip_space(pos)
        if pos >= self.limit:
            return Token(TokenKind.EOF, pos, pos)
        char = self.text[pos]
        if char.isascii() and char.isdigit():
            return self._number(pos)
        if char == "." and self._digit_follows(pos + 1):
            return self._number(pos)
        if char == '"':
            if self.at(pos, _PLUS_MINUS) and self._operand_at(pos + len(_PLUS_MINUS)):
                return Token(TokenKind.OP, pos, pos + 4, "+-", _PLUS_MINUS)
            return self._string(pos)
        if char == "@":
            return self._arbitrary_constant(pos)
        if char == "#":
            return self._hash(pos)
        if char == "_":
            raise self.error(pos, "a name")
        if char in names.NON_LETTERS and (
            char in names.CONSTANT_GLYPHS or char in names.FUNCTION_GLYPHS
        ):
            return self._glyph_name(pos)
        if is_letter(char):
            return self._name(pos)
        if char == names.PRIME_MARK:
            # A derivative mark is lexed with its name, so this is the quote, which the
            # parser accepts only where an operand may start.
            return Token(TokenKind.OP, pos, pos + 1, QUOTE)
        return self._operator(pos)

    def _operand_at(self, pos: int) -> bool:
        """Whether an operand starts at `pos`, after any blanks.

        `"+-"` before an operand is the plus-or-minus sign; anywhere else, as in
        `["+-"inf, "+-"]`, it is the two-character string.
        """
        pos = self.skip_space(pos)
        if pos >= self.limit:
            return False
        char = self.text[pos]
        return (
            char.isascii()
            and (char.isalnum() or char in "\"([{-|#@?_'")
            or char == "." and self._digit_follows(pos + 1)
            or not char.isascii()
            and (
                is_letter(char)
                or char in names.CONSTANT_GLYPHS
                or char in names.FUNCTION_GLYPHS
                or names.OPERATOR_GLYPHS.get(char) in ("+-", "|", "NOT")
            )
        )

    def _digit_follows(self, pos: int) -> bool:
        return pos < self.limit and self.text[pos].isascii() and self.text[pos].isdigit()

    # -- numerals -----------------------------------------------------------

    def _number(self, pos: int) -> Token:
        """A numeral in the current input base.

        Above base 10, `A`..`Z` are digits, but a numeral must start with `0`-`9`:
        `0ff` is 255 while `ff` is `f*f`. There is no exponent notation and no digit
        separator.
        """
        base = self.state.input_base
        end = pos
        while end < self.limit:
            value = _digit_value(self.text[end])
            if value is None or value >= base:
                break
            end += 1
        integer = self.text[pos:end]
        fraction = ""
        has_point = end < self.limit and self.text[end] == "."
        if has_point:
            end += 1
            start = end
            while end < self.limit:
                value = _digit_value(self.text[end])
                if value is None or value >= base:
                    break
                end += 1
            fraction = self.text[start:end]
        if not integer and not fraction:
            # A digit invalid in this base, such as `2` in base 2.
            raise self.error(pos, "a numeral")
        if end < self.limit and self.text[end] == ".":
            raise self.error(end, "a numeral")
        written = self.text[pos:end]
        value = _number_value(integer, fraction, base)
        return Token(TokenKind.NUMBER, pos, end, value, written)

    # -- atomic name forms --------------------------------------------------

    def _string(self, pos: int) -> Token:
        """A string literal, or a quoted name where a name is expected.

        There are no escapes, so `"a""b"` is two adjacent strings, and a backslash is a
        backslash. An unterminated string is not an error; Derive closes it at end of
        input. A string can't hold a line end.
        """
        end = self.text.find('"', pos + 1)
        if end < 0 or end >= self.limit:
            end = self.limit
        contents = self.text[pos + 1 : end]
        broken = next(
            (index for index, char in enumerate(contents) if char in "\n\r"), None
        )
        if broken is not None:
            raise self.error(pos + 1 + broken, "'\"'")
        return Token(TokenKind.STRING, pos, min(end + 1, self.limit), contents)

    def _arbitrary_constant(self, pos: int) -> Token:
        """`@1`, an arbitrary constant, or a bare `@`, which older files use as a name."""
        end = pos + 1
        while self._digit_follows(end):
            end += 1
        return Token(TokenKind.NAME, pos, end, self.text[pos:end])

    def _hash(self, pos: int) -> Token:
        """`#3` is a label; `#e` and `#i` are constants."""
        end = pos + 1
        while self._digit_follows(end):
            end += 1
        if end > pos + 1:
            return Token(TokenKind.LABEL, pos, end, self.text[pos + 1 : end])
        if end < self.limit and self.text[end] in "eEiI":
            spelling = self.text[pos : end + 1]
            canonical = self.state.lookup(spelling)
            if canonical is not None:
                return Token(TokenKind.NAME, pos, end + 1, canonical, spelling)
        raise self.error(end, "digits")

    def _glyph_name(self, pos: int) -> Token:
        """A glyph such as `√`, `∞` or `°` that is an alternative spelling of a name.

        Glyphs aren't identifier characters, so `a√b` is `a*SQRT(b)`.
        """
        glyph = self.text[pos]
        canonical = names.FUNCTION_GLYPHS.get(glyph)
        is_function = canonical is not None
        if canonical is None:
            canonical = names.CONSTANT_GLYPHS[glyph]
        return _named(
            Token(
                TokenKind.NAME, pos, pos + 1, canonical, glyph, is_function=is_function
            )
        )

    # -- names --------------------------------------------------------------

    def scan_run(self, pos: int) -> int:
        """The end of the maximal run of `[letter | digit | _]` at `pos`.

        A letter that may stand for a constant or a function, such as `π`, is a run of
        its own when it begins one.
        """
        if pos < self.limit and self.text[pos] in names.DISPLACEABLE_GLYPHS:
            return pos + 1
        end = pos
        while end < self.limit and _is_continuation(self.text[end]):
            end += 1
        return end

    def scan_balanced(self, pos: int) -> int | None:
        """The index just past the `)` matching the `(` at `pos`."""
        if not self.at(pos, "("):
            return None
        depth = 0
        index = pos
        in_string = False
        while index < self.limit:
            char = self.text[index]
            if in_string:
                in_string = char != '"'
            elif char == '"':
                in_string = True
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return index + 1
            index += 1
        return None

    def count_arguments(self, pos: int) -> int | None:
        """The number of comma-separated arguments in the parenthesized group at `pos`.

        This scans characters rather than parsing, because the parser needs the count to
        decide whether `f(x)` is a product and `f(x,y)` an error.
        """
        close = self.scan_balanced(pos)
        if close is None:
            return None
        depth = 0
        commas = 0
        in_string = False
        for index in range(pos, close):
            char = self.text[index]
            if in_string:
                in_string = char != '"'
            elif char == '"':
                in_string = True
            elif char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
            elif char == "," and depth == 1:
                commas += 1
        if self.skip_space(pos + 1) == close - 1:
            return 0
        return commas + 1

    def name_run_at(self, pos: int) -> Token:
        """The whole run at `pos` as one name, regardless of input mode.

        Used for the parameters of a definition head, which are registered before the
        body is lexed. A `?` directly after the run belongs to the name.
        """
        pos = self.skip_space(pos)
        if pos >= self.limit or not is_letter(self.text[pos]):
            raise self.error(pos, "a parameter name")
        end = self._questioned(self.scan_run(pos))
        return Token(TokenKind.NAME, pos, end, self.text[pos:end])

    def name_end(self, pos: int) -> int:
        """The end of the name at `pos` taken whole, with a `?` that follows it."""
        return self._questioned(self.scan_run(pos))

    def assignment_at(self, pos: int) -> bool:
        """Whether `:=`, in any spelling, is at `pos`."""
        return self._assignment_at(pos)

    def _questioned(self, end: int) -> int:
        """`end`, or past the `?` that directly follows it."""
        return end + 1 if self.at(end, _QUESTION) else end

    def _name(self, pos: int) -> Token:
        end = self.scan_run(pos)
        run = self.text[pos:end]
        question = self.at(end, _QUESTION)

        # Step 1: binding lookahead. Without it, Character mode would lex `tera:=10^12`
        # as `t*e*r*a`, an illegal left side, and `c1_:epsilonReal` would stop at `c`. A
        # binding takes the run whole, with a `?` that follows it. A keyword operator is
        # never bound: in `a SUB (j+1) := 2` the `SUB` is no definition head.
        if run not in names.KEYWORD_OPERATORS:
            for stop in (end + 1, end) if question else (end,):
                binding = self._binding(pos, stop)
                if binding is not None:
                    return binding

        # In `BB'(r)` the primes are part of the name, so it parses as an ordinary call.
        primed = self._primed_name(pos, run, end)
        if primed is not None:
            return primed

        # Step 2: longest match against known names. In Word mode only the whole run is
        # tried, so `xk` is `xk` even if `x` is declared (SOLVE.MTH's iteration variable
        # relies on this). In Character mode shorter prefixes are tried too, so `terax`
        # reads as `tera*x`. A `?` directly after the run is tried with the whole run
        # first: `PRIME?` is a name, while `x?` is `x` times the unknown.
        longest = len(run)
        shortest = 1 if self.state.input_mode == InputMode.CHARACTER else longest
        within = pos > self.start and _is_continuation(self.text[pos - 1])
        # Another spelling of a word operator, such as `row`, is the operator only where
        # one can stand. The token is what the text reads as otherwise, and carries the
        # operator for the parser to take instead.
        word: tuple[str, int] | None = None
        for length in range(longest, shortest - 1, -1):
            if length < len(run) and run[length] == "_":
                # Underscores belong to the preceding name, so a declared `x` doesn't
                # match the start of `x_`. Derive's utility files rely on this.
                continue
            spellings = [run[:length]]
            if question and length == longest:
                spellings.insert(0, run + _QUESTION)
            for written in spellings:
                if length == longest and not within:
                    library = self._library_call(written, pos + len(written))
                    if library is not None:
                        return Token(
                            TokenKind.NAME,
                            pos,
                            pos + len(written),
                            library,
                            written,
                            is_function=True,
                        )
                info = self.state.resolve(written)
                if info is None or (
                    (within or length < longest)
                    and info.canonical in names.RESERVED_WORDS
                ):
                    # Derive 6's new keywords are words of their own, not found inside
                    # a longer run as `SUB` and `AND` are: `returns` is no `RETURN`.
                    continue
                stop = pos + len(written)
                if info.is_keyword_operator:
                    if info.canonical in names.INFIX_WORDS and written != info.canonical:
                        word = word or (info.canonical, stop)
                        continue
                    return Token(TokenKind.OP, pos, stop, info.canonical, written)
                return _named(
                    Token(
                        TokenKind.NAME,
                        pos,
                        stop,
                        info.canonical,
                        written,
                        is_function=info.is_function,
                    ),
                    word,
                )

        # Step 3: mode fallback. A Word-mode name is the whole run, with its `?`.
        if self.state.input_mode == InputMode.WORD:
            stop = self._questioned(end)
        else:
            stop = pos + 1
            while stop < self.limit and _is_mark(self.text[stop]):
                stop += 1
            while stop < self.limit and self.text[stop] == "_":
                stop += 1
        written = self.text[pos:stop]
        canonical = self.state.canonical_variable(written)
        return _named(Token(TokenKind.NAME, pos, stop, canonical, written), word)

    def _library_call(self, written: str, stop: int) -> str | None:
        """The library function a whole run spelled `written` calls, ending at `stop`.

        A library function is a name only where it is called: directly before `(`, or
        in Word mode after spaces too. Anywhere else the run reads as if the library
        did not exist, so in Character mode `star + 1` stays a product of letters.
        """
        if self.state.input_mode == InputMode.WORD:
            stop = self.skip_space(stop)
        if not self.at(stop, "("):
            return None
        return self.state.library_function(written)

    def _binding(self, pos: int, stop: int) -> Token | None:
        """The text from `pos` to `stop` as a name being bound, if it is one.

        That is the target of an assignment or a domain declaration, or the name in a
        definition head `F(x, y) :=`. The name is the exact spelling (`bound_name`).
        """
        written = self.text[pos:stop]
        after = self.skip_space(stop)
        target = self._subscripted_end(after)
        if self._lvalue_operator_at(after) or (
            target is not None and self._assigning_at(self.skip_space(target))
        ):
            name = self.state.bound_name(written)
            return Token(TokenKind.NAME, pos, stop, name, written)
        close = self.scan_balanced(after)
        if close is not None and self._assignment_at(self.skip_space(close)):
            return Token(
                TokenKind.NAME,
                pos,
                stop,
                self.state.bound_name(written),
                written,
                is_function=True,
                is_defhead=True,
            )
        return None

    def _assignment_at(self, pos: int) -> bool:
        return any(self.at(pos, op) for op in _ASSIGN_OPS)

    def _assigning_at(self, pos: int) -> bool:
        """Whether an assignment or an update operator such as `:+` is at `pos`."""
        return self._assignment_at(pos) or any(self.at(pos, op) for op in _UPDATE_OPS)

    def _lvalue_operator_at(self, pos: int) -> bool:
        """Whether the operator at `pos` makes the preceding run a left-hand side."""
        return self._assigning_at(pos) or any(
            self.at(pos, op) for op in names.DOMAIN_OPERATORS
        )

    def _subscripted_end(self, pos: int) -> int | None:
        """The end of the subscript links at `pos`, as in `SUB i SUB (j+1)`, or None if
        none begins there.

        A name followed by links and an assignment is the target of the assignment, a
        binding position like the name in `x := u`, so in Character mode `counts SUB 2 :=
        u` takes `counts` whole. This scans characters rather than tokens: an index is a
        signed number, name, call or bracketed expression.
        """
        linked = False
        while True:
            pos = self.skip_space(pos)
            link = self._subscript_at(pos)
            if link is None:
                return pos if linked else None
            while link is not None:
                pos = link
                link = self._subscript_at(self.skip_space(pos))
            end = self._index_end(self.skip_space(pos))
            if end is None:
                return None
            pos = end
            linked = True

    def _subscript_at(self, pos: int) -> int | None:
        """The end of the subscript operator at `pos`, or None if there is none."""
        for glyph in _SUBSCRIPT_GLYPHS:
            if self.at(pos, glyph):
                return pos + len(glyph)
        end = pos + 3
        if end > self.limit or self.text[pos:end].upper() != "SUB":
            return None
        if end < self.limit and _is_continuation(self.text[end]):
            return None
        return end

    def _index_end(self, pos: int) -> int | None:
        """The end of the index at `pos`; see `_subscripted_end`."""
        if self.at(pos, "-") or self.at(pos, "+"):
            pos = self.skip_space(pos + 1)
        if pos >= self.limit:
            return None
        char = self.text[pos]
        if char in _OPENERS:
            return self._bracketed_end(pos)
        if char == '"':
            end = self.text.find('"', pos + 1)
            return None if end < 0 or end >= self.limit else end + 1
        if char.isascii() and char.isdigit():
            end = pos
            while end < self.limit and (self.text[end] in "0123456789."):
                end += 1
            return end
        if not is_letter(char):
            return None
        end = self._questioned(self.scan_run(pos))
        if self.at(end, "("):
            return self._bracketed_end(end)
        return end

    def _bracketed_end(self, pos: int) -> int | None:
        """The index just past the bracket closing the one at `pos`, of any kind."""
        depth = 0
        index = pos
        in_string = False
        while index < self.limit:
            char = self.text[index]
            if in_string:
                in_string = char != '"'
            elif char == '"':
                in_string = True
            elif char in _OPENERS:
                depth += 1
            elif char in _CLOSERS:
                depth -= 1
                if depth == 0:
                    return index + 1
            index += 1
        return None

    def _primed_name(self, pos: int, run: str, end: int) -> Token | None:
        """A derivative name such as `BB'` or `BB''`.

        The primes must immediately follow the name and an argument list must follow
        the primes. The name is a known function, or a name nothing is declared as,
        which is then an undefined function: `y'(x)`. In Character mode an undeclared
        name is one letter, as it would be without the marks, so `ay'(x)` is `a*y'(x)`.
        Otherwise the apostrophe is not a derivative mark.
        """
        stop = end
        while stop < self.limit and self.text[stop] == names.PRIME_MARK:
            stop += 1
        if stop == end:
            return None
        after = self.skip_space(stop)
        if not self.at(after, "("):
            return None
        known = self.state.resolve(run)
        if known is None:
            if run in names.KEYWORD_OPERATORS or not self._undeclared_whole(pos, end):
                return None
            canonical = self.state.canonical_variable(run)
        elif not known.is_function:
            return None
        else:
            canonical = known.canonical
        marks = self.text[end:stop]
        return Token(
            TokenKind.NAME,
            pos,
            stop,
            canonical + marks,
            run + marks,
            is_function=True,
        )

    def _undeclared_whole(self, pos: int, end: int) -> bool:
        """Whether the undeclared run from `pos` to `end` is one name in this mode."""
        if self.state.input_mode == InputMode.WORD:
            return True
        stop = pos + 1
        while stop < end and _is_mark(self.text[stop]):
            stop += 1
        while stop < end and self.text[stop] == "_":
            stop += 1
        return stop == end

    # -- operators ----------------------------------------------------------

    def _operator(self, pos: int) -> Token:
        for op in _OPERATORS_BY_LENGTH:
            if not self.at(pos, op):
                continue
            if op == "~":
                # Preprocessing already removed the legal (line continuation) ones.
                raise self.error(pos, "an operator")
            canonical = names.OPERATOR_GLYPHS.get(op, op)
            surface = op if canonical != op else None
            return Token(TokenKind.OP, pos, pos + len(op), canonical, surface)
        raise self.error(pos, "an operand")


def _named(token: Token, word: tuple[str, int] | None = None) -> Token:
    """`token`, carrying the word operator it may stand for (see `Token.word`).

    That is `word`, the operator another spelling of the run names, or for the constant
    `deg` in any spelling, the postfix `DEG`.
    """
    if word is None and token.value == "deg":
        word = ("DEG", token.end)
    if word is None:
        return token
    return replace(token, word=word[0], word_end=word[1])


def numeral_value(written: str, base: int) -> str:
    """The value of the numeral `written` in `base`, as its token has it: its decimal
    value as text (see `_number_value`)."""
    integer, _, fraction = written.partition(".")
    return _number_value(integer, fraction, base)


def _number_value(integer: str, fraction: str, base: int) -> str:
    """The numeral's decimal value, as text.

    A radix point doesn't change the value, so `1.0`, `1.` and `1.000` are all `1`.
    In base 10 redundant zeros are dropped (`007` is `7`, `1.50` is `1.5`), along with
    a trailing point. Other bases are converted.
    """
    if base == 10:
        return _decimal(integer, fraction)
    value = Fraction(int(integer or "0", base))
    for power, digit in enumerate(fraction, start=1):
        value += Fraction(int(digit, base), base**power)
    return _exact(value)


def _decimal(integer: str, fraction: str) -> str:
    """A base-ten numeral without redundant zeros."""
    integer = integer.lstrip("0")
    fraction = fraction.rstrip("0")
    return (integer or "0") + ("." + fraction if fraction else "")


def _exact(value: Fraction) -> str:
    """`value` exactly, as a decimal numeral or, if it doesn't terminate, a ratio.

    `0.1` in hexadecimal is `0.0625`, while `0.1` in base three is `1/3`. Nothing is
    rounded.
    """
    denominator, twos, fives = value.denominator, 0, 0
    while denominator % 2 == 0:
        denominator, twos = denominator // 2, twos + 1
    while denominator % 5 == 0:
        denominator, fives = denominator // 5, fives + 1
    if denominator != 1:
        return f"{value.numerator}/{value.denominator}"
    places = max(twos, fives)
    scaled = str(value.numerator * 10**places // value.denominator)
    if not places:
        return scaled
    scaled = scaled.rjust(places + 1, "0")
    return _decimal(scaled[:-places], scaled[-places:])

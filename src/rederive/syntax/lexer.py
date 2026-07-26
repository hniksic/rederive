"""The pull lexer.

The only module that consults `ParseState` for name resolution. Lexing a name
is not a self-contained lexical rule: it needs the live symbol table, and in
Character mode it needs lookahead as well, because `tera:=10^12` must take
`tera` whole where a bare `tera` would split into four letters.

Every entry point is a pure function of a position and the state, so the parser
can look ahead simply by lexing at another offset and can back up by forgetting
one.
"""

from __future__ import annotations

import unicodedata
from fractions import Fraction

from rederive.syntax import names
from rederive.syntax.errors import DeriveSyntaxError
from rederive.syntax.source import Source
from rederive.syntax.state import CaseMode, InputMode, ParseState, ascii_upper
from rederive.syntax.tokens import Token, TokenKind

_SPACE = " \t"
_ASSIGN_OPS = (":==", ":=")
_LETTER_CATEGORIES = frozenset(("Lu", "Ll", "Lt", "Lm", "Lo"))
_CONTINUATION_CATEGORIES = _LETTER_CATEGORIES | frozenset(("Mn", "Mc", "Nd"))
_MARK_CATEGORIES = frozenset(("Mn", "Mc"))
_OPERATORS_BY_LENGTH = tuple(sorted(names.OPERATORS, key=len, reverse=True))
_PLUS_MINUS = '"+-"'


def is_letter(char: str) -> bool:
    """Whether `char` may begin an identifier."""
    return unicodedata.category(char) in _LETTER_CATEGORIES


def _is_continuation(char: str) -> bool:
    return char == "_" or unicodedata.category(char) in _CONTINUATION_CATEGORIES


def is_name(text: str) -> bool:
    """Whether `text` is one name and nothing else.

    What a Declare command may declare. The question is lexical rather than
    grammatical: a declared name may be multi-character even in Character
    input mode, where `area` on its own would otherwise read as `a*r*e*a`.
    """
    normalized = unicodedata.normalize("NFC", text)
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
    """Turns positions in a `Source` into tokens.

    `limit` bounds the token stream, so that a file can be lexed one expression
    at a time while offsets stay indices into the whole `Source.text`.
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
        if char == "\n":
            return Token(TokenKind.EOF, pos, pos)
        if char.isascii() and char.isdigit():
            return self._number(pos)
        if char == "." and self._digit_follows(pos + 1):
            return self._number(pos)
        if char == '"':
            if self.at(pos, _PLUS_MINUS):
                return Token(TokenKind.OP, pos, pos + 4, "+-", _PLUS_MINUS)
            return self._string(pos)
        if char == "@":
            return self._arbitrary_constant(pos)
        if char == "#":
            return self._hash(pos)
        if char == "_":
            raise self.error(pos, "a name")
        if char in names.CONSTANT_GLYPHS or char in names.FUNCTION_GLYPHS:
            return self._glyph_name(pos)
        if is_letter(char):
            return self._name(pos)
        return self._operator(pos)

    def _digit_follows(self, pos: int) -> bool:
        return pos < self.limit and self.text[pos].isascii() and self.text[pos].isdigit()

    # -- numerals -----------------------------------------------------------

    def _number(self, pos: int) -> Token:
        """A numeral, in the current input base.

        Digits are those valid in the base, plus `A`..`Z` above base 10, and a
        numeral must begin with a digit: `0ff` is 255 while `ff` is `f*f`.
        There is no exponent notation and no digit separators.
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
            # A digit that is not a digit in this base, such as `2` in base 2.
            raise self.error(pos, "a numeral")
        if end < self.limit and self.text[end] == ".":
            raise self.error(end, "a numeral")
        written = self.text[pos:end]
        value = _number_value(integer, fraction, base)
        return Token(TokenKind.NUMBER, pos, end, value, written)

    # -- atomic name forms --------------------------------------------------

    def _string(self, pos: int) -> Token:
        """A string literal, or a quoted name where a name is expected.

        There is no escape mechanism, so `"a""b"` is two adjacent strings. An
        unterminated string is not an error: Derive closes it at end of input.
        """
        end = self.text.find('"', pos + 1)
        if end < 0 or end >= self.limit:
            contents = self.text[pos + 1 : self.limit]
            return Token(TokenKind.STRING, pos, self.limit, contents)
        return Token(TokenKind.STRING, pos, end + 1, self.text[pos + 1 : end])

    def _arbitrary_constant(self, pos: int) -> Token:
        end = pos + 1
        while self._digit_follows(end):
            end += 1
        if end == pos + 1:
            raise self.error(end, "digits")
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
        """`√`, `∞`, `°` and the like: alternative spellings of ordinary names.

        None of these is an identifier character, so `a√b` is `a*SQRT(b)`.
        """
        glyph = self.text[pos]
        canonical = names.FUNCTION_GLYPHS.get(glyph)
        is_function = canonical is not None
        if canonical is None:
            canonical = names.CONSTANT_GLYPHS[glyph]
        return Token(
            TokenKind.NAME, pos, pos + 1, canonical, glyph, is_function=is_function
        )

    # -- names --------------------------------------------------------------

    def scan_run(self, pos: int) -> int:
        """The end of the maximal run of `[letter | digit | _]` at `pos`."""
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
        """How many comma-separated groups the parenthesised group at `pos` has.

        A character scan, not a parse: the parser needs the count before it can
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
            elif char in "([":
                depth += 1
            elif char in ")]":
                depth -= 1
            elif char == "," and depth == 1:
                commas += 1
        if self.skip_space(pos + 1) == close - 1:
            return 0
        return commas + 1

    def name_run_at(self, pos: int) -> Token:
        """The whole run at `pos` as one name, whatever the input mode.

        Used for the formal parameters of a definition head, which are taken
        whole and registered before the body is lexed.
        """
        pos = self.skip_space(pos)
        if pos >= self.limit or not is_letter(self.text[pos]):
            raise self.error(pos, "a parameter name")
        end = self.scan_run(pos)
        return Token(TokenKind.NAME, pos, end, self.text[pos:end])

    def _name(self, pos: int) -> Token:
        end = self.scan_run(pos)
        run = self.text[pos:end]
        after = self.skip_space(end)

        # Step 1: definition-head lookahead. Without it a Character-mode
        # `tera:=10^12` would lex as `t*e*r*a`, which is not a legal left side,
        # and `c1_:epsilonReal` would stop at `c`.
        if self._lvalue_operator_at(after):
            return Token(TokenKind.NAME, pos, end, self.new_variable(run), run)
        close = self.scan_balanced(after)
        if close is not None and self._assignment_at(self.skip_space(close)):
            canonical = self.state.canonical_function(run)
            return Token(
                TokenKind.NAME,
                pos,
                end,
                canonical,
                run,
                is_function=True,
                is_defhead=True,
            )

        # An upper-case run before `(` is a function reference: upper case is
        # the canonical spelling of a function, so `SIDE(w)` is a call even
        # where `quux(z)` is a product of letters.
        if self._is_call_head(run, after):
            return Token(TokenKind.NAME, pos, end, run, run, is_function=True)

        # Step 2: longest match against known names.
        for length in range(len(run), 0, -1):
            if length < len(run) and run[length] == "_":
                # An underscore run belongs to the name in front of it, so a
                # declared `x` does not match the first letter of `x_`: they
                # are different names, which is what the `_` convention in
                # Derive's own utility files is for.
                continue
            info = self.state.resolve(run[:length])
            if info is None:
                continue
            written = run[:length]
            stop = pos + length
            if info.is_keyword_operator:
                return Token(TokenKind.OP, pos, stop, info.canonical, written)
            return Token(
                TokenKind.NAME,
                pos,
                stop,
                info.canonical,
                written,
                is_function=info.is_function,
            )

        # Step 3: mode fallback.
        if self.state.input_mode == InputMode.WORD:
            stop = end
        else:
            stop = pos + 1
            while stop < self.limit and _is_mark(self.text[stop]):
                stop += 1
            while stop < self.limit and self.text[stop] == "_":
                stop += 1
        written = self.text[pos:stop]
        canonical = self.state.canonical_variable(written)
        return Token(TokenKind.NAME, pos, stop, canonical, written)

    def _assignment_at(self, pos: int) -> bool:
        return any(self.at(pos, op) for op in _ASSIGN_OPS)

    def _lvalue_operator_at(self, pos: int) -> bool:
        """Whether what follows makes the run before it a left-hand side."""
        return self._assignment_at(pos) or any(
            self.at(pos, op) for op in names.DOMAIN_OPERATORS
        )

    def _is_call_head(self, run: str, after: int) -> bool:
        """Whether an upper-case run before `(` names a function.

        Upper case is the canonical spelling of a function, so `SIDE(w)` is a
        call where `quux(z)` is a product of letters. A setting keyword gives
        way here, since Derive's own files define `EXACT(...)`; a constant or a
        declared variable does not.
        """
        if not self.at(after, "("):
            return False
        if ascii_upper(run) != run or not any("A" <= char <= "Z" for char in run):
            return False
        known = self.state.resolve(run)
        return known is None or known.is_function or known.is_displaceable

    def new_variable(self, run: str) -> str:
        """The canonical spelling of a name this expression is declaring.

        A pre-defined name keeps its inventory spelling; anything else keeps
        the spelling it is written with here, which is what the symbol table
        will record.
        """
        known = self.state.lookup(run)
        if known is not None:
            return known
        return unicodedata.normalize("NFC", run)

    # -- operators ----------------------------------------------------------

    def _operator(self, pos: int) -> Token:
        for op in _OPERATORS_BY_LENGTH:
            if not self.at(pos, op):
                continue
            if op == "~":
                # Preprocessing removed the legal ones.
                raise self.error(pos, "an operator")
            canonical = names.OPERATOR_GLYPHS.get(op, op)
            surface = op if canonical != op else None
            return Token(TokenKind.OP, pos, pos + len(op), canonical, surface)
        raise self.error(pos, "an operand")


def _number_value(integer: str, fraction: str, base: int) -> str:
    """The numeral's decimal value, as text.

    A radix point is spelling rather than value, so `1.0`, `1.` and `1.000`
    are all the number Derive shows as `1`. Base 10 therefore drops the zeros
    that say nothing on either side - `007` is `7` and `1.50` is `1.5` - and
    the point goes with the last of the trailing ones. Other bases are
    converted.
    """
    if base == 10:
        return _decimal(integer, fraction)
    value = Fraction(int(integer or "0", base))
    for power, digit in enumerate(fraction, start=1):
        value += Fraction(int(digit, base), base**power)
    return _exact(value)


def _decimal(integer: str, fraction: str) -> str:
    """A base-ten numeral written with none of its redundant zeros."""
    integer = integer.lstrip("0")
    fraction = fraction.rstrip("0")
    return (integer or "0") + ("." + fraction if fraction else "")


def _exact(value: Fraction) -> str:
    """`value` in full: a numeral, or a ratio when no numeral holds it.

    A fraction terminates in base ten exactly when its denominator divides a
    power of ten, so `0.1` in hexadecimal is the numeral `0.0625` while `0.1`
    in base three is one third, which no numeral spells. Neither is rounded:
    the digits a converted numeral is worth are all of them.
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

"""The parser's public API: sources, declarations, spans and errors."""

from __future__ import annotations

import pytest
from sexpr import to_sexpr

from rederive.syntax import (
    ClearWhat,
    DeriveSyntaxError,
    DomainDeclaration,
    FunctionDeclaration,
    InputMode,
    Kind,
    ParseState,
    SettingDeclaration,
    Source,
    VariableDeclaration,
    VariableInfo,
    parse_expression,
    parse_source,
)


def parse(text, state=None):
    return parse_expression(text, state or ParseState())


def sexpr(text, state=None):
    return to_sexpr(parse(text, state).node)


# -- source preprocessing ---------------------------------------------------


def test_line_endings_are_normalised():
    source = Source.from_file("1+2\r\n3+4\r5+6\n")
    assert source.text == "1+2\n3+4\n5+6\n"


def test_comments_are_stripped():
    source = Source.from_file("1+2 ; a comment\n3+4\n")
    assert source.text == "1+2 \n3+4\n"


def test_a_semicolon_in_a_string_is_data():
    source = Source.from_file('"a ; b" + 1 ; but this goes\n')
    assert source.text == '"a ; b" + 1 \n'


def test_a_semicolon_is_a_syntax_error_on_the_author_line():
    # `;` is a file-level construct only.
    with pytest.raises(DeriveSyntaxError):
        parse("1+2 ; a comment")


def test_continuation_lines_are_spliced():
    source = Source.from_file("SUM(ELEME~\nNT(u,n+1-k))\n")
    assert source.text == "SUM(ELEMENT(u,n+1-k))\n"


def test_a_splice_may_cut_a_string_literal_in_half():
    source = Source.from_file('"over [a,b] with n ~\nsubdivisions."\n')
    assert source.text == '"over [a,b] with n subdivisions."\n'


def test_comments_are_stripped_before_splicing():
    # The `~` inside the comment is gone before splicing looks for one, so
    # both lines load.
    source = Source.from_file("; a comment ~\n1+2\n")
    assert source.text == "\n1+2\n"


def test_offsets_map_back_to_the_original_text():
    source = Source.from_file("; heading\nx + ~\ny\n")
    offset = source.text.index("y")
    assert source.locate(offset) == (3, 1)


def test_the_author_line_has_an_identity_offset_map():
    source = Source.from_line("x + y")
    assert source.text == source.original
    assert source.locate(4) == (1, 5)


# -- whole sources ----------------------------------------------------------


def test_parse_source_yields_one_result_per_expression():
    source = Source.from_file("1+2\n\n; nothing here\n3+4\n")
    results = list(parse_source(source, ParseState()))
    assert [to_sexpr(result.node) for result in results] == ["(+ 1 2)", "(+ 3 4)"]


def test_parse_source_offsets_index_the_whole_source():
    source = Source.from_file("1+2\n3+4\n")
    second = list(parse_source(source, ParseState()))[1]
    assert source.text[second.node.start : second.node.end] == "3+4"


def test_a_setting_takes_effect_once_the_caller_applies_it():
    source = Source.from_file("InputMode:=Word\nxyz\n")
    state = ParseState()
    written = []
    for result in parse_source(source, state):
        for declaration in result.declarations:
            state.declare(declaration)
        written.append(to_sexpr(result.node))
    assert written == ["(:= InputMode Word)", "xyz"]
    assert state.input_mode is InputMode.WORD


def test_word_mode_takes_a_keyword_operator_inside_a_name():
    # In Character mode `xSUBy` is a subscript; in Word mode the run is one
    # name, folded to lower case because nothing has declared it.
    assert sexpr("xSUBy") == "(sub x y)"
    assert sexpr("xSUBy", ParseState(input_mode=InputMode.WORD)) == "xsuby"


def test_word_mode_takes_a_run_whole_over_a_name_declared_inside_it():
    # A declared name matching the front of a run is what Character mode is
    # for. In Word mode the run is the name: `xk` is `xk` however long `x` has
    # been declared, which is what SOLVE.MTH's iteration variable needs.
    state = ParseState()
    state.declare(VariableDeclaration("x", has_value=True))
    assert sexpr("xk", state) == "(juxt x k)"
    state.input_mode = InputMode.WORD
    assert sexpr("xk", state) == "xk"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("erfi(1)", "(call ERFI 1)"),
        ("ei(x)", "(call EI x)"),
        ("si(x)", "(call SI x)"),
        ("ci(x)", "(call CI x)"),
        ("digamma(x)", "(call DIGAMMA x)"),
        ("polygamma(1, x)", "(call POLYGAMMA 1 x)"),
        ("polylog(2, x)", "(call POLYLOG 2 x)"),
        ("lambertw(1)", "(call LAMBERTW 1)"),
        ("sinh_int(x)", "(call SINH_INT x)"),
    ],
    ids=str,
)
def test_a_special_function_of_rederives_own_is_typed_in_any_case(text, expected):
    # Character mode, where a run of letters nothing names is a product of them:
    # `erfi(1)` would be `ERF(i)*1` and `si(x)` would be `s*i*x`.
    assert sexpr(text) == expected


def test_a_special_function_of_rederives_own_is_found_inside_a_word():
    # Longest match takes these names in front of the letters after them, as it
    # takes `ERF` in `erfx`.
    assert sexpr("erfix") == "(apply ERFI x)"
    assert sexpr("cit") == "(apply CI t)"
    assert sexpr("xerfi(1)") == "(juxt x (call ERFI 1))"
    assert sexpr("velocity") == "(juxt v e l o (apply CI t) y)"


def test_a_word_reads_around_every_known_function_name_as_derive_reads_it():
    # Derive's rule for every function it knows, its own or one the author defines:
    # with `CI(x) :=` and `EI(x) :=` defined there, Derive 4.11 reads each of these
    # as below, and these names are built in here.
    assert sexpr("height") == "(juxt h (apply EI g) h t)"
    assert sexpr("xci(t)") == "(juxt x (call CI t))"
    assert sexpr("velocity") == "(juxt v e l o (apply CI t) y)"


@pytest.mark.parametrize(
    ("text", "mode", "expected"),
    [
        ("FOO(x)", InputMode.CHARACTER, "(juxt f o o x)"),
        ("foo(x)", InputMode.CHARACTER, "(juxt f o o x)"),
        ("FOO(x)", InputMode.WORD, "(juxt foo x)"),
        ("foo(x)", InputMode.WORD, "(juxt foo x)"),
    ],
    ids=str,
)
def test_case_does_not_make_an_unknown_name_a_function(text, mode, expected):
    # Upper case is how a function name is printed and not what makes a name one: in
    # Derive 4.11 `FOO(x)` reads as `f*o*o*x` in Character mode, as `foo(x)` does,
    # until `FOO` is defined.
    assert sexpr(text, ParseState(input_mode=mode)) == expected


def test_an_unknown_name_with_several_arguments_is_no_call():
    with pytest.raises(DeriveSyntaxError):
        sexpr("FOO(x, y)")


def test_a_setting_the_caller_ignores_changes_nothing():
    source = Source.from_file("InputMode:=Word\nxyz\n")
    state = ParseState()
    written = [to_sexpr(result.node) for result in parse_source(source, state)]
    assert written == ["(:= InputMode Word)", "(juxt x y z)"]


# -- declarations -----------------------------------------------------------


def test_an_assignment_declares_a_variable():
    result = parse("tera:=10^12")
    assert result.declarations == (VariableDeclaration("tera", has_value=True),)


def test_an_empty_right_hand_side_declares_a_variable_without_a_value():
    declarations = parse("meter:=").declarations
    assert declarations == (VariableDeclaration("meter", has_value=False),)


def test_a_definition_declares_the_function_and_not_its_parameters():
    # The parameters are known names while the body is read and no longer, so
    # what the line declares is the function alone.
    result = parse("LINE(x,y,m,t):=y+m*(t-x)")
    assert result.declarations[-1] == FunctionDeclaration(
        "LINE", ("x", "y", "m", "t"), has_body=True
    )
    assert not any(
        isinstance(declaration, VariableDeclaration)
        for declaration in result.declarations
    )


def test_a_domain_declaration_is_reported():
    assert parse("x :epsilon Real").declarations == (DomainDeclaration("x", "Real"),)


def test_a_setting_is_reported_rather_than_applied():
    state = ParseState()
    assert parse("InputBase:=Hexadecimal", state).declarations == (
        SettingDeclaration("InputBase", "Hexadecimal"),
    )
    assert state.input_base == 10


def test_a_setting_value_stands_where_its_function_could_not():
    """`NORMAL` and `EXPAND` are built-in functions and setting values too.

    The original takes all of these and rejects a bare `SIN`.
    """
    assert parse("DisplayFormat := Normal").declarations == (
        SettingDeclaration("DisplayFormat", "Normal"),
    )
    assert sexpr("Trigonometry := Expand") == "(:= Trigonometry Expand)"
    # However it was spelled, and wherever a value can stand.
    assert parse("DisplayFormat := NORMAL").declarations == (
        SettingDeclaration("DisplayFormat", "Normal"),
    )
    assert sexpr("x := Normal") == "(:= x Normal)"
    assert sexpr("2 + Normal") == "(+ 2 Normal)"
    # Given an argument list it is the function again.
    assert sexpr("Normal(3)") == "(call NORMAL 3)"
    with pytest.raises(DeriveSyntaxError):
        parse("SIN")


@pytest.mark.parametrize("text", ["APPROX(x):=x+1", "sin(x):=x+1", "SUBS(u):="], ids=str)
def test_a_builtin_function_cannot_be_defined(text):
    # Derive refuses these at the `:=`, so a builtin always means itself.
    with pytest.raises(DeriveSyntaxError) as raised:
        parse(text)
    assert raised.value.offset == text.index(":=")


@pytest.mark.parametrize("text", ["ZETA(s,m):=s+m", "FLOOR(x):=x", "EI(x):=x"], ids=str)
def test_a_builtin_an_older_utility_file_defines_can_be_defined(text):
    # Derive 1.62's SPECIAL.MTH defines ZETA, 2.01's MISC.MTH defines FLOOR, and
    # EXP_INT.MTH defines EI, a function rederive adds.
    assert sexpr(text).startswith("(fundef ")


def test_the_parameters_of_a_definition_do_not_outlive_it():
    # `mx` has to be one name for the body of `FIT3(mx,mf):=mx+mf` to be lexed
    # at all, and is `m*x` again once the definition is read: a parameter is a
    # name the body uses and not one the session gained.
    state = ParseState()
    parse("FIT3(mx,mf):=mx+mf", state)
    assert sexpr("mx+mf", state) == "(+ (juxt m x) (juxt m f))"


def test_a_parameter_hands_back_the_variable_it_shadowed():
    # Value, domain and all: `tera` was the user's before the definition
    # borrowed the name, and is the user's afterwards.
    state = ParseState()
    parse("tera:=10^12", state)
    state.declare(VariableDeclaration("tera", has_value=True))
    state.declare(DomainDeclaration("tera", "Real"))
    parse("FIT3(tera):=tera^2", state)
    assert state.variables == {"tera": VariableInfo("tera", True, "Real")}
    assert sexpr("terab", state) == "(juxt tera b)"


def test_clearing_the_table_makes_a_name_split_again():
    state = ParseState()
    parse("tera:=10^12", state)
    state.declare(VariableDeclaration("tera", has_value=True))
    assert sexpr("terab", state) == "(juxt tera b)"
    state.clear(ClearWhat.VARIABLES)
    assert sexpr("terab", state) == "(juxt t e r a b)"


def test_lookup_answers_in_canonical_spelling():
    state = ParseState()
    assert state.lookup("sin") == "SIN"
    assert state.lookup("PI") == "pi"
    assert state.lookup("nowhere") is None
    assert state.is_function("cos") is True
    assert state.is_function("pi") is False


def test_a_declared_function_outranks_a_setting_keyword():
    # Derive's own utility files define `EXACT(...)`, so the keyword value
    # `Exact` cannot keep the name to itself.
    # Until then the keyword is what the name is, and a keyword is no function.
    state = ParseState()
    assert sexpr("EXACT(a)", state) == "(juxt Exact a)"
    parse("EXACT(p,q):=p+q", state)
    assert sexpr("EXACT(a,b)", state) == "(call EXACT a b)"


def test_declarations_from_nested_assignments_are_reported():
    result = parse("[a:=,b:=1]")
    assert result.declarations == (
        VariableDeclaration("a", has_value=False),
        VariableDeclaration("b", has_value=True),
    )


# -- the tree ---------------------------------------------------------------


def test_a_parenthesised_operand_excludes_its_parentheses():
    text = "x (x + 1)"
    node = parse(text).node
    assert (node.kind, text[node.start : node.end]) == (Kind.PRODUCT, "x (x + 1)")
    spans = [(child.kind, text[child.start : child.end]) for child in node.children]
    assert spans == [(Kind.NAME, "x"), (Kind.SUM, "x + 1")]


def test_juxtaposition_records_that_no_operator_was_written():
    # A product spells each of its gaps; juxtaposition is the one written
    # as nothing, and stands in the run as a space.
    juxtaposed = parse("x y").node
    explicit = parse("x*y").node
    assert (juxtaposed.value, juxtaposed.surface) == ("*", " ")
    assert (explicit.value, explicit.surface) == ("*", None)
    assert parse("2*3 4").node.surface == "* "


def test_a_greek_variable_has_one_name_however_it_is_spelled():
    # Derive held Greek letters in the IBM PC character set, whose Greek run
    # decodes to `ß`, `Θ`, `Φ`, `Ω` and the micro sign. Text that came through
    # that decoding has to name the same variables the Greek letters name.
    for spellings in [
        ("beta", "BETA", "β", "ß"),
        ("theta", "θ", "Θ"),
        ("phi", "φ", "Φ"),
        ("omega", "ω", "Ω"),
        ("mu", "μ", "µ"),
    ]:
        canonical = {sexpr(spelling) for spelling in spellings}
        assert len(canonical) == 1, f"{spellings} gave {canonical}"


def test_a_greek_letter_with_no_alias_is_an_ordinary_name():
    # Only the ten letters Derive pre-defines have a spelled-out form.
    assert sexpr("λ") == "λ"
    assert sexpr("lambda") == "(juxt l a m b d a)"


def test_a_glyph_operator_keeps_the_spelling_it_was_written_with():
    node = parse("a ≤ b").node
    assert (node.kind, node.value, node.surface) == (Kind.REL, "<=", "≤")


def test_a_relation_chain_nests_to_the_left():
    # `(1<=a)<=b`: two operands per node, one operator each. Transfer Save
    # drops the parentheses of `(a=b)<c` and keeps those of `a=(b<c)`.
    node = parse("1<=a<=b").node
    assert node.kind is Kind.REL
    assert node.value == "<="
    left, right = node.children
    assert (left.kind, left.value) == (Kind.REL, "<=")
    assert [child.value for child in left.children] == ["1", "a"]
    assert right.value == "b"


def test_authored_expressions_stay_inert():
    assert sexpr("2+3") == "(+ 2 3)"


# -- the derivative prime ----------------------------------------------------
#
# `BB'(r)` is what the original prints for the derivative of a function nobody
# defined, and what it reads back. The marks are part of the name, so the
# grammar above knows nothing about them: what follows one is a call.


def primed(text):
    """`text` parsed with `BB` declared as a function of one argument and `CC` as one
    of two."""
    state = ParseState(input_mode=InputMode.WORD)
    state.declare(FunctionDeclaration("BB", ("u",), has_body=False))
    state.declare(FunctionDeclaration("CC", ("u", "v"), has_body=False))
    return sexpr(text, state)


def test_a_prime_on_an_applied_function_is_part_of_its_name():
    assert primed("BB'(r)") == "(call BB' r)"
    assert primed("BB''(r)") == "(call BB'' r)"
    assert primed("BB'(r^2)") == "(call BB' (^ r 2))"
    assert primed("CC'(u, v)") == "(call CC' u v)"


def test_a_prime_binds_no_tighter_than_the_call_it_is_part_of():
    # An atom, so the product and the power around it read as they always did.
    assert primed("2BB'(r)") == "(juxt 2 (call BB' r))"
    assert primed("BB'(r)^2") == "(^ (call BB' r) 2)"


@pytest.mark.parametrize(
    "text", ["BB'", "BB' + 1", "BB '(r)", "b'(r)", "DD'(r)", "'(2 + 3)"]
)
def test_an_apostrophe_that_derives_nothing_is_the_error_it_always_was(text):
    """Three things make a prime, and anything else is no notation at all.

    The marks have to touch the name, there has to be an argument list after
    them, since a derivative is a function and not a value, and what carries
    them has to be a name that could be applied at all. `b` is a letter here
    and a letter is no function, and `DD` is a name nothing defines.
    """
    with pytest.raises(DeriveSyntaxError):
        primed(text)


# -- errors -----------------------------------------------------------------


def test_an_error_reports_where_the_parser_stopped():
    with pytest.raises(DeriveSyntaxError) as caught:
        parse("(x+1")
    assert caught.value.offset == 4
    assert str(caught.value) == "Syntax error detected at cursor"


def test_an_error_carries_its_source():
    with pytest.raises(DeriveSyntaxError) as caught:
        parse("2 +")
    assert caught.value.source is not None
    assert caught.value.source.locate(caught.value.offset) == (1, 4)


def test_an_error_in_a_file_locates_in_the_original_text():
    source = Source.from_file("1+2\n3 +\n")
    results = parse_source(source, ParseState())
    next(results)
    with pytest.raises(DeriveSyntaxError) as caught:
        next(results)
    assert source.locate(caught.value.offset) == (2, 4)

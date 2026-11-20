"""The parser's public API: sources, declarations, spans and errors."""

from __future__ import annotations

import pytest
from sexpr import to_sexpr

from rederive.syntax import (
    CaseMode,
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


def test_lines_are_spliced_before_comments_are_stripped():
    # The splice comes first, so a `;` in a string split by a wrap is still data.
    source = Source.from_file('"a, ~\nb; c" ; comment\n')
    assert source.text == '"a, b; c" \n'


def test_an_annotation_line_ending_in_a_tilde_takes_the_next_line_with_it():
    # Derive never writes one; its own reader would join them the same way.
    source = Source.from_file("; a comment ~\n1+2\n3\n")
    assert source.text == "\n3\n"
    assert source.annotations == {1: "a comment 1+2"}


def test_an_annotation_belongs_to_the_expression_line_after_it():
    source = Source.from_file(";Simp(#1)\n\n;Approx(#1)\nx + ~\ny\n\nz\n")
    assert source.text == "\n\n\nx + y\n\nz\n"
    assert source.annotations == {3: "Approx(#1)"}


def test_a_semicolon_in_square_brackets_is_no_comment():
    source = Source.from_file("[a, b; c, d] ; rows\n")
    assert source.text == "[a, b; c, d] \n"


def test_escapes_are_expanded_after_lines_are_joined():
    source = Source.from_file("\x7f03~\nb8 ; \x7f03b1\n")
    assert source.text == "θ \n"


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
    # In Character mode `xSUBy` is a subscript; in Word mode it is one undeclared
    # name, folded to lower case.
    assert sexpr("xSUBy") == "(sub x y)"
    assert sexpr("xSUBy", ParseState(input_mode=InputMode.WORD)) == "xsuby"


def test_word_mode_takes_a_run_whole_over_a_name_declared_inside_it():
    # Only Character mode splits off a declared name at the start of a run. In Word
    # mode `xk` stays `xk` even with `x` declared, as SOLVE.MTH's iteration variable
    # needs.
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
    # In Character mode an unknown run of letters is a product, so without these
    # names `erfi(1)` would be `ERF(i)*1` and `si(x)` would be `s*i*x`.
    assert sexpr(text) == expected


def test_a_special_function_of_rederives_own_is_found_inside_a_word():
    # Longest match picks these names out of a word, as it does `ERF` in `erfx`.
    assert sexpr("erfix") == "(apply ERFI x)"
    assert sexpr("cit") == "(apply CI t)"
    assert sexpr("xerfi(1)") == "(juxt x (call ERFI 1))"
    assert sexpr("velocity") == "(juxt v e l o (apply CI t) y)"


def test_a_word_reads_around_every_known_function_name_as_derive_reads_it():
    # Derive applies this to every known function, built in or user-defined: with
    # `CI(x) :=` and `EI(x) :=` defined, Derive 4.11 reads these as below. Here the
    # names are built in.
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
    # Upper case doesn't make a name a function: until `FOO` is defined, Derive 4.11
    # reads both `FOO(x)` and `foo(x)` as `f*o*o*x` in Character mode.
    assert sexpr(text, ParseState(input_mode=mode)) == expected


def test_an_unknown_name_with_no_argument_or_several_is_a_call():
    # It can't be a product, and the call belongs to the last name of the run.
    assert sexpr("foo(a,b)") == "(juxt f o (call o a b))"
    word = ParseState(input_mode=InputMode.WORD)
    assert sexpr("foo(a,b)", word) == "(call foo a b)"
    assert sexpr("foo()", word) == "(call foo)"
    assert sexpr("Plot_Thing(a, 1+1)", word) == "(call Plot_Thing a (+ 1 1))"
    # The call declares nothing.
    assert parse("foo(a,b)", word).declarations == ()
    assert "foo" not in word.functions


def test_a_function_named_only_by_folding_case_is_a_variable_where_it_cant_apply():
    # Recorded worksheets iterate over `n` beside a function `N`, and multiply `fre`
    # beside a function `FRE`: where nothing follows to apply to, the spelling is a
    # variable of its own. Where it can apply, it is the function.
    state = ParseState(input_mode=InputMode.WORD)
    state.declare(FunctionDeclaration("N", ("x",), True))
    state.declare(FunctionDeclaration("FRE", ("l",), True))
    assert sexpr("ITERATES(N(n), n, 1, 25)", state) == "(call ITERATES (call N n) n 1 25)"
    assert sexpr("fr_ := fre*liste", state) == "(:= fr_ (* fre liste))"
    assert sexpr("fre(x)", state) == "(call FRE x)"


def test_a_variable_assigned_is_known_to_the_rest_of_the_line():
    # Undeclared, `S` would read as the variable `s` in the second assignment.
    word = ParseState(input_mode=InputMode.WORD)
    assert sexpr("[S := 2299160, s := S]", word) == "(vec (:= S 2299160) (:= s S))"
    # In Character mode a name assigned reads whole after its assignment.
    assert sexpr("[w1 := 2, w1^2]") == "(vec (:= w1 2) (^ w1 2))"


def test_an_unknown_name_before_a_parenthesis_in_a_file_is_a_call():
    # Derive writes every product with `*`, so a name directly before `(` is a function
    # defined later in the file, or never. It is declared, so later lines agree.
    state = ParseState(input_mode=InputMode.WORD, reading_file=True)
    result = parse("G(x):=H(x+1)+2", state)
    assert to_sexpr(result.node) == "(fundef G (params x) (+ (call H (+ x 1)) 2))"
    assert FunctionDeclaration("H", ("_0",)) in result.declarations
    assert sexpr("K(x):=NOSUCH(x)+1", state) == (
        "(fundef K (params x) (+ (call NOSUCH x) 1))"
    )
    # A blank between them, or a name known as a variable, still multiplies.
    assert sexpr("foo (x)", state) == "(juxt foo x)"
    state.declare(VariableDeclaration("a", has_value=True))
    assert sexpr("a(b+c)", state) == "(juxt a (+ b c))"


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


def test_a_definition_reports_its_new_parameters_as_variables():
    # The function is declared, and each parameter that wasn't a known name is reported
    # as a variable with no value, for the session to apply when the line is entered.
    result = parse("LINE(x,y,m,t):=y+m*(t-x)")
    assert result.declarations[-1] == FunctionDeclaration(
        "LINE", ("x", "y", "m", "t"), has_body=True
    )
    variables = {
        declaration
        for declaration in result.declarations
        if isinstance(declaration, VariableDeclaration)
    }
    assert variables == {VariableDeclaration(name) for name in "xymt"}


def test_a_parameter_that_was_known_is_not_reported():
    # A valued variable keeps its value, and a function its meaning.
    state = ParseState()
    state.declare(VariableDeclaration("v", has_value=True))
    state.declare(FunctionDeclaration("A", ("t",)))
    result = parse("FF(v,a):=v+a", state)
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
    """`NORMAL` and `EXPAND` are both built-in functions and setting values.

    The original accepts all of these and rejects a bare `SIN`.
    """
    assert parse("DisplayFormat := Normal").declarations == (
        SettingDeclaration("DisplayFormat", "Normal"),
    )
    assert sexpr("Trigonometry := Expand") == "(:= Trigonometry Expand)"
    # In any case, and anywhere a value can appear.
    assert parse("DisplayFormat := NORMAL").declarations == (
        SettingDeclaration("DisplayFormat", "Normal"),
    )
    assert sexpr("x := Normal") == "(:= x Normal)"
    assert sexpr("2 + Normal") == "(+ 2 Normal)"
    # With an argument list it is the function.
    assert sexpr("Normal(3)") == "(call NORMAL 3)"
    with pytest.raises(DeriveSyntaxError):
        parse("SIN")


@pytest.mark.parametrize(
    "text",
    [
        "APPROX(x):=x+1",
        "SIN(x) := x + 1",
        "SUBS(u):=",
        "FIRST(v) := v SUB 1",
        "SORT(v) := v",
        "DIMENSION(v) := 7",
        "PRIME(n) := n",
        "EI(x, m) := x + m",
        "SI(z) := z",
        "POLYGAMMA(n, z) := n + z",
        "LAMBERTW(x) := x",
        "DIGAMMA(z) := z",
        "CEILING(a, d) := -FLOOR(-a, d)",
        "ROUND(a, d) := FLOOR(a/d + 1/2)",
        "INVERSE_MOD(a, m) := 0",
        "RANK(A) := 0",
        "POLYLOG(n, z, m) := SUM(z^k/k^n, k, 1, m)",
        "Real(x) := x",
        "Angle(x) := x",
    ],
    ids=str,
)
def test_a_builtin_function_cannot_be_defined(text):
    # Derive rejects these at the `:=`, so a builtin can't be redefined, nor an old
    # name it still accepts, nor a value word or a setting.
    with pytest.raises(DeriveSyntaxError) as raised:
        parse(text)
    assert raised.value.offset == text.index(":=")


@pytest.mark.parametrize(
    "text",
    [
        "ZETA(s,m):=s+m",
        "FLOOR(x):=x",
        "MOD(a,b):=a",
        "DELETE_ELEMENT(v,n):=v",
        "REVERSE_VECTOR(v):=v",
        "VAR(x,n,xvar):=x+n+xvar",
    ],
    ids=str,
)
def test_a_builtin_an_older_utility_file_defines_can_be_defined(text):
    # Derive 1.62's SPECIAL.MTH defines ZETA, 2.01's MISC.MTH defines FLOOR, and a 6.10
    # worksheet its own VAR.
    assert sexpr(text).startswith("(fundef ")


@pytest.mark.parametrize(
    "text",
    [
        "sin(x):=x+1",
        "perm(v,n):=v+n",
        "ei(x):=x+1",
        "Power?(x):=x",
        "EXACT(p,q):=p+q",
        "HEX(a):=a+1",
    ],
    ids=str,
)
def test_another_spelling_of_a_reserved_name_is_free(text):
    assert sexpr(text).startswith("(fundef ")


@pytest.mark.parametrize(
    "text",
    [
        "pi:=3",
        "PI:=3",
        "ROW:=3",
        "Real:=3",
        "exit:=1",
        "SIN:=3",
        "FIRST:=3",
        "pi:epsilon Real",
        "Angle:epsilon Real",
    ],
    ids=str,
)
def test_a_reserved_name_cannot_be_bound(text):
    with pytest.raises(DeriveSyntaxError) as raised:
        parse(text)
    assert raised.value.offset == text.index(":")


def test_a_set_operator_an_older_utility_file_defines_becomes_a_function():
    # Files from before sets were built in define `UNION(u, v)` on vectors.
    state = ParseState()
    assert sexpr("{1} UNION {2}", state) == "(union (set 1) (set 2))"
    for declaration in parse_expression("UNION(u,v):=APPEND(u,v)", state).declarations:
        state.declare(declaration)
    assert sexpr("UNION([1],[2])", state) == "(call UNION (vec 1) (vec 2))"


def test_parsing_a_definition_leaves_its_parameters_undeclared():
    # `mx` is one name inside the body of `FIT3(mx,mf):=mx+mf`, but reverts to `m*x`
    # afterwards: parsing alone declares nothing. The session declares the parameters
    # when the line is entered.
    state = ParseState()
    parse("FIT3(mx,mf):=mx+mf", state)
    assert sexpr("mx+mf", state) == "(+ (juxt m x) (juxt m f))"


def test_a_parameter_named_like_a_built_in_function_is_an_error():
    with pytest.raises(DeriveSyntaxError):
        parse("FF(INT):=INT+1")
    with pytest.raises(DeriveSyntaxError):
        parse("F(x, COS):=x")


def test_a_parameter_spelled_otherwise_hides_the_built_in_in_the_body():
    state = ParseState()
    result = parse("FF(int):=int+1", state)
    assert to_sexpr(result.node) == "(fundef FF (params int) (+ int 1))"
    for declaration in result.declarations:
        state.declare(declaration)
    # Afterwards `int` is still the function's other spelling, not a variable.
    assert sexpr("int(x,x)", state) == "(call INT x x)"


def test_a_parameter_hands_back_the_variable_it_shadowed():
    # The user's `tera`, with its value and domain, is restored after the definition
    # uses the name as a parameter.
    state = ParseState()
    parse("tera:=10^12", state)
    state.declare(VariableDeclaration("tera", has_value=True))
    state.declare(DomainDeclaration("tera", "Real"))
    parse("FIT3(tera):=tera^2", state)
    assert state.variables == {"tera": VariableInfo("tera", True, "Real")}
    assert sexpr("terab", state) == "(juxt tera b)"


def test_a_definition_that_fails_to_parse_leaves_its_name_undeclared():
    # The name is declared while the body is parsed, so the body can recurse, but a
    # body that doesn't parse must not leave it behind: a later call would then name a
    # function that was never defined.
    state = ParseState()
    with pytest.raises(DeriveSyntaxError):
        parse("BIS_AUX(u,x,a,b):=u+", state)
    assert "BIS_AUX" not in state.functions
    state.declare(SettingDeclaration("InputMode", "Word"))
    assert sexpr("BIS_AUX(u)", state) == "(juxt bis_aux u)"


def test_a_definition_that_fails_to_parse_keeps_the_earlier_one():
    state = ParseState()
    parse("f(a,b):=a+b", state)
    state.declare(FunctionDeclaration("f", ("a", "b"), has_body=True))
    with pytest.raises(DeriveSyntaxError):
        parse("f(a):=a+", state)
    assert sexpr("f(1,2)", state) == "(call f 1 2)"


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


def test_a_function_outranks_a_setting_keyword():
    # Derive's utility files define `EXACT(...)`, which the library pre-defines, so the
    # name before `(` is the function, not the keyword `Exact`; so is a user's
    # definition of it. Another spelling, or the name without `(`, is the keyword.
    state = ParseState()
    assert sexpr("EXACT(a)", state) == "(call EXACT a)"
    assert sexpr("Exact(a)", state) == "(juxt Exact a)"
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
    # A product records each operator as written; juxtaposition is recorded as a
    # space.
    juxtaposed = parse("x y").node
    explicit = parse("x*y").node
    assert (juxtaposed.value, juxtaposed.surface) == ("*", " ")
    assert (explicit.value, explicit.surface) == ("*", None)
    assert parse("2*3 4").node.surface == "* "


def test_a_greek_variable_has_one_name_however_it_is_spelled():
    # Derive stored Greek letters in the IBM PC character set, some of which decode to
    # `ß`, `Θ`, `Φ`, `Ω` and the micro sign. Those must name the same variables as the
    # Greek letters.
    for spellings in [
        ("beta", "β", "ß"),
        ("theta", "θ", "Θ"),
        ("phi", "φ", "Φ"),
        ("omega", "ω", "Ω"),
        ("mu", "μ", "µ"),
    ]:
        canonical = {sexpr(spelling) for spelling in spellings}
        assert len(canonical) == 1, f"{spellings} gave {canonical}"


def test_a_greek_letter_is_named_in_lower_case_or_capitalized():
    # The case of a Greek name is respected in either case mode.
    assert sexpr("λ") == "λ"
    assert sexpr("lambda+Lambda") == "(+ λ Λ)"
    assert sexpr("LAMBDA") == "(juxt l a m b d a)"
    assert sexpr("BETA") == "(juxt b e t a)"
    # So `gamma` is the letter, and `GAMMA` the function.
    assert sexpr("gamma+GAMMA(x)") == "(+ γ (call GAMMA x))"
    # `pi` is the constant in any spelling.
    assert sexpr("Pi") == "pi"


def test_a_greek_name_binds_its_letter():
    state = ParseState()
    result = parse("gamma(k):=k+1", state)
    assert to_sexpr(result.node) == "(fundef γ (params k) (+ k 1))"
    for declaration in result.declarations:
        state.declare(declaration)
    assert sexpr("gamma(2)", state) == "(call γ 2)"


def test_a_letter_standing_for_a_constant_becomes_the_name_a_definition_gives_it():
    state = ParseState()
    assert sexpr("π(1)", state) == "(juxt pi 1)"
    for declaration in parse("π(a):=a+1", state).declarations:
        state.declare(declaration)
    assert sexpr("π(1)", state) == "(call π 1)"
    assert sexpr("pi", state) == "pi"


def test_a_glyph_operator_keeps_the_spelling_it_was_written_with():
    node = parse("a ≤ b").node
    assert (node.kind, node.value, node.surface) == (Kind.REL, "<=", "≤")


def test_a_relation_chain_nests_to_the_left():
    # `(1<=a)<=b`: two operands and one operator per node. Transfer Save accordingly
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
# The original prints the derivative of an undefined function as `BB'(r)` and reads it
# back. The marks are part of the name, so the grammar needs no special rule: what
# follows is an ordinary call.


def primed(text):
    """Parse `text` with `BB` declared as a function of one argument and `CC` of two."""
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
    # It's an atom, so surrounding products and powers parse as usual.
    assert primed("2BB'(r)") == "(juxt 2 (call BB' r))"
    assert primed("BB'(r)^2") == "(^ (call BB' r) 2)"


def test_a_prime_on_an_undeclared_name_is_the_derivative_of_an_undefined_function():
    assert primed("DD'(r)") == "(call dd' r)"
    assert primed("u'(t)+w''(t)") == "(+ (call u' t) (call w'' t))"
    assert sexpr("(1-x^2)*y'(x)") == "(* (- 1 (^ x 2)) (call y' x))"
    # In Character mode an undeclared name is one letter, marks or not.
    assert sexpr("ay'(x)") == "(juxt a (call y' x))"


@pytest.mark.parametrize(
    "text", ["BB'", "BB' + 1", "b '(r)", "x'", "(a+b)'", "'-x", "'"]
)
def test_an_apostrophe_that_neither_derives_nor_quotes_is_an_error(text):
    """A prime requires a name it touches and an argument list after it; a quote, an
    operand after it that starts with no sign."""
    with pytest.raises(DeriveSyntaxError):
        primed(text)


# -- the quote ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("'(2+3)", "(quote (+ 2 3))"),
        ("' (2+3)", "(quote (+ 2 3))"),
        ("TERMS('(x+x+x))", "(call TERMS (quote (+ x x x)))"),
        ("'x", "(quote x)"),
        ("''(1+2)", "(quote (quote (+ 1 2)))"),
        # The quote takes one primary: what follows applies to the quote.
        ("'p4^2", "(^ (quote p4) 2)"),
        ("'n_*r-SKAL(n_,b_)", "(- (* (quote n_) r) (call SKAL n_ b_))"),
        ("[a='(q_*b)+r_]", "(vec (rel a = (+ (quote (* q_ b)) r_)))"),
        ("'v SUB 1", "(sub (quote v) 1)"),
        ("RETURN 'InvLaplace(f,s,t)", "(call RETURN (quote (call InvLaplace f s t)))"),
        ("['IF(x>=0,g),f SUB 2]", "(vec (quote (call IF (rel x >= 0) g)) (sub f 2))"),
        ("s_:='(s^2+b*s+c=0)", "(:= s_ (quote (rel (+ (^ s 2) (* b s) c) = 0)))"),
        (
            "list:=APPEND([['(equ=0)]],list)",
            "(:= list (call APPEND (vec (vec (quote (rel equ = 0)))) list))",
        ),
        ('"it\'s"', '"it\'s"'),
    ],
)
def test_an_apostrophe_where_an_operand_starts_is_the_quote(text, expected):
    assert sexpr(text, ParseState(input_mode=InputMode.WORD)) == expected


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


# -- names ---------------------------------------------------------------------


def _read(lines, **modes):
    """The s-expression of the last of `lines`, each entered in turn."""
    state = ParseState(**modes)
    for line in lines[:-1]:
        for declaration in parse(line, state).declarations:
            state.declare(declaration)
    return sexpr(lines[-1], state)


_WORD = {"input_mode": InputMode.WORD}


@pytest.mark.parametrize(
    ("lines", "modes", "expected"),
    [
        # A name may end in `?` when that spelling is known, or in Word mode; otherwise
        # `?` is the unknown value.
        (["PRIME?(7)"], {}, "(call PRIME? 7)"),
        (["x?"], {}, "(juxt x ?)"),
        (["x ?"], {}, "(juxt x ?)"),
        (["IF(a>0,1,?)"], {}, "(call IF (rel a > 0) 1 ?)"),
        (["lyap?+1"], _WORD, "(+ lyap? 1)"),
        (["EVEN?(n)"], _WORD, "(call EVEN? n)"),
        (["Natural?(n):=n>=0", "ok?:=3", "Natural?(ok?)"], {}, "(call Natural? ok?)"),
        # A letter and its underscores, and `@` with its digits, are one name.
        (["2x_+@1+@"], {}, "(+ (juxt 2 x_) @1 @)"),
        # Every character from U+0080 on that is not reserved is a letter.
        (["m₀+1"], {}, "(+ (juxt m ₀) 1)"),
        (["m₀+1"], _WORD, "(+ m₀ 1)"),
        (["Lotfußpunkt*2"], _WORD, "(* lotfußpunkt 2)"),
        (["[l:=■,f:=‧]", "l+f"], {}, "(+ l f)"),
        # Character mode cuts a run into the longest known names, else single letters.
        (["xyz"], {}, "(juxt x y z)"),
        (["x1+a2"], {}, "(+ (juxt x 1) (juxt a 2))"),
        (["xsinx"], {}, "(juxt x (apply SIN x))"),
        # Derive 6's new keywords are found only as a whole run.
        (["diff+arrow"], {}, "(+ (apply DIF f) (juxt a r r o w))"),
        (["x1:=", "velocity:=3", "x1+2velocity"], {}, "(+ x1 (juxt 2 velocity))"),
        (["F(base,counter):=base+counter", "base counter"], {}, "(juxt base counter)"),
        (["velocity time"], _WORD, "(juxt velocity time)"),
        (["x1e3"], _WORD, "x1e3"),
        # No exponent notation.
        (["1.5e3"], {}, "(juxt 1.5 e 3)"),
        (["1.5e3"], _WORD, "(juxt 1.5 e3)"),
        (["2(3+5)"], {}, "(juxt 2 (+ 3 5))"),
        # A backslash and a semicolon in a string are data.
        (['"a\\b"+"c;d"'], {}, '(+ "a\\b" "c;d")'),
        ([""""\""""], {}, '""'),
        # `"+-"` is the sign before an operand, else a string.
        (['["+-"inf,"+-"]'], {}, '(vec (pm inf) "+-")'),
        (['x="+-"SQRT(7)+2'], {}, "(rel x = (+ (pm (call SQRT 7)) 2))"),
        # Line ends are blanks, so an expression may be entered over several lines.
        (
            ['IF(x<0,"neg",\n   IF(x>0,"pos",\n   "zero")\n  )'],
            {},
            '(call IF (rel x < 0) "neg" (call IF (rel x > 0) "pos" "zero"))',
        ),
        # Names are case-distinct, and resolve to the exact spelling first.
        (["perm(v,n):=v+n", "[perm(1,2),PERM(5,2),Perm(5,2)]"], {}, (
            "(vec (call perm 1 2) (call PERM 5 2) (call PERM 5 2))"
        )),
        (["H(s):=2*s", "h(s):=s+5", "[H(1),h(1)]"], {}, "(vec (call H 1) (call h 1))"),
        (["Step:=1/40", "Step+STEP(3)"], {}, "(+ Step (call STEP 3))"),
        (["rE:=[1,2,3]", "rE SUB 2"], {}, "(sub rE 2)"),
        (["C:=5", "c:=7", "C-c"], {}, "(- C c)"),
        (["foo:=3", "FOO+Foo"], {}, "(+ foo foo)"),
        (["A+a"], {}, "(+ a a)"),
        (["sin(0)+Sin(0)+SIN(0)"], {}, (
            "(+ (call SIN 0) (call SIN 0) (call SIN 0))"
        )),
        (["first([a,b,c])"], {}, "(call FIRST (vec a b c))"),
        (
            ["r+R"],
            {"input_mode": InputMode.WORD, "case_mode": CaseMode.SENSITIVE},
            "(+ r R)",
        ),
        (
            ["Sin(x)"],
            {"input_mode": InputMode.WORD, "case_mode": CaseMode.SENSITIVE},
            "(juxt Sin x)",
        ),
        # A setting line in another case spelling, when the value is the setting's.
        (["inputmode:=word"], {}, "(:= InputMode Word)"),
        (["inputmode:=word", "velocity time"], {}, "(juxt velocity time)"),
        (["angle:=30", "angle+1"], {}, "(+ angle 1)"),
        # Old names Derive 6 accepts: some are read as the new name, some kept.
        (["DIMENSION(v)"], {}, "(call DIM v)"),
        (["dim(n):=DIMENSION(n)"], {}, "(fundef dim (params n) (call DIM n))"),
        (["NUMBER(n)"], {}, "(call NUMBER n)"),
        # Derive 6's built-ins are calls even where Rederive does not evaluate them.
        (["PROG(x+1)"], _WORD, "(call PROG (+ x 1))"),
        (["f(x):=x", "f()"], {}, "(call f)"),
    ],
    ids=lambda value: repr(value) if isinstance(value, list) else "",
)
def test_a_name_reads_as_derive_6_reads_it(lines, modes, expected):
    assert _read(lines, **modes) == expected


@pytest.mark.parametrize("text", ["1..5", '"a\nb"', "a+b ; note"], ids=repr)
def test_these_are_refused(text):
    with pytest.raises(DeriveSyntaxError):
        parse(text)


def test_a_setting_line_in_another_case_spelling_needs_a_name_no_user_has():
    state = ParseState()
    state.declare(VariableDeclaration("inputmode", has_value=True))
    assert sexpr("inputmode:=word", state) == "(:= inputmode Word)"
    sensitive = ParseState(case_mode=CaseMode.SENSITIVE)
    assert sexpr("inputmode:=Word", sensitive) == "(:= inputmode Word)"


def test_an_empty_definition_gives_a_yielding_name_back_to_the_built_in():
    state = ParseState()
    for line in ("ZETA(s):=s+1", "ZETA(s):="):
        for declaration in parse(line, state).declarations:
            state.declare(declaration)
    assert "ZETA" not in state.functions
    assert state.is_function("ZETA")


# -- operators and statements of Derive 6 ---------------------------------------


@pytest.mark.parametrize(
    ("lines", "modes", "expected"),
    [
        # Precedence, loosest first: `IFF`, `IMP`, `XOR`, `OR`, `AND`, `NOT`; relations
        # and `SUBSET`; the set operators; sums; `DOTPRODUCT`; `CROSSPRODUCT`; signs.
        (['p OR q IMP r IFF s XOR t'], {}, '(iff (imp (or p q) r) (xor s t))'),
        (['p IFF q IFF r'], {}, '(iff (iff p q) r)'),
        (['NOT p OR q AND r'], {}, '(or (not p) (and q r))'),
        (['r` UNION s INTERSECTION t'], {}, '(union (transpose r) (intersection s t))'),
        (['a\\b\\c'], {}, '(setminus (setminus a b) c)'),
        (['a UNION b\\c'], {}, '(union a (setminus b c))'),
        (['u INTERSECTION v=u'], {}, '(rel (intersection u v) = u)'),
        (['a UNION b SUBSET c'], {}, '(subset (union a b) c)'),
        (['a-b DOTPRODUCT c+d'], {}, '(sum a - (dotproduct b c) + d)'),
        (['a*b DOTPRODUCT c*d'], {}, '(dotproduct (* a b) (* c d))'),
        (['a DOTPRODUCT b CROSSPRODUCT c'], {}, '(dotproduct a (crossproduct b c))'),
        (['a CROSSPRODUCT b CROSSPRODUCT c'], {}, '(crossproduct (crossproduct a b) c)'),
        (['-a CROSSPRODUCT b'], {}, '(crossproduct (neg a) b)'),
        # Glyphs of the operators.
        (['A ∪ B ∩ C'], _WORD, '(union a (intersection b c))'),
        (['p ↔ q'], {}, '(iff p q)'),
        (['u ⊆ v'], {}, '(subset u v)'),
        (['u • v'], {}, '(dotproduct u v)'),
        (['u ⨯ v'], {}, '(crossproduct u v)'),
        # Word operators. Another spelling is the operator only where an operator can
        # stand, and only if no name has that spelling.
        (['m ROW 2'], _WORD, '(row m 2)'),
        (['m COL [1,...,3]'], _WORD, '(col m (vec 1 ... 3))'),
        (['a SUB [2,3] SUB SUB [3,...,7]'], _WORD, (
            '(subsub (sub a (vec 2 3)) (vec 3 ... 7))'
        )),
        (['MQ SUB  SUB i'], _WORD, '(subsub mq i)'),
        (['MQ↓↓i'], _WORD, '(subsub mq i)'),
        (['B SUB SUB SUB k'], _WORD, '(subsubsub b k)'),
        (['p iff q'], _WORD, '(iff p q)'),
        (['m row 2'], {}, '(row m 2)'),
        (['m row'], {}, '(juxt m r o w)'),
        (['row+1'], _WORD, '(+ row 1)'),
        (['row:=5', 'm ROW row'], _WORD, '(row m row)'),
        (['row:=5', 'm row 2'], _WORD, '(juxt m row 2)'),
        # `DEG` is postfix after an operand, and the constant `deg` elsewhere.
        (['a/1 DEG'], {}, '(/ a (degrees 1))'),
        (['n*t DEG/2'], {}, '(/ (* n (degrees t)) 2)'),
        (['(-135) DEG'], {}, '(degrees (neg 135))'),
        (['100°'], {}, '(degrees 100)'),
        (['37.875*deg'], {}, '(* 37.875 deg)'),
        (['x deg'], {}, '(degrees x)'),
        (['DEG'], {}, 'deg'),
        # Subscript links and postfix operators apply left to right, tighter than `^`.
        (['k SUB i!'], {}, '(fact (sub k i))'),
        (['a` SUB 1'], {}, '(sub (transpose a) 1)'),
        (['m SUB 1` SUB 1'], {}, '(sub (transpose (sub m 1)) 1)'),
        (['hp` ROW [2,...,DIM(hp`)]'], _WORD, (
            '(row (transpose hp) (vec 2 ... (call DIM (transpose hp))))'
        )),
        (['R(u):=', '(R(u)) COL [2]` SUB 1'], _WORD, (
            '(sub (transpose (col (call R u) (vec 2))) 1)'
        )),
        (['v SUB 1^v SUB 2'], {}, '(^ (sub v 1) (sub v 2))'),
        (['x^v SUB 2'], {}, '(^ x (sub v 2))'),
        (['nodes SUB DIM(nodes) SUB 1'], _WORD, '(sub (sub nodes (call DIM nodes)) 1)'),
        (['v SUB -1'], {}, '(sub v (neg 1))'),
        (['F(x):=[x,x^2]', 'F(x) SUB 1'], {}, '(sub (call F x) 1)'),
        (['m!^3'], {}, '(^ (fact m) 3)'),
        (['9%/12'], {}, '(/ (pct 9) 12)'),
        (['"Derive" SUB [1,2,6,3]'], {}, '(sub "Derive" (vec 1 2 6 3))'),
        (['123456789 SUB [1,3,5,-3,-1]'], {}, (
            '(sub 123456789 (vec 1 3 5 (neg 3) (neg 1)))'
        )),
        # `RETURN` takes everything up to the next comma or closing bracket; `exit` is a
        # keyword in any case.
        (['IF(v=[], RETURN w)'], {}, '(call IF (rel v = (vec)) (call RETURN w))'),
        (['RETURN n/NN(a,b)'], _WORD, '(call RETURN (/ n (call NN a b)))'),
        (['(o:=s) AND RETURN [true,c]'], {}, '(and (:= o s) (call RETURN (vec true c)))'),
        (['RETURN "+-"SQRT(u)'], {}, '(call RETURN (pm (call SQRT u)))'),
        (['RETURN(a)+1'], {}, '(call RETURN (+ a 1))'),
        (['IF(i>n,exit)'], {}, '(call IF (rel i > n) exit)'),
        (['IF(i>n,EXIT)'], _WORD, '(call IF (rel i > n) exit)'),
        (['(s:=ADJOIN(a,s)) AND exit'], _WORD, '(and (:= s (call ADJOIN a s)) exit)'),
        # Assignments and updates; a name with subscripts is a target.
        (['a:=1+b:=2'], {}, '(:= a (+ 1 (:= b 2)))'),
        (['(m:=c SUB 3) AND c:=d SUB 4'], {}, '(and (:= m (sub c 3)) (:= c (sub d 4)))'),
        (['IF(1>j:=j-1,exit)'], {}, '(call IF (rel 1 > (:= j (- j 1))) exit)'),
        (['counts SUB 2 SUB k:=counts SUB 2 SUB k+1'], _WORD, (
            '(:= (sub (sub counts 2) k) (+ (sub (sub counts 2) k) 1))'
        )),
        (['counts SUB 2 SUB k:=1'], {}, '(:= (sub (sub counts 2) k) 1)'),
        (['syl SUB i SUB (j+i-1):=cp1 SUB j'], _WORD, (
            '(:= (sub (sub syl i) (sum j + i - 1)) (sub cp1 j))'
        )),
        (['a1 SUB [1,3]:=[r1,r2]'], _WORD, '(:= (sub a1 (vec 1 3)) (vec r1 r2))'),
        (['b SUB j:+t*b SUB i'], {}, '(:+ (sub b j) (* t (sub b i)))'),
        (['k:+1'], {}, '(:+ k 1)'),
        (['t_:*(n-k_+1)/n'], {}, '(:* t_ (/ (sum n - k_ + 1) n))'),
        (['[n:/2,k_:-1]'], {}, '(vec (:/ n 2) (:- k_ 1))'),
        (['x:==3'], {}, '(:= x 3)'),
        (['[steps:=,stn:=]'], _WORD, '(vec (:= steps) (:= stn))'),
        (['a SUB SUB j:=0'], {}, '(:= (subsub a j) 0)'),
        # `;` separates the rows of a matrix; domains.
        (['[a, b; c, d]'], {}, '(vec (vec a b) (vec c d))'),
        (['[a; b; c]'], {}, '(vec (vec a) (vec b) (vec c))'),
        (['[2,3,5;7,1,4;4,6,8] SUB 3 SUB 2'], {}, (
            '(sub (sub (vec (vec 2 3 5) (vec 7 1 4) (vec 4 6 8)) 3) 2)'
        )),
        (['[a, b; c]'], {}, '(vec (vec a b) (vec c))'),
        (['[1,...,3; 4,5,6]'], {}, '(vec (vec 1 ... 3) (vec 4 5 6))'),
        (['x^[0,...,n-1]'], {}, '(^ x (vec 0 ... (- n 1)))'),
        (['x :epsilon Vector'], {}, '(domain x Vector)'),
        (['x :epsilon Logical'], {}, '(domain x Logical)'),
        (['s :epsilon Real(1,inf)'], {}, '(domain s Real (ivl "(" 1 inf ")"))'),
        (['RS:epsilonVect'], _WORD, '(domain RS ?)'),
        # Definition heads: defaults, the variadic head `F v :=` where a statement can
        # stand, and no parameters.
        (['SUMSQ(x, y := 0, z := 0) := x^2+y^2+z^2'], _WORD, (
            '(fundef SUMSQ (params x (:= y 0) (:= z 0)) (+ (^ x 2) (^ y 2) (^ z 2)))'
        )),
        (['FOO(x, y := 3, z) :='], _WORD, '(fundef FOO (params x (:= y 3) z))'),
        (['pslq(x,ds:=1,d:=PrecisionDigits,n):=x'], _WORD, (
            '(fundef pslq (params x (:= ds 1) (:= d PrecisionDigits) n) x)'
        )),
        (['f(x, step := 0.1, end) := x'], _WORD, (
            '(fundef f (params x (:= step 0.1) end) x)'
        )),
        (['P1(x, d := 2*x) := x + d'], {}, (
            '(fundef P1 (params x (:= d (* 2 x))) (+ x d))'
        )),
        (['FOO v := SUM(v)'], _WORD, '(fundef FOO (variadic v) (call SUM v))'),
        (['[a := 1, G w := w SUB 1]'], _WORD, (
            '(vec (:= a 1) (fundef G (variadic w) (sub w 1)))'
        )),
        (['PROG(Block commands := commands, 1)'], _WORD, (
            '(call PROG (fundef Block (variadic commands) commands) 1)'
        )),
        (['x y := 5'], {}, '(fundef x (variadic y) 5)'),
        (['SIN x := 3'], {}, '(apply SIN (:= x 3))'),
        (['F() := 5'], {}, '(fundef F (params) 5)'),
        (['IRR(cf, ppy := 1, i, "IRR"):=1'], _WORD, (
            '(fundef IRR (params cf (:= ppy 1) i "IRR") 1)'
        )),
        # Plus-or-minus between two terms adds the plus-or-minus of the second.
        (['2*s_=n"+-"1'], _WORD, '(rel (* 2 s_) = (+ n (pm 1)))'),
        (['a ± b - c'], {}, '(sum a + (pm b) - c)'),
    ],
    ids=lambda value: repr(value) if isinstance(value, list) else "",
)
def test_an_operator_reads_as_derive_6_reads_it(lines, modes, expected):
    assert _read(lines, **modes) == expected


@pytest.mark.parametrize(
    "text",
    [
        "IF(c, RETURN)",
        "RETURN",
        "x:+",
        "[x:-, 1]",
        "[a, b;]",
        "[;a]",
        "[a;;b]",
        "{a; b}",
        "(a; b)",
        "F(x, y := ) := x",
        "F(SIN) := 1",
        "F(x, y:=1",
    ],
    ids=repr,
)
def test_these_forms_are_refused(text):
    with pytest.raises(DeriveSyntaxError):
        parse(text)


def test_a_definition_with_defaults_declares_its_parameter_names():
    result = parse("f(x, n := 0, tmp) := x + n")
    declared = [
        declaration
        for declaration in result.declarations
        if isinstance(declaration, FunctionDeclaration)
    ]
    assert declared[-1] == FunctionDeclaration("f", ("x", "n", "tmp"), has_body=True)
    assert VariableDeclaration("tmp") in result.declarations


def test_a_variadic_head_declares_a_variadic_function():
    state = ParseState(input_mode=InputMode.WORD)
    result = parse("FOO v := SUM(v)", state)
    assert result.declarations[-1] == FunctionDeclaration(
        "FOO", ("v",), has_body=True, variadic=True
    )
    for declaration in result.declarations:
        state.declare(declaration)
    assert sexpr("FOO(2, 3, 5)", state) == "(call FOO 2 3 5)"


def test_an_update_declares_nothing():
    # It changes a variable that has a value already.
    assert not parse("k :+ 1").declarations
    assert not parse("a SUB i := 3").declarations


def test_a_word_operator_spelled_otherwise_is_a_name_in_operand_position():
    # Where no operator can stand, `col` is whatever the text reads as otherwise.
    assert sexpr("col+1") == "(+ (juxt c o l) 1)"
    word = ParseState(input_mode=InputMode.WORD)
    assert sexpr("[col, iff]", word) == "(vec col iff)"
    sensitive = ParseState(input_mode=InputMode.WORD, case_mode=CaseMode.SENSITIVE)
    assert sexpr("m row 2", sensitive) == "(juxt m row 2)"

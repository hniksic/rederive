"""Load coverage: worksheets and math files of the kinds Derive 5 and 6 users wrote,
each holding the constructs that a published file of its kind holds.

A worksheet is built byte by byte (`dfwsheet.Sheet`), a version 5 one in that
version's font (`derive5`); a math file is written as Derive writes one. The state
blocks and expressions are those of the files, trimmed to what the checks need. Each
test loads one and checks what must hold afterwards: the entries, their annotations,
the definitions, values, domains and settings, and the load report.
"""

import pytest
from dfwsheet import OLD, Sheet, derive5, rtf

from rederive.engine.client import DomainKind
from rederive.model.session import Session
from rederive.syntax import write_expression

#: Derive 6's settings, saved at the end of every version 6 state block.
SETTINGS = {
    "Precision": "Exact",
    "PrecisionDigits": "10",
    "Notation": "Rational",
    "NotationDigits": "10",
    "Branch": "Principal",
    "Exponential": "Auto",
    "Logarithm": "Auto",
    "Trigonometry": "Auto",
    "Trigpower": "Auto",
    "Angle": "Radian",
    "CaseMode": "Insensitive",
    "VariableOrder": "[x,y,z]",
    "OutputBase": "Decimal",
    "InputBase": "Decimal",
    "InputMode": "Character",
    "DisplayFormat": "Normal",
    "TimesOperator": "Dot",
    "DisplaySteps": "false",
}


def settings(**changed):
    """The setting lines of a version 6 state block, with `changed` values."""
    return [f"{name}:={changed.get(name, value)}" for name, value in SETTINGS.items()]


@pytest.fixture(autouse=True)
def here(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def session():
    return Session()


def texts(session):
    return [entry.text for entry in session.entries]


def annotations(session):
    return [entry.annotation for entry in session.entries]


def body(session, name):
    """The body of function `name` as the writer writes it."""
    return write_expression(session.functions[name].body)


def math_file(path, *lines, end=b"\r\n"):
    """A math file of physical `lines`, CR LF ended, in UTF-8 unless given as bytes."""
    path.write_bytes(
        b"".join(
            (line if isinstance(line, bytes) else line.encode()) + b"\r\n"
            for line in lines
        )
        + (b"" if end == b"\r\n" else end)
    )
    return path


def names_in(node):
    """The names in a tree."""
    found = {str(node.value)} if node.kind.name == "NAME" else set()
    for child in node.children:
        found |= names_in(child)
    return found


async def simplified(session, text):
    return (await session.simplify(text)).text


# -- version 5 worksheets ----------------------------------------------------------


def test_hom_test_the_smallest_worksheet(session, tmp_path):
    before = dict(session.settings.values())
    sheet = Sheet(OLD)
    sheet.expression("(y*#e^(y/x)-x)/(x*#e^(y/x))")
    sheet.expression(
        "LIM(LIM((y*#e^(y/x)-x)/(x*#e^(y/x)),x,k*x),y,k*y)=y/x-#e^(-y/x)",
        "User=Simp(User)",
    )
    sheet.expression(
        "(y*#e^(y/x)-x)/(x*#e^(y/x))-(y/x-#e^(-y/x))=0", "User=Simp(User)"
    )
    assert not session.load(sheet.write(tmp_path / "hom_test.dfw"))
    assert annotations(session) == ["User"] * 3
    assert dict(session.settings.values()) == before


async def test_benno_german_annotations_and_a_string(session, tmp_path):
    state = [b'v(i):=IF(i = 1, ["*"], IF(i = 11, ["III"], APPEND(["*"], v(i - 1))))']
    sheet = Sheet(OLD, state=state)
    sheet.expression(
        'v(i):=IF(i=1,["*"],IF(i=11,["III"],APPEND(["*"],v(i-1))))', "Benutzer"
    )
    sheet.expression('"Simplify #3"')
    sheet.expression("VECTOR([v(i)],i,1,11)", "Benutzer")
    assert not session.load(sheet.write(tmp_path / "benno.dfw"))
    assert annotations(session) == ["User"] * 3
    assert await simplified(session, "v(2)") == '["*", "*"]'


async def test_sumfact_a_program_saved_with_blank_separated_products(session, tmp_path):
    state = [
        derive5(
            "sumfactpow(u, k, x := 1, n := 0, s := 0, a):=LOOP(IF(u = 0, RETURN ê^x s),"
            " a := SUBST(u, k, n), u := QUOTIENT(u, k - n, k), s :+ a x^n, n :+ 1)"
        )
    ]
    sheet = Sheet(OLD, state=state)
    sheet.expression(
        "sumfactpow(u,k,x:=1,n:=0,s:=0,a):=LOOP(IF(u=0,RETURN(s*EXP(x))),"
        "a:=SUBST(u,k,n),u:=QUOTIENT(u,k-n,k),s:+a*x^n,n:+1)"
    )
    sheet.expression(
        "(sumfactpow(k^2+1,k,EXP(#i*x))-sumfactpow(k^2+1,k,EXP(-#i*x)))/(2*#i)"
    )
    sheet.expression(
        "#e^COS(x)*(COS(SIN(x))*(SIN(2*x)+SIN(x))+SIN(SIN(x))*(COS(2*x)+COS(x)+1))",
        "Simp(#2)",
    )
    assert not session.load(sheet.write(tmp_path / "sumfact.dfw"))
    assert "RETURN(#e^x*s)" in body(session, "sumfactpow")
    assert "s:+a*x^n" in body(session, "sumfactpow")
    assert await simplified(session, "sumfactpow(k^2 + 1, k, y)") == (
        "(y^2 + y + 1)*#e^y"
    )


def test_plots_text_objects_and_where_the_cross_was(session, tmp_path):
    state = [
        "hCross:=APPROX(6269230769230769/500000000000000)",
        "vCross:=APPROX(4230769230769231/12500000000000000)",
    ]
    sheet = Sheet(OLD, state=state)
    sheet.text(rtf('Settings #1 are frozen, one double click will "defrost" it:'))
    sheet.text(rtf("Now I need other settings (scaling, region, ....)"))
    sheet.text(
        rtf(
            "I save this file as a dfw-file and opening this file in the future I "
            "don´t need to take care for the previous settings"
        )
    )
    sheet.plot()
    report = session.load(sheet.write(tmp_path / "plots.dfw"))
    assert str(report) == "1 plot not loaded"
    assert len(session.entries) == 3
    assert texts(session)[0] == (
        "\"Settings #1 are frozen, one double click will 'defrost' it:\""
    )
    assert not session.assignments and not session.state.variables


async def test_net5k_v5_repeated_declarations_and_exact_decimals(session, tmp_path):
    state = [
        "apv(u, v):=APPEND(u, v)",
        "fc(m):=COS(4·m·π/5)",
        "fs(m):=SIN(4·m·π/5)",
        "n5(x, y, l, k, m, n):=IF(n < 0, [], APPEND([[x, y]], n5(x + l·fc(m), y + "
        "l·fs(m), k·l, k, m + 1, n - 1), n5(x + 190983/1000000·l·fc(m) - "
        "117557/200000·l·fs(m), y, k·l, k, m, n - 1)))",
        "hCross:=APPROX(39615384615384617/10000000000000000)",
        *["k:="] * 3,
        *["l:="] * 3,
        *["m:="] * 4,
        *["n:="] * 3,
        "vCross:=APPROX(10576923076923077/2500000000000000)",
        *["x:="] * 3,
        *["y:="] * 3,
    ]
    sheet = Sheet(OLD, state=[derive5(line) for line in state])
    sheet.expression("[fc(m):=COS(4*m*pi/5),fs(m):=SIN(4*m*pi/5)]")
    sheet.expression(
        "n5(x,y,l,k,m,n):=IF(n<0,[],APPEND([[x,y]],n5(x+l*fc(m),y+l*fs(m),k*l,k,m+1,"
        "n-1),n5(x+0.190983*l*fc(m)-0.587785*l*fs(m),y,k*l,k,m,n-1)))"
    )
    sheet.expression("n5(-2,-0.5,4,0.25,0,1)")
    assert not session.load(sheet.write(tmp_path / "net5k_v5.dfw"))
    assert await simplified(session, "fc(1)") == "-SQRT(5)/4 - 1/4"
    assert "190983/1000000" in body(session, "n5")


async def test_write_derive_greek_names_and_string_named_variables(session, tmp_path):
    state = [
        derive5("write(text):=VECTOR(σ(text↓k, 3·(k - 1)), k, 1, DIM(text))"),
        b"\xba(\xbf, \xad):=[\xad + \xbf\x991\x991, - \xbf\x991\x992]",
        derive5(
            "σ(λ, ε):=[ε + λ↓1↓1, λ↓1↓2; ε + λ↓2↓1, λ↓2↓2; ε + λ↓3↓1, λ↓3↓2; "
            "ε + λ↓4↓1, λ↓4↓2]"
        ),
        derive5('"A":=[τ, 3·τ; 13·τ/10 + 7/20, 1; τ + 1, 3 - 3·τ; 0, 0]'),
        derive5('"E":=[0, 3·τ; 2·τ, 0; 2·τ, 3/2; 2·τ, 3]'),
        derive5('"O":=[1 + COS(2·π·τ), 3/2 + 3/2·SIN(2·π·τ); 1, 0; 1, 0; 1, 0]'),
        b"hCross:=APPROX(5571428571428571/500000000000000)",
        b'text:="TICALC"',
        derive5("ε:="),
        derive5("λ:="),
    ]
    sheet = Sheet(OLD, state=state)
    sheet.expression(
        "sigma(lambda,epsilon):=[[epsilon+lambda SUB 1 SUB 1,lambda SUB 1 SUB 2],"
        "[epsilon+lambda SUB 2 SUB 1,lambda SUB 2 SUB 2],[epsilon+lambda SUB 3 SUB 1,"
        "lambda SUB 3 SUB 2],[epsilon+lambda SUB 4 SUB 1,lambda SUB 4 SUB 2]]"
    )
    sheet.expression('["E":=[[0,3*tau],[2*tau,0],[2*tau,3/2],[2*tau,3]]]')
    sheet.expression("write(text):=VECTOR(sigma(text SUB k,3*(k-1)),k,1,DIM(text))")
    sheet.expression('write("TICALC")')
    sheet.expression('"A"*[[COS(pi/4),SIN(pi/4)],[-SIN(pi/4),COS(pi/4)]]')
    sheet.embedded()
    report = session.load(sheet.write(tmp_path / "write_derive.dfw"))
    assert str(report) == "1 embedded object not loaded"
    assert {"write", "σ", "ρ"} <= set(session.functions)
    assert {'"A"', '"E"', '"O"'} <= set(session.assignments)
    assert session.entries[3].text == 'write("TICALC")'
    assert await simplified(session, "#4") != '"TICALC"'
    assert session.take_notice() is None


async def test_partfrac_a_declaration_only_the_state_has(session, tmp_path):
    state = [
        "x:\xad Real (0, \x96)".encode("latin-1"),
        derive5("list:=[1/(2·(x - 1)), - 1/(2·(x + 1))]"),
        b"res_:=",
        b"s1_:=",
    ]
    sheet = Sheet(OLD, state=state)
    sheet.expression("1/(1+x^8)")
    sheet.expression("INT(1/(1+x^8),x)", "Int(#1,x)")
    sheet.expression("fee([1,2,3,4])=1+2+3+4", "User=Simp(User)")
    assert not session.load(sheet.write(tmp_path / "partfrac.dfw"))
    assert annotations(session) == ["User", "Int(#1,x)", "User"]
    domain = session.domains["x"]
    assert domain.kind is DomainKind.REAL
    assert write_expression(domain.low) == "0" and not domain.closed_low
    assert await simplified(session, "SQRT(x^2)") == "x"


async def test_partfrac_a_definition_with_a_quote(session, tmp_path):
    state = [
        b"fee(list, s1_, res_):=PROG(res_ := FIRST(list), list := REST(list), "
        b"LOOP(IF(list = [], RETURN res_), s1_ := FIRST(list), res_ := '(res_ + s1_), "
        b"list := REST(list)))"
    ]
    sheet = Sheet(OLD, state=state)
    assert not session.load(sheet.write(tmp_path / "partfrac.dfw"))
    assert await simplified(session, "fee([1, 2, 3, 4])") == "1 + 2 + 3 + 4"


async def test_beispiel2_rows_in_the_state_and_a_localized_solve(session, tmp_path):
    state = [
        b"HESSE(xo, yo):=LIM(LIM([DIF(DIF(f(x, y), x), x), DIF(DIF(f(x, y), x), y); "
        b"DIF(DIF(f(x, y), y), x), DIF(DIF(f(x, y), y), y)], x, xo), y, yo)",
        b"PARTDIFF(xo, yo):=LIM(LIM(GRAD(f(x, y)), y, yo), x, xo)",
        derive5("f(x, y):=(x^2 - a^2)·(y^2 - b^2)"),
        b"xo:=",
        b"yo:=",
    ]
    sheet = Sheet(OLD, state=state)
    sheet.expression("f(x,y):=(x^2-a^2)*(y^2-b^2)", "Benutzer")
    sheet.expression(
        "PARTDIFF(xo,yo)=[2*xo*(yo^2-b^2),2*yo*(xo^2-a^2),0]",
        "Benutzer=Simp(Benutzer)",
    )
    sheet.expression("SOLVE(2*xo*(yo^2-b^2),[xo,yo],Real)", derive5("Löse(#5',[xo,yo])"))
    sheet.expression("xo=0 OR yo=-b OR yo=b", derive5("Simp(Löse(#5',[xo,yo]))"))
    sheet.expression("HESSE(0,0)=[[-2*b^2,0],[0,-2*a^2]]", "Benutzer=Simp(Benutzer)")
    assert not session.load(sheet.write(tmp_path / "beispiel2.dfw"))
    assert annotations(session) == [
        "User",
        "User",
        "Löse(#5',[xo,yo])",
        "Simp(Löse(#5',[xo,yo]))",
        "User",
    ]
    assert await simplified(session, "HESSE(0, 0)") == "[[-2*b^2, 0], [0, -2*a^2]]"


#: The string-named variables of a compiler's state block.
STRING_NAMES = "by1 by2 by3 from1 from2 start1 to1 to2 to3"


async def test_example_a_worksheet_named_as_a_math_file(session, tmp_path):
    state = [
        b"f(t):=2 SIN(3 t) - 4 COS(8 t) + 5 SIN(15 t) + 3",
        *[f'"${name}":='.encode() for name in STRING_NAMES.split()],
        derive5("data:=VECTOR(f(2 π/64 k), k, 0, 63)"),
        b"m:=n/2",
        b"n:=DIM(data)",
    ]
    sheet = Sheet(OLD, state=state)
    sheet.expression("f(t):=2*SIN(3*t)-4*COS(8*t)+5*SIN(15*t)+3")
    sheet.expression("data:=VECTOR(f(2*pi/64*k),k,0,63)")
    sheet.expression('LOAD("C:\\DfD\\DNL\\Mth00\\compiler\\Fastf.mth")')
    sheet.expression('FFT(a,b,m,inv,k,tr,"$start1","$by3"):=PROG(n:=2^m,n)')
    folder = tmp_path / "compiler"
    folder.mkdir()
    path = sheet.write(folder / "example.mth")
    assert not session.load(path)
    assert session.file is None
    assert len(session.entries) == 4
    assert body(session, "f") == "2*SIN(3*t)-4*COS(8*t)+5*SIN(15*t)+3"
    assert write_expression(session.assignments["data"]) == "VECTOR(f(2*pi/64*k),k,0,63)"
    assert "FASTF" not in session.functions
    math_file(folder / "Fastf.mth", "FASTF(x):=x")
    assert await simplified(session, "#3") == "true"
    assert "FASTF" in session.functions


# -- version 6 worksheets ----------------------------------------------------------


def test_spheres_a_minimal_version_6_worksheet(session, tmp_path):
    sheet = Sheet(state=settings())
    sheet.expression(
        "PRODUCT(PRODUCT(PRODUCT((x-2*a)^2+(y-2*b)^2+(z-2*c)^2-1,a,-1,1),b,-1,1),"
        "c,-1,1)=0"
    )
    sheet.expression("(x^2+y^2+z^2-1)*(x^2+y^2+z^2+4*z+3)", "Simp(#1)")
    assert not session.load(sheet.write(tmp_path / "spheres.dfw"))
    assert texts(session)[1] == "(x^2+y^2+z^2-1)*(x^2+y^2+z^2+4*z+3)"
    assert annotations(session) == ["User", "Simp(#1)"]
    assert session.settings["CaseMode"] == "Insensitive"


def test_dug9_setting_objects_and_case_sensitive_names(session, tmp_path):
    sheet = Sheet(state=settings(CaseMode="Sensitive", InputMode="Word"))
    sheet.expression("CaseMode:=Sensitive")
    sheet.expression("InputMode:=Word")
    sheet.text(rtf("Solve for p1q2, p1q3, p2q1, p2q3, p3q1 and p3q2."))
    sheet.expression("[(p1-p1q2-p1q3)/Rp1q1+p2q1/Rp2q1+p3q1/Rp3q1+q1=0]")
    assert not session.load(sheet.write(tmp_path / "DUG9.dfw"))
    assert session.settings["CaseMode"] == "Sensitive"
    assert session.settings["InputMode"] == "Word"
    assert texts(session)[3] == "[(p1-p1q2-p1q3)/Rp1q1+p2q1/Rp2q1+p3q1/Rp3q1+q1=0]"
    assert {"Rp1q1", "p1q2", "q1"} <= names_in(session.entries[3].node)


def test_scheuer2_a_solve_pair_and_boolean_results(session, tmp_path):
    sheet = Sheet(state=settings(CaseMode="Sensitive", InputMode="Word"))
    sheet.expression("[CaseMode:=Sensitive,InputMode:=Word]")
    sheet.expression("PL=(U0/(RI+RL))^2*RL")
    sheet.expression("SOLVE(PL=(U0/(RI+RL))^2*RL,RL)", "Solve(#2,RL)")
    sheet.expression(
        "RL=-SQRT(U0^2-4*PL*RI)*ABS(U0)/(2*PL)-(2*PL*RI-U0^2)/(2*PL) OR "
        "RL=SQRT(U0^2-4*PL*RI)*ABS(U0)/(2*PL)-(2*PL*RI-U0^2)/(2*PL)",
        "Simp(Solve(#2,RL))",
    )
    sheet.text(rtf("Set the discriminant = 0"))
    sheet.expression(
        "SOLVE(U0^2-4*PLmax*RI=0,PLmax)=(PLmax=U0^2/(4*RI))", "User=Simp(User)"
    )
    sheet.text(rtf("expr #3 = expr #6 and solve for RL:"))
    sheet.expression(
        "SOLVE((U0/(RI+RL))^2*RL=U0^2/(4*RI),RL)=(RL=RI OR U0=0)", "User=Simp(User)"
    )
    assert not session.load(sheet.write(tmp_path / "scheuer2.dfw"))
    assert len(session.entries) == 8
    assert annotations(session)[2:4] == ["Solve(#2,RL)", "Simp(Solve(#2,RL))"]
    strings = [entry.number for entry in session.entries if entry.text.startswith('"')]
    assert strings == [5, 7]
    found = names_in(session.entries[1].node) | names_in(session.entries[5].node)
    assert {"RL", "U0", "PLmax"} <= found
    assert all(entry.node.kind.name != "STRING" for entry in session.entries[:4])


def test_binormal_a_compressed_state(session, tmp_path):
    state = [
        "BINORMAL(u,v,r):=([VECTOR(1,z,1,16+1)]·VECTOR([z,1/√(2·\ue7c2)·\ue7c0^(-z^2/2),"
        "NORMAL((√r·z+u)/√(1-r))·NORMAL((√r·z+v)/√(1-r))·(1/√(2·\ue7c2))·"
        "\ue7c0^(-z^2/2)],z,-4,4,1/2))↓1↓3/2",
        *settings(DisplayFormat="Compressed"),
    ]
    sheet = Sheet(state=state)
    sheet.expression("1/(2*pi*SQRT(1-ρ^2))*#e^(-(x^2-2*x*y*ρ+y^2)/(2*(1-ρ^2)))")
    sheet.expression(
        "1/(2*pi*SQRT(1-(0.3)^2))*#e^(-(x^2-2*x*y*(0.3)+y^2)/(2*(1-(0.3)^2)))",
        "Sub(#1)",
    )
    sheet.expression("BINORMAL((38-40)/(2.5),(53-50)/(5.5),0.29)")
    sheet.expression("0.1769063132", "Approx(#3)")
    assert not session.load(sheet.write(tmp_path / "BINORMAL.dfw"))
    assert " SUB 1 SUB 3" in body(session, "BINORMAL")
    assert session.settings["DisplayFormat"] == "Compressed"
    assert annotations(session) == ["User", "Sub(#1)", "User", "Approx(#3)"]


async def test_polyfactor_an_asterisk_state_and_a_forward_annotation(session, tmp_path):
    state = [
        "m1:=[4096/(15752961*\ue7c2^4) + 1/4, 14336/(50625*\ue7c2^4); "
        "14336/(50625*\ue7c2^4), 50176/(1185921*\ue7c2^4) + 1/4]",
        *settings(TimesOperator="Asterisk"),
    ]
    sheet = Sheet(state=state)
    sheet.text(rtf("The polynomial"))
    sheet.expression("x^4-x^3+x^2*(62337182076360565082813671875*pi^8-1)", "Expd(#2)")
    sheet.expression("TimesOperator:=Asterisk")
    assert not session.load(sheet.write(tmp_path / "Polyfactor.dfw"))
    assert annotations(session)[1] == "Expd(#2)"
    assert await simplified(session, "[DIM(m1), DIM(m1 SUB 1)]") == "[2, 2]"
    assert session.settings["TimesOperator"] == "Asterisk"


async def test_mauve2_forward_references_and_decimal_settings(session, tmp_path):
    state = [
        "appr_area(r, n):=∑(area(r, n, k), k, 1, n)",
        "area(r, n, k):=2·radius(r, n, k)·r/n",
        "radius(r, n, k):=√(r^2 - (k·r/n)^2)",
        *settings(Precision="Approximate", Notation="Decimal"),
    ]
    sheet = Sheet(state=state)
    sheet.expression("radius(r,n,k):=SQRT(r^2-(k*r/n)^2)")
    sheet.expression("Precision:=Approximate")
    sheet.expression("Notation:=Decimal")
    sheet.expression(
        "[2*appr_area(1,10),2*appr_area(1,100),2*appr_area(1,1000),2*appr_area(1,10000)]"
    )
    sheet.expression("[2.904518326,3.120417031,3.139555466,3.141391477]", "Approx(#4)")
    assert not session.load(sheet.write(tmp_path / "mauve2.dfw"))
    assert session.settings["Precision"] == "Approximate"
    assert session.settings["Notation"] == "Decimal"
    assert await simplified(session, "2*appr_area(1, 10)") == "2.904518326"


async def test_desph_ge_3_a_domain_with_decimal_ends(session, tmp_path):
    state = [
        "d:epsilon Real (-0.5, 0.5)",
        *settings(Trigonometry="Expand", Trigpower="Sines"),
    ]
    sheet = Sheet(state=state)
    sheet.expression("ARCTAN(ABS(-1/2+d)^(-1)*(3+d*(-4)+d^2*(-4))^(1/2)*1/2)")
    sheet.expression("d:epsilonReal (- 1/2, 1/2)")
    sheet.expression("ARCTAN(SQRT(-4*d^2-4*d+3)/(1-2*d))", "Simp(#1')")
    assert not session.load(sheet.write(tmp_path / "desph_ge_3.dfw"))
    assert annotations(session)[2] == "Simp(#1')"
    assert session.entries[0].node.children[0].value == "ARCTAN"
    assert session.domains["d"].kind is DomainKind.REAL
    assert await simplified(session, "[SQRT((d - 1)^2), SQRT((d + 1)^2)]") == (
        "[1 - d, d + 1]"
    )


async def test_compl_int_values_only_the_state_has(session, tmp_path):
    state = ["f(x):=SIN(x)^n/(SIN(x)^n + COS(x)^n)", "a:=0", "b:=\ue7c2/2", *settings()]
    paragraphs = (
        "b",
        "∫ (F(x) + F(a + b - x)) dx",
        "b a",
        "∫ F(x) dx \ue7cc\ue7cd \ue7cc⎯⎯⎯⎯\ue7cb",
        "a 2",
    )
    sheet = Sheet(state=state)
    sheet.expression("INT(SIN(x)^n/(SIN(x)^n+COS(x)^n),x,0,pi/2)")
    sheet.expression("pi/4", "Simp(#1)")
    sheet.text(rtf(*paragraphs))
    sheet.expression("INT(1/2,x,0,pi/2)", "Step(#1)")
    sheet.expression("f(x):=SIN(x)^n/(SIN(x)^n+COS(x)^n)")
    sheet.expression("f(x)+f(0+pi/2-x)=1", "User=Simp(User)")
    assert not session.load(sheet.write(tmp_path / "compl_int.dfw"))
    assert await simplified(session, "[a, b]") == "[0, pi/2]"
    assert texts(session)[2:7] == [f'"{paragraph}"' for paragraph in paragraphs]
    assert annotations(session)[7] == "Step(#1)"


async def test_extremum_prob2_a_load_entered_and_sorted_definitions(session, tmp_path):
    state = [
        "DER_2ND(a, s, r, h):=SUBST(∂(F_ONE(a, s, r, h), r, 2), r, EXTR_ONE(a, s, r, h))",
        "EXTREM(goal, side, x, y, extr):=[x = EXTR_ONE(goal, side, x, y), y = "
        "EXTR_TWO(goal, side, x, y); extr = EXTRVAL(goal, side, x, y), EXTR_DEC(goal, "
        "side, x, y); F_ONE(goal, side, x, y), ∂(F_ONE(goal, side, x, y), x)]",
        "EXTRVAL(a, s, r, h):=SUBST(a, [r, h], [EXTR_ONE(a, s, r, h), EXTR_TWO(a, s, r, "
        "h)])",
        'EXTR_DEC(a, s, r, h):=IF(DER_2ND(a, s, r, h) < 0, "Maximum", IF(DER_2ND(a, s, '
        'r, h) > 0, "Minimum", "Point of Inflection"))',
        "EXTR_ONE(a, s, r, h):=ITERATE(∂(F_ONE(a, s, r, h), r), r, 0, -1)",
        "EXTR_TWO(a, s, r, h):=ITERATE(SUBST(s, r, EXTR_ONE(a, s, r, h)), h, 0, -1)",
        "F_ONE(a, s, r, h):=LIM(a, h, ITERATE(s, h, 0, -1))",
        "ar:=x·y - 9800",
        "extr:=",
        "goal:=",
        "side:=",
        "water:=(x + 4)·(y + 2) - x·y",
        *settings(),
    ]
    sheet = Sheet(state=state)
    sheet.expression('LOAD("D:\\DfD\\DNL\\DNL93\\MTH12\\MaxMin_ut.mth")')
    sheet.expression("[water:=(x+4)*(y+2)-x*y,ar:=x*y-9800]")
    sheet.expression("EXTREM(water,ar,x,y)")
    sheet.expression(
        '[[x=140,y=70],[extr=568,"Minimum"],[2*(x^2+4*x+19600)/x,2*(x^2-19600)/x^2]]',
        "Simp(#3)",
    )
    sheet.text(
        rtf(
            "The dimension of the field is 70m by 140m,",
            "the minimum area of the ditch is 568 m^2.",
        )
    )
    assert not session.load(sheet.write(tmp_path / "extremum_prob2.dfw"))
    names = "DER_2ND EXTREM EXTRVAL EXTR_DEC EXTR_ONE EXTR_TWO F_ONE".split()
    assert set(names) <= set(session.functions)
    assert await simplified(session, "#1") == (
        'LOAD("D:\\DfD\\DNL\\DNL93\\MTH12\\MaxMin_ut.mth")'
    )
    assert session.take_notice() == "MaxMin_ut.mth could not be found"
    assert set(names) <= set(session.functions)


async def test_variations_a_utility_file_in_the_state(session, tmp_path):
    state = [
        "ADJOIN_ELEMENT(e, v):=ADJOIN(e, v)",
        "APPEND_COLUMNS Args:=IF((DIM(Args) = 1 ∧ VECTOR?(FIRST(Args))) ∧ "
        "EVERY(MATRIX?(a_), a_, FIRST(Args)), APPEND(MAP_LIST(a_`, a_, FIRST(Args)))`, "
        "APPEND(MAP_LIST(a_`, a_, Args))`)",
        "KRONECKER(i, j):=IF(i = j, 1, 0)",
        "MATPROD(a, b, i, j):=PROG(b ≔ b COL j, a ≔ a↓i, a·b)",
        "POLAR_TO_RECTANGULAR(r, θ):=[r·COS(θ), r·SIN(θ)]",
        "vars_(k, b, k_, m_ ≔ 0, n_, s_ ≔ [], t_):=LOOP(IF(m_ = 2^k, RETURN "
        "REVERSE(s_)), k_ ≔ k, n_ ≔ m_, t_ ≔ [], LOOP(IF(k_ = 0, exit), t_ ≔ "
        "ADJOIN(b↓(MOD(n_, 2) + 1), t_), n_ ≔ FLOOR(n_, 2), k_ :- 1), DISPLAY(t_), "
        "s_ ≔ ADJOIN(t_, s_), m_ :+ 1)",
        "Args:=",
        "cylindrical:=[r, θ, z; 1, r, 1]",
        "i_:=[1, 0, 0]",
        "α:=",
        "θ:=",
        *settings(),
    ]
    sheet = Sheet(state=state)
    sheet.expression(
        "vars_(k,b,k_,m_:=0,n_,s_:=[],t_):=LOOP(IF(m_=2^k,RETURN(REVERSE(s_))),k_:=k,"
        "n_:=m_,t_:=[],LOOP(IF(k_=0,exit),t_:=ADJOIN(b SUB (MOD(n_,2)+1),t_),"
        "n_:=FLOOR(n_,2),k_:-1),DISPLAY(t_),s_:=ADJOIN(t_,s_),m_:+1)"
    )
    sheet.text(rtf(""))
    assert not session.load(sheet.write(tmp_path / "variations.dfw"))
    assert len(session.entries) == 1
    assert session.functions["APPEND_COLUMNS"].variadic
    assert await simplified(session, "KRONECKER(2, 2)") == "1"
    product = "MATPROD([[1, 2], [3, 4]], [[5, 6], [7, 8]], 1, 2)"
    assert await simplified(session, product) == "22"
    count = len(session.entries)
    assert await simplified(session, "vars_(1, [p, q])") == "[[p], [q]]"
    assert texts(session)[count:-1] == ['"[p]"', '"[q]"']


async def test_mypower_an_inline_domain_and_a_setting_in_a_body(session, tmp_path):
    state = [
        "EXACT_EIGENVECTOR(M, λ, r_, c_, trig_):=PROG(c_ ≔ M - "
        "λ·IDENTITY_MATRIX(DIM(M)), r_ ≔ c_, IF(r_ = [], PROG(trig_ ≔ Trigonometry, "
        "Trigonometry ≔ Collect, r_ ≔ c_, Trigonometry ≔ trig_)), r_)",
        "MINOR(M, i, j):=DELETE(DELETE(M, i)`, j)`",
        "SUBTRACT_ELEMENTS(v, i, j, s ≔ 1):=REPLACE(v↓i - s·v↓j, v, i)",
        "mypower(a, t_):=PROG(t_ ≔ APPEND(VECTOR(EXACT_EIGENVECTOR(a, e_)`, e_, "
        "EIGENVALUES(a)))`, n \ue7c8 Integer [0, ∞), a ≔ 1/t_·a·t_, t_·a^n/t)",
        "n:epsilon Integer [0, ∞)",
        "A:=[1/2, 1/2; - 1/2, 1/2]",
        "λ:=",
        "μ:=",
        *settings(),
    ]
    sheet = Sheet(state=state)
    sheet.expression(
        "mypower(a,t_):=PROG(t_:=APPEND(VECTOR(EXACT_EIGENVECTOR(a,e_)`,e_,"
        "EIGENVALUES(a)))`,n:epsilonInteger [0, inf),a:=t_^(-1)*a*t_,t_*a^n*t^(-1))"
    )
    assert not session.load(sheet.write(tmp_path / "mypower.dfw"))
    domain = session.domains["n"]
    assert domain.kind is DomainKind.INTEGER and domain.closed_low
    assert await simplified(session, "MINOR([[1, 2], [3, 4]], 1, 1)") == "[[4]]"
    assert await simplified(session, "SUBTRACT_ELEMENTS([5, 2], 1, 2)") == "[3, 2]"
    assert "mypower" in session.functions


async def test_horb_2_definitions_of_runge_kutta(session, tmp_path):
    state = [
        "RHS_(y, yp, et):=[yp, 2·(V(x) - et)·y]",
        "RK(r, v, v0, h, n):=RK_AUX0(h·APPEND([1], r), v, v0, n)",
        "RK_AUX0(p, v, v0, n):=ITERATES(u_ + RK_AUX1(p, v, u_, SUBST(p, v, u_)), u_, "
        "v0, n)",
        "RK_AUX1(p, v, u_, c1):=RK_AUX2(p, v, u_, c1, SUBST(p, v, u_ + c1/2))",
        "RK_AUX2(p, v, u_, c1, c2):=RK_AUX3(p, v, u_, c1, c2, SUBST(p, v, u_ + c2/2))",
        "RK_AUX3(p, v, u_, c1, c2, c3):=(c1 + SUBST(p, v, u_ + c3) + 2·(c2 + c3))/6",
        "V(x):=1/2·x^4",
        *[f"{name}:=" for name in "c1 c2 c3 et u_ v0 yp".split()],
        "hCross:=APPROX(15583499999999999/5000000000000000)",
        *settings(),
    ]
    sheet = Sheet(state=state)
    sheet.expression("RK(r,v,v0,h,n):=RK_AUX0(h*APPEND([1],r),v,v0,n)")
    sheet.expression("V(x):=0.5*x^4")
    sheet.expression("(RK(RHS_(y,yp,0.5),[x,y,yp],[0,1,0],0.1,50)) SUB  SUB [1,2]")
    assert not session.load(sheet.write(tmp_path / "horb_2.dfw"))
    assert body(session, "V") == "1/2*x^4"
    assert await simplified(session, "RK([y], [x, y], [0, 1], 1/2, 1)") == (
        "[[0, 1], [1/2, 211/128]]"
    )


async def test_pts_dots_damaged_annotations_and_rows_of_vectors(session, tmp_path):
    state = [
        "hCross:=APPROX(38929/10000)",
        "list:=[[-2, -2, -2]; [1, 1, 1]]",
        "list1:=[]",
        "vCross:=APPROX(61071/10000)",
        *settings(),
    ]
    sheet = Sheet(state=state)
    sheet.expression("list:=[[[-2,-2,-2]],[[1,1,1]]]")
    sheet.expression("VECTOR(VECTOR([[[a,b,a-b],[a,b,a-b]]],a,-1,2),b,-2,1)")
    sheet.expression("[[[[[-1,-2,1],[-1,-2,1]]]]]", "Simp(User6)")
    sheet.expression("[[[[-1,-2,1],[0,-2,2],[1,-2,3],[2,-2,4]]]]", "Simp(User0)")
    sheet.expression("[[-1,-2,1],[0,-2,2]]", "Simp(User2)")
    sheet.embedded()
    path = sheet.write(tmp_path / "pts_dots.dfw")
    assert str(session.load(path)) == "1 embedded object not loaded"
    assert await simplified(session, "list") == "[[[-2, -2, -2]], [[1, 1, 1]]]"
    session.merge(path)
    assert annotations(session)[-3:] == ["Simp(User6)", "Simp(User0)", "Simp(User2)"]


async def test_induction_1_names_that_differ_only_in_case(session, tmp_path):
    state = [
        "S(n):=∑(r^3, r, 1, n)",
        "T(n):=n^2·(n + 1)^2/4",
        "s(n):=n^2·(n + 1)^2/4",
        *settings(),
    ]
    sheet = Sheet(state=state)
    sheet.expression("r^3")
    sheet.expression("SUM(r^3,r,1,n)", "Sum(#1,r)")
    sheet.expression("S(n):=SUM(r^3,r,1,n)")
    sheet.expression("TABLE([S(n),T(n)],n,1,10)")
    sheet.expression("[[1,1,1],[2,9,9],[10,3025,3025]]", "Simp(#4)")
    assert not session.load(sheet.write(tmp_path / "induction_1.dfw"))
    assert {"S", "s", "T"} <= set(session.functions)
    assert await simplified(session, "S(3)") == "36"
    assert await simplified(session, "s(3)") == "36"
    assert body(session, "S") == "SUM(r^3,r,1,n)"


async def test_col_pade_a_compressed_state_with_a_program(session, tmp_path):
    state = [
        "A_(d,n):=VECTOR(LIM(APPEND(VECTOR(x^k,k,0,n),VECTOR(-y·x^k,k,0,DIM(d)-n-2)),"
        "[x,y],[d↓j↓1,d↓j↓2]),j,1,DIM(d))",
        "PADE(y,x,x0,n,d,j_≔0,t_≔0):=PROG(LOOP(t_:+LIM(y,x,x0)·x^j_,j_:+1,"
        "IF(j_>n+d,exit),y≔∂(y,x)/j_),t_)",
        "RECURRENCE(u,v,v0,m):=REVERSE(ITERATE(APPEND([u],v),v,v0,m-DIM(v0)))",
        "d:=[-1,5/4;-1/2,32/25;1,1/2;3/2,0;3,1/8]",
        *[f"{name}:=" for name in "T_C j_ t_ v0 x0".split()],
        *settings(DisplayFormat="Compressed"),
    ]
    sheet = Sheet(state=state)
    sheet.text(
        rtf(
            "COL_PADE.MTH by Adam Marlewski (Poznan,Poland), 1992.07.22",
            "adapted for DERIVE version 6.10 by Josef Böhm, November 2005",
        )
    )
    sheet.expression("d:=[[-1,5/4],[-1/2,32/25],[1,1/2],[3/2,0],[3,1/8]]")
    sheet.expression("COL_PADE(d,1)=(2*x-3)/(x^3-3)", "User=Simp(User)")
    sheet.expression(
        "61*x^4/1200-37*x^3/1200-1771*x^2/4800-413*x/1200+1909/1600", "Simp(#19)"
    )
    sheet.expression("DisplayFormat:=Compressed")
    sheet.expression("PADE(d,2)")
    assert not session.load(sheet.write(tmp_path / "COL_PADE.dfw"))
    assert annotations(session)[4] == "Simp(#19)"
    assert await simplified(session, "PADE(1/(1 - x), x, 0, 2, 0)") == "x^2 + x + 1"
    assert await simplified(session, "[DIM(d), DIM(d SUB 1)]") == "[5, 2]"


async def test_col_pade_a_held_recurrence(session, tmp_path):
    state = ["CHEBYSHEV_T_LIST(n,x):=RECURRENCE('(2·x·v_↓1-v_↓2),v_,[x,1],n)"]
    assert not session.load(Sheet(state=state).write(tmp_path / "COL_PADE.dfw"))
    # The held recurrence stays held at each pass of the iteration.
    assert await simplified(session, "CHEBYSHEV_T_LIST(4, x)") == (
        "[1, x, 2*x*v_ SUB 1 - v_ SUB 2, 2*x*v_ SUB 1 - v_ SUB 2]"
    )


async def test_hill_new_a_user_hex_and_old_function_names(session, tmp_path):
    state = [
        "AUX(number, base):=VECTOR(SELCT(REM((SEQU(number, base))↓k, base)), k, 1, "
        "DIM(SEQU(number, base)) - 2)",
        "CONV(a, x_):=[REVERSE(AUX(a, x_))]",
        "HEX(a):=IF(a = 0, [0, 0], IF(a > 15, (CONV(a, 16))↓1, APPEND([0], (CONV(a, "
        "16))↓1)))",
        "REM(number, base):=MOD(number, base)",
        "SELCT(r):=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, A, B, C, D, E, F]↓(r + 1)",
        "SEQU(number, base):=ITERATES(FLOOR(x, base), x, number)",
        *settings(CaseMode="Sensitive"),
    ]
    sheet = Sheet(state=state)
    sheet.expression("CaseMode:=Sensitive")
    sheet.expression(
        "AUX(number,base):=VECTOR(SELCT(REM((SEQU(number,base)) SUB k,base)),k,1,"
        "DIMENSION(SEQU(number,base))-2)"
    )
    sheet.expression("CONV(a,x_):=[REVERSE_VECTOR(AUX(a,x_))]")
    sheet.expression(
        "HEX(a):=IF(a=0,[0,0],IF(a>15,(CONV(a,16)) SUB 1,APPEND([0],(CONV(a,16)) SUB 1)))"
    )
    sheet.expression("[HEX(56),HEX(13)]=[[3,8],[0,D]]", "User=Simp(User)")
    assert not session.load(sheet.write(tmp_path / "Hill_new.dfw"))
    assert await simplified(session, "[HEX(56), HEX(13)]") == "[[3, 8], [0, D]]"


async def test_titbits3_state_definitions_of_number_theory(session, tmp_path):
    state = [
        "CYCLOTOMIC(n, x):=∏((x^(n/d_) - 1)^MOEBIUS_MU(d_), d_, DIVISORS(n))",
        "DIVISORS(n):=SORT(VECTOR(∏(u_), u_, {[1]}·∏(VECTOR(MAP_LIST([v_↓1^k_], k_, "
        "{0, ..., v_↓2}), v_, FACTORS(n)))))",
        "MOEBIUS_MU(n):=∏(- MAX(2 - v_↓2, 0), v_, FACTORS(n))",
        *settings(),
    ]
    sheet = Sheet(state=state)
    sheet.expression("MOEBIUS_MU(n):=PRODUCT(-MAX(2-v_ SUB 2,0),v_,FACTORS(n))")
    sheet.expression(
        "CYCLOTOMIC(n,x):=PRODUCT((x^(n/d_)-1)^MOEBIUS_MU(d_),d_,DIVISORS(n))"
    )
    sheet.expression("CYCLOTOMIC(12,x)=x^4-x^2+1", "User=Simp(User)")
    assert not session.load(sheet.write(tmp_path / "titbits3.dfw"))
    assert {"CYCLOTOMIC", "DIVISORS", "MOEBIUS_MU"} <= set(session.functions)
    assert await simplified(session, "[MOEBIUS_MU(6), MOEBIUS_MU(12)]") == "[1, 0]"
    assert await simplified(session, "CYCLOTOMIC(12, x)") == "x^4 - x^2 + 1"


async def test_titbits3_typed_again_uses_the_pre_defined_divisors(session):
    session.author("MOEBIUS_MU(n):=PRODUCT(-MAX(2-v_ SUB 2,0),v_,FACTORS(n))")
    session.author("CYCLOTOMIC(n,x):=PRODUCT((x^(n/d_)-1)^MOEBIUS_MU(d_),d_,DIVISORS(n))")
    assert "DIVISORS" not in session.functions
    assert await simplified(session, "CYCLOTOMIC(12, x)") == "x^4 - x^2 + 1"


async def test_titbits3_divisors_by_products_of_vectors(session, tmp_path):
    state = [
        "DIVISORS(n):=SORT(VECTOR(∏(u_), u_, {[1]}·∏(VECTOR(MAP_LIST([v_↓1^k_], k_, "
        "{0, ..., v_↓2}), v_, FACTORS(n)))))",
    ]
    assert not session.load(Sheet(state=state).write(tmp_path / "titbits3.dfw"))
    assert await simplified(session, "DIVISORS(12)") == "[1, 2, 3, 4, 6, 12]"


def test_appel_ti_d6_handheld_imports(session, tmp_path):
    sheet = Sheet(state=settings())
    sheet.expression("PROG(a:=[[2],[1]],b:=[[9/2],[1/2]])", "Handheld")
    sheet.expression("b:=[[9/2],[1/2]]", "Simp(#1)")
    sheet.expression("line1:=a+t*(b-a)", "Handheld")
    sheet.expression("line1:=[[5*t/2+2],[1-t/2]]", "Simp(#3)")
    sheet.expression("line1 SUB 1 SUB 1", "Handheld")
    sheet.expression("5*t/2+2", "Simp(#5)")
    assert not session.load(sheet.write(tmp_path / "APPEL_TI_D6.dfw"))
    assert annotations(session)[::2] == ["Handheld"] * 3
    assert session.entries[1].node.kind.name == "ASSIGN"


async def test_bessel_b_new_a_load_of_an_installation_path(session, tmp_path):
    state = [
        "ALPHA(L_, NMAX, z):=IF(L_ = -1, 1, IF(L_ = 0, B(L_, NMAX, z), IF(L_ ≥ 1, B(L_, "
        "NMAX, z)·ALPHA(L_ - 1, NMAX, z) + A(L_)·ALPHA(L_ - 2, NMAX, z))))",
        "PASO(L_, z):=IF(L_ = 0, - COS(z)/z, IF(L_ = 1, (- SIN(z) - COS(z)/z)/z, (2·L_ - "
        "1)/z·PASO(L_ - 1, z) - PASO(L_ - 2, z)))",
        "α(l_, nmax, z):=IF(l_ = -1, 1, IF(l_ = 0, B(l_, nmax, z), 0))",
        *[f"{name}:=" for name in "L_ NMAX N_ l_ n_ nmax".split()],
        "nlim:=5",
        *settings(PrecisionDigits="6", NotationDigits="6", InputMode="Word"),
    ]
    sheet = Sheet(state=state)
    sheet.expression(
        'LOAD("C:\\Programme\\TI Education\\Derive 6\\Math\\BesselFunctions.mth")'
    )
    sheet.expression(
        "PASO(L_,z):=IF(L_=0,-COS(z)/z,IF(L_=1,(-SIN(z)-COS(z)/z)/z,"
        "(2*L_-1)/z*PASO(L_-1,z)-PASO(L_-2,z)))"
    )
    sheet.expression("InputMode:=Word")
    assert not session.load(sheet.write(tmp_path / "BESSEL_B_new.dfw"))
    assert {"ALPHA", "α", "PASO"} <= set(session.functions)
    assert {"L_", "l_", "NMAX", "nmax"} <= set(session.state.variables)
    assert await simplified(session, "#1") == (
        'LOAD("C:\\Programme\\TI Education\\Derive 6\\Math\\BesselFunctions.mth")'
    )
    assert session.take_notice() == "BesselFunctions.mth could not be found"
    math_file(tmp_path / "BesselFunctions.mth", "PASO(L_,z):=L_")
    assert await simplified(session, "#1") == "true"
    assert await simplified(session, "PASO(3, z)") == "3"


# -- math files --------------------------------------------------------------------


async def test_gauss_newton_bare_annotations_and_settings_vectors(session, tmp_path):
    path = math_file(
        tmp_path / "Gauss_Newton.mth",
        ";",
        "[InputMode:=Word,Precision:=Approximate,PrecisionDigits:=10,Notation:=Scienti~",
        "fic,NotationDigits:=8]",
        "",
        ";",
        '[output:=,iter:=,msg:="",feq:=,rsq:=,arsq:=,anova:=,\x7f03b2_:=,xpxi:=,'
        "vars:=,pa~",
        "rm1:=,ivars1:=,eqq:=,se_:=,dfe_:=,r_:=]",
        "",
        "position(e,u):=PROG(d:=DIM(u),i_:=1,LOOP(IF(e=u SUB i_,RETURN(i_)),IF(i_>d,RE~",
        "TURN(0)),i_:+1))",
        "",
        "GAUSS_NEWTON(eq,parm,\x7f03b20_,data,n_:=30,sn_:=10,cc_:=10^(-8)):="
        "PROG(eqq:=RHS~",
        "(eq),eqq)",
    )
    assert not session.load(path)
    assert session.settings["Precision"] == "Approximate"
    assert session.settings["Notation"] == "Scientific"
    assert session.settings["NotationDigits"] == 8
    assert annotations(session)[:2] == ["", ""]
    assert "β_" in session.state.variables
    assert session.functions["GAUSS_NEWTON"].names[2] == "β0_"
    assert await simplified(session, "position(3, [1, 2, 3])") == "3"
    assert await simplified(session, "POSITION(3, [1, 2, 3])") == "3"


def test_logarithmic_spiral_settings_domains_and_wrapped_escapes(session, tmp_path):
    lines = []
    for name, value in SETTINGS.items():
        changed = {"PrecisionDigits": "64", "CaseMode": "Sensitive", "InputMode": "Word"}
        value = changed.get(name, value)
        lines += [f"{name}:={value}", ""]
    path = math_file(
        tmp_path / "Logarithmic_Spiral_Surfaces.mth",
        *lines,
        "\x7f03c6:epsilonReal [-pi, +pi]",
        "",
        "u:epsilonVector",
        "",
        "SpiralTorusYZ_U(b,m,\x7f03b8,\x7f03c6):=[m*b^\x7f03b8*COS(\x7f03c6),"
        "b^\x7f03b8*((2*pi*m*SIN(~",
        "\x7f03c6)/SQRT(LN(b)^2+4*pi^2)+1)*COS(2*pi*\x7f03b8)+m*LN(b)*SIN(\x7f03c6)*"
        "SIN(2*pi*\x7f0~",
        "3b8)/SQRT(LN(b)^2+4*pi^2))]",
        "",
        "h(b,m,n):=PROG(\x7f03b4m:epsilonReal,"
        'WRITE(APPEND("m := ",APPROX(STRING(m),n))),~',
        "\x7f03b4m:=m-APPROX(f(b,m),2*n)/APPROX(g(b,m),2*n),RETURN(APPROX(\x7f03b4m,n)))",
    )
    assert not session.load(path)
    domain = session.domains["φ"]
    assert domain.kind is DomainKind.REAL and domain.closed_low and domain.closed_high
    assert write_expression(domain.high) == "+pi"
    assert session.functions["SpiralTorusYZ_U"].names == ("b", "m", "θ", "φ")
    assert "SIN(2*pi*θ)" in body(session, "SpiralTorusYZ_U")
    assert session.settings["PrecisionDigits"] == 64
    assert "h" in session.functions


def test_scheuer1_dos_annotations_kept_as_written(session, tmp_path):
    path = math_file(
        tmp_path / "SCHEUER1.MTH",
        "InputMode:=Word",
        "",
        "CaseMode:=Sensitive",
        "",
        "Rg:=R1+R2*RL/(R2+RL)",
        "",
        "Ig:=U/Rg",
        "",
        ";Sub(#5)",
        "Ig:=U/(Rp-R2+R2*RL/(R2+RL))",
        "",
        ";Simp(#9/User)",
        "-R2*RL/(R2^2-R2*Rp-RL*Rp)",
    )
    assert not session.load(path)
    assert annotations(session)[4:] == ["Sub(#5)", "Simp(#9/User)"]
    assert {"RL", "Rp", "R2"} <= names_in(session.entries[4].node)


def test_circles_a_settings_block_with_numbers_for_the_radix(session, tmp_path):
    path = math_file(
        tmp_path / "CIRCLES.MTH",
        "Precision:=Exact",
        "PrecisionDigits:=6",
        "Notation:=Rational",
        "NotationDigits:=3",
        "CaseMode:=Insensitive",
        "VariableOrder:=[x,y,z]",
        "OutputBase:=10",
        "",
        "InputBase:=10",
        "",
        "InputMode:=Character",
        "",
        "DisplayFormat:=Compressed",
        "",
        "TimesOperator:=Dot",
        "",
        '"Circles1"',
        "",
        "VECTOR(VECTOR(SQRT(1-(x+5-2*a)^2)-5+2*b,a,0,5),b,0,5)",
    )
    session.settings.assign("InputMode", "Word")
    session.load(path)
    assert session.settings.base("OutputBase") == 10
    assert session.settings["InputMode"] == "Character"
    assert '"Circles1"' in texts(session)
    assert texts(session)[-1] == "VECTOR(VECTOR(SQRT(1-(x+5-2*a)^2)-5+2*b,a,0,5),b,0,5)"


def test_f_n1_strings_of_a_tilde_and_a_backslash(session, tmp_path):
    path = math_file(
        tmp_path / "F_N1.MTH",
        'n1:=[["~",[[1.6,7],[1.6,6.6],[2,6.6],[2,7],[1.6,7],[inf,~',
        "inf],[4,7],[4,6.6],[4.4,6.6],[4.4,7],[4,7]]]]",
        "",
        'char_names_n1:=["#","$","@","[","\\","]","^","`","{","|","}","~"]',
        end=b"\x1a",
    )
    assert not session.load(path)
    names = session.entries[1].node.children[1].children
    assert len(names) == 12
    assert [names[4].value, names[11].value] == ["\\", "~"]


async def test_hypothesis_testing_a_semicolon_in_a_wrapped_string(session, tmp_path):
    path = math_file(
        tmp_path / "Hypothesis Testing1.mth",
        b'HT(pz,alpha):=PROG(IF(pz<alpha,con:="Reject Ho",con:="Do Not Reject Ho"),~',
        b'o1:=["Ho:","Ha:","x1:","s1:","n1:","x2:","s2:","n2:"],o1:=APPEND(o1,["s(x1-x2~',
        b');","Z:","Z\xe0:","P(Z):","\xe0:","Conclusion:"]),o1)',
    )
    assert not session.load(path)
    answer = await simplified(session, "HT(0, 1)")
    assert '"s(x1-x2);", "Z:", "Zα:", "P(Z):", "α:"' in answer


async def test_hypothesis_testing_a_quoted_if(session, tmp_path):
    path = math_file(tmp_path / "h.mth", "FPlot(f,g):=['IF(x>=0,g),f SUB 2]")
    assert not session.load(path)
    assert await simplified(session, "FPlot([a, b], c)") == "[IF(x >= 0, c), b]"


async def test_fuster_ode2_a_derive_6_utility_file(session, tmp_path):
    path = math_file(
        tmp_path / "fuster_ode2.mth",
        '"SecondOrderODEs.mth -- Copyright (c) 1990-2003 by Texas Instruments Incorpor~',
        'ated"',
        "",
        "LIN_SOLVE(a,b):=ELEMENT(ROW_REDUCE(a,b)`,1+DIM(a))",
        "",
        "DISC_AUX(p,q,x):=p^2/4-q+DIF(p,x)/2",
        "",
        "LIN2_HOM_AUX(d,x):=IF(d<0,[COS(SQRT(-d)*x),SIN(SQRT(-d)*x)],[#e^(SQRT(d)*x),#~",
        "e^(-SQRT(d)*x)],[#e^(SQRT(d)*x),#e^(-SQRT(d)*x)])",
    )
    assert not session.load_utility(path)
    assert texts(session) == []
    assert {"LIN_SOLVE", "DISC_AUX", "LIN2_HOM_AUX"} == set(session.functions)
    assert await simplified(session, "LIN2_HOM_AUX(-1, x)") == "[COS(x), SIN(x)]"
    assert await simplified(session, "LIN_SOLVE([[1, 1], [1, -1]], [3, 1])") == "[2, 1]"
    # The rest of the file's functions are the pre-defined library's.
    session.author("InputMode := Word")
    assert await simplified(session, "DSOLVE2(0, 1, 0, x, c1, c2)") == (
        "c1*COS(x) + c2*SIN(x)"
    )


async def test_ds_gauss_declarations_and_a_load_of_a_missing_file(session, tmp_path):
    path = math_file(
        tmp_path / "DS-GAUSS.MTH",
        '"Elementary matrix program. Copyright 1998 by David Stegenga."',
        "",
        "[[],a,1,1,1,0]",
        "",
        "ROW_SWAP(v,k):=",
        "",
        "ROW_SCALE(v):=",
        "",
        'LOAD("..\\math\\vector")',
        "",
        "COLS(a):=DIMENSION(a`)",
        "",
        "ROWS(a):=DIMENSION(a)",
    )
    assert str(session.load(path)) == "vector could not be found"
    assert {"ROW_SWAP", "ROW_SCALE"} <= set(session.state.functions)
    assert await simplified(session, "ROWS([[1, 2], [3, 4]])") == "2"

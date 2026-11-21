"""A session computing through the engine in a worker process, as the shipped app does.

Almost every other test computes in-process. Here one worksheet's worth of commands,
touching every kind of value and effect that must cross the worker's protocol, runs
through a real worker: library functions, programs that loop, show progress, write a
notice, hold a value, change a setting, draw random numbers and load a file; a step
limit and an abort; sets, strings, the Vector, Set and Logical domains; answers that
keep a command's form (`FACTOR`, `SOLVE`, matrix factorizations). The answers must be
those of an in-process session, and the worksheet saved with its state must read back
into a fresh session that computes the same.
"""

import threading

import pytest
from dfwsheet import OLD, Sheet, derive5
from screen import entries

from rederive.engine.remote import EngineAborted, EngineStepLimit, RemoteEngine
from rederive.model.session import Session
from rederive.syntax import write_expression
from rederive.ui.app import RederiveApp

#: The worksheet: `>` lines are entered, the others simplified to the answer given. The
#: notice is what the command left to say, if anything.
SCENARIO = [
    "> InputMode := Word",
    # A library function, defined by its text in the worker.
    ("FIBONACCI(20)", "6765"),
    ("TAYLOR(SIN(x), x, 0, 5)", "x^5/120 - x^3/6 + x"),
    ("CHEBYCHEV_T(3, x)", "4*x^3 - 3*x"),
    # Programs, with their effects.
    "> count(n, k, s) := PROG(k := 0, s := 0, LOOP(IF(k >= n, RETURN(s)), k :+ 1, "
    "s :+ k^2))",
    ("count(10)", "385"),
    "> noisy(n, k) := PROG(k := 0, LOOP(IF(k = n, RETURN(k)), k :+ 1, DISPLAY(k)))",
    ("noisy(3)", "3"),
    '> wr(n) := PROG(WRITE("done"), n + 1)',
    ("wr(4)", "5", "done"),
    "> h := 'x + x",
    ("h", "x + x"),
    "> setp(n) := PROG(Precision := Approximate, n/3)",
    ("setp(1)", "0.3333333333"),
    ("Precision", "Approximate"),
    "> Precision := Exact",
    ("glob := 7", "glob := 7"),
    "> bump(n) := PROG(glob :+ n, glob)",
    ("bump(3)", "10"),
    ("glob", "10"),
    ("RANDOM(1000)", "0"),
    ("RANDOM(1000)", "618"),
    ('PROG(LOAD("util1.mth"), ADD1(3, 4))', "7"),
    ("ADD1(5, 6)", "11"),
    # Sets, strings and the domains.
    ("{1, 2, 3} UNION {3, 4}", "{1, 2, 3, 4}"),
    ('APPEND("ab", "cd")', '"abcd"'),
    ('"abc" = "abd"', "false"),
    "> v :epsilon Vector",
    "> s :epsilon Set",
    "> p :epsilon Logical",
    ("IF(v = v, 1, 2)", "1"),
    ("s UNION s", "s"),
    ("p AND NOT p", "false"),
    ("IF((n AND d) = 0, 1, 2)", "IF((d AND n) = 0, 1, 2)"),
    # Answers in a command's form.
    ("FACTOR(12)", "2^2*3"),
    ("SOLVE(x^2 = 4, x)", "x = -2 OR x = 2"),
    ("SOLVE(x^2 = 4, x) AND x > 0", "x = 2"),
    ("SOLUTIONS(x^2 = 4, x)", "[2, -2]"),
    (
        "FACTOR([[1, 2], [3, 4]], Turing)",
        "[[0, 1], [1, 0]] . [[1, 0], [1/3, 1]] . [[3, 0], [0, 2/3]] . [[1, 4/3], [0, 1]]"
        " . [[1, 0], [0, 1]]",
    ),
    (
        "FACTOR([[4, 2], [2, 3]], Schmidt)",
        "[[2*SQRT(5)/5, -SQRT(5)/5], [SQRT(5)/5, 2*SQRT(5)/5]] . [[2*SQRT(5), "
        "7*SQRT(5)/5], [0, 4*SQRT(5)/5]]",
    ),
]


@pytest.fixture(scope="module")
def engine():
    """One worker for the module: starting one costs a second of imports."""
    worker = RemoteEngine()
    worker.start()
    yield worker
    worker.shutdown()


@pytest.fixture(autouse=True)
def here(tmp_path, monkeypatch):
    """A directory holding the utility file the scenario loads."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "util1.mth").write_text("ADD1(x, y) := x + y\n")


async def played(session):
    """The scenario's answers and notices from `session`, and the texts the programs
    showed while running."""
    heard = []
    session.runner.progress = lambda text, displayed: heard.append(text)
    answers = []
    for step in SCENARIO:
        if isinstance(step, str):
            session.author(step[1:].strip())
            continue
        answers.append(((await session.simplify(step[0])).text, session.take_notice()))
    return answers, heard


def expected():
    """The scenario's answers and notices."""
    return [
        (step[1], step[2] if len(step) > 2 else None)
        for step in SCENARIO
        if not isinstance(step, str)
    ]


async def test_a_session_computes_through_the_worker_as_it_does_in_process(engine):
    remote, remote_heard = await played(Session(runner=engine))
    local, local_heard = await played(Session())
    assert remote == expected()
    assert local == remote
    assert remote_heard == local_heard == ["1", "2", "3", "done"]


async def test_a_worksheet_saved_with_its_state_reads_back_and_computes_alike(
    engine, tmp_path
):
    session = Session(runner=engine)
    await played(session)
    session.settings.assign("SaveState", "Save")
    session.settings.assign("PrecisionDigits", "12")
    session.settings.assign("Branch", "Real")
    path = tmp_path / "scenario.mth"
    session.save(path)

    fresh = Session(runner=engine)
    assert not fresh.load(path)
    for setting in ("Precision", "PrecisionDigits", "Branch", "InputMode"):
        assert fresh.settings[setting] == session.settings[setting], setting
    # The file starts with the settings, read back as entries, then the worksheet.
    written = [write_expression(entry.node) for entry in session.entries]
    read = [write_expression(entry.node) for entry in fresh.entries]
    assert read[-len(written) :] == written
    for name in ("v", "s", "p"):
        assert fresh.declared_as(name) == session.declared_as(name)
    for text, answer in [
        ("count(10)", "385"),
        ("h", "x + x"),
        # Reading the file assigns what the worksheet assigns; a program's effect
        # came from simplifying, which reading doesn't repeat.
        ("glob", "7"),
        ("IF(v = v, 1, 2)", "1"),
        ("p AND NOT p", "false"),
        ("FACTOR(12)", "2^2*3"),
    ]:
        assert (await fresh.simplify(text)).text == answer, text


async def test_a_runaway_program_is_stopped_by_the_step_limit_or_an_abort(engine):
    session = Session(runner=engine, step_limit=1000)
    session.author("gl := 1")
    session.author("spin(n) := LOOP(gl := gl + 1)")
    with pytest.raises(EngineStepLimit):
        await session.simplify("spin(0)")
    assert (await session.simplify("gl")).text == "1"

    worker = RemoteEngine()
    worker.start()
    try:
        session = Session(runner=worker)
        session.author("spin(n) := LOOP(n := n + 1)")
        assert (await session.simplify("1 + 1")).text == "2"
        threading.Timer(1.0, worker.abort).start()
        with pytest.raises(EngineAborted):
            await session.simplify("spin(0)")
        assert (await session.simplify("count := 3")).text == "count := 3"
    finally:
        worker.shutdown()


# -- the terminal app ----------------------------------------------------------


async def settled(pilot, app, count, patience=60.0):
    """Wait until the worksheet holds `count` entries: the worker answers on another
    thread, after the key that asked has been handled."""
    for _ in range(int(patience / 0.05)):
        if len(app.session.entries) >= count:
            await pilot.pause()
            return entries(app)
        await pilot.pause(0.05)
    raise AssertionError(f"still {entries(app)}")


async def test_the_app_runs_a_loaded_worksheet_and_the_new_commands_through_the_worker(
    engine, tmp_path
):
    session = Session(runner=engine)
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
    assert not session.load(sheet.write(tmp_path / "sumfact.dfw"))
    session.author("InputMode := Word")
    app = RederiveApp(session=session)
    async with app.run_test(size=(100, 40)) as pilot:
        # A program that shows its progress, simplified through the worker.
        session.author("PROG(DISPLAY(sumfactpow(k^2 + 1, k, y)), 1)")
        app.place_windows()
        await pilot.press("s", "enter")
        assert (await settled(pilot, app, 5))[-2:] == ['"(y^2 + y + 1)*#e^y"', "1"]
        # Calculus tAble, then simplified.
        session.author("x^2")
        app.place_windows()
        await pilot.press("c", "a", "enter", "enter", "5", "enter")
        assert (await settled(pilot, app, 7))[-1] == "TABLE(x^2, x, 1, 5)"
        await pilot.press("s", "enter")
        assert (await settled(pilot, app, 8))[-1] == (
            "[[1, 1], [2, 4], [3, 9], [4, 16], [5, 25]]"
        )
        # Factor on a matrix, with scHmidt.
        session.author("[[5, 1], [12, 3]]")
        app.place_windows()
        await pilot.press("f", "enter", "h")
        assert (await settled(pilot, app, 10))[-1] == (
            "[[5/13, -12/13], [12/13, 5/13]] . [[13, 41/13], [0, 3/13]]"
        )
        # Declare Variable Set, which the worker then reads.
        await pilot.press("d", "v", "w", "enter", "s")
        assert entries(app)[-1] == "w :epsilon Set"
        session.author("w UNION w")
        app.place_windows()
        await pilot.press("s", "enter")
        assert (await settled(pilot, app, 13))[-1] == "w"
        # Options Defaults.
        session.author("Precision := Approximate")
        await pilot.press("o", "d", "y")
        assert app.settings["Precision"] == "Exact"
        assert entries(app)[-2:] == ["Precision := Exact", "InputMode := Character"]

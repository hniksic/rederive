"""Work budgets: an optional step past its work is interrupted, always at the same point,
and nothing it left behind changes a later answer.

`within` is the one place the engine bounds a sympy call (see `rederive.engine.budget`):
by the objects it constructs, not by time. These tests check its contract (the answer
or None, nesting, any thread), that a count is the same in every run and process
whatever the machine's load, and that interrupting computations at many points leaves
sympy's caches and the engine's own state as they should be: the same inputs computed
afterwards give the same answers as before.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import threading

import pytest
import sympy as sp
from simplifying import simp
from sympy.core import random as sympy_random
from sympy.core.cache import clear_cache
from sympy.polys.densearith import dup_mul
from sympy.polys.densebasic import dup_from_dict
from sympy.polys.densetools import dup_eval
from sympy.polys.domains import ZZ

from rederive.engine.budget import counted, within
from rederive.engine.computing import Context


def _work(steps: int) -> int:
    """Construct sympy objects, a few dozen per step, from the state a command starts
    from: caches empty, random generators seeded, and the variable a new object."""
    clear_cache()
    sympy_random.seed(0)
    variable = sp.Symbol("x")
    total = sp.Integer(0)
    for k in range(steps):
        total = sp.expand(total + (variable + k) ** 3)
    return steps


def test_an_operation_within_its_budget_gives_its_value():
    assert within(10**6, lambda: 42) == 42
    assert within(10**6, lambda: _work(5)) == 5


def test_an_operation_past_its_budget_gives_none():
    assert within(100, lambda: _work(1000)) is None


def test_an_inner_budget_runs_out_alone():
    def outer() -> tuple:
        inner = within(100, lambda: _work(1000))
        return inner, _work(2)

    assert within(10**6, outer) == (None, 2)


def test_an_outer_budget_reaches_its_own_boundary():
    reached = []

    def outer() -> str:
        within(10**9, lambda: _work(10**6))
        reached.append("after the inner operation")
        return "finished"

    assert within(1000, outer) is None
    assert not reached


def test_a_swallowed_interruption_is_raised_again():
    def swallowing() -> str:
        try:
            _work(10**6)
        except BaseException:
            pass
        _work(10**6)
        return "finished"

    assert within(1000, swallowing) is None


def test_exceptions_pass_through():
    def failing() -> None:
        raise ValueError("no")

    with pytest.raises(ValueError):
        within(10**6, failing)


def test_a_budget_holds_off_the_main_thread_and_counts_only_its_own_thread():
    """Another thread's work neither counts against a budget nor escapes its own."""
    found = []
    stop = threading.Event()

    def busy() -> None:
        # Polynomial arithmetic, which counts as work but leaves sympy's caches alone.
        while not stop.is_set():
            dup_mul([1, 2, 3], [4, 5], ZZ)

    thread = threading.Thread(target=lambda: found.append(within(100, lambda: _work(10))))
    other = threading.Thread(target=busy)
    other.start()
    try:
        thread.start()
        thread.join()
        _, alone = counted(lambda: _work(10))
        assert counted(lambda: _work(10))[1] == alone
    finally:
        stop.set()
        other.join()
    assert found == [None]


def test_arithmetic_on_huge_numbers_counts_by_their_size():
    """An operation on numbers of hundreds of thousands of digits costs far more than
    one on short ones, and counts so: sympy's integers, the rationals of its polynomial
    arithmetic, and a primality test. The size of most numbers is looked at in one of
    several operations, so each side does several."""
    huge = sp.Integer(3) ** 400_000
    short = sp.Integer(3) ** 40
    shortly = max(counted(lambda: [short * (short + k) for k in range(64)])[1], 1)
    assert counted(lambda: [huge * (huge + k) for k in range(64)])[1] > 100 * shortly
    x = sp.Symbol("x")
    coefficients = [sp.Rational(3 ** (5000 + k), 7 ** (4000 + k)) for k in range(8)]
    long = sp.Poly(sum(c * x**k for k, c in enumerate(coefficients)), x, domain="QQ")
    plain = sp.Poly(sum(k * x**k for k in range(8)), x, domain="QQ")
    products = [counted(lambda p=p: [p * p for _ in range(32)])[1] for p in (long, plain)]
    assert products[0] > 100 * products[1]
    prime = 2**4423 - 1
    small = counted(lambda: sp.isprime(997))[1]
    assert counted(lambda: sp.isprime(prime))[1] > 1000 * small


def test_filling_a_dense_polynomial_counts_by_its_degree():
    """A polynomial of a huge degree is filled with a zero per degree, each in about a
    hundredth of the time of constructing an object, which counts ten units: a unit per
    ten coefficients."""
    terms = {(1_000_000,): ZZ(1), (0,): ZZ(1)}
    assert 50_000 <= counted(lambda: dup_from_dict(terms, ZZ))[1] <= 200_000
    assert counted(lambda: dup_from_dict({(10,): ZZ(1)}, ZZ))[1] < 10


def test_evaluating_a_long_polynomial_at_a_long_number_counts_by_both_sizes():
    """Horner's rule makes the partial value longer at each coefficient: a polynomial
    of degree twenty thousand at a 64-bit number takes about a second, the time of a
    million units, in one call."""
    point = ZZ(2**64 - 59)
    terms = [ZZ(k % 1000 + 1) for k in range(20_000)]
    assert 500_000 <= counted(lambda: dup_eval(terms, point, ZZ))[1] <= 2_000_000
    assert counted(lambda: dup_eval(terms[:10], point, ZZ))[1] < 10


def test_interrupting_at_many_points_changes_no_later_answer():
    expected = _answers()
    approximate = simp("APPROX(pi*SQRT(2), 30)")
    chooser = random.Random(1234)
    for _ in range(60):
        text = chooser.choice(_INPUTS)
        if chooser.random() < 0.5:
            # From scratch, so that the interruption falls amid sympy's own work.
            clear_cache()
        within(chooser.randrange(10, 20_000), lambda: simp(text))
    assert _answers() == expected
    assert simp("APPROX(pi*SQRT(2), 30)") == approximate
    assert simp("2/3 + 0.5", Context()) == simp("2/3 + 1/2")


#: Inputs that exercise the caches an interruption could leave half-written: sympy's
#: assumptions and function caches, polynomial domains, integration, solving, limits,
#: the engine's memo of simplified subexpressions, and mpmath's precision.
_INPUTS = [
    "INT(x^2*SIN(x)^3, x)",
    "INT(1/(x^4 + 1), x)",
    "SOLVE(x^3 - 6*x^2 + 11*x - 6 = 0, x)",
    "SOLVE([x + y = 3, x*y = 2], [x, y])",
    "LIM((1 + 1/n)^n, n, inf)",
    "SUM(1/k^2, k, 1, inf)",
    "FACTOR(x^6 - 1)",
    "(x^3 - 1)/(x - 1) + SQRT(8)",
    "SIN(x)^2 + COS(x)^2 + TAN(3*x)/TAN(x)",
    "SQRT(5 + 2*SQRT(6))",
    "DIF(ATAN(x/a), x)",
    "EXPAND((x + y + 1)^6)",
]


def _answers() -> list[str]:
    return [simp(text) for text in _INPUTS]


# -- the same count everywhere ---------------------------------------------------------

#: Simplifies inputs after other commands, so that the caches had other contents, and
#: prints what simplifying each constructs inside budgets, and how far a runaway step
#: gets before its budget stops it.
_SCRIPT = """
import json, sys
import sympy as sp
from rederive.engine import budget
from rederive.engine.computing import simplify
from rederive.syntax import ParseState, parse_expression

def simplified(text):
    return simplify(parse_expression(text, ParseState()).node).text

for text in sys.argv[2:]:
    simplified(text)
counts = []
real = budget.within

def counting(limit, compute):
    value, count = budget.counted(lambda: real(limit, compute))
    counts.append(count)
    return value

budget.within = counting
for module in ("pipeline", "normal", "evaluate"):
    setattr(sys.modules["rederive.engine." + module], "within", counting)
answer = simplified(sys.argv[1])
budget.within = real
x = sp.Symbol("x")
progress = []

def runaway():
    sp.core.cache.clear_cache()
    total = sp.Integer(0)
    while True:
        total = sp.expand(total + (x + len(progress)) ** 3)
        progress.append(total)

real(200_000, runaway)
print(json.dumps([answer, counts, len(progress)]))
"""

_INPUT = "(SQRT(a) + SQRT(b))/(SQRT(a) - SQRT(b)) + SIN(x)^4 - COS(x)^4 + 2*COS(x)^2"


def _measured(*before: str, load: bool = False) -> list:
    """What `_SCRIPT` prints, run in a fresh interpreter with a fixed hash seed, as the
    worker runs, after simplifying `before`; with `load`, beside processes that keep
    cores busy."""
    environment = {**os.environ, "PYTHONHASHSEED": "0"}
    hogs = []
    if load:
        hogs = [
            subprocess.Popen([sys.executable, "-c", "while True: pass"])
            for _ in range(min(4, os.cpu_count() or 1))
        ]
    try:
        output = subprocess.run(
            [sys.executable, "-c", _SCRIPT, _INPUT, *before],
            capture_output=True,
            text=True,
            check=True,
            env=environment,
            timeout=600,
        ).stdout
    finally:
        for hog in hogs:
            hog.kill()
            hog.wait()
    return json.loads(output)


def test_the_work_of_a_command_is_the_same_in_every_run():
    """In fresh processes, under load or not, and after other commands: the steps of
    a command construct the same number of objects, and a step that doesn't end is
    stopped at the same point."""
    first = _measured()
    assert first[1] and first[2] > 0
    assert _measured(load=True) == first
    assert _measured("INT(x^2*SIN(x)^3, x)", "SOLVE(x^3 - 2*x = 1, x)") == first

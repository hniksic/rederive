"""An answer must not depend on the process it was computed in.

Python randomizes string hashing per process, so the iteration order of a set or a dict
built from one changes from run to run. Sympy expressions hash through their names, so
any step that picks among equals in that order (which stand-in a subexpression gets,
which rewrite is tried first) can give a different, equally valid, answer in each run.
Sympy's own random generators start from the system's entropy in each process unless
seeded (see `rederive.engine.fresh`). A result that changes between runs is a defect.
These cases are computed in fresh interpreters under several hash seeds and must agree.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

#: Inputs whose simplification goes through the steps that choose among equals:
#: rationalizing denominators over hidden roots, combining powers, solving systems,
#: integrating by substitution, trigonometric and logarithmic rewrites.
CASES = [
    "Trigonometry:=Collect",
    "Trigpower:=Cosines",
    "y=r*SQRT(1-(SQRT(2)*SQRT((x+r)/r)/2)^2)"
    "*(2*(SQRT(2)*SQRT((x+r)/r)/2)-1/(SQRT(2)*SQRT((x+r)/r)/2))",
    "Trigonometry:=Auto",
    "Trigpower:=Auto",
    "(SQRT(a) + SQRT(b))/(SQRT(a) - SQRT(b))",
    "1/(SQRT(x + 1) - SQRT(x - 1))",
    "(x^(3/2) + 1)/(SQRT(x^3) - 1)",
    "SQRT(x*y)*SQRT(x/y) + SQRT(x^2*y^2)",
    "SIN(a + b)*COS(a - b) + COS(a + b)*SIN(a - b)",
    "SIN(x)^4 - COS(x)^4 + 2*COS(x)^2",
    "LN(a*b) - LN(a) - LN(b)",
    "#e^(a + b)*#e^(-a)/(#e^b + #e^(-b))",
    "SOLVE([x + y + z = 6, x - y = 1, x*z = 3], [x, y, z])",
    "SOLVE(a*x^2 + b*x + c = 0, x)",
    "SOLVE(x^3 - 3*a*x + b = 0, x)",
    "INT(1/(SQRT(x) + SQRT(x + 1)), x)",
    "INT(x*ABS(x - a), x)",
    "INT(SIN(x)^3*COS(x)^2, x)",
    "INT(1/(x^3 + a^3), x)",
    "DIF(ATAN(x/a) + ASIN(y/a), x)",
    "LIM((SQRT(x + a) - SQRT(a))/x, x, 0)",
    "SUM(k*q^k, k, 1, n)",
    "FACTOR(x^4 + y^4, Radical)",
    "EXPAND((a - b + c)^4)",
    "(a^2 - b^2)/(a + b) + (c^3 - d^3)/(c - d)",
    "[SQRT(u*v), SQRT(v*w), SQRT(u*w)]*SQRT(u*v*w)",
    "ABS(x*y)*SIGN(x)*SIGN(y)",
    "x^n*y^n*(x*y)^(-n)",
]

_SCRIPT = """
import asyncio, json, sys
from rederive.model.session import Session
from rederive.syntax.writer import write_expression
session = Session()
session.author("InputMode := Word")
answers = []
for text in json.loads(sys.argv[1]):
    entry = session.author(text)
    answer = asyncio.run(session.simplify(f"#{entry.number}"))
    answers.append(write_expression(answer.node))
print(json.dumps(answers))
"""


def _started(seed: str) -> subprocess.Popen:
    environment = dict(os.environ, PYTHONHASHSEED=seed)
    return subprocess.Popen(
        [sys.executable, "-c", _SCRIPT, json.dumps(CASES)],
        env=environment,
        stdout=subprocess.PIPE,
        text=True,
    )


def test_answers_are_the_same_under_every_hash_seed():
    processes = [_started(seed) for seed in ("1", "2", "3", "4")]
    first, *others = (
        json.loads(process.communicate(timeout=600)[0]) for process in processes
    )
    assert len(first) == len(CASES)
    for other in others:
        assert other == first

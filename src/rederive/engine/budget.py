"""Work budgets for optional steps that may not end in reasonable time.

What the user asks for runs until it ends or the user interrupts it, as in Derive: a
`SOLVE`, an integral or a factorization has no limit, and Esc (which kills the worker)
is the only way to stop it. But Simplify also takes steps nobody asked for, each an
identity that only makes the answer shorter or tidier: a rewrite rule, a cancellation
in the normal form. On some inputs a moment's change away from one done in
milliseconds, such a step can run for hours inside sympy, which can't be told to stop.
`within` gives such a step a budget: it runs the step, and if the step does more work
than the budget allows, it is interrupted and `within` returns None, and the step is
left undone. Leaving an identity unapplied is always sound.

The work is counted, not timed, so that an answer is a function of the input alone:
the same on a fast machine and a slow one, under load or idle, on every platform and
on any thread. The work is counted in calls, by `sys.monitoring` in the thread whose
budget runs: a construction of a sympy object (a call of `Basic.__new__`) counts for
ten units, a call of a function of sympy's polynomial arithmetic for one, and filling
a dense polynomial for one per ten coefficients (see `_counting`). Arithmetic on numbers
of more than 2048 bits costs far more than a call: an operation of polynomial
arithmetic on such coefficients, a sympy number of such a size as it is made, and a
primality test count by the size of their numbers (see `_sized`, `_testing`), and so
does evaluating a polynomial of a high degree at a number (see `_horner`). Other
work that one call does without making further such calls is seen only as that call,
and takes that much longer to run out of its budget. A count is reproducible when
three things are fixed: string hashing (the worker runs with a fixed `PYTHONHASHSEED`,
see `remote.py`), sympy's random generators, and its caches and the modules it has
imported (each command starts from the same state of all three, see `fresh.py`). Counts
change with the version of sympy, so a budget is set above the most any step needs in
the test suite and the recorded worksheets, and only a step that would run far longer
is cut off.

Budgets nest: an operation inside another gets at most what is left of the outer one,
and when the outer runs out the interruption passes through the inner boundary to the
outer one.

An interruption is an exception raised at a unit of work, in whatever runs. It leaves
nothing half-done that a later computation could see: sympy's caches store a value
only once it is computed, and the assumptions an object learns are stored as
deduced, each a true fact. What a `finally` clause would have put back is put back
here: the operation runs in a copy of the context variables, and sympy's global
evaluation flags and mpmath's working precision are restored after an interruption.
An exception handler that swallows the interruption doesn't stop it: the next unit of
work raises it again.
"""

from __future__ import annotations

import contextvars
import math
import sys
import threading
from collections.abc import Callable
from typing import Any, TypeVar

__all__ = [
    "CONVERGING",
    "DECIDING",
    "EXTENDED",
    "NORMALIZING",
    "NORMAL_FORM",
    "RESPELLED",
    "REWRITING",
    "TRIGONOMETRIC",
    "OutOfWork",
    "budgeted",
    "counted",
    "within",
]

_T = TypeVar("_T")

# The budgets, in units of work. Each is somewhat above the most that such a step needs
# where it ends, in the test suite and in the recorded worksheets that compute (sympy
# 1.14, the worker's hash seed, the caches emptied at the start of each command), so
# that no budget changes an answer there, and a step that would run far longer is cut
# off after seconds.

#: One rule or step of Simplify, and a display's conditionals folded together. The most
#: a rule needs
#: that ends is 1.3 million units in the suite and 4.0 million in the worksheets (a
#: difference of two answers simplified); a fold, 43,000.
REWRITING = 5_000_000

#: One rule of Simplify that rewrites trigonometric calls through sympy's `trigsimp`
#: (directly, or over a common half angle). The most such a rule needs where its result
#: is kept is 1.3 million units in the suite and 1.4 million in the worksheets; where it
#: changes nothing, it runs up to 4.9 million units first, a dozen seconds.
TRIGONOMETRIC = 2_000_000

#: A limit that judges whether a definite integral converges at an end. None that ends
#: needs as much as 0.2 million units, in the suite or the worksheets.
CONVERGING = 1_000_000

#: One cancellation or combination of quotients in the normal form, and the normal form
#: of one whole expression. The most a cancellation needs is 0.5 million units in the
#: suite and 4.0 million in the worksheets, a whole normal form 7.2 million.
NORMALIZING = 5_000_000
NORMAL_FORM = 10_000_000

#: A calculus head retried with its operands spelled as an answer spells them, once the
#: head as written did not evaluate. The most a retry needs is 0.7 million units, in the
#: suite; none in the worksheets.
RESPELLED = 1_000_000


#: A square-free factorization of a radicand over the field its numeric roots span,
#: done to take powers out of a root. The most one needs is 15,000 units, in the suite.
EXTENDED = 1_000_000

#: A test of whether a system of equations is consistent, or a condition on its
#: parameters zero, done to decide between no solution and a condition: undecided, the
#: condition stays. The most such a test needs is 75,000 units, in the suite.
DECIDING = 1_000_000


class OutOfWork(BaseException):
    """The operation of the innermost `within` whose budget is spent was interrupted.

    A `BaseException`, so that no handler in the engine that turns a failure into an
    unevaluated call takes it for one before it reaches its `within`.
    """


class _Meter:
    """The work of one thread: the units of work done while a budget runs, and the
    count at which each running budget is spent, outermost first."""

    __slots__ = ("count", "ends", "end", "calls")

    def __init__(self) -> None:
        self.count = 0
        self.ends: list[int] = []
        #: The count at which the innermost budget is spent (the least of `ends`).
        self.end = 0
        #: The calls counting by size since the outermost budget started (see
        #: `_worked`).
        self.calls = 0


_threads = threading.local()

#: The `sys.monitoring` tool id taken for counting, None until the first budget, -1
#: where none is free.
_tool: int | None = None
_install = threading.Lock()


def within(limit: int, compute: Callable[[], _T]) -> _T | None:
    """`compute()`, or None if it does more than `limit` units of work.

    Only the operation's own boundary may catch the interruption: an outer budget that
    ran out too is passed on to its own `within`.
    """
    meter = _meter()
    level = len(meter.ends)
    globals_before = _globals()
    end = meter.count + limit
    if level:
        end = min(end, meter.end)
    if not level:
        meter.calls = 0
    meter.ends.append(end)
    meter.end = end
    try:
        return contextvars.copy_context().run(compute)
    except OutOfWork:
        _restore(globals_before)
        if level and meter.count >= meter.ends[level - 1]:
            raise
        return None
    finally:
        del meter.ends[level:]
        meter.end = meter.ends[-1] if meter.ends else 0


def counted(compute: Callable[[], _T]) -> tuple[_T, int]:
    """`compute()` and the units of work it did, with no budget of its own (inside one,
    it is still interrupted where that one runs out): for measuring what a step needs."""
    meter = _meter()
    level = len(meter.ends)
    if not level:
        meter.calls = 0
    meter.ends.append(meter.end if level else sys.maxsize)
    meter.end = meter.ends[-1]
    before = meter.count
    try:
        value = compute()
    finally:
        del meter.ends[level:]
        meter.end = meter.ends[-1] if meter.ends else 0
    return value, meter.count - before


def budgeted() -> bool:
    """Whether a budget runs in this thread, so that the work done counts."""
    meter = getattr(_threads, "meter", None)
    return meter is not None and bool(meter.ends)


def _meter() -> _Meter:
    """This thread's meter, with the counting switched on in the process."""
    meter = getattr(_threads, "meter", None)
    if meter is None:
        if _tool is None:
            _counting()
        meter = _threads.meter = _Meter()
    return meter


def _worked(code: object, offset: int) -> None:
    """Count the work of one call in a thread whose budget runs, and interrupt the
    operation if that spends it."""
    meter = getattr(_threads, "meter", None)
    if meter is None or not meter.ends:
        return
    if code is _constructing:
        meter.count += _CONSTRUCTION
    elif code is _filling:
        meter.count += 1 + _degree(sys._getframe(1).f_locals.get("f")) // _FILLED
    elif code is _evaluating:
        meter.count += 1 + _horner(sys._getframe(1).f_locals)
    elif code in _sized:
        # The size is looked at in one call of every `_SAMPLED` and counts for all of
        # them: these calls are frequent, and reading a frame's arguments costs more
        # than the call often does. The calls are counted from the start of the
        # outermost budget, so the count stays a function of the work alone.
        base, size = _sized[code]
        meter.calls += 1
        meter.count += base
        if not meter.calls % _SAMPLED:
            meter.count += _SAMPLED * size(sys._getframe(1).f_locals)
    elif code in _testing:
        meter.count += 1 + _primality_size(sys._getframe(1).f_locals.get("n"))
    else:
        meter.count += 1
    if meter.count >= meter.end:
        raise OutOfWork


def _degree(terms: object) -> int:
    """The degree of a univariate polynomial given by its terms, `dup_from_dict`'s
    argument: the length of the dense list it fills."""
    if not isinstance(terms, dict) or not terms:
        return 1
    return 1 + max(
        monomial if isinstance(monomial, int) else monomial[0] for monomial in terms
    )


def _horner(arguments: Any) -> int:
    """The units of work of `dup_eval`, a dense polynomial `f` evaluated at `a` by
    Horner's rule, beyond its call: the partial value grows by the size of `a` at each
    of the `n` coefficients, so the multiplications cost `n^2` times that size, which
    for an integer of a thousand bits and a polynomial of degree ten thousand takes
    seconds."""
    terms, point = arguments.get("f"), arguments.get("a")
    size = getattr(point, "bit_length", None)
    if not isinstance(terms, list) or size is None:
        return 0
    return len(terms) ** 2 * size() // _EVALUATED


def _bits(arguments: Any, names: tuple[str, ...]) -> int:
    """The bits of the integers `names` among a call's `arguments`, together."""
    bits = 0
    for name in names:
        value = arguments.get(name)
        if isinstance(value, int):
            bits += value.bit_length()
    return bits


def _pieces(bits: int) -> int:
    """The units of work of one operation on numbers of `bits` bits together, beyond
    the call it is part of: none up to 2048 bits, and beyond, the number `k` of
    2048-bit pieces to the power 3/2, which multiplying them costs."""
    pieces = bits >> 11
    return pieces * math.isqrt(pieces)


def _number_size(names: tuple[str, ...]) -> Callable[[Any], int]:
    """The units of work of a sympy integer or rational made from the integers `names`,
    the result of an operation on numbers (see `_pieces`)."""

    def size(arguments: Any) -> int:
        return _pieces(_bits(arguments, names))

    return size


def _coefficient_bits(coefficient: object) -> int:
    """The bits of an integer or rational coefficient of a dense polynomial, 0 for any
    other."""
    if isinstance(coefficient, int):
        return coefficient.bit_length()
    numerator = getattr(coefficient, "numerator", None)
    if isinstance(numerator, int):
        denominator = coefficient.denominator  # type: ignore[attr-defined]
        return numerator.bit_length() + denominator.bit_length()
    return 0


def _dense_size(operands: tuple[str, ...], product: bool) -> Callable[[Any], int]:
    """The units of work of an operation on dense polynomials `operands`, each a list of
    coefficients or a coefficient, beyond its call: an operation on numbers the size of
    their leading coefficients (see `_pieces`) for each pair of coefficients
    multiplied in a product, or each coefficient of the longest otherwise. Rationals
    over each, as polynomial arithmetic over the rationals keeps them in lowest terms,
    count four times. A polynomial with coefficients of millions of digits takes
    seconds per operation."""

    def size(arguments: Any) -> int:
        bits = 0
        rational = False
        lengths = []
        for name in operands:
            value = arguments.get(name)
            if isinstance(value, list):
                if not value:
                    return 0
                lengths.append(len(value))
                value = value[0]
            rational = rational or not isinstance(value, int)
            bits += _coefficient_bits(value)
        units = _pieces(bits)
        if not units:
            return 0
        terms = math.prod(lengths) if product else max(lengths, default=1)
        return terms * units * (4 if rational else 1)

    return size


def _primality_size(number: object) -> int:
    """The units of work of a primality test on `number` beyond its call: the cube of
    the number of its 128-bit pieces, which is how modular powers grow."""
    if not isinstance(number, int):
        return 0
    return (abs(number).bit_length() >> 7) ** 3


#: The units of work one construction of a sympy object counts for, against one for a
#: call of polynomial arithmetic: a step constructs from 5,000 to 75,000 objects in a
#: second, and calls polynomial arithmetic up to 700,000 times, so that a unit stands
#: for a more even share of time whichever it does.
_CONSTRUCTION = 10

#: The code of `Basic.__new__`, once counting is on.
_constructing: object = None

#: The code of the functions whose work grows with the size of the numbers they work
#: on, each with the units its call counts for and a function of its arguments giving
#: the units beyond: the constructors of sympy's integers and rationals, which a cached
#: number does not reach (see `_number_size`), and the arithmetic of dense polynomials
#: (see `_dense_size`).
_sized: dict[object, tuple[int, Callable[[Any], int]]] = {}

#: One in how many calls that count by size `_worked` looks at the size in.
_SAMPLED = 16

#: The functions of dense polynomial arithmetic that count by the size of their
#: coefficients (see `_dense_size`): name, the operands, and whether the operation
#: multiplies each coefficient of one by each of the other.
_DENSE = (
    ("dup_mul", ("f", "g"), True),
    ("dup_sqr", ("f", "f"), True),
    ("dup_mul_term", ("f", "c"), False),
    ("dup_mul_ground", ("f", "c"), False),
    ("dup_quo_ground", ("f", "c"), False),
    ("dup_exquo_ground", ("f", "c"), False),
    ("dup_add", ("f", "g"), False),
    ("dup_sub", ("f", "g"), False),
)

#: The code of the primality tests, which count by the size of the number tested (see
#: `_primality_size`).
_testing: frozenset = frozenset()

#: The code of `dup_from_dict`, which fills a dense list of coefficients, one per
#: degree, without calling anything: it counts a unit for each `_FILLED` of them, before
#: it starts.
_filling: object = None

#: How many coefficients `dup_from_dict` fills in the time of a unit of work: a
#: coefficient takes a tenth of what a call of polynomial arithmetic does.
_FILLED = 10

#: The code of `dup_eval`, which evaluates a dense polynomial at a number in one
#: call, by the size of both (see `_horner`).
_evaluating: object = None

#: The products of a polynomial's length squared and the bits of the point at which it
#: is evaluated (see `_horner`) that take the time of a unit of work.
_EVALUATED = 25_000


#: The modules of sympy's polynomial arithmetic and primality tests, each call of a
#: function of which is a unit of work (see `_counting`).
_POLYNOMIAL_MODULES = (
    "sympy.polys.densearith",
    "sympy.polys.densetools",
    "sympy.polys.euclidtools",
    "sympy.polys.factortools",
    "sympy.polys.sqfreetools",
    "sympy.polys.galoistools",
    "sympy.ntheory.primetest",
)

#: The arithmetic of sparse polynomials (`sympy.polys.rings.PolyElement`), each call of
#: which is a unit of work.
_SPARSE_ARITHMETIC = (
    "__add__ __radd__ __sub__ __rsub__ __mul__ __rmul__ __pow__ __truediv__ div rem "
    "quo exquo gcd cofactors cancel"
).split()


#: The zero polynomials of several variables, which a dense polynomial of a huge degree
#: is filled with one by one: each is a unit of work (see also `_filling`).
_DENSE_ZEROS = ("dmp_zero", "dmp_zeros")


def _counting() -> None:
    """Switch the counting of work on, once per process.

    The units are the constructions of sympy objects (calls of `Basic.__new__`), and
    the calls of the functions of polynomial arithmetic: factoring over an extension or
    a polynomial gcd can run for minutes constructing hardly any object. The events
    stay on, and a thread that runs no budget counts nothing, which costs a few
    percent of the time of polynomial arithmetic.
    """
    global _tool, _constructing, _filling, _evaluating, _testing
    with _install:
        if _tool is not None:
            return
        import importlib
        import types

        from sympy.core.basic import Basic
        from sympy.core.numbers import Integer, Rational
        from sympy.polys import densearith, densebasic, densetools
        from sympy.polys.rings import PolyElement

        monitoring = sys.monitoring
        free = [tool for tool in (4, 3) if monitoring.get_tool(tool) is None]
        if not free:
            _tool = -1
            return
        tool = free[0]
        monitoring.use_tool_id(tool, "rederive work budget")
        monitoring.register_callback(tool, monitoring.events.PY_START, _worked)
        _constructing = Basic.__new__.__code__
        codes = [_constructing]
        for name in _POLYNOMIAL_MODULES:
            module = importlib.import_module(name)
            codes.extend(
                value.__code__
                for value in vars(module).values()
                if isinstance(value, types.FunctionType) and value.__module__ == name
            )
        codes.extend(getattr(PolyElement, name).__code__ for name in _SPARSE_ARITHMETIC)
        codes.extend(getattr(densebasic, name).__code__ for name in _DENSE_ZEROS)
        _filling = densebasic.dup_from_dict.__code__
        codes.append(_filling)
        _evaluating = densetools.dup_eval.__code__
        _sized.update(
            {
                Integer.__new__.__wrapped__.__code__: (0, _number_size(("i",))),
                Rational.__new__.__wrapped__.__code__: (0, _number_size(("p", "q"))),
            }
        )
        codes.extend(_sized)
        for name, operands, product in _DENSE:
            code = getattr(densearith, name).__code__
            _sized[code] = (1, _dense_size(operands, product))
        primality = importlib.import_module("sympy.ntheory.primetest")
        _testing = frozenset(
            value.__code__
            for value in vars(primality).values()
            if isinstance(value, types.FunctionType)
            and value.__module__ == primality.__name__
            and "n" in value.__code__.co_varnames[: value.__code__.co_argcount]
        )
        for code in codes:
            monitoring.set_local_events(tool, code, monitoring.events.PY_START)
        _tool = tool


def _globals() -> tuple[Any, ...]:
    """The process-wide settings sympy and mpmath change and put back in `finally`
    clauses, which an interruption may skip."""
    sympy = sys.modules.get("sympy.core.parameters")
    mpmath = sys.modules.get("mpmath")
    flags = None
    if sympy is not None:
        found = sympy.global_parameters
        flags = (found.evaluate, found.distribute, found.exp_is_pow)
    return flags, None if mpmath is None else mpmath.mp.prec


def _restore(saved: tuple[Any, ...]) -> None:
    """Put back the settings `_globals` saved, where they changed."""
    flags, precision = saved
    if flags is not None:
        found = sys.modules["sympy.core.parameters"].global_parameters
        if (found.evaluate, found.distribute, found.exp_is_pow) != flags:
            found.evaluate, found.distribute, found.exp_is_pow = flags
    if precision is not None:
        mpmath = sys.modules["mpmath"]
        if mpmath.mp.prec != precision:
            mpmath.mp.prec = precision

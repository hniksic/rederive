"""Turning an expression into numbers: the closure, the sampler, the grid.

Pure functions over numpy arrays, with no Qt anywhere near them, because this
is where the corner cases live and corner cases want to be testable without a
screen. Everything here runs on the host's sampling thread and never on its Qt
thread.

Three ideas carry the whole module.

The first is that an expression is converted *once*. The algebraic form of what
is being plotted must not affect how fast it draws, so the tree goes through the
engine's own pre-pass and converter and comes out a lambdified closure that is
kept for the life of the curve; re-sampling a zoom then costs microseconds and
sympy is never asked anything again. Where numpy has no reading of a head at
all - `x!` is the plainest of them - sympy is asked after all, a point at a
time, because what plots should be what approximates and not what numpy has a
name for. That rung is under a probe and is reached only where the fast one has
already failed, so nothing it exists for is paid for by anything else.

The second is that everything is evaluated in the complex plane and masked back
to the reals afterwards. `SQRT(x)` over a range that includes the negative half
is not an error, it is a curve with a beginning: evaluating on `complex128` and
keeping the values whose imaginary part is numerical noise draws exactly the
real part of the graph and nothing else. Masking is never silent, though - a
curve with no finite point in view says so on the status bar, which is what the
finite count is read for.

The third is that sampling is adaptive in *screen* space. A tolerance in the
units of the expression means nothing: a quarter of a pixel is what a person can
see, so the deviation of a midpoint from its chord is measured after the view
transform, and every curve is therefore sampled to the same visible accuracy
whatever it is a curve of. That is also what makes the discontinuity guard
possible - refinement that bottoms out at subpixel width with the samples still
far apart, and no nearer to each other than before it was cut, has found a jump
rather than a slope, and a jump is where a NaN goes so that nothing draws a
stroke across it. The last of those conditions is what keeps the guard off a
curve that is merely steeper than the screen: bisecting a slope halves its
height and bisecting a jump does not, so `x·SIN(x)` zoomed out until it
oscillates faster than the pixels is drawn as the band it is.

There are two samplers rather than one because a curve parametrized by t is a
different measurement from a curve parametrized by its own abscissa: the error
of a midpoint is a distance in the plane and not a height, and an interval is
narrow when its two ends land on the same pixel rather than when its parameters
are close. Everything else - the tolerance, the depth cap, the gap through a
jump - is the same argument twice.

`features` is the other half of a plot as a measuring instrument. What it finds
is found on the samples that are already drawn and then refined on the closure,
so a root is the function's own root to full precision and not the pixel the
curve happened to cross the axis in.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from rederive.engine.context import Context
from rederive.model.expr import Node

__all__ = [
    "CROSSING",
    "INITIAL_POINTS",
    "MAXIMUM",
    "MAX_DEPTH",
    "MINIMUM",
    "ROOT",
    "Boundary",
    "Feature",
    "Sampled",
    "Unplottable",
    "closure",
    "difference",
    "features",
    "finite_fraction",
    "grid_boundary",
    "grid_eval",
    "mask",
    "number",
    "pair",
    "points",
    "polar_pair",
    "sample_adaptive",
    "sample_curve",
]

#: How close to real a complex answer has to be to count as real, relative to
#: its own magnitude. Slack enough to swallow the rounding of a long numeric
#: chain, tight enough that a genuinely complex branch is dropped.
REAL_TOLERANCE = 1e-9

#: The uniform pass, before any refinement: enough points that a curve appears
#: in outline at once, few enough that a slow expression is not paid for twice.
INITIAL_POINTS = 129

#: How far refinement may bisect. Twelve levels take an interval of the initial
#: grid to a four-thousandth of itself, which is subpixel on any canvas.
MAX_DEPTH = 12

#: How far a midpoint may sit from its chord before the interval is bisected,
#: in pixels. A quarter of one is under what a person can see and over what
#: floating point noise produces.
TOLERANCE_PX = 0.25

#: How narrow an interval has to get before refinement stops caring, in pixels.
#: Below this the two samples land on the same pixel column.
NARROWEST_PX = 0.25

#: How tall a jump has to be, in pixels, to be called a discontinuity when
#: refinement has already bottomed out on it. The visible height is the other
#: sufficient reason, and it is the one a pole meets.
JUMP_PX = 8.0

#: How much of its height a jump has to keep when it is bisected, as a fraction
#: of the height the interval had before the split, for it to still count as a
#: jump. A discontinuity keeps all of it however finely it is cut, because the
#: whole of the step is in whichever half the step is in. A slope hands half of
#: it to each half, and goes on doing so however steep it is.
HELD_FRACTION = 0.75

#: How many samples one curve may end up with. Refinement only bisects where
#: it must, so no ordinary expression comes near this; it is what stops a
#: pathological one from eating the host's memory.
MAX_POINTS = 20001

#: How many points the exact rung is offered before it is given up on. One is
#: not enough: the first sample of a curve can perfectly well be a pole, and a
#: pole is a value the evaluator found rather than a sign that it cannot
#: evaluate. Spread across the sample, these settle the question for the cost
#: of a rounding error on a plot that was going to be slow anyway.
PROBE_POINTS = 8


class Unplottable(Exception):
    """The expression cannot be turned into numbers, and this is why.

    Raised by `closure` for everything that goes wrong before the first sample:
    an operator with no numeric reading, a conversion that fails, a lambdify
    that will not compile. The host reports it as an event against the curve
    rather than drawing nothing.
    """


def closure(
    node: Node, context: Context, variables: Sequence[str] = ()
) -> Callable[..., np.ndarray]:
    """A numeric function of `variables` for the expression `node` names.

    The pre-pass is the worker's own - `substitute` resolves labels, assigned
    variables and defined functions on the tree, before sympy sees any of it -
    so a plot of `#3` plots what `#3` means and a plot of `f(x)` plots the
    function the worksheet defined.

    What comes back takes arrays and returns an array of the same shape: real
    where the value is real and finite, NaN everywhere else. It never raises,
    whatever it is handed, because it is called from the middle of a sampling
    loop where an exception would cost the whole curve rather than one point:
    a vectorized call that raises is retried point by point, a point that
    raises is asked of sympy itself, and a point sympy will not answer is NaN.
    """
    return _lambdified(_converted(node, context), variables)


def difference(
    node: Node, context: Context, variables: Sequence[str] = ()
) -> Callable[..., np.ndarray]:
    """`u - v` as a function, for the implicit plot of the equation `u = v`.

    An implicit curve is the zero contour of the difference of the two sides,
    so the equation is turned into one expression here rather than anywhere
    else: the marching squares that draws it wants a scalar field, and `u = v`
    is not one until it has been subtracted.
    """
    expression = _converted(node, context)
    sides = (getattr(expression, "lhs", None), getattr(expression, "rhs", None))
    if sides[0] is not None and sides[1] is not None:
        expression = sides[0] - sides[1]
    return _lambdified(expression, variables)


def mask(
    node: Node, context: Context, variables: Sequence[str] = ()
) -> Callable[..., np.ndarray]:
    """A truth value per point, for the shaded region of an inequality.

    Evaluated on real inputs rather than complex ones, which is the one place
    this module leaves the complex plane: `x > 0` has no reading for a complex
    x, sympy says so by raising, and a region is drawn of the reals anyway.
    What comes back is a boolean array, false wherever the answer was not a
    truth value at all.
    """
    inside = _lambdified(_converted(node, context), variables, real=True)

    def truth(*arguments: np.ndarray) -> np.ndarray:
        values = inside(*arguments)
        return np.asarray(np.nan_to_num(values, nan=0.0) > 0.5, dtype=bool)

    return truth


def pair(
    node: Node, context: Context, variables: Sequence[str] = ()
) -> tuple[Callable[..., np.ndarray], Callable[..., np.ndarray]]:
    """The two closures of a parametric pair, converted once together.

    `[x(t), y(t)]` is one expression to the user and two functions to the
    sampler, and the conversion that makes them is the expensive half, so it
    happens once for the pair.
    """
    expression = _converted(node, context)
    try:
        elements = list(expression)
    except TypeError:
        raise Unplottable("a parametric plot needs two expressions") from None
    if len(elements) != 2:
        raise Unplottable("a parametric plot needs two expressions")
    return (
        _lambdified(elements[0], variables),
        _lambdified(elements[1], variables),
    )


def polar_pair(
    f: Callable[..., np.ndarray], degrees: bool = False
) -> tuple[Callable[..., np.ndarray], Callable[..., np.ndarray]]:
    """`r = f(θ)` as the parametric pair the sampler draws it from.

    The angle is the expression's own, so the turn from θ to a point has to be
    made in the units the worksheet is set to: in degree mode `SIN(θ)` is a
    function of degrees, and composing it with a radian cosine would draw a
    rose with the wrong number of petals.
    """
    turn = np.pi / 180.0 if degrees else 1.0

    def horizontal(t: np.ndarray) -> np.ndarray:
        return f(t) * np.cos(np.asarray(t, dtype=np.float64) * turn)

    def vertical(t: np.ndarray) -> np.ndarray:
        return f(t) * np.sin(np.asarray(t, dtype=np.float64) * turn)

    return horizontal, vertical


def points(node: Node, context: Context) -> tuple[np.ndarray, np.ndarray]:
    """The columns of a constant matrix, as the x and y of a data plot.

    A single point is a one-row matrix and reads the same way, which is what
    makes `[1, 2]` and a hundred thousand rows the same plot kind.
    """
    import sympy as sp

    expression = _converted(node, context)
    try:
        rows = np.array(sp.Matrix(expression).evalf().tolist(), dtype=np.complex128)
    except Exception as error:
        raise Unplottable(_said(error)) from None
    if rows.ndim != 2 or rows.shape[1] != 2:
        raise Unplottable("a data plot needs a matrix of two columns")
    return _real_part(rows[:, 0]), _real_part(rows[:, 1])


def number(node: Node | None, context: Context, default: float) -> float:
    """What a bound written as an expression is worth, `default` where nothing.

    A range or a domain is typed into a plot window's toolbar, where `-π` is
    what a person types; turning it into a float is arithmetic, so it happens
    here, on the sampling thread, and never where the text was typed.
    """
    if node is None:
        return default
    try:
        value = float(np.asarray(closure(node, context, ())()))
    except Exception:
        return default
    return value if np.isfinite(value) else default


def _converted(node: Node, context: Context) -> object:
    """The engine's reading of `node` as a sympy object.

    The pre-pass is the worker's own - `substitute` resolves labels, assigned
    variables and defined functions on the tree, before sympy sees any of it.
    """
    from rederive.engine.substitute import substitute
    from rederive.engine.to_sympy import to_sympy

    try:
        return to_sympy(substitute(node, context), context)
    except Exception as error:
        raise Unplottable(_said(error)) from None


def _lambdified(
    expression: object, variables: Sequence[str], real: bool = False
) -> Callable[..., np.ndarray]:
    """`expression` as a numeric function of `variables`, in that order."""
    import sympy as sp

    named = {
        symbol.name: symbol
        for symbol in getattr(expression, "free_symbols", set())
        if isinstance(symbol, sp.Symbol)
    }
    symbols = [named.get(name, sp.Symbol(name)) for name in variables]
    try:
        function = sp.lambdify(symbols, expression, modules="numpy")
    except Exception as error:
        raise Unplottable(_said(error)) from None
    return _Numeric(function, expression, symbols, real=real)


def _said(error: Exception) -> str:
    """One line naming an exception, for a message line to carry."""
    text = str(error).strip().splitlines()
    return text[0] if text else type(error).__name__


def _exact(expression: Any, symbols: Sequence[Any]) -> list[Callable[..., Any]]:
    """The evaluators to fall back on, in the order they are worth trying.

    Lambdify's numpy namespace is narrower than the arithmetic the engine
    itself does, and the gap is not a short list of special cases: a head numpy
    has no entry for is printed as a bare name and picks up whatever python
    happens to have under it, which for `factorial` is `math.factorial` and
    refuses every float it is handed. Anything the rest of the program can
    approximate should therefore be plottable, and these are what make that
    true - the same value the engine's precision mode would show, arrived at
    the same way.

    Mpmath first, because it is the same arithmetic evalf runs on with the
    dispatch taken off the front, and measures three to seventy times faster
    per point for it. Sympy's own evaluation second, for the expression mpmath
    has no reading of at all: it is the slower of the two and the one that
    cannot be wrong about what the engine would say.
    """
    import sympy as sp

    candidates: list[Callable[..., Any]] = []
    try:
        candidates.append(sp.lambdify(symbols, expression, modules="mpmath"))
    except Exception:
        pass
    candidates.append(_Evalf(expression, symbols))
    return candidates


class _Evalf:
    """One point of `expression`, evaluated by sympy the way the engine does."""

    def __init__(self, expression: Any, symbols: Sequence[Any]) -> None:
        self._expression = expression
        self._symbols = list(symbols)

    def __call__(self, *arguments: Any) -> complex:
        values = [_as_number(argument) for argument in arguments]
        bound = dict(zip(self._symbols, values, strict=True))
        return complex(self._expression.subs(bound).evalf())


def _as_number(value: Any) -> Any:
    """A numpy scalar as the sympy number it stands for.

    A real input stays real rather than becoming a complex with nothing in its
    imaginary part, because the two are different questions to ask sympy: the
    factorial of 2.5 is a number and the factorial of `2.5 + 0i` is a branch.
    """
    import sympy as sp

    number = complex(value)
    if number.imag:
        return sp.Float(number.real) + sp.Float(number.imag) * sp.I
    return sp.Float(number.real)


def _probes(size: int) -> list[int]:
    """Which points the exact rung is tried at, spread across the sample."""
    if size <= PROBE_POINTS:
        return list(range(size))
    step = size / PROBE_POINTS
    return [int(index * step) for index in range(PROBE_POINTS)]


class _Numeric:
    """A lambdified expression, evaluated in the complex plane and masked back.

    The wrapper is where every promise of section 8 is kept: complex inputs so
    that a radical has a real part to find, the real mask, non-finite to NaN,
    numpy's warnings silenced - a plot of `1/x` divides by zero by design and
    the terminal is not the place to hear about it - and the retries, which
    some sympy functions lambdify imperfectly enough to need.

    The complex plane is where the evaluation starts and not where it has to
    end. A few of numpy's functions are defined over the reals and nowhere
    else - `arctan2`, `floor`, `ceiling`, `Mod` - and they refuse a complex
    argument by its dtype rather than by its value, so a plot of `ATAN(-y, x)`
    would be an empty window if the first attempt were the only one. The real
    parts of the inputs are therefore tried next, which is the reading those
    functions do have, and nothing is lost by it: an input whose imaginary part
    is more than noise is already outside the graph and comes back NaN.

    Below both of those is the exact rung, which is what makes the set of
    things that plot the set of things that approximate rather than the set of
    things numpy has a name for. It is reached only where the lambdified
    function has already failed at a point, so no expression numpy can evaluate
    pays anything for it, and it is gated on a probe so that no expression
    nothing can evaluate pays for it twice: a few points decide whether an
    exact evaluator answers at all, and where none does the sample is NaN as
    before.
    """

    def __init__(
        self,
        function: Callable[..., object],
        expression: Any,
        symbols: Sequence[Any],
        real: bool = False,
    ) -> None:
        self._function = function
        self._expression = expression
        self._symbols = list(symbols)
        self._arity = len(self._symbols)
        #: Whether the inputs stay real. Only a relation asks for that: `x > 0`
        #: has no reading in the complex plane, and sympy says so by raising.
        self._dtype = np.float64 if real else np.complex128
        #: The exact evaluators, built at most once. Kept for the life of the
        #: curve like the lambdified function is, since building one means
        #: lambdifying again and a zoom must not pay for that.
        self._candidates: list[Callable[..., Any]] | None = None

    def __call__(self, *arguments: np.ndarray) -> np.ndarray:
        inputs = [np.asarray(argument, dtype=self._dtype) for argument in arguments]
        shape = np.broadcast(*inputs).shape if inputs else ()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(all="ignore"):
                values = self._values(inputs, shape)
                if values.shape != shape:
                    # A constant expression lambdifies to a function that
                    # ignores its arguments and answers with one number.
                    values = np.broadcast_to(values, shape).copy()
                return _real_part(values)

    def _values(self, inputs: list[np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
        """What the function answers over `inputs`, however it can be got.

        The vectorized call over each reading of the inputs in turn, and a
        point at a time if none of them evaluated.
        """
        for attempt in self._attempts(inputs):
            try:
                return np.asarray(self._function(*attempt), dtype=np.complex128)
            except Exception:
                continue
        return self._pointwise(inputs, shape)

    def _attempts(self, inputs: list[np.ndarray]) -> list[list[np.ndarray]]:
        """The inputs as they stand, and then their real parts.

        One reading when the inputs are real already, since the second would be
        the first call over again.
        """
        if self._dtype is np.float64 or not inputs:
            return [inputs]
        return [inputs, [_real_part(value) for value in inputs]]

    def _pointwise(
        self, inputs: list[np.ndarray], shape: tuple[int, ...]
    ) -> np.ndarray:
        """One point at a time, for a function the vectorized call choked on.

        The lambdified function first, since where it answers it answers
        fastest, and the exact rung under it for the points it refuses. The
        probe is run at the first such point rather than up front, so a
        function that only stumbles over a pole never builds an evaluator it
        has no use for.
        """
        spreads = [
            [np.broadcast_to(value, shape).ravel() for value in attempt]
            for attempt in self._attempts(inputs)
        ]
        answers = np.full(int(np.prod(shape)) if shape else 1, np.nan, np.complex128)
        exact: Callable[..., Any] | None = None
        probed = False
        for index in range(answers.size):
            value = _at(self._function, spreads, index)
            if value is None:
                if not probed:
                    exact, probed = self._probed(spreads, answers.size), True
                if exact is not None:
                    value = _at(exact, spreads, index)
            answers[index] = np.nan if value is None else value
        return answers.reshape(shape)

    def _probed(
        self, spreads: list[list[np.ndarray]], size: int
    ) -> Callable[..., Any] | None:
        """Whichever exact evaluator answers over this sample, or none of them.

        Asked afresh for every sampling rather than remembered, because it is a
        question about the points in view: a head undefined over the whole of
        one window can be the graph of the next one along. What is remembered
        is the evaluators themselves, which is the part that costs anything to
        make.
        """
        if self._candidates is None:
            self._candidates = _exact(self._expression, self._symbols)
        for candidate in self._candidates:
            if any(_at(candidate, spreads, place) is not None for place in _probes(size)):
                return candidate
        return None


def _at(
    function: Callable[..., Any], spreads: list[list[np.ndarray]], index: int
) -> complex | None:
    """What `function` is worth at one point, or None where it will not say.

    None rather than NaN, because the two mean different things here: NaN is a
    value the evaluator found and a point the graph does not pass through,
    while None is the evaluator declining and the reason to try the next one.
    """
    for spread in spreads:
        try:
            return complex(function(*(value[index] for value in spread)))
        except Exception:
            continue
    return None


def _real_part(values: np.ndarray) -> np.ndarray:
    """The real values of `values`, with everything else NaN.

    A value counts as real when its imaginary part is noise beside its real
    part, which is what the relative tolerance is for: the imaginary residue of
    a long numeric chain grows with the answer.
    """
    real = values.real
    imaginary = np.abs(values.imag)
    keep = np.isfinite(real) & np.isfinite(imaginary)
    keep &= imaginary <= REAL_TOLERANCE * (1.0 + np.abs(real))
    return np.where(keep, real, np.nan).astype(np.float64, copy=False)


def sample_adaptive(
    f: Callable[..., np.ndarray],
    xrange: tuple[float, float],
    yrange: tuple[float, float],
    size_px: tuple[float, float],
    report: Callable[[np.ndarray, np.ndarray], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample `f` over `xrange` to subpixel accuracy on a canvas that size.

    Both ranges and both pixel counts are needed because the tolerance is in
    screen space: a quarter pixel of deviation means one thing on a canvas
    showing y from -5 to 5 and another on one showing it from -5000 to 5000.

    `report` is called with the uniform pass as soon as it exists, so that a
    slow curve appears in outline while the refinement is still running. The
    arrays it is handed are not the ones returned.

    What comes back is x and y with NaN wherever the curve has no real value
    and wherever it jumps, so that a plot item drawn with `connect="finite"`
    leaves gaps rather than strokes. There is no accuracy setting: the answer
    is always subpixel, and zooming is how detail is asked for.
    """
    a, b = float(xrange[0]), float(xrange[1])
    width, height = float(size_px[0]), float(size_px[1])
    span = float(yrange[1]) - float(yrange[0])
    if not np.isfinite(a) or not np.isfinite(b) or b <= a or width <= 0:
        return np.empty(0), np.empty(0)
    # How many pixels one unit of each axis is worth, which is the whole of the
    # view transform the tolerance is measured through. A view with no height
    # at all cannot say how far a midpoint has strayed, so refinement then
    # rests on the uniform pass alone.
    across = width / (b - a)
    up = height / span if span > 0 and height > 0 else 0.0
    xs = np.linspace(a, b, INITIAL_POINTS)
    ys = f(xs)
    if report is not None:
        report(xs, ys)
    xs, ys, standing = _refine(f, xs, ys, across, up)
    return _cut(xs, ys, standing, abs(span), up)


def _refine(
    f: Callable[..., np.ndarray],
    xs: np.ndarray,
    ys: np.ndarray,
    across: float,
    up: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bisect until every interval is within tolerance, or cannot be.

    The flag array runs alongside the intervals and says which of them are
    still worth a midpoint. An interval whose midpoint came in within tolerance
    is settled and never looked at again; one that did not passes the flag to
    both its halves. Whatever is still flagged when the loop ends - because
    twelve levels were not enough, or because the interval is now narrower than
    a pixel - is where refinement gave up.

    Giving up is not by itself a discontinuity, though, which is what the second
    array is for. A slope too steep to draw and a jump are both intervals whose
    midpoint never comes near its chord, and the one thing that tells them apart
    is what bisection does to the height: a jump is entirely inside one half and
    the other half loses it, while a slope splits its height evenly however
    steep it is. So each half is told whether it kept the height its parent had,
    and an interval that has kept it at every level down is one bisection has
    made no progress on at all. That is the set the discontinuity guard wants,
    and it is what comes back - `TAN(x)` keeps its poles, and `x·SIN(x)` zoomed
    out to where it oscillates faster than the pixels draws as a solid band
    rather than as a comb of gaps.
    """
    flags = np.ones(max(len(xs) - 1, 0), dtype=bool)
    held = flags.copy()
    for _ in range(MAX_DEPTH):
        if len(xs) > MAX_POINTS:
            break
        wide = np.diff(xs) * across > NARROWEST_PX
        splitting = np.nonzero(flags & wide)[0]
        if not splitting.size:
            break
        middles = (xs[splitting] + xs[splitting + 1]) / 2.0
        values = f(middles)
        chords = (ys[splitting] + ys[splitting + 1]) / 2.0
        children = _strayed(ys[splitting], ys[splitting + 1], values, chords, up)
        heights, kept = np.abs(ys[splitting + 1] - ys[splitting]), held[splitting]
        right = np.abs(ys[splitting + 1] - values)
        left = np.abs(values - ys[splitting])
        # `insert` puts its value before the element it is given, so the copy is
        # the left half of the split and what was there is the right half.
        held[splitting] = kept & (right > HELD_FRACTION * heights)
        held = np.insert(held, splitting, kept & (left > HELD_FRACTION * heights))
        flags[splitting] = children
        flags = np.insert(flags, splitting, children)
        xs = np.insert(xs, splitting + 1, middles)
        ys = np.insert(ys, splitting + 1, values)
    return xs, ys, flags & held


def _strayed(
    left: np.ndarray,
    right: np.ndarray,
    middle: np.ndarray,
    chord: np.ndarray,
    up: float,
) -> np.ndarray:
    """Which of these midpoints justify another bisection.

    Two reasons, and they are different questions. A midpoint that is more than
    a quarter pixel off its chord means the curve bends between the samples, and
    bisecting is how the bend is drawn. An interval with one real endpoint and
    one non-real one straddles the edge of the curve's domain - `SQRT(x)` at 0,
    or a pole - and is bisected to sub-pixel width whatever the values are, so
    that the curve begins where it begins rather than a grid step early.

    An interval with nothing real at either end or in the middle is outside the
    domain altogether, and there is nothing there to resolve.
    """
    here = np.isfinite(middle)
    edges = np.isfinite(left) & np.isfinite(right)
    straddling = (np.isfinite(left) != np.isfinite(right)) | (edges != here)
    strayed = np.zeros(middle.shape, dtype=bool)
    settled = edges & here
    deviation = np.abs(np.where(settled, middle - chord, 0.0)) * up
    strayed[settled] = deviation[settled] > TOLERANCE_PX
    strayed |= straddling
    return strayed


def _cut(
    xs: np.ndarray,
    ys: np.ndarray,
    standing: np.ndarray,
    span: float,
    up: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Put a NaN through every jump, so nothing draws a stroke across one.

    A jump is an interval bisection got nowhere on - it is subpixel wide, the
    midpoint is still nowhere near the chord, and every split left the whole
    height on one side - across which the value moves either more than the whole
    visible height, which is what a pole does, or more than a few pixels, which
    is what a step function does. Both are the same fact: the curve is not there
    in between, and joining the two samples would draw a line the function has
    no points on.

    `TAN(x)` and `SIGN(x)` are the two acceptance cases, and they arrive here by
    the two different clauses.
    """
    if len(xs) < 2:
        return xs, ys
    steps = np.abs(np.diff(ys))
    joined = np.isfinite(ys[:-1]) & np.isfinite(ys[1:])
    jumping = (steps > span) if span > 0 else np.zeros(steps.shape, dtype=bool)
    jumping |= steps * up > JUMP_PX
    cutting = np.nonzero(standing & joined & jumping)[0]
    if not cutting.size:
        return xs, ys
    middles = (xs[cutting] + xs[cutting + 1]) / 2.0
    return (
        np.insert(xs, cutting + 1, middles),
        np.insert(ys, cutting + 1, np.nan),
    )


@dataclass(frozen=True)
class Sampled:
    """A parametrized curve as the window draws and rides it.

    The parameter comes back beside the points because trace lives on it: the
    marker of a parametric curve sits at a t, steps by a fraction of the
    t-range, and snaps to the nearest sampled point, none of which the x and y
    arrays can answer on their own.

    `gave_up` is the parameter value refinement bottomed out at, where it did.
    A curve through a pole has one and the window says so; a curve that is a
    curve everywhere has None.
    """

    ts: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    gave_up: float | None = None


def sample_curve(
    fx: Callable[..., np.ndarray],
    fy: Callable[..., np.ndarray],
    trange: tuple[float, float],
    xrange: tuple[float, float],
    yrange: tuple[float, float],
    size_px: tuple[float, float],
    report: Callable[[np.ndarray, np.ndarray], None] | None = None,
) -> Sampled:
    """Sample the curve `(fx(t), fy(t))` over `trange`, subpixel on that canvas.

    The same argument as `sample_adaptive` with the error measured as a
    distance rather than a height: a parametric curve can double back, stand
    still, or leave the view and return, so nothing may be assumed about how t
    and the abscissa are related. An interval is settled when its midpoint is
    within a quarter pixel of the chord *in the plane* and its two ends are
    within a pixel of each other; an interval that is neither after twelve
    bisections is a jump, and a NaN goes through it.

    That is what makes `[t, 1/t]` terminate with a gap: the two branches never
    get close on the screen however finely t is cut, so refinement stops at the
    depth cap rather than chasing the pole, and `gave_up` carries the t to say
    so at.
    """
    a, b = float(trange[0]), float(trange[1])
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        return Sampled(np.empty(0), np.empty(0), np.empty(0))
    across, up = _pixels_per_unit(xrange, yrange, size_px)
    ts = np.linspace(a, b, INITIAL_POINTS)
    xs, ys = fx(ts), fy(ts)
    if report is not None:
        report(xs, ys)
    ts, xs, ys, standing = _refine_curve(fx, fy, ts, xs, ys, across, up)
    return _cut_curve(ts, xs, ys, standing, across, up)


def _pixels_per_unit(
    xrange: tuple[float, float],
    yrange: tuple[float, float],
    size_px: tuple[float, float],
) -> tuple[float, float]:
    """How many pixels one unit of each axis is worth: the view transform.

    A degenerate view - no width, no height, no canvas - answers zero, and a
    tolerance measured through zero is a tolerance nothing exceeds, which
    leaves refinement resting on the uniform pass. That is the right failure:
    a window with no size to draw in has no accuracy to be short of.
    """
    width, height = float(size_px[0]), float(size_px[1])
    span_x = float(xrange[1]) - float(xrange[0])
    span_y = float(yrange[1]) - float(yrange[0])
    across = width / span_x if span_x > 0 and width > 0 else 0.0
    up = height / span_y if span_y > 0 and height > 0 else 0.0
    return across, up


def _refine_curve(
    fx: Callable[..., np.ndarray],
    fy: Callable[..., np.ndarray],
    ts: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    across: float,
    up: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bisect in t until every interval is a straight line on the screen.

    The second array that comes back is the explicit sampler's, for the same
    reason and by the same measure with the height read as a distance in the
    plane: an interval whose two ends stayed as far apart as its parent's, at
    every level down, is one bisection has made no progress on, and only those
    are jumps. A curve that merely moves faster than the pixels is drawn.
    """
    flags = np.ones(max(len(ts) - 1, 0), dtype=bool)
    held = flags.copy()
    for _ in range(MAX_DEPTH):
        if len(ts) > MAX_POINTS:
            break
        reach = _apart(xs[:-1], ys[:-1], xs[1:], ys[1:], across, up)
        splitting = np.nonzero(flags & (reach > NARROWEST_PX))[0]
        if not splitting.size:
            break
        middles = (ts[splitting] + ts[splitting + 1]) / 2.0
        mx, my = fx(middles), fy(middles)
        chord_x = (xs[splitting] + xs[splitting + 1]) / 2.0
        chord_y = (ys[splitting] + ys[splitting + 1]) / 2.0
        strayed = _apart(mx, my, chord_x, chord_y, across, up) > TOLERANCE_PX
        # An interval with one end on the curve and one off it straddles the
        # edge of the domain and is bisected whatever the distances say; one
        # with nothing real anywhere near it is outside the domain altogether,
        # and there is nothing there to resolve.
        here = np.isfinite(mx) & np.isfinite(my)
        left = np.isfinite(xs[splitting]) & np.isfinite(ys[splitting])
        right = np.isfinite(xs[splitting + 1]) & np.isfinite(ys[splitting + 1])
        edges = left & right
        straddling = (left != right) | (edges != here)
        children = np.where(edges & here, strayed, straddling)
        reaches, kept = reach[splitting], held[splitting]
        second = _apart(mx, my, xs[splitting + 1], ys[splitting + 1], across, up)
        first = _apart(xs[splitting], ys[splitting], mx, my, across, up)
        # `insert` puts its value before the element it is given, so the copy is
        # the first half of the split and what was there is the second.
        held[splitting] = kept & (second > HELD_FRACTION * reaches)
        held = np.insert(held, splitting, kept & (first > HELD_FRACTION * reaches))
        flags[splitting] = children
        flags = np.insert(flags, splitting, children)
        ts = np.insert(ts, splitting + 1, middles)
        xs = np.insert(xs, splitting + 1, mx)
        ys = np.insert(ys, splitting + 1, my)
    return ts, xs, ys, flags & held


def _apart(
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    across: float,
    up: float,
) -> np.ndarray:
    """How far apart two points are on the screen, in pixels.

    A point the curve does not reach is infinitely far from anything, which is
    what keeps a domain edge being refined rather than settled by a NaN that
    fails every comparison it is put in.
    """
    dx = (np.asarray(x2) - np.asarray(x1)) * across
    dy = (np.asarray(y2) - np.asarray(y1)) * up
    away = np.hypot(dx, dy)
    return np.where(np.isnan(away), np.inf, away)


def _cut_curve(
    ts: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    standing: np.ndarray,
    across: float,
    up: float,
) -> Sampled:
    """Put a NaN through every jump refinement could not close.

    Two samples a subpixel step of t apart and half a screen apart in the plane
    are not two ends of a segment, they are two branches, and drawing the
    segment would be drawing a line the curve has no points on - as long as
    bisection got no nearer to closing them, which is what `standing` says.
    """
    if len(ts) < 2:
        return Sampled(ts, xs, ys)
    away = _apart(xs[:-1], ys[:-1], xs[1:], ys[1:], across, up)
    ends = np.isfinite(xs) & np.isfinite(ys)
    joined = ends[:-1] & ends[1:]
    cutting = np.nonzero(standing & joined & (away > JUMP_PX))[0]
    gave_up = float(ts[cutting[0]]) if cutting.size else None
    if not cutting.size:
        return Sampled(ts, xs, ys, gave_up)
    middles = (ts[cutting] + ts[cutting + 1]) / 2.0
    return Sampled(
        np.insert(ts, cutting + 1, middles),
        np.insert(xs, cutting + 1, np.nan),
        np.insert(ys, cutting + 1, np.nan),
        gave_up,
    )


def grid_eval(
    f: Callable[..., np.ndarray],
    xrange: tuple[float, float],
    yrange: tuple[float, float],
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate `f` over a rectangle, for the plots that are areas rather lines.

    Returns the two axis vectors and the `nx` by `ny` value array indexed
    `[i, j]` for x[i], y[j] - the order `isocurve` and `ImageItem` read, and
    the order a surface mesh is built in. Non-real and non-finite values are
    NaN here as everywhere, which is what leaves a hole in a contour and a hole
    in a mesh rather than a wrong answer.
    """
    xs = np.linspace(float(xrange[0]), float(xrange[1]), max(int(nx), 2))
    ys = np.linspace(float(yrange[0]), float(yrange[1]), max(int(ny), 2))
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    return xs, ys, f(grid_x, grid_y)


#: How many times a straddling cell edge is halved to find where the surface
#: stops being real. Six halvings put the crossing within 1/64 of a cell,
#: comfortably under a pixel at any grid the window draws.
BOUNDARY_BISECTIONS = 6


@dataclass(frozen=True)
class Boundary:
    """Where a surface stops being real, refined along the edges of its grid.

    One entry per cell edge whose endpoints straddle the domain of definition:
    `across` holds the x of the crossing on each edge running along x (indexed
    `[i, j]` for the edge from x[i] to x[i+1] at y[j]), `along` the y of the
    crossing on each edge running along y (indexed `[i, j]` for the edge from
    y[j] to y[j+1] at x[i]), and the `_z` arrays the value the surface takes
    there, found on the defined side of the crossing. Edges that do not
    straddle hold NaN. The mesh reads these to put its boundary vertices on
    the boundary, and a wire mesh ends its lines on them.
    """

    across: np.ndarray
    across_z: np.ndarray
    along: np.ndarray
    along_z: np.ndarray


def grid_boundary(
    f: Callable[..., np.ndarray],
    xs: np.ndarray,
    ys: np.ndarray,
    values: np.ndarray,
) -> Boundary:
    """Where `f` stops being real along the grid's edges, to a fraction of a cell.

    For every cell edge with one defined and one undefined endpoint, the
    crossing is bisected on the closure: the defined end moves to every
    midpoint that answers a real value, so what remains after six halvings is
    a real point within a sixty-fourth of a cell of where the surface ends -
    the same bargain the curve sampler strikes at the edge of its domain.
    Depends only on the domain and the grid, never on the box, so it runs on
    the sampling thread beside `grid_eval` and is cached with its arrays.
    """
    zs = np.asarray(values, dtype=np.float64)
    if zs.ndim != 2:
        empty = np.empty((0, 0))
        return Boundary(empty, empty, empty, empty)
    defined = np.isfinite(zs)
    nx, ny = zs.shape
    across = np.full((max(nx - 1, 0), ny), np.nan)
    across_z = np.full_like(across, np.nan)
    along = np.full((nx, max(ny - 1, 0)), np.nan)
    along_z = np.full_like(along, np.nan)
    ii, jj = np.nonzero(defined[:-1, :] != defined[1:, :])
    if ii.size:
        first = defined[ii, jj]
        starts = np.where(first, zs[ii, jj], zs[ii + 1, jj])
        found, worth = _bisected(
            lambda middle: f(middle, ys[jj]),
            np.where(first, xs[ii], xs[ii + 1]),
            np.where(first, xs[ii + 1], xs[ii]),
            starts,
        )
        across[ii, jj], across_z[ii, jj] = found, worth
    ii, jj = np.nonzero(defined[:, :-1] != defined[:, 1:])
    if ii.size:
        first = defined[ii, jj]
        starts = np.where(first, zs[ii, jj], zs[ii, jj + 1])
        found, worth = _bisected(
            lambda middle: f(xs[ii], middle),
            np.where(first, ys[jj], ys[jj + 1]),
            np.where(first, ys[jj + 1], ys[jj]),
            starts,
        )
        along[ii, jj], along_z[ii, jj] = found, worth
    return Boundary(across, across_z, along, along_z)


def _bisected(
    f: Callable[[np.ndarray], np.ndarray],
    real: np.ndarray,
    lost: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Halve every edge between its real end and its lost one, all at once.

    Each halving is one vectorized call over every straddling edge, so the
    whole refinement costs six evaluations however many edges there are. What
    comes back is the real end as it stands after the halvings, and the value
    the surface takes there - always a real one, because the end only ever
    moves to a midpoint that answered one.
    """
    for _ in range(BOUNDARY_BISECTIONS):
        middle = (real + lost) / 2.0
        answered = np.asarray(f(middle), dtype=np.float64)
        settled = np.isfinite(answered)
        values = np.where(settled, answered, values)
        real, lost = np.where(settled, middle, real), np.where(settled, lost, middle)
    return real, values


#: What a feature is called, which is what the status bar says it found.
ROOT = "root"
MAXIMUM = "maximum"
MINIMUM = "minimum"
CROSSING = "intersection"

#: How many bisections a refinement is worth. Fifty halvings take an interval
#: of the initial grid below the resolution of a float, so this is "until it
#: cannot be improved" written as a number.
REFINEMENTS = 50

#: How near zero a refined root has to come, relative to the values it was
#: found between. A bisection that ran into a pole ends up with the interval
#: closed and the function still enormous, and that is not a root.
ROOT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Feature:
    """One notable point of a curve, as trace mode snaps to it.

    `other` is the index into the curves the intersections were sought against,
    so the window can name the curve that was crossed, and -1 for the features
    that are about one curve alone.
    """

    x: float
    y: float
    kind: str
    other: int = -1


def features(
    xs: np.ndarray,
    ys: np.ndarray,
    f: Callable[..., np.ndarray],
    others: Sequence[Callable[..., np.ndarray]] = (),
) -> tuple[Feature, ...]:
    """The roots, extrema and intersections of a sampled curve, in x order.

    Found on the samples that are drawn and refined on the closure, which is
    the only honest way round: the samples say where to look - a sign change
    of y, of its successive differences, of the difference against another
    curve - and the function itself says exactly where the point is. A feature
    read off the pixel grid would move when the window was resized.

    A sign change across a gap is a pole and not a root, and is skipped: only
    intervals whose two ends are both finite are considered, so the NaN the
    sampler put through the pole of `TAN(x)` is what keeps it from being
    called a zero. A refinement that closes on a value that is not near zero
    is dropped for the same reason - it found the pole of a curve nobody had
    gapped.
    """
    if len(xs) < 3:
        return ()
    found = list(_zeros(xs, ys, f, ROOT))
    found += _extrema(xs, ys, f)
    for index, other in enumerate(others):
        try:
            values = ys - np.asarray(other(xs), dtype=np.float64)
        except Exception:
            continue
        crossings = _zeros(xs, values, _minus(f, other), CROSSING)
        found += [
            Feature(feature.x, _value(f, feature.x), CROSSING, index)
            for feature in crossings
        ]
    return tuple(sorted(found, key=lambda feature: feature.x))


def _minus(
    f: Callable[..., np.ndarray], g: Callable[..., np.ndarray]
) -> Callable[..., np.ndarray]:
    """The difference of two curves, whose roots are their intersections."""
    return lambda x: f(x) - g(x)


def _zeros(
    xs: np.ndarray,
    ys: np.ndarray,
    f: Callable[..., np.ndarray],
    kind: str,
) -> list[Feature]:
    """Where a sampled function changes sign, refined to the function's own root."""
    found: list[Feature] = []
    finite = np.isfinite(ys)
    for index in range(len(xs) - 1):
        if not finite[index]:
            continue
        if ys[index] == 0.0:
            found.append(Feature(float(xs[index]), 0.0, kind))
            continue
        if not finite[index + 1] or ys[index] * ys[index + 1] >= 0.0:
            continue
        place = _bisect(f, float(xs[index]), float(xs[index + 1]))
        if place is None:
            continue
        scale = max(abs(float(ys[index])), abs(float(ys[index + 1])))
        if abs(_value(f, place)) > ROOT_TOLERANCE * (1.0 + scale):
            continue  # a pole nobody gapped, not a zero
        found.append(Feature(place, 0.0, kind))
    return found


def _bisect(f: Callable[..., np.ndarray], a: float, b: float) -> float | None:
    """Halve a bracketed sign change until there is nothing left to halve.

    Bisection rather than anything cleverer because it is the one method that
    cannot leave the bracket: a curve steep enough to be interesting is a curve
    a secant step would jump out of.
    """
    fa = _value(f, a)
    fb = _value(f, b)
    if not np.isfinite(fa) or not np.isfinite(fb) or fa * fb > 0.0:
        return None
    for _ in range(REFINEMENTS):
        middle = (a + b) / 2.0
        if middle <= a or middle >= b:
            break
        value = _value(f, middle)
        if not np.isfinite(value):
            return None
        if value == 0.0:
            return middle
        if (value > 0.0) == (fa > 0.0):
            a, fa = middle, value
        else:
            b, fb = middle, value
    return (a + b) / 2.0


def _extrema(
    xs: np.ndarray, ys: np.ndarray, f: Callable[..., np.ndarray]
) -> list[Feature]:
    """Where the sampled slope changes sign, refined by parabolic fit.

    The bracket is three consecutive samples with the middle one highest or
    lowest, which is what a sign change of the successive differences comes to,
    and the refinement is the vertex of the parabola through them, iterated.
    A parabola is what a smooth extremum looks like from close enough, so a
    handful of fits reaches the accuracy the closure can offer; the golden
    step is the fallback for a fit that leaves the bracket, which happens where
    the curve is not smooth at all.
    """
    found: list[Feature] = []
    finite = np.isfinite(ys)
    for index in range(1, len(xs) - 1):
        if not (finite[index - 1] and finite[index] and finite[index + 1]):
            continue
        bracket = (float(xs[index - 1]), float(xs[index]), float(xs[index + 1]))
        if not bracket[0] < bracket[1] < bracket[2]:
            continue
        left, middle, right = ys[index - 1], ys[index], ys[index + 1]
        if middle > left and middle >= right:
            place, kind = _peak(f, bracket, True), MAXIMUM
        elif middle < left and middle <= right:
            place, kind = _peak(f, bracket, False), MINIMUM
        else:
            continue
        if place is None:
            continue
        found.append(Feature(place, _value(f, place), kind))
    return found


#: The golden ratio's complement, which is where a bracket is cut when the
#: parabola through it says something useless.
GOLDEN = 0.3819660112501051


def _peak(
    f: Callable[..., np.ndarray], bracket: tuple[float, float, float], up: bool
) -> float | None:
    """The turning point in the bracket, to the accuracy the closure can give.

    The middle of the bracket is the sample that stood out, not the midpoint of
    the interval: adaptive sampling puts its points where the curve needed
    them, and that is the best first guess there is.
    """
    sign = 1.0 if up else -1.0
    a, b, c = bracket
    fa, fb, fc = _value(f, a) * sign, _value(f, b) * sign, _value(f, c) * sign
    if not (np.isfinite(fa) and np.isfinite(fb) and np.isfinite(fc)):
        return None
    for _ in range(REFINEMENTS):
        place = _vertex(a, b, c, fa, fb, fc)
        if place is None or not a < place < c or place == b:
            wider = c - b > b - a
            place = b + GOLDEN * ((c - b) if wider else (a - b))
        if place <= a or place >= c or place == b:
            break
        value = _value(f, place) * sign
        if not np.isfinite(value):
            break
        if value > fb:
            if place < b:
                a, fa, b, fb, c, fc = a, fa, place, value, b, fb
            else:
                a, fa, b, fb, c, fc = b, fb, place, value, c, fc
        else:
            a, fa, c, fc = (place, value, c, fc) if place < b else (a, fa, place, value)
        if c - a <= abs(b) * 1e-15 + 1e-300:
            break
    return b


def _vertex(
    a: float, b: float, c: float, fa: float, fb: float, fc: float
) -> float | None:
    """Where the parabola through three points turns, or None where it does not."""
    left, right = (b - a) * (fb - fc), (b - c) * (fb - fa)
    bottom = 2.0 * (right - left)
    if bottom == 0.0 or not np.isfinite(bottom):
        return None
    place = b - ((b - c) * right - (b - a) * left) / bottom
    return float(place) if np.isfinite(place) else None


def _value(f: Callable[..., np.ndarray], x: float) -> float:
    """The closure at one point, as a float, NaN where it has no value there."""
    try:
        return float(np.asarray(f(np.array([x], dtype=np.float64)))[0])
    except Exception:
        return float("nan")


def finite_fraction(values: Iterable[float] | np.ndarray) -> float:
    """How much of a sampled curve is real and finite, between 0 and 1.

    Zero is the one figure that has to be acted on: it is the empty picture the
    window explains rather than shows.
    """
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return 0.0
    return float(np.count_nonzero(np.isfinite(array)) / array.size)

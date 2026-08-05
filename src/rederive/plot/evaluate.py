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
sympy is never asked anything again.

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
far apart has found a jump rather than a slope, and a jump is where a NaN goes
so that nothing draws a stroke across it.

`features` - the roots, extrema and intersections trace mode snaps to - belongs
here too and is not written yet; it is the one function of section 8 that the
2D window does not call.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from rederive.engine.context import Context
from rederive.model.expr import Node

__all__ = [
    "INITIAL_POINTS",
    "MAX_DEPTH",
    "Unplottable",
    "closure",
    "grid_eval",
    "sample_adaptive",
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

#: How many samples one curve may end up with. Refinement only bisects where
#: it must, so no ordinary expression comes near this; it is what stops a
#: pathological one from eating the host's memory.
MAX_POINTS = 20001


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
    a vectorized call that raises is retried point by point, and a point that
    raises is NaN.
    """
    import sympy as sp

    from rederive.engine.substitute import substitute
    from rederive.engine.to_sympy import to_sympy

    try:
        expression = to_sympy(substitute(node, context), context)
    except Exception as error:
        raise Unplottable(_said(error)) from None
    named = {
        symbol.name: symbol
        for symbol in expression.free_symbols
        if isinstance(symbol, sp.Symbol)
    }
    symbols = [named.get(name, sp.Symbol(name)) for name in variables]
    try:
        function = sp.lambdify(symbols, expression, modules="numpy")
    except Exception as error:
        raise Unplottable(_said(error)) from None
    return _Numeric(function, len(symbols))


def _said(error: Exception) -> str:
    """One line naming an exception, for a message line to carry."""
    text = str(error).strip().splitlines()
    return text[0] if text else type(error).__name__


class _Numeric:
    """A lambdified expression, evaluated in the complex plane and masked back.

    The wrapper is where every promise of section 8 is kept: complex inputs so
    that a radical has a real part to find, the real mask, non-finite to NaN,
    numpy's warnings silenced - a plot of `1/x` divides by zero by design and
    the terminal is not the place to hear about it - and the pointwise retry,
    which some sympy functions lambdify imperfectly enough to need.
    """

    def __init__(self, function: Callable[..., object], arity: int) -> None:
        self._function = function
        self._arity = arity

    def __call__(self, *arguments: np.ndarray) -> np.ndarray:
        inputs = [np.asarray(argument, dtype=np.complex128) for argument in arguments]
        shape = np.broadcast(*inputs).shape if inputs else ()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(all="ignore"):
                try:
                    values = np.asarray(self._function(*inputs), dtype=np.complex128)
                except Exception:
                    values = self._pointwise(inputs, shape)
                if values.shape != shape:
                    # A constant expression lambdifies to a function that
                    # ignores its arguments and answers with one number.
                    values = np.broadcast_to(values, shape).copy()
                return _real_part(values)

    def _pointwise(
        self, inputs: list[np.ndarray], shape: tuple[int, ...]
    ) -> np.ndarray:
        """One point at a time, for a function the vectorized call choked on."""
        spread = [np.broadcast_to(value, shape).ravel() for value in inputs]
        answers = np.full(int(np.prod(shape)) if shape else 1, np.nan, np.complex128)
        for index in range(answers.size):
            try:
                answers[index] = complex(
                    self._function(*(value[index] for value in spread))
                )
            except Exception:
                answers[index] = np.nan
        return answers.reshape(shape)


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
    xs, ys, unresolved = _refine(f, xs, ys, across, up)
    return _cut(xs, ys, unresolved, abs(span), up)


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
    a pixel - is where refinement gave up, and that is exactly the set the
    discontinuity guard is interested in.
    """
    flags = np.ones(max(len(xs) - 1, 0), dtype=bool)
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
        flags[splitting] = children
        flags = np.insert(flags, splitting, children)
        xs = np.insert(xs, splitting + 1, middles)
        ys = np.insert(ys, splitting + 1, values)
    return xs, ys, flags


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
    unresolved: np.ndarray,
    span: float,
    up: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Put a NaN through every jump, so nothing draws a stroke across one.

    A jump is an interval refinement could not settle - it is subpixel wide and
    the midpoint is still nowhere near the chord - across which the value moves
    either more than the whole visible height, which is what a pole does, or
    more than a few pixels, which is what a step function does. Both are the
    same fact: the curve is not there in between, and joining the two samples
    would draw a line the function has no points on.

    `TAN(x)` and `SIGN(x)` are the two acceptance cases, and they arrive here by
    the two different clauses.
    """
    if len(xs) < 2:
        return xs, ys
    steps = np.abs(np.diff(ys))
    joined = np.isfinite(ys[:-1]) & np.isfinite(ys[1:])
    jumping = (steps > span) if span > 0 else np.zeros(steps.shape, dtype=bool)
    jumping |= steps * up > JUMP_PX
    cutting = np.nonzero(unresolved & joined & jumping)[0]
    if not cutting.size:
        return xs, ys
    middles = (xs[cutting] + xs[cutting + 1]) / 2.0
    return (
        np.insert(xs, cutting + 1, middles),
        np.insert(ys, cutting + 1, np.nan),
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


def finite_fraction(values: Iterable[float] | np.ndarray) -> float:
    """How much of a sampled curve is real and finite, between 0 and 1.

    Zero is the one figure that has to be acted on: it is the empty picture the
    window explains rather than shows.
    """
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return 0.0
    return float(np.count_nonzero(np.isfinite(array)) / array.size)

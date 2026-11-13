"""Numeric evaluation of plotted expressions: closures, samplers and grids.

Pure numpy functions with no Qt, so the many corner cases can be tested without a
screen. Everything here runs on the host's sampling thread, never the Qt thread.

Three ideas shape the module.

An expression is converted *once*. The tree goes through the engine's pre-pass and
converter into a lambdified closure kept for the life of the curve, so resampling on
zoom costs microseconds and never calls sympy. Where numpy can't evaluate a head (e.g.
`x!`), sympy is used point by point, so anything that can be approximated can be
plotted. That fallback is only reached after the fast path fails, and is gated by a
probe, so it costs nothing otherwise.

Everything is evaluated in the complex plane and masked back to the reals. `SQRT(x)`
over a range including negative x is a curve that starts at zero, not an error:
evaluating in `complex128` and keeping values with negligible imaginary parts draws
exactly the real part of the graph. Masking isn't silent: a curve with no finite point
in view says so on the status bar.

Sampling is adaptive in *screen* space. A midpoint's deviation from its chord is
measured in pixels after the view transform, so every curve is sampled to the same
visible accuracy. This also enables the discontinuity guard: if refinement reaches
subpixel width with the samples still far apart, and bisection didn't reduce the gap,
it found a jump, which gets a NaN so no stroke is drawn across it. The last condition
distinguishes jumps from steep slopes, since bisecting a slope halves its height; so
`x·SIN(x)` zoomed out until it oscillates faster than the pixels is drawn as a band.

There are separate samplers for functions and parametric curves, because for the
latter a midpoint's error is a distance in the plane rather than a height, and an
interval is narrow when its ends land on the same pixel rather than when its
parameters are close. Otherwise (tolerance, depth cap, gaps at jumps) they work the
same.

`features` finds roots and extrema on the drawn samples and then refines them on the
closure, so they are accurate to full precision rather than to a pixel.
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

#: How small a value's imaginary part must be, relative to its magnitude, to count as
#: real: loose enough to absorb rounding, tight enough to drop truly complex values.
REAL_TOLERANCE = 1e-9

#: Points in the initial uniform pass, enough to show the curve's outline at once
#: without making slow expressions much slower.
INITIAL_POINTS = 129

#: Maximum bisection depth. Twelve levels shrink an initial interval to 1/4096, which
#: is subpixel on any canvas.
MAX_DEPTH = 12

#: How far in pixels a midpoint may deviate from its chord before the interval is
#: bisected: below what is visible, above floating-point noise.
TOLERANCE_PX = 0.25

#: Interval width in pixels below which refinement stops, as both samples fall in the
#: same pixel column.
NARROWEST_PX = 0.25

#: Height in pixels of an unrefinable interval that makes it a discontinuity. Poles
#: meet it.
JUMP_PX = 8.0

#: Fraction of its height one half of a bisected interval must keep to still count as
#: a jump. A discontinuity keeps all of it in one half however finely it is cut, while
#: a slope splits it between the halves.
HELD_FRACTION = 0.75

#: Maximum samples per curve. Ordinary expressions never come near it; it stops
#: pathological ones from exhausting memory.
MAX_POINTS = 20001

#: How many points to probe the exact evaluators at before giving up on them. One isn't
#: enough, since a single point could be a pole. Spread across the sample, a few points
#: settle it cheaply.
PROBE_POINTS = 8


class Unplottable(Exception):
    """The expression can't be evaluated numerically.

    Raised by `closure` for failures before the first sample: an operator with no
    numeric meaning, a failed conversion, or a failed lambdify. The host reports it
    as an event for the curve.
    """


def closure(
    node: Node, context: Context, variables: Sequence[str] = ()
) -> Callable[..., np.ndarray]:
    """A numeric function of `variables` for the expression `node`.

    The engine's `evaluated` first resolves labels, assigned variables and defined
    functions, so plotting `#3` or `f(x)` plots what they mean in the worksheet.

    The result maps arrays to an array of the same shape: real where the value is real
    and finite, NaN elsewhere. It never raises, since an exception mid-sampling would
    lose the whole curve: a failed vectorized call is retried point by point, a failed
    point is evaluated by sympy, and a point sympy can't evaluate is NaN.
    """
    return _lambdified(_converted(node, context), variables)


def difference(
    node: Node, context: Context, variables: Sequence[str] = ()
) -> Callable[..., np.ndarray]:
    """`u - v` as a function, for the implicit plot of the equation `u = v`.

    An implicit curve is the zero contour of the difference, which gives marching
    squares the scalar field it needs.
    """
    expression = _converted(node, context)
    sides = (getattr(expression, "lhs", None), getattr(expression, "rhs", None))
    if sides[0] is not None and sides[1] is not None:
        expression = sides[0] - sides[1]
    return _lambdified(expression, variables)


def mask(
    node: Node, context: Context, variables: Sequence[str] = ()
) -> Callable[..., np.ndarray]:
    """A boolean per point, for the shaded region of an inequality.

    Unlike everything else here, this is evaluated on real inputs, since `x > 0`
    raises for complex x. Points where the result isn't a truth value are false.
    """
    inside = _lambdified(_converted(node, context), variables, real=True)

    def truth(*arguments: np.ndarray) -> np.ndarray:
        values = inside(*arguments)
        return np.asarray(np.nan_to_num(values, nan=0.0) > 0.5, dtype=bool)

    return truth


def pair(
    node: Node, context: Context, variables: Sequence[str] = ()
) -> tuple[Callable[..., np.ndarray], Callable[..., np.ndarray]]:
    """The two closures of a parametric pair `[x(t), y(t)]`, converted together once."""
    from rederive.engine.vectors import Vector

    expression = _converted(node, context)
    if not isinstance(expression, Vector):
        raise Unplottable("a parametric plot needs two expressions")
    elements = list(expression.args)
    if len(elements) != 2 or any(isinstance(e, Vector) for e in elements):
        raise Unplottable("a parametric plot needs two expressions")
    return (
        _lambdified(elements[0], variables),
        _lambdified(elements[1], variables),
    )


def polar_pair(
    f: Callable[..., np.ndarray], degrees: bool = False
) -> tuple[Callable[..., np.ndarray], Callable[..., np.ndarray]]:
    """`r = f(θ)` as a parametric pair for the sampler.

    θ is in the worksheet's angle unit: in degree mode `SIN(θ)` takes degrees, so the
    conversion to a point must use degrees too.
    """
    turn = np.pi / 180.0 if degrees else 1.0

    def horizontal(t: np.ndarray) -> np.ndarray:
        return f(t) * np.cos(np.asarray(t, dtype=np.float64) * turn)

    def vertical(t: np.ndarray) -> np.ndarray:
        return f(t) * np.sin(np.asarray(t, dtype=np.float64) * turn)

    return horizontal, vertical


def points(node: Node, context: Context) -> tuple[np.ndarray, np.ndarray]:
    """The two columns of a constant matrix, as a data plot's x and y.

    A single point like `[1, 2]` is read as a one-row matrix.
    """
    from rederive.engine.vectors import Vector, is_matrix, matrix_of

    expression = _converted(node, context)
    if isinstance(expression, Vector) and not is_matrix(expression):
        expression = Vector(expression)
    matrix = matrix_of(expression)
    if matrix is None:
        raise Unplottable("a data plot needs a matrix of two columns")
    try:
        rows = np.array(matrix.evalf().tolist(), dtype=np.complex128)
    except Exception as error:
        raise Unplottable(_said(error)) from None
    if rows.ndim != 2 or rows.shape[1] != 2:
        raise Unplottable("a data plot needs a matrix of two columns")
    return _real_part(rows[:, 0]), _real_part(rows[:, 1])


def number(node: Node | None, context: Context, default: float) -> float:
    """The value of a bound typed as an expression (like `-π`), or `default`.

    Evaluated here, on the sampling thread, rather than where it was typed.
    """
    if node is None:
        return default
    try:
        value = float(np.asarray(closure(node, context, ())()))
    except Exception:
        return default
    return value if np.isfinite(value) else default


def _converted(node: Node, context: Context) -> object:
    """`node` as a sympy object, as evaluated by the engine.

    The engine's `evaluated` resolves labels, assigned variables and defined functions
    first.
    """
    from rederive.engine.evaluate import evaluated

    try:
        return evaluated(node, context)
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
    """A one-line description of an exception, for the message line."""
    text = str(error).strip().splitlines()
    return text[0] if text else type(error).__name__


def _exact(expression: Any, symbols: Sequence[Any]) -> list[Callable[..., Any]]:
    """The fallback evaluators, in the order to try them.

    Lambdify's numpy namespace covers less than the engine does. A head numpy lacks is
    printed as a bare name and resolves to whatever Python has under it; for
    `factorial` that is `math.factorial`, which rejects floats. These fallbacks make
    anything the engine can approximate plottable, with the same values.

    Mpmath comes first: it is the arithmetic evalf uses, without the dispatch overhead,
    and 3 to 70 times faster per point. Sympy's evalf is second, for expressions mpmath
    can't handle; it is slower but always agrees with the engine.
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
    """Evaluates `expression` at one point with sympy, as the engine does."""

    def __init__(self, expression: Any, symbols: Sequence[Any]) -> None:
        self._expression = expression
        self._symbols = list(symbols)

    def __call__(self, *arguments: Any) -> complex:
        values = [_as_number(argument) for argument in arguments]
        bound = dict(zip(self._symbols, values, strict=True))
        return complex(self._expression.subs(bound).evalf())


def _as_number(value: Any) -> Any:
    """A numpy scalar as a sympy number.

    Real inputs stay real rather than becoming complex with a zero imaginary part,
    since sympy treats them differently: the factorial of 2.5 is a number, while that
    of `2.5 + 0i` is a branch.
    """
    import sympy as sp

    number = complex(value)
    if number.imag:
        return sp.Float(number.real) + sp.Float(number.imag) * sp.I
    return sp.Float(number.real)


def _probes(size: int) -> list[int]:
    """Indices at which to probe the exact evaluators, spread across the sample."""
    if size <= PROBE_POINTS:
        return list(range(size))
    step = size / PROBE_POINTS
    return [int(index * step) for index in range(PROBE_POINTS)]


class _Numeric:
    """A lambdified expression evaluated in the complex plane and masked to the reals.

    This implements the evaluation rules of design section 8: complex inputs so
    radicals have a real part, masking to the reals, non-finite values to NaN, numpy
    warnings silenced (plotting `1/x` divides by zero by design), and retries for
    functions sympy lambdifies imperfectly.

    Some numpy functions (`arctan2`, `floor`, `ceiling`, `Mod`) reject complex dtypes
    outright, so `ATAN(-y, x)` would plot nothing. The real parts of the inputs are
    therefore tried next. Nothing is lost, since inputs with a significant imaginary
    part are off the graph anyway.

    Last come the exact evaluators, which make anything the engine can approximate
    plottable. They are only used at points where the lambdified function failed, and
    gated by a probe, so a few points decide whether any exact evaluator works; if
    none does, those samples are NaN.
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
        #: Whether inputs are real. Only relations need that, since sympy raises on
        #: `x > 0` for complex x.
        self._dtype = np.float64 if real else np.complex128
        #: The exact evaluators, built at most once and kept for the life of the curve
        #: so zooming doesn't lambdify again.
        self._candidates: list[Callable[..., Any]] | None = None

    def __call__(self, *arguments: np.ndarray) -> np.ndarray:
        inputs = [np.asarray(argument, dtype=self._dtype) for argument in arguments]
        shape = np.broadcast(*inputs).shape if inputs else ()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(all="ignore"):
                values = self._values(inputs, shape)
                if values.shape != shape:
                    # A constant expression lambdifies to a function returning a
                    # scalar.
                    values = np.broadcast_to(values, shape).copy()
                return _real_part(values)

    def _values(self, inputs: list[np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
        """Evaluate over `inputs` by whatever means works.

        Tries the vectorized call on each variant of the inputs, then falls back to
        point-by-point evaluation.
        """
        for attempt in self._attempts(inputs):
            try:
                return np.asarray(self._function(*attempt), dtype=np.complex128)
            except Exception:
                continue
        return self._pointwise(inputs, shape)

    def _attempts(self, inputs: list[np.ndarray]) -> list[list[np.ndarray]]:
        """The inputs as given, then their real parts (unless already real)."""
        if self._dtype is np.float64 or not inputs:
            return [inputs]
        return [inputs, [_real_part(value) for value in inputs]]

    def _pointwise(
        self, inputs: list[np.ndarray], shape: tuple[int, ...]
    ) -> np.ndarray:
        """Evaluate point by point, when the vectorized call failed.

        Each point tries the lambdified function first, then the exact evaluator. The
        probe runs at the first failing point rather than up front, so a function that
        only fails at a pole doesn't build an evaluator needlessly.
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
        """The first exact evaluator that works on this sample, or None.

        Probed anew for each sampling, since the answer depends on the points in view:
        a head undefined over one window may be defined over the next. The evaluators
        themselves, which are expensive to build, are cached.
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
    """The value of `function` at one point, or None if it fails.

    None differs from NaN: NaN is a valid result (a point off the graph), while None
    means the evaluator failed and the next one should be tried.
    """
    for spread in spreads:
        try:
            return complex(function(*(value[index] for value in spread)))
        except Exception:
            continue
    return None


def _real_part(values: np.ndarray) -> np.ndarray:
    """The real values of `values`, with everything else NaN.

    A value is real if its imaginary part is negligible relative to its magnitude,
    since rounding residue grows with the value.
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
    """Sample `f` over `xrange` to subpixel accuracy on a canvas of the given size.

    Both ranges and the pixel size are needed because the tolerance is in screen space.

    `report` is called with the uniform pass right away, so a slow curve appears in
    outline during refinement. The arrays it receives are not the ones returned.

    Returns x and y, with NaN where the curve has no real value and at jumps, so that
    a plot item with `connect="finite"` leaves gaps there. There is no accuracy
    setting: the result is always subpixel, and zooming reveals more detail.
    """
    a, b = float(xrange[0]), float(xrange[1])
    width, height = float(size_px[0]), float(size_px[1])
    span = float(yrange[1]) - float(yrange[0])
    if not np.isfinite(a) or not np.isfinite(b) or b <= a or width <= 0:
        return np.empty(0), np.empty(0)
    # Pixels per unit on each axis, which is all of the view transform the tolerance
    # needs. With zero view height, deviation can't be measured, so only the uniform
    # pass is used.
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
    """Bisect until every interval is within tolerance or can't be refined further.

    `flags` marks the intervals still needing a midpoint. An interval whose midpoint is
    within tolerance is settled; otherwise both halves inherit the flag. Intervals
    still flagged at the end (depth exhausted or narrower than a pixel) are where
    refinement gave up.

    Giving up doesn't by itself mean a discontinuity, which is what `held` is for. Both
    steep slopes and jumps have midpoints far from their chords, but bisection treats
    them differently: a jump stays entirely in one half, while a slope splits its height
    between the halves. So each half records whether it kept its parent's height, and
    an interval that kept it at every level is a jump. That set is returned, so
    `TAN(x)` keeps its poles, while `x·SIN(x)` zoomed out until it oscillates faster
    than the pixels is drawn as a solid band rather than full of gaps.
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
        # `insert` places its value before the given index, so the inserted value is
        # the left half and the existing one becomes the right half.
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
    """Which of these midpoints call for another bisection.

    There are two reasons. A midpoint more than `TOLERANCE_PX` off its chord means the
    curve bends there. An interval with one real and one non-real endpoint straddles
    the edge of the domain (`SQRT(x)` at 0, or a pole), and is bisected to subpixel
    width regardless, so the curve starts in the right place.

    An interval with nothing real at either end or in the middle is outside the domain,
    with nothing to resolve.
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
    """Insert a NaN at every jump, so no stroke is drawn across it.

    A jump is an interval bisection couldn't resolve (see `_refine`) across which the
    value changes by more than the visible height (a pole) or by more than `JUMP_PX`
    (a step). Either way the curve has no points in between.

    `TAN(x)` and `SIGN(x)` are the test cases for the two conditions.
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
    """A sampled parametric curve, for drawing and tracing.

    The parameter values are kept because tracing uses them: the marker sits at a t,
    steps by a fraction of the t range, and snaps to the nearest sample.

    `gave_up` is the parameter value where refinement gave up at a jump, such as a
    pole, which the window reports; otherwise None.
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
    """Sample the curve `(fx(t), fy(t))` over `trange` to subpixel accuracy.

    Like `sample_adaptive`, but with error measured as a distance in the plane, since a
    parametric curve can double back, stop, or leave the view and return. An interval
    is settled when its midpoint is within `TOLERANCE_PX` of the chord and its ends are
    close on the screen; one that is neither after `MAX_DEPTH` bisections is a jump and
    gets a NaN.

    So `[t, 1/t]` ends with a gap: its branches never come close on the screen, so
    refinement stops at the depth cap and `gave_up` records the t.
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
    """Pixels per unit on each axis.

    A degenerate view (zero width, height or canvas size) gives zero, so nothing
    exceeds the tolerance and only the uniform pass is used.
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
    """Bisect in t until every interval is straight on the screen.

    The jump detection works as in `_refine`, using screen distance instead of height:
    only intervals whose ends stayed as far apart as their parent's at every level are
    jumps, so a curve that merely moves faster than the pixels is still drawn.
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
        # Intervals straddling the domain edge are bisected regardless of distance;
        # those with nothing real in them are outside the domain and left alone.
        here = np.isfinite(mx) & np.isfinite(my)
        left = np.isfinite(xs[splitting]) & np.isfinite(ys[splitting])
        right = np.isfinite(xs[splitting + 1]) & np.isfinite(ys[splitting + 1])
        edges = left & right
        straddling = (left != right) | (edges != here)
        children = np.where(edges & here, strayed, straddling)
        reaches, kept = reach[splitting], held[splitting]
        second = _apart(mx, my, xs[splitting + 1], ys[splitting + 1], across, up)
        first = _apart(xs[splitting], ys[splitting], mx, my, across, up)
        # `insert` places its value before the given index, so the inserted value is
        # the first half and the existing one becomes the second.
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
    """The screen distance between two points, in pixels.

    A NaN point is infinitely far from anything, so domain edges keep being refined
    rather than being settled by NaN failing every comparison.
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
    """Insert a NaN at every jump refinement couldn't close.

    Two samples a subpixel step of t apart but far apart on the screen, which bisection
    didn't bring closer (as `standing` says), belong to separate branches.
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
    """Evaluate `f` over a rectangle, for plots that are areas rather than lines.

    Returns the two axis vectors and the `nx` by `ny` value array indexed `[i, j]` for
    x[i], y[j], as `isocurve`, `ImageItem` and the surface mesh expect. Non-real and
    non-finite values are NaN, leaving holes in contours and meshes.
    """
    xs = np.linspace(float(xrange[0]), float(xrange[1]), max(int(nx), 2))
    ys = np.linspace(float(yrange[0]), float(yrange[1]), max(int(ny), 2))
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    return xs, ys, f(grid_x, grid_y)


#: How many times to halve a cell edge to find where the surface stops being real. Six
#: halvings put the crossing within 1/64 of a cell, well under a pixel.
BOUNDARY_BISECTIONS = 6


@dataclass(frozen=True)
class Boundary:
    """Where a surface stops being real, refined along its grid's cell edges.

    `across` holds the x of the crossing on each edge along x (indexed `[i, j]` for the
    edge from x[i] to x[i+1] at y[j]), and `along` the y of the crossing on each edge
    along y (indexed `[i, j]` for the edge from y[j] to y[j+1] at x[i]). The `_z` arrays
    hold the surface's value there, on the defined side. Edges that don't straddle the
    boundary hold NaN. The mesh and wireframe place their boundary vertices on these.
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

    For each cell edge with one defined and one undefined end, the crossing is
    bisected on the closure. The defined end moves to each midpoint with a real
    value, ending within 1/64 of a cell of the boundary. It depends only on the domain
    and grid, not the box, so it runs on the sampling thread with `grid_eval` and is
    cached with its arrays.
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
    """Bisect all edges between their real and non-real ends at once.

    Each halving is one vectorized call over all edges. Returns the final real ends and
    the surface's values there, which are always real, since an end only moves to a
    midpoint with a real value.
    """
    for _ in range(BOUNDARY_BISECTIONS):
        middle = (real + lost) / 2.0
        answered = np.asarray(f(middle), dtype=np.float64)
        settled = np.isfinite(answered)
        values = np.where(settled, answered, values)
        real, lost = np.where(settled, middle, real), np.where(settled, lost, middle)
    return real, values


#: Feature names, as shown on the status bar.
ROOT = "root"
MAXIMUM = "maximum"
MINIMUM = "minimum"
CROSSING = "intersection"

#: Maximum refinement iterations. Fifty halvings take an initial interval below float
#: resolution, so this effectively means "until it can't improve".
REFINEMENTS = 50

#: How close to zero a refined root must be, relative to the bracketing values. A
#: bisection that converges on a pole leaves the function huge, and is rejected.
ROOT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Feature:
    """A notable point of a curve, which trace mode snaps to.

    For an intersection, `other` is the index of the other curve, so the window can
    name it; it is -1 for other features.
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

    The samples show where to look (a sign change of y, of its differences, or of the
    difference with another curve), and the closure gives the exact location. Features
    read off the samples alone would move when the window was resized.

    A sign change across a gap is a pole, not a root, and is skipped: only intervals
    with both ends finite are considered, so the NaN the sampler put at the pole of
    `TAN(x)` keeps it from counting as a zero. A refinement that converges on a value
    not near zero is also dropped, as an unmarked pole.
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
    """Where a sampled function changes sign, refined to the function's root."""
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
    """Bisect a bracketed sign change to float precision.

    Bisection is used because it can't leave the bracket, which a secant step on a
    steep curve could.
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

    The bracket is three consecutive samples with the middle one highest or lowest. It
    is refined by iterating the vertex of the parabola through them, which converges
    quickly at a smooth extremum. A golden-section step is the fallback when the fit
    leaves the bracket, as happens where the curve isn't smooth.
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


#: The golden-section ratio, for cutting a bracket when the parabolic fit fails.
GOLDEN = 0.3819660112501051


def _peak(
    f: Callable[..., np.ndarray], bracket: tuple[float, float, float], up: bool
) -> float | None:
    """The turning point in the bracket, to the closure's accuracy.

    The bracket's middle is the extreme sample, not the interval's midpoint, which is
    the best first guess since adaptive sampling placed it.
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
    """The vertex of the parabola through three points, or None if there is none."""
    left, right = (b - a) * (fb - fc), (b - c) * (fb - fa)
    bottom = 2.0 * (right - left)
    if bottom == 0.0 or not np.isfinite(bottom):
        return None
    place = b - ((b - c) * right - (b - a) * left) / bottom
    return float(place) if np.isfinite(place) else None


def _value(f: Callable[..., np.ndarray], x: float) -> float:
    """The closure's value at one point as a float, or NaN."""
    try:
        return float(np.asarray(f(np.array([x], dtype=np.float64)))[0])
    except Exception:
        return float("nan")


def finite_fraction(values: Iterable[float] | np.ndarray) -> float:
    """The fraction of a sampled curve that is real and finite.

    Zero means an empty picture, which the window explains with a message.
    """
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return 0.0
    return float(np.count_nonzero(np.isfinite(array)) / array.size)

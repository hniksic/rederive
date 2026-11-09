"""Plotting: classification, evaluation, and the plot host.

Classification is pure. The interesting cases are ambiguous shapes, such as a 2-vector
that is a point, a parametric pair or a pair of surfaces depending on how many variables
it holds.

The evaluator tests cover the classic plotter bugs: a stroke bridging a pole, the
negative half of a square root drawn as zero, a spike between two samples.

The host is exercised over a real pipe with Qt running offscreen. It is skipped where Qt
is unavailable, and every wait has a deadline so a hang can't stall the suite.

The plot session tests (window placement, the receiver, preferences, refusals) run
against a backend that draws nothing, and surface geometry is tested on arrays alone,
which also shows that the backend seam is real.
"""

import asyncio
import dataclasses
import threading

import numpy as np
import pytest
from fakeplot import FakeBackend, InlineExecutor, offered, ticked
from screen import band, content, highlighted, message, prompt

from rederive.engine.context import Context
from rederive.model.plotting import Unplottable, classify
from rederive.model.session import Session
from rederive.plot import actions, controls, evaluate
from rederive.plot import protocol as plots
from rederive.plot.model import Plot, Surface
from rederive.plot.protocol import PlotKind
from rederive.plot.proxy import PlotError
from rederive.syntax import ParseState, parse_expression

#: Canvas size for tests that don't care. The sampler needs one because its tolerance is
#: in pixels.
CANVAS = (800.0, 600.0)

#: The default framing, used by most tests.
VIEW = (-5.0, 5.0)


def parsed(text):
    return parse_expression(text, ParseState()).node


def closure(text, variables=("x",)):
    return evaluate.closure(parsed(text), Context(), variables)


def sampled(text, xrange=VIEW, yrange=VIEW, size=CANVAS):
    return evaluate.sample_adaptive(closure(text), xrange, yrange, size)


# -- classification -----------------------------------------------------------


def kind_of(text, variables=()):
    """Classify `text`, given the variables the engine would find in it."""
    return classify(parsed(text), variables, text)


@pytest.mark.parametrize(
    ("text", "variables", "kind"),
    [
        ("SIN(x)", ("x",), PlotKind.CURVE),
        ("3", (), PlotKind.CURVE),
        ("x*y", ("x", "y"), PlotKind.SURFACE),
        ("z = x^2 + y^2", ("x", "y", "z"), PlotKind.SURFACE),
        ("x^2 + y^2 = 4", ("x", "y"), PlotKind.IMPLICIT),
        ("y = SIN(x)", ("x", "y"), PlotKind.IMPLICIT),
        ("x < 3", ("x",), PlotKind.REGION),
        ("x^2 + y^2 <= 4", ("x", "y"), PlotKind.REGION),
        ("x > 0 AND y > 0", ("x", "y"), PlotKind.REGION),
        ("NOT (x < 3)", ("x",), PlotKind.REGION),
        ("[1, 2]", (), PlotKind.DATA),
        ("[[1, 2], [3, 4], [5, 6]]", (), PlotKind.DATA),
        ("[SIN(t), COS(t)]", ("t",), PlotKind.PARAMETRIC),
        ("[x + y, x - y]", ("x", "y"), PlotKind.SURFACES),
        ("[x, x^2, x^3]", ("x",), PlotKind.FAMILY),
        ("[1, 2, 3]", (), PlotKind.FAMILY),
        ("[x*y, x + y, x - y]", ("x", "y"), PlotKind.SURFACES),
    ],
)
def test_the_classification_table_reads_the_shape_off_the_tree(text, variables, kind):
    assert kind_of(text, variables).kind is kind


def test_an_equation_solved_for_a_lone_variable_names_the_vertical_axis():
    plotted = kind_of("z = x^2 + y^2", ("x", "y", "z"))
    assert plotted.kind is PlotKind.SURFACE
    assert plotted.vertical == "z"
    assert plotted.variables == ("x", "y")


def test_a_variable_on_both_sides_is_not_solved_for_anything():
    # No reading of `z = x + y + z` names an axis, so it is refused, not read as a
    # surface.
    with pytest.raises(Unplottable):
        kind_of("z = x + y + z", ("x", "y", "z"))


def test_the_abscissa_of_a_curve_is_the_variable_it_holds():
    assert kind_of("SIN(t)", ("t",)).variables == ("t",)


def test_a_constant_two_vector_is_one_point_and_a_variable_one_is_a_path():
    assert kind_of("[1, 2]", ()).kind is PlotKind.DATA
    assert kind_of("[t, t^2]", ("t",)).kind is PlotKind.PARAMETRIC
    assert kind_of("[t, t^2]", ("t",)).variables == ("t",)


def test_three_variables_are_refused_by_name():
    with pytest.raises(Unplottable) as refusal:
        kind_of("a*x*y", ("x", "y", "a"))
    assert "depends on x, y, a" in str(refusal.value)
    assert "one or two variables" in str(refusal.value)


def test_a_shape_that_is_not_a_plot_says_so_without_naming_variables():
    with pytest.raises(Unplottable) as refusal:
        kind_of("[[1, 2, 3], [4, 5, 6]]", ())
    assert "not an expression this can plot" in str(refusal.value)


async def test_a_function_applied_to_an_equation_is_plotted_as_the_equation():
    # `IM(u = v)` is a call that evaluates to an equation, and the equation is drawn as
    # an implicit curve, not as a surface of the equality's truth value.
    session = Session()
    session.author("IM(2^(x + #i*y) = (x + #i*y)^6)")
    target = session.target("#1")
    assert await session.relation("#1") == "="
    plotted = classify(target, await session.variables("#1"), "#1", "=")
    assert plotted.kind is PlotKind.IMPLICIT
    assert plotted.variables == ("x", "y")


async def test_an_expression_that_evaluates_to_no_relation_reports_none():
    session = Session()
    session.author("SIN(x)")
    assert await session.relation("#1") is None


def test_a_tree_written_as_a_relation_keeps_its_own_operator():
    assert kind_of("x < 3", ("x",)).kind is PlotKind.REGION
    assert classify(parsed("x = 1"), ("x",), "x = 1", None).kind is PlotKind.IMPLICIT


async def test_classification_reads_what_a_label_means_and_not_what_it_says():
    # The command resolves `#1` before classifying, so the expression is classified,
    # not the label.
    session = Session()
    session.author("SIN(x)")
    target = session.target("#1")
    assert classify(target, await session.variables("#1"), "#1").kind is PlotKind.CURVE


# -- evaluation ---------------------------------------------------------------


def test_a_closure_masks_the_complex_half_of_a_square_root():
    values = closure("SQRT(x)")(np.array([-4.0, -1.0, 0.0, 1.0, 4.0]))
    assert np.isnan(values[0]) and np.isnan(values[1])
    assert list(values[2:]) == [0.0, 1.0, 2.0]


def test_a_closure_evaluates_a_function_that_has_only_a_real_reading():
    # `arctan2` refuses complex dtypes, which would make every point of `ATAN(-y, x)`
    # NaN.
    values = closure("ATAN(-y, x + 1/10)", ("x", "y"))(
        np.array([1.0, -1.0]), np.array([1.0, 1.0])
    )
    assert np.allclose(values, [np.arctan2(-1.0, 1.1), np.arctan2(-1.0, -0.9)])


def test_a_closure_evaluates_the_other_functions_of_the_reals_alone():
    # The same complex-dtype refusal, elsewhere in numpy.
    for text, answers in [
        ("FLOOR(x)", [-2.0, 1.0]),
        ("MOD(x, 2)", [0.5, 1.5]),
    ]:
        assert list(closure(text)(np.array([-1.5, 1.5]))) == answers


def test_a_closure_answers_nan_where_the_value_is_not_finite():
    values = closure("1/x")(np.array([-1.0, 0.0, 1.0]))
    assert list(values[[0, 2]]) == [-1.0, 1.0]
    assert np.isnan(values[1])


def test_a_closure_plots_the_heads_numpy_has_no_name_for():
    # lambdify's numpy namespace has no factorial, so `x!` would use `math.factorial`,
    # which refuses floats, and every sample would be NaN. Anything that approximates
    # must plot, so sympy evaluates the points numpy can't, as a general rule rather
    # than a list of known heads.
    for text, place, expected in [
        ("x!", 2.5, 3.32335097),
        ("ZETA(x)", 0.5, -1.46035450),
        ("SI(x)", 2.0, 1.60541298),
        ("EI(x)", 1.0, 1.89511782),
        ("SINH_INT(x)", 2.0, 2.50156743),
    ]:
        value = closure(text)(np.array([place]))[0]
        assert abs(value - expected) < 1e-6, f"{text} at {place} is {value}"


def test_a_closure_over_a_slow_head_still_gaps_its_poles():
    # The probe must tell an evaluator that can't answer from one that answers "no
    # value here", which look alike at a single point. The first samples of `x!` are
    # its poles at negative integers, so probing only the first sample would give up
    # on the curve beyond them.
    values = closure("x!")(np.array([-3.0, -2.0, -1.0, 0.5, 2.5]))
    assert np.isnan(values[:3]).all()
    assert abs(values[3] - 0.88622693) < 1e-6
    assert abs(values[4] - 3.32335097) < 1e-6


def test_a_closure_numpy_can_evaluate_never_builds_the_slow_rung():
    # The slow fallback must cost nothing for expressions numpy handles, so ordinary
    # curves stay fast to zoom.
    f = closure("SIN(x)")
    evaluate.sample_adaptive(f, VIEW, VIEW, CANVAS)
    assert f._candidates is None


def test_a_constant_closure_answers_one_value_per_point():
    values = closure("3")(np.linspace(-1.0, 1.0, 7))
    assert list(values) == [3.0] * 7


def test_a_closure_over_two_variables_takes_a_grid():
    xs, ys, values = evaluate.grid_eval(
        closure("x + y", ("x", "y")), (0.0, 1.0), (0.0, 1.0), 3, 5
    )
    assert values.shape == (3, 5)
    assert values[0][0] == 0.0
    assert values[2][4] == 2.0


def test_the_sampler_leaves_a_gap_at_a_pole_instead_of_a_stroke():
    # No drawn segment may cross the pole, so there must be a NaN between the samples
    # on either side.
    xs, ys = sampled("TAN(x)")
    assert np.isnan(ys).any()
    crossing = _spans(xs, ys, np.pi / 2)
    assert crossing == [], f"TAN(x) bridges its pole: {crossing}"


def test_the_sampler_leaves_a_gap_at_a_step():
    xs, ys = sampled("SIGN(x)")
    assert _spans(xs, ys, 0.0) == []


def test_the_sampler_gaps_the_pole_of_a_reciprocal():
    xs, ys = sampled("1/x")
    assert _spans(xs, ys, 0.0) == []


def test_the_sampler_draws_only_the_real_half_of_a_square_root():
    xs, ys = sampled("SQRT(x)")
    assert not np.isfinite(ys[xs < -0.01]).any()
    assert np.isfinite(ys[xs > 0.01]).all()


def test_the_sampler_says_nothing_is_real_where_nothing_is():
    xs, ys = sampled("SQRT(-1 - x^2)")
    assert evaluate.finite_fraction(ys) == 0.0


def test_a_spike_narrower_than_a_pixel_resolves_when_it_is_zoomed_to():
    # This is why we re-sample on every view change: at the default framing the spike
    # falls between two samples, and zooming in reveals it.
    text = "1/(1 + 10000*(x - 1)^2)"
    _, wide = sampled(text)
    _, close = sampled(text, xrange=(0.99, 1.01), yrange=(-0.1, 1.1))
    assert np.nanmax(wide) < 0.5
    assert np.nanmax(close) > 0.99


def test_a_curve_with_a_pole_in_it_still_terminates():
    # A pole at the range midpoint makes naive bisection recurse forever.
    xs, ys = sampled("1/x", xrange=(-1.0, 1.0))
    assert len(xs) < evaluate.MAX_POINTS
    assert np.isfinite(ys).any()


def test_the_uniform_pass_is_reported_before_the_refinement():
    seen = []
    evaluate.sample_adaptive(
        closure("SIN(x)"), VIEW, VIEW, CANVAS, report=lambda xs, ys: seen.append(xs)
    )
    assert len(seen) == 1
    assert len(seen[0]) == evaluate.INITIAL_POINTS


def test_a_tree_that_will_not_lambdify_is_refused_in_words():
    # Classification refuses an assignment long before this; what's tested here is
    # that the failure is a message, not a traceback.
    with pytest.raises(evaluate.Unplottable):
        evaluate.closure(parsed("x := 3"), Context(), ("x",))


def test_a_function_with_no_numeric_reading_is_all_gaps_rather_than_a_crash():
    # `FOO` is undefined, so every point is unevaluable: a curve with no real values.
    values = closure("FOO(x)")(np.array([1.0, 2.0]))
    assert evaluate.finite_fraction(values) == 0.0


def test_a_parametric_pair_through_a_pole_terminates_with_a_gap():
    # The two branches never come close on screen however finely t is cut, so
    # refinement must stop at the depth cap rather than chase the pole forever. The
    # pole is at 1, not 0, so it falls between uniform samples and must be found
    # rather than handed over as the NaN of `1/0`.
    fx, fy = evaluate.pair(parsed("[t, 1/(t - 1)]"), Context(), ("t",))
    drawn = evaluate.sample_curve(fx, fy, VIEW, VIEW, VIEW, CANVAS)
    assert len(drawn.ts) < evaluate.MAX_POINTS
    assert drawn.gave_up is not None and abs(drawn.gave_up - 1.0) < 0.01
    # The gap is at the pole, not along the branches.
    gapped = drawn.ts[np.isnan(drawn.xs)]
    assert gapped.size and np.abs(gapped - 1.0).max() < 0.05


def test_a_curve_faster_than_the_pixels_is_drawn_rather_than_gapped():
    # Zoomed out, the flanks of `x·SIN(x)` are steeper than the screen and refinement
    # bottoms out on them. Treating that as a jump would cut the curve into a comb. A
    # slope halves its height with each bisection, which distinguishes it from a
    # discontinuity.
    for span in (400.0, 2000.0):
        view = (-span / 2, span / 2)
        xs, ys = sampled("x*SIN(x)", xrange=view, yrange=view, size=(900.0, 600.0))
        assert not np.isnan(ys).any(), f"x·SIN(x) over {view} is cut into pieces"
    # The same curve as a parametric pair, which uses the other sampler.
    fx, fy = evaluate.pair(parsed("[t, t*SIN(t)]"), Context(), ("t",))
    wide = (-1000.0, 1000.0)
    drawn = evaluate.sample_curve(fx, fy, wide, wide, wide, (900.0, 600.0))
    assert not np.isnan(drawn.xs).any()
    assert drawn.gave_up is None


def test_a_jump_is_still_a_jump_where_the_curve_around_it_is_steep():
    # A pole keeps its full height in whichever half it lands in, however deep the
    # bisection, so `TAN(x)` is gapped at every pole even though its steep branches
    # are drawn.
    xs, ys = sampled("TAN(x)")
    for pole in (-3 * np.pi / 2, -np.pi / 2, np.pi / 2, 3 * np.pi / 2):
        assert _spans(xs, ys, pole) == [], f"TAN(x) bridges the pole at {pole}"


def test_a_polar_curve_turns_its_angle_in_the_unit_it_was_written_in():
    # `r = 1` at 90 is the top of the unit circle in degrees but not in radians, so the
    # angle unit must be a parameter.
    one = closure("1")
    for degrees, expected in ((True, 90.0), (False, np.pi / 2)):
        horizontal, vertical = evaluate.polar_pair(one, degrees=degrees)
        assert abs(float(horizontal(np.array([expected]))[0])) < 1e-9
        assert abs(float(vertical(np.array([expected]))[0]) - 1.0) < 1e-9


def test_a_constant_matrix_becomes_the_two_columns_of_a_data_plot():
    xs, ys = evaluate.points(parsed("[[1, 2], [3, 4], [5, 6]]"), Context())
    assert list(xs) == [1.0, 3.0, 5.0]
    assert list(ys) == [2.0, 4.0, 6.0]
    # A single point reads the same way, as a one-row matrix.
    xs, ys = evaluate.points(parsed("[1, 2]"), Context())
    assert (list(xs), list(ys)) == ([1.0], [2.0])


def test_an_equation_is_evaluated_as_the_difference_of_its_sides():
    # An implicit curve is the zero contour of `u - v`, so the equation must become a
    # single expression before marching squares sees it.
    g = evaluate.difference(parsed("x^2 + y^2 = 4"), Context(), ("x", "y"))
    assert float(g(np.array([2.0]), np.array([0.0]))[0]) == 0.0
    assert float(g(np.array([0.0]), np.array([0.0]))[0]) == -4.0


def test_an_inequality_is_a_truth_value_at_every_point():
    inside = evaluate.mask(parsed("x^2 + y^2 <= 4"), Context(), ("x", "y"))
    answer = inside(np.array([0.0, 3.0]), np.array([0.0, 0.0]))
    assert answer.dtype == bool
    assert list(answer) == [True, False]


def test_a_bound_written_as_an_expression_is_worth_a_number():
    # Toolbar fields accept expressions like `-π`, evaluated by the engine.
    assert evaluate.number(parsed("-π"), Context(), 0.0) == pytest.approx(-np.pi)
    assert evaluate.number(None, Context(), 1.5) == 1.5
    # A non-numeric answer falls back to the default rather than being refused.
    assert evaluate.number(parsed("1/0"), Context(), 2.5) == 2.5


def test_features_finds_the_roots_and_extrema_of_a_sine():
    f = closure("SIN(x)")
    xs, ys = evaluate.sample_adaptive(f, VIEW, VIEW, CANVAS)
    found = evaluate.features(xs, ys, f)
    roots = [item.x for item in found if item.kind == evaluate.ROOT]
    assert roots == pytest.approx([-np.pi, 0.0, np.pi], abs=1e-9)
    peaks = [item.x for item in found if item.kind == evaluate.MAXIMUM]
    dips = [item.x for item in found if item.kind == evaluate.MINIMUM]
    assert peaks == pytest.approx([-3 * np.pi / 2, np.pi / 2], abs=1e-6)
    assert dips == pytest.approx([-np.pi / 2, 3 * np.pi / 2], abs=1e-6)


def test_features_finds_where_two_curves_cross():
    f, g = closure("SIN(x)"), closure("COS(x)")
    xs, ys = evaluate.sample_adaptive(f, VIEW, VIEW, CANVAS)
    crossings = [
        item
        for item in evaluate.features(xs, ys, f, [g])
        if item.kind == evaluate.CROSSING
    ]
    assert [item.x for item in crossings] == pytest.approx(
        [-3 * np.pi / 4, np.pi / 4, 5 * np.pi / 4], abs=1e-9
    )
    # A crossing carries its height and the index of the other curve.
    assert [item.other for item in crossings] == [0, 0, 0]
    assert crossings[1].y == pytest.approx(np.sqrt(0.5))


def test_the_pole_of_a_tangent_is_not_called_a_root():
    # `TAN(x)` changes sign across every pole, so a sign-only scan would report four
    # false roots. The NaN the sampler puts at each pole keeps them out.
    f = closure("TAN(x)")
    xs, ys = evaluate.sample_adaptive(f, VIEW, VIEW, CANVAS)
    found = evaluate.features(xs, ys, f)
    roots = [item.x for item in found if item.kind == evaluate.ROOT]
    assert roots == pytest.approx([-np.pi, 0.0, np.pi], abs=1e-9)
    assert not any(abs(abs(root) - np.pi / 2) < 0.1 for root in roots)


def test_a_feature_scan_over_nothing_finds_nothing():
    assert evaluate.features(np.empty(0), np.empty(0), closure("SIN(x)")) == ()


def _spans(xs, ys, place):
    """Return the drawn segments crossing `place`, i.e. strokes bridging a discontinuity.

    A segment is drawn between adjacent samples when both are finite.
    """
    left, right = xs[:-1], xs[1:]
    straddles = (left < place) & (right > place)
    finite = np.isfinite(ys[:-1]) & np.isfinite(ys[1:])
    return [
        (float(left[index]), float(right[index]))
        for index in np.nonzero(straddles & finite)[0]
    ]


# -- the solid: the mesh, the box, the ticks -----------------------------------
#
# Pure functions, so the 3D window's logic is tested without a graphics card: which faces
# survive a hole in the domain, where the box stands, and its tick labels. GL rendering
# itself is exercised only by the offscreen host below.


@pytest.fixture(scope="module")
def solid():
    """The surface geometry module, which needs no toolkit."""
    from rederive.plot import surface

    return surface


@pytest.fixture(scope="module")
def space(qt):
    """The 3D window module, skipping if OpenGL is unavailable."""
    try:
        from rederive.plot.qt import window3d
    except ImportError as missing:  # no PyOpenGL, or no libGL to load it with
        pytest.skip(f"pyqtgraph.opengl will not import: {missing}")
    return window3d


def _grid(f, span=(-2.0, 2.0), n=33):
    """Evaluate `f` on a grid over a square domain, as the window does."""
    xs = np.linspace(span[0], span[1], n)
    ys = np.linspace(span[0], span[1], n)
    across, along = np.meshgrid(xs, ys, indexing="ij")
    with np.errstate(all="ignore"):
        values = f(across, along)
    return xs, ys, np.where(np.isfinite(values), values, np.nan)


def test_a_whole_grid_becomes_two_triangles_a_cell(solid):
    xs, ys, values = _grid(lambda x, y: x + y, n=8)
    box = solid.Box((-2.0, 2.0), (-2.0, 2.0), (-4.0, 4.0))
    vertexes, faces, shading = solid.mesh(xs, ys, values, box)
    assert vertexes.shape == (64, 3)
    assert faces.shape == (2 * 7 * 7, 3)
    assert shading.shape == (64,)


def test_the_faces_of_a_dome_stop_at_the_edge_of_its_domain(solid):
    # `SQRT(1-x^2-y^2)` over [-2, 2] is a dome over the unit disc and undefined outside
    # it. Every face must lie inside the circle, and enough must remain to form a dome.
    xs, ys, values = _grid(lambda x, y: np.sqrt(1 - x**2 - y**2))
    box = solid.Box((-2.0, 2.0), (-2.0, 2.0), (0.0, 1.0))
    vertexes, faces, _ = solid.mesh(xs, ys, values, box)
    assert faces.size
    corners = vertexes[faces.reshape(-1)]
    # Convert to data coordinates. The world is a square of side WORLD representing
    # the domain.
    scale = (2.0 - -2.0) / solid.WORLD
    radius = np.hypot(corners[:, 0] * scale, corners[:, 1] * scale)
    assert radius.max() <= 1.0
    # The disc covers about a quarter of the square, so about a quarter of the faces
    # survive; fewer than a tenth would not be a dome.
    assert len(faces) > 0.1 * 2 * 32 * 32


def test_a_surface_with_no_real_values_has_no_faces_at_all(solid):
    xs, ys, values = _grid(lambda x, y: np.sqrt(-1 - x**2 - y**2))
    box = solid.Box((-2.0, 2.0), (-2.0, 2.0), (0.0, 1.0))
    _, faces, _ = solid.mesh(xs, ys, values, box)
    assert not faces.size
    assert solid.extent([values]) is None


def test_a_vertex_beyond_the_clip_drops_its_faces_like_a_hole(solid):
    xs, ys, values = _grid(lambda x, y: x + y, span=(-2.0, 2.0), n=9)
    whole = solid.Box((-2.0, 2.0), (-2.0, 2.0), (-4.0, 4.0))
    clipped = solid.Box((-2.0, 2.0), (-2.0, 2.0), (-1.0, 1.0))
    assert len(solid.mesh(xs, ys, values, clipped)[1]) < len(
        solid.mesh(xs, ys, values, whole)[1]
    )
    heights = solid.mesh(xs, ys, values, clipped)[0][:, 2]
    assert heights.max() <= solid.HALF + 1e-6


def test_the_extent_is_the_data_until_a_spike_would_crush_it(solid):
    values = np.linspace(-2.0, 2.0, 400)
    (low, high), clipped = solid.extent([values])
    assert (low, high) == (-2.0, 2.0)
    assert clipped is False
    # With one pole among 400 ordinary values, the box uses percentiles and reports
    # that it clipped.
    spiked = values.copy()
    spiked[0] = 1e6
    (low, high), clipped = solid.extent([spiked])
    assert clipped is True
    assert high < 10.0


def test_a_flat_surface_still_gets_a_box_to_stand_in(solid):
    (low, high), clipped = solid.extent([np.full(100, 3.0)])
    assert low < 3.0 < high
    assert clipped is False


def test_every_range_is_stretched_onto_the_same_cube(solid):
    # Each range is stretched onto the same cube, so a surface with a tiny range
    # doesn't look flat and one with a huge range doesn't look like a wall.
    for span in ((0.0, 0.001), (0.0, 1.0), (-1e6, 1e6)):
        box = solid.Box((-5.0, 5.0), (-5.0, 5.0), span)
        assert box.up(span[0]) == pytest.approx(-solid.HALF)
        assert box.up(span[1]) == pytest.approx(solid.HALF)
    # A surface and its scaled copy look the same; only the z-axis labels show the
    # magnitude.
    xs, ys, values = _grid(lambda x, y: 1.0 / (9.0 + x**2 + y**2))
    box = solid.Box((-2.0, 2.0), (-2.0, 2.0), solid.extent([values])[0])
    scaled = 100.0 * values
    lifted = solid.Box((-2.0, 2.0), (-2.0, 2.0), solid.extent([scaled])[0])
    assert solid.mesh(xs, ys, values, box)[0] == pytest.approx(
        solid.mesh(xs, ys, scaled, lifted)[0]
    )


def test_the_box_reports_its_center_and_its_lengths(solid):
    box = solid.Box((-2.0, 6.0), (-5.0, 5.0), (0.0, 1.0))
    assert box.center == (2.0, 0.0, 0.5)
    assert box.lengths == (8.0, 10.0, 1.0)


def test_the_tick_marks_of_an_axis_are_round_numbers(solid):
    assert solid.ticks(-5.0, 5.0) == [-4.0, -2.0, 0.0, 2.0, 4.0]
    assert solid.ticks(0.0, 1.0) == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    assert solid.ticks(3.0, 3.0) == []
    # Zero comes out as exactly zero, not a rounding error.
    assert 0.0 in solid.ticks(-0.3, 0.3)
    assert all(abs(value) > 1e-9 or value == 0.0 for value in solid.ticks(-0.3, 0.3))


def test_a_vertex_is_darkest_at_the_floor_of_the_box(solid):
    colors = solid.brightened(np.array([0.0, 1.0]), "#ffffff")
    assert colors.shape == (2, 4)
    assert colors[0][0] < colors[1][0]
    assert colors[1][0] == pytest.approx(1.0)
    assert list(colors[:, 3]) == [1.0, 1.0]


def test_the_shading_of_a_surface_runs_from_dim_to_full(solid):
    xs, ys, values = _grid(lambda x, y: x + y, n=16)
    box = solid.Box((-2.0, 2.0), (-2.0, 2.0), (-4.0, 4.0))
    shading = solid.mesh(xs, ys, values, box)[2]
    assert 0.0 < shading.min() < shading.max() <= 1.0


# -- the refined boundary: a mesh that stops where the surface does ------------
#
# Section 9's geometry, still without a graphics card: the sampler bisects the domain's
# edge, the mesh fills partial cells from the crossings, and the clip trims by
# interpolation.


def _dome(x, y):
    with np.errstate(all="ignore"):
        return np.sqrt(1 - x**2 - y**2)


def test_the_domes_rim_is_refined_onto_the_unit_circle(solid):
    # At the default 64 x 64 grid, the rim of `SQRT(1-x^2-y^2)` over [-2, 2] lies on
    # the unit circle to within a fraction of a cell, on the defined side, without the
    # jagged z values of a grid-aligned boundary.
    xs, ys, values = _grid(_dome, n=64)
    boundary = evaluate.grid_boundary(_dome, xs, ys, values)
    cell = xs[1] - xs[0]
    crossed = np.isfinite(boundary.across)
    assert crossed.any()
    radius = np.hypot(
        boundary.across[crossed], np.broadcast_to(ys, boundary.across.shape)[crossed]
    )
    assert radius.max() <= 1.0 + 1e-9
    assert radius.min() >= 1.0 - cell / 8
    assert boundary.across_z[crossed].max() <= 0.1
    # The mesh built from the crossings reaches the circle but never goes past it.
    box = solid.Box((-2.0, 2.0), (-2.0, 2.0), (0.0, 1.0))
    vertexes, faces, shading = solid.mesh(xs, ys, values, box, boundary)
    assert len(faces) > len(solid.mesh(xs, ys, values, box)[1])
    used = vertexes[np.unique(faces.reshape(-1))]
    scale = (2.0 - -2.0) / solid.WORLD
    reach = np.hypot(used[:, 0] * scale, used[:, 1] * scale)
    assert reach.max() <= 1.0 + 1e-6
    assert reach.max() >= 1.0 - cell / 8
    assert np.isfinite(shading).all()


def test_a_grid_aligned_boundary_stays_straight(solid):
    # `SQRT(x)` ends exactly on the grid line x = 0, and refinement must not make it
    # wobble.
    def half(x, y):
        with np.errstate(all="ignore"):
            return np.sqrt(x) + 0.0 * y

    xs, ys, values = _grid(half)
    boundary = evaluate.grid_boundary(half, xs, ys, values)
    crossed = np.isfinite(boundary.across)
    assert crossed.any()
    assert (boundary.across[crossed] == 0.0).all()
    assert not np.isfinite(boundary.along).any()
    box = solid.Box((-2.0, 2.0), (-2.0, 2.0), (0.0, 1.5))
    vertexes, faces, _ = solid.mesh(xs, ys, values, box, boundary)
    used = vertexes[np.unique(faces.reshape(-1))]
    assert used[:, 0].min() >= -1e-9  # x = 0 is the world's own origin here


def test_the_ambiguous_cell_is_resolved_one_fixed_way(solid):
    # A cell with two diagonally opposite defined corners is filled as two separate
    # triangles, each with one defined corner, whichever diagonal it is.
    def diagonal(x, y):
        return np.where(np.abs(x - y) < 0.3, 0.5, np.nan)

    def other(x, y):
        return np.where(np.abs(x + y - 1.0) < 0.3, 0.5, np.nan)

    xs = np.array([0.0, 1.0])
    ys = np.array([0.0, 1.0])
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    box = solid.Box((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    corners = {
        (-solid.HALF, -solid.HALF),
        (solid.HALF, -solid.HALF),
        (solid.HALF, solid.HALF),
        (-solid.HALF, solid.HALF),
    }
    for f in (diagonal, other):
        values = f(grid_x, grid_y)
        boundary = evaluate.grid_boundary(f, xs, ys, values)
        vertexes, faces, _ = solid.mesh(xs, ys, values, box, boundary)
        assert len(faces) == 2
        for face in faces:
            touched = [
                tuple(np.round(vertexes[k][:2], 6).tolist()) in corners for k in face
            ]
            assert sum(touched) == 1


def test_an_unbounded_boundary_value_is_trimmed_by_the_clip(solid):
    # `LOG(x)` goes to minus infinity at its boundary, so the refined boundary vertex
    # is huge. The clip trims it, and the mesh ends exactly on the clip plane rather
    # than outside the box or a cell short.
    def logged(x, y):
        with np.errstate(all="ignore"):
            return np.log(x) + 0.0 * y

    xs, ys, values = _grid(logged)
    boundary = evaluate.grid_boundary(logged, xs, ys, values)
    crossed = np.isfinite(boundary.across)
    assert boundary.across_z[crossed].min() < -4.0
    box = solid.Box((-2.0, 2.0), (-2.0, 2.0), (-2.0, 1.0))
    vertexes, faces, shading = solid.mesh(xs, ys, values, box, boundary)
    used = vertexes[np.unique(faces.reshape(-1))]
    top = solid.HALF
    assert used[:, 2].min() >= -top - 1e-5
    assert used[:, 2].max() <= top + 1e-5
    assert used[:, 2].min() == pytest.approx(-top, abs=1e-5)
    assert np.isfinite(shading).all()


# -- the wire: the grid of the samples, on request -------------------------------
#
# Section 10's geometry, also pure: row and column polylines from the same arrays,
# thinned when drawn but not when sampled, ending on the refined boundary and trimmed by
# the same clip.


def test_the_wire_is_thinned_rows_of_dense_polylines(solid):
    # At 64 x 64 the full grid would be gray fuzz, so only every k-th row and column
    # is drawn (16 to 24 each way), but each drawn line keeps full sample spacing.
    xs, ys, values = _grid(lambda x, y: x + y, n=64)
    box = solid.Box((-2.0, 2.0), (-2.0, 2.0), (-4.0, 4.0))
    points, shades = solid.wire(xs, ys, values, box)
    assert len(points) % 2 == 0
    assert len(shades) == len(points)
    segments = points.reshape(-1, 2, 3)
    step = solid.WORLD / 63
    for axis in (0, 1):
        lines = segments[np.isclose(segments[:, 0, axis], segments[:, 1, axis])]
        drawn = np.unique(np.round(lines[:, 0, axis], 5))
        assert 16 <= len(drawn) <= 24
        # The first and last rows are always drawn.
        assert np.isclose(drawn.min(), -solid.HALF)
        assert np.isclose(drawn.max(), solid.HALF)
        # Dense polylines: every segment spans exactly one grid step.
        along = lines[:, 1, 1 - axis] - lines[:, 0, 1 - axis]
        assert np.abs(along) == pytest.approx(np.full(len(along), step), rel=1e-4)
    assert 0.0 < shades.min() <= shades.max() <= 1.0


def test_the_wires_rim_ends_on_the_refined_boundary(solid):
    # The wire dome has a clean circular rim: it ends on the unit circle, not a grid
    # step short, and no segment goes past it. Holes in the wire match the solid.
    xs, ys, values = _grid(_dome, n=64)
    boundary = evaluate.grid_boundary(_dome, xs, ys, values)
    box = solid.Box((-2.0, 2.0), (-2.0, 2.0), (0.0, 1.0))
    points, shades = solid.wire(xs, ys, values, box, boundary)
    cell = xs[1] - xs[0]
    scale = (2.0 - -2.0) / solid.WORLD
    radius = np.hypot(points[:, 0] * scale, points[:, 1] * scale)
    assert radius.max() <= 1.0 + 1e-6
    assert radius.max() >= 1.0 - cell / 8
    assert np.isfinite(shades).all()
    # Without the refinement the wire stops at the last defined sample.
    bare = solid.wire(xs, ys, values, box)[0]
    short = np.hypot(bare[:, 0] * scale, bare[:, 1] * scale)
    assert short.max() < radius.max()


def test_a_wire_segment_beyond_the_clip_is_trimmed_to_it(solid):
    # The clip trims the wire by interpolation as it does the faces, including
    # unbounded boundary ends, so the wire ends exactly on the clip plane.
    def logged(x, y):
        with np.errstate(all="ignore"):
            return np.log(x) + 0.0 * y

    xs, ys, values = _grid(logged, n=64)
    boundary = evaluate.grid_boundary(logged, xs, ys, values)
    box = solid.Box((-2.0, 2.0), (-2.0, 2.0), (-2.0, 1.0))
    points, _ = solid.wire(xs, ys, values, box, boundary)
    top = solid.HALF
    assert points[:, 2].min() >= -top - 1e-5
    assert points[:, 2].max() <= top + 1e-5
    assert points[:, 2].min() == pytest.approx(-top, abs=1e-5)


# -- window titles ---------------------------------------------------------------
#
# A window is titled by its contents, not its protocol number. These are pure functions,
# so no toolkit is needed.


def test_a_window_is_titled_by_its_contents():
    titled = plots.titled(plots.WindowKind.TWO_D, ("SIN(x)", "COS(x)"))
    assert titled == "SIN(x), COS(x) - Rederive plot"
    titled = plots.titled(plots.WindowKind.THREE_D, ("x^2 - y^2",), current=True)
    assert titled == "x^2 - y^2 - Rederive 3D plot (current)"


def test_an_empty_window_is_titled_by_its_kind_alone():
    assert plots.titled(plots.WindowKind.TWO_D) == "Rederive plot"
    assert plots.titled(plots.WindowKind.THREE_D) == "Rederive 3D plot"


def test_a_title_that_outgrows_a_taskbar_is_cut_with_an_ellipsis():
    titled = plots.titled(plots.WindowKind.TWO_D, ("SIN(x)",) * 12)
    assert titled.endswith("... - Rederive plot")
    assert len(titled) <= plots.TITLE_WIDTH + len(" - Rederive plot")


# -- the host over a real pipe -------------------------------------------------


def _toolkit():
    """Whether the host's toolkit is installed, checked without importing it."""
    from importlib.util import find_spec

    return find_spec("pyqtgraph") is not None and find_spec("PySide6") is not None


@pytest.fixture
def host(monkeypatch):
    """A plot host in a child process, rendering offscreen.

    Offscreen means no windows flash up, and the test runs without a display. Every
    wait has a deadline and the host is always shut down, so a host that won't start
    fails the test instead of hanging.
    """
    if not _toolkit():
        pytest.skip("pyqtgraph and PySide6 are not installed")
    from rederive.plot import proxy as proxy_module

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr(proxy_module, "READY_TIMEOUT", 60.0)
    monkeypatch.setattr(proxy_module, "REPLY_TIMEOUT", 20.0)
    events = []
    proxy = proxy_module.PlotProxy(events.append)
    proxy.events_seen = events
    try:
        yield proxy
    finally:
        proxy.shutdown()


async def _add(session, host, text, **keywords):
    """Plot an expression as the Plot command does, returning its window number."""
    entry = session.author(text)
    label = f"#{entry.number}"
    plotted = classify(entry.node, await session.variables(label), text)
    fields = {
        "worksheet": id(session),
        "node": entry.node,
        "context": session.context,
        "kind": plotted.kind,
        "label": label,
        "text": text,
        "options": plotted.options,
    }
    placed = await host.add(plots.Add(**{**fields, **keywords}))
    return placed.window


async def test_a_host_takes_a_plot_and_describes_what_it_holds(host):
    session = Session()
    assert await _add(session, host, "SIN(x)") == 1
    assert await _add(session, host, "x^2 - 3") == 1
    described = host.describe()
    assert len(described) == 1
    window = described[0]
    assert window.number == 1
    assert window.kind is plots.WindowKind.TWO_D
    assert window.title == "SIN(x), x^2 - 3 - Rederive plot (current)"
    assert window.current is True
    assert [plot.label for plot in window.plots] == ["#1", "#2"]
    assert [plot.text for plot in window.plots] == ["SIN(x)", "x^2 - 3"]


async def test_re_plotting_a_label_replaces_its_curve_and_says_so(host):
    session = Session()
    await _add(session, host, "SIN(x)")
    entry = session.entries[0]
    placed = host.plot(
        plots.Add(
            worksheet=id(session),
            node=entry.node,
            context=session.context,
            kind=PlotKind.CURVE,
            label="#1",
            text="SIN(x)",
            options=plots.Options(variables=("x",)),
        )
    )
    # The reply says the curve was replaced, which the acknowledgement message uses.
    assert placed == plots.Placed(1, replaced=True)
    assert len(host.describe()[0].plots) == 1


async def test_a_new_window_takes_the_next_number_and_becomes_current(host):
    session = Session()
    assert await _add(session, host, "SIN(x)") == 1
    assert await _add(session, host, "COS(x)", window=plots.Where.NEW) == 2
    described = host.describe()
    assert [window.number for window in described] == [1, 2]
    assert [window.current for window in described] == [False, True]
    assert described[0].title == "SIN(x) - Rederive plot"
    assert described[1].title == "COS(x) - Rederive plot (current)"
    # The next plot lands in the receiver, which is the newest window.
    assert await _add(session, host, "TAN(x)") == 2


def test_a_family_becomes_one_curve_per_element(host):
    session = Session()
    entry = session.author("[x, x^2, x^3]")
    host.plot(
        plots.Add(
            worksheet=id(session),
            node=entry.node,
            context=session.context,
            kind=PlotKind.FAMILY,
            label="#1",
            text="[x, x^2, x^3]",
            options=plots.Options(variables=("x",), texts=("x", "x^2", "x^3")),
        )
    )
    window = host.describe()[0]
    assert [plot.label for plot in window.plots] == ["#1.1", "#1.2", "#1.3"]
    assert [plot.text for plot in window.plots] == ["x", "x^2", "x^3"]


async def test_one_window_takes_every_two_dimensional_kind(host):
    # A parametric pair, a rose, points, a contour and a shaded region all share one
    # window and legend. One test, because each test costs a process start.
    session = Session()
    await _add(session, host, "[SIN(t), COS(t)]", options=plots.Options(variables=("t",)))
    await _add(session, host, "[[1, 2], [3, 4]]")
    await _add(session, host, "x^2 + y^2 = 4")
    await _add(session, host, "y < x^2")
    # Polar is a view mode, not a plot kind, so a request of polar kind lands as a
    # curve.
    await _add(
        session,
        host,
        "2*COS(3*t)",
        kind=PlotKind.POLAR,
        options=plots.Options(variables=("t",)),
    )
    window = host.describe()[0]
    assert [plot.kind for plot in window.plots] == [
        PlotKind.PARAMETRIC,
        PlotKind.DATA,
        PlotKind.IMPLICIT,
        PlotKind.REGION,
        PlotKind.CURVE,
    ]
    assert not [plot.label for plot in window.plots if plot.hidden]


async def test_a_surface_opens_a_solid_window_of_its_own(host):
    # Curves and surfaces never share a window, and each kind has its own current
    # window, so the next plot of either kind lands with the last. One test, because
    # each test costs a process start.
    session = Session()
    assert await _add(session, host, "SIN(x)") == 1
    assert await _add(session, host, "x^2 - y^2") == 2
    flat, solid = host.describe()
    assert (flat.kind, solid.kind) == (
        plots.WindowKind.TWO_D,
        plots.WindowKind.THREE_D,
    )
    assert solid.title == "x^2 - y^2 - Rederive 3D plot (current)"
    assert flat.title == "SIN(x) - Rederive plot (current)"
    assert [plot.kind for plot in solid.plots] == [PlotKind.SURFACE]
    # The window reports its domain, here the default.
    assert solid.xrange == (-5.0, 5.0)
    assert solid.yrange == (-5.0, 5.0)
    assert await _add(session, host, "SIN(x*y)") == 2
    assert await _add(session, host, "COS(x)") == 1


def test_a_vector_of_surfaces_becomes_one_surface_per_element(host):
    session = Session()
    entry = session.author("[x + y, x - y]")
    host.plot(
        plots.Add(
            worksheet=id(session),
            node=entry.node,
            context=session.context,
            kind=PlotKind.SURFACES,
            label="#1",
            text="[x + y, x - y]",
            options=plots.Options(variables=("x", "y"), texts=("x + y", "x - y")),
        )
    )
    window = host.describe()[0]
    assert [plot.label for plot in window.plots] == ["#1.1", "#1.2"]
    assert [plot.text for plot in window.plots] == ["x + y", "x - y"]
    assert [plot.kind for plot in window.plots] == [PlotKind.SURFACE] * 2


async def test_a_curve_that_will_not_evaluate_reports_itself(host):
    session = Session()
    # A window with nothing to draw in reports over the pipe instead; the event is
    # the only sign.
    await _add(session, host, "SIN(x)")
    assert host.describe()[0].plots[0].label == "#1"


async def test_the_host_is_started_once_and_stopped_when_it_is_asked_to(host):
    session = Session()
    await _add(session, host, "SIN(x)")
    await _add(session, host, "COS(x)")
    assert host.starts == 1
    assert host.running
    host.shutdown()
    assert not host.running


def test_describing_with_no_host_answers_nothing_without_starting_one():
    from rederive.plot import proxy as proxy_module

    proxy = proxy_module.PlotProxy()
    assert proxy.describe() == ()
    assert proxy.starts == 0


async def test_letting_a_demonstration_go_is_waited_for_off_the_loop(monkeypatch):
    """Waits for the host must not block the app's loop.

    A key pressed in the plot window can end a demonstration. The key arrives on the
    proxy's reader thread, which hands it to the loop and waits for it to be taken, so
    blocking the loop on the host's reply deadlocks until the reply timeout, after which
    a live host is declared dead.
    """
    from rederive.plot import proxy as proxy_module

    proxy = proxy_module.PlotProxy()
    holding = threading.Event()
    monkeypatch.setattr(type(proxy), "running", property(lambda self: True))
    monkeypatch.setattr(proxy, "_ask", lambda request: holding.wait(5.0))
    letting_go = asyncio.create_task(proxy.release())
    # The loop must keep turning while the host is awaited.
    await asyncio.sleep(0)
    assert not letting_go.done()
    holding.set()
    await letting_go


async def test_letting_go_with_no_host_says_nothing_and_starts_none():
    from rederive.plot import proxy as proxy_module

    proxy = proxy_module.PlotProxy()
    await proxy.release()
    assert proxy.starts == 0


def test_a_host_that_will_not_start_reports_its_own_words(monkeypatch):
    # A missing package or a display Qt can't open are the usual start failures, and
    # their messages must be shown, not reported as a process death. Nothing is spawned;
    # this tests how the proxy handles the handshake reply.
    from rederive.plot import proxy as proxy_module

    proxy = proxy_module.PlotProxy()

    def refusing():
        proxy._replies[plots.READY] = plots.Refused("no module named pyqtgraph")

    monkeypatch.setattr(proxy, "_spawn", refusing)
    with pytest.raises(PlotError) as refused:
        proxy.plot(
            plots.Add(
                worksheet=0,
                node=parsed("SIN(x)"),
                context=Context(),
                kind=PlotKind.CURVE,
            )
        )
    assert str(refused.value) == "no module named pyqtgraph"


def test_a_plot_with_the_extra_not_installed_says_what_to_install(monkeypatch):
    # The `plot` extra is opt-in, so its absence is a normal install. The user should
    # be told what to install, not shown an ImportError, and without spawning a process
    # first (`starts` stays 0).
    from rederive.plot import proxy as proxy_module

    proxy = proxy_module.PlotProxy()
    monkeypatch.setattr(proxy_module, "_installed", lambda name: False)
    with pytest.raises(PlotError) as refused:
        proxy.plot(
            plots.Add(
                worksheet=0,
                node=parsed("SIN(x)"),
                context=Context(),
                kind=PlotKind.CURVE,
            )
        )
    assert str(refused.value) == proxy_module.UNINSTALLED
    assert proxy.starts == 0


async def test_a_host_takes_the_preferences_before_the_plot_that_follows(host):
    """A preference request over a real pipe, followed by a plot that still lands.

    `Describe` reports a window's contents, not how they were drawn, so this only
    tests the round trip: the host accepts the request and the next plot still lands
    where expected.
    """
    session = Session()
    host.prefer(plots.Prefer(grid=16, connected=True, point_size=9))
    assert await _add(session, host, "SIN(x)") == 1
    assert [plot.label for plot in host.describe()[0].plots] == ["#1"]


# -- the windows' own fields ----------------------------------------------------
#
# The 2D range fields and 3D domain fields are tested in-process and offscreen, with
# sampling run on the calling thread. This tests the wiring from typed expression to
# re-sampled plot; the host tests above already cover the pipe.


class InlineSession:
    """A plot session that runs each sampling job before `sample` returns."""

    def __init__(self):
        #: Points windows have sent to the worksheet, as (worksheet, text) pairs.
        self.authored = []
        #: Sticky preference values windows have sent back, merged.
        self.adjustments = {}
        #: One entry per key a window sent back while a demonstration waited in it. The
        #: key itself doesn't matter.
        self.steps = []

    def sample(self, key, work, done, report=None):
        try:
            answer = work(lambda *arguments: None)
        except Exception as error:
            answer = error
        done(answer)

    def trouble(self, window, label, message):
        pass

    def touched(self, number):
        pass

    def closed(self, number):
        pass

    def author(self, worksheet, text):
        self.authored.append((worksheet, text))

    def stepped(self):
        self.steps.append(None)

    def adjusted(self, **values):
        self.adjustments.update(values)


class QueuedSession(InlineSession):
    """A session that queues each job until the test runs it, as a thread would.

    This shows that a window submitted an evaluation rather than doing it inline,
    which `InlineSession` can't distinguish.
    """

    def __init__(self):
        super().__init__()
        #: Submitted jobs not yet run, as (key, work, done) triples.
        self.jobs = []

    def sample(self, key, work, done, report=None):
        self.jobs.append((key, work, done))

    def run(self):
        """Run the jobs queued so far, leaving anything they queue for later."""
        queued, self.jobs = self.jobs, []
        for _key, work, done in queued:
            try:
                answer = work(lambda *arguments: None)
            except Exception as error:
                answer = error
            done(answer)


@pytest.fixture(scope="module")
def qt():
    """An offscreen Qt application in this process, skipping if Qt is unavailable."""
    if not _toolkit():
        pytest.skip("pyqtgraph and PySide6 are not installed")
    import os

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    import pyqtgraph as pg

    return pg.mkQApp()


@pytest.fixture
def flat(qt):
    from rederive.plot.qt.window2d import Window2D

    window = Window2D(1, InlineSession())
    yield window
    window.close()


def _plot(text, kind, variables, label="#1"):
    return Plot(
        worksheet=1,
        label=label,
        text=text,
        kind=kind,
        node=parsed(text),
        context=Context(),
        options=plots.Options(variables=variables),
        state=ParseState(),
    )


def test_a_demonstrations_picture_stands_over_the_program_until_it_is_let_go(flat, qt):
    """A demonstration's window stays on top of the program, which fills the screen."""
    from pyqtgraph.Qt import QtCore

    on_top = QtCore.Qt.WindowType.WindowStaysOnTopHint
    flat.present(demonstrating=True)
    assert flat.windowFlags() & on_top
    # A second step reuses the same window.
    flat.present(demonstrating=True)
    assert flat.windowFlags() & on_top
    flat.release()
    assert not flat.windowFlags() & on_top and flat.isVisible()
    # A window nothing is waiting on is left as it is.
    flat.release(finished=True)
    assert flat.isVisible()


def test_a_gallery_that_ran_out_closes_the_window_it_was_shown_in(flat, qt):
    """The window belonged to the demonstration, not to a user's plot."""
    flat.present(demonstrating=True)
    flat.release(finished=True)
    assert not flat.isVisible()


def _typed(key):
    from pyqtgraph.Qt import QtCore, QtGui

    return QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress, key, QtCore.Qt.KeyboardModifier.NoModifier
    )


def test_a_key_this_window_has_no_use_for_goes_back_to_the_program(flat, qt):
    """Keys the window doesn't use are forwarded to the program.

    The desktop focuses the new window regardless, so the user often presses the
    message line's key there. Unused keys like Space and Enter are sent back rather
    than dropped.
    """
    from pyqtgraph.Qt import QtCore

    flat.present(demonstrating=True)
    flat.keyPressEvent(_typed(QtCore.Qt.Key.Key_Space))
    flat.keyPressEvent(_typed(QtCore.Qt.Key.Key_Return))
    assert flat.session.steps == [None, None]
    # Without a waiting demonstration there's nothing to forward to.
    flat.release()
    flat.keyPressEvent(_typed(QtCore.Qt.Key.Key_Space))
    assert flat.session.steps == [None, None]


def test_the_windows_own_keys_stay_the_windows_during_a_demonstration(flat, qt):
    """A gallery can be traced and panned while it runs."""
    from pyqtgraph.Qt import QtCore

    flat.add(_plot("SIN(x)", PlotKind.CURVE, ("x",)))
    flat.present(demonstrating=True)
    flat.keyPressEvent(_typed(QtCore.Qt.Key.Key_T))
    assert flat._tracing is not None
    flat.keyPressEvent(_typed(QtCore.Qt.Key.Key_Left))
    assert flat.session.steps == []


def test_escape_in_a_plot_window_closes_the_picture(flat, qt):
    """Esc closes the window but not the demonstration, which the program's Esc stops.

    While tracing, the first Esc removes the marker and the second closes the window.
    """
    from pyqtgraph.Qt import QtCore

    flat.add(_plot("SIN(x)", PlotKind.CURVE, ("x",)))
    flat.present(demonstrating=True)
    flat.keyPressEvent(_typed(QtCore.Qt.Key.Key_T))
    flat.keyPressEvent(_typed(QtCore.Qt.Key.Key_Escape))
    assert flat._tracing is None and flat.isVisible()
    flat.keyPressEvent(_typed(QtCore.Qt.Key.Key_Escape))
    assert not flat.isVisible()
    assert flat.session.steps == []


def test_a_plot_of_ones_own_stays_among_the_other_windows(flat, qt):
    flat.present()
    from pyqtgraph.Qt import QtCore

    assert not flat.windowFlags() & QtCore.Qt.WindowType.WindowStaysOnTopHint


def test_a_parametric_plot_draws_at_once_over_one_turn(flat):
    plot = _plot("[SIN(t), COS(t)]", PlotKind.PARAMETRIC, ("t",))
    plot = flat.add(plot)
    # It draws without any prompt.
    assert plot.trange == pytest.approx((-np.pi, np.pi))
    assert np.isfinite(plot.ys).any()
    # The toolbar shows the range, labeled with the parameter.
    assert all(action.isVisible() for action in flat._range_actions)
    assert flat.range_name.text().strip() == "t:"
    assert (flat.range_low.text(), flat.range_high.text()) == ("-3.14159", "3.14159")


def test_the_range_fields_read_expressions_and_resample_the_plot(flat):
    plot = _plot("[SIN(t), COS(t)]", PlotKind.PARAMETRIC, ("t",))
    plot = flat.add(plot)
    flat.range_low.setText("0")
    flat.range_high.setText("2π")
    flat._range_edited()
    assert plot.trange == pytest.approx((0.0, 2 * np.pi))
    assert plot.bounds is None
    assert float(plot.ts[0]) == pytest.approx(0.0)
    # Text that doesn't parse reverts to the displayed range.
    flat.range_high.setText("2*")
    flat._range_edited()
    assert flat.range_high.text() == "6.28319"
    assert plot.trange == pytest.approx((0.0, 2 * np.pi))
    # So does a bound that parses but isn't a number.
    flat.range_low.setText("1/0")
    flat._range_edited()
    assert plot.trange == pytest.approx((0.0, 2 * np.pi))


def test_a_window_of_functions_offers_no_range_fields(flat):
    plot = _plot("SIN(x)", PlotKind.CURVE, ("x",))
    plot = flat.add(plot)
    assert not any(action.isVisible() for action in flat._range_actions)


def test_the_polar_toggle_rereads_curves_and_restores_them(flat):
    plot = _plot("SIN(x)", PlotKind.CURVE, ("x",))
    plot = flat.add(plot)
    flat.polar_toggle.trigger()
    # The curve is now r = f(θ) over one full turn (the x-range is not a θ range),
    # with the range fields labeled θ and the readout in polar coordinates.
    assert plot.kind is PlotKind.POLAR
    assert plot.trange == pytest.approx((-np.pi, np.pi))
    assert np.isfinite(plot.ys).any()
    assert float(np.nanmax(np.abs(plot.xs))) < 1.5
    assert flat.range_name.text().strip() == "θ:"
    assert "r:" in flat._readout(1.0, 0.0)
    # Toggling back restores the cartesian curve, sampled over the view.
    flat.polar_toggle.trigger()
    assert plot.kind is PlotKind.CURVE
    assert float(plot.xs.min()) < -4 and float(plot.xs.max()) > 4
    assert not any(action.isVisible() for action in flat._range_actions)
    assert "r:" not in flat._readout(1.0, 0.0)


def test_the_kinds_with_no_polar_reading_ignore_the_toggle(flat):
    pair = _plot("[SIN(t), COS(t)]", PlotKind.PARAMETRIC, ("t",))
    points = _plot("[[1, 2], [3, 4]]", PlotKind.DATA, (), label="#2")
    pair = flat.add(pair)
    points = flat.add(points)
    before = pair.trange
    flat.polar_toggle.trigger()
    assert pair.kind is PlotKind.PARAMETRIC
    assert points.kind is PlotKind.DATA
    assert pair.trange == before


def test_a_curve_added_to_a_polar_window_is_read_polar_from_the_start(flat):
    flat.polar_toggle.trigger()
    plot = _plot("2*COS(3*t)", PlotKind.CURVE, ("t",))
    plot = flat.add(plot)
    assert plot.kind is PlotKind.POLAR
    assert plot.trange == pytest.approx((-np.pi, np.pi))
    assert flat.range_name.text().strip() == "θ:"


def test_the_range_fields_adjust_a_reread_curve_like_a_born_polar_one(flat):
    plot = _plot("SIN(x)", PlotKind.CURVE, ("x",))
    plot = flat.add(plot)
    flat.polar_toggle.trigger()
    flat.range_low.setText("0")
    flat.range_high.setText("2π")
    flat._range_edited()
    assert plot.trange == pytest.approx((0.0, 2 * np.pi))
    # The adjusted range belonged to that polar reading; toggling again returns to one
    # full turn.
    flat.polar_toggle.trigger()
    flat.polar_toggle.trigger()
    assert plot.trange == pytest.approx((-np.pi, np.pi))


def test_a_polar_window_is_ruled_by_rings_instead_of_lines(flat):
    # Polar shows rings instead of grid lines, and the Grid toggle controls whichever
    # applies, as in the page's pane.
    assert flat.item.ctrl.xGridCheck.isChecked()
    assert not flat.rings.isVisible()
    flat.polar_toggle.trigger()
    assert not flat.item.ctrl.xGridCheck.isChecked()
    assert flat.rings.isVisible()
    flat.toggle_grid()
    assert not flat.rings.isVisible()
    # A gridless window stays gridless after leaving polar: the toggle belongs to the
    # window, not the mode.
    flat.polar_toggle.trigger()
    assert not flat.item.ctrl.xGridCheck.isChecked()
    flat.toggle_grid()
    assert flat.item.ctrl.xGridCheck.isChecked()
    assert not flat.rings.isVisible()


def test_the_rings_stand_at_numbers_a_reader_can_add_up(qt):
    from rederive.plot.qt.window2d import _ruled

    # About five rings to the view's edge, spaced at 1, 2 or 5 times a power of ten. A
    # view of zero extent gets no rings.
    assert [_ruled(reach) for reach in (5.0, 10.0, 47.0, 0.5)] == [1.0, 2.0, 10.0, 0.1]
    assert _ruled(0.0) == 0.0


def test_the_clear_button_empties_its_own_window(flat):
    # The toolbar's Clear acts on its own window, and the range fields disappear with
    # the parametric plot that owned them.
    flat.add(_plot("SIN(x)", PlotKind.CURVE, ("x",)))
    flat.add(_plot("[SIN(t), COS(t)]", PlotKind.PARAMETRIC, ("t",), label="#2"))
    assert all(action.isVisible() for action in flat._range_actions)
    flat.clear_action.trigger()
    assert flat.plots == []
    assert not any(action.isVisible() for action in flat._range_actions)
    assert flat.item.getAxis("bottom").labelText == ""


def test_the_delete_key_clears_the_window_it_is_pressed_in(flat):
    from pyqtgraph.Qt import QtCore, QtGui

    flat.add(_plot("SIN(x)", PlotKind.CURVE, ("x",)))
    flat.keyPressEvent(
        QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_Delete,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
    )
    assert flat.plots == []


def _pressed_return(window):
    from pyqtgraph.Qt import QtCore, QtGui

    window.keyPressEvent(
        QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_Return,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
    )


def test_enter_while_tracing_sends_the_refined_point_home(flat):
    flat.add(_plot("SIN(x)", PlotKind.CURVE, ("x",)))
    # Without a trace marker, Enter sends nothing.
    _pressed_return(flat)
    assert flat.session.authored == []
    flat.trace()
    # Tab snaps to the next feature right of center (the maximum at π/2), refined on
    # the closure, and Enter sends it to the worksheet.
    flat.snap(False)
    _pressed_return(flat)
    assert flat.session.authored == [(1, "[1.570796, 1.000000]")]
    # The status line confirms it.
    assert flat.status.text() == "Sent [1.570796, 1.000000] to the worksheet"


def test_the_feature_scan_is_asked_for_rather_than_done_on_the_qt_thread(qt):
    from rederive.plot.qt.window2d import Window2D

    # The scan evaluates the closure, so like all evaluation it goes to the sampler,
    # and Tab on a dense curve can't freeze the window.
    session = QueuedSession()
    window = Window2D(1, session)
    try:
        window.add(_plot("SIN(x)", PlotKind.CURVE, ("x",)))
        session.run()  # the sampling the curve arrived asking for
        window.trace()
        window.snap(False)
        # Nothing is scanned yet: the job is queued under its own key, and the marker
        # hasn't moved.
        keys = [key for key, _work, _done in session.jobs]
        assert [key[:2] for key in keys] == [(1, "features")]
        assert window.plots[0].features is None
        assert "x = 0.000000" in window.status.text()
        # Once the job runs, the marker moves to the maximum at π/2, refined on the
        # closure rather than read off a sample.
        session.run()
        assert window.plots[0].features
        assert window.status.text().endswith("maximum at x = 1.570796")
    finally:
        window.close()


def test_a_feature_scan_answering_after_a_re_sample_is_dropped(qt):
    from rederive.plot.qt.window2d import Window2D

    # The answer refers to the samples it was computed over, so a view that has since
    # changed discards it.
    session = QueuedSession()
    window = Window2D(1, session)
    try:
        window.add(_plot("SIN(x)", PlotKind.CURVE, ("x",)))
        session.run()
        window.trace()
        window.snap(False)
        # Re-sample while the scan is queued, as a drag would.
        window.resample()
        session.run()
        assert window.plots[0].features is None
        assert "maximum" not in window.status.text()
    finally:
        window.close()


def test_the_point_sent_home_is_the_text_ctrl_c_copies(flat, qt):
    flat.add(_plot("SIN(x)", PlotKind.CURVE, ("x",)))
    flat.trace()
    flat.snap(False)
    flat.copy_image()
    copied = qt.clipboard().text()
    flat.send_home()
    assert copied.startswith("[")
    assert flat.session.authored == [(1, copied)]


def test_the_title_tracks_the_plot_list(flat):
    # The title follows every change to the plot list: add, replace, remove, clear.
    assert flat.windowTitle() == "Rederive plot"
    flat.add(_plot("SIN(x)", PlotKind.CURVE, ("x",)))
    assert flat.windowTitle() == "SIN(x) - Rederive plot"
    flat.add(_plot("COS(x)", PlotKind.CURVE, ("x",), label="#2"))
    assert flat.windowTitle() == "SIN(x), COS(x) - Rederive plot"
    # Re-plotting a label replaces its curve, and the title shows the new text.
    flat.add(_plot("x^2 - 3", PlotKind.CURVE, ("x",)))
    assert flat.windowTitle() == "COS(x), x^2 - 3 - Rederive plot"
    flat.remove(flat.plots[0])
    assert flat.windowTitle() == "x^2 - 3 - Rederive plot"
    flat.clear_action.trigger()
    assert flat.windowTitle() == "Rederive plot"


def test_a_data_plots_toggles_hand_the_sticky_values_back(flat):
    # Per section 7, the right-click controls write back, so the next data plot
    # arrives with these settings.
    plot = _plot("[[1, 2], [3, 4]]", PlotKind.DATA, ())
    plot = flat.add(plot)
    flat._pointed = plot
    flat._toggle_connected()
    assert flat.session.adjustments == {"connected": True}
    flat._set_point_size(8.0)
    assert flat.session.adjustments == {"connected": True, "point_size": 8.0}
    flat._toggle_connected()
    assert flat.session.adjustments["connected"] is False


def test_the_framing_lock_hands_nothing_back(flat):
    # Equal scales is deliberately not sticky: a one-off framing choice must not become
    # the default for the next plot.
    flat.equal.trigger()
    assert not flat.equal.isChecked()
    assert flat.session.adjustments == {}


def test_the_axis_label_follows_the_polar_mode(flat):
    # In polar mode the horizontal axis is not θ, so it must not be labeled with the
    # parameter.
    plot = _plot("SIN(t)", PlotKind.CURVE, ("t",))
    plot = flat.add(plot)
    axis = flat.item.getAxis("bottom")
    assert axis.labelText == "t"
    flat.polar_toggle.trigger()
    assert axis.labelText == ""
    flat.polar_toggle.trigger()
    assert axis.labelText == "t"


# -- the stroke weight: curves out-weigh the furniture ---------------------------


def test_a_curve_out_weighs_the_axes_and_the_grid(flat):
    from rederive.plot.qt.window2d import CURVE_WIDTH

    plot = _plot("2*x + 3", PlotKind.CURVE, ("x",))
    plot = flat.add(plot)
    pen = plot.item.opts["pen"]
    # Two logical pixels, scaled by screen density, and cosmetic so zooming doesn't
    # change the stroke width.
    assert pen.widthF() == CURVE_WIDTH * flat.devicePixelRatioF()
    assert pen.isCosmetic()
    # The axes stay hairlines, thinner than the curve.
    for line in flat.axes:
        assert line.pen.widthF() == 1.0
    # The legend swatch matches the curve's weight and color.
    assert flat.legend.entries == ((plot.named, plot.color, False),)


def test_a_data_plots_line_takes_the_weight_and_its_points_do_not(flat):
    from rederive.plot.qt.window2d import CURVE_WIDTH

    plot = _plot("[[1, 2], [3, 4]]", PlotKind.DATA, ())
    plot = flat.add(plot)
    flat._pointed = plot
    flat._toggle_connected()
    pen = plot.item.opts["pen"]
    assert pen.widthF() == CURVE_WIDTH * flat.devicePixelRatioF()
    assert pen.isCosmetic()
    # Point size is a separate setting: the symbol outline stays at 1, so a thicker
    # line doesn't thicken the points.
    assert plot.item.opts["symbolPen"].widthF() == 1.0


def test_the_export_pens_carry_the_same_weight(flat):
    from rederive.plot.qt.window2d import CURVE_WIDTH, _on_paper

    plot = _plot("SIN(x)", PlotKind.CURVE, ("x",))
    plot = flat.add(plot)
    with _on_paper(flat):
        # Exporters render without high-DPI scaling, so export pens use the plain
        # constant, matching a 1x screen.
        pen = plot.item.opts["pen"]
        assert pen.widthF() == CURVE_WIDTH
        assert pen.isCosmetic()
    assert plot.item.opts["pen"].widthF() == CURVE_WIDTH * flat.devicePixelRatioF()


# -- one spelling of a control --------------------------------------------------
#
# Each window's controls (labels, order, keys) are one table in `plot.controls`, and a
# menu's current state is that table evaluated against a window snapshot. Neither needs
# a display, so menus can be tested without a toolkit, and backends take all their
# wording from the table.


def test_a_menu_is_described_off_a_snapshot_and_needs_no_window():
    # The full 2D menu for a click over nothing, so it has no item-specific tail.
    assert offered(controls.Flat()) == [
        ("Set range...", ""),
        ("View all", "A"),
        ("Home framing", "Home"),
        ("Back", "Backspace"),
        ("Trace", "T"),
        ("Grid", "G"),
        ("Legend", "L"),
        ("Copy image", "Ctrl+C"),
        ("Export...", "Ctrl+S"),
        ("Clear", "Del"),
        ("Close", "Q"),
    ]
    # The 3D menu is about the camera; state shown on the toolbar is not repeated.
    assert offered(controls.Solid()) == [
        ("Home view", "Home"),
        ("Face the xy plane", "1"),
        ("Face the xz plane", "2"),
        ("Face the yz plane", "3"),
        ("Rotate", "R"),
        ("View...", ""),
        ("Copy image", "Ctrl+C"),
        ("Export...", "Ctrl+S"),
        ("Remove", ""),
        ("Clear", "Del"),
        ("Close", "Q"),
    ]


def test_the_tail_of_a_menu_is_about_whatever_the_click_was_over():
    curve = controls.Pointed(named="#1: SIN(x)", kind=PlotKind.CURVE)
    points = controls.Pointed(named="#2", kind=PlotKind.DATA)
    # A curve only offers Remove; the two point options appear only for a data plot.
    tail = [label for label, _ in offered(controls.Flat(pointed=curve))[11:]]
    assert tail == ["Remove #1: SIN(x)"]
    tail = [label for label, _ in offered(controls.Flat(pointed=points))[11:]]
    assert tail == ["Remove #2", "Connect points", "Point size"]
    # The connect entry offers the opposite of the current state.
    connected = dataclasses.replace(points, connected=True)
    assert "Disconnect points" in [
        label for label, _ in offered(controls.Flat(pointed=connected))
    ]
    # The offered sizes are a fixed list.
    sizes = next(
        entry
        for entry in controls.menu(controls.Flat(pointed=points))
        if entry.name == "points.size"
    )
    assert [item.label for item in sizes.items] == ["3 px", "5 px", "8 px", "12 px"]


def test_the_ticks_of_a_menu_are_the_windows_and_not_the_menus():
    # Tracing and the grid can be changed outside the menu, so every check mark is
    # read off the snapshot.
    assert ticked(controls.Flat()) == ["Grid", "Legend"]
    assert ticked(controls.Flat(tracing=True, grid=False)) == ["Trace", "Legend"]
    assert ticked(controls.Solid()) == []
    assert ticked(controls.Solid(spinning=True)) == ["Rotate"]


def test_the_remove_submenu_lists_what_a_3d_window_holds():
    entry = next(
        one
        for one in controls.menu(controls.Solid(surfaces=("#1: x*y", "#2: x+y")))
        if one.name == "surface.remove"
    )
    assert [item.label for item in entry.items] == ["#1: x*y", "#2: x+y"]
    # Each item's value is the surface's index in the window's list, which the backend
    # removes it by.
    assert [item.value for item in entry.items] == [0, 1]
    # In an empty window the submenu is disabled rather than omitted.
    empty = next(
        one for one in controls.menu(controls.Solid()) if one.name == "surface.remove"
    )
    assert empty.items == () and not empty.enabled


def test_a_legend_rows_menu_is_the_canvas_menus_own_words():
    # A separate menu, since half the canvas menu is about the view and a legend row
    # is about one plot, but with the canvas menu's wording so both read as the same
    # target.
    points = controls.Pointed(named="#2", kind=PlotKind.DATA, connected=True)
    card = controls.card(controls.Flat(pointed=points))
    assert [entry.label for entry in card] == ["Remove #2", "Disconnect points"]
    # A 3D row offers a per-surface override of the toolbar's wire toggle, labeled
    # with the opposite of the current state.
    solid = controls.Pointed(named="#1", kind=PlotKind.SURFACE, wire=True)
    card = controls.card(controls.Solid(pointed=solid))
    assert [entry.label for entry in card] == ["Draw solid", "Remove #1"]
    assert [entry.label for entry in controls.card(controls.Solid())] == []


def test_a_page_has_its_own_spelling_of_a_key_and_says_where_it_has_none():
    # Each binding has a desktop and a web spelling. Keys the browser reserves, like
    # Ctrl+W (close tab), are omitted from the web one.
    close = controls.control("close", controls.FLAT)
    assert (close.keys, close.web) == (("Q", "Ctrl+W"), ("q",))
    trace = controls.control("trace", controls.FLAT)
    assert (trace.keys, trace.web) == (("T", "F3"), ("t", "F3"))
    assert [entry.keys for entry in controls.menu(controls.Flat(), page=True)][:4] == [
        (),
        ("a",),
        ("Home", "0", "h"),
        ("Backspace",),
    ]


# -- the windows' own menus -----------------------------------------------------
#
# Both windows show a right-click menu rendered from that table instead of pyqtgraph's,
# with each entry's key shown. These test that the menu is entirely ours, matches the
# description, that a key fires exactly once, and that keys typed into a toolbar field
# are text.


def _entries(menu):
    """A menu's entries as (text, shortcut) pairs, without separators."""
    return [
        (action.text(), action.shortcut().toString())
        for action in menu.actions()
        if not action.isSeparator()
    ]


def _entry(menu, text):
    """The menu entry with this text."""
    return next(action for action in menu.actions() if action.text() == text)


def _pressed(window, key, target=None):
    """Deliver a key press as the window would receive it."""
    from pyqtgraph.Qt import QtCore, QtTest

    QtTest.QTest.keyClick(target if target is not None else window, key)
    QtCore.QCoreApplication.processEvents()


def test_the_canvas_menu_is_the_windows_own_and_not_pyqtgraphs(flat):
    # Every entry is ours, in the agreed order, with its key. The item-specific tail
    # comes last.
    assert _entries(flat.menu) == [
        ("Set range...", ""),
        ("View all", "A"),
        ("Home framing", "Home"),
        ("Back", "Backspace"),
        ("Trace", "T"),
        ("Grid", "G"),
        ("Legend", "L"),
        ("Copy image", "Ctrl+C"),
        ("Export...", "Ctrl+S"),
        ("Clear", "Del"),
        ("Close", "Q"),
        ("Remove", ""),
        ("Connect points", ""),
        ("Point size", ""),
    ]
    # Nothing from pyqtgraph leaks in: the view box has no menu, and the plot item's
    # is disabled, which stops the scene from appending `Plot Options` and its own
    # `Export...`.
    assert flat.canvas.menu is None
    assert not flat.item.menuEnabled()
    words = {text for text, _ in _entries(flat.menu)}
    assert not words & {"Mouse Mode", "Plot Options", "X axis", "Y axis", "View All"}


def _shown(menu):
    """An opened menu's visible entries, as (text, checked) pairs."""
    return [
        (action.text(), action.isChecked())
        for action in menu.actions()
        if not action.isSeparator() and action.isVisible()
    ]


def test_the_menu_a_window_opens_is_the_description_rendered(flat):
    # The window provides a snapshot, `plot.controls` says what the menu should show,
    # and the menu shows exactly that.
    plot = flat.add(_plot("[[1, 2], [3, 4]]", PlotKind.DATA, ()))
    flat.toggle_grid()
    flat._pointed = plot
    flat.menu.aboutToShow.emit()
    described = [
        (entry.label, entry.checked) for entry in controls.menu(flat.snapshot())
    ]
    assert _shown(flat.menu) == described
    assert (f"Remove {plot.named}", False) in described
    # A click over nothing offers the same menu without the tail.
    flat._pointed = None
    flat.menu.aboutToShow.emit()
    assert _shown(flat.menu) == [
        (entry.label, entry.checked) for entry in controls.menu(flat.snapshot())
    ]


class _Click:
    """A fake right-click event, as the view box passes it to the menu."""

    def __init__(self, point):
        self._point = point

    def pos(self):
        return self._point

    def screenPos(self):
        return self._point


def test_a_right_click_raises_our_menu_whole(flat):
    from pyqtgraph.Qt import QtCore

    flat.add(_plot("SIN(x)", PlotKind.CURVE, ("x",)))
    built = len(flat.menu.actions())
    flat.canvas.raiseContextMenu(_Click(QtCore.QPointF(0.0, 0.0)))
    try:
        assert flat.menu.isVisible()
        # Stock `raiseContextMenu` lets the scene append `Plot Options` and its own
        # `Export...`. Here the menu stays exactly as built.
        assert len(flat.menu.actions()) == built
        assert "Plot Options" not in {action.text() for action in flat.menu.actions()}
    finally:
        flat.menu.close()


def test_the_menus_keys_and_the_windows_keys_are_one_table(flat):
    from pyqtgraph.Qt import QtCore

    # Keys dispatch through the menu's own table, so entries and their keys can't
    # drift apart. Secondary keys count too: Home framing also answers to 0, and Trace
    # to F3.
    assert set(flat._keyed.values()) <= set(flat.menu.actions())
    keys = QtCore.Qt.Key
    for key, text in (
        (keys.Key_A, "View all"),
        (keys.Key_0, "Home framing"),
        (keys.Key_F3, "Trace"),
    ):
        assert flat._keyed[(0, int(key))].text() == text


def test_view_all_is_this_windows_autoscale_and_is_remembered(flat):
    # `View all` is our framing, not pyqtgraph's auto-range: it pushes the old range
    # onto the history, releases equal scales, and frames the drawn samples.
    flat.add(_plot("SIN(x)/3", PlotKind.CURVE, ("x",)))
    flat.canvas.setRange(xRange=(-1.0, 1.0), yRange=(-1.0, 1.0), padding=0)
    framed = tuple(tuple(span) for span in flat.canvas.viewRange())
    _entry(flat.menu, "View all").trigger()
    assert flat._framing.history.last == framed
    assert not flat.equal.isChecked()
    (left, right), (low, high) = flat.canvas.viewRange()
    # The framing fits the curve (a third of a unit tall over the sampled x range),
    # not the previous square.
    assert right - left > 9.0
    assert 0.3 < high < 0.4 and -0.4 < low < -0.3


def test_the_switched_entries_are_read_off_the_window_as_the_menu_opens(flat):
    flat.add(_plot("SIN(x)", PlotKind.CURVE, ("x",)))
    trace, grid, legend = (
        _entry(flat.menu, name) for name in ("Trace", "Grid", "Legend")
    )
    flat.menu.aboutToShow.emit()
    assert (trace.isChecked(), grid.isChecked(), legend.isChecked()) == (
        False,
        True,
        True,
    )
    # These can change outside the menu, so the check marks are read from the window
    # each time the menu opens.
    flat.trace()
    flat.toggle_grid()
    flat.menu.aboutToShow.emit()
    assert (trace.isChecked(), grid.isChecked(), legend.isChecked()) == (
        True,
        False,
        True,
    )


def test_the_set_range_dialog_frames_the_view_on_what_is_typed(flat):
    flat.add(_plot("SIN(x)", PlotKind.CURVE, ("x",)))
    flat.set_range()
    dialog = flat._bounds
    # It opens with the current view, so changing one edge leaves the other three.
    assert dialog.typed[:2] == ("-5", "5")
    framed = tuple(tuple(span) for span in flat.canvas.viewRange())
    for name, text in (("left", "-1"), ("right", "2π"), ("bottom", "-3"), ("top", "4")):
        dialog.fields[name].setText(text)
    dialog.accept()
    (left, right), (bottom, top) = flat.canvas.viewRange()
    # The bounds are expressions, like the toolbar's range fields, and the view is
    # framed to exactly their values, with no padding and no equal-scales widening.
    assert (left, bottom, top) == pytest.approx((-1.0, -3.0, 4.0))
    assert right == pytest.approx(2 * np.pi)
    assert not flat.equal.isChecked()
    # The previous framing is one Backspace away, and the status shows the new one.
    assert flat._framing.history.last == framed
    assert flat.status.text().startswith("Showing -1 ≤ x ≤ 6.28319")


def test_bounds_that_are_not_a_range_are_refused_in_words(flat):
    flat.set_range()
    dialog = flat._bounds
    dialog.fields["left"].setText("3")
    dialog.fields["right"].setText("1")
    dialog.accept()
    assert flat.status.text() == "A range runs from a lower bound to a higher one"
    assert flat.canvas.viewRange()[0] == pytest.approx([-5.0, 5.0])
    # Unparseable text gets a generic message, not one naming the field, and so does
    # text that parses but isn't a number.
    flat.set_range()
    flat._bounds.fields["left"].setText("2*")
    flat._bounds.accept()
    assert flat.status.text() == "The bounds are expressions, like -π or 2π"
    flat.set_range()
    flat._bounds.fields["left"].setText("1/0")
    flat._bounds.accept()
    assert flat.status.text() == "The bounds have to be worth numbers, like -π or 2π"
    assert flat.canvas.viewRange()[0] == pytest.approx([-5.0, 5.0])


def _answers_the_save_dialog(monkeypatch, name, chosen="PNG image (*.png)"):
    """Make the save dialog return this name and filter."""
    from pyqtgraph.Qt import QtWidgets

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(name), chosen),
    )


def test_export_writes_the_png_ctrl_c_copies_and_leaves_the_window_dark(
    flat, tmp_path, monkeypatch
):
    from rederive.plot.qt.window2d import BACKGROUND

    plot = _plot("SIN(x)", PlotKind.CURVE, ("x",))
    plot = flat.add(plot)
    target = tmp_path / "curve.png"
    _answers_the_save_dialog(monkeypatch, target)
    flat.export()
    # Check the PNG signature.
    assert target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert flat.status.text() == f"Saved {target}"
    # The light export colors are used only for the file; the window stays dark.
    assert flat.plot.backgroundBrush().color().name() == BACKGROUND
    assert plot.item.opts["pen"].color().name() == plot.color


def test_export_writes_an_svg_when_the_name_asks_for_one(flat, tmp_path, monkeypatch):
    plot = _plot("SIN(x)", PlotKind.CURVE, ("x",))
    plot = flat.add(plot)
    target = tmp_path / "curve.SVG"
    # The typed extension decides the format, regardless of the filter or case.
    _answers_the_save_dialog(monkeypatch, target)
    flat.export()
    written = target.read_text()
    assert "<svg" in written
    # The legend is an overlay, not part of the scene, so the names are written into
    # the file separately, here as SVG text.
    assert plot.named in written
    assert flat.status.text() == f"Saved {target}"


def test_a_name_with_no_extension_takes_the_chosen_filters(flat, tmp_path, monkeypatch):
    flat.add(_plot("SIN(x)", PlotKind.CURVE, ("x",)))
    _answers_the_save_dialog(monkeypatch, tmp_path / "curve", "SVG image (*.svg)")
    flat.export()
    written = tmp_path / "curve.svg"
    assert "<svg" in written.read_text()
    assert flat.status.text() == f"Saved {written}"


def test_a_cancelled_export_writes_nothing_and_says_nothing(flat, tmp_path, monkeypatch):
    flat.add(_plot("SIN(x)", PlotKind.CURVE, ("x",)))
    flat.say("Ready")
    _answers_the_save_dialog(monkeypatch, "")
    flat.export()
    assert list(tmp_path.iterdir()) == []
    assert flat.status.text() == "Ready"


def test_q_closes_the_window_as_the_menu_advertises(flat):
    from pyqtgraph.Qt import QtCore, QtGui

    # `Q` is Derive's key for leaving a plot window, and `Close` shows it (along with
    # Ctrl+W) so the menu teaches it.
    flat.show()
    flat.keyPressEvent(
        QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_Q,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
    )
    assert flat.isHidden()


def test_a_key_the_menu_names_does_its_thing_exactly_once(flat):
    from pyqtgraph.Qt import QtCore

    # The menu only displays the key; the key handler fires it. A Qt shortcut would
    # fire as well and reframe the window twice.
    flat.show()
    fired = []
    _entry(flat.menu, "View all").triggered.connect(lambda: fired.append("A"))
    _pressed(flat, QtCore.Qt.Key.Key_A)
    assert fired == ["A"]


def test_a_key_typed_into_a_range_field_is_text_and_not_a_command(flat):
    from pyqtgraph.Qt import QtCore

    flat.add(_plot("[SIN(t), COS(t)]", PlotKind.PARAMETRIC, ("t",)))
    flat.show()
    fired = []
    for name in ("View all", "Clear"):
        _entry(flat.menu, name).triggered.connect(
            lambda _=False, name=name: fired.append(name)
        )
    flat.range_low.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
    _pressed(flat, QtCore.Qt.Key.Key_A, target=flat.range_low)
    _pressed(flat, QtCore.Qt.Key.Key_Delete, target=flat.range_low)
    assert fired == []
    # The letter went into the field, and Del deleted only text, not the plots.
    assert flat.range_low.text().endswith("a")
    assert len(flat.plots) == 1


@pytest.fixture
def deep(space):
    window = space.Window3D(1, InlineSession())
    yield window
    window.close()


def test_a_solid_demonstration_stands_over_the_program_too(deep, qt):
    """A 3D demonstration window behaves like a 2D one."""
    from pyqtgraph.Qt import QtCore

    on_top = QtCore.Qt.WindowType.WindowStaysOnTopHint
    deep.present(demonstrating=True)
    assert deep.windowFlags() & on_top
    # Space is forwarded to the program; the arrows still rotate the view.
    deep.keyPressEvent(_typed(QtCore.Qt.Key.Key_Space))
    deep.keyPressEvent(_typed(QtCore.Qt.Key.Key_Left))
    assert deep.session.steps == [None]
    deep.release()
    assert not deep.windowFlags() & on_top


def test_the_domain_fields_read_expressions(deep):
    surface = Surface(
        worksheet=1,
        label="#1",
        text="x*y",
        kind=PlotKind.SURFACE,
        node=parsed("x*y"),
        context=Context(),
        options=plots.Options(variables=("x", "y")),
        state=ParseState(),
    )
    surface = deep.add(surface)
    deep.fields["x0"].setText("-π")
    deep.fields["x1"].setText("π")
    deep._edited()
    assert deep.xdomain == pytest.approx((-np.pi, np.pi))
    assert deep.fields["x0"].text() == "-3.14159"
    # A field that doesn't parse reverts, leaving the domain unchanged.
    deep.fields["y0"].setText("*")
    deep._edited()
    assert deep.ydomain == (-5.0, 5.0)
    assert deep.fields["y0"].text() == "-5"
    # An inverted domain is refused and reverted too.
    deep.fields["y0"].setText("10")
    deep._edited()
    assert deep.ydomain == (-5.0, 5.0)
    assert deep.fields["y0"].text() == "-5"


def test_an_edited_grid_hands_the_new_count_back_to_the_host(deep):
    deep.fields["nx"].setText("32")
    deep.fields["ny"].setText("32")
    deep._edited()
    assert deep.grid == (32, 32)
    assert deep.session.adjustments == {"grid": 32}
    # The sticky value is a single count, so a rectangular grid sends its finer axis.
    # The domain fields are framing and send nothing.
    deep.fields["ny"].setText("48")
    deep._edited()
    assert deep.session.adjustments == {"grid": 48}
    deep.fields["x0"].setText("-2")
    deep._edited()
    assert deep.session.adjustments == {"grid": 48}


def test_a_surfaces_boundary_arrives_beside_its_arrays(deep):
    # The boundary is refined on the sampling thread along with `grid_eval` and cached
    # on the surface for the mesh and the wire.
    surface = Surface(
        worksheet=1,
        label="#1",
        text="SQRT(1-x^2-y^2)",
        kind=PlotKind.SURFACE,
        node=parsed("SQRT(1-x^2-y^2)"),
        context=Context(),
        options=plots.Options(variables=("x", "y")),
        state=ParseState(),
    )
    surface = deep.add(surface)
    assert isinstance(surface.boundary, evaluate.Boundary)
    assert np.isfinite(surface.boundary.across).any()
    assert np.isfinite(surface.boundary.along).any()


def test_an_all_nan_surface_still_says_no_real_values(deep):
    surface = Surface(
        worksheet=1,
        label="#5",
        text="SQRT(-1-x^2-y^2)",
        kind=PlotKind.SURFACE,
        node=parsed("SQRT(-1-x^2-y^2)"),
        context=Context(),
        options=plots.Options(variables=("x", "y")),
        state=ParseState(),
    )
    surface = deep.add(surface)
    assert deep.status.text() == "#5: no real values over this domain"


def test_the_3d_clear_button_empties_its_own_window(deep):
    surface = Surface(
        worksheet=1,
        label="#1",
        text="x*y",
        kind=PlotKind.SURFACE,
        node=parsed("x*y"),
        context=Context(),
        options=plots.Options(variables=("x", "y")),
        state=ParseState(),
    )
    surface = deep.add(surface)
    assert deep.windowTitle() == "x*y - Rederive 3D plot"
    deep.clear_action.trigger()
    assert deep.plots == []
    # A cleared window's title falls back to the empty-window one.
    assert deep.windowTitle() == "Rederive 3D plot"


def _surface(text, label="#1"):
    return Surface(
        worksheet=1,
        label=label,
        text=text,
        kind=PlotKind.SURFACE,
        node=parsed(text),
        context=Context(),
        options=plots.Options(variables=("x", "y")),
        state=ParseState(),
    )


def _pressed_m(window):
    from pyqtgraph.Qt import QtCore, QtGui

    window.keyPressEvent(
        QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_M,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
    )


def test_the_mesh_box_and_the_m_key_flip_every_surface(deep):
    # The toolbar box (key M) toggles every surface between wire and solid, and the
    # result is sent back as the sticky value.
    one = _surface("x*y")
    two = _surface("x+y", label="#2")
    one = deep.add(one)
    two = deep.add(two)
    assert one.wire and two.wire
    deep.mesh_action.trigger()
    assert not one.wire and not two.wire
    assert deep.session.adjustments == {"wire": False}
    assert one.item.visible() and not one.wires.visible()
    _pressed_m(deep)
    assert one.wire and two.wire
    assert deep.session.adjustments == {"wire": True}
    # The solid stays visible under the wire, for hidden-line removal.
    assert one.item.visible() and one.wires.visible()


def test_the_wire_draws_at_the_curves_weight(deep):
    from rederive.plot.qt.window2d import CURVE_WIDTH

    # The wire uses the same stroke width constant as 2D curves.
    surface = _surface("x*y")
    surface = deep.add(surface)
    assert surface.wires.width == CURVE_WIDTH


def test_the_legend_override_moves_one_surface_and_nothing_sticky(deep):
    # As with data plot points, the per-surface right-click is a one-off override:
    # one surface goes solid, the other stays wire, and nothing sticky is sent back.
    one = _surface("x*y")
    two = _surface("x+y", label="#2")
    one = deep.add(one)
    two = deep.add(two)
    deep.toggle_wire(one)
    assert not one.wire and two.wire
    assert one.item.visible() and two.wires.visible()
    assert deep.session.adjustments == {}
    deep.toggle_wire(one)
    assert one.wire
    assert deep.session.adjustments == {}


def test_a_surface_arrives_in_the_look_the_window_was_left_in(space):
    # Sticky per section 7: a window created solid (wire is the default) opens with
    # the box unchecked and draws new surfaces solid, while a replacement keeps the
    # look of the surface it replaces.
    window = space.Window3D(1, InlineSession(), wire=False)
    try:
        assert not window.mesh_action.isChecked()
        one = _surface("x*y")
        one = window.add(one)
        assert not one.wire
        window.toggle_wire(one)
        replaced = _surface("x^2-y^2")
        replaced = window.add(replaced)
        assert replaced.wire
    finally:
        window.close()


def test_the_wire_darkens_for_export_like_every_other_color(deep, solid, space):
    surface = _surface("x*y")
    surface = deep.add(surface)
    points, shades = solid.wire(
        surface.xs, surface.ys, surface.values, deep.box_now, surface.boundary
    )
    deep._papered = True
    deep._draw(surface)
    assert np.allclose(surface.wires.color, solid.brightened(shades, surface.paper))
    deep._papered = False
    deep._draw(surface)
    assert np.allclose(surface.wires.color, solid.brightened(shades, surface.color))


def _gl_state(item):
    """The GL state an item draws under, from a pyqtgraph private attribute.

    There is no public accessor, and the polygon offset hidden-line removal relies on
    isn't available anywhere else.
    """
    return item._GLGraphicsItem__glOpts


def test_a_wire_hides_behind_a_solid_painted_in_the_canvas(deep, solid, space):
    import pyqtgraph as pg

    # Hidden-line removal: both items draw, with the solid in the background color and
    # pushed back by a polygon offset, so it hides lines behind it but not lines on it.
    surface = _surface("x^2+y^2")
    surface = deep.add(surface)
    assert surface.item.visible() and surface.wires.visible()
    assert surface.item.opts["color"] == pg.mkColor(space.BACKGROUND)
    # The flat color is used only when a mesh has no vertex colors, so the occluder
    # has none.
    assert not surface.item.opts["meshdata"].hasVertexColor()
    state = _gl_state(surface.item)
    assert state["glPolygonOffset"] == space.WIRE_OFFSET
    assert state[space.GL.GL_POLYGON_OFFSET_FILL] is True
    # The occluder uses the solid's own triangles, so the silhouette matches.
    faces = solid.mesh(
        surface.xs, surface.ys, surface.values, deep.box_now, surface.boundary
    )[1]
    assert len(surface.item.opts["meshdata"].faces()) == len(faces)


def test_the_occluder_goes_white_with_the_canvas_for_an_export(deep, solid, space):
    import pyqtgraph as pg

    # Exports have a white background, so a dark occluder would show as a black
    # surface.
    surface = _surface("x^2+y^2")
    surface = deep.add(surface)
    with space._on_paper(deep):
        assert surface.item.opts["color"] == pg.mkColor("w")
    assert surface.item.opts["color"] == pg.mkColor(space.BACKGROUND)


def test_a_surface_drawn_solid_again_is_the_stock_item_it_was(deep, solid, space):
    from pyqtgraph.opengl.GLGraphicsItem import GLOptions

    # The occluder setup applies only while the wire is on. Back to solid, the item
    # uses the stock opaque state and its shaded colors, however often it is toggled.
    surface = _surface("x^2+y^2")
    surface = deep.add(surface)
    for _ in range(2):
        deep.toggle_wire(surface)
        assert _gl_state(surface.item) == GLOptions["opaque"]
        assert surface.item.visible() and not surface.wires.visible()
        shading = solid.mesh(
            surface.xs, surface.ys, surface.values, deep.box_now, surface.boundary
        )[2]
        assert np.allclose(
            surface.item.opts["meshdata"].vertexColors(),
            solid.brightened(shading, surface.color),
        )
        deep.toggle_wire(surface)
        assert space.GL.GL_POLYGON_OFFSET_FILL in _gl_state(surface.item)


def test_the_wire_loses_the_pixels_behind_the_surface(qt, deep):
    # Rendered check, where OpenGL is available: a wire bowl has fewer lit pixels with
    # the occluder than without, because its far side is hidden.
    surface = _surface("x^2+y^2")
    surface = deep.add(surface)
    deep.resize(400, 300)
    deep.show()
    qt.processEvents()
    # The GL context is created on first paint, so only now do we know whether there is
    # one; offscreen there usually isn't.
    if deep.view.broken:
        pytest.skip(f"no OpenGL to render with: {deep.view.broken}")
    hidden = _lit_pixels(deep)
    surface.item.setVisible(False)
    through = _lit_pixels(deep)
    assert 0 < hidden < through


def _lit_pixels(window):
    """Count the view's pixels that have color (as opposed to gray).

    The wire is the surface's hue at varying brightness, and everything else (canvas,
    box, labels) is gray, so saturation tells them apart.

    A frame buffer that reads back empty means OpenGL without a graphics card rendered
    nowhere, so the test is skipped.
    """
    from pyqtgraph.Qt import QtGui

    image = window.view.readQImage().convertToFormat(QtGui.QImage.Format.Format_RGB32)
    pixels = image.constBits()
    if pixels is None:
        pytest.skip("the OpenGL frame buffer read back empty")
    pixels = np.frombuffer(pixels, dtype=np.uint8)
    pixels = pixels.reshape(image.height(), image.width(), 4)[:, :, 2::-1]
    spread = pixels.astype(np.int16).max(axis=2) - pixels.astype(np.int16).min(axis=2)
    return int(np.count_nonzero(spread > 25))


def _right_click(view, start, end):
    """Send the view a right-button press at `start` and release at `end`."""
    from pyqtgraph.Qt import QtCore, QtGui

    for kind, at in (
        (QtCore.QEvent.Type.MouseButtonPress, start),
        (QtCore.QEvent.Type.MouseButtonRelease, end),
    ):
        ev = QtGui.QMouseEvent(
            kind,
            QtCore.QPointF(*at),
            QtCore.QPointF(*at),
            QtCore.Qt.MouseButton.RightButton,
            QtCore.Qt.MouseButton.RightButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        if kind == QtCore.QEvent.Type.MouseButtonPress:
            view.mousePressEvent(ev)
        else:
            view.mouseReleaseEvent(ev)


def test_the_3d_menu_is_the_cameras_list_with_its_keys(deep):
    from pyqtgraph.Qt import QtCore, QtGui

    # The 3D menu is about the camera; toolbar state such as the `mesh` box is not
    # repeated.
    assert _entries(deep.menu) == [
        ("Home view", "Home"),
        ("Face the xy plane", "1"),
        ("Face the xz plane", "2"),
        ("Face the yz plane", "3"),
        ("Rotate", "R"),
        ("View...", ""),
        ("Copy image", "Ctrl+C"),
        ("Export...", "Ctrl+S"),
        ("Remove", ""),
        ("Clear", "Del"),
        ("Close", "Q"),
    ]
    assert "mesh" not in {text for text, _ in _entries(deep.menu)}
    # Keys dispatch through the same table, so a key does what its entry says.
    deep.keyPressEvent(
        QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_1,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
    )
    assert deep.status.text() == "Facing the xy plane"


def test_the_3d_menu_a_window_opens_is_the_description_rendered(deep):
    # As in 2D, the menu shows exactly what `plot.controls` says for the snapshot.
    deep.add(_surface("x*y"))
    deep.spin()
    deep.menu.aboutToShow.emit()
    assert _shown(deep.menu) == [
        (entry.label, entry.checked) for entry in controls.menu(deep.snapshot())
    ]
    assert ("Rotate", True) in _shown(deep.menu)


def test_the_3d_menu_lists_every_surface_under_remove(deep):
    entry = _entry(deep.menu, "Remove")
    deep.menu.aboutToShow.emit()
    # With no surfaces the submenu is disabled rather than omitted.
    assert not entry.isEnabled()
    one, two = _surface("x*y"), _surface("x+y", label="#2")
    one = deep.add(one)
    two = deep.add(two)
    deep.menu.aboutToShow.emit()
    listed = deep._remove_menu.actions()
    assert [action.text() for action in listed] == [one.named, two.named]
    assert entry.isEnabled()
    listed[0].trigger()
    assert deep.plots == [two]


def test_the_rotate_entry_is_read_off_the_timer_that_turns_the_picture(deep):
    rotate = _entry(deep.menu, "Rotate")
    deep.spin()
    deep.menu.aboutToShow.emit()
    assert rotate.isChecked()
    # The entry stops it, like `R`, since both are the same command.
    rotate.trigger()
    deep.menu.aboutToShow.emit()
    assert not rotate.isChecked()
    assert not deep._spin.isActive()


def test_the_inspector_is_the_view_as_the_nine_numbers_it_is(deep):
    from rederive.plot import forms

    # A form of named fields for the box's center and size and the camera position. It
    # opens with the current values, so changing one leaves the other eight.
    deep.add(_surface("x*y"))
    deep.inspector()
    sheet = deep._inspector
    assert sheet.said(("cx", "lx")) == ("0", "10")
    assert sheet.fields["azimuth"].text() == str(int(actions.CAMERA.azimuth))
    # Applying sets the box, and a typed z range overrides autoscaling.
    for name, text in (("cz", "0"), ("lz", "2"), ("distance", "30")):
        sheet.fields[name].setText(text)
    sheet.answered(forms.Role.APPLY)
    assert deep.zrange == pytest.approx((-1.0, 1.0))
    assert deep.view.cameraParams()["distance"] == pytest.approx(30.0)
    # Reset returns z to autoscaling, and the fields update.
    sheet.answered(forms.Role.RESET)
    assert deep.zrange != (-1.0, 1.0)
    assert sheet.fields["lz"].text() != "2"
    # A non-numeric field is refused with a message and the form reverts.
    sheet.fields["cx"].setText("π")
    sheet.answered(forms.Role.APPLY)
    assert deep.status.text() == forms.NUMBERS
    assert sheet.fields["cx"].text() == "0"


def test_the_inspector_follows_the_camera_while_it_is_open(deep):
    deep.inspector()
    sheet = deep._inspector
    was = sheet.fields["azimuth"].text()
    deep.face("xy")
    # The form also acts as a live readout.
    assert sheet.fields["azimuth"].text() != was
    assert sheet.fields["elevation"].text() == "90"


def test_a_right_click_that_stays_put_is_what_asks_for_the_3d_menu(deep):
    asked = []
    deep.view.asked.connect(lambda point: asked.append(point))
    _right_click(deep.view, (10.0, 10.0), (11.0, 10.0))
    assert len(asked) == 1
    # A right drag is not a click and keeps its stock meaning, which in the GL view
    # (left orbits, middle pans) is nothing.
    _right_click(deep.view, (10.0, 10.0), (60.0, 40.0))
    assert len(asked) == 1


# -- the receiver ---------------------------------------------------------------
#
# The receiver is the window the user last touched, learned from the windows'
# activation events. None of this draws, so it runs on a fake backend, which also
# confirms the session uses only what `plot.backend` defines.


@pytest.fixture
def registry():
    """A plot session over a fake backend with no toolkit.

    Tests which window an `Add` lands in and how the receiver follows the user, so
    replies are return values and events are collected in a list.
    """
    from rederive.plot.session import PlotSession

    session = PlotSession(FakeBackend(), InlineExecutor())
    #: Every event the session has reported to the app.
    session.reported = []
    session.events = session.reported.append
    return session


@pytest.fixture
def windowed(qt):
    """The same session over the Qt backend, offscreen.

    Activation and the framing lock are handled by the window, not the session, so
    they need a real window.
    """
    from rederive.plot.qt.backend import QtBackend, ThreadExecutor
    from rederive.plot.session import PlotSession

    session = PlotSession(QtBackend(), ThreadExecutor())
    session.reported = []
    session.events = session.reported.append
    yield session
    for window in list(session.windows.values()):
        window.close()


def _landing(text, label, **keywords):
    """A curve `Add` request, as the app would send it."""
    return plots.Add(
        worksheet=1,
        node=parsed(text),
        context=Context(),
        kind=PlotKind.CURVE,
        label=label,
        text=text,
        options=plots.Options(variables=("x",)),
        **keywords,
    )


def test_a_plots_arrival_counts_as_a_touch(registry):
    placed = registry.add(_landing("SIN(x)", "#1"))
    assert placed == plots.Placed(1)
    window = registry.windows[1]
    assert window.current and window.presented == 1
    assert window.title == "SIN(x) - Rederive plot (current)"


def test_a_demonstrations_step_empties_the_window_first(registry):
    """In a gallery, each new picture replaces the previous one."""
    registry.add(_landing("SIN(x)", "#1"))
    registry.add(_landing("COS(x)", "#2"))
    landed = registry.add(_landing("TAN(x)", "#3", demonstrating=True))
    window = registry.windows[landed.window]
    assert [plot.label for plot in window.plots] == ["#3"]


def test_a_demonstrations_step_is_shown_without_taking_the_keyboard(registry):
    """A demonstration window doesn't take focus, since the program awaits a key.

    An ordinary plot window is brought to the front to be worked in and does take
    focus.
    """
    commanded = registry.windows[registry.add(_landing("SIN(x)", "#1")).window]
    assert commanded.presented == 1 and not commanded.quietly
    landed = registry.add(_landing("COS(x)", "#2", demonstrating=True))
    shown = registry.windows[landed.window]
    assert shown.presented == 2 and shown.quietly


def test_a_demonstration_that_is_over_lets_every_window_go(registry):
    """Releasing a demonstration releases all its windows, e.g. both 2D and 3D."""
    registry.add(_landing("SIN(x)", "#1", demonstrating=True))
    registry.add(_landing("COS(x)", "#2", window=plots.Where.NEW, demonstrating=True))
    assert all(window.demonstrating for window in registry.windows.values())
    registry.handle(plots.Release(), lambda reply: None)
    assert not any(window.demonstrating for window in registry.windows.values())
    assert all(window.released == 1 for window in registry.windows.values())


def test_a_demonstration_that_ran_out_closes_the_windows_it_drew_in(registry):
    """Only demonstration windows close; a window opened for a user's plot stays."""
    theirs = registry.add(_landing("SIN(x)", "#1")).window
    shown = registry.add(_landing("COS(x)", "#2", window=plots.Where.NEW)).window
    registry.add(_landing("TAN(x)", "#3", window=shown, demonstrating=True))
    registry.handle(plots.Release(finished=True), lambda reply: None)
    assert sorted(registry.windows) == [theirs]


def test_touching_a_window_makes_it_the_receiver(registry):
    one = registry.add(_landing("SIN(x)", "#1")).window
    two = registry.add(_landing("COS(x)", "#2", window=plots.Where.NEW)).window
    assert registry.windows[two].current and not registry.windows[one].current
    registry.touched(one)
    assert registry.windows[one].current and not registry.windows[two].current
    assert registry.windows[one].title.endswith("(current)")
    # The next plot follows the touch.
    assert registry.add(_landing("TAN(x)", "#3")).window == one


def test_the_activation_event_is_what_feeds_the_receiver(windowed, qt):
    # Activating a window (by click, raise or alt-tab) reaches the registry directly.
    one = windowed.add(_landing("SIN(x)", "#1")).window
    windowed.add(_landing("COS(x)", "#2", window=plots.Where.NEW))
    windowed.windows[one].activateWindow()
    qt.processEvents()
    assert windowed.windows[one].current
    assert windowed.add(_landing("TAN(x)", "#3")).window == one


def test_closing_the_receiver_hands_it_to_the_last_activated_survivor(registry):
    one = registry.add(_landing("SIN(x)", "#1")).window
    two = registry.add(_landing("COS(x)", "#2", window=plots.Where.NEW)).window
    three = registry.add(_landing("TAN(x)", "#3", window=plots.Where.NEW)).window
    registry.touched(one)
    registry.touched(three)
    registry.windows[three].close()
    # The receiver passes to the last-activated survivor, not the newest window.
    assert registry.windows[one].current and not registry.windows[two].current
    assert registry.add(_landing("x^2", "#4")).window == one


def test_a_plot_for_a_window_that_is_not_there_is_refused_by_number(registry):
    registry.add(_landing("SIN(x)", "#1"))
    assert registry.add(_landing("COS(x)", "#2", window=7)) == plots.Refused(
        "there is no plot window 7"
    )


def test_a_curve_sent_to_a_surface_window_is_refused_in_words(registry):
    solid = registry.add(_surfaced("x*y", "#1")).window
    refused = registry.add(_landing("SIN(x)", "#2", window=solid))
    assert isinstance(refused, plots.Refused)
    assert refused.message == f"window {solid} is a 3D window"


def test_a_kind_no_window_draws_is_refused_before_a_window_is_opened(
    registry, monkeypatch
):
    # Every kind is drawn today, so this removes one from the drawn set. No window
    # must be opened for the refused plot.
    monkeypatch.setattr(plots, "DRAWN", frozenset({PlotKind.CURVE}))
    refused = registry.add(_surfaced("x*y", "#1"))
    assert refused == plots.Refused("surface plots are not implemented yet")
    assert registry.windows == {}


# -- the sticky preferences in the registry --------------------------------------
#
# The write-back half of section 7: a control changed in a window updates the
# preferences for the next plot, and the change is reported so the app can persist it.


def _surfaced(text, label, **keywords):
    """A surface `Add` request, as the app would send it."""
    return plots.Add(
        worksheet=1,
        node=parsed(text),
        context=Context(),
        kind=PlotKind.SURFACE,
        label=label,
        text=text,
        options=plots.Options(variables=("x", "y")),
        **keywords,
    )


def _points(text, label, **keywords):
    """A data plot `Add` request with no options of its own."""
    return plots.Add(
        worksheet=1,
        node=parsed(text),
        context=Context(),
        kind=PlotKind.DATA,
        label=label,
        text=text,
        **keywords,
    )


def test_a_toggle_left_in_one_window_shapes_the_next_data_plot(registry):
    first = registry.windows[registry.add(_points("[[1, 2], [3, 4]]", "#1")).window]
    assert first.plots[0].connected is False
    # A window reports a changed control...
    registry.adjusted(connected=True)
    registry.adjusted(point_size=8.0)
    # ...the session keeps the values for the next plot...
    assert registry.preferences == plots.Prefer(connected=True, point_size=8.0)
    # ...and tells the app, so the values persist.
    assert registry.reported[0] == plots.Preferred(plots.Prefer(connected=True))
    # A data plot without its own options uses them, even in a new window.
    fresh = registry.add(_points("[[5, 6], [7, 8]]", "#2", window=plots.Where.NEW))
    arrived = registry.windows[fresh.window].plots[0]
    assert arrived.connected is True
    assert arrived.point_size == 8.0


def test_a_released_framing_lock_is_this_windows_alone(windowed):
    one = windowed.add(_landing("SIN(x)", "#1")).window
    windowed.windows[one].equal.trigger()
    assert not windowed.windows[one].equal.isChecked()
    # Nothing was kept or reported, and the next window opens with equal scales.
    assert windowed.preferences == plots.Prefer()
    assert windowed.reported == []
    two = windowed.add(_landing("COS(x)", "#2", window=plots.Where.NEW)).window
    assert windowed.windows[two].equal.isChecked()


def test_what_was_handed_back_is_what_the_next_window_is_opened_with(registry):
    # Section 10's stickiness round trip: the session keeps an adjusted value, tells
    # the app, and builds the next window from the current preferences, which is the
    # only way preferences reach a window.
    registry.adjusted(grid=32)
    registry.adjusted(wire=True)
    assert registry.reported == [
        plots.Preferred(plots.Prefer(grid=32)),
        plots.Preferred(plots.Prefer(grid=32, wire=True)),
    ]
    registry.add(_surfaced("x*y", "#1", window=plots.Where.NEW))
    opened = registry.backend.opened[-1]
    assert opened.preferences == plots.Prefer(grid=32, wire=True)


def test_the_grid_handed_back_is_the_next_surface_windows_grid(windowed, space):
    windowed.adjusted(grid=32)
    window = windowed._target(plots.Where.NEW, plots.WindowKind.THREE_D)
    assert window.grid == (32, 32)


def test_the_wire_handed_back_is_the_next_surface_windows_look(windowed, space):
    # The Qt side of the round trip: the window opens with the mesh box checked.
    windowed.adjusted(wire=True)
    window = windowed._target(plots.Where.NEW, plots.WindowKind.THREE_D)
    assert window.wired
    assert window.mesh_action.isChecked()


# -- the preferences -----------------------------------------------------------
#
# Sticky plot values live in the settings store. They must translate into the host's
# request, reach a host that may not be running yet, and serve as defaults for plots
# without their own options. None of this needs a window.


def test_the_preferences_translate_the_words_the_settings_hold():
    from rederive.model.plotting import preferences
    from rederive.model.settings import Settings

    settings = Settings()
    # The settings defaults match the request defaults, so an untouched session
    # behaves like one where a control was changed and changed back.
    assert preferences(settings) == plots.Prefer()
    settings.apply(
        {
            "PlotGrid": 128,
            "PlotPoints": "Connected",
            "PlotPointSize": 8,
            "PlotMesh": "Wire",
        }
    )
    assert preferences(settings) == plots.Prefer(
        grid=128, connected=True, point_size=8.0, wire=True
    )


def test_what_a_host_reports_round_trips_through_the_settings():
    from rederive.model.plotting import learned, preferences
    from rederive.model.settings import Settings

    reported = plots.Prefer(grid=32, connected=True, point_size=8.0, wire=True)
    settings = Settings()
    settings.apply(learned(reported))
    # The two translations are inverses, so the next host gets exactly what was set.
    assert preferences(settings) == reported


def test_the_preferences_travel_in_a_state_file():
    from rederive.model import state
    from rederive.model.plotting import preferences
    from rederive.model.settings import Settings

    written = Settings()
    written.apply({"PlotGrid": 32, "PlotPoints": "Connected", "PlotMesh": "Wire"})
    read = Settings()
    assert state.read(state.write(written), read) == (0, "")
    assert preferences(read) == plots.Prefer(grid=32, connected=True, wire=True)


def test_equal_scales_is_nobodys_setting_and_nothing_persists_it():
    # A new window always opens with equal scales and the 1:1 toggle is a one-off, so
    # there is no setting for it and nothing in the state file.
    from rederive.model import state
    from rederive.model.settings import DEFAULTS, Settings

    assert "PlotScales" not in DEFAULTS
    assert "PlotScales" not in state.write(Settings())


def test_a_plot_with_no_opinion_is_drawn_the_way_the_preferences_say():
    from rederive.plot.session import preferred

    request = plots.Add(
        worksheet=0,
        node=parsed("[[1, 2], [3, 4]]"),
        context=Context(),
        kind=PlotKind.DATA,
    )
    filled = preferred(request, plots.Prefer(connected=True, point_size=12.0))
    assert (filled.options.connected, filled.options.point_size) == (True, 12.0)


def test_a_plot_that_has_an_opinion_keeps_it():
    """Options set on a plot override preferences, so a chosen point size survives."""
    from rederive.plot.session import preferred

    request = plots.Add(
        worksheet=0,
        node=parsed("[[1, 2], [3, 4]]"),
        context=Context(),
        kind=PlotKind.DATA,
        options=plots.Options(connected=False, point_size=3.0),
    )
    assert preferred(request, plots.Prefer(connected=True, point_size=12.0)) is request


def test_the_preferences_go_in_front_of_the_next_request_and_only_once(monkeypatch):
    """A changed preference is sent with the next plot, not immediately.

    This lets the settings watcher call `prefer` from the event loop, and a session
    that never plots again starts no process.
    """
    from rederive.plot import proxy as proxy_module

    sent = []

    def delivering(self, request):
        sent.append(request)
        return plots.Placed(1)

    monkeypatch.setattr(proxy_module.PlotProxy, "_require", lambda self: None)
    monkeypatch.setattr(proxy_module.PlotProxy, "_deliver", delivering)
    proxy = proxy_module.PlotProxy()
    request = plots.Add(
        worksheet=0, node=parsed("SIN(x)"), context=Context(), kind=PlotKind.CURVE
    )
    proxy.prefer(plots.Prefer(grid=16))
    assert sent == []
    proxy.plot(request)
    assert sent == [plots.Prefer(grid=16), request]
    # A second plot doesn't resend it.
    proxy.plot(request)
    assert sent[2:] == [request]
    # Nor does setting the same preferences again.
    proxy.prefer(plots.Prefer(grid=16))
    proxy.plot(request)
    assert sent[3:] == [request]


def test_a_restarted_host_hears_the_preferences_again(monkeypatch):
    """A restarted host knows only the defaults, so preferences are resent with its
    first plot."""
    from rederive.plot import proxy as proxy_module

    sent = []

    class Alive:
        def is_alive(self):
            return True

    def spawning(self):
        self._replies[plots.READY] = plots.Done()
        self._process = Alive()
        self.starts += 1

    def delivering(self, request):
        sent.append(request)
        return plots.Placed(1)

    monkeypatch.setattr(proxy_module.PlotProxy, "_spawn", spawning)
    monkeypatch.setattr(proxy_module.PlotProxy, "_deliver", delivering)
    proxy = proxy_module.PlotProxy()
    request = plots.Add(
        worksheet=0, node=parsed("SIN(x)"), context=Context(), kind=PlotKind.CURVE
    )
    proxy.prefer(plots.Prefer(connected=True))
    proxy.plot(request)
    assert sent == [plots.Prefer(connected=True), request]
    # The host dies; the next plot starts a new one and resends the preferences.
    proxy._process = None
    proxy.plot(request)
    assert proxy.starts == 2
    assert sent[2:] == [plots.Prefer(connected=True), request]


# -- the Plot command in the algebra window ------------------------------------
#
# The app is driven with a fake proxy instead of a host, since this tests the wiring:
# which request a keystroke sends and what the message line says about the answer. The
# host is tested above, and starting one per test would require Qt for all of these.


class Answering:
    """A plot proxy that records what it was asked and answers plausibly."""

    def __init__(self, windows=(), refuse=None):
        self.sent = []
        self.windows = windows
        self.refuse = refuse
        self.events = None
        #: The reply to `add`; tests override it to report a replaced curve.
        self.placed = plots.Placed(1)
        #: How many times the app asked for window contents, which a plain plot must
        #: never do.
        self.described = 0
        #: Preferences received, newest last. The real proxy sends them with the next
        #: request; the app only has to hand them over.
        self.preferences = []
        #: How many times the app ended a demonstration (so its window no longer
        #: stays on top), and whether the last one ran to completion and closed it.
        self.released = 0
        self.finished = None

    def _answer(self, request):
        self.sent.append(request)
        if self.refuse is not None:
            raise PlotError(self.refuse)

    async def add(self, request):
        self._answer(request)
        return self.placed

    def describe(self):
        self.described += 1
        return self.windows

    def prefer(self, preferences):
        self.preferences.append(preferences)

    async def release(self, finished=False):
        self.released += 1
        self.finished = finished

    def shutdown(self):
        pass


@pytest.fixture
def app(monkeypatch):
    """The app, with a fake host that answers every request without drawing.

    `DISPLAY` is set explicitly, since without a display (e.g. on a build runner) the
    Plot command refuses outright. That refusal has its own test below.
    """
    from rederive.ui.app import RederiveApp

    monkeypatch.setenv("DISPLAY", ":0")
    made = RederiveApp()
    made.plots = Answering()
    return made


async def authored(pilot, app, text):
    app.session.author(text)
    await pilot.pause()


async def test_the_plot_menu_offers_two_words_with_plot_first(app):
    async with app.run_test() as pilot:
        await pilot.press("p")
        assert band(app) == [" PLOT: Plot New"]
        assert highlighted(app) == "Plot"
        assert message(app) == "Enter option"


async def test_p_p_plots_the_highlighted_expression_with_no_question(app):
    async with app.run_test() as pilot:
        await authored(pilot, app, "SIN(x)")
        await pilot.press("p", "p")
        assert message(app) == "Plotting #1"
        request = app.plots.sent[0]
        assert request.label == "#1"
        assert request.text == "SIN(x)"
        assert request.kind is PlotKind.CURVE
        # No window given: the plot goes to the receiver of its kind.
        assert request.window is None
        assert request.options.variables == ("x",)
        # The submenu is done and the command menu is back.
        assert band(app)[0].startswith(" COMMAND:")


async def test_plot_new_asks_for_a_window_of_its_own(app):
    async with app.run_test() as pilot:
        await authored(pilot, app, "SIN(x)")
        await pilot.press("p", "n")
        assert app.plots.sent[0].window is plots.Where.NEW


async def test_plotting_with_nothing_highlighted_is_refused(app):
    async with app.run_test() as pilot:
        await pilot.press("p", "p")
        assert message(app) == "Plot: no expression to plot"
        assert app.plots.sent == []


async def test_a_surface_is_sent_with_no_question_asked(app):
    async with app.run_test() as pilot:
        await authored(pilot, app, "x^2 - y^2")
        await pilot.press("p", "p")
        assert message(app) == "Plotting #1"
        request = app.plots.sent[0]
        assert request.kind is PlotKind.SURFACE
        assert request.options.variables == ("x", "y")
        # No prompt: the window owns the surface's domain.
        assert band(app)[0].startswith(" COMMAND:")


async def test_a_vector_of_surfaces_carries_the_text_of_every_element(app):
    async with app.run_test() as pilot:
        await authored(pilot, app, "[x + y, x - y]")
        await pilot.press("p", "p")
        request = app.plots.sent[0]
        assert request.kind is PlotKind.SURFACES
        assert request.options.texts == ("x + y", "x - y")


async def test_a_kind_no_window_draws_yet_is_refused_before_it_is_sent(app, monkeypatch):
    # Every kind is drawn today, so this removes one from the drawn set. The refusal
    # covers any future kind that classification knows before a window can draw it.
    monkeypatch.setattr(plots, "DRAWN", frozenset({PlotKind.CURVE}))
    async with app.run_test() as pilot:
        await authored(pilot, app, "x*y")
        await pilot.press("p", "p")
        assert message(app) == "Plot: surface plots are not implemented yet"
        assert app.plots.sent == []


async def test_an_expression_of_three_variables_is_refused_by_name(app):
    async with app.run_test() as pilot:
        await authored(pilot, app, "a*x*y")
        await pilot.press("p", "p")
        assert message(app) == (
            "Plot: a*x*y depends on x, y, a - reduce to one or two variables"
        )


async def test_a_parametric_pair_is_sent_with_no_question_asked(app):
    async with app.run_test() as pilot:
        await authored(pilot, app, "[SIN(t), COS(t)]")
        await pilot.press("p", "p")
        # No prompt: it draws over one turn, and the range is adjusted in the plot
        # window's toolbar.
        assert message(app) == "Plotting #1"
        request = app.plots.sent[0]
        assert request.kind is PlotKind.PARAMETRIC
        assert request.options.variables == ("t",)
        # The parse state travels with the plot, so the window's range fields parse
        # bounds with the worksheet's grammar.
        assert request.state is app.session.state
        assert band(app)[0].startswith(" COMMAND:")


async def test_a_plot_is_sent_without_asking_any_window_its_mode(app):
    async with app.run_test() as pilot:
        # Polar is a window view mode, so a univariate expression is always sent as a
        # curve, without a `Describe` round trip.
        await authored(pilot, app, "SIN(x)")
        await pilot.press("p", "p")
        assert app.plots.sent[0].kind is PlotKind.CURVE
        assert app.plots.described == 0


async def test_a_family_carries_the_text_of_every_element(app):
    async with app.run_test() as pilot:
        await authored(pilot, app, "[x, x^2, x^3]")
        await pilot.press("p", "p")
        request = app.plots.sent[0]
        assert request.kind is PlotKind.FAMILY
        assert request.options.texts == ("x", "x^2", "x^3")


async def test_a_highlighted_subexpression_is_what_gets_plotted(app):
    async with app.run_test() as pilot:
        await authored(pilot, app, "SIN(x) + 3")
        # Right selects the first term.
        app.session.move_right()
        await pilot.press("p", "p")
        request = app.plots.sent[0]
        assert request.label == "#1'"
        assert request.text == "SIN(x)"


async def test_a_plot_that_replaces_a_curve_says_replotting(app):
    async with app.run_test() as pilot:
        await authored(pilot, app, "SIN(x)")
        # The host reports a replaced curve, and the message says so.
        app.plots.placed = plots.Placed(1, replaced=True)
        await pilot.press("p", "p")
        assert message(app) == "Replotting #1"


# -- the plot galleries -------------------------------------------------------
#
# The original shipped three files of captioned expressions to look at. They work like
# demonstrations, except each step's expression is plotted rather than simplified. The
# page's demo menu runs them.


@pytest.fixture
def gallery(tmp_path):
    path = tmp_path / "plot2d.mth"
    path.write_text('"A spike:"\n\n2*SIN(x)\n\n"A saddle:"\n\ny^2-x^2\n')
    return path


async def test_a_gallery_draws_one_step_at_a_time(app, gallery):
    async with app.run_test() as pilot:
        app.demonstrate(str(gallery), plotting=True)
        await pilot.pause()
        # The caption replaces the menu band and waits there.
        assert band(app) == [" A spike:"]
        # The message line is a demonstration's, not the Plot command's.
        assert message(app) == "Press any key to continue, ESC to stop"
        request = app.plots.sent[0]
        assert request.label == "#1"
        assert request.kind is PlotKind.CURVE
        await pilot.press("space")
        assert band(app) == [" A saddle:"]
        # The second step is a surface and goes to the surface receiver; galleries don't
        # choose windows.
        assert app.plots.sent[1].kind is PlotKind.SURFACE
        assert app.plots.sent[1].window is None
        # After the last step, the command menu returns.
        await pilot.press("space")
        assert band(app)[0].startswith(" COMMAND:")
        assert message(app) == "Enter option"


async def test_a_gallery_that_ran_out_takes_its_window_with_it(app, gallery):
    """A finished gallery closes its window."""
    async with app.run_test() as pilot:
        app.demonstrate(str(gallery), plotting=True)
        await pilot.pause()
        assert app.plots.released == 0
        await pilot.press("space", "space")
        await pilot.pause()
        assert (app.plots.released, app.plots.finished) == (1, True)


async def test_a_gallery_somebody_stopped_leaves_its_picture_standing(app, gallery):
    """Esc stops the gallery but keeps the current picture open."""
    async with app.run_test() as pilot:
        app.demonstrate(str(gallery), plotting=True)
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert (app.plots.released, app.plots.finished) == (1, False)


async def test_a_script_has_no_window_to_let_go_of(app, tmp_path):
    """A non-plot script sends no release, which would needlessly start a host."""
    path = tmp_path / "show.dmo"
    path.write_text("; adds two numbers\n2 + 3\n")
    async with app.run_test() as pilot:
        app.demonstrate(str(path))
        await pilot.pause()
        await pilot.press("space")
        assert app.plots.released == 0


async def test_a_key_pressed_in_the_picture_steps_the_gallery(app, gallery):
    """A key pressed in the plot window advances the gallery, as one in the app does."""
    async with app.run_test() as pilot:
        app.demonstrate(str(gallery), plotting=True)
        await pilot.pause()
        app._plot_event(plots.Stepped())
        await pilot.pause()
        assert band(app) == [" A saddle:"]


async def test_a_key_from_a_window_with_nothing_waiting_is_dropped(app, gallery):
    """A key that arrives over the pipe after the demonstration ended is ignored."""
    async with app.run_test() as pilot:
        app.demonstrate(str(gallery), plotting=True)
        await pilot.pause()
        await pilot.press("escape")
        app._plot_event(plots.Stepped())
        assert band(app)[0].startswith(" COMMAND:")


async def test_a_window_closed_under_a_step_does_not_step_it(app, gallery):
    """Closing a plot window must not advance the demonstration.

    Most desktops pass the closing click through to the program underneath, where a
    demonstration would take it as a step. The window's close report lets the app
    ignore that click.
    """
    async with app.run_test() as pilot:
        app.demonstrate(str(gallery), plotting=True)
        await pilot.pause()
        app._plot_event(plots.Closed(1))
        await pilot.click(content(app), offset=(2, 0))
        assert band(app) == [" A spike:"]
        assert len(app.plots.sent) == 1
        # The next click advances as usual.
        await pilot.click(content(app), offset=(2, 0))
        assert band(app) == [" A saddle:"]


async def test_a_window_closed_with_no_demonstration_waiting_is_nothing(app, gallery):
    """Without a waiting demonstration, a close report doesn't swallow a later click."""
    async with app.run_test() as pilot:
        app._plot_event(plots.Closed(1))
        app.demonstrate(str(gallery), plotting=True)
        await pilot.pause()
        await pilot.click(content(app), offset=(2, 0))
        assert band(app) == [" A saddle:"]


async def test_a_gallery_step_is_the_only_picture_the_window_keeps(app, gallery):
    """Each gallery step replaces the previous picture, since they are unrelated."""
    async with app.run_test() as pilot:
        app.demonstrate(str(gallery), plotting=True)
        await pilot.pause()
        await pilot.press("space")
        assert [request.demonstrating for request in app.plots.sent] == [True, True]


async def test_a_gallery_line_that_will_not_draw_is_walked_past(app, gallery):
    """An unplottable line is skipped, like an unparseable one."""
    gallery.write_text('"Not a picture:"\n\nx+y+z\n\n"A spike:"\n\n2*SIN(x)\n')
    async with app.run_test() as pilot:
        app.demonstrate(str(gallery), plotting=True)
        await pilot.pause()
        assert band(app) == [" A spike:"]
        assert [request.text for request in app.plots.sent] == ["2*SIN(x)"]


async def test_a_gallery_leaves_its_expressions_on_the_worksheet(app, gallery):
    """Every step is authored into the worksheet, so it can be replotted."""
    async with app.run_test() as pilot:
        app.demonstrate(str(gallery), plotting=True)
        await pilot.pause()
        await pilot.press("space", "space")
        assert [entry.text for entry in app.session.entries] == ["2*SIN(x)", "y^2-x^2"]


async def test_a_gallery_without_a_display_is_refused_as_a_plot_is(
    app, gallery, monkeypatch
):
    import rederive.ui.app as app_module

    monkeypatch.setattr(app_module, "available", lambda: False)
    async with app.run_test() as pilot:
        app.demonstrate(str(gallery), plotting=True)
        await pilot.pause()
        assert message(app) == "Plot: needs a graphical display"
        assert app.plots.sent == []
        assert app.session.entries == []


async def test_plotting_without_a_display_is_refused_but_still_offered(
    app, monkeypatch
):
    import rederive.ui.app as app_module

    monkeypatch.setattr(app_module, "available", lambda: False)
    async with app.run_test() as pilot:
        await authored(pilot, app, "SIN(x)")
        await pilot.press("p", "p")
        assert message(app) == "Plot: needs a graphical display"
        assert app.plots.sent == []


async def test_the_options_plot_screen_is_gone(app):
    async with app.run_test() as pilot:
        # There is no Options Plot screen: sticky values are set by the plot windows'
        # controls, so the menu doesn't offer it and `l` does nothing.
        await pilot.press("o")
        assert band(app) == [
            " OPTIONS: Color Input Mute Notation Output Precision Radix"
        ]
        await pilot.press("l")
        assert band(app) == [
            " OPTIONS: Color Input Mute Notation Output Precision Radix"
        ]


async def test_a_control_the_host_reports_becomes_the_sticky_default(app):
    async with app.run_test() as pilot:
        app._plot_event(
            plots.Preferred(
                plots.Prefer(grid=32, connected=True, point_size=8.0, wire=True)
            )
        )
        await pilot.pause()
        # The values go to the settings store, which outlives the host and is saved in
        # state files...
        assert app.settings["PlotGrid"] == 32
        assert app.settings["PlotPoints"] == "Connected"
        assert app.settings["PlotPointSize"] == 8
        assert app.settings["PlotMesh"] == "Wire"
        # ...and are handed back to the proxy for any future host.
        assert app.plots.preferences[-1] == plots.Prefer(
            grid=32, connected=True, point_size=8.0, wire=True
        )
        # Nothing is authored or announced; the user is looking at the control.
        assert app.session.entries == []


async def test_what_a_host_reported_is_what_a_state_file_carries(app, tmp_path):
    async with app.run_test() as pilot:
        app._plot_event(
            plots.Preferred(plots.Prefer(connected=True, point_size=8.0, wire=True))
        )
        await pilot.pause()
        path = tmp_path / "kept.ini"
        app.session.save_state(path)
        lines = path.read_text().splitlines()
        assert "PlotPoints := Connected" in lines
        assert "PlotPointSize := 8" in lines
        assert "PlotMesh := Wire" in lines


async def test_a_state_file_that_moves_a_preference_reaches_the_plots(app, tmp_path):
    async with app.run_test() as pilot:
        app.settings.apply({"PlotPoints": "Connected", "PlotPointSize": 9})
        path = tmp_path / "kept.ini"
        app.session.save_state(path)
        app.settings.apply({"PlotPoints": "Discrete", "PlotPointSize": 5})
        assert app.session.load_state(path) == 0
        await pilot.pause()
        assert app.plots.preferences[-1] == plots.Prefer(connected=True, point_size=9.0)


async def test_a_curve_the_host_could_not_draw_reaches_the_message_line(app):
    async with app.run_test() as pilot:
        app._plot_event(plots.Trouble(1, "#3", "no numeric reading of FOO"))
        await pilot.pause()
        assert message(app) == "Plot: #3: no numeric reading of FOO"


async def test_a_point_sent_home_is_authored_into_its_worksheet(app):
    async with app.run_test() as pilot:
        app._plot_event(plots.Traced(id(app.session), "[1.570796, 1.000000]"))
        await pilot.pause()
        [entry] = app.session.entries
        # This enters exactly what pasting the Ctrl-C text on the author line would.
        assert entry.text == Session().author("[1.570796, 1.000000]").text
        assert message(app) == "Entered [1.570796, 1.000000] as #1"


async def test_a_point_lands_in_its_own_worksheet_not_the_active_one(app):
    async with app.run_test() as pilot:
        # An overlay makes a second worksheet active; the point still goes to the
        # worksheet its plot came from.
        first = app.session
        app.windows.open(app.windows.kind, first.copy())
        assert app.session is not first
        app._plot_event(plots.Traced(id(first), "[1.000000, 2.000000]"))
        await pilot.pause()
        assert len(first.entries) == 1
        assert app.session.entries == []
        assert message(app) == "Entered [1.000000, 2.000000] as #1"


async def test_a_point_from_a_closed_worksheet_finds_the_active_one(app):
    async with app.run_test() as pilot:
        # The plot's worksheet (an overlay) has since closed, so the point lands in the
        # active worksheet rather than being dropped.
        app._plot_event(plots.Traced(-1, "[1.000000, 2.000000]"))
        await pilot.pause()
        assert len(app.session.entries) == 1
        assert message(app) == "Entered [1.000000, 2.000000] as #1"


async def test_a_point_arriving_mid_edit_leaves_the_line_standing(app):
    async with app.run_test() as pilot:
        # With a half-typed Author line, the entry is added and the prompt is left
        # untouched; F3 can bring the point into the line.
        await pilot.press("a", "s", "i", "n")
        app._plot_event(plots.Traced(id(app.session), "[1.000000, 2.000000]"))
        await pilot.pause()
        assert len(app.session.entries) == 1
        assert message(app) == "Entered [1.000000, 2.000000] as #1"
        assert prompt(app) == ("AUTHOR expression:", "sin")


async def test_a_point_arriving_mid_command_waits_its_turn(app):
    from rederive.ui.app import MODE_COMPUTE, MODE_MENU

    async with app.run_test() as pilot:
        # While a command computes, its thread owns the worksheets, so the point waits.
        app.mode = MODE_COMPUTE
        app._plot_event(plots.Traced(id(app.session), "[1.000000, 2.000000]"))
        await pilot.pause()
        assert app.session.entries == []
        app.mode = MODE_MENU
        for _ in range(100):
            await asyncio.sleep(0.05)
            await pilot.pause()
            if app.session.entries:
                break
        assert len(app.session.entries) == 1
        assert message(app) == "Entered [1.000000, 2.000000] as #1"


async def test_the_whole_loop_lands_a_parametric_plot_in_a_real_window(monkeypatch):
    """`P` `P` puts a curve in a real window, with no prompts.

    The only test combining the app and a real host; others stub one side. The seam
    can break silently (the app's screen is not a file descriptor, and a spawn that
    inherits one is refused), so it's worth the process start.
    """
    if not _toolkit():
        pytest.skip("pyqtgraph and PySide6 are not installed")
    from rederive.plot import proxy as proxy_module
    from rederive.ui.app import RederiveApp

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    # `DISPLAY` satisfies the command's check, while the host draws offscreen.
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(proxy_module, "READY_TIMEOUT", 60.0)
    monkeypatch.setattr(proxy_module, "REPLY_TIMEOUT", 20.0)
    app = RederiveApp()
    try:
        async with app.run_test() as pilot:
            app.session.author("[3*SIN(3*t), 3*COS(2*t)]")
            await pilot.pause()
            await pilot.press("p", "p")
            for _ in range(100):
                await pilot.pause()
                if message(app).startswith(("Plotting #", "Plot:")):
                    break
                await asyncio.sleep(0.05)
            assert message(app) == "Plotting #1"
            window = app.plots.describe()[0]
            # The window title is the plot text as the app wrote it.
            assert window.title == f"{window.plots[0].text} - Rederive plot (current)"
            assert window.plots[0].kind is PlotKind.PARAMETRIC
    finally:
        app.plots.shutdown()


def test_availability_needs_a_display_on_linux(monkeypatch):
    import sys

    from rederive import plot

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert plot.available() is False
    monkeypatch.setenv("DISPLAY", ":0")
    assert plot.available() is True
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv("DISPLAY", raising=False)
    assert plot.available() is True

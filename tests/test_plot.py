"""Plotting: what an expression is a plot of, what it evaluates to, and the host.

Three test plans in one file, because they are three halves of one command.

Classification is pure and is the bulk of it: the table of section 5 has eleven
rows and the interesting ones are the shapes a reader would argue about - a
2-vector that is a point, a parameter pair or a pair of surfaces depending on
how many variables it holds.

The evaluator is where the corner cases live. Every one of these is a picture
that has been drawn wrong by some plotter: the bridging stroke across a pole,
the negative half of a square root drawn as zero, the spike between two
samples that no zoom ever reveals.

The host is exercised over a real pipe with a real Qt toolkit, offscreen. It is
the one test here that starts a process, so it is skipped where there is no Qt
to start it with, and every wait in it has a deadline: a test that hangs the
suite is worse than a test that is not run.
"""

import asyncio

import numpy as np
import pytest
from screen import band, highlighted, message, prompt

from rederive.engine.context import Context
from rederive.model.plotting import Unplottable, classify
from rederive.model.session import Session
from rederive.plot import evaluate
from rederive.plot import protocol as plots
from rederive.plot.protocol import PlotKind
from rederive.plot.proxy import PlotError
from rederive.syntax import ParseState, parse_expression

#: How wide and tall the canvas is taken to be where a test does not care. The
#: sampler needs a size because its tolerance is in pixels.
CANVAS = (800.0, 600.0)

#: The default framing, which is what most of these are sampled over.
VIEW = (-5.0, 5.0)


def parsed(text):
    return parse_expression(text, ParseState()).node


def closure(text, variables=("x",)):
    return evaluate.closure(parsed(text), Context(), variables)


def sampled(text, xrange=VIEW, yrange=VIEW, size=CANVAS):
    return evaluate.sample_adaptive(closure(text), xrange, yrange, size)


# -- classification -----------------------------------------------------------


def kind_of(text, variables=()):
    """What `text` classifies as, with the variables the engine would find."""
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
    # `z = x + y + z` is an equation in three variables and no reading of it
    # names an axis, so it is refused rather than read as a surface.
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


def test_classification_reads_what_a_label_means_and_not_what_it_says():
    # The command resolves `#1` before classifying, so what is classified is
    # the expression, whatever it was written as.
    session = Session()
    session.author("SIN(x)")
    target = session.target("#1")
    assert classify(target, session.variables("#1"), "#1").kind is PlotKind.CURVE


# -- evaluation ---------------------------------------------------------------


def test_a_closure_masks_the_complex_half_of_a_square_root():
    values = closure("SQRT(x)")(np.array([-4.0, -1.0, 0.0, 1.0, 4.0]))
    assert np.isnan(values[0]) and np.isnan(values[1])
    assert list(values[2:]) == [0.0, 1.0, 2.0]


def test_a_closure_answers_nan_where_the_value_is_not_finite():
    values = closure("1/x")(np.array([-1.0, 0.0, 1.0]))
    assert list(values[[0, 2]]) == [-1.0, 1.0]
    assert np.isnan(values[1])


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
    # The acceptance case: no segment of the drawn curve may cross the pole,
    # which means a NaN between the sample below it and the sample above.
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
    # The whole argument for re-sampling on every view change: at the default
    # framing this spike is between two samples, and zooming toward it is what
    # brings it into the data.
    text = "1/(1 + 10000*(x - 1)^2)"
    _, wide = sampled(text)
    _, close = sampled(text, xrange=(0.99, 1.01), yrange=(-0.1, 1.1))
    assert np.nanmax(wide) < 0.5
    assert np.nanmax(close) > 0.99


def test_a_curve_with_a_pole_in_it_still_terminates():
    # `1/x` over a range whose midpoint is the pole is the shape that makes a
    # naive bisection recurse forever.
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
    # An assignment is not an expression, and nothing numeric comes of it.
    # Classification refuses one long before this, so what is under test is
    # that the failure is a message and not a traceback.
    with pytest.raises(evaluate.Unplottable):
        evaluate.closure(parsed("x := 3"), Context(), ("x",))


def test_a_function_with_no_numeric_reading_is_all_gaps_rather_than_a_crash():
    # `FOO` is nobody's function, so every point of it is unevaluable - which
    # is a curve with no real values, and the window says so.
    values = closure("FOO(x)")(np.array([1.0, 2.0]))
    assert evaluate.finite_fraction(values) == 0.0


def test_a_parametric_pair_through_a_pole_terminates_with_a_gap():
    # `[t, 1/(t - 1)]` is the acceptance case: the two branches never come close
    # on the screen however finely t is cut, so refinement has to stop at the
    # depth cap rather than chase the pole down forever. The pole is at 1 rather
    # than at 0 so that it falls between two samples of the uniform pass and the
    # sampler has to find it, instead of being handed the NaN of `1/0`.
    fx, fy = evaluate.pair(parsed("[t, 1/(t - 1)]"), Context(), ("t",))
    drawn = evaluate.sample_curve(fx, fy, VIEW, VIEW, VIEW, CANVAS)
    assert len(drawn.ts) < evaluate.MAX_POINTS
    assert drawn.gave_up is not None and abs(drawn.gave_up - 1.0) < 0.01
    # And the gap is at the pole rather than anywhere along the branches.
    gapped = drawn.ts[np.isnan(drawn.xs)]
    assert gapped.size and np.abs(gapped - 1.0).max() < 0.05


def test_a_curve_faster_than_the_pixels_is_drawn_rather_than_gapped():
    # `x·SIN(x)` zoomed out is the regression: its flanks are steeper than the
    # screen, so refinement bottoms out on them, and reading that as a jump cut
    # the band into a comb of strokes. A slope loses half its height to each
    # half when it is bisected, which is what tells it from a discontinuity.
    for span in (400.0, 2000.0):
        view = (-span / 2, span / 2)
        xs, ys = sampled("x*SIN(x)", xrange=view, yrange=view, size=(900.0, 600.0))
        assert not np.isnan(ys).any(), f"x·SIN(x) over {view} is cut into pieces"
    # And the same curve as a parametric pair, which is the other sampler.
    fx, fy = evaluate.pair(parsed("[t, t*SIN(t)]"), Context(), ("t",))
    wide = (-1000.0, 1000.0)
    drawn = evaluate.sample_curve(fx, fy, wide, wide, wide, (900.0, 600.0))
    assert not np.isnan(drawn.xs).any()
    assert drawn.gave_up is None


def test_a_jump_is_still_a_jump_where_the_curve_around_it_is_steep():
    # The other half of the argument: a pole keeps its whole height in whichever
    # half of the interval it lands in, however far down the bisection goes, so
    # `TAN(x)` is gapped at every pole even though the branches leading up to it
    # are steeper than the screen and are drawn.
    xs, ys = sampled("TAN(x)")
    for pole in (-3 * np.pi / 2, -np.pi / 2, np.pi / 2, 3 * np.pi / 2):
        assert _spans(xs, ys, pole) == [], f"TAN(x) bridges the pole at {pole}"


def test_a_polar_curve_turns_its_angle_in_the_unit_it_was_written_in():
    # `r = 1` at 90 is the top of the unit circle in degrees and something
    # else entirely in radians, which is the whole reason the turn is a
    # parameter of the composition.
    one = closure("1")
    for degrees, expected in ((True, 90.0), (False, np.pi / 2)):
        horizontal, vertical = evaluate.polar_pair(one, degrees=degrees)
        assert abs(float(horizontal(np.array([expected]))[0])) < 1e-9
        assert abs(float(vertical(np.array([expected]))[0]) - 1.0) < 1e-9


def test_a_constant_matrix_becomes_the_two_columns_of_a_data_plot():
    xs, ys = evaluate.points(parsed("[[1, 2], [3, 4], [5, 6]]"), Context())
    assert list(xs) == [1.0, 3.0, 5.0]
    assert list(ys) == [2.0, 4.0, 6.0]
    # A single point is a one-row matrix and reads exactly the same way.
    xs, ys = evaluate.points(parsed("[1, 2]"), Context())
    assert (list(xs), list(ys)) == ([1.0], [2.0])


def test_an_equation_is_evaluated_as_the_difference_of_its_sides():
    # An implicit curve is the zero contour of `u - v`, so the equation has to
    # become one expression before marching squares ever sees it.
    g = evaluate.difference(parsed("x^2 + y^2 = 4"), Context(), ("x", "y"))
    assert float(g(np.array([2.0]), np.array([0.0]))[0]) == 0.0
    assert float(g(np.array([0.0]), np.array([0.0]))[0]) == -4.0


def test_an_inequality_is_a_truth_value_at_every_point():
    inside = evaluate.mask(parsed("x^2 + y^2 <= 4"), Context(), ("x", "y"))
    answer = inside(np.array([0.0, 3.0]), np.array([0.0, 0.0]))
    assert answer.dtype == bool
    assert list(answer) == [True, False]


def test_a_bound_written_as_an_expression_is_worth_a_number():
    # The toolbar fields take `-π` because that is what a person types; what it
    # is worth is arithmetic, and arithmetic happens in the process that has it.
    assert evaluate.number(parsed("-π"), Context(), 0.0) == pytest.approx(-np.pi)
    assert evaluate.number(None, Context(), 1.5) == 1.5
    # An answer with no number in it falls back rather than refusing: the
    # picture is worth more than the complaint.
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
    # The crossing carries the height of the curve it belongs to, and says
    # which of the others it was with.
    assert [item.other for item in crossings] == [0, 0, 0]
    assert crossings[1].y == pytest.approx(np.sqrt(0.5))


def test_the_pole_of_a_tangent_is_not_called_a_root():
    # `TAN(x)` changes sign across every pole, and a scan that only looked at
    # signs would report four roots that are not there. The NaN the sampler put
    # through the pole is what keeps them out.
    f = closure("TAN(x)")
    xs, ys = evaluate.sample_adaptive(f, VIEW, VIEW, CANVAS)
    found = evaluate.features(xs, ys, f)
    roots = [item.x for item in found if item.kind == evaluate.ROOT]
    assert roots == pytest.approx([-np.pi, 0.0, np.pi], abs=1e-9)
    assert not any(abs(abs(root) - np.pi / 2) < 0.1 for root in roots)


def test_a_feature_scan_over_nothing_finds_nothing():
    assert evaluate.features(np.empty(0), np.empty(0), closure("SIN(x)")) == ()


def _spans(xs, ys, place):
    """The drawn segments that cross `place`, which a discontinuity forbids.

    A segment is drawn between two adjacent samples when both are finite, so
    the ones that straddle the point and have finite ends are exactly the
    bridging strokes.
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
# Pure functions, so the whole of the interesting behaviour of a 3D window is
# tested without a graphics card: which faces survive a hole in the domain,
# where the box stands, and what the numbers along its edges say. GL rendering
# itself is only ever exercised by the offscreen host below.


@pytest.fixture(scope="module")
def solid():
    """The 3D window module, or a skip where there is no OpenGL to import it."""
    if not _toolkit():
        pytest.skip("pyqtgraph and PySide6 are not installed")
    try:
        from rederive.plot import window3d
    except ImportError as missing:  # no PyOpenGL, or no libGL to load it with
        pytest.skip(f"pyqtgraph.opengl will not import: {missing}")
    return window3d


def _grid(f, span=(-2.0, 2.0), n=33):
    """A grid of `f` over a square domain, the way the window evaluates one."""
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
    # The acceptance case: `SQRT(1-x^2-y^2)` over [-2, 2] is a dome over the
    # unit disc and nothing at all outside it. Every face has to lie inside the
    # circle, and there have to be enough of them left to be a dome.
    xs, ys, values = _grid(lambda x, y: np.sqrt(1 - x**2 - y**2))
    box = solid.Box((-2.0, 2.0), (-2.0, 2.0), (0.0, 1.0))
    vertexes, faces, _ = solid.mesh(xs, ys, values, box)
    assert faces.size
    corners = vertexes[faces.reshape(-1)]
    # Back to data coordinates, where the unit circle is the boundary. The
    # world is a square of side WORLD standing for the domain.
    scale = (2.0 - -2.0) / solid.WORLD
    radius = np.hypot(corners[:, 0] * scale, corners[:, 1] * scale)
    assert radius.max() <= 1.0
    # A quarter of the square is the disc, so about a quarter of the faces of a
    # whole grid survive; fewer than a tenth would not be a dome.
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
    assert heights.max() <= clipped.height / 2 + 1e-6


def test_the_extent_is_the_data_until_a_spike_would_crush_it(solid):
    values = np.linspace(-2.0, 2.0, 400)
    (low, high), clipped = solid.extent([values])
    assert (low, high) == (-2.0, 2.0)
    assert clipped is False
    # One pole among four hundred ordinary values: the box is the percentiles
    # and the window says so.
    spiked = values.copy()
    spiked[0] = 1e6
    (low, high), clipped = solid.extent([spiked])
    assert clipped is True
    assert high < 10.0


def test_a_flat_surface_still_gets_a_box_to_stand_in(solid):
    (low, high), clipped = solid.extent([np.full(100, 3.0)])
    assert low < 3.0 < high
    assert clipped is False


def test_the_box_keeps_the_true_height_until_it_would_be_a_tower(solid):
    # A hemisphere of radius 1 over a floor of 4 is a quarter as tall as it is
    # wide, and is drawn that way - which is what makes it read as a dome.
    dome = solid.Box((-2.0, 2.0), (-2.0, 2.0), (0.0, 1.0))
    assert dome.height == pytest.approx(solid.WORLD * 0.25)
    # A z range of five times the domain would be a tower nothing fits in.
    tower = solid.Box((-5.0, 5.0), (-5.0, 5.0), (-25.0, 25.0))
    assert tower.height == solid.WORLD
    # And a sheet is exaggerated rather than drawn as a line.
    sheet = solid.Box((-5.0, 5.0), (-5.0, 5.0), (-0.001, 0.001))
    assert sheet.height == pytest.approx(solid.WORLD * solid.MIN_HEIGHT)


def test_the_box_reports_its_center_and_its_lengths(solid):
    box = solid.Box((-2.0, 6.0), (-5.0, 5.0), (0.0, 1.0))
    assert box.center == (2.0, 0.0, 0.5)
    assert box.lengths == (8.0, 10.0, 1.0)


def test_the_tick_marks_of_an_axis_are_round_numbers(solid):
    assert solid.ticks(-5.0, 5.0) == [-4.0, -2.0, 0.0, 2.0, 4.0]
    assert solid.ticks(0.0, 1.0) == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    assert solid.ticks(3.0, 3.0) == []
    # Zero comes out as zero rather than as a rounding error of one.
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
# Section 9's geometry, still without a graphics card: the sampler bisects the
# domain's edge on the closure, the mesh fills the partial cells from the
# crossings, and the clip trims the result by interpolation alone.


def _dome(x, y):
    with np.errstate(all="ignore"):
        return np.sqrt(1 - x**2 - y**2)


def test_the_domes_rim_is_refined_onto_the_unit_circle(solid):
    # The acceptance case: at the default 64 x 64 grid, the rim of
    # `SQRT(1-x^2-y^2)` over [-2, 2] sits on the unit circle to within a
    # fraction of a cell, on the defined side, with none of the z teeth the
    # grid-aligned boundary had.
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
    # The mesh built from the crossings reaches the circle and never leaves it.
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
    # `SQRT(x)` over a domain crossing zero ends exactly on the grid line at
    # x = 0, and the refinement must leave it there: no wobble.
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
    # Two diagonally opposite defined corners: filled as two separate
    # triangles, each holding exactly one defined corner, whichever diagonal
    # the definedness runs along.
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
    # `LOG(x)` runs to minus infinity at its boundary, so the refined boundary
    # vertex is enormous; the clip trims it like any other, and the mesh ends
    # exactly on the clip plane rather than outside the box or a cell short.
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
    top = box.height / 2
    assert used[:, 2].min() >= -top - 1e-5
    assert used[:, 2].max() <= top + 1e-5
    assert used[:, 2].min() == pytest.approx(-top, abs=1e-5)
    assert np.isfinite(shading).all()


# -- the wire: the grid of the samples, on request -------------------------------
#
# Section 10's geometry, pure like the mesh's: row and column polylines built
# from the same arrays, thinned in the drawing and never in the sampling,
# ending on section 9's refined boundary and trimmed by the same clip.


def test_the_wire_is_thinned_rows_of_dense_polylines(solid):
    # At 64 x 64 the full grid would be gray fuzz, so only every k-th row and
    # column is drawn - roughly 16 to 24 each way - while every drawn line
    # keeps the full sample spacing: each segment is one grid step long.
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
        # The first and last sample rows are always among the lines drawn.
        assert np.isclose(drawn.min(), -solid.HALF)
        assert np.isclose(drawn.max(), solid.HALF)
        # Dense polylines: every segment spans exactly one grid step.
        along = lines[:, 1, 1 - axis] - lines[:, 0, 1 - axis]
        assert np.abs(along) == pytest.approx(np.full(len(along), step), rel=1e-4)
    assert 0.0 < shades.min() <= shades.max() <= 1.0


def test_the_wires_rim_ends_on_the_refined_boundary(solid):
    # The acceptance case: the dome in wire has a clean circular rim, the wire
    # ending on the unit circle rather than a grid step short of it, and no
    # segment reaching past it - holes appear in the wire exactly where they
    # appear in the solid.
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
    # The clip trims the wire by interpolation as it trims the faces - a
    # boundary end whose limit is unbounded included - so the wire ends on the
    # clip plane rather than outside the box or a sample short of it.
    def logged(x, y):
        with np.errstate(all="ignore"):
            return np.log(x) + 0.0 * y

    xs, ys, values = _grid(logged, n=64)
    boundary = evaluate.grid_boundary(logged, xs, ys, values)
    box = solid.Box((-2.0, 2.0), (-2.0, 2.0), (-2.0, 1.0))
    points, _ = solid.wire(xs, ys, values, box, boundary)
    top = box.height / 2
    assert points[:, 2].min() >= -top - 1e-5
    assert points[:, 2].max() <= top + 1e-5
    assert points[:, 2].min() == pytest.approx(-top, abs=1e-5)


# -- window titles ---------------------------------------------------------------
#
# A window is titled by what it holds; the number that keys the protocol is no
# part of the name. Pure functions of the contents, so no toolkit is needed.


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
    """Whether the toolkit the host needs is installed, without importing it here."""
    from importlib.util import find_spec

    return find_spec("pyqtgraph") is not None and find_spec("PySide6") is not None


@pytest.fixture
def host(monkeypatch):
    """A plot host in a child process, drawing into nothing.

    Offscreen, so that a test suite on a machine with a display does not have
    windows flashing over it, and so that one without a display runs the test
    at all. Every wait has a deadline and the host is always shut down, so a
    host that will not start fails the test instead of hanging the suite.
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


def _add(session, host, text, **keywords):
    """Plot one expression the way the Plot command does, and say where it went."""
    entry = session.author(text)
    label = f"#{entry.number}"
    plotted = classify(entry.node, session.variables(label), text)
    fields = {
        "worksheet": id(session),
        "node": entry.node,
        "context": session.context,
        "kind": plotted.kind,
        "label": label,
        "text": text,
        "options": plotted.options,
    }
    return host.add(plots.Add(**{**fields, **keywords})).window


def test_a_host_takes_a_plot_and_describes_what_it_holds(host):
    session = Session()
    assert _add(session, host, "SIN(x)") == 1
    assert _add(session, host, "x^2 - 3") == 1
    described = host.describe()
    assert len(described) == 1
    window = described[0]
    assert window.number == 1
    assert window.kind is plots.WindowKind.TWO_D
    assert window.title == "SIN(x), x^2 - 3 - Rederive plot (current)"
    assert window.current is True
    assert [plot.label for plot in window.plots] == ["#1", "#2"]
    assert [plot.text for plot in window.plots] == ["SIN(x)", "x^2 - 3"]


def test_re_plotting_a_label_replaces_its_curve_and_says_so(host):
    session = Session()
    _add(session, host, "SIN(x)")
    entry = session.entries[0]
    placed = host.add(
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
    # The reply says the plot replaced a curve rather than adding one, which
    # is the word the acknowledgement message turns on.
    assert placed == plots.Placed(1, replaced=True)
    assert len(host.describe()[0].plots) == 1


def test_a_new_window_takes_the_next_number_and_becomes_current(host):
    session = Session()
    assert _add(session, host, "SIN(x)") == 1
    assert _add(session, host, "COS(x)", window=plots.Where.NEW) == 2
    described = host.describe()
    assert [window.number for window in described] == [1, 2]
    assert [window.current for window in described] == [False, True]
    assert described[0].title == "SIN(x) - Rederive plot"
    assert described[1].title == "COS(x) - Rederive plot (current)"
    # The receiver is where the next plot lands: the new window's arrival was
    # the last touch.
    assert _add(session, host, "TAN(x)") == 2


def test_a_family_becomes_one_curve_per_element(host):
    session = Session()
    entry = session.author("[x, x^2, x^3]")
    host.add(
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


def test_one_window_takes_every_two_dimensional_kind(host):
    # One window and one plot list for all of them, which is the design's own
    # claim: a parametric pair, a rose, a matrix of points, a contour and a
    # shaded region are one picture with one legend. Sent in a single test
    # because each of these costs a process to start.
    session = Session()
    _add(session, host, "[SIN(t), COS(t)]", options=plots.Options(variables=("t",)))
    _add(session, host, "[[1, 2], [3, 4]]")
    _add(session, host, "x^2 + y^2 = 4")
    _add(session, host, "y < x^2")
    # Polar is the view's mode rather than the plot's: a request still
    # spelling the polar kind lands as the curve this window reads it as.
    _add(
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


def test_a_surface_opens_a_solid_window_of_its_own(host):
    # One window per kind: a curve and a surface never share a window, and each
    # is the current one for its own kind, so the next plot of either lands
    # where the last one did. Sent in one test because each costs a process.
    session = Session()
    assert _add(session, host, "SIN(x)") == 1
    assert _add(session, host, "x^2 - y^2") == 2
    flat, solid = host.describe()
    assert (flat.kind, solid.kind) == (
        plots.WindowKind.TWO_D,
        plots.WindowKind.THREE_D,
    )
    assert solid.title == "x^2 - y^2 - Rederive 3D plot (current)"
    assert flat.title == "SIN(x) - Rederive plot (current)"
    assert [plot.kind for plot in solid.plots] == [PlotKind.SURFACE]
    # The window reports the domain it evaluates over, which is the default one.
    assert solid.xrange == (-5.0, 5.0)
    assert solid.yrange == (-5.0, 5.0)
    assert _add(session, host, "SIN(x*y)") == 2
    assert _add(session, host, "COS(x)") == 1


def test_a_vector_of_surfaces_becomes_one_surface_per_element(host):
    session = Session()
    entry = session.author("[x + y, x - y]")
    host.add(
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


def test_a_curve_that_will_not_evaluate_reports_itself(host):
    session = Session()
    # A window with no host of its own to draw in reports over the pipe rather
    # than drawing nothing; the event is the only thing that says so.
    _add(session, host, "SIN(x)")
    assert host.describe()[0].plots[0].label == "#1"


def test_the_host_is_started_once_and_stopped_when_it_is_asked_to(host):
    session = Session()
    _add(session, host, "SIN(x)")
    _add(session, host, "COS(x)")
    assert host.starts == 1
    assert host.running
    host.shutdown()
    assert not host.running


def test_describing_with_no_host_answers_nothing_without_starting_one():
    from rederive.plot import proxy as proxy_module

    proxy = proxy_module.PlotProxy()
    assert proxy.describe() == ()
    assert proxy.starts == 0


def test_a_host_that_will_not_start_reports_its_own_words(monkeypatch):
    # A missing wheel and a display Qt cannot open are the two ordinary ways a
    # host fails to start, and both are worth reading; the death of a process
    # is what they must not be reported as. Nothing is spawned here: what is
    # under test is what the proxy makes of the handshake it gets back.
    from rederive.plot import proxy as proxy_module

    proxy = proxy_module.PlotProxy()

    def refusing():
        proxy._replies[plots.READY] = plots.Refused("no module named pyqtgraph")

    monkeypatch.setattr(proxy, "_spawn", refusing)
    with pytest.raises(PlotError) as refused:
        proxy.add(
            plots.Add(
                worksheet=0,
                node=parsed("SIN(x)"),
                context=Context(),
                kind=PlotKind.CURVE,
            )
        )
    assert str(refused.value) == "no module named pyqtgraph"


def test_a_host_takes_the_preferences_before_the_plot_that_follows(host):
    """The new request over a real pipe, in front of a plot that still lands.

    What a preference does to a picture is a thing to be looked at rather than
    asserted about - `Describe` reports what is in a window and not what it was
    built with - so this is the round trip: the host understands the request,
    keeps whatever it says, and the next plot goes where it would have gone.
    """
    session = Session()
    host.prefer(plots.Prefer(grid=16, connected=True, point_size=9))
    assert _add(session, host, "SIN(x)") == 1
    assert [plot.label for plot in host.describe()[0].plots] == ["#1"]


# -- the windows' own fields ----------------------------------------------------
#
# The 2D range fields and the 3D domain fields are exercised in-process,
# offscreen, with a host whose sampling thread is the caller: what is under
# test is the wiring from a typed expression to a re-sampled plot, and the
# pipe is already crossed by the host tests above.


class InlineHost:
    """A host that runs each sampling job before `sample` returns."""

    def __init__(self):
        #: The points windows have sent home, as (worksheet, text) pairs.
        self.authored = []
        #: The sticky preference values windows have handed back, merged.
        self.adjustments = {}

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

    def adjusted(self, **values):
        self.adjustments.update(values)


@pytest.fixture(scope="module")
def qt():
    """A Qt application in this process, offscreen, or a skip without one."""
    if not _toolkit():
        pytest.skip("pyqtgraph and PySide6 are not installed")
    import os

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    import pyqtgraph as pg

    return pg.mkQApp()


@pytest.fixture
def flat(qt):
    from rederive.plot.window2d import Window2D

    window = Window2D(1, InlineHost())
    yield window
    window.close()


def _plot(text, kind, variables, label="#1"):
    from rederive.plot.window2d import Plot

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


def test_a_parametric_plot_draws_at_once_over_one_turn(flat):
    plot = _plot("[SIN(t), COS(t)]", PlotKind.PARAMETRIC, ("t",))
    flat.add(plot)
    # The picture is there with no question asked anywhere on the way.
    assert plot.trange == pytest.approx((-np.pi, np.pi))
    assert np.isfinite(plot.ys).any()
    # And the range that drew it is on the toolbar, named by its parameter.
    assert all(action.isVisible() for action in flat._range_actions)
    assert flat.range_name.text().strip() == "t:"
    assert (flat.range_low.text(), flat.range_high.text()) == ("-3.14159", "3.14159")


def test_the_range_fields_read_expressions_and_resample_the_plot(flat):
    plot = _plot("[SIN(t), COS(t)]", PlotKind.PARAMETRIC, ("t",))
    flat.add(plot)
    flat.range_low.setText("0")
    flat.range_high.setText("2π")
    flat._range_edited()
    assert plot.trange == pytest.approx((0.0, 2 * np.pi))
    assert plot.bounds is None
    assert float(plot.ts[0]) == pytest.approx(0.0)
    # Text that does not parse reverts to the range on the screen.
    flat.range_high.setText("2*")
    flat._range_edited()
    assert flat.range_high.text() == "6.28319"
    assert plot.trange == pytest.approx((0.0, 2 * np.pi))
    # A bound that parses but is worth no number falls back the same way.
    flat.range_low.setText("1/0")
    flat._range_edited()
    assert plot.trange == pytest.approx((0.0, 2 * np.pi))


def test_a_window_of_functions_offers_no_range_fields(flat):
    plot = _plot("SIN(x)", PlotKind.CURVE, ("x",))
    flat.add(plot)
    assert not any(action.isVisible() for action in flat._range_actions)


def test_the_polar_toggle_rereads_curves_and_restores_them(flat):
    plot = _plot("SIN(x)", PlotKind.CURVE, ("x",))
    flat.add(plot)
    flat.polar_toggle.trigger()
    # The curve is now r = f(θ) over one full turn - the x-range it was viewed
    # at is not a θ range - with the range fields named by the angle, and the
    # readout in the coordinates the picture is measured in.
    assert plot.kind is PlotKind.POLAR
    assert plot.trange == pytest.approx((-np.pi, np.pi))
    assert np.isfinite(plot.ys).any()
    assert float(np.nanmax(np.abs(plot.xs))) < 1.5
    assert flat.range_name.text().strip() == "θ:"
    assert "r:" in flat._readout(1.0, 0.0)
    # Flipping back restores the cartesian reading, sampled over the view.
    flat.polar_toggle.trigger()
    assert plot.kind is PlotKind.CURVE
    assert float(plot.xs.min()) < -4 and float(plot.xs.max()) > 4
    assert not any(action.isVisible() for action in flat._range_actions)
    assert "r:" not in flat._readout(1.0, 0.0)


def test_the_kinds_with_no_polar_reading_ignore_the_toggle(flat):
    pair = _plot("[SIN(t), COS(t)]", PlotKind.PARAMETRIC, ("t",))
    points = _plot("[[1, 2], [3, 4]]", PlotKind.DATA, (), label="#2")
    flat.add(pair)
    flat.add(points)
    before = pair.trange
    flat.polar_toggle.trigger()
    assert pair.kind is PlotKind.PARAMETRIC
    assert points.kind is PlotKind.DATA
    assert pair.trange == before


def test_a_curve_added_to_a_polar_window_is_read_polar_from_the_start(flat):
    flat.polar_toggle.trigger()
    plot = _plot("2*COS(3*t)", PlotKind.CURVE, ("t",))
    flat.add(plot)
    assert plot.kind is PlotKind.POLAR
    assert plot.trange == pytest.approx((-np.pi, np.pi))
    assert flat.range_name.text().strip() == "θ:"


def test_the_range_fields_adjust_a_reread_curve_like_a_born_polar_one(flat):
    plot = _plot("SIN(x)", PlotKind.CURVE, ("x",))
    flat.add(plot)
    flat.polar_toggle.trigger()
    flat.range_low.setText("0")
    flat.range_high.setText("2π")
    flat._range_edited()
    assert plot.trange == pytest.approx((0.0, 2 * np.pi))
    # And a reread over one turn is the default again the next time the
    # toggle comes on, the adjusted range having been the polar reading's.
    flat.polar_toggle.trigger()
    flat.polar_toggle.trigger()
    assert plot.trange == pytest.approx((-np.pi, np.pi))


def test_the_clear_button_empties_its_own_window(flat):
    # The toolbar's clear acts on the window it is drawn in - there is nothing
    # to infer and nothing to report - and the range fields go away with the
    # parametrized plot that owned them.
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
    # Enter with no marker up is not the send key: nothing goes anywhere.
    _pressed_return(flat)
    assert flat.host.authored == []
    flat.trace()
    # Tab snaps to the next feature to the right of center - the maximum at
    # π/2 - refined on the closure, and Enter sends that number home.
    flat.snap(False)
    _pressed_return(flat)
    assert flat.host.authored == [(1, "[1.570796, 1.000000]")]
    # The window says so where the user is looking.
    assert flat.status.text() == "Sent [1.570796, 1.000000] to the worksheet"


def test_the_point_sent_home_is_the_text_ctrl_c_copies(flat, qt):
    flat.add(_plot("SIN(x)", PlotKind.CURVE, ("x",)))
    flat.trace()
    flat.snap(False)
    flat.copy_image()
    copied = qt.clipboard().text()
    flat.send_home()
    assert copied.startswith("[")
    assert flat.host.authored == [(1, copied)]


def test_the_title_tracks_the_plot_list(flat):
    # The title bar names what the window holds, so it follows every change to
    # the plot list: add, replace in place, remove, clear.
    assert flat.windowTitle() == "Rederive plot"
    flat.add(_plot("SIN(x)", PlotKind.CURVE, ("x",)))
    assert flat.windowTitle() == "SIN(x) - Rederive plot"
    flat.add(_plot("COS(x)", PlotKind.CURVE, ("x",), label="#2"))
    assert flat.windowTitle() == "SIN(x), COS(x) - Rederive plot"
    # Re-plotting a label replaces its curve, and the title reads the new text.
    flat.add(_plot("x^2 - 3", PlotKind.CURVE, ("x",)))
    assert flat.windowTitle() == "COS(x), x^2 - 3 - Rederive plot"
    flat.remove(flat.plots[0])
    assert flat.windowTitle() == "x^2 - 3 - Rederive plot"
    flat.clear_action.trigger()
    assert flat.windowTitle() == "Rederive plot"


def test_a_data_plots_toggles_hand_the_sticky_values_back(flat):
    # The reversal section 7 asks for: the right-click controls write back, so
    # the way this plot is left is the way the next data plot arrives.
    plot = _plot("[[1, 2], [3, 4]]", PlotKind.DATA, ())
    flat.add(plot)
    flat._pointed = plot
    flat._toggle_connected()
    assert flat.host.adjustments == {"connected": True}
    flat._set_point_size(8.0)
    assert flat.host.adjustments == {"connected": True, "point_size": 8.0}
    flat._toggle_connected()
    assert flat.host.adjustments["connected"] is False


def test_the_framing_lock_hands_nothing_back(flat):
    # Equal scales is deliberately not sticky: releasing it is this window's
    # exception, and a one-off framing choice must not silently become the
    # default that reshapes the next circle.
    flat.equal.trigger()
    assert not flat.equal.isChecked()
    assert flat.host.adjustments == {}


def test_the_axis_label_follows_the_polar_mode(flat):
    # The known wart, fixed: a window in polar mode must not label its
    # abscissa with the parameter's letter - the horizontal axis of a polar
    # picture is not θ.
    plot = _plot("SIN(t)", PlotKind.CURVE, ("t",))
    flat.add(plot)
    axis = flat.item.getAxis("bottom")
    assert axis.labelText == "t"
    flat.polar_toggle.trigger()
    assert axis.labelText == ""
    flat.polar_toggle.trigger()
    assert axis.labelText == "t"


# -- the stroke weight: curves out-weigh the furniture ---------------------------


def test_a_curve_out_weighs_the_axes_and_the_grid(flat):
    from rederive.plot.window2d import CURVE_WIDTH

    plot = _plot("2*x + 3", PlotKind.CURVE, ("x",))
    flat.add(plot)
    pen = plot.item.opts["pen"]
    # Two logical pixels - the screen's density multiplied in - and cosmetic,
    # so zooming the view never fattens or thins the stroke.
    assert pen.widthF() == CURVE_WIDTH * flat.devicePixelRatioF()
    assert pen.isCosmetic()
    # The furniture keeps its hairlines: the origin axes locate the curve, and
    # the curve plainly out-weighs them.
    for line in flat.axes:
        assert line.pen.widthF() == 1.0
    # The legend's sample is a stroke of the same weight in the same color, so
    # the swatch on the card shows what the canvas does.
    assert flat.legend.entries == ((plot.named, plot.color, False),)


def test_a_data_plots_line_takes_the_weight_and_its_points_do_not(flat):
    from rederive.plot.window2d import CURVE_WIDTH

    plot = _plot("[[1, 2], [3, 4]]", PlotKind.DATA, ())
    flat.add(plot)
    flat._pointed = plot
    flat._toggle_connected()
    pen = plot.item.opts["pen"]
    assert pen.widthF() == CURVE_WIDTH * flat.devicePixelRatioF()
    assert pen.isCosmetic()
    # The point size is its own setting: the symbol outline stays at one, so
    # fattening the connecting line does not fatten every point with it.
    assert plot.item.opts["symbolPen"].widthF() == 1.0


def test_the_export_pens_carry_the_same_weight(flat):
    from rederive.plot.window2d import CURVE_WIDTH, _on_paper

    plot = _plot("SIN(x)", PlotKind.CURVE, ("x",))
    flat.add(plot)
    with _on_paper(flat):
        # The exporters render the scene with no high-DPI scale, so the paper
        # pens take the plain constant: the file carries the weight a 1x
        # screen shows.
        pen = plot.item.opts["pen"]
        assert pen.widthF() == CURVE_WIDTH
        assert pen.isCosmetic()
    assert plot.item.opts["pen"].widthF() == CURVE_WIDTH * flat.devicePixelRatioF()


@pytest.fixture
def deep(qt, solid):
    window = solid.Window3D(1, InlineHost())
    yield window
    window.close()


def test_the_domain_fields_read_expressions(deep, solid):
    surface = solid.Surface(
        worksheet=1,
        label="#1",
        text="x*y",
        kind=PlotKind.SURFACE,
        node=parsed("x*y"),
        context=Context(),
        options=plots.Options(variables=("x", "y")),
        state=ParseState(),
    )
    deep.add(surface)
    deep.fields["x0"].setText("-π")
    deep.fields["x1"].setText("π")
    deep._edited()
    assert deep.xdomain == pytest.approx((-np.pi, np.pi))
    assert deep.fields["x0"].text() == "-3.14159"
    # A field that does not parse reverts, the domain untouched.
    deep.fields["y0"].setText("*")
    deep._edited()
    assert deep.ydomain == (-5.0, 5.0)
    assert deep.fields["y0"].text() == "-5"
    # An inverted domain is refused in words and reverted the same way.
    deep.fields["y0"].setText("10")
    deep._edited()
    assert deep.ydomain == (-5.0, 5.0)
    assert deep.fields["y0"].text() == "-5"


def test_an_edited_grid_hands_the_new_count_back_to_the_host(deep, solid):
    deep.fields["nx"].setText("32")
    deep.fields["ny"].setText("32")
    deep._edited()
    assert deep.grid == (32, 32)
    assert deep.host.adjustments == {"grid": 32}
    # The sticky value is one count per axis, so a rectangular grid hands on
    # its finer axis; the domain fields are a framing and hand back nothing.
    deep.fields["ny"].setText("48")
    deep._edited()
    assert deep.host.adjustments == {"grid": 48}
    deep.fields["x0"].setText("-2")
    deep._edited()
    assert deep.host.adjustments == {"grid": 48}


def test_a_surfaces_boundary_arrives_beside_its_arrays(deep, solid):
    # The refinement runs on the sampling thread with `grid_eval` and is cached
    # on the surface, where the mesh - and section 10's wire - can read it.
    surface = solid.Surface(
        worksheet=1,
        label="#1",
        text="SQRT(1-x^2-y^2)",
        kind=PlotKind.SURFACE,
        node=parsed("SQRT(1-x^2-y^2)"),
        context=Context(),
        options=plots.Options(variables=("x", "y")),
        state=ParseState(),
    )
    deep.add(surface)
    assert isinstance(surface.boundary, evaluate.Boundary)
    assert np.isfinite(surface.boundary.across).any()
    assert np.isfinite(surface.boundary.along).any()


def test_an_all_nan_surface_still_says_no_real_values(deep, solid):
    surface = solid.Surface(
        worksheet=1,
        label="#5",
        text="SQRT(-1-x^2-y^2)",
        kind=PlotKind.SURFACE,
        node=parsed("SQRT(-1-x^2-y^2)"),
        context=Context(),
        options=plots.Options(variables=("x", "y")),
        state=ParseState(),
    )
    deep.add(surface)
    assert deep.status.text() == "#5: no real values over this domain"


def test_the_3d_clear_button_empties_its_own_window(deep, solid):
    surface = solid.Surface(
        worksheet=1,
        label="#1",
        text="x*y",
        kind=PlotKind.SURFACE,
        node=parsed("x*y"),
        context=Context(),
        options=plots.Options(variables=("x", "y")),
        state=ParseState(),
    )
    deep.add(surface)
    assert deep.windowTitle() == "x*y - Rederive 3D plot"
    deep.clear_action.trigger()
    assert deep.plots == []
    # A cleared window's title falls back to the empty-window one.
    assert deep.windowTitle() == "Rederive 3D plot"


def _surface(solid, text, label="#1"):
    return solid.Surface(
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


def test_the_mesh_box_and_the_m_key_flip_every_surface(deep, solid):
    # The toolbar box flips every surface in the window to wire and back, M is
    # its key, and the look it leaves is handed back as the sticky value.
    one = _surface(solid, "x*y")
    two = _surface(solid, "x+y", label="#2")
    deep.add(one)
    deep.add(two)
    assert not one.wire and not two.wire
    deep.mesh_action.trigger()
    assert one.wire and two.wire
    assert deep.host.adjustments == {"wire": True}
    # The solid item steps aside and the wire draws in its place.
    assert not one.item.visible() and one.wires.visible()
    _pressed_m(deep)
    assert not one.wire and not two.wire
    assert deep.host.adjustments == {"wire": False}
    assert one.item.visible() and not one.wires.visible()


def test_the_wire_draws_at_the_curves_weight(deep, solid):
    from rederive.plot.window2d import CURVE_WIDTH

    # The one constant of the 2D window's strokes reaches the wire too, so a
    # wire surface carries the weight a curve does.
    surface = _surface(solid, "x*y")
    deep.add(surface)
    assert surface.wires.width == CURVE_WIDTH


def test_the_legend_override_moves_one_surface_and_nothing_sticky(deep, solid):
    # The two-level shape a data plot's points already use: the per-surface
    # right-click is the exception, so one surface goes wire while the other
    # stays solid, and no sticky value is handed back.
    one = _surface(solid, "x*y")
    two = _surface(solid, "x+y", label="#2")
    deep.add(one)
    deep.add(two)
    deep.toggle_wire(one)
    assert one.wire and not two.wire
    assert one.wires.visible() and two.item.visible()
    assert deep.host.adjustments == {}
    deep.toggle_wire(one)
    assert not one.wire
    assert deep.host.adjustments == {}


def test_a_surface_arrives_in_the_look_the_window_was_left_in(qt, solid):
    # Sticky in the sense of section 7: a window built with the wire look
    # opens with the box checked and gives it to every surface that arrives,
    # while a replacement keeps the look of the surface it replaces.
    window = solid.Window3D(1, InlineHost(), wire=True)
    try:
        assert window.mesh_action.isChecked()
        one = _surface(solid, "x*y")
        window.add(one)
        assert one.wire
        window.toggle_wire(one)
        replaced = _surface(solid, "x^2-y^2")
        window.add(replaced)
        assert not replaced.wire
    finally:
        window.close()


def test_the_wire_darkens_for_export_like_every_other_color(deep, solid):
    surface = _surface(solid, "x*y")
    deep.add(surface)
    deep.toggle_wire(surface)
    points, shades = solid.wire(
        surface.xs, surface.ys, surface.values, deep.box_now, surface.boundary
    )
    deep._papered = True
    deep._draw(surface)
    assert np.allclose(surface.wires.color, solid.brightened(shades, surface.paper))
    deep._papered = False
    deep._draw(surface)
    assert np.allclose(surface.wires.color, solid.brightened(shades, surface.color))


# -- the receiver ---------------------------------------------------------------
#
# One pointer, and it is the window the user last touched: the host learns it
# from the windows' own activation events. Exercised on a real Host built in
# this process, offscreen, because activation is a conversation between the
# window and the registry - the pipe carries none of it, and a child process's
# windows cannot be touched from a test.


@pytest.fixture
def registry(qt):
    """A real Host in this process, windows and receiver bookkeeping included.

    The pipe goes nowhere: what is under test is which window an `Add` lands
    in and how the receiver follows the user's touch, and the replies come
    back as return values rather than up a pipe.
    """
    import multiprocessing

    from rederive.plot.host import Host

    ours, theirs = multiprocessing.Pipe()
    host = Host(theirs)
    #: The app's end of the pipe, for a test that reads events off it.
    host.appside = ours
    yield host
    for window in list(host.windows.values()):
        window.close()
    ours.close()
    theirs.close()


def _landing(text, label, **keywords):
    """One curve the way the app would send it, for the registry to place."""
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
    placed = registry._add(_landing("SIN(x)", "#1"))
    assert placed == plots.Placed(1)
    window = registry.windows[1]
    assert window.current
    assert window.windowTitle() == "SIN(x) - Rederive plot (current)"


def test_touching_a_window_makes_it_the_receiver(registry):
    one = registry._add(_landing("SIN(x)", "#1")).window
    two = registry._add(_landing("COS(x)", "#2", window=plots.Where.NEW)).window
    assert registry.windows[two].current and not registry.windows[one].current
    registry.touched(one)
    assert registry.windows[one].current and not registry.windows[two].current
    assert registry.windows[one].windowTitle().endswith("(current)")
    # The next plot follows the touch.
    assert registry._add(_landing("TAN(x)", "#3")).window == one


def test_the_activation_event_is_what_feeds_the_receiver(registry, qt):
    # The wiring itself: activating the window - what a click, a raise or an
    # alt-tab comes to - reaches the registry with no request on the way.
    one = registry._add(_landing("SIN(x)", "#1")).window
    registry._add(_landing("COS(x)", "#2", window=plots.Where.NEW))
    registry.windows[one].activateWindow()
    qt.processEvents()
    assert registry.windows[one].current
    assert registry._add(_landing("TAN(x)", "#3")).window == one


def test_closing_the_receiver_hands_it_to_the_last_activated_survivor(registry):
    one = registry._add(_landing("SIN(x)", "#1")).window
    two = registry._add(_landing("COS(x)", "#2", window=plots.Where.NEW)).window
    three = registry._add(_landing("TAN(x)", "#3", window=plots.Where.NEW)).window
    registry.touched(one)
    registry.touched(three)
    registry.windows[three].close()
    # The last-activated survivor of the kind, not the newest window.
    assert registry.windows[one].current and not registry.windows[two].current
    assert registry._add(_landing("x^2", "#4")).window == one


# -- the sticky preferences in the registry --------------------------------------
#
# The write-back half of section 7, on the same in-process Host: a control
# moved in a window updates the preferences the next plot is built with, and
# the change goes up the pipe so the app can outlive this host with it.


def _points(text, label, **keywords):
    """One data plot the way the app would send it, with no opinion of its own."""
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
    window = registry.windows[registry._add(_points("[[1, 2], [3, 4]]", "#1")).window]
    plot = window.plots[0]
    assert plot.connected is False
    window._pointed = plot
    window._toggle_connected()
    window._set_point_size(8.0)
    # The host keeps the values the next plot is filled from...
    assert registry.preferences == plots.Prefer(connected=True, point_size=8.0)
    # ...and the app is told, so the values survive this host.
    assert registry.appside.poll(1.0)
    number, event = registry.appside.recv()
    assert number == plots.EVENT
    assert event == plots.Preferred(plots.Prefer(connected=True))
    # A data plot with no opinion of its own follows suit, in a fresh window
    # as in this one.
    fresh = registry._add(_points("[[5, 6], [7, 8]]", "#2", window=plots.Where.NEW))
    arrived = registry.windows[fresh.window].plots[0]
    assert arrived.connected is True
    assert arrived.point_size == 8.0


def test_a_released_framing_lock_is_this_windows_alone(registry):
    one = registry._add(_landing("SIN(x)", "#1")).window
    registry.windows[one].equal.trigger()
    assert not registry.windows[one].equal.isChecked()
    # Nothing was kept and nothing was reported: the next window opens with
    # equal scales, as every window does.
    assert registry.preferences == plots.Prefer()
    assert not registry.appside.poll(0.05)
    two = registry._add(_landing("COS(x)", "#2", window=plots.Where.NEW)).window
    assert registry.windows[two].equal.isChecked()


def test_the_grid_handed_back_is_the_next_surface_windows_grid(registry, solid):
    registry.adjusted(grid=32)
    window = registry._target(plots.Where.NEW, plots.WindowKind.THREE_D)
    assert window.grid == (32, 32)


def test_the_wire_handed_back_is_the_next_surface_windows_look(registry, solid):
    # The round trip of section 10's stickiness: the mesh box moved somewhere,
    # the host kept the value and told the app, and the next 3D window opens
    # with the box checked - so the next surface arrives as wire.
    registry.adjusted(wire=True)
    assert registry.appside.poll(1.0)
    _, event = registry.appside.recv()
    assert event == plots.Preferred(plots.Prefer(wire=True))
    window = registry._target(plots.Where.NEW, plots.WindowKind.THREE_D)
    assert window.wired
    assert window.mesh_action.isChecked()


# -- the gallery ---------------------------------------------------------------


def test_every_line_of_the_gallery_is_a_captioned_plot():
    """The shipped worksheet, read as the Plot command reads it.

    A gallery whose lines have stopped classifying as what their captions say
    they are is worse than no gallery, so every line goes through the same
    classification the command does. `VECTOR(...)` is the one line that is not a
    plot as it stands - its caption says to simplify it into a matrix first -
    and it is here as the reading it has before that, a constant.
    """
    from rederive.model import worksheet

    session = Session()
    assert session.load(worksheet.reading("gallery")) == 0
    kinds = []
    for entry in session.entries:
        assert entry.annotation, entry.text
        plotted = classify(entry.node, session.variables(f"#{entry.number}"), entry.text)
        kinds.append(plotted.kind)
    assert set(kinds) >= {
        PlotKind.CURVE,
        PlotKind.FAMILY,
        PlotKind.PARAMETRIC,
        PlotKind.IMPLICIT,
        PlotKind.REGION,
        PlotKind.SURFACE,
    }


# -- the preferences -----------------------------------------------------------
#
# The sticky plot values live in the settings store, screenless, and they have
# three things to do: translate into the request the host understands, reach a
# host that may not be running yet, and be the default a plot with no opinion
# of its own is drawn with. All three are pure and none of them needs a window.


def test_the_preferences_translate_the_words_the_settings_hold():
    from rederive.model.plotting import preferences
    from rederive.model.settings import Settings

    settings = Settings()
    # The settings' defaults and the request's defaults are the same picture,
    # which is what makes a session that never touched a sticky control behave
    # like one that touched it and put it back.
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
    # The two translations are inverses, so what a toggle moved is exactly
    # what the next host is handed.
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
    # The resolved open question: a new window always opens with equal scales
    # and the 1:1 toggle serves the exception, so there is no setting to hold
    # it and no line of a state file that carries it.
    from rederive.model import state
    from rederive.model.settings import DEFAULTS, Settings

    assert "PlotScales" not in DEFAULTS
    assert "PlotScales" not in state.write(Settings())


def test_a_plot_with_no_opinion_is_drawn_the_way_the_preferences_say():
    from rederive.plot.host import preferred

    request = plots.Add(
        worksheet=0,
        node=parsed("[[1, 2], [3, 4]]"),
        context=Context(),
        kind=PlotKind.DATA,
    )
    filled = preferred(request, plots.Prefer(connected=True, point_size=12.0))
    assert (filled.options.connected, filled.options.point_size) == (True, 12.0)


def test_a_plot_that_has_an_opinion_keeps_it():
    """Which is what makes a point size chosen in the window survive a replot."""
    from rederive.plot.host import preferred

    request = plots.Add(
        worksheet=0,
        node=parsed("[[1, 2], [3, 4]]"),
        context=Context(),
        kind=PlotKind.DATA,
        options=plots.Options(connected=False, point_size=3.0),
    )
    assert preferred(request, plots.Prefer(connected=True, point_size=12.0)) is request


def test_the_preferences_go_in_front_of_the_next_request_and_only_once(monkeypatch):
    """Nothing is sent when a preference changes; it travels with the next plot.

    Which is what lets the settings watcher call `prefer` straight from the
    event loop, and what means a session that changes a preference and never
    plots again starts no process over it.
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
    proxy.add(request)
    assert sent == [plots.Prefer(grid=16), request]
    # And a second plot does not say it again.
    proxy.add(request)
    assert sent[2:] == [request]
    # Nor does one that has heard the same preferences twice.
    proxy.prefer(plots.Prefer(grid=16))
    proxy.add(request)
    assert sent[3:] == [request]


def test_a_restarted_host_hears_the_preferences_again(monkeypatch):
    """A fresh host knows only the defaults, so the sticky values go in front
    of its first plot however many hosts heard them before."""
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
    proxy.add(request)
    assert sent == [plots.Prefer(connected=True), request]
    # The host dies; the next plot starts another, and the preferences it was
    # never told go in front of that plot too.
    proxy._process = None
    proxy.add(request)
    assert proxy.starts == 2
    assert sent[2:] == [plots.Prefer(connected=True), request]


# -- the Plot command in the algebra window ------------------------------------
#
# The app is driven with a proxy that answers instead of a host, because what
# is under test is the wiring: which request a keystroke sends, and what the
# message line says about the answer. The host has its own tests above, and
# starting one per keystroke would put a Qt toolkit behind every one of these.


class Answering:
    """A plot proxy that records what it was asked and answers plausibly."""

    def __init__(self, windows=(), refuse=None):
        self.sent = []
        self.windows = windows
        self.refuse = refuse
        self.events = None
        #: What `add` answers, which a test overrides to say a plot replaced
        #: a curve rather than adding one.
        self.placed = plots.Placed(1)
        #: How many times the app has asked what the windows hold, which a
        #: plain plot must never do: nothing app-side needs a window's state
        #: to send one.
        self.described = 0
        #: The preferences it has been handed, newest last. The real proxy holds
        #: one and sends it in front of the next request; what the app is
        #: responsible for is handing it over at all.
        self.preferences = []

    def _answer(self, request):
        self.sent.append(request)
        if self.refuse is not None:
            raise PlotError(self.refuse)

    def add(self, request):
        self._answer(request)
        return self.placed

    def describe(self):
        self.described += 1
        return self.windows

    def prefer(self, preferences):
        self.preferences.append(preferences)

    def shutdown(self):
        pass


@pytest.fixture
def app():
    from rederive.ui.app import RederiveApp

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
        # No window named: the plot goes to the receiver of its kind.
        assert request.window is None
        assert request.options.variables == ("x",)
        # The command menu is back, the submenu being finished with.
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
        # No field line: a surface is drawn over a domain the window owns, and
        # the command asks nothing at all.
        assert band(app)[0].startswith(" COMMAND:")


async def test_a_vector_of_surfaces_carries_the_text_of_every_element(app):
    async with app.run_test() as pilot:
        await authored(pilot, app, "[x + y, x - y]")
        await pilot.press("p", "p")
        request = app.plots.sent[0]
        assert request.kind is PlotKind.SURFACES
        assert request.options.texts == ("x + y", "x - y")


async def test_a_kind_no_window_draws_yet_is_refused_before_it_is_sent(app, monkeypatch):
    # Every kind the vocabulary has is drawn today, so the refusal is exercised
    # against a kind taken out of the drawn set: it is the machinery that will
    # name the next kind classification learns before a window can draw it.
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
        # No field line: the picture appears over one turn, and the range is
        # adjusted in the plot window's own toolbar.
        assert message(app) == "Plotting #1"
        request = app.plots.sent[0]
        assert request.kind is PlotKind.PARAMETRIC
        assert request.options.variables == ("t",)
        # The parse state travels with the plot, so the window's range fields
        # read a typed bound with the grammar the worksheet reads.
        assert request.state is app.session.state
        assert band(app)[0].startswith(" COMMAND:")


async def test_a_plot_is_sent_without_asking_any_window_its_mode(app):
    async with app.run_test() as pilot:
        # Polar is the window's own view mode, so a univariate expression is
        # always sent as the curve it classifies as, and no `Describe` goes
        # down the pipe on the way.
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
        # Right steps into the expression, onto its first term.
        app.session.move_right()
        await pilot.press("p", "p")
        request = app.plots.sent[0]
        assert request.label == "#1'"
        assert request.text == "SIN(x)"


async def test_a_plot_that_replaces_a_curve_says_replotting(app):
    async with app.run_test() as pilot:
        await authored(pilot, app, "SIN(x)")
        # The host says the plot replaced a curve already there, and the
        # message says so in one word - which is how replacement teaches
        # itself.
        app.plots.placed = plots.Placed(1, replaced=True)
        await pilot.press("p", "p")
        assert message(app) == "Replotting #1"


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
        # The settings screen is retired: the sticky values live with the plot
        # windows' own controls, so the menu no longer offers a word for it,
        # and the letter that used to open it answers nothing.
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
        # The values land in the settings store, which is where they outlive
        # the host and where a state file finds them...
        assert app.settings["PlotGrid"] == 32
        assert app.settings["PlotPoints"] == "Connected"
        assert app.settings["PlotPointSize"] == 8
        assert app.settings["PlotMesh"] == "Wire"
        # ...and the proxy is handed them back, for the host after this one.
        assert app.plots.preferences[-1] == plots.Prefer(
            grid=32, connected=True, point_size=8.0, wire=True
        )
        # Nothing is recorded and nothing is said: the user is looking at the
        # control they just moved.
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
        # The direct route enters exactly what pasting the Ctrl-C text on the
        # author line would have entered: the same text, read the same way.
        assert entry.text == Session().author("[1.570796, 1.000000]").text
        assert message(app) == "Entered [1.570796, 1.000000] as #1"


async def test_a_point_lands_in_its_own_worksheet_not_the_active_one(app):
    async with app.run_test() as pilot:
        # An overlay opens a second worksheet over the first, and the second is
        # the active one; the point still goes to the worksheet its plot came
        # from.
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
        # No worksheet on screen carries this id: the overlay the plot came
        # from has closed since. The point was sent to be computed with, so it
        # lands in the active worksheet rather than being dropped.
        app._plot_event(plots.Traced(-1, "[1.000000, 2.000000]"))
        await pilot.pause()
        assert len(app.session.entries) == 1
        assert message(app) == "Entered [1.000000, 2.000000] as #1"


async def test_a_point_arriving_mid_edit_leaves_the_line_standing(app):
    async with app.run_test() as pilot:
        # Author, with a line half-typed: the entry goes in under the prompt,
        # which stands untouched - and the point is one F3 away from the line.
        await pilot.press("a", "s", "i", "n")
        app._plot_event(plots.Traced(id(app.session), "[1.000000, 2.000000]"))
        await pilot.pause()
        assert len(app.session.entries) == 1
        assert message(app) == "Entered [1.000000, 2.000000] as #1"
        assert prompt(app) == ("AUTHOR expression:", "sin")


async def test_a_point_arriving_mid_command_waits_its_turn(app):
    from rederive.ui.app import MODE_COMPUTE, MODE_MENU

    async with app.run_test() as pilot:
        # While a command computes, the computing thread owns the worksheets,
        # so the point waits for the command to let go rather than racing it.
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
    """`P` `P` and a curve in a window that exists, with nothing asked between.

    The one test that has the app and a real host in it at once, and the only
    place the seam between them is exercised: everything above either answers
    the app with a stub or drives the host without one. It earns a process
    because that seam has a way of breaking silently - the app's screen is not
    a file descriptor, and a spawn that inherits one is refused - and because
    the acceptance case for this phase is a picture, not a request.
    """
    if not _toolkit():
        pytest.skip("pyqtgraph and PySide6 are not installed")
    from rederive.plot import proxy as proxy_module
    from rederive.ui.app import RederiveApp

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
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
            # The window is titled by what it holds, spelled as the app wrote it.
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

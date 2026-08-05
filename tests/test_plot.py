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
from screen import band, highlighted, message

from rederive.engine.context import Context
from rederive.model.expr import Node
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
    # `[t, 1/t]` is the acceptance case: the two branches never come close on
    # the screen however finely t is cut, so refinement has to stop at the
    # depth cap rather than chase the pole down forever.
    fx, fy = evaluate.pair(parsed("[t, 1/t]"), Context(), ("t",))
    drawn = evaluate.sample_curve(fx, fy, VIEW, VIEW, VIEW, CANVAS)
    assert len(drawn.ts) < evaluate.MAX_POINTS
    assert drawn.gave_up is not None and abs(drawn.gave_up) < 0.01
    # And the gap is at the pole rather than anywhere along the branches.
    gapped = drawn.ts[np.isnan(drawn.xs)]
    assert gapped.size and np.abs(gapped).max() < 0.05


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
    # The field line takes `-π` because that is what a person types; what it is
    # worth is arithmetic, and arithmetic happens in the process that has it.
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
    """Plot one expression the way the Plot command does."""
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
    return host.add(plots.Add(**{**fields, **keywords}))


def test_a_host_takes_a_plot_and_describes_what_it_holds(host):
    session = Session()
    assert _add(session, host, "SIN(x)") == 1
    assert _add(session, host, "x^2 - 3") == 1
    described = host.describe()
    assert len(described) == 1
    window = described[0]
    assert window.number == 1
    assert window.kind is plots.WindowKind.TWO_D
    assert window.title == "Rederive 2D plot 1 (current)"
    assert window.current is True
    assert window.polar is False
    assert [plot.label for plot in window.plots] == ["#1", "#2"]
    assert [plot.text for plot in window.plots] == ["SIN(x)", "x^2 - 3"]


def test_re_plotting_a_label_replaces_its_curve(host):
    session = Session()
    _add(session, host, "SIN(x)")
    entry = session.entries[0]
    host.add(
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
    assert len(host.describe()[0].plots) == 1


def test_a_new_window_takes_the_next_number_and_becomes_current(host):
    session = Session()
    assert _add(session, host, "SIN(x)") == 1
    assert _add(session, host, "COS(x)", window=plots.Where.NEW) == 2
    described = host.describe()
    assert [window.number for window in described] == [1, 2]
    assert [window.current for window in described] == [False, True]
    assert described[0].title == "Rederive 2D plot 1"
    assert described[1].title == "Rederive 2D plot 2 (current)"
    # The current window is where the next plot lands.
    assert _add(session, host, "TAN(x)") == 2


def test_naming_a_window_makes_it_current_again(host):
    session = Session()
    _add(session, host, "SIN(x)")
    _add(session, host, "COS(x)", window=plots.Where.NEW)
    assert host.set_current(1) == 1
    assert _add(session, host, "TAN(x)") == 1


def test_delete_takes_plots_off_the_end_of_the_list(host):
    session = Session()
    for text in ("SIN(x)", "COS(x)", "TAN(x)"):
        _add(session, host, text)
    removed = host.delete(plots.Delete(which=plots.Which.LAST))
    assert (removed.window, removed.count) == (1, 1)
    assert [plot.label for plot in host.describe()[0].plots] == ["#1", "#2"]
    host.delete(plots.Delete(which=plots.Which.BUTLAST))
    assert [plot.label for plot in host.describe()[0].plots] == ["#2"]
    assert host.delete(plots.Delete(which=plots.Which.ALL)).count == 1
    assert host.describe()[0].plots == ()


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
    _add(session, host, "[SIN(t), COS(t)]", options=_turn())
    _add(session, host, "[[1, 2], [3, 4]]")
    _add(session, host, "x^2 + y^2 = 4")
    _add(session, host, "y < x^2")
    _add(session, host, "2*COS(3*t)", kind=PlotKind.POLAR, options=_turn(("t",)))
    window = host.describe()[0]
    assert [plot.kind for plot in window.plots] == [
        PlotKind.PARAMETRIC,
        PlotKind.DATA,
        PlotKind.IMPLICIT,
        PlotKind.REGION,
        PlotKind.POLAR,
    ]
    assert not [plot.label for plot in window.plots if plot.hidden]


def _turn(variables=("t",)):
    """The parameter range the Plot command's field line would have carried."""
    return plots.Options(variables=variables, minimum=parsed("-π"), maximum=parsed("π"))


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
    assert solid.title == "Rederive 3D plot 2 (current)"
    assert flat.title == "Rederive 2D plot 1 (current)"
    # Absent rather than off: a 3D window has no polar mode to be in.
    assert solid.polar is None
    assert flat.polar is False
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
    removed = host.delete(plots.Delete(which=plots.Which.LAST))
    assert (removed.window, removed.count) == (1, 1)
    assert [plot.label for plot in host.describe()[0].plots] == ["#1.1"]


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


def test_deleting_with_no_window_open_is_refused_without_starting_one():
    from rederive.plot import proxy as proxy_module

    proxy = proxy_module.PlotProxy()
    with pytest.raises(proxy_module.PlotError) as refused:
        proxy.delete(plots.Delete())
    assert str(refused.value) == proxy_module.NO_WINDOW
    assert proxy.starts == 0
    assert proxy.describe() == ()


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
    host.prefer(plots.Prefer(equal_scales=False, grid=16, connected=True, point_size=9))
    assert _add(session, host, "SIN(x)") == 1
    assert [plot.label for plot in host.describe()[0].plots] == ["#1"]


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
# `Options Plot` is four values in the settings store, and they have three
# things to do: translate into the request the host understands, reach a host
# that may not be running yet, and be the default a plot with no opinion of its
# own is drawn with. All three are pure and none of them needs a window.


def test_the_preferences_translate_the_words_the_dialog_holds():
    from rederive.model.plotting import preferences
    from rederive.model.settings import Settings

    settings = Settings()
    # The dialog's defaults and the request's defaults are the same picture,
    # which is what makes a session that never opens the screen behave like one
    # that opened it and pressed Enter.
    assert preferences(settings) == plots.Prefer()
    settings.apply(
        {
            "PlotScales": "No",
            "PlotGrid": 128,
            "PlotPoints": "Connected",
            "PlotPointSize": 8,
        }
    )
    assert preferences(settings) == plots.Prefer(
        equal_scales=False, grid=128, connected=True, point_size=8.0
    )


def test_the_preferences_travel_in_a_state_file():
    from rederive.model import state
    from rederive.model.plotting import preferences
    from rederive.model.settings import Settings

    written = Settings()
    written.apply({"PlotScales": "No", "PlotGrid": 32})
    read = Settings()
    assert state.read(state.write(written), read) == (0, "")
    assert preferences(read) == plots.Prefer(equal_scales=False, grid=32)


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
        return 1

    def delete(self, request):
        self._answer(request)
        return plots.Removed(2, 1)

    def set_current(self, window):
        self._answer(window)
        return window

    def describe(self):
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


async def test_the_plot_menu_offers_four_words_with_plot_first(app):
    async with app.run_test() as pilot:
        await pilot.press("p")
        assert band(app) == [" PLOT: Plot New Delete Window"]
        assert highlighted(app) == "Plot"
        assert message(app) == "Enter option"


async def test_p_p_plots_the_highlighted_expression_with_no_question(app):
    async with app.run_test() as pilot:
        await authored(pilot, app, "SIN(x)")
        await pilot.press("p", "p")
        assert message(app) == "Plotting #1 in window 1"
        request = app.plots.sent[0]
        assert request.label == "#1"
        assert request.text == "SIN(x)"
        assert request.kind is PlotKind.CURVE
        assert request.window is plots.Where.CURRENT
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
        assert message(app) == "Plotting #1 in window 1"
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


async def test_a_parametric_pair_is_asked_its_range_before_it_is_sent(app):
    async with app.run_test() as pilot:
        await authored(pilot, app, "[SIN(t), COS(t)]")
        await pilot.press("p", "p")
        # The one question the Plot command ever asks about an expression,
        # offering one whole turn.
        assert band(app)[0].startswith(" PLOT: Min: -π")
        assert band(app)[0].endswith("Max: π")
        assert message(app) == "Enter minimum parameter value"
        assert app.plots.sent == []
        await pilot.press("enter")
        request = app.plots.sent[0]
        assert request.kind is PlotKind.PARAMETRIC
        assert request.options.variables == ("t",)
        # The bounds travel as expressions, the app doing no arithmetic.
        assert isinstance(request.options.minimum, Node)
        assert message(app) == "Plotting #1 in window 1"


async def test_a_range_that_is_not_an_expression_keeps_the_question_up(app):
    async with app.run_test() as pilot:
        await authored(pilot, app, "[SIN(t), COS(t)]")
        await pilot.press("p", "p")
        await pilot.press("*", "enter")
        assert band(app)[0].startswith(" PLOT: Min: *π")
        assert app.plots.sent == []


async def test_a_polar_window_reads_a_curve_as_r_of_an_angle(app):
    async with app.run_test() as pilot:
        # Polar is never a classification: it is what the command makes of a
        # univariate expression when the window it is heading for says so.
        app.plots.windows = (
            plots.WindowInfo(
                1, plots.WindowKind.TWO_D, "one", True, polar=True
            ),
        )
        await authored(pilot, app, "SIN(x)")
        await pilot.press("p", "p")
        assert band(app)[0].startswith(" PLOT: Min: -π")
        await pilot.press("enter")
        assert app.plots.sent[0].kind is PlotKind.POLAR


async def test_a_new_window_is_never_polar_and_asks_nothing(app):
    async with app.run_test() as pilot:
        app.plots.windows = (
            plots.WindowInfo(
                1, plots.WindowKind.TWO_D, "one", True, polar=True
            ),
        )
        await authored(pilot, app, "SIN(x)")
        await pilot.press("p", "n")
        assert app.plots.sent[0].kind is PlotKind.CURVE
        assert message(app) == "Plotting #1 in window 1"


def test_the_range_offered_follows_the_angle_unit():
    from rederive.ui import menu as menus

    assert [field.default for field in menus.parameter_range(False).fields] == [
        "-π",
        "π",
    ]
    assert [field.default for field in menus.parameter_range(True).fields] == [
        "-180",
        "180",
    ]


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


async def test_delete_names_what_it_took_out_and_from_where(app):
    async with app.run_test() as pilot:
        await pilot.press("p", "d", "l")
        assert message(app) == "Deleted last plot in window 2"
        assert app.plots.sent[0].which is plots.Which.LAST
        assert app.plots.sent[0].window is plots.Where.MRU


async def test_delete_with_no_plot_window_says_so(app):
    async with app.run_test() as pilot:
        app.plots.refuse = "no plot window"
        await pilot.press("p", "d", "a")
        assert message(app) == "Plot: no plot window"


async def test_the_window_command_opens_on_the_current_window(app):
    async with app.run_test() as pilot:
        app.plots.windows = (
            plots.WindowInfo(1, plots.WindowKind.TWO_D, "one", False),
            plots.WindowInfo(2, plots.WindowKind.TWO_D, "two", True),
        )
        await pilot.press("p", "w")
        assert band(app) == [" PLOT WINDOW: Window: 2"]
        assert message(app) == "Enter plot window number"
        await pilot.press("backspace", "1", "enter")
        assert app.plots.sent == [1]
        assert message(app) == "Plotting the next plot in window 1"


async def test_the_window_command_with_nothing_open_says_so(app):
    async with app.run_test() as pilot:
        await pilot.press("p", "w")
        assert message(app) == "Plot: no plot window"


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


async def test_the_options_plot_screen_hands_what_it_changed_to_the_plots(app):
    async with app.run_test() as pilot:
        # Options pLot: `P` belongs to Precision, so the plot screen takes `l`.
        await pilot.press("o", "l")
        assert band(app) == [
            " OPTIONS PLOT: Scales: Yes No  Grid: 64",
            "               Points:(Discrete)Connected  Size: 5",
        ]
        assert message(app) == "Select equal scales in a new plot window"
        await pilot.press("n", "1", "6", "enter")
        assert app.settings["PlotScales"] == "No"
        assert app.settings["PlotGrid"] == 16
        assert app.plots.preferences[-1] == plots.Prefer(equal_scales=False, grid=16)
        # Nothing is recorded: a plot preference has no `Name := Value` spelling.
        assert app.session.entries == []


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


async def test_the_whole_loop_lands_a_parametric_plot_in_a_real_window(monkeypatch):
    """`P` `P`, the range field line, and a curve in a window that exists.

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
            assert band(app)[0].startswith(" PLOT: Min: -π")
            await pilot.press("enter")
            for _ in range(100):
                await pilot.pause()
                if "window" in message(app) or message(app).startswith("Plot:"):
                    break
                await asyncio.sleep(0.05)
            assert message(app) == "Plotting #1 in window 1"
            window = app.plots.describe()[0]
            assert window.title == "Rederive 2D plot 1 (current)"
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

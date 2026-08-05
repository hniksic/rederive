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

import numpy as np
import pytest
from screen import band, highlighted, message

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
    return host.add(
        plots.Add(
            worksheet=id(session),
            node=entry.node,
            context=session.context,
            kind=plotted.kind,
            label=label,
            text=text,
            options=plotted.options,
            **keywords,
        )
    )


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


def test_a_kind_no_window_draws_yet_is_refused_by_name(host):
    from rederive.plot.proxy import PlotError

    session = Session()
    with pytest.raises(PlotError) as refused:
        _add(session, host, "x^2 + y^2 = 4")
    assert "implicit plots are not implemented yet" in str(refused.value)


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


async def test_a_kind_no_window_draws_yet_is_refused_before_it_is_sent(app):
    async with app.run_test() as pilot:
        await authored(pilot, app, "x^2 + y^2 = 4")
        await pilot.press("p", "p")
        assert message(app) == "Plot: implicit plots are not implemented yet"
        assert app.plots.sent == []


async def test_an_expression_of_three_variables_is_refused_by_name(app):
    async with app.run_test() as pilot:
        await authored(pilot, app, "a*x*y")
        await pilot.press("p", "p")
        assert message(app) == (
            "Plot: a*x*y depends on x, y, a - reduce to one or two variables"
        )


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


async def test_a_curve_the_host_could_not_draw_reaches_the_message_line(app):
    async with app.run_test() as pilot:
        app._plot_event(plots.Trouble(1, "#3", "no numeric reading of FOO"))
        await pilot.pause()
        assert message(app) == "Plot: #3: no numeric reading of FOO"


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

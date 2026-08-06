"""Plotting in a browser, asked without one: the panes, the requests, the answers.

Three questions this file exists to ask. What a pane does when a plot lands in
it and what it then asks the engine worker for; what the worker makes of such a
request, which is where the arrays are and so where every kind either draws or
does not; and that the split holds - the main thread routes and the arrays never
come through it.

The page is faked and the runtime bridge is faked, and neither is a pretend
browser: the first records what it was told to draw and the second is the pair of
identity functions Pyodide's are on a desktop. What is real is everything
between them, which is the whole of what stage 5 wrote in Python.
"""

import asyncio

import numpy as np
import pytest
from fakepage import FakeEngine, FakePage, bridge

from rederive.engine.context import Angle, Context
from rederive.plot import protocol as plots
from rederive.plot.model import Plot
from rederive.plot.protocol import PlotKind
from rederive.plot.proxy import PlotError
from rederive.plot.session import PlotSession
from rederive.plot.web import protocol as asking
from rederive.plot.web import sampler
from rederive.plot.web.backend import WebBackend, WorkerExecutor
from rederive.syntax import ParseState, parse_expression

bridge()

VIEW = (-5.0, 5.0)
CANVAS = (800.0, 640.0)


def parsed(text):
    return parse_expression(text, ParseState()).node


def _landing(text, label="#1", kind=PlotKind.CURVE, variables=("x",), **keywords):
    """One plot the way the app sends it, for a pane to take."""
    return plots.Add(
        worksheet=1,
        node=parsed(text),
        context=keywords.pop("context", Context()),
        kind=kind,
        label=label,
        text=text,
        options=plots.Options(variables=variables, **keywords.pop("options", {})),
        **keywords,
    )


def _plot(text, kind=PlotKind.CURVE, variables=("x",), context=None):
    """One plot as the worker is sent one."""
    return Plot(
        worksheet=1,
        label="#1",
        text=text,
        kind=kind,
        node=parsed(text),
        context=context or Context(),
        options=plots.Options(variables=variables),
    )


def _sample(text, kind=PlotKind.CURVE, variables=("x",), context=None, **keywords):
    return asking.Sample(
        pane=1,
        plot=1,
        generation=1,
        model=_plot(text, kind, variables, context),
        xrange=keywords.pop("xrange", VIEW),
        yrange=keywords.pop("yrange", VIEW),
        size=keywords.pop("size", CANVAS),
        **keywords,
    )


def _answered(request, held=None):
    """One sampling, run where the worker would run it."""
    table = sampler.methods(lambda _partial: None)
    return table[asking.SAMPLE](request)


@pytest.fixture
def page():
    return FakePage()


@pytest.fixture
def engine():
    return FakeEngine()


@pytest.fixture
def session(page, engine):
    """A plot session over panes, sampling nowhere: the requests are the point."""
    executor = WorkerExecutor(engine)
    backend = WebBackend(page, engine)
    page.attend(backend.handed({"landed": executor.landed}))
    engine.lost = executor.lost
    return PlotSession(backend, executor)


# -- what a pane does with a plot -------------------------------------------------


async def test_a_plot_opens_a_pane_and_lands_in_it(session, page):
    assert session.add(_landing("SIN(x)")) == plots.Placed(1)
    pane = page.panes[1]
    assert pane.presented == 1
    assert pane.title == "SIN(x) - Rederive plot (current)"
    assert [spec["name"] for spec in pane.plots.values()] == ["#1  SIN(x)"]


async def test_a_pane_names_a_plot_with_what_the_app_wrote(session, page):
    # Not with anything this side rendered: the page is handed the one line the
    # syntax writer made, cut to a legend's width by the plot model.
    session.add(_landing("SIN(x)/x + COS(x)", label="#7"))
    spec = next(iter(page.panes[1].plots.values()))
    assert spec["name"] == "#7  SIN(x)/x + COS(x)"
    assert spec["kind"] == "curve"
    assert spec["color"] == "#ffffff"


async def test_replotting_a_label_keeps_its_color_and_replaces_its_curve(session, page):
    session.add(_landing("SIN(x)", "#1"))
    session.add(_landing("COS(x)", "#2"))
    session.add(_landing("TAN(x)", "#2"))
    pane = page.panes[1]
    assert len(pane.plots) == 2
    colors = [spec["color"] for spec in pane.plots.values()]
    assert colors == ["#ffffff", "#ffff55"]


async def test_a_family_becomes_one_curve_per_element(session, page):
    request = _landing(
        "[SIN(x), COS(x)]",
        kind=PlotKind.FAMILY,
        options={"texts": ("SIN(x)", "COS(x)")},
    )
    session.add(request)
    names = [spec["name"] for spec in page.panes[1].plots.values()]
    assert names == ["#1.1  SIN(x)", "#1.2  COS(x)"]


async def test_a_solid_opens_no_pane_and_says_what_is_missing(session, page):
    with pytest.raises(ValueError, match="3D plots are not in the browser yet"):
        session.add(_landing("x*y", kind=PlotKind.SURFACE, variables=("x", "y")))
    assert page.panes == {}


async def test_describe_reads_the_view_off_the_page(session, page):
    session.add(_landing("SIN(x)"))
    page.panes[1].shown = [-2.0, 8.0, -1.0, 3.0, 900.0, 500.0]
    described = session.describe().windows
    assert len(described) == 1
    assert described[0].xrange == (-2.0, 8.0)
    assert described[0].yrange == (-1.0, 3.0)
    assert described[0].plots[0].label == "#1"


async def test_the_default_framing_is_the_one_the_desktop_opens_on(session, page):
    session.add(_landing("SIN(x)"))
    assert list(page.panes[1].say["home"](800, 640)) == [-5.0, 5.0, -4.0, 4.0]


async def test_a_closed_pane_hands_its_kind_on_and_says_so(session, page):
    heard = []
    session.events = heard.append
    session.add(_landing("SIN(x)"))
    page.panes[1].say["closed"]()
    assert heard == [plots.Closed(1)]
    assert session.windows == {}


async def test_a_legend_row_hides_a_curve_without_removing_it(session, page):
    session.add(_landing("SIN(x)"))
    pane = page.panes[1]
    pane.say["hide"](1, True)
    assert pane.plots[1]["hidden"] is True
    assert session.describe().windows[0].plots[0].hidden is True


async def test_the_menus_remove_takes_a_plot_out_for_good(session, page):
    session.add(_landing("SIN(x)"))
    page.panes[1].say["drop"](1)
    assert page.panes[1].plots == {}
    assert session.describe().windows[0].plots == ()


async def test_joining_a_data_plots_points_is_sticky(session, page):
    heard = []
    session.events = heard.append
    session.add(_landing("[[1, 2], [3, 4]]", kind=PlotKind.DATA, variables=()))
    page.panes[1].say["connect"](1, True)
    assert page.panes[1].plots[1]["connected"] is True
    assert heard[-1] == plots.Preferred(plots.Prefer(connected=True))


async def test_the_polar_toggle_rereads_every_curve_in_the_pane(session, page):
    session.add(_landing("SIN(3x)"))
    pane = page.panes[1]
    pane.say["polar"](True)
    assert pane.plots[1]["kind"] == "polar"
    pane.say["polar"](False)
    assert pane.plots[1]["kind"] == "curve"


# -- what a pane asks the worker for ----------------------------------------------


async def test_adding_a_plot_asks_for_it_over_the_view_the_page_shows(
    session, page, engine
):
    session.add(_landing("SIN(x)"))
    await settle()
    number, method, request = engine.sent[0]
    assert method == asking.SAMPLE
    assert request.xrange == (-5.0, 5.0)
    assert request.yrange == (-4.0, 4.0)
    assert request.size == (800.0, 640.0)
    assert request.model.text == "SIN(x)"
    # And the pane was told which generation to expect back.
    assert page.panes[1].started == [(1, 1, True)]


async def test_the_request_carries_no_pane_bookkeeping(session, engine):
    session.add(_landing("SIN(x)"))
    await settle()
    _, _, request = engine.sent[0]
    assert type(request.model) is Plot
    assert not hasattr(request.model, "serial")


async def test_a_view_change_samples_every_plot_again(session, page, engine):
    session.add(_landing("SIN(x)"))
    session.add(_landing("COS(x)", "#2"))
    await drain(page, engine)
    engine.sent.clear()
    page.panes[1].say["changed"]()
    await settle()
    # One in flight, so only the first is out; the second waits its turn.
    assert len(engine.sent) == 1
    assert page.panes[1].started[-2:] == [(1, 2, False), (2, 2, False)]


async def test_a_data_plot_is_not_sampled_again_on_a_view_change(session, page, engine):
    session.add(_landing("[[1, 2], [3, 4]]", kind=PlotKind.DATA, variables=()))
    await drain(page, engine)
    engine.sent.clear()
    page.panes[1].say["changed"]()
    await settle()
    assert engine.sent == []


async def test_the_angle_unit_of_the_worksheet_travels_with_the_request(
    session, engine
):
    session.add(_landing("SIN(x)", context=Context(angle=Angle.DEGREE)))
    await settle()
    assert engine.sent[0][2].degrees is True


async def test_a_reading_is_asked_of_the_side_that_holds_the_closures(
    session, page, engine
):
    session.add(_landing("SIN(x)"))
    await drain(page, engine)
    engine.sent.clear()
    page.panes[1].say["traced"](1, 1.25)
    await settle()
    number, method, request = engine.sent[0]
    assert method == asking.TRACE
    assert request.at == 1.25
    assert request.model.text == "SIN(x)"


async def test_a_feature_scan_names_the_curves_it_is_to_cross(session, page, engine):
    session.add(_landing("SIN(x)"))
    session.add(_landing("COS(x)", "#2"))
    await drain(page, engine)
    engine.sent.clear()
    page.panes[1].say["scanned"](1)
    await settle()
    _, method, request = engine.sent[0]
    assert method == asking.FEATURES
    assert [one.text for one in request.others] == ["COS(x)"]


async def test_a_point_sent_home_is_authored_under_its_own_worksheet(session, page):
    heard = []
    session.events = heard.append
    session.add(_landing("SIN(x)"))
    page.panes[1].say["author"](1, "[1.000000, 0.841471]")
    assert heard == [plots.Traced(1, "[1.000000, 0.841471]")]


async def test_a_sampling_that_would_not_evaluate_reports_itself(session, page):
    heard = []
    session.events = heard.append
    session.add(_landing("SIN(x)"))
    await settle()
    page.handlers["landed"](
        session.backend.engine.sent[-1][0], "the curve would not evaluate"
    )
    assert heard == [plots.Trouble(1, "#1", "the curve would not evaluate")]


async def test_only_one_request_is_with_the_worker_at_a_time(session, page, engine):
    session.add(_landing("SIN(x)"))
    session.add(_landing("COS(x)", "#2"))
    session.add(_landing("TAN(x)", "#3"))
    await settle()
    assert len(engine.sent) == 1
    page.handlers["landed"](engine.sent[-1][0], "")
    await settle()
    assert len(engine.sent) == 2
    page.handlers["landed"](engine.sent[-1][0], "")
    await settle()
    assert len(engine.sent) == 3


async def test_a_terminated_worker_lets_go_of_the_sampling_it_took(
    session, page, engine
):
    heard = []
    session.events = heard.append
    session.add(_landing("SIN(x)"))
    session.add(_landing("COS(x)", "#2"))
    await settle()
    assert len(engine.sent) == 1
    engine.lost("Aborted")
    assert heard == [plots.Trouble(1, "#1", "Aborted")]
    # And the queue moves again, so the curve behind it is not lost with it.
    page.panes[1].say["changed"]()
    await settle()
    assert len(engine.sent) == 2


async def test_a_dragged_view_costs_one_sampling_per_curve_and_not_one_per_frame(
    session, page, engine
):
    session.add(_landing("SIN(x)"))
    await drain(page, engine)
    engine.sent.clear()
    for _ in range(60):
        page.panes[1].say["changed"]()
        await settle()
    # One went out; the fifty-nine behind it displaced each other in the queue.
    assert len(engine.sent) == 1
    page.handlers["landed"](engine.sent[-1][0], "")
    await settle()
    assert len(engine.sent) == 2
    page.handlers["landed"](engine.sent[-1][0], "")
    await settle()
    assert len(engine.sent) == 2


async def settle():
    """Let whatever the executor queued actually reach the worker.

    A request is posted from a coroutine, because a worker that is still
    booting has to be waited for; in a browser the loop is the browser's and
    turns constantly, and here it has to be turned by hand.
    """
    await asyncio.sleep(0)


async def drain(page, engine):
    """Answer for everything in flight, so the queue empties."""
    for _ in range(20):
        await settle()
        if not engine.sent:
            return
        number = engine.sent[-1][0]
        before = len(engine.sent)
        page.handlers["landed"](number, "")
        await settle()
        if len(engine.sent) == before:
            return


# -- what the worker makes of a request --------------------------------------------


def test_a_curve_comes_back_as_two_arrays_of_single_precision():
    answer = _answered(_sample("SIN(x)"))
    assert answer["shape"] == "stroke"
    assert answer["xs"].dtype == np.float32
    assert answer["ys"].dtype == np.float32
    assert answer["xs"].ndim == 1 and answer["xs"].flags["C_CONTIGUOUS"]
    assert answer["xs"].size == answer["ys"].size > 100
    assert np.allclose(answer["ys"], np.sin(answer["xs"].astype(np.float64)), atol=1e-6)


def test_a_curve_is_sampled_over_the_view_it_was_asked_about():
    answer = _answered(_sample("x^2", xrange=(10.0, 20.0)))
    assert float(answer["xs"].min()) == pytest.approx(10.0)
    assert float(answer["xs"].max()) == pytest.approx(20.0)


def test_the_rectangle_a_curve_wants_comes_back_with_it():
    answer = _answered(_sample("x", xrange=(-1.0, 3.0)))
    left, right, low, high = answer["bounds"]
    assert (left, right) == pytest.approx((-1.0, 3.0))
    assert (low, high) == pytest.approx((-1.0, 3.0))


def test_a_curve_with_nothing_real_in_view_says_so_rather_than_drawing_nothing():
    answer = _answered(_sample("SQRT(x)", xrange=(-10.0, -1.0)))
    assert answer["empty"] is True
    assert "no real values" in answer["words"]


def test_a_pole_comes_back_as_a_gap_and_not_as_a_stroke():
    answer = _answered(_sample("1/x"))
    assert np.isnan(answer["ys"]).any()


def test_a_data_plot_is_the_columns_of_its_matrix():
    answer = _answered(
        _sample("[[1, 2], [3, 4]]", kind=PlotKind.DATA, variables=())
    )
    assert list(answer["xs"]) == [1.0, 3.0]
    assert list(answer["ys"]) == [2.0, 4.0]


def test_a_parametric_pair_comes_back_as_a_path_and_its_range():
    answer = _answered(
        _sample("[COS(t), SIN(t)]", kind=PlotKind.PARAMETRIC, variables=("t",))
    )
    assert answer["trange"] == pytest.approx([-np.pi, np.pi])
    radius = np.hypot(answer["xs"].astype(np.float64), answer["ys"].astype(np.float64))
    assert np.allclose(radius[np.isfinite(radius)], 1.0, atol=1e-5)


def test_a_polar_curve_is_composed_where_the_numbers_are():
    # r = 2 is a circle of radius 2, and what crosses is x and y: turning the
    # angle into a point is two lines of numpy and no part of a drawing library.
    answer = _answered(_sample("2", kind=PlotKind.POLAR, variables=("θ",)))
    radius = np.hypot(answer["xs"].astype(np.float64), answer["ys"].astype(np.float64))
    assert np.allclose(radius[np.isfinite(radius)], 2.0, atol=1e-5)


def test_a_polar_curve_in_degrees_turns_by_degrees():
    answer = _answered(
        _sample(
            "θ",
            kind=PlotKind.POLAR,
            variables=("θ",),
            context=Context(angle=Angle.DEGREE),
            degrees=True,
        )
    )
    # One turn of 180 degrees, so the radius runs out to 180 rather than to π.
    radius = np.hypot(answer["xs"].astype(np.float64), answer["ys"].astype(np.float64))
    assert float(np.nanmax(radius)) == pytest.approx(180.0, rel=1e-3)


def test_an_implicit_curve_comes_back_as_a_contour_of_segments():
    answer = _answered(
        _sample("x^2 + y^2 = 4", kind=PlotKind.IMPLICIT, variables=("x", "y"))
    )
    xs = answer["xs"].astype(np.float64)
    ys = answer["ys"].astype(np.float64)
    real = np.isfinite(xs) & np.isfinite(ys)
    assert real.any()
    assert np.allclose(np.hypot(xs[real], ys[real]), 2.0, atol=0.05)


def test_an_implicit_curve_with_no_solution_in_view_says_so():
    answer = _answered(
        _sample(
            "x^2 + y^2 = 4",
            kind=PlotKind.IMPLICIT,
            variables=("x", "y"),
            xrange=(10.0, 20.0),
            yrange=(10.0, 20.0),
        )
    )
    assert answer["empty"] is True
    assert "no solutions in view" in answer["words"]


def test_a_region_comes_back_as_one_byte_a_cell_over_its_rectangle():
    answer = _answered(_sample("y > x", kind=PlotKind.REGION, variables=("x", "y")))
    assert answer["shape"] == "region"
    assert answer["mask"].dtype == np.uint8
    assert answer["mask"].size == answer["nx"] * answer["ny"]
    grid = answer["mask"].reshape(answer["ny"], answer["nx"])
    # The rows arrive from the top, which is how an image is read: the top-left
    # corner is high y and low x, where y > x is true.
    assert grid[0, 0] == 1
    assert grid[-1, -1] == 0


def test_an_expression_that_will_not_evaluate_answers_with_what_went_wrong():
    answer = _answered(_sample("SIN(x)", variables=("x",), kind=PlotKind.PARAMETRIC))
    assert answer["trouble"]
    assert "two expressions" in answer["trouble"]


def test_a_reading_is_the_functions_own_value_and_its_sentence():
    table = sampler.methods(lambda _partial: None)
    answer = table[asking.TRACE](
        asking.Trace(pane=1, plot=1, generation=1, model=_plot("SIN(x)"), at=1.0)
    )
    assert answer["found"] is True
    assert answer["y"] == pytest.approx(np.sin(1.0))
    assert answer["point"] == "[1.000000, 0.841471]"
    assert answer["words"] == "Tracing #1  SIN(x)   x = 1.000000   y = 0.841471"


def test_a_reading_off_the_end_of_a_curve_is_an_answer_and_not_an_error():
    table = sampler.methods(lambda _partial: None)
    answer = table[asking.TRACE](
        asking.Trace(pane=1, plot=1, generation=1, model=_plot("SQRT(x)"), at=-1.0)
    )
    assert answer["found"] is False
    assert "not real and finite at x = -1.000000" in answer["words"]


def test_a_feature_scan_finds_the_roots_and_names_them():
    table = sampler.methods(lambda _partial: None)
    answer = table[asking.FEATURES](
        asking.Features(
            pane=1,
            plot=1,
            generation=1,
            model=_plot("SIN(x)"),
            others=(),
            xrange=VIEW,
            yrange=VIEW,
            size=CANVAS,
        )
    )
    roots = [one for one in answer["features"] if one["words"].startswith("root")]
    assert [round(one["x"], 6) for one in roots] == [
        pytest.approx(-np.pi, abs=1e-5),
        pytest.approx(0.0, abs=1e-5),
        pytest.approx(np.pi, abs=1e-5),
    ]


def test_a_feature_scan_names_the_curve_a_crossing_is_with():
    table = sampler.methods(lambda _partial: None)
    answer = table[asking.FEATURES](
        asking.Features(
            pane=1,
            plot=1,
            generation=1,
            model=_plot("SIN(x)"),
            others=(_plot("COS(x)"),),
            xrange=VIEW,
            yrange=VIEW,
            size=CANVAS,
        )
    )
    crossings = [one for one in answer["features"] if "intersection" in one["words"]]
    assert crossings
    assert "intersection with #1  COS(x)" in crossings[0]["words"]


# -- the closure cache --------------------------------------------------------------


def test_the_same_expression_is_lambdified_once_however_often_it_is_drawn():
    held = sampler.Closures()
    first = sampler._riding(held, _plot("SIN(x)"), False)
    assert first.closure is None
    sampler._sampled(held, _sample("SIN(x)"), lambda _partial: None)
    second = sampler._riding(held, _plot("SIN(x)"), False)
    assert second.closure is not None


def test_the_cache_is_keyed_by_content_and_not_by_who_asked():
    held = sampler.Closures()
    sampler._sampled(held, _sample("SIN(x)"), lambda _partial: None)
    # Another pane, another plot, another generation - the same expression.
    elsewhere = asking.Sample(
        pane=9,
        plot=4,
        generation=17,
        model=_plot("SIN(x)"),
        xrange=VIEW,
        yrange=VIEW,
        size=CANVAS,
    )
    assert sampler._riding(held, elsewhere.model, False).closure is not None


def test_two_readings_of_one_tree_are_two_entries():
    # `x^2 + y^2 = 4` is a difference where it is a contour and a truth value
    # where it is a region, and a cache that confused the two would shade the
    # wrong side of a curve.
    held = sampler.Closures()
    contoured = _plot("x^2 + y^2 = 4", PlotKind.IMPLICIT, ("x", "y"))
    shaded = _plot("x^2 + y^2 = 4", PlotKind.REGION, ("x", "y"))
    assert held.keyed(contoured, ("x", "y"), "difference") != held.keyed(
        shaded, ("x", "y"), "mask"
    )


def test_the_cache_forgets_the_oldest_rather_than_growing_forever():
    held = sampler.Closures(held=2)
    for name in "abc":
        held.put(name.encode(), name)
    assert held.get(b"a") is None
    assert held.get(b"c") == "c"


# -- the whole of it, over an inline executor ----------------------------------------


async def test_the_five_calls_answer_as_the_app_expects_them_to(page, engine):
    from rederive.web.plots import WebPlots

    plots_object = WebPlots(page, engine)
    placed = await plots_object.add(_landing("SIN(x)"))
    assert placed == plots.Placed(1, replaced=False)
    again = await plots_object.add(_landing("COS(x)", "#1"))
    assert again == plots.Placed(1, replaced=True)
    assert len(plots_object.describe()) == 1
    plots_object.shutdown()
    assert page.stopped == 1


async def test_a_solid_is_a_plot_error_in_the_words_the_message_line_prints(
    page, engine
):
    from rederive.web.plots import WebPlots

    plots_object = WebPlots(page, engine)
    with pytest.raises(PlotError, match="3D plots are not in the browser yet"):
        await plots_object.add(
            _landing("x*y", kind=PlotKind.SURFACE, variables=("x", "y"))
        )

"""Plotting in a browser, asked without one: the panes, the requests, the answers.

Three questions this file exists to ask. What a pane does when a plot lands in
it and what it then asks the engine worker for; what the worker makes of such a
request, which is where the arrays are and so where every kind either draws or
does not; and that the split holds - the main thread routes and the arrays never
come through it.

The solids are asked the same three questions and one more, which is the only
one a 3D picture has that a flat one has not: that what crosses is exactly what
`plot/surface.py` builds, since that file is what the desktop draws from too and
two programs drawing one surface differently would be two programs.

The page is faked and the runtime bridge is faked, and neither is a pretend
browser: the first records what it was told to draw and the second is the pair of
identity functions Pyodide's are on a desktop. What is real is everything
between them, which is the whole of what stage 5 wrote in Python.
"""

import asyncio
import types

import numpy as np
import pytest
from fakepage import FakeEngine, FakePage, FakeSolids, bridge

from rederive.engine.context import Angle, Context
from rederive.plot import controls
from rederive.plot import protocol as plots
from rederive.plot.model import Plot, Surface, written
from rederive.plot.protocol import PlotKind
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
def solids():
    return FakeSolids()


@pytest.fixture
def engine():
    return FakeEngine()


@pytest.fixture
def session(page, solids, engine):
    """A plot session over panes, sampling nowhere: the requests are the point."""
    executor = WorkerExecutor(engine)
    backend = WebBackend(page, solids, engine)
    page.attend(backend.handed({"landed": executor.landed}))
    solids.attend(backend.handed({"landed": executor.landed}))
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


async def test_a_pane_that_reframes_for_a_new_curve_says_it_in_pythons_words(
    session, page
):
    # The page decides that a curve arrived with none of itself in view and
    # which way it reframed; what is said about it is the sentence the desktop
    # says in the same circumstance, naming the curve as the plot list does.
    session.add(_landing("SIN(x) + 40"))
    page.panes[1].say["fitted"](1, True)
    assert page.panes[1].message == "#1  SIN(x) + 40: y autoscaled to fit"
    page.panes[1].say["fitted"](1, False)
    assert page.panes[1].message == "#1  SIN(x) + 40: autoscaled to fit"


async def test_a_legend_row_hides_a_curve_without_removing_it(session, page):
    session.add(_landing("SIN(x)"))
    pane = page.panes[1]
    pane.say["hide"](1, True)
    assert pane.plots[1]["hidden"] is True
    assert session.describe().windows[0].plots[0].hidden is True


async def test_the_menus_remove_takes_a_plot_out_for_good(session, page):
    session.add(_landing("SIN(x)"))
    pane = page.panes[1]
    # The menu is raised over the curve, which is what the entry is about, and
    # what comes back afterwards is the name of a control and nothing else.
    assert "Remove #1  SIN(x)" in _labels(pane.say["menu"](_flat(pointed=1)))
    pane.say["command"]("plot.remove", None)
    assert pane.plots == {}
    assert session.describe().windows[0].plots == ()


async def test_joining_a_data_plots_points_is_sticky(session, page):
    heard = []
    session.events = heard.append
    session.add(_landing("[[1, 2], [3, 4]]", kind=PlotKind.DATA, variables=()))
    pane = page.panes[1]
    assert _labels(pane.say["card"](_flat(pointed=1))) == [
        "Remove #1  [[1, 2], [3, 4]]",
        "Connect points",
    ]
    pane.say["command"]("points.connect", None)
    assert pane.plots[1]["connected"] is True
    assert heard[-1] == plots.Preferred(plots.Prefer(connected=True))


async def test_the_polar_toggle_rereads_every_curve_in_the_pane(session, page):
    session.add(_landing("SIN(3x)"))
    pane = page.panes[1]
    pane.say["command"]("view.polar", None)
    assert pane.plots[1]["kind"] == "polar"
    # The page is told, since the rings it draws and the r and θ it reads the
    # pointer out in follow the mode, and so is the button that shows it.
    assert (pane.polar, pane.buttons["view.polar"]) == (True, True)
    assert pane.message == "Polar: curves are read as r = f(θ)"
    pane.say["command"]("view.polar", None)
    assert pane.plots[1]["kind"] == "curve"
    assert pane.message == "Polar off: curves are read as y = f(x)"


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


# -- what a pane offers -------------------------------------------------------------
#
# The commands of a pane, their words, their keys and the order a menu stands
# them in are `plot/controls.py`'s, and a pane renders them. So the table below
# is checked in rather than derived: it is what a person would otherwise have to
# read off two files and compare by eye, and it is asserted against both sides -
# against the description the desktop's windows render, and against the menu and
# the key ladder this backend actually hands the page.


#: What a 2D window offers, command by command: the words a menu writes it with,
#: the key a desktop presses it by, the key a page presses it by, and whether
#: the browser can do it yet. A platform short of a key is a platform that has
#: none for that command - Ctrl+W closes a tab, so the page answers to the
#: letter alone - and a command that is not served is left off the browser's
#: menu rather than offered dead.
FLAT_CONTROLS = (
    ("range.set", "Set range...", "", "", False),
    ("view.all", "View all", "A", "a", True),
    ("view.home", "Home framing", "Home", "Home", True),
    ("view.back", "Back", "Backspace", "Backspace", False),
    ("trace", "Trace", "T", "t", True),
    ("grid", "Grid", "G", "g", True),
    ("legend", "Legend", "L", "l", True),
    ("image.copy", "Copy image", "Ctrl+C", "Ctrl+C", False),
    ("image.export", "Export...", "Ctrl+S", "Ctrl+S", False),
    ("clear", "Clear", "Del", "Delete", False),
    ("close", "Close", "Q", "q", True),
    ("plot.remove", "Remove", "", "", True),
    ("points.connect", "Connect points", "", "", True),
    ("points.size", "Point size", "", "", False),
)

#: And what a 3D window offers, on the same terms.
SOLID_CONTROLS = (
    ("camera.home", "Home view", "Home", "Home", True),
    ("camera.xy", "Face the xy plane", "1", "1", True),
    ("camera.xz", "Face the xz plane", "2", "2", True),
    ("camera.yz", "Face the yz plane", "3", "3", True),
    ("camera.spin", "Rotate", "R", "r", True),
    ("view.inspect", "View...", "", "v", True),
    ("image.copy", "Copy image", "Ctrl+C", "Ctrl+C", False),
    ("image.export", "Export...", "Ctrl+S", "Ctrl+S", False),
    ("surface.remove", "Remove", "", "", True),
    ("clear", "Clear", "Del", "Delete", False),
    ("close", "Close", "Q", "q", True),
)


def _flat(pointed=None, **fields):
    """The snapshot a 2D pane hands over when a menu opens, as the page builds it."""
    state = {"tracing": False, "grid": True, "legend": True, "equal": True}
    return types.SimpleNamespace(pointed=pointed, **{**state, **fields})


def _deep(pointed=None, **fields):
    """The same for a 3D pane, which has a camera where the other has a view."""
    state = {"spinning": False, "boxed": True, "legend": True}
    return types.SimpleNamespace(pointed=pointed, **{**state, **fields})


def _labels(entries):
    return [entry["label"] for entry in entries]


def _listing(entries):
    """The `Remove` entry of a 3D menu, which is a submenu of what the pane holds."""
    return next(entry for entry in entries if entry["name"] == "surface.remove")


def _described(table):
    """The table as `plot/controls.py` has it, in the shape checked in above."""
    return tuple(
        (
            one.name,
            controls.plain(one),
            (controls.keys(one) or ("",))[0],
            (controls.keys(one, page=True) or ("",))[0],
        )
        for one in controls.listed(table)
    )


@pytest.mark.parametrize(
    ("checked", "table"),
    [(FLAT_CONTROLS, controls.FLAT), (SOLID_CONTROLS, controls.SOLID)],
)
def test_neither_backend_may_move_a_word_a_key_or_a_place_by_itself(checked, table):
    # The whole of the checked-in table against the description both backends
    # render: a menu that changed its words, its keys or its order in one of
    # them changed them in the description, and the description is here.
    assert _described(table) == tuple(row[:4] for row in checked)


async def test_the_browser_serves_what_the_table_says_it_serves(session, page, solids):
    # The other half of the same claim, asked of this backend: what a pane may
    # be asked for is the served column, and a pane is handed exactly that much.
    session.add(_landing("SIN(x)"))
    session.add(_plotted("x*y", "#2"))
    for checked, pane in (
        (FLAT_CONTROLS, page.panes[1]),
        (SOLID_CONTROLS, solids.panes[2]),
    ):
        named = [row[0] for row in checked]
        offered = [one["name"] for one in pane.commands if one["name"] in named]
        assert offered == [row[0] for row in checked if row[4]]


async def test_a_pane_renders_the_menu_the_desktop_renders_less_what_it_cannot_do(
    session, page
):
    session.add(_landing("SIN(x)"))
    entries = page.panes[1].say["menu"](_flat())
    assert [(entry["name"], entry["label"]) for entry in entries] == [
        ("view.all", "View all"),
        ("view.home", "Home framing"),
        ("trace", "Trace"),
        ("grid", "Grid"),
        ("legend", "Legend"),
        ("close", "Close"),
    ]
    assert {entry["name"] for entry in entries} <= {
        row[0] for row in FLAT_CONTROLS if row[4]
    }
    # The keys are the page's own spelling of the same bindings, and the rules
    # fall around what is left: a group the browser can serve one entry of has
    # one entry under its rule.
    assert [entry["keys"][0] for entry in entries] == ["a", "Home", "t", "g", "l", "q"]
    assert [entry["name"] for entry in entries if entry["separated"]] == [
        "trace",
        "close",
    ]


async def test_a_menu_is_read_off_the_pane_as_it_opens(session, page):
    session.add(_landing("SIN(x)"))
    pane = page.panes[1]

    def ticked(state):
        return [entry["label"] for entry in pane.say["menu"](state) if entry["checked"]]

    assert ticked(_flat()) == ["Grid", "Legend"]
    assert ticked(_flat(tracing=True, grid=False)) == ["Trace", "Legend"]


async def test_a_control_the_page_names_reaches_the_thing_it_names(session, page):
    session.add(_landing("SIN(x)"))
    pane = page.panes[1]
    for name in ("view.all", "view.home", "trace", "grid", "legend"):
        pane.say["command"](name, None)
    assert pane.done == ["autoscale", "home", "trace", "grid", "legend"]
    # The two zooms are keys with no entry, and the factor is `plot/actions.py`'s
    # rather than the page's.
    pane.say["command"]("view.zoom.in", None)
    pane.say["command"]("view.zoom.out", None)
    assert pane.done[-2:] == [("zoom", 0.5), ("zoom", 2.0)]
    # A name the pane has nothing for is a menu somebody left open, and does
    # nothing at all rather than raising into a click handler.
    pane.say["command"]("image.export", None)
    assert len(pane.done) == 7


async def test_a_solid_pane_offers_the_camera_and_what_it_holds(session, solids):
    session.add(_plotted("SIN(x y)"))
    session.add(_plotted("x^2 - y^2", "#2"))
    pane = solids.panes[1]
    entries = pane.say["menu"](_deep())
    assert _labels(entries) == [
        "Home view",
        "Face the xy plane",
        "Face the xz plane",
        "Face the yz plane",
        "Rotate",
        "View...",
        "Remove",
        "Close",
    ]
    # The submenu lists what the pane holds, in the order it holds them, and an
    # item names its surface by where it stands in that list.
    listing = _listing(entries)
    assert _labels(listing["items"]) == ["#1  SIN(x y)", "#2  x^2 - y^2"]
    pane.say["command"]("surface.remove", listing["items"][1]["value"])
    assert _labels(_listing(pane.say["menu"](_deep()))["items"]) == ["#1  SIN(x y)"]


async def test_the_camera_a_preset_looks_from_is_the_desktops(session, solids):
    session.add(_plotted("SIN(x y)"))
    pane = solids.panes[1]
    # A fresh pane opens on the camera `plot/actions.py` names, and Home view
    # returns to it; a preset turns the camera and leaves it standing where it
    # was, which is what a distance of zero means.
    assert pane.camera == (30.0, -60.0, 23.0)
    pane.say["command"]("camera.xz", None)
    assert pane.camera == (0.0, -90.0, 0.0)
    assert pane.message == "Facing the xz plane"
    pane.say["command"]("camera.home", None)
    assert pane.camera == (30.0, -60.0, 23.0)
    # And what a turning pane says names the key that stops it in the page's own
    # spelling of it, which is not the desktop's.
    pane.say["command"]("camera.spin", None)
    assert pane.spinning == "Rotating - r stops it"


async def test_a_solids_legend_row_offers_the_menu_about_that_surface(session, solids):
    session.add(_plotted("SIN(x y)"))
    pane = solids.panes[1]
    # `Draw as wire mesh` is the per-surface exception and is not served yet, so
    # the row offers the one thing that is rather than an entry that would do
    # nothing.
    assert _labels(pane.say["card"](_deep(pointed=1))) == ["Remove #1  SIN(x y)"]
    pane.say["command"]("plot.remove", None)
    assert pane.plots == {}


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


async def test_the_five_calls_answer_as_the_app_expects_them_to(page, solids, engine):
    from rederive.web.plots import WebPlots

    plots_object = WebPlots(page, solids, engine)
    placed = await plots_object.add(_landing("SIN(x)"))
    assert placed == plots.Placed(1, replaced=False)
    again = await plots_object.add(_landing("COS(x)", "#1"))
    assert again == plots.Placed(1, replaced=True)
    assert len(plots_object.describe()) == 1
    plots_object.shutdown()
    assert page.stopped == 1


async def test_a_solid_opens_a_pane_of_its_own_beside_the_flat_ones(
    page, solids, engine
):
    from rederive.web.plots import WebPlots

    plots_object = WebPlots(page, solids, engine)
    await plots_object.add(_landing("SIN(x)"))
    await plots_object.add(_landing("x*y", kind=PlotKind.SURFACE, variables=("x", "y")))
    assert list(page.panes) == [1]
    assert list(solids.panes) == [2]
    assert [one.kind for one in plots_object.describe()] == ["2D", "3D"]


# -- a pane a solid stands in ------------------------------------------------------


def _solid(text, label="#1", variables=("x", "y"), context=None):
    """One surface as the worker is sent one."""
    return Surface(
        worksheet=1,
        label=label,
        text=text,
        kind=PlotKind.SURFACE,
        node=parsed(text),
        context=context or Context(),
        options=plots.Options(variables=variables),
    )


def _grid(text, grid=(16, 16), domain=(-5.0, 5.0), zrange=None, **keywords):
    return asking.Grid(
        pane=1,
        plot=1,
        generation=1,
        model=_solid(text, **keywords),
        xdomain=domain,
        ydomain=domain,
        grid=grid,
        zrange=zrange,
    )


def _standing(request):
    """One evaluation, run where the worker would run it."""
    table = sampler.methods(lambda _partial: None)
    return table[asking.GRID](request)


def _plotted(text, label="#1"):
    return _landing(text, label=label, kind=PlotKind.SURFACE, variables=("x", "y"))


async def test_a_surface_opens_a_pane_of_its_own_and_lands_in_it(session, solids, page):
    assert session.add(_plotted("SIN(x y)")) == plots.Placed(1)
    assert page.panes == {}
    pane = solids.panes[1]
    assert pane.presented == 1
    assert pane.title == "SIN(x y) - Rederive 3D plot (current)"
    assert [spec["name"] for spec in pane.plots.values()] == ["#1  SIN(x y)"]
    # The solid palette, which is the curve palette turned by one: a white
    # surface is a white shape with a white shape behind it.
    assert [spec["color"] for spec in pane.plots.values()] == ["#ffff55"]


async def test_a_surface_is_asked_for_over_the_domain_the_pane_opens_on(
    session, solids, engine
):
    session.add(_plotted("SIN(x y)"))
    await settle()
    _number, method, request = engine.sent[0]
    assert method == asking.GRID
    assert request.xdomain == (-5.0, 5.0)
    assert request.ydomain == (-5.0, 5.0)
    assert request.grid == (64, 64)
    # Alone in the picture, the surface says what the box is rather than being
    # told - which is what makes one surface cost one evaluation.
    assert request.zrange is None
    assert solids.panes[1].started == [(1, 1)]


async def test_the_grid_a_pane_opens_on_is_the_sticky_one(session, engine):
    session.prefer(plots.Prefer(grid=24))
    session.add(_plotted("SIN(x y)"))
    await settle()
    assert engine.sent[0][2].grid == (24, 24)


async def test_a_typed_grid_evaluates_again_and_the_next_pane_opens_on_it(
    session, solids, engine
):
    heard = []
    session.events = heard.append
    session.add(_plotted("SIN(x y)"))
    await drain(page_of(solids), engine)
    engine.sent.clear()
    solids.panes[1].say["framed"]("-2", "2", "-2", "2", "32", "32")
    await settle()
    assert heard[-1] == plots.Preferred(plots.Prefer(grid=32))
    request = engine.sent[0][2]
    assert request.xdomain == (-2.0, 2.0)
    assert request.grid == (32, 32)
    assert solids.panes[1].message == "Evaluating over the new domain at 32 by 32"
    # And `Describe` reports the domain, which is what a 3D window is framed by.
    assert session.describe().windows[0].xrange == (-2.0, 2.0)


async def test_a_domain_field_that_will_not_read_is_put_back(session, solids, engine):
    session.add(_plotted("SIN(x y)"))
    await drain(page_of(solids), engine)
    engine.sent.clear()
    pane = solids.panes[1]
    pane.say["framed"]("-π", "5", "-5", "5", "64", "64")
    assert pane.message == "The domain is four numbers, the grid two"
    assert pane.fields == ("-5", "5", "-5", "5", "64", "64")
    pane.say["framed"]("5", "-5", "-5", "5", "64", "64")
    assert pane.message == "The domain runs from a lower bound to a higher one"
    await settle()
    assert engine.sent == []


async def test_the_mesh_box_flips_every_surface_and_is_sticky(session, solids):
    heard = []
    session.events = heard.append
    session.add(_plotted("SIN(x y)"))
    session.add(_plotted("x^2 - y^2", "#2"))
    pane = solids.panes[1]
    assert [spec["wire"] for spec in pane.plots.values()] == [True, True]
    pane.say["command"]("mesh", None)
    assert [spec["wire"] for spec in pane.plots.values()] == [False, False]
    assert pane.buttons["mesh"] is False
    assert heard[-1] == plots.Preferred(plots.Prefer(wire=False))


async def test_a_legend_row_hides_a_surface_without_removing_it(session, solids):
    session.add(_plotted("SIN(x y)"))
    solids.panes[1].say["hide"](1, True)
    assert solids.panes[1].plots[1]["hidden"] is True
    assert session.describe().windows[0].plots[0].hidden is True


async def test_one_surface_in_a_pane_is_drawn_in_the_box_it_asked_for(
    session, solids, engine
):
    session.add(_plotted("SIN(x y)"))
    await settle()
    engine.sent.clear()
    # What the page says when it has drawn: the extent the values wanted, and
    # the box they were placed in. The same, for a surface standing alone.
    solids.panes[1].say["stood"](1, [-1.0, 1.0], [-1.0, 1.0])
    await settle()
    assert engine.sent == []


async def test_two_surfaces_are_drawn_in_one_box(session, solids, engine):
    session.add(_plotted("SIN(x y)"))
    session.add(_plotted("x^2 - y^2", "#2"))
    pane = solids.panes[1]
    await settle()
    pane.say["stood"](1, [-1.0, 1.0], [-1.0, 1.0])
    await answered(solids, engine)
    at = len(engine.sent)
    # The second surface reaches higher, so the box grows and every mesh built
    # to the box that is gone - both of them - is asked for again in the new one.
    pane.say["stood"](2, [-25.0, 25.0], [-1.0, 1.0])
    await answered(solids, engine)
    asked = engine.sent[at:]
    assert asked and all(one[2].zrange == (-25.0, 25.0) for one in asked)


async def answered(solids, engine):
    """Acknowledge the evaluation with the worker, which sends the next one."""
    await settle()
    if engine.sent:
        solids.handlers["landed"](engine.sent[-1][0], "")
    await settle()


async def test_a_surface_that_would_not_evaluate_reports_itself(
    session, solids, engine
):
    heard = []
    session.events = heard.append
    session.add(_plotted("SIN(x y)"))
    await settle()
    solids.handlers["landed"](engine.sent[-1][0], "the surface would not evaluate")
    assert heard == [plots.Trouble(1, "#1", "the surface would not evaluate")]


def page_of(solids):
    """The one handler a drain needs, which for a solid is the same table."""
    return _Landing(solids)


class _Landing:
    """Enough of a page for `drain`: the acknowledgement, and nothing else."""

    def __init__(self, solids):
        self.handlers = solids.handlers


# -- what the worker makes of a surface ---------------------------------------------


def _geometry(text, grid=(16, 16), domain=(-5.0, 5.0), zrange=None):
    """The same surface, built the way the desktop's window builds one."""
    from rederive.plot import evaluate
    from rederive.plot.surface import Box, brightened, extent, mesh, wire

    node = parsed(text)
    closure = evaluate.closure(node, Context(), ("x", "y"))
    xs, ys, values = evaluate.grid_eval(closure, domain, domain, *grid)
    boundary = evaluate.grid_boundary(closure, xs, ys, values)
    found = extent([values])
    box = Box(domain, domain, zrange or found[0])
    vertexes, faces, shading = mesh(xs, ys, values, box, boundary)
    points, shades = wire(xs, ys, values, box, boundary)
    return {
        "box": box,
        "clipped": found[1],
        "wanted": found[0],
        "vertices": vertexes,
        "faces": faces,
        "colors": brightened(shading, "#ffff55"),
        "wire": points,
        "wirecolors": brightened(shades, "#ffff55"),
    }


def test_a_surface_crosses_as_the_geometry_plot_surface_builds():
    answer = _standing(_grid("SIN(x y)"))
    built = _geometry("SIN(x y)")
    assert np.array_equal(answer["vertices"], built["vertices"].ravel())
    assert np.array_equal(answer["faces"], built["faces"].ravel())
    assert np.array_equal(answer["colors"], built["colors"][:, :3].ravel())
    assert np.array_equal(answer["wire"], built["wire"].ravel())
    assert answer["vertices"].dtype == np.float32
    assert answer["faces"].dtype == np.uint32
    assert answer["vertices"].ndim == 1 and answer["vertices"].flags["C_CONTIGUOUS"]


def test_a_surfaces_box_is_the_one_the_desktops_inspector_reads():
    answer = _standing(_grid("SIN(x y)"))
    box = _geometry("SIN(x y)")["box"]
    assert answer["center"] == pytest.approx(list(box.center))
    assert answer["lengths"] == pytest.approx(list(box.lengths))
    assert answer["height"] == pytest.approx(box.height)
    assert answer["world"] == 10.0
    # And the numbers arrive spelled, in the function the desktop's fields and
    # tick labels are spelled by: the page shows what Python wrote.
    assert answer["reading"]["lengths"] == [written(one) for one in box.lengths]
    assert answer["reading"]["lengths"][:2] == ["10", "10"]


def test_a_grid_of_n_by_n_is_n_squared_vertices_and_twice_the_faces():
    answer = _standing(_grid("x + y", grid=(8, 8)))
    assert answer["vertices"].size == 3 * 8 * 8
    assert answer["faces"].size == 3 * 2 * 7 * 7


def test_a_dome_stops_where_it_stops_rather_than_a_grid_step_short():
    # `SQRT(1-x^2-y^2)` is a dome over the unit disc and nothing outside it, so
    # the mesh has fewer faces than a full grid and every vertex of it is in the
    # box - which is what the refined boundary is for.
    answer = _standing(_grid("SQRT(1 - x^2 - y^2)", grid=(16, 16), domain=(-2.0, 2.0)))
    assert 0 < answer["faces"].size < 3 * 2 * 15 * 15
    assert answer["trouble"] == ""


def test_a_spike_clips_the_box_to_the_percentiles_and_says_so():
    answer = _standing(_grid("1/(x^2 + y^2)", grid=(32, 32)))
    assert answer["clipped"] is True
    assert answer["words"].startswith("z clipped to the 1st-99th percentile")


def test_a_surface_placed_in_a_box_it_was_given_asks_for_nothing():
    # The box a pane holding two surfaces hands down: the mesh stands in it, and
    # what the values themselves wanted is reported beside it rather than used.
    answer = _standing(_grid("SIN(x y)", zrange=(-4.0, 4.0)))
    assert answer["zrange"] == pytest.approx([-4.0, 4.0])
    assert answer["wanted"] == pytest.approx([-1.0, 1.0], abs=1e-4)
    assert answer["clipped"] is False


def test_a_surface_with_nothing_real_over_the_domain_says_so():
    answer = _standing(_grid("SQRT(-1 - x^2 - y^2)"))
    assert answer["empty"] is True
    assert answer["words"] == "#1: no real values over this domain"
    assert answer["vertices"].size == 0


def test_a_surface_that_will_not_evaluate_answers_with_what_went_wrong():
    answer = _standing(_grid("1/0 + x y"))
    assert answer["trouble"]
    assert "vertices" not in answer


def test_a_surfaces_closure_is_cached_by_content_like_a_curves():
    held = sampler.Closures()
    first = sampler._standing(held, _solid("SIN(x y)"))
    assert first.closure is None
    sampler._gridded(held, _grid("SIN(x y)"))
    assert sampler._standing(held, _solid("SIN(x y)")).closure is not None

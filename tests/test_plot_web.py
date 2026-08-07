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
import re
import types
from pathlib import Path

import numpy as np
import pytest
from fakepage import (
    FakeEngine,
    FakePage,
    FakePane,
    FakeSolid,
    FakeSolids,
    bridge,
    jsnull,
)

from rederive.engine.context import Angle, Context
from rederive.plot import controls
from rederive.plot import protocol as plots
from rederive.plot.model import Plot, Surface, written
from rederive.plot.protocol import PlotKind
from rederive.plot.session import PlotSession
from rederive.plot.web import protocol as asking
from rederive.plot.web import backend, sampler
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
    # Two colors, as the desktop's windows hold: the second is what the same
    # curve is drawn in on the white ground a picture leaving the pane is drawn
    # on. Both are `plot/model.py`'s, so a plot copied out of a browser is the
    # color it is copied out of a window in - and a white curve is black there,
    # white on white paper being an empty picture.
    assert spec["paper"] == "#000000"


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
    ("range.set", "Set range...", "", "", True),
    ("view.all", "View all", "A", "a", True),
    ("view.home", "Home framing", "Home", "Home", True),
    ("view.back", "Back", "Backspace", "Backspace", True),
    ("trace", "Trace", "T", "t", True),
    ("grid", "Grid", "G", "g", True),
    ("legend", "Legend", "L", "l", True),
    ("image.copy", "Copy image", "Ctrl+C", "Ctrl+C", True),
    ("image.export", "Export...", "Ctrl+S", "Ctrl+S", True),
    ("clear", "Clear", "Del", "Delete", True),
    ("close", "Close", "Q", "q", True),
    ("plot.remove", "Remove", "", "", True),
    ("points.connect", "Connect points", "", "", True),
    ("points.size", "Point size", "", "", True),
)

#: And what a 3D window offers, on the same terms.
SOLID_CONTROLS = (
    ("camera.home", "Home view", "Home", "Home", True),
    ("camera.xy", "Face the xy plane", "1", "1", True),
    ("camera.xz", "Face the xz plane", "2", "2", True),
    ("camera.yz", "Face the yz plane", "3", "3", True),
    ("camera.spin", "Rotate", "R", "r", True),
    ("view.inspect", "View...", "", "v", True),
    ("image.copy", "Copy image", "Ctrl+C", "Ctrl+C", True),
    ("image.export", "Export...", "Ctrl+S", "Ctrl+S", True),
    ("surface.remove", "Remove", "", "", True),
    ("clear", "Clear", "Del", "Delete", True),
    ("close", "Close", "Q", "q", True),
)


#: The entries a canvas menu only has when the click was over a plot, which is
#: what the tail of a menu is about and what a menu raised over nothing is short
#: of.
_POINTED = ("plot.remove", "points.connect", "points.size")


def _flat(pointed=jsnull, **fields):
    """The snapshot a 2D pane hands over when a menu opens, as the page builds it.

    A click over no curve is the page's own `null` rather than Python's `None`:
    the snapshot is built in JavaScript, it has the field, and what it has put
    in the field is nothing. Pyodide hands the two over as two different
    things, so a test that said `None` here would be asking the seam an easier
    question than the page asks it.
    """
    state = {"tracing": False, "grid": True, "legend": True, "equal": True}
    return types.SimpleNamespace(pointed=pointed, **{**state, **fields})


def _deep(pointed=jsnull, **fields):
    """The same for a 3D pane, which has a camera where the other has a view."""
    state = {"spinning": False, "boxed": True, "names": True, "legend": True}
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


async def test_a_pane_renders_the_menu_the_desktop_renders(session, page):
    session.add(_landing("SIN(x)"))
    entries = page.panes[1].say["menu"](_flat())
    assert [(entry["name"], entry["label"]) for entry in entries] == [
        ("range.set", "Set range..."),
        ("view.all", "View all"),
        ("view.home", "Home framing"),
        ("view.back", "Back"),
        ("trace", "Trace"),
        ("grid", "Grid"),
        ("legend", "Legend"),
        ("image.copy", "Copy image"),
        ("image.export", "Export..."),
        ("clear", "Clear"),
        ("close", "Close"),
    ]
    # Which is now the whole of the menu the desktop offers a window holding a
    # curve: nothing is left off, so the two lists are one list.
    assert [entry["name"] for entry in entries] == [
        row[0] for row in FLAT_CONTROLS if row[4] and row[0] not in _POINTED
    ]
    # The keys are the page's own spelling of the same bindings, and the rules
    # fall where the description puts them.
    assert [entry["keys"][0] if entry["keys"] else "" for entry in entries] == [
        "", "a", "Home", "Backspace", "t", "g", "l", "Ctrl+C", "Ctrl+S", "Delete", "q",
    ]
    assert [entry["name"] for entry in entries if entry["separated"]] == [
        "trace",
        "image.copy",
    ]


async def test_a_menu_is_read_off_the_pane_as_it_opens(session, page):
    session.add(_landing("SIN(x)"))
    pane = page.panes[1]

    def ticked(state):
        return [entry["label"] for entry in pane.say["menu"](state) if entry["checked"]]

    assert ticked(_flat()) == ["Grid", "Legend"]
    assert ticked(_flat(tracing=True, grid=False)) == ["Trace", "Legend"]


async def test_a_menu_raised_over_nothing_is_a_menu_about_the_view(
    session, page, solids
):
    """The page's `null` is nothing, in either kind of pane.

    A right click on empty canvas is the ordinary way a menu is asked for, and
    what the snapshot then carries where a plot would be is the page's own word
    for having none. Read as an answer rather than as nothing it is a serial to
    be looked up, and looking it up is where a menu stops being a menu - so
    both kinds are asked here, both being handed the same word.
    """
    session.add(_landing("SIN(x)"))
    session.add(_plotted("SIN(x y)"))
    flat, deep = page.panes[1], solids.panes[2]
    assert _labels(flat.say["menu"](_flat()))[0] == "Set range..."
    assert _labels(deep.say["menu"](_deep()))[0] == "Home view"
    # And the tail that is about one plot is absent rather than about a plot
    # nobody pointed at, which is the same answer as for a field with nothing
    # in it at all.
    assert "Remove #1  SIN(x)" not in _labels(flat.say["menu"](_flat()))
    assert _labels(flat.say["card"](_flat())) == []
    assert _labels(deep.say["card"](_deep())) == []


async def test_a_control_the_page_names_reaches_the_thing_it_names(session, page):
    session.add(_landing("SIN(x)"))
    pane = page.panes[1]
    for name in ("view.all", "trace", "grid", "legend", "image.copy", "image.export"):
        # With the value the page sends where a control carries none, which is
        # `null` and not an omitted argument.
        pane.say["command"](name, jsnull)
    assert pane.done == [
        "autoscale", "trace", "grid", "legend", "copy", "export"
    ]
    # The two zooms are keys with no entry, and the factor is `plot/actions.py`'s
    # rather than the page's.
    pane.say["command"]("view.zoom.in", None)
    pane.say["command"]("view.zoom.out", None)
    assert pane.done[-2:] == [("zoom", 0.5), ("zoom", 2.0)]
    # A name the pane has nothing for is a menu somebody left open, and does
    # nothing at all rather than raising into a click handler.
    pane.say["command"]("nothing.at.all", None)
    assert len(pane.done) == 8


async def test_home_is_the_framing_the_desktop_opens_on_with_the_lock_back_on(
    session, page
):
    session.add(_landing("SIN(x)"))
    pane = page.panes[1]
    pane.say["command"]("view.home", None)
    # `plot/view.py` says what a fresh window shows and `plot/actions.py` says
    # that Home relocks the scales, and the pane is told both rather than
    # working either out.
    assert pane.done[-1] == ("reframe", (-5.0, 5.0, -4.0, 4.0), True)


async def test_view_all_says_which_of_the_two_things_happened(session, page):
    session.add(_landing("SIN(x)"))
    pane = page.panes[1]
    pane.say["command"]("view.all", None)
    assert pane.message == "Framed on everything drawn"
    # The framing is the page's, since the rectangle each curve wants came off
    # the worker with its samples; the sentence about it is Python's, and there
    # is one for the case where there was nothing to frame.
    pane.framable = False
    pane.say["command"]("view.all", None)
    assert pane.message == "Nothing to frame yet"


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
        "Copy image",
        "Export...",
        "Remove",
        "Clear",
        "Close",
    ]
    # Which is the whole of the menu the desktop offers a window holding a
    # surface: nothing is left off, so the two lists are one list.
    assert [entry["name"] for entry in entries] == [
        row[0] for row in SOLID_CONTROLS if row[4]
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
    # The per-surface exception first and what removes the surface second, in
    # the words and the order the desktop's legend offers them.
    assert _labels(pane.say["card"](_deep(pointed=1))) == [
        "Draw solid",
        "Remove #1  SIN(x y)",
    ]
    pane.say["command"]("plot.remove", None)
    assert pane.plots == {}


async def test_one_surfaces_wire_is_its_own_exception_and_not_the_panes(
    session, solids
):
    heard = []
    session.events = heard.append
    session.add(_plotted("SIN(x y)"))
    session.add(_plotted("x^2 - y^2", "#2"))
    pane = solids.panes[1]
    pane.say["command"]("mesh", None)
    assert [spec["wire"] for spec in pane.plots.values()] == [False, False]
    heard.clear()
    # The row menu flips one surface and nothing else: the `mesh` button stands
    # where it was and nothing goes back to the sticky preferences, since an
    # exception is not what the next surface should arrive as.
    pane.say["card"](_deep(pointed=1))
    assert _labels(pane.say["card"](_deep(pointed=1))) == [
        "Draw as wire mesh",
        "Remove #1  SIN(x y)",
    ]
    pane.say["command"]("surface.wire", None)
    assert [spec["wire"] for spec in pane.plots.values()] == [True, False]
    assert pane.buttons["mesh"] is False
    assert heard == []
    # And the entry then reads the other way round, off the surface itself.
    assert _labels(pane.say["card"](_deep(pointed=1)))[0] == "Draw solid"


async def test_del_clears_a_solid_pane_and_takes_the_axis_names_with_it(
    session, solids
):
    session.add(_plotted("SIN(x y)"))
    pane = solids.panes[1]
    assert pane.axes == ("x", "y", "z")
    pane.say["command"]("clear", None)
    assert pane.plots == {}
    assert session.describe().windows[0].plots == ()


async def test_the_axes_of_a_solid_are_named_by_the_first_surfaces_expression(
    session, solids
):
    # A pane with nothing in it is told the three letters every reader supplies
    # anyway; a surface names them after its own variables, and the vertical one
    # after whatever the expression called it.
    session.add(_landing("u^2 + v^2", kind=PlotKind.SURFACE, variables=("u", "v")))
    assert solids.panes[1].axes == ("u", "v", "z")


async def test_a_solids_picture_leaves_the_pane_and_says_what_happened(
    session, solids
):
    session.add(_plotted("SIN(x y)"))
    pane = solids.panes[1]
    for name in ("image.copy", "image.export"):
        pane.say["command"](name, None)
    assert pane.done[-2:] == ["copy", "export"]
    # And the sentences are the flat pane's own, since what they are about is a
    # browser rather than a picture.
    pane.say["copied"]("", "")
    assert pane.message == "Copied the plot to the clipboard"
    pane.say["copied"]("", "NotAllowedError: Write permission denied")
    assert pane.message == (
        "The browser did not allow the clipboard:"
        " NotAllowedError: Write permission denied"
    )
    pane.say["exported"]("plot1.png", 1520, 1240, "")
    assert pane.message == "Downloaded plot1.png, 1520 by 1240 pixels"


# -- the two files the page draws with -----------------------------------------------
#
# Nothing here runs any JavaScript. What it reads is the source of the page's
# two plotting modules, for the one thing about a pane that decides whether the
# seam works at all and that no amount of Python can see from this side: a pane
# is an object, everything Python asks of one it asks by name, and a field of
# that name is that name. `this.named = true` written beside `named(across,
# along, up)` is a pane whose axes cannot be told what they are called, and the
# fake page cannot show it - the fake is Python, where the same collision is
# spelled differently and where a fake that had it would be a fake nobody would
# write.

#: Where the page's own files are, which is beside the tests rather than in
#: the package: they are the page, and the package is what the page loads.
PAGE = Path(__file__).resolve().parent.parent / "web"

#: And where the desktop's windows are, for the one thing a pane has to agree
#: with them about that neither side can be asked for without a toolkit.
WINDOWS = Path(__file__).resolve().parent.parent / "src" / "rederive" / "plot" / "qt"

#: Which module draws which kind of pane, under what name on either side of the
#: seam. The backend's class and the page's carry the same name because they
#: are the two halves of one thing.
DRAWN = (("plot2d.js", "Pane", FakePane), ("plot3d.js", "Solid", FakeSolid))


def _drawing(module, named):
    """One class of one of the page's modules, as its own run of the file."""
    source = (PAGE / module).read_text(encoding="utf-8")
    start = source.index(f"\nclass {named} {{\n")
    return source[start : source.index("\n}\n", start)]


def _methods(body):
    """The methods a class declares, which are its members at one indent."""
    return set(re.findall(r"^  (?:async )?([A-Za-z_$][\w$]*)\(", body, re.M))


def _fields(body):
    """The fields it puts on itself, wherever in it that is done."""
    return set(re.findall(r"this\.([A-Za-z_$][\w$]*)\s*=[^=]", body))


def _asked(named):
    """Every call one backend pane makes into the page it draws through."""
    source = Path(backend.__file__).read_text(encoding="utf-8")
    start = source.index(f"\nclass {named}:\n")
    body = source[start : source.find("\n\nclass ", start + 1)]
    return set(re.findall(r"self\.page\.([A-Za-z_]\w*)\(", body))


@pytest.mark.parametrize("module, named, fake", DRAWN)
def test_no_field_of_a_pane_carries_the_name_of_one_of_its_methods(module, named, fake):
    """A pane's own bookkeeping may not be spelled like anything it does.

    Python reaches a pane's methods by name and JavaScript has one namespace
    for both, so a field assigned in the constructor silently replaces the
    method it is named after - and the call that finds it says only that a
    boolean is not callable, from the far side of a proxy.
    """
    body = _drawing(module, named)
    assert _methods(body) & _fields(body) == set()


@pytest.mark.parametrize("module, named, fake", DRAWN)
def test_every_call_the_backend_makes_on_a_pane_is_one_the_pane_answers(
    module, named, fake
):
    """And the seam is the same shape on all three sides of it.

    The page is what a browser calls, the fake is what these tests call, and a
    name the backend uses that either is short of is a call that would go
    nowhere - in a browser, or in every test at once.
    """
    called = _asked(named)
    assert called, "the backend calls into a pane by name, so there is a list"
    assert called <= _methods(_drawing(module, named))
    assert called <= {name for name in dir(fake) if not name.startswith("_")}


# -- the things about a pane that only its own source says -------------------------
#
# Still no JavaScript. What is read below is the shape of a handful of rules that
# hold nowhere else: a page hands the keyboard to the document after a press on
# anything it cannot focus, a wheel is turned by a distance rather than a number
# of times, what a fresh pane opens showing follows from how tall it stands, and
# a marker on a function follows the pointer. None of them is visible from Python
# - the fake page has no DOM, no pointer, no focus and no size - and each has
# already been got wrong once, so what can be asserted about them is asserted
# here.


def _handler(source, opening):
    """One event handler of a page module, from its listener to its brace."""
    body = source[source.index(opening) :]
    return body[: body.index("});")]


def _method(body, named):
    """One method of a page's class, from its name to the brace that closes it."""
    start = body.index(f"\n  {named}(")
    return body[start : body.index("\n  }\n", start)]


def test_a_press_on_a_menu_entry_is_cancelled_so_a_command_keeps_the_keyboard():
    """A command that raises a dialog must be left holding what it focused.

    The browser's answer to a press on something it cannot focus - a menu
    entry, the space beside a field - is to hand the keyboard to the document,
    and it does it after the handler has run. So a dialog raised from a menu
    entry has its first field taken off it again unless the press was
    cancelled, and an overlay whose keys go to the document behind it answers
    to neither Enter nor Escape.
    """
    controls_js = (PAGE / "controls.js").read_text(encoding="utf-8")
    assert "press.preventDefault();" in _handler(
        controls_js, "item.addEventListener('pointerdown'"
    )
    forms_js = (PAGE / "forms.js").read_text(encoding="utf-8")
    assert "event.preventDefault();" in _handler(
        forms_js, "overlay.addEventListener('pointerdown'"
    )


@pytest.mark.parametrize("module", ("plot2d.js", "plot3d.js"))
def test_a_tool_button_takes_the_keyboard_back_before_its_command_runs(module):
    """And the same rule for the buttons, which a page can focus.

    A button keeps the keyboard once it is pressed, so a pane has to take it
    back; taking it back after the command has run is what leaves `view...`
    raising an inspector that cannot be typed into.
    """
    source = (PAGE / module).read_text(encoding="utf-8")
    run = _handler(source, "controls.bar(this.tools")
    assert run.index("this.element.focus()") < run.index("this.say.command(")


def test_the_wheel_zooms_by_how_far_it_was_turned_rather_than_once_per_event():
    """A trackpad is a wheel that reports a stream of very small turns.

    A factor applied whole per event is right for a mouse, which fires one
    event per detent, and compounds into a zoom of several hundred for a
    trackpad. So the factor is raised to the distance scrolled, in whichever of
    the three units the page reported it in, and one turn of a wheel comes to
    about what one turn of a wheel comes to on a desktop - where pyqtgraph
    scales by 1.02 raised to the wheel's angle in eighths of a degree, and a
    detent is fifteen degrees of it.
    """
    source = (PAGE / "plot2d.js").read_text(encoding="utf-8")
    listener = _handler(source, "over.addEventListener('wheel'")
    assert "wheeled(event)" in listener
    turned = source[source.index("function wheeled(event)") :]
    turned = turned[: turned.index("\n}\n")]
    assert "event.deltaY" in turned and "event.deltaMode" in turned
    factor = float(re.search(r"const WHEEL_FACTOR = ([\d.]+);", source).group(1))
    assert 0.8 < factor / 1.02**15 < 1.25


def test_a_fresh_pane_opens_no_flatter_than_a_fresh_window():
    """Because what a fresh view shows follows from the shape of the picture.

    x runs from -5 to 5 and the ordinate follows from equal scales, so two
    programs whose pictures are different shapes open on different rectangles -
    which `plot/view.py` says is two programs. A pane has to stand taller than
    a window for its width rather than merely as tall, since it spends more of
    itself on chrome: a title bar of its own, and wider gutters for the numbers
    along the edges. How much taller only a browser can measure; that it is at
    least as tall is measurable here.
    """
    source = (PAGE / "plot2d.js").read_text(encoding="utf-8")
    pane = tuple(
        int(re.search(rf"const PANE_{name} = (\d+);", source).group(1))
        for name in ("WIDTH", "HEIGHT")
    )
    # By path rather than by import: this file is the browser's half of the
    # tests and runs where no toolkit is installed.
    window = (WINDOWS / "window2d.py").read_text(encoding="utf-8")
    opens = re.search(r"self\.resize\((\d+), (\d+)\)", window)
    assert opens is not None, "the desktop window opens at a size, and this reads it"
    assert pane[1] / pane[0] >= int(opens[2]) / int(opens[1])


def test_a_marker_on_a_function_follows_the_pointer_and_one_on_a_curve_does_not():
    """Which is what makes trace feel like pointing at the curve.

    A marker riding a parametric curve stays where it is - its place is a
    parameter value and the pointer's x is not one - and a marker on a function
    goes where the pointer is. `Window2D._moved` is the same rule on the
    desktop, and a pane that gave up as soon as a marker was up would be a pane
    whose trace mode answered only to the keyboard.

    The pointer readout gives way to the marker, because a pane has one line to
    say things on where the window has two. What a person reads while a marker
    is up is the curve's own value, and a readout would write over it.
    """
    moved = _method(_drawing("plot2d.js", "Pane"), "_pointed")
    assert "if (this.tracing !== null) return;" not in moved
    riding = moved.index("this.tracing !== null")
    assert "PARAMETRIZED.has(plot.kind)" in moved[riding:]
    assert "this.traceAt = at.x" in moved[riding:]
    assert "this._retrace()" in moved[riding:]
    assert moved.index("this.said(") > moved.index("return;", riding)


def test_the_page_calls_the_same_two_kinds_parametrized_that_the_protocol_does():
    """One vocabulary of kinds, as there is one spelling of a control.

    Which kinds are ridden at a parameter rather than at an abscissa decides
    what the pointer does, what an arrow key steps by and what a click snaps to.
    A page that read a kind of its own into that list would be a second answer
    to a question `plot/protocol.py` has already answered.
    """
    source = (PAGE / "plot2d.js").read_text(encoding="utf-8")
    named = re.search(r"const PARAMETRIZED = new Set\(\[([^\]]*)\]\);", source)
    assert named is not None, "the page names the parametrized kinds in one place"
    spelled = {
        re.search(rf"const {name.strip()} = '([^']+)';", source).group(1)
        for name in named.group(1).split(",")
    }
    assert spelled == {str(kind) for kind in plots.PARAMETRIZED}


# -- the commands that are more than a menu entry ----------------------------------
#
# Everything below is a 2D command with something behind it: a history, a form,
# a choice, a picture leaving the pane. What they have in common is where the
# work is divided - the view stays on the page, the words and the arithmetic
# stay in Python, and what a person typed is read by the engine and by nothing
# else - so what is asked of each is that it crossed the seam the right way.


async def test_backspace_steps_back_through_where_the_view_has_been(session, page):
    session.add(_landing("SIN(x)"))
    pane = page.panes[1]
    # A gesture says where the view stood before it moved it, and that is the
    # whole of what a drag tells this side. The history is Python's, because
    # `plot/view.py` already says what a step through one comes to.
    pane.say["remembered"](-5.0, 5.0, -4.0, 4.0)
    pane.shown = [-1.0, 1.0, -1.0, 1.0, 800.0, 640.0]
    pane.say["command"]("view.back", None)
    assert pane.done[-1] == ("reframe", (-5.0, 5.0, -4.0, 4.0), False)
    pane.say["command"]("view.forward", None)
    assert pane.done[-1] == ("reframe", (-1.0, 1.0, -1.0, 1.0), False)
    # A step past the end of the history is a command that found nothing to do,
    # and the view is left where it is rather than moved to nowhere.
    standing = len(pane.done)
    pane.say["command"]("view.forward", None)
    assert len(pane.done) == standing


async def test_a_typed_range_is_read_by_the_engine_so_minus_pi_is_an_answer(
    session, page, engine
):
    session.add(_landing("SIN(x)"))
    pane = page.panes[1]
    await drain(page, engine)
    engine.sent.clear()
    # `Set range...` puts up the form `plot/forms.py` describes, filled from the
    # view as it stands: a form opened to change one edge is three edges right.
    pane.say["command"]("range.set", None)
    assert pane.form["name"] == "range"
    assert [one["label"] for one in pane.form["fields"]] == [
        "Left",
        "Right",
        "Bottom",
        "Top",
    ]
    assert pane.values == ("-5", "5", "-4", "4")
    # What is typed is parsed where it was typed and evaluated where the numbers
    # are, which is why the fields take expressions in a page as on a desktop.
    engine.answers.append((-np.pi, np.pi, -1.0, 1.0))
    pane.say["typed"]("range", ["-π", "π", "-1", "1"])
    await settle()
    _, method, request = engine.sent[-1]
    assert method == asking.NUMBERS
    assert len(request.nodes) == 4
    assert pane.done[-1] == ("reframe", (-np.pi, np.pi, -1.0, 1.0), False)
    assert pane.message.startswith("Showing -3.14159 ≤ x ≤ 3.14159")


@pytest.mark.parametrize(
    ("texts", "worth", "said"),
    [
        (["(", "5", "-4", "4"], None, "The bounds are expressions, like -π or 2π"),
        (
            ["1/0", "5", "-4", "4"],
            (float("nan"), 5.0, -4.0, 4.0),
            "The bounds have to be worth numbers, like -π or 2π",
        ),
        (
            ["5", "-5", "-4", "4"],
            (5.0, -5.0, -4.0, 4.0),
            "A range runs from a lower bound to a higher one",
        ),
    ],
)
async def test_a_range_that_is_not_one_says_in_what_way(
    session, page, engine, texts, worth, said
):
    # Never silence, in every way a range can fail to be one: a text that will
    # not parse never reaches the worker, and one that parses to nothing worth a
    # number and one that runs backwards each have a sentence of their own.
    session.add(_landing("SIN(x)"))
    pane = page.panes[1]
    await drain(page, engine)
    engine.sent.clear()
    if worth is not None:
        engine.answers.append(worth)
    pane.say["typed"]("range", texts)
    await settle()
    assert pane.message == said
    assert len(engine.sent) == (0 if worth is None else 1)
    assert not any(one[0] == "reframe" for one in pane.done if isinstance(one, tuple))


async def test_del_clears_the_pane_and_takes_the_axis_names_with_it(session, page):
    session.add(_landing("SIN(x)"))
    pane = page.panes[1]
    assert pane.axes == ("x", "#1")
    pane.say["command"]("clear", None)
    assert pane.plots == {}
    assert pane.axes == ("", "")
    assert session.describe().windows[0].plots == ()


async def test_the_axes_are_named_after_what_is_plotted_against(session, page):
    session.add(_landing("SIN(x)"))
    pane = page.panes[1]
    assert pane.axes == ("x", "#1")
    # Two curves share the abscissa, and the ordinate then belongs to neither -
    # a stack of names down the side of the canvas is what the legend is for.
    session.add(_landing("COS(x)", "#2"))
    assert pane.axes == ("x", "")


async def test_the_point_size_menu_offers_the_sizes_and_is_sticky(session, page):
    heard = []
    session.events = heard.append
    session.add(_landing("[[1, 2], [3, 4]]", kind=PlotKind.DATA, variables=()))
    pane = page.panes[1]
    sizes = next(
        entry
        for entry in pane.say["menu"](_flat(pointed=1))
        if entry["name"] == "points.size"
    )
    assert [item["label"] for item in sizes["items"]] == ["3 px", "5 px", "8 px", "12 px"]
    pane.say["command"]("points.size", sizes["items"][2]["value"])
    assert pane.plots[1]["size"] == 8.0
    assert heard[-1] == plots.Preferred(plots.Prefer(point_size=8.0))


async def test_equal_scales_is_a_control_and_the_page_is_the_side_that_holds_it(
    session, page
):
    session.add(_landing("SIN(x)"))
    pane = page.panes[1]
    # A button on the tool row rather than an entry, since a toggle whose state
    # is worth reading is worth reading without opening anything.
    assert ("1:1", "Equal scales on both axes") == next(
        (one["bar"], one["hint"])
        for one in pane.commands
        if one["name"] == "scales.equal"
    )
    pane.say["command"]("scales.equal", None)
    # The lock is view state - a rubber band releases it without asking anybody
    # - so the page flips it and the menu reads it back off the snapshot.
    assert pane.done[-1] == "equalize"


async def test_a_picture_that_left_the_pane_says_so_and_a_refused_one_says_why(
    session, page
):
    session.add(_landing("SIN(x)"))
    pane = page.panes[1]
    pane.say["copied"]("", "")
    assert pane.message == "Copied the plot to the clipboard"
    # While tracing it is the traced point that goes, which is the one reading a
    # plot makes that the algebra window can take back.
    pane.say["copied"]("[1.000000, 0.841471]", "")
    assert pane.message == "Copied [1.000000, 0.841471]"
    # And a clipboard the browser refused is the case this exists for: Chromium
    # gates an image on permission, and a copy that did nothing and said nothing
    # would look exactly like a key bound to nothing.
    pane.say["copied"]("", "NotAllowedError: Write permission denied")
    assert pane.message == (
        "The browser did not allow the clipboard:"
        " NotAllowedError: Write permission denied"
    )


async def test_an_export_is_a_download_and_says_what_size_of_picture_it_was(
    session, page
):
    session.add(_landing("SIN(x)"))
    pane = page.panes[1]
    # Pixels rather than the desktop's vector file, and the sentence says the
    # size rather than leaving it to be discovered.
    pane.say["exported"]("plot1.png", 1440, 1000, "")
    assert pane.message == "Downloaded plot1.png, 1440 by 1000 pixels"
    pane.say["exported"]("plot1.png", 0, 0, "the download was blocked")
    assert pane.message == "The browser refused the download: the download was blocked"


async def test_the_parameter_range_is_a_strip_the_engine_reads(session, page, engine):
    session.add(_landing("SIN(x)"))
    pane = page.panes[1]
    # The strip is `plot/forms.py`'s, handed over when the pane is made, and it
    # stands only while there is a plot for it to be about.
    assert pane.strip["name"] == "trange"
    assert pane.parameters == (None, "", "", "")
    session.add(
        _landing("[COS(t), SIN(t)]", "#2", kind=PlotKind.PARAMETRIC, variables=("t",))
    )
    assert pane.parameters == (2, "t", "-3.14159", "3.14159")
    await drain(page, engine)
    engine.sent.clear()
    # A bound typed into it is an expression like any other, and the plot is
    # sampled again over the trees it came to.
    pane.say["ranged"](2, "0", "2π")
    await settle()
    assert engine.sent[0][2].model.bounds is not None
    # One that will not read is refused in the parameter's own name, and the
    # fields go back to the range the curve is actually drawn over.
    pane.say["ranged"](2, "(", "2π")
    assert pane.message == "The t bounds are expressions, like -π or 2π"
    assert pane.parameters == (2, "t", "-3.14159", "3.14159")
    # And the range a sampling came back with is what they then show, spelled
    # by Python: nothing on the page writes a number into a field.
    pane.say["spanned"](2, 0.0, 6.28318)
    assert pane.parameters == (2, "t", "0", "6.28318")


async def test_copy_image_is_heard_as_an_event_rather_than_as_a_key(session, page):
    # The one control whose key does not go on the page's key ladder. Cancelling
    # the Ctrl+C keydown cancels the copy the browser was about to offer, so the
    # description says which event carries it and the page listens for that.
    session.add(_landing("SIN(x)"))
    offered = {one["name"]: one for one in page.panes[1].commands}
    assert offered["image.copy"]["event"] == "copy"
    assert offered["image.copy"]["keys"] == ["Ctrl+C"]
    assert [one["name"] for one in offered.values() if one["event"]] == ["image.copy"]


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
    # The parameter each sample came from crosses with the samples, which is what
    # lets a click on a curve that doubles back name a place on it: a point in
    # the plane is not a parameter, so the page snaps to the nearest sample and
    # rides on from the parameter that one came from.
    assert answer["ts"].shape == answer["xs"].shape
    turned = answer["ts"].astype(np.float64)
    assert np.allclose(np.cos(turned), answer["xs"].astype(np.float64), atol=1e-5)
    assert np.all(np.diff(turned) >= 0)


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
    engine.answers.append((-2.0, 2.0, -2.0, 2.0))
    solids.panes[1].say["framed"]("-2", "2", "-2", "2", "32", "32")
    # Twice: what the bounds are worth comes back first, and the evaluation
    # over the domain they came to goes out behind it.
    await settle()
    await settle()
    assert heard[-1] == plots.Preferred(plots.Prefer(grid=32))
    request = engine.sent[-1][2]
    assert request.xdomain == (-2.0, 2.0)
    assert request.grid == (32, 32)
    assert solids.panes[1].message == "Evaluating over the new domain at 32 by 32"
    # And `Describe` reports the domain, which is what a 3D window is framed by.
    assert session.describe().windows[0].xrange == (-2.0, 2.0)


async def test_a_typed_domain_is_read_by_the_engine_so_minus_pi_is_an_answer(
    session, solids, engine
):
    session.add(_plotted("SIN(x y)"))
    pane = solids.panes[1]
    # The strip is `plot/forms.py`'s, handed over when the pane is made, and the
    # bounds in it are expressions rather than floats - parsed where they were
    # typed and evaluated where the numbers are, exactly as a flat pane's are.
    assert pane.strip["name"] == "domain"
    assert [one["name"] for one in pane.strip["fields"]] == [
        "x0",
        "x1",
        "y0",
        "y1",
        "nx",
        "ny",
    ]
    await drain(page_of(solids), engine)
    engine.sent.clear()
    engine.answers.append((-np.pi, np.pi, -5.0, 5.0))
    pane.say["framed"]("-π", "π", "-5", "5", "64", "64")
    await settle()
    await settle()
    assert engine.sent[0][1] == asking.NUMBERS
    assert engine.sent[-1][2].xdomain == pytest.approx((-np.pi, np.pi))
    assert pane.fields[:2] == (written(-np.pi), written(np.pi))


@pytest.mark.parametrize(
    ("texts", "worth", "said"),
    [
        (
            ("(", "5", "-5", "5", "64", "64"),
            None,
            "The domain bounds are expressions, like -π or 2π",
        ),
        (("-5", "5", "-5", "5", "x", "64"), None, "The grid is a pair of numbers"),
        (
            ("5", "-5", "-5", "5", "64", "64"),
            (5.0, -5.0, -5.0, 5.0),
            "The domain runs from a lower bound to a higher one",
        ),
        (
            ("1/0", "5", "-5", "5", "64", "64"),
            (float("nan"), 5.0, -5.0, 5.0),
            "The bounds have to be worth numbers, like -π or 2π",
        ),
    ],
)
async def test_a_domain_that_is_not_one_says_in_what_way(
    session, solids, engine, texts, worth, said
):
    # Never silence, in every way a domain can fail to be one, and the fields go
    # back to what the pane is actually evaluating over: a text that will not
    # parse never reaches the worker, a grid that is not a pair of counts is
    # refused before it does, and one that parses to nothing worth a number and
    # one that runs backwards each have a sentence of their own.
    session.add(_plotted("SIN(x y)"))
    await drain(page_of(solids), engine)
    engine.sent.clear()
    pane = solids.panes[1]
    if worth is not None:
        engine.answers.append(worth)
    pane.say["framed"](*texts)
    await settle()
    assert pane.message == said
    assert pane.fields == ("-5", "5", "-5", "5", "64", "64")
    assert not any(one[1] == asking.GRID for one in engine.sent)


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
    # What the page says when it has drawn: what the values measure in z, and
    # the box they were placed in. The same, for a surface standing alone.
    solids.panes[1].say["stood"](1, [-1.0, 1.0, -1.0, 1.0], [-1.0, 1.0])
    await settle()
    assert engine.sent == []


async def test_two_surfaces_are_drawn_in_one_box(session, solids, engine):
    session.add(_plotted("SIN(x y)"))
    session.add(_plotted("x^2 - y^2", "#2"))
    pane = solids.panes[1]
    await settle()
    pane.say["stood"](1, [-1.0, 1.0, -1.0, 1.0], [-1.0, 1.0])
    await answered(solids, engine)
    at = len(engine.sent)
    # The second surface reaches higher, so the box grows and every mesh built
    # to the box that is gone - both of them - is asked for again in the new one.
    pane.say["stood"](2, [-25.0, 25.0, -25.0, 25.0], [-1.0, 1.0])
    await answered(solids, engine)
    asked = engine.sent[at:]
    assert asked and all(one[2].zrange == (-25.0, 25.0) for one in asked)


async def test_a_pooled_box_that_cuts_a_spike_says_so_in_the_desktops_words(
    session, solids
):
    session.add(_plotted("SIN(x y)"))
    pane = solids.panes[1]
    # The whole of these values runs far past their middle 98%, which is what a
    # spike is: the box takes the middle, and the sentence about it is the one
    # `plot/actions.py` writes for both windows.
    pane.say["stood"](1, [-500.0, 500.0, -3.0, 3.0], [-500.0, 500.0])
    assert pane.message == "z clipped to the 1st-99th percentile: -3 to 3"


def test_a_pooled_box_is_the_box_a_window_holding_the_arrays_would_build():
    from rederive.plot import view as framing
    from rederive.plot.surface import extent, spanned

    # The desktop has the values and the browser has four numbers a surface, and
    # the two arrive at one box because the pooling is one function that both of
    # them call. A window measures and pools in one breath; a pane measures in
    # the worker, sends the summary and pools on the main thread.
    gentle = np.linspace(-1.0, 1.0, 400)
    spiked = np.concatenate([np.linspace(-3.0, 3.0, 399), [900.0]])
    assert framing.pooled([spanned(gentle), spanned(spiked)]) == extent(
        [gentle, spiked]
    )
    (low, high), clipped = extent([gentle, spiked])
    assert clipped is True
    assert (low, high) == pytest.approx((-3.0, 3.0), abs=0.1)


async def answered(solids, engine):
    """Acknowledge the evaluation with the worker, which sends the next one."""
    await settle()
    if engine.sent:
        solids.handlers["landed"](engine.sent[-1][0], "")
    await settle()


async def test_the_inspector_is_the_form_the_desktops_dialog_is(session, solids):
    session.add(_plotted("SIN(x y)"))
    pane = solids.panes[1]
    # The same nine fields in the same three groups, and the three answers the
    # desktop's dialog offers - which is the one form that names its own.
    assert pane.form["name"] == "view"
    assert [one["name"] for one in pane.form["fields"]] == [
        *("cx", "cy", "cz", "lx", "ly", "lz"),
        *("azimuth", "elevation", "distance"),
    ]
    assert [group["heads"] for group in pane.form["groups"]] == [
        ["x", "y", "z"],
        ["azimuth", "elevation", "distance"],
    ]
    assert [one["label"] for one in pane.form["buttons"]] == [
        "Apply",
        "Autoscale z",
        "Close",
    ]
    # It opens on the box the pane is standing in, spelled by Python: the domain
    # and the z are this side's, and the camera is the page's to fill in.
    pane.say["command"]("view.inspect", None)
    assert pane.inspected == ("0", "0", "0", "10", "10", "2")


async def test_the_inspector_sets_the_box_and_the_camera(session, solids, engine):
    session.add(_plotted("SIN(x y)"))
    pane = solids.panes[1]
    await drain(page_of(solids), engine)
    engine.sent.clear()
    # A center and a length an axis, and where to look from: the domain moves,
    # the surface is evaluated over it, and the camera goes where it was told.
    pane.say["typed"](
        "view", ["1", "0", "0", "4", "10", "2", "45", "20", "30"], "apply"
    )
    await settle()
    assert pane.fields == ("-1", "3", "-5", "5", "64", "64")
    assert engine.sent[-1][2].xdomain == (-1.0, 3.0)
    assert pane.camera == (20.0, 45.0, 30.0)
    # And a box that is not numbers is refused rather than swallowed.
    pane.say["typed"]("view", ["a"] * 9, "apply")
    assert pane.message == "The view is described in numbers"


async def test_a_typed_z_outranks_the_data_until_autoscale_gives_it_back(
    session, solids, engine
):
    session.add(_plotted("SIN(x y)"))
    pane = solids.panes[1]
    pane.say["stood"](1, [-1.0, 1.0, -1.0, 1.0], [-1.0, 1.0])
    await drain(page_of(solids), engine)
    engine.sent.clear()
    # Somebody who asks for -4 to 4 in z wants the surface cut off at 4, so the
    # mesh is built to it and the next measurement does not autoscale it away.
    pane.say["typed"](
        "view", ["0", "0", "0", "10", "10", "8", "0", "0", "20"], "apply"
    )
    await settle()
    assert engine.sent[0][2].zrange == (-4.0, 4.0)
    assert pane.inspected == ("0", "0", "0", "10", "10", "8")
    pane.say["stood"](1, [-1.0, 1.0, -1.0, 1.0], [-4.0, 4.0])
    await drain(page_of(solids), engine)
    engine.sent.clear()
    # `Autoscale z` hands the extent back to the data, and the surface is asked
    # for again in the box its own values want.
    pane.say["typed"]("view", [""] * 9, "reset")
    await settle()
    assert engine.sent[0][2].zrange is None
    assert pane.inspected == ("0", "0", "0", "10", "10", "2")


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


def test_the_writing_on_the_box_is_chosen_and_spelled_where_the_numbers_are():
    from rederive.plot.surface import ticks

    answer = _standing(_grid("SIN(x y)"))
    box = _geometry("SIN(x y)")["box"]
    # A tick a page draws is a coordinate in the box's own world units and a
    # string: the ruler is `plot/surface.py`'s, the spelling is `written`, and
    # the page projects a point it was handed and works out neither.
    across = answer["ticks"]["x"]
    assert [one["text"] for one in across] == [
        written(value) for value in ticks(*box.x)
    ]
    assert [one["at"] for one in across] == pytest.approx(
        [float(box.across(value)) for value in ticks(*box.x)]
    )
    assert [one["text"] for one in answer["ticks"]["z"]] == [
        written(value) for value in ticks(*box.z)
    ]


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

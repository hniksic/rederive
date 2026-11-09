"""Fakes of the browser environment, for testing the browser backend without a browser.

The browser backend depends on two things a desktop lacks: a page, which it calls to
create panes and draw in them, and a runtime bridge that converts Python values for
JavaScript. Both interfaces are narrow, so they are faked here in plain Python, letting
tests check what a pane does when a plot is added, what it asks the worker for, and how
it handles the answer.

As in the browser, the page is two modules: 2D and 3D plots share no drawing code, so
they get separate fakes.

`pyodide` is faked rather than avoided. The backend uses it only to proxy callbacks and
convert structures, which are identity functions on a desktop. Nothing here draws.

The tab is faked to the same depth. `js` provides a store, a clipboard, an address and
two objects that take listeners, which is all the storage and file actions touch. The
store can be given a size limit to test what a save does when the browser is full.
"""

import sys
import types


class FakePane:
    """A 2D pane that records the calls the backend makes on it.

    Drawing belongs to the page, so methods that commands reach (framing the view,
    toggling the grid, tracing a curve) only record that they were called. Tests check
    which method a control's name was dispatched to, which is all the dispatch has to
    get right.
    """

    def __init__(self, number, debounce, commands, strip, handlers):
        self.number = number
        self.debounce = debounce
        #: The commands the pane may offer, as described by `plot/controls.py`.
        self.commands = commands
        #: The parameter range strip, as `plot/forms.py` describes it.
        self.strip = strip
        self.say = handlers
        self.plots = {}
        self.order = []
        self.title = ""
        self.current = False
        self.presented = 0
        self.dismissed = 0
        self.message = ""
        self.polar = False
        #: Which toolbar buttons are lit, keyed by control.
        self.buttons = {}
        #: Every drawing call made by a command, in order.
        self.done = []
        #: The view the pane is showing, which the backend reads back for a `Describe`
        #: and for every sampling it requests.
        self.shown = [-5.0, 5.0, -4.0, 4.0, 800.0, 640.0]
        #: Every `starting` the backend announced, as (plot, generation, fresh).
        self.started = []
        #: Whether there is anything to frame. `View all` reports this as a flag; the
        #: message about it comes from Python.
        self.framable = True
        #: The axis names, and which plot the parameter range fields refer to.
        self.axes = ("", "")
        self.parameters = (None, "", "", "")
        #: The form currently shown over the pane, as passed to the page.
        self.form = None
        self.values = ()

    def autoscale(self):
        self.done.append("autoscale")
        if self.framable:
            self.shown = [-2.0, 2.0, -1.0, 1.0, *self.shown[4:]]
        return self.framable

    def reframe(self, x0, x1, y0, y1, equal):
        self.done.append(("reframe", (x0, x1, y0, y1), equal))
        self.shown = [x0, x1, y0, y1, *self.shown[4:]]

    def equalize(self):
        self.done.append("equalize")

    def named(self, across, up):
        self.axes = (across, up)

    def parametrized(self, serial, name, low, high):
        self.parameters = (serial, name, low, high)

    def ask(self, form, values):
        self.form = form
        self.values = tuple(values)
        self.done.append("ask")

    def copy(self):
        self.done.append("copy")

    def export(self):
        self.done.append("export")

    def zoom(self, factor):
        self.done.append(("zoom", factor))

    def trace(self):
        self.done.append("trace")

    def grid(self):
        self.done.append("grid")

    def legend(self):
        self.done.append("legend")

    def polarized(self, polar):
        self.polar = bool(polar)
        self.done.append(("polarized", self.polar))

    def lit(self, name, on):
        self.buttons[name] = bool(on)

    def said(self, message):
        self.message = message

    def add(self, serial, spec):
        self.plots[serial] = spec
        self.order.append(serial)

    def respec(self, serial, spec):
        self.plots[serial] = spec

    def remove(self, serial):
        self.plots.pop(serial, None)
        self.order = [one for one in self.order if one != serial]

    def starting(self, serial, generation, fresh):
        self.started.append((serial, generation, fresh))

    def present(self, quietly=False):
        self.presented += 1
        self.quietly = quietly

    def retitle(self, title, current):
        self.title = title
        self.current = current

    def view(self):
        return self.shown

    def dismiss(self):
        self.dismissed += 1


class FakePage:
    """The page's 2D plotting module."""

    def __init__(self):
        self.panes = {}
        self.handlers = None
        self.stopped = 0
        #: The appearance settings given to the panes, as described by
        #: `plot/appearance.py`.
        self.look = None

    def dress(self, description):
        self.look = description

    def open(self, number, debounce, commands, strip, handlers):
        pane = FakePane(number, debounce, commands, strip, handlers)
        self.panes[number] = pane
        return pane

    def attend(self, handlers):
        self.handlers = handlers

    def stop(self):
        self.stopped += 1


class FakeSolid:
    """A 3D pane that records the calls the backend makes on it."""

    def __init__(self, number, commands, strip, form, handlers):
        self.number = number
        #: The commands the pane may offer, as described by `plot/controls.py`.
        self.commands = commands
        #: The domain strip and the inspector, as `plot/forms.py` describes them.
        self.strip = strip
        self.form = form
        self.say = handlers
        self.plots = {}
        self.order = []
        self.title = ""
        self.current = False
        self.presented = 0
        self.dismissed = 0
        self.message = ""
        #: The last camera position, as (elevation, azimuth, distance).
        self.camera = None
        self.spinning = ""
        #: Which toolbar buttons are lit, keyed by control.
        self.buttons = {}
        #: Every drawing call made by a command, in order.
        self.done = []
        #: The contents of the six toolbar fields.
        self.fields = ()
        #: The three axis names, and the box the inspector was last opened with or
        #: filled with.
        self.axes = ()
        self.inspected = None
        #: Every `starting` the backend announced, as (surface, generation).
        self.started = []

    def look(self, elevation, azimuth, distance):
        self.camera = (elevation, azimuth, distance)
        self.done.append("look")

    def spin(self, rotating):
        self.spinning = rotating
        self.done.append("spin")

    def inspect(self, box):
        self.inspected = tuple(box)
        self.done.append("inspect")

    def showing(self, box):
        self.inspected = tuple(box)

    def named(self, across, along, up):
        self.axes = (across, along, up)

    def box(self):
        self.done.append("box")

    def naming(self):
        self.done.append("names")

    def copy(self):
        self.done.append("copy")

    def export(self):
        self.done.append("export")

    def legend(self):
        self.done.append("legend")

    def lit(self, name, on):
        self.buttons[name] = bool(on)

    def add(self, serial, spec):
        self.plots[serial] = spec
        self.order.append(serial)

    def respec(self, serial, spec):
        self.plots[serial] = spec

    def remove(self, serial):
        self.plots.pop(serial, None)
        self.order = [one for one in self.order if one != serial]

    def starting(self, serial, generation):
        self.started.append((serial, generation))

    def present(self, quietly=False):
        self.presented += 1
        self.quietly = quietly

    def retitle(self, title, current):
        self.title = title
        self.current = current

    def domain(self, *fields):
        self.fields = fields

    def said(self, message):
        self.message = message

    def dismiss(self):
        self.dismissed += 1


class FakeSolids:
    """The page's 3D plotting module."""

    def __init__(self):
        self.panes = {}
        self.handlers = None
        self.stopped = 0
        #: The appearance settings given to the panes, as described by
        #: `plot/appearance.py`.
        self.look = None

    def dress(self, description):
        self.look = description

    def open(self, number, commands, strip, form, handlers):
        pane = FakeSolid(number, commands, strip, form, handlers)
        self.panes[number] = pane
        return pane

    def attend(self, handlers):
        self.handlers = handlers

    def stop(self):
        self.stopped += 1


class FakeEngine:
    """The engine worker, as the executor posts to it.

    Requests are recorded rather than sent, so a test can inspect them and supply the
    answers.
    """

    def __init__(self):
        self.sent = []
        self.lost = None
        #: Answers returned by successive `value` requests, in order. When empty, the
        #: worker answers with nothing, which counts as a refusal.
        self.answers = []
        self._number = 0

    def numbered(self):
        self._number += 1
        return self._number

    async def ask(self, number, method, args):
        self.sent.append((number, method, args[0]))

    async def value(self, method, args):
        """A request whose answer is returned to Python rather than to the page."""
        self.sent.append((self.numbered(), method, args[0]))
        return self.answers.pop(0) if self.answers else ()


#: The error a page reports when the archive can't be reached, formatted the way a
#: browser formats it: error name, message, and stack.
SHUT = "Error: the server said 503\n    at got (demos.js:214:11)"


class JsNull:
    """JavaScript `null` as Pyodide passes it: falsy, but not `None`.

    Pyodide distinguishes `undefined` from `null`: a missing field arrives as `None`, a
    field set to `null` as this object. It is falsy, so truth tests pass, but not
    `None`, so identity tests fail. A menu snapshot with no curve under it is built in
    JavaScript and uses `null`, so the fake page does too, testing the backend against
    what it really receives.
    """

    def __bool__(self):
        return False

    def __repr__(self):
        return "jsnull"


#: The singleton, like the one in `pyodide.ffi`.
jsnull = JsNull()


class Proxy:
    """A Python callable proxied for JavaScript: callable and destroyable.

    Pyodide's proxies have a lifetime, and that's the part worth faking: a backend that
    stopped destroying its proxies would leak them in a browser yet pass every test that
    only calls them.
    """

    def __init__(self, call):
        self._call = call
        self.destroyed = 0

    def __call__(self, *arguments, **keywords):
        return self._call(*arguments, **keywords)

    def destroy(self):
        self.destroyed += 1


class Store:
    """localStorage: a string store with an optional size limit.

    `room` is the largest value, in characters, it accepts before raising, mimicking a
    browser's quota. None means no limit.
    """

    def __init__(self, room=None):
        self.items = {}
        self.room = room

    def getItem(self, key):
        return self.items.get(key)

    def setItem(self, key, value):
        if self.room is not None and len(value) > self.room:
            raise RuntimeError("QuotaExceededError")
        self.items[key] = value


class Promise:
    """The result of a clipboard write, which may be rejected."""

    def __init__(self, refuse=None):
        self.refuse = refuse

    def catch(self, handler):
        if self.refuse is not None:
            handler(self.refuse)


class Clipboard:
    """The page's clipboard, which may allow or refuse writes, recording what it got."""

    def __init__(self, allow=True, raises=False):
        self.allow = allow
        self.raises = raises
        self.written = []

    def writeText(self, text):
        if self.raises:
            raise RuntimeError("NotAllowedError")
        self.written.append(text)
        return Promise(None if self.allow else "NotAllowedError")


class FakeTab:
    """The page as called by `WebStorage`, `Files` and `Demos`.

    These are two modules in a browser but one here, since the files module and the demo
    menu share no calls; a test tells them apart by which list was written to.
    """

    def __init__(self):
        self.downloads = []
        self.notices = []
        self.fragment = None
        self.handlers = None
        #: The result of downloading each file name through the page: bytes, or the
        #: exception to raise. A missing name means the archive is unreachable.
        self.archive = {}
        #: Every file the page was asked to fetch, in order.
        self.fetched = []

    async def brought(self, file):
        """The download `demos.js` performs, as awaited from Python."""
        self.fetched.append(file)
        answer = self.archive.get(file)
        if isinstance(answer, BaseException):
            raise answer
        if answer is None:
            raise RuntimeError(SHUT)
        return answer

    def download(self, name, text):
        self.downloads.append((name, text))

    def said(self, text):
        self.notices.append(text)

    def address(self, fragment):
        self.fragment = fragment

    def attend(self, handlers):
        self.handlers = handlers


def bridge():
    """Install fake `js` and `pyodide.ffi` modules for the backend to import.

    Idempotent. The modules are left in place, since on a desktop there is nothing for
    them to collide with.
    """
    if "pyodide.ffi" in sys.modules:
        return
    js = types.ModuleType("js")
    js.Object = types.SimpleNamespace(fromEntries=dict)
    ffi = types.ModuleType("pyodide.ffi")
    ffi.create_proxy = Proxy
    ffi.to_js = lambda value, dict_converter=None: value
    ffi.jsnull = jsnull
    pyodide = types.ModuleType("pyodide")
    pyodide.ffi = ffi
    sys.modules["js"] = js
    sys.modules["pyodide"] = pyodide
    sys.modules["pyodide.ffi"] = ffi


def browser(store=None, clipboard=None, hash=""):
    """Install and return a fresh tab: an empty store, a clipboard, and an address.

    Everything is replaced on each call, so no test sees another test's worksheets.
    """
    bridge()
    js = sys.modules["js"]
    js.localStorage = Store() if store is None else store
    js.navigator = types.SimpleNamespace(
        clipboard=Clipboard() if clipboard is None else clipboard,
        userAgent="a test",
    )
    js.location = types.SimpleNamespace(hash=hash, href=f"https://demo.test/#{hash}")
    js.window = types.SimpleNamespace(addEventListener=lambda *given: None)
    js.document = types.SimpleNamespace(addEventListener=lambda *given: None)
    return js

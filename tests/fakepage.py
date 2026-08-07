"""A browser with no browser in it, for the tests that are about the browser.

The browser backend talks to two things a desktop has not got: a page, which it
calls into to make a pane and draw in one, and a runtime bridge, which turns a
Python value into something JavaScript can read. Both are narrow, and both are
here as ordinary Python - so what a pane does when a plot lands in it, what it
asks the worker for, and what it says when an answer comes back are questions
that can be asked with no browser at all.

The page is two modules, as it is in a browser: a flat picture and a solid one
share no drawing, so they share no fake either.

`pyodide` is faked rather than avoided. The two calls the backend makes of it -
proxying a callback and converting a structure - are identity functions on a
desktop, and the fake says so; what it is not is a pretend browser, and nothing
here draws anything.

The tab itself is faked for the same reasons and to the same depth. `js` is a
module with a store, a clipboard, an address and two things that take listeners,
because that is the whole of what the storage and the file gestures touch - and
a store with a ceiling on it is the one way to ask what a save does when the
browser will not keep any more.
"""

import sys
import types


class FakePane:
    """One pane as the backend speaks to one, recording what it is told.

    The picture is the page's on this side of the seam too, so the methods a
    command reaches - framing the view, flipping the grid, riding a curve - do
    nothing here but say that they were called. What a test then asks is which
    of them the name of a control arrived at, which is the whole of what the
    dispatch has to get right.
    """

    def __init__(self, number, debounce, commands, strip, handlers):
        self.number = number
        self.debounce = debounce
        #: What the pane was told it may offer, as `plot/controls.py` says it.
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
        #: Which buttons of the tool row are lit, by the control each one is.
        self.buttons = {}
        #: Every call a command reached the picture by, in order.
        self.done = []
        #: What the pane would be showing, which the backend reads back for a
        #: `Describe` and for every sampling it asks for.
        self.shown = [-5.0, 5.0, -4.0, 4.0, 800.0, 640.0]
        #: Every `starting` the backend announced, as a plot and a generation.
        self.started = []
        #: Whether there is anything to frame, which is what `View all` reports
        #: rather than says: the sentence about it is Python's.
        self.framable = True
        #: What the axes are named, and which plot the range fields are about.
        self.axes = ("", "")
        self.parameters = (None, "", "", "")
        #: The form now up over the pane, as the page was handed it.
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

    def present(self):
        self.presented += 1

    def retitle(self, title, current):
        self.title = title
        self.current = current

    def view(self):
        return self.shown

    def dismiss(self):
        self.dismissed += 1


class FakePage:
    """The page's plotting module, as the backend calls into it."""

    def __init__(self):
        self.panes = {}
        self.handlers = None
        self.stopped = 0

    def open(self, number, debounce, commands, strip, handlers):
        pane = FakePane(number, debounce, commands, strip, handlers)
        self.panes[number] = pane
        return pane

    def attend(self, handlers):
        self.handlers = handlers

    def stop(self):
        self.stopped += 1


class FakeSolid:
    """One 3D pane as the backend speaks to one, recording what it is told."""

    def __init__(self, number, commands, handlers):
        self.number = number
        #: What the pane was told it may offer, as `plot/controls.py` says it.
        self.commands = commands
        self.say = handlers
        self.plots = {}
        self.order = []
        self.title = ""
        self.current = False
        self.presented = 0
        self.dismissed = 0
        self.message = ""
        #: Where the camera was last put, as elevation, azimuth and distance.
        self.camera = None
        self.spinning = ""
        #: Which buttons of the tool row are lit, by the control each one is.
        self.buttons = {}
        #: Every call a command reached the picture by, in order.
        self.done = []
        #: What the six fields of the tool row are showing.
        self.fields = ()
        #: Every `starting` the backend announced, as a surface and a generation.
        self.started = []

    def look(self, elevation, azimuth, distance):
        self.camera = (elevation, azimuth, distance)
        self.done.append("look")

    def spin(self, rotating):
        self.spinning = rotating
        self.done.append("spin")

    def inspect(self):
        self.done.append("inspect")

    def box(self):
        self.done.append("box")

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

    def present(self):
        self.presented += 1

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
    """The page's solid-drawing module, as the backend calls into it."""

    def __init__(self):
        self.panes = {}
        self.handlers = None
        self.stopped = 0

    def open(self, number, commands, handlers):
        pane = FakeSolid(number, commands, handlers)
        self.panes[number] = pane
        return pane

    def attend(self, handlers):
        self.handlers = handlers

    def stop(self):
        self.stopped += 1


class FakeEngine:
    """The engine worker, as the executor posts to one.

    Every request is kept rather than sent, so that a test can read what the
    worker would have been asked and answer for it.
    """

    def __init__(self):
        self.sent = []
        self.lost = None
        #: What the next request whose answer is a value comes back with, in
        #: the order they are asked for. Empty is a worker that answers with
        #: nothing at all, which is a refusal like any other.
        self.answers = []
        self._number = 0

    def numbered(self):
        self._number += 1
        return self._number

    async def ask(self, number, method, args):
        self.sent.append((number, method, args[0]))

    async def value(self, method, args):
        """One request whose answer is Python's rather than the page's."""
        self.sent.append((self.numbered(), method, args[0]))
        return self.answers.pop(0) if self.answers else ()


class Proxy:
    """A Python callable as JavaScript holds one: callable, and destroyable.

    Pyodide's is a handle with a lifetime, and the lifetime is the half worth
    faking - a backend that stopped destroying its proxies would leak them in a
    browser and pass every test that only ever called them.
    """

    def __init__(self, call):
        self._call = call
        self.destroyed = 0

    def __call__(self, *arguments, **keywords):
        return self._call(*arguments, **keywords)

    def destroy(self):
        self.destroyed += 1


class Store:
    """localStorage as a browser keeps one: strings, and a ceiling to hit.

    `room` is how many characters it will take before it refuses, which is what
    a browser does with its five megabytes and the only interesting thing about
    the ceiling. None is a store with no ceiling at all.
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
    """What a clipboard write answers with: something that may be refused."""

    def __init__(self, refuse=None):
        self.refuse = refuse

    def catch(self, handler):
        if self.refuse is not None:
            handler(self.refuse)


class Clipboard:
    """The page's clipboard, granted or refused, and what it was given."""

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
    """The page's file module, as `WebStorage` and `Files` call into it."""

    def __init__(self):
        self.downloads = []
        self.notices = []
        self.fragment = None
        self.handlers = None

    def download(self, name, text):
        self.downloads.append((name, text))

    def said(self, text):
        self.notices.append(text)

    def address(self, fragment):
        self.fragment = fragment

    def attend(self, handlers):
        self.handlers = handlers


def bridge():
    """Put a desktop-shaped `js` and `pyodide.ffi` where the backend looks.

    Idempotent, and it leaves the modules in place: they are two functions and
    an object with one attribute, and there is nothing on a desktop for them to
    collide with.
    """
    if "pyodide.ffi" in sys.modules:
        return
    js = types.ModuleType("js")
    js.Object = types.SimpleNamespace(fromEntries=dict)
    ffi = types.ModuleType("pyodide.ffi")
    ffi.create_proxy = Proxy
    ffi.to_js = lambda value, dict_converter=None: value
    pyodide = types.ModuleType("pyodide")
    pyodide.ffi = ffi
    sys.modules["js"] = js
    sys.modules["pyodide"] = pyodide
    sys.modules["pyodide.ffi"] = ffi


def browser(store=None, clipboard=None, hash=""):
    """A tab with nothing in it yet: an empty store, a clipboard, an address.

    Returned as well as installed, since a test asks the tab what it was told
    as often as it tells it anything. Everything is replaced on every call - a
    store left over from another test is a store with somebody else's
    worksheets in it.
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

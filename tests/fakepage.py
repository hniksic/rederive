"""A browser with no browser in it, for the tests that are about the plot panes.

The browser backend talks to two things a desktop has not got: a page, which it
calls into to make a pane and draw in one, and a runtime bridge, which turns a
Python value into something JavaScript can read. Both are narrow, and both are
here as ordinary Python - so what a pane does when a plot lands in it, what it
asks the worker for, and what it says when an answer comes back are questions
that can be asked with no browser at all.

`pyodide` is faked rather than avoided. The two calls the backend makes of it -
proxying a callback and converting a structure - are identity functions on a
desktop, and the fake says so; what it is not is a pretend browser, and nothing
here draws anything.
"""

import sys
import types


class FakePane:
    """One pane as the backend speaks to one, recording what it is told."""

    def __init__(self, number, debounce, handlers):
        self.number = number
        self.debounce = debounce
        self.say = handlers
        self.plots = {}
        self.order = []
        self.title = ""
        self.current = False
        self.presented = 0
        self.dismissed = 0
        #: What the pane would be showing, which the backend reads back for a
        #: `Describe` and for every sampling it asks for.
        self.shown = [-5.0, 5.0, -4.0, 4.0, 800.0, 640.0]
        #: Every `starting` the backend announced, as a plot and a generation.
        self.started = []

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

    def open(self, number, debounce, handlers):
        pane = FakePane(number, debounce, handlers)
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
        self._number = 0

    def numbered(self):
        self._number += 1
        return self._number

    async def ask(self, number, method, args):
        self.sent.append((number, method, args[0]))


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

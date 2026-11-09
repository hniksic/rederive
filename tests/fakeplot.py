"""A plot backend that does no drawing, for tests of plot bookkeeping.

Which window an `Add` lands in, which window is the receiver, how a preference affects
the next window, and what a refusal says don't involve a toolkit, so this backend lets
them be tested without one. As a second implementation of `plot.backend`, it also checks
the protocol: its windows answer exactly the calls the session makes and no others.

Window menus can be tested the same way: a menu is the controls table evaluated against
a snapshot, so it needs no display, toolkit or window.
"""

from rederive.plot import controls, protocol


def offered(state):
    """The (label, first key) pairs a menu offers in this state, in menu order.

    A control with no key on this platform gets an empty key rather than being omitted,
    since it is still a menu entry.
    """
    return [
        (entry.label, entry.keys[0] if entry.keys else "")
        for entry in controls.menu(state)
    ]


def ticked(state):
    """The labels of the menu entries that are checked."""
    return [entry.label for entry in controls.menu(state) if entry.checked]


class FakeWindow:
    """A window reduced to what the session sees of it."""

    def __init__(self, session, kind, number, preferences):
        self.session = session
        self.kind = kind
        self.number = number
        #: The preferences this window was opened with; this is how a sticky grid or
        #: wire look reaches a real window.
        self.preferences = preferences
        self.plots = []
        self.current = False
        self.title = ""
        #: How many times a plot was added here and the window brought to the front.
        self.presented = 0
        #: Whether a demonstration is keeping this window in front, and how many times
        #: one has been released.
        self.demonstrating = False
        self.released = 0

    def add(self, plot):
        existing = self.find(plot.worksheet, plot.label)
        if existing is not None:
            plot.color = existing.color
            self.plots[self.plots.index(existing)] = plot
        else:
            self.plots.append(plot)
        self.retitle()
        return plot

    def remove(self, plot):
        if plot in self.plots:
            self.plots.remove(plot)
        self.retitle()

    def find(self, worksheet, label):
        for plot in self.plots:
            if plot.worksheet == worksheet and plot.label == label:
                return plot
        return None

    def present(self, demonstrating=False):
        self.presented += 1
        self.quietly = demonstrating
        self.demonstrating = demonstrating

    def release(self, finished=False):
        if not self.demonstrating:
            return
        self.released += 1
        self.demonstrating = False
        if finished:
            self.close()

    def retitle(self, current=None):
        if current is not None:
            self.current = current
        self.title = protocol.titled(
            self.kind, tuple(plot.text or plot.label for plot in self.plots), self.current
        )

    def describe(self):
        return protocol.WindowInfo(
            number=self.number,
            kind=self.kind,
            title=self.title,
            current=self.current,
            plots=tuple(
                protocol.PlotInfo(
                    worksheet=plot.worksheet,
                    label=plot.label,
                    text=plot.text,
                    kind=plot.kind,
                    hidden=False,
                )
                for plot in self.plots
            ),
            xrange=(-5.0, 5.0),
            yrange=(-5.0, 5.0),
        )

    def close(self):
        """Simulate the window manager closing the window by notifying the session."""
        self.session.closed(self.number)


class FakeBackend:
    """A backend whose windows do bookkeeping and no drawing."""

    def __init__(self):
        self.windows = {}
        #: Every window opened, in order, for tests that check what a window was
        #: built with.
        self.opened = []
        self.stopped = False

    def open(self, session, kind, number, preferences):
        window = FakeWindow(session, kind, number, preferences)
        self.windows[number] = window
        self.opened.append(window)
        return window

    def close(self, number):
        self.windows.pop(number, None)

    def stop(self):
        self.stopped = True
        self.windows.clear()


class InlineExecutor:
    """An executor that runs each job synchronously, before `submit` returns.

    The desktop uses a thread and the browser will use a worker; running jobs in the
    caller turns a sampling in a test into a plain function call.
    """

    def __init__(self):
        self.sampler = None

    def serving(self, sampler):
        self.sampler = sampler

    def wake(self):
        while (job := self.sampler.take()) is not None:
            work, done = job
            done(work())

    def deliver(self, done, answer):
        done(answer)

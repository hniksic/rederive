"""A plot backend with nothing under it, for the tests that are about bookkeeping.

Which window an `Add` lands in, which window is the receiver, what a preference
does to the next one and what a refusal says are questions with no toolkit in
them, and this is what lets them be asked without one. It is the second
implementation of `plot.backend` that the desktop program does not have yet, so
it is also where the protocol gets read back: a window here answers exactly the
calls the session makes and no others.

What a window's menu offers is now such a question too. The controls are a
table and a menu is that table answered against a snapshot, so what a menu
should read can be said on a machine with no display, no toolkit and no window
at all - which is exactly the claim the seam makes.
"""

from rederive.plot import controls, protocol


def offered(state):
    """What a menu offers a window in this state: the words and the first key.

    One pair per entry, in the order the menu draws them. A control with no key
    on this platform says so with an empty one rather than being left out, since
    a menu entry that answers to nothing is still a menu entry.
    """
    return [
        (entry.label, entry.keys[0] if entry.keys else "")
        for entry in controls.menu(state)
    ]


def ticked(state):
    """Which of a menu's entries are switched on, by the words they read."""
    return [entry.label for entry in controls.menu(state) if entry.checked]


class FakeWindow:
    """One window as the session sees one, and nothing else at all."""

    def __init__(self, session, kind, number, preferences):
        self.session = session
        self.kind = kind
        self.number = number
        #: The preferences this window was opened with, which is where a sticky
        #: grid or wire look would reach a real one.
        self.preferences = preferences
        self.plots = []
        self.current = False
        self.title = ""
        #: How many times a plot has landed here and been brought to the front.
        self.presented = 0

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

    def present(self, quietly=False):
        self.presented += 1
        self.quietly = quietly

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
        """What a window manager's close comes to: the session is told."""
        self.session.closed(self.number)


class FakeBackend:
    """Windows that are bookkeeping and no drawing."""

    def __init__(self):
        self.windows = {}
        #: Every window opened, in order, for a test that asks what the next
        #: one was built with.
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
    """An executor that runs each job where it was queued, before `submit` returns.

    The desktop's is a thread and the browser's will be a worker; this one is
    the caller, which is what makes a sampling in a test a function call with a
    return value.
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

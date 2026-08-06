#!/usr/bin/env python3
"""Record the plotting session the README shows, as one animated WebP.

Plotting happens in two windows at once - the expression is authored in the
terminal and drawn in a window of its own - so the film shows both, the algebra
window standing over the plot window it sends its work to. The algebra window is
kept short, tall enough for the expression being plotted and the menu under it,
because the picture is what the section is about.

The two are filmed apart and put together afterwards, which is what the two
passes are.

The **first pass** drives the app headlessly through Textual's pilot exactly as
`readme_demo.py` does, a keystroke at a time, with one difference: the plot host
is replaced by a stand-in that keeps every request instead of sending it down a
pipe. Nothing is drawn yet, and the app is none the wiser - it authors, it
classifies, and it says `Plotting #3` on the message line.

The **second pass** replays those requests into a real plot host built in this
process, and photographs its windows one plot at a time. The windows are laid
out and rendered but never mapped onto the screen, so a recording does not throw
windows over whatever else is running. Every picture is therefore drawn by the
program's own windows from the program's own requests, and a curve in the film is
a curve Rederive drew.

The plot window is then dressed in the same chrome Rich draws a terminal
screenshot in, titled the way the window manager titles it, and the two are
stacked into one frame.

Needs a display - the 3D windows want OpenGL, which nothing offscreen provides
here - and Inkscape on the path, and Pillow, which is asked for on the command
line rather than carried as a dependency of a project that never imports it. A
display that reports two device pixels per logical one is what the sizes below
are chosen for; one that does not gives a softer picture. Run from the
repository root:

    uv run --with pillow python tools/plot_demo.py [-o plot.webp]
"""

import argparse
import asyncio
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from PIL import Image
from textual.widgets import Input

import readme_demo
from readme_demo import FINALE, KEY, READ, RESULT

from rederive.plot import protocol as plots
from rederive.plot.proxy import PlotProxy
from rederive.ui.app import RederiveApp

#: How tall the algebra window stands, in rows. Five go to the menu, the
#: message and the status line, and the eight that are left are the worksheet:
#: enough for the tallest expression the script authors to stand whole, which
#: is what decides this number. The film is about what the plot window does
#: with an expression, not about the worksheet it came from, so the window is
#: no taller than it has to be.
ROWS = 13

#: How big a plot window is opened, in logical pixels. Landscape, because it
#: stands under a terminal of the same width and the whole picture has to stay
#: readable at half size in a README.
WINDOW = (900, 520)

#: How far a spin turns the camera between two frames, in degrees, and how long
#: such a frame stands. A surface turning is what says the 3D window is a view
#: of a solid thing rather than a picture of one, and it is also every byte of
#: the file: a frame that moves a whole wire mesh is a frame stored whole, and
#: costs what a hundred keystrokes do. So the steps are few and wide rather
#: than many and small. A spin turns one way and comes back, so it ends on the
#: view it started from and the film loops without a jump.
SPIN_DEGREES = 4.0
SPIN = 0.12

#: The three actions the polar flip is filmed as. The `polar` toggle is a
#: control of the picture and not a command of the app, so no keystroke stands
#: for it and nothing moves in the algebra window while it happens - and a
#: picture that rearranges itself with nothing visible causing it reads as a
#: fault. So the pointer that does it is drawn in: it reaches the control, the
#: control lights under it and the curve is reread, and it leaves again.
REACH = "reach"
POLAR = "polar"
AWAY = "away"

#: The pointer, drawn tip-first from the point it is pointing at, and how tall
#: it stands in the units the chrome is drawn in. White on a dark toolbar, with
#: an outline so that it survives the light one under it.
POINTER = "M0 0 L0 17 L4.4 13.1 L7.2 19.4 L10.2 18 L7.4 11.8 L12.8 11.6 Z"
POINTER_HEIGHT = 19.4
POINTER_SIZE = 22.0

#: The chrome Rich draws around a screenshot, in the units it writes one in: the
#: whole width, the padding a screenshot sits in, the title bar over it and the
#: strip under it. Read off an exported screenshot, and what makes the plot
#: window below match the terminal window above.
FRAME_WIDTH = 994
INSET = 9
TITLE_BAR = 41
FOOT = 10
#: The gap between the two windows in a finished frame, in output pixels.
GAP = 14

#: How long to wait for the sampling behind one plot, how long to let the
#: windows paint once it is in, and how long a camera move needs, which is one
#: repaint of a mesh that is already sampled. The wait has a deadline because a
#: job displaced by a newer one never reports back, and the film is not worth a
#: hang.
PATIENCE = 120.0
QUIET = 0.4
TURN = 0.15


class Recorder(readme_demo.Recorder):
    """The pilot with two cameras: the screen, and which plot is beside it.

    The plot pictures are not taken here - nothing is drawn during the first
    pass - so what a frame carries is the number of the picture that will stand
    beside it, and the requests that make those pictures are collected in the
    order they are asked for.
    """

    def __init__(self, pilot: Any) -> None:
        super().__init__(pilot)
        #: What the second pass replays: a plot request, the polar toggle, or
        #: so many degrees of camera turned over the surface last drawn.
        self.actions: list[plots.Add | float | str] = []
        #: Which picture stands beside each frame. Zero is the plot window
        #: before anything has been plotted into it.
        self.pictures: list[int] = []

    def snap(self, hold: float) -> None:
        super().snap(hold)
        self.pictures.append(len(self.actions))

    def asked(self, request: plots.Add) -> plots.Placed:
        """Take one plot request off the app, in place of the host.

        Answered as a plot that landed somewhere new, which is the truth for
        every plot in this script: each one is a fresh expression, so none of
        them replaces a curve already drawn.
        """
        self.actions.append(request)
        return plots.Placed(window=len(self.actions), replaced=False)

    async def plot(self, new: bool = False, hold: float = RESULT) -> None:
        """Plot the highlighted expression, into its own window or the last one."""
        await self.key("p", KEY)
        await self.key("n" if new else "p", hold)

    def polar(self, hold: float = RESULT) -> None:
        """Film the pointer reaching the last window's polar toggle and pressing it.

        Three frames: the pointer on the control, the control lit with the curve
        redrawn under it, and the picture alone once the pointer has gone.
        """
        for action, held in ((REACH, 0.9), (POLAR, 1.6), (AWAY, hold)):
            self.actions.append(action)
            self.snap(held)

    def spin(self, frames: int, last: float = RESULT) -> None:
        """Turn the camera over the surface just drawn, a frame at a step.

        Half the frames turn one way and half turn back, which leaves the view
        where it was found. The screen does not change over any of them, and is
        re-exported for each rather than remembered: the app is idle, so every
        export is the same picture, and the encoder stores such a frame as the
        rectangle that moved.
        """
        for index in range(frames):
            self.actions.append(-SPIN_DEGREES if index * 2 < frames else SPIN_DEGREES)
            self.snap(last if index == frames - 1 else SPIN)


class Standin(PlotProxy):
    """The plot host as far as the app can tell: it takes requests and keeps them.

    Nothing is spawned and nothing is drawn. The app classifies the expression,
    hands the request over and words its message line off the answer, which is
    all of the plot command that belongs in the algebra window.
    """

    def __init__(self, recorder: Recorder) -> None:
        super().__init__()
        self._recorder = recorder

    def add(self, request: plots.Add) -> plots.Placed:
        return self._recorder.asked(request)

    def prefer(self, preferences: plots.Prefer) -> None:
        pass

    def describe(self) -> tuple[plots.WindowInfo, ...]:
        return ()

    def shutdown(self) -> None:
        pass


async def play(rec: Recorder) -> None:
    """The session the animation shows.

    Six expressions and no two of a kind: a function, an equation, a pair in one
    parameter, a curve read the other way round, and two surfaces. Each gets a
    window to itself, so what the film argues is that Rederive reads what was
    written rather than plotting what it was handed. Two of the six are pictures
    everyone already knows, and they are there so the other four register as
    strange.
    """
    # The empty worksheet, and the plot window waiting under it.
    rec.snap(READ)

    # An envelope anyone can read, with the oscillation packing infinitely
    # tight against the origin inside it.
    await rec.author("x sin(2/x)")
    await rec.plot()

    # An equation is not a function, and what is drawn is where it holds.
    await rec.author("x^2 + y^2 = 4")
    await rec.plot(new=True)

    # A pair of expressions in one parameter is a path, not two curves - and
    # this one is a heart nobody would read off the formula.
    await rec.author("[sin(2t) + sin t, -cos(2t) - cos t]")
    await rec.plot(new=True)

    # The dullest curve in the film, until the window is told to read it as
    # r = f(θ) and the same samples fall into an eight-petal rose.
    await rec.author("2sin(4x)")
    await rec.plot(new=True)
    rec.polar(3.4)

    # Two variables make a surface, and open a window that can hold one.
    await rec.author("y(3x^2 - y^2)")
    await rec.plot()
    rec.spin(8)

    # A sombrero closes the show, turning for as long as the loop pauses.
    await rec.author("cos((x^2 + y^2)/4)/(3 + x^2 + y^2)")
    await rec.plot(new=True)
    rec.spin(16, last=FINALE)


async def record() -> Recorder:
    """The first pass: the algebra window, and the requests it sends out."""
    app = RederiveApp()
    async with app.run_test(size=(80, ROWS)) as pilot:
        rec = Recorder(pilot)
        # No host is started and no window is opened: the plot requests are
        # kept, and the second pass is what draws them.
        app.plots = Standin(rec)
        # A cursor that blinks on the wall clock would flicker at random in
        # frames spaced by script time, so it is held steady instead.
        app.query_one("#prompt-input", Input).cursor_blink = False
        # Let the opening screen finish painting before the first frame.
        await pilot.pause()
        await play(rec)
        return rec


# -- the plot windows ----------------------------------------------------------


class _Silent:
    """The pipe a host talks home on, with nobody at the other end.

    The host reports closed windows and moved controls to the app it was spawned
    by. Here it was spawned by nothing, and everything it has to say is dropped.
    """

    def send(self, message: Any) -> None:
        pass


def photograph(
    actions: list[plots.Add | float | str], directory: Path
) -> list[tuple[Path, str, tuple[float, float] | None]]:
    """Every picture the film shows, taken off the program's own plot windows.

    A host is built here rather than spawned, and asked to take the requests the
    first pass collected - the same handler the pipe would have delivered them
    to. Its windows are laid out and rendered but never mapped onto the screen,
    so this draws real pictures without putting a window in front of whatever
    the machine is doing.

    The first picture is the window before anything has been plotted, which is
    what stands beside the frames where the first expression is still being
    typed. The plot that follows lands in it, since a window that exists is
    where the next curve goes.
    """
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets

    from rederive.plot.qt import theme
    from rederive.plot.host import Host

    pg.setConfigOptions(antialias=True, imageAxisOrder="row-major")
    QtCore.QCoreApplication.setAttribute(
        QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True
    )
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    theme.dress(application)

    class Filmed(Host):
        """A host whose windows are the right size and are never shown."""

        pending = 0

        def _open(self, kind: plots.WindowKind) -> Any:
            window = super()._open(kind)
            window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, True)
            window.resize(*WINDOW)
            return window

        def sample(self, key: Any, work: Any, done: Any, report: Any = None) -> None:
            """Count the sampling out, so a picture is taken of a finished curve."""
            self.pending += 1

            def finished(answer: Any) -> None:
                self.pending -= 1
                done(answer)

            super().sample(key, work, finished, report)

    host = Filmed(_Silent())

    def settle() -> None:
        """Wait out the sampling, and let the windows paint what came of it."""
        deadline = time.monotonic() + PATIENCE
        while host.pending and time.monotonic() < deadline:
            application.processEvents()
            time.sleep(0.01)
        quiet()

    def quiet(seconds: float = QUIET) -> None:
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            application.processEvents()
            time.sleep(0.01)

    shots: list[tuple[Path, str, tuple[float, float] | None]] = []
    #: Where the pointer stands in the picture being taken, in its own pixels,
    #: or None for the frames it is not in - which is all but three of them.
    pointer: tuple[float, float] | None = None

    def shoot(window: Any) -> None:
        path = directory / f"{len(shots):04d}.png"
        window.grab().save(str(path))
        shots.append((path, window.windowTitle(), pointer))

    def hover(window: Any, under: bool) -> tuple[float, float] | None:
        """Put the polar toggle under the pointer, or take it out again.

        The button is left believing the pointer is on it, which is what the
        style sheet's `:hover` reads, so the control lights the way it would
        under a hand. Where it lands is asked of the button rather than
        guessed, and answered in the picture's pixels rather than the window's.
        """
        button = next(
            found
            for found in window.findChildren(QtWidgets.QToolButton)
            if found.defaultAction() is window.polar_toggle
        )
        button.setAttribute(QtCore.Qt.WidgetAttribute.WA_UnderMouse, under)
        button.update()
        if not under:
            return None
        middle = button.mapTo(
            window, QtCore.QPoint(button.width() // 2, button.height() // 2)
        )
        ratio = window.devicePixelRatioF()
        return (middle.x() * ratio, middle.y() * ratio)

    window = host._target(plots.Where.NEW, plots.WindowKind.TWO_D)
    window.show()
    settle()
    shoot(window)
    for action in actions:
        if isinstance(action, plots.Add):
            placed = host._add(action)
            if not isinstance(placed, plots.Placed):
                raise SystemExit(f"{action.label} was refused: {placed}")
            window = host.windows[placed.window]
            settle()
        elif action == REACH:
            pointer = hover(window, True)
            quiet(TURN)
        elif action == POLAR:
            window.polar_toggle.trigger()
            settle()
        elif action == AWAY:
            pointer = hover(window, False)
            quiet(TURN)
        else:
            window.view.orbit(action, 0.0)
            quiet(TURN)
        shoot(window)
    return shots


def framed(picture: Path, title: str, pointer: tuple[float, float] | None) -> str:
    """One plot window in the chrome Rich draws a terminal screenshot in.

    The same rounded card, the same three lights and the same title face, so
    that the two windows in a frame read as two windows of one desktop. The
    title is the window's own, which is the plots it holds - the window manager
    would be showing exactly this over it.

    The pointer, where there is one, is drawn over the picture rather than
    grabbed with it: a window renders without the cursor that is on it, and the
    one control the film works by hand needs a hand to be seen working it.
    """
    with Image.open(picture) as image:
        width, height = image.size
    inner = FRAME_WIDTH - 2 * INSET
    tall = round(inner * height / width, 1)
    hand = ""
    if pointer is not None:
        scale = POINTER_SIZE / POINTER_HEIGHT
        at = (INSET + pointer[0] * inner / width, TITLE_BAR + pointer[1] * tall / height)
        hand = (
            f'\n  <path d="{POINTER}" fill="#ffffff" stroke="#111318"'
            f' stroke-width="1.4" stroke-linejoin="round"'
            f' transform="translate({at[0]:.1f},{at[1]:.1f}) scale({scale:.3f})"/>'
        )
    return f"""<svg viewBox="0 0 {FRAME_WIDTH} {TITLE_BAR + tall + FOOT}"
     xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
  <rect fill="#292929" stroke="rgba(255,255,255,0.35)" stroke-width="1" rx="8"
        x="1" y="1" width="{FRAME_WIDTH - 2}" height="{TITLE_BAR + tall + FOOT - 2}"/>
  <text fill="#c5c8c6" text-anchor="middle" font-family="arial" font-size="18px"
        font-weight="bold" x="{FRAME_WIDTH / 2}" y="27">{escape(title)}</text>
  <g transform="translate(26,22)">
    <circle cx="0" cy="0" r="7" fill="#ff5f57"/>
    <circle cx="22" cy="0" r="7" fill="#febc2e"/>
    <circle cx="44" cy="0" r="7" fill="#28c840"/>
  </g>
  <image xlink:href="{picture.name}" x="{INSET}" y="{TITLE_BAR}"
         width="{inner}" height="{tall}"/>{hand}
</svg>
"""


def render(svgs: list[str], width: int, directory: Path) -> list[Path]:
    """Every SVG as a PNG `width` pixels across, in one run of Inkscape."""
    if not shutil.which("inkscape"):
        raise SystemExit("inkscape rasterizes the frames, and is not on PATH")
    paths = []
    for index, svg in enumerate(svgs):
        path = directory / f"panel{index:04d}.svg"
        path.write_text(svg)
        paths.append(path)
    subprocess.run(
        ["inkscape", "--export-type=png", f"--export-width={width}", *map(str, paths)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return [path.with_suffix(".png") for path in paths]


# -- the two, stacked ----------------------------------------------------------


def compose(
    screens: list[Path], panels: list[Path], pictures: list[int], directory: Path
) -> list[Path]:
    """Every frame of the film: the algebra window over the plot window it feeds.

    Transparent white between and around them, which is what Inkscape leaves
    outside the rounded corners and what the encoder then flattens to white.
    """
    below = [Image.open(path).convert("RGBA") for path in panels]
    frames = []
    for index, (screen, picture) in enumerate(zip(screens, pictures, strict=True)):
        with Image.open(screen) as opened:
            above = opened.convert("RGBA")
        plot = below[picture]
        width = max(above.width, plot.width)
        frame = Image.new(
            "RGBA", (width, above.height + GAP + plot.height), (255, 255, 255, 0)
        )
        frame.paste(above, ((width - above.width) // 2, 0), above)
        frame.paste(plot, ((width - plot.width) // 2, above.height + GAP), plot)
        path = directory / f"frame{index:04d}.png"
        frame.save(path)
        frames.append(path)
    return frames


def screens(frames: list[tuple[str, float]], width: int, directory: Path) -> list[Path]:
    """The terminal of every frame as a PNG, each distinct screen drawn once.

    A frame that only turns a surface leaves the screen alone, and dozens of
    them in a row are one picture as far as the terminal is concerned.
    """
    directory.mkdir()
    order: dict[str, int] = {}
    for svg, _ in frames:
        order.setdefault(svg, len(order))
    drawn = readme_demo.rasterize(list(order), width, directory)
    return [drawn[order[svg]] for svg, _ in frames]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "plot.webp",
        help="where to write the animation (default: plot.webp beside README.md)",
    )
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=1400,
        help="pixels across, twice the width the README shows it at (default: 1400)",
    )
    arguments = parser.parse_args()
    rec = asyncio.run(record())
    with tempfile.TemporaryDirectory() as name:
        directory = Path(name)
        plots_at = directory / "plots"
        plots_at.mkdir()
        shots = photograph(rec.actions, plots_at)
        panels = render(
            [framed(*shot) for shot in shots],
            arguments.width,
            plots_at,
        )
        above = screens(rec.frames, arguments.width, directory / "screens")
        frames = compose(above, panels, rec.pictures, directory)
        readme_demo.encode(frames, [hold for _, hold in rec.frames], arguments.output)
    seconds = sum(hold for _, hold in rec.frames)
    size = arguments.output.stat().st_size
    print(f"{arguments.output}: {len(frames)} frames, {seconds:.1f}s, {size} bytes")


if __name__ == "__main__":
    main()

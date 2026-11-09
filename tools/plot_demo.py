#!/usr/bin/env python3
"""Record the plotting session the README shows, as one animated WebP.

The animation shows the terminal above the plot window it sends plots to. The terminal
is kept short, just tall enough for the expression and the menu, since the plots are
the focus.

The two are recorded in separate passes and combined afterwards.

The **first pass** drives the app headlessly through Textual's pilot, as
`readme_demo.py` does, but with a stand-in plot host that records each request instead
of sending it to a host process. The app behaves normally, down to showing `Plotting #3`
on the message line.

The **second pass** replays the requests into a real plot session in this process and
captures its windows after each plot. The windows are rendered but never shown on
screen, so recording doesn't cover other windows. Every curve in the animation is drawn
by Rederive itself.

Each plot window is framed in the same chrome Rich draws around terminal screenshots,
with the window's title, and stacked under the terminal frame.

Requires a display (3D windows need OpenGL, unavailable offscreen), Inkscape on the
path, and Pillow, supplied on the command line rather than as a project dependency.
The sizes assume a display with two device pixels per logical pixel; others give a
softer picture. Run from the repository root:

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

#: Terminal height in rows: five for the menu, message and status lines, and eight for
#: the worksheet, enough for the tallest expression in the script.
ROWS = 13

#: Plot window size in logical pixels. Landscape, to sit under a terminal of the same
#: width and stay readable at half size in the README.
WINDOW = (900, 520)

#: Camera rotation per spin frame, in degrees, and the frame duration. Rotation shows
#: the surface is 3D, but each such frame is stored nearly whole and costs as much as a
#: hundred keystrokes, so there are few, wide steps. A spin turns one way and back, so
#: the loop has no jump.
SPIN_DEGREES = 4.0
SPIN = 0.12

#: The three steps of the polar toggle animation. The toggle is a plot window control,
#: not an app command, so nothing in the terminal shows it being used, and a picture
#: changing for no visible reason looks like a bug. So a pointer is drawn: it reaches the
#: control, the control highlights and the curve is redrawn, and the pointer leaves.
REACH = "reach"
POLAR = "polar"
AWAY = "away"

#: Pointer outline, with its tip at the origin, and its size in chrome units. It is
#: white with a dark outline, so it shows on light and dark toolbars.
POINTER = "M0 0 L0 17 L4.4 13.1 L7.2 19.4 L10.2 18 L7.4 11.8 L12.8 11.6 Z"
POINTER_HEIGHT = 19.4
POINTER_SIZE = 22.0

#: Geometry of Rich's screenshot chrome, in its SVG units: total width, padding, title
#: bar and bottom strip. Taken from an exported screenshot, so the plot window matches
#: the terminal above it.
FRAME_WIDTH = 994
INSET = 9
TITLE_BAR = 41
FOOT = 10
#: Gap between the two windows in a frame, in output pixels.
GAP = 14

#: Timeout for a plot's sampling, time to let windows paint afterwards, and time for a
#: camera move to repaint. The timeout exists because a job superseded by a newer one
#: never reports back.
PATIENCE = 120.0
QUIET = 0.4
TURN = 0.15


class Recorder(readme_demo.Recorder):
    """A recorder that also tracks which plot picture goes with each frame.

    Nothing is drawn in the first pass, so each frame records the index of its plot
    picture, and the plot requests are collected in order for the second pass.
    """

    def __init__(self, pilot: Any) -> None:
        super().__init__(pilot)
        #: Actions for the second pass: plot requests, polar toggle steps, or camera
        #: rotations in degrees.
        self.actions: list[plots.Add | float | str] = []
        #: Plot picture index for each frame; zero is the empty plot window.
        self.pictures: list[int] = []

    def snap(self, hold: float) -> None:
        super().snap(hold)
        self.pictures.append(len(self.actions))

    def asked(self, request: plots.Add) -> plots.Placed:
        """Record a plot request in place of the host.

        Every plot in this script is a new expression, so the answer always says it
        replaced nothing.
        """
        self.actions.append(request)
        return plots.Placed(window=len(self.actions), replaced=False)

    async def plot(self, new: bool = False, hold: float = RESULT) -> None:
        """Plot the highlighted expression in a new window or the last one."""
        await self.key("p", KEY)
        await self.key("n" if new else "p", hold)

    def polar(self, hold: float = RESULT) -> None:
        """Record the pointer pressing the last window's polar toggle.

        Three frames: the pointer on the control, the control highlighted with the curve
        redrawn, and the picture after the pointer leaves.
        """
        for action, held in ((REACH, 0.9), (POLAR, 1.6), (AWAY, hold)):
            self.actions.append(action)
            self.snap(held)

    def spin(self, frames: int, last: float = RESULT) -> None:
        """Rotate the camera around the last surface, one step per frame.

        Half the frames turn one way and half turn back, restoring the view. The
        terminal is idle, so each frame re-exports the same screen.
        """
        for index in range(frames):
            self.actions.append(-SPIN_DEGREES if index * 2 < frames else SPIN_DEGREES)
            self.snap(last if index == frames - 1 else SPIN)


class Standin(PlotProxy):
    """A stand-in plot host that records requests without spawning or drawing.

    The app still classifies the expression and shows its message line, which is all
    the terminal needs to show.
    """

    def __init__(self, recorder: Recorder) -> None:
        super().__init__()
        self._recorder = recorder

    async def add(self, request: plots.Add) -> plots.Placed:
        return self._recorder.asked(request)

    def prefer(self, preferences: plots.Prefer) -> None:
        pass

    def describe(self) -> tuple[plots.WindowInfo, ...]:
        return ()

    def shutdown(self) -> None:
        pass


async def play(rec: Recorder) -> None:
    """The session the animation shows.

    Six expressions of different kinds: a function, an equation, a parametric pair, a
    curve switched to polar, and two surfaces. Each gets its own window, showing that
    Rederive plots each according to what it is. Two are familiar pictures, so the
    others stand out.
    """
    # The empty worksheet, and the plot window waiting under it.
    rec.snap(READ)

    # A simple envelope with oscillation packing infinitely tight toward the origin.
    await rec.author("x sin(2/x)")
    await rec.plot()

    # An equation is drawn where it holds.
    await rec.author("x^2 + y^2 = 4")
    await rec.plot(new=True)

    # A pair of expressions in one parameter is a path, here an unexpected heart.
    await rec.author("[sin(2t) + sin t, -cos(2t) - cos t]")
    await rec.plot(new=True)

    # A dull curve, until read as r = f(θ) it becomes an eight-petal rose.
    await rec.author("2sin(4x)")
    await rec.plot(new=True)
    rec.polar(3.4)

    # Two variables make a surface, which opens a 3D window.
    await rec.author("y(3x^2 - y^2)")
    await rec.plot()
    rec.spin(8)

    # A sombrero closes the show, turning while the loop pauses.
    await rec.author("cos((x^2 + y^2)/4)/(3 + x^2 + y^2)")
    await rec.plot(new=True)
    rec.spin(16, last=FINALE)


async def record() -> Recorder:
    """First pass: record the terminal and collect the plot requests."""
    app = RederiveApp()
    async with app.run_test(size=(80, ROWS)) as pilot:
        rec = Recorder(pilot)
        # Collect plot requests instead of starting a host; the second pass draws them.
        app.plots = Standin(rec)
        # A blinking cursor would flicker randomly across frames, so disable blinking.
        app.query_one("#prompt-input", Input).cursor_blink = False
        # Let the opening screen finish painting before the first frame.
        await pilot.pause()
        await play(rec)
        return rec


# -- the plot windows ----------------------------------------------------------


def photograph(
    actions: list[plots.Add | float | str], directory: Path
) -> list[tuple[Path, str, tuple[float, float] | None]]:
    """Second pass: replay the plot requests and capture the program's plot windows.

    The plot session runs in this process instead of a host process, with the same
    routing. Its windows are rendered but never shown on screen.

    The first picture is the empty window, shown while the first expression is typed;
    the first plot then goes into that window.
    """
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets

    from rederive.plot.qt import theme
    from rederive.plot.qt.backend import QtBackend, ThreadExecutor
    from rederive.plot.session import PlotSession

    pg.setConfigOptions(antialias=True, imageAxisOrder="row-major")
    QtCore.QCoreApplication.setAttribute(
        QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True
    )
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    theme.dress(application)

    class Filmed(QtBackend):
        """A backend whose windows have a fixed size and are never shown on screen."""

        def open(self, session: Any, kind: Any, number: int, preferences: Any) -> Any:
            window = super().open(session, kind, number, preferences)
            window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, True)
            window.resize(*WINDOW)
            return window

    class Counted(PlotSession):
        """A session that counts pending samplings, so pictures show finished curves."""

        pending = 0

        def sample(self, key: Any, work: Any, done: Any, report: Any = None) -> None:
            self.pending += 1

            def finished(answer: Any) -> None:
                self.pending -= 1
                done(answer)

            super().sample(key, work, finished, report)

    host = Counted(Filmed(), ThreadExecutor())

    def settle() -> None:
        """Wait for sampling to finish, then let the windows paint."""
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
    #: Pointer position in the current picture's pixels, or None when not shown.
    pointer: tuple[float, float] | None = None

    def shoot(window: Any) -> None:
        path = directory / f"{len(shots):04d}.png"
        window.grab().save(str(path))
        shots.append((path, window.windowTitle(), pointer))

    def hover(window: Any, under: bool) -> tuple[float, float] | None:
        """Set or clear the hover state on the polar toggle and return the pointer spot.

        Setting `WA_UnderMouse` makes the style sheet's `:hover` apply, so the control
        highlights as under a real pointer. The position is the button's center, in
        the picture's device pixels.
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
            placed = host.add(action)
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
    """Frame a plot window picture in the chrome Rich draws around a screenshot.

    The rounded card, three buttons and title font match, so both windows look like
    part of one desktop. The title is the window's own, listing its plots.

    The pointer, if any, is drawn over the picture, since a window grab doesn't
    include the cursor.
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
    """Rasterize every SVG to a PNG `width` pixels wide, in one Inkscape run."""
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


# -- composing frames ----------------------------------------------------------


def compose(
    screens: list[Path], panels: list[Path], pictures: list[int], directory: Path
) -> list[Path]:
    """Compose each frame: the terminal above its plot window.

    The background is transparent white, matching what Inkscape leaves outside the
    rounded corners; the encoder flattens it to white.
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
    """Rasterize the terminal for every frame, rendering each distinct screen once.

    Spin frames repeat the same screen many times in a row.
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

#!/usr/bin/env python3
"""Record the animated session the README shows, as one animated WebP.

The app is driven headlessly through Textual's pilot, a keystroke at a time, and
every keystroke's screen is exported as Rich draws it. Each frame's SVG is
rasterized by Inkscape, and the frames are encoded as one animated WebP, every
frame held for as long as the script asks. The format is a raster one because a
browser plays it for free: the same session as an animated SVG is a couple of
thousand text nodes and a CSS animation per frame, all of them live for as long
as the README is on screen, which makes the page sluggish.

The script is the `play` coroutine: the two examples the Usage section walks
through, then a quadratic solved symbolically, some trigonometry, a matrix
inverse, and the Basel sum as the closing number. Every result it shows is
computed by the real engine while the recording runs, so the animation cannot
drift from what the program actually answers.

Needs Inkscape on the path, and Pillow, which is asked for on the command line
rather than carried as a dependency of a project that never imports it. Run
from the repository root:

    uv run --with pillow python tools/readme_demo.py [-o demo.webp]
"""

import argparse
import asyncio
import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image
from textual.widgets import Input

from rederive.ui.app import MODE_COMPUTE, RederiveApp

#: How long a frame stands, by what it shows: a character typed, a menu or
#: prompt answered, an expression put up to be read, and a result to be taken
#: in. The finale stays longest, being where the loop pauses before starting
#: over.
TYPE = 0.06
KEY = 0.7
READ = 1.5
RESULT = 2.4
FINALE = 5.0


class Recorder:
    """The pilot with a camera: every action ends by filming the screen."""

    def __init__(self, pilot):
        self.pilot = pilot
        #: The film: an exported SVG and how long it stands, per frame.
        self.frames: list[tuple[str, float]] = []

    async def settle(self):
        """Wait out whatever the engine is computing."""
        while self.pilot.app.mode == MODE_COMPUTE:
            await asyncio.sleep(0.02)
        await self.pilot.pause()

    def snap(self, hold):
        self.frames.append((self.pilot.app.export_screenshot(), hold))

    async def key(self, key, hold=KEY):
        await self.pilot.press(key)
        await self.settle()
        self.snap(hold)

    async def type(self, text, hold=KEY):
        """Type `text` a character per frame, and hold the completed line."""
        for character in text:
            await self.pilot.press(character)
            self.snap(TYPE)
        self.frames[-1] = (self.frames[-1][0], hold)

    async def author(self, text, hold=READ):
        await self.key("a", KEY)
        await self.type(text)
        await self.key("enter", hold)

    async def simplify(self, hold=RESULT):
        await self.key("s", KEY)
        await self.key("enter", hold)


async def play(rec: Recorder):
    """The session the animation shows."""
    # The opening notice.
    rec.snap(READ)

    # The Usage section's first example: author and simplify.
    await rec.author("((ax+b)^2 - (ax-b)^2) / ((cx+d)^2 - (cx-d)^2)")
    await rec.simplify()

    # Its second: the Gaussian integral through the Calculus menu.
    await rec.author("#e^(-x^2)")
    await rec.key("c")
    await rec.key("i")
    await rec.key("enter")  # the offered expression
    await rec.key("enter")  # the offered variable
    await rec.type("0")
    await rec.key("tab", 0.4)
    await rec.type("inf")
    await rec.key("enter", READ)
    await rec.simplify()

    # A quadratic solved symbolically: the formula, both roots.
    await rec.author("ax^2 + bx + c = 0")
    await rec.key("l")
    await rec.key("enter")  # the offered expression
    await rec.type("x")
    await rec.key("enter", 2.8)

    # Trigonometry: a multiple angle collapsed, a half angle folded in.
    await rec.author("sin(6x)/sin(3x)")
    await rec.simplify(2.2)
    await rec.author("(cos(a/2) + sin(a/2))^2")
    await rec.simplify(2.2)

    # A symbolic matrix inverted.
    await rec.author("[[a, b], [c, d]]^-1")
    await rec.simplify(2.8)

    # The Basel problem closes the show.
    await rec.author("1/k^2")
    await rec.key("c")
    await rec.key("s")
    await rec.key("enter")  # the offered expression
    await rec.key("enter")  # the offered variable
    await rec.key("tab", 0.4)  # the lower limit is already 1
    await rec.type("inf")
    await rec.key("enter", READ)
    await rec.simplify(FINALE)


async def record() -> list[tuple[str, float]]:
    app = RederiveApp()
    async with app.run_test(size=(80, 24)) as pilot:
        # A cursor that blinks on the wall clock would flicker at random in
        # frames spaced by script time, so it is held steady instead.
        app.query_one("#prompt-input", Input).cursor_blink = False
        # Let the opening screen finish painting before the first frame.
        await pilot.pause()
        rec = Recorder(pilot)
        await play(rec)
        return rec.frames


#: One text style Rich wrote out, e.g. `.terminal-123-r4 { fill: #ffff55 }`.
STYLE = re.compile(r"\.terminal-\d+-r\d+ \{ (?P<style>[^}]*?) \}")
#: The generated id every name in a frame is prefixed with.
PREFIX = re.compile(r"terminal-\d+-")
#: A face Rich offers to fetch over the network. Nothing is fetched here: the
#: rasterizer takes the font-family from the installed fonts, Fira Code if it
#: is one of them and the default monospace face otherwise.
FONT_FACE = re.compile(r"\s*@font-face \{.*?\n\s*\}", re.DOTALL)
#: The group holding everything drawn inside the terminal.
CONTENT = re.compile(r'<g transform="translate\(9, 41\)"[^>]*>')
#: The rect the terminal text is clipped to, which sizes the backdrop.
CLIP = re.compile(
    r'id="t-clip-terminal">\s*<rect x="0" y="0" width="([\d.]+)" height="([\d.]+)"'
)
#: A cell background. The black ones are the empty screen, painted as one
#: backdrop instead of a rect per run of cells.
BLACK = re.compile(r'<rect fill="#000000"[^>]*/>')

#: The box-drawing characters the layouts are built from: fraction bars, matrix
#: brackets, the integral sign's arcs, the rule over the menu. A terminal
#: stretches these to fill the whole character cell so they join up; a font in
#: an SVG does not reach across Rich's line spacing, and the integral sign
#: comes apart. Each one is therefore redrawn as strokes spanning the exact
#: cell, and its place in the text is blanked to keep the spacing.
BOXED = set("─═│┌┐└┘╭╮╰╯")
#: One run of styled text, as Rich writes it out.
RUN = re.compile(
    r'(?P<head><text class="t-(?P<class>r\d+)" x="(?P<x>[\d.]+)" y="(?P<y>[\d.]+)"'
    r'[^>]*>)(?P<body>[^<]*)</text>'
)
FILL = re.compile(r"fill: (#[0-9a-fA-F]{6})")

#: The screen's geometry as Rich lays it out: a character cell's width and
#: height, and how far a line's baseline sits below the top of its cell.
CELL = 12.2
LINE = 24.4
ASCENT = 18.5
#: How the strokes are drawn: the line weight, the gap between the two strokes
#: of `═`, and an arc's radius, which is half a cell so that the quarter turn
#: ends exactly on the cell's edge.
WEIGHT = 2.0
GAP = 2.0
RADIUS = CELL / 2


def _fmt(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _drawn(character: str, x0: float, top: float, count: int = 1) -> str:
    """The path commands for `count` cells of `character` starting at `x0`.

    Only the two horizontals are ever drawn more than a cell at a time; the
    rest take `count` of one. Corners meet edge centers exactly, and verticals
    span the full cell height, so strokes in adjacent cells join without gaps.
    """
    f = _fmt
    cx, mid, bottom = x0 + CELL / 2, top + LINE / 2, top + LINE
    x1, r = x0 + CELL * count, RADIUS
    match character:
        case "─":
            return f"M{f(x0)} {f(mid)}H{f(x1)}"
        case "═":
            return (
                f"M{f(x0)} {f(mid - GAP)}H{f(x1)}M{f(x0)} {f(mid + GAP)}H{f(x1)}"
            )
        case "│":
            return f"M{f(cx)} {f(top)}V{f(bottom)}"
        case "┌":
            return f"M{f(x0 + CELL)} {f(mid)}H{f(cx)}V{f(bottom)}"
        case "┐":
            return f"M{f(x0)} {f(mid)}H{f(cx)}V{f(bottom)}"
        case "└":
            return f"M{f(cx)} {f(top)}V{f(mid)}H{f(x0 + CELL)}"
        case "┘":
            return f"M{f(cx)} {f(top)}V{f(mid)}H{f(x0)}"
        case "╭":
            return (
                f"M{f(cx)} {f(bottom)}V{f(mid + r)}"
                f"A{f(r)} {f(r)} 0 0 1 {f(cx + r)} {f(mid)}"
            )
        case "╮":
            return (
                f"M{f(x0)} {f(mid)}A{f(r)} {f(r)} 0 0 1 {f(cx)} {f(mid + r)}"
                f"V{f(bottom)}"
            )
        case "╰":
            return (
                f"M{f(cx)} {f(top)}V{f(mid - r)}"
                f"A{f(r)} {f(r)} 0 0 0 {f(cx + r)} {f(mid)}"
            )
        case "╯":
            return (
                f"M{f(cx)} {f(top)}V{f(mid - r)}"
                f"A{f(r)} {f(r)} 0 0 1 {f(cx - r)} {f(mid)}"
            )
    raise ValueError(character)


def _vectorized(match: re.Match, styles: dict[str, str]) -> tuple[str, str]:
    """One text run with its box characters blanked, and the strokes drawing them."""
    body = html.unescape(match.group("body"))
    if not BOXED & set(body):
        return match.group(0), ""
    top = float(match.group("y")) - ASCENT
    x = float(match.group("x"))
    color = FILL.search(styles[match.group("class")]).group(1)
    characters = list(body)
    parts = []
    at = 0
    while at < len(characters):
        character = characters[at]
        if character not in BOXED:
            at += 1
            continue
        count = 1
        if character in "─═":
            while at + count < len(characters) and characters[at + count] == character:
                count += 1
        parts.append(_drawn(character, x + at * CELL, top, count))
        characters[at : at + count] = "\xa0" * count
        at += count
    body = escape("".join(characters)).replace("\xa0", "&#160;")
    path = (
        f'<path fill="none" stroke="{color}" stroke-width="{_fmt(WEIGHT)}" '
        f'd="{"".join(parts)}"/>'
    )
    return f"{match.group('head')}{body}</text>", path


def redraw(svg: str) -> str:
    """One frame ready to rasterize: no web fonts, box drawing as strokes.

    Rich stamps every screenshot's ids with a hash of its content; the stamp is
    cut off so that a frame reads the same as any other. The empty screen
    arrives as a black rect per run of cells, and the runs meet at fractional
    coordinates where a seam then shows, so one backdrop is painted instead.
    The box-drawing characters leave the text and come back as strokes,
    appended last so they are drawn over the cells they span.
    """
    styles = {
        match.group(0).split()[0].split("-")[-1]: match.group("style")
        for match in STYLE.finditer(svg)
    }
    svg = FONT_FACE.sub("", PREFIX.sub("t-", svg))
    opening = CONTENT.search(svg)
    width, height = CLIP.search(svg).groups()
    head = svg[: opening.end()]
    content = BLACK.sub("", svg[opening.end() :])
    paths: list[str] = []

    def vectorize(match: re.Match) -> str:
        text, path = _vectorized(match, styles)
        if path:
            paths.append(path)
        return text

    content = RUN.sub(vectorize, content)
    # The strokes belong to the terminal's group, which is the last to close.
    at = content.rindex("</g>")
    backdrop = f'<rect fill="#000000" x="0" y="0" width="{width}" height="{height}"/>'
    return head + backdrop + content[:at] + "".join(paths) + content[at:]


def rasterize(frames: list[str], width: int, directory: Path) -> list[Path]:
    """Every frame as a PNG `width` pixels across, in one run of Inkscape."""
    if not shutil.which("inkscape"):
        raise SystemExit("inkscape rasterizes the frames, and is not on PATH")
    paths = []
    for index, svg in enumerate(frames):
        path = directory / f"{index:04d}.svg"
        path.write_text(redraw(svg))
        paths.append(path)
    subprocess.run(
        ["inkscape", "--export-type=png", f"--export-width={width}", *map(str, paths)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return [path.with_suffix(".png") for path in paths]


def encode(pngs: list[Path], holds: list[float], output: Path):
    """The frames as one looping animated WebP, each held for its own time.

    Lossless, the terminal being flat-colored text that a lossy codec would
    only smear, and `minimize_size` so that consecutive frames are stored as
    the rectangle that changed - which for a keystroke is one character cell.
    The compression method is the default four: six takes twenty times as long
    to save half a percent.
    """
    images = [Image.open(png).convert("RGB") for png in pngs]
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=[round(hold * 1000) for hold in holds],
        loop=0,
        lossless=True,
        method=4,
        minimize_size=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "demo.webp",
        help="where to write the animation (default: demo.webp beside README.md)",
    )
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=1400,
        help="pixels across, twice the width the README shows it at (default: 1400)",
    )
    arguments = parser.parse_args()
    frames = asyncio.run(record())
    with tempfile.TemporaryDirectory() as directory:
        pngs = rasterize([svg for svg, _ in frames], arguments.width, Path(directory))
        encode(pngs, [hold for _, hold in frames], arguments.output)
    seconds = sum(hold for _, hold in frames)
    size = arguments.output.stat().st_size
    print(f"{arguments.output}: {len(frames)} frames, {seconds:.1f}s, {size} bytes")


if __name__ == "__main__":
    main()

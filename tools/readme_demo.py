#!/usr/bin/env python3
"""Record the animated session the README shows, as one self-contained SVG.

The app is driven headlessly through Textual's pilot, a keystroke at a time, and
every keystroke's screen is exported as Rich draws it. The frames are then fused
into a single SVG: the window chrome, the clip paths and the text styles are
written once, each frame becomes a group of its own, and a stepped CSS animation
shows the groups one after another - which is the one kind of animation an
<img> tag on GitHub will play. A viewer with animations off sees the final
frame, the worksheet with everything on it.

The script is the `play` coroutine: the two examples the Usage section walks
through, then a quadratic solved symbolically, some trigonometry, a matrix
inverse, and the Basel sum as the closing number. Every result it shows is
computed by the real engine while the recording runs, so the animation cannot
drift from what the program actually answers.

Run from the repository root:

    uv run python tools/readme_demo.py [-o demo.svg]
"""

import argparse
import asyncio
import html
import re
from pathlib import Path
from xml.sax.saxutils import escape

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
#: The group holding everything drawn inside the terminal.
CONTENT = re.compile(r'<g transform="translate\(9, 41\)"[^>]*>')
#: The rect the terminal text is clipped to, which sizes the backdrop.
CLIP = re.compile(
    r'id="t-clip-terminal">\s*<rect x="0" y="0" width="([\d.]+)" height="([\d.]+)"'
)
#: A cell background. The black ones are the empty screen, painted once for
#: the whole animation instead of once per frame.
BLACK = re.compile(r'<rect fill="#000000"[^>]*/>')
#: A text element; the ones holding nothing visible are dropped.
TEXT = re.compile(r"<text[^>]*>([^<]*)</text>")
#: A class attribute naming one of a frame's text styles.
CLASS = re.compile(r'class="t-(r\d+)"')

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


def dissect(svg: str) -> tuple[dict[str, str], str]:
    """One frame's text styles by name, and its content with names normalized.

    Rich stamps every screenshot's ids with a hash of its content, so nothing
    in one frame can refer to another frame's defs until the stamp is cut off.
    The empty text runs and the black cell backgrounds are dropped here too:
    the first paint nothing, and the second are one backdrop drawn per frame.
    The box-drawing characters leave the text and come back as strokes.
    """
    styles = {
        match.group(0).split()[0].split("-")[-1]: match.group("style")
        for match in STYLE.finditer(svg)
    }
    content = svg[CONTENT.search(svg).end() : svg.rindex("</g>")]
    content = PREFIX.sub("t-", content)
    content = BLACK.sub("", content)
    paths: list[str] = []

    def vectorize(match: re.Match) -> str:
        text, path = _vectorized(match, styles)
        if path:
            paths.append(path)
        return text

    content = RUN.sub(vectorize, content)
    content = TEXT.sub(
        lambda m: "" if not m.group(1).replace("&#160;", "").strip() else m.group(0),
        content,
    )
    return styles, content + "".join(paths)


def keyframes(index: int, start: float, end: float, total: float, last: bool) -> str:
    """The animation showing frame `index` from `start` to `end` seconds.

    The timing function is step-end, so opacity holds each keyframe's value
    until the next one: hidden to `start`, shown to `end`, hidden to the loop's
    end. The last frame stays up instead, and is also the one frame visible
    without animation, so a still viewer sees the finished worksheet.
    """
    a = f"{start / total:.4%}".rstrip("%")
    b = f"{end / total:.4%}".rstrip("%")
    on, off = "{opacity:1}", "{opacity:0}"
    if index == 0:
        frames = f"0%{on}{b}%{off}100%{off}"
    elif last:
        frames = f"0%{off}{a}%{on}100%{on}"
    else:
        frames = f"0%{off}{a}%{on}{b}%{off}100%{off}"
    return (
        f"@keyframes k{index}{{{frames}}}\n"
        f"#f{index}{{animation:k{index} {total:.2f}s step-end infinite}}"
    )


def assemble(frames: list[tuple[str, float]]) -> str:
    """All the frames as one animated SVG."""
    first = frames[0][0]
    # Everything outside the terminal group is identical from frame to frame:
    # the fonts, the window, the clip paths. It is kept from the first frame,
    # its content styles cut out and replaced with the shared ones.
    head = PREFIX.sub("t-", first[: CONTENT.search(first).start()])
    head = re.sub(r"\.t-r\d+ \{[^}]*\}\n?", "", head)

    classes: dict[str, str] = {}
    groups = []
    total = sum(hold for _, hold in frames)
    clock = 0.0
    animations = []
    for index, (svg, hold) in enumerate(frames):
        styles, content = dissect(svg)
        remap = {}
        for name, style in styles.items():
            remap[name] = classes.setdefault(style, f"g{len(classes)}")
        content = CLASS.sub(lambda m: f'class="{remap[m.group(1)]}"', content)
        groups.append(f'<g class="f" id="f{index}">{content}</g>')
        animations.append(
            keyframes(index, clock, clock + hold, total, index == len(frames) - 1)
        )
        clock += hold

    width, height = CLIP.search(head).groups()
    # The finale is the one frame shown without animations: a renderer that
    # plays none - a thumbnail, a feed reader - shows the finished worksheet.
    # While the animation runs it takes precedence over the static rule.
    shared = "\n".join(
        [".f{opacity:0}", f"#f{len(frames) - 1}{{opacity:1}}"]
        + [f".{name}{{{style}}}" for style, name in classes.items()]
        + animations
    )
    head = head.replace("</style>", shared + "\n</style>")
    backdrop = f'<rect fill="#000000" x="0" y="0" width="{width}" height="{height}"/>'
    opening = PREFIX.sub("t-", CONTENT.search(first).group(0))
    return f"{head}{opening}\n{backdrop}\n" + "\n".join(groups) + "\n</g>\n</svg>\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "demo.svg",
        help="where to write the animation (default: demo.svg beside README.md)",
    )
    arguments = parser.parse_args()
    frames = asyncio.run(record())
    svg = assemble(frames)
    arguments.output.write_text(svg)
    seconds = sum(hold for _, hold in frames)
    print(f"{arguments.output}: {len(frames)} frames, {seconds:.1f}s, {len(svg)} bytes")


if __name__ == "__main__":
    main()

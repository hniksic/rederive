#!/usr/bin/env python3
"""Assemble the web build: a directory a static file server can serve as it is.

    python3 tools/build_web.py                 # into build/web
    python3 tools/build_web.py --output DIR
    python3 tools/build_web.py --offline       # from what is already cached

What it gathers, and where each part comes from:

* the Pyodide runtime and the sympy, mpmath and numpy wheels, from the CDN's
  precompiled `pyc` channel;
* xterm.js and its fit, unicode11, WebGL and canvas addons, uPlot and three.js,
  from npm;
* this package, built with `uv build`, and the pure-Python wheels it and
  Textual stand on, at the versions the working environment holds - so what
  the page runs is what the suite was run against;
* `web/`, which is the page itself.

**Nothing is fetched at run time.** Every URL the page names is under the
directory this writes, which is what lets the build work offline, behind a
proxy, and on a static host that is not allowed to reach anywhere else.

Two things about that directory are worth knowing before it is served.

It is large and it compresses well: about 62 MB as it sits, about 24 MB over a
connection whose server compresses, and nearly all of it is the precompiled
sympy wheel, numpy and the stdlib. The JavaScript is a rounding error beside
them - three.js, the biggest piece of it, is 0.8 MB and 0.2 MB compressed.
`python -m http.server` does not compress at all, which is fine for a look and
wrong for a page anyone else is going to load; GitHub Pages does.

And it needs no headers. There is no `SharedArrayBuffer` here and no
`setInterruptBuffer`, so no COOP or COEP is called for: the abort is a
terminated worker, which needs nothing of the host.

The `pyc` channel is a decision rather than a knob, and stage 0 is where it was
made: four times faster to a usable prompt and four times faster to recover
from an Esc, for 4.7 MB more over the wire. `--channel full` is here for a
Pyodide release that has no precompiled build, and costs what stage 0 measured.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

#: Which Pyodide the page runs on. Pinned rather than tracked: anything older
#: than 314 ships sympy 1.13.3, which is below this package's floor, and a
#: release the build has not been tried on is not one to hand a user.
PYODIDE = "314.0.3"

#: Where a release's files live. The `pyc` channel is a directory beside the
#: default one holding the same packages with their bytecode already compiled.
CDN = "https://cdn.jsdelivr.net/pyodide/v{version}/{channel}/"

#: The runtime, which is the same set of files on either channel. `pyodide.mjs`
#: is the module entry, and the only one anything here names by hand: a worker
#: cannot load Pyodide any other way, `importScripts` being refused outright.
RUNTIME = (
    "pyodide.mjs",
    "pyodide.asm.mjs",
    "pyodide.asm.wasm",
    "python_stdlib.zip",
    "pyodide-lock.json",
)

#: What the engine worker loads out of the distribution, by the names the lock
#: file knows them under. Numpy is here because every plot sample is one, and
#: it is loaded by the worker alone: the main thread routes plots and draws a
#: terminal, and loads no mathematics at all.
DISTRIBUTED = ("sympy", "mpmath", "numpy")

#: The terminal, the four addons the page loads it with, and the two libraries
#: the plot panes draw with. unicode11 is insurance rather than necessity -
#: stage 0 measured every character the display can emit as one cell either way
#: - and canvas is the fallback for a machine that cannot have a WebGL context,
#: the two being pixel-identical. uPlot is 21 kB and draws the axes, their
#: numbers and the ruling; the curves are drawn on its canvas by `plot2d.js`.
#: three.js is the largest thing here that is not Python - 190 kB over a
#: compressing connection for the module, its core and one control - and it is
#: what a solid is put on a card by.
NPM = (
    "@xterm/xterm@^6.0.0",
    "@xterm/addon-fit@^0.11.0",
    "@xterm/addon-unicode11@^0.9.0",
    "@xterm/addon-webgl@^0.19.0",
    "@xterm/addon-canvas@^0.7.0",
    "uplot@^1.6.32",
    "three@^0.185.0",
)

#: Which file of each package the page loads, and which directory of the built
#: page it lands in. The UMD and IIFE builds, because they are what these
#: packages ship for a plain `<script>` tag and what their own examples use.
#: three.js is the exception and is taken as modules, which is the only form it
#: ships in: the minified module, the core it imports beside it, and the one
#: control out of the examples tree that the panes use. Nothing else of that
#: tree is copied - it is a few megabytes of loaders and post-processing for a
#: example that orbits a mesh.
BUNDLED = {
    "@xterm/xterm": ("lib/xterm.js", "css/xterm.css"),
    "@xterm/addon-fit": ("lib/addon-fit.js",),
    "@xterm/addon-unicode11": ("lib/addon-unicode11.js",),
    "@xterm/addon-webgl": ("lib/addon-webgl.js",),
    "@xterm/addon-canvas": ("lib/addon-canvas.js",),
    "uplot": ("dist/uPlot.iife.min.js", "dist/uPlot.min.css"),
    "three": (
        "build/three.module.min.js",
        "build/three.core.min.js",
        "examples/jsm/controls/OrbitControls.js",
    ),
}

#: Which directory each package's files are served from. One name per package,
#: because a page that names `xterm/xterm.js` and `uplot/uPlot.min.css` says
#: where each thing came from.
SERVED = {"uplot": "uplot", "three": "three"}

#: What the app instance needs beside this package: Textual and what Textual
#: stands on, every one of them pure Python. Sympy is not here and must not be
#: - the main thread draws the screen and knows no mathematics, and the worker
#: gets its own copy out of the Pyodide distribution.
DEPENDENCIES = (
    "textual",
    "rich",
    "markdown-it-py",
    "mdit-py-plugins",
    "mdurl",
    "linkify-it-py",
    "uc-micro-py",
    "platformdirs",
    "pygments",
    "typing_extensions",
)

def main(arguments: list[str] | None = None) -> int:
    parsed = _command_line(arguments)
    output = Path(parsed.output).resolve()
    cache = Path(parsed.cache).resolve()
    pyodide = _pyodide(cache, parsed.version, parsed.channel, parsed.offline)
    terminal = _terminal(cache, parsed.offline)
    wheels = _wheels(cache, parsed.offline)

    _fresh(output)
    for name in RUNTIME:
        _copy(pyodide / name, output / "pyodide" / name)
    distributed = _distributed(pyodide)
    for name in distributed:
        _copy(pyodide / name, output / "pyodide" / name)
    for package, source in terminal:
        _copy(source, output / SERVED.get(package, "xterm") / source.name)
    for source in wheels:
        _copy(source, output / "wheels" / source.name)
    for source in sorted((ROOT / "web").iterdir()):
        if source.is_file():
            _copy(source, output / source.name)

    package = [name for name in wheels if name.name.startswith("rederive-")]
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "pyodide": "pyodide/",
                "channel": parsed.channel,
                "version": parsed.version,
                # This package's own version, read off the wheel just built. The
                # loading screen says it, and there is no Python running yet for
                # the page to ask.
                "rederive": package[0].name.split("-")[1] if package else "",
                # What each Pyodide loads. The page's own instance takes the
                # app and no mathematics; the worker takes the engine and the
                # sympy that answers with it.
                "page": [f"wheels/{name.name}" for name in wheels],
                "worker": [f"wheels/{name.name}" for name in package],
                "packages": list(DISTRIBUTED),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"{output}: {_megabytes(output):.1f} MB", file=sys.stderr)
    return 0


def _command_line(arguments: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the web directory.")
    parser.add_argument("--output", default=str(ROOT / "build" / "web"))
    parser.add_argument("--cache", default=str(ROOT / "build" / "vendor"))
    parser.add_argument("--version", default=PYODIDE, help="which Pyodide to vendor")
    parser.add_argument("--channel", default="pyc", choices=("pyc", "full"))
    parser.add_argument(
        "--offline",
        action="store_true",
        help="build from what is cached already, fetching nothing",
    )
    return parser.parse_args(arguments)


# -- the pieces ---------------------------------------------------------------


def _pyodide(cache: Path, version: str, channel: str, offline: bool) -> Path:
    """The runtime and the packages the worker loads, downloaded once and kept.

    The lock file says what a package's wheel is called, so the two are fetched
    in that order and nothing here has to know a wheel's version.
    """
    directory = cache / f"pyodide-{version}-{channel}"
    directory.mkdir(parents=True, exist_ok=True)
    base = CDN.format(version=version, channel=channel)
    for name in RUNTIME:
        _fetch(base + name, directory / name, offline)
    for name in _distributed(directory):
        _fetch(base + name, directory / name, offline)
    return directory


def _distributed(directory: Path) -> list[str]:
    """The wheel file names of the packages the worker takes from Pyodide."""
    lock = json.loads((directory / "pyodide-lock.json").read_text())
    packages = lock["packages"]
    return [packages[name]["file_name"] for name in DISTRIBUTED]


def _terminal(cache: Path, offline: bool) -> list[tuple[str, Path]]:
    """The page's JavaScript, installed with npm and read out of the tree."""
    directory = cache / "npm"
    directory.mkdir(parents=True, exist_ok=True)
    if not offline:
        # `--legacy-peer-deps` because addon-canvas 0.7.0 still declares a peer
        # of xterm ^5 while xterm is at 6.0.0, which npm refuses outright. It
        # is a stale range and not an incompatibility: stage 0 loaded that
        # addon on xterm 6 in both browsers and found it drew pixel for pixel
        # what the WebGL renderer drew.
        install = ["npm", "install", "--silent", "--no-package-lock"]
        _run([*install, "--legacy-peer-deps", *NPM], directory)
    found = []
    for package, files in BUNDLED.items():
        for name in files:
            path = directory / "node_modules" / Path(package) / name
            if not path.is_file():
                raise SystemExit(f"{path}: not installed - run without --offline")
            found.append((package, path))
    return found


def _wheels(cache: Path, offline: bool) -> list[Path]:
    """This package and what the app instance stands on, as wheels.

    Built and downloaded at the versions this environment has, so that the
    browser runs the Textual the suite ran against rather than whatever PyPI
    resolves to on the day.

    This package is built every time, `--offline` or not. It is the one thing
    here that comes from the working tree rather than from a network, and a
    build that served yesterday's wheel while the page beside it was today's
    would be the worst kind of stale: everything looks rebuilt and half of it is.
    """
    directory = cache / "wheels"
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("rederive-*.whl"):
        stale.unlink()
    _run(["uv", "build", "--wheel", "--out-dir", str(directory)], ROOT)
    if not offline:
        held = _versions()
        pinned = [f"{name}=={held[_normal(name)]}" for name in DEPENDENCIES]
        _run(
            [
                "uv",
                "run",
                "--with",
                "pip",
                "--",
                "python",
                "-m",
                "pip",
                "download",
                "--no-deps",
                "--only-binary",
                ":all:",
                "--dest",
                str(directory),
                *pinned,
            ],
            ROOT,
        )
    found = sorted(directory.glob("*.whl"))
    if not found:
        raise SystemExit(f"{directory}: no wheels - run without --offline")
    return found


def _versions() -> dict[str, str]:
    """What version of everything the project's own environment holds.

    Asked of `uv run` rather than of this interpreter, since this script may be
    run by any Python on the machine and the answer that matters is the one the
    suite runs against.
    """
    program = (
        "import json;from importlib.metadata import distributions;"
        "print(json.dumps({d.metadata['Name']: d.version for d in distributions()}))"
    )

    answer = subprocess.run(
        ["uv", "run", "python", "-c", program],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return {_normal(name): held for name, held in json.loads(answer.stdout).items()}


def _normal(name: str) -> str:
    """One spelling of a package name, since a wheel and its metadata differ.

    `pygments` is `Pygments` to its own metadata and `typing_extensions` is
    `typing-extensions`, and neither difference is worth a table.
    """
    return name.lower().replace("_", "-")


# -- doing things -------------------------------------------------------------


def _fetch(url: str, path: Path, offline: bool) -> None:
    """Download one file, unless it is already here."""
    if path.is_file():
        return
    if offline:
        raise SystemExit(f"{path}: not cached - run without --offline")
    print(f"fetching {url}", file=sys.stderr)
    with urllib.request.urlopen(url) as answer, open(path, "wb") as file:
        shutil.copyfileobj(answer, file)


def _run(command: list[str], where: Path) -> None:
    print(" ".join(command), file=sys.stderr)
    subprocess.run(command, cwd=where, check=True)


def _fresh(output: Path) -> None:
    """Empty the output directory, so that nothing stale is ever served."""
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _megabytes(directory: Path) -> float:
    return sum(f.stat().st_size for f in directory.rglob("*") if f.is_file()) / 1e6


if __name__ == "__main__":
    sys.exit(main())

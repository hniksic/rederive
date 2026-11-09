#!/usr/bin/env python3
"""Assemble the web build: a directory a static file server can serve as is.

    python3 tools/build_web.py                 # into build/web
    python3 tools/build_web.py --output DIR
    python3 tools/build_web.py --offline       # from what is already cached

Contents and their sources:

* the Pyodide runtime and the sympy, mpmath and numpy wheels, from the CDN's
  precompiled `pyc` channel;
* xterm.js with its fit, unicode11, WebGL and canvas addons, uPlot and three.js, from
  npm;
* this package, built with `uv build`, and the pure-Python wheels it and Textual
  depend on, at the versions in the working environment, so the page runs what the
  test suite ran against;
* `web/`, the page itself.

**Nothing is fetched at run time.** Every URL the page uses is inside the output
directory, so the build works offline, behind a proxy, and on a restricted static host.

The output is about 62 MB, or about 24 MB compressed, almost all of it the sympy wheel,
numpy and the stdlib (three.js, the largest JavaScript piece, is 0.8 MB). `python -m
http.server` doesn't compress, which is fine for local testing; GitHub Pages does.

No special headers are needed. Without `SharedArrayBuffer` or `setInterruptBuffer`,
there's no need for COOP or COEP; abort works by terminating the worker.

The `pyc` channel was chosen deliberately: it reaches a usable prompt four times faster
and recovers from Esc four times faster, for 4.7 MB more download. `--channel full` is
for a Pyodide release without a precompiled build.
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

#: Pinned Pyodide version. Releases before 314 ship sympy 1.13.3, below this package's
#: minimum, and untested releases shouldn't reach users.
PYODIDE = "314.0.3"

#: CDN location of a release. The `pyc` channel holds the same packages with bytecode
#: precompiled.
CDN = "https://cdn.jsdelivr.net/pyodide/v{version}/{channel}/"

#: Runtime files, the same on either channel. `pyodide.mjs` is the module entry point;
#: a worker can't load Pyodide via `importScripts`.
RUNTIME = (
    "pyodide.mjs",
    "pyodide.asm.mjs",
    "pyodide.asm.wasm",
    "python_stdlib.zip",
    "pyodide-lock.json",
)

#: Packages the engine worker loads from the distribution, by lock-file name. Numpy is
#: for plot sampling and is loaded only by the worker; the main thread loads no math.
DISTRIBUTED = ("sympy", "mpmath", "numpy")

#: xterm.js with four addons, plus the plot libraries. unicode11 is insurance (every
#: character the display emits measured as one cell either way), and canvas is a
#: pixel-identical fallback when WebGL is unavailable. uPlot (21 kB) draws the axes,
#: tick labels and grid, while `plot2d.js` draws the curves on its canvas. three.js
#: (190 kB compressed) renders 3D surfaces.
NPM = (
    "@xterm/xterm@^6.0.0",
    "@xterm/addon-fit@^0.11.0",
    "@xterm/addon-unicode11@^0.9.0",
    "@xterm/addon-webgl@^0.19.0",
    "@xterm/addon-canvas@^0.7.0",
    "uplot@^1.6.32",
    "three@^0.185.0",
)

#: Files the page loads from each package: the UMD and IIFE builds meant for plain
#: `<script>` tags. three.js ships only as modules, so it contributes the minified
#: module, its core, and the one control the panes use from the examples tree; the rest
#: of that tree is megabytes of unneeded loaders and effects.
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

#: Output directory per package (default `xterm`), so paths show where each file came
#: from.
SERVED = {"uplot": "uplot", "three": "three"}

#: Pure-Python dependencies of the main-thread app: Textual and its dependencies. Sympy
#: must not be here; the main thread does no math, and the worker gets sympy from the
#: Pyodide distribution.
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
                # This package's version, from the wheel just built, for the loading
                # screen to show before Python is running.
                "rederive": package[0].name.split("-")[1] if package else "",
                # What each Pyodide instance loads: the page gets the app without math
                # packages, and the worker gets the engine and sympy.
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
    """Download and cache the runtime and the worker's packages.

    The runtime, including the lock file, is fetched first, since the lock file gives
    the wheel names.
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
    """Return the wheel file names of the packages the worker loads from Pyodide."""
    lock = json.loads((directory / "pyodide-lock.json").read_text())
    packages = lock["packages"]
    return [packages[name]["file_name"] for name in DISTRIBUTED]


def _terminal(cache: Path, offline: bool) -> list[tuple[str, Path]]:
    """Install the page's JavaScript with npm and return the files to bundle."""
    directory = cache / "npm"
    directory.mkdir(parents=True, exist_ok=True)
    if not offline:
        # `--legacy-peer-deps` because addon-canvas 0.7.0 declares a peer dependency on
        # xterm ^5, which npm rejects with xterm 6.0.0. The range is just stale: the
        # addon was tested on xterm 6 in both browsers and matches the WebGL renderer
        # pixel for pixel.
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
    """Build this package's wheel and download the app's dependency wheels.

    Dependencies are pinned to the versions in this environment, so the browser runs
    the Textual the suite ran against rather than whatever PyPI resolves to today.

    This package is rebuilt every time, even with `--offline`, since it comes from the
    working tree and a stale wheel next to a fresh page would be confusing.
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
    """Return the installed package versions in the project's environment.

    Queried via `uv run`, since this script may run under any Python, and the
    project's environment is what the suite runs against.
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
    """Normalize a package name, e.g. `Pygments` -> `pygments` and `typing_extensions`
    -> `typing-extensions`."""
    return name.lower().replace("_", "-")


# -- helpers -----------------------------------------------------------------


def _fetch(url: str, path: Path, offline: bool) -> None:
    """Download one file, unless it is already cached."""
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
    """Empty the output directory so nothing stale is served."""
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

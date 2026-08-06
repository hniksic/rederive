# -*- mode: python ; coding: utf-8 -*-
"""What a release ships: a program that runs with no Python installed.

There are two forms of it, out of one analysis, because the two things people want
are different. Built from the repository root:

    uv run --group packaging pyinstaller packaging/rederive.spec

leaves `dist/rederive`, one file that can be downloaded and run wherever it lands.
It pays for that on every launch: a single file has nowhere to keep an unpacked
interpreter, so it writes one to a temporary directory each time it starts and
takes it away again on the way out. With Qt in it that is a quarter of a gigabyte
a launch, and the form is kept anyway - it is for trying the program rather than
keeping it, and anyone keeping it has the form below.

    uv run --group packaging pyinstaller packaging/rederive.spec \
        --distpath dist/tree -- --tree

leaves `dist/tree/rederive/`, the same program as a launcher with an `_internal`
directory beside it. Nothing is unpacked at run time because nothing is packed;
this is the form a release tars up for people who want the program installed
rather than tried. `RELEASING.md` says what is done with either.

Either form is built out of an environment carrying the `full` extra, which is what
`RELEASING.md` and the build workflow ask for. The plot toolkit and the two
libraries behind the clipboard and the memory gauge are opt-in for an install from
source, and a bundle built without them is a program that starts, computes and
refuses to plot - which is a supported way to install Rederive and not a way to
release it. `smoke.py` catches it: `--version` has to name a Qt.

`packaging/smoke.py` is what says whether a build works, and it should be run
against both: a bundle can import cleanly and still be missing a file it only reads
when the user asks for help.

Four things about this program decide what has to be written here.

The two data files are the first. Nothing in a bundle is found by walking a source
tree, so `help.txt` and `rederive.tcss` have to be named and placed under the package
directories that read them - `importlib.resources` for the one, the path Textual
works out from the class's module for the other. Both are read at start-up, so
forgetting either does not ship quietly: the program dies on the first frame.

Sympy is the second. What the analysis finds by reading imports is about two thirds
of it, and the third it leaves out is almost all parts this program never reaches -
`physics`, `parsing`, `stats`, benchmarks, deprecated shims. Almost. A handful of
live algorithms are in there too, reached by imports written inside the functions
that need them, and an engine that meets one in a bundle without it has its worker
die rather than answer. The corpus does not reach any of them, so this is insurance
and not a fix: it costs about eight megabytes, which is the cheaper side of the
trade.

PyOpenGL is the third, and it goes wrong the way `help.txt` would rather than the way
sympy might. It chooses its platform backend - `egl`, `glx`, the Windows and macOS
ones - by name at run time, so the analysis reads no import for any of them and a
bundle built without this carries none. The build says nothing and the program starts,
computes and draws a 2D plot; it dies the first time someone asks for a 3D one, with a
`'NoneType' object is not callable` out of `OpenGL.platform`. Taking the package whole
is what avoids that.

The test dependencies are the fourth, and they are only weight. Nothing in the app
imports them; they are excluded because the environment the build runs in has them
installed and the analysis would otherwise follow them in.

UPX is off deliberately. Compressed executables are what a great deal of Windows
malware looks like, and an unsigned binary already starts with as much suspicion as
it can carry.
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).parent  # noqa: F821 - PyInstaller defines SPECPATH
PACKAGE = ROOT / "src" / "rederive"

#: Which of the two forms to build. PyInstaller hands a spec whatever follows `--`
#: on its own command line, which is the only way to ask a spec a question.
TREE = "--tree" in sys.argv

a = Analysis(  # noqa: F821
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=[
        (str(PACKAGE / "model" / "help.txt"), "rederive/model"),
        (str(PACKAGE / "ui" / "rederive.tcss"), "rederive/ui"),
    ],
    hiddenimports=collect_submodules("sympy") + collect_submodules("OpenGL"),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "_pytest", "pytest_asyncio", "xdist", "setuptools"],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)  # noqa: F821

SETTINGS = dict(
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

if TREE:
    # The directory the tarball carries: the launcher, and `_internal` beside it
    # holding everything the one file would otherwise unpack on every run.
    exe = EXE(  # noqa: F821
        pyz, a.scripts, [], exclude_binaries=True, name="rederive", **SETTINGS
    )
    COLLECT(exe, a.binaries, a.datas, name="rederive")  # noqa: F821
else:
    exe = EXE(  # noqa: F821
        pyz, a.scripts, a.binaries, a.datas, [], name="rederive",
        runtime_tmpdir=None, **SETTINGS
    )

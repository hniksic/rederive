# -*- mode: python ; coding: utf-8 -*-
"""The single-file executable: one download that runs with no Python installed.

Built from the repository root:

    uv run --group packaging pyinstaller packaging/rederive.spec

which leaves `dist/rederive` (or `dist/rederive.exe`). `packaging/smoke.py` is what
says whether the result works; a bundle can import cleanly and still be missing a
file it only reads when the user asks for help.

Three things about this program decide what has to be written here.

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

The test dependencies are the third, and they are only weight. Nothing in the app
imports them; they are excluded because the environment the build runs in has them
installed and the analysis would otherwise follow them in.

UPX is off deliberately. Compressed executables are what a great deal of Windows
malware looks like, and an unsigned binary already starts with as much suspicion as
it can carry.
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).parent  # noqa: F821 - PyInstaller defines SPECPATH
PACKAGE = ROOT / "src" / "rederive"

a = Analysis(  # noqa: F821
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=[
        (str(PACKAGE / "model" / "help.txt"), "rederive/model"),
        (str(PACKAGE / "ui" / "rederive.tcss"), "rederive/ui"),
    ],
    hiddenimports=collect_submodules("sympy"),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "_pytest", "pytest_asyncio", "xdist", "setuptools"],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="rederive",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

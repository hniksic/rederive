"""Catalog of the original's demonstrations and where to download them.

Derive shipped six demo scripts (alternating comments and expressions) and three
galleries of expressions to plot. They are still the best way to show off the program,
but they aren't ours to ship, so this module only lists them: file name, menu title and
menu word, kind, and download URLs.

Fetching lives elsewhere, since the browser (`web/demos.js`) and the desktop
(`rederive.fetch`) download differently. Both share this list.

A script step is authored and simplified; a gallery step is authored and plotted.
Scripts happen to be `.DMO` files and galleries `.MTH`.

`SOURCES` are Internet Archive URLs that extract one file from an archived image: a zip
of the Derive 3.14 diskette and a rar of Derive 4. Both contain all the demos, so they
can be queried in parallel and the first answer used.

`IMAGE` is the whole 3.14 diskette as a single download, a fallback for when the
archive's extractor (which both `SOURCES` depend on) is down. It is about 300 KB rather
than a few, and must be unzipped locally.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DEMOS", "GALLERIES", "IMAGE", "SCRIPTS", "SOURCES", "Demo", "named"]

#: URL templates for a file inside each archived image. The file name is substituted
#: unescaped, which is fine since every catalog name is plain ASCII like `ALGEBRA.DMO`.
SOURCES = (
    "https://archive.org/download/derive314cas/DERIVE_3.14.zip/{file}",
    "https://archive.org/download/derive-for-dos-v-4/"
    "DERIVE%20for%20DOS%20V4.rar/DERIVE%20for%20DOS%2F{file}",
)

#: The whole Derive 3.14 diskette, containing every demonstration. It bypasses the
#: archive's extractor, which does fail in practice: the `derive-v-3` item's extractor
#: has consistently answered 503.
IMAGE = "https://archive.org/download/derive314cas/DERIVE_3.14.zip"

#: Menu headings for the two kinds of demonstration.
SCRIPTS = "Demonstrations"
GALLERIES = "Plot galleries"


@dataclass(frozen=True)
class Demo:
    """One of the original's demonstration files, and what to call it."""

    #: File name, used for downloading and storing it.
    file: str
    #: Menu title, naming the subject.
    title: str
    #: One-word title for the keyboard menu. Its capital letter is the key that selects
    #: it, as in the program's other menus.
    word: str
    #: Whether its steps are drawn rather than simplified.
    plotting: bool = False

    @property
    def group(self) -> str:
        """The menu heading it appears under."""
        return GALLERIES if self.plotting else SCRIPTS

    @property
    def urls(self) -> tuple[str, ...]:
        """All URLs the file can be downloaded from, in no particular order.

        The caller is expected to query them in parallel and take the first answer.
        """
        return tuple(source.format(file=self.file) for source in SOURCES)


#: The catalog in menu order: the six scripts in the original's order, then the three
#: galleries.
DEMOS: tuple[Demo, ...] = (
    Demo("ARITH.DMO", "Arithmetic", "Arithmetic"),
    Demo("ALGEBRA.DMO", "Algebra", "alGebra"),
    Demo("TRIG.DMO", "Trigonometry", "Trigonometry"),
    Demo("FUNCTION.DMO", "Elementary functions", "Functions"),
    Demo("CALCULUS.DMO", "Calculus", "Calculus"),
    Demo("MATRIX.DMO", "Vectors and matrices", "Matrices"),
    Demo("PLOT2D.MTH", "2D plots", "2D-plots", plotting=True),
    Demo("PLOTPARA.MTH", "Parametric plots", "Parametric", plotting=True),
    Demo("PLOT3D.MTH", "3D plots", "3D-plots", plotting=True),
)


def named(file: str) -> Demo | None:
    """Return the demonstration with file name `file`, or None.

    The match is case-sensitive, against the catalog's uppercase names.
    """
    for demo in DEMOS:
        if demo.file == file:
            return demo
    return None

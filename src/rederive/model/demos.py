"""The original's own demonstrations: what they are, and where they still are.

Derive shipped a tour of itself - six scripts of a comment and an expression,
over and over, and three galleries of expressions worth drawing - and those
files are still the best thing to show the program off with. They are not
Rederive's to ship, so what is here is a catalogue rather than the files: the
name each one has, what to call it in a menu, which of the two kinds of
demonstration it is, and the addresses it can be downloaded from.

Fetching is deliberately not here. A browser downloads with `fetch` and a
desktop with whatever a desktop has, and neither can do the other's; what both
need is the same nine names and the same URLs, which is why the list is Python
rather than living in the page that first wanted it. What a browser does when a
source will not answer *it* in particular is the browser's own business and is
in `web/demos.js`.

The two kinds differ in what a step does. A script's step is authored and
simplified; a gallery's is authored and drawn, which is what the galleries were
written for - the original's own first line is "Interesting expressions to plot
in a 2D-plot window". The extensions say the same thing by accident: the scripts
are `.DMO` and the galleries are `.MTH`.

**About the addresses.** Both are Internet Archive items opened one file at a
time out of an archive image - a zip of the Derive 3.14 diskette, a rar of
Derive 4. That endpoint is also the only one of theirs a page may read: the
whole-file download carries no CORS headers at all, so a browser cannot take
the zip and unpack it. Two images carry the same set, so every demonstration
can be asked of both at once and whichever answers first is the one that runs -
an image that is slow or refusing today costs nothing while the other is up.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DEMOS", "GALLERIES", "SCRIPTS", "SOURCES", "Demo", "named"]

#: Where a file of a DOS release is reached, by the name it has inside the
#: image. Formatted with the file name and nothing else: every name in the
#: catalogue is bare ASCII of the shape `ALGEBRA.DMO`, so there is nothing in
#: one for a URL to have to escape.
SOURCES = (
    "https://archive.org/download/derive314cas/DERIVE_3.14.zip/{file}",
    "https://archive.org/download/derive-for-dos-v-4/"
    "DERIVE%20for%20DOS%20V4.rar/DERIVE%20for%20DOS%2F{file}",
)

#: What a menu offers the two kinds under. A gallery is a demonstration too,
#: and the heading is what says how it differs.
SCRIPTS = "Demonstrations"
GALLERIES = "Plot galleries"


@dataclass(frozen=True)
class Demo:
    """One of the original's demonstration files, and what to call it."""

    #: The name the file has, which is the name it is downloaded and kept under.
    file: str
    #: What a menu calls it. The original named these by their subject and so
    #: does this, in the words a menu has room for.
    title: str
    #: Whether its steps are drawn rather than simplified.
    plotting: bool = False

    @property
    def group(self) -> str:
        """The heading it is offered under."""
        return GALLERIES if self.plotting else SCRIPTS

    @property
    def urls(self) -> tuple[str, ...]:
        """Every address the file can be had from, in no particular order.

        All of them, rather than one and a fallback: the caller is expected to
        ask them at once and take whichever answers, since asking one after the
        other adds the waits together.
        """
        return tuple(source.format(file=self.file) for source in SOURCES)


#: The catalogue, in the order a menu offers it: the six scripts as the original
#: ordered them - arithmetic first, calculus late - and then the three
#: galleries, from the flat to the solid.
DEMOS: tuple[Demo, ...] = (
    Demo("ARITH.DMO", "Arithmetic"),
    Demo("ALGEBRA.DMO", "Algebra"),
    Demo("TRIG.DMO", "Trigonometry"),
    Demo("FUNCTION.DMO", "Elementary functions"),
    Demo("CALCULUS.DMO", "Calculus"),
    Demo("MATRIX.DMO", "Vectors and matrices"),
    Demo("PLOT2D.MTH", "2D plots", plotting=True),
    Demo("PLOTPARA.MTH", "Parametric plots", plotting=True),
    Demo("PLOT3D.MTH", "3D plots", plotting=True),
)


def named(file: str) -> Demo | None:
    """The demonstration `file` names, or nothing where it names none.

    Matched as the catalogue spells it, which is how the original spelled it:
    these names are capitals and there is no second file of any of them.
    """
    for demo in DEMOS:
        if demo.file == file:
            return demo
    return None

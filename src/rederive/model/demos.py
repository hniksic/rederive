"""The original's own demonstrations: what they are, and where they still are.

Derive shipped a tour of itself - six scripts of a comment and an expression,
over and over, and three galleries of expressions worth drawing - and those
files are still the best thing to show the program off with. They are not
Rederive's to ship, so what is here is a catalogue rather than the files: the
name each one has, what to call it in a menu - a page's title and a menu band's
one word - which of the two kinds of demonstration it is, and the addresses it
can be downloaded from.

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

**About the addresses.** `SOURCES` are Internet Archive items opened one file
at a time out of an archive image - a zip of the Derive 3.14 diskette, a rar of
Derive 4. Two images carry the same set, so every demonstration can be asked of
both at once and whichever answers first is the one that runs: an image that is
slow or refusing today costs nothing while the other is up.

`IMAGE` is the same diskette taken whole, and it is there because both of those
go through one piece of machinery - the archive's own extractor - and that is
the piece observed to fail. It costs three hundred kilobytes rather than one,
and a zip has to be opened at the other end, so it is nobody's first choice; it
is the road that is still there when the others are not.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DEMOS", "GALLERIES", "IMAGE", "SCRIPTS", "SOURCES", "Demo", "named"]

#: Where a file of a DOS release is reached, by the name it has inside the
#: image. Formatted with the file name and nothing else: every name in the
#: catalogue is bare ASCII of the shape `ALGEBRA.DMO`, so there is nothing in
#: one for a URL to have to escape.
SOURCES = (
    "https://archive.org/download/derive314cas/DERIVE_3.14.zip/{file}",
    "https://archive.org/download/derive-for-dos-v-4/"
    "DERIVE%20for%20DOS%20V4.rar/DERIVE%20for%20DOS%2F{file}",
)

#: The whole of the Derive 3.14 diskette, as one file. Three hundred kilobytes
#: against a demonstration's one, and every demonstration is inside it.
#:
#: The road that does not go through the archive's extractor, and so the one
#: left when the extractor is the thing that is down - which is not a
#: hypothetical: the `derive-v-3` item's extractor answers nothing but 503 and
#: has for as long as this was being written. Whoever takes this road pays for
#: it in bytes and in having to open a zip, and gets all nine files for the
#: one download.
IMAGE = "https://archive.org/download/derive314cas/DERIVE_3.14.zip"

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
    #: The same thing in one word, for a menu whose words are typed at rather
    #: than clicked: the capital in it is the letter that chooses it, as the
    #: capital in every option of the program's own menus is.
    word: str
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
    """The demonstration `file` names, or nothing where it names none.

    Matched as the catalogue spells it, which is how the original spelled it:
    these names are capitals and there is no second file of any of them.
    """
    for demo in DEMOS:
        if demo.file == file:
            return demo
    return None

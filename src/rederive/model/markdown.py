"""Writing the worksheet as a Markdown document with LaTeX math, for reading elsewhere.

The document is written for renderers such as GitHub, Jupyter, Obsidian and Pandoc, and
is never read back. Each entry gets its number, its annotation as a line of text when
the annotations are saved, and the expression as a `$$` display block, each a paragraph
of its own:

    **\\#3:**

    Simp(\\#2)

    $$
    \\frac{1}{3}
    $$

A string standing alone, which is what the text of a Derive for Windows worksheet loads
as, is a paragraph of text instead of a formula.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from rederive.display import DisplayOptions, write_latex
from rederive.model.expr import Kind, Node

SUFFIX = ".md"

#: Characters that Markdown, or the math in it, may read as markup anywhere in a line.
_MARKUP = re.compile(r"([\\`*_\[\]<>#|~$&])")

#: What makes a line start a list or a rule, or turns it into a heading's underline: a
#: marker at the start, possibly after a number, and then a blank or nothing.
_LEADING = re.compile(r"^(\d*)([-+=.)])(?=\s|$)")


def write(
    entries: Iterable[tuple[int, str | None, Node]], options: DisplayOptions
) -> str:
    """The document for `entries`, each a number, an annotation or None, and a tree."""
    paragraphs: list[str] = []
    for number, annotation, node in entries:
        paragraphs.append(rf"**\#{number}:**")
        if annotation is not None:
            paragraphs.append(escaped(annotation))
        if node.kind is Kind.STRING:
            paragraphs.append(escaped(str(node.value)))
        else:
            paragraphs.append(f"$$\n{write_latex(node, options)}\n$$")
    return "\n\n".join(paragraphs) + "\n"


def escaped(text: str) -> str:
    """`text` as a paragraph that Markdown shows as it is."""
    text = _MARKUP.sub(r"\\\1", " ".join(text.split()))
    return _LEADING.sub(r"\1\\\2", text)

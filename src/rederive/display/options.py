"""Options passed to the renderer.

Plain data built by whoever owns the settings. The display layer doesn't import
`rederive.model.settings`, so a pane can render with throwaway options without needing a
`Settings` object.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The three values of the `TimesOperator` setting.
ASTERISK = "Asterisk"
DOT = "Dot"
IMPLICIT = "Implicit"


@dataclass(frozen=True)
class DisplayOptions:
    """The settings a render depends on.

    `compressed` is `DisplayFormat := Compressed`, and `notation_digits` is
    `NotationDigits`, the significant digits shown for numbers with a radix point.
    `height` is the number of rows the pane can show, or `None` for no limit; a render
    that would be taller turns its outermost divisions and matrices into linear forms
    until it fits.
    """

    times: str = DOT
    compressed: bool = False
    output_base: int = 10
    notation_digits: int = 6
    height: int | None = None

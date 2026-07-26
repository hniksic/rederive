"""What the caller tells the renderer.

Plain data, built by whoever owns the settings. The display layer does not
import `rederive.model.settings`: a pane that renders one expression with a
throwaway set of options must not have to own a `Settings` object to do it.
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

    `compressed` is `DisplayFormat := Compressed`, `notation_digits` is
    `NotationDigits`, the significant digits a number with a radix point is
    shown to, and `height` is the number of rows the pane can show. `None`
    means lay out freely; a render that would be taller degrades its outermost
    divisions and matrices to linear forms until it fits.
    """

    times: str = DOT
    compressed: bool = False
    output_base: int = 10
    notation_digits: int = 6
    height: int | None = None

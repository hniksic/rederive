"""Integration: what Simplify makes of an `INT` head.

`integral` is the one way in, and its answer is a function of the head alone. The
work is split four ways: `indefinite` dispatches an integral over the methods that
may answer it, `definite` puts right what sympy gets wrong between limits, `proof`
is what an answer has to pass before it is taken, and `form` chooses which
antiderivative is the one printed.
"""

from __future__ import annotations

from rederive.engine.integration.indefinite import integral
from rederive.engine.integration.proof import agrees, proved_zero

__all__ = ["agrees", "integral", "proved_zero"]

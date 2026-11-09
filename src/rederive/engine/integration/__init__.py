"""Integration: how Simplify evaluates an `INT` head.

`integral` is the entry point; its answer depends only on the head. `indefinite` tries
each integration method in turn, `definite` corrects what sympy gets wrong between
limits, `proof` checks an answer before it is accepted, and `form` chooses which
antiderivative is printed.
"""

from __future__ import annotations

from rederive.engine.integration.indefinite import integral
from rederive.engine.integration.proof import agrees, proved_zero

__all__ = ["agrees", "integral", "proved_zero"]

"""Model layer: expression trees and the session state, free of any UI.

The session is not re-exported here, and cannot be. `rederive.syntax` is built
on `model.expr`, and the session is built on the syntax package, so importing
the session from this `__init__` would make `import rederive.syntax` reach
back into a half-initialised model package. `Entry` and `Session` come from
`rederive.model.session`, the way `Settings` comes from
`rederive.model.settings`.
"""

from rederive.model.expr import Kind, Node

__all__ = ["Kind", "Node"]

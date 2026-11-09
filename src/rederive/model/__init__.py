"""Model layer: expression trees and the session state, free of any UI.

The session can't be re-exported here: `rederive.syntax` depends on `model.expr`, and
the session depends on `rederive.syntax`, so importing it here would create a circular
import. Import `Entry` and `Session` from `rederive.model.session` and `Settings` from
`rederive.model.settings`.
"""

from rederive.model.expr import Kind, Node

__all__ = ["Kind", "Node"]

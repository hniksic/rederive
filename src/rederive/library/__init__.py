"""The pre-defined utility library: Derive's utility functions, in force in every session.

Derive 6 loaded a utility file by itself the first time one of its functions was used.
Rederive has no files to load; instead every session starts with the definitions of
those functions made, as ordinary definitions in Derive's own language (`texts`). They
are not built-ins:

- A definition of one of the names, typed, read from a file or restored from a
  worksheet's state, replaces the library's for the session, and an empty definition
  leaves an arbitrary function. Clearing the functions or variables removes the
  library's too. None of it comes back until a new session.
- They call each other by name, at the time of the call, so a user's redefinition of one
  is what the others call.
- A library function is a function name only directly before `(` (and, in Word mode,
  before spaces and `(`); in Character mode only when the whole run of letters is its
  name. So `star + 1` is still a product of letters and `star(3)` is a call. The two
  library variables are variable names, as if declared.
- The definitions are never entries of the worksheet, so a save never writes them, and
  a load report never mentions them.

This package is plain data, imported by the parse state. Each definition is parsed where
it is first called, once per process (`engine.context`), so starting a session costs
nothing and the engine worker never receives the library with a request.

`PENDING` holds the functions whose definitions need what Rederive does not compute
yet. They are not pre-defined until it does: a call of one stays a call.
"""

from __future__ import annotations

import re

from rederive.library.texts import TEXTS

#: The name a definition text defines, and what follows it: `(` for a function, `:=` for
#: a variable, and a parameter name for a variadic head such as `F v :=`.
_HEAD = re.compile(r"([^\s(:]+)\s*(\(|:=|\w)")

#: Library functions whose definitions need what Rederive does not compute yet, with the
#: reason. They are left out of `FUNCTIONS` until it does.
PENDING: dict[str, str] = {}


def _defined() -> tuple[dict[str, str], dict[str, str], frozenset[str]]:
    """Each definition text by the name it defines, the group of each name, and the
    names that are variables."""
    texts: dict[str, str] = {}
    groups: dict[str, str] = {}
    variables: set[str] = set()
    for group, definitions in TEXTS.items():
        for text in definitions:
            found = _HEAD.match(text)
            assert found is not None, text
            name = found.group(1)
            assert name not in texts, name
            texts[name] = text
            groups[name] = group
            if found.group(2) == ":=":
                variables.add(name)
    return texts, groups, frozenset(variables)


_TEXTS, GROUPS, VARIABLES = _defined()

#: Every definition the library holds, pending or not.
DEFINED: frozenset[str] = frozenset(_TEXTS)

#: The library functions in force when a session starts.
FUNCTIONS: frozenset[str] = DEFINED - VARIABLES - frozenset(PENDING)

#: Every library name in force when a session starts.
NAMES: frozenset[str] = FUNCTIONS | VARIABLES


def text(name: str) -> str:
    """The definition text of the library name `name`."""
    return _TEXTS[name]

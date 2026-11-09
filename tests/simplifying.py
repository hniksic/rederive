"""Shared helpers for tests that check what Simplify makes of a line.

`test_simplify.py` and its siblings, `test_integrate.py`, `test_calculus.py`,
`test_matrices.py`, `test_logic.py`, `test_approximate.py` and `test_functions.py` all
state cases as input text and the expected worksheet text. `simp` does that round trip;
`declaring` and `declared` build the parse state and context; the contexts defined here
are the ones several of those files use.
"""

from __future__ import annotations

from rederive.engine.computing import (
    Context,
    Domain,
    DomainKind,
    domain_of_node,
    simplify,
)
from rederive.syntax import FunctionDeclaration, ParseState, parse_expression


def parse(text, state=None):
    return parse_expression(text, state or ParseState()).node


def declaring(*names):
    """Return a parse state in which each of `names` is declared a function.

    An undeclared name followed by a parenthesis parses as a product (of its letters, in
    Character mode), so cases about functions must declare them first.
    """
    state = ParseState()
    for name in names:
        state.declare(FunctionDeclaration(name))
    return state


def simp(text, context=None, state=None):
    """Simplify `text` and return it as the worksheet would show it.

    Unless `state` is given, the text is parsed with the context's functions declared,
    as in the session that defined them.
    """
    context = context or Context()
    state = state or declaring(*context.functions)
    return simplify(parse(text, state), context, state).text


def declared(*declarations, **settings):
    """Return a context with the domains given by these declarations."""
    domains = {}
    for declaration in declarations:
        name, domain = domain_of_node(parse(declaration))
        domains[name] = domain
    return Context(domains=domains, **settings)


COMPLEX_X = Context(domains={"x": Domain(DomainKind.COMPLEX)})
POSITIVE_X = declared("x :epsilon Real (0, inf)")

"""How the Simplify tests read and simplify a line.

The files that test what Simplify makes of a line - `test_simplify.py` and its
siblings, `test_integrate.py`, `test_calculus.py`, `test_matrices.py`, `test_logic.py`,
`test_approximate.py` and `test_functions.py` - all state their cases the same way: the
author's text in, the worksheet's text out. `simp` is that round trip, `declaring` and
`declared` are the parse state and the context a case is read in, and the contexts here
are the ones more than one of those files asks a case under.
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
    """A parse state in which each of `names` is a function, as defining it makes it.

    A name nothing defines is no function - before a parenthesis it is a product, of
    its letters in Character mode - so a case about a function, with a body or
    without one as `F(x) :=` leaves it, reads its text with the function declared.
    """
    state = ParseState()
    for name in names:
        state.declare(FunctionDeclaration(name))
    return state


def simp(text, context=None, state=None):
    """`text` simplified, as the worksheet would show it.

    Read, unless a state is given, with the functions the context defines declared,
    which is how the session that defined them reads a line.
    """
    context = context or Context()
    state = state or declaring(*context.functions)
    return simplify(parse(text, state), context, state).text


def declared(*declarations, **settings):
    """A context whose domains are what these declarations say."""
    domains = {}
    for declaration in declarations:
        name, domain = domain_of_node(parse(declaration))
        domains[name] = domain
    return Context(domains=domains, **settings)


COMPLEX_X = Context(domains={"x": Domain(DomainKind.COMPLEX)})
POSITIVE_X = declared("x :epsilon Real (0, inf)")

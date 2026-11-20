"""Plotting what needs the evaluator: the form a plot draws, and its value at a point.

A plot starts as Derive's does, from the expression simplified: programs run, the
globals they assign are assigned for the session, and the answer is drawn. Most
answers are closed forms, which the plot host converts once and evaluates with numpy
at every sample, as it always has (see `plot.evaluate`).

Some aren't. A function defined by a program whose flow depends on its argument has
no symbolic value: `stp(x) := PROG(IF(x < 0, RETURN 0), IF(x < 1, RETURN x), 1)`
simplified with `x` unknown would run past both tests and answer 1. A recursion stands
on its symbolic argument, `PRIMEPI(x)` sums a test that only a number decides, and
`x + RANDOM(1)` simplified draws one number for the whole curve. These are evaluated
anew at each point, programs running, by `Pointwise`.

`plot_form` makes the choice. Its simplification is strict (see `Context.strict`): a
program test that stays undecided, or an iteration that carries an undecided
conditional from pass to pass, ends it at once, as does running past `SETTLING`
seconds. What ends that way is drawn point by point, and so is an answer that is no
closed form: one that still calls a function or program, holds a call that stands,
decides on integers (`numeric_form`), or drew a random number.

Point evaluations are for the picture only: each starts from the session's state, and
whatever its programs assign, define or show is dropped with it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from time import monotonic
from typing import Any

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.core.relational import Relational

from rederive.engine import pipeline
from rederive.engine.boundary import SETTLING, PlotForm
from rederive.engine.context import (
    Context,
    FileNeeded,
    Precision,
    StepLimitExceeded,
    Store,
    Unsettled,
    reporting,
)
from rederive.engine.evaluate import (
    Environment,
    Suspended,
    evaluate,
    evaluated,
    released,
    value_of,
)
from rederive.engine.fresh import afresh
from rederive.engine.ordering import main_order
from rederive.engine.variables import evaluated_names, relation_in, variables_in
from rederive.engine.vectors import Vector
from rederive.model.expr import Kind, Node
from rederive.syntax import ParseState
from rederive.syntax.names import (
    BINDING_FUNCTIONS,
    CONSTANTS,
    EXIT,
    PROGRAM_FUNCTIONS,
    SETTINGS,
)
from rederive.syntax.writer import write_expression

__all__ = ["POINT_STEPS", "Pointwise", "number_of", "numeric_form", "plot_form"]

#: Program steps one point may take (see `Context.step_limit`). A program that ends
#: on numbers takes a few hundred; one past this at a point is given up there, and the
#: point is left out of the picture.
POINT_STEPS = 20_000

#: Modules whose functions decide on integers: the number theory of `FIBONACCI`,
#: `DIVISOR_SIGMA` and the like, whose value at a non-integer is no point of the graph.
_ON_INTEGERS = ("sympy.functions.combinatorial.numbers", "sympy.ntheory")


@afresh
def plot_form(
    node: Node,
    context: Context | None = None,
    state: ParseState | None = None,
    running: bool = True,
) -> PlotForm:
    """What the plot of `node` draws, and how (see `PlotForm`).

    `running` simplifies first, as the Plot command does. Without it, the expression is
    taken as evaluated where nothing runs, as a plot was before it simplified, which
    the command falls back to when the simplification took too long (see `SETTLING`):
    such an expression is drawn as it stands if that is a closed form.
    """
    context = context or Context()
    if not running:
        return _as_evaluated(node, context)
    strict = dataclasses.replace(
        context, strict=True, deadline=monotonic() + SETTLING
    )
    try:
        result = pipeline.simplify(node, strict, state)
    except (Unsettled, StepLimitExceeded):
        return _pointwise(node, context)
    if result.needs is not None:
        return PlotForm(result.node, result.text, needs=result.needs)
    taken = {
        field.name: getattr(result, field.name) for field in dataclasses.fields(result)
    }
    value = None if result.random is not None else _closed(result.exact, context)
    if value is None:
        return dataclasses.replace(_pointwise(node, context), **_effects(taken))
    return PlotForm(
        **taken,
        variables=main_order(variables_in(value), context.order),
        relation=relation_in(value),
    )


def _as_evaluated(node: Node, context: Context) -> PlotForm:
    """The form of `node` as evaluated where nothing runs: itself, if that is a
    closed form, or else to be evaluated at each point."""
    if _calls_programs(node, context):
        return _pointwise(node, context)
    value, inert = evaluated_names(node, context)
    try:
        closed = not _nested_conditionals(value) and numeric_form(released(value))
    except Exception:
        closed = False
    if not closed:
        return _pointwise(node, context)
    return PlotForm(
        node,
        write_expression(node, spaced=True),
        variables=main_order(variables_in(value, inert), context.order),
        relation=relation_in(value),
    )


def _nested_conditionals(value: sp.Basic) -> bool:
    """Whether `value` holds an undecided conditional that holds another, as an
    iteration over one builds: written out, their cases multiply."""
    return any(
        type(found).display.atoms(Suspended) for found in value.atoms(Suspended)
    )


def _effects(fields: dict[str, Any]) -> dict[str, Any]:
    """The fields of a simplification that are its effects on the session."""
    return {
        name: fields[name] for name in ("performed", "displayed", "written", "random")
    }


def _pointwise(node: Node, context: Context) -> PlotForm:
    """`node` to be evaluated at each point, with its variables: those of its value
    where nothing runs and those it is written in, but not the names its standing
    programs assign, which are theirs, not the plot's.

    The variables as written count too, since where nothing runs a program's
    assignments to its own locals are still made, which can decide its tests: `slow(x,
    k_) := PROG(k_ := 0, LOOP(IF(k_ > 9, RETURN x), k_ :+ 1))` comes to no value of `x`
    there.
    """
    value, inert = evaluated_names(node, context)
    names = set(variables_in(value, inert)) | (_written_names(node, context) - inert)
    return PlotForm(
        node,
        write_expression(node, spaced=True),
        variables=main_order(list(names), context.order),
        relation=relation_in(value),
        pointwise=True,
    )


def _written_names(node: Node, context: Context) -> set[str]:
    """The variables `node` is written in: its names that are no function, constant,
    setting or assigned variable, nor the variable of a binding call they are in."""
    if node.kind is Kind.NAME:
        name = str(node.value)
        if (
            name in CONSTANTS
            or name in SETTINGS
            or name == EXIT
            or context.assigned(name) is not None
            or context.function(name) is not None
        ):
            return set()
        return {name}
    if node.kind is not Kind.CALL:
        return set().union(*(_written_names(child, context) for child in node.children))
    head, *arguments = node.children
    found = set().union(*(_written_names(argument, context) for argument in arguments))
    if str(head.value) in BINDING_FUNCTIONS and len(arguments) >= 2:
        index = arguments[1]
        if index.kind is Kind.NAME:
            body = _written_names(arguments[0], context) - {str(index.value)}
            rest = arguments[2:]
            found = body.union(*(_written_names(part, context) for part in rest))
    return found


def _closed(node: Node, context: Context) -> sp.Basic | None:
    """The value of the simplified tree `node`, if it is a closed form; else None."""
    if _calls_programs(node, context):
        return None
    try:
        value = evaluated(node, context.fresh())
        return value if numeric_form(value) else None
    except Exception:
        return None


def _calls_programs(node: Node, context: Context) -> bool:
    """Whether `node` calls a defined function (the session's or the library's) or a
    program construct, which a closed form doesn't."""
    if node.kind is Kind.CALL:
        name = str(node.children[0].value)
        if name in PROGRAM_FUNCTIONS or context.function(name) is not None:
            return True
    return any(_calls_programs(child, context) for child in node.children)


def numeric_form(value: Any) -> bool:
    """Whether `value` is a closed form the plot host evaluates as numbers: a number,
    variables and functions with numeric code, as is or in a vector.

    Not one is a call that stands (a built-in on symbolic arguments, an undecided test
    shown as a call), a function deciding on integers (see `_ON_INTEGERS`), or anything
    neither numpy nor mpmath can evaluate.
    """
    if isinstance(value, Vector):
        return bool(value.args) and all(numeric_form(element) for element in value.args)
    if not isinstance(value, sp.Basic):
        return False
    for part in sp.preorder_traversal(value):
        if isinstance(part, AppliedUndef):
            return False
        if isinstance(part, sp.Function) and type(part).__module__.startswith(
            _ON_INTEGERS
        ):
            return False
    symbols = sorted(value.free_symbols, key=sp.default_sort_key)
    for modules in ("numpy", "mpmath"):
        try:
            sp.lambdify(symbols, value, modules=modules)
            return True
        except Exception:
            continue
    return False


class Pointwise:
    """An expression evaluated anew at each point, programs running.

    `at` gives the values of `parts` (expressions in the same variables, evaluated in
    order, so `u` and `v` of an equation `u = v`) with `variables` given numbers, as
    Simplify gives them under Approximate precision at the session's digits; a part that
    doesn't come to a number is None, and so is every part where evaluating fails, runs
    past `steps` steps (`POINT_STEPS` by default), or asks for a file.

    Each point is evaluated as a command of its own whose effects are dropped, except
    that `RANDOM`'s generator moves on from point to point, so that each point draws.
    """

    def __init__(
        self,
        parts: Sequence[Node],
        context: Context,
        variables: Sequence[str],
        steps: int = POINT_STEPS,
    ) -> None:
        self.parts = tuple(parts)
        self.variables = tuple(variables)
        approximate = context.with_precision(Precision.APPROXIMATE)
        self._context = dataclasses.replace(
            approximate, step_limit=steps, strict=False, deadline=None
        )
        #: The state of `RANDOM`'s generator for the next point.
        self.random_state = context.random_state
        #: How many points were evaluated, and how many of them ran past the step
        #: limit, so that a program that never ends anywhere can be told from one that
        #: ends at some points.
        self.points = 0
        self.overrun = 0

    def at(self, *values: float) -> tuple[complex | None, ...]:
        """The parts' values with the variables at `values` (see the class)."""
        given = {
            name: sp.Rational(value)
            for name, value in zip(self.variables, values, strict=True)
        }
        store = Store(values=given)
        context = dataclasses.replace(
            self._context, store=store, random_state=self.random_state
        )
        self.points += 1
        try:
            with reporting(None):
                found = tuple(number_of(_evaluated(part, context)) for part in self.parts)
        except StepLimitExceeded:
            self.overrun += 1
            return (None,) * len(self.parts)
        except (Exception, FileNeeded, Unsettled):
            return (None,) * len(self.parts)
        if store.random is not None:
            self.random_state = store.random
        return found

    def value(self) -> sp.Basic:
        """The first part evaluated once, its variables standing, for a plot whose value
        is a whole (the matrix of a data plot)."""
        context = dataclasses.replace(
            self._context, store=Store(), random_state=self.random_state
        )
        with reporting(None):
            return _evaluated(self.parts[0], context)


def _evaluated(node: Node, context: Context) -> sp.Basic:
    """`node` evaluated as Simplify evaluates it, programs running, before rewriting:
    at a point everything comes to a number, which needs no rewriting."""
    environment = Environment(
        context,
        decide=lambda test: pipeline._decide_simplified(test, context),
        work_out=lambda head: pipeline._evaluate(head, context),
        command=pipeline._commanded,
        simplify=pipeline._argument,
        running=True,
    )
    return value_of(released(evaluate(node, environment)))


def number_of(value: sp.Basic) -> complex | None:
    """`value` as a number, a truth value as 1 or 0, or None if it is neither.

    A relation between numbers, as a region's inequality comes to at a point, is
    decided first.
    """
    if isinstance(value, sp.logic.boolalg.Boolean) and not value.free_symbols:
        value = _truth(value)
    if value is sp.true:
        return 1.0
    if value is sp.false:
        return 0.0
    if not isinstance(value, sp.Expr) or value.free_symbols:
        return None
    try:
        return complex(value if isinstance(value, sp.Rational) else value.evalf())
    except (TypeError, ValueError, OverflowError):
        return None


def _truth(value: sp.Basic) -> sp.Basic:
    """A relation between numbers, or a logical combination of them, decided: exactly
    where sympy can, else by the numbers' values, as a picture needs."""
    if isinstance(value, Relational):
        left, right = value.args
        decided = type(value)(left, right)
        if decided in (sp.true, sp.false):
            return decided
        try:
            difference = complex((left - right).evalf())
        except (TypeError, ValueError):
            return value
        if difference.imag:
            return sp.false if isinstance(value, sp.Eq) else value
        holds = {
            sp.Eq: difference.real == 0,
            sp.Ne: difference.real != 0,
            sp.Lt: difference.real < 0,
            sp.Le: difference.real <= 0,
            sp.Gt: difference.real > 0,
            sp.Ge: difference.real >= 0,
        }.get(type(value))
        return value if holds is None else sp.true if holds else sp.false
    if isinstance(value, (sp.And, sp.Or, sp.Not, sp.Xor)):
        return value.func(*(_truth(argument) for argument in value.args))
    return value

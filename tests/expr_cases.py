"""Reading `tests/corpus/parser-cases.txt`.

One case per line, directive lines that set the state for the cases that
follow, `##` group headers that reset it, and ` # ` comments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from rederive.syntax import (
    CaseMode,
    FunctionDeclaration,
    InputMode,
    ParseState,
    VariableDeclaration,
)

CORPUS = Path(__file__).parent / "corpus" / "parser-cases.txt"

_SEPARATOR = re.compile(r"\s+=>\s+")
_DIRECTIVE = re.compile(r"^\[(state|declared):(.*)\]$")
_COMMENT = re.compile(r"\s+#\s")
_FUNCTION = re.compile(r"^fun\s+(\S+)/(\d+)$")
_VARIABLE = re.compile(r"^var\s+(\S+)$")


@dataclass(frozen=True)
class Case:
    line: int
    group: str
    text: str
    expected: str
    state_spec: str = ""
    declared_spec: str = ""

    @property
    def id(self) -> str:
        return f"L{self.line}:{self.text}"

    def state(self) -> ParseState:
        """A fresh state, as the directives in force describe it."""
        state = ParseState()
        for setting in _items(self.state_spec):
            name, _, value = setting.partition("=")
            match name.strip():
                case "InputMode":
                    state.input_mode = InputMode(value.strip())
                case "CaseMode":
                    state.case_mode = CaseMode(value.strip())
                case "InputBase":
                    state.input_base = int(value.strip())
                case other:
                    raise AssertionError(f"unknown setting in corpus: {other}")
        for declaration in _items(self.declared_spec):
            function = _FUNCTION.match(declaration)
            variable = _VARIABLE.match(declaration)
            if function is not None:
                arity = int(function.group(2))
                params = tuple(f"p{index}" for index in range(arity))
                state.declare(
                    FunctionDeclaration(function.group(1), params, has_body=True)
                )
            elif variable is not None:
                state.declare(VariableDeclaration(variable.group(1), has_value=True))
            else:
                raise AssertionError(f"unreadable declaration in corpus: {declaration}")
        return state


@dataclass
class _Directives:
    group: str = ""
    state_spec: str = ""
    declared_spec: str = ""
    cases: list[Case] = field(default_factory=list)


def load(path: Path = CORPUS) -> list[Case]:
    """Every case in the corpus, in file order."""
    context = _Directives()
    preamble = True
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("##"):
            preamble = False
            context = _Directives(group=line.lstrip("# ").strip(), cases=context.cases)
            continue
        if preamble:
            # The file header. Past it, a leading `#` is a label or a constant.
            continue
        directive = _DIRECTIVE.match(line.strip())
        if directive is not None:
            kind, body = directive.group(1), directive.group(2)
            if kind == "state":
                context.state_spec = body
                context.declared_spec = ""
            else:
                context.declared_spec = body
            continue
        split = _SEPARATOR.search(line)
        if split is None:
            raise AssertionError(f"unreadable corpus line {number}: {line}")
        text = line[: split.start()]
        expected = _COMMENT.split(line[split.end() :], maxsplit=1)[0].strip()
        context.cases.append(
            Case(
                number,
                context.group,
                text,
                expected,
                context.state_spec,
                context.declared_spec,
            )
        )
    return context.cases


def _items(spec: str) -> list[str]:
    return [item.strip() for item in spec.split(";") if item.strip()]

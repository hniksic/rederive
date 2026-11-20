"""Factoring a matrix into a product of matrices: Factor's tUring and scHmidt amounts.

`FACTOR(A, Turing)` writes an m-by-n matrix as `P . L . D . U . R`: a permutation, a
unit lower triangular matrix, the diagonal of pivots, a unit upper triangular matrix,
and the reduced row echelon form of `A` (the identity when `A` is square and of full
rank). `FACTOR(A, Schmidt)` is the Gram-Schmidt factorization `Q . R`, Q with
orthonormal columns and R upper triangular with a positive diagonal, or `Q . R . P` with
a column permutation P when the columns of `A` are dependent.

Both are exact over the entries given. Elimination treats a symbolic pivot as nonzero,
so the factorization is the generic one: it holds wherever its entries are defined.
"""

from __future__ import annotations

from collections.abc import Callable

import sympy as sp

from rederive.engine.boundary import Amount
from rederive.engine.vectors import from_matrix, matrix_of

__all__ = ["MATRIX_AMOUNTS", "matrix_factors", "matrix_product"]

#: The amounts that factor a matrix rather than its elements.
MATRIX_AMOUNTS = (Amount.TURING, Amount.SCHMIDT)


def matrix_factors(value: sp.Basic, amount: Amount) -> list[sp.Basic] | None:
    """The factors of the matrix `value` by `amount`, as vectors of rows, or None when
    `value` is not a matrix or has no such factorization (the zero matrix has no
    Gram-Schmidt factors)."""
    matrix = matrix_of(value)
    if matrix is None:
        return None
    factors = _turing(matrix) if amount is Amount.TURING else _schmidt(matrix)
    if factors is None:
        return None
    return [from_matrix(factor) for factor in factors]


def matrix_product(
    value: sp.Basic,
    amount: Amount,
    finish: Callable[[sp.Expr], sp.Expr] | None = None,
) -> sp.Basic | None:
    """The factors of the matrix `value` by `amount` as their product `P . L . D . U .
    R`, held unevaluated, or None (see `matrix_factors`). `finish` is applied to each
    entry, as Factor rounds its factors."""
    from rederive.engine.to_sympy import Dot

    factors = matrix_factors(value, amount)
    if factors is None:
        return None
    if finish is not None:
        factors = [
            from_matrix(matrix_of(factor).applyfunc(finish)) for factor in factors
        ]
    product = factors[0]
    for factor in factors[1:]:
        product = Dot(product, factor)
    return product


def _turing(matrix: sp.ImmutableMatrix) -> list[sp.Matrix]:
    """`[P, L, D, U, R]` with `matrix = P . L . D . U . R`.

    Elimination goes column by column. The pivot of a column is chosen among the entries
    in and below the current row: a number before a symbolic entry, and among numbers
    the one of largest magnitude (the magnitude only chooses; any nonzero pivot gives
    an exact factorization). A row left without a pivot gets 1 in `D`.
    """
    rows, columns = matrix.shape
    work = [list(matrix.row(row)) for row in range(rows)]
    origin = list(range(rows))
    lower = sp.zeros(rows, rows)
    pivots: list[tuple[int, sp.Expr]] = []
    for col in range(columns):
        done = len(pivots)
        if done == rows:
            break
        candidates = [row for row in range(done, rows) if not _vanishes(work[row][col])]
        if not candidates:
            continue
        chosen = _pivot_row(candidates, [work[row][col] for row in candidates])
        work[done], work[chosen] = work[chosen], work[done]
        origin[done], origin[chosen] = origin[chosen], origin[done]
        for col_before in range(done):
            lower[done, col_before], lower[chosen, col_before] = (
                lower[chosen, col_before],
                lower[done, col_before],
            )
        pivot = work[done][col]
        for row in range(done + 1, rows):
            multiplier = _tidy(work[row][col] / pivot)
            lower[row, done] = multiplier
            work[row] = [
                _tidy(entry - multiplier * above)
                for entry, above in zip(work[row], work[done], strict=True)
            ]
            work[row][col] = sp.S.Zero
        pivots.append((col, pivot))
    permutation = sp.zeros(rows, rows)
    for position, row in enumerate(origin):
        permutation[row, position] = 1
    unit_lower = lower + sp.eye(rows)
    diagonal = sp.diag(
        *(pivot for _, pivot in pivots), *([sp.S.One] * (rows - len(pivots)))
    )
    scaled = [
        [_tidy(entry / pivots[row][1]) for entry in work[row]]
        if row < len(pivots)
        else work[row]
        for row in range(rows)
    ]
    unit_upper = sp.eye(rows)
    for row in range(len(pivots)):
        for later in range(row + 1, len(pivots)):
            unit_upper[row, later] = scaled[row][pivots[later][0]]
    echelon = (unit_upper.inv() * sp.Matrix(scaled)).applyfunc(_tidy)
    return [permutation, unit_lower, diagonal, unit_upper, echelon]


def _pivot_row(rows: list[int], entries: list[sp.Expr]) -> int:
    """Which of `rows` holds the pivot: the first number of largest magnitude, or the
    first entry if none is a number."""
    numbers = [
        (row, entry) for row, entry in zip(rows, entries, strict=True) if entry.is_number
    ]
    if not numbers:
        return rows[0]
    best, _ = max(numbers, key=lambda pair: abs(complex(sp.N(pair[1], 30))))
    return best


def _schmidt(matrix: sp.ImmutableMatrix) -> list[sp.Matrix] | None:
    """`[Q, R]` with `matrix = Q . R`, or `[Q, R, P]` with `matrix = Q . R . P` when the
    columns are dependent; None for the zero matrix.

    Each column in turn is made orthogonal to the unit vectors found so far; one left
    with nothing is dependent and goes after the independent ones, which `P` records.
    """
    rows, columns = matrix.shape
    # Over real entries nothing is conjugated: a norm of symbolic entries would be, and
    # the conjugate of its root does not simplify.
    real = all(entry.is_real for entry in matrix)
    units: list[sp.Matrix] = []
    independent: list[int] = []
    dependent: list[int] = []
    norms: list[sp.Expr] = []
    for col in range(columns):
        column = sp.Matrix(matrix.col(col))
        residual = column
        for unit in units:
            residual = residual - _inner(unit, column, real) * unit
        residual = residual.applyfunc(_tidy)
        if all(_vanishes(entry) for entry in residual):
            dependent.append(col)
            continue
        norm = _tidy(sp.sqrt(sp.factor(_inner(residual, residual, real))))
        units.append(residual.applyfunc(lambda entry, norm=norm: _tidy(entry / norm)))
        norms.append(norm)
        independent.append(col)
    if not units:
        return None
    order = independent + dependent
    orthonormal = sp.Matrix.hstack(*units)
    triangular = sp.zeros(len(units), columns)
    for row, unit in enumerate(units):
        triangular[row, row] = norms[row]
        for position in range(row + 1, columns):
            triangular[row, position] = _tidy(
                _inner(unit, sp.Matrix(matrix.col(order[position])), real)
            )
    if not dependent:
        return [orthonormal, triangular]
    permutation = sp.zeros(columns, columns)
    for position, col in enumerate(order):
        permutation[position, col] = 1
    return [orthonormal, triangular, permutation]


def _inner(left: sp.Matrix, right: sp.Matrix, real: bool) -> sp.Expr:
    """The inner product of two columns, conjugating the left one unless they are
    `real`."""
    pairs = zip(left, right, strict=True)
    return _tidy(sum(a * b if real else sp.conjugate(a) * b for a, b in pairs))


def _vanishes(entry: sp.Expr) -> bool:
    """Whether `entry` is zero for every value of its variables. One that is zero only
    for some, like `x - 2`, is not: a symbolic pivot is taken to be nonzero."""
    if entry == 0:
        return True
    known = entry.is_zero
    if known is None:
        known = sp.simplify(entry).is_zero
    return bool(known)


def _tidy(entry: sp.Expr) -> sp.Expr:
    """An entry in lowest terms, a number with the radicals of its denominator
    rationalized and in rectangular form.

    Only a number is rationalized: `radsimp` takes `ABS(x + 6)` for `x + 6`.
    """
    entry = sp.cancel(sp.together(entry))
    if not entry.is_number:
        return entry
    if entry.has(sp.I):
        entry = sp.expand(entry)
    if any(not power.exp.is_Integer for power in entry.atoms(sp.Pow)):
        entry = sp.radsimp(entry)
    return entry

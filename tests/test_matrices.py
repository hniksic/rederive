"""What Simplify makes of vectors and matrices.

Covers vector and matrix arithmetic, variables declared nonscalar, linear algebra,
vector calculus, `VECTOR` and subscripts, iteration, sample statistics, functions that
split an expression into a vector of parts, and number theory. Cases are inline, as in
`test_simplify.py`, so that a failure shows the mathematics. The matrix helpers from
the utility files and demos are tested here too, since they rely on this arithmetic.
"""

from __future__ import annotations

import time

import pytest
from simplifying import declared, declaring, parse, simp

from rederive.engine.computing import Context, Definition, StepLimitExceeded
from rederive.model.session import Session


def test_a_conditional_whose_branches_are_vectors_simplifies_elementwise():
    # The chosen branch is a vector, and is simplified elementwise like a written one.
    taken = "IF(a > 0, [1, 2], [3, 4], [SQRT(b)*SQRT(b), 2*b/b])"
    assert simp(taken) == simp("[SQRT(b)*SQRT(b), 2*b/b]") == "[b, 2]"


def test_a_symbolic_power_of_a_matrix_is_its_closed_form():
    # A unipotent matrix to a symbolic power is a polynomial in `n`.
    assert simp("[[1, 1], [0, 1]]^n") == "[[1, n], [0, 1]]"
    # A diagonalizable one is a sum of powers of its eigenvalues, each coefficient
    # written over a rational denominator.
    assert simp("[[1, 1], [1, 0]]^n") == (
        "[[(5 - SQRT(5))*(1/2 - SQRT(5)/2)^n/10 + (SQRT(5) + 5)*(SQRT(5)/2 + 1/2)^n/10,"
        " SQRT(5)*(SQRT(5)/2 + 1/2)^n/5 - SQRT(5)*(1/2 - SQRT(5)/2)^n/5],"
        " [SQRT(5)*(SQRT(5)/2 + 1/2)^n/5 - SQRT(5)*(1/2 - SQRT(5)/2)^n/5,"
        " (5 - SQRT(5))*(SQRT(5)/2 + 1/2)^n/10 + (SQRT(5) + 5)*(1/2 - SQRT(5)/2)^n/10]]"
    )


def test_an_integer_power_of_an_integer_matrix_is_exact():
    assert simp("[[0, 1], [1, 1]]^10") == "[[34, 55], [55, 89]]"
    assert simp("[[0, 1], [1, 1]]^200") == simp(
        "[[FIBONACCI(199), FIBONACCI(200)], [FIBONACCI(200), FIBONACCI(201)]]"
    )
    assert simp("[[2, -1, 0], [0, 3, 1], [1, 0, -1]]^0") == (
        "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"
    )
    assert simp("[[2, -1, 0], [0, 3, 1], [1, 0, -1]]^3") == simp(
        "[[2, -1, 0], [0, 3, 1], [1, 0, -1]] . [[2, -1, 0], [0, 3, 1], [1, 0, -1]]"
        " . [[2, -1, 0], [0, 3, 1], [1, 0, -1]]"
    )
    assert simp("[[1, 2], [3, 4]]^-1") == "[[-2, 1], [3/2, -1/2]]"


def test_a_matrix_root_is_simplified_like_a_matrix_power():
    # SQRT of a square matrix is its power to 1/2, whose entries are simplified.
    assert simp("SQRT([[10, 7], [7, 17]])") == "[[3, 1], [1, 4]]"
    assert simp("SQRT([[5, 4], [4, 5]])") == simp("[[5, 4], [4, 5]]^(1/2)")
    assert simp("SQRT([[5, 4], [4, 5]])") == "[[2, 1], [1, 2]]"
    # Of any other matrix, it is the root of each element.
    assert simp("SQRT([[1, 4, 9], [4, 5, 6]])") == "[[1, 2, 3], [2, SQRT(5), SQRT(6)]]"


def test_an_integer_power_of_a_vector_is_a_power_of_its_dot_square():
    # An odd power keeps one factor of the vector, so `^-1` divides it by its square.
    assert simp("[x, y + 2]^2 - 9") == "x^2 + y^2 + 4*y - 5"
    assert simp("[1, 2]^3") == "[5, 10]"
    assert simp("[1, 2]^-1") == "[1/5, 2/5]"
    assert simp("[[1, 2]]^2") == "[[1, 2]]^2"


def test_a_symbolic_power_of_a_vector_stands():
    # An integer power of a vector is a power of its dot square, so a symbolic power
    # cannot be taken element by element: `[a, b]^2` would then disagree with `[a, b]^k`
    # at k = 2.
    assert simp("[a, b]^k") == "[a, b]^k"
    assert simp("SUBST([a, b]^k, k, 2)") == "a^2 + b^2"


def test_a_singular_matrix_inverse_comes_back_unsimplified():
    assert simp("[[1, 2], [2, 4]]^-1") == "[[1, 2], [2, 4]]^-1"
    assert simp("[[1, 2], [3, 4]]^-1") == "[[-2, 1], [3/2, -1/2]]"


# -- vectors and matrices -----------------------------------------------------

VECTORS = [
    ("[2*x, -5, x^2] + [-x, 8, 2*x]", "[x, 3, x^2 + 2*x]"),
    ("[2, a, 5] . [2*a, 3, -1]", "7*a - 5"),
    ("[2, 3]*[4, 5]", "23"),
    ("DET([[1, 2], [3, 4]])", "-2"),
    ("TRACE([[1, 2], [3, 4]])", "5"),
    ("DIMENSION([1, 2, 3])", "3"),
    ("[[1, 2], [3, 4]]`", "[[1, 3], [2, 4]]"),
    ("SUM([1, 2, 3])", "6"),
    # Like `SUM`, `MAX` and `MIN` of one vector take its elements; of a matrix, each
    # row's.
    ("MAX([3, 1, 2])", "3"),
    ("MIN([[1, 2], [3, 0]])", "[1, 0]"),
    # A set's elements, as a vector's; there is no extremum of nothing.
    ("[MAX({3, 1, 2}), MIN({3, 1, 2})]", "[3, 1]"),
    ("[MAX([]), MIN({})]", "[MAX([]), MIN({})]"),
    ("[MIN([0.1, 21.3, 5]), MAX([0.1, 21.3, 5])]", "[1/10, 213/10]"),
    ("MAX(#i, 1)", "MAX(#i, 1)"),
    ("[[1, 2], [3, 4]] . [[a], [b]]", "[[a + 2*b], [3*a + 4*b]]"),
    # A flat vector right of a matrix acts as a column and the answer is flat again; a
    # written column stays a column.
    ("[[a, b], [c, d]] . [2, 3]", "[2*a + 3*b, 2*c + 3*d]"),
    ("[2, 3] . [[a, b], [c, d]]", "[2*a + 3*c, 2*b + 3*d]"),
    # The cross product of 3-vectors, and of 2-vectors: the latter is the third component
    # of the 3D product (the other two are zero), returned as a number.
    ("CROSS([1, 2, 3], [a, b, c])", "[2*c - 3*b, 3*a - c, b - 2*a]"),
    ("CROSS([1, 2], [a, b])", "b - 2*a"),
    # A vector index selects one element for each index, so selecting from a matrix
    # gives a matrix; one index out of range keeps the access unevaluated.
    ("[[2, 3, 5], [7, 1, 4]] SUB [2, 1]", "[[7, 1, 4], [2, 3, 5]]"),
    ("[[2, 3, 5], [7, 1, 4]] SUB [2, 3]", "ELEMENT([[2, 3, 5], [7, 1, 4]], [2, 3])"),
    (
        "ELEMENT([[2, 3, 5], [7, 1, 4]], [2, 3])",
        "ELEMENT([[2, 3, 5], [7, 1, 4]], [2, 3])",
    ),
    ("[[2, 3, 5], [7, 1, 4]] SUB [2]", "[[7, 1, 4]]"),
    # A vector of matrices stays one rather than being flattened into a 2x4. It has two
    # elements, and `ELEMENT` takes one index per dimension.
    ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", "[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]"),
    ("DIMENSION([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])", "2"),
    ("ELEMENT([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 2)", "[[5, 6], [7, 8]]"),
    ("ELEMENT([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 2, 1, 2)", "6"),
    ("VECTOR([[k, k], [k, k]], k, 2)", "[[[1, 1], [1, 1]], [[2, 2], [2, 2]]]"),
    # A chain of `SUB`s is `ELEMENT` with one index per dimension. The shipped utility
    # files use it to index Christoffel symbols.
    ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]] SUB 2 SUB 1 SUB 2", "6"),
    ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]] SUB 2", "[[5, 6], [7, 8]]"),
    # An out-of-range index keeps the access unevaluated, so a generator that substitutes
    # a valid index later still has it.
    (
        "[[[1, 2], [3, 4]], [[5, 6], [7, 8]]] SUB 4",
        "ELEMENT([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 4)",
    ),
    ("VECTOR([[[1, 2], [3, 4]], [[5, 6], [7, 8]]] SUB i SUB 1 SUB 2, i, 2)", "[2, 6]"),
    # More indices than dimensions keep the access unevaluated.
    ("ELEMENT([a, b, c], 3, 1)", "ELEMENT([a, b, c], 3, 1)"),
    # Ordinary matrix indexing; an index past the end keeps the head.
    ("ELEMENT([[2, 3, 5], [7, 1, 4]], 2, 3)", "4"),
    ("ELEMENT([[2, 3, 5], [7, 1, 4]], 2)", "[7, 1, 4]"),
    (
        "ELEMENT([[2, 3, 5], [7, 1, 4]], 2, 4)",
        "ELEMENT([[2, 3, 5], [7, 1, 4]], 2, 4)",
    ),
    # Incompatible shapes keep the operator unevaluated rather than guessing a product.
    ("[[1, 2], [3, 4]] . [1, 2, 3]", "[[1, 2], [3, 4]] . [1, 2, 3]"),
    # Nor do they add. Terms of each length are summed, and the sum of different
    # lengths stays.
    ("[1, 2] + [1, 2, 3]", "[1, 2] + [1, 2, 3]"),
    ("[1, 2] - [1, 2, 3]", "[1, 2] + [-1, -2, -3]"),
    ("[1, 2] + [3, 4] + [1, 2, 3]", "[4, 6] + [1, 2, 3]"),
    ("x + 1 + [1, 2] + [1, 2, 3]", "x + 1 + [1, 2] + [1, 2, 3]"),
    # Only the outer lengths matter, as in Derive (D): the elements are added by the
    # same rule, so a row plus a scalar is added element by element.
    ("[[1, 2], [3, 4]] + [1, 2]", "[[2, 3], [5, 6]]"),
    (
        "[[1, 2], [3, 4]] + [[1, 2, 3], [4, 5, 6]]",
        "[[1, 2] + [1, 2, 3], [3, 4] + [4, 5, 6]]",
    ),
    ("x*([1, 2] + [1, 2, 3])", "([1, 2] + [1, 2, 3])*x"),
    ("([1, 2] + [1, 2, 3]) + [1, 1]", "[2, 3] + [1, 2, 3]"),
    ("[1, 2] + [1, 2, 3] - ([1, 2] + [1, 2, 3])", "[0, 0] + [0, 0, 0]"),
    ("DET([1, 2] + [1, 2, 3])", "DET([1, 2] + [1, 2, 3])"),
    ("[1, 2] + 5", "[6, 7]"),
    # A scalar added to a square matrix is a multiple of the identity, so a polynomial
    # in a matrix has its constant term: Cayley-Hamilton gives the zero matrix, and
    # `A - lambda` is the matrix whose determinant is the characteristic polynomial.
    ("[[1, 2], [3, 4]] + 7", "[[8, 2], [3, 11]]"),
    ("[[1, 2], [3, 4]] - 7", "[[-6, 2], [3, -3]]"),
    ("VECTOR([k, k^2], k, 1, 2) - 1", "[[0, 1], [2, 3]]"),
    ("[[1, -2], [2, 3]]^2 - 4*[[1, -2], [2, 3]] + 7", "[[0, 0], [0, 0]]"),
    ("DET([[1, 2], [3, 4]] - x)", "x^2 - 5*x - 2"),
    # Any other shape has the scalar added to each element.
    ("[[1, 2], [3, 4], [5, 6]] + 7", "[[8, 9], [10, 11], [12, 13]]"),
    ("[[1, [2, 3]], 4] + 1", "[[2, [3, 4]], 5]"),
    # Like `FLOOR`, the remainder of a vector is taken element by element.
    ("MOD([[7, 11], [3, 8]], 2)", "[[1, 1], [1, 0]]"),
    # Only square matrices have powers and inverses to divide by.
    ("[[1, 2, 3], [4, 5, 6]]^(-1)", "[[1, 2, 3], [4, 5, 6]]^-1"),
    ("[[1, 2, 3], [4, 5, 6]]^2", "[[1, 2, 3], [4, 5, 6]]^2"),
    ("[[1, 2, 3], [4, 5, 6]]^n", "[[1, 2, 3], [4, 5, 6]]^n"),
    # A vector's power is one of its dot square (see below).
    ("[1, 2]^2", "5"),
    # Dividing by a flat vector multiplies by its inverse `v/(v . v)`.
    ("[1, 2]/[3, 4]", "11/25"),
    ("2/[1, 2]", "[2/5, 4/5]"),
    ("[[1, 2], [3, 4]]/[1, 2]", "[[1, 2], [3, 4]]/[1, 2]"),
    ("[[1, 2], [3, 4]]/[[1, 2], [3, 5]]", "[[1, 0], [-3, 2]]"),
    ("[1, 2]/[[1, 2], [3, 4]]", "[1, 0]"),
    ("[[1, 2], [3, 4]]/[[1, 2], [2, 4]]", "[[1, 2], [3, 4]] . [[1, 2], [2, 4]]^-1"),
    ("CROSS([1, 2], [1, 2, 3])", "CROSS([1, 2], [1, 2, 3])"),
    # An element of the empty vector is undefined, as Derive gives `[] SUB 1`: loops
    # that run past the end of a vector test that value and go on. Past the end of
    # another vector the access stands.
    ("ELEMENT([1, 2], 3)", "ELEMENT([1, 2], 3)"),
    ("[] SUB 1", "?"),
    ("[] SUB -1", "?"),
    ("11 /= [] SUB 1", "true"),
    ("[] SUB 0", "ELEMENT([], 0)"),
    # Anything not declared nonscalar is a scalar (8.5), so `.` between scalars is a
    # product. MATRIX.DMO's vector algebra step.
    ("a . b", "a*b"),
    ("a . b - b . a", "0"),
    ("a . (b + c) - (b` . a`)`", "a*c"),
    ("a . [1, 2]", "a . [1, 2]"),
    # Ragged: not a matrix, and left alone.
    ("[[1, 2], [3]]", "[[1, 2], [3]]"),
]


#: Rows, columns and deeper levels: `A ROW i` and `A COL j` of a matrix, `A SUB SUB j`
#: one level down, a further `SUB` per level deeper. An index counts from the right if
#: negative, and a vector of indices selects one of each.
PARTS = [
    ("[a, b, c; d, e, f; g, h, i] ROW 2", "[d, e, f]"),
    ("[a, b, c; d, e, f; g, h, i] COL 2", "[b, e, h]"),
    ("[[a, b, c], [d, e, f], [g, h, i]] SUB SUB 2", "[b, e, h]"),
    ("[[a, b, c], [d, e, f], [g, h, i]] ROW [3, 1]", "[[g, h, i], [a, b, c]]"),
    ("[[a, b, c], [d, e, f], [g, h, i]] COL [3, 1]", "[[c, a], [f, d], [i, g]]"),
    ("[[a, b, c], [d, e, f], [g, h, i]] COL [2]", "[[b], [e], [h]]"),
    ("[[a, b, c], [d, e, f], [g, h, i]] SUB SUB (-1)", "[c, f, i]"),
    ("[[a, b, c], [d, e, f], [g, h, i]] SUB SUB 2 SUB 3", "h"),
    ("[[a, b, c], [d, e, f], [g, h, i]] ROW [2, ..., 3]", "[[d, e, f], [g, h, i]]"),
    ("[[a, b, c], [d, e, f], [g, h, i]] COL [1, ..., 2]", "[[a, b], [d, e], [g, h]]"),
    ("DIM([[a, b, c], [d, e, f], [g, h, i]] SUB SUB 1)", "3"),
    ("([[a, b, c], [d, e, f], [g, h, i]] ROW [1, 2])`", "[[a, d], [b, e], [c, f]]"),
    ("([[1, 5, 5, 1], [0, 0, 3, 0], [1, 1, 1, 1]] ROW [1, 2])`", (
        "[[1, 0], [5, 0], [5, 3], [1, 0]]"
    )),
    ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]] SUB SUB SUB 1", "[[1, 3], [5, 7]]"),
    ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]] SUB SUB 1", "[[1, 2], [5, 6]]"),
    # Where an element on the way has no such index, or `ROW` and `COL` have no
    # matrix, the operation stands.
    ("[1, 2, 3] SUB SUB 1", "[1, 2, 3] SUB SUB 1"),
    ("[1, 2, 3] ROW 1", "[1, 2, 3] ROW 1"),
    ("[[1, 2], [3]] SUB SUB 2", "[[1, 2], [3]] SUB SUB 2"),
    ("[[1, 2], [3]] ROW 1", "[[1, 2], [3]] ROW 1"),
    ("[[a, b, c], [d, e, f], [g, h, i]] SUB [2, 3] SUB SUB [2, 3]", "[[e, f], [h, i]]"),
    ("m ROW k", "m ROW k"),
    ("m SUB SUB (-1)", "m SUB SUB (-1)"),
    # Rows written with `;`.
    ("DET([1, 2; 3, 4])", "-2"),
    ("[2, 3, 5; 7, 1, 4; 4, 6, 8] SUB 3 SUB 2", "6"),
    # The vector products: `DOTPRODUCT` conjugates its right operand.
    ("[a + b*#i, c + d*#i] DOTPRODUCT [a + b*#i, c + d*#i]", "a^2 + b^2 + c^2 + d^2"),
    ("[1, 2] DOTPRODUCT [3, 4]", "11"),
    ("[1, 2] DOTPRODUCT [1, 2, 3]", "[1, 2] DOTPRODUCT [1, 2, 3]"),
    ("[[1, 2]] DOTPRODUCT [[1, 2]]", "[[1, 2]] DOTPRODUCT [[1, 2]]"),
    ("[a, b, c] CROSSPRODUCT [d, e, f]", "[b*f - c*e, c*d - a*f, a*e - b*d]"),
    ("[a, b] CROSSPRODUCT [c, d]", "a*d - b*c"),
    ("[a, b, c] CROSSPRODUCT [d, e]", "[a, b, c] CROSSPRODUCT [d, e]"),
    ("u CROSSPRODUCT v + 1", "u CROSSPRODUCT v + 1"),
]


@pytest.mark.parametrize(("text", "expected"), PARTS, ids=str)
def test_rows_columns_and_vector_products(text, expected):
    assert simp(text) == expected


#: Indexing: `v SUB i` and `ELEMENT` count from the left, or from the right for a
#: negative index, and a vector of indices selects one element for each, recursively.
#: Where an index isn't one of the positions the access stands.
INDEXING = [
    ("[a, b, c, d] SUB 2", "b"),
    ("[[a, b, c], [d, e, f], [g, h, i]] SUB 2 SUB 3", "f"),
    ("[a, b, c, d] SUB [4, 2]", "[d, b]"),
    ("[[a, b, c], [d, e, f], [g, h, i]] SUB [3, 1]", "[[g, h, i], [a, b, c]]"),
    ("ELEMENT([a, b, c, d], 2)", "b"),
    ("ELEMENT([[a, b, c], [d, e, f], [g, h, i]], 2, 3)", "f"),
    ("ELEMENT([a, b, c, d], [4, 2])", "[d, b]"),
    ("[a, b, c, d] SUB (-1)", "d"),
    ("[a, b, c, d] SUB (-4)", "a"),
    ("ELEMENT([a, b, c], -2)", "b"),
    ("[a, b, c] SUB 0", "ELEMENT([a, b, c], 0)"),
    ("[a, b, c] SUB 4", "ELEMENT([a, b, c], 4)"),
    ("[a, b, c] SUB (-4)", "ELEMENT([a, b, c], -4)"),
    ("[a, b, c] SUB [1, 4]", "ELEMENT([a, b, c], [1, 4])"),
    ("[a, b, c] SUB (3/2)", "ELEMENT([a, b, c], 3/2)"),
    ("[a, b, c] SUB k", "ELEMENT([a, b, c], k)"),
    ("VECTOR([a, b, c] SUB k, k, 3)", "[a, b, c]"),
    ("[a, b, c] SUB []", "[]"),
    ("[a, b, c] SUB [3, 3, 1]", "[c, c, a]"),
    ("[a, b, c] SUB [1, [2, 3]]", "[a, [b, c]]"),
    (
        "[p, q, q] SUB [[1, 2, 2], [2, 1, 2], [2, 2, 1]]",
        "[[p, q, q], [q, p, q], [q, q, p]]",
    ),
    ("ELEMENT([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 2, 1, 2)", "6"),
    ("[a, b, c, d, e] SUB [2, ..., 4]", "[b, c, d]"),
    ("[1, 4, 9, 16, 25] SUB [1, 3, ..., 5]", "[1, 9, 25]"),
    ("[a, b, c] SUB [1, ..., 0]", "[]"),
    (
        "[[-7, 2], [-6, 2.1], [-5, 2.2], [-4, 2.3], [-3, 2.4]] SUB [1, 2, 3, 4]",
        "[[-7, 2], [-6, 21/10], [-5, 11/5], [-4, 23/10]]",
    ),
    ("x SUB 1", "x SUB 1"),
    # A set of pairs is a function given by its graph, looked up by the first element.
    ("{[2, 3], [3, 5], [1, 7]} SUB 2", "3"),
    ("{[2, 3], [3, 5], [1, 7]} SUB 4", "{[1, 7], [2, 3], [3, 5]} SUB 4"),
    ("{[1, a], [1, b]} SUB 1", "{[1, a], [1, b]} SUB 1"),
    ("{[x, 1], [y, 2]} SUB y", "2"),
    ("{[2, 3, 4]} SUB 2", "3"),
    ("{1, 2} SUB 1", "{1, 2} SUB 1"),
    ("VECTOR({[1, a], [2, b]} SUB k, k, 2)", "[a, b]"),
    # Strings are sequences of characters and integers of digits, which keep the sign.
    ('"Derive" SUB 3', '"r"'),
    ('"Derive" SUB [1, 2, 6, 3]', '"Deer"'),
    ("123456789 SUB [1, 3, 5, -3, -1]", "13579"),
    ("3141590 SUB 7", "0"),
    ("VECTOR(3141590 SUB i, i, DIM(3141590))", "[3, 1, 4, 1, 5, 9, 0]"),
    ("1203 SUB [3, 4]", "3"),
    ("(-123) SUB 1", "-1"),
    ('"0.07s" SUB (-1)', '"s"'),
    ('"0000000" SUB [1, ..., 3]', '"000"'),
    ('"abc" SUB [1, ..., 0]', '""'),
    ("(2/3) SUB 1", "(2/3) SUB 1"),
]


@pytest.mark.parametrize(("text", "expected"), INDEXING, ids=str)
def test_indexing(text, expected):
    assert simp(text) == expected


#: The vector manipulation functions, on vectors, strings, integers and sets. Where an
#: argument isn't one they take, or an index isn't a position, the call stands.
MANIPULATION = [
    ("DIM([[a, b, c], [1, 2, 3]])", "2"),
    ("DIM([[a, b, c], [1, 2, 3]] SUB 1)", "3"),
    ('DIM("Derive")', "6"),
    ("DIM(123456789)", "9"),
    ("DIM(2^257 - 1)", "78"),
    ("DIM(16937389168607)", "14"),
    ("DIM(2^65536)", "19729"),
    ("DIM({a, b, c})", "3"),
    ("DIM({x, x + x, 2*x})", "2"),
    ("DIM([])", "0"),
    ("DIM({})", "0"),
    ('DIM("")', "0"),
    ("DIM(0)", "1"),
    ("DIM(-123)", "3"),
    ("DIM(x)", "DIM(x)"),
    ("DIM(2/3)", "DIM(2/3)"),
    ("DIMENSION([a, b])", "2"),
    ("ADJOIN(FIRST([a, b, c]), REST([d, e, f]))", "[a, e, f]"),
    ("FIRST([[1, 2], [3, 4]])", "[1, 2]"),
    ("REST([a, b, c, d], 2)", "[c, d]"),
    ("REST([a, b, c, d], -1)", "[a, b, c]"),
    ("REST([a, b, c], 0)", "[a, b, c]"),
    ("REST([a, b, c], 5)", "[]"),
    ("REST([a, b, c], -3)", "[]"),
    ("REST([a], 1)", "[]"),
    ("FIRST([])", "?"),
    ("REST([])", "[]"),
    ("REST([], 2)", "[]"),
    ("REST({})", "{}"),
    ('REST("")', '""'),
    ("REST([a, b], k)", "REST([a, b], k)"),
    ('FIRST("Derive")', '"D"'),
    ('REST("Derive")', '"erive"'),
    ('REST("a")', '""'),
    ("FIRST(905)", "9"),
    ("REST(905)", "5"),
    ("REST(5)", "REST(5)"),
    ("FIRST(-123)", "-1"),
    ("REST(-123)", "-23"),
    ("FIRST({3, 1, 2})", "1"),
    ("REST({3, 1, 2})", "{2, 3}"),
    ("FIRST({})", "FIRST({})"),
    ("ADJOIN(0, [1, 2])", "[0, 1, 2]"),
    ("ADJOIN([1, 2], [[3, 4]])", "[[1, 2], [3, 4]]"),
    ("ADJOIN(4, {1, 2})", "{1, 2, 4}"),
    ('ADJOIN("0", "7s")', '"07s"'),
    ('ADJOIN("", ?)', 'ADJOIN("", ?)'),
    ("ADJOIN(1, 23)", "123"),
    # Beside a string, an empty vector is the empty string; another vector is not.
    ('ADJOIN([], " a")', '" a"'),
    ('APPEND([], "a", [])', '"a"'),
    ('APPEND([1], "a")', 'APPEND([1], "a")'),
    ("ADJOIN(a, b)", "ADJOIN(a, b)"),
    (
        "ITERATES([n + 1, ADJOIN(0, v) + APPEND(v, [0])], [n, v], [0, [1]], 4)",
        "[[0, [1]], [1, [1, 1]], [2, [1, 2, 1]], [3, [1, 3, 3, 1]],"
        " [4, [1, 4, 6, 4, 1]]]",
    ),
    ("APPEND([a, b], [c, d], [e, f])", "[a, b, c, d, e, f]"),
    ("APPEND([[a, b], [c, d], [e, f]])", "[a, b, c, d, e, f]"),
    ("APPEND([[0], [], [1, 2]])", "[0, 1, 2]"),
    ("APPEND([[1], 2, [3]])", "[1, 2, 3]"),
    ("APPEND([1, 2])", "[1, 2]"),
    ('APPEND([["x", "y"]], [[1, 2], [3, 4]])', '[["x", "y"], [1, 2], [3, 4]]'),
    ('APPEND("a", "b", "c", 3)', '"abc3"'),
    ('APPEND("", "", "x")', '"x"'),
    ('APPEND("a", -5)', '"a-5"'),
    ("APPEND(12, 345)", "12345"),
    ("APPEND(-12, 34)", "-1234"),
    ("APPEND(10, 0)", "100"),
    ("APPEND([a], b)", "APPEND([a], b)"),
    ('APPEND("a", [1])', 'APPEND("a", [1])'),
    ('APPEND("a", 5/2)', 'APPEND("a", 5/2)'),
    ("APPEND({1}, {2})", "APPEND({1}, {2})"),
    ("APPEND()", "APPEND()"),
    ("DELETE([a, b, c, d], 2)", "[a, c, d]"),
    ("DELETE([a, b, c, d], [2, -1])", "[a, c]"),
    ("DELETE([a, b, c, d, e], [1, 3, 5])", "[b, d]"),
    ("DELETE([a, b, c], {1, 3})", "[b]"),
    ("DELETE([a, b, c], -1)", "[a, b]"),
    ("DELETE([a, b, c])", "[b, c]"),
    ("DELETE([a, b, c], [1, 1])", "[b, c]"),
    ("DELETE([a, b, c], [])", "[a, b, c]"),
    ("DELETE([[1, 2], [3, 4]], 1)", "[[3, 4]]"),
    ('DELETE("Derive", 2)', '"Drive"'),
    ('DELETE("Derive", -5)', '"Drive"'),
    ('DELETE("a", 1)', '""'),
    ("DELETE(12345, 1)", "2345"),
    ("DELETE(1023, 1)", "23"),
    ("DELETE(7, 1)", "DELETE(7, 1)"),
    # An index past the end designates no element, and nothing is deleted.
    ("DELETE([a, b, c], 4)", "[a, b, c]"),
    ("DELETE([a, b, c], -4)", "[a, b, c]"),
    ("DELETE([a, b, c], 0)", "DELETE([a, b, c], 0)"),
    ("DELETE([a, b, c], [1, 5])", "[b, c]"),
    ("DELETE(x, 1)", "DELETE(x, 1)"),
    ("DELETE_ELEMENT([a, b, c], 2)", "[a, c]"),
    ("INSERT(d, [a, b, c], 2)", "[a, d, b, c]"),
    ("INSERT(d, [a, b, c])", "[d, a, b, c]"),
    ("INSERT(d, [a, b, c], 4)", "[a, b, c, d]"),
    ("INSERT(d, [a, b, c], 0)", "[a, b, c, d]"),
    ("INSERT(d, [a, b, c], -1)", "[a, b, d, c]"),
    ("INSERT(d, [a, b, c], -3)", "[d, a, b, c]"),
    ("INSERT(d, [a, b, c], -4)", "INSERT(d, [a, b, c], -4)"),
    ("INSERT(d, [a, b, c], 5)", "INSERT(d, [a, b, c], 5)"),
    ("INSERT(0, [1, 2])", "[0, 1, 2]"),
    ('INSERT(".", "007s", -3)', '"0.07s"'),
    ("INSERT(\"'\", \"1.23\", 3)", "\"1.'23\""),
    ("INSERT(5, 123, 2)", "1523"),
    ("INSERT(d, {a}, 1)", "INSERT(d, {a}, 1)"),
    ("INSERT_ELEMENT(d, [a, b, c], 2)", "[a, d, b, c]"),
    ("REPLACE(d, [a, b, c], 2)", "[a, d, c]"),
    ("REPLACE(d, [a, b, c])", "[d, b, c]"),
    # A negative position counts from the right; one that designates no element
    # replaces nothing, as DELETE deletes nothing there.
    ("REPLACE(d, [a, b, c], -1)", "[a, b, d]"),
    ("REPLACE(d, [a, b, c], -3)", "[d, b, c]"),
    ("REPLACE(d, [a, b, c], -4)", "[a, b, c]"),
    ("REPLACE(d, [a, b, c], 0)", "[a, b, c]"),
    ("REPLACE(d, [a, b, c], 4)", "[a, b, c]"),
    ("REPLACE(d, [a, b, c], 1/2)", "REPLACE(d, [a, b, c], 1/2)"),
    ("REPLACE(d, [a, b, c], n)", "REPLACE(d, [a, b, c], n)"),
    (
        "REPLACE([p, q, r], [[a, b, c, d], [e, f, g, h], [i, j, k, l]]`, 3)`",
        "[[a, b, p, d], [e, f, q, h], [i, j, r, l]]",
    ),
    ('REPLACE("x", "abc", 2)', '"axc"'),
    ('REPLACE("xy", "abc", 2)', '"axyc"'),
    ("REPLACE(9, 123, -1)", "129"),
    ("REPLACE_ELEMENT(d, [a, b, c], 2)", "[a, d, c]"),
    ("REVERSE([a, b, c])", "[c, b, a]"),
    ("REVERSE(1234567890)", "987654321"),
    ('REVERSE("1000")', '"0001"'),
    ('REVERSE("Derive")', '"evireD"'),
    ("REVERSE(-123)", "-321"),
    ("REVERSE([[1, 2], [3, 4]])", "[[3, 4], [1, 2]]"),
    ("REVERSE([])", "[]"),
    ("REVERSE([a, b, c]) * [1, 2, 3]", "3*a + 2*b + c"),
    ("REVERSE({1, 2})", "REVERSE({1, 2})"),
    ("REVERSE(x)", "REVERSE(x)"),
    ("REVERSE_VECTOR([a, b, c])", "[c, b, a]"),
    # Elements are compared simplified; a string or integer is searched for as a run.
    ("POSITION(c, [a, b, c, d, c, e])", "3"),
    ("POSITION(f, [a, b, c, d, c, e])", "false"),
    ("POSITION(c, [a, b, c, d, c, e], 4)", "5"),
    ("POSITION(c, [a, b, c, d, c, e], -2)", "5"),
    ("POSITION(c, [a, b, c], 4)", "false"),
    ("POSITION(c, [a, b, c], 0)", "POSITION(c, [a, b, c], 0)"),
    ("POSITION(2*x, [x, x + x])", "2"),
    ("POSITION([1, 2], [[3, 4], [1, 2]])", "2"),
    ("[POSITION(1, [a, b]) = false]", "[true]"),
    ('POSITION(".", "3.53\'171")', "2"),
    ('POSITION("\'", "3.53\'171")', "5"),
    ('POSITION("ri", "Derive")', "3"),
    ('POSITION("4", 14242415)', "2"),
    ('POSITION("11", 12345)', "false"),
    ("POSITION(0, 1203)", "3"),
    ('POSITION("0000", 1234500006)', "6"),
    ("POSITION(1, {1, 2})", "POSITION(1, {1, 2})"),
    ("POSITION(c, x)", "POSITION(c, x)"),
    (
        "DIM([[1, 2], [2, 1], [3, 4], [1, 2], [5, 6]]) - POSITION(FIRST(REVERSE("
        "[[1, 2], [2, 1], [3, 4], [1, 2], [5, 6]])), [[1, 2], [2, 1], [3, 4], [1, 2], "
        "[5, 6]])",
        "0",
    ),
]


@pytest.mark.parametrize(("text", "expected"), MANIPULATION, ids=str)
def test_vector_manipulation(text, expected):
    assert simp(text) == expected


#: `SORT` orders numbers, exactly, then variables and strings by name, then any other
#: expression, then vectors lexicographically. With a test and two variables bound in
#: it, it sorts stably by the test. An order it can't decide leaves the call standing.
SORTED = [
    (
        "SORT([3, 2, y, pi, #e, SQRT(2), z, x, 1, 0, 2, 0, 0, -3])",
        "[-3, 0, 0, 0, 1, SQRT(2), 2, 2, #e, 3, pi, x, y, z]",
    ),
    ("SORT({[2, 3], [3, 1], [2, 1], [1, 5]})", "[[1, 5], [2, 1], [2, 3], [3, 1]]"),
    ("SORT([-5, -2, 0, 3, 4], ABS(x) < ABS(y), x, y)", "[0, -2, 3, 4, -5]"),
    (
        "SORT([pi, SQRT(3), SQRT(2), #e, SQRT(5), SQRT(2), #e])",
        "[SQRT(2), SQRT(2), SQRT(3), SQRT(5), #e, #e, pi]",
    ),
    ("SORT([pi, -SQRT(2), 0, -100.25, 100*#e])", "[-401/4, -SQRT(2), 0, pi, 100*#e]"),
    ("SORT([[2, 1], [1, 5], [1, 2]])", "[[1, 2], [1, 5], [2, 1]]"),
    ("SORT({3, 1, 2})", "[1, 2, 3]"),
    ("SORT([3 + #i, 3 - #i, 2])", "[2, 3 - #i, 3 + #i]"),
    ("SORT([inf, 1, -inf])", "[-inf, 1, inf]"),
    ('SORT(["b", "a", x])', '["a", "b", x]'),
    ("SORT([x + 1, x, 1])", "[1, x, x + 1]"),
    ("SORT([[1, 2], 3, b])", "[3, b, [1, 2]]"),
    ("SORT([[1, 2, 3], [1, 2]])", "[[1, 2], [1, 2, 3]]"),
    ("SORT([1, ?])", "SORT([1, ?])"),
    ("SORT([3, 1, 2], x > y, x, y)", "[3, 2, 1]"),
    ("SORT({3, 1, 2}, x > y, x, y)", "[3, 2, 1]"),
    ("SORT([[1, 3], [2, 1]], x SUB 2 < y SUB 2, x, y)", "[[2, 1], [1, 3]]"),
    (
        "SORT([[1, b], [2, a]], x SUB 2 < y SUB 2, x, y)",
        "SORT([[1, b], [2, a]], x SUB 2 < y SUB 2, x, y)",
    ),
    ("SORT([2, -2, 1], ABS(x) < ABS(y), x, y)", "[1, 2, -2]"),
    ("SORT(x)", "SORT(x)"),
]


@pytest.mark.parametrize(("text", "expected"), SORTED, ids=str)
def test_sorting(text, expected):
    assert simp(text) == expected


async def test_a_sorting_test_has_its_variables_bound():
    # An assigned `x` is not the `x` of the test, and a function passes its vector on.
    session = Session()
    session.author("x := 5")
    session.author("F(v) := SORT(v, x > y, x, y)")
    assert (await session.simplify("F([1, 3, 2])")).text == "[3, 2, 1]"
    assert (await session.simplify("SORT([1, 3, 2], x > y, x, y)")).text == "[3, 2, 1]"


#: The rank, where every pivot is decided exactly: a pivot that is zero for some values
#: of its variables leaves the call standing.
RANKS = [
    ("RANK([[2, 3, 5], [4, 6, 10], [1, 2, 3]])", "2"),
    ("RANK(IDENTITY_MATRIX(3))", "3"),
    ("RANK([[0, 0], [0, 0]])", "0"),
    ("RANK([[1, x], [x, x^2]])", "1"),
    ("RANK([[1, 0], [0, x^2 + 1]])", "2"),
    ("RANK([[1, 0], [0, x - 2]])", "RANK([[1, 0], [0, x - 2]])"),
    ("RANK([[a, b], [c, d]])", "RANK([[a, b], [c, d]])"),
    ("RANK([1, 2])", "RANK([1, 2])"),
]


@pytest.mark.parametrize(("text", "expected"), RANKS, ids=str)
def test_rank(text, expected):
    assert simp(text) == expected


@pytest.mark.parametrize(("text", "expected"), VECTORS, ids=str)
def test_vectors_and_matrices(text, expected):
    assert simp(text) == expected


#: A vector's shape is its nesting alone: a one-row matrix is not the vector it holds,
#: elements can be anything, including vectors of any rank, and nothing depends on
#: whether an element was written as a vector or computed. Checked in Derive 4.11
#: where noted (D).
NESTING = [
    ("DIMENSION([[1, 2]])", "1"),  # D
    ("ITERATES(v, v, [1, 2], 0)", "[[1, 2]]"),  # D
    ("SUM([[1, 2], [3, 4]])", "[4, 6]"),  # D
    ("SIGN([3, 4])", "[3/5, 4/5]"),  # D
    ("VECTOR(RHS(e), e, [x = 2, x = -2])", "[2, -2]"),
    ("ROW_REDUCE([LIM([x, 1], x, 2), [3, 4]])", "[[1, 0], [0, 1]]"),
    ("[ELEMENT([[0, 1], [2, 3]], 1), [5, 6]]`", "[[0, 5], [1, 6]]"),
    ("LIM([x, [x, x]], x, 1)", "[1, [1, 1]]"),
    ("ITERATE([p, m + 1], [p, m], [[1], 1], 2)", "[[1], 3]"),
    # TENSOR.DOC 3.3: a scalar multiplies every element of a tensor.
    (
        "1/2*[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]",
        "[[[1/2, 1], [3/2, 2]], [[5/2, 3], [7/2, 4]]]",
    ),
    # A one-row matrix transposed is a column, and a column times a row their outer
    # product.
    ("[[1, 2]]`", "[[1], [2]]"),
    ("[[1, 2]]`*[VECTOR(k, k, 3)]", "[[1, 2, 3], [2, 4, 6]]"),
    # A flat vector is its own transpose (D).
    ("[1, 2]`", "[1, 2]"),
    ("[[1], [2]]`", "[[1, 2]]"),
    ("[[1, 2]] . [3, 4]", "[11]"),
    # Ragged rows transpose as Derive does it: row j holds the j-th elements of the
    # rows that have one.
    ("[[1, 2], [3]]`", "[[1, 3], [2]]"),
    ("[[u, a], [v, b], [w, c, d, e]]`", "[[u, v, w], [a, b, c], [d], [e]]"),
    # A vector mixing rows and other elements has no transpose.
    ("[[1, 2], 3]`", "[[1, 2], 3]`"),
    # Derive 6 applies elementary functions and adds a scalar element by element,
    # where DOS Derive left both standing.
    ("SIN([1, 2])", "[SIN(1), SIN(2)]"),
    ("[1, 2] + 2", "[3, 4]"),
    ("[[1, 2, 3]] - 1", "[[0, 1, 2]]"),
    # A square matrix plus a scalar adds it on the diagonal.
    ("[[1, 2], [3, 4]] + 7", "[[8, 2], [3, 11]]"),
    # The elements of a vector sum or product are combined as vectors where they are.
    ("SUM(VECTOR([k, 1], k, 3))", "[6, 3]"),
    ("PRODUCT([[1, 2], [3, 4]])", "11"),
    ("SIGN([[3, 4]])", "[[3/5, 4/5]]"),
    ("[[1, 2], [3, 4]] . [[1], [1]]", "[[3], [7]]"),
    ("ROW_REDUCE([[1, 2], [3, 4]], [5, 6])", "[[1, 0, -4], [0, 1, 9/2]]"),
    # Arithmetic on an element that is an equation applies to both sides, as it does
    # to an equation operand.
    ("[x = 1] + [y = 2]", "[x + y = 3]"),
    ("x*[y = 1]", "[x*y = x]"),
    # Vectors are equal only if their shapes are.
    ("[[[1, 2]] = [[1, 2]]]", "[true]"),
    ("[[1, 2]] = [1, 2]", "false"),
    ("[[] = []]", "[true]"),
    # A vector of equations isn't related element by element.
    ("[x = 1, y = 2] = [1, 2]", "[x = 1, y = 2] = [1, 2]"),
]


@pytest.mark.parametrize(("text", "expected"), NESTING, ids=str)
def test_a_vector_is_shaped_by_its_nesting_alone(text, expected):
    assert simp(text) == expected


#: The elementary, special and integer functions apply to each element of a vector, at
#: every depth, and of a set; several vectors of one length pair up. `EXP` and `SQRT`
#: of a square matrix are its matrix exponential and square root instead.
ELEMENTWISE = [
    ("SIN([0, pi])", "[0, 0]"),
    ("SQRT([4, 9])", "[2, 3]"),
    ("LN([1, #e])", "[0, 1]"),
    ("EXP([1, 2])", "[#e, #e^2]"),
    ("EXP([LN(2), LN(3)])", "[2, 3]"),
    ("COS([[0, pi], [pi, 0]])", "[[1, -1], [-1, 1]]"),
    ("PHASE([#i, -1])", "[pi/2, pi]"),
    ("RE([1 + 2*#i, 3; #i, 4])", "[[1, 3], [0, 4]]"),
    ("CONJ([[#i, 2]])", "[[-#i, 2]]"),
    ("GAMMA([1, 2, 3])", "[1, 1, 2]"),
    ("[1, 2, 3]!", "[1, 2, 6]"),
    ("FLOOR([1.5, 2.5])", "[1, 2]"),
    ("CEILING([1.2, 3])", "[2, 3]"),
    # A set's results are a set, duplicates merged.
    ("SIN({0, pi})", "{0}"),
    ("LN({1, #e})", "{0, 1}"),
    ("FLOOR({1/2, 3/2, 7/4})", "{0, 1}"),
    # One collection among several arguments is mapped over, the others fixed.
    ("LOG([8, 4], 2)", "[3, 2]"),
    ("MOD(10, [3, 4])", "[1, 2]"),
    (
        "MOD(k*[[73], [110], [32]], 127)",
        "[[MOD(73*k, 127)], [MOD(110*k, 127)], [MOD(32*k, 127)]]",
    ),
    # Vectors of one length pair up; of different lengths, the call stands.
    ("LOG([8, 9], [2, 3])", "[3, 2]"),
    ("ATAN([1, 1], [1, -1])", "[pi/4, 3*pi/4]"),
    ("MOD([1, 2], [3, 4, 5])", "MOD([1, 2], [3, 4, 5])"),
    # Equations are mapped as before, and ABS is a vector's length.
    ("SIN(x = 0)", "SIN(x) = 0"),
    ("ABS([3, 4])", "5"),
    # A scalar to a vector is the power to each element.
    ("2^[1, 2, 3]", "[2, 4, 8]"),
    ("x^[0, 1, 2]", "[1, x, x^2]"),
    ("[a, b, c]^(-2)", "1/(a^2 + b^2 + c^2)"),
    # The matrix exponential and root of a square matrix.
    ("EXP([[0, 1], [0, 0]])", "[[1, 1], [0, 1]]"),
    ("EXP([[a, 0], [0, b]])", "[[#e^a, 0], [0, #e^b]]"),
    ("EXP([[0, x], [-x, 0]])", "[[COS(x), SIN(x)], [-SIN(x), COS(x)]]"),
    ("#e^[[1, 0], [0, 2]]", "[[#e, 0], [0, #e^2]]"),
    ("EXP([[a]])", "[[#e^a]]"),
    ("EXP([[1, 2]])", "[[#e, #e^2]]"),
    ("SQRT([[4, 0], [0, 9]])", "[[2, 0], [0, 3]]"),
    ("SQRT([[0, 1], [0, 0]])", "SQRT([[0, 1], [0, 0]])"),
]


@pytest.mark.parametrize(("text", "expected"), ELEMENTWISE, ids=str)
def test_a_function_applies_to_each_element(text, expected):
    assert simp(text) == expected


def test_a_vector_of_solutions_is_a_vector_to_generate_over():
    # The solutions in Derive's order, 2 before -2.
    assert simp("VECTOR(e^2 + e, e, SOLUTIONS(x^2 = 4, x))") == "[6, 2]"


# -- sets ---------------------------------------------------------------------

#: A set keeps each element once, numbers first in ascending order. Elements are
#: expressions, the same element only when they simplify to the same expression.
SETS = [
    ("{a, b, b, c}", "{a, b, c}"),
    ("{x, 1, y, 2/3, -5}", "{-5, 2/3, 1, x, y}"),
    ("{1, 3, 5, 7, 9} UNION {2, 3, 5, 8, 13}", "{1, 2, 3, 5, 7, 8, 9, 13}"),
    ("{1, ..., 10}\\{2, 3, 5, 7}", "{1, 4, 6, 8, 9, 10}"),
    ("{1, 2, 3} INTERSECTION {2, 3, 4}", "{2, 3}"),
    ("{1, 2} UNION {3} INTERSECTION {3}", "{1, 2, 3}"),
    # Elements are expressions, different unless they simplify to the same one.
    ("{1, 2}\\{x}", "{1, 2}"),
    ("{1, 2, x}\\{2, 5}", "{1, x}"),
    ("{a, b} INTERSECTION {c}", "{}"),
    ("{a, b, c} INTERSECTION {b, d}", "{b}"),
    ("{x + x, 2*x}", "{2*x}"),
    ("{SIN(x)^2 + COS(x)^2, 2}\\{1}", "{2}"),
    ("s INTERSECTION {a}", "s INTERSECTION {a}"),
    ("{1, ..., n}\\{1}", "{1, ..., n}\\{1}"),
    ("POWER_SET({a, b, c})", "{{}, {a}, {b}, {c}, {a, b}, {a, c}, {b, c}, {a, b, c}}"),
    ("POWER_SET({1, 2, 3}, 2)", "{{1, 2}, {1, 3}, {2, 3}}"),
    ("POWER_SET({a, b, c}, 2)", "{{a, b}, {a, c}, {b, c}}"),
    ("{a, b}*{c, d}", "{[a, c], [a, d], [b, c], [b, d]}"),
    ("{2, 3}*{1}*{4, 5}", "{[2, 1, 4], [2, 1, 5], [3, 1, 4], [3, 1, 5]}"),
    # Vector elements are tuples the product extends, so it is associative.
    ("({a}*{b})*{c, d}", "{[a, b, c], [a, b, d]}"),
    ("{[1]}*{[2], [3, 4]}", "{[1, 2], [1, 3, 4]}"),
    ("{1, 2}^2", "{[1, 1], [1, 2], [2, 1], [2, 2]}"),
    ("2*{1, 2}", "{2, 4}"),
    ("{1, 2}*[3, 4]", "[{3, 6}, {4, 8}]"),
    ("{1, 2} + 1", "{1, 2} + 1"),
    ("{a, b}``", "{a, b}"),
    ("{a, b} INTERSECTION {a, b}`", "{}"),
    ("{a, b} UNION {a, b}`", "{}`"),
    ("DIMENSION({1, 2, 3})", "3"),
    ("DIMENSION({a, b})", "2"),
    ("DIMENSION({SIN(x)^2 + COS(x)^2, 1})", "1"),
    ("[{1, 2} = {2, 1}]", "[true]"),
    ("{1, 2} = {1, 2, 3}", "false"),
    ("[{a, b} = {b, a}]", "[true]"),
    ("{a, b} = {a, c}", "false"),
    ("[{a, b} = {a, c}]", "[false]"),
    ("{a, b} /= {a, c}", "true"),
    ("[{{x}} = {{1}}]", "[false]"),
    # The ellipsis counts by the step between the first two terms, or by one.
    ("[0, 2, ..., 10]", "[0, 2, 4, 6, 8, 10]"),
    ("[x, x + 1, ..., x + 3]", "[x, x + 1, x + 2, x + 3]"),
    ("[1/2, ..., 3]", "[1/2, 3/2, 5/2]"),
    ("{10, ..., 1}", "{}"),
    ("{1, ..., n}", "{1, ..., n}"),
    # `SUBSET` of two explicit sets compares their elements simplified.
    ("{1, 2} SUBSET {1, 2, 3}", "true"),
    ("{1, 4} SUBSET {1, 2, 3}", "false"),
    ("{} SUBSET {a}", "true"),
    ("{x} SUBSET {x, y}", "true"),
    ("{2*x} SUBSET {x + x}", "true"),
    ("{a, b} SUBSET {a, c}", "false"),
    ("{[a, b]} SUBSET {[a, c]}", "false"),
    ("{1, 2} SUBSET {1, ..., 5}", "true"),
    ("{1, 2} SUBSET {1, ..., n}", "{1, 2} SUBSET {1, ..., n}"),
    ("{a} SUBSET {a}`", "{a} SUBSET {a}`"),
    ("s SUBSET t", "s SUBSET t"),
    ("[1, 2] SUBSET {1, 2}", "[1, 2] SUBSET {1, 2}"),
    # `MEMBER?` compares simplified elements by identity, and is never undecided for an
    # explicit set or vector.
    ("MEMBER?(2*x, {x, x + x, x^2})", "true"),
    ("MEMBER?(3, [1, 2])", "false"),
    ("MEMBER?(a, [b])", "false"),
    ("MEMBER?([1, 2], [[1, 2], [3, 4]])", "true"),
    ("MEMBER?(1, 123)", "MEMBER?(1, 123)"),
    ("MEMBER?(x, y)", "MEMBER?(x, y)"),
    # On truth values the set operations are the logical ones, and the logical
    # operations on explicit sets are the set ones; variables are neither.
    ("true UNION false", "true"),
    ("true INTERSECTION false", "false"),
    ("(x = 1) UNION (y = 2)", "x = 1 OR y = 2"),
    ("true`", "false"),
    ("{1, 2} AND {2, 3}", "{2}"),
    ("{1} OR {2}", "{1, 2}"),
    ("NOT {1}", "{1}`"),
    ("p UNION q", "p UNION q"),
    ("p OR q", "p OR q"),
    # A set to a set has no meaning.
    ("{m}^{n}", "{m}^{n}"),
    ("{0, 1}^2", "{[0, 0], [0, 1], [1, 0], [1, 1]}"),
    ("FIRST({m}^2)", "[m, m]"),
    ("SORT({1, ..., 8} \\ {1, 5, 7, 8})", "[2, 3, 4, 6]"),
]


@pytest.mark.parametrize(("text", "expected"), SETS, ids=str)
def test_sets(text, expected):
    assert simp(text) == expected


# -- declared nonscalars ------------------------------------------------------

NONSCALAR = declared(*(f"{name} :epsilon Nonscalar" for name in "abcd"))

#: Behavior of declared nonscalars beyond the nine rules of section 8.8. All come from
#: the original; none are in the manual.
NONSCALARS = [
    # 8.4: multiplying two matrices gives the matrix product, however it was written.
    ("a*b", "a . b"),
    # A matrix times its inverse is left as written: there is no notation for an
    # identity matrix of unknown dimension.
    ("a . a^-1", "a . a^-1"),
    ("a^-1 . a", "a^-1 . a"),
    # Longer products nest to the right, with parentheses printed. `a . (b . c)` is the
    # fixed point; `(a . b) . c` is not.
    ("((a . b) . c) . d", "a . (b . (c . d))"),
    ("a . b . c . d", "a . (b . (c . d))"),
    ("a . (b . c)", "a . (b . c)"),
    # The associativity rule of 8.8.
    ("a . (b . c) - (a . b) . c", "0"),
    # The determinant of a product (the manual states this only for an inverse).
    ("DET(a . b)", "DET(a)*DET(b)"),
    ("TRACE(a + b)", "TRACE(a) + TRACE(b)"),
    # Distribution works inside a longer product too.
    ("a . (b + c) . a", "a . (b . a) + a . (c . a)"),
    # A scalar commutes with everything, so a negated product needs no parentheses.
    ("a . b - b . a", "a . b - b . a"),
    ("-(a . b)", "-a . b"),
    # A distributed product cancels against the rest of the sum, as MATRIX.DMO's
    # vector algebra step shows.
    ("a . (b + c) - (b` . a`)`", "a . c"),
    # Dividing by a matrix means multiplying by its inverse.
    ("1/a", "a^-1"),
    ("x/a", "x*a^-1"),
]


@pytest.mark.parametrize(("text", "expected"), NONSCALARS, ids=str)
def test_the_algebra_of_a_declared_nonscalar(text, expected):
    assert simp(text, NONSCALAR) == expected


def test_a_scalars_transpose_collapses_at_the_first_backquote():
    # 8.5: the transpose of a scalar is the scalar, and anything not declared nonscalar
    # is a scalar.
    assert simp("x`") == "x"
    assert simp("x``") == "x"
    assert simp("(x + y)`") == "x + y"
    # A nonscalar's transpose stays, and two cancel.
    assert simp("a`", NONSCALAR) == "a`"
    assert simp("a``", NONSCALAR) == "a"


def test_a_default_domain_of_nonscalar_does_not_make_every_variable_a_matrix():
    # `default :epsilon Nonscalar` applies to everything, including function arguments,
    # where a matrix would be meaningless. So the default sets the assumptions but not
    # the matrix shape an explicitly declared name gets.
    default = declared("default :epsilon Nonscalar")
    assert simp("SIN(x)", default) == "SIN(x)"
    assert simp("x . y", default) == "x . y"


#: Row echelon form, characteristic polynomial and eigenvalues: the manual's examples,
#: and each function's accepted argument forms.
LINEAR_ALGEBRA = [
    # The manual's two augmented examples. The second is singular: the system for the
    # last column is inconsistent, shown by the pivot in that column.
    ("ROW_REDUCE([[1,2],[5,6]],[[3,4],[7,8]])", "[[1, 0, -1, -2], [0, 1, 2, 3]]"),
    ("ROW_REDUCE([[1,1],[2,2]],[[1,1],[2,1]])", "[[1, 1, 1, 0], [0, 0, 0, 1]]"),
    # The manual's exercise, solved exactly: the augmented column holds the solution of
    # the 3 x 3 system.
    (
        "ROW_REDUCE([[5,3,-7],[2,-8,1],[-1,9,4]],[[4],[6],[5]])",
        "[[1, 0, 0, 879/302], [0, 1, 0, 53/302], [0, 0, 1, 239/151]]",
    ),
    # A vector is adjoined as a column, same as a one-column matrix.
    (
        "ROW_REDUCE([[5,3,-7],[2,-8,1],[-1,9,4]],[4,6,5])",
        "[[1, 0, 0, 879/302], [0, 1, 0, 53/302], [0, 0, 1, 239/151]]",
    ),
    # With one argument: a nonsingular matrix reduces to the identity, a rank-deficient
    # one keeps a zero row.
    ("ROW_REDUCE([[1,2],[3,4]])", "[[1, 0], [0, 1]]"),
    ("ROW_REDUCE([[1,2,3],[4,5,6],[7,8,9]])", "[[1, 0, -1], [0, 1, 2], [0, 0, 0]]"),
    # Symbolic entries give the generic case, assuming no pivot is zero, which matches
    # the answer via the inverse.
    (
        "ROW_REDUCE([[a,b],[c,d]],[e,f])",
        "[[1, 0, (d*e - b*f)/(a*d - b*c)], [0, 1, (a*f - c*e)/(a*d - b*c)]]",
    ),
    # The manual's characteristic polynomial, term for term.
    ("CHARPOLY([[2,3],[a,b]],z)", "z^2 - z*(b + 2) - 3*a + 2*b"),
    # The variable defaults to `w`, which isn't on the order list, so `a` is the main
    # variable: the normal form separates `-3*a` and writes the rest in terms of `b`.
    # Derive answers `(2 - w)*(b - w) - 3*a` because its determinant stays as a product.
    # Sympy returns the polynomial multiplied out; given that sum, Derive answers
    # exactly as this does.
    ("CHARPOLY([[2,3],[a,b]])", "-3*a + b*(2 - w) + w^2 - 2*w"),
    # The definition is `DET(A - z*IDENTITY_MATRIX(3))`, so in odd dimension the cube is
    # negated; a sum of more than two terms keeps the leading minus.
    ("CHARPOLY([[1,2,3],[4,5,6],[7,8,10]],z)", "-z^3 + 16*z^2 + 12*z - 3"),
    # Any value may stand for the variable: at 0 the polynomial is the determinant.
    ("CHARPOLY([[-4,1],[-5,0]],0)", "5"),
    ("CHARPOLY([[1,2],[3,4]],x + 1)", "x^2 - 3*x - 6"),
    # Eigenvalues are given as solved equations: the manual's example, and a 2 x 2
    # needing the quadratic formula. For a pair differing only in a radical's sign,
    # Derive lists the sum first; this follows sympy's order, which reverses that case
    # but otherwise agrees with Derive.
    ("EIGENVALUES([[2,3],[0,b]],z)", "[z = 2, z = b]"),
    ("EIGENVALUES([[5,0],[0,2]],z)", "[z = 2, z = 5]"),
    ("EIGENVALUES([[1,2],[3,4]],x)", "[x = SQRT(33)/2 + 5/2, x = 5/2 - SQRT(33)/2]"),
    # A repeated eigenvalue is listed once; multiplicity shows in the eigenvectors, not
    # as a second solution.
    ("EIGENVALUES([[1,0,0],[0,1,0],[0,0,2]],z)", "[z = 1, z = 2]"),
    # A symmetric matrix has real eigenvalues. This one's characteristic polynomial is a
    # casus irreducibilis, whose three real roots Cardano's formula writes with `#i`;
    # Viete's trigonometric form writes them without it. They come in the order of
    # Derive's trigonometric formula: for this depressed cubic (`q < 0`) the largest,
    # the smallest, the middle one.
    (
        "EIGENVALUES([[0,1,1],[1,1,1],[1,1,-1]],z)",
        "[z = 4*SQRT(3)*COS(ACOS(3*SQRT(3)/8)/3)/3, "
        "z = -4*SQRT(3)*COS(pi/3 - ACOS(3*SQRT(3)/8)/3)/3, "
        "z = -4*SQRT(3)*SIN(pi/6 - ACOS(3*SQRT(3)/8)/3)/3]",
    ),
    # A characteristic polynomial with a factor that has no radical roots gives the
    # roots of the other factors, and that factor's equation.
    (
        "EIGENVALUES([[-a, -b, -c, -d, -e, 0], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], "
        "[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 3]], z)",
        "[z = 3, z^5 + a*z^4 + b*z^3 + c*z^2 + d*z + e = 0]",
    ),
    # A non-square matrix has no characteristic polynomial, and the quintic from the
    # manual's 5 x 5 exercise has no radical roots (the manual's reason exact eigenvalues
    # stop at 4 x 4). Both stay unevaluated.
    ("CHARPOLY([[1,2],[3,4],[5,6]],z)", "CHARPOLY([[1, 2], [3, 4], [5, 6]], z)"),
    (
        "EIGENVALUES([[-2,1,1,1,1],[1,-3,-1,0,-1],[1,-1,1,0,-3],[1,0,0,3,0],"
        "[1,-1,-3,0,2]],z)",
        "EIGENVALUES([[-2, 1, 1, 1, 1], [1, -3, -1, 0, -1], [1, -1, 1, 0, -3], "
        "[1, 0, 0, 3, 0], [1, -1, -3, 0, 2]], z)",
    ),
]


@pytest.mark.parametrize(("text", "expected"), LINEAR_ALGEBRA, ids=str)
def test_linear_algebra(text, expected):
    assert simp(text) == expected


#: The Derive 6.1 help's examples of vector products, eigenvalues, the other matrix
#: built-ins and truth tables, and results recorded in Derive 5 and 6 worksheets.
DERIVE_6_MATRICES = [
    # The conjugating dot product of vectors of one length, and the cross product of
    # two 3- or 2-element vectors; anything else stands.
    (
        "[a + b*#i, c + d*#i] DOTPRODUCT [e + f*#i, g + h*#i]",
        "a*(e - #i*f) + b*(f + #i*e) + c*(g - #i*h) + d*(h + #i*g)",
    ),
    ("[a + b*#i, c + d*#i] DOTPRODUCT [a + b*#i, c + d*#i]", "a^2 + b^2 + c^2 + d^2"),
    ("[1, 2] DOTPRODUCT [1, 2, 3]", "[1, 2] DOTPRODUCT [1, 2, 3]"),
    ("[a, b, c] CROSSPRODUCT [d, e, f]", "[b*f - c*e, c*d - a*f, a*e - b*d]"),
    ("[a, b] CROSSPRODUCT [c, d]", "a*d - b*c"),
    ("[a, b, c] CROSSPRODUCT [d, e]", "[a, b, c] CROSSPRODUCT [d, e]"),
    # The length of a vector is the root of the sum of its elements' squared moduli.
    ("ABS([24, 24, 12])", "36"),
    ("ABS([3, 4*#i])", "5"),
    ("ABS([])", "0"),
    # Without a variable, the eigenvalues themselves, each once; approximating, the
    # zeros of a factor without radical ones too.
    ("EIGENVALUES([[1, 2], [1, 1]])", "[SQRT(2) + 1, 1 - SQRT(2)]"),
    ("EIGENVALUES([[3, 2], [-2, -1]])", "[1]"),
    ("EIGENVALUES([[1, 2], [2, -2]])", "[2, -3]"),
    ("EIGENVALUES([[1, 2], [2, -2]], w)", "[w = 2, w = -3]"),
    (
        "EIGENVALUES([[0, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 0, 0, 0], "
        "[0, 0, 1, 0, 0], [0, 0, 0, 1, 0]])",
        "EIGENVALUES([[0, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 0, 0, 0], "
        "[0, 0, 1, 0, 0], [0, 0, 0, 1, 0]])",
    ),
    ("CHARPOLY([[2, 3], [a, b]], z)", "z^2 - z*(b + 2) - 3*a + 2*b"),
    ("TRACE([[a, b, c], [d, e, f], [g, h, i]])", "a + e + i"),
    ("DET([[t^2 + 5*t, t^2 - 5*t], [2*t + 5, 2*t - 5]])", "10*t^2"),
    ("DET([[10, 2, 0], [2, 8, 2], [0, 2, 10]])", "720"),
    ("[[2, 3], [a, b]]^2", "[[3*a + 4, 3*b + 6], [a*(b + 2), 3*a + b^2]]"),
    ("[[2, 3], [4*a, 6*a]]^(-1)", "[[2, 3], [4*a, 6*a]]^-1"),
    ("[[a, b, c], [1, 2, 3]]`", "[[a, 1], [b, 2], [c, 3]]"),
    ("[[x = 1, x = 2], [x = 3, x = 4]]`", "[[x = 1, x = 3], [x = 2, x = 4]]"),
    ("[a, b, c]`", "[a, b, c]"),
    ("ROW_REDUCE([[1, 1], [2, 2]], [[1, 1], [2, 1]])", "[[1, 1, 1, 0], [0, 0, 0, 1]]"),
    ("ROW_REDUCE([[1, 0], [0, x - 2]])", "[[1, 0], [0, 1]]"),
    (
        "ROW_REDUCE([[2, 6, -3, -6], [4, 3, 3, 6], [4, -3, 9, k]])",
        "[[1, 0, 3/2, 0], [0, 1, -1, 0], [0, 0, 0, 1]]",
    ),
    (
        "ROW_REDUCE([[2, 3, 1], [1, 2, -1], [2, 1, 2]], IDENTITY_MATRIX(3))",
        "[[1, 0, 0, -1, 1, 1], [0, 1, 0, 4/5, -2/5, -3/5], [0, 0, 1, 3/5, -4/5, -1/5]]",
    ),
    (
        "ROW_REDUCE([[9, 3, 0, 0], [3, 5, 4, 0], [0, 4, 5, 3], [0, 0, 3, 9]])",
        "[[1, 0, 0, 1], [0, 1, 0, -3], [0, 0, 1, 3], [0, 0, 0, 0]]",
    ),
    (
        "[[0, 0, 1, 3], [0, 2, 2, 0], [3, 1, 0, 0]] * "
        "[[0, 0, 3], [0, 2, 1], [1, 2, 0], [3, 0, 0]]",
        "[[10, 2, 0], [2, 8, 2], [0, 2, 10]]",
    ),
    # The first row is the variables and the expressions, then a row per assignment.
    (
        "TRUTH_TABLE(a, b, NOT(a AND b), NOT(a) OR NOT(b))",
        "[[a, b, NOT (a AND b), NOT a OR NOT b], [true, true, false, false], "
        "[true, false, true, true], [false, true, true, true], "
        "[false, false, true, true]]",
    ),
    (
        "TRUTH_TABLE(a, b, a IFF b)",
        "[[a, b, a IFF b], [true, true, true], [true, false, false], "
        "[false, true, false], [false, false, true]]",
    ),
    (
        "TRUTH_TABLE(p, q, p AND q, p OR q, p XOR q, p IMP q, p IFF q)",
        "[[p, q, p AND q, p OR q, p XOR q, p IMP q, p IFF q], "
        "[true, true, true, true, false, true, true], "
        "[true, false, false, true, true, false, false], "
        "[false, true, false, true, true, true, false], "
        "[false, false, false, false, false, true, true]]",
    ),
]

#: Approximations, which read back as the exact rationals they print as, so their
#: answers are not fixed points of Simplify in exact precision.
DERIVE_6_APPROXIMATED_MATRICES = [
    (
        "APPROX(EIGENVALUES([[0, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 0, 0, 0], "
        "[0, 0, 1, 0, 0], [0, 0, 0, 1, 0]]), 10)",
        "[1.167303978, -0.7648844336 - 0.3524715460*#i, "
        "-0.7648844336 + 0.3524715460*#i, 0.1812324444 - 1.083954101*#i, "
        "0.1812324444 + 1.083954101*#i]",
    ),
    (
        "APPROX([[5, 3, -7], [2, -8, 1], [-1, 9, 4]]^(-1) . [4, 6, 5], 10)",
        "[2.910596026, 0.1754966887, 1.582781456]",
    ),
]


@pytest.mark.parametrize(("text", "expected"), DERIVE_6_MATRICES, ids=str)
def test_derive_6_matrices(text, expected):
    assert simp(text) == expected


@pytest.mark.parametrize(("text", "expected"), DERIVE_6_APPROXIMATED_MATRICES, ids=str)
def test_derive_6_approximated_matrices(text, expected):
    assert simp(text, Context()) == expected


@pytest.mark.parametrize(
    "matrix",
    [
        "[[2,3],[a,b]]",
        "[[1,2,3],[4,5,6],[7,8,10]]",
        # The symmetric matrix from the manual's last eigenvalue exercise.
        "[[-2,1,1,1,1],[1,-3,-1,0,-1],[1,-1,1,0,-3],[1,0,0,3,0],[1,-1,-3,0,2]]",
    ],
    ids=str,
)
def test_the_characteristic_polynomial_is_the_determinant_defining_it(matrix):
    # The manual defines CHARPOLY as `DET(A - z*I)`, so they must agree, including in
    # odd dimensions where it isn't monic. The difference is compared rather than the
    # texts, because CHARPOLY's terms are collected in the variable and the
    # determinant's are not.
    written_out = f"DET({matrix} - z*IDENTITY_MATRIX(DIMENSION({matrix})))"
    assert simp(f"CHARPOLY({matrix},z) - {written_out}") == "0"


#: Vector calculus: the manual's examples for each of the six operators, and their
#: accepted argument forms.
VECTOR_CALCULUS = [
    # The default coordinate system is 3D Cartesian in x, y, z, so a gradient always has
    # three elements by default.
    ("GRAD(x*y^2*z^3)", "[y^2*z^3, 2*x*y*z^3, 3*x*y^2*z^2]"),
    ("GRAD(x^2 + y^2)", "[2*x, 2*y, 0]"),
    # A vector of variables gives custom Cartesian coordinates, any number of them.
    ("GRAD(c*w + x^2 + y^3 + z^4, [w, x, y, z])", "[c, 2*x, 3*y^2, 4*z^3]"),
    ("GRAD(x*y, [x, y])", "[y, x]"),
    ("DIV([y^2*z^3, 2*x*y*z^3, 3*x*y^2*z^2])", "x*(6*y^2*z + 2*z^3)"),
    # Degree one in the main variable, so the normal form writes `x` times its
    # coefficient, keeping the numeric content, as the manual's answer does. Derive is
    # inconsistent here: it gives `x*(2*y + 6)` for `2*x*y + 6*x` but `2*x^2*(y + 3)`
    # when expanding `((x + 1)^2 + y)^2`, so no single form matches everything.
    # LAPLACIAN is DIV of GRAD, so the two agree.
    ("LAPLACIAN(x*y^2*z^3)", "x*(6*y^2*z + 2*z^3)"),
    ("CURL([y^2, 2*x*z, 0])", "[-2*x, 0, 2*z - 2*y]"),
    # The curl of a plane field is a number, not the vector `[0, 0, w]`.
    ("CURL([v^2, u], [[u, v], [1, 1]])", "1 - 2*v"),
    ("CURL([-y, x], [u, v])", "0"),
    # A two-row second argument is a coordinate geometry matrix: variables above, scale
    # factors below. This is the manual's spherical example, written out rather than
    # loaded from VECTOR.MTH. The manual's `COT(phi)*COS(theta)` comes out as
    # `COS(theta)*COT(phi)` in the engine's canonical form.
    (
        "GRAD(r*SIN(theta)*COS(phi), [[r, theta, phi], [1, r*SIN(phi), r]])",
        "[SIN(theta)*COS(phi), COS(theta)*COT(phi), -SIN(theta)*SIN(phi)]",
    ),
    # Cylindrical coordinates, where the scale factors change the answers: `DIV` of a
    # radial field grows with the circumference.
    ("DIV([r^2, 0, 0], [[r, theta, z], [1, r, 1]])", "3*r"),
    ("LAPLACIAN(r^2, [[r, theta, z], [1, r, 1]])", "4"),
    ("CURL([0, r, 0], [[r, theta, z], [1, r, 1]])", "[0, 0, 2]"),
    ("POTENTIAL([y^2*z^3, 2*x*y*z^3, 3*x*y^2*z^2])", "x*y^2*z^3"),
    ("VECTOR_POTENTIAL([x, 0, y - z])", "[-y^2/2, -x*z, 0]"),
    # The line integrals start at the origin by default. Starting at [1, 1, 1] instead
    # subtracts the potential's value there, an additive constant.
    ("POTENTIAL([2*x, 2*y, 2*z])", "x^2 + y^2 + z^2"),
    ("POTENTIAL([2*x, 2*y, 2*z], [1, 1, 1])", "x^2 + y^2 + z^2 - 3"),
    # The manual's warning about a bad starting point: this field is infinite at the
    # origin, so the potential from there is too; the manual recommends starting at 1
    # for a logarithm. Legs diverging in opposite directions have no sum.
    ("POTENTIAL([1/x, 1/y, 1/z], [1, 1, 1])", "LN(x) + LN(y) + LN(z)"),
    ("POTENTIAL([1/x, 1/y, 1/z])", "inf"),
    ("POTENTIAL([-1/x, 1/y, 0])", "POTENTIAL([-1/x, 1/y, 0])"),
    # A field with fewer elements than the system's coordinates (three by default) is
    # invalid. A two-element field needs an explicit planar system.
    ("CURL([-y, x])", "CURL([-y, x])"),
    ("DIV([x, y])", "DIV([x, y])"),
    ("CURL([-y, x], [[x, y], [1, 1]])", "2"),
    # Not every field has a potential, and Derive doesn't check. POTENTIAL computes one
    # line integral; `[-y, x, 0]` circulates, so the gradient of the result isn't the
    # field. The manual leaves the check to the user, and CURL is the easier test.
    ("CURL([-y, x, 0])", "[0, 0, 2]"),
    ("POTENTIAL([-y, x, 0])", "x*y"),
    ("GRAD(POTENTIAL([-y, x, 0]))", "[y, x, 0]"),
    # Likewise for a vector potential, where nonzero divergence is the obstruction: the
    # curl of the answer for `[x, 0, 0]` is not the field.
    ("DIV([x, 0, 0])", "1"),
    ("VECTOR_POTENTIAL([x, 0, 0])", "[0, -x*z, 0]"),
    ("CURL(VECTOR_POTENTIAL([x, 0, 0]))", "[x, 0, -z]"),
    # The gradients of a vector's elements are the columns of the answer, so row `i`
    # holds the derivatives along coordinate `i`. This is the transpose of VECTOR.MTH's
    # JACOBIAN, the orientation SOLVE.MTH's NEWTONS needs: `APPEND(GRAD(u, x), [u])``
    # is the augmented matrix of one Newton step.
    ("GRAD([x*y, x + y], [x, y])", "[[y, 1], [x, 1]]"),
    ("GRAD([1, 2])", "[[0, 0], [0, 0], [0, 0]]"),
    # Invalid arguments: a single variable is not a coordinate system, and curl and
    # vector potential exist only in 2D and 3D. Each call stays unevaluated.
    ("GRAD(u, x)", "GRAD(u, x)"),
    ("CURL([a, b, c, d])", "CURL([a, b, c, d])"),
    ("VECTOR_POTENTIAL([a, b])", "VECTOR_POTENTIAL([a, b])"),
    ("DIV([1, 2, 3, 4])", "DIV([1, 2, 3, 4])"),
]


@pytest.mark.parametrize(("text", "expected"), VECTOR_CALCULUS, ids=str)
def test_vector_calculus(text, expected):
    assert simp(text) == expected


@pytest.mark.parametrize(
    "field",
    ["[y^2*z^3, 2*x*y*z^3, 3*x*y^2*z^2]", "[1, 2*y, 3*z^2]", "[y*z, x*z, x*y]"],
    ids=str,
)
def test_a_conservative_field_is_the_gradient_of_its_potential(field):
    # The manual's check, on fields that pass it: POTENTIAL is a line integral and is a
    # true potential exactly when the field's curl is zero.
    assert simp(f"CURL({field})") == "[0, 0, 0]"
    assert simp(f"GRAD(POTENTIAL({field})) - {field}") == "[0, 0, 0]"


@pytest.mark.parametrize(
    "field",
    ["[-2*x, 0, 2*z - 2*y]", "[x, 0, y - z]", "[y, z, x]"],
    ids=str,
)
def test_a_solenoidal_field_is_the_curl_of_its_vector_potential(field):
    # The same check for VECTOR_POTENTIAL, whose condition is zero divergence. Valid
    # vector potentials differ by a gradient, so the curl is compared, not the vector.
    assert simp(f"DIV({field})") == "0"
    assert simp(f"CURL(VECTOR_POTENTIAL({field})) - {field}") == "[0, 0, 0]"


#: The four ways to specify the values of a generated vector's variable. The first
#: three are the manual's examples.
GENERATED = [
    ("VECTOR(x^2, x, 5)", "[1, 4, 9, 16, 25]"),
    ("VECTOR(j!, j, 0, 4)", "[1, 1, 2, 6, 24]"),
    ("VECTOR(k^2, k, [2, 3, 5, 7, 11])", "[4, 9, 25, 49, 121]"),
    # With a step, the count rounds down: four elements, the last short of pi/4. Exact;
    # the manual shows the approximated decimals.
    ("VECTOR(SIN(z), z, 0, pi/4, 1/5)", "[0, SIN(1/5), SIN(2/5), SIN(3/5)]"),
    # A nested call generates a matrix.
    (
        "VECTOR(VECTOR(j + k, k, 1, 4), j, 1, 3)",
        "[[2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]]",
    ),
    # The variable ranges over a matrix's rows, as with any element access.
    ("VECTOR(k, k, [[1, 2], [3, 4]])", "[[1, 2], [3, 4]]"),
    # A vector of variables runs over a grid, the first fastest.
    (
        "VECTOR(a + 10*b, [a, b], [0, 0], [1, 4], [1, 2])",
        "[[0, 1], [20, 21], [40, 41]]",
    ),
    # A range shorter than the step has one element; a reversed range has none.
    ("VECTOR(x^2, x, 3, 3)", "[9]"),
    ("VECTOR(x, x, 5, 1)", "[]"),
    # Symbolic bounds, a non-variable second argument, or a zero step: the call stays.
    ("VECTOR(x^2, x, n)", "VECTOR(x^2, x, n)"),
    ("VECTOR(x^2, 5, 1, 3)", "VECTOR(x^2, 5, 1, 3)"),
    ("VECTOR(x^2, x, 1, 3, 0)", "VECTOR(x^2, x, 1, 3, 0)"),
]


@pytest.mark.parametrize(("text", "expected"), GENERATED, ids=str)
def test_generated_vectors(text, expected):
    assert simp(text) == expected


#: `TABLE` takes its values as `VECTOR` does and makes a row of each value and the entry
#: there, or the entry's elements if it is a vector.
TABLES = [
    ("TABLE(x^2, x, 5)", "[[1, 1], [2, 4], [3, 9], [4, 16], [5, 25]]"),
    (
        "TABLE([x^2, x^3, x^4], x, 5)",
        "[[1, 1, 1, 1], [2, 4, 8, 16], [3, 9, 27, 81], [4, 16, 64, 256], "
        "[5, 25, 125, 625]]",
    ),
    ("TABLE(j!, j, 0, 4)", "[[0, 1], [1, 1], [2, 2], [3, 6], [4, 24]]"),
    (
        "TABLE([SIN(z), COS(z)], z, 0, pi/4, 0.2)",
        "[[0, 0, 1], [1/5, SIN(1/5), COS(1/5)], [2/5, SIN(2/5), COS(2/5)], "
        "[3/5, SIN(3/5), COS(3/5)]]",
    ),
    ("TABLE(k^2, k, [2, 3, 5, 7, 11])", "[[2, 4], [3, 9], [5, 25], [7, 49], [11, 121]]"),
    # Over a set, the rows follow its order.
    ("TABLE(k^2, k, {3, 2})", "[[2, 4], [3, 9]]"),
    ("TABLE([k, k^2], k, 2, 3)", "[[2, 2, 4], [3, 3, 9]]"),
    # Elements of the entry that are vectors stay elements of the row.
    ("TABLE([[1], k], k, 1, 2)", "[[1, [1], 1], [2, [1], 2]]"),
    ("TABLE(1/k, k, 10, 30, 10)", "[[10, 1/10], [20, 1/20], [30, 1/30]]"),
    ("(TABLE(k, k, 3))`", "[[1, 2, 3], [1, 2, 3]]"),
    (
        "(TABLE(n^2/2 - n/2, n, 0, 10)) SUB SUB 2",
        "[0, 0, 1, 3, 6, 10, 15, 21, 28, 36, 45]",
    ),
    ("MAX((TABLE(k*(4 - k), k, 1, 3)) SUB SUB 2)", "4"),
    # An entry that is a vector only once evaluated, as a call of a function returning
    # one, is the second column: Derive spreads only a vector written as such.
    (
        "TABLE(VECTOR(k, k, 0, n), n, 0, 2)",
        "[[0, [0]], [1, [0, 1]], [2, [0, 1, 2]]]",
    ),
    ("TABLE(x^2)", "TABLE(x^2)"),
    ("TABLE(x^2, x)", "TABLE(x^2, x)"),
    ("TABLE(x^2, x, n)", "TABLE(x^2, x, n)"),
]


@pytest.mark.parametrize(("text", "expected"), TABLES, ids=str)
def test_tables(text, expected):
    assert simp(text) == expected


def test_a_table_approximated():
    # The decimals shown are not what they read back as, so this is no fixed point.
    shown = simp("APPROX(TABLE([SIN(z), COS(z)], z, 0, pi/4, 0.2), 10)")
    assert shown == (
        "[[0, 0, 1], [0.2, 0.1986693307, 0.9800665778], "
        "[0.4, 0.3894183423, 0.9210609940], [0.6, 0.5646424733, 0.8253356149]]"
    )


#: `SELECT` takes values in the same four ways, and returns the values that pass the
#: test.
SELECTED = [
    # The manual's two examples.
    ("SELECT(PRIME(k), k, [3, 5, 7, 9, 11])", "[3, 5, 7, 11]"),
    ("SELECT(PRIME(k), k, 1, 20)", "[2, 3, 5, 7, 11, 13, 17, 19]"),
    # A count, and a range with a step.
    ("SELECT(k > 1, k, 5)", "[2, 3, 4, 5]"),
    ("SELECT(k > 1, k, 1, 10, 3)", "[4, 7, 10]"),
    ("SELECT(k^2 > 5, k, 1, 5)", "[3, 4, 5]"),
    ("SELECT(k > 5, k, 1, 3)", "[]"),
    # A non-variable second argument leaves the call as is.
    ("SELECT(k > 1, 5, [1, 2])", "SELECT(k > 1, 5, [1, 2])"),
    # As in Derive, an element whose test is undecided is dropped. ODE.MTH's `FREE`
    # relies on this to find a variable among `VARIABLES(u)`.
    ("SELECT(PRIME(k), k, x, x + 2)", "[]"),
    ("SELECT(k = y, k, [x, y])", "[y]"),
    # A test that isn't a truth value is not true.
    ("SELECT(k - 1, k, 1, 3)", "[]"),
    # A vector of numbers is not a number.
    ("SELECT(k /= 0, k, [0, [1, 1], 0])", "[[1, 1]]"),
]


def test_a_shown_definition_keeps_a_selection_its_parameters_decide():
    """The test is on a parameter, so a call decides it; nothing is dropped yet."""
    assert simp("F(n) := SELECT(k = n, k, [1, 2, 3])") == (
        "F(n) := SELECT(k = n, k, [1, 2, 3])"
    )


@pytest.mark.parametrize(("text", "expected"), SELECTED, ids=str)
def test_selected_elements(text, expected):
    assert simp(text) == expected


#: The two iteration functions, which differ only in returning the whole sequence or
#: just the last value.
ITERATION = [
    # `n` updates make `n + 1` iterates; ITERATE keeps the last of them.
    ("ITERATES(x^2, x, 2, 3)", "[2, 4, 16, 256]"),
    ("ITERATE(x^2, x, 2, 3)", "256"),
    ("ITERATES(x^2, x, 2, 0)", "[2]"),
    # The manual's POWER: `n` copies of `x` multiplied into an accumulator.
    ("ITERATE(a*x, a, 1, 3)", "x^3"),
    # Without a count, iteration runs until a value repeats, ending with the repeat.
    # ITERATE expects a fixed point and answers `?` for a longer cycle.
    ("ITERATES(1/x, x, 2)", "[2, 1/2, 2]"),
    ("ITERATE(1/x, x, 2)", "?"),
    # Vectors of variables and starting values: the manual's Fibonacci without
    # subscripts.
    ("ITERATE([k, j + k], [j, k], [0, 1], 10)", "[55, 89]"),
    ("ITERATES([k, j + k], [j, k], [0, 1], 3)", "[[0, 1], [1, 1], [1, 2], [2, 3]]"),
    # A negative count iterates the inverse, which MISC.MTH relies on to define
    # `INVERSE(u, x) := ITERATE(u, x, x, -1)`.
    ("ITERATES(TAN(x), x, x, -1)", "[x, ATAN(x)]"),
    ("ITERATES(x + 1, x, 0, -2)", "[0, -1, -2]"),
    # As in Derive, the principal inverse is taken: a power is undone by the reciprocal
    # power, a call by its inverse function. MISC.MTH's INT_SUBST depends on this (the
    # manual inverts `t^2` to `SQRT(t)`, 9.21, p.285).
    ("ITERATES(x^2, x, 2, -1)", "[2, SQRT(2)]"),
    # An inverse the notation has no name for (sympy's inverse error function) is no
    # answer: the iteration stays as written, which reads back. ODE.MTH's INVERT
    # relies on this to see that there is no inverse.
    ("ITERATE(ERF(x), x, x, -1)", "ITERATE(ERF(x), x, x, -1)"),
    # Of several solutions, the one without a sign or an added constant is principal.
    # ODE.MTH's CHANGE_X inverts this to change the variable of `2*SIN(x)^2*y'' +
    # SIN(2*x)*y' + 8*y = 0`.
    (
        "ITERATE(LN(1 - COS(x)) - LN(COS(x) + 1), x, x, -1)",
        "ACOS((#e^(-x/2) - #e^(x/2))/(#e^(x/2) + #e^(-x/2)))",
    ),
    # A count that isn't an integer is truncated toward zero, as in Derive (D).
    # ORTH_POL.MTH counts `n/2` steps, which for an odd order is a half.
    ("ITERATES(x + 1, x, 0, 5/2)", "[0, 1, 2]"),
    ("ITERATES(x + 1, x, 0, -5/2)", "[0, -1, -2]"),
    ("ITERATE(x + 1, x, 0, 5/2)", "2"),
    # A count that isn't a number runs the iteration as if it were omitted, as the
    # manual says and Derive does: SOLVE.MTH's `NEWTON` called without its count.
    ("ITERATES(1/x, x, 2, n)", "[2, 1/2, 2]"),
    ("ITERATE(1/x, x, 2, n)", "?"),
    # Exact numbers may take hundreds of steps to settle.
    ("ITERATE(FLOOR(x, 2), x, 10^100)", "0"),
    # So may vectors of them: a binary power `[result, square, exponent]` halves the
    # exponent at each step.
    ("ITERATE([IF(MOD(v SUB 3, 2) = 1, v SUB 1*v SUB 2, v SUB 1), v SUB 2^2, "
     "FLOOR(v SUB 3, 2)], v, [1, 1, 2^200]) SUB 3", "0"),
    # An uncounted iteration whose iterates outgrow what can be iterated on, in size
    # or in digits (Newton's method for SQRT(3), exactly), stays unevaluated; Derive
    # runs until memory runs out.
    ("ITERATES(x^2, x, 2)", "ITERATES(x^2, x, 2)"),
    ("ITERATES((x^2 + 3)/(2*x), x, 2, n)", "ITERATES((x^2 + 3)/(2*x), x, 2, n)"),
    # So does a count that is an expression other than a variable.
    ("ITERATES(x + 1, x, 0, n + 1)", "ITERATES(x + 1, x, 0, n + 1)"),
    # An infinite count is no count.
    ("ITERATES(1/x, x, 2, inf)", "[2, 1/2, 2]"),
]


@pytest.mark.parametrize(("text", "expected"), ITERATION, ids=str)
def test_iteration(text, expected):
    assert simp(text) == expected


@pytest.mark.parametrize(
    "text",
    ["ITERATE(a*x, a, 1, n)", "ITERATES(x + 1, x, 0, n)", "ITERATES(k + 1, k, 0)"],
)
def test_an_uncounted_iteration_that_never_repeats_runs_until_stopped(text):
    # As in Derive, there is no limit on the number of steps; here the step limit
    # stops it.
    with pytest.raises(StepLimitExceeded):
        simp(text, Context(step_limit=1000))


def test_an_element_access_that_could_not_be_taken_is_taken_once_generated():
    # An access with a symbolic index converts to an inert head. Once each element is
    # generated the index is a number, so the access is retried; the demo's outer
    # product depends on this.
    assert simp("VECTOR([[a, b, c] SUB i], i, 3)") == "[[a], [b], [c]]"
    assert simp("VECTOR([a, b, c] SUB i + 1, i, 2)") == "[a + 1, b + 1]"


#: A calculus head over the variable of a counted call. As in Derive, the head is
#: evaluated with the variable symbolic, and only then is each value substituted.
COUNTED_CALCULUS = [
    ("VECTOR(DIF(x^2, x), x, 1, 3)", "[2, 4, 6]"),
    ("VECTOR(INT(x, x), x, 1, 3)", "[1/2, 2, 9/2]"),
    ("SUM(DIF(x^2, x), x, 1, 3)", "12"),
    ("PRODUCT(DIF(x^2, x), x, 1, 3)", "48"),
    # Newton's method for SQRT(3): SOLVE.MTH's `NEWTON` written out.
    ("ITERATES(x - (x^2 - 3)/DIF(x^2 - 3, x), x, 2, 3)", "[2, 7/4, 97/56, 18817/10864]"),
    # A head over another variable is unaffected.
    ("VECTOR(DIF(x^2*y, x), y, 1, 2)", "[2*x, 4*x]"),
]


@pytest.mark.parametrize(("text", "expected"), COUNTED_CALCULUS, ids=str)
def test_a_calculus_head_in_a_counted_call_is_worked_out_before_counting(
    text, expected
):
    assert simp(text) == expected


def test_a_parameter_a_counted_call_differentiates_is_taken_as_supplied():
    # SOLVE.MTH's `NEWTON(u, x, x0, n)`: the iteration substitutes values for `x`, and
    # `DIF(u, x)` differentiates the supplied `u` with respect to `x`.
    newton = Definition(("u", "x", "a", "n"), parse("ITERATES(x - u/DIF(u, x), x, a, n)"))
    context = Context(functions={"NEWTON": newton})
    assert simp("NEWTON(x^2 - 3, x, 2, 3)", context) == "[2, 7/4, 97/56, 18817/10864]"
    assert simp("NEWTON(t^2 - 3, t, 2, 3)", context) == "[2, 7/4, 97/56, 18817/10864]"


def test_a_function_iterated_over_its_own_variable_is_applied_once_per_step():
    # The argument is evaluated with the iterate substituted, and the parameter stands
    # for that value: it isn't substituted into a second time.
    context = Context(functions={"F": Definition(("x",), parse("x^2"))})
    assert simp("ITERATES(F(x), x, x, 3)", context) == "[x, x^2, x^4, x^8]"


# -- statistics ---------------------------------------------------------------

#: The statistical functions' three argument forms (separate arguments, a vector, or a
#: matrix with one sample per row), and the manual's convention: variance is the
#: unbiased sample variance, over `n - 1`, and standard deviation is its square root.
STATISTICS = [
    # The manual's pair: the same sample written both ways.
    ("AVERAGE(2, 3, 4)", "3"),
    ("AVERAGE([2, 3, 4])", "3"),
    ("RMS([2, 3, 5])", "SQRT(114)/3"),
    # The manual's matrix example: a vector of per-row results.
    ("RMS([[3, 4, 5], [5, 12, 13]])", "[5*SQRT(6)/3, 13*SQRT(6)/3]"),
    ("AVERAGE([[1, 2], [3, 4]])", "[3/2, 7/2]"),
    # One sample through all four functions pins the convention: over `n - 1` the
    # variance is 20/3; over `n` it would be 5.
    ("AVERAGE(2, 4, 6, 8)", "5"),
    ("RMS(2, 4, 6, 8)", "SQRT(30)"),
    ("VAR(2, 4, 6, 8)", "20/3"),
    ("STDEV(2, 4, 6, 8)", "2*SQRT(15)/3"),
    # A single value is a valid sample: its mean is itself, its RMS its magnitude.
    ("AVERAGE(x)", "x"),
    ("RMS(x)", "ABS(x)"),
    # An unevaluated call is not a sample. Treating it as a sample of one would give
    # `ABS` of a vector not yet built, breaking definitions containing
    # `RMS(VECTOR(...))`.
    ("RMS(VECTOR(u, k, n))", "RMS(VECTOR(u, k, n))"),
    # A sample of one has zero deviation; like Derive, answer 0 rather than the
    # formula's `0/0`.
    ("VAR([5])", "0"),
    ("VAR(x)", "0"),
    ("STDEV([5])", "0"),
]


@pytest.mark.parametrize(("text", "expected"), STATISTICS, ids=str)
def test_statistics(text, expected):
    assert simp(text) == expected


# -- what an expression is made of, and a vector rebuilt ----------------------

STRUCTURE = [
    # Terms are syntactic, so the power isn't multiplied out: the manual's example, and
    # why it suggests composing with EXPAND.
    ("TERMS(x*(a + b)^2 + c)", "[x*(a + b)^2, c]"),
    ("TERMS(x^2 + x + 1)", "[x^2, x, 1]"),
    # In printed order: the main variable's higher powers first, whatever the degree.
    ("TERMS(x^5*y^3 - x^3*y^6)", "[x^5*y^3, -x^3*y^6]"),
    ("TERMS(y + x^2*y^5)", "[x^2*y^5, y]"),
    # Expanded in a variable, the terms come by its powers, highest first.
    ("TERMS(t*(x*y - t^3), t)", "[-t^4, t*x*y]"),
    ("TERMS(EXPAND(t*(x*y - t^3), t))", "[-t^4, t*x*y]"),
    ("TERMS(EXPAND(t*(x*y - t^3)))", "[t*x*y, -t^4]"),
    # A non-sum has one term, and a vector distributes.
    ("TERMS(x*y)", "[x*y]"),
    ("TERMS([x + 1, y + 2])", "[[x, 1], [y, 2]]"),
    # Zero is the empty sum, so a sum over its terms is zero again.
    ("TERMS(0)", "[]"),
    # With an amount, the argument is expanded first.
    ("TERMS(x*(a + b)^2 + c, Trivial)", "[a^2*x, 2*a*b*x, b^2*x, c]"),
    ("DIM(TERMS(EXPAND((x + 1)^3)))", "4"),
    # The argument is taken apart as Simplify gives it, its like terms collected, while
    # an expansion asked for stays expanded: a sum built with `SUM` has one term per
    # power.
    ("TERMS(x*(2*y + z) + x*(3*y + z) + x^2*z)", "[x^2*z, x*(5*y + 2*z)]"),
    ("TERMS(EXPAND(x*(2*y + z) + x*(3*y + z) + x^2*z))", "[x^2*z, 5*x*y, 2*x*z]"),
    ("DIM(TERMS(SUM(x^k*(k*y + z), k, 1, 3) + SUM(x^k*(y + k*z), k, 1, 3)))", "3"),
    ("TERMS(FACTOR(x^2 - 1))", "[(x + 1)*(x - 1)]"),
    # A logical formula's terms are the disjuncts of its disjunctive normal form; a
    # relation is one term.
    ("TERMS(p AND (q OR r))", "[p AND q, p AND r]"),
    ("TERMS(FACTORS(NOT(NOT(a) OR b) OR NOT(c)))", "[[a, NOT c], [NOT b, NOT c]]"),
    ("[TERMS(p OR q), TERMS(p AND q)]", "[[p, q], [p AND q]]"),
    ("TERMS(x = 1)", "[x = 1]"),
    ("TERMS({x + 1})", "{[x, 1]}"),
    # Most main first: `x`, `y`, `z` from the order list, then the rest alphabetically,
    # the same order other commands use.
    ("VARIABLES(x^2 + a*y)", "[x, y, a]"),
    ("VARIABLES(a*b*z)", "[z, a, b]"),
    ("VARIABLES(5)", "[]"),
    ("VARIABLES(a = 3 AND b = 7)", "[a, b]"),
    ("VARIABLES(2 + pi)", "[]"),
    # Strings follow the variables, in code-point order.
    ('VARIABLES(["∫", x, "a"])', '[x, "a", "∫"]'),
    ('MEMBER?("∫", VARIABLES(STRING(INT(1/SQRT(1 - x^4), x))))', "true"),
    # The argument is simplified first, as in Derive: this one cancels to `a^2`.
    # ODE.MTH's SYMMETRIC_TEST relies on this.
    ("VARIABLES((a^2*y^2 + a^2*y*z)/(y^2 + y*z))", "[a]"),
    # A quotient in `x` and `SQRT(y)` cancels; ODE.MTH's HOMOGENEOUS_4 asks whether
    # this exponent is free of `x` and `y`. A standing integral is held as it is.
    ("VARIABLES(x*(-(x + SQRT(y))/y - x/y)/(y*(x*(x + SQRT(y))/y^2 - x/(2*y^(3/2)))))",
     "[]"),
    ("VARIABLES(SIN(x)^2 + COS(x)^2 + y)", "[y]"),
    ("VARIABLES(INT(SIN(x^3)/LN(x), x) + (a*y + y)/(a + 1))", "[x, y]"),
    # A term an infinity absorbs keeps its variables, as in Derive; one that vanishes
    # otherwise does not.
    ("VARIABLES(a + inf)", "[a]"),
    ("VARIABLES(b - inf + 2*a)", "[a, b]"),
    ("VARIABLES([y + inf, x])", "[x, y]"),
    ("VARIABLES(a - a + 0*b + inf)", "[]"),
    # NUMBER is a predicate and always answers a truth value. The manual's example is
    # the second pair: true exactly when the sum of two squares is a perfect square.
    ("NUMBER(2/3)", "true"),
    ("NUMBER(x)", "false"),
    ("NUMBER(SQRT(9))", "true"),
    ("NUMBER(SQRT(2))", "false"),
    # The utility files use this to detect an argument that wasn't supplied.
    ("IF(NUMBER(d), d, 1)", "1"),
    # Indices count from 1, and a matrix's elements are its rows.
    ("DELETE_ELEMENT([a, b, c], 2)", "[a, c]"),
    ("DELETE_ELEMENT([[1, 2], [3, 4], [5, 6]], 2)", "[[1, 2], [5, 6]]"),
    ("DELETE_ELEMENT([a], 1)", "[]"),
    # The new value comes first, then the vector; the index defaults to 1.
    ("REPLACE_ELEMENT(d, [a, b, c], 2)", "[a, d, c]"),
    ("REPLACE_ELEMENT(d, [a, b, c])", "[d, b, c]"),
    ("REPLACE_ELEMENT([7, 8], [[1, 2], [3, 4]], 1)", "[[7, 8], [3, 4]]"),
    # A symbolic index leaves the call as is; one past the end deletes nothing.
    ("DELETE_ELEMENT([a, b, c], n)", "DELETE([a, b, c], n)"),
    ("DELETE_ELEMENT([a, b, c], 4)", "[a, b, c]"),
    # Inserts before the nth element; the index defaults to 1. One past the end is valid
    # here, unlike above, since that is how to append.
    ("INSERT_ELEMENT(d, [a, b, c], 2)", "[a, d, b, c]"),
    ("INSERT_ELEMENT(d, [a, b, c])", "[d, a, b, c]"),
    ("INSERT_ELEMENT(d, [a, b, c], 4)", "[a, b, c, d]"),
    # Two past the end is invalid and the call stays, as with DELETE_ELEMENT. Derive
    # answers `[a, b, c]`, silently dropping the element.
    ("INSERT_ELEMENT(d, [a, b, c], 5)", "INSERT(d, [a, b, c], 5)"),
    # A matrix is reversed by rows.
    ("REVERSE_VECTOR([a, b, c])", "[c, b, a]"),
    ("REVERSE_VECTOR([[1, 2], [3, 4]])", "[[3, 4], [1, 2]]"),
    # Sorting, which NUMBER.MTH's FAREY and DIVISORS need: numbers by value, then
    # variables alphabetically (the 6.1 help's example), vectors element by element.
    ("SORT([3, 1, 2])", "[1, 2, 3]"),
    (
        "SORT([3, 2, y, pi, #e, SQRT(2), z, x, 1, 0, 2, 0, 0, -3])",
        "[-3, 0, 0, 0, 1, SQRT(2), 2, 2, #e, 3, pi, x, y, z]",
    ),
    ("SORT([[2, 3], [3, 1], [2, 1], [1, 5]])", "[[1, 5], [2, 1], [2, 3], [3, 1]]"),
    ("SORT([#i, 1, -#i, 0])", "[-#i, 0, #i, 1]"),
    ("SORT([])", "[]"),
    # Numbers compare exactly, however close: fractions directly, other constants by the
    # sign of their difference.
    ("SORT([1/2, -3/4, 1/3, 2/5, 1/3])", "[-3/4, 1/3, 1/3, 2/5, 1/2]"),
    (
        "SORT([10^40 + 1, 10^40, 10^40 - 1])",
        "[9999999999999999999999999999999999999999, "
        "10000000000000000000000000000000000000000, "
        "10000000000000000000000000000000000000001]",
    ),
    (
        "SORT([1/3 + 10^-40, 1/3, 1/3 - 10^-40])",
        "[9999999999999999999999999999999999999997"
        "/30000000000000000000000000000000000000000, 1/3, "
        "10000000000000000000000000000000000000003"
        "/30000000000000000000000000000000000000000]",
    ),
    ("SORT([pi + 10^-40, pi])", "[pi, pi + 1/10000000000000000000000000000000000000000]"),
    (
        "SORT([pi, 22/7, 355/113, SQRT(10), #e^2/SQRT(5), LN(23)])",
        "[LN(23), pi, 355/113, 22/7, SQRT(10), SQRT(5)*#e^2/5]",
    ),
    ("SORT([SQRT(2)*SQRT(3), SQRT(6), 2])", "[2, SQRT(6), SQRT(6)]"),
    ("SORT([1 + SQRT(2)*#i, 1 + #i, #i])", "[#i, 1 + #i, 1 + SQRT(2)*#i]"),
    ("SORT([[2, pi], [2, 3], [1]])", "[[1], [2, 3], [2, pi]]"),
    # Expressions other than numbers and variables come after both. A pair of numbers
    # whose order can't be decided exactly (these two are equal) leaves the call as
    # written.
    ("SORT([x + 1, 2])", "[2, x + 1]"),
    ("SORT([ATAN(1/2) + ATAN(1/3), pi/4])", "SORT([ATAN(1/2) + ATAN(1/3), pi/4])"),
    # Concatenation. Matrices stack by rows, except that a single matrix is flattened,
    # as the manual specifies.
    ("APPEND([a, b], [c, d], [e, f])", "[a, b, c, d, e, f]"),
    ("APPEND([[a, b], [c, d], [e, f]])", "[a, b, c, d, e, f]"),
    ("APPEND([[a, b], [c, d]], [[e, f], [g, h]])", "[[a, b], [c, d], [e, f], [g, h]]"),
    ("APPEND([a, b])", "[a, b]"),
    # Elements are concatenated whatever they are, so a matrix and a vector give a
    # vector of rows and then elements. A row is appended as a one-row matrix, the way
    # `NEWTONS` appends its system below its gradients.
    ("APPEND([[a, b], [c, d]], [e, f])", "[[a, b], [c, d], e, f]"),
    ("APPEND([[a, b], [c, d]], [[e, f]])", "[[a, b], [c, d], [e, f]]"),
    ("APPEND([e, f], [[a, b], [c, d]])", "[e, f, [a, b], [c, d]]"),
    # A one-row matrix is not a vector, and elements needn't be numbers.
    ("APPEND([[1, 2]], [[3, 4]])", "[[1, 2], [3, 4]]"),
    ("APPEND([x = 1], [y = 2])", "[x = 1, y = 2]"),
    ("APPEND([], [[[1, 2, 3] = x]])", "[[[1, 2, 3] = x]]"),
    ("APPEND([[1, 3], [1, 2]], [VECTOR(k, k, 1, 2)])", "[[1, 3], [1, 2], [1, 2]]"),
    # A single vector of vectors loses one level, whatever its rank.
    ("APPEND([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])", "[[1, 2], [3, 4], [5, 6], [7, 8]]"),
]


@pytest.mark.parametrize(("text", "expected"), STRUCTURE, ids=str)
def test_structure(text, expected):
    assert simp(text) == expected


#: Section 6.14's other decomposition functions: syntactic factors, the sides of a
#: relation, and polynomial division and gcd.
DECOMPOSITION = [
    # Factors as base and exponent pairs, read off the product as it prints (sums last,
    # in reverse), nothing factored first: a sum is one factor, a coefficient the first
    # pair.
    ("FACTORS(3*x*(x + 1)^2)", "[[3, 1], [x, 1], [x + 1, 2]]"),
    ("FACTORS(x^2 - 1)", "[[x^2 - 1, 1]]"),
    ("FACTORS(x/y)", "[[x, 1], [y, -1]]"),
    ("FACTORS(6*x^2/y)", "[[6, 1], [x, 2], [y, -1]]"),
    # Exponentials come first, then other functions, then the rest.
    (
        "FACTORS(4*t^2*#e^(3*t)*SIN(4*t))",
        "[[4, 1], [#e^(3*t), 1], [SIN(4*t), 1], [t, 2]]",
    ),
    ("FACTORS(-x)", "[[-1, 1], [x, 1]]"),
    ("FACTORS(SQRT(x))", "[[x, 1/2]]"),
    ("FACTORS(x^n)", "[[x, n]]"),
    ("FACTORS(x + 1)", "[[x + 1, 1]]"),
    ("FACTORS(SQRT(2))", "[[2, 1/2]]"),
    # A rational number is factored into primes, the denominator's with negative
    # exponents. One is the empty product.
    ("FACTORS(10!)", "[[2, 8], [3, 4], [5, 2], [7, 1]]"),
    ("FACTORS(360)", "[[2, 3], [3, 2], [5, 1]]"),
    ("FACTORS(1000)", "[[2, 3], [5, 3]]"),
    ("FACTORS(-12)", "[[-1, 1], [2, 2], [3, 1]]"),
    ("FACTORS(12/5)", "[[2, 2], [3, 1], [5, -1]]"),
    ("FACTORS(1)", "[]"),
    ("FACTORS(0)", "[[0, 1]]"),
    ("FACTORS(FACTOR(360))", "[[2, 3], [3, 2], [5, 1]]"),
    # The factors of the argument as Simplify gives it, unless it was expanded.
    ("FACTORS(x*y + x*z)", "[[x, 1], [y + z, 1]]"),
    ("DIM(FACTORS(EXPAND(x*(y + z))))", "1"),
    ("(FACTORS(360)) SUB SUB 2", "[3, 2, 1]"),
    ("PRODUCT(v_ SUB 2 + 1, v_, FACTORS(360))", "24"),
    ("PRODUCT(v SUB 1^v SUB 2, v, FACTORS(3*x^2/y))", "3*x^2/y"),
    # With an amount, the argument is factored first. The Derive 6.1 help's examples.
    ("FACTORS((x^2 - 1)*(x + 3)^2)", "[[x^2 - 1, 1], [x + 3, 2]]"),
    ("FACTORS((x^2 - 1)*(x + 3)^2, Rational)", "[[x + 3, 2], [x - 1, 1], [x + 1, 1]]"),
    ("FACTORS(4*x^2 + 12*x*y + 5*y^2, Rational)", "[[2*x + 5*y, 1], [2*x + y, 1]]"),
    # A logical formula gives the clauses of its conjunctive normal form.
    ("FACTORS(p OR (q AND r))", "[p OR q, p OR r]"),
    ("FACTORS(TERMS(NOT(NOT(a) OR b) OR NOT(c)))", "[[a, NOT b], [NOT c]]"),
    # Vectors and sets distribute.
    ("FACTORS([12, 9])", "[[[2, 2], [3, 1]], [[3, 2]]]"),
    ("FACTORS([x*y, z^2])", "[[[x, 1], [y, 1]], [[z, 2]]]"),
    # Sides of a relation, distributing over a vector, so `RHS(SOLVE([u], x))` gives
    # the roots.
    ("LHS(2*x + 3 = 5)", "2*x + 3"),
    ("RHS(2*x + 3 = 5)", "5"),
    ("LHS([x = 1, y = 2])", "[x, y]"),
    ("RHS([x = 1, y = 2])", "[1, 2]"),
    # A non-relation `u` reads as `u = 0`, as in Derive.
    ("LHS(2*x + 3)", "2*x + 3"),
    ("RHS(3)", "0"),
    ("RHS(x + y)", "0"),
    # The operands of a logical or set operator; more than two read as the first and
    # the rest, in printed order. A negation has no sides.
    (
        "LHS([2*x + y = 5, x < y - 3, NOT p AND q, a UNION b])",
        "[2*x + y, x, NOT p, a]",
    ),
    ("RHS([2*x + y = 5, x < y - 3, NOT p AND q, a UNION b])", "[5, y - 3, q, b]"),
    ("RHS(p = 1 AND q = 3/4 AND s = 4)", "q = 3/4 AND s = 4"),
    ("RHS(RHS(RHS(p = 1 AND q = 3/4 AND s = 4)))", "4"),
    ("[LHS(a AND b), LHS(p IMP q), RHS(p IMP q)]", "[a, p, q]"),
    ("LHS(NOT p)", "LHS(NOT p)"),
    # A relation that is the argument is taken apart, not decided.
    ("RHS(x = x + 1)", "x + 1"),
    ("RHS(2*x + 3 < 5)", "5"),
    # Division in the main variable: the manual's pair.
    ("QUOTIENT(x^4 + 3*x^3 + 5*x + 6, x^2 - 5)", "x^2 + 3*x + 5"),
    ("REMAINDER(x^4 + 3*x^3 + 5*x + 6, x^2 - 5)", "20*x + 31"),
    ("QUOTIENT(x, x^2)", "0"),
    ("REMAINDER(x, x^2)", "x"),
    # Other variables act as field coefficients: in `x` the divisor is a constant, which
    # divides exactly.
    ("QUOTIENT(x*y + 1, y)", "x + 1/y"),
    ("REMAINDER(x*y + 1, y)", "0"),
    # Numbers form a field, so the quotient is the fraction and the remainder zero.
    ("QUOTIENT(7, 2)", "7/2"),
    ("REMAINDER(7, 2)", "0"),
    ("QUOTIENT(-7, 2)", "-7/2"),
    # A polynomial in a kernel is divided in that kernel.
    ("QUOTIENT(SIN(x), SIN(x)^2)", "0"),
    ("REMAINDER(SIN(x), SIN(x)^2)", "SIN(x)"),
    ("QUOTIENT(SIN(x), x)", "SIN(x)/x"),
    # A third argument names the variable to divide in.
    ("QUOTIENT(x^2 + y, x + y, y)", "1"),
    ("REMAINDER(x^2 + y, x + y, y)", "x^2 - x"),
    # A quotient that cancels to a polynomial divides as that polynomial.
    ("QUOTIENT(z^3, (z^2 - 1)/(z + 1))", "z^2 + z + 1"),
    ("REMAINDER(z^3, (z^2 - 1)/(z + 1))", "1"),
    (
        "REMAINDER(z^16 - z^15 + z^14 - z^13, (z^17 + 1)/(z + 1))",
        "-z^12 + z^11 - z^10 + z^9 - z^8 + z^7 - z^6 + z^5 - z^4 + z^3 - z^2 + z - 1",
    ),
    # The manual's gcd example, and gcd of numbers.
    ("POLY_GCD(x^3 + 3*x^2 + 5*x + 6, x^3 + 2*x - 3)", "x^2 + x + 3"),
    ("POLY_GCD(12, 18)", "6"),
    # Powers whose exponents differ by an integer have a common factor.
    ("POLY_GCD(x^(n + 1), x^n)", "x^n"),
    (
        "POLY_GCD(x^4 - 7*x^3 + 18*x^2 - 22*x + 12, 3*x^3 - 13*x^2 + 8*x + 12)",
        "x^2 - 5*x + 6",
    ),
    ("QUOTIENT(0, x + 1)", "0"),
    # Complex coefficients divide like any others.
    ("QUOTIENT(x^2 + 4, x + 2*#i)", "x - 2*#i"),
    ("QUOTIENT(a*x^2 + b, x - c, x)", "a*x + a*c"),
]


@pytest.mark.parametrize(("text", "expected"), DECOMPOSITION, ids=str)
def test_what_an_expression_decomposes_into(text, expected):
    assert simp(text) == expected


#: The reduced Groebner basis, lexicographic in the variables as listed: each element a
#: primitive polynomial with integer coefficients and a positive leading coefficient,
#: listed by ascending leading monomial.
GROEBNER_BASES = [
    (
        "GROEBNER_BASIS([x*z^3 + 2*y + 2, y*z^3 - y*z^2 - z^2 + y^2 + y], [x, y, z])",
        "[y^2 + y*(z^3 - z^2 + 1) - z^2, x*z^3 + 2*y + 2]",
    ),
    (
        "GROEBNER_BASIS([x^2 + y^2 + z^2 - 2, x^2 + y^2 - z, x^2 + y^2 - 1], [x, y, z])",
        "[z - 1, x^2 + y^2 - 1]",
    ),
    (
        "GROEBNER_BASIS([x^2 + y^2 - z + 1, -x^2 - y^2 - 1 - z], [x, y, z])",
        "[z, x^2 + y^2 + 1]",
    ),
    # The unit ideal, of a system without solutions.
    ("GROEBNER_BASIS([x^2 + y^2 - z + 1, x^2 + y^2 - z + 3], [x, y, z])", "[1]"),
    ("GROEBNER_BASIS([0], [x])", "[]"),
    (
        "GROEBNER_BASIS([a^2 + b^2 + c^2 - 9, a^3 + b^3 + c^3 - 24, a + b + c - 3],"
        " [a, b, c])",
        "[c^3 - 3*c^2 + 1, b^2 + b*(c - 3) + c^2 - 3*c, a + b + c - 3]",
    ),
    # Denominators are cleared, the content removed and the leading coefficient made
    # positive: not monic.
    (
        "GROEBNER_BASIS([x^2 + y^2 - 5, x^2 + y + z - 10, x*z + y^2 - 20], [z, x, y])",
        "[y^6 - 2*y^5 + 7*y^4 - 70*y^2 + 50*y + 275, 605*x + y^5 + 23*y^4 - 23*y^3 +"
        " 30*y^2 + 75*y - 1100, -y^2 + y + z - 5]",
    ),
    (
        "GROEBNER_BASIS([x^2 + y^2 - 5, x^2 + y + z - 10, x*z + y^2 - 20], [x, y, z])",
        "[z^6 - 18*z^5 - 23*z^4 + 2120*z^3 - 8070*z^2 - 60500*z + 366025, 20719*y -"
        " 22*z^5 + 477*z^4 - 2192*z^3 - 19734*z^2 + 208759*z - 538545, 12534995*x -"
        " 34*z^5 + 13922*z^4 - 287803*z^3 + 1254080*z^2 + 12213450*z - 111707200]",
    ),
    ("GROEBNER_BASIS([x/2 - 1/3], [x])", "[3*x - 2]"),
    # Other symbols are coefficients.
    ("GROEBNER_BASIS([a*x^2 - 1, x*y - b], [x, y])", "[y^2 - a*b^2, a*b*x - y]"),
    # A set gives a set, and an equation stands for the difference of its sides.
    ("GROEBNER_BASIS({x - y, x + y}, [x, y])", "{x, y}"),
    ("GROEBNER_BASIS([x^2 = y, x - 1], [x, y])", "[y - 1, x - 1]"),
    (
        "FIRST(GROEBNER_BASIS([u^2 - 2, v^2 - 3, w^2 - 5, x - u - v - w], [u, v, w, x]))",
        "x^8 - 40*x^6 + 352*x^4 - 960*x^2 + 576",
    ),
    # What is no polynomial in the variables, or no variable, leaves the call.
    ("GROEBNER_BASIS([SIN(x)], [x])", "GROEBNER_BASIS([SIN(x)], [x])"),
    ("GROEBNER_BASIS([1/x], [x])", "GROEBNER_BASIS([1/x], [x])"),
    ("GROEBNER_BASIS([x - 1], [2])", "GROEBNER_BASIS([x - 1], [2])"),
    ("GROEBNER_BASIS([x - 1], [x, x])", "GROEBNER_BASIS([x - 1], [x, x])"),
]


@pytest.mark.parametrize(("text", "expected"), GROEBNER_BASES, ids=str)
def test_the_groebner_basis_of_an_ideal(text, expected):
    assert simp(text) == expected


#: The prime numbers.
NUMBER_THEORY = [
    ("PRIME(7)", "true"),
    ("PRIME(8)", "false"),
    ("PRIME(1)", "false"),
    # The second argument is the number of rounds of a probabilistic test. This test is
    # deterministic, so it is ignored.
    ("PRIME(7, 20)", "true"),
    ("PRIME(1991)", "false"),
    ("PRIME(1993)", "true"),
    # Every number that isn't a prime is false; anything else is undecided, so the
    # call stays.
    (
        "[PRIME?(1), PRIME?(-7), PRIME?(7/2), PRIME?(pi), PRIME?(#i)]",
        "[false, false, false, false, false]",
    ),
    ("PRIME(x)", "PRIME(x)"),
    ("PRIME?(n)", "PRIME?(n)"),
    ("PRIME?(22564845703)", "false"),
    ("PRIME?(2521008887)", "true"),
    ("PRIME?(16022236204009818131831320183)", "true"),
    ("PRIME?(3908408957924020300919472370957356345933709)", "true"),
    ("PRIME?(70990461585528724931289825118059422005340095813)", "true"),
    (
        "VECTOR(PRIME?(p_), p_, "
        "[32381, 49919, 57347, 53653, 38321, 11351, 26731, 74873])",
        "[true, true, true, true, true, true, true, true]",
    ),
    (
        "VECTOR(PRIME?(p_), p_, [2, 3, 7, 43, 1807, 3263443, 10650056950807])",
        "[true, true, true, true, false, true, false]",
    ),
    ("SELECT(PRIME(k), k, [3, 5, 7, 9, 11])", "[3, 5, 7, 11]"),
    (
        "ITERATE(IF(PRIME(6*k + 1) AND PRIME(12*k + 1) AND PRIME(18*k + 1), k, k + 1), "
        "k, 1000)",
        "1025",
    ),
    # The prime functions apply to each element of a vector or set.
    ("PRIME?([2, 4, {5, 9}])", "[true, false, {false, true}]"),
    ("NEXT_PRIME({1, 2})", "{2, 3}"),
    # Strictly larger or smaller; the argument is any real number.
    ("NEXT_PRIME(1000)", "1009"),
    ("NEXT_PRIME(7)", "11"),
    ("NEXT_PRIME(7/2)", "5"),
    ("NEXT_PRIME(10.5)", "11"),
    ("NEXT_PRIME(-5)", "2"),
    ("NEXT_PRIME(pi)", "5"),
    ("NEXT_PRIME(1000, 6)", "1009"),
    ("NEXT_PRIME(10^100) - 10^100", "267"),
    ("NEXT_PRIME(3212345)", "3212347"),
    ("PREVIOUS_PRIME(1000)", "997"),
    ("PREVIOUS_PRIME(3)", "2"),
    ("PREVIOUS_PRIME(2)", "?"),
    ("PREVIOUS_PRIME(10.5)", "7"),
    ("PREVIOUS_PRIME(x)", "PREVIOUS_PRIME(x)"),
]


@pytest.mark.parametrize(("text", "expected"), NUMBER_THEORY, ids=str)
def test_number_theory(text, expected):
    assert simp(text) == expected


#: Rounding, residues and the other functions of integers.
INTEGERS = [
    ("FLOOR(3/2 + 5/2*#i)", "1 + 2*#i"),
    ("FLOOR(7, 0)", "?"),
    ("FLOOR(x, 10)", "FLOOR(x/10)"),
    # CEILING is exact like FLOOR, and stands as the FLOOR form.
    ("[CEILING(5.73), CEILING(573, 100), CEILING(-5.73), CEILING(6)]", "[6, 6, -5, 6]"),
    ("CEILING(SQRT(2))", "2"),
    ("CEILING(EXP(0.43*SQRT(LN(100)*LN(LN(100)))))", "4"),
    ("CEILING(x)", "-FLOOR(-x)"),
    ("CEILING(x, 4)", "-FLOOR(-x/4)"),
    ("CEILING(7, 0)", "?"),
    # ROUND takes a tie up, toward +inf, and rounds each coordinate of a complex
    # number.
    ("[ROUND(5.73), ROUND(573, 100), ROUND(7, 2)]", "[6, 6, 4]"),
    ("[ROUND(5/2), ROUND(-5/2), ROUND(1/10000000000)]", "[3, -2, 0]"),
    ("ROUND(1.2*SQRT(193707721))", "16701"),
    ("ROUND(5/2 - 7/5*#i)", "3 - #i"),
    ("ROUND(x)", "FLOOR(x + 1/2)"),
    # Rounded exactly, however long the exact exponent.
    (
        "VECTOR(ROUND(2^(1.30076870414817691055252567828266106688423996320151467218595488"
        "^n)), n, 1, 12)",
        "[2, 3, 5, 7, 13, 29, 79, 293, 1619, 14947, 269237, 11570443]",
    ),
    (
        "VECTOR(ROUND((10^100 + 267)^(0.38562256641529^n)), n, 2, 6)",
        "[742123822526586, 542489, 163, 7, 2]",
    ),
    (
        "VECTOR(ROUND(2.1322219996628413452^((1/0.38562256641529)^n)), n, 0, 3)",
        "[2, 7, 163, 542489]",
    ),
    # MOD lies in [0, n) for a positive n, MODS in [-n/2, n/2).
    ("[MOD(5.73), MODS(5.73)]", "[73/100, -27/100]"),
    ("[MOD(573, 100), MODS(573, 100), MODS(50, 100)]", "[73, -27, -50]"),
    ("[MOD(7, 0), MOD(-7, 3), MOD(7, -3)]", "[7, 2, -2]"),
    ("MOD(3^(-1), 5)", "1/3"),
    (
        "VECTOR(MODS(k, 13), k, 100, 120)",
        "[-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3]",
    ),
    # A power is reduced by modular powering, however big it would be.
    (
        "VECTOR(MOD(3^(10^n), 10007), n, 1, 10)",
        "[9014, 4995, 6823, 5079, 2749, 8517, 594, 6074, 7480, 493]",
    ),
    (
        "VECTOR(MODS(3^(10^n), 10007), n, 1, 10)",
        "[-993, 4995, -3184, -4928, 2749, -1490, 594, -3933, -2527, 493]",
    ),
    ("MOD(5^62032454, 3397745833)", "123456789"),
    ("MODS(2^((10^100 + 266)/2), 10^100 + 267)", "-1"),
    # Symbolic, MOD stands, normalized by identities that hold for every value.
    ("MOD(2*x + 3, 5)", "MOD(2*x + 3, 5)"),
    ("MOD(x)", "MOD(x)"),
    ("MOD(t, 2292816/364913)", "MOD(364913*t, 2292816)/364913"),
    ("MOD(6*x, 4)", "2*MOD(3*x, 2)"),
    ("MODS(6*x, 4)", "2*MODS(3*x, 2)"),
    ("x - FLOOR(x)", "MOD(x)"),
    ("y - 5*FLOOR(y/5)", "MOD(y, 5)"),
    # Gaussian integers: the quotient is rounded in each coordinate.
    ("MOD(5 + 3*#i, 4 + #i)", "1 + 2*#i"),
    ("[MOD(-5 + 3*#i, 4 + #i), MOD(-6 + 7*#i, 4 + #i)]", "[0, 0]"),
    ("MODS(3 + 5*#i, 4 + #i)", "0"),
    ("MODS(5 + 3*#i, 4 + #i)", "1 + 2*#i"),
    ("MODS(-5 + 5*#i, 4 + #i)", "2*#i"),
    ("MOD([[12, 3], [5, 6]] . [[11], [4]], 26)", "[[14], [1]]"),
    # The polynomial with its coefficients reduced; a denominator is inverted.
    ("POLY_MOD(3*x^3 + 6*x + 3, 3)", "0"),
    ("POLY_MOD(8*x^2 - 3*x + 10, 7)", "x^2 + 4*x + 3"),
    ("POLY_MODS(8*x^2 - 3*x + 10, 7)", "x^2 - 3*x + 3"),
    ("POLY_MOD(x*y*5 + 12*y^2, 5)", "2*y^2"),
    # A quotient whose denominator divides out is the polynomial it cancels to.
    ("POLY_MOD(8*(3*x^4 + 6)*(4*x^2*y - 24*y)/(2*y), 7)", "6*x^6 + 6*x^4 + 5*x^2 + 5"),
    ("POLY_MOD(x/2 + 1, 7)", "4*x + 1"),
    ("POLY_MOD(x/7 + 1, 7)", "POLY_MOD(x/7 + 1, 7)"),
    ("POLY_MOD(SIN(x), 5)", "POLY_MOD(SIN(x), 5)"),
    ("POLY_MOD(REMAINDER((x + 1)^7, x^2 + 1, x), 7)", "6*x + 1"),
    ("POLY_MOD(3*x + 8 = 9, 7)", "3*x + 1 = 2"),
    # As MOD(c, 0) is c, modulus 0 only expands; a constant is reduced as MOD does.
    ("POLY_MOD((x + 1)^2/2, 0)", "x^2/2 + x + 1/2"),
    ("[POLY_MOD(6, 4), POLY_MODS(7, 4), POLY_MOD(5, 0)]", "[2, -1, 5]"),
    # Shifts in bits, a right shift rounding toward -inf.
    ("[SHIFT(5, 3), SHIFT(41, -3), SHIFT(-5, -1)]", "[40, 5, -3]"),
    ("SHIFT(1, 100)", "1267650600228229401496703205376"),
    ("SHIFT(x, 2)", "SHIFT(x, 2)"),
    ("MOD(SHIFT(4^2 - 2, -3) + ((4^2 - 2) AND 7), 7)", "0"),
    # Inverses and powers modulo m.
    ("[INVERSE_MOD(37, 53), INVERSE_MOD(4, 31), INVERSE_MOD(3, 97)]", "[43, 8, 65]"),
    ("INVERSE_MOD(1234567, 234567)", "9640"),
    ("INVERSE_MOD(65537, 140400)", "68273"),
    ("[INVERSE_MOD(6, 9), INVERSE_MOD(5, 1)]", "[?, 0]"),
    ("INVERSE_MOD(x, 7)", "INVERSE_MOD(x, 7)"),
    ("[POWER_MOD(2, 500, 100), POWER_MOD(3, -1, 5), POWER_MOD(2, -3, 7)]", "[76, 2, 1]"),
    ("POWER_MOD(6, -1, 9)", "?"),
    ("POWER_MOD(5, 62032454, 3397745833)", "123456789"),
    # GCD and LCM of several vectors pair their elements.
    ("GCD([6, 15], [10, 4])", "[2, 1]"),
    ("[GCD(1/2, 3/4), GCD(-4, 6)]", "[1/4, 2]"),
    ("LCM(x, y)", "ABS(x*y)/GCD(y, x)"),
    ("GCD([[4, 6], [10, 15]])", "[2, 5]"),
    ("DIM(LCM([1, ..., 3000]))", "1304"),
]


@pytest.mark.parametrize(("text", "expected"), INTEGERS, ids=str)
def test_functions_of_integers(text, expected):
    assert simp(text) == expected


def test_a_power_inside_mod_is_never_built():
    # Each of these powers has 200,000 bits; built, the test would take seconds.
    start = time.perf_counter()
    assert simp("EVERY(MOD(x^14701, 15251) = x, x, 0, 15250)") == "true"
    assert time.perf_counter() - start < 10


def test_the_utility_files_appended_columns():
    # ``APPEND_COLUMNS(A, B) := APPEND(A`, B`)` ``, from VECTOR.MTH and a manual exercise.
    # It relies on appending matrices stacking rows rather than flattening.
    columns = "APPEND([[a, b], [c, d]]`, [[e, f], [g, h]]`)`"
    assert simp(columns) == "[[a, b, e, f], [c, d, g, h]]"


def test_the_utility_files_matrix_minor():
    # ``MINOR(a,i,j) := DELETE_ELEMENT(DELETE_ELEMENT(a,i)`,j)` ``, from VECTOR.MTH and a
    # manual exercise: delete row `i`, transpose, delete column `j`, transpose back.
    body = parse("DELETE_ELEMENT(DELETE_ELEMENT(a,i)`,j)`")
    context = Context(functions={"MINOR": Definition(("a", "i", "j"), body)})
    assert simp("MINOR([[1,2,3],[4,5,6],[7,8,9]],2,3)", context) == "[[1, 2], [7, 8]]"


def test_the_utility_files_trapezoidal_integral_of_data():
    # NUMERIC.MTH's INT_DATA: the iterates are rows, and the answer is a matrix of the
    # abscissas beside the running integrals, built by transposing.
    body = parse(
        "[ELEMENT(a`,1),ELEMENT(ITERATES([1+ELEMENT(v,1),ELEMENT(v,2)+(ELEMENT(a,"
        "ELEMENT(v,1),2)+ELEMENT(a,1+ELEMENT(v,1),2))*(ELEMENT(a,1+ELEMENT(v,1),1)"
        "-ELEMENT(a,ELEMENT(v,1),1))/2],v,[1,0],DIMENSION(a)-1)`,2)]`"
    )
    context = Context(functions={"INT_DATA": Definition(("a",), body)})
    assert simp("INT_DATA([[0,0],[1,1],[2,4],[3,9]])", context) == (
        "[[0, 0], [1, 1/2], [2, 3], [3, 19/2]]"
    )


def test_the_utility_files_legendre_polynomial():
    # ORTH_POL.MTH's LEGENDRE_P sums the rows of a matrix of iterates and takes the
    # first element of that vector.
    body = parse(
        "2^(-n)*(SUM(ITERATES([v SUB 1*(2*v SUB 2-n-1)*(2*v SUB 2-n-2)/(2*x^2*v SUB 2*"
        "(2*v SUB 2-2*n-1)),v SUB 2+1],v,[COMB(2*n,n)*x^n,1],n/2))) SUB 1"
    )
    context = Context(functions={"P": Definition(("n", "x"), body)})
    assert simp("P(4, x)", context) == "35*x^4/8 - 15*x^2/4 + 3/8"


def test_the_utility_files_approximate_eigenvector_is_a_unit_vector():
    # VECTOR.MTH's APPROX_EIGENVECTOR: one step of inverse iteration, normalized by
    # SIGN. The solution is the last column of the row-reduced augmented matrix.
    body = parse(
        "SIGN(ROW_REDUCE(a-u*IDENTITY_MATRIX(DIMENSION(a)),"
        "VECTOR(3000*COS(pi*m)/m,m,DIMENSION(a)))` SUB (DIMENSION(a)+1))"
    )
    context = Context(functions={"E": Definition(("a", "u"), body)})
    assert simp("E([[2, 0], [0, 3]], 201/100)", context) == (
        "[198*SQRT(39205)/39205, SQRT(39205)/39205]"
    )


#: The number theory of Derive 6's utility library, built on the pairs of `FACTORS`.
LIBRARY_NUMBER_THEORY = (
    "EULER_PHI(n) := n*PRODUCT(1 - 1/v_ SUB 1, v_, FACTORS(n))",
    "MOEBIUS_MU(n) := PRODUCT(-MAX(2 - v_ SUB 2, 0), v_, FACTORS(n))",
    "SQUAREFREE(n) := EVERY(v_ SUB 2 = 1, v_, FACTORS(n))",
    "DIVISORS(n, d_ := [1]) := PROG(MAP(d_ := APPEND(VECTOR(d_*v_ SUB 1^k_, k_, 0, "
    "v_ SUB 2)), v_, FACTORS(n)), SORT(d_))",
    "CYCLOTOMIC(n, x) := PRODUCT((x^(n/d_) - 1)^MOEBIUS_MU(d_), d_, DIVISORS(n))",
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("EULER_PHI(15251)", "15000"),
        ("EULER_PHI(360)", "96"),
        ("[MOEBIUS_MU(30), MOEBIUS_MU(12), MOEBIUS_MU(1)]", "[-1, 0, 1]"),
        ("[SQUAREFREE(30), SQUAREFREE(12)]", "[true, false]"),
        ("DIVISORS(12)", "[1, 2, 3, 4, 6, 12]"),
        ("CYCLOTOMIC(12, x)", "x^4 - x^2 + 1"),
    ],
    ids=str,
)
async def test_the_utility_librarys_number_theory(text, expected):
    session = Session(step_limit=-1)
    for definition in LIBRARY_NUMBER_THEORY:
        session.author(definition)
    assert (await session.simplify(text)).text == expected


async def test_a_definition_over_the_factors_of_its_parameter_stands():
    # The factors of `n` are still to come, so neither FACTORS nor the product over
    # them is computed: the definition simplifies to itself.
    definition = "EULER_PHI(n) := n*PRODUCT(1 - 1/v_ SUB 1, v_, FACTORS(n))"
    assert (await Session().simplify(definition)).text == definition


def test_the_utility_files_legendre_polynomial_of_odd_order():
    # LEGENDRE_P counts `n/2` iterations, floored for an odd order.
    body = parse(
        "2^(-n)*(SUM(ITERATES([v SUB 1*(2*v SUB 2-n-1)*(2*v SUB 2-n-2)/(2*x^2*v SUB 2*"
        "(2*v SUB 2-2*n-1)),v SUB 2+1],v,[COMB(2*n,n)*x^n,1],n/2))) SUB 1"
    )
    context = Context(functions={"P": Definition(("n", "x"), body)})
    assert simp("P(3, x)", context) == "5*x^3/2 - 3*x/2"


def test_a_fit_takes_its_labels_from_the_first_row_of_a_single_matrix():
    # As in Derive; CURV_FIT.MTH's `LSL(m_, x) := FIT(APPEND([[x, a*x + b]], m_))`.
    assert simp("FIT([[x, a*x + b], [0, 1], [1, 3]])") == "2*x + 1"
    body = parse("FIT(APPEND([[x,a*x+b]],m_))")
    context = Context(functions={"LSL": Definition(("m_", "x"), body)})
    assert simp("LSL([[0, 1], [1, 3], [2, 5]], t)", context) == "2*t + 1"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("INVERSE(x^3 + 1, x)", "(x - 1)^(1/3)"),
        ("INVERSE(SIN(x) + 1, x)", "ASIN(x - 1)"),
        ("INVERSE(3*COS(2*x), x)", "ACOS(x/3)/2"),
        # Two terms in `x` have no principal inverse to choose.
        ("INVERSE(x^2 + x, x)", "ITERATE(x^2 + x, x, x, -1)"),
    ],
    ids=str,
)
def test_the_utility_files_inverse_takes_the_principal_branch(text, expected):
    # MISC.MTH's `INVERSE(u, x) := ITERATE(u, x, x, -1)`.
    inverse = Definition(("u", "x"), parse("ITERATE(u,x,x,-1)"))
    context = Context(functions={"INVERSE": inverse})
    assert simp(text, context) == expected


def test_an_argument_that_cancels_to_a_constant_is_passed_as_the_constant():
    # Derive simplifies an argument before the call. ODE.MTH's HOMOGENEOUS_4 computes a
    # scaling exponent that is -2, a quotient in `x` and `SQRT(y)`, and its body tests
    # it and raises `x` to it.
    context = Context(functions={"F": Definition(("n",), parse("IF(n=-2,x^n,0)"))})
    text = "F(x*(-(x + SQRT(y))/y - x/y)/(y*(x*(x + SQRT(y))/y^2 - x/(2*y^(3/2)))))"
    assert simp(text, context) == "1/x^2"


def test_the_demos_matrix_generator():
    # The demo's `MATRIX(z,i,m,j,n) := VECTOR(VECTOR(z,j,1,n),i,1,m)`.
    body = parse("VECTOR(VECTOR(z,j,1,n),i,1,m)")
    context = Context(functions={"MATRIX": Definition(("z", "i", "m", "j", "n"), body)})
    assert simp("MATRIX(i-j,i,2,j,3)", context) == "[[0, -1, -2], [1, 0, -1]]"


def test_the_demos_outer_product():
    # `OUTER(v,w) := VECTOR([v SUB i],i,DIMENSION(v)) . [w]`. The length depends on the
    # argument, so nothing is generated until the definition is applied.
    body = parse("VECTOR([v SUB i],i,DIMENSION(v)) . [w]")
    context = Context(functions={"OUTER": Definition(("v", "w"), body)})
    assert simp("OUTER([a,b,c],[2,3,4])", context) == (
        "[[2*a, 3*a, 4*a], [2*b, 3*b, 4*b], [2*c, 3*c, 4*c]]"
    )

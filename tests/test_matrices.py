"""What Simplify makes of vectors and matrices.

Covers vector and matrix arithmetic, variables declared nonscalar, linear algebra,
vector calculus, `VECTOR` and subscripts, iteration, sample statistics, functions that
split an expression into a vector of parts, and number theory. Cases are inline, as in
`test_simplify.py`, so that a failure shows the mathematics. The matrix helpers from
the utility files and demos are tested here too, since they rely on this arithmetic.
"""

from __future__ import annotations

import pytest
from simplifying import declared, declaring, parse, simp

from rederive.engine.computing import Context


def test_a_conditional_whose_branches_are_vectors_simplifies_elementwise():
    # The chosen branch is a vector, and is simplified elementwise like a written one.
    taken = "IF(a > 0, [1, 2], [3, 4], [SQRT(b)*SQRT(b), 2*b/b])"
    assert simp(taken) == simp("[SQRT(b)*SQRT(b), 2*b/b]") == "[b, 2]"


def test_a_symbolic_power_of_a_matrix_is_its_closed_form():
    # A unipotent matrix to a symbolic power is a polynomial in `n`.
    assert simp("[[1, 1], [0, 1]]^n") == "[[1, n], [0, 1]]"


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
    ("[[1, 2], [3, 4]] . [[a], [b]]", "[[a + 2*b], [3*a + 4*b]]"),
    # A flat vector right of a matrix acts as a column and the answer is flat again; a
    # written column stays a column.
    ("[[a, b], [c, d]] . [2, 3]", "[2*a + 3*b, 2*c + 3*d]"),
    ("[2, 3] . [[a, b], [c, d]]", "[2*a + 3*c, 2*b + 3*d]"),
    # The cross product of 3-vectors, and of 2-vectors: the latter is the third component
    # of the 3D product (the other two are zero), returned as a number.
    ("CROSS([1, 2, 3], [a, b, c])", "[2*c - 3*b, 3*a - c, b - 2*a]"),
    ("CROSS([1, 2], [a, b])", "b - 2*a"),
    # A vector index reaches through as many dimensions as it has elements.
    ("[[2, 3, 5], [7, 1, 4]] SUB [2, 3]", "4"),
    ("ELEMENT([[2, 3, 5], [7, 1, 4]], [2, 3])", "4"),
    ("[[2, 3, 5], [7, 1, 4]] SUB [2]", "[7, 1, 4]"),
    # A vector of matrices stays one rather than being flattened into a 2x4. It has two
    # elements, and `ELEMENT` takes one index per dimension.
    ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", "[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]"),
    ("DIMENSION([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])", "2"),
    ("ELEMENT([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 2)", "[[5, 6], [7, 8]]"),
    ("ELEMENT([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 2, 1, 2)", "6"),
    ("VECTOR([[k, k], [k, k]], k, 2)", "[[[1, 1], [1, 1]], [[2, 2], [2, 2]]]"),
    # `SUB` is equivalent to `ELEMENT`, one dimension per index, however the indices are
    # written. The shipped utility files use it to index Christoffel symbols.
    ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]] SUB 2 SUB 1 SUB 2", "6"),
    ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]] SUB [2, 1, 2]", "6"),
    ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]] SUB 2", "[[5, 6], [7, 8]]"),
    # An out-of-range index keeps the access unevaluated, so a generator that substitutes
    # a valid index later still has it.
    (
        "[[[1, 2], [3, 4]], [[5, 6], [7, 8]]] SUB 4",
        "ELEMENT([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 4)",
    ),
    ("VECTOR([[[1, 2], [3, 4]], [[5, 6], [7, 8]]] SUB i SUB 1 SUB 2, i, 2)", "[2, 6]"),
    # More indices than dimensions keep the access unevaluated.
    ("[a, b, c, d] SUB [4, 2]", "ELEMENT([a, b, c, d], [4, 2])"),
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
    # same rule, so a row plus a scalar stays, and so do rows of different lengths.
    ("[[1, 2], [3, 4]] + [1, 2]", "[[1, 2] + 1, [3, 4] + 2]"),
    (
        "[[1, 2], [3, 4]] + [[1, 2, 3], [4, 5, 6]]",
        "[[1, 2] + [1, 2, 3], [3, 4] + [4, 5, 6]]",
    ),
    ("x*([1, 2] + [1, 2, 3])", "([1, 2] + [1, 2, 3])*x"),
    ("([1, 2] + [1, 2, 3]) + [1, 1]", "[2, 3] + [1, 2, 3]"),
    ("[1, 2] + [1, 2, 3] - ([1, 2] + [1, 2, 3])", "[0, 0] + [0, 0, 0]"),
    ("DET([1, 2] + [1, 2, 3])", "DET([1, 2] + [1, 2, 3])"),
    ("[1, 2] + 5", "[1, 2] + 5"),
    # Only square matrices have powers and inverses to divide by.
    ("[[1, 2, 3], [4, 5, 6]]^(-1)", "[[1, 2, 3], [4, 5, 6]]^(-1)"),
    ("[[1, 2, 3], [4, 5, 6]]^2", "[[1, 2, 3], [4, 5, 6]]^2"),
    ("[[1, 2, 3], [4, 5, 6]]^n", "[[1, 2, 3], [4, 5, 6]]^n"),
    ("[1, 2]^2", "[1, 2]^2"),
    ("[1, 2]/[3, 4]", "[1, 2]/[3, 4]"),
    ("2/[1, 2]", "2/[1, 2]"),
    ("[[1, 2], [3, 4]]/[1, 2]", "[[1, 2], [3, 4]]/[1, 2]"),
    ("[[1, 2], [3, 4]]/[[1, 2], [3, 5]]", "[[1, 0], [-3, 2]]"),
    ("[1, 2]/[[1, 2], [3, 4]]", "[1, 0]"),
    ("[[1, 2], [3, 4]]/[[1, 2], [2, 4]]", "[[1, 2], [3, 4]] . [[1, 2], [2, 4]]^-1"),
    ("CROSS([1, 2], [1, 2, 3])", "CROSS([1, 2], [1, 2, 3])"),
    ("ELEMENT([1, 2], 3)", "ELEMENT([1, 2], 3)"),
    ("a . b", "a . b"),
    # Ragged: not a matrix, and left alone.
    ("[[1, 2], [3]]", "[[1, 2], [3]]"),
]


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
    # Only rows of one length transpose.
    ("[[1, 2], [3]]`", "[[1, 2], [3]]`"),
    # Elementary functions and scalar addition don't distribute over a vector (D).
    ("SIN([1, 2])", "SIN([1, 2])"),
    ("[1, 2] + 2", "[1, 2] + 2"),
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
    ("[[1, 2]] = [[1, 2]]", "true"),
    ("[[1, 2]] = [1, 2]", "false"),
    ("[] = []", "true"),
    # A vector of equations isn't related element by element.
    ("[x = 1, y = 2] = [1, 2]", "[x = 1, y = 2] = [1, 2]"),
]


@pytest.mark.parametrize(("text", "expected"), NESTING, ids=str)
def test_a_vector_is_shaped_by_its_nesting_alone(text, expected):
    assert simp(text) == expected


def test_a_solved_vector_is_a_vector_to_generate_over():
    # Derive gives `[2, -2]` (D), its solutions coming in the other order.
    assert simp("VECTOR(RHS(e), e, SOLVE(x^2 = 4, x))") == "[-2, 2]"


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
    # Eigenvalues are given as solved equations: the manual's example, and a 2 x 2
    # needing the quadratic formula. For a pair differing only in a radical's sign,
    # Derive lists the sum first; this follows sympy's order, which reverses that case
    # but otherwise agrees with Derive.
    ("EIGENVALUES([[2,3],[0,b]],z)", "[z = 2, z = b]"),
    ("EIGENVALUES([[5,0],[0,2]],z)", "[z = 2, z = 5]"),
    ("EIGENVALUES([[1,2],[3,4]],x)", "[x = 5/2 - SQRT(33)/2, x = SQRT(33)/2 + 5/2]"),
    # A repeated eigenvalue is listed once; multiplicity shows in the eigenvectors, not
    # as a second solution.
    ("EIGENVALUES([[1,0,0],[0,1,0],[0,0,2]],z)", "[z = 1, z = 2]"),
    # A symmetric matrix has real eigenvalues. This one's characteristic polynomial is a
    # casus irreducibilis, whose three real roots Cardano's formula writes with `#i`;
    # Viete's trigonometric form writes them without it.
    (
        "EIGENVALUES([[0,1,1],[1,1,1],[1,1,-1]],z)",
        "[z = -4*SQRT(3)*SIN(pi/6 - ACOS(3*SQRT(3)/8)/3)/3, "
        "z = -4*SQRT(3)*COS(pi/3 - ACOS(3*SQRT(3)/8)/3)/3, "
        "z = 4*SQRT(3)*COS(ACOS(3*SQRT(3)/8)/3)/3]",
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
    # An uncounted iteration that never repeats stays unevaluated (Derive runs until
    # memory runs out), as does a symbolic count.
    ("ITERATES(x^2, x, 2)", "ITERATES(x^2, x, 2)"),
    ("ITERATE(a*x, a, 1, n)", "ITERATE(a*x, a, 1, n)"),
]


@pytest.mark.parametrize(("text", "expected"), ITERATION, ids=str)
def test_iteration(text, expected):
    assert simp(text) == expected


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
    newton = (("u", "x", "a", "n"), parse("ITERATES(x - u/DIF(u, x), x, a, n)"))
    context = Context(functions={"NEWTON": newton})
    assert simp("NEWTON(x^2 - 3, x, 2, 3)", context) == "[2, 7/4, 97/56, 18817/10864]"
    assert simp("NEWTON(t^2 - 3, t, 2, 3)", context) == "[2, 7/4, 97/56, 18817/10864]"


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
    # A non-sum has one term, and a vector distributes.
    ("TERMS(x*y)", "[x*y]"),
    ("TERMS([x + 1, y + 2])", "[[x, 1], [y, 2]]"),
    # Most main first: `x`, `y`, `z` from the order list, then the rest alphabetically,
    # the same order other commands use.
    ("VARIABLES(x^2 + a*y)", "[x, y, a]"),
    ("VARIABLES(a*b*z)", "[z, a, b]"),
    ("VARIABLES(5)", "[]"),
    # The argument is simplified first, as in Derive: this one cancels to `a^2`.
    # ODE.MTH's SYMMETRIC_TEST relies on this.
    ("VARIABLES((a^2*y^2 + a^2*y*z)/(y^2 + y*z))", "[a]"),
    # A quotient in `x` and `SQRT(y)` cancels; ODE.MTH's HOMOGENEOUS_4 asks whether
    # this exponent is free of `x` and `y`. A standing integral is held as it is.
    ("VARIABLES(x*(-(x + SQRT(y))/y - x/y)/(y*(x*(x + SQRT(y))/y^2 - x/(2*y^(3/2)))))",
     "[]"),
    ("VARIABLES(SIN(x)^2 + COS(x)^2 + y)", "[y]"),
    ("VARIABLES(INT(SIN(x^3)/LN(x), x) + (a*y + y)/(a + 1))", "[x, y]"),
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
    # A symbolic or out-of-range index leaves the call as is.
    ("DELETE_ELEMENT([a, b, c], n)", "DELETE_ELEMENT([a, b, c], n)"),
    ("DELETE_ELEMENT([a, b, c], 4)", "DELETE_ELEMENT([a, b, c], 4)"),
    # Inserts before the nth element; the index defaults to 1. One past the end is valid
    # here, unlike above, since that is how to append.
    ("INSERT_ELEMENT(d, [a, b, c], 2)", "[a, d, b, c]"),
    ("INSERT_ELEMENT(d, [a, b, c])", "[d, a, b, c]"),
    ("INSERT_ELEMENT(d, [a, b, c], 4)", "[a, b, c, d]"),
    # Two past the end is invalid and the call stays, as with DELETE_ELEMENT. Derive
    # answers `[a, b, c]`, silently dropping the element.
    ("INSERT_ELEMENT(d, [a, b, c], 5)", "INSERT_ELEMENT(d, [a, b, c], 5)"),
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
    # An element with no place in the order leaves the call as written, and so does a
    # pair of numbers whose order can't be decided exactly (these two are equal).
    ("SORT([x + 1, 2])", "SORT([x + 1, 2])"),
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
    # Syntactic factors, the counterpart of `TERMS`: nothing is factored first, so a sum
    # or a number is one factor.
    ("FACTORS(3*x*(x + 1)^2)", "[(x + 1)^2, x, 3]"),
    ("FACTORS(x^2 - 1)", "[x^2 - 1]"),
    ("FACTORS(12)", "[12]"),
    ("FACTORS(2/3)", "[2/3]"),
    ("FACTORS(x/y)", "[x, 1/y]"),
    # Composed with FACTOR, a number's prime powers (see test_factor.py for how they
    # are shown), values like any other once used. One is the empty product.
    ("FACTORS(FACTOR(360)) SUB 1", "8"),
    ("VECTOR(PRIME(k), k, FACTORS(FACTOR(360)))", "[false, false, true]"),
    ("FACTORS(1)", "[]"),
    # Most main first; for the same variable, compound before simple.
    ("FACTORS(3*x*y)", "[x, y, 3]"),
    ("FACTORS(x^2*y^3)", "[x^2, y^3]"),
    ("FACTORS(x*SIN(x))", "[SIN(x), x]"),
    ("FACTORS((x + 1)*(x + 2))", "[x + 2, x + 1]"),
    # A vector distributes, as it does for TERMS.
    ("FACTORS([x*y, 2*z])", "[[x, y], [z, 2]]"),
    # Sides of a relation, distributing over a vector, so `RHS(SOLVE(u, x))` gives the
    # roots.
    ("LHS(2*x + 3 = 5)", "2*x + 3"),
    ("RHS(2*x + 3 = 5)", "5"),
    ("LHS([x = 1, y = 2])", "[x, y]"),
    ("RHS([x = 1, y = 2])", "[1, 2]"),
    # A non-relation `u` reads as `u = 0`, as in Derive.
    ("LHS(2*x + 3)", "2*x + 3"),
    ("RHS(3)", "0"),
    ("RHS(x + y)", "0"),
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
    # The manual's gcd example, and gcd of numbers.
    ("POLY_GCD(x^3 + 3*x^2 + 5*x + 6, x^3 + 2*x - 3)", "x^2 + x + 3"),
    ("POLY_GCD(12, 18)", "6"),
]


@pytest.mark.parametrize(("text", "expected"), DECOMPOSITION, ids=str)
def test_what_an_expression_decomposes_into(text, expected):
    assert simp(text) == expected


#: The two number-theory functions.
NUMBER_THEORY = [
    ("PRIME(7)", "true"),
    ("PRIME(8)", "false"),
    ("PRIME(1)", "false"),
    # The second argument is the number of rounds of a probabilistic test. This test is
    # deterministic, so it is ignored.
    ("PRIME(7, 20)", "true"),
    # A non-integer is undecided, not false, so the call stays.
    ("PRIME(x)", "PRIME(x)"),
    # Strictly larger; the argument needn't be an integer.
    ("NEXT_PRIME(1000)", "1009"),
    ("NEXT_PRIME(7)", "11"),
    ("NEXT_PRIME(7/2)", "5"),
]


@pytest.mark.parametrize(("text", "expected"), NUMBER_THEORY, ids=str)
def test_number_theory(text, expected):
    assert simp(text) == expected


def test_the_utility_files_appended_columns():
    # ``APPEND_COLUMNS(A, B) := APPEND(A`, B`)` ``, from VECTOR.MTH and a manual exercise.
    # It relies on appending matrices stacking rows rather than flattening.
    columns = "APPEND([[a, b], [c, d]]`, [[e, f], [g, h]]`)`"
    assert simp(columns) == "[[a, b, e, f], [c, d, g, h]]"


def test_the_utility_files_matrix_minor():
    # ``MINOR(a,i,j) := DELETE_ELEMENT(DELETE_ELEMENT(a,i)`,j)` ``, from VECTOR.MTH and a
    # manual exercise: delete row `i`, transpose, delete column `j`, transpose back.
    body = parse("DELETE_ELEMENT(DELETE_ELEMENT(a,i)`,j)`")
    context = Context(functions={"MINOR": (("a", "i", "j"), body)})
    assert simp("MINOR([[1,2,3],[4,5,6],[7,8,9]],2,3)", context) == "[[1, 2], [7, 8]]"


def test_the_utility_files_trapezoidal_integral_of_data():
    # NUMERIC.MTH's INT_DATA: the iterates are rows, and the answer is a matrix of the
    # abscissas beside the running integrals, built by transposing.
    body = parse(
        "[ELEMENT(a`,1),ELEMENT(ITERATES([1+ELEMENT(v,1),ELEMENT(v,2)+(ELEMENT(a,"
        "ELEMENT(v,1),2)+ELEMENT(a,1+ELEMENT(v,1),2))*(ELEMENT(a,1+ELEMENT(v,1),1)"
        "-ELEMENT(a,ELEMENT(v,1),1))/2],v,[1,0],DIMENSION(a)-1)`,2)]`"
    )
    context = Context(functions={"INT_DATA": (("a",), body)})
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
    context = Context(functions={"P": (("n", "x"), body)})
    assert simp("P(4, x)", context) == "35*x^4/8 - 15*x^2/4 + 3/8"


def test_the_utility_files_approximate_eigenvector_is_a_unit_vector():
    # VECTOR.MTH's APPROX_EIGENVECTOR: one step of inverse iteration, normalized by
    # SIGN. The solution is the last column of the row-reduced augmented matrix.
    body = parse(
        "SIGN(ROW_REDUCE(a-u*IDENTITY_MATRIX(DIMENSION(a)),"
        "VECTOR(3000*COS(pi*m)/m,m,DIMENSION(a)))` SUB (DIMENSION(a)+1))"
    )
    context = Context(functions={"E": (("a", "u"), body)})
    assert simp("E([[2, 0], [0, 3]], 201/100)", context) == (
        "[198*SQRT(39205)/39205, SQRT(39205)/39205]"
    )


#: NUMBER.MTH's factorizations, which factor with `FACTORS(FACTOR(n))` and sort with SORT.
NUMBER_THEORY_FILE = {
    "PRIME_FACTORIZATION": (
        ("n",),
        "VECTOR(ITERATE(IF(PRIME(p_),[p_,k_],[p_^(k_/(k_+1)),k_+1],"
        "[p_^(k_/(k_+1)),k_+1]),[p_,k_],[f_,1]),f_,FACTORS(FACTOR(n)))",
    ),
    "MOEBIUS_MU": (("n",), "PRODUCT(IF(PRIME(k_),-1,0),k_,FACTORS(FACTOR(n)))"),
    "DIVISORS": (
        ("n",),
        "SORT((ITERATE(IF(DIMENSION(v_)=0,[u_,v_],[APPEND([u_]`*[ITERATES(v_ SUB 1"
        " SUB 1*b_,b_,1,v_ SUB 1 SUB 2)]),DELETE_ELEMENT(v_)]),[u_,v_],[[1],"
        "PRIME_FACTORIZATION(n)])) SUB 1)",
    ),
    "CYCLOTOMIC": (("n", "x"), "PRODUCT((x^(n/d_)-1)^MOEBIUS_MU(d_),d_,DIVISORS(n))"),
}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("PRIME_FACTORIZATION(360)", "[[2, 3], [3, 2], [5, 1]]"),
        ("PRIME_FACTORIZATION(1)", "[]"),
        ("MOEBIUS_MU(30)", "-1"),
        ("MOEBIUS_MU(12)", "0"),
        ("MOEBIUS_MU(1)", "1"),
        ("DIVISORS(12)", "[1, 2, 3, 4, 6, 12]"),
        ("CYCLOTOMIC(12, x)", "x^4 - x^2 + 1"),
    ],
    ids=str,
)
def test_the_utility_files_factorizations(text, expected):
    functions = {
        name: (parameters, parse(body, declaring(*NUMBER_THEORY_FILE)))
        for name, (parameters, body) in NUMBER_THEORY_FILE.items()
    }
    assert simp(text, Context(functions=functions)) == expected


def test_the_utility_files_legendre_polynomial_of_odd_order():
    # LEGENDRE_P counts `n/2` iterations, floored for an odd order.
    body = parse(
        "2^(-n)*(SUM(ITERATES([v SUB 1*(2*v SUB 2-n-1)*(2*v SUB 2-n-2)/(2*x^2*v SUB 2*"
        "(2*v SUB 2-2*n-1)),v SUB 2+1],v,[COMB(2*n,n)*x^n,1],n/2))) SUB 1"
    )
    context = Context(functions={"P": (("n", "x"), body)})
    assert simp("P(3, x)", context) == "5*x^3/2 - 3*x/2"


def test_a_fit_takes_its_labels_from_the_first_row_of_a_single_matrix():
    # As in Derive; CURV_FIT.MTH's `LSL(m_, x) := FIT(APPEND([[x, a*x + b]], m_))`.
    assert simp("FIT([[x, a*x + b], [0, 1], [1, 3]])") == "2*x + 1"
    body = parse("FIT(APPEND([[x,a*x+b]],m_))")
    context = Context(functions={"LSL": (("m_", "x"), body)})
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
    context = Context(functions={"INVERSE": (("u", "x"), parse("ITERATE(u,x,x,-1)"))})
    assert simp(text, context) == expected


def test_an_argument_that_cancels_to_a_constant_is_passed_as_the_constant():
    # Derive simplifies an argument before the call. ODE.MTH's HOMOGENEOUS_4 computes a
    # scaling exponent that is -2, a quotient in `x` and `SQRT(y)`, and its body tests
    # it and raises `x` to it.
    context = Context(functions={"F": (("n",), parse("IF(n=-2,x^n,0)"))})
    text = "F(x*(-(x + SQRT(y))/y - x/y)/(y*(x*(x + SQRT(y))/y^2 - x/(2*y^(3/2)))))"
    assert simp(text, context) == "1/x^2"


def test_the_demos_matrix_generator():
    # The demo's `MATRIX(z,i,m,j,n) := VECTOR(VECTOR(z,j,1,n),i,1,m)`.
    body = parse("VECTOR(VECTOR(z,j,1,n),i,1,m)")
    context = Context(functions={"MATRIX": (("z", "i", "m", "j", "n"), body)})
    assert simp("MATRIX(i-j,i,2,j,3)", context) == "[[0, -1, -2], [1, 0, -1]]"


def test_the_demos_outer_product():
    # `OUTER(v,w) := VECTOR([v SUB i],i,DIMENSION(v)) . [w]`. The length depends on the
    # argument, so nothing is generated until the definition is applied.
    body = parse("VECTOR([v SUB i],i,DIMENSION(v)) . [w]")
    context = Context(functions={"OUTER": (("v", "w"), body)})
    assert simp("OUTER([a,b,c],[2,3,4])", context) == (
        "[[2*a, 3*a, 4*a], [2*b, 3*b, 4*b], [2*c, 3*c, 4*c]]"
    )

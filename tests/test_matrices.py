"""What Simplify makes of vectors and matrices.

The arithmetic of vectors and matrices, a variable declared nonscalar, linear algebra,
vector calculus, the vectors `VECTOR` generates and the elements a subscript selects,
iteration, statistics over a sample, the functions that take an expression apart into a
vector of its pieces, and number theory. Each case is the author's text in and the
worksheet's text out, written inline as in `test_simplify.py` so that a failure names
the mathematics. `[2, a, 5] . [2*a, 3, -1]` is `7*a - 5`; `[[1, 1], [0, 1]]^n` is
`[[1, n], [0, 1]]`, the closed form of a symbolic power; `[[1, 2], [2, 4]]^-1` stands,
since a singular matrix has no inverse to give; and `a*b` of two declared nonscalars is
the matrix product `a . b`, however it was written. The utility files' and the demos'
matrix helpers are here too, since what they lean on is this arithmetic.
"""

from __future__ import annotations

import pytest
from simplifying import declared, parse, simp

from rederive.engine.computing import Context


def test_a_conditional_whose_branches_are_vectors_simplifies_elementwise():
    # The clause taken is a vector, and a vector is simplified element by
    # element whether it was written or arrived at.
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
    # A flat vector to the right of a matrix is the column it stands for, and
    # the answer is flat again; written as a column it stays one.
    ("[[a, b], [c, d]] . [2, 3]", "[2*a + 3*b, 2*c + 3*d]"),
    ("[2, 3] . [[a, b], [c, d]]", "[2*a + 3*c, 2*b + 3*d]"),
    # The cross product of two vectors of three elements, and of two of two:
    # the second is the third component of the first, the one the other two
    # come to zero in, and it is a number rather than a vector.
    ("CROSS([1, 2, 3], [a, b, c])", "[2*c - 3*b, 3*a - c, b - 2*a]"),
    ("CROSS([1, 2], [a, b])", "b - 2*a"),
    # An index that is a vector reaches through as many dimensions as it holds.
    ("[[2, 3, 5], [7, 1, 4]] SUB [2, 3]", "4"),
    ("ELEMENT([[2, 3, 5], [7, 1, 4]], [2, 3])", "4"),
    ("[[2, 3, 5], [7, 1, 4]] SUB [2]", "[7, 1, 4]"),
    # Two dimensions are all a matrix has, so a vector of matrices stays a
    # vector of matrices rather than being flattened into one 2x4. It counts
    # two elements, and `ELEMENT` takes an index for every dimension there is.
    ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]", "[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]"),
    ("DIMENSION([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])", "2"),
    ("ELEMENT([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 2)", "[[5, 6], [7, 8]]"),
    ("ELEMENT([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 2, 1, 2)", "6"),
    ("VECTOR([[k, k], [k, k]], k, 2)", "[[[1, 1], [1, 1]], [[2, 2], [2, 2]]]"),
    # `SUB` is the other spelling of `ELEMENT` and reaches as deep, one
    # dimension per index, however the index is written. That is the spelling
    # the shipped utility files index a table of Christoffel symbols with.
    ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]] SUB 2 SUB 1 SUB 2", "6"),
    ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]] SUB [2, 1, 2]", "6"),
    ("[[[1, 2], [3, 4]], [[5, 6], [7, 8]]] SUB 2", "[[5, 6], [7, 8]]"),
    # An index that is no index yet keeps the access whole, so that a generator
    # writing a number in later has one to make.
    (
        "[[[1, 2], [3, 4]], [[5, 6], [7, 8]]] SUB 4",
        "ELEMENT([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 4)",
    ),
    ("VECTOR([[[1, 2], [3, 4]], [[5, 6], [7, 8]]] SUB i SUB 1 SUB 2, i, 2)", "[2, 6]"),
    # More indices than dimensions reach a scalar with an index left over, and
    # keep the access whole, however the indices are written.
    ("[a, b, c, d] SUB [4, 2]", "ELEMENT([a, b, c, d], [4, 2])"),
    ("ELEMENT([a, b, c], 3, 1)", "ELEMENT([a, b, c], 3, 1)"),
    # Two indices on an ordinary matrix reach what they always did, and an
    # index past the end is no index and keeps the head.
    ("ELEMENT([[2, 3, 5], [7, 1, 4]], 2, 3)", "4"),
    ("ELEMENT([[2, 3, 5], [7, 1, 4]], 2)", "[7, 1, 4]"),
    (
        "ELEMENT([[2, 3, 5], [7, 1, 4]], 2, 4)",
        "ELEMENT([[2, 3, 5], [7, 1, 4]], 2, 4)",
    ),
    # Shapes that will not multiply keep the operator, unevaluated: better the
    # expression back than a guess at which product was meant.
    ("[[1, 2], [3, 4]] . [1, 2, 3]", "[[1, 2], [3, 4]] . [1, 2, 3]"),
    # Nor will they add. Each shape's terms are added together and the sum of the
    # shapes stands. A matrix and a vector are two shapes like any other, as a
    # vector and a scalar are: there is no sum to distribute over the rows.
    ("[1, 2] + [1, 2, 3]", "[1, 2] + [1, 2, 3]"),
    ("[1, 2] - [1, 2, 3]", "[1, 2] + [-1, -2, -3]"),
    ("[1, 2] + [3, 4] + [1, 2, 3]", "[4, 6] + [1, 2, 3]"),
    ("x + 1 + [1, 2] + [1, 2, 3]", "x + 1 + [1, 2] + [1, 2, 3]"),
    ("[[1, 2], [3, 4]] + [1, 2]", "[[1, 2], [3, 4]] + [1, 2]"),
    (
        "[[1, 2], [3, 4]] + [[1, 2, 3], [4, 5, 6]]",
        "[[1, 2], [3, 4]] + [[1, 2, 3], [4, 5, 6]]",
    ),
    ("x*([1, 2] + [1, 2, 3])", "([1, 2] + [1, 2, 3])*x"),
    ("([1, 2] + [1, 2, 3]) + [1, 1]", "[2, 3] + [1, 2, 3]"),
    ("[1, 2] + [1, 2, 3] - ([1, 2] + [1, 2, 3])", "[0, 0] + [0, 0, 0]"),
    ("DET([1, 2] + [1, 2, 3])", "DET([1, 2] + [1, 2, 3])"),
    ("[1, 2] + 5", "[1, 2] + 5"),
    # Only a square matrix has powers, and only a square matrix has an inverse to
    # divide by.
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
    # Ragged: no matrix to be, and nothing to do to it either.
    ("[[1, 2], [3]]", "[[1, 2], [3]]"),
]


@pytest.mark.parametrize(("text", "expected"), VECTORS, ids=str)
def test_vectors_and_matrices(text, expected):
    assert simp(text) == expected


# -- declared nonscalars ------------------------------------------------------

NONSCALAR = declared(*(f"{name} :epsilon Nonscalar" for name in "abcd"))

#: What a declared nonscalar is worth past the nine rules section 8.8 states.
#: Every one of these comes from the original and none of them is in the manual.
NONSCALARS = [
    # 8.4: multiplication between two matrices is the matrix product, however
    # it was written.
    ("a*b", "a . b"),
    # A matrix beside its own inverse is left exactly as it was written. There
    # is no notation for an identity matrix of unknown dimension, so an answer
    # holding one could not be shown at all.
    ("a . a^-1", "a . a^-1"),
    ("a^-1 . a", "a^-1 . a"),
    # A longer product nests to the right, and the parentheses are printed.
    # `a . (b . c)` is the fixed point; `(a . b) . c` is not.
    ("((a . b) . c) . d", "a . (b . (c . d))"),
    ("a . b . c . d", "a . (b . (c . d))"),
    ("a . (b . c)", "a . (b . c)"),
    # Which is the associativity rule of 8.8, seen from the other side.
    ("a . (b . c) - (a . b) . c", "0"),
    # The determinant of a product, which the manual states only for an
    # inverse.
    ("DET(a . b)", "DET(a)*DET(b)"),
    ("TRACE(a + b)", "TRACE(a) + TRACE(b)"),
    # Distribution reaches inside a longer product too.
    ("a . (b + c) . a", "a . (b . a) + a . (c . a)"),
    # A scalar commutes with everything, so a negated product needs no
    # parentheses and a difference of two of them reads like any other sum.
    ("a . b - b . a", "a . b - b . a"),
    ("-(a . b)", "-a . b"),
    # Dividing by a matrix is multiplying by its inverse, that being the only
    # reading the notation has for it.
    ("1/a", "a^-1"),
    ("x/a", "x*a^-1"),
]


@pytest.mark.parametrize(("text", "expected"), NONSCALARS, ids=str)
def test_the_algebra_of_a_declared_nonscalar(text, expected):
    assert simp(text, NONSCALAR) == expected


def test_a_scalars_transpose_collapses_at_the_first_backquote():
    # 8.5: the transpose of a scalar is the scalar, and everything no
    # declaration calls nonscalar is one.
    assert simp("x`") == "x"
    assert simp("x``") == "x"
    assert simp("(x + y)`") == "x + y"
    # A nonscalar's does not, and takes two to come back.
    assert simp("a`", NONSCALAR) == "a`"
    assert simp("a``", NONSCALAR) == "a"


def test_a_default_domain_of_nonscalar_does_not_make_every_variable_a_matrix():
    # `default :epsilon Nonscalar` widens the domain of everything at once,
    # and everything includes the argument of every function: a matrix there
    # would convert and mean nothing. So the default is worth its assumptions
    # and not the shape a declared name gets.
    default = declared("default :epsilon Nonscalar")
    assert simp("SIN(x)", default) == "SIN(x)"
    assert simp("x . y", default) == "x . y"


#: Row echelon form, characteristic polynomial and eigenvalues: the manual's own
#: examples, and the argument forms each of the three accepts.
LINEAR_ALGEBRA = [
    # Both of the manual's augmented examples. The second is singular: its right
    # column is inconsistent and the one before it is consistent, which is what
    # the reduced form says by putting the pivot in the last column.
    ("ROW_REDUCE([[1,2],[5,6]],[[3,4],[7,8]])", "[[1, 0, -1, -2], [0, 1, 2, 3]]"),
    ("ROW_REDUCE([[1,1],[2,2]],[[1,1],[2,1]])", "[[1, 1, 1, 0], [0, 0, 0, 1]]"),
    # The manual's exercise, exactly rather than approximately: the augmented
    # column holds the solution of the 3 x 3 system.
    (
        "ROW_REDUCE([[5,3,-7],[2,-8,1],[-1,9,4]],[[4],[6],[5]])",
        "[[1, 0, 0, 879/302], [0, 1, 0, 53/302], [0, 0, 1, 239/151]]",
    ),
    # A vector is adjoined as the column it stands for, so it says the same
    # thing as a one-column matrix does.
    (
        "ROW_REDUCE([[5,3,-7],[2,-8,1],[-1,9,4]],[4,6,5])",
        "[[1, 0, 0, 879/302], [0, 1, 0, 53/302], [0, 0, 1, 239/151]]",
    ),
    # One argument reduces the matrix alone: a nonsingular one reduces to the
    # identity, and a rank-deficient one keeps a zero row.
    ("ROW_REDUCE([[1,2],[3,4]])", "[[1, 0], [0, 1]]"),
    ("ROW_REDUCE([[1,2,3],[4,5,6],[7,8,9]])", "[[1, 0, -1], [0, 1, 2], [0, 0, 0]]"),
    # Symbolic entries reduce the generic case, the one where no pivot is zero -
    # the answer the inverse gives for the same system.
    (
        "ROW_REDUCE([[a,b],[c,d]],[e,f])",
        "[[1, 0, (d*e - b*f)/(a*d - b*c)], [0, 1, (a*f - c*e)/(a*d - b*c)]]",
    ),
    # The manual's characteristic polynomial, term for term.
    ("CHARPOLY([[2,3],[a,b]],z)", "z^2 - z*(b + 2) - 3*a + 2*b"),
    # The variable defaults to `w` when none is written, and `w` is not on the
    # order list, so the most main variable of the answer is `a` and the normal
    # form takes `-3*a` off and writes what is left about `b`. Derive answers
    # `(2 - w)*(b - w) - 3*a`: its determinant of `[[2 - w, 3], [a, b - w]]`
    # stays folded, and a product is not a sum for the normal form to touch.
    # Sympy computes the characteristic polynomial by recurrence and hands back
    # a multiplied-out one, which no longer folds. Given that multiplied-out sum
    # Derive answers exactly as this does.
    ("CHARPOLY([[2,3],[a,b]])", "-3*a + b*(2 - w) + w^2 - 2*w"),
    # `DET(A - z*IDENTITY_MATRIX(3))` is the definition, so the cube is negated
    # in an odd dimension; a sum of more than two terms keeps the leading minus.
    ("CHARPOLY([[1,2,3],[4,5,6],[7,8,10]],z)", "-z^3 + 16*z^2 + 12*z - 3"),
    # Eigenvalues are answered as solved equations. The manual's example, and a
    # 2 x 2 whose eigenvalues the quadratic formula has to be used on. Where a
    # pair differs only in the sign of a radical Derive lists the sum first;
    # this is sympy's canonical order, which agrees with Derive on every other
    # case and reverses that one.
    ("EIGENVALUES([[2,3],[0,b]],z)", "[z = 2, z = b]"),
    ("EIGENVALUES([[5,0],[0,2]],z)", "[z = 2, z = 5]"),
    ("EIGENVALUES([[1,2],[3,4]],x)", "[x = 5/2 - SQRT(33)/2, x = SQRT(33)/2 + 5/2]"),
    # A repeated eigenvalue is one eigenvalue: the multiplicity is the number of
    # parameters its eigenvector carries, not a second solution.
    ("EIGENVALUES([[1,0,0],[0,1,0],[0,0,2]],z)", "[z = 1, z = 2]"),
    # A symmetric matrix has real eigenvalues, and this one's characteristic
    # polynomial is the casus irreducibilis: three real roots that Cardano's
    # formula can only write through `#i`. Viete's cubic writes them as the
    # cosines they are, so nothing here is spelled with an imaginary unit.
    (
        "EIGENVALUES([[0,1,1],[1,1,1],[1,1,-1]],z)",
        "[z = -4*SQRT(3)*SIN(pi/6 - ACOS(3*SQRT(3)/8)/3)/3, "
        "z = -4*SQRT(3)*COS(pi/3 - ACOS(3*SQRT(3)/8)/3)/3, "
        "z = 4*SQRT(3)*COS(ACOS(3*SQRT(3)/8)/3)/3]",
    ),
    # No square matrix, no characteristic polynomial - and no radicals for the
    # quintic the manual's 5 x 5 exercise leads to, which is the manual's own
    # account of why exact eigenvalues stop at 4 x 4. Both come back unchanged.
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
        # The symmetric matrix the manual's last eigenvalue exercise poses.
        "[[-2,1,1,1,1],[1,-3,-1,0,-1],[1,-1,1,0,-3],[1,0,0,3,0],[1,-1,-3,0,2]]",
    ],
    ids=str,
)
def test_the_characteristic_polynomial_is_the_determinant_defining_it(matrix):
    # The manual defines CHARPOLY as the determinant of the difference of the
    # matrix and a variable times the identity matrix, so the two must agree -
    # including in odd dimensions, where that determinant is not monic. The
    # difference is what is asserted rather than the two texts: the polynomial
    # comes out with its terms collected in the variable and the determinant does
    # not, which is a spelling and not a disagreement.
    written_out = f"DET({matrix} - z*IDENTITY_MATRIX(DIMENSION({matrix})))"
    assert simp(f"CHARPOLY({matrix},z) - {written_out}") == "0"


#: Differential and integral vector calculus: the manual's own examples for each
#: of the six operators, and the argument forms they accept.
VECTOR_CALCULUS = [
    # The default coordinate system is three-dimensional Cartesian in x, y, z,
    # so a gradient written without one has three elements however few variables
    # the expression mentions.
    ("GRAD(x*y^2*z^3)", "[y^2*z^3, 2*x*y*z^3, 3*x*y^2*z^2]"),
    ("GRAD(x^2 + y^2)", "[2*x, 2*y, 0]"),
    # A vector of variables names Cartesian coordinates of the caller's
    # choosing, and there may be any number of them.
    ("GRAD(c*w + x^2 + y^3 + z^4, [w, x, y, z])", "[c, 2*x, 3*y^2, 4*z^3]"),
    ("GRAD(x*y, [x, y])", "[y, x]"),
    ("DIV([y^2*z^3, 2*x*y*z^3, 3*x*y^2*z^2])", "x*(6*y^2*z + 2*z^3)"),
    # Degree one in the primary variable, so the normal form writes it as `x`
    # times its coefficient, and the coefficient keeps its numeric content -
    # which is the manual's own answer. Derive is not consistent about that:
    # the same coefficient `2*y + 6` comes back `x*(2*y + 6)` authored as
    # `2*x*y + 6*x` and `2*x^2*(y + 3)` when it falls out of expanding
    # `((x + 1)^2 + y)^2`, both taken from the original, so there is no one form
    # that matches everything. LAPLACIAN is DIV of GRAD, so the two agree.
    ("LAPLACIAN(x*y^2*z^3)", "x*(6*y^2*z + 2*z^3)"),
    ("CURL([y^2, 2*x*z, 0])", "[-2*x, 0, 2*z - 2*y]"),
    # The curl of a plane field is one number, not the space vector `[0, 0, w]`
    # that number is the last element of.
    ("CURL([v^2, u], [[u, v], [1, 1]])", "1 - 2*v"),
    ("CURL([-y, x], [u, v])", "0"),
    # A two-row second argument is a coordinate geometry matrix: variables
    # above, scale factors below. The manual's spherical example, written out
    # rather than loaded from VECTOR.MTH, which is where the name lives. The
    # manual prints the middle element as `COT(phi)*COS(theta)`; a cotangent
    # comes out of the engine's canonical form as one over a tangent.
    (
        "GRAD(r*SIN(theta)*COS(phi), [[r, theta, phi], [1, r*SIN(phi), r]])",
        "[SIN(theta)*COS(phi), COS(theta)/TAN(phi), -SIN(theta)*SIN(phi)]",
    ),
    # Cylindrical coordinates, where the scale factors are what make the answers
    # differ from the Cartesian ones: `DIV` of a purely radial field grows with
    # the circumference the flux crosses.
    ("DIV([r^2, 0, 0], [[r, theta, z], [1, r, 1]])", "3*r"),
    ("LAPLACIAN(r^2, [[r, theta, z], [1, r, 1]])", "4"),
    ("CURL([0, r, 0], [[r, theta, z], [1, r, 1]])", "[0, 0, 2]"),
    ("POTENTIAL([y^2*z^3, 2*x*y*z^3, 3*x*y^2*z^2])", "x*y^2*z^3"),
    ("VECTOR_POTENTIAL([x, 0, y - z])", "[-y^2/2, -x*z, 0]"),
    # The starting coordinates of the line integrals, which default to the
    # origin. Starting at [1, 1, 1] rather than the origin subtracts the value
    # the potential would have had there, an additive constant and no more.
    ("POTENTIAL([2*x, 2*y, 2*z])", "x^2 + y^2 + z^2"),
    ("POTENTIAL([2*x, 2*y, 2*z], [1, 1, 1])", "x^2 + y^2 + z^2 - 3"),
    # The manual's warning about a bad starting point, exactly: this field is
    # infinite at the origin, so the potential from the origin is infinite too,
    # and 1 is the alternative the manual recommends for a logarithm. Legs that
    # diverge opposite ways have no sum.
    ("POTENTIAL([1/x, 1/y, 1/z], [1, 1, 1])", "LN(x) + LN(y) + LN(z)"),
    ("POTENTIAL([1/x, 1/y, 1/z])", "inf"),
    ("POTENTIAL([-1/x, 1/y, 0])", "POTENTIAL([-1/x, 1/y, 0])"),
    # A field of fewer elements than the system has coordinates is no field on
    # it, and the default system has three. Two elements are a planar curl only
    # when a planar system is given to work in.
    ("CURL([-y, x])", "CURL([-y, x])"),
    ("DIV([x, y])", "DIV([x, y])"),
    ("CURL([-y, x], [[x, y], [1, 1]])", "2"),
    # Not every field has a potential, and Derive does not check. POTENTIAL
    # computes one line integral and hands back its value; `[-y, x, 0]`
    # circulates, so the gradient of that value is not the field it came from.
    # The manual leaves that comparison to the caller, and CURL is the easier
    # test.
    ("CURL([-y, x, 0])", "[0, 0, 2]"),
    ("POTENTIAL([-y, x, 0])", "x*y"),
    ("GRAD(POTENTIAL([-y, x, 0]))", "[y, x, 0]"),
    # The same for a vector potential, whose obstruction is a nonzero
    # divergence: `[x, 0, 0]` spreads out, and the curl of the answer is not it.
    ("DIV([x, 0, 0])", "1"),
    ("VECTOR_POTENTIAL([x, 0, 0])", "[0, -x*z, 0]"),
    ("CURL(VECTOR_POTENTIAL([x, 0, 0]))", "[x, 0, -z]"),
    # A vector of expressions has one gradient per element, and those gradients
    # stand side by side as the columns of the answer: row `i` holds the rates
    # along coordinate `i` across the whole field. That is the transpose of
    # VECTOR.MTH's JACOBIAN, and it is the way round SOLVE.MTH's NEWTONS needs -
    # `APPEND(GRAD(u, x), [u])`` is the augmented matrix of one Newton step.
    ("GRAD([x*y, x + y], [x, y])", "[[y, 1], [x, 1]]"),
    ("GRAD([1, 2])", "[[0, 0], [0, 0], [0, 0]]"),
    # Arguments the operators can make nothing of: a single variable is no
    # coordinate system, curl and vector potential are defined in the plane and
    # in space and nowhere else. Each comes back the call it was written as.
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
    # What the manual says to check, on fields that pass it: POTENTIAL is only
    # a line integral, and it answers the question asked of it exactly when the
    # curl of the field is zero.
    assert simp(f"CURL({field})") == "[0, 0, 0]"
    assert simp(f"GRAD(POTENTIAL({field})) - {field}") == "[0, 0, 0]"


@pytest.mark.parametrize(
    "field",
    ["[-2*x, 0, 2*z - 2*y]", "[x, 0, y - z]", "[y, z, x]"],
    ids=str,
)
def test_a_solenoidal_field_is_the_curl_of_its_vector_potential(field):
    # The same check for VECTOR_POTENTIAL, whose condition is a vanishing
    # divergence. Two equally valid vector potentials differ by a gradient, so
    # it is the curl that has to be compared and not the vector itself.
    assert simp(f"DIV({field})") == "0"
    assert simp(f"CURL(VECTOR_POTENTIAL({field})) - {field}") == "[0, 0, 0]"


#: The four ways of saying which values a generated vector's variable takes.
#: The first three are the manual's own examples.
GENERATED = [
    ("VECTOR(x^2, x, 5)", "[1, 4, 9, 16, 25]"),
    ("VECTOR(j!, j, 0, 4)", "[1, 1, 2, 6, 24]"),
    ("VECTOR(k^2, k, [2, 3, 5, 7, 11])", "[4, 9, 25, 49, 121]"),
    # A step, and the count rounded down: four elements, the last one short of
    # pi/4. Exact, the manual's decimals being what it approximates to.
    ("VECTOR(SIN(z), z, 0, pi/4, 1/5)", "[0, SIN(1/5), SIN(2/5), SIN(3/5)]"),
    # A vector of vectors is a matrix, so a nested call generates one.
    (
        "VECTOR(VECTOR(j + k, k, 1, 4), j, 1, 3)",
        "[[2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]]",
    ),
    # The variable ranges over a matrix's rows, as every other element access
    # does.
    ("VECTOR(k, k, [[1, 2], [3, 4]])", "[[1, 2], [3, 4]]"),
    # A range the step cannot cross once has one element; one going the wrong
    # way has none.
    ("VECTOR(x^2, x, 3, 3)", "[9]"),
    ("VECTOR(x, x, 5, 1)", "[]"),
    # Bounds that are no sequence yet, and a second argument that is no
    # variable: the call itself is the answer.
    ("VECTOR(x^2, x, n)", "VECTOR(x^2, x, n)"),
    ("VECTOR(x^2, 5, 1, 3)", "VECTOR(x^2, 5, 1, 3)"),
    ("VECTOR(x^2, x, 1, 3, 0)", "VECTOR(x^2, x, 1, 3, 0)"),
]


@pytest.mark.parametrize(("text", "expected"), GENERATED, ids=str)
def test_generated_vectors(text, expected):
    assert simp(text) == expected


#: `SELECT` says which values in the same four ways, and answers with the
#: values themselves rather than with anything computed from them.
SELECTED = [
    # The manual's own two examples.
    ("SELECT(PRIME(k), k, [3, 5, 7, 9, 11])", "[3, 5, 7, 11]"),
    ("SELECT(PRIME(k), k, 1, 20)", "[2, 3, 5, 7, 11, 13, 17, 19]"),
    # A count, and a range with a step.
    ("SELECT(k > 1, k, 5)", "[2, 3, 4, 5]"),
    ("SELECT(k > 1, k, 1, 10, 3)", "[4, 7, 10]"),
    ("SELECT(k^2 > 5, k, 1, 5)", "[3, 4, 5]"),
    ("SELECT(k > 5, k, 1, 3)", "[]"),
    # A second argument that is no variable binds nothing, and a test nobody
    # can decide is no answer: both come back the call they were written as.
    # Derive answers the second with the empty vector, having dropped the
    # element it could not judge; this engine would rather say so.
    ("SELECT(k > 1, 5, [1, 2])", "SELECT(k > 1, 5, [1, 2])"),
    ("SELECT(PRIME(k), k, x, x + 2)", "SELECT(PRIME(k), k, x, x + 2)"),
]


@pytest.mark.parametrize(("text", "expected"), SELECTED, ids=str)
def test_selected_elements(text, expected):
    assert simp(text) == expected


#: The two iteration functions, which differ only in how much of the sequence
#: they answer with.
ITERATION = [
    # `n` updates make `n + 1` iterates; ITERATE keeps the last of them.
    ("ITERATES(x^2, x, 2, 3)", "[2, 4, 16, 256]"),
    ("ITERATE(x^2, x, 2, 3)", "256"),
    ("ITERATES(x^2, x, 2, 0)", "[2]"),
    # The manual's POWER: `n` copies of `x` multiplied into an accumulator.
    ("ITERATE(a*x, a, 1, 3)", "x^3"),
    # Uncounted, the sequence runs until a value comes round, and the repeat is
    # the last element. ITERATE wants a cycle of length one and answers `?`
    # where it finds a longer one.
    ("ITERATES(1/x, x, 2)", "[2, 1/2, 2]"),
    ("ITERATE(1/x, x, 2)", "?"),
    # A vector of variables and a vector of their starting values, which is how
    # the manual writes Fibonacci without subscripts.
    ("ITERATE([k, j + k], [j, k], [0, 1], 10)", "[55, 89]"),
    ("ITERATES([k, j + k], [j, k], [0, 1], 3)", "[[0, 1], [1, 1], [1, 2], [2, 3]]"),
    # A negative count iterates the inverse, which is what MISC.MTH defines
    # `INVERSE(u, x) := ITERATE(u, x, x, -1)` on.
    ("ITERATES(TAN(x), x, x, -1)", "[x, ATAN(x)]"),
    ("ITERATES(x + 1, x, 0, -2)", "[0, -1, -2]"),
    # Where there is a choice of inverses the principal one is taken, as Derive
    # takes it: a power is undone by the reciprocal power and a call by the
    # function that inverts it. MISC.MTH's INT_SUBST is what rests on it - the
    # manual substitutes `t^2` by inverting it to `SQRT(t)` (9.21, p.285) - so
    # declining the choice would leave one of the manual's own sessions unread.
    ("ITERATES(x^2, x, 2, -1)", "[2, SQRT(2)]"),
    # An iteration that neither comes round nor was counted comes back as
    # written: Derive runs one until memory is gone, which is no answer to
    # give. Neither is a count that is still a name.
    ("ITERATES(x^2, x, 2)", "ITERATES(x^2, x, 2)"),
    ("ITERATE(a*x, a, 1, n)", "ITERATE(a*x, a, 1, n)"),
]


@pytest.mark.parametrize(("text", "expected"), ITERATION, ids=str)
def test_iteration(text, expected):
    assert simp(text) == expected


def test_an_element_access_that_could_not_be_taken_is_taken_once_generated():
    # An access into a vector converts before the index is a number, and becomes
    # an inert one. Each generated element has a number for it, so the access is
    # worth making again - which is how the demo's outer product comes out.
    assert simp("VECTOR([[a, b, c] SUB i], i, 3)") == "[[a], [b], [c]]"
    assert simp("VECTOR([a, b, c] SUB i + 1, i, 2)") == "[a + 1, b + 1]"


#: A calculus head over the variable a counted call ranges over. The head is worked
#: out with the variable standing for itself, and only then is each value written in,
#: which is Derive's order.
COUNTED_CALCULUS = [
    ("VECTOR(DIF(x^2, x), x, 1, 3)", "[2, 4, 6]"),
    ("VECTOR(INT(x, x), x, 1, 3)", "[1/2, 2, 9/2]"),
    ("SUM(DIF(x^2, x), x, 1, 3)", "12"),
    ("PRODUCT(DIF(x^2, x), x, 1, 3)", "48"),
    # Newton's method for SQRT(3), which is SOLVE.MTH's `NEWTON` written out.
    ("ITERATES(x - (x^2 - 3)/DIF(x^2 - 3, x), x, 2, 3)", "[2, 7/4, 97/56, 18817/10864]"),
    # A head over another variable is untouched by the counting.
    ("VECTOR(DIF(x^2*y, x), y, 1, 2)", "[2*x, 4*x]"),
]


@pytest.mark.parametrize(("text", "expected"), COUNTED_CALCULUS, ids=str)
def test_a_calculus_head_in_a_counted_call_is_worked_out_before_counting(
    text, expected
):
    assert simp(text) == expected


def test_a_parameter_a_counted_call_differentiates_is_taken_as_supplied():
    # SOLVE.MTH's `NEWTON(u, x, x0, n)`: the iteration writes each value in for `x`,
    # and the `DIF(u, x)` inside differentiates the `u` the call supplied, in `x`.
    newton = (("u", "x", "a", "n"), parse("ITERATES(x - u/DIF(u, x), x, a, n)"))
    context = Context(functions={"NEWTON": newton})
    assert simp("NEWTON(x^2 - 3, x, 2, 3)", context) == "[2, 7/4, 97/56, 18817/10864]"
    assert simp("NEWTON(t^2 - 3, t, 2, 3)", context) == "[2, 7/4, 97/56, 18817/10864]"


# -- statistics ---------------------------------------------------------------

#: The three forms the statistical functions take - written-out arguments, one
#: vector, one matrix whose rows are each a sample - and the convention the
#: manual fixes: the variance is the unbiased sample one, over `n - 1`, and the
#: standard deviation is its square root.
STATISTICS = [
    # Both spellings of one sample, which is the manual's own pair.
    ("AVERAGE(2, 3, 4)", "3"),
    ("AVERAGE([2, 3, 4])", "3"),
    ("RMS([2, 3, 5])", "SQRT(114)/3"),
    # The manual's matrix example: the statistic of each row, as a vector.
    ("RMS([[3, 4, 5], [5, 12, 13]])", "[5*SQRT(6)/3, 13*SQRT(6)/3]"),
    ("AVERAGE([[1, 2], [3, 4]])", "[3/2, 7/2]"),
    # One sample answered by all four, which is what pins the convention down:
    # over `n - 1` the variance is 20/3, and over `n` it would be 5.
    ("AVERAGE(2, 4, 6, 8)", "5"),
    ("RMS(2, 4, 6, 8)", "SQRT(30)"),
    ("VAR(2, 4, 6, 8)", "20/3"),
    ("STDEV(2, 4, 6, 8)", "2*SQRT(15)/3"),
    # The formulas are written over `n` arguments for any `n`, so a sample of
    # one is a sample: its mean is itself, and its root mean square is its
    # magnitude.
    ("AVERAGE(x)", "x"),
    ("RMS(x)", "ABS(x)"),
    # A call nobody could make yet is no sample at all. Reading one as a sample
    # of one would answer `ABS` of a vector that has not been built, which is
    # what a definition holding `RMS(VECTOR(...))` would become.
    ("RMS(VECTOR(u, k, n))", "RMS(VECTOR(u, k, n))"),
    # A sample of one deviates from its own average by nothing, and Derive
    # answers the zero that sums to rather than the `0/0` the formula reads as.
    ("VAR([5])", "0"),
    ("VAR(x)", "0"),
    ("STDEV([5])", "0"),
]


@pytest.mark.parametrize(("text", "expected"), STATISTICS, ids=str)
def test_statistics(text, expected):
    assert simp(text) == expected


# -- what an expression is made of, and a vector rebuilt ----------------------

STRUCTURE = [
    # Syntactic terms, so the power is not multiplied out: the manual's own
    # example, and the reason it tells the caller to compose with EXPAND.
    ("TERMS(x*(a + b)^2 + c)", "[x*(a + b)^2, c]"),
    ("TERMS(x^2 + x + 1)", "[x^2, x, 1]"),
    # What is no sum has one term, and a vector distributes.
    ("TERMS(x*y)", "[x*y]"),
    ("TERMS([x + 1, y + 2])", "[[x, 1], [y, 2]]"),
    # Most main to least: the order list `x`, `y`, `z` first and the rest
    # alphabetically, which is the order every other command offers them in.
    ("VARIABLES(x^2 + a*y)", "[x, y, a]"),
    ("VARIABLES(a*b*z)", "[z, a, b]"),
    ("VARIABLES(5)", "[]"),
    # NUMBER is a predicate and answers a truth-value whichever way it comes
    # out. The manual's example is the second pair: true exactly where the sum
    # of two squares is a perfect square.
    ("NUMBER(2/3)", "true"),
    ("NUMBER(x)", "false"),
    ("NUMBER(SQRT(9))", "true"),
    ("NUMBER(SQRT(2))", "false"),
    # Which is what the utility files use it for: an argument still standing as
    # the variable it was written as is an argument nobody supplied.
    ("IF(NUMBER(d), d, 1)", "1"),
    # Counting from 1, and a matrix's elements are its rows.
    ("DELETE_ELEMENT([a, b, c], 2)", "[a, c]"),
    ("DELETE_ELEMENT([[1, 2], [3, 4], [5, 6]], 2)", "[[1, 2], [5, 6]]"),
    ("DELETE_ELEMENT([a], 1)", "[]"),
    # The value to write in comes first and the vector second; the index
    # defaults to 1.
    ("REPLACE_ELEMENT(d, [a, b, c], 2)", "[a, d, c]"),
    ("REPLACE_ELEMENT(d, [a, b, c])", "[d, b, c]"),
    ("REPLACE_ELEMENT([7, 8], [[1, 2], [3, 4]], 1)", "[[7, 8], [3, 4]]"),
    # An index that is no index yet, and one that is past the end: both come
    # back the call they were written as.
    ("DELETE_ELEMENT([a, b, c], n)", "DELETE_ELEMENT([a, b, c], n)"),
    ("DELETE_ELEMENT([a, b, c], 4)", "DELETE_ELEMENT([a, b, c], 4)"),
    # Inserted before the nth element, and the index defaults to 1. One past
    # the end is an index here where it is none above, because that is how an
    # element is added to the end.
    ("INSERT_ELEMENT(d, [a, b, c], 2)", "[a, d, b, c]"),
    ("INSERT_ELEMENT(d, [a, b, c])", "[d, a, b, c]"),
    ("INSERT_ELEMENT(d, [a, b, c], 4)", "[a, b, c, d]"),
    # Two past it is no index at all, and the call comes back as written -
    # DELETE_ELEMENT's rule above, and this engine's throughout. Derive answers
    # `[a, b, c]`, quietly dropping the element it was asked to insert.
    ("INSERT_ELEMENT(d, [a, b, c], 5)", "INSERT_ELEMENT(d, [a, b, c], 5)"),
    # Reversed, and a matrix's elements are its rows there as everywhere.
    ("REVERSE_VECTOR([a, b, c])", "[c, b, a]"),
    ("REVERSE_VECTOR([[1, 2], [3, 4]])", "[[3, 4], [1, 2]]"),
    # Concatenation: the elements of each vector, run together. Matrices stack,
    # their elements being their rows - except for a single matrix, which the
    # manual makes the exception and flattens.
    ("APPEND([a, b], [c, d], [e, f])", "[a, b, c, d, e, f]"),
    ("APPEND([[a, b], [c, d], [e, f]])", "[a, b, c, d, e, f]"),
    ("APPEND([[a, b], [c, d]], [[e, f], [g, h]])", "[[a, b], [c, d], [e, f], [g, h]]"),
    ("APPEND([a, b])", "[a, b]"),
    # A row of a matrix is a vector and a matrix of one row is that vector too,
    # so a vector as wide as a matrix it is appended to is one more row of it.
    # `NEWTONS` writes its system under its gradients that way.
    ("APPEND([[a, b], [c, d]], [e, f])", "[[a, b], [c, d], [e, f]]"),
    ("APPEND([e, f], [[a, b], [c, d]])", "[[e, f], [a, b], [c, d]]"),
]


@pytest.mark.parametrize(("text", "expected"), STRUCTURE, ids=str)
def test_structure(text, expected):
    assert simp(text) == expected


#: Section 6.14's other decomposition functions: what an expression is written
#: as the product of, which side of a relation is which, and the polynomial
#: division and gcd that sit beside `TERMS` in the manual.
DECOMPOSITION = [
    # Syntactic factors, `TERMS`'s counterpart: nothing is factored first, so a
    # sum is one factor and so is a number - `FACTOR` is what factors.
    ("FACTORS(3*x*(x + 1)^2)", "[(x + 1)^2, x, 3]"),
    ("FACTORS(x^2 - 1)", "[x^2 - 1]"),
    ("FACTORS(12)", "[12]"),
    ("FACTORS(2/3)", "[2/3]"),
    ("FACTORS(x/y)", "[x, 1/y]"),
    # Most main first, and the compound before the simple where two are about
    # the same variable.
    ("FACTORS(3*x*y)", "[x, y, 3]"),
    ("FACTORS(x^2*y^3)", "[x^2, y^3]"),
    ("FACTORS(x*SIN(x))", "[SIN(x), x]"),
    ("FACTORS((x + 1)*(x + 2))", "[x + 2, x + 1]"),
    # A vector distributes, as it does for TERMS.
    ("FACTORS([x*y, 2*z])", "[[x, y], [z, 2]]"),
    # The sides of a relation, and a vector of them - which is what makes
    # `RHS(SOLVE(u, x))` the vector of roots.
    ("LHS(2*x + 3 = 5)", "2*x + 3"),
    ("RHS(2*x + 3 = 5)", "5"),
    ("LHS([x = 1, y = 2])", "[x, y]"),
    ("RHS([x = 1, y = 2])", "[1, 2]"),
    # What is no relation has no sides to take and answers with itself.
    ("LHS(2*x + 3)", "2*x + 3"),
    ("RHS(3)", "3"),
    # Division in the main variable: the manual's own pair.
    ("QUOTIENT(x^4 + 3*x^3 + 5*x + 6, x^2 - 5)", "x^2 + 3*x + 5"),
    ("REMAINDER(x^4 + 3*x^3 + 5*x + 6, x^2 - 5)", "20*x + 31"),
    ("QUOTIENT(x, x^2)", "0"),
    ("REMAINDER(x, x^2)", "x"),
    # The other variables ride along in the coefficients as a field would: in
    # `x` the divisor is a constant, and a constant divides exactly.
    ("QUOTIENT(x*y + 1, y)", "x + 1/y"),
    ("REMAINDER(x*y + 1, y)", "0"),
    # Two numbers have no variable to divide in and are a field on their own,
    # so the quotient is the fraction and nothing is left over.
    ("QUOTIENT(7, 2)", "7/2"),
    ("REMAINDER(7, 2)", "0"),
    ("QUOTIENT(-7, 2)", "-7/2"),
    # And what is a polynomial in a kernel is divided in the kernel.
    ("QUOTIENT(SIN(x), SIN(x)^2)", "0"),
    ("REMAINDER(SIN(x), SIN(x)^2)", "SIN(x)"),
    ("QUOTIENT(SIN(x), x)", "SIN(x)/x"),
    # The manual's gcd example, and the same function over numbers.
    ("POLY_GCD(x^3 + 3*x^2 + 5*x + 6, x^3 + 2*x - 3)", "x^2 + x + 3"),
    ("POLY_GCD(12, 18)", "6"),
]


@pytest.mark.parametrize(("text", "expected"), DECOMPOSITION, ids=str)
def test_what_an_expression_decomposes_into(text, expected):
    assert simp(text) == expected


#: The two number-theoretic functions, both of them predicates over the
#: integers in the sense that they answer rather than compute.
NUMBER_THEORY = [
    ("PRIME(7)", "true"),
    ("PRIME(8)", "false"),
    ("PRIME(1)", "false"),
    # The manual's second argument is how many rounds of a probabilistic test
    # to run. This test is not probabilistic, so the count changes nothing.
    ("PRIME(7, 20)", "true"),
    # What is no integer is not false but undecided, and comes back as written.
    ("PRIME(x)", "PRIME(x)"),
    # Strictly larger, and the argument need be no integer.
    ("NEXT_PRIME(1000)", "1009"),
    ("NEXT_PRIME(7)", "11"),
    ("NEXT_PRIME(7/2)", "5"),
]


@pytest.mark.parametrize(("text", "expected"), NUMBER_THEORY, ids=str)
def test_number_theory(text, expected):
    assert simp(text) == expected


def test_the_utility_files_appended_columns():
    # ``APPEND_COLUMNS(A, B) := APPEND(A`, B`)` ``, VECTOR.MTH's definition and
    # the exercise the manual sets. It works only because appending two
    # matrices stacks their rows rather than flattening them.
    columns = "APPEND([[a, b], [c, d]]`, [[e, f], [g, h]]`)`"
    assert simp(columns) == "[[a, b, e, f], [c, d, g, h]]"


def test_the_utility_files_matrix_minor():
    # ``MINOR(a,i,j) := DELETE_ELEMENT(DELETE_ELEMENT(a,i)`,j)` ``, VECTOR.MTH's
    # definition and the exercise the manual sets: delete row `i`, transpose,
    # delete what was column `j`, transpose back.
    body = parse("DELETE_ELEMENT(DELETE_ELEMENT(a,i)`,j)`")
    context = Context(functions={"MINOR": (("a", "i", "j"), body)})
    assert simp("MINOR([[1,2,3],[4,5,6],[7,8,9]],2,3)", context) == "[[1, 2], [7, 8]]"


def test_the_demos_matrix_generator():
    # `MATRIX(z,i,m,j,n) := VECTOR(VECTOR(z,j,1,n),i,1,m)`, the definition the
    # demo builds out of two nested generated vectors.
    body = parse("VECTOR(VECTOR(z,j,1,n),i,1,m)")
    context = Context(functions={"MATRIX": (("z", "i", "m", "j", "n"), body)})
    assert simp("MATRIX(i-j,i,2,j,3)", context) == "[[0, -1, -2], [1, 0, -1]]"


def test_the_demos_outer_product():
    # `OUTER(v,w) := VECTOR([v SUB i],i,DIMENSION(v)) . [w]`. The generator's
    # length is a call on its own argument, so nothing generates until the
    # definition is applied.
    body = parse("VECTOR([v SUB i],i,DIMENSION(v)) . [w]")
    context = Context(functions={"OUTER": (("v", "w"), body)})
    assert simp("OUTER([a,b,c],[2,3,4])", context) == (
        "[[2*a, 3*a, 4*a], [2*b, 3*b, 4*b], [2*c, 3*c, 4*c]]"
    )

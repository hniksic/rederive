"""Static name inventories.

Pure data, no logic. These sets are what make the name-resolution procedure of
`lexer.py` work: an incomplete list produces silently wrong parses, not errors.
"""

from __future__ import annotations

# The functions Derive itself defines.
DERIVE_FUNCTIONS: frozenset[str] = frozenset(
    """
    ABS ACOS ACOSH ACOT ACOTH ACSC ACSCH APPEND APPROX ASEC ASECH ASIN ASINH
    ATAN ATANH AVERAGE CHARPOLY CHI COMB CONJ COS COSH COT COTH CROSS CSC CSCH
    CURL DELETE_ELEMENT DENOMINATOR DET DIF DIMENSION DIV EIGENVALUES ELEMENT ERF
    ERFC EXP EXPAND FACTOR FACTORS FIT FLOOR FVAL GAMMA GCD GRAD IDENTITY_MATRIX
    IF IM INSERT_ELEMENT INT INTERVAL ITERATE ITERATES LAPLACIAN LCM LHS LIM LN LOG
    MAX MIN MOD MODS NEXT_PRIME NORMAL NPER NUMBER NUMERATOR PERM PHASE PMT POLY_GCD
    POTENTIAL PRIME PRODUCT PVAL QUOTIENT RANDOM RATE RE REMAINDER REPLACE_ELEMENT
    REVERSE_VECTOR RHS RMS ROOT_OF ROOT_SUM ROW_REDUCE SELECT SEC SECH SIGN SIN
    SINH SOLVE SQRT STDEV STEP SUM TAN TANH TAYLOR TERMS TRACE TRUTH_TABLE VAR
    VARIABLES VECTOR VECTOR_POTENTIAL ZETA
    """.split()
)

# The special functions rederive adds to Derive's own. Each arises from rederive's own
# operations on elementary input - `INT(EXP(x^2), x)` is `SQRT(pi)*ERFI(x)/2`,
# `DIF(GAMMA(x), x)` is `GAMMA(x)*DIGAMMA(x)`, `INT(SIN(x)/x, x)` is `SI(x)` - and each
# is standard across computer algebra systems, so an answer that holds one is written in
# a name the author can type back, in either case. The lexer reads them exactly as it
# reads Derive's own, inside a longer run of letters as well: Derive finds every function
# it knows that way, its own or one the author defines, so with `CI` known `velocity` is
# `v*e*l*o*CI(t)*y`.
EXTENDED_FUNCTIONS: frozenset[str] = frozenset(
    """
    CI COSH_INT DIGAMMA EI ERFI LAMBERTW POLYGAMMA POLYLOG SI SINH_INT
    """.split()
)

# The notation rederive adds for what a derivative leaves behind: `SUBS(u, [x], [a])` is
# `u` with `x` set to `a` after `u` is worked out, which is how a derivative of a function
# of several arguments, without a body, is written at a point other than its variable:
# `DIF(G(x^2, y), x)` is `2*x*SUBS(DIF(G(x_, y), x_), [x_], [x^2])`. An answer can hold
# one, so it is a name the author can type back, in either case, like the others.
NOTATION_FUNCTIONS: frozenset[str] = frozenset("SUBS".split())

BUILTIN_FUNCTIONS: frozenset[str] = (
    DERIVE_FUNCTIONS | EXTENDED_FUNCTIONS | NOTATION_FUNCTIONS
)

# The functions whose second argument names a variable of their own: the index
# of a sum, the variable a derivative is taken with respect to, the one a
# generated vector steps through. Inside the first argument that name stands for
# itself, whatever else it may have been assigned - and everything after it is
# read outside the binding, which is what lets `ITERATES(TAN(x), x, x, -1)` mean
# the outer `x` in its third argument and the bound one in its first.
BINDING_FUNCTIONS: frozenset[str] = frozenset(
    """
    DIF INT ITERATE ITERATES LIM PRODUCT ROOT_OF ROOT_SUM SELECT SUM TAYLOR
    VECTOR
    """.split()
)

# The binding functions whose name covers every argument rather than the first
# alone. A sum's limits are outer expressions and a root's index is a number, but
# the polynomial a root sum runs over is written in the very variable its summand
# names: `ROOT_SUM(p, t, u)` means the same `t` in `p` as in `u`, and an assigned
# `t` may reach neither.
BINDING_THROUGHOUT: frozenset[str] = frozenset("ROOT_SUM".split())

# `e` and `i` are not here: they are ordinary free variables. `?` is the
# unknown value and has its own token.
CONSTANTS: frozenset[str] = frozenset(
    "pi inf deg #e #i true false euler_gamma unit_circle".split()
)

# Pre-defined variable names, so they lex as one token in Character mode.
# `gamma`, `zeta` and `chi` are absent on purpose: those are the functions
# GAMMA, ZETA and CHI. `pi` is a constant, not a variable.
GREEK_VARIABLES: frozenset[str] = frozenset(
    "alpha beta delta epsilon theta mu phi sigma tau omega".split()
)

KEYWORD_OPERATORS: frozenset[str] = frozenset("SUB AND OR XOR IMP NOT".split())

# The derivative mark, written on a function name and applied: `BB'(r)` is the
# derivative of `BB`, taken at `r`, and one mark is one order. It is part of the
# name rather than an operator of its own, which is why it is here and not in
# `OPERATORS`. Transposition is the backtick, so the apostrophe is free for
# this and means nothing anywhere else.
PRIME_MARK = "'"

# Settings the parser must accept as ordinary assignments. The first three
# change how later expressions lex; the rest are passed through untouched.
PARSING_SETTINGS: frozenset[str] = frozenset("InputMode CaseMode InputBase".split())

SETTINGS: frozenset[str] = PARSING_SETTINGS | frozenset(
    """
    Precision PrecisionDigits Notation NotationDigits OutputBase DisplayFormat
    TimesOperator Angle Trigonometry Trigpower Exponential Logarithm Branch
    ArrowKeyMode
    """.split()
)

# The keyword values settings take. They must be reserved because they appear
# on the left of `:=` and as bare keyword arguments: `FACTOR(w, Rational, y2)`.
SETTING_VALUES: frozenset[str] = frozenset(
    """
    Character Word Insensitive Sensitive LineEdit Subexpression Binary Octal
    Decimal Hexadecimal Exact Approximate Mixed Rational Scientific Normal
    Compressed Asterisk Dot Implicit Degree Radian Auto Collect Expand Sines
    Cosines Principal Real Any Integer Complex Nonscalar Trivial Squarefree
    raDical
    """.split()
)

DOMAINS: frozenset[str] = frozenset("Integer Real Complex Nonscalar".split())

# Glyph spellings of function names. Resolved in the lexer, so they introduce
# no grammar of their own: `√x^2` obeys the bare-application rule exactly as
# `SQRT x^2` does.
FUNCTION_GLYPHS: dict[str, str] = {
    "√": "SQRT",  # square root sign
    "Γ": "GAMMA",  # capital gamma
    "Σ": "SUM",  # capital sigma
    "Π": "PRODUCT",  # capital pi
}

# Glyph spellings of constants.
CONSTANT_GLYPHS: dict[str, str] = {
    "π": "pi",  # small pi
    "∞": "inf",  # infinity
    "°": "deg",  # degree sign
    "ê": "#e",  # e with circumflex
    "î": "#i",  # i with circumflex
}

# The spelling each pre-defined Greek variable prints as: the letter itself,
# lower case, as Derive's manual sets it.
GREEK_CANONICAL: dict[str, str] = {
    "alpha": "α",
    "beta": "β",
    "delta": "δ",
    "epsilon": "ε",
    "theta": "θ",
    "mu": "μ",
    "sigma": "σ",
    "tau": "τ",
    "phi": "φ",
    "omega": "ω",
}

# Every spelling that resolves to one of those names. Derive held Greek letters
# in the IBM PC character set, whose Greek run decodes to `ß` for beta, `Θ` for
# theta, `Φ` for phi, `Ω` for omega and the micro sign for mu - Derive's own
# GRAPHICS.MTH writes `Θn` for a theta-subscripted parameter. Text that came
# through that decoding has to name the same variables as the Greek letters do.
GREEK_GLYPHS: dict[str, str] = {glyph: name for name, glyph in GREEK_CANONICAL.items()} | {
    "ß": "beta",  # sharp s, U+00DF
    "Θ": "theta",  # capital theta
    "µ": "mu",  # micro sign, U+00B5
    "Φ": "phi",  # capital phi
    "Ω": "omega",  # capital omega
}

# Operator glyphs and their canonical ASCII spelling. `·` U+00B7 is
# multiplication and `∙` U+2219 is the dot product: different characters with
# different meanings.
OPERATOR_GLYPHS: dict[str, str] = {
    "≤": "<=",
    "≥": ">=",
    "·": "*",
    "∙": ".",
    "±": "+-",
    "│": "|",
    ":ε": ":epsilon",
    ":∈": ":epsilon",
}

# The domain operator, in every spelling. `ε` on its own is the name
# `epsilon`, not an operator: `q ε Real` is `q*epsilon*Real`.
DOMAIN_OPERATORS: tuple[str, ...] = (":epsilon", ":ε", ":∈")

# Longest match wins, so this list is ordered longest first within each
# leading character.
OPERATORS: tuple[str, ...] = (
    ":==",
    ":=",
    ":epsilon",
    ":ε",
    ":∈",
    "<=",
    ">=",
    "/=",
    "≤",
    "≥",
    "=",
    "<",
    ">",
    "+",
    "-",
    "*",
    "/",
    "^",
    ".",
    "!",
    "%",
    "`",
    "|",
    "~",
    "(",
    ")",
    "[",
    "]",
    ",",
    "?",
    "·",
    "∙",
    "±",
    "│",
)

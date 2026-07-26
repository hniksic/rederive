"""Static name inventories.

Pure data, no logic. These sets are what make the name-resolution procedure of
`lexer.py` work: an incomplete list produces silently wrong parses, not errors.
"""

from __future__ import annotations

BUILTIN_FUNCTIONS: frozenset[str] = frozenset(
    """
    ABS ACOS ACOSH ACOT ACOTH ACSC ACSCH APPEND APPROX ASEC ASECH ASIN ASINH
    ATAN ATANH AVERAGE CHARPOLY CHI COMB CONJ COS COSH COT COTH CROSS CSC CSCH
    CURL DELETE_ELEMENT DENOMINATOR DET DIF DIMENSION DIV EIGENVALUES ELEMENT ERF
    ERFC EXP EXPAND FACTOR FACTORS FIT FLOOR FVAL GAMMA GCD GRAD IDENTITY_MATRIX
    IF IM INSERT_ELEMENT INT ITERATE ITERATES LAPLACIAN LCM LHS LIM LN LOG MAX MIN
    MOD MODS NEXT_PRIME NORMAL NPER NUMBER NUMERATOR PERM PHASE PMT POLY_GCD
    POTENTIAL PRIME PRODUCT PVAL QUOTIENT RANDOM RATE RE REMAINDER REPLACE_ELEMENT
    REVERSE_VECTOR RHS RMS ROW_REDUCE SELECT SEC SECH SIGN SIN SINH SOLVE SQRT
    STDEV STEP SUM TAN TANH TAYLOR TERMS TRACE TRUTH_TABLE VAR VARIABLES
    VECTOR VECTOR_POTENTIAL ZETA
    """.split()
)

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

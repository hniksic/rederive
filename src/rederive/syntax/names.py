"""Static name inventories.

Pure data used by name resolution in `lexer.py`. An incomplete list causes silently
wrong parses, not errors.
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

# Special functions rederive adds to Derive's. They arise from operations on elementary
# input (`INT(EXP(x^2), x)` is `SQRT(pi)*ERFI(x)/2`, `DIF(GAMMA(x), x)` is
# `GAMMA(x)*DIGAMMA(x)`, `INT(SIN(x)/x, x)` is `SI(x)`), and use names standard across
# computer algebra systems, so the user can type answers back in either case. The lexer
# treats them like Derive's own functions, including inside a longer run of letters: with
# `CI` known, `velocity` is `v*e*l*o*CI(t)*y`.
EXTENDED_FUNCTIONS: frozenset[str] = frozenset(
    """
    CI COSH_INT DIGAMMA EI ERFI LAMBERTW POLYGAMMA POLYLOG SI SINH_INT
    """.split()
)

# Notation rederive adds for derivatives: `SUBS(u, [x], [a])` is `u`, evaluated, with `x`
# set to `a`. It expresses the derivative of an undefined multi-argument function at a
# point: `DIF(G(x^2, y), x)` is `2*x*SUBS(DIF(G(x_, y), x_), [x_], [x^2])`. Answers can
# contain it, so the user can type it back.
NOTATION_FUNCTIONS: frozenset[str] = frozenset("SUBS".split())

BUILTIN_FUNCTIONS: frozenset[str] = (
    DERIVE_FUNCTIONS | EXTENDED_FUNCTIONS | NOTATION_FUNCTIONS
)

# Builtin functions the user can't redefine; Derive rejects `SIN(x):=x+1` with a syntax
# error at the `:=`. The exceptions are functions defined by older utility files (2.01's
# MISC.MTH and VECTOR.MTH, 1.62's SPECIAL.MTH) from before they were built in, so those
# files still load. Rederive's extended functions stay definable, since EXP_INT.MTH,
# PROBABIL.MTH and ZETA.MTH define most of them.
UNDEFINABLE_FUNCTIONS: frozenset[str] = (
    DERIVE_FUNCTIONS - frozenset("DELETE_ELEMENT FLOOR MOD REVERSE_VECTOR ZETA".split())
) | NOTATION_FUNCTIONS

# Functions whose second argument names a bound variable, such as a sum's index or a
# derivative's variable. Within the first argument that name stands for itself, even if
# it has a value. Later arguments are outside the binding, so `ITERATES(TAN(x), x, x, -1)`
# uses the bound `x` in its first argument and the outer `x` in its third.
BINDING_FUNCTIONS: frozenset[str] = frozenset(
    """
    DIF INT ITERATE ITERATES LIM PRODUCT ROOT_OF ROOT_SUM SELECT SUM TAYLOR
    VECTOR
    """.split()
)

# Binding functions whose variable is bound in every argument, not just the first:
# `ROOT_SUM(p, t, u)` uses the same bound `t` in the polynomial `p` and the summand `u`.
BINDING_THROUGHOUT: frozenset[str] = frozenset("ROOT_SUM".split())

# `e` and `i` aren't here because they are ordinary variables. `?`, the unknown value,
# has its own token.
CONSTANTS: frozenset[str] = frozenset(
    "pi inf deg #e #i true false euler_gamma unit_circle".split()
)

# Predefined variable names, so they lex as one token in Character mode. `gamma`, `zeta`
# and `chi` are deliberately absent because they are functions, and `pi` is a constant.
GREEK_VARIABLES: frozenset[str] = frozenset(
    "alpha beta delta epsilon theta mu phi sigma tau omega".split()
)

KEYWORD_OPERATORS: frozenset[str] = frozenset("SUB AND OR XOR IMP NOT".split())

# The derivative mark: `BB'(r)` is the derivative of `BB` at `r`, one mark per order. It
# is part of the name, not an operator, so it's not in `OPERATORS`. (Transposition uses
# the backtick.)
PRIME_MARK = "'"

# Settings the parser accepts as ordinary assignments. These three change how later
# expressions lex; the other `SETTINGS` pass through untouched.
PARSING_SETTINGS: frozenset[str] = frozenset("InputMode CaseMode InputBase".split())

SETTINGS: frozenset[str] = PARSING_SETTINGS | frozenset(
    """
    Precision PrecisionDigits Notation NotationDigits OutputBase DisplayFormat
    TimesOperator Angle Trigonometry Trigpower Exponential Logarithm Branch
    ArrowKeyMode
    """.split()
)

# Keyword values of settings. They are reserved because they appear on the left of `:=`
# and as bare keyword arguments, as in `FACTOR(w, Rational, y2)`.
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

# Glyph spellings of function names. They are resolved in the lexer and have no grammar
# of their own: `√x^2` parses exactly like `SQRT x^2`.
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

# How each predefined Greek variable prints: as the lowercase letter, as in Derive's
# manual.
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

# Every glyph that resolves to one of those names. Derive stored Greek letters in the IBM
# PC character set, whose Greek characters decode to `ß` for beta, `Θ` for theta, `Φ` for
# phi, `Ω` for omega and the micro sign for mu (GRAPHICS.MTH writes `Θn`, for example).
# Text decoded that way must name the same variables.
GREEK_GLYPHS: dict[str, str] = {glyph: name for name, glyph in GREEK_CANONICAL.items()} | {
    "ß": "beta",  # sharp s, U+00DF
    "Θ": "theta",  # capital theta
    "µ": "mu",  # micro sign, U+00B5
    "Φ": "phi",  # capital phi
    "Ω": "omega",  # capital omega
}

# Operator glyphs and their canonical ASCII spelling. Note that `·` U+00B7 is
# multiplication and `∙` U+2219 is the dot product.
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

# All spellings of the domain operator. A lone `ε` is the name `epsilon`, not an
# operator: `q ε Real` is `q*epsilon*Real`.
DOMAIN_OPERATORS: tuple[str, ...] = (":epsilon", ":ε", ":∈")

# Longest match wins, so within each leading character this lists longer operators first.
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

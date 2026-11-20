"""Static name inventories.

Pure data used by name resolution in `lexer.py`. An incomplete list causes silently
wrong parses, not errors.
"""

from __future__ import annotations

# The functions of Derive 6 that Rederive evaluates, by their Derive 6 names.
DERIVE_FUNCTIONS: frozenset[str] = frozenset(
    """
    ABS ACOS ACOSH ACOT ACOTH ACSC ACSCH ADJOIN APPEND APPROX ARCCOS ARCCOT ARCCSC
    ARCSEC ARCSIN ARCTAN ASEC ASECH ASIN ASINH ATAN ATANH AVERAGE CEILING CHARPOLY CHI
    CODES_TO_NAME COMB CONJ COS COSH COT COTH CROSS CSC CSCH CURL DELETE DENOMINATOR DET
    DIF DILOG DIM DIV EIGENVALUES ELEMENT ERF ERFC EVEN? EVERY EXP EXPAND FACTOR FACTORS
    FIRST FIT FLOOR FVAL GAMMA GCD GRAD GROEBNER_BASIS IDENTICAL? IDENTITY_MATRIX IF IM
    INSERT INT INTEGER? INTEGER_TYPE? INTERVAL INVERSE_MOD ITERATE ITERATES LAPLACIAN
    LCM LHS LIM LN LOG LOGICAL? LOGICAL_TYPE? MAP MAP_LIST MATRIX? MAX MEMBER? MIN MOD
    MODS NAME_TO_CODES NEXT_PRIME NORMAL NPER NSOLUTIONS NSOLVE NUMBER NUMBER?
    NUMBER_TYPE? NUMERATOR ODD? PERM PHASE PMT POLY_GCD POLY_MOD POLY_MODS POSITION
    POTENTIAL POWER? POWER_MOD POWER_SET PREVIOUS_PRIME PRIME PRIME? PRODUCT PRODUCT?
    PVAL QUOTIENT RANDOM RANK RATE RATIONAL? RE REAL_TYPE? REMAINDER REPLACE REST
    REVERSE RHS RMS ROOT_OF ROOT_SUM ROUND ROW_REDUCE SEC SECH SELECT SET? SET_TYPE?
    SHIFT SIGN SIN SINH SOLUTIONS SOLVE SOME SORT SQRT STDEV STEP STRING STRING? SUBST
    SUBST_DIFF SUM SUM? TABLE TAN TANH TAYLOR TERMS TRACE TRUTH_TABLE VAR VARIABLE?
    VARIABLES VARIANCE VECTOR VECTOR? VECTOR_POTENTIAL VECTOR_TYPE? ZETA
    """.split()
)

# The built-in functions of Derive 6 that Rederive does not evaluate yet. They are names
# all the same: the lexer knows them, so `DIM` is one name in Character mode and `PROG(x)`
# a call in any mode, a definition of one is refused, and a call stays as written with
# its arguments simplified. A built-in is added by moving its name to `DERIVE_FUNCTIONS`
# and giving it a converter in `engine/to_sympy.py`'s `FUNCTIONS`.
PENDING_FUNCTIONS: frozenset[str] = frozenset()

# The program constructs of Derive 6, which the evaluator runs rather than converts: their
# arguments are statements, evaluated in an order and as often as the construct says, so
# they have no converter. With them are the built-ins done for their effect on the
# session or the screen: showing text, and loading a file. A definition of one is
# refused, like any built-in's.
PROGRAM_FUNCTIONS: frozenset[str] = frozenset(
    "ASSIGN DISPLAY LOAD LOOP PROG RETURN WRITE".split()
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

# The functions Rederive evaluates.
BUILTIN_FUNCTIONS: frozenset[str] = (
    DERIVE_FUNCTIONS | EXTENDED_FUNCTIONS | NOTATION_FUNCTIONS
)

# Old names of Derive's functions that Derive 6 still accepts, which are replaced by the
# Derive 6 name when an expression is read, so an entry and a saved file show the new
# name.
REWRITTEN_ALIASES: dict[str, str] = {
    "DIMENSION": "DIM",
    "DELETE_ELEMENT": "DELETE",
    "REPLACE_ELEMENT": "REPLACE",
    "INSERT_ELEMENT": "INSERT",
    "REVERSE_VECTOR": "REVERSE",
}

# Old names that keep their spelling in the tree and compute as the Derive 6 function
# named here. `ELEMENT` and `CROSS` have converters of their own.
KEPT_ALIASES: dict[str, str] = {
    "PRIME": "PRIME?",
    "NUMBER": "NUMBER?",
    "VAR": "VARIANCE",
}

# Every function name the lexer knows without a declaration.
KNOWN_FUNCTIONS: frozenset[str] = (
    BUILTIN_FUNCTIONS
    | PROGRAM_FUNCTIONS
    | PENDING_FUNCTIONS
    | frozenset(REWRITTEN_ALIASES)
)

# Built-ins that a definition of the same spelling takes over for the session: functions
# older utility files defined before they were built in, which those files still define.
# An empty definition gives the name back to the built-in.
YIELDING_FUNCTIONS: frozenset[str] = frozenset(
    "DELETE_ELEMENT FLOOR MOD REVERSE_VECTOR VAR ZETA".split()
)

# Builtin functions the user can't redefine in their exact spelling; Derive rejects
# `SIN(x):=x+1` with a syntax error at the `:=`. Another spelling, `sin(x) := x + 1`, is
# another name.
UNDEFINABLE_FUNCTIONS: frozenset[str] = KNOWN_FUNCTIONS - YIELDING_FUNCTIONS

# Functions whose second argument names a bound variable, such as a sum's index or a
# derivative's variable. Within the first argument that name stands for itself, even if
# it has a value. Later arguments are outside the binding, so `ITERATES(TAN(x), x, x, -1)`
# uses the bound `x` in its first argument and the outer `x` in its third.
BINDING_FUNCTIONS: frozenset[str] = frozenset(
    """
    DIF EVERY INT ITERATE ITERATES LIM MAP MAP_LIST PRODUCT ROOT_OF ROOT_SUM SELECT
    SOME SUBST_DIFF SUM TABLE TAYLOR VECTOR
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

KEYWORD_OPERATORS: frozenset[str] = frozenset("SUB AND OR XOR IMP NOT".split())

# The word operators of Derive 6. Each is recognized only as a whole run of letters, and
# only in its upper-case spelling, except where an operator can stand: there another
# spelling (`row`, `iff`) is the operator too, unless a user name has that exact
# spelling. Anywhere else such a spelling is an ordinary name. `DEG` is postfix, the
# others infix. The spelling `deg` is the constant in operand position.
WORD_OPERATORS: frozenset[str] = frozenset(
    "IFF SUBSET ROW COL DOTPRODUCT CROSSPRODUCT DEG".split()
)

# The word operators that stand between two operands.
INFIX_WORDS: frozenset[str] = WORD_OPERATORS - {"DEG"}

# The keyword that ends a `LOOP` or a `PROG`. It stands alone as a primary, and is
# spelled in lower case; `EXIT` is the same keyword.
EXIT = "exit"

# Words that can't be bound, and that are found only as a whole run, never inside a
# longer one: `returns` is no `RETURN`.
RESERVED_WORDS: frozenset[str] = WORD_OPERATORS | frozenset(
    ("RETURN", EXIT, EXIT.upper())
)

# Keyword operators a definition displaces. Utility files from before sets were built
# in define `UNION(u, v)` and `INTERSECTION(u, v)` on vectors, and keep working.
DISPLACEABLE_OPERATORS: frozenset[str] = frozenset("UNION INTERSECTION".split())

# The operators on sets, as the parser's node values: the union, the intersection and
# the difference `s \ t`, loosest first.
SET_OPERATORS: tuple[str, ...] = ("UNION", "INTERSECTION", "\\")

# The ellipsis that stands for the elements of an arithmetic sequence in a vector or a
# set: `[0, 2, ..., 10]`.
ELLIPSIS = "..."

# The apostrophe. Directly after a name and before `(` or another mark it is a derivative
# mark: `BB'(r)` is the derivative of `BB` at `r`, one mark per order, and the marks are
# part of the name. Where an operand may start it is the quote, `'u`. So it is not in
# `OPERATORS`. (Transposition uses the backtick.)
PRIME_MARK = "'"

# Settings the parser accepts as ordinary assignments. These three change how later
# expressions lex; the other `SETTINGS` pass through untouched.
PARSING_SETTINGS: frozenset[str] = frozenset("InputMode CaseMode InputBase".split())

# The state variables of Derive 6. An assignment to one is a setting, never a variable,
# whether or not Rederive has the setting: `DisplaySteps` and `SimplifyRelations` are
# recognized and have no effect, and `ScientificThreshold`, which Derive reserved
# without using, is ignored.
SETTINGS: frozenset[str] = PARSING_SETTINGS | frozenset(
    """
    Precision PrecisionDigits Notation NotationDigits OutputBase DisplayFormat
    TimesOperator Angle Trigonometry Trigpower Exponential Logarithm Branch
    ArrowKeyMode VariableOrder DisplaySteps SimplifyRelations ScientificThreshold
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
    raDical Turing Schmidt
    """.split()
)

# The values each setting takes, for telling a setting line in another case spelling
# (`inputmode:=word`) from an assignment to a variable (`angle:=30`). `NUMBER` stands for
# an integer, `VECTOR` for a vector, and `true` and `false` for themselves.
NUMBER = "#number"
VECTOR = "#vector"
_DIRECTIONS = frozenset("Auto Collect Expand".split())
_RADICES = frozenset(("Decimal", "Binary", "Octal", "Hexadecimal", NUMBER))
SETTING_CHOICES: dict[str, frozenset[str]] = {
    "Precision": frozenset("Exact Approximate Mixed".split()),
    "PrecisionDigits": frozenset((NUMBER,)),
    "Notation": frozenset("Rational Decimal Mixed Scientific".split()),
    "NotationDigits": frozenset((NUMBER,)),
    "Branch": frozenset("Principal Real Any".split()),
    "Exponential": _DIRECTIONS,
    "Logarithm": _DIRECTIONS,
    "Trigonometry": _DIRECTIONS,
    "Trigpower": frozenset("Auto Sines Cosines".split()),
    "Angle": frozenset("Radian Degree".split()),
    "CaseMode": frozenset("Insensitive Sensitive".split()),
    "VariableOrder": frozenset((VECTOR,)),
    "OutputBase": _RADICES,
    "InputBase": _RADICES,
    "InputMode": frozenset("Character Word".split()),
    "DisplayFormat": frozenset("Normal Compressed".split()),
    "TimesOperator": frozenset("Dot Asterisk Implicit".split()),
    "DisplaySteps": frozenset("true false".split()),
    "SimplifyRelations": frozenset("true false".split()),
    "ArrowKeyMode": frozenset("LineEdit Subexpression".split()),
    "ScientificThreshold": frozenset((NUMBER,)),
}

DOMAINS: frozenset[str] = frozenset(
    "Integer Real Complex Nonscalar Vector Set Logical".split()
)

# Glyph spellings of function names. They are resolved in the lexer and have no grammar
# of their own: `√x^2` parses exactly like `SQRT x^2`.
#
# Derive 6 spells built-in names and operators with the characters its help lists as
# "Reserved Unicode Characters", some of them in the Private Use Area. They appear in its
# worksheets and, written as `\x7fXXXX` escapes, in its math files.
FUNCTION_GLYPHS: dict[str, str] = {
    "√": "SQRT",  # square root sign
    "Γ": "GAMMA",  # capital gamma
    "Σ": "SUM",  # capital sigma
    "Π": "PRODUCT",  # capital pi
    "∑": "SUM",  # n-ary summation, U+2211
    "∏": "PRODUCT",  # n-ary product, U+220F
    "∂": "DIF",  # partial differential
    "∫": "INT",  # integral
    "\ue7c4": "GAMMA",  # Derive's gamma function
    "\ue7c5": "ZETA",  # Derive's zeta function
    "\ue7c6": "CHI",  # Derive's chi function
    "\ue7c7": "DIGAMMA",  # Derive's digamma function
    "╥": "PRODUCT",  # the DOS character set's stand-in for a capital pi, byte 0xD2
}

# Glyph spellings of constants.
CONSTANT_GLYPHS: dict[str, str] = {
    "π": "pi",  # small pi
    "∞": "inf",  # infinity
    "°": "deg",  # degree sign
    "ê": "#e",  # e with circumflex
    "î": "#i",  # i with circumflex
    "\ue7c0": "#e",  # Derive's base of natural logarithms
    "\ue7c1": "#i",  # Derive's imaginary unit
    "\ue7c2": "pi",  # Derive's pi
    "\ue7c3": "euler_gamma",  # Derive's Euler gamma
}

# The Greek letters by name: the lower-case spelling names the small letter, the
# capitalized one the capital. They are predeclared, so they lex as one name in Character
# mode, and their case is respected in every case mode: `GAMMA` is the function, `gamma`
# the letter. `pi` is not among them, since it is the constant in any spelling.
GREEK_LETTERS: dict[str, str] = {
    spelling: letter
    for name, small, capital in (
        ("alpha", "α", "Α"),
        ("beta", "β", "Β"),
        ("gamma", "γ", "Γ"),
        ("delta", "δ", "Δ"),
        ("epsilon", "ε", "Ε"),
        ("zeta", "ζ", "Ζ"),
        ("eta", "η", "Η"),
        ("theta", "θ", "Θ"),
        ("iota", "ι", "Ι"),
        ("kappa", "κ", "Κ"),
        ("lambda", "λ", "Λ"),
        ("mu", "μ", "Μ"),
        ("nu", "ν", "Ν"),
        ("xi", "ξ", "Ξ"),
        ("omicron", "ο", "Ο"),
        ("rho", "ρ", "Ρ"),
        ("sigma", "σ", "Σ"),
        ("tau", "τ", "Τ"),
        ("upsilon", "υ", "Υ"),
        ("phi", "φ", "Φ"),
        ("chi", "χ", "Χ"),
        ("psi", "ψ", "Ψ"),
        ("omega", "ω", "Ω"),
    )
    for spelling, letter in ((name, small), (name.capitalize(), capital))
}

# The name of each Greek letter, which result lines use, and files where the letter
# itself would read as something else.
GREEK_NAMES: dict[str, str] = {letter: name for name, letter in GREEK_LETTERS.items()}

# Characters of the IBM PC character set read as Greek letters. Derive for DOS stored
# Greek letters in it, and its Greek characters decode to `ß` for beta, `Θ` for theta,
# `Φ` for phi, `Ω` for omega and the micro sign for mu (GRAPHICS.MTH writes `Θn`, for
# example). Text decoded that way must name the same variables. A declaration that uses
# one of these characters as a name takes it for that name.
DOS_GREEK: dict[str, str] = {
    "ß": "β",  # sharp s, U+00DF
    "Θ": "θ",  # capital theta
    "µ": "μ",  # micro sign, U+00B5
    "Φ": "φ",  # capital phi
    "Ω": "ω",  # capital omega
}

# Letters that read as a constant or a built-in function until a declaration uses the
# letter as a name, as `π(a) := a + 1` does; from then on they are that name.
DISPLACEABLE_GLYPHS: frozenset[str] = frozenset("πΓΣΠêî")

# Operator glyphs and their canonical spelling, ASCII or a keyword operator. Note that
# `·` U+00B7 is multiplication, and `∙` U+2219 and Derive's `⋅` U+22C5 the dot product.
OPERATOR_GLYPHS: dict[str, str] = {
    "≤": "<=",
    "≥": ">=",
    "≠": "/=",
    "≔": ":=",
    # Read as an ordinary assignment, as Derive 6 shows and treats it.
    ":==": ":=",
    "·": "*",
    "∙": ".",
    "⋅": ".",
    "±": "+-",
    "│": "|",
    ":ε": ":epsilon",
    ":∈": ":epsilon",
    "\ue7c8": ":epsilon",  # Derive's domain declaration sign
    "↓": "SUB",
    "¬": "NOT",
    "∧": "AND",
    "∨": "OR",
    "⊻": "XOR",
    "→": "IMP",
    "↔": "IFF",
    "⊆": "SUBSET",
    "∪": "UNION",
    "∩": "INTERSECTION",
    "•": "DOTPRODUCT",
    "⨯": "CROSSPRODUCT",
    # The minus sign, which text pasted from documents carries.
    "\u2212": "-",
    # The subscript arrow, byte 0x19 of the DOS character set, which DOS files store for
    # `SUB`.
    "\x19": "SUB",
}

# All spellings of the domain operator. A lone `ε` is the name `epsilon`, not an
# operator: `q ε Real` is `q*epsilon*Real`.
DOMAIN_OPERATORS: tuple[str, ...] = (":epsilon", ":ε", ":∈", "\ue7c8")

# Longest match wins, so within each leading character this lists longer operators first.
OPERATORS: tuple[str, ...] = (
    ":==",
    ":=",
    ":+",
    ":-",
    ":*",
    ":/",
    ":epsilon",
    ":ε",
    ":∈",
    "<=",
    ">=",
    "/=",
    "≤",
    "≥",
    "≠",
    "≔",
    "=",
    "<",
    ">",
    "+",
    "-",
    "*",
    "/",
    "^",
    "...",
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
    "{",
    "}",
    "\\",
    ",",
    ";",
    "?",
    "·",
    "∙",
    "⋅",
    "±",
    "│",
    "\ue7c8",
    "↓",
    "¬",
    "∧",
    "∨",
    "⊻",
    "→",
    "↔",
    "⊆",
    "∪",
    "∩",
    "•",
    "⨯",
    "\u2212",
    "\x19",
)

# Characters at or above U+0080 that are not letters: the glyphs of constants, functions
# and operators, the minus sign U+2212, and the rest of Derive's block in the Private Use
# Area. Every other such character is a letter, as in Derive 6 and in
# Derive for DOS, so names such as `Lotfußpunkt`, `■`, `m₀` or `F21` followed by a
# no-break space read whole. The letters `DISPLACEABLE_GLYPHS` begin a name or stand for
# a constant or a function, depending on the declarations.
NON_LETTERS: frozenset[str] = (
    ((frozenset(CONSTANT_GLYPHS) | frozenset(FUNCTION_GLYPHS)) - DISPLACEABLE_GLYPHS)
    | frozenset(op for op in OPERATOR_GLYPHS if len(op) == 1)
    | frozenset("\u2212")
    | frozenset(chr(code) for code in range(0xE7C0, 0xE7C9))
)

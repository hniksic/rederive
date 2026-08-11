"""Rederive - a remake of the DOS-era Derive computer algebra system."""

import sys

__version__ = "1.2.2"

# CPython refuses to convert between an int and a decimal numeral of more than
# 4300 digits, the conversion being quadratic and the limit a defence against
# untrusted input. A computer algebra system has no untrusted input: the numbers
# are the ones the user asked for, and `5000!` alone is 16326 digits. Lifting the
# limit here, above everything that converts a numeral, is what makes a number the
# printer can write a number the parser can read back - without it Factor on a
# large enough integer fails where Simplify has just produced it. A computation
# that runs away is held by the worker's memory cap and by Esc, neither of which
# is a digit count.
sys.set_int_max_str_digits(0)

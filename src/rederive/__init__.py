"""Rederive - a remake of the DOS-era Derive computer algebra system."""

import sys

__version__ = "1.6.0"

# CPython refuses int <-> decimal conversions beyond 4300 digits, as a defense against
# untrusted input. A CAS has no untrusted input, and `5000!` alone has 16326 digits.
# Without lifting the limit, the parser couldn't read back large numbers the printer
# writes, so e.g. Factor would fail on an integer Simplify just produced. Runaway
# computations are still bounded by the worker's memory cap and by Esc.
sys.set_int_max_str_digits(0)

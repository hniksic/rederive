"""Test-wide configuration.

Textual's pilot sleeps 20 ms twice per simulated keystroke, comparing wall clock
against process time to guess whether the app has settled.  The guess costs far more
than the app it watches, and buys little: it is a wall-clock heuristic, and a process
that a busy machine has descheduled looks idle to it, so what it settles is not
something a test can rest on either way.  Shrink the granularity so a test costs what
its work costs.

What pilot.press does promise is a drain: every widget's message queue emptied once.
A drain is not a layout pass.  Showing a widget or setting how tall it stands writes a
style, and the size that follows is assigned later, on the screen's own idle, so a test
that reads `size` or `region` has to wait for the layout itself - `laid_out` in
`screen.py` is how.
"""

import textual._wait as _wait

# Read before assigning, so a rename upstream fails loudly rather than quietly
# creating an attribute nobody looks at.
assert _wait.SLEEP_GRANULARITY and _wait.SLEEP_IDLE
_wait.SLEEP_GRANULARITY = 0.002
_wait.SLEEP_IDLE = _wait.SLEEP_GRANULARITY / 20.0

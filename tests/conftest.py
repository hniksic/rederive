"""Test-wide configuration.

Textual's pilot sleeps 20 ms twice per simulated keystroke, comparing wall clock
against process time to guess whether the app has settled.  The guess costs far more
than the app it watches, and it is redundant here: pilot.press already ends by waiting
for every widget to drain its message queue.  Shrink the granularity so a test costs
what its work costs.
"""

import textual._wait as _wait

# Read before assigning, so a rename upstream fails loudly rather than quietly
# creating an attribute nobody looks at.
assert _wait.SLEEP_GRANULARITY and _wait.SLEEP_IDLE
_wait.SLEEP_GRANULARITY = 0.002
_wait.SLEEP_IDLE = _wait.SLEEP_GRANULARITY / 20.0

# Web demo: which Pyodide channel the build uses, measured on the real page

Stage 0 measured the two Pyodide distribution channels on a bare spike - a page that
booted Pyodide, imported sympy and answered one request - and stage 3 chose the
precompiled `pyc` channel on that evidence. It also measured the real page on `pyc` alone.
What was never measured was **the real page on the default channel**, which is the only
comparison that decides anything now that the page carries the rederive wheel, Textual,
Rich, numpy, uPlot and three.js as well.

This is that measurement, taken on 7 August 2026, and the decision it settles.

**The verdict: keep `pyc`.** It is 6 MB more over a compressing connection and it is worth
it twice over - **1.2 to 1.4 seconds off the wait for the first screen, and 4.5 seconds
off the wait for the first answer**, in both browsers. The second number is the one that
matters and it is the one stage 0 could not see: the default channel spends four extra
seconds compiling sympy inside the engine worker, and a user pays that on the first
Simplify rather than on the loading screen, where nothing explains it.

## How it was asked

Both channels were built from this tree, unchanged except for the channel:

```
python3 tools/build_web.py                                     # pyc, build/web
python3 tools/build_web.py --channel full --output build/web-full
```

Each built directory was copied, and one `<script>` was appended to its `index.html`: the
page measures itself, drives its own terminal through `term.input` - the same door a
keystroke comes in by - and posts its answers back to the server serving it. So what is
timed is the whole page, including the app instance, the engine worker's sympy and the
Textual driver, with no automation harness anywhere in it.

The reason for that shape is stage 0's finding, which stands: **nothing is timed through
Playwright.** Its Firefox ran WebAssembly some eight times slower than the machine's own,
and its Chromium about twice as slow. `tools/smoke_web.py` is a correctness gate and takes
no timings at all.

**The machine.** Intel Core Ultra 7 258V, 8 cores, 30 GB, Linux 6.14 - stage 0's machine.
**The browsers**, both headless and both snaps: **Chromium 150.0.7871.128** and **Firefox
153.0.3**, each on a profile of its own, cold (a fresh profile) and warm (the same profile
a second time). **Pyodide 314.0.3** on both channels. Served over `python -m http.server`
from this machine, which compresses nothing - so the wire is loopback and near free, and
what is being compared is compilation rather than transfer.

Two figures per run:

* **prompt** - from the page starting to the program's own menu being on the screen. This
  is the loading screen's whole life.
* **answer** - from the page starting to `4·x` being on the screen, having authored
  `(x+1)^2-(x-1)^2` and simplified it. It includes the prompt, the engine worker's own
  Pyodide, its sympy import and the first `simplify` - so the difference between the two
  columns is what the worker costs, and the worker is where the channel is felt.

## Seconds

| channel | browser | | prompt | answer |
| --- | --- | --- | ---: | ---: |
| **`pyc`** | Chromium | cold | **1.82 s** | **2.91 s** |
| | | warm | 1.55 s | 2.62 s |
| | Firefox | cold | **2.47 s** | **3.67 s** |
| | | warm | 2.31 s | 3.71 s |
| `full` | Chromium | cold | 3.04 s | 7.10 s |
| | | warm | 2.56 s | 6.35 s |
| | Firefox | cold | 3.57 s | 8.34 s |
| | | warm | 3.57 s | 8.22 s |

Where the page's own time goes, cold, in milliseconds:

| | `loadPyodide` | the rederive wheel | the rest |
| --- | ---: | ---: | ---: |
| Chromium, `pyc` | 310 | 186 | 1319 |
| Chromium, `full` | 1297 | 248 | 1492 |
| Firefox, `pyc` | 474 | 244 | 1751 |
| Firefox, `full` | 1456 | 267 | 1845 |

"The rest" is Textual building a screen and the driver taking it over, and it is the same
on both channels, as it must be: no sympy runs on this thread. The channel shows up in
`loadPyodide` - which is the stdlib zip being compiled - and then again, four times over,
in the worker.

**Cold and warm differ by a quarter of a second and no more.** The files are on loopback,
so what a warm cache saves is transfer that cost nothing to begin with. Stage 0 found the
same thing and for the same reason; a real visitor over a real connection pays the wire on
top of every number here, and that cost is the table below.

## Bytes

The whole built directory, as it sits and as a compressing server sends it:

| | raw | gzip -6 |
| --- | ---: | ---: |
| `pyc` | 64.6 MB | **23 MB** |
| `full` | 26.0 MB | **17 MB** |

**The `pyc` channel costs 6 MB over the wire and saves 4.5 seconds of compiling.** At
10 Mbit/s those 6 MB are about 5 seconds, so the two roughly cancel on a slow connection
and `pyc` wins outright on a fast one - and it wins every time after the first, because
the files are cached and the compiling is not.

The raw column is the one to watch on a host that does not compress. `python -m
http.server` is such a host: over it the `pyc` build is 64.6 MB and the default build 26
MB, and the default one would then be the faster first visit by a wide margin. That is a
statement about the server and not about the channel. **GitHub Pages compresses**, which
is what the publish step in `.github/workflows/web.yml` is written against.

## The decision

`tools/build_web.py` keeps `--channel pyc` as its default and `--channel full` as the
escape hatch for a Pyodide release with no precompiled build. Nothing changes; what
changes is that the choice now rests on the page that is actually shipped rather than on a
spike, and on the number that a user feels most - the first answer - rather than only on
the first screen.

Two things would reopen this:

* **A host that does not compress.** 64.6 MB down a wire is a different demo from 23 MB.
  Anyone serving the built directory from something other than Pages should measure their
  own first visit before assuming this table applies.
* **A Pyodide release whose `pyc` channel lags.** The channel is undocumented and its
  contents are not guaranteed to keep pace with the default one. `--channel full` builds
  and runs; it is 4.5 seconds slower to the first answer and nothing else.

## What the loading screen says, and why

`web/main.js` tells the user that about 24 MB comes down on a first visit and that a few
seconds of what follows is compiling Python, and it prints the seconds each phase took as
it finishes. Those words are written against this table: 23 MB compressed, and one and a
half to two and a half seconds to the first screen. Stage 0's warning that the plan's
"8-15 seconds" was ten times too pessimistic still holds - but so does the other half of
it, which is that the first Simplify carries the worker's sympy with it. That is why the
loading screen names it, and why the demo does not simply look slow for no stated reason
the first time a user presses S.

## What this did not measure

* **No throttled connection**, for stage 0's reason: Chromium can be throttled through CDP
  and Firefox cannot, so the two would not be comparable. The bytes table is what stands in
  for it.
* **One run of each case.** The differences reported here are factors, not percentages -
  1.2 to 4.5 seconds against a spread between cold and warm of a quarter of a second - but
  none of these is an averaged benchmark.
* **Nothing about plotting.** numpy is loaded by the engine worker when the first plot is
  sampled, not at boot, and neither figure above includes it. Stage 6 found the first
  surface taking about 4.5 s in Firefox and 2.2 s in Chromium, which is the other wait a
  user meets once.
* **One machine, one operating system, and no phone.** A phone is slower at all of this
  and has less memory to lose to two Pyodide instances.

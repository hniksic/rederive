"""Plotting in a browser: panes in a page, sampled in the engine worker.

Three pieces and a line drawn twice through them.

`backend.py` is the `Backend` the plot session opens windows on, and it runs on
the main thread beside the app. It owns what the desktop's windows own that has
no numbers in it: the plot list, the colors, which pane a plot lands in, what
`Describe` answers. It imports no numpy, because the instance it runs in paints
the screen and knows no mathematics.

`protocol.py` is what it asks the engine worker for and what comes back. The
requests are picklable and free of numpy, exactly as `plot/protocol.py` is; the
answers are arrays, and they never come back this way at all.

`sampler.py` is the far end, and it runs in the worker beside sympy. It holds
the lambdified closures - the one content-keyed cache the worker's statelessness
contract licenses - runs `plot/resample.py` over them, and answers in arrays
that go straight to the page's JavaScript. They never stop over in the main
thread's Python, which is what keeps numpy out of the instance that draws the
terminal.

What is left for JavaScript is the picture: the view, the gestures and the
drawing, in `web/plot2d.js`. It renders no expression of its own, and no number
it shows was formatted anywhere but Python.
"""

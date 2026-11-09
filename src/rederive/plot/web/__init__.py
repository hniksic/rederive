"""Plotting in a browser: panes in a page, sampled in the engine worker.

`backend.py` is the `Backend` the plot session opens windows on. It runs on the main
thread beside the app and owns the parts without numbers: the plot list, colors, which
pane a plot lands in, and what `Describe` answers. It imports no numpy, since the main
thread's instance only draws the screen.

`protocol.py` defines the requests sent to the engine worker: a curve over a view, a
reading, a scan, and a surface over a domain. Like `plot/protocol.py`, they are picklable
and free of numpy. The answers are arrays and don't come back through this path.

`sampler.py` runs in the worker beside sympy. It caches the lambdified closures (the one
content-keyed cache the worker's statelessness contract allows), runs `plot/resample.py`
over them, and sends arrays straight to the page's JavaScript, bypassing the main
thread's Python so it stays free of numpy.

JavaScript handles the view, gestures and drawing: `web/plot2d.js` for 2D plots and
`web/plot3d.js` for 3D. Neither renders expressions, and all numbers they show are
formatted in Python.

For a 3D surface, what crosses to the page is not the sampled grid but the geometry
`plot/surface.py` builds from it (triangles, wireframe and per-vertex colors), so
three.js gets exactly what pyqtgraph gets on the desktop.
"""

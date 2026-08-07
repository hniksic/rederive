"""The colors and measurements a picture is drawn at, for both backends at once.

A plot window and a plot pane are two programs drawing one picture, and every
number here is one they have to agree about: the ground the canvas is painted,
the gray the ruling is tinted out of, how wide the stroke that draws
mathematics is, how far the arrow keys move a trace marker, how far out of a
box a tick number stands. Written twice, they are numbers the two backends can
drift apart on, and the drift is invisible until the two pictures are put side
by side.

So they are written here, and the backends read them. The desktop's windows
import this module; the page's panes are handed what is in it when the browser
backend starts, in `plot/web/backend.py`, and `web/plot2d.js` and
`web/plot3d.js` draw with what they were handed rather than with constants of
their own.

Nothing in here is a toolkit's. The colors are strings and tuples, the lengths
are logical pixels and world units, and the module imports nothing at all -
which is what lets the browser backend, forbidden Qt and numpy alike, read the
same numbers the Qt windows draw with.

What is *not* here is anything only one backend has. A page counts its own
double-click, measures its own wheel and cascades its own panes, and a desktop
has a window manager, a painter path and a style sheet for those; a number with
one home belongs in that home. The rule is agreement, not accumulation.
"""

from __future__ import annotations

__all__ = [
    "AXIS_COLOR",
    "BACKGROUND",
    "BOX_COLOR",
    "CHIP_OFFSET_PX",
    "CLICK_SLOP_PX",
    "CURVE_WIDTH",
    "EDGE_ON",
    "GRID_ALPHA",
    "HAIRLINE_ALPHA",
    "HIT_PX",
    "LABEL_OUT",
    "LEGEND_FADED",
    "LIGHT",
    "MARKER_PX",
    "MARKER_WIDTH",
    "NAME_OUT",
    "NUDGE_FAST_PX",
    "NUDGE_PX",
    "PAPER_AXIS",
    "POLAR_CLOSEST_PX",
    "POLAR_RINGS",
    "POLAR_SPOKES",
    "REGION_ALPHA",
    "STEP_FAST_SHARE",
    "STEP_SHARE",
    "TICK_COLOR",
    "TICK_OUT",
    "WIRE_OFFSET",
]

#: The ground a picture is painted on, and the gray its furniture is drawn in.
#: Near-black rather than black, so that a curve in a black-adjacent color
#: still reads and so that neither a window nor a pane looks like a hole in
#: what is behind it. What is *around* the picture - the bar, the fields, the
#: status line - is the chrome's business and is each backend's own: a style
#: sheet on the desktop, `plot/qt/theme.py`, and the page's own in its CSS.
BACKGROUND = "#0c0c10"
AXIS_COLOR = "#909090"

#: How much of that gray the ruling is drawn at. A grid is read against rather
#: than read, so it is the faintest thing on the canvas that is still there.
GRID_ALPHA = 0.18

#: The axis lines through the origin, on the white ground every image export is
#: taken on. A dark picture pasted into a document is a black rectangle, so
#: what leaves either backend is drawn as it would be printed - and this is the
#: one gray of the furniture that has to change to stay readable there. The
#: ruling and the tick marks keep `AXIS_COLOR` at its own alpha, which reads on
#: either ground.
PAPER_AXIS = "#404040"

#: How a polar picture is ruled: about this many rings out to the edge of the
#: view, a spoke every full turn divided this many ways, and no grid at all
#: where the rings would fall closer together than this many pixels, a grid
#: that dense being a wash over the picture rather than something to read
#: against.
POLAR_RINGS = 5
POLAR_SPOKES = 12
POLAR_CLOSEST_PX = 2.0

#: How wide the stroke that draws mathematics is, in logical pixels. The curve
#: pen, a data plot's connecting polyline, the legend sample drawn with the
#: item's own pen, the export pens and a solid's wire all take their width from
#: here, so the weight is the same on screen, in the swatch and in a pasted
#: PNG. The furniture keeps its hairlines - axis lines at one pixel, the grid
#: at its alpha - which is what makes a curve read as the subject rather than
#: as more scaffolding.
CURVE_WIDTH = 2.0

#: How far the pointer may move between a right-button press and its release
#: and still count as a click that opens the context menu rather than a
#: rubber-band zoom of no area.
CLICK_SLOP_PX = 4.0

#: How near a curve the pointer has to be, in pixels, to be pointing at it.
HIT_PX = 6.0

#: How solid a region's fill is. Enough to read as shading, little enough that
#: a curve crossing it is still a curve.
REGION_ALPHA = 0.25

#: How much of a hidden legend row is left standing. Dimmed rather than struck
#: through: a hidden curve is a curve that is still in the picture, and the row
#: has to read as one of the list rather than as a mistake in it.
LEGEND_FADED = 0.4

#: The trace marker: a ring rather than a filled dot, so the point it names is
#: still visible under it, and the pen that draws the ring. The hairline down
#: to the axis is drawn in the same color at this much of it, and the value
#: chip rides this far from the point so that a pointer over the marker never
#: covers what the marker says.
MARKER_PX = 11.0
MARKER_WIDTH = 2.0
HAIRLINE_ALPHA = 0.45
CHIP_OFFSET_PX = 12.0

#: How far the arrow keys move the trace marker, in pixels, plain and with
#: Shift. How far they pan a view that has no marker on it is a property of the
#: view rather than of the marker, and is `plot/actions.py`'.
NUDGE_PX = 1.0
NUDGE_FAST_PX = 10.0

#: How far the arrow keys move the marker along a parametric curve, as a
#: fraction of the parameter range, plain and with Shift. A five-hundredth is
#: about a pixel on a curve that crosses the picture once.
STEP_SHARE = 1 / 500
STEP_FAST_SHARE = 1 / 50

#: The box a surface stands in and the marks along its edges, as red, green,
#: blue and alpha out of 255. A gray that reads against the ground without
#: competing with the surface standing in it, and the ticks a little more
#: present than the box, being what the numbers are read off.
BOX_COLOR = (150, 150, 150, 110)
TICK_COLOR = (150, 150, 150, 200)

#: How far the wire's occluder is pushed away from the camera, as OpenGL's
#: (factor, units) pair - which is what WebGL reads it as too. The occluder is
#: the surface's own triangles, so the lines lie exactly on it and the depth
#: test would decide their pixels by rounding - a wire stitched out of dashes.
#: A small shove backwards settles it: the lines win everywhere they touch the
#: surface, and the shove is far too slight to let a line on the far side
#: through.
WIRE_OFFSET = (1.0, 2.0)

#: How far a tick mark, its number and the name of an axis stand out of the
#: box, in the world units the box is measured in.
TICK_OUT = 0.35
LABEL_OUT = 1.05
NAME_OUT = 2.3

#: How nearly an axis has to point at the camera before its numbers are
#: dropped, as the cosine of the angle between them: about five degrees. Facing
#: the xy plane makes the whole z axis one point of the screen, and five
#: numbers stacked on that point are five numbers about nothing.
EDGE_ON = 0.996

#: Where the light a surface is shaded to comes from. `plot/surface.py` bakes
#: it into the vertex colors, which is what makes a fold in the middle of a
#: surface show and what both backends draw; the page's card then shines a lamp
#: of its own from the same direction, so that turning the picture moves a
#: highlight across it. Two directions would light one surface twice from two
#: places, and the relief would fight the highlight.
LIGHT = (0.40, -0.60, 0.69)

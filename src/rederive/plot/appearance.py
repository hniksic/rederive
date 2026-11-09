"""Colors and measurements shared by both plot backends.

The Qt windows and the browser panes draw the same pictures, so values such as the
background color, grid tint, curve width, trace-marker step and tick offsets are defined
once here to keep them from drifting apart.

The Qt windows import this module. The browser backend (`plot/web/backend.py`) passes
its contents to the page on startup, and `web/plot2d.js` and `web/plot3d.js` draw with
those values rather than their own constants.

Values are plain strings, tuples and numbers (logical pixels and world units), and the
module imports nothing, so the browser backend can use it without Qt or numpy.

Settings used by only one backend belong in that backend, not here.
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

#: The canvas background and the gray of axes and grid. Near-black rather than black,
#: so that dark curves remain visible and the canvas doesn't look like a hole. The
#: surrounding chrome (toolbar, fields, status line) is styled by each backend:
#: `plot/qt/theme.py` on the desktop, CSS in the page.
BACKGROUND = "#0c0c10"
AXIS_COLOR = "#909090"

#: Opacity of the grid, the faintest thing on the canvas.
GRID_ALPHA = 0.18

#: Color of the axis lines in image exports, which use a white background so they
#: paste well into documents. Grid and tick marks keep `AXIS_COLOR`, which reads on
#: either background.
PAPER_AXIS = "#404040"

#: The polar grid: about this many rings out to the edge of the view, and this many
#: spokes per turn. The grid is omitted when rings would be closer than
#: `POLAR_CLOSEST_PX` pixels, since it would just smear over the picture.
POLAR_RINGS = 5
POLAR_SPOKES = 12
POLAR_CLOSEST_PX = 2.0

#: Width in logical pixels of curves, data polylines, legend samples, export pens and
#: surface wireframes, so the weight matches on screen and in exports. Axes and grid
#: stay hairlines so curves stand out.
CURVE_WIDTH = 2.0

#: How far the pointer may move between right-button press and release and still count
#: as a click that opens the context menu, rather than a rubber-band zoom.
CLICK_SLOP_PX = 4.0

#: How near a curve, in pixels, the pointer must be to point at it.
HIT_PX = 6.0

#: Opacity of a region's fill: visible as shading, but faint enough for curves to show
#: through.
REGION_ALPHA = 0.25

#: Opacity of a hidden curve's legend row. Dimmed rather than struck through, since
#: the curve is still part of the plot.
LEGEND_FADED = 0.4

#: The trace marker: ring size (a ring rather than a dot, so the point stays visible)
#: and pen width, the opacity of the hairline down to the axis, and how far the value
#: chip sits from the point so the pointer doesn't cover it.
MARKER_PX = 11.0
MARKER_WIDTH = 2.0
HAIRLINE_ALPHA = 0.45
CHIP_OFFSET_PX = 12.0

#: How far the arrow keys move the trace marker in pixels, plain and with Shift.
#: Panning without a marker is set in `plot/actions.py`.
NUDGE_PX = 1.0
NUDGE_FAST_PX = 10.0

#: How far the arrow keys move the marker along a parametric curve, as a fraction of
#: the parameter range, plain and with Shift. 1/500 is about a pixel on a curve that
#: crosses the picture once.
STEP_SHARE = 1 / 500
STEP_FAST_SHARE = 1 / 50

#: RGBA colors of the 3D bounding box and its ticks: a gray that doesn't compete with
#: the surface, with ticks slightly stronger since values are read off them.
BOX_COLOR = (150, 150, 150, 110)
TICK_COLOR = (150, 150, 150, 200)

#: Polygon offset (OpenGL/WebGL factor, units) pushing the surface's occluding
#: triangles away from the camera. The wireframe lies exactly on them, so without the
#: offset the depth test would break it into dashes; the offset is too small to let
#: lines on the far side show through.
WIRE_OFFSET = (1.0, 2.0)

#: How far tick marks, tick labels and axis names stand out from the box, in world
#: units.
TICK_OUT = 0.35
LABEL_OUT = 1.05
NAME_OUT = 2.3

#: Cosine of the angle (about five degrees) below which an axis pointing at the camera
#: loses its tick labels. Facing the xy plane, for example, collapses the z axis to a
#: point, and its labels would just pile up.
EDGE_ON = 0.996

#: Direction of the light used to shade surfaces. `plot/surface.py` bakes it into the
#: vertex colors, which makes folds visible in both backends. The page adds its own
#: light from the same direction for a moving highlight; a different direction would
#: make the highlight fight the shading.
LIGHT = (0.40, -0.60, 0.69)

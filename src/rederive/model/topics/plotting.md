# Plotting

Plot draws the highlighted expression in a plot window.

## What an expression plots as

| Command | What it does |
|---|---|
| Plot Plot | draw in the most recently used plot window, opening one if needed |
| Plot New | draw in a new window; later plots then go to it |

The expression itself decides what kind of picture it makes, and with it which of the two
kinds of window it opens.

| In 2D | Drawn as |
|---|---|
| one variable, or none | a curve: `SIN(x)/x`, `x^2 - 1`, `3` |
| y = f(x) | the curve of f(x): `y = 1/x` |
| u = v | the curve where the two sides are equal |
| an inequality | the area where it holds: `x^2 + y^2 < 9` |
| [u, v] | a parametric curve: `[3SIN(3t), 3SIN(4t)]` |
| [u, v, w] and longer | each as it is: `[x = 4, x - 6, x^2]` |
| [a, b] of numbers | a point; a vector of pairs draws them all |
| vectors of those | a line of points for each, as ISOMETRICS gives |

| In 3D | Drawn as |
|---|---|
| two variables | a surface: `1/(9 + x^2 + y^2)` |
| z = f(x, y) | a surface, z on the vertical axis |
| a vector of either form | a surface for each, in one picture |

To draw two curves together, Plot each expression in turn. Plotting an expression again
replaces its old curve rather than adding one. An expression in three or more variables
cannot be plotted. The axes take their names from the expression's own variables.

The expression is simplified first, so a program it calls runs before anything is drawn.
Where that leaves no formula, as with a program whose tests depend on the variable, the
program runs again at each point, which is slower.

## The 2D window

| Mouse | Action |
|---|---|
| wheel | zoom about the pointer; Ctrl only x, Shift only y |
| drag | move the view |
| right drag | mark a rectangle to zoom into |
| double click | center the view there |
| click on a curve | [trace](help:plotting/tracing) it |
| click a legend entry | hide that plot, or show it again |
| right click | the window's menu |

The menu also acts on the plot the click was over: remove it, or join a data plot's points
and change their size.

| Key | Action |
|---|---|
| Home or 0 | back to the starting view |
| A | fit everything drawn into the view |
| + and - | zoom in and out |
| arrows | move the view |
| Backspace | the previous view; Shift-Backspace forward again |
| T or F3 | trace on and off |
| G and L | show or hide the grid and the legend |
| Del | remove every plot |
| Ctrl-C | copy the picture; while tracing, the marked point |
| Ctrl-S | save the picture as PNG, or as SVG, which scales in print |
| Q | close the window |

Set range... on the menu lets you type the four bounds of the view, as expressions: -π is
a valid bound. A fresh window shows -5 to 5 with the same scale on both axes, so a circle
is round. After A fits everything in, the two scales may differ; the 1:1 button or Home
makes them equal again. A plot drawn entirely outside the current view is fitted into it
automatically.

The polar button switches the window to polar coordinates: every curve is then drawn as
r = f(θ), with its variable as the angle. A parametric or polar curve is first drawn for
parameter values from -π to π; type other bounds into the fields at the top of the window
to change that.

Copies and saved files use dark lines on a white background, and include the legend.

## Tracing

Trace follows a curve with a marker and shows the coordinates under it. On an ordinary
curve the marker follows the pointer; the keys move it too.

| Key | Action |
|---|---|
| Left, Right | along the curve; faster with Shift |
| Up, Down | onto another curve |
| Tab | to the next root, maximum, minimum or intersection with another curve; Shift-Tab to the previous one |
| Enter | add the marked point to the worksheet as a new expression |
| Ctrl-C | copy the marked point |
| Esc | stop tracing |

The points Tab finds are exact enough to calculate with. Where the curve has no value,
like `SIN(x)/x` at x = 0, tracing shows that too.

## The 3D window

The mouse and keys move only the camera.

| Key | Action |
|---|---|
| drag | turn the picture; Shift-drag slides it |
| wheel | nearer and further |
| arrows | turn a step at a time |
| Home or 0 | back to the starting view |
| 1, 2, 3 | look straight at the xy, xz or yz plane |
| R | spin slowly; R again stops |
| M | draw the surfaces as wire mesh, or shaded |
| B | show or hide the box around the surfaces |
| X | show or hide the numbers and names along the box edges |
| L | show or hide the legend |
| Del | remove every surface |
| Ctrl-C, Ctrl-S, Q | as in a 2D window; the picture is saved as PNG |

The surfaces are computed over the domain given in the fields at the top, on a grid of the
given fineness; changing either recomputes them. The view... button shows the plotted
region and the camera as numbers you can edit. A surface running off to infinity has its
spike cut short, so that the rest of the shape stays visible.

The grid and the wire or shaded drawing chosen here, and the joining and size of points
chosen in a 2D window, become the defaults for new plots.

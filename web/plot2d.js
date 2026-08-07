// The 2D plot panes: uPlot for the frame, and everything a picture needs that a
// chart library has never heard of.
//
// What uPlot is here for is the furniture - the axes, their ticks and numbers,
// the ruling, the plot rectangle and the canvas - and it is very good at it.
// What it is not is a way to draw mathematics. A chart library holds one shared
// abscissa and wants it sorted, and neither is true of what this program draws:
// every curve is sampled adaptively over its own abscissa, and a parametric
// curve doubles back. So the series are empty and every stroke is drawn in a
// draw hook, on uPlot's own canvas, in uPlot's own coordinates. That is also
// where the polar grid, the region shading, the axis lines through the origin
// and the trace marker go, none of which a chart library has.
//
// Nothing here renders an expression and nothing here evaluates one. A curve's
// name is the one line Python wrote with the syntax writer; the reading under
// the trace marker, the sentence about a feature and the point sent home to the
// worksheet are all written where the numbers are, in the engine worker, and
// this file shows what it is handed. The one exception is the pointer readout,
// which is the coordinates of the pointer and not a value of anything.
//
// The samples never pass through Python. They come off the worker as typed
// arrays, are drawn from here, and what goes back to the main thread's Python is
// the request number and whatever would not evaluate.
//
// Nor does this file own what a pane offers. The commands, their words, their
// keys and the order a menu stands them in are `plot/controls.py`'s, the same
// table the desktop's window renders; a pane is handed the ones this backend
// serves when it is made and the menu as it should currently read when a right
// click asks for one, and it answers with the name of a control. What is kept
// here is the view - the framing, the toggles, what the pointer is over - since
// a drag that had to ask Python where it was would be a drag nobody could use.
// The one thing kept here that a command needs is therefore handed over when
// the command runs: a gesture says where the view stood before it moved it, so
// that Back has a history to step through without a keystroke ever crossing.
//
// The dialogs are `plot/forms.py`'s in the same way, drawn by `forms.js`, and
// what is typed into one goes back as text - the parsing and the arithmetic are
// Python's, which is what lets a bound be written `-π` here as it is on a
// desktop.
//
// Copying and exporting are the one pair this file answers outright. A canvas
// has no painter path to write a vector file out of, so what leaves a pane is
// pixels, and the sentence Python says about it says so. What leaves is drawn
// on paper rather than on the pane's near-black, as the desktop's window draws
// it: a dark picture pasted into a document is a black rectangle.

import * as controls from './controls.js';
import * as sheets from './forms.js';
import * as place from './place.js';

// The colors the canvas itself is drawn in, which are `plot/qt/window2d.py`'s.
// Near-black rather than black, so that a curve in a black-adjacent color still
// reads and so that the pane does not look like a hole in the page. Everything
// around the canvas - the bar, the legend card, the status line - is the
// chrome's business and is in the style sheet, as `theme.py` is on the desktop.
const BACKGROUND = '#0c0c10';
const AXIS_COLOR = '#909090';
const GRID_COLOR = 'rgba(144, 144, 144, 0.18)';
const MUTED = '#7d8595';

// And the colors a picture that is leaving the pane is drawn in, which are
// `plot/qt/window2d.py`'s `_on_paper`: a white ground, every curve in the
// second color the plot list holds for it, the numbers on the axes in black
// and the axes through the origin in a gray that reads on paper. A dark
// picture pasted into a document is a black rectangle, so what leaves is drawn
// the way it would be printed. The ruling and the tick marks are left as they
// are, being the desktop's `#909090` at its own alpha on either ground.
const PAPER = '#ffffff';
const PAPER_TEXT = '#000000';
const PAPER_AXIS = '#404040';

// The stroke that draws mathematics, in logical pixels, and the furniture's
// hairlines. The weight is what makes a curve read as the subject rather than
// as more scaffolding.
const CURVE_WIDTH = 2;
const MARKER_PX = 11;
const MARKER_WIDTH = 2;
const HAIRLINE_ALPHA = 0.45;
const CHIP_OFFSET_PX = 12;
const REGION_ALPHA = 0.25;

// How far the arrow keys move the marker: a pixel at a time along a function,
// whose parameter is the abscissa, and a five-hundredth of the parameter range
// along a curve whose parameter is not on the screen at all.
const NUDGE_PX = 1;
const NUDGE_FAST_PX = 10;
const STEP_SHARE = 1 / 500;
const STEP_FAST_SHARE = 1 / 50;

// How far the pointer may move between a right-button press and its release and
// still count as a click that opens the menu rather than a rubber band.
const CLICK_SLOP_PX = 4;

// How far a wheel zooms, and what one turn of one is worth in the units a page
// reports a scroll in.
//
// The factor is raised to the distance scrolled rather than applied whole per
// event, because a mouse fires one event per detent and a trackpad fires a
// stream of small ones: a step per event is right for the first and turns a
// short two-finger drag into a zoom of several hundred for the second. The
// desktop is proportional in the same way - pyqtgraph scales by 1.02 raised to
// the wheel's angle - but that rate is in Qt's eighth-of-a-degree units, which a
// page has no equivalent of, so the number here is this file's own and it is
// only the outcome that is comparable: one detent is 1.25 in a pane and 1.02^15,
// or 1.35, in a window.
//
// A page reports a scroll in pixels, in lines or in pages, and no two browsers
// agree on which; nothing measures a line or a page for us, so those two are
// what a line of text and a screenful of them come to. The cap is what stops one
// oversized event - a page of scroll, or a driver that batches a flick - from
// throwing the view away in a single frame.
const WHEEL_FACTOR = 1.25;
const WHEEL_DETENT_PX = 100;
const WHEEL_LINE_PX = 40;
const WHEEL_PAGE_PX = 800;
const WHEEL_MAX_DETENTS = 3;

// How much of a hidden legend row is left standing. Dimmed rather than struck
// through: a hidden curve is a curve that is still in the pane.
const LEGEND_FADED = 0.4;

// How the plot list is written into an exported picture: the size of the words
// and how far in from the corner they stand, in logical pixels. The card itself
// is a floating element and no part of the canvas, so an export that did not
// write the names would be a picture that had lost half of what it showed.
const LEGEND_PX = 12;
const LEGEND_MARGIN_PX = 10;

// How big a fresh pane is. Where it is put is `place.js`'s, which both kinds of
// pane share so that they cascade off one another.
//
// The height is not a matter of taste. A fresh view is x in [-5, 5] with the
// ordinate following from equal scales, so what a pane opens showing is decided
// by the shape of its picture - and `plot/view.py` says that a browser opening
// on a different rectangle from a desktop would be two programs. The desktop
// window is 760 by 560 and gives 723 by 482 of that to the picture, a picture
// two thirds as tall as it is wide; a pane spends more of itself on chrome,
// having a title bar of its own that a window gets from the window manager and
// wider gutters for the numbers along its edges, so it has to stand taller to
// leave a picture of the same shape behind. These are the numbers that do: they
// leave about 640 by 427, which is two thirds, and `2*x + 3` traced at the
// middle of the view is on the picture here exactly as it is there.
const PANE_WIDTH = 720;
const PANE_HEIGHT = 590;

// The kinds, spelled as `plot/protocol.py` spells them. The page is told which
// kind a plot is and reads nothing else into it.
const CURVE = 'curve';
const FAMILY = 'family';
const PARAMETRIC = 'parametric';
const POLAR = 'polar';
const DATA = 'data';

// Which kinds a marker can ride, which a feature scan works on, and which are
// ridden at a parameter rather than at an abscissa. All three are
// `plot/protocol.py`'s answer: a region has no f(x) a marker could be at a point
// of, and a curve that doubles back has no abscissa that names one place on it.
const RIDEABLE = new Set([CURVE, FAMILY, PARAMETRIC, POLAR]);
const SCANNED = new Set([CURVE, FAMILY]);
const PARAMETRIZED = new Set([PARAMETRIC, POLAR]);

// How close together in time and in place two clicks have to be to be the one
// gesture that centres the view. The page counts them itself because a press on
// the picture is cancelled - that is what makes a drag a drag rather than a
// selection - and a cancelled press is one no `dblclick` follows.
const DOUBLE_MS = 400;
const DOUBLE_PX = 6;

// The one place a plot pane is put, laid over the terminal and letting the
// pointer through everywhere it has no pane.
let root = null;
let landed = () => {};
let terminal = null;

// Every open pane by its number, which is the number the plot session names it
// by at both ends. Exported because `main.js` hangs this module off
// `window.rederive`, where the console and a driving script can read what the
// page is actually showing; nothing in the program writes to it.
export const panes = new Map();

// The keyboard back to the terminal, as `files.js` does it for its buttons.
// Nothing else on the page reads keys, so there is nowhere else to put them.
function focus() {
  if (terminal !== null) terminal.focus();
}

function surface() {
  if (root === null) {
    root = document.getElementById('panes');
  }
  return root;
}

// Holding on to a pointer for the length of a drag, so that a finger or a mouse
// that leaves the element it started on is still heard. A browser may refuse -
// a pointer it has already released is not one to capture - and a refusal is
// not worth an exception in the page: what capture buys is the end of the drag,
// not the drag.
function capture(element, event) {
  try {
    element.setPointerCapture(event.pointerId);
  } catch (refused) {
    /* the drag goes on without it, and ends wherever the element does */
  }
}

// -- what Python calls ---------------------------------------------------------

// Called by the page as it builds itself, so that a pane which closes while it
// holds the keyboard knows where to hand it back. The terminal is the only
// other thing on the page that takes keys.
//
// The copy event is listened for here for a reason of its own. Ctrl+C reaches a
// pane as a `copy` and not as a key press: a page that cancels the keydown
// cancels the copy the browser was about to offer it, so the key ladder leaves
// the stroke alone and this is where it arrives. The listener is the document's
// because a copy event is raised where the selection is rather than on whatever
// holds the keyboard, and it goes to the pane that has focus - which is also
// what keeps it away from the terminal, xterm.js having its own copy.
export function wire(term) {
  terminal = term;
  document.addEventListener('copy', (event) => {
    const pane = holding();
    if (pane !== null) pane._copied(event);
  });
}

// The pane the keyboard is in, or null while it is anywhere else.
function holding() {
  for (const pane of panes.values()) {
    if (pane.element.contains(document.activeElement)) return pane;
  }
  return null;
}

// One pane, opened where the plot session asked for one. What comes back is
// what the session's window handle calls into: everything here is a method
// Python names, and nothing else on this side is reachable from there.
export function open(number, debounce, commands, strip, handlers) {
  const pane = new Pane(number, debounce, commands, strip, handlers);
  panes.set(number, pane);
  return pane;
}

// The one thing the page says that is about no pane in particular. The executor
// on the Python side is what it reaches, and hearing it is what lets the next
// sampling go out.
export function attend(handlers) {
  landed = handlers.landed;
}

// Every pane at once, for a program that is ending.
export function stop() {
  for (const pane of [...panes.values()]) pane.dismiss();
  panes.clear();
}

// One message off the engine worker. Everything the engine itself says is a
// pickle and belongs to Python; a sampling answer is typed arrays and belongs
// here, and the two are told apart by the field naming the pane.
export function heard(message) {
  if (message === null || typeof message !== 'object' || message.pane === undefined) {
    return;
  }
  const pane = panes.get(message.pane);
  if (pane !== undefined) pane.answered(message);
  // The acknowledgement goes out whether or not there was anything to draw:
  // it is what lets go of the request in flight, and a pane closed mid-sample
  // must not stop the queue.
  if (message.number !== undefined) landed(message.number, message.trouble || '');
}

// -- one pane -------------------------------------------------------------------

class Pane {
  constructor(number, debounce, commands, strip, handlers) {
    this.number = number;
    // How long the view has to stand still before the curves are sampled for
    // it, which is `plot/resample.py`'s figure and not this file's: long enough
    // that a drag is one re-sample rather than sixty, short enough that letting
    // go of the mouse feels like the end of the gesture.
    this.debounce = debounce;
    // What this pane offers, as `plot/controls.py` describes it: the keys, the
    // buttons and the words are read off this and are spelled nowhere here.
    this.commands = commands;
    // The parameter range as `plot/forms.py` describes it, which is the one
    // strip of fields a flat pane has.
    this.description = strip;
    this.say = handlers;
    this.plots = new Map();
    this.order = [];
    this.shown = { x0: -5, x1: 5, y0: -5, y1: 5 };
    this.polar = false;
    // Whether the two axes show the same number of units per pixel, which is
    // what a fresh pane opens on and what makes a circle round.
    this.equal = true;
    this.ruled = true;
    this.listed = true;
    this.tracing = null;
    this.traceAt = 0;
    this.tracePoint = null;
    // When and where the last click on the picture was, which is how the second
    // one is known to be the second.
    this.doubling = null;
    // The traced point as the worker spelled it, which is what Ctrl+C carries
    // while the marker is up.
    this.tracedText = '';
    this.features = null;
    this.message = '';
    this.reading = '';
    this.announce = '';
    this.waiting = false;
    this.pending = null;
    this.featureBackwards = false;
    // Which plot the parameter range fields are about, or null while the pane
    // holds nothing with a parameter in it. Python says which.
    this.ranged = null;
    // The dialog now up over the pane, so that a second `Set range...` replaces
    // its own overlay rather than standing a second one on top of it.
    this.sheet = null;
    // Whether what is being drawn is going to leave the pane. Set for the one
    // redraw a photograph is taken off and cleared again before anything else
    // is painted, so the screen never shows the paper colors.
    this.papered = false;
    this._build();
    this._frame();
    this._home();
  }

  // -- the furniture ---------------------------------------------------------

  _build() {
    const pane = document.createElement('div');
    pane.className = 'plot-pane';
    pane.tabIndex = 0;
    place.pane(pane, PANE_WIDTH, PANE_HEIGHT);
    pane.innerHTML =
      '<div class="plot-bar"><span class="plot-title"></span>' +
      '<button class="plot-close" title="Close">×</button></div>' +
      '<div class="plot-tools"></div>' +
      '<div class="plot-canvas"><div class="plot-legend"></div></div>' +
      '<div class="plot-status"></div>';
    this.element = pane;
    this.bar = pane.querySelector('.plot-bar');
    this.titleText = pane.querySelector('.plot-title');
    this.tools = pane.querySelector('.plot-tools');
    this.canvas = pane.querySelector('.plot-canvas');
    this.card = pane.querySelector('.plot-legend');
    this.status = pane.querySelector('.plot-status');
    // The keyboard is taken back from the button before the command runs rather
    // than after it, so that a command which raises a dialog keeps the focus it
    // gives to the dialog's first field.
    this.buttons = controls.bar(this.tools, this.commands, (name) => {
      this.element.focus();
      this.say.command(name, null);
    });
    // The parameter range stands after the buttons and only while there is a
    // plot for it to be about: a range of nothing is not a control.
    this.range = sheets.strip(this.tools, this.description, () => this._ranged());
    this.range.show(false);
    surface().appendChild(pane);
    pane.querySelector('.plot-close').addEventListener('click', () => {
      this.dismiss();
      this.say.closed();
    });
    pane.addEventListener('pointerdown', () => {
      this.raise();
      this.say.touched();
    });
    pane.addEventListener('keydown', (event) => this._pressed(event));
    this._movable();
    this._watch();
    pane.focus();
  }

  // Dragging by the title bar, which is what a pane has instead of a window
  // manager. Nothing here is a resize: the pane's own corner does that, and the
  // observer below hears about it.
  //
  // The pointer is captured by the bar, so the drag follows a finger or a mouse
  // that has left it - which every drag does, a bar being twenty pixels tall.
  //
  // The keyboard is handed over by hand, as it is over the picture. Cancelling
  // the press is what stops the drag from selecting the title as text, and it
  // also cancels the focus the press would have given the pane - so a pane
  // taken hold of by its bar would answer to none of its keys.
  _movable() {
    let from = null;
    this.bar.addEventListener('pointerdown', (event) => {
      if (event.button !== 0 || event.target.tagName === 'BUTTON') return;
      from = { x: event.clientX, y: event.clientY, left: this.element.offsetLeft,
               top: this.element.offsetTop };
      capture(this.bar, event);
      this.element.focus();
      event.preventDefault();
    });
    this.bar.addEventListener('pointermove', (event) => {
      if (from === null) return;
      this.element.style.left = `${from.left + event.clientX - from.x}px`;
      this.element.style.top = `${from.top + event.clientY - from.y}px`;
    });
    const dropped = () => { from = null; };
    this.bar.addEventListener('pointerup', dropped);
    this.bar.addEventListener('pointercancel', dropped);
  }

  _watch() {
    const observer = new ResizeObserver(() => this._resized());
    observer.observe(this.canvas);
    this.observer = observer;
  }

  _resized() {
    const width = Math.max(this.canvas.clientWidth, 120);
    const height = Math.max(this.canvas.clientHeight, 120);
    if (this.plot === undefined) return;
    this.plot.setSize({ width, height });
    // Through `_look`, so that a pane made taller keeps its scales equal rather
    // than stretching what it was showing.
    const { x0, x1, y0, y1 } = this.shown;
    this._look(x0, x1, y0, y1);
  }

  // uPlot with no series to speak of. The two it is given are the least it will
  // take, and neither is drawn: the data is a rectangle that keeps the library
  // happy while every scale it uses is set by hand.
  _frame() {
    const font = '11px "DejaVu Sans", "Liberation Sans", system-ui, sans-serif';
    const axis = {
      stroke: MUTED,
      font,
      // What the axis is a plot against, where Python has a name for it. Null
      // rather than empty, since an axis with a label of no words still takes
      // the room a label needs.
      label: null,
      labelFont: font,
      labelSize: 14,
      grid: { show: true, stroke: GRID_COLOR, width: 1 },
      ticks: { show: true, stroke: GRID_COLOR, width: 1, size: 4 },
    };
    this.plot = new uPlot(
      {
        width: Math.max(this.canvas.clientWidth, 120),
        height: Math.max(this.canvas.clientHeight, 120),
        padding: [10, 14, 4, 4],
        cursor: { show: false, drag: { x: false, y: false } },
        legend: { show: false },
        scales: { x: { time: false, auto: false }, y: { auto: false } },
        series: [{}, { show: false }],
        axes: [axis, { ...axis }],
        hooks: { draw: [(u) => this._draw(u)] },
      },
      [[0, 1], [0, 0]],
      this.canvas,
    );
    this.plot.root.style.background = BACKGROUND;
    this._gestures(this.plot.over);
  }

  // The default framing is Python's - `plot/view.py` says what a fresh window
  // shows, and a browser opening on a different rectangle from a desktop would
  // be two programs. Every framing after this one arrives through `reframe`,
  // which is the same call with the numbers already worked out.
  _home() {
    this._locked(true);
    const [x0, x1, y0, y1] = this.say.home(
      this.plot.bbox.width / devicePixelRatio,
      this.plot.bbox.height / devicePixelRatio,
    );
    this.shown = { x0, x1, y0, y1 };
    this._scaled();
    this.moved();
  }

  _scaled() {
    this.plot.setScale('x', { min: this.shown.x0, max: this.shown.x1 });
    this.plot.setScale('y', { min: this.shown.y0, max: this.shown.y1 });
  }

  // In front of the others, and still holding the keyboard. Moving an element
  // in the DOM takes it out and puts it back, and an element taken out is an
  // element blurred - so the pane that was just clicked would lose the keys the
  // click was meant to give it. Hence the check and the refocus.
  raise() {
    const layer = surface();
    if (layer.lastElementChild !== this.element) {
      layer.appendChild(this.element);
      this.element.focus();
    }
  }

  // -- what the plot session calls -------------------------------------------

  add(serial, spec) {
    this.plots.set(serial, {
      serial,
      ...spec,
      xs: null,
      ys: null,
      // The parameter each sample came from, which only a curve ridden at one
      // has and which is how a click on such a curve names a place on it.
      ts: null,
      region: null,
      regionPaper: null,
      extent: null,
      bounds: null,
      generation: 0,
      fresh: false,
    });
    if (!this.order.includes(serial)) this.order.push(serial);
    this._relabel();
  }

  respec(serial, spec) {
    const plot = this.plots.get(serial);
    if (plot === undefined) return;
    Object.assign(plot, spec);
    this._relabel();
    this.plot.redraw();
  }

  remove(serial) {
    this.plots.delete(serial);
    this.order = this.order.filter((one) => one !== serial);
    if (this.tracing === serial) this._traceOff();
    this._relabel();
    this.plot.redraw();
  }

  // A sampling has been asked for. What is worth remembering is which one, so
  // that an answer about a view that has moved on can be dropped.
  starting(serial, generation, fresh) {
    const plot = this.plots.get(serial);
    if (plot === undefined) return;
    plot.generation = generation;
    plot.fresh = plot.fresh || fresh;
  }

  present() {
    this.element.style.display = '';
    this.raise();
    this.element.focus();
  }

  retitle(title, current) {
    this.titleText.textContent = title;
    this.element.classList.toggle('current', Boolean(current));
  }

  // What the axes are plots against, as Python worked it out off the plot list.
  // The room uPlot leaves for an axis depends on whether it has a label, so a
  // name that changed lays the picture out again rather than merely redrawing
  // it - and one that did not change does neither.
  named(across, up) {
    const [one, two] = [across || null, up || null];
    if (this.plot.axes[0].label === one && this.plot.axes[1].label === two) return;
    this.plot.axes[0].label = one;
    this.plot.axes[1].label = two;
    this._resized();
  }

  // Which plot the parameter range fields are about, under what name, and the
  // two bounds as Python spells them. Nothing on this side writes a number into
  // a field: what stands there is what the range came back as.
  parametrized(serial, name, low, high) {
    this.ranged = serial === null || serial === undefined ? null : serial;
    this.range.show(this.ranged !== null);
    if (this.ranged === null) return;
    this.range.rename('parameter', { parameter: name });
    const fields = this.range.fields;
    if (!fields.get('low').matches(':focus')) fields.get('low').value = low;
    if (!fields.get('high').matches(':focus')) fields.get('high').value = high;
  }

  // Look at exactly this rectangle, with the scales locked or not as the
  // framing that asked for it left them. Every framing Python decides - the
  // history, the four typed bounds, Home - arrives here.
  reframe(x0, x1, y0, y1, equal) {
    this._locked(Boolean(equal));
    this._look(x0, x1, y0, y1);
  }

  // The `1:1` button: hold the two axes to the same scale, or let them go.
  equalize() {
    this._locked(!this.equal);
    const { x0, x1, y0, y1 } = this.shown;
    this._look(x0, x1, y0, y1);
  }

  // One form, put up over the pane and filled with the values it opens on.
  // What is asked for and in what words is Python's; what comes back is the
  // text, in the order the description lists the fields.
  ask(form, values) {
    if (this.sheet !== null) this.sheet.close();
    this.sheet = sheets.ask(
      this.element,
      form,
      values,
      (said, role) => this.say.typed(form.name, said, role),
      () => {
        this.sheet = null;
        // The keyboard goes back to the pane, which is where it was before the
        // menu that raised the dialog took it away.
        this.element.focus();
      },
    );
  }

  // The view as Python reads it: four bounds and the size of the canvas, which
  // is the tolerance every sampling is measured against.
  view() {
    return [
      this.shown.x0,
      this.shown.x1,
      this.shown.y0,
      this.shown.y1,
      this.plot.bbox.width / devicePixelRatio,
      this.plot.bbox.height / devicePixelRatio,
    ];
  }

  // -- the picture, off the pane ----------------------------------------------

  // The command's copy, which is what the menu entry reaches. The key reaches
  // `_copied` below instead, and both end on the same two roads: a traced point
  // is text, and everything else is the picture.
  copy() {
    const text = this.tracing !== null ? this.tracedText : '';
    if (!text) {
      this._copyImage();
      return;
    }
    const clipboard = navigator.clipboard;
    if (clipboard === undefined || clipboard.writeText === undefined) {
      this.say.copied(text, 'navigator.clipboard is unavailable');
      return;
    }
    clipboard.writeText(text).then(
      () => this.say.copied(text, ''),
      (refused) => this.say.copied(text, String(refused)),
    );
  }

  // Ctrl+C, arriving as the event it has to arrive as. The default is cancelled
  // so that a pane with nothing selected does not copy an empty string over
  // whatever was on the clipboard.
  //
  // A traced point goes through the event's own clipboard, which is the road
  // that needs no permission of anybody: the browser is offering to be written
  // into, and text is what it will take. A picture cannot go that way - a
  // DataTransfer carries strings - so it goes through `navigator.clipboard`,
  // which is granted or refused, and a refusal is a sentence and not a silence.
  _copied(event) {
    if (controls.evented(this.commands, 'copy') === null) return;
    event.preventDefault();
    const text = this.tracing !== null ? this.tracedText : '';
    if (text && event.clipboardData !== null) {
      event.clipboardData.setData('text/plain', text);
      this.say.copied(text, '');
      return;
    }
    this._copyImage();
  }

  _copyImage() {
    const clipboard = navigator.clipboard;
    if (typeof ClipboardItem === 'undefined' || clipboard === undefined
        || clipboard.write === undefined) {
      this.say.copied('', 'ClipboardItem is unavailable');
      return;
    }
    this._photograph().then((shot) => {
      shot.toBlob((blob) => {
        if (blob === null) {
          this.say.copied('', 'canvas.toBlob gave nothing');
          return;
        }
        clipboard.write([new ClipboardItem({ 'image/png': blob })]).then(
          () => this.say.copied('', ''),
          (refused) => this.say.copied('', String(refused)),
        );
      }, 'image/png');
    }, (refused) => this.say.copied('', String(refused)));
  }

  // Export, which in a tab is a download: an object URL and a link clicked from
  // here, since that is the only way a file leaves a page.
  //
  // A PNG and deliberately not an SVG. The desktop's export is a painter path
  // replayed onto a vector device, and there is no such path here: every stroke
  // of this picture is drawn straight onto a canvas, a shaded region *is* a
  // bitmap, and writing an SVG would mean a second renderer for every kind -
  // parity of code where the plan asks for parity of capability. So what leaves
  // is the picture at the size it is drawn, and Python's sentence says the size
  // rather than letting anybody discover it later.
  export() {
    const name = `plot${this.number}.png`;
    this._photograph().then((shot) => {
      shot.toBlob((blob) => {
        if (blob === null) {
          this.say.exported(name, 0, 0, 'canvas.toBlob gave nothing');
          return;
        }
        try {
          const address = URL.createObjectURL(blob);
          const link = document.createElement('a');
          link.href = address;
          link.download = name;
          document.body.appendChild(link);
          link.click();
          link.remove();
          // Freed on the next turn of the loop: revoked while the download is
          // still starting, the file is taken away from it.
          setTimeout(() => URL.revokeObjectURL(address), 0);
        } catch (refused) {
          this.say.exported(name, 0, 0, String(refused));
          return;
        }
        this.say.exported(name, shot.width, shot.height, '');
      }, 'image/png');
    }, (refused) => this.say.exported(name, 0, 0, String(refused)));
  }

  // The pane as one image: the ground it is drawn on, the picture, and the
  // names of what is in it, all in the colors a picture is read on paper in.
  //
  // The ground is painted rather than inherited - the canvas itself is
  // transparent and the color is the element's - so a picture pasted into a
  // document is not a picture of white curves on nothing. The legend is a card
  // floating over the canvas rather than anything drawn into it, so the names
  // are written on afterwards, exactly as the desktop's export writes them.
  _photograph() {
    return this._onPaper(() => {
      const source = this.plot.ctx.canvas;
      const shot = document.createElement('canvas');
      shot.width = source.width;
      shot.height = source.height;
      const ctx = shot.getContext('2d');
      ctx.fillStyle = PAPER;
      ctx.fillRect(0, 0, shot.width, shot.height);
      ctx.drawImage(source, 0, 0);
      this._namePlots(ctx, shot.width, devicePixelRatio);
      return shot;
    });
  }

  // The pane in paper colors for as long as a photograph takes, which is
  // `plot/qt/window2d.py`'s `_on_paper` on a canvas.
  //
  // What the curves, the shading and the axes through the origin are drawn in
  // the pane reads off `papered` as it draws them. What the numbers along the
  // axes are drawn in is uPlot's, so its two strokes are swapped and put back.
  //
  // The picture is waited for rather than assumed: uPlot commits a redraw in a
  // microtask rather than in the call, and a photograph taken in the call
  // would be a photograph of the colors the pane had before. Waiting for a
  // microtask is not waiting for a frame - the two swaps still fall between
  // two paints - so the file is on paper and the pane on the screen stays dark
  // throughout, exactly as the desktop's window does.
  async _onPaper(take) {
    const strokes = this.plot.axes.map((axis) => axis.stroke);
    this.papered = true;
    for (const axis of this.plot.axes) axis.stroke = () => PAPER_TEXT;
    this.plot.redraw(false, true);
    try {
      await null;
      return take();
    } finally {
      this.papered = false;
      this.plot.axes.forEach((axis, at) => { axis.stroke = strokes[at]; });
      this.plot.redraw(false, true);
    }
  }

  _namePlots(ctx, wide, ratio) {
    if (!this.listed || this.order.length === 0) return;
    const size = LEGEND_PX * ratio;
    ctx.save();
    ctx.font = `${size}px "DejaVu Sans", "Liberation Sans", system-ui, sans-serif`;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'top';
    let down = LEGEND_MARGIN_PX * ratio;
    for (const serial of this.order) {
      const plot = this.plots.get(serial);
      if (plot === undefined || plot.hidden) continue;
      ctx.fillStyle = this._ink(plot);
      ctx.fillText(plot.name, wide - LEGEND_MARGIN_PX * ratio, down);
      down += size * 1.4;
    }
    ctx.restore();
  }

  dismiss() {
    if (this.sheet !== null) this.sheet.close();
    // Where the keys go next. A pane that had them - the close button it was
    // shut with is inside it - would otherwise leave them on nothing, and the
    // program would look as though it had stopped listening.
    const held = this.element.contains(document.activeElement);
    if (this.observer) this.observer.disconnect();
    if (this.plot) this.plot.destroy();
    this.element.remove();
    panes.delete(this.number);
    if (held) focus();
  }

  // -- what the worker says ---------------------------------------------------

  answered(message) {
    if (message.reply === 'trace') return this._readTrace(message);
    if (message.reply === 'features') return this._readFeatures(message);
    const plot = this.plots.get(message.plot);
    if (plot === undefined || message.generation !== plot.generation) return;
    if (message.trouble) {
      this.said(`${plot.label}: ${message.trouble}`);
      return;
    }
    if (message.shape === 'region') {
      // Both washes at once, the truth grid being what either is made of and
      // arriving only here. Made once when the samples land rather than once a
      // frame, since a drag redraws sixty times over a grid the worker sent
      // once - and a picture leaving the pane has its own wash ready.
      plot.region = shading(message, plot.color);
      plot.regionPaper = shading(message, plot.paper);
      plot.extent = message.extent;
      plot.xs = null;
      plot.ts = null;
    } else {
      plot.xs = typed(message.xs, plot.label);
      plot.ys = typed(message.ys, plot.label);
      plot.ts = message.ts ? typed(message.ts, plot.label) : null;
      plot.bounds = message.bounds && message.bounds.length ? message.bounds : null;
      if (message.trange) {
        plot.trange = message.trange;
        // The range came back with the arrays, so this side is the first to
        // know it; what the fields say about it is Python's to spell, as every
        // other number in a field of this pane is.
        this.say.spanned(plot.serial, message.trange[0], message.trange[1]);
      }
    }
    if (!message.partial && plot.fresh) {
      plot.fresh = false;
      this._fit(plot);
    }
    if (message.words) this.said(message.words);
    this.plot.redraw();
    if (!message.partial && this.tracing === plot.serial) this._retrace();
  }

  _readTrace(message) {
    const plot = this.plots.get(message.plot);
    this.waiting = false;
    if (plot === undefined || this.tracing !== plot.serial) return;
    this.tracePoint = message.found ? { x: message.x, y: message.y } : null;
    this.tracedText = message.point || '';
    this.reading = message.found ? message.words : '';
    // A feature was asked for by moving the marker to it, so the reading comes
    // back after the sentence that named it; the sentence is what the status
    // line is for and wins where there is one.
    this.said(this.announce || message.words);
    this.announce = '';
    this.plot.redraw();
    // A step asked for while the last one was still out is remembered rather
    // than dropped, so that holding an arrow key moves the marker rather than
    // moving it once and stopping.
    const next = this.pending;
    this.pending = null;
    if (next !== null) this._ask(next);
  }

  _readFeatures(message) {
    const plot = this.plots.get(message.plot);
    if (plot === undefined) return;
    if (message.words) this.said(message.words);
    this.features = { generation: message.generation, found: message.features };
    this._step(this.featureBackwards);
  }

  said(text) {
    this.message = text || '';
    this.status.textContent = this.message;
  }

  // -- drawing ----------------------------------------------------------------

  // What one plot is drawn in: the color it is read on the screen in, or the
  // one it is read on paper in while a picture is being taken. Both are the
  // plot list's own, so a curve copied out of a pane is the color the same
  // curve copied out of a window is.
  _ink(plot) {
    return this.papered ? plot.paper : plot.color;
  }

  _draw(u) {
    const ctx = u.ctx;
    const ratio = devicePixelRatio;
    ctx.save();
    ctx.beginPath();
    ctx.rect(u.bbox.left, u.bbox.top, u.bbox.width, u.bbox.height);
    ctx.clip();
    if (this.polar) this._polarGrid(u, ctx, ratio);
    this._axisLines(u, ctx, ratio);
    for (const serial of this.order) {
      const plot = this.plots.get(serial);
      if (plot === undefined || plot.hidden) continue;
      if (plot.region !== null) this._region(u, ctx, plot);
    }
    for (const serial of this.order) {
      const plot = this.plots.get(serial);
      if (plot === undefined || plot.hidden || plot.xs === null) continue;
      if (plot.kind === DATA) this._points(u, ctx, plot, ratio);
      else this._stroke(u, ctx, plot, ratio);
    }
    this._marker(u, ctx, ratio);
    ctx.restore();
  }

  // The axis lines go through the origin, because that is where mathematics
  // puts them, and at the edge nearest it where the view is framed away - the
  // clamp is `plot/view.py`'s `axis_at`, which is one expression and is written
  // here rather than asked for once per frame.
  _axisLines(u, ctx, ratio) {
    const across = (this.shown.x1 - this.shown.x0) / Math.max(u.bbox.width / ratio, 1);
    const up = (this.shown.y1 - this.shown.y0) / Math.max(u.bbox.height / ratio, 1);
    const at = (low, high, pixel) => Math.min(Math.max(0, low + pixel), high - pixel);
    ctx.save();
    ctx.strokeStyle = this.papered ? PAPER_AXIS : AXIS_COLOR;
    ctx.lineWidth = ratio;
    ctx.beginPath();
    const x = u.valToPos(at(this.shown.x0, this.shown.x1, across), 'x', true);
    const y = u.valToPos(at(this.shown.y0, this.shown.y1, up), 'y', true);
    ctx.moveTo(x, u.bbox.top);
    ctx.lineTo(x, u.bbox.top + u.bbox.height);
    ctx.moveTo(u.bbox.left, y);
    ctx.lineTo(u.bbox.left + u.bbox.width, y);
    ctx.stroke();
    ctx.restore();
  }

  // The rings and spokes a polar picture is read against. Not a library
  // requirement and never was: the curve is sampled as x = r cos θ, y = r sin θ
  // where the numbers are, and what is left is a few dozen lines of arcs.
  _polarGrid(u, ctx, ratio) {
    const reach = Math.max(
      Math.abs(this.shown.x0), Math.abs(this.shown.x1),
      Math.abs(this.shown.y0), Math.abs(this.shown.y1),
    );
    const step = ruled(reach);
    if (!(step > 0)) return;
    const origin = { x: u.valToPos(0, 'x', true), y: u.valToPos(0, 'y', true) };
    const unit = Math.abs(u.valToPos(step, 'x', true) - origin.x);
    if (!(unit > 2)) return;
    ctx.save();
    ctx.strokeStyle = GRID_COLOR;
    ctx.lineWidth = ratio;
    for (let radius = unit; radius <= reach / step * unit + unit; radius += unit) {
      ctx.beginPath();
      ctx.arc(origin.x, origin.y, radius, 0, 2 * Math.PI);
      ctx.stroke();
    }
    const far = Math.hypot(u.bbox.width, u.bbox.height);
    for (let turn = 0; turn < 12; turn += 1) {
      const angle = (turn * Math.PI) / 6;
      ctx.beginPath();
      ctx.moveTo(origin.x, origin.y);
      ctx.lineTo(origin.x + far * Math.cos(angle), origin.y - far * Math.sin(angle));
      ctx.stroke();
    }
    ctx.restore();
  }

  // One stroke, with a gap wherever the samples say there is one. A NaN is how
  // the sampler marks a jump, a pole and the end of a contour segment, and a
  // canvas that was handed one would quietly join the two sides of it - so the
  // path is broken here and started again on the far side.
  _stroke(u, ctx, plot, ratio) {
    ctx.save();
    ctx.strokeStyle = this._ink(plot);
    ctx.lineWidth = CURVE_WIDTH * ratio;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    let down = false;
    const xs = plot.xs;
    const ys = plot.ys;
    for (let index = 0; index < xs.length; index += 1) {
      const x = xs[index];
      const y = ys[index];
      if (!Number.isFinite(x) || !Number.isFinite(y)) {
        down = false;
        continue;
      }
      const px = u.valToPos(x, 'x', true);
      const py = u.valToPos(y, 'y', true);
      if (down) ctx.lineTo(px, py);
      else ctx.moveTo(px, py);
      down = true;
    }
    ctx.stroke();
    ctx.restore();
  }

  _points(u, ctx, plot, ratio) {
    if (plot.connected) this._stroke(u, ctx, plot, ratio);
    ctx.save();
    ctx.fillStyle = this._ink(plot);
    const radius = (plot.size / 2) * ratio;
    for (let index = 0; index < plot.xs.length; index += 1) {
      const x = plot.xs[index];
      const y = plot.ys[index];
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      ctx.beginPath();
      ctx.arc(u.valToPos(x, 'x', true), u.valToPos(y, 'y', true), radius, 0, 2 * Math.PI);
      ctx.fill();
    }
    ctx.restore();
  }

  // A region is the shading the worker's truth grid was turned into, stretched
  // over the rectangle it was evaluated on. Stretched rather than resampled on
  // purpose: the boundary of a region is only known to the grid's accuracy, and
  // a fill drawn sharper than that would be claiming more than was computed.
  _region(u, ctx, plot) {
    const off = this.papered ? plot.regionPaper : plot.region;
    const [x0, x1, y0, y1] = plot.extent;
    const left = u.valToPos(x0, 'x', true);
    const right = u.valToPos(x1, 'x', true);
    const top = u.valToPos(y1, 'y', true);
    const bottom = u.valToPos(y0, 'y', true);
    ctx.save();
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, left, top, right - left, bottom - top);
    ctx.restore();
  }

  // A ring rather than a dot, so the curve goes on through the point it names;
  // a dashed hairline down to the abscissa, because half of what a reading means
  // is where along the axis it is; and the chip, which says what the status line
  // says without the eye having to travel to the bottom of the pane for it.
  _marker(u, ctx, ratio) {
    if (this.tracing === null || this.tracePoint === null) return;
    const plot = this.plots.get(this.tracing);
    if (plot === undefined) return;
    const px = u.valToPos(this.tracePoint.x, 'x', true);
    const py = u.valToPos(this.tracePoint.y, 'y', true);
    ctx.save();
    ctx.strokeStyle = this._ink(plot);
    ctx.globalAlpha = HAIRLINE_ALPHA;
    ctx.setLineDash([4 * ratio, 4 * ratio]);
    ctx.lineWidth = ratio;
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(px, u.valToPos(0, 'y', true));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    ctx.lineWidth = MARKER_WIDTH * ratio;
    ctx.beginPath();
    ctx.arc(px, py, (MARKER_PX / 2) * ratio, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.restore();
    this._chip(u, px, py, plot);
  }

  // The chip stands beside the point and says what the status line says,
  // without the eye having to travel to the bottom of the pane for it. The
  // name of the curve is left off - the legend and the status line both say
  // which curve is being ridden, and the chip has room for numbers.
  _chip(u, px, py, plot) {
    const chip = this.chipElement || (this.chipElement = this._chipElement());
    if (!this.reading) {
      chip.style.display = 'none';
      return;
    }
    chip.style.display = '';
    chip.style.color = plot.color;
    chip.textContent = this.reading.split('   ').slice(1).join('\n');
    // Up and to the right of the marker, which is where the pointer is not,
    // and around to the other side wherever that would put it off the picture.
    const ratio = devicePixelRatio;
    const wide = this.canvas.clientWidth;
    const overRight = px / ratio + CHIP_OFFSET_PX + chip.offsetWidth > wide;
    const overTop = py / ratio - CHIP_OFFSET_PX - chip.offsetHeight < 0;
    chip.style.left = `${px / ratio + (overRight ? -CHIP_OFFSET_PX : CHIP_OFFSET_PX)}px`;
    chip.style.top = `${py / ratio + (overTop ? CHIP_OFFSET_PX : -CHIP_OFFSET_PX)}px`;
    chip.style.transform = `translate(${overRight ? '-100%' : '0'}, ${
      overTop ? '0' : '-100%'
    })`;
  }

  _chipElement() {
    const chip = document.createElement('div');
    chip.className = 'plot-chip';
    this.canvas.appendChild(chip);
    return chip;
  }

  _relabel() {
    if (!this.listed || this.order.length === 0) {
      this.card.style.display = 'none';
      return;
    }
    this.card.style.display = '';
    this.card.textContent = '';
    for (const serial of this.order) {
      const plot = this.plots.get(serial);
      if (plot === undefined) continue;
      const row = document.createElement('div');
      row.className = 'plot-row';
      row.style.opacity = plot.hidden ? LEGEND_FADED : 1;
      const swatch = document.createElement('span');
      swatch.className = 'plot-swatch';
      swatch.style.background = plot.color;
      const name = document.createElement('span');
      // What the app wrote, cut to a line by Python. Nothing here renders any
      // part of an expression.
      name.textContent = plot.name;
      row.append(swatch, name);
      // The left button takes a curve out of the picture and puts it back; the
      // right one opens the menu about that curve, which is the same menu the
      // canvas offers about whatever a click there was over.
      row.addEventListener('pointerdown', (event) => {
        event.stopPropagation();
        if (event.button === 0) this.say.hide(serial, !plot.hidden);
      });
      row.addEventListener('contextmenu', (event) => {
        event.preventDefault();
        event.stopPropagation();
        this._card(event, serial);
      });
      this.card.appendChild(row);
    }
  }

  // -- the mouse ---------------------------------------------------------------

  // Pointer events rather than mouse ones, which is the whole of what a finger
  // needs from a flat picture: the same handler hears a mouse, a finger and a
  // pen, and a drag that leaves the picture is still delivered because the
  // pointer is captured. The one gesture a finger has that a mouse has not is
  // the second finger, and it is what a touch screen has instead of the wheel.
  _gestures(over) {
    let panning = null;
    let banding = null;
    //: Where every pointer now down on the picture is, which is how a pan
    //: knows it has become a pinch.
    const down = new Map();
    let pinch = null;
    const spread = () => {
      const [first, second] = [...down.values()];
      return {
        apart: Math.hypot(first.x - second.x, first.y - second.y),
        x: (first.x + second.x) / 2,
        y: (first.y + second.y) / 2,
      };
    };
    over.addEventListener('contextmenu', (event) => event.preventDefault());
    over.addEventListener('pointerdown', (event) => {
      this.element.focus();
      down.set(event.pointerId, { x: event.offsetX, y: event.offsetY });
      capture(over, event);
      if (down.size === 2) {
        // A second finger is a pinch and not a pan, so whatever the first one
        // had started is given up rather than left half done.
        panning = null;
        banding = null;
        this._band(null);
        pinch = spread();
        this._remember();
        event.preventDefault();
        return;
      }
      if (event.button === 0) {
        panning = { x: event.offsetX, y: event.offsetY, view: { ...this.shown } };
        this._remember();
      } else if (event.button === 2) {
        banding = { x: event.offsetX, y: event.offsetY, moved: 0 };
        this._band(banding);
      }
      event.preventDefault();
    });
    over.addEventListener('pointermove', (event) => {
      if (down.has(event.pointerId)) {
        down.set(event.pointerId, { x: event.offsetX, y: event.offsetY });
      }
      if (pinch !== null && down.size === 2) {
        const now = spread();
        if (pinch.apart > 0 && now.apart > 0) {
          this._zoom(pinch.apart / now.apart, now.x, now.y, over);
        }
        pinch = now;
        return;
      }
      if (panning !== null) {
        const across = (this.shown.x1 - this.shown.x0) / over.clientWidth;
        const up = (this.shown.y1 - this.shown.y0) / over.clientHeight;
        const dx = (event.offsetX - panning.x) * across;
        const dy = (event.offsetY - panning.y) * up;
        this._look(
          panning.view.x0 - dx, panning.view.x1 - dx,
          panning.view.y0 + dy, panning.view.y1 + dy,
        );
        return;
      }
      if (banding !== null) {
        banding.moved = Math.max(
          banding.moved,
          Math.abs(event.offsetX - banding.x) + Math.abs(event.offsetY - banding.y),
        );
        this._band(banding, event.offsetX, event.offsetY);
        return;
      }
      this._pointed(event.offsetX, event.offsetY);
    });
    // A gesture is over before it is acted on, rather than after. What the
    // release comes to - a rectangle to look at, or a menu - can be a long way
    // from here, and a band still on the picture because something along that
    // road went wrong would follow the pointer about with no button held.
    const release = (event) => {
      down.delete(event.pointerId);
      if (down.size < 2) pinch = null;
      const banded = banding;
      const panned = panning;
      panning = null;
      banding = null;
      if (banded === null) {
        // A left press let go where it was pressed is a click and not a pan of
        // no distance, and a click on the picture is a gesture of its own.
        if (panned !== null && event.type === 'pointerup') {
          const moved = Math.abs(event.offsetX - panned.x) +
            Math.abs(event.offsetY - panned.y);
          if (moved <= CLICK_SLOP_PX) this._clicked(event);
        }
        return;
      }
      this._band(null);
      if (banded.moved > CLICK_SLOP_PX) {
        this._zoomTo(banded, event.offsetX, event.offsetY, over);
      } else {
        this._menu(event);
      }
    };
    over.addEventListener('pointerup', release);
    over.addEventListener('pointercancel', release);
    over.addEventListener('pointerleave', () => {
      panning = null;
      banding = null;
      this._band(null);
    });
    over.addEventListener('wheel', (event) => {
      event.preventDefault();
      this._remember();
      // Shift holds the width and Ctrl the height, which is how one axis is
      // stretched against the other, and is what the desktop's wheel takes them
      // to mean as well.
      this._zoom(
        wheeled(event),
        event.offsetX,
        event.offsetY,
        over,
        !event.shiftKey,
        !event.ctrlKey,
      );
    }, { passive: false });
  }

  // Zoom about a point of the picture: the wheel's gesture and the pinch's,
  // which differ in where the factor comes from and in nothing else.
  _zoom(factor, x, y, over, wide = true, tall = true) {
    const at = {
      x: this.shown.x0 + (x / over.clientWidth) * (this.shown.x1 - this.shown.x0),
      y: this.shown.y1 - (y / over.clientHeight) * (this.shown.y1 - this.shown.y0),
    };
    if (!wide || !tall) this._unlock();
    this._look(
      wide ? at.x + (this.shown.x0 - at.x) * factor : this.shown.x0,
      wide ? at.x + (this.shown.x1 - at.x) * factor : this.shown.x1,
      tall ? at.y + (this.shown.y0 - at.y) * factor : this.shown.y0,
      tall ? at.y + (this.shown.y1 - at.y) * factor : this.shown.y1,
    );
  }

  // Where a place on the picture is, in the units the picture is drawn in.
  _value(x, y) {
    const over = this.plot.over;
    return {
      x: this.shown.x0 + (x / over.clientWidth) * (this.shown.x1 - this.shown.x0),
      y: this.shown.y1 - (y / over.clientHeight) * (this.shown.y1 - this.shown.y0),
    };
  }

  // The pointer readout: where the pointer is, which is the one number on this
  // side that is not a value of anything and so is the one this file may write.
  // And the marker, while one is up and is the kind that follows the pointer.
  //
  // A marker riding a parametric curve stays where it is: its place is a
  // parameter value, and the pointer's x is not one - the arrow keys and a click
  // are how it moves. A marker on a function follows the pointer, which is what
  // makes trace feel like pointing at the curve.
  //
  // The readout gives way to the marker, because a pane has one line to say
  // things on where the desktop's window has two: there the pointer's position
  // is a widget of its own beside the status line the trace sentence is written
  // on, and here they would be the same line. What a person is reading while a
  // marker is up is the curve's own value, which arrives on that line with every
  // move the marker makes; a readout would take it away again as fast as it came.
  _pointed(x, y) {
    const at = this._value(x, y);
    if (this.tracing !== null) {
      const plot = this.plots.get(this.tracing);
      if (plot !== undefined && !PARAMETRIZED.has(plot.kind)) {
        this.traceAt = at.x;
        this._retrace();
      }
      return;
    }
    if (this.polar) {
      this.said(
        `r: ${Math.hypot(at.x, at.y).toFixed(6)}` +
        `   θ: ${Math.atan2(at.y, at.x).toFixed(6)}`,
      );
      return;
    }
    this.said(`x: ${at.x.toFixed(6)}   y: ${at.y.toFixed(6)}`);
  }

  // A left click on the picture: the middle of the view, or a curve.
  //
  // Two clicks in the same place put that place in the middle of the canvas, and
  // one on a curve takes hold of it. Both are the desktop's gestures, and there
  // as here the first of a pair of clicks is still a click: a double click over
  // a curve traces it and then centres on it.
  _clicked(event) {
    const at = this._value(event.offsetX, event.offsetY);
    const before = this.doubling;
    const twice = before !== null &&
      event.timeStamp - before.when < DOUBLE_MS &&
      Math.abs(event.offsetX - before.x) + Math.abs(event.offsetY - before.y) < DOUBLE_PX;
    this.doubling = twice
      ? null
      : { when: event.timeStamp, x: event.offsetX, y: event.offsetY };
    if (twice) {
      // The arithmetic and the history are `plot/actions.py`'s, as they are for
      // every other framing: what crosses is the point that was clicked.
      this.say.centred(at.x, at.y);
      return;
    }
    const plot = this._pointedPlot(event);
    if (plot !== null) this._takeHold(plot, at);
  }

  // Trace `plot`, at the place the click was pointing at.
  //
  // Which place that is depends on what the curve is parametrized by. A function
  // is ridden at an abscissa and the click names one; a parametric or polar curve
  // is ridden at a parameter, which a point in the plane does not name at all -
  // so the click snaps to the nearest point that was actually sampled, and the
  // parameter it came from is where the marker goes.
  _takeHold(plot, at) {
    if (!RIDEABLE.has(plot.kind)) return;
    const held = this.tracing === plot.serial;
    if (!held) this.features = null;
    this.tracing = plot.serial;
    if (!PARAMETRIZED.has(plot.kind)) {
      this.traceAt = at.x;
    } else {
      const nearest = this._nearest(plot, at);
      if (nearest !== null) this.traceAt = nearest;
      else if (!held) this.traceAt = middle(plot.trange);
    }
    this._retrace();
  }

  // The parameter of the sampled point nearest a place on the picture, or null
  // where the curve has no samples to measure against. Measured in pixels, which
  // is what the eye measures it in: how near a curve looks is how near it is.
  _nearest(plot, at) {
    if (plot.xs === null || plot.ts === null) return null;
    const over = this.plot.over;
    const across = (this.shown.x1 - this.shown.x0) / Math.max(over.clientWidth, 1);
    const up = (this.shown.y1 - this.shown.y0) / Math.max(over.clientHeight, 1);
    let best = null;
    for (let index = 0; index < plot.ts.length; index += 1) {
      const away = Math.hypot(
        (plot.xs[index] - at.x) / across, (plot.ys[index] - at.y) / up,
      );
      if (Number.isFinite(away) && (best === null || away < best.away)) {
        best = { away, at: plot.ts[index] };
      }
    }
    return best === null ? null : best.at;
  }

  _band(state, x, y) {
    if (this.bandElement === undefined) {
      this.bandElement = document.createElement('div');
      this.bandElement.className = 'plot-band';
      this.canvas.appendChild(this.bandElement);
    }
    if (state === null || x === undefined) {
      this.bandElement.style.display = 'none';
      return;
    }
    this.bandElement.style.display = '';
    this.bandElement.style.left = `${Math.min(state.x, x)}px`;
    this.bandElement.style.top = `${Math.min(state.y, y)}px`;
    this.bandElement.style.width = `${Math.abs(x - state.x)}px`;
    this.bandElement.style.height = `${Math.abs(y - state.y)}px`;
  }

  _zoomTo(band, x, y, over) {
    const value = (px, py) => ({
      x: this.shown.x0 + (px / over.clientWidth) * (this.shown.x1 - this.shown.x0),
      y: this.shown.y1 - (py / over.clientHeight) * (this.shown.y1 - this.shown.y0),
    });
    const one = value(band.x, band.y);
    const two = value(x, y);
    this._remember();
    // A rubber band is a rectangle the user drew, and drawing one is asking for
    // that rectangle rather than for a square-scaled version of it.
    this._unlock();
    this._look(
      Math.min(one.x, two.x), Math.max(one.x, two.x),
      Math.min(one.y, two.y), Math.max(one.y, two.y),
    );
  }

  // -- the view ----------------------------------------------------------------

  _look(x0, x1, y0, y1) {
    if (!(x1 > x0) || !(y1 > y0)) return;
    this.shown = this.equal ? this._squared(x0, x1, y0, y1) : { x0, x1, y0, y1 };
    this._scaled();
    this.moved();
  }

  // Equal scales, so a circle is round. Whichever axis is showing less per
  // pixel is widened about its own centre until the two agree, which is what an
  // aspect lock does: a range is only ever grown, so nothing asked for goes off
  // the picture. Zooming one axis alone releases the lock - that gesture is a
  // request for unequal scales - and Home puts it back.
  _squared(x0, x1, y0, y1) {
    const wide = Math.max(this.plot.bbox.width / devicePixelRatio, 1);
    const tall = Math.max(this.plot.bbox.height / devicePixelRatio, 1);
    const unit = Math.max((x1 - x0) / wide, (y1 - y0) / tall);
    const across = (unit * wide - (x1 - x0)) / 2;
    const up = (unit * tall - (y1 - y0)) / 2;
    return { x0: x0 - across, x1: x1 + across, y0: y0 - up, y1: y1 + up };
  }

  // A gesture that asked for one axis has asked for unequal scales, and gets
  // them until the framing is put back to what a pane opens on.
  _unlock() {
    this._locked(false);
  }

  // The aspect lock, and the button that shows where it stands. The lock is the
  // page's own state - a rubber band releases it without asking anybody - which
  // is why the button is lit from here and not from Python.
  _locked(equal) {
    this.equal = equal;
    this.lit('scales.equal', equal);
  }

  // Say where the view stands, before a gesture moves it off. Once per gesture
  // rather than once per frame, which is what makes a history of somewhere the
  // user was rather than of every pixel they passed through; Python keeps it,
  // because stepping back through one is a command and `plot/view.py` already
  // says what a step comes to.
  _remember() {
    this.say.remembered(this.shown.x0, this.shown.x1, this.shown.y0, this.shown.y1);
  }

  // A view change asks for a sampling, once the view has stopped changing. The
  // keyed job queue on the Python side collapses whatever still gets through.
  moved() {
    clearTimeout(this.timer);
    this.timer = setTimeout(() => this.say.changed(), this.debounce);
    if (this.tracing !== null) this._retrace();
  }

  // The keyed zoom, which is about the middle of the picture: the wheel and the
  // pinch are about the point they happened at, and a key happened nowhere.
  zoom(factor) {
    const over = this.plot.over;
    this._remember();
    this._zoom(factor, over.clientWidth / 2, over.clientHeight / 2, over);
  }

  // Every plot's rectangle, unioned, and whether there was one to find. The
  // rectangles come off the worker with the samples, so nothing here scans an
  // array for one; what is said about the answer is Python's, which is why this
  // reports rather than speaks.
  autoscale() {
    let box = null;
    for (const plot of this.plots.values()) {
      if (plot.hidden || plot.bounds === null) continue;
      box = box === null ? [...plot.bounds] : [
        Math.min(box[0], plot.bounds[0]), Math.max(box[1], plot.bounds[1]),
        Math.min(box[2], plot.bounds[2]), Math.max(box[3], plot.bounds[3]),
      ];
    }
    if (box === null) return false;
    this._unlock();
    this._look(...padded(box));
    return true;
  }

  // A new plot with points but none of them in view. Exactly when the
  // alternative is an empty picture, and never otherwise: a pane whose framing
  // moved under every added curve would be one nobody could compare two curves
  // in. A function is framed in y alone - its x range is the view, which is what
  // it was sampled over.
  _fit(plot) {
    if (plot.bounds === null) return;
    const [left, right, low, high] = plot.bounds;
    const inside =
      right >= this.shown.x0 && left <= this.shown.x1 &&
      high >= this.shown.y0 && low <= this.shown.y1;
    if (inside) return;
    // The aspect lock goes when a window reframes itself to fit: a range chosen
    // to hold a curve is not a range equal scales would have given. What is
    // said about it is Python's - a graph is reframed in y alone and says so -
    // and this side says which of the two happened.
    this._unlock();
    const graph = plot.kind === CURVE || plot.kind === FAMILY;
    if (graph) {
      const pad = 0.1 * (high - low || 1);
      this._look(this.shown.x0, this.shown.x1, low - pad, high + pad);
    } else {
      this._look(...padded([left, right, low, high]));
    }
    this.say.fitted(plot.serial, graph);
  }

  // -- trace and features -------------------------------------------------------

  trace() {
    if (this.tracing !== null) {
      this._traceOff();
      return;
    }
    const rideable = this.order.filter((serial) => {
      const plot = this.plots.get(serial);
      return plot !== undefined && !plot.hidden && RIDEABLE.has(plot.kind);
    });
    if (rideable.length === 0) return;
    this.tracing = rideable[0];
    this.features = null;
    const plot = this.plots.get(this.tracing);
    this.traceAt = PARAMETRIZED.has(plot.kind)
      ? middle(plot.trange)
      : (this.shown.x0 + this.shown.x1) / 2;
    this._retrace();
  }

  _traceOff() {
    this.tracing = null;
    this.tracePoint = null;
    this.reading = '';
    this.announce = '';
    this.features = null;
    this.said('');
    this.plot.redraw();
  }

  _traceCurve(step) {
    const rideable = this.order.filter((serial) => {
      const plot = this.plots.get(serial);
      return plot !== undefined && !plot.hidden && RIDEABLE.has(plot.kind);
    });
    if (rideable.length === 0 || this.tracing === null) return;
    const here = Math.max(rideable.indexOf(this.tracing), 0);
    this.tracing = rideable[(here + step + rideable.length) % rideable.length];
    this.features = null;
    this._retrace();
  }

  _traceStep(backwards, fast) {
    const plot = this.plots.get(this.tracing);
    if (plot === undefined) return;
    const direction = backwards ? -1 : 1;
    if (PARAMETRIZED.has(plot.kind)) {
      const [low, high] = plot.trange || [-Math.PI, Math.PI];
      const share = fast ? STEP_FAST_SHARE : STEP_SHARE;
      this.traceAt = clamp(this.traceAt + direction * share * (high - low), low, high);
    } else {
      const pixels = fast ? NUDGE_FAST_PX : NUDGE_PX;
      const across = (this.shown.x1 - this.shown.x0) /
        Math.max(this.plot.bbox.width / devicePixelRatio, 1);
      this.traceAt += direction * pixels * across;
    }
    this._retrace();
  }

  // One reading in flight at a time, the latest asked for winning: an arrow key
  // held down would otherwise post a request per repeat and read them all back
  // in order, which is a marker arriving where the key was a second ago.
  _retrace() {
    if (this.tracing === null) return;
    this._ask({ serial: this.tracing, at: this.traceAt });
  }

  _ask(asking) {
    if (this.waiting) {
      this.pending = asking;
      return;
    }
    this.waiting = true;
    this.pending = null;
    this.say.traced(asking.serial, asking.at);
  }

  snap(backwards) {
    const plot = this.plots.get(this.tracing);
    if (plot === undefined) return;
    if (!SCANNED.has(plot.kind)) {
      this.said(`Tracing ${plot.name}: features are found on function curves`);
      return;
    }
    this.featureBackwards = backwards;
    if (this.features !== null && this.features.generation === plot.generation) {
      this._step(backwards);
      return;
    }
    this.say.scanned(plot.serial);
  }

  // Past the marker rather than at it, and by more than a pixel, so that
  // pressing Tab twice moves twice.
  _step(backwards) {
    const plot = this.plots.get(this.tracing);
    if (plot === undefined || this.features === null) return;
    const step = (this.shown.x1 - this.shown.x0) /
      Math.max(this.plot.bbox.width / devicePixelRatio, 1);
    const found = this.features.found;
    const next = backwards
      ? [...found].reverse().find((one) => one.x < this.traceAt - step)
      : found.find((one) => one.x > this.traceAt + step);
    if (next === undefined) {
      this.said(
        `Tracing ${plot.name}: no further feature to the ${backwards ? 'left' : 'right'}`,
      );
      return;
    }
    this.traceAt = next.x;
    // The sentence is the worker's, written where the number was found.
    this.announce = `Tracing ${plot.name}: ${next.words}`;
    this._retrace();
    this.said(this.announce);
  }

  sendHome() {
    if (this.tracing === null || !this.tracedText) return;
    this.say.author(this.tracing, this.tracedText);
    this.said(`Sent ${this.tracedText} to the worksheet`);
  }

  // -- the keyboard and the menu ------------------------------------------------

  // The gestures first, since they are what a key means while a marker is up
  // and are named by no entry of any menu; then the ladder, which is the keys
  // the description says this pane answers to. A key that is in neither is left
  // to the browser.
  _pressed(event) {
    // A key typed into a field stays in the field it was typed into, which is
    // the bargain the desktop's toolbar fields make as well. The fields stop
    // their own keys from getting here; this is what keeps a field added later
    // from having to remember to.
    if (event.target.tagName === 'INPUT') return;
    const key = event.key;
    if (key === 'Tab') {
      event.preventDefault();
      this.snap(event.shiftKey);
      return;
    }
    if (key === 'Enter' || key === 'Escape') {
      event.preventDefault();
      if (key === 'Enter') this.sendHome();
      else this._traceOff();
      return;
    }
    if (key === 'ArrowLeft' || key === 'ArrowRight') {
      event.preventDefault();
      if (this.tracing !== null) this._traceStep(key === 'ArrowLeft', event.shiftKey);
      else this._pan(key === 'ArrowLeft' ? -0.25 : 0.25, 0);
      return;
    }
    if (key === 'ArrowUp' || key === 'ArrowDown') {
      event.preventDefault();
      if (this.tracing !== null) this._traceCurve(key === 'ArrowUp' ? -1 : 1);
      else this._pan(0, key === 'ArrowUp' ? 0.25 : -0.25);
      return;
    }
    const command = controls.pressed(this.commands, event);
    if (command !== null) {
      event.preventDefault();
      this.say.command(command, null);
    }
  }

  _pan(across, up) {
    const dx = across * (this.shown.x1 - this.shown.x0);
    const dy = up * (this.shown.y1 - this.shown.y0);
    this._remember();
    this._look(
      this.shown.x0 + dx, this.shown.x1 + dx, this.shown.y0 + dy, this.shown.y1 + dy,
    );
  }

  grid() {
    this.ruled = !this.ruled;
    for (const axis of this.plot.axes) axis.grid.show = this.ruled;
    this.plot.redraw(false, true);
  }

  legend() {
    this.listed = !this.listed;
    this._relabel();
  }

  // Which way the polar reading stands, which is Python's to say: what a curve
  // is read as decides what is sampled for it, and this side draws the rings
  // and reads the pointer out in r and θ.
  polarized(polar) {
    this.polar = Boolean(polar);
    this.plot.redraw();
  }

  // One button of the tool row, on or off, by the name of the control it is.
  lit(name, on) {
    controls.lit(this.buttons, name, on);
  }

  // A parameter range field was left or Enter was pressed in one: hand both
  // bounds over as they were typed. What they read as is Python's, since `-π`
  // is an answer here and reading one is a parse and an evaluation.
  _ranged() {
    if (this.ranged === null) return;
    this.say.ranged(
      this.ranged,
      this.range.fields.get('low').value.trim(),
      this.range.fields.get('high').value.trim(),
    );
  }

  // -- the menus -----------------------------------------------------------

  // The canvas menu, which is about the view and about whatever the click was
  // over. Both halves are Python's answer to the snapshot below; what is left
  // here is where to put the popup.
  _menu(event) {
    const at = this._pointedPlot(event);
    const state = this._state(at === null ? null : at.serial);
    this._offer(event, this.say.menu(state));
  }

  // The menu one legend row offers, which is about that curve alone.
  _card(event, serial) {
    this._offer(event, this.say.card(this._state(serial)));
  }

  // Either menu, put up where the click was: what comes back is the name of a
  // control, which goes where every other one goes.
  _offer(event, entries) {
    controls.menu(this.canvas, event, entries, (name, value) =>
      this.say.command(name, value));
  }

  // This pane as the description of its controls has to read it: the little of
  // the view state a menu depends on, and the serial of whatever the click was
  // over. Small on purpose, and it crosses only when a menu opens - the drag
  // that moved the view never asked anybody anything.
  _state(pointed) {
    return {
      tracing: this.tracing !== null,
      grid: this.ruled,
      legend: this.listed,
      equal: this.equal,
      pointed,
    };
  }

  // Which curve the pointer is on, for the menu that is about one. The nearest
  // sample within a few pixels, which is the same hit test the desktop's stroke
  // does for itself.
  _pointedPlot(event) {
    const over = this.plot.over;
    const at = this._value(event.offsetX, event.offsetY);
    const across = (this.shown.x1 - this.shown.x0) / over.clientWidth;
    const up = (this.shown.y1 - this.shown.y0) / over.clientHeight;
    let best = null;
    for (const serial of this.order) {
      const plot = this.plots.get(serial);
      if (plot === undefined || plot.hidden || plot.xs === null) continue;
      for (let index = 0; index < plot.xs.length; index += 1) {
        const dx = (plot.xs[index] - at.x) / across;
        const dy = (plot.ys[index] - at.y) / up;
        const distance = Math.hypot(dx, dy);
        if (distance < 8 && (best === null || distance < best.distance)) {
          best = { plot, distance };
        }
      }
    }
    return best === null ? null : best.plot;
  }
}

// -- odds and ends ---------------------------------------------------------------

// A typed array is what a 1-D contiguous float32 crosses as, and anything else
// means the conversion took the slow road. Saying so is cheaper than wondering
// why a drag stutters.
function typed(array, label) {
  if (!ArrayBuffer.isView(array)) {
    console.warn(`${label}: samples arrived as ${array && array.constructor
      && array.constructor.name}, not a typed array`);
    return Float32Array.from(array || []);
  }
  return array;
}

function middle(range) {
  return range === undefined || range === null ? 0 : (range[0] + range[1]) / 2;
}

// What one wheel event zooms by: how far it scrolled, as a number of detents,
// and the factor raised to that. The distance arrives in pixels, in lines or in
// pages, and `deltaMode` is which of the three - 0, 1 and 2 in that order.
function wheeled(event) {
  const perUnit = [1, WHEEL_LINE_PX, WHEEL_PAGE_PX][event.deltaMode] || 1;
  const detents = (event.deltaY * perUnit) / WHEEL_DETENT_PX;
  return WHEEL_FACTOR ** clamp(detents, -WHEEL_MAX_DETENTS, WHEEL_MAX_DETENTS);
}

function clamp(value, low, high) {
  return Math.min(Math.max(value, low), high);
}

function padded([left, right, low, high]) {
  const wide = (right - left) || 1;
  const tall = (high - low) || 1;
  return [left - 0.1 * wide, right + 0.1 * wide, low - 0.1 * tall, high + 0.1 * tall];
}

// A round step at about a tenth of the reach, which is what the rings of a
// polar grid are spaced by.
function ruled(reach) {
  if (!(reach > 0)) return 0;
  const rough = reach / 5;
  const power = Math.pow(10, Math.floor(Math.log10(rough)));
  for (const step of [1, 2, 5, 10]) {
    if (step * power >= rough) return step * power;
  }
  return 10 * power;
}

function rgb(color) {
  const value = parseInt(color.slice(1), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

// One truth grid as the image a region is filled with: one pixel a cell, in the
// plot's own color at a quarter opacity, transparent where the inequality is
// false. Made once when the samples arrive rather than once a frame, since a
// drag redraws sixty times over a grid the worker only sent once.
function shading(message, color) {
  const [red, green, blue] = rgb(color);
  const image = new ImageData(message.nx, message.ny);
  const pixels = image.data;
  for (let at = 0; at < message.mask.length; at += 1) {
    const to = at * 4;
    pixels[to] = red;
    pixels[to + 1] = green;
    pixels[to + 2] = blue;
    pixels[to + 3] = message.mask[at] ? Math.round(REGION_ALPHA * 255) : 0;
  }
  const off = document.createElement('canvas');
  off.width = message.nx;
  off.height = message.ny;
  off.getContext('2d').putImageData(image, 0, 0);
  return off;
}

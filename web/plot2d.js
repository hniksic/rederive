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

import * as controls from './controls.js';

// The colors the canvas itself is drawn in, which are `plot/qt/window2d.py`'s.
// Near-black rather than black, so that a curve in a black-adjacent color still
// reads and so that the pane does not look like a hole in the page. Everything
// around the canvas - the bar, the legend card, the status line - is the
// chrome's business and is in the style sheet, as `theme.py` is on the desktop.
const BACKGROUND = '#0c0c10';
const AXIS_COLOR = '#909090';
const GRID_COLOR = 'rgba(144, 144, 144, 0.18)';
const MUTED = '#7d8595';

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

// How much of a hidden legend row is left standing. Dimmed rather than struck
// through: a hidden curve is a curve that is still in the pane.
const LEGEND_FADED = 0.4;

// Where a fresh pane is put and how big it is, and how far the next one is
// offset so that two panes are two panes and not one on top of another.
const PANE_WIDTH = 720;
const PANE_HEIGHT = 500;
const PANE_STEP = 28;

// The kinds, spelled as `plot/protocol.py` spells them. The page is told which
// kind a plot is and reads nothing else into it.
const CURVE = 'curve';
const FAMILY = 'family';
const PARAMETRIC = 'parametric';
const POLAR = 'polar';
const DATA = 'data';

// Which kinds a marker can ride, and which a feature scan works on. Both are
// `plot/model.py`'s answer: a region has no f(x) a marker could be at a point of.
const RIDEABLE = new Set([CURVE, FAMILY, PARAMETRIC, POLAR]);
const SCANNED = new Set([CURVE, FAMILY]);

// The one place a plot pane is put, laid over the terminal and letting the
// pointer through everywhere it has no pane.
let root = null;
let opened = 0;
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
export function wire(term) {
  terminal = term;
}

// One pane, opened where the plot session asked for one. What comes back is
// what the session's window handle calls into: everything here is a method
// Python names, and nothing else on this side is reachable from there.
export function open(number, debounce, commands, handlers) {
  const pane = new Pane(number, debounce, commands, handlers);
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
  constructor(number, debounce, commands, handlers) {
    this.number = number;
    // How long the view has to stand still before the curves are sampled for
    // it, which is `plot/resample.py`'s figure and not this file's: long enough
    // that a drag is one re-sample rather than sixty, short enough that letting
    // go of the mouse feels like the end of the gesture.
    this.debounce = debounce;
    // What this pane offers, as `plot/controls.py` describes it: the keys, the
    // buttons and the words are read off this and are spelled nowhere here.
    this.commands = commands;
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
    this.features = null;
    this.message = '';
    this.reading = '';
    this.announce = '';
    this.waiting = false;
    this.pending = null;
    this.featureBackwards = false;
    this._build();
    this._frame();
    this._home();
  }

  // -- the furniture ---------------------------------------------------------

  _build() {
    const pane = document.createElement('div');
    pane.className = 'plot-pane';
    pane.tabIndex = 0;
    const offset = (opened++ % 6) * PANE_STEP;
    pane.style.left = `${40 + offset}px`;
    pane.style.top = `${40 + offset}px`;
    pane.style.width = `${PANE_WIDTH}px`;
    pane.style.height = `${PANE_HEIGHT}px`;
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
    this.buttons = controls.bar(this.tools, this.commands, (name) => {
      this.say.command(name, null);
      this.element.focus();
    });
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
  _movable() {
    let from = null;
    this.bar.addEventListener('pointerdown', (event) => {
      if (event.button !== 0 || event.target.tagName === 'BUTTON') return;
      from = { x: event.clientX, y: event.clientY, left: this.element.offsetLeft,
               top: this.element.offsetTop };
      capture(this.bar, event);
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
  // be two programs.
  _home() {
    this.equal = true;
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
      region: null,
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

  dismiss() {
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
      plot.region = shading(message, plot.color);
      plot.extent = message.extent;
      plot.xs = null;
    } else {
      plot.xs = typed(message.xs, plot.label);
      plot.ys = typed(message.ys, plot.label);
      plot.bounds = message.bounds && message.bounds.length ? message.bounds : null;
      if (message.trange) plot.trange = message.trange;
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
    ctx.strokeStyle = AXIS_COLOR;
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
    ctx.strokeStyle = plot.color;
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
    ctx.fillStyle = plot.color;
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
    const off = plot.region;
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
    ctx.strokeStyle = plot.color;
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
        event.preventDefault();
        return;
      }
      if (event.button === 0) {
        panning = { x: event.offsetX, y: event.offsetY, view: { ...this.shown } };
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
      this._pointed(event.offsetX, event.offsetY, over);
    });
    const release = (event) => {
      down.delete(event.pointerId);
      if (down.size < 2) pinch = null;
      if (banding !== null) {
        this._band(null);
        if (banding.moved > CLICK_SLOP_PX) {
          this._zoomTo(banding, event.offsetX, event.offsetY, over);
        } else {
          this._menu(event);
        }
      }
      panning = null;
      banding = null;
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
      // Shift holds the width and Ctrl the height, which is how one axis is
      // stretched against the other.
      this._zoom(
        event.deltaY > 0 ? 1.25 : 0.8,
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

  // The pointer readout: where the pointer is, which is the one number on this
  // side that is not a value of anything and so is the one this file may write.
  _pointed(x, y, over) {
    if (this.tracing !== null) return;
    const at = {
      x: this.shown.x0 + (x / over.clientWidth) * (this.shown.x1 - this.shown.x0),
      y: this.shown.y1 - (y / over.clientHeight) * (this.shown.y1 - this.shown.y0),
    };
    if (this.polar) {
      this.said(
        `r: ${Math.hypot(at.x, at.y).toFixed(6)}` +
        `   θ: ${Math.atan2(at.y, at.x).toFixed(6)}`,
      );
      return;
    }
    this.said(`x: ${at.x.toFixed(6)}   y: ${at.y.toFixed(6)}`);
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
    this.equal = false;
  }

  // A view change asks for a sampling, once the view has stopped changing. The
  // keyed job queue on the Python side collapses whatever still gets through.
  moved() {
    clearTimeout(this.timer);
    this.timer = setTimeout(() => this.say.changed(), this.debounce);
    if (this.tracing !== null) this._retrace();
  }

  home() {
    this._home();
  }

  // The keyed zoom, which is about the middle of the picture: the wheel and the
  // pinch are about the point they happened at, and a key happened nowhere.
  zoom(factor) {
    const over = this.plot.over;
    this._zoom(factor, over.clientWidth / 2, over.clientHeight / 2, over);
  }

  // Every plot's rectangle, unioned. The rectangles come off the worker with the
  // samples, so nothing here scans an array to find one.
  autoscale() {
    let box = null;
    for (const plot of this.plots.values()) {
      if (plot.hidden || plot.bounds === null) continue;
      box = box === null ? [...plot.bounds] : [
        Math.min(box[0], plot.bounds[0]), Math.max(box[1], plot.bounds[1]),
        Math.min(box[2], plot.bounds[2]), Math.max(box[3], plot.bounds[3]),
      ];
    }
    if (box === null) {
      this.said('Nothing to frame yet');
      return;
    }
    this._unlock();
    this._look(...padded(box));
    this.said('Framed on everything drawn');
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
    this.traceAt = plot.kind === PARAMETRIC || plot.kind === POLAR
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
    if (plot.kind === PARAMETRIC || plot.kind === POLAR) {
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
    const at = {
      x: this.shown.x0 + (event.offsetX / over.clientWidth) * (this.shown.x1 - this.shown.x0),
      y: this.shown.y1 - (event.offsetY / over.clientHeight) * (this.shown.y1 - this.shown.y0),
    };
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

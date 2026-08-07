// The 3D plot panes: three.js for the card, and a surface that was built in Python.
//
// Almost nothing here is about a surface. What a solid is - the box it stands
// in, its triangles, where its mesh stops at a hole, how bright each vertex is -
// is `plot/surface.py`'s, and it is the same file the desktop's window draws
// from; what arrives on this side is three flat arrays of numbers and a fourth
// of indices. So this file is a scene, a camera, two materials and the rule
// for putting new numbers into buffers that are already on the card.
//
// That is also why turning a surface costs nothing. The mesh is built once per
// domain and grid; every orbit, every wheel click and every preset is a camera
// move over vertices that have not changed, and nothing the mouse can do starts
// an evaluation. The two fields in the tool row are the only things that do,
// because they are the two that change what was computed rather than where it
// is looked at from.
//
// Nothing here renders an expression and nothing here evaluates one. A
// surface's name is the line Python wrote with the syntax writer, and the
// numbers behind the picture - the box's center, its lengths, the z it was
// clipped to, every number along its edges - arrive spelled by the same
// function the desktop's inspector spells them with. What this file writes for
// itself is where the camera is, which is a fact about the page and not a value
// of anything.
//
// The writing on the box is DOM and not geometry. A tick number is a `<span>`
// put where the same projection the card draws with says the point is, which
// costs one vector each per camera move and comes out at the screen's own
// resolution - where a font baked onto the card would cost a texture, a shader
// and a picture that softens as it is turned. What Python chose and spelled is
// where the ticks fall and what they read; this side projects a point it was
// handed and writes nothing.
//
// Nor does it own what a pane offers. The commands, their words, their keys and
// the order a menu stands them in are `plot/controls.py`'s, the same table the
// desktop's window renders, and where the camera goes for each of them is
// `plot/actions.py`'s; a pane is handed the ones this backend serves when it is
// made and the menu as it should currently read when a right click asks for
// one, and it answers with the name of a control. The domain it is evaluated
// over and the box and camera of its inspector are `plot/forms.py`'s in the
// same way, drawn by `forms.js`, and what is typed into either goes back as
// text - so `-π` is an answer in a page as it is on a desktop.

import * as THREE from 'three';
import { OrbitControls } from './three/OrbitControls.js';
import * as controls from './controls.js';
import * as sheets from './forms.js';
import * as place from './place.js';

// The canvas colors, which are `plot/qt/window3d.py`'s: the same near-black the
// 2D pane draws on, and a box drawn in a gray that reads against it without
// competing with the surface standing in it.
const BACKGROUND = '#0c0c10';
const BOX_COLOR = 0x969696;
const BOX_ALPHA = 0.43;
const TICK_COLOR = 0x969696;
const TICK_ALPHA = 0.78;

// The light the card adds to the light Python baked in. Half of a vertex's
// color is already how the surface lies to a fixed lamp - that is what makes a
// fold in the middle of a surface show - so what is asked of three.js is the
// rest: a broad ambient that keeps the baked shading readable, and one
// directional lamp from `surface.py`'s own direction, so that turning the
// picture moves a highlight across it the way turning a solid does.
//
// The numbers are near π rather than near one because three.js lights are in
// physical units and a diffuse surface gives back a π-th of what falls on it.
// What they are chosen to add up to is the desktop's own brightness - the
// vertex color itself, which is what pyqtgraph draws with no lighting at all -
// so the two programs draw one surface at one weight, with the lamp for relief.
const AMBIENT = 2.4;
const DIRECTIONAL = 0.95;
const LAMP = [0.4, -0.6, 0.69];

// The field of view is measured across the picture, as pyqtgraph measures it
// and as three.js does not - three.js takes the angle up it - so the two are
// converted below. A pane and a window of the same shape would otherwise frame
// the same box at two different sizes. Where the camera stands is not here at
// all: `plot/actions.py` says where a pane opens and where each preset looks
// from, and a pane is told.
const FIELD_OF_VIEW = 60;
const NEAR = 0.5;
const FAR = 500;

// How far one arrow key turns the camera, and how fast a rotation turns it by
// itself. A rotation is for reading a shape from every side while the eye stays
// still, so it is slow: a turn takes half a minute.
const ORBIT_DEGREES = 5;
const SPIN_DEGREES = 0.4;

// How far the wire's occluder is pushed away from the camera. The lines lie on
// the very faces they are being depth-tested against, so a shove backwards is
// what makes the wire a wire rather than a stitch - and the shove is far too
// slight to let a line on the far side of the shape through. The desktop's
// numbers, in the same (factor, units) the depth buffer reads them as.
const WIRE_OFFSET = [1, 2];
const WIRE_WIDTH = 2;

// How big a fresh pane is. Where it is put is `place.js`'s, which both kinds of
// pane share so that they cascade off one another.
const PANE_WIDTH = 760;
const PANE_HEIGHT = 620;

// How far a tick mark and its number stand out of the box, and how far out the
// name of an axis does, in the world units the box is measured in. The
// desktop's numbers, so that the two pictures are furnished alike.
const TICK_OUT = 0.35;
const LABEL_OUT = 1.05;
const NAME_OUT = 2.3;

// How nearly an axis has to point at the camera before its numbers are dropped,
// as the cosine of the angle between them: about five degrees. Facing the xy
// plane makes the whole z axis one point of the screen, and five numbers
// stacked on that point are five numbers about nothing.
const EDGE_ON = 0.996;

// The three axes, in the order the box reports its ticks and its names in.
const AXES = ['x', 'y', 'z'];

// How the plot list is written into an exported picture: the size of the words
// and how far in from the corner they stand, in logical pixels. The card is a
// picture of the scene alone, and the legend is a floating element over it, so
// an export that did not write the names would be a picture that had lost half
// of what it showed.
const LEGEND_PX = 12;
const LEGEND_MARGIN_PX = 10;

// How much of a hidden legend row is left standing. Dimmed rather than struck
// through: a hidden surface is a surface that is still in the pane.
const LEGEND_FADED = 0.4;

// Why a picture cannot leave a pane that has no card in it. The refusal a
// browser with no usable WebGL earns, said where the pane is looked at rather
// than swallowed: a copy that did nothing and said nothing would look exactly
// like a key bound to nothing.
const NO_CARD = 'this pane has no 3D drawing to take a picture of';

// How far the pointer may move between a right-button press and its release and
// still count as a click that opens the menu rather than a pan of the camera.
const CLICK_SLOP_PX = 4;

// The one place a plot pane is put, laid over the terminal and letting the
// pointer through everywhere it has no pane. The 2D panes are in the same
// layer, so a solid and a curve stack the way two windows do.
let root = null;
let landed = () => {};
let terminal = null;

// Every open 3D pane by its number, which is the number the plot session names
// it by at both ends. Exported because `main.js` hangs this module off
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
// The copy event is listened for here for the reason `plot2d.js` listens for
// its own. Ctrl+C reaches a pane as a `copy` and not as a key press: a page
// that cancels the keydown cancels the copy the browser was about to offer it,
// so the key ladder leaves the stroke alone and this is where it arrives. The
// listener is the document's because a copy event is raised where the selection
// is rather than on whatever holds the keyboard, and it goes to the pane that
// has focus - which is what keeps it away from the terminal and from a flat
// pane alike.
export function wire(term) {
  terminal = term;
  document.addEventListener('copy', (event) => {
    const pane = holding();
    if (pane !== null) pane._copied(event);
  });
}

// The 3D pane the keyboard is in, or null while it is anywhere else.
function holding() {
  for (const pane of panes.values()) {
    if (pane.element.contains(document.activeElement)) return pane;
  }
  return null;
}

// One pane, opened where the plot session asked for one. What comes back is
// what the session's window handle calls into: everything here is a method
// Python names, and nothing else on this side is reachable from there.
export function open(number, commands, strip, form, handlers) {
  const pane = new Solid(number, commands, strip, form, handlers);
  panes.set(number, pane);
  return pane;
}

// The one thing the page says that is about no pane in particular. The executor
// on the Python side is what it reaches, and hearing it is what lets the next
// evaluation go out - the queue is one queue for curves and surfaces alike.
export function attend(handlers) {
  landed = handlers.landed;
}

// Every pane at once, for a program that is ending.
export function stop() {
  for (const pane of [...panes.values()]) pane.dismiss();
  panes.clear();
}

// One message off the engine worker, and whether it was a surface's.
//
// Answering that is the whole reason this returns anything: a sampling answer
// is acknowledged exactly once, and both drawing modules are listening to the
// same worker. So a message this file takes is a message `plot2d.js` never
// sees, and the acknowledgement below is the only one it gets.
export function heard(message) {
  if (message === null || typeof message !== 'object' || message.reply !== 'grid') {
    return false;
  }
  const pane = panes.get(message.pane);
  if (pane !== undefined) pane.answered(message);
  // The acknowledgement goes out whether or not there was anything to draw:
  // it is what lets go of the request in flight, and a pane closed mid-evaluation
  // must not stop the queue.
  if (message.number !== undefined) landed(message.number, message.trouble || '');
  return true;
}

// -- one pane -------------------------------------------------------------------

class Solid {
  constructor(number, commands, strip, form, handlers) {
    this.number = number;
    // What this pane offers, as `plot/controls.py` describes it: the keys, the
    // buttons and the words are read off this and are spelled nowhere here.
    this.commands = commands;
    // The domain and the grid as a strip of fields, and the box and the camera
    // as a form: both are `plot/forms.py`'s, and neither is laid out here.
    this.description = strip;
    this.viewForm = form;
    this.say = handlers;
    this.plots = new Map();
    this.order = [];
    this.standing = null;
    this.listed = true;
    this.boxed = true;
    // Whether the numbers along the box edges and the names of the axes are
    // drawn, which is the one furnishing with nothing else to say its state.
    // Spelled apart from `named` below, which is what Python calls to say what
    // the three axes are: a field and a method of one name are one name, and
    // the field would answer the call.
    this.lettered = true;
    // What the three axes are called. Python says, taking them from the first
    // surface's own expression.
    this.axes = [...AXES];
    // The inspector now up over the pane, so that a second `View...` closes it
    // rather than standing a second one on top of it.
    this.sheet = null;
    this.message = '';
    this.spinning = null;
    this._build();
    this._scene();
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
      '<div class="plot-canvas"><div class="plot-legend"></div>' +
      '<div class="plot-marks"></div></div>' +
      '<div class="plot-status"></div>';
    this.element = pane;
    this.bar = pane.querySelector('.plot-bar');
    this.titleText = pane.querySelector('.plot-title');
    this.tools = pane.querySelector('.plot-tools');
    this.canvas = pane.querySelector('.plot-canvas');
    this.card = pane.querySelector('.plot-legend');
    // Where the writing on the box goes: a layer over the card that the pointer
    // passes straight through, holding one element per number and per name.
    this.writing = pane.querySelector('.plot-marks');
    this.labels = [];
    this.status = pane.querySelector('.plot-status');
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
    this._toolbar();
    this._movable();
    this._menued();
    this._watch();
    pane.focus();
  }

  // A right click on the picture opens the menu, unless it was a drag: the
  // right button also pans the camera, and a menu at the end of a pan would be
  // a menu nobody asked for.
  _menued() {
    let from = null;
    this.canvas.addEventListener('contextmenu', (event) => event.preventDefault());
    this.canvas.addEventListener('pointerdown', (event) => {
      from = event.button === 2 ? { x: event.clientX, y: event.clientY } : null;
    });
    this.canvas.addEventListener('pointerup', (event) => {
      if (from === null || event.button !== 2) return;
      const moved =
        Math.abs(event.clientX - from.x) + Math.abs(event.clientY - from.y);
      from = null;
      if (moved <= CLICK_SLOP_PX) this._menu(event);
    });
  }

  // The domain, the grid, and the buttons the description gives a word to.
  // Everything else about a 3D pane is the camera, and the camera is the mouse.
  //
  // The strip is `plot/forms.py`'s, down to the word standing in front of each
  // field and the hairline between one answer and the next: a domain in x, a
  // domain in y and a grid are three answers rather than six numbers in a row,
  // and the desktop's toolbar is laid out from the same description.
  _toolbar() {
    this.strip = sheets.strip(this.tools, this.description, () => this._framed());
    this.fields = this.description.fields.map((one) => one.name);
    // The keyboard is taken back from the button before the command runs rather
    // than after it, so that `view...` keeps the focus it gives to the first
    // field of the inspector it raises.
    this.buttons = controls.bar(this.tools, this.commands, (name) => {
      this.element.focus();
      this.say.command(name, null);
    });
  }

  // Dragging by the title bar, which is what a pane has instead of a window
  // manager. Nothing here is a resize: the pane's own corner does that, and the
  // observer below hears about it.
  //
  // The pointer is captured by the bar, so the drag follows a finger or a mouse
  // that has left it - which every drag does, a bar being twenty pixels tall.
  // The camera needs nothing of the sort: OrbitControls reads pointers itself
  // and has read a finger since long before this pane existed.
  //
  // The keyboard is handed over by hand. Cancelling the press is what stops the
  // drag from selecting the title as text, and it also cancels the focus the
  // press would have given the pane - so a pane taken hold of by its bar would
  // answer to none of its keys.
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
    if (this.renderer === undefined) return;
    const width = Math.max(this.canvas.clientWidth, 120);
    const height = Math.max(this.canvas.clientHeight, 120);
    this.renderer.setSize(width, height, false);
    this._lens(width, height);
    this._anchor();
    this._paint();
  }

  // The angle up the picture that shows what `FIELD_OF_VIEW` degrees across it
  // would, which is what the desktop's view is set up with.
  _lens(width, height) {
    const across = (FIELD_OF_VIEW * Math.PI) / 360;
    this.camera.fov = (360 * Math.atan((Math.tan(across) * height) / width)) / Math.PI;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  // The scene, and a pane that says so where there is no card to draw on.
  //
  // A page whose WebGL context cannot be had is the browser's answer to the
  // desktop's machine with no usable OpenGL, and it gets the same treatment:
  // no picture, but a pane, a plot list, a title and a line naming what
  // happened. Never a blank rectangle and no explanation.
  _scene() {
    try {
      // The drawing buffer is kept because a picture leaving the pane is read
      // off it, and a card that cleared after every frame would hand back a
      // blank rectangle to whoever asked for one a moment later.
      this.renderer = new THREE.WebGLRenderer({
        antialias: true,
        preserveDrawingBuffer: true,
      });
    } catch (error) {
      this.said(`3D drawing is not available: ${error && error.message}`);
      return;
    }
    this.renderer.setPixelRatio(devicePixelRatio);
    this.renderer.setSize(
      Math.max(this.canvas.clientWidth, 120),
      Math.max(this.canvas.clientHeight, 120),
      false,
    );
    // The shading is Python's and is already in the numbers: a vertex color is
    // the surface's own color scaled by how bright that vertex is, worked out
    // where the arrays were. Encoding it a second time on the way to the screen
    // would make the browser's picture a lighter one than the desktop's, so the
    // card writes what it was given.
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    this.renderer.setClearColor(new THREE.Color(BACKGROUND), 1);
    this.renderer.domElement.className = 'plot-card';
    this.canvas.appendChild(this.renderer.domElement);
    this.world = new THREE.Scene();
    this.world.add(new THREE.AmbientLight(0xffffff, AMBIENT));
    const lamp = new THREE.DirectionalLight(0xffffff, DIRECTIONAL);
    lamp.position.set(LAMP[0], LAMP[1], LAMP[2]).multiplyScalar(100);
    this.world.add(lamp);
    this.camera = new THREE.PerspectiveCamera(FIELD_OF_VIEW, 1, NEAR, FAR);
    this._lens(
      Math.max(this.canvas.clientWidth, 120),
      Math.max(this.canvas.clientHeight, 120),
    );
    // z is up here, as it is in the arrays Python placed: the floor of the box
    // is the domain and the height is the value. three.js opens on y up, and a
    // camera left that way would orbit around the wrong axis.
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = false;
    this.controls.addEventListener('change', () => {
      this._anchor();
      this._followed();
      this._paint();
    });
    this.frame = this._frame();
    this.world.add(this.frame);
    this.marks = this._marks();
    this.world.add(this.marks);
  }

  // The box the picture stands in: twelve edges, and the numbers along three of
  // them. A surface floating in the dark is a shape nobody can tell the size
  // of, and a box with no writing on it is a shape nobody can tell the scale
  // of - so both are drawn, the edges as geometry and the writing as DOM.
  _frame() {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array(24 * 3), 3),
    );
    const material = new THREE.LineBasicMaterial({
      color: BOX_COLOR,
      transparent: true,
      opacity: BOX_ALPHA,
    });
    const lines = new THREE.LineSegments(geometry, material);
    lines.visible = false;
    return lines;
  }

  // The dashes standing out of the three edges the numbers are written along.
  // Rebuilt on every camera move, which is a few dozen coordinates and no
  // geometry at all: the edges that carry them change as the picture turns.
  _marks() {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
    const lines = new THREE.LineSegments(
      geometry,
      new THREE.LineBasicMaterial({
        color: TICK_COLOR,
        transparent: true,
        opacity: TICK_ALPHA,
      }),
    );
    lines.visible = false;
    return lines;
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
    const plot = new Standing(serial, spec);
    this.plots.set(serial, plot);
    if (!this.order.includes(serial)) this.order.push(serial);
    if (this.world !== undefined) plot.enter(this.world);
    this._relabel();
    this._paint();
  }

  respec(serial, spec) {
    const plot = this.plots.get(serial);
    if (plot === undefined) return;
    plot.respec(spec);
    this._relabel();
    this._paint();
  }

  // One button of the tool row, on or off, by the name of the control it is.
  // Which way the `mesh` box stands is Python's to say: the look is sticky, and
  // what a fresh pane opens on is what the last one was left in.
  lit(name, on) {
    controls.lit(this.buttons, name, on);
  }

  remove(serial) {
    const plot = this.plots.get(serial);
    if (plot === undefined) return;
    plot.leave();
    this.plots.delete(serial);
    this.order = this.order.filter((one) => one !== serial);
    this._relabel();
    this._paint();
  }

  // An evaluation has been asked for. What is worth remembering is which one,
  // so that an answer about a domain that has moved on can be dropped.
  starting(serial, generation) {
    const plot = this.plots.get(serial);
    if (plot !== undefined) plot.generation = generation;
  }

  present() {
    this.element.style.display = '';
    this.raise();
    this.element.focus();
    // A pane made while the page was busy may never have been measured.
    this._resized();
  }

  retitle(title, current) {
    this.titleText.textContent = title;
    this.element.classList.toggle('current', Boolean(current));
  }

  // The six numbers of the tool row, as Python spells them and in the order the
  // description stands its fields in. It owns the domain and the grid; these
  // fields show what it has and hand back what was typed.
  domain(...values) {
    this.fields.forEach((name, at) => {
      const edit = this.strip.fields.get(name);
      if (edit !== undefined) edit.value = values[at];
    });
  }

  // What the three axes are called, which is the first surface's own reading of
  // its expression. Python takes them off the plot list; nothing here invents a
  // letter, and an empty pane is told the three every reader supplies anyway.
  named(across, along, up) {
    this.axes = [across, along, up];
    this._anchor();
  }

  // The inspector, opened over the pane or taken away again. The six box
  // numbers arrive spelled by Python, which owns the domain and the z; the
  // three camera numbers are the page's, being where a reader is standing
  // rather than a value of anything. Whoever owns the number writes it.
  inspect(box) {
    if (this.sheet !== null) {
      this.sheet.close();
      return;
    }
    const at = this.where();
    this.sheet = sheets.ask(
      this.element,
      this.viewForm,
      [...box, rounded(at.azimuth), rounded(at.elevation), rounded(at.distance)],
      (said, role) => this.say.typed(this.viewForm.name, said, role),
      () => {
        this.sheet = null;
        this.lit('view.inspect', false);
        this.element.focus();
      },
    );
    this.lit('view.inspect', true);
  }

  // The box fields filled again after an answer moved the picture, wherever the
  // inspector is still up. The camera half follows the mouse instead, in
  // `_followed` below, which is what the form's own subtitle promises.
  showing(box) {
    if (this.sheet === null) return;
    const written = {};
    this.viewForm.fields.forEach((one, at) => {
      if (at < box.length) written[one.name] = box[at];
    });
    this.sheet.fill(written);
  }

  said(text) {
    this.message = text || '';
    this.status.textContent = this.message;
  }

  dismiss() {
    if (this.sheet !== null) this.sheet.close();
    // Where the keys go next. A pane that had them - the close button it was
    // shut with is inside it - would otherwise leave them on nothing, and the
    // program would look as though it had stopped listening.
    const held = this.element.contains(document.activeElement);
    if (this.observer) this.observer.disconnect();
    if (this.spinning !== null) cancelAnimationFrame(this.spinning);
    for (const plot of this.plots.values()) plot.leave();
    if (this.controls) this.controls.dispose();
    if (this.renderer) this.renderer.dispose();
    this.element.remove();
    panes.delete(this.number);
    if (held) focus();
  }

  // -- what the worker says ---------------------------------------------------

  answered(message) {
    const plot = this.plots.get(message.plot);
    if (plot === undefined || message.generation !== plot.generation) return;
    if (message.trouble) {
      this.said(`${plot.label}: ${message.trouble}`);
      return;
    }
    plot.upload(message);
    if (message.world) this._stand(message);
    this.said(message.words || '');
    this._relabel();
    this._paint();
    // What the surface's values measure in z, and the box it was drawn in. The
    // pane on the Python side is what pools the first into the second, since
    // one box for the window is a rule about a picture and not about a surface.
    this.say.stood(message.plot, message.span, message.zrange);
  }

  // The box every surface in the pane stands in, as the answer just described
  // it. Nothing here works it out: the floor, the height and the numbers behind
  // them are Python's, and what is done with them is twelve lines and a scale.
  _stand(message) {
    this.standing = message;
    if (this.frame === undefined) return;
    const half = message.world / 2;
    const up = message.height / 2;
    const corners = [
      [-half, -half, -up], [half, -half, -up], [half, half, -up], [-half, half, -up],
      [-half, -half, up], [half, -half, up], [half, half, up], [-half, half, up],
    ];
    const edges = [
      [0, 1], [1, 2], [2, 3], [3, 0],
      [4, 5], [5, 6], [6, 7], [7, 4],
      [0, 4], [1, 5], [2, 6], [3, 7],
    ];
    const points = this.frame.geometry.attributes.position;
    edges.forEach(([from, to], at) => {
      points.array.set(corners[from], at * 6);
      points.array.set(corners[to], at * 6 + 3);
    });
    points.needsUpdate = true;
    this.frame.geometry.computeBoundingSphere();
    this.frame.visible = this.boxed;
    this._anchor();
  }

  // -- drawing ----------------------------------------------------------------

  _paint() {
    if (this.renderer === undefined) return;
    this.renderer.render(this.world, this.camera);
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
      row.addEventListener('pointerdown', (event) => {
        event.stopPropagation();
        if (event.button === 0) this.say.hide(serial, !plot.hidden);
      });
      // The right button on a row opens the menu about that surface, which is
      // the same one the desktop's legend offers on the same click: hiding is
      // one button and what is done to the surface itself is the other.
      row.addEventListener('contextmenu', (event) => {
        event.preventDefault();
        event.stopPropagation();
        this._card(event, serial);
      });
      this.card.appendChild(row);
    }
  }

  // -- the writing on the box ---------------------------------------------------

  // Put the tick marks and the numbers on the box edges nearest the camera.
  //
  // Three edges carry them: the two bottom edges of the side the camera is on,
  // which is where a number sits in front of the picture rather than behind it,
  // and for the upright axis the far vertical edge, which is the one the
  // surface does not stand in front of. Which edges those are changes as the
  // view turns, so this runs on every camera move; it is a few dozen positions
  // and no geometry at all.
  //
  // An axis pointing at the camera is left unnumbered. Facing the xy plane
  // makes the whole z axis one point of the screen, and five numbers stacked on
  // that point are five numbers about nothing; the axis keeps its name, which
  // is all there is to say about it from there.
  //
  // Where each number falls and what it reads is Python's, in the same function
  // and off the same ruler the desktop's window uses. What is worked out here
  // is which edge it stands against and where that lands on the screen.
  _anchor() {
    const box = this.standing;
    if (this.camera === undefined || box === null) {
      this._write([], []);
      return;
    }
    const half = box.world / 2;
    const floor = -box.height / 2;
    const at = this.camera.position;
    const nearY = at.y < 0 ? -half : half;
    const nearX = at.x < 0 ? -half : half;
    const farX = -nearX;
    const farY = -nearY;
    const outY = Math.sign(nearY);
    const outX = Math.sign(nearX);
    // The upright edge is a corner, so its numbers stand out along both floor
    // directions at once rather than along either.
    const up = [Math.sign(farX) * 0.7, Math.sign(farY) * 0.7];
    const headOn = this._headOn();
    const ticks = box.ticks || {};
    const segments = [];
    const written = [];
    const ruler = (axis, foot, out) => {
      if (headOn[AXES.indexOf(axis)]) return;
      for (const tick of ticks[axis] || []) {
        const from = foot(tick.at);
        segments.push(from, stepped(from, out, TICK_OUT));
        written.push({ where: stepped(from, out, LABEL_OUT), text: tick.text });
      }
    };
    ruler('x', (value) => [value, nearY, floor], [0, outY, 0]);
    ruler('y', (value) => [nearX, value, floor], [outX, 0, 0]);
    ruler('z', (value) => [farX, farY, value], [up[0], up[1], 0]);
    const names = [
      [0, nearY + outY * NAME_OUT, floor],
      [nearX + outX * NAME_OUT, 0, floor],
      [farX + up[0] * NAME_OUT, farY + up[1] * NAME_OUT, -floor * 0.9],
    ];
    this.axes.forEach((name, axis) => {
      if (name) written.push({ where: names[axis], text: name, name: true });
    });
    this._write(segments, written);
  }

  // Which axes point so nearly at the camera that they have no length on the
  // screen. The camera looks along the line from itself to the box's center, so
  // an axis is edge-on exactly when it is parallel to that line - which is what
  // each of the three presets makes one of them.
  _headOn() {
    const at = this.camera.position;
    const length = Math.hypot(at.x, at.y, at.z);
    if (!length) return [false, false, false];
    return [at.x, at.y, at.z].map((value) => Math.abs(value) / length > EDGE_ON);
  }

  // The dashes as geometry and the words as DOM, both from what `_anchor` just
  // worked out.
  //
  // What a frame costs is one projected point and two style writes per number,
  // which for a box of five ticks an axis and three names is under twenty of
  // each. The elements are pooled and their text is left alone where it has not
  // changed: a rotation runs this sixty times a second, and a layer emptied and
  // refilled that often is a layer the browser lays out that often.
  _write(segments, written) {
    if (this.marks !== undefined) {
      const flat = new Float32Array(segments.length * 3);
      segments.forEach((point, at) => flat.set(point, at * 3));
      this.marks.geometry.setAttribute('position', new THREE.BufferAttribute(flat, 3));
      this.marks.geometry.computeBoundingSphere();
      // The dashes belong to the box and the words to the naming, which is how
      // the desktop's two toggles divide them: a box with no numbers beside it
      // still says where its divisions fall.
      this.marks.visible = this.boxed && segments.length > 0;
    }
    const shown = this.lettered ? written : [];
    while (this.labels.length < shown.length) {
      const label = document.createElement('span');
      label.className = 'plot-mark';
      this.writing.appendChild(label);
      this.labels.push(label);
    }
    this.labels.forEach((label, at) => {
      if (at >= shown.length) {
        label.style.display = 'none';
        return;
      }
      const one = shown[at];
      const spot = this._project(one.where);
      label.style.display = '';
      label.classList.toggle('name', Boolean(one.name));
      if (label.textContent !== one.text) label.textContent = one.text;
      label.style.left = `${spot.x}px`;
      label.style.top = `${spot.y}px`;
    });
  }

  // One point of the box on the screen, through the very projection the card
  // draws the box with - which is what makes a number sit against the edge it
  // is about however the picture is turned.
  _project(where) {
    const point = new THREE.Vector3(where[0], where[1], where[2]);
    point.project(this.camera);
    const wide = this.canvas.clientWidth;
    const tall = this.canvas.clientHeight;
    return {
      x: (point.x * 0.5 + 0.5) * wide,
      y: (-point.y * 0.5 + 0.5) * tall,
    };
  }

  // -- the camera --------------------------------------------------------------

  // Where the camera stands, in the three numbers a person turns a picture by.
  // The page's own arithmetic, because a camera is a fact about the page: no
  // part of it is a value of anything that was evaluated.
  where() {
    const at = this.camera.position;
    const distance = Math.hypot(at.x, at.y, at.z);
    return {
      azimuth: (Math.atan2(at.y, at.x) * 180) / Math.PI,
      elevation: distance ? (Math.asin(at.z / distance) * 180) / Math.PI : 0,
      distance,
    };
  }

  // The camera put where those three numbers say, which is where every preset,
  // every arrow key and a fresh pane leaves it. A distance of zero is no
  // distance at all: the presets turn the camera and leave it standing where it
  // was, so how far out the reader is looking from is not theirs to say.
  look(elevation, azimuth, distance) {
    if (this.camera === undefined) return;
    const far = distance || this.where().distance;
    const up = (elevation * Math.PI) / 180;
    const around = (azimuth * Math.PI) / 180;
    this.camera.position.set(
      far * Math.cos(up) * Math.cos(around),
      far * Math.cos(up) * Math.sin(around),
      far * Math.sin(up),
    );
    this.controls.target.set(0, 0, 0);
    this.controls.update();
    this._anchor();
    this._followed();
    this._paint();
  }

  orbit(across, up) {
    const at = this.where();
    this.look(
      Math.max(Math.min(at.elevation + up, 89.5), -89.5),
      at.azimuth + across,
      at.distance,
    );
  }

  // Turn the picture slowly, or stop turning it. Slow on purpose - a rotation
  // is for reading a shape from every side while the eye stays still. The
  // sentence is Python's, and names the key that stops it in the page's own
  // spelling.
  spin(rotating) {
    if (this.spinning !== null) {
      cancelAnimationFrame(this.spinning);
      this.spinning = null;
      this.said('');
      return;
    }
    const turn = () => {
      this.orbit(SPIN_DEGREES, 0);
      this.spinning = requestAnimationFrame(turn);
    };
    this.said(rotating);
    this.spinning = requestAnimationFrame(turn);
  }

  // -- the fields, the toggles and the keyboard ---------------------------------

  _framed() {
    this.say.framed(
      ...this.fields.map((name) => {
        const edit = this.strip.fields.get(name);
        return edit === undefined ? '' : edit.value.trim();
      }),
    );
  }

  // The camera half of the inspector, kept up to date while it is open, which
  // is what the form's own subtitle promises: it is a readout as well as a
  // field. The three numbers are the page's own, so the page spells them.
  _followed() {
    if (this.sheet === null) return;
    const at = this.where();
    this.sheet.fill({
      azimuth: String(rounded(at.azimuth)),
      elevation: String(rounded(at.elevation)),
      distance: String(rounded(at.distance)),
    });
  }

  legend() {
    this.listed = !this.listed;
    this._relabel();
  }

  // The box the picture stands in, on or off. It is drawn from the numbers the
  // last answer carried, so a pane with nothing in it has nothing to show.
  box() {
    this.boxed = !this.boxed;
    if (this.frame !== undefined) {
      this.frame.visible = this.boxed && this.standing !== null;
    }
    this._anchor();
    this._paint();
  }

  // The numbers along the box edges and the names of the axes, or neither.
  naming() {
    this.lettered = !this.lettered;
    this._anchor();
    this._paint();
  }

  // -- the picture, off the pane ------------------------------------------------

  // The command's copy, which is what the menu entry reaches. The key reaches
  // `_copied` below instead, and both end on the same road: a 3D pane has no
  // reading to carry, so what leaves it is always the picture.
  copy() {
    this._copyImage();
  }

  // Ctrl+C, arriving as the event it has to arrive as. The default is cancelled
  // so that a pane with nothing selected does not copy an empty string over
  // whatever was on the clipboard.
  _copied(event) {
    if (controls.evented(this.commands, 'copy') === null) return;
    event.preventDefault();
    this._copyImage();
  }

  _copyImage() {
    const shot = this._photograph();
    const clipboard = navigator.clipboard;
    if (shot === null) {
      this.say.copied('', NO_CARD);
      return;
    }
    if (typeof ClipboardItem === 'undefined' || clipboard === undefined
        || clipboard.write === undefined) {
      this.say.copied('', 'ClipboardItem is unavailable');
      return;
    }
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
  }

  // Export, which in a tab is a download: an object URL and a link clicked from
  // here, since that is the only way a file leaves a page.
  //
  // A PNG, which is also what the desktop's 3D window exports: there is no
  // painter path behind a card, so a picture of a solid is pixels on either
  // side of the program. The sentence Python says names the size.
  export() {
    const shot = this._photograph();
    const name = `plot${this.number}.png`;
    if (shot === null) {
      this.say.exported(name, 0, 0, NO_CARD);
      return;
    }
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
  }

  // The pane as one image: the card, and the names of what is standing on it.
  //
  // The scene is painted first, because a drawing buffer is only guaranteed to
  // hold what was last rendered into it. The legend and the writing on the box
  // are DOM over the card rather than anything drawn into it, so the plot list
  // is written on afterwards, exactly as the desktop's export writes it. A pane
  // with no card at all answers with nothing, and the caller says so rather
  // than handing back a blank rectangle.
  _photograph() {
    if (this.renderer === undefined) return null;
    this._paint();
    const source = this.renderer.domElement;
    const shot = document.createElement('canvas');
    shot.width = source.width;
    shot.height = source.height;
    const ctx = shot.getContext('2d');
    ctx.fillStyle = BACKGROUND;
    ctx.fillRect(0, 0, shot.width, shot.height);
    ctx.drawImage(source, 0, 0);
    this._namePlots(ctx, shot.width, shot.width / Math.max(source.clientWidth, 1));
    return shot;
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
      ctx.fillStyle = plot.color;
      ctx.fillText(plot.name, wide - LEGEND_MARGIN_PX * ratio, down);
      down += size * 1.4;
    }
    ctx.restore();
  }

  // -- the menus -----------------------------------------------------------

  // The canvas menu, which is where to look from and what the pane holds. Every
  // word of it is Python's answer to the snapshot below.
  _menu(event) {
    this._offer(event, this.say.menu(this._state(null)));
  }

  // The menu one legend row offers, which is about that surface alone.
  _card(event, serial) {
    this._offer(event, this.say.card(this._state(serial)));
  }

  // Either menu, put up where the click was: what comes back is the name of a
  // control, which goes where every other one goes.
  _offer(event, entries) {
    controls.menu(this.canvas, event, entries, (name, value) =>
      this.say.command(name, value));
  }

  // This pane as the description of its controls has to read it: the panels and
  // the camera, which are the page's, and the serial of whatever the click was
  // over. What the surfaces are called and how they are drawn is Python's and
  // is not sent back to it.
  _state(pointed) {
    return {
      spinning: this.spinning !== null,
      boxed: this.boxed,
      names: this.lettered,
      legend: this.listed,
      pointed,
    };
  }

  // The gestures first - the arrow keys turn the camera and no menu entry names
  // them - and then the ladder, which is the keys the description says this
  // pane answers to.
  _pressed(event) {
    const turns = {
      ArrowLeft: [ORBIT_DEGREES, 0],
      ArrowRight: [-ORBIT_DEGREES, 0],
      ArrowUp: [0, ORBIT_DEGREES],
      ArrowDown: [0, -ORBIT_DEGREES],
    };
    const turn = turns[event.key];
    if (turn !== undefined) {
      event.preventDefault();
      this.orbit(turn[0], turn[1]);
      return;
    }
    const command = controls.pressed(this.commands, event);
    if (command !== null) {
      event.preventDefault();
      this.say.command(command, null);
    }
  }
}

// -- one surface on the card -------------------------------------------------------

// A surface as the card holds it: two drawings of one set of numbers.
//
// The solid is the triangles, colored a vertex at a time and lit; the lines are
// the wire grid of the samples. Both are uploaded whichever look is on, because
// the toggle between them has to be a toggle - a round trip to the worker to
// see the same numbers drawn differently would be a wait for nothing.
//
// A wire surface has its hidden lines removed, the way Derive's plotter removed
// them: a see-through grid is a shape a reader has to solve rather than see.
// The solid stays under the lines as an occluder, painted in the color of the
// canvas and shoved a hair back by the polygon offset - an invisible body for
// the depth buffer to hide the far lines behind.
class Standing {
  constructor(serial, spec) {
    this.serial = serial;
    this.generation = 0;
    this.vertices = 0;
    this.triangles = 0;
    this.geometry = new THREE.BufferGeometry();
    this.strands = new THREE.BufferGeometry();
    this.lit = new THREE.MeshStandardMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      roughness: 1,
      metalness: 0,
      flatShading: false,
    });
    this.occluder = new THREE.MeshBasicMaterial({
      color: new THREE.Color(BACKGROUND),
      side: THREE.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: WIRE_OFFSET[0],
      polygonOffsetUnits: WIRE_OFFSET[1],
    });
    this.mesh = new THREE.Mesh(this.geometry, this.lit);
    this.mesh.visible = false;
    // The width is asked for and will not be given: WebGL draws every line one
    // pixel wide whatever a material says, which is the same refusal the
    // desktop's core profile makes. The wire reads anyway, because what makes
    // it read is the solid hiding the half of it that is behind.
    this.lines = new THREE.LineSegments(
      this.strands,
      new THREE.LineBasicMaterial({ vertexColors: true, linewidth: WIRE_WIDTH }),
    );
    this.lines.visible = false;
    this.respec(spec);
  }

  respec(spec) {
    this.label = spec.label;
    this.name = spec.name;
    this.color = spec.color;
    this.wire = Boolean(spec.wire);
    this.hidden = Boolean(spec.hidden);
    this.mesh.material = this.wire ? this.occluder : this.lit;
    this._shown();
  }

  enter(world) {
    world.add(this.mesh);
    world.add(this.lines);
  }

  leave() {
    if (this.mesh.parent) this.mesh.parent.remove(this.mesh);
    if (this.lines.parent) this.lines.parent.remove(this.lines);
    this.geometry.dispose();
    this.strands.dispose();
    this.lit.dispose();
    this.occluder.dispose();
    this.lines.material.dispose();
  }

  // The geometry Python built, put where the card can see it.
  //
  // A domain moved with the grid left alone gives a mesh of exactly the shape
  // the last one had - the same vertices in the same order, standing in
  // different places - so the buffers are written through and the index, which
  // says which of them make triangles, is left alone. That is the difference
  // between re-sampling a surface and building one: the triangles are already
  // on the card and are not sent again.
  upload(message) {
    const vertices = typed(message.vertices, this.label);
    const faces = typed(message.faces, this.label, Uint32Array);
    const colors = typed(message.colors, this.label);
    this.vertices = vertices.length / 3;
    this.triangles = faces.length / 3;
    if (this._rewritable(vertices, faces)) {
      const position = this.geometry.attributes.position;
      position.array.set(vertices);
      position.needsUpdate = true;
      const color = this.geometry.attributes.color;
      color.array.set(colors);
      color.needsUpdate = true;
    } else {
      this.geometry.setIndex(new THREE.BufferAttribute(faces, 1));
      this.geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
      this.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    }
    // The normals are the card's own business either way: the lighting three.js
    // adds is over the surface's real slope, and a rewritten position is a new
    // slope. It is one pass over the triangles and it is not what a re-sample
    // costs.
    this.geometry.computeVertexNormals();
    this.geometry.computeBoundingSphere();
    this._strands(message);
    this._shown();
  }

  // Whether the mesh on the card is the same mesh, standing somewhere else.
  //
  // The same size is not the same triangles: a surface with a hole in it keeps
  // only the faces that reach, so a hole that moved is a different mesh over
  // the same number of vertices. Hence the comparison, which is a pass over an
  // array that is already here and is what buys the pass over one that is not.
  _rewritable(vertices, faces) {
    const position = this.geometry.attributes.position;
    const index = this.geometry.index;
    if (position === undefined || index === null) return false;
    if (position.array.length !== vertices.length) return false;
    if (index.array.length !== faces.length) return false;
    for (let at = 0; at < faces.length; at += 1) {
      if (index.array[at] !== faces[at]) return false;
    }
    return true;
  }

  _strands(message) {
    const points = typed(message.wire, this.label);
    const colors = typed(message.wirecolors, this.label);
    const position = this.strands.attributes.position;
    if (position !== undefined && position.array.length === points.length) {
      position.array.set(points);
      position.needsUpdate = true;
      const color = this.strands.attributes.color;
      color.array.set(colors);
      color.needsUpdate = true;
    } else {
      this.strands.setAttribute('position', new THREE.BufferAttribute(points, 3));
      this.strands.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    }
    this.strands.computeBoundingSphere();
  }

  _shown() {
    const drawn = !this.hidden && this.triangles > 0;
    this.mesh.visible = drawn;
    this.lines.visible = !this.hidden && this.wire && this.vertices > 0;
  }
}

// -- odds and ends ---------------------------------------------------------------

// A typed array is what a 1-D contiguous array crosses as, and anything else
// means the conversion took the slow road. Saying so is cheaper than wondering
// why a surface takes a second to appear - and a surface is thousands of
// numbers where a curve is hundreds.
function typed(array, label, kind = Float32Array) {
  if (!ArrayBuffer.isView(array)) {
    console.warn(`${label}: geometry arrived as ${array && array.constructor
      && array.constructor.name}, not a typed array`);
    return kind.from(array || []);
  }
  return array;
}

// A camera angle as the inspector shows one. The numbers of the box are spelled
// in Python, where they mean something; these are the page's own.
function rounded(value) {
  return Number(value.toFixed(2));
}

// One point of the box stepped out of it, along a direction and by a distance,
// which is where a tick mark ends and where its number stands.
function stepped(from, out, far) {
  return [from[0] + out[0] * far, from[1] + out[1] * far, from[2] + out[2] * far];
}

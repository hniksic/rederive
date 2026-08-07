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
// clipped to - arrive spelled by the same function the desktop's inspector
// spells them with. What this file writes for itself is where the camera is,
// which is a fact about the page and not a value of anything.
//
// Nor does it own what a pane offers. The commands, their words, their keys and
// the order a menu stands them in are `plot/controls.py`'s, the same table the
// desktop's window renders, and where the camera goes for each of them is
// `plot/actions.py`'s; a pane is handed the ones this backend serves when it is
// made and the menu as it should currently read when a right click asks for
// one, and it answers with the name of a control.

import * as THREE from 'three';
import { OrbitControls } from './three/OrbitControls.js';
import * as controls from './controls.js';

// The canvas colors, which are `plot/qt/window3d.py`'s: the same near-black the
// 2D pane draws on, and a box drawn in a gray that reads against it without
// competing with the surface standing in it.
const BACKGROUND = '#0c0c10';
const BOX_COLOR = 0x969696;
const BOX_ALPHA = 0.43;

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

// Where a fresh pane is put and how big it is, and how far the next one is
// offset so that two panes are two panes and not one on top of another.
const PANE_WIDTH = 760;
const PANE_HEIGHT = 620;
const PANE_STEP = 28;

// The six numbers of the tool row, in the order they stand in it.
const FIELDS = ['x0', 'x1', 'y0', 'y1', 'nx', 'ny'];

// How much of a hidden legend row is left standing. Dimmed rather than struck
// through: a hidden surface is a surface that is still in the pane.
const LEGEND_FADED = 0.4;

// How far the pointer may move between a right-button press and its release and
// still count as a click that opens the menu rather than a pan of the camera.
const CLICK_SLOP_PX = 4;

// The one place a plot pane is put, laid over the terminal and letting the
// pointer through everywhere it has no pane. The 2D panes are in the same
// layer, so a solid and a curve stack the way two windows do.
let root = null;
let opened = 0;
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
export function wire(term) {
  terminal = term;
}

// One pane, opened where the plot session asked for one. What comes back is
// what the session's window handle calls into: everything here is a method
// Python names, and nothing else on this side is reachable from there.
export function open(number, commands, handlers) {
  const pane = new Solid(number, commands, handlers);
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
  constructor(number, commands, handlers) {
    this.number = number;
    // What this pane offers, as `plot/controls.py` describes it: the keys, the
    // buttons and the words are read off this and are spelled nowhere here.
    this.commands = commands;
    this.say = handlers;
    this.plots = new Map();
    this.order = [];
    this.standing = null;
    this.listed = true;
    this.boxed = true;
    this.inspecting = false;
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
    const offset = (opened++ % 6) * PANE_STEP;
    pane.style.left = `${40 + offset}px`;
    pane.style.top = `${40 + offset}px`;
    pane.style.width = `${PANE_WIDTH}px`;
    pane.style.height = `${PANE_HEIGHT}px`;
    pane.innerHTML =
      '<div class="plot-bar"><span class="plot-title"></span>' +
      '<button class="plot-close" title="Close">×</button></div>' +
      '<div class="plot-tools"></div>' +
      '<div class="plot-canvas"><div class="plot-legend"></div>' +
      '<div class="plot-readout"></div></div>' +
      '<div class="plot-status"></div>';
    this.element = pane;
    this.bar = pane.querySelector('.plot-bar');
    this.titleText = pane.querySelector('.plot-title');
    this.tools = pane.querySelector('.plot-tools');
    this.canvas = pane.querySelector('.plot-canvas');
    this.card = pane.querySelector('.plot-legend');
    this.readout = pane.querySelector('.plot-readout');
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
  _toolbar() {
    this.fields = {};
    const put = (text) => {
      const label = document.createElement('span');
      label.className = 'plot-label';
      label.textContent = text;
      this.tools.appendChild(label);
    };
    const field = (name) => {
      const edit = document.createElement('input');
      edit.className = 'plot-field';
      edit.spellcheck = false;
      // Enter and leaving the field are the same answer, which is what a
      // toolbar field means everywhere: the numbers now in the six of them.
      edit.addEventListener('change', () => this._framed());
      edit.addEventListener('keydown', (event) => {
        event.stopPropagation();
        if (event.key === 'Enter') this._framed();
      });
      this.fields[name] = edit;
      this.tools.appendChild(edit);
    };
    put('x:');
    field('x0');
    put('…');
    field('x1');
    put('y:');
    field('y0');
    put('…');
    field('y1');
    put('grid:');
    field('nx');
    put('×');
    field('ny');
    this.buttons = controls.bar(this.tools, this.commands, (name) => {
      this.say.command(name, null);
      this.element.focus();
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
    if (this.renderer === undefined) return;
    const width = Math.max(this.canvas.clientWidth, 120);
    const height = Math.max(this.canvas.clientHeight, 120);
    this.renderer.setSize(width, height, false);
    this._lens(width, height);
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
      this.renderer = new THREE.WebGLRenderer({ antialias: true });
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
      this._readout();
      this._paint();
    });
    this.frame = this._frame();
    this.world.add(this.frame);
  }

  // The box the picture stands in: twelve edges and nothing else.
  //
  // The desktop draws numbers along three of them, which wants a font on a
  // card; here the numbers are in the tool row and the readout, where they are
  // also editable. What the edges are for is the same either way - a surface
  // floating in the dark is a shape nobody can tell the size of.
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

  // The six numbers of the tool row, as Python spells them. It owns the domain
  // and the grid; these fields show what it has and hand back what was typed.
  domain(x0, x1, y0, y1, nx, ny) {
    const values = [x0, x1, y0, y1, nx, ny];
    FIELDS.forEach((name, at) => { this.fields[name].value = values[at]; });
  }

  said(text) {
    this.message = text || '';
    this.status.textContent = this.message;
  }

  dismiss() {
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
    this._readout();
    this._paint();
    // What the surface asks for in z, and what it was drawn in. The pane on the
    // Python side is what decides between the two, since one box for the window
    // is a rule about a picture and not about a surface.
    this.say.stood(message.plot, message.wanted, message.zrange);
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

  // The numbers behind the picture: the box as Python wrote it, the camera as
  // the page has it, and how much geometry there is to look at. Everything here
  // is reachable with the mouse and the fields; what it adds is exactness, and
  // a 3D picture is one of the few places where that is worth a panel.
  _readout() {
    if (!this.inspecting) {
      this.readout.style.display = 'none';
      return;
    }
    this.readout.style.display = '';
    if (this.camera === undefined) return;
    const box = this.standing;
    const camera = this.where();
    const lines = [];
    if (box !== null) {
      lines.push(`center   ${box.reading.center.join('   ')}`);
      lines.push(`length   ${box.reading.lengths.join('   ')}`);
      lines.push(`z        ${box.reading.z.join(' … ')}`);
    }
    lines.push(
      `camera   azimuth ${rounded(camera.azimuth)}` +
      `   elevation ${rounded(camera.elevation)}` +
      `   distance ${rounded(camera.distance)}`,
    );
    for (const serial of this.order) {
      const plot = this.plots.get(serial);
      if (plot === undefined) continue;
      lines.push(
        `${plot.label}   ${plot.vertices} vertices   ${plot.triangles} triangles`,
      );
    }
    this.readout.textContent = lines.join('\n');
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
    this._readout();
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
    this.say.framed(...FIELDS.map((name) => this.fields[name].value.trim()));
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
    this._paint();
  }

  // The panel of numbers behind the picture, which is what this pane has where
  // the desktop opens an inspector.
  inspect() {
    this.inspecting = !this.inspecting;
    this.lit('view.inspect', this.inspecting);
    this._readout();
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

// A camera angle as the readout shows one. The numbers of the box are spelled
// in Python, where they mean something; these are the page's own.
function rounded(value) {
  return Number(value.toFixed(2));
}

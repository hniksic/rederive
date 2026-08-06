// The page: a terminal, a runtime, and the program between them.
//
// What runs here is the app instance - Textual, Rich and this package, and no
// mathematics at all. The mathematics is a Web Worker away, has a Pyodide of
// its own, and is reached only by the Python in `rederive.web.engine`; nothing
// in this file knows what a request looks like or what an answer means. That
// is the same split the desktop program has, with `postMessage` where the pipe
// was.
//
// The one message this file does look at is a plot sampling. Its answer is
// typed arrays and is for the canvas rather than for Python, so the worker's
// messages are listened to twice: Python's own handler takes the pickles and
// steps over these, and the drawing modules take these and step over the
// pickles. The arrays therefore go from the interpreter that computed them to
// the canvas that draws them, and never become a Python object on this thread.
//
// There are two drawing modules, a flat picture and a solid one sharing no
// drawing at all, and one of them is asked first: a sampling is acknowledged
// exactly once, and the acknowledgement is what lets the next one go out.
//
// Nor does this file render anything. The one spelling of an expression is the
// one Python wrote; what crosses to JS is the bytes a terminal would have been
// sent, and xterm.js draws them exactly as a terminal would.
//
// Every URL named below is relative to this file. Nothing is fetched from a
// CDN, which is what `tools/build_web.py` is for.

import * as files from './files.js';
import * as plots from './plot2d.js';
import * as solids from './plot3d.js';

const SCREEN = 'screen';
const MANIFEST = 'manifest.json';

// What the page says for itself while there is nothing else to look at. It is
// written against a measurement rather than an apology - see
// `research/web-demo-channel.md` - and the one thing it must not do is imply
// that something is wrong: a first visit spends most of its time on the wire.
const OPENING = [
  'Rederive',
  '',
  'The whole program runs in this tab - the interpreter, the algebra and the',
  'plotting - with nothing behind it but a file server. About 24 MB comes down',
  'the first time, over a connection that compresses, and is cached afterwards.',
  'What follows that is compiling Python a desktop would have compiled once.',
  '',
];

// And what it says once the screen is up, about the two waits that are still to
// come. Both are sympy and numpy warming up inside the engine worker, and both
// happen exactly once.
const AFTERWARDS =
  'The first Simplify and the first plot each take a moment longer than the ones\r\n' +
  'after them, while the engine worker warms up.\r\n';

// The font the display is measured in. DejaVu is what stage 0 checked the
// typeset math against; the rest are what a machine without it is likely to
// have. Box drawing does not come from the font at all under the canvas and
// WebGL renderers - xterm draws those itself, to the cell - which is why the
// fraction bars and integral rails join whatever this resolves to.
const FONT = '"DejaVu Sans Mono", "Liberation Mono", Menlo, Consolas, monospace';
const FONT_SIZE = 16;

// What the page says while it is loading, and where it says it: in the
// terminal itself, since that is the one thing on the page and the program
// takes it over with the alternate screen the moment it starts.
const say = (term, text) => term.write(text + '\r\n');

// When each phase finished, for a measurement that does not need a stopwatch
// held to the screen. Read it out of the console, or off `window.rederive`.
const timings = {};
const mark = (name, since) => {
  timings[name] = Math.round(performance.now() - since);
  return performance.now();
};

// One line of the loading screen, opened when a phase starts and closed with
// what it cost. A user watching a page load wants to know that it is still
// going and roughly how far along it is, and the honest way to say both is to
// name each piece as it is fetched and to print the seconds it took.
const starting = (term, what) => term.write('  ' + what.padEnd(22, '.') + ' ');
const finished = (term, name, since) => {
  const at = mark(name, since);
  term.write((timings[name] / 1000).toFixed(1) + ' s\r\n');
  return at;
};

function terminal() {
  const term = new Terminal({
    fontFamily: FONT,
    fontSize: FONT_SIZE,
    // The program owns the whole screen and keeps no history above it: a
    // scrollback buffer here would only ever hold what the alternate screen
    // covered up.
    scrollback: 0,
    // What the unicode11 addon needs before it may be loaded.
    allowProposedApi: true,
    theme: { background: '#000000', foreground: '#d0d0d0' },
  });
  term.loadAddon(new Unicode11Addon.Unicode11Addon());
  term.unicode.activeVersion = '11';
  const fit = new FitAddon.FitAddon();
  term.loadAddon(fit);
  const screen = document.getElementById(SCREEN);
  term.open(screen);
  render(term, true);
  fit.fit();
  term.focus();
  // A click anywhere puts the keyboard back on the terminal, since there is
  // nothing else on the page for it to be on.
  screen.addEventListener('mousedown', () => term.focus());
  tapping(term, screen);
  // Refitting posts a resize, which the driver turns into the event a program
  // on a desktop would have got from SIGWINCH.
  let pending = null;
  window.addEventListener('resize', () => {
    clearTimeout(pending);
    pending = setTimeout(() => fit.fit(), 100);
  });
  return term;
}

// A tap is a click, which is the whole of what a touch screen needs from this
// program: the menus are words on a band, and pointing at one is how the mouse
// works them on a desktop.
//
// xterm.js hears nothing from a finger - it reports mouse buttons and has no
// touch handling at all - so the report is written by hand and fed back in as
// if the terminal had sent it. SGR, because that is the encoding the driver
// turned on, and it is the only one that says which button was released.
//
// The touch is cancelled rather than let through: a tap the browser is allowed
// to finish produces a second, synthetic mouse press a moment later, and the
// program would see every tap twice. Cancelling it also takes away the focus a
// tap would have given, which is why the terminal is focused by hand - and
// focusing it inside the gesture is what puts a phone's keyboard up.
function tapping(term, element) {
  let at = null;
  const within = (value, most) => Math.min(most, Math.max(1, Math.ceil(value)));
  const cell = (touch) => {
    const screen = element.querySelector('.xterm-screen') || element;
    const box = screen.getBoundingClientRect();
    return {
      column: within((touch.clientX - box.left) / (box.width / term.cols), term.cols),
      row: within((touch.clientY - box.top) / (box.height / term.rows), term.rows),
    };
  };
  element.addEventListener('touchstart', (event) => {
    if (event.touches.length !== 1) return;
    at = cell(event.touches[0]);
    event.preventDefault();
    term.focus();
  }, { passive: false });
  element.addEventListener('touchend', (event) => {
    if (at === null) return;
    // Press and release in one go. A finger has no button to hold down, and a
    // press with no release would leave the program dragging.
    term.input(`\x1b[<0;${at.column};${at.row}M`, false);
    term.input(`\x1b[<0;${at.column};${at.row}m`, false);
    at = null;
    event.preventDefault();
  }, { passive: false });
  element.addEventListener('touchcancel', () => { at = null; });
}

// WebGL, with canvas behind it. The two are pixel-identical - they share the
// glyph drawing and differ in how the result reaches the screen - so the
// fallback costs nothing in appearance, and the DOM renderer, which does not
// draw box glyphs itself, is not used at all.
//
// A dropped context is the reason this is a function rather than four lines:
// the addon stops drawing and the terminal is left blank forever unless
// something puts a renderer back. One more attempt at WebGL, then canvas,
// which needs no context to lose.
function render(term, retry) {
  try {
    const webgl = new WebglAddon.WebglAddon();
    webgl.onContextLoss(() => {
      webgl.dispose();
      render(term, false);
    });
    term.loadAddon(webgl);
    return 'webgl';
  } catch (error) {
    console.warn('WebGL is unavailable, drawing on canvas instead:', error);
  }
  if (retry) {
    try {
      term.loadAddon(new CanvasAddon.CanvasAddon());
      return 'canvas';
    } catch (error) {
      console.warn('canvas is unavailable too, leaving the DOM renderer:', error);
    }
  }
  return 'dom';
}

// One engine worker, booting as it is handed over. What it is told is where
// its runtime and its wheels are and nothing else: the protocol on top of it
// is Python's at both ends, and every message after this one is a pickle.
//
// A module worker, because Pyodide refuses to load in a classic one at all.
//
// The plot listener goes on here rather than anywhere else, because a worker is
// replaced whenever one dies and a listener attached elsewhere would be
// listening to a worker that is gone. It runs beside Python's own handler and
// neither knows about the other: each recognizes its own messages and lets the
// rest by.
function spawn(manifest) {
  const worker = new Worker('worker.js', { type: 'module' });
  worker.addEventListener('message', (event) => {
    if (!solids.heard(event.data)) plots.heard(event.data);
  });
  worker.postMessage({
    indexURL: new URL(manifest.pyodide, location.href).href,
    packages: manifest.packages,
    wheels: manifest.worker.map((name) => new URL(name, location.href).href),
  });
  return worker;
}

async function main() {
  const term = terminal();
  files.wire(term);
  for (const line of OPENING) say(term, line);
  const started = performance.now();
  const manifest = await (await fetch(MANIFEST)).json();
  let at = performance.now();

  starting(term, 'the interpreter');
  const { loadPyodide } = await import('./' + manifest.pyodide + 'pyodide.mjs');
  const pyodide = await loadPyodide({ indexURL: manifest.pyodide });
  at = finished(term, 'runtime', at);
  starting(term, 'the program');
  // The app instance and nothing under it: no sympy here, ever. The engine
  // worker loads its own, in the interpreter where the computing happens.
  await pyodide.loadPackage(manifest.page, { messageCallback: () => {} });
  at = finished(term, 'package', at);
  say(term, '');
  term.write(AFTERWARDS);
  say(term, '');
  starting(term, 'the screen');

  pyodide.globals.set('TERMINAL', term);
  pyodide.globals.set('SPAWN', () => spawn(manifest));
  pyodide.globals.set('PLOTS', plots);
  pyodide.globals.set('SOLIDS', solids);
  pyodide.globals.set('FILES', files);
  window.rederive = { term, pyodide, timings, plots, solids, files };
  term.onRender(() => {
    if (timings.prompt === undefined) mark('prompt', started);
  });

  // `run_async` and never `run`: there is one thread here and the loop is the
  // browser's, so the app is awaited rather than run inside a loop of its own.
  await pyodide.runPythonAsync(`
from rederive.web.boot import start

await start(TERMINAL, SPAWN, PLOTS, SOLIDS, FILES)
`);
  mark('session', started);
  say(term, '');
  say(term, 'Rederive has ended. Reload the page to start it again.');
}

main().catch((error) => {
  // Never silence: whatever went wrong goes where the user is looking as well
  // as into the console, since a page that stops loading and says nothing is
  // indistinguishable from a page that is still loading.
  console.error(error);
  const term = window.rederive && window.rederive.term;
  const said = '\r\n' + String(error && error.stack ? error.stack : error) + '\r\n';
  if (term) term.write(said.replace(/\n/g, '\r\n'));
  else document.body.textContent = said;
});

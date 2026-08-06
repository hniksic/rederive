// The page: a terminal, a runtime, and the program between them.
//
// What runs here is the app instance - Textual, Rich and this package, and no
// mathematics at all. The mathematics is a Web Worker away, has a Pyodide of
// its own, and is reached only by the Python in `rederive.web.engine`; nothing
// in this file knows what a request looks like or what an answer means. That
// is the same split the desktop program has, with `postMessage` where the pipe
// was.
//
// Nor does this file render anything. The one spelling of an expression is the
// one Python wrote; what crosses to JS is the bytes a terminal would have been
// sent, and xterm.js draws them exactly as a terminal would.
//
// Every URL named below is relative to this file. Nothing is fetched from a
// CDN, which is what `tools/build_web.py` is for.

const SCREEN = 'screen';
const MANIFEST = 'manifest.json';

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
  term.open(document.getElementById(SCREEN));
  render(term, true);
  fit.fit();
  term.focus();
  // A click anywhere puts the keyboard back on the terminal, since there is
  // nothing else on the page for it to be on.
  document.getElementById(SCREEN).addEventListener('mousedown', () => term.focus());
  // Refitting posts a resize, which the driver turns into the event a program
  // on a desktop would have got from SIGWINCH.
  let pending = null;
  window.addEventListener('resize', () => {
    clearTimeout(pending);
    pending = setTimeout(() => fit.fit(), 100);
  });
  return term;
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
function spawn(manifest) {
  const worker = new Worker('worker.js', { type: 'module' });
  worker.postMessage({
    indexURL: new URL(manifest.pyodide, location.href).href,
    packages: manifest.packages,
    wheels: manifest.worker.map((name) => new URL(name, location.href).href),
  });
  return worker;
}

async function main() {
  const term = terminal();
  say(term, 'Rederive');
  say(term, 'Loading the runtime...');
  const started = performance.now();
  const manifest = await (await fetch(MANIFEST)).json();
  let at = performance.now();

  const { loadPyodide } = await import('./' + manifest.pyodide + 'pyodide.mjs');
  const pyodide = await loadPyodide({ indexURL: manifest.pyodide });
  at = mark('runtime', at);
  say(term, 'Loading the program...');
  // The app instance and nothing under it: no sympy here, ever. The engine
  // worker loads its own, in the interpreter where the computing happens.
  await pyodide.loadPackage(manifest.page, { messageCallback: () => {} });
  at = mark('package', at);

  pyodide.globals.set('TERMINAL', term);
  pyodide.globals.set('SPAWN', () => spawn(manifest));
  window.rederive = { term, pyodide, timings };
  term.onRender(() => {
    if (timings.prompt === undefined) mark('prompt', started);
  });

  // `run_async` and never `run`: there is one thread here and the loop is the
  // browser's, so the app is awaited rather than run inside a loop of its own.
  await pyodide.runPythonAsync(`
from rederive.web.boot import start

await start(TERMINAL, SPAWN)
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

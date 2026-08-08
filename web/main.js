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
// CDN, which is what `tools/build_web.py` is for. The one thing this page ever
// asks another host for is one of the original's demonstrations, and only when
// the user has chosen one: see `demos.js`.

import * as demos from './demos.js';
import * as files from './files.js';
import * as plots from './plot2d.js';
import * as solids from './plot3d.js';

const SCREEN = 'screen';
const MANIFEST = 'manifest.json';

// The title page, in the words the program opens with itself. What loads here
// stands where the greeting will stand and says the same things, so that the
// screen the program takes over is the one that was already there.
const TITLE = 'R E D E R I V E';
const TAGLINE = 'A Mathematical Assistant';

// What the page says for itself while there is nothing else to look at. The
// first paragraph is what the program is, since a minute spent waiting is the
// one minute a visitor will read anything at all; the second is what the wait
// is for, written against a measurement rather than an apology. The one thing
// neither may do is imply that something is wrong: a first visit spends most
// of its time on the wire.
//
// A paragraph is one string and is broken to the terminal it finds, since a
// phone is forty columns wide and prose hard-wrapped at seventy-six reads on
// one as every second line half empty.
const OPENING = [
  'Rederive simplifies, solves, expands and plots, symbolically and' +
    ' numerically, and typesets its answers. The menu along the foot of the' +
    ' screen is the whole of the interface.',
  'All of it runs in this tab. About 24 MB comes down the first time and is' +
    ' cached afterwards, and what follows that is compiling Python a desktop' +
    ' would have compiled once.',
];

// The widest the prose is set, however wide the window is: text is read across,
// and a line that runs the whole of a desktop screen is not.
const COLUMNS = 72;

// Four quadrants, turning, and how long each is held. They are block elements
// rather than the braille a spinner is usually made of because xterm draws
// these itself under the canvas and WebGL renderers, so no font on the machine
// has to have them.
const SPINNER = ['▘', '▝', '▗', '▖'];
const TICK = 125;

// Bold for the name, dim for what is said around it, green for a phase that is
// done. The loading screen is the one place this file writes any style at all:
// everything after it is the terminal the program itself has written.
const BOLD = '\x1b[1m';
const DIM = '\x1b[2m';
const DONE = '\x1b[32m';
const PLAIN = '\x1b[0m';

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
};

// The turning of whichever phase is running, which nothing else on the page may
// be writing over: it lives out here so that the handler that reports a failure
// can stop it before it says anything.
let ticking = null;
const stop = () => {
  clearInterval(ticking);
  ticking = null;
};

// Prose broken to a width, on spaces, which is the one thing a terminal will
// not do for itself.
function wrapped(text, width) {
  const lines = [];
  let line = '';
  for (const word of text.split(' ')) {
    if (line && line.length + 1 + word.length > width) {
      lines.push(line);
      line = word;
    } else {
      line = line ? line + ' ' + word : word;
    }
  }
  return line ? lines.concat(line) : lines;
}

// The loading screen, and what drives it a phase at a time.
//
// The block is placed as a whole: its height is counted before a line of it is
// written, so that the page can be a title page rather than output run down
// from the top corner. High rather than halfway, which is how the program
// spaces its own greeting, and hard against the top of a window too short to
// spare the room.
//
// The version is the one thing here the page cannot know for itself, so the
// name goes up first and everything under it waits for the manifest.
function opening(term) {
  const width = Math.min(COLUMNS, Math.max(24, term.cols - 4));
  const left = ' '.repeat(Math.max(0, Math.floor((term.cols - width) / 2)));
  const paragraphs = OPENING.map((text) => wrapped(text, width));
  // The two names, the version, the prose and the three phases, with the blank
  // lines that space them.
  const height = 5 + paragraphs.reduce((n, lines) => n + lines.length + 1, 0) + 3;
  const centred = (text) =>
    left + ' '.repeat(Math.max(0, Math.floor((width - text.length) / 2))) + text;

  term.write('\r\n'.repeat(Math.max(0, Math.floor((term.rows - height) / 3))));
  say(term, BOLD + centred(TITLE) + PLAIN);
  say(term, DIM + centred(TAGLINE) + PLAIN);

  // One line of the loading screen, opened when a phase starts and closed with
  // what it cost. A user watching a page load wants to know that it is still
  // going and roughly how far along it is, and the honest way to say both is to
  // name each piece as it is fetched, to keep something turning while it is,
  // and to print the seconds it took. The line is rewritten from its start
  // each time rather than appended to, which is what lets the count run.
  let phase = null;
  const named = (what) =>
    '\r\x1b[K' + left + '  ' + what + ' ' +
    DIM + '.'.repeat(Math.max(2, 22 - what.length)) + PLAIN + ' ';
  const opened = (what) => {
    phase = { what, since: performance.now() };
    term.write(named(what));
  };

  return {
    version(release) {
      say(term, '');
      // A build that names no version leaves the line blank rather than the
      // block a line shorter than it was placed for.
      say(term, release ? DIM + centred('Version ' + release) + PLAIN : '');
      say(term, '');
      for (const lines of paragraphs) {
        for (const text of lines) say(term, left + text);
        say(term, '');
      }
    },

    starting(what) {
      opened(what);
      ticking = setInterval(() => {
        const spent = performance.now() - phase.since;
        const turn = SPINNER[Math.floor(spent / TICK) % SPINNER.length];
        term.write(named(phase.what) + turn + ' ' + (spent / 1000).toFixed(1) + ' s');
      }, TICK);
    },

    finished(name) {
      stop();
      mark(name, phase.since);
      const spent = (timings[name] / 1000).toFixed(1);
      term.write(named(phase.what) + DONE + spent + ' s' + PLAIN + '\r\n');
    },

    // The last phase stands still. What follows it is the program taking the
    // screen for itself, and a timer still turning would be writing over it.
    last(what) {
      opened(what);
    },
  };
}

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
  demos.wire(term);
  plots.wire(term);
  solids.wire(term);
  const screen = opening(term);
  const started = performance.now();
  const manifest = await (await fetch(MANIFEST)).json();
  screen.version(manifest.rederive);

  screen.starting('the interpreter');
  const { loadPyodide } = await import('./' + manifest.pyodide + 'pyodide.mjs');
  const pyodide = await loadPyodide({ indexURL: manifest.pyodide });
  screen.finished('runtime');
  screen.starting('the program');
  // The app instance and nothing under it: no sympy here, ever. The engine
  // worker loads its own, in the interpreter where the computing happens.
  await pyodide.loadPackage(manifest.page, { messageCallback: () => {} });
  screen.finished('package');
  screen.last('the screen');

  pyodide.globals.set('TERMINAL', term);
  pyodide.globals.set('SPAWN', () => spawn(manifest));
  pyodide.globals.set('PLOTS', plots);
  pyodide.globals.set('SOLIDS', solids);
  pyodide.globals.set('FILES', files);
  pyodide.globals.set('DEMOS', demos);
  window.rederive = { term, pyodide, timings, plots, solids, files, demos };
  term.onRender(() => {
    if (timings.prompt === undefined) mark('prompt', started);
  });

  // `run_async` and never `run`: there is one thread here and the loop is the
  // browser's, so the app is awaited rather than run inside a loop of its own.
  await pyodide.runPythonAsync(`
from rederive.web.boot import start

await start(TERMINAL, SPAWN, PLOTS, SOLIDS, FILES, DEMOS)
`);
  mark('session', started);
  say(term, '');
  say(term, 'Rederive has ended. Reload the page to start it again.');
}

main().catch((error) => {
  // Never silence: whatever went wrong goes where the user is looking as well
  // as into the console, since a page that stops loading and says nothing is
  // indistinguishable from a page that is still loading. The loading screen is
  // stopped first, so that nothing is still counting under the report.
  stop();
  console.error(error);
  const term = window.rederive && window.rederive.term;
  const said = '\r\n' + String(error && error.stack ? error.stack : error) + '\r\n';
  if (term) term.write(said.replace(/\n/g, '\r\n'));
  else document.body.textContent = said;
});

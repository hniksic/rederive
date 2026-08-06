// The engine worker: a Pyodide of its own, and Python answering on it.
//
// This file is the boot and nothing else. It loads the runtime, loads sympy and
// this package, and hands the message channel to `rederive.web.worker.serve`,
// which owns it from then on - so every message after the one that starts this
// is a pickle between two Pythons, and nothing here knows what any of them say.
//
// A module worker and `await import`, because Pyodide 314 refuses to load under
// `importScripts` outright: a classic worker gets `Classic web workers are not
// supported` and nothing else.
//
// The one message this file may send is a string, which is how the page says
// that there is no worker to be had - a runtime that would not load, a wheel
// that would not install. `rederive.web.engine` reads a string as a death
// before the handshake, and three of those in a row are what put the engine
// down rather than spinning on a broken install.

let booting = false;

self.onmessage = (event) => {
  if (booting) return;
  booting = true;
  boot(event.data);
};

async function boot(where) {
  try {
    const { loadPyodide } = await import(where.indexURL + 'pyodide.mjs');
    // The packages come down while the runtime is starting, which is the whole
    // reason they are named here rather than loaded afterwards.
    const pyodide = await loadPyodide({
      indexURL: where.indexURL,
      packages: where.packages,
    });
    await pyodide.loadPackage(where.wheels, { messageCallback: () => {} });
    // `serve` imports the engine - sympy and the four hundred modules under it
    // - and only then says it is ready, so a page that has heard the handshake
    // knows the next request pays for nothing but its own answer.
    await pyodide.runPythonAsync(`
from rederive.web.worker import serve

serve()
`);
  } catch (error) {
    // The last line of it, which for a Python exception is the one naming what
    // went wrong: a message line has one line to say this in, and the rest of
    // the traceback is in the console for whoever wants it.
    const said = String((error && error.message) || error).trim().split('\n');
    self.postMessage('The engine could not be started: ' + said[said.length - 1].trim());
  }
}

// What a tab has instead of a filesystem: a download, a picker, an address.
//
// A page cannot open a file and cannot write one. What it has is three
// gestures, and this file is the three of them and the page's menu they are
// on, and nothing else - the store that gives them names, and every decision
// about what goes where, is Python's in `rederive.platform.web` and
// `rederive.web.files`.
//
// * **Save downloads.** A worksheet leaves the tab the only way anything
//   leaves a tab, as a file the browser puts wherever the user keeps them.
// * **A file comes in by being handed over**, picked with the button or
//   dropped on the page. Neither can be asked for by the program: a picker
//   opens on a click and on nothing else, which is why this is page furniture
//   and not a command.
// * **A link is the address.** The worksheet goes in the fragment, which is
//   the half of a URL no server is ever sent, so a shared link carries the
//   mathematics and tells nobody about it.
//
// The controls are on one menu in a corner rather than in a bar, because the
// terminal is the page and a bar would take a row of it away from the program.
// It is faded until pointed at, and the keyboard goes back to the terminal the
// moment anything on it has been used. Beside the gestures the menu carries
// the one piece of housekeeping a tab that remembers its settings needs, which
// is a way to make it forget them.

const NOTICE_SECONDS = 8;

// Long enough for the browser to have started the download. Revoking sooner
// cancels it; never revoking leaks the whole worksheet for the life of the tab.
const REVOKE_MS = 60000;

import { settle } from './keys.js';
import { dropdown } from './menu.js';

let terminal = null;
let handlers = null;
let notice = null;
let timer = null;

// Called by the page as it builds itself, and before Python knows anything
// about any of this: the DOM is wired here, and what the controls *do* arrives
// later, with `attend`. A control pressed in between says so rather than
// throwing, since the interesting few seconds of this page are the ones before
// the program has started.
export function wire(term) {
  terminal = term;
  notice = document.getElementById('notice');
  const menu = dropdown(term, named('page'), named('gestures'));
  // A choice closes the menu, whatever it was: the one that leaves the page
  // included, since the tab it opens takes the pointer with it.
  named('gestures').addEventListener('click', () => menu.show(false));
  const picker = document.getElementById('picker');
  named('pick').addEventListener('click', () => {
    // The value is cleared first so that picking the same file twice is two
    // events: `change` says nothing when what was picked has not changed.
    picker.value = '';
    picker.click();
  });
  picker.addEventListener('change', () => {
    take(picker.files);
    focus();
  });
  named('link').addEventListener('click', () => {
    if (ready()) handlers.linked();
    focus();
  });
  named('reset').addEventListener('click', () => {
    if (ready()) handlers.reset();
    focus();
  });
  document.addEventListener('dragover', (event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy';
  });
  document.addEventListener('drop', (event) => {
    event.preventDefault();
    take(event.dataTransfer.files);
    focus();
  });
}

// Python's side of the controls: what to do with a file that arrived, what to
// put in the address, and what a reset of the settings is. Held rather than
// called, since none of it happens until the user asks.
export function attend(given) {
  handlers = given;
}

// A worksheet on its way out of the tab. The name is the one the user typed at
// the prompt, so the file lands under the name the program thinks it has.
export function download(name, text) {
  const url = URL.createObjectURL(
    new Blob([text], { type: 'text/plain;charset=utf-8' }),
  );
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = name;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), REVOKE_MS);
}

// The worksheet, in the one part of a URL that is never sent anywhere.
// `replaceState` rather than assigning to `location.hash`, which would put an
// entry in the history and turn the browser's Back button into something that
// undoes a save.
//
// What was written is kept as well, for the Back that a touch screen turns into
// Esc: that steps down to the entry under this one, which has never heard of
// the worksheet, and `written` is how it is put back there.
let wrote = null;

export function address(fragment) {
  wrote = fragment === '' ? location.pathname : '#' + fragment;
  history.replaceState(null, '', wrote);
}

// The address last written, or null where nothing has been.
export function written() {
  return wrote;
}

// What this file has instead of the program's message line, for the things
// that happen beside the program rather than inside it: a file dropped on the
// page, a link made, a store with no room left. It fades by itself, because
// nothing here is worth a dismissal.
export function said(text) {
  if (notice === null) return;
  notice.textContent = text;
  notice.hidden = false;
  clearTimeout(timer);
  timer = setTimeout(() => { notice.hidden = true; }, NOTICE_SECONDS * 1000);
}

// -- what the gestures produce ------------------------------------------------

// Every file handed over, read as text and passed on one at a time. A file
// that will not read is reported and the rest still arrive: a drop of six
// worksheets should not be lost to one of them.
function take(files) {
  if (!ready()) return;
  for (const file of files) {
    file
      .text()
      .then((text) => handlers.arrived(file.name, text))
      .catch((error) => said(`${file.name} could not be read: ${error}`));
  }
}

function ready() {
  if (handlers !== null) return true;
  said('Rederive is still starting; try again once the screen is up');
  return false;
}

function named(id) {
  return document.getElementById(id);
}

// The keyboard belongs to the terminal. A button keeps focus after a click,
// and the next keystroke would go to it rather than to the program.
function focus() {
  if (terminal !== null) settle(terminal);
}

// The original's own demonstrations, fetched from where they survive.
//
// Derive shipped a tour of itself - six scripts of a comment and an expression,
// and three galleries of expressions worth drawing - and those files are still
// what the program is best shown off with. They are not Rederive's to ship, so
// they are downloaded when one is asked for.
//
// The list, the titles and every address come from `rederive.model.demos` and
// arrive with `attend`, because a desktop wanting the same menu wants the same
// nine files. What is *here* is the browser's half and only that: a menu, a
// download, and one thing a page needs that nothing else does - a relay, for
// the road that is closed to pages. Until Python has attended there is no menu,
// which is why the button says so rather than opening an empty one.
//
// **Two roads, and the second one exists for a reason that has been seen to
// happen.** Ordinarily a demonstration is one file of a few kilobytes, taken
// out of a DOS diskette image by the archive itself; two images carry the same
// set, so both are asked at once and the first answer wins. But both of those
// go through one piece of machinery - the archive's extractor - and that is
// the piece that fails: the extractor for one of the two items answers nothing
// but 503. When it takes both images down at once, the whole diskette is
// fetched instead and opened here. That download is three hundred kilobytes
// against one, which is why it is the second road and not the first, and it
// brings back all nine demonstrations, which is why none of them is ever
// downloaded twice.
//
// The whole diskette is the one address a page may not read directly - the
// archive sends no CORS headers with it - so that road, and only that road,
// goes through a relay.
//
// What a file *is* stays Python's, as everything else across this seam does.
// The page knows a title and a URL, and hands over bytes.

import { said } from './files.js';

// Python says things about a demonstration too - which command runs it again,
// why one was refused - and it says them through this module, that being the
// object it was handed. One notice box, one way of writing in it, whichever
// side has something to say.
export { said };

// Public relays: they fetch a URL, follow the redirect the archive answers
// with, and hand back the body with the header that lets a page read it. Free
// servers promised to nobody, so they are asked all at once and the first to
// answer is the one used - on any given day some of them are rate-limited or
// down, and that is expected rather than a problem to be solved.
const RELAYS = [
  'https://api.cors.lol/?url={url}',
  'https://api.codetabs.com/v1/proxy?quest={url}',
  'https://corsproxy-8uo5.onrender.com/?url={url}',
];

// What a demonstration file may be before it is taken for one, and what the
// whole diskette may be. A relay that answers with its own error page answers
// it with a 200, so a page that only looked at the status would put an HTML
// apology on the worksheet.
const LARGEST_FILE = 65536;
const LARGEST_IMAGE = 4 * 1024 * 1024;

let terminal = null;
let handlers = null;
let menu = null;
let button = null;
let warmed = false;

// Called as the page builds itself, before Python knows any of this exists.
// What the button opens does not exist yet either: the menu is built when the
// list for it arrives.
export function wire(term) {
  terminal = term;
  button = document.getElementById('demo');
  menu = document.getElementById('gallery');
  button.addEventListener('click', (event) => {
    event.stopPropagation();
    if (handlers === null) {
      said('Rederive is still starting; the demonstrations arrive with it');
      return;
    }
    if (menu.hidden) warm();
    show(menu.hidden);
  });
  // A menu closes on anything that is not a choice: the next click anywhere,
  // Esc, or the keyboard leaving it. The terminal is behind all of this and
  // gets the keystroke either way, which is why Esc is not swallowed.
  document.addEventListener('click', () => show(false));
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') show(false);
  });
}

// Python's side: the demonstrations on offer, where the whole diskette is,
// whether one is here already, and what to do with what comes back. The menu
// is built here because this is when there is anything to build it out of.
export function attend(given) {
  handlers = given;
  fill(menu, given.demos);
}

function fill(where, demos) {
  let group = '';
  for (const demo of demos) {
    if (demo.group !== group) {
      group = demo.group;
      const heading = document.createElement('div');
      heading.className = 'gallery-heading';
      heading.textContent = group;
      where.appendChild(heading);
    }
    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'gallery-item';
    item.textContent = demo.title;
    item.title = `${demo.file}, from the original Derive`;
    item.addEventListener('click', () => {
      show(false);
      chose(demo);
    });
    where.appendChild(item);
  }
}

// The menu is open, so a download is likely: open the connection it would need
// now, while the list is being read. Once per page - a preconnect that is
// already made is not made again, and the browser drops an unused one itself.
function warm() {
  if (warmed) return;
  warmed = true;
  const origins = new Set(
    handlers.demos.flatMap((demo) => demo.urls).map((url) => new URL(url).origin),
  );
  for (const origin of origins) {
    const link = document.createElement('link');
    link.rel = 'preconnect';
    link.href = origin;
    document.head.appendChild(link);
  }
}

// A menu that is already where it is asked to be is left alone, which is what
// makes the two listeners in `wire` safe: they hear every click and every
// Escape in the page, and a closed menu must not take the keyboard back from a
// plot pane somebody is working in.
function show(open) {
  if (menu === null || open === !menu.hidden) return;
  menu.hidden = !open;
  button.setAttribute('aria-expanded', String(open));
  if (!open && terminal !== null) terminal.focus();
}

// A demonstration was asked for: fetch it if it is not here already, and hand
// it over. The download is said out loud, because it is the one thing this page
// does that can take a visible while or fail outright - and it is not said at
// all when there is nothing to wait for.
async function chose(demo) {
  let bytes = null;
  if (!handlers.kept(demo.file)) {
    said(`Fetching ${demo.file} from the Internet Archive...`);
    try {
      bytes = await fetched(demo);
    } catch (error) {
      said(`${demo.file} could not be downloaded: ${reason(error)}`);
      return;
    }
  }
  // Said rather than thrown. What is on the other side of this call is Python,
  // and a refusal from it inside an async function is an unhandled rejection -
  // a line in a console nobody has open, and a page where nothing happened.
  try {
    handlers.chose(demo.file, bytes);
  } catch (error) {
    said(`${demo.file} could not be started: ${error}`);
    throw error;
  }
  if (terminal !== null) terminal.focus();
}

// The file: out of whichever image answers first, or out of the whole diskette
// when neither will. Both images are asked at once rather than in turn, since
// the slow case is a server thinking and asking one after the other would add
// the waits together.
async function fetched(demo) {
  try {
    return await Promise.any(demo.urls.map((url) => got(url, LARGEST_FILE)));
  } catch (refused) {
    said(`The archive will not open ${demo.file}; fetching the whole diskette...`);
    try {
      return await unpacked(demo.file);
    } catch {
      // The archive's own refusal rather than the diskette's: the first road
      // is the one that was meant to work, and its failure is the one worth
      // reading.
      throw refused;
    }
  }
}

// The whole diskette, opened here. Every demonstration in it is kept, since
// they have all been paid for; the one that was asked for is handed back.
async function unpacked(file) {
  const relayed = RELAYS.map((relay) =>
    got(relay.replace('{url}', encodeURIComponent(handlers.image)), LARGEST_IMAGE),
  );
  const found = await entries(await Promise.any(relayed));
  for (const demo of handlers.demos) {
    const bytes = found.get(demo.file);
    if (bytes !== undefined && demo.file !== file) handlers.keep(demo.file, bytes);
  }
  const wanted = found.get(file);
  if (wanted === undefined) throw new Error(`the diskette has no ${file}`);
  return wanted;
}

async function got(url, largest) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`the server said ${response.status}`);
  // Bytes rather than text: these are DOS files, and which of the two
  // encodings one is written in is a question Python already answers.
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (!bytes.length || bytes.length > largest || bytes[0] === 0x3c) {
    throw new Error('what came back was not a file of the original');
  }
  return bytes;
}

// -- the diskette --------------------------------------------------------------
//
// Enough of a zip reader for one written in 1996: the central directory, and
// deflate. No library, because a browser is already a deflate implementation
// and every file in this one is either stored or deflated - a zip needs a
// reader in the megabytes only to cover what nothing here uses.

const SIGNATURE_END = 0x06054b50;
const SIGNATURE_ENTRY = 0x02014b50;
const STORED = 0;
const DEFLATED = 8;

// The end record is last in the file and holds a comment nobody writes, so it
// is found by looking back from the end rather than by seeking to it.
const LOOKBACK = 1024;

async function entries(zip) {
  const view = new DataView(zip.buffer, zip.byteOffset, zip.byteLength);
  let at = -1;
  for (let back = 22; back <= Math.min(LOOKBACK, zip.length); back++) {
    if (view.getUint32(zip.length - back, true) === SIGNATURE_END) {
      at = zip.length - back;
      break;
    }
  }
  if (at < 0) throw new Error('the diskette is not a zip file');
  let entry = view.getUint32(at + 16, true);
  const count = view.getUint16(at + 10, true);
  const found = new Map();
  for (let index = 0; index < count; index++) {
    if (view.getUint32(entry, true) !== SIGNATURE_ENTRY) break;
    const method = view.getUint16(entry + 10, true);
    const compressed = view.getUint32(entry + 20, true);
    const nameLength = view.getUint16(entry + 28, true);
    const extraLength = view.getUint16(entry + 30, true);
    const commentLength = view.getUint16(entry + 32, true);
    const start = view.getUint32(entry + 42, true);
    const name = new TextDecoder().decode(
      zip.subarray(entry + 46, entry + 46 + nameLength),
    );
    // The local header repeats the name and carries its own extra field, whose
    // length is its own: where the bytes start is only knowable from here.
    const local =
      30 + view.getUint16(start + 26, true) + view.getUint16(start + 28, true);
    const body = zip.subarray(start + local, start + local + compressed);
    if (method === STORED) found.set(name, body);
    else if (method === DEFLATED) found.set(name, await inflated(body));
    entry += 46 + nameLength + extraLength + commentLength;
  }
  return found;
}

async function inflated(body) {
  const stream = new Blob([body])
    .stream()
    .pipeThrough(new DecompressionStream('deflate-raw'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

// What went wrong, in one clause. `Promise.any` reports every failure at once,
// and the first of them is as much as a message line has room to say.
function reason(error) {
  const failures = error.errors;
  return failures !== undefined && failures.length ? failures[0] : error;
}

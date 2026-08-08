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
// download, and the relays a page needs because a page can be refused for
// being one. Until Python has attended there is no menu, which is why the
// button says so rather than opening an empty one.
//
// **Every part of this is about the wait**, because the wait is the whole of
// what stands between a click and the demonstration:
//
// * A demonstration already in the page is not downloaded at all. The store
//   outlives the tab, so the second viewing of one starts at once.
// * Opening the menu warms the connection to the archives. A TLS handshake
//   started while the eye is still going down a list of nine titles is a
//   handshake nobody waits for.
// * Every address is asked at once and the first answer wins. The files live
//   inside archive images of two DOS releases and both carry the same set, so
//   an image that is slow or refusing today costs nothing while the other is
//   up.
// * The relays join that same race a moment later, if nothing has answered.
//   Late rather than at once, because they are strangers' servers, free and
//   promised to nobody: a direct answer is the ordinary case and no third
//   party needs to be told which demonstration anyone chose.
//
// What a file *is* stays Python's, as everything else across this seam does.
// The page knows a title and a URL, and hands over bytes.

import { said } from './files.js';

// Public relays that fetch a URL and answer with the CORS headers a page needs.
// Each takes the address it is to fetch as one query field, percent-encoded.
// Nothing here is relied on: they are worth one attempt each, and the day none
// of them answers either is the day this button says so.
const RELAYS = [
  'https://api.cors.lol/?url={url}',
  'https://api.codetabs.com/v1/proxy?quest={url}',
  'https://corsproxy-8uo5.onrender.com/?url={url}',
];

// How long the archives are given before the relays join the race. Generous,
// because a relay fetches the same archive through one more hop and so only
// ever wins where the direct road is blocked rather than slow: a download that
// is merely taking its time is not worth telling a stranger about. The wait is
// not paid where it would matter - the relays start the instant every archive
// has failed, however early that is.
const HEDGE_MS = 1500;

// What a demonstration file may be before it is taken for one. A relay that
// answers with its own error page answers it with a 200, so a page that only
// looked at the status would put an HTML apology on the worksheet: these files
// are a few kilobytes of DOS text and nothing about them starts with a tag.
const LARGEST = 65536;

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

// Python's side: the demonstrations on offer, whether one is here already, and
// what to do with one that has been downloaded. The menu is built here because
// this is when there is anything to build it out of.
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

// The menu is open, so a download is likely: open the connections it would
// need now, while the list is being read. Once per page - a preconnect that is
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
  handlers.chose(demo.file, bytes);
  if (terminal !== null) terminal.focus();
}

// The file, from whichever address answers first. One race, entered by every
// archive at once and by the relays a moment later: asking one after another
// would add the waits together, and asking the relays at once would tell three
// strangers about a download that was never going to need them.
function fetched(demo) {
  const answered = { yet: false };
  const direct = demo.urls.map((url) =>
    got(url).then((bytes) => {
      answered.yet = true;
      return bytes;
    }),
  );
  // The relays enter when the archives have had their moment, or the instant
  // every one of them has failed - whichever comes first, so a refusal costs
  // nothing at all and a slow answer costs one wait rather than two.
  const late = Promise.race([after(HEDGE_MS), Promise.allSettled(direct)]);
  const relayed = RELAYS.map((relay, index) =>
    late.then(() => {
      // The race is already won; the losers of it ask nobody anything.
      if (answered.yet) throw new Error('the archive answered first');
      // Round robin over the archives rather than all of them through the
      // first: what a relay is for is an archive that will not answer a page,
      // and sending every relay to the same one would relay the same refusal.
      const url = demo.urls[index % demo.urls.length];
      return got(relay.replace('{url}', encodeURIComponent(url)));
    }),
  );
  return Promise.any([...direct, ...relayed]);
}

async function got(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`the server said ${response.status}`);
  // Bytes rather than text: these are DOS files, and which of the two
  // encodings one is written in is a question Python already answers.
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (!plausible(bytes)) throw new Error('what came back was not a math file');
  return bytes;
}

// Whether what came back can be a demonstration at all: something, not too
// much of it, and not a web page.
function plausible(bytes) {
  return bytes.length > 0 && bytes.length <= LARGEST && bytes[0] !== 0x3c;
}

function after(milliseconds) {
  return new Promise((wake) => setTimeout(wake, milliseconds));
}

// What went wrong, in one clause. `Promise.any` reports every failure at once,
// and the first of them is as much as a message line has room to say.
function reason(error) {
  const failures = error.errors;
  return failures !== undefined && failures.length ? failures[0] : error;
}

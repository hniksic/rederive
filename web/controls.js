// The controls of a plot pane, drawn from what Python describes them as.
//
// Nothing here knows what any command is called, what key presses it, or in
// what order a menu offers them. `plot/controls.py` is where that is written
// down, once, for the desktop windows and the panes alike; a pane is handed the
// controls it may offer when it is made and the menu as it should currently
// read when a right click asks for one, and this file turns either into DOM.
//
// So the whole of what is here is rendering: a row of buttons, a popup of
// lines, and the rule for deciding whether a key event is one of the keys a
// control answers to. A word in this file that a user can read would be a word
// the two backends could disagree about, and there is none.
//
// The gestures are not controls and are not here. A drag, a wheel, a pinch and
// the arrow keys mean one thing while a marker is up and another while it is
// not, no menu entry names them, and they belong to the pane that hears them.

// What a key of the table is spelled with when something has to be held down
// with it. The page's own alphabet, as `plot/controls.py` writes it: the keys
// a browser keeps for itself are simply absent from the description, so nothing
// here has to know which those are.
const HELD = { Ctrl: 'ctrlKey', Shift: 'shiftKey', Alt: 'altKey' };

// What a submenu is marked with where a key would otherwise be written, and
// what a ticked entry carries.
const DEEPER = '▸';
const TICK = '✓';

// Whether one key of the description is the key this event is.
//
// A single character matches whichever case it arrives in, because the page
// reports a letter as the letter that was typed and a control answers to its
// letter either way. Anything longer is a name - `Home`, `F3`, `Delete` - and
// is matched exactly. A modifier the key does not name must not be down, save
// for Shift, which is how `+` reaches a keyboard that spells it as a shifted
// `=`.
function matches(key, event) {
  const parts = key.split('+');
  const wanted = parts.pop();
  for (const [name, held] of Object.entries(HELD)) {
    const asked = parts.includes(name);
    if (asked !== Boolean(event[held]) && (asked || name !== 'Shift')) return false;
  }
  return wanted.length === 1
    ? wanted.toLowerCase() === event.key.toLowerCase()
    : wanted === event.key;
}

// Which control a key press is, or null where it is none of them. The order is
// the description's, so a key claimed by two controls goes to the one the table
// names first - and the table is the only place that can be argued with.
//
// A control the description says arrives as an event is passed over here, keys
// and all. Cancelling the keydown of such a stroke is exactly what must not
// happen - it cancels the thing the browser was about to offer - so the ladder
// does not claim it and `evented` below is how the pane hears it instead.
export function pressed(commands, event) {
  for (const one of commands) {
    if (one.event) continue;
    for (const key of one.keys) {
      if (matches(key, event)) return one.name;
    }
  }
  return null;
}

// Which control a page event of this name is, or null where none of them is
// heard that way. Copy image is the whole of the list, and which event carries
// it is the description's to say rather than this file's to know.
export function evented(commands, name) {
  for (const one of commands) {
    if (one.event === name) return one.name;
  }
  return null;
}

// The tool row: one button for every control the description gives a button's
// worth of words to. A toggle standing here shows its own state where it
// stands, which is worth more than an entry that has to be opened to be read,
// and `lit` below is how the state gets onto it.
//
// What comes back is the buttons by the name of the control each is, since the
// pane is the side that hears a toggle change and has to light it.
export function bar(where, commands, run) {
  const buttons = new Map();
  for (const one of commands) {
    if (!one.bar) continue;
    const button = document.createElement('button');
    button.className = 'plot-toggle';
    button.textContent = one.bar;
    button.title = one.hint;
    button.addEventListener('click', () => run(one.name));
    where.appendChild(button);
    buttons.set(one.name, button);
  }
  return buttons;
}

// One button of the tool row, on or off, by the name of the control it draws.
export function lit(buttons, name, on) {
  const button = buttons.get(name);
  if (button !== undefined) button.classList.toggle('on', Boolean(on));
}

// The menu a right click raises, where the click was and inside the element it
// is put in.
//
// Every entry is drawn as the description has it - the words, the key that does
// the same thing with no menu open, the tick, the greying and the rules between
// the groups - and a click hands the name of the control back. The pane never
// sees the words again.
export function menu(where, event, entries, run) {
  const box = where.getBoundingClientRect();
  const element = document.createElement('div');
  element.className = 'plot-menu';
  const away = () => {
    element.remove();
    window.removeEventListener('pointerdown', away, true);
  };
  for (const entry of entries) element.appendChild(line(entry, run, away));
  // In the capture phase, so that a click anywhere else takes the menu away
  // before whatever was under it hears the press.
  window.addEventListener('pointerdown', away, true);
  where.appendChild(element);
  place(element, box, event);
  return element;
}

// Where a menu stands: at the pointer, and back inside the picture wherever
// that would put an entry outside it. The element it is put in has its
// overflow clipped - a pane is a window and a picture does not spill out of one
// - so a menu hanging over the edge is a menu whose last entries cannot be
// read or reached. Measured after it is in the DOM, since how tall it is is
// how many entries the description gave it.
function place(element, box, event) {
  const wide = element.offsetWidth;
  const tall = element.offsetHeight;
  const left = event.clientX - box.left;
  const top = event.clientY - box.top;
  element.style.left = `${Math.max(0, Math.min(left, box.width - wide))}px`;
  element.style.top = `${Math.max(0, Math.min(top, box.height - tall))}px`;
}

// One line of a menu, and the submenu under it where it has one.
function line(entry, run, away) {
  const item = document.createElement('div');
  item.className = 'plot-item';
  if (entry.separated) item.classList.add('ruled');
  if (!entry.enabled) item.classList.add('off');
  const tick = document.createElement('span');
  tick.className = 'plot-tick';
  tick.textContent = entry.checked ? TICK : '';
  const label = document.createElement('span');
  label.className = 'plot-text';
  label.textContent = entry.label;
  const keys = document.createElement('span');
  keys.className = 'plot-keys';
  keys.textContent = entry.items.length ? DEEPER : entry.keys[0] || '';
  item.append(tick, label, keys);
  if (!entry.enabled) return item;
  if (entry.items.length) {
    const deeper = document.createElement('div');
    deeper.className = 'plot-menu plot-deeper';
    for (const one of entry.items) deeper.appendChild(line(one, run, away));
    item.appendChild(deeper);
    return item;
  }
  item.addEventListener('pointerdown', (press) => {
    press.stopPropagation();
    away();
    run(entry.name, entry.value === undefined ? null : entry.value);
  });
  return item;
}

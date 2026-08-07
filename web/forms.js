// The dialogs of a plot pane, drawn from what Python describes them as.
//
// `plot/forms.py` says what a form asks for - the fields, the words in front of
// them, the headings they stand under and the answers at the bottom - and this
// file turns one into an overlay. There is no word here a user can read, for
// the same reason there is none in `controls.js`: a word written twice is a word
// the two backends can disagree about, and the desktop draws these same fields
// as a QDialog.
//
// An overlay rather than a window, because a page has no window. It lies over
// the pane that opened it, takes the keyboard while it is up and hands it back
// when it goes, and it stops its own keys from reaching the pane underneath -
// a pane whose ladder heard the letters being typed into a field would frame
// the view every time somebody wrote an `a`.

// What a strip's divider is drawn as, and how much of a form's own layout is
// worth keeping: a page has no button box, so the answers stand in a row in the
// order the description gives them.
const APPLY = 'apply';
const CLOSE = 'close';

// The pair a form that names no buttons takes, since accepting and cancelling
// are the two answers every dialog has and neither is worth describing. The
// words are the platform's on a desktop and the browser has no platform to ask,
// so they are the plainest two there are.
const STANDARD = [
  { label: 'OK', role: APPLY, primary: true },
  { label: 'Cancel', role: CLOSE, primary: false },
];

// Put one form up over an element, filled with the values it opens on.
//
// `done` is handed what the fields say, in the order the description lists
// them, and the overlay closes behind it. Nothing here argues with what was
// typed: a form collects strings, and the side that asked for them is the side
// that knows what they have to be. `gone` is told whichever way it closed, so
// that the pane can stop holding a dialog that is no longer there. What comes
// back is how to close it from outside.
export function ask(where, form, values, done, gone) {
  const held = document.activeElement;
  const overlay = document.createElement('div');
  overlay.className = 'plot-form';
  const sheet = document.createElement('div');
  sheet.className = 'plot-sheet';
  overlay.appendChild(sheet);
  sheet.appendChild(headed(form));
  const fields = new Map();
  for (const group of form.groups) sheet.appendChild(grouped(form, group, fields));
  form.fields.forEach((one, at) => {
    const edit = fields.get(one.name);
    if (edit !== undefined) edit.value = values[at] === undefined ? '' : values[at];
  });

  let up = true;
  const away = () => {
    if (!up) return;
    up = false;
    overlay.remove();
    // The keyboard goes back where it was, which is the pane: a dialog that
    // closed onto nothing would leave the program looking as though it had
    // stopped listening.
    if (held !== null && held.focus !== undefined) held.focus();
    if (gone !== undefined) gone();
  };
  const apply = () => {
    const said = form.fields.map((one) => {
      const edit = fields.get(one.name);
      return edit === undefined ? '' : edit.value.trim();
    });
    away();
    done(said);
  };
  sheet.appendChild(answers(form, apply, away));

  // The overlay owns every key while it is up. Enter is the primary answer and
  // Escape is the way out, which is what a dialog means by both everywhere;
  // everything else stays in the field it was typed into rather than reaching
  // the pane's ladder underneath.
  overlay.addEventListener('keydown', (event) => {
    event.stopPropagation();
    if (event.key === 'Enter') {
      event.preventDefault();
      apply();
    } else if (event.key === 'Escape') {
      event.preventDefault();
      away();
    }
  });
  // A click outside the sheet is the same answer as Cancel, and a click inside
  // it must not reach the pane, whose own listeners would raise and refocus it.
  overlay.addEventListener('pointerdown', (event) => {
    event.stopPropagation();
    if (event.target === overlay) away();
  });
  where.appendChild(overlay);
  const first = form.fields.length ? fields.get(form.fields[0].name) : undefined;
  if (first !== undefined) {
    first.focus();
    first.select();
  }
  return away;
}

// What the form is about, and the one thing worth knowing that the fields
// themselves would not say.
function headed(form) {
  const holder = document.createElement('div');
  holder.className = 'plot-form-head';
  const title = document.createElement('div');
  title.className = 'plot-form-title';
  title.textContent = form.title;
  const subtitle = document.createElement('div');
  subtitle.className = 'plot-form-sub';
  subtitle.textContent = form.subtitle;
  holder.append(title, subtitle);
  return holder;
}

// One run of fields, laid out as the thing it describes: the columns carry the
// heads where the description gives them, each row its own name where it has
// one, and a field its own label where neither names it.
function grouped(form, group, fields) {
  const box = document.createElement('div');
  box.className = 'plot-group';
  if (group.title) {
    const title = document.createElement('div');
    title.className = 'plot-group-title';
    title.textContent = group.title;
    box.appendChild(title);
  }
  const labelled = group.rows.some((row) => row.label);
  const grid = document.createElement('div');
  grid.className = 'plot-grid';
  const wide = Math.max(...group.rows.map((row) => row.fields.length), 1);
  grid.style.gridTemplateColumns = `${labelled ? 'auto ' : ''}repeat(${wide}, auto)`;
  if (group.heads.length) {
    if (labelled) grid.appendChild(cell('plot-form-sub', ''));
    for (const head of group.heads) grid.appendChild(cell('plot-form-sub', head));
  }
  for (const row of group.rows) {
    if (labelled) grid.appendChild(cell('plot-label', row.label));
    for (const name of row.fields) {
      grid.appendChild(entry(named(form, name), fields));
    }
  }
  box.appendChild(grid);
  return box;
}

// One field, with the word in front of it where the description gives it one.
function entry(one, fields) {
  const holder = document.createElement('label');
  holder.className = 'plot-entry';
  if (one.label) {
    const label = document.createElement('span');
    label.className = 'plot-label';
    label.textContent = one.label;
    holder.appendChild(label);
  }
  const edit = document.createElement('input');
  edit.className = 'plot-field';
  edit.spellcheck = false;
  // The width is how much text is expected in it rather than a size, which is
  // what the description says it is; a bound is longer than a grid count.
  edit.style.width = `${one.width}px`;
  holder.appendChild(edit);
  fields.set(one.name, edit);
  return holder;
}

// The answers the form offers, the one it is for being the accented one.
function answers(form, apply, away) {
  const row = document.createElement('div');
  row.className = 'plot-answers';
  const offered = form.buttons.length ? form.buttons : STANDARD;
  for (const one of offered) {
    const button = document.createElement('button');
    button.className = 'plot-toggle';
    if (one.primary) button.classList.add('primary');
    button.type = 'button';
    button.textContent = one.label;
    button.addEventListener('click', one.role === CLOSE ? away : apply);
    row.appendChild(button);
  }
  return row;
}

function named(form, name) {
  return form.fields.find((one) => one.name === name) || { name, label: '', width: 90 };
}

function cell(className, text) {
  const one = document.createElement('div');
  one.className = className;
  one.textContent = text;
  return one;
}

// A run of fields along a tool row, built once from the description and shown
// only while there is something for it to be about.
//
// The same fields as a form's and a different arrangement, which is the whole
// of the difference between the two: a strip is a word, a field, another word,
// another field, in the order the description stands them in.
export function strip(where, description, edited) {
  const fields = new Map();
  const words = new Map();
  const pieces = [];
  for (const piece of description.pieces) {
    let element;
    if (piece.divider) {
      element = document.createElement('span');
      element.className = 'plot-divider';
    } else if (piece.entry) {
      element = document.createElement('input');
      element.className = 'plot-field';
      element.spellcheck = false;
      const one = named(description, piece.entry);
      element.style.width = `${one.width}px`;
      // Enter and leaving the field are the same answer, which is what a
      // toolbar field means everywhere. The keys stay in the field they were
      // typed into rather than reaching the pane's ladder.
      element.addEventListener('change', edited);
      element.addEventListener('keydown', (event) => {
        event.stopPropagation();
        if (event.key === 'Enter') edited();
      });
      fields.set(piece.entry, element);
    } else {
      element = document.createElement('span');
      element.className = 'plot-label';
      element.textContent = piece.word;
      if (piece.name) words.set(piece.name, { element, word: piece.word });
    }
    where.appendChild(element);
    pieces.push(element);
  }
  return {
    fields,
    // Whether the strip stands at all, which is what a range of nothing comes
    // to: a control about no plot is not a control.
    show(on) {
      for (const element of pieces) element.style.display = on ? '' : 'none';
    },
    // One of the strip's words rewritten as it changes what it is about, which
    // the description marks by giving the piece a name.
    rename(name, values) {
      const held = words.get(name);
      if (held === undefined) return;
      held.element.textContent = held.word.replace(
        /\{(\w+)\}/g,
        (whole, key) => (key in values ? values[key] : whole),
      );
    },
  };
}

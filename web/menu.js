// A menu under a button, of which the corner has two.
//
// The button and what it opens are one control, and this is the behaviour the
// pair shares: a click on the button opens or closes the menu, a click
// anywhere else or an Escape closes it, and no two menus stand open at once.
// What is on a menu is its own module's business - `demos.js` fills one with
// the demonstrations as they arrive, `files.js` has the page's gestures on the
// other - and neither module has to know the other's menu exists.

import { settle } from './keys.js';

// Every menu made, so that opening one can close the rest.
const menus = [];

// The document listeners are shared, and set up when the first menu is made:
// the next click anywhere closes whatever is open, and so does Esc. The
// terminal is behind all of this and gets the keystroke either way, which is
// why Esc is not swallowed.
//
// Heard on the way down rather than on the way up, because the keyboard is the
// terminal's while a menu stands open and xterm stops a key it has handled from
// going any further. A listener on the bubble would never hear the Esc that
// closes this.
let listening = false;

function listen() {
  if (listening) return;
  listening = true;
  document.addEventListener('click', () => closeAll());
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') closeAll();
  }, true);
}

function closeAll() {
  for (const menu of menus) menu.show(false);
}

// Make `button` open `panel`. `opening`, if given, is called before the panel
// is shown and may return false to keep it closed and say why itself.
//
// What comes back is the pair's own `show`, for a module that wants to close
// its menu once a choice has been made on it.
export function dropdown(term, button, panel, opening = null) {
  listen();
  const menu = {
    show(open) {
      // A menu that is already where it is asked to be is left alone, which is
      // what makes the shared listeners safe: they hear every click and every
      // Escape in the page, and a closed menu must not take the keyboard back
      // from a plot pane somebody is working in.
      if (open === !panel.hidden) return;
      panel.hidden = !open;
      button.setAttribute('aria-expanded', String(open));
      // The keyboard goes back to the program as the menu closes. A button
      // keeps focus after a click, and the next keystroke would go to it.
      if (!open && term !== null) settle(term);
    },
  };
  menus.push(menu);
  button.addEventListener('click', (event) => {
    event.stopPropagation();
    if (!panel.hidden) {
      menu.show(false);
      return;
    }
    if (opening !== null && opening() === false) return;
    closeAll();
    menu.show(true);
  });
  return menu;
}

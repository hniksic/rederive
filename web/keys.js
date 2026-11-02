// Where the keyboard goes: to the terminal, except where it is drawn on the
// screen.
//
// A desk's keyboard is there whatever has the focus, so the terminal keeps it
// and everything on the page that borrows it gives it straight back. A phone
// draws its keyboard over half the page whenever the terminal has the focus,
// so there the terminal has it only while the program is waiting for text,
// which is something only the program knows and says through `typing`.

// Whether the page is being used with a finger.
export const TOUCH = matchMedia('(pointer: coarse)');

let waiting = false;

// Whether the program is waiting for text, as it last said.
export function wanted() {
  return waiting;
}

// What the program calls when it starts or stops waiting for text: on a touch
// screen, the keyboard going up and coming down.
//
// Putting the keyboard up from here works because it follows the tap that
// asked for it closely enough for a browser to count it as that tap's.
export function typing(term) {
  return (on) => {
    waiting = on;
    if (!TOUCH.matches) return;
    if (on) term.focus();
    else term.blur();
  };
}

// The keyboard handed back to the terminal, by whatever borrowed it - or left
// down, on a touch screen with nothing to type.
export function settle(term) {
  if (!TOUCH.matches || waiting) term.focus();
}

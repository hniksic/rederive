// Where a floating pane opens.
//
// A pane is what a page has instead of a window, and what is underneath one is
// the program: the worksheet's lines and the expression list are written down
// the left of the terminal, so a pane opening at the left corner covers the very
// expression it was asked to draw. It opens against the right edge instead, a
// little in from it and near the top, leaving the algebra showing beside the
// picture of it; each pane after it steps in and down, so that two panes are two
// panes and not one on top of another.
//
// The whole of this is arithmetic on a DOM element and there is no desktop
// spelling to share: a window manager puts the desktop's windows where it likes
// and nothing under `plot/` has an opinion about it. What the two pane modules
// share is this file, so a flat pane and a solid one open the same way and step
// off each other rather than each cascading on its own.

//: How far in from the right edge and down from the top the first pane opens,
//: how far each one after it steps, and how many step before the cascade starts
//: over at the corner again.
const MARGIN_PX = 40;
const STEP_PX = 28;
const CASCADE = 6;

//: How much of a pane must stay on the page. A title bar off the top or off the
//: side is one nothing can take hold of, and a pane cannot be dragged back from
//: where it cannot be picked up - so a page too narrow or too short for the
//: cascade gets a pane at the corner rather than a pane it has lost.
const HELD_PX = 120;

//: What is left clear at the foot of the page. The program's menu and its
//: message line are the bottom rows of the terminal and are what a person reads
//: while a plot is up, so a pane with no room to stand clear of them rises off
//: them rather than sitting on them.
const FOOT_PX = 72;

//: How many panes have opened, over both kinds.
let opened = 0;

// Put one pane where the next one goes, at the size it opens at.
export function pane(element, width, height) {
  const wide = document.documentElement.clientWidth;
  const tall = document.documentElement.clientHeight;
  const step = (opened++ % CASCADE) * STEP_PX;
  element.style.left = `${Math.round(held(wide - MARGIN_PX - width - step, wide))}px`;
  element.style.top =
    `${Math.round(held(Math.min(MARGIN_PX + step, tall - FOOT_PX - height), tall))}px`;
  element.style.width = `${width}px`;
  element.style.height = `${height}px`;
}

// One coordinate, kept where the bar can still be reached: never off the near
// edge, and never so far along that there is nothing of the pane left to grab.
function held(where, room) {
  return Math.max(0, Math.min(where, Math.max(room - HELD_PX, 0)));
}

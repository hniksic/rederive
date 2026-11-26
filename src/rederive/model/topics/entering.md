# Entering expressions

## Writing expressions

```
3x, x y, 2(x + 1)     products
[1, 2, 3]             a vector
[1, 2; 3, 4]          a matrix; a semicolon ends each row
{1, 2, 3}             a set
[0, 2, ..., 10]       [0, 2, 4, 6, 8, 10]
"text"                a string
|x|                   ABS(x)
#3                    expression 3
```

Names ignore case: SIN, Sin and sin are the same function. Each letter is a variable, so
`ax` is a times x. A built-in name, and a name you have given a value or a definition, is
read whole. To have every run of letters read as one name, set
[Options Input](help:settings/numbers-input-and-display) Mode to Word.

`e` and `i` are variables like any other. The constants are written `#e` and `#i`.

Numbers are exact: `0.1` is 1/10.

## Operators and constants

| Operator | Also | Meaning |
|---|---|---|
| `+`, `-`, `/`, `^` | | sum, difference or negation, quotient, power |
| `*` | `·` | product; `x y` and `3x` multiply too |
| `!` | | `n!` is the factorial of n |
| `%` | | `u%` is u/100 |
| `±` | `"+-"`, Alt-minus | `x ± 1` is both x + 1 and x - 1 |
| `.` | `∙` | dot product of vectors, product of matrices |
| `` ` `` | | after a matrix, its transpose; after a set, its complement |
| `SUB` | `↓`, Alt-V | `v SUB i` is element i of the vector v, and `x SUB 1` a subscripted name; `m SUB SUB j` is column j of a matrix |
| `=`, `/=`, `<`, `<=`, `>`, `>=` | `≠`, `≤`, `≥` | relations |
| `NOT`, `AND`, `OR`, `XOR` | `¬`, `∧`, `∨`, `⊻` | not, and, or, exclusive or |
| `IMP`, `IFF` | `→`, `↔` | implication, equivalence |
| `:=` | `≔` | assignment and definition; see [Programming](help:programming/defining-functions-and-variables) |
| `:epsilon` | `:∈` | `x :epsilon Real` declares what x ranges over |
| `'` | | `'u` is u kept as written: `'(2 + 3)` stays `2 + 3` |

Two vectors are equal if they have the same length and equal elements, two strings if
they have the same characters, upper and lower case being different. Neither can be
compared with `<`, `<=`, `>` or `>=`. The operators on matrices and sets are listed with
[matrices](help:functions/vectors-and-matrices) and
[sets](help:functions/sets).

| Constant | Also | Meaning |
|---|---|---|
| `pi` | `π`, Alt-P | the circumference of a circle over its diameter |
| `#e` | `ê`, Alt-E | the base of natural logarithms |
| `#i` | `î`, Alt-I | the square root of -1 |
| `inf` | `∞`, Alt-0 | infinity; `-inf` is negative infinity |
| `deg`, `DEG` | `°` | one degree, pi/180: `SIN(30 deg)` |
| `euler_gamma` | | Euler's constant |
| `true`, `false` | | the truth values |
| `unit_circle` | | an unspecified point on the complex unit circle |
| `?` | | an unknown value, written in answers that have none |

Greek letters are typed by name, as in `alpha` or `Omega`, or with Alt: Alt-A alpha, Alt-B
beta, Alt-D delta, Alt-N epsilon, Alt-H theta, Alt-M mu, Alt-S sigma, Alt-T tau, Alt-F
phi, Alt-O omega, Alt-G Gamma. Alt-Q types the square root sign. The name is what the
program prints back either way.

## Line editing

The keys below apply to every line the program reads, from the Author line to a line
naming a file.

| Key | Action |
|---|---|
| F1 | display this help |
| F6 | make Left and Right walk the [parts of the highlighted expression](help:entering/highlighting-subexpressions) instead of moving along the line; press again to switch back |
| Enter | accept the line |
| Ctrl-Enter | accept it, and simplify the expression the command enters |
| Shift-Enter, Alt-Enter | start a new row of the Author line |
| Esc | abandon the command; on an open list of names, close the list |
| Tab | on a line naming a file, complete the name and list the matches |
| Ins | switch between inserting characters and typing over them |
| Ctrl-C | copy the selection, or the highlighted expression |
| Ctrl-X | cut the selection |
| Ctrl-V | paste the clipboard onto the line |

| Key | Selection |
|---|---|
| Shift-Left, Shift-Right | select one more character, one fewer |
| Shift-Home, Shift-End | select to the start of the line, to the end |

The label a command puts on its line comes up selected so that typing replaces it; Ctrl-C
and Ctrl-X leave it alone.

| Key | Deletion |
|---|---|
| Ctrl-W | delete the word before the cursor |
| Ctrl-U, Ctrl-K | delete everything before the cursor, from the cursor on |
| Ctrl-Y | put back what the last deletion took out |
| Alt-Y | after Ctrl-Y, put back an earlier deletion instead |

The status line shows Ovr while typing overwrites the character at the cursor, and Lin
while Left and Right move along the line.

## Highlighting subexpressions

These keys move the highlight, also while a line is being typed, so an expression can be
highlighted and then copied onto the line. Only the variable line of Factor and Expand
keeps the highlight where it is.

| Key | Highlight |
|---|---|
| Up, Down | the previous, the next expression |
| Ctrl-Up, Ctrl-Down | the same, also on an Author line of several rows (Alt on a Mac) |
| PgUp, PgDn | a screenful back, a screenful forward |
| Ctrl-Home, Ctrl-End | the first, the last expression; also Ctrl-PgUp, Ctrl-PgDn |
| Ctrl-Left, Ctrl-Right | scroll sideways over an expression wider than the pane |
| <, > | the same, where the Ctrl keys never reach the program |

Press F6, or set Options Input Use to Subexpression, and the sideways keys walk into the
highlighted expression instead of along the line:

| Key | Part highlighted |
|---|---|
| Right | into the expression, or on to the part beside the one highlighted |
| Left | back to the part before it |
| Down | deeper into the highlighted part |
| Up | back out to the part holding it |
| Home, End | the first, the last part at this depth; on a whole expression, the first, the last expression |

What is highlighted is what a command acts on, so a subexpression can be factored,
approximated or substituted for on its own without the expression around it being
disturbed.

A click highlights the whole expression, and each further click goes one part deeper
towards what the pointer is over. A click just outside the highlighted part goes back out
one level.

# Commands

A command works on the highlighted expression unless another is named.

## Computing

| Command | What it does |
|---|---|
| Author | enter an expression as the next numbered entry, unsimplified |
| Simplify | work an expression out, exactly unless Precision says otherwise |
| approX | work an expression out as a number, to the precision now set |
| Expand | expand an expression over the variables named, into a sum |
| Factor | factor an expression over the variables named, into a product |
| soLve | solve an expression for its unknowns |
| Build | put an expression together from ones already on the screen |
| Manage Substitute | put a value in place of a variable, or of a subexpression |
| Plot | draw the highlighted expression in a [plot window](help:plotting) |
| Calculus | build a calculus expression; Simplify works it out |
| Calculus Differentiate | derivative of any order; a negative order integrates |
| Calculus Integrate | definite or indefinite integral |
| Calculus Limit | limit at a point, from the left, right or both sides |
| Calculus Product | product over a range of an index |
| Calculus Sum | sum over a range of an index |
| Calculus Taylor | Taylor polynomial about a point, to a given degree |
| Calculus Vector | vector of values over a range of an index |
| Calculus tAble | table of an index and the values over its range |

Factor asks how far to go: Trivial, Squarefree, Rational, raDical or Complex. When
Precision is Approximate, soLve asks for the interval to search.

## Declaring

| Command | What it does |
|---|---|
| Declare | give a name a meaning |
| Declare Function | define a function of any number of variables |
| Declare Variable | give a variable a value, or a domain to simplify over |
| Declare Matrix | enter a matrix element by element |
| Declare vectoR | enter a vector element by element |

The domains are Integer, Real, Complex, vecTor, Set and Logical. An Integer or Real
variable can be limited to Positive, Negative, nonpoSitive, nonneGative or an Interval.
The same can be typed: `x :epsilon Real (0, inf)` declares x positive.

## Worksheet and windows

| Command | What it does |
|---|---|
| Jump | move the highlight to a numbered expression |
| Remove | take a block of expressions out of the worksheet |
| Unremove | put back the last block Remove took out, asking where |
| moVe | move a block of expressions elsewhere in the worksheet |
| Manage Renumber | number the expressions again from one |
| Manage Annotate | attach a note to an expression, shown on the status line |
| Help | this help |
| Window | split the screen, and move between the pieces |
| Window Split Horizontal | divide this window in two, one above the other |
| Window Split Vertical | divide it in two, one beside the other |
| Window Open | open another worksheet over this one |
| Window Flip | show the next of the worksheets opened over this one |
| Window Close | close a window |
| Window Next | make the next window active |
| Window Previous | make the previous window active |
| Window Goto | make a numbered window the active one |

| Key | Action |
|---|---|
| F1, Shift-F1 | switch to the next, the previous window; while typing a line, F1 is this help |
| F2, Shift-F2 | show the next, the previous of the worksheets opened over this window |
| Esc | abandon the question being asked; stop a running computation |

## Files and demos

| Command | What it does |
|---|---|
| Transfer | reading and writing files, and starting over |
| Transfer Load | read a worksheet, settings, data or definitions from a file |
| Transfer Merge | read a worksheet in after the expressions already there |
| Transfer Save | write the worksheet to a file, in the format you choose |
| Transfer Clear | start over, throwing away the expressions, the definitions, or both |
| Transfer Demo | run one of Derive's demonstrations, or one of your own |
| Quit | leave the program, after asking for confirmation |

Load, Merge and Load Utility read files written for Derive (.mth), and Derive for Windows
worksheets (.dfw) without their plots. The functions of Derive's own utility files need no
loading: they are [already defined](help:functions/utility-library).

Transfer Demo lists Derive's demonstrations and downloads each from the Internet Archive
when first run. A file of the same name in the working directory is used instead.

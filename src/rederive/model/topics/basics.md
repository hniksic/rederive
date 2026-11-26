# Basics

## First steps

Enter an expression with Author. It is displayed as typed and stays unevaluated until a
command works on it, most often Simplify.

```
press A, type 1/2 + 2/3, press Enter    #1 shows 1/2 + 2/3
press S, then Enter                     #2 shows 7/6
press A, type x^3 - x, press Enter      #3 shows x^3 - x
press F, then Enter at each question    #4 shows it factored
```

Each answer is added as a new numbered expression.

Simplify does whatever the expression calls for: arithmetic, canceling, derivatives,
integrals, sums. For a particular form use Factor or Expand, and for a decimal number
approX. soLve solves an equation. The Calculus menu builds a derivative, integral, limit,
sum or Taylor polynomial, which Simplify then works out.

For an integral, author `#e^(-x^2)`, press C for Calculus and I for Integrate, and enter
the limits `0` and `inf`. The integral is added as written, and Simplify gives
`SQRT(pi)/2`.

Transfer Demo runs Derive's demonstrations of arithmetic, algebra, calculus and plotting.

## Menus and the highlight

| Key | Menu |
|---|---|
| Space, Tab | move the highlight to the next option |
| Backspace, Shift-Tab | move it back |
| Enter | run the highlighted option |
| a letter | run the option that shows it as a capital: l for soLve |
| Esc | leave a submenu, or abandon a command |

Up and Down move the highlight between expressions, and Jump goes to one by its number. A
command works on the highlighted expression unless another is named: on a command's line,
`#3` refers to expression 3.

With the mouse, click a menu option to run it, or an expression to highlight it. Click
again to highlight a part of the expression, one level deeper. The wheel scrolls the
worksheet, and with Shift scrolls it sideways. Drag selects text, for Ctrl-C to copy.

## Exact and approximate

Simplify answers exactly: `SQRT(2)` stays as it is, and `0.1 + 0.2` is `3/10`. approX
gives a decimal approximation: `SQRT(2)` becomes `1.414213562`.
[Options Precision](help:settings/numbers-input-and-display) sets whether Simplify answers
exactly or approximately, and to how many digits.

## Variables and their domains

A variable that has not been declared stands for any real number, and Simplify uses an
identity only where it holds for every value the variable may take. So `SQRT(x^2)`
simplifies to `ABS(x)`, not to `x`.

[Declare Variable](help:commands/declaring) limits what a variable ranges over. Once x is
declared positive, `SQRT(x^2)` simplifies to `x`.

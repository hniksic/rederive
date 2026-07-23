# Usenet / Google Groups discussions mentioning Derive

Research notes: Google Groups' internal search engine for old Usenet
archives (sci.math.symbolic, sci.math, k12.ed.math, comp.soft-sys.math.*,
alt.folklore.computers) turned out to index/surface mostly unrelated or
much more recent threads when searched for "Derive" - the term collides
constantly with the common verb "to derive" and with unrelated software
(Clojure's `derive`, Elixir's `@derive`, etc.), which drowns out genuine
software-discussion hits. The items below are the genuine hits found.

---

## 1. sci.math.symbolic - "Mathematica sucks!" thread

Source: https://groups.google.com/g/sci.math.symbolic/c/3aO4jdeDF_s
Date of relevant post: November 27, 1996

Thread starter W. J. Vlymen posted a harsh critique of Mathematica's
complexity and unintuitive interface (asking for alternatives). Several
people replied with recommendations. The Derive-relevant reply:

> **John Feth**, Nov 27, 1996 - recommends trying "DeriveXM or Derive for
> Windows", describing them as having "a short learning curve, good
> succinct manuals, and a very simple and intuitive interface." He notes
> that you cannot create elaborate visualizations with Derive, but it
> offers a straightforward symbolic program for users who prefer
> simplicity over flashy graphics.

Other responses in the same thread recommended MathCad (Peter Somlo) and
Theorist 2 (Achim Recktenwald) as other "simpler than Mathematica"
alternatives - suggesting Derive was grouped in Usenet's collective mind
with MathCad/Theorist as the "easy, unintimidating" tier of math software,
opposite Mathematica's "powerful but baroque" reputation.

---

## 2. k12.ed.math - incidental mentions of Derive as a study/checking tool

Source: https://groups.google.com/g/k12.ed.math (search results for "Derive")

- Thread "Complex number question", **Sky Rookie**, 5/16/03: a poster
  verifies a result involving Euler's formula by checking it "in Derive,
  so I know the answer" - i.e., using Derive informally as an answer-key /
  sanity-check tool outside of any assigned coursework, years after its
  commercial peak.

This is a small but telling data point: by 2003, Derive had become
background infrastructure some people just had installed and reached for
casually, rather than a name brand thing they discussed at length.

---

## 3. Narkive-mirrored Usenet threads (fetch access was unreliable)

Narkive.com mirrors old Usenet hierarchies. Several threads directly on
topic were located via search but the narkive servers returned HTTP 503
(rate-limiting/blocking automated fetches) on repeated attempts, so only
search-engine-surfaced snippets could be captured, not the full threads.
Recorded here for completeness/follow-up:

- "Derive vs Mathematica vs Math Lab vs Maples. Which software you think
  is the best and easiest to use?"
  https://software.comp.narkive.com/LUsM5Ycp/derive-vs-mathematica-vs-math-lab-vs-maples-which-software-you-think-is-the-best-and-easiest-to-use
  Surfaced snippet: "Mathematica is described as the broader and more
  modern system with everything you need built-in and continues to
  develop at a high rate... Derive is noted as dead, and Maple is heading
  in that direction."

- "Derive vs Maple" (Spanish-language Usenet mirror)
  https://es.ciencia.matematicas.narkive.com/firtFGaN/derive-vs-maple

- "Derive vs Matlab" (Spanish-language Usenet mirror)
  https://es.ciencia.matematicas.narkive.com/on3vpR1r/derive-vs-matlab

- "Derive Software no longer available!?"
  https://alt.math.recreational.narkive.com/whPc3tXY/derive-software-no-longer-available
  This thread (also cross-posted/discussed on The Math Forum, see below)
  is where much of the "why did TI kill it" grief and confusion lives.
  Surfaced snippet: users express bafflement that "Texas Instruments would
  buy a product that had a very good reputation and kill it," and that
  after acquisition Derive "was killed off soon afterwards" and became
  "difficult to find a copy left on store shelves anywhere now."

- Related thread also appears on The Math Forum @ Drexel's mailing-list
  archive: http://mathforum.org/kb/thread.jspa?forumID=13&messageID=5807115&threadID=1593791
  ("Re: Derive Software no longer available!?") - this server returned
  HTTP 522 (origin down) on every attempt; content not retrievable during
  this research session.

---

## 4. PhysicsForums - CAS choice discussions (semi-Usenet-successor community)

Source: https://www.physicsforums.com/threads/advice-on-choosing-a-cas-mathematica-maple-derive-matlab.38102/
Date: original post by "agro", Aug 4, 2004

A student asks for advice choosing between Mathematica, Maple, Derive,
MATLAB or alternatives for calculus work. Notable reply:

> **graphic7** (Gold Member): recommends Maple over Mathematica, noting
> "Maple releases all their algorithms to the general public, but
> Mathematica does not." Suggests that for basic calculus a scientific
> calculator might suffice, and recommends free/open alternatives
> (Octave, Scilab) for numerical work, arguing "Learning how to use the
> computer for numerical calculations is all what the computer is about" -
> i.e. skepticism toward symbolic CAS generally.

No one in the visible thread reported firsthand Derive experience -
notable in itself: by 2004, in a physics-student forum, Derive wasn't
even part of the live conversation anymore, only listed as one of the
"legacy" options in the thread title.

Source: https://www.physicsforums.com/threads/open-source-mathematica-derive-like-program.150056/
Date: original post by "haki", Jan 4, 2007

> **haki** asks for a free/open-source alternative "as intuitive as
> Derive" for entering an integral and getting a simplified symbolic
> result - using Derive's interface as the implicit gold standard for ease
> of use, seven years after Derive's last independent release and the
> same year TI discontinued it outright.
>
> **HallsofIvy** pushes back that "a program like Mathematica and Derive
> require one heck of a lot of work, typically by a large crew of
> programmers" (skeptical that a free equivalent could exist).
>
> **chroot** (staff) recommends Maxima and Octave as "extremely good free
> replacements for the commercial packages." The OP later reports being
> satisfied with Maxima.

This is a good, quotable illustration of Derive's lingering reputation:
years after its commercial death, "as easy as Derive" was still the
benchmark people reached for when describing what they wanted from a CAS
interface.

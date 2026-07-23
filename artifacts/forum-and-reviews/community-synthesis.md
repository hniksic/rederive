# Derive: Community & Qualitative "Feel" Research - Synthesis

Compiled from Usenet/Google Groups, retrocomputing forums, download-site
reviews, blogs, a founder interview, magazine reviews, and math-education
literature. See the companion files in this directory for full excerpts
and source URLs:

- `usenet-groups-google-threads.md`
- `magazine-reviews.md`
- `blogs-and-firsthand-accounts.md`
- `forums-and-communities.md`
- `academic-education-papers.md`

## Research note on source availability

Genuine 1990s Usenet discussion of Derive turned out to be much harder to
surface than expected. Google Groups' internal search buries old
sci.math.symbolic/comp.soft-sys.math.* threads under a flood of unrelated
modern hits (the word "derive" collides with the common English verb and
with programming-language features like Clojure's `derive`). Several
promising narkive.com Usenet mirrors and one mathforum.org thread were
identified by title but returned server errors (503/522) on every fetch
attempt and could not be read directly - only search-engine-surfaced
paraphrases were recoverable. The clearest, best-dated primary-source
Usenet quote actually retrieved is a November 1996 sci.math.symbolic post
recommending Derive over Mathematica for its "short learning curve" and
"simple and intuitive interface." Given more time/access, the narkive and
mathforum.org threads (documented with URLs in the files above) are the
highest-value follow-up target.

---

## What people loved

1. **Simplicity and a short learning curve, positioned explicitly against
   Mathematica's intimidating complexity.** The single clearest and most
   recurring theme across every era of source, from a 1996 Usenet
   recommendation ("short learning curve, good succinct manuals, a very
   simple and intuitive interface") to a 2007 forum request for something
   "as intuitive as Derive" to a 2004 magazine review calling it ideal for
   "independent learners." Derive's whole identity, in the minds of
   people who talked about it publicly, was "the CAS that doesn't fight
   you."

2. **A tiny footprint that felt astonishing in retrospect.** Multiple
   independent sources - a 2024 retrocomputing blog post, the Scientific
   Computing World review, University of Hawaii's own department page -
   specifically call out how little memory/disk space Derive needed
   (the blog post marvels that the whole program was about 400KB), and
   frame this as a *feature*, not a limitation: it ran on hardware too
   weak for the competition, which mattered enormously for
   budget-constrained schools.

3. **Genuine reputation for reliability.** More than one source
   independently uses almost the same words: Derive had "a reputation for
   having very few bugs." In a category (symbolic computation) where
   users expect edge-case weirdness, "it just works and gives you a
   trustworthy answer" was a real differentiator.

4. **Pedagogical transparency features were praised, late in its life.**
   Derive 6's "Display Steps" (show the simplification rules applied, not
   just the final answer) and the slider-bar-driven live graph animation
   were called out by reviewers as directly answering the "students just
   get an answer without understanding it" criticism leveled at CAS tools
   generally. Academic research (Pierce & Stacey, 2001) independently
   found that using Derive in a calculus classroom pushed students toward
   *more* discussion of mathematical meaning with peers and instructors,
   not less - the opposite of the "CAS makes students lazy" fear.

5. **Longtime enthusiast loyalty that outlasted the product by well over
   a decade.** Unattributed download-site reviews from years after
   discontinuation call it "Best program I ever used" and, in 2013 ("even
   by today's standards"), "A masterpiece... incredibly powerful,
   surprisingly light and practical to use." An active preservation
   community (WinWorld forum, as recently as 2025-2026) is still
   uploading manuals and disk images and helping people get DOS-era
   versions running under DOSBox.

6. **Price and accessibility, both financial and cognitive.** Soft
   Warehouse's business model from its earliest Atari-era products
   (Algicalc, Polycalc, Calculus Demon, all $22.95 in 1982) through Derive
   itself was consistently "cheap, narrowly useful, explicitly
   student/teacher-targeted" - the opposite end of the market from
   Mathematica's famously expensive licensing.

## What people criticized

1. **Poor discoverability of advanced functionality.** The most specific,
   concrete criticism found: "most functions aren't accessible via the
   front menu" - e.g. finding something like the Gamma function required
   digging into help files instead of browsing a menu. For a product
   whose whole reputation rested on being simple and menu-driven, this was
   a real inconsistency reviewers flagged.

2. **Narrower raw mathematical power than Maple or Mathematica.**
   Consistently acknowledged, including by favorable reviewers - Derive
   was described as roughly comparable to Wolfram's stripped-down
   "Mathematica Lite" product, not to full Mathematica or Maple. Nobody
   claimed Derive competed on power; its entire pitch was elsewhere
   (ease, price, footprint).

3. **Weaker/less flashy graphics and visualization** relative to
   Mathematica in particular - noted as far back as a 1996 Usenet
   recommendation ("you cannot create elaborate visualizations with
   Derive... but...") and echoed in modern software-directory summaries
   calling its graphics "not as advanced or visually appealing" as newer
   tools.

4. **Confusion, even anger, over Texas Instruments' handling of the
   acquisition.** A recurring Usenet/forum thread topic (title alone:
   "Derive Software no longer available!?") captures users baffled that
   TI "would buy a product that had a very good reputation and kill it
   off soon afterwards," leaving it "difficult to find a copy... on store
   shelves anywhere." This abrupt, seemingly wasteful discontinuation
   (folded into TI-Nspire, June 2007) is probably the single most
   emotionally charged community memory about Derive's *end*, as opposed
   to its use.

## Common use cases people described

- **Personal answer-checking / sanity-checking**, informally, alongside
  homework - not just assigned coursework use (e.g. a 2003 k12.ed.math
  poster verifying a complex-number identity "in Derive, so I know the
  answer").
- **Undergraduate coursework**, especially early calculus/algebra -
  Derive 2.55 used personally by at least one now-blogging alumnus;
  Derive 6 installed on University of Hawaii's own teaching-lab machines
  with locally authored video tutorials and lab manuals.
- **Worksheet preparation by teachers** (explicitly praised by UK
  reviewers Evaluate and Schoolzone) - i.e. Derive was as much a
  *teacher's authoring tool* for generating classroom materials as a
  student-facing program.
- **Bridging PC and calculator work**, via Derive 6's ability to
  send/receive worksheets to and from TI-89/TI-92+/Voyage 200 - though
  "syntax differences" meant this bridge wasn't perfectly seamless.
- **The TI-92/TI-89 calculator experience** (Derive's engine, not the PC
  product, but the same underlying CAS and the same UX DNA) was
  extremely formative for a whole generation of US students, some of whom
  describe carrying the large TI-92 into math competitions for its
  "intimidation factor," and others who had it explicitly banned from
  standardized tests and from some college courses because its symbolic
  solving was seen as undermining the point of the test.

## Why people chose Derive over the competition

- **Price** - both for the software itself and implicitly (its low system
  requirements meant schools didn't need to buy new hardware to run it).
- **Ran acceptably on weak/old hardware** - explicitly called out across
  eras, from its DOS-era memory footprint to it still being praised for
  "practical to use" performance "even by today's standards" in a 2013
  review.
- **Simplicity/low intimidation factor** for students and non-specialist
  teachers, versus Mathematica's programming-language-like depth and
  Maple's own learning curve.
- **The TI calculator halo effect** - once Soft Warehouse's engine was
  inside the wildly popular TI-92/TI-89/Voyage 200 line, an entire
  generation of students experienced "Derive-grade" symbolic computation
  as a completely normal part of math class, even if they never heard the
  brand name "Derive" itself.

## Why people say they eventually stopped using it

- **Texas Instruments discontinued it outright in 2007**, folding its
  capability into TI-Nspire CAS rather than continuing the standalone
  product - widely perceived by the user community as a company killing a
  well-regarded product for reasons users didn't understand or agree
  with.
- **The broader field moved on**: free/open alternatives (Maxima, Octave,
  Scilab, and later Sage) arose specifically to fill the gap once
  proprietary CAS products (Derive included) stopped being actively
  developed or became hard to obtain; by the mid-2000s, forum posters
  were already treating Derive as the *reference point* for "intuitive"
  rather than as a live option to actually buy.
- **It was never positioned to compete on raw power**, so as
  research/professional needs grew, users graduated to Maple/Mathematica
  (for power) or to free tools (for cost/openness), leaving Derive mostly
  as a fond memory of "the one that was easy in school."

## Notable quotes worth keeping close at hand

- "a short learning curve, good succinct manuals, and a very simple and
  intuitive interface" - Usenet, sci.math.symbolic, Nov 1996
- "a reputation for having very very very few bugs" - recurring
  characterization across multiple secondary sources
- "most functions aren't accessible via the front menu" - Scientific
  Computing World, Mar/Apr 2004
- "far too good to be limited to the educational field" - Scientific
  Computing World, Mar/Apr 2004 (the review's own headline/thesis)
- "as intuitive as Derive" - PhysicsForums, Jan 2007, used as the
  aspirational bar for a free alternative
- "Best program I ever used" / "A masterpiece! Even by today's standards
  (2013) it is incredibly powerful, surprisingly light and practical to
  use!" - unattributed long-tail user reviews on software-download
  mirrors
- "why would Texas Instruments buy a product that had a very good
  reputation and kill it" - paraphrased community reaction to the 2007
  discontinuation

# Blogs, interviews, and other firsthand accounts

## Darren Goossens - "Derive calculus and algebra software" (blog post)

Source: https://darrengoossens.wordpress.com/2024/06/10/derive-calculus-and-algebra-software/
Date: June 10, 2024 (post is retrospective; describes university use much
earlier)

Darren Goossens reflects on Derive as a program he used personally during
his university years. Key points:

- Calls it "a remarkably powerful piece of kit" - and is struck, looking
  back from 2024, that the whole program occupied only about 400
  kilobytes on disk. The tiny footprint clearly still impresses a modern
  reader/author decades later.
- Describes the DOS version's functionality: integration and
  differentiation of polynomials and a "wide range of capabilities"
  reachable through the menu system, plus straightforward but genuinely
  useful plotting features.
- States he personally used version 2.55 as an undergraduate.
- Notes the commercial history matter-of-factly: distributed by
  Chartwell-Yorke before Texas Instruments bought it and ended
  development around 2007.
- Provides a practical "how to relive this" angle: instructions for
  running the DOS-era 2.x versions today via DOSBox on Linux, including
  the mount process, with screenshots of v2.60 running successfully.
- On abandonware ethics: says he feels only "measured ambivalence,"
  reasoning that "the likelihood of any party suffering financial or
  other losses through use of this old software is pretty small," while
  still seeing value in keeping access to old computing tools alive.
- Mentions the Derive user group / newsletter community is still active
  online in some form.

This is a good example of the "retrocomputing enthusiast rediscovering
Derive" genre: technical nostalgia focused on how *small and complete*
the program feels by modern standards, not on any specific classroom
memory.

---

## ANTIC: The Atari 8-bit Podcast - Interview 137, David Stoutemyer

Sources:
- https://archive.org/details/DavidStoutemyer-TheSoftWarehouse-Interview
- https://ataripodcast.libsyn.com/antic-interview-137-david-stoutemyer-the-soft-warehouse
Date recorded: January 29, 2016. Hosts: Randy Kindig, Kevin (Kay) Savetz,
Brad Arnold.

This is an audio interview (not a transcript - only metadata/description
text and an auto-generated speech-recognition subtitle track were
retrievable in this session) with David Stoutemyer, co-founder of Soft
Warehouse, the company that built muMATH and then Derive.

Recovered facts about the earliest, pre-Derive chapter of the company's
history (relevant "why did this company/product line exist and succeed"
backstory for the requirements document):

- Soft Warehouse's very first commercial products were for the Atari 8-bit
  line, distributed through the Atari Program Exchange (APX) in 1982:
  **Algicalc** ($22.95) - described in APX's own catalog copy as "a
  valuable tool for students and teachers of algebra and calculus" for
  symbolic operations; it won third prize in APX's education category
  that season. **Polycalc** ($22.95) - polynomial-focused, supporting
  fractional and negative powers of variables (broader than Algicalc's
  single-variable focus). **Calculus Demon** ($22.95) - automatic
  symbolic partial derivatives and indefinite integrals.
- This confirms Soft Warehouse's business DNA from day one: cheap
  ($20-ish), narrowly-scoped, explicitly student/teacher-targeted
  symbolic-math utilities sold through a hobbyist-computer software
  exchange - years before Derive packaged the same ethos ("affordable,
  classroom-friendly symbolic math") into a single general product for
  MS-DOS.

Also referenced (but not independently retrieved in this session): David
Stoutemyer's own written technical retrospective, "Ways to Implement
Computer Algebra Compactly: A Personal History," presented at CCA 2008
(PDF at orcca.on.ca) - this is the founder's own account of the *design
philosophy* decisions (compactness, small memory footprint, muLISP
internals) that shaped how Derive felt to end users, though the PDF
recovered in this session was a slide deck whose text did not extract
cleanly via automated tooling; a manual read of the original slides is
recommended if design-philosophy quotes are needed later.

---

## William Stein - "Mathematical Software and Me: A Very Personal Recollection"

Source: http://sagemath.blogspot.com/2009/12/mathematical-software-and-me-very.html
Date: December 2009

Not about Derive directly (Stein doesn't mention using it), but highly
relevant context for how a serious research mathematician of that
generation experienced the *broader* commercial-CAS landscape Derive
competed in, including the exact "why not just use one of the big
proprietary packages" sentiment Derive's marketing pitched against:

> On first encountering Mathematica on Windows 3.1 in 1992 (a pirated
> copy), Stein found it "frustrating, since there was no way to
> interactively change the viewpoint" for 3D plots.
>
> Of Maple, Mathematica, and MathCad collectively: "I viewed Maple,
> Mathematica, and MATHCAD as software that didn't really go beyond
> scientific calculators in any exciting way."

This is useful negative-space evidence for the requirements document: it
shows that among technically sophisticated users, the "big three"
GUI/symbolic packages (the ones Derive was constantly compared against
in reviews) were not universally beloved even by people who used them -
some serious users found them underwhelming compared to what dedicated
research CAS tools (Magma, PARI, etc.) could do, reinforcing that
Derive's own niche was explicitly "simple and non-intimidating," not
"as powerful as possible."

---

## University of Hawaii Department of Mathematics - Derive page

Source: https://math.hawaii.edu/wordpress/derive/

Short institutional page, notable mainly because Soft Warehouse (Derive's
original developer) was itself based in Honolulu, Hawaii - so this is the
"hometown university" connection. Confirms:
- Derive was implemented in muLISP (also a Soft Warehouse product).
- Discontinued June 29, 2007, superseded by TI-Nspire; final version
  Derive 6.1 for MS-Windows.
- Emphasizes the low system requirements as a key practical advantage:
  Derive "required comparably little memory," making it usable on older
  or otherwise underpowered lab computers.
- States Derive 6 was installed on the department's own teaching-lab
  computers, and that the department produced its own instructional video
  tutorials and a lab manual to support students learning the software -
  i.e. a firsthand example of exactly the kind of grassroots
  teacher-authored support material that grew up around Derive at many
  institutions.

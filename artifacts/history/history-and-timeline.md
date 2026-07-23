# Derive: History and Product Timeline

Research notes compiled for the Derive-remake requirements document. Every
non-obvious factual claim below is footnoted with a source URL. Where
sources conflict, both versions are given rather than silently picking one.
Several primary/secondary source documents were downloaded alongside this
file into `/home/hniksic/work/derive/artifacts/history/` — see the "Local
source files" section at the end for the full list.

---

## 1. Executive summary

Derive was a symbolic/numeric computer algebra system (CAS) sold from 1988
to 2007. It was written by **David R. Stoutemyer** and **Albert D. Rich**,
who had previously written the pioneering microcomputer CAS **muMATH**
(1979–1983) as **The Soft Warehouse**, a small partnership/company they
founded in Honolulu, Hawaii, in January 1979.[^rich-memoriam][^grokipedia]
Derive was a ground-up rewrite of the muMATH engine, released for MS-DOS in
1988, and became one of the best-selling educational CAS products of the
1990s, especially in Europe.[^wiki-derive][^mannheim] Texas Instruments (TI)
began collaborating with Soft Warehouse in the early-to-mid 1990s to embed
Derive's algebra engine into the TI-92 (1995) and TI-89 (1998) graphing
calculators, and acquired Soft Warehouse outright in August/September
1999.[^ticalc-acquire][^ti92-guidebook] TI continued to sell Derive for
Windows for a few more years (final release 6.10, October 2004) before
discontinuing it on 29 June 2007 in favor of the from-scratch **TI-Nspire
CAS** platform.[^wiki-derive][^grokipedia]

---

## 2. Quick-reference timeline

| Date | Event | Source |
|---|---|---|
| 1976 | Albert Rich hand-assembles a LISP interpreter for the 8080/Z-80 on a home-built IMSAI 8080 computer | [^rich-memoriam][^grokipedia] |
| 1977 | David Stoutemyer (University of Hawaii engineering professor) hires/partners with Rich; work starts on a compact LISP dialect (muLISP) and symbolic math on top of it | [^uhcl-thesis][^rich-memoriam] |
| **January 1979** | Stoutemyer and Rich found **The Soft Warehouse** as a Honolulu partnership | [^grokipedia][^uhcl-thesis] |
| 1979 | **muMATH-79** / **muLISP-79** released for 8080/Z80 CP/M and TRS-80 (TRS-DOS); licensed to Microsoft for distribution | [^wiki-mumath][^mumath-manual] |
| 1980 | muMATH-80 adds Apple II (Z80 SoftCard) support | [^wiki-mumath] |
| Nov 1980 | Gregg Williams reviews muSIMP/muMATH-79 in *BYTE* magazine, p.324 | [^byte-biblio] |
| Oct 1982 | David D. Shochat profiles muMATH in *Creative Computing* | [^wiki-mumath] |
| 1982 | Soft Warehouse sells three programs (Algicalc, Polycalc, Calculus Demon) through the Atari Program Exchange (APX), $22.95 each | [^antic-podcast] |
| 1983 | Klaus Aspetsberger and Gerhard Funk run what is claimed to be the world's first classroom experiment with a CAS (muMATH), at an Austrian high school | [^roanes-lozano] |
| **1983** | **muMATH-83** — final muMATH version, ported to MS-DOS/IBM PC (8086/8088), published by Microsoft | [^wiki-mumath][^grokipedia] |
| 1985 | Partnership incorporates as **Soft Warehouse, Inc.** | [^uhcl-thesis] |
| 1987 | Soft Warehouse ends its distribution arrangement with Microsoft, takes back full control of its products | [^uhcl-thesis] |
| **1988** | **Derive 1.0** released for MS-DOS — a menu-driven, ground-up successor to muMATH (month given as October by one source, November by another) | [^grokipedia][^mannheim][^wiki-derive] |
| 1989 | Derive 1.0 manual, 3rd edition, published (Aug 1989); Student Edition of Derive published by Addison-Wesley (1990), ISBN 0201506645 | [^amazon-manual][^archive-student] |
| Nov 1990 | **Derive 2.0** released | [^mannheim] |
| Sept 1991 | Soft Warehouse moves into new, larger, "award-winning" offices | [^uhcl-thesis] |
| Autumn 1991 | Austria's Federal Ministry of Education buys a country-wide secondary-school license for Derive; ACDCA (Austrian Center for Didactics of Computer Algebra) founded | [^mannheim][^roanes-lozano] |
| 1991 | **Derive User Group (DUG)** founded by Austrian teacher **Josef Böhm**; first issue of the *Derive Newsletter* | [^austromath][^roanes-lozano] |
| Summer 1992 | **Derive 2.5** released | [^mannheim] |
| ~1992 | ~40,000 Derive licenses worldwide in use (~10,000 in Europe, ~6,000 in Germany/Austria) | [^mannheim] |
| 1994 | 1st International Derive Conference, Plymouth, UK | [^roanes-lozano] |
| 1994 | **Derive 3.0x** released for MS-DOS/OS-2; **Derive XM** (extended-memory edition, up to 4 GB, 386/486+) also available | [^mannheim][^winworld-3x] |
| ~1993–95 | TI and Soft Warehouse begin joint development of a symbolic-algebra engine for a new handheld ("seven-year collaboration" per one account) | [^scientific-computing][^grokipedia] |
| **1995** | **TI-92** launched — first symbolic (CAS) calculator from Texas Instruments; its CAS explicitly credited to "the authors of the DERIVE program, who are with Soft Warehouse, Inc., Honolulu, HI" | [^ti92-guidebook][^wiki-ti92] |
| 1996 | Derive 3.13 (DOS) released; a Windows-hosted release ("Derive for Windows" / reported by some sources as "Derive 4", Oct 1996) also appears | [^archive-314][^grokipedia][^roanes-lozano] |
| 1996 | 2nd International Derive and TI-92 Conference, Bonn, Germany | [^roanes-lozano] |
| 1997 | Soft Warehouse celebrates its 18th anniversary in the software business | [^uhcl-thesis] |
| 1997 | International Journal of Computer Algebra in Mathematics Education (IJCAME) founded, growing out of the 1994 International Derive Journal | [^roanes-lozano] |
| 1998 | **TI-89** launched (CAS-capable, smaller/cheaper than TI-92) | [^wiki-ti89] |
| 1998 | International Derive & TI-89/92 Conference, Gettysburg, USA | [^roanes-lozano] |
| **Aug/Sept 1999** | **Texas Instruments acquires Soft Warehouse, Inc.** | [^ticalc-acquire][^grokipedia] |
| 2000 | **Derive 5.00** released — much-improved native Windows GUI; Theresa Shelby joins the team, works mainly on the graphical interface | [^roanes-lozano][^chartwellyorke] |
| 2000 | International Derive & TI-89/92 Conference, Liverpool, UK | [^roanes-lozano] |
| 2002 | Voyage 200 (successor to TI-92 Plus) released | [^wiki-ti92] |
| 2002 | International Derive & TI-89/92 Conference, Vienna, Austria | [^roanes-lozano] |
| ~2000–2003 | **Derive 6** released — Windows 2000/XP, worksheet exchange with TI-89/TI-92+/Voyage 200, step-by-step "Display Steps," slider-bar animation, mouse 3D rotation, Unicode fonts | [^scientific-computing][^chartwellyorke] |
| 2004 | International Derive & TI-89/92 Conference, Montreal, Canada | [^roanes-lozano] |
| **October 2004** | **Derive 6.10** — final release of Derive | [^wiki-derive] |
| 2004 | TI InterActive! 1.3.0.9 (a related but separate TI product) reaches its final version | [^wiki-tiinteractive] |
| 2006 | International Derive and TI-CAS Conference, Dresden, Germany (name changed again) | [^roanes-lozano] |
| **29 June 2007** | Texas Instruments **discontinues Derive**, in favor of TI-Nspire CAS | [^wiki-derive][^grokipedia] |
| 25 Sept 2007 | **TI-Nspire** launched (developed from scratch by TI; TI-Nspire CAS incorporates Derive-derived algorithms) | [^wiki-tinspire][^roanes-lozano] |
| 2008 | International Derive and TI-CAS Conference, Buffelspoort, South Africa | [^roanes-lozano] |
| 2010 | International TI-Nspire & Derive Conference, Málaga, Spain (DUG conference series continues under new names through 2016) | [^roanes-lozano] |
| 2023 | Albert D. Rich dies, 11 August 2023 | [^rich-memoriam] |
| 2024 | Josef Böhm, DUG founder, dies | [^austromath][^grokipedia] |

---

## 3. Predecessor: muMATH and muSIMP (1976–1988)

Computer algebra in the 1970s meant Macsyma, REDUCE, and similar systems
running on mainframes at a handful of universities.[^grokipedia] Albert D.
Rich — a University of Texas math graduate (BA 1971) and former U.S. Navy
nuclear-submarine officer — built an IMSAI 8080 microcomputer after his
discharge and, by the end of 1976, had hand-assembled a LISP interpreter for
the Intel 8080/Zilog Z80.[^rich-memoriam] In 1977 David R. Stoutemyer, then
an engineering professor at the University of Hawaiʻi at Mānoa (Caltech BS,
MIT MS in mechanical engineering, Stanford PhD in computer science),
recruited Rich to speed up the interpreter and add infinite-precision
arithmetic.[^rich-memoriam][^uhcl-thesis] Their collaboration produced
**muLISP** (a compact LISP dialect) and, on top of it, the **muSIMP**
programming language, in which they wrote **muMATH**, a symbolic-math
package.[^wiki-mumath]

The two formed **The Soft Warehouse** as a partnership in **January
1979**, working out of Rich's house; Stoutemyer put in $2,000 and Rich
contributed a home-built computer.[^grokipedia][^uhcl-thesis] The company's
first-year revenue was about $26,000.[^uhcl-thesis]

- **muMATH-79** (1979): ran on 8080/Z80 CP/M machines and the TRS-80
  (TRS-DOS), in as little as 48 KB of RAM; licensed to Microsoft for
  distribution as a Microsoft Consumer Products title.[^wiki-mumath][^mumath-manual]
  The archived TRS-80 manual (1980) credits the TRS-80 port to Gregory
  Whitten of Microsoft and the instruction booklet to William Barden Jr.
  and Gregory Whitten; catalog number 1208, part number 13HO8, Microsoft
  Consumer Products, Bellevue, WA.[^mumath-manual]
- **muMATH-80**: adds Apple II support (via the Z80 SoftCard).[^wiki-mumath]
- **muMATH-83**: final version; ported to MS-DOS/IBM PC (8086/8088, ~300 KB)
  as well as continuing CP/M-80 support; adds packages for limits, series
  summation, ODEs, and vector algebra/calculus.[^uhcl-thesis][^wiki-mumath]

muMATH could do algebraic simplification, trigonometric/logarithmic
expansion, equation solving, exact-rational and arbitrary-precision
arithmetic, symbolic matrix/vector operations, differentiation,
integration, limits and summation — all in well under 100 KB of memory, an
engineering feat that later academic surveys single out
explicitly.[^wiki-mumath][^kajler-soiffer] Norbert Kajler and Neil Soiffer's
1998 *Journal of Symbolic Computation* survey states plainly: "MuMath...
was the first CAS available on a personal computer. Derive..., its
successor, is a very compact and easy to use system for the IBM PC... Derive
is also the first CAS to run on a portable calculator [handheld
palmtop]."[^kajler-soiffer]

Contemporary press coverage: Gregg Williams reviewed muSIMP/muMATH-79 in
*BYTE* (November 1980, p.324), and David D. Shochat covered it in *Creative
Computing* (October 1982); Stuart Edwards wrote about a muMATH-based
automatic-unit-conversion calculator application in BYTE (December
1983).[^byte-biblio][^wiki-mumath] Also on the Soft Warehouse credit sheet
from this era: Rich and Stoutemyer's own conference paper, "Capabilities of
the muMATH-78 Computer Algebra System for the Intel-8080 Microprocessor,"
presented at EUROSAM '79.[^roanes-lozano]

Before Derive existed, Soft Warehouse also sold three smaller,
special-purpose programs through the **Atari Program Exchange (APX)**:
**Algicalc** and **Polycalc** (both $22.95, Summer 1982 catalog — Algicalc
won 3rd prize in the education category) and **Calculus Demon** ($22.95,
Fall 1982 catalog, for symbolic partial derivatives and indefinite
integrals).[^antic-podcast] These were built on the same muMATH/muLISP
technology stack running on Atari 8-bit hardware.

Remarkably early adoption: in **1983**, Austrian teachers Klaus Aspetsberger
and Gerhard Funk ran what the literature calls probably the world's first
classroom experiment with a CAS, using muMATH in an Austrian high
school.[^roanes-lozano] This Austrian connection would prove durable — see
§8 on the Derive User Group.

By the mid-1980s the IBM PC's rise made muMATH's command-line interface feel
dated, motivating Stoutemyer and Rich to design a completely new,
menu-driven, more accessible system: Derive.[^grokipedia]

---

## 4. Soft Warehouse, Inc.: the company

- The partnership incorporated as **Soft Warehouse, Inc.** in
  **1985**.[^uhcl-thesis] (Note: some AI-generated/aggregator sources state
  the 1979 date as the "founding," which is technically the *partnership*
  date; 1985 is the *incorporation* date per the 1998 UHCL master's
  thesis — the most detailed single source located for this period.)
- In **1987**, Soft Warehouse ended its Microsoft distribution deal and took
  direct control of its products going forward.[^uhcl-thesis]
- By the time it introduced Derive in 1988, the thesis notes the company
  "sold more copies of Derive in one year than it sold of muMATH in 10
  years."[^uhcl-thesis]
- Growth: from a $26,000 first year, the seven-employee operation reportedly
  grew to about **$1.6 million** in annual revenue roughly seven years
  later (i.e., around 1992).[^uhcl-thesis] (This figure is cited in the
  thesis to a magazine source abbreviated "Kamhis 5" that could not be
  independently located/verified during this research pass — flagged as
  lower-confidence.)
- **September 1991**: Soft Warehouse moves into new, "award-winning"
  offices in Honolulu.[^uhcl-thesis]
- **January 1997**: the company marks its 18th anniversary (dating itself
  to the January 1979 partnership).[^uhcl-thesis]
- A European arm, **Soft Warehouse Europe**, was based in Hagenberg,
  Austria (per the imprint on Bernhard Kutzler's 1994 book *Mathematics on
  the PC — Introduction to DERIVE*), and continued to support/sell Derive
  in Europe even after the 1999 TI acquisition — the 2003 Scientific
  Computing World review of Derive 6 explicitly says the product was "…
  supported in the USA by Texas Instruments and in Europe by Soft Warehouse
  Europe."[^roanes-lozano][^scientific-computing]
- Company products circa the mid-to-late 1990s: Derive for Windows, Derive
  for DOS, muLISP-90, muLISP XM, sold through an international dealer
  network.[^uhcl-thesis]

---

## 5. Derive on DOS: versions 1–3 and variants (1988–1996)

Multiple partially-conflicting version dates were found; the following
table merges them, noting the source for each data point.

| Version | Date | Notes / source |
|---|---|---|
| 1.0 | 1988 (Oct. per Grokipedia; Nov. per the 1990s Mannheim CAS page) | First MS-DOS release, menu-driven, ran from a single 5¼" floppy without installation, no hard disk required[^grokipedia][^mannheim][^roanes-lozano] |
| 1.0 manual, 3rd ed. | Aug 1989 | "Derive 1.0 – A Mathematical Assistant Program," Soft Warehouse, Honolulu[^amazon-manual][^grokipedia] |
| 1.62 | 1988 | Archived at archive.org (`derivecas162`)[^archive-162] |
| Student Edition | 1990 | Published by Addison-Wesley, Reading, MA; ISBN 0201506645/9780201506648; manual by David C. Arney; system needs: IBM PC/PS2, DOS 2.1+, 512 KB RAM[^archive-student] |
| 2.0 | Nov. 1990 | [^mannheim] |
| 2.01 | 1990 | Archived on WinWorld[^winworld-2x] |
| 2.5 | Summer 1992 | [^mannheim] |
| 2.60 | 1994 | Archived on WinWorld[^winworld-2x] |
| 3.05 | 1994 | Archived on WinWorld[^winworld-3x] |
| 3.13 / 3.14 | March 1996 | Archived on archive.org and WinWorld, playable via DOSBox[^archive-314][^winworld-3x] |
| Derive XM | mid-1990s | "Industrial strength," extended-memory (up to 4 GB) edition; identical menus/features to standard Derive but scales to much larger problems; requires 386/486 with 2+ MB extended memory[^xm-edition][^mannheim] |
| 4.11 / 4.13 | 1996 | DOS release; single combined 16-bit/32-bit executable that auto-detects extended memory, replacing the earlier two-executable (standard/XM) scheme; contemporary review calls it able to solve exact integrals (e.g. ∫cos(x)⁷/(7^x+1)dx from −π/2 to π/2 = 16/35) that "Mathematica ... at the time" could only approximate[^vetusware-411][^palmtoppaper] |

Requirements/footprint through this era stayed tiny by design: the
mid-1990s German university page on Derive lists Derive 3 needing only a
286 CPU and 512 KB RAM.[^mannheim] Standard graphics adapters were
supported, and — notably for a remake targeting "look and feel" — Derive
ran on early palmtop PCs including the **HP 95LX**, **HP 200LX**, and
**Poqet PC**, with a dedicated PC-Card edition for the HP
95LX.[^mannheim][^palmtoppaper] A first-hand 1990s review of the palmtop
edition survives in *Palmtop Paper* magazine (David Sargeant, "Derive: The
Mathematical Assistant") and is a good primary source for period screen
shots and workflow description; it also confirms Derive could export
expressions to BASIC, Pascal, C, and Fortran source code, and states
plainly: "Derive, originally called muMath, was marketed by Microsoft in
the early 1980's. MuMath soon returned to the parent company, SoftWarehouse
of Honolulu, Hawaii, where it was renamed Derive."[^palmtoppaper]

**Market penetration** (Germany-focused but internationally sourced,
c.1992): roughly 40,000 Derive licenses worldwide, of which about 10,000
were in Europe and 6,000 in Germany and Austria combined.[^mannheim] Austria
is the standout case: the Austrian Federal Ministry of Education bought a
**country-wide secondary-school license** starting **autumn 1991**, and
Derive became "standard tool in mathematics instruction there" — the
Austrians also created the dedicated **ACDCA** (Austrian Center for
Didactics of Computer Algebra) research center.[^mannheim][^roanes-lozano]

---

## 6. Transition to Windows: versions 4/5/6 (1996–2004)

This is the murkiest stretch of the version history, and sources genuinely
disagree:

- **One narrative** (Wikipedia's summary as fetched here, and the
  Scientific Computing World retrospective) says **Derive 5** (released
  2000) was "the first Windows version, released after Texas Instruments
  bought Soft Warehouse in August 1999."[^wiki-derive][^scientific-computing]
- **Another narrative** (Grokipedia's aggregation, and the 2018
  Roanes-Lozano/Galán-García/Solano-Macías academic paper) says **Derive 4**
  already added a Windows version alongside the continuing DOS line, in
  1996 — three years *before* the TI acquisition.[^grokipedia][^roanes-lozano]
- A mid-1990s German university CAS-survey page independently describes a
  "**Derive for Windows**" product (32-bit, for Windows 3.x/95/NT)
  co-existing with "Derive 3" and "Derive XM" for MS-DOS/OS-2 — consistent
  with a separate, perhaps unevenly-numbered, early Windows port shipping
  well before 1999.[^mannheim]

The likely reconciliation (not confirmed by a single authoritative source):
Soft Warehouse shipped an initial, less-polished Windows port around
1996 (sometimes lumped into "Derive 4"/"Derive for Windows"), while the
**post-acquisition Derive 5 (2000)** was a substantially rebuilt, more
professional native Windows GUI that some retrospectives treat as the
"real" first Windows version because it's the one people remember. Anyone
using this document to plan a remake's version-numbering scheme should
treat the DOS→Windows transition as fuzzy in the historical record rather
than pin a single date.

What is well corroborated:

- **Derive 5.00** (2000): major Windows GUI overhaul. **Theresa Shelby**
  joined the Soft Warehouse/TI team around this time and worked mainly on
  the graphical interface.[^roanes-lozano] Compatible with Windows 95/98/NT;
  very modest disk footprint (~3 MB).[^grokipedia]
- **Derive 6**: Windows 2000/XP-era release. Major new features per the
  2003 Scientific Computing World review:
  - "**Display Steps**" — shows the transformation rules applied during
    simplification (a pedagogical feature)
  - Slider-bar controls for dynamically exploring graph parameters
  - "Module Method" for pre-built teacher worksheets
  - Unicode 16-bit font support (multi-language)
  - Mouse-driven rotation of 3D plots
  - A syntax "bracket checker"
  - Customizable menus/shortcuts/toolbar icons
  - Two-pane Windows help
  - ~250 built-in functions plus 300+ more specialist functions
  - Graphics module credited to **David Parker**
  - Tighter TI-calculator integration: worksheets could be sent to/from
    the TI-89, TI-92+, and Voyage 200 (though syntax sometimes needed
    hand-editing — e.g. Derive's `SOLVE` becomes `cSolve` on the Voyage
    200)[^scientific-computing]
  - Distributed on a CD but the whole program still fit in roughly
    2 MB[^roanes-lozano]
- Official companion manuals by **Bernhard Kutzler and Vlasta Kokol-Voljc**:
  *Introduction to Derive 5* (Texas Instruments, 2000) and *Introduction to
  Derive 6* (Texas Instruments, 2003); a full **Derive 6 User Manual**
  (~500+ pages) was also produced under TI in
  2001.[^roanes-lozano][^grokipedia]
- **Derive 6.10** (October 2004): final release; mainly bug fixes and
  minor polish; starting at 6.1 the Unicode-font Windows-compatibility
  issues of 6.01 (problems on Windows 98/Me) were resolved.[^wiki-derive][^chartwellyorke]
- A related but separate TI product of this era, **TI InterActive!**
  (final version 1.3.0.9, 6 July 2004), combined graphing-calculator-style
  functionality with a document/text-editor interface and an embedded IE
  browser; Wikipedia's article does not establish a direct code
  relationship to Derive, but the two are grouped together in period CAS
  comparisons.[^wiki-tiinteractive]

Contemporary comparison, from the Scientific Computing World review of
Derive 6: mathematically "obviously narrower than the heavyweight
packages, such as Maple and Mathematica," but favorably compared to
"Wolfram's 'Mathematica Lite' CalculationCenter." As a speed demonstration,
the review notes that factoring the Mersenne number **M67** — a
computation that took the mathematician Frank Nelson Cole roughly 20 years
of "Sunday afternoons" to do by hand around 1903 — takes Derive 6 **0.281
seconds** on a Pentium-class PC.[^scientific-computing]

---

## 7. Texas Instruments: collaboration, acquisition, and calculator integration

TI's relationship with Soft Warehouse predates the 1999 acquisition by
several years. TI's own **TI-92 Guidebook** front matter states plainly:
"The TI-92 Symbolic Manipulation was jointly developed by TI and the
authors of the DERIVE program, who are with Soft Warehouse, Inc., Honolulu,
HI" (copyright notice on the guidebook spans 1995–1998, 2001, consistent
with development having started before the 1995 launch).[^ti92-guidebook]
One secondary source calls this a "seven-year collaboration" between TI and
Soft Warehouse, whose programmers "still work for TI" (written a few years
after the acquisition).[^scientific-computing]

- **TI-92** (1995): TI's first symbolic-manipulation ("CAS") graphing
  calculator; came with a CAS based on Derive, plus geometry based on
  Cabri II (Université Joseph Fourier, Grenoble); QWERTY keyboard led U.S.
  testing bodies to classify it as a "computer" rather than a "calculator,"
  which barred it from many standardized tests (e.g. AP
  exams).[^wiki-ti92][^ti92-guidebook]
- **TI-92 II** (1996), **TI-92 Plus** (1998, adds flash memory), **Voyage
  200** (2002) followed in the same product line, all Motorola
  68000-based.[^wiki-ti92]
- **TI-89** (September 1998): smaller/cheaper CAS calculator without a full
  QWERTY layout, addressing the standardized-test problem; **TI-89
  Titanium** replaced it 1 June 2004.[^wiki-ti89]
- **Full acquisition**: Texas Instruments acquired Soft Warehouse, Inc. in
  **August 1999**. The trade site ticalc.org posted a contemporaneous news
  item dated **7 August 1999**: "Texas Instruments has just recently
  acquired Soft Warehouse, Inc. The company's primary product is Derive, a
  CAS... upon which the TI-89, TI-92, and TI-92 Plus are
  based."[^ticalc-acquire] Grokipedia's aggregated account (citing a TI
  investor-relations press release this research could not directly fetch
  due to bot-protection blocking automated retrieval; noted here as
  lower-confidence pending direct verification) gives a more specific
  closing date of **23 September 1999**.[^grokipedia]
- Reaction at the time (from the ticalc.org comment thread, Aug 1999) was
  mixed/muted among calculator hobbyists — some found it "dull," others
  correctly intuited it meant deeper CAS integration into future TI
  calculators.[^ticalc-acquire]
- Rationale offered by secondary sources for the *acquisition itself*: to
  let TI "integrate [Soft Warehouse] with its graphing calculator
  range."[^scientific-computing]

---

## 8. Discontinuation (2007) and TI-Nspire

- Texas Instruments discontinued Derive as a standalone product on **29
  June 2007**.[^wiki-derive][^grokipedia]
- Replacement: **TI-Nspire**, launched **25 September 2007**, described by
  Wikipedia as developed "from the TI PLT SHH1 prototype model, the TI-92
  series... and the TI-89 series," with the CAS variant (TI-Nspire CAS)
  incorporating Derive-derived algorithms.[^wiki-tinspire][^roanes-lozano]
  Wikipedia's own TI-Nspire article, however, does not connect it
  explicitly to Derive/Soft Warehouse — that link comes from the CAS
  history literature (Roanes-Lozano et al.) rather than from TI's own
  Nspire documentation.[^wiki-tinspire][^roanes-lozano]
- Reasons for discontinuing the *PC software* specifically (as opposed to
  the underlying algorithms, which lived on) are not spelled out in any
  single authoritative TI statement located during this research pass.
  The best-supported synthesis, from Grokipedia's sourced narrative: TI
  "increasingly prioritized embedding CAS functionality into handheld
  hardware over sustaining the PC-based application, aligning with
  broader industry trends toward portable computing devices," amid
  "growing competition from integrated educational technologies and
  emerging open-source alternatives like Maxima."[^grokipedia] Take this
  as informed synthesis rather than a quoted TI rationale — no primary TI
  statement explaining the decision was found.
- Aftermath in the education market: despite discontinuation, Derive
  continued to be used in some university labs into the 2010s (e.g.
  University of Hawaiʻi's own math department ran Derive 6 in its teaching
  labs and produced instructional videos), and legacy copies remain
  available through DOSBox/emulation and archival
  sites.[^math-hawaii][^grokipedia]

---

## 9. People

- **David R. Stoutemyer** — born 17 June 1942, Washington, D.C. BS
  engineering, Caltech; MS mechanical engineering, MIT; PhD computer
  science, Stanford (1970). Joined University of Hawaiʻi at Mānoa in 1966
  (general engineering, then electrical engineering, then information &
  computer science); still an adjunct professor there as of the 2019 ACA
  conference bio. Co-founded Soft Warehouse with Rich; after the TI
  acquisition worked directly for TI before moving to
  consulting.[^uhcl-thesis][^aca2019]
- **Albert D. Rich** (Albert Rich III) — born 26 April 1949 in Altadena,
  California; raised in Nacogdoches, Texas; BA in mathematics, University
  of Texas at Austin (1971), where he developed a unification algorithm
  for propositional-calculus theorem proving with professor Laurent
  Siklóssy. Served as a commissioned U.S. Navy officer (nuclear power
  school, submarine duty aboard USS *James Monroe*) until 1976. Built the
  first muLISP interpreter later that year. Co-founded Soft Warehouse in
  1979. After Derive's acquisition by TI, devoted himself to **Rubi**, a
  public-domain rule-based indefinite-integration system
  (rulebasedintegration.org). Died **11 August 2023**.[^rich-memoriam]
- **Theresa Shelby** — joined the Derive team around 2000, primarily
  responsible for the graphical interface work in Derive
  5.[^roanes-lozano]
- **David Parker** — credited with the graphics module in Derive
  6.[^scientific-computing]
- **Bernhard Kutzler** and **Vlasta Kokol-Voljc** — authored the official
  "Introduction to Derive 5/6" books for TI and were closely associated
  with Soft Warehouse Europe (Hagenberg, Austria); Kutzler's 1994 book
  *Mathematics on the PC — Introduction to DERIVE* was published under the
  Soft Warehouse Europe imprint.[^roanes-lozano]
- **Josef Böhm** (1945–2024) — Austrian secondary-school teacher; founded
  the **International Derive User Group (DUG)** in 1991 and edited its
  newsletter continuously (four issues/year) until his death; the DUG
  archive (austromath.at/dug) survives him and remains
  online.[^austromath][^grokipedia]
- **William Barden Jr.** and **Gregory Whitten** — authored the muMATH/
  muSIMP TRS-80 instruction manual for Microsoft (1980); Whitten
  specifically implemented the TRS-80 port of muMATH-79 while at
  Microsoft.[^mumath-manual]
- **Note on "David R. Barton"**: the research brief for this document asked
  about a "David R. Barton" alongside Stoutemyer and Rich. No connection
  between anyone by that name and Soft Warehouse/Derive/muMATH was found.
  There *is* a well-documented **David Barton** in computer-algebra
  history, but he is a *different* person: a developer (with Stephen
  Bourne and John Fitch, from ~1968) of **CAMAL**, the Cambridge Algebra
  System, used for celestial-mechanics and general-relativity computation
  at Cambridge — unrelated to Soft Warehouse or
  Derive.[^camal] Treat "David R. Barton" and Derive as unconnected unless
  a future source turns up a specific link.

---

## 10. Market context and competitors

Derive existed in a CAS market that, by the early-to-mid 1990s, had several
recognizable tiers:

- **Mainframe-era pioneers**: Macsyma (MIT Project MAC, from 1968; licensed
  to Symbolics 1982; Symbolics Macsyma spun off 1992; discontinued 1999;
  survives as the open-source **Maxima**) and REDUCE (Anthony Hearn, from
  1963/1966).[^roanes-lozano]
- **PC-era heavyweights**: **Maple** (University of Waterloo /
  Waterloo Maple Inc., released 1986) and **Mathematica** (Stephen Wolfram
  / Wolfram Research, released June 1988 on Macintosh, MS-DOS version
  following within six months).[^uhcl-thesis] Both remain actively
  developed today; both were positioned as more powerful, more expensive,
  and more resource-hungry than Derive.[^grokipedia]
- **MuPAD**: a German academic CAS, free for single-user Linux but paid on
  Windows; generally considered to have a clumsier interface than
  Maple/Mathematica's notebook-style UI (later became the symbolic engine
  behind MATLAB's Symbolic Math Toolbox).
- **MathCad** (Mathsoft): more numerically/engineering oriented,
  live-document style; frequently mentioned in the same breath as Derive
  in period retrospectives as a comparison point for "easier, cheaper,
  education/engineering-flavored" CAS-adjacent tools.[^palmtoppaper]

Derive's competitive niche, repeatedly described the same way across
sources: **low hardware requirements**, **low price/education focus**, and
a **menu-driven, non-command-line interface** that lowered the barrier to
entry relative to Maple/Mathematica's more programming-language-like
front ends — at the cost of narrower mathematical scope and raw power.[^grokipedia][^kajler-soiffer]
A rigorous, systematic capability comparison across CAS packages (including
Derive, Macsyma, Maple, Mathematica, MuPAD, Axiom, REDUCE) was published as
Michael Wester's 1999 book *Computer Algebra Systems: A Practical Guide*
(28 test-problem domains); the book's own index page is archived but the
actual comparative test results were not retrieved in this pass and would
need a follow-up look at Wester's original PostScript/PDF test
files.[^wester-cas]

A 1998 University of Houston–Clear Lake master's thesis by Jackeline M.
Gascon-Brewton, *A History of the Development of Computer Algebra Systems*
— which the thesis says drew directly on internal-company documents
supplied by Soft Warehouse, Macsyma Inc., Wolfram Research, and Waterloo
Maple — is one of the single richest sources located for cross-system
comparison and Derive's specific corporate history, and has been
downloaded in full (see Local source files, below).[^uhcl-thesis]

A 2018 academic bibliometric study (Roanes-Lozano, Galán-García &
Solano-Macías, *Some reflections about the success and impact of the
computer algebra system DERIVE with a 10-year time perspective*) tracked
citation counts across Google Scholar/Scopus/MathEduc and found Derive
citations **peaked in 1990–1994**, with a **surprising revival in
2010–2014** — i.e., Derive continued generating academic-literature
interest for roughly a decade after its 2007 discontinuation, unlike a
"steep decline" one might expect.[^roanes-lozano]

---

## 11. Community, conferences, and journals

- **Derive User Group (DUG)**, founded 1991 by Josef Böhm, Austria; over
  500 members at its peak; publishes *The Derive Newsletter* (DNL) four
  times a year (title changed over time: "...the Bulletin of the Derive
  User Group" → "...+ TI92" (1996) → "...+ CAS-TI" (2003)); now a free
  online-only archive at austromath.at/dug, still
  maintained.[^austromath][^roanes-lozano] An early issue (**DNL #7**,
  September 1992, "revised reprint 2005") was downloaded for this research
  and confirms ~450 members worldwide by mid-1992, contributions arriving
  from the US, Canada, Australia, Hong Kong, and Brazil, and that David and
  Karen Stoutemyer personally attended DUG/European conferences in this
  period.[^dnl07]
- **Biennial Derive/TI conference series** (name changed repeatedly as
  scope grew): 1st International Derive Conference, Plymouth UK (1994) →
  2nd International Derive and TI-92 Conference, Bonn (1996) →
  International Derive & TI-89/92 Conference: Gettysburg USA (1998),
  Liverpool UK (2000), Vienna (2002), Montreal (2004) → International
  Derive and TI-CAS Conference: Dresden (2006), Buffelspoort South Africa
  (2008) → International TI-Nspire & Derive Conference: Málaga (2010) →
  "Conference for CAS in Education & Research" (TIME conference series):
  Tartu Estonia (2012), Krems Austria (2014), Mexico City
  (2016).[^roanes-lozano]
- **ACDCA** (Austrian Center for Didactics of Computer Algebra) ran a
  parallel annual "Summer Academy" series from 1992 (Krems 1992, 1993;
  Honolulu 1995 — note the direct nod to Soft Warehouse's home city; Særo
  Hus/Kungsbacka 1996; Gösing 1998; Portorož 2000), merging with the
  Derive conference series from 2004 onward under the "TIME" (Technology
  and its Integration in Mathematics Education) banner.[^roanes-lozano]
- **Journals**: *The International Derive Journal* (founded 1994, Plymouth
  University) → *International Journal of Computer Algebra in Mathematics
  Education* (IJCAME, 1997) → *International Journal of Technology in
  Mathematics Education* (IJTME) — indexed in Scopus and Emerging Sources
  Citation Index, still active today.[^roanes-lozano]

---

## 12. Legacy and modern preservation

- Derive is preserved on the Internet Archive in multiple forms: v1.62
  (1988), v3.14 (1996, with a scanned user manual and DOSBox playability),
  and the Addison-Wesley "Student Edition" (1990).[^archive-162][^archive-314][^archive-student]
- **WinWorld** hosts downloadable copies of Derive 2.x and 3.x with basic
  metadata.[^winworld-2x][^winworld-3x]
- Derive is called out by name (alongside Macsyma and Mathematica) as part
  of the **Smithsonian's National Museum of American History** collection
  on classic educational/mathematical software.[^scientific-computing][^nmah]
- The University of Hawaiʻi at Mānoa Department of Mathematics — Derive's
  original home turf — kept Derive 6 installed in teaching labs and
  produced its own lab manual and instructional videos well after TI's
  2007 discontinuation.[^math-hawaii]
- Enthusiast/community continuation: DOSBox-based emulation, a fan-hosted
  mirror of the Derive 6.1 online help
  (`waluigibsod.github.io/derive6.1-online-help`), and ongoing forum
  threads (e.g. on hpmuseum.org) about keeping old Derive files
  usable.[^grokipedia]

---

## 13. Open questions / lower-confidence items (flag for follow-up)

1. **Windows version numbering** — genuine source conflict on whether the
   first Windows-hosted Derive shipped as part of "Derive 4" (1996, pre-TI)
   or only arrived with "Derive 5" (2000, post-TI). See §6.
2. **Exact TI acquisition closing date** — ticalc.org's contemporaneous
   post (7 Aug 1999) says the deal had "just recently" happened;
   Grokipedia cites a specific 23 September 1999 closing date sourced to a
   TI investor-relations PDF that this research pass could not fetch (the
   URL, `investor.ti.com/static-files/3a58a383-ec35-42dd-9b54-caf257b4292c`,
   consistently reset/blocked automated HTTP requests — worth a manual
   browser check).
3. **"$1.6 million" 1992-era revenue figure and "Kamhis 5" citation** — could
   not independently verify or find the original "Kamhis" source cited by
   the 1998 UHCL thesis.
4. **Exact Derive 6 initial release year** — sources agree on "Windows
   2000/XP era" and the Oct 2004 final 6.10, but no source gives a clean
   "Derive 6.0 released on [date]" data point; Kutzler & Kokol-Voljc's
   *Introduction to Derive 6* is dated 2003, suggesting that as the most
   likely initial-release year.
5. **Original retail pricing** — despite considerable searching, no
   reliable period source for Derive's US list price (student vs.
   commercial/site license) at any point in its lifetime was found in this
   pass. Given the Austrian government bought a "country-wide" license and
   ~40,000 units were in the field by 1992, pricing was clearly
   education-accessible, but no dollar figure could be confirmed.
6. **TI press release text** — the actual original 1999 TI/Soft Warehouse
   acquisition press release was not retrievable (see item 2); the
   ticalc.org contemporaneous news post is the best located substitute for
   primary-source flavor.

---

## 14. Local source files

Downloaded into `/home/hniksic/work/derive/artifacts/history/` alongside
this document:

| File | What it is |
|---|---|
| `albert-rich-memoriam.html` | Obituary/memoriam for Albert D. Rich, ACA (Applications of Computer Algebra) "In Memoriam" page |
| `mumath-musimp-1980-manual.txt` | Full OCR text of the 1980 TRS-80 muMATH/muSIMP manual (archive.org) |
| `uhcl-thesis-gascon-brewton-1998-history-of-cas.pdf` / `.txt` | Jackeline M. Gascon-Brewton, *A History of the Development of Computer Algebra Systems* (M.S. thesis, University of Houston–Clear Lake, Dec. 1998) — covers Macsyma, Derive, Mathematica, Maple |
| `roanes-lozano-2018-derive-10-year-perspective.pdf` / `.txt` | Roanes-Lozano, Galán-García & Solano-Macías, *Some reflections about the success and impact of the computer algebra system DERIVE with a 10-year time perspective* (2018) |
| `grokipedia-derive-cas.html` / `grokipedia-derive-cas-dump.txt` | Grokipedia's "Derive (computer algebra system)" article (AI-aggregated, with sourced citations — treat claims as needing cross-check, which was done wherever feasible in this document) |
| `kajler-soiffer-cas-ui-survey.pdf` | Kajler & Soiffer, "A Survey of User Interfaces for Computer Algebra Systems," *J. Symbolic Computation* 25 (1998), 127–159 |
| `ti92-guidebook-1995-1998-2001.pdf` / `.txt` | Official TI-92 Guidebook (copyright 1995–1998, 2001), including the Derive/Soft Warehouse credit line |
| `derive-newsletter-07-dug-1992.pdf` / `.txt` | Derive Newsletter (DUG) issue #7, September 1992 (revised reprint 2005) |
| `ticalc-news-13888.html` | ticalc.org news post, "TI Acquires Soft Warehouse, Inc.," 7 August 1999, with contemporary reader comments |
| `palmtoppaper-derive-hp95lx.html` | *Palmtop Paper* magazine article, "Derive: The Mathematical Assistant," on the HP 95LX/200LX palmtop edition |
| `scientific-computing-world-derive6-review.html` | Scientific Computing World, "Derive 6: Far too good just for students" (2003 review) |
| `cas-derive-mannheim-1990s.html` | University of Mannheim CAS-info page on Derive (German, page last-modified 1997 — a genuine mid-1990s snapshot with version dates and license-count figures) |

Note: an attempt to download the original TI/Soft Warehouse acquisition
press release from `investor.ti.com` failed — the endpoint answers `HEAD`
requests normally but resets every `GET` from both `curl` and the
research agent's fetch tool, suggesting bot-detection on that investor
relations mirror. Its existence and content are known only second-hand via
Grokipedia's citation list.

---

## Footnotes / source URLs

[^rich-memoriam]: https://math.unm.edu/~aca/ACA/Memoriam/Albert_Rich/
[^grokipedia]: https://grokipedia.com/page/derive_computer_algebra_system
[^wiki-derive]: https://en.wikipedia.org/wiki/Derive_(computer_algebra_system)
[^mannheim]: https://krum.rz.uni-mannheim.de/ca-info/cas-derive.html
[^uhcl-thesis]: University of Houston–Clear Lake M.S. thesis by Jackeline M. Gascon-Brewton (1998), retrieved via https://uhcl-ir.tdl.org/server/api/core/bitstreams/1539bc96-ef3c-416a-81d9-c908750fe4e1/content
[^wiki-mumath]: https://en.wikipedia.org/wiki/MuMATH
[^mumath-manual]: https://archive.org/stream/MuMath_and_MuSimp_1980_Soft_Warehouse/MuMath_and_MuSimp_1980_Soft_Warehouse_djvu.txt
[^byte-biblio]: https://ftp.math.utah.edu/pub/tex/bib/byte1980.pdf
[^antic-podcast]: https://ataripodcast.libsyn.com/antic-interview-137-david-stoutemyer-the-soft-warehouse and https://archive.org/details/DavidStoutemyer-TheSoftWarehouse-Interview
[^roanes-lozano]: Roanes-Lozano, Galán-García & Solano-Macías (2018), retrieved via https://docta.ucm.es/bitstreams/7f18ecbd-cac6-4dd7-a824-4449b386a5b4/download
[^amazon-manual]: https://www.amazon.com/Derive-User-Manual-Version-3/dp/B001DCXFW4 (and search-result metadata referencing the Aug 1989 3rd-edition Derive 1.0 manual)
[^archive-student]: https://archive.org/details/studenteditionof0000unse
[^austromath]: https://www.austromath.at/dug/
[^winworld-3x]: https://winworldpc.com/product/derive/3x
[^scientific-computing]: https://www.scientific-computing.com/feature/derive-6-far-too-good-just-students
[^ti92-guidebook]: local file `ti92-guidebook-1995-1998-2001.pdf`, sourced from https://sites.science.oregonstate.edu/math/home/programs/undergrad/TI_Manuals/ti92Guidebook.pdf
[^wiki-ti92]: https://en.wikipedia.org/wiki/TI-92_series
[^archive-314]: https://archive.org/details/derive314cas
[^wiki-ti89]: https://en.wikipedia.org/wiki/TI-89_series
[^ticalc-acquire]: https://www.ticalc.org/archives/news/articles/1/13/13888.html
[^chartwellyorke]: https://www.chartwellyorke.com/derive.html
[^wiki-tinspire]: https://en.wikipedia.org/wiki/TI-Nspire_series
[^wiki-tiinteractive]: https://en.wikipedia.org/wiki/TI_InterActive!
[^archive-162]: https://archive.org/details/derivecas162
[^winworld-2x]: https://winworldpc.com/product/derive/2x
[^xm-edition]: search-aggregated description of Derive XM (extended memory edition), corroborated by https://www.dorn.org/uni/sls/kap05/e08_01de.htm and https://krum.rz.uni-mannheim.de/ca-info/cas-derive.html
[^vetusware-411]: https://vetusware.com/download/Derive%204.11%204.11/?id=8610
[^palmtoppaper]: local file `palmtoppaper-derive-hp95lx.html`, sourced from https://www.palmtoppaper.com/ptphtml/48/48c0000c.htm
[^kajler-soiffer]: local file `kajler-soiffer-cas-ui-survey.pdf`, Kajler & Soiffer, J. Symbolic Computation 25 (1998) 127–159, sourced from https://people.eecs.berkeley.edu/~fateman/temp/kajler-soiffer.pdf
[^aca2019]: http://aca2019.etsmtl.ca/david-stoutemyer/
[^camal]: https://en.wikipedia.org/wiki/Cambridge_Algebra_System
[^math-hawaii]: https://math.hawaii.edu/wordpress/derive/
[^nmah]: https://americanhistory.si.edu/collections/object/nmah_472821
[^wester-cas]: https://math.unm.edu/~wester/cas_review.html
[^dnl07]: local file `derive-newsletter-07-dug-1992.pdf` / `.txt`, sourced from https://www.austromath.at/dug/dnl07.pdf

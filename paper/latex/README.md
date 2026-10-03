# LaTeX build: `paper/manuscript.md` → `main.tex` → `main.pdf`

```bash
python paper/latex/build.py            # main.tex, main.pdf, arxiv/ and arxiv.zip
python paper/latex/build.py --draft    # keeps the DRAFT note as a box on page 1
python paper/latex/build.py --no-pdf   # main.tex only (no LaTeX needed)
```

`manuscript.md` is the source and is only read. After every change to it, rerun
the command; do not edit `main.tex` by hand.

## Tools used (tested 4 October 2026, macOS 27)

| Tool | Version | Install |
|---|---|---|
| pandoc | 3.12 | `brew install pandoc` (Windows: `winget install JohnMacFarlane.Pandoc`) |
| TeX Live via TinyTeX | TeX Live 2026, pdfTeX 3.141592653-2.6-1.40.29 | TinyTeX installer, then `tlmgr install microtype newunicodechar caption xurl booktabs etoolbox` |
| Python | 3.12 (standard library only) | |

Any full TeX Live or MiKTeX distribution works as well. `build.py` finds
`pdflatex` and `pandoc` on `PATH`, in `~/Library/TinyTeX` or `~/.TinyTeX`, or
through the `PDFLATEX` / `PANDOC` environment variables. The document is built
with plain `pdflatex`, which is what arXiv uses.

## What `build.py` does

- **Front matter.** It drops the leading DRAFT block quote unless `--draft` is
  given. It takes the title from the first `#` heading and the abstract from
  `## Abstract`. The authors and affiliation are fixed in `template.tex`.
- **Template.** `template.tex` uses `article`, 11pt, A4, single column, with
  `amsmath`, `booktabs`, `graphicx` and `hyperref`. Section numbers come from the
  Markdown headings themselves.
- **Tables.**
  - A `**Table N. Title.** Source: ...` paragraph becomes the caption of the
    table that follows it, typeset with `booktabs` rules.
  - Tables with lines longer than 72 characters (Table 1 and the two uncaptioned
    tables) get proportional column widths and `\small`.
  - LaTeX numbers the captioned tables 1, 2, 3; the build checks that this
    matches the numbers written in the Markdown and every "Table N" in the text.
- **Figures.**
  - The image line plus the `**Figure N.** ...` paragraph after it become one
    figure with that caption.
  - The PDF version from `paper/figures_manuscript/` is used when one exists.
  - Numbering is checked the same way as for tables.
- **Math.** `MATH_REWRITES` in `build.py` rewrites δmin and delta_min, D C D^T,
  α = …, ±, ≥, ≤, −, √, p_Holm, Spearman rho and µV into TeX math, outside code
  spans. Any other Greek or math characters fall back to `\newunicodechar`
  definitions in the template. The report lists any non-ASCII character that is
  covered by neither.
- **Inline code** (paths, identifiers) is set with `\nolinkurl` (`filters.lua`),
  so long paths break instead of running into the margin.
- **References.** These use option (a): the plain list under `## References`
  becomes `thebibliography`, sorted by first author, with no numeric labels,
  because the text cites as Author (Year).
  - The build compares the "Author (Year)" citations in the text with that list,
    in both directions.
  - `citation_report.md` shows, for option (b), which reference entries map to
    a key in `paper/literature/references.bib`. Entries that do not map are
    listed, never added.

## Report printed at the end

- Page count.
- Overfull and underfull boxes.
- Missing figures and undefined references.
- Table and figure numbering.
- Citations in the text without a reference entry, and the reverse.
- How many references are missing from `references.bib`.
- Whether `arxiv/` compiles on its own.

## arXiv bundle

`arxiv/` holds `main.tex` and `figures/*.pdf`, and `arxiv.zip` is the same
folder zipped. The build compiles `arxiv/` on its own, then removes the
auxiliary files.

No `.bbl` file is needed, because the bibliography is inline (`thebibliography`
in `main.tex`). Upload `arxiv.zip` as is.

## Open points

- **Authors and order.** The current list is "Emre Atay Tümer, Hasan Berk Berber,
  Sevgi Şengül Ayan", Antalya Bilim University. It is provisional: the order and
  final form will be settled with the advisor. Change them in `template.tex`
  (`\author{...}` and `pdfauthor`).
- **Option (b), natbib with `references.bib`.** This is not used yet: 31 of the 39
  reference entries are not in `references.bib`, which so far holds only what the
  systematic search found (see `citation_report.md`). Those entries would have to
  be added from Crossref first.

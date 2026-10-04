# LaTeX build: `paper/manuscript.md` → `main.tex` → `main.pdf`

```bash
python paper/latex/build.py            # main.tex, main.pdf, arxiv/ and arxiv.zip
python paper/latex/build.py --draft    # keeps the DRAFT note as a box on page 1
python paper/latex/build.py --no-pdf   # main.tex only (no LaTeX needed)
python paper/latex/build.py --bib plain   # reference list as written (option a)
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
- **References.** By default the build uses option (b): natbib with plainnat, from
  `paper/literature/references.bib`. All 39 entries of the reference list are in
  that file, verified against Crossref, arXiv, Europe PMC or MIT DSpace; the
  comment above each entry says how.
  - Each entry of the list under `## References` is matched to a key, by DOI and
    otherwise by first author surname and year.
  - In the text, "Author (Year)" becomes `\citet{key}`. "Author Year", written
    inside the manuscript's own parentheses, becomes `\citealp{key}` without a
    comma, so the punctuation stays as written.
  - The cited entries are written to `main.bib` for plainnat. Titles are
    brace-protected; URL, ISSN, ISBN and month are dropped; Crossref chapters
    become `incollection`. Then bibtex runs between the pdflatex runs.
  - natbib prints the author names from `references.bib`: one surname, two, or
    the first plus "et al.". Where that differs from the manuscript text (for
    example "Roy (2019)" for a paper with six authors), `citation_report.md`
    lists it, together with entries in the manuscript's list whose author list
    or year differs from `references.bib`.
  - If any entry has no key, the build falls back to option (a) and says so.
    Option (a) typesets the list as written, as `thebibliography`, without
    numbers. `--bib plain` forces it.
  - In both modes the build compares the citations in the text with the list,
    in both directions. Nothing is ever added to `references.bib` by the build.

## Report printed at the end

- Page count.
- Overfull and underfull boxes.
- Missing figures and undefined references.
- Table and figure numbering.
- Citations in the text without a reference entry, and the reverse.
- Which bibliography mode was used, bibtex warnings, and how many author names
  natbib prints differently from the text.
- Whether `arxiv/` compiles on its own.

## arXiv bundle

`arxiv/` holds `main.tex` and `figures/*.pdf`, and `arxiv.zip` is the same
folder zipped. The build compiles `arxiv/` on its own, then removes the
auxiliary files.

With natbib, `arxiv/` also holds `main.bbl`, because arXiv does not run bibtex.
In option (a) no `.bbl` is needed, because the bibliography is inline. Upload
`arxiv.zip` as is.

## Open points

- **Authors and order.** The current list is "Emre Atay Tümer, Hasan Berk Berber,
  Sevgi Şengül Ayan", Antalya Bilim University. It is provisional: the order and
  final form will be settled with the advisor. Change them in `template.tex`
  (`\author{...}` and `pdfauthor`).
- **Manuscript text vs natbib.** Five author-name differences and two reference
  list entries with incomplete author lists are listed in `citation_report.md`.
  They can be fixed in `manuscript.md`; the PDF already prints the names from
  `references.bib`.

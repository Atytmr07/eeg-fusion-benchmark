"""Build paper/latex/main.tex and main.pdf from paper/manuscript.md.

    python paper/latex/build.py            # main.tex, main.pdf, arxiv/ bundle
    python paper/latex/build.py --draft    # keep the DRAFT note at the top
    python paper/latex/build.py --no-pdf   # LaTeX only

manuscript.md is read, never written. Steps:

1. Split off the leading DRAFT block quote, the title, the author lines and the
   abstract; title and abstract go to template.tex as metadata, the authors are
   fixed in the template.
2. Rewrite the Markdown pandoc would not render as intended: "**Table N. ...**"
   paragraphs become table captions, "**Figure N.** ..." paragraphs become figure
   captions (PDF figure if one exists), wide tables get proportional column widths
   and a smaller font, and Unicode math (δmin, ±, D C D^T, ...) becomes TeX math.
3. Turn the plain reference list into a thebibliography environment (option (a)
   in the task; references.bib is only used for the report in step 5).
4. Run pandoc with template.tex and filters.lua, then pdflatex twice.
5. Report pages, LaTeX warnings, table/figure numbering, citations in the text
   without a reference entry and vice versa, and which citations could be mapped
   to keys in paper/literature/references.bib (written to citation_report.md).
6. Copy main.tex and the figures to arxiv/ and check that it compiles there.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
MANUSCRIPT = PAPER / "manuscript.md"
SUPPLEMENT = PAPER / "supplementary.md"
FIGDIR = PAPER / "figures_manuscript"
BIBFILE = PAPER / "literature" / "references.bib"
ARXIV = HERE / "arxiv"

# Unicode and ASCII math in the running text -> TeX math. Applied outside code
# spans only, in this order.
MATH_REWRITES = [
    (r"(?:δmin|delta_min)\s*(?:<=|≤)\s*(\d[\d.]*\d|\d)", r"$\\delta_{\\min} \\leq \1$"),
    (r"(?:δmin|delta_min)\s*=\s*(\d[\d.]*\d|\d)", r"$\\delta_{\\min} = \1$"),
    (r"δmin|delta_min", r"$\\delta_{\\min}$"),
    (r"\bD C D\^T\b", r"$D\\,C\\,D^{\\top}$"),
    (r"\bp_Holm\b\s*(=|<|>)\s*(\d[\d.]*\d)", r"$p_{\\mathrm{Holm}} \1 \2$"),
    (r"\bp_Holm\b", r"$p_{\\mathrm{Holm}}$"),
    (r"(\d) uV\b", "\\1\u00a0$\\\\mu$V"),  # no-break space
    (r"α\s*=\s*(\d[\d.]*\d|\d)", r"$\\alpha = \1$"),
    (r"\|z\|\s*>\s*(\d+)", r"$|z| > \1$"),
    (r"Spearman rho\s*=\s*(\d[\d.]*\d)", r"Spearman $\\rho = \1$"),
    (r"±\s*(\d[\d.]*)", r"$\\pm \1$"),
    (r"≥\s*(\d[\d.]*)", r"$\\geq \1$"),
    (r"≤\s*(\d[\d.]*)", r"$\\leq \1$"),
    (r"−(\d[\d.]*)", r"$-\1$"),
    (r"√\s*\(([^()]+)\)", r"$\\sqrt{\1}$"),
    (r"√\s*(\w+)", r"$\\sqrt{\1}$"),
]
# Characters pdflatex handles directly (T1 + utf8) or template.tex maps.
SAFE_NON_ASCII = set("\u00a0şŞšŠüÜéÉçÇğĞöÖıİäÄàáèíóúñ’‘“”–—…") | set("δαβμσ±≥≤≈×−√→")


def run(cmd, cwd=None, check=True):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, errors="replace")
    if check and p.returncode != 0:
        sys.stderr.write(p.stdout[-4000:] + p.stderr[-4000:])
        raise SystemExit(f"command failed: {' '.join(map(str, cmd))}")
    return p


def find_tool(name: str) -> str:
    env = os.environ.get(name.upper())
    if env:
        return env
    found = shutil.which(name)
    if found:
        return found
    for pattern in ("~/Library/TinyTeX/bin/*/", "~/.TinyTeX/bin/*/",
                    "/Library/TeX/texbin/", "/usr/local/texlive/*/bin/*/"):
        for d in sorted(glob.glob(os.path.expanduser(pattern))):
            if os.path.exists(os.path.join(d, name)):
                return os.path.join(d, name)
    raise SystemExit(f"{name} not found; see paper/latex/README.md")


# ---------------------------------------------------------------- splitting

def split_manuscript(text: str):
    lines = text.splitlines()
    i = 0
    draft = []
    while i < len(lines) and (lines[i].startswith(">") or (draft and not lines[i].strip())):
        if lines[i].startswith(">"):
            draft.append(lines[i][1:].lstrip())
        i += 1
        if draft and i < len(lines) and not lines[i].strip():
            i += 1
            break
    while i < len(lines) and not lines[i].startswith("# "):
        i += 1
    title = lines[i][2:].strip()
    i += 1
    rest = "\n".join(lines[i:])
    m = re.search(r"^## Abstract\s*$", rest, re.M)
    if not m:
        raise SystemExit("no '## Abstract' heading")
    after = rest[m.end():]
    n = re.search(r"^## ", after, re.M)
    abstract = after[: n.start()].strip()
    body = after[n.start():]
    r = re.search(r"^## References\s*$", body, re.M)
    refs = body[r.end():].strip() if r else ""
    body = body[: r.start()] if r else body
    return "\n".join(draft).strip(), title, abstract, body, refs


# --------------------------------------------------------- text rewriting

CODE_SPAN = re.compile(r"(`+)(.+?)\1", re.S)


def outside_code(text: str, fn) -> str:
    out, pos = [], 0
    for m in CODE_SPAN.finditer(text):
        out.append(fn(text[pos:m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(fn(text[pos:]))
    return "".join(out)


def rewrite_math(text: str) -> str:
    def fn(chunk):
        chunk = chunk.replace("$", r"\$")
        for pat, rep in MATH_REWRITES:
            chunk = re.sub(pat, rep, chunk)
        return chunk
    return outside_code(text, fn)


def paragraphs(body: str):
    return re.split(r"\n\s*\n", body)


def is_table(par: str) -> bool:
    rows = par.strip().splitlines()
    return len(rows) >= 2 and rows[0].lstrip().startswith("|") and \
        re.fullmatch(r"\s*\|[\s:|-]+\|\s*", rows[1]) is not None


def cells(row: str):
    return [c.strip() for c in row.strip().strip("|").split("|")]


TEXT_PT = 455.0                              # \linewidth of template.tex (A4, 2.5 cm)
CHAR_PT = {"small": 4.9, "footnotesize": 4.3}   # mean character width (lmodern), with margin


def widen_table(par: str) -> tuple[str, str | None]:
    r"""Proportional column widths for tables pandoc would wrap anyway, and the font
    size to set them in (None: leave the table as it is).

    Widths follow the cell lengths (square-root damped, so short label columns such as
    author names still get room), but no column is narrower than its longest word, so
    headers like "recordings" do not run into the next column. If the longest words
    alone do not fit at \small, the table is set in \footnotesize."""
    rows = par.strip().splitlines()
    if max(len(r) for r in rows) <= 72:
        return par, None
    def printed(c: str) -> str:       # measure a citation as about what natbib prints
        c = re.sub(r"`\\cite\w*\{[^}]*\}`\{=latex\}", "Author et al. (2000)", c)
        return re.sub(r"[*`]", "", c)
    data = [[printed(c) for c in cells(r)] for i, r in enumerate(rows) if i != 1]
    ncol = len(data[0])
    lens, longest = [], []
    for j in range(ncol):
        col = [d[j] for d in data if j < len(d)]
        n = [len(c) for c in col]
        lens.append(max(6, sum(n) / len(n) * 0.5 + max(n) * 0.5) ** 0.5)
        longest.append(max((len(w) for c in col for w in c.split()), default=1))
    for size in CHAR_PT:
        avail = (TEXT_PT - 12.0 * ncol) / CHAR_PT[size]       # characters, minus tabcolsep
        need = [(w + 1.5) / avail for w in longest]
        if sum(need) <= 1:
            break
    else:                             # even the longest words do not fit: share by them
        need = [x / sum(need) for x in need]
    share = [l / sum(lens) for l in lens]
    fixed: set[int] = set()
    while True:                       # raise columns below their minimum, rescale the rest
        rest = [j for j in range(ncol) if j not in fixed]
        room = 1 - sum(need[j] for j in fixed)
        tot = sum(share[j] for j in rest)
        w = [need[j] if j in fixed else share[j] * room / tot for j in range(ncol)]
        low = {j for j in rest if w[j] < need[j]}
        if not low or len(fixed | low) == ncol:
            break
        fixed |= low
    rows[1] = "|" + "|".join("-" * max(3, round(200 * x)) for x in w) + "|"
    return "\n".join(rows), size


TABLE_CAP = re.compile(r"^\*\*Table (\d+)\.\s*(.+?)\*\*\s*(.*)$", re.S)
FIG_CAP = re.compile(r"^\*\*Figure (\d+)\.\*\*\s*(.+)$", re.S)
IMAGE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$")


def figure_path(src: str) -> str:
    base = Path(src).name
    pdf = FIGDIR / (Path(base).stem + ".pdf")
    return pdf.name if pdf.exists() else base


def restructure(body: str, report: dict) -> str:
    pars = paragraphs(body)
    out = []
    i = 0
    tables, figures = [], []
    pending_caption = None
    while i < len(pars):
        p = pars[i].strip("\n")
        flat = " ".join(p.split())
        tm = TABLE_CAP.match(flat)
        if tm and i + 1 < len(pars) and is_table(pars[i + 1]):
            tables.append(int(tm.group(1)))
            title, extra = tm.group(2).strip(), tm.group(3).strip()
            pending_caption = f"**{title}**" + (f" {extra}" if extra else "")
            i += 1
            continue
        if is_table(p):
            table, size = widen_table(p)
            block = table
            if pending_caption:
                block += f"\n\n: {pending_caption}"
                pending_caption = None
            if size:
                block = f"```{{=latex}}\n\\begingroup\\{size}\n```\n\n" + block + \
                        "\n\n```{=latex}\n\\endgroup\n```"
            out.append(block)
            i += 1
            continue
        im = IMAGE.match(p.strip())
        if im:
            caption = im.group(1)
            if i + 1 < len(pars):
                fm = FIG_CAP.match(" ".join(pars[i + 1].split()))
                if fm:
                    figures.append(int(fm.group(1)))
                    caption = fm.group(2).strip()
                    i += 1
            path = figure_path(im.group(2))
            if not (FIGDIR / path).exists():
                report["missing_figures"].append(im.group(2))
            n = len(figures) if figures else 0
            out.append(f"![{caption}]({path}){{#fig:{n} width=80%}}")
            i += 1
            continue
        out.append(p)
        i += 1
    report["tables"] = tables
    report["figures"] = figures
    return "\n\n".join(out) + "\n"


# ------------------------------------------------------------- supplement

SUPP_TABLE = re.compile(r"^## Table S(\d+)\.\s*(.+)$")
SUPP_HEADER = r"""```{=latex}
\clearpage
\section*{Supplementary Material}
\setcounter{table}{0}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\theHtable}{S\arabic{table}}
\setcounter{figure}{0}
\renewcommand{\thefigure}{S\arabic{figure}}
\renewcommand{\theHfigure}{S\arabic{figure}}
```"""


def supplement_markdown(text: str, report: dict) -> str:
    """paper/supplementary.md as an appendix after the references: tables numbered S1,
    S2, ... by LaTeX, each "## Table Sk. Title" heading and the paragraph under it
    becoming the caption of the table that follows. The document title and the
    "Generated by ... do not edit by hand" note are repository notes and are dropped."""
    out, nums, pending = [SUPP_HEADER], [], None
    for par in paragraphs(text):
        p = par.strip()
        flat = " ".join(p.split())
        if not p or p.startswith("# ") or re.fullmatch(r"\*Generated by .*\*", flat):
            continue
        m = SUPP_TABLE.match(flat)
        if m:
            nums.append(int(m.group(1)))
            pending = {"title": m.group(2).strip(), "text": []}
            continue
        if is_table(p):
            table, size = widen_table(p)
            block = table
            if pending:
                title = pending["title"]
                title += "" if title[-1] in ".:?!" else "."
                block += f"\n\n: **{title}**" + (" " + " ".join(pending["text"])
                                                   if pending["text"] else "")
                pending = None
            if size:
                block = f"```{{=latex}}\n\\begingroup\\{size}\n```\n\n" + block + \
                        "\n\n```{=latex}\n\\endgroup\n```"
            out.append(block)
            continue
        if pending is not None:
            pending["text"].append(flat)
            continue
        out.append(p)
    report["supp_tables"] = nums
    return "\n\n".join(out) + "\n"


def check_supplement(report: dict, body: str):
    nums = report.get("supp_tables", [])
    issues = []
    if nums != list(range(1, len(nums) + 1)):
        issues.append(f"supplementary tables are numbered S{nums}, LaTeX numbers them "
                      f"S1..S{len(nums)}")
    flat = " ".join(CODE_SPAN.sub(" ", body).split())
    cited = set()
    for a, b in re.findall(r"\bTables?\s+S(\d+)(?:\s+(?:to|-|and)\s+S(\d+))?", flat):
        cited |= set(range(int(a), int(b or a) + 1))
    for n in sorted(cited - set(nums)):
        issues.append(f"the manuscript refers to Table S{n}, which the supplement lacks")
    report["numbering"] += issues


# --------------------------------------------------------------- references

def parse_references(refs: str):
    items = []
    for line in refs.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        entry = line[2:].strip()
        surname = entry.split(",")[0].strip()
        y = re.search(r"\((\d{4})[a-z]?\)", entry)
        items.append({"entry": entry, "surname": surname, "year": y.group(1) if y else ""})
    items.sort(key=lambda d: (strip_accents(d["surname"]).lower(), d["year"]))
    seen = {}
    for d in items:
        k = re.sub(r"[^A-Za-z]", "", strip_accents(d["surname"])) + d["year"]
        seen[k] = seen.get(k, 0) + 1
        d["key"] = k if seen[k] == 1 else f"{k}{'abcdefgh'[seen[k] - 1]}"
    return items


def bibliography_markdown(items) -> str:
    def linkify(e):
        e = re.sub(r"doi:(10\.\S+?)(?=[.,;]?(\s|$))",
                   lambda m: f"[doi:{m.group(1)}](https://doi.org/{m.group(1)})", e)
        e = re.sub(r"arXiv:(\d{4}\.\d{4,5})",
                   lambda m: f"[arXiv:{m.group(1)}](https://arxiv.org/abs/{m.group(1)})", e)
        return e
    parts = ["```{=latex}\n\\begin{thebibliography}{}\n```"]
    for d in items:
        parts.append(f"`\\bibitem{{{d['key']}}}`{{=latex}} {linkify(rewrite_math(d['entry']))}")
    parts.append("```{=latex}\n\\end{thebibliography}\n```")
    return "\n\n".join(parts) + "\n"


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


# Capital initial, including accented Latin capitals (Örnek, Šoškić, Łukasz).
UPPER = "A-Z" + "".join(c for c in map(chr, range(0xC0, 0x250)) if c.isupper())
NAME = rf"(?:(?:[Vv]an den|[Vv]an|[Vv]on|[Dd]e|Del|del|Le)\s)?[{UPPER}][\w'’-]+"
CITE = re.compile(
    rf"(?P<lead>\(|;\s|\b)(?P<first>{NAME})"
    rf"(?P<rest>\s+et\s+al\.|(?:,\s+{NAME})*,?\s+and\s+{NAME})?"
    rf",?\s+(?P<open>\()?(?P<year>(?:19|20)\d\d)[a-z]?(?P<close>\))?")
NOT_NAMES = {"Table", "Figure", "Section", "Sections", "January", "February", "March",
             "April", "May", "June", "July", "August", "September", "October",
             "November", "December", "In", "The", "Since", "From", "Until", "By",
             "Through", "PhysioNet", "Python", "Fall", "Spring", "Summer", "Winter"}


def text_citations(body: str):
    flat = " ".join(CODE_SPAN.sub(" ", body).split())
    found = {}
    for m in CITE.finditer(flat):
        first = m.group("first")
        if first.split()[-1] in NOT_NAMES:
            continue
        citation_like = m.group("rest") or (m.group("open") and m.group("close")) \
            or m.group("lead") in ("(",) or m.group("lead").startswith(";") or m.group("close")
        if not citation_like:
            continue
        key = (strip_accents(first).lower(), m.group("year"))
        found.setdefault(key, m.group(0).lstrip("(; ").strip())
    return found


def _braced(text: str, i: int) -> int:
    """Index just past the brace group opening at text[i] == '{'."""
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return j + 1
    raise ValueError("unbalanced braces in references.bib")


def parse_bib(text: str) -> dict:
    """Entries of a BibTeX file: key -> {type, fields (raw values), authors, surname,
    year, doi}. Handles both Crossref's one-line and the multi-line layout."""
    entries = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        end = _braced(text, text.index("{", m.start())) - 1
        body, fields, pos = text[m.end():end], {}, 0
        for f in re.finditer(r"(\w+)\s*=\s*", body):
            if f.start() < pos:
                continue
            k, v0 = f.group(1).lower(), f.end()
            if body[v0] == "{":
                v1 = _braced(body, v0)
                fields[k] = body[v0 + 1:v1 - 1].strip()
            else:
                v1 = v0 + re.match(r"[^,}]*", body[v0:]).end()
                fields[k] = body[v0:v1].strip()
            pos = v1
        authors = [a.strip() for a in re.split(r"\s+and\s+", fields.get("author", "")) if a.strip()]
        entries[m.group(2)] = {"type": m.group(1).lower(), "fields": fields, "authors": authors,
                               "surname": surname_of(authors[0]) if authors else "",
                               "year": fields.get("year", "")[:4],
                               "doi": fields.get("doi", "").lower()}
    return entries


PARTICLES = {"van", "von", "de", "del", "der", "den", "da", "di", "le", "la"}


def surname_of(name: str) -> str:
    """Last name of a BibTeX author ("Last, First" or "First [von] Last")."""
    if "," in name:
        return name.split(",")[0].strip()
    toks = name.split()
    i = len(toks) - 1
    while i > 0 and toks[i - 1].lower() in PARTICLES:
        i -= 1
    return " ".join(toks[i:])


def _norm(s: str) -> str:
    return strip_accents(s).lower().strip()


def map_references(refs, bib: dict):
    """Attach the references.bib key to each manuscript reference: by DOI, else by
    first author surname and year."""
    for d in refs:
        doi = re.search(r"doi:(10\.\S+?)(?=[.,;]?(\s|$))", d["entry"])
        doi = doi.group(1).lower() if doi else ""
        key = next((k for k, b in bib.items() if doi and b["doi"] == doi), None)
        if key is None:
            key = next((k for k, b in bib.items() if _norm(b["surname"]) == _norm(d["surname"])
                        and b["year"] == d["year"]), None)
        d["bibkey"] = key
    return [f"{d['surname']} ({d['year']})" for d in refs if d["bibkey"] is None]


def natbib_authors(entry: dict) -> str:
    """Author part as natbib's author-year styles print it."""
    s = [surname_of(a) for a in entry["authors"]]
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        return f"{s[0]} and {s[1]}"
    return f"{s[0]} et al."


def natbib_body(body: str, refs, bib: dict, report: dict) -> str:
    r"""Replace "Author (Year)" by \citet and "Author Year" (inside the manuscript's own
    parentheses) by \citealp, so the surrounding punctuation stays as written."""
    index = {(_norm(d["surname"]), d["year"]): d["bibkey"] for d in refs if d["bibkey"]}
    changed = {}

    def repl(m):
        first = m.group("first")
        lead, year = m.group("lead"), m.group("year")
        if first.split()[-1] in NOT_NAMES:
            return m.group(0)
        if not (m.group("rest") or m.group("close") or lead == "(" or lead.startswith(";")):
            return m.group(0)
        key = index.get((_norm(first), year))
        if key is None:
            return m.group(0)
        written = " ".join((first + (m.group("rest") or "")).split())
        printed = natbib_authors(bib[key])
        if _norm(written.replace(",", "")) != _norm(printed.replace(",", "")):
            changed[f"{written} {year}"] = f"{printed} {year}"
        if m.group("open") and m.group("close"):
            return f"{lead}`\\citet{{{key}}}`{{=latex}}"
        if m.group("open"):
            return m.group(0)
        return f"{lead}`\\citealp{{{key}}}`{{=latex}}" + (m.group("close") or "")

    out = outside_code(body, lambda chunk: CITE.sub(repl, chunk))
    report["natbib_changed"] = changed
    return out


def write_main_bib(refs, bib: dict, path: Path):
    """The cited entries, prepared for plainnat: titles brace-protected so the style
    keeps their capitals, url dropped when a DOI is printed, ISSN/publisher/month of
    journal articles dropped, arXiv identifiers moved to note."""
    out = ["% Generated by build.py from paper/literature/references.bib; do not edit."]
    for d in refs:
        e = bib[d["bibkey"]]
        f = dict(e["fields"])
        for k in ("issn", "month", "collection", "series", "abstract"):
            f.pop(k, None)
        etype = e["type"]
        if etype == "inbook" and f.get("booktitle"):
            etype = "incollection"  # Crossref's chapters; plainnat's inbook drops booktitle
        f.pop("isbn", None)
        if etype == "article":
            f.pop("publisher", None)
        f = {k: v.replace("\u2010", "-").replace("\u00a0", " ") for k, v in f.items()}
        if "pages" in f:  # plainnat recognises a range only by "-"
            f["pages"] = f["pages"].replace("\u2013", "--")
        if f.get("doi"):
            f.pop("url", None)
        if f.pop("archiveprefix", "").lower() == "arxiv" and f.get("eprint"):
            f["note"] = f"arXiv:{f.pop('eprint')}"
        if "title" in f:
            f["title"] = "{" + f["title"] + "}"
        body = ",\n".join(f"  {k} = {{{v}}}" for k, v in f.items())
        out.append(f"@{etype}{{{d['bibkey']},\n{body}\n}}\n")
    path.write_text("\n".join(out), encoding="utf-8")


def citation_report(body: str, refs, report: dict):
    cited = text_citations(body)
    ref_index = {(strip_accents(d["surname"]).lower(), d["year"]): d for d in refs}
    report["cited_not_in_refs"] = sorted(v for k, v in cited.items() if k not in ref_index)
    cited_keys = set(cited)
    report["refs_not_cited"] = sorted(
        f"{d['surname']} ({d['year']})" for k, d in ref_index.items() if k not in cited_keys)

    bib = parse_bib(BIBFILE.read_text(encoding="utf-8")) if BIBFILE.exists() else {}
    report["bib"] = bib
    report["bib_unmapped"] = map_references(refs, bib)
    lines = ["# Citation report (generated by build.py)", "",
             "Each entry of the reference list in manuscript.md is matched to a key in",
             "`paper/literature/references.bib`, by DOI and otherwise by first author",
             "surname and year. When every entry matches, the PDF uses natbib with those",
             "entries (option b); otherwise build.py falls back to typesetting the list",
             "as written (option a). Nothing is added to references.bib by the build.", "",
             "| Reference | Cited in text | references.bib key |", "|---|---|---|"]
    for d in refs:
        incited = (strip_accents(d["surname"]).lower(), d["year"]) in cited_keys
        lines.append(f"| {d['surname']} ({d['year']}) | {'yes' if incited else '**no**'} | "
                     f"{d['bibkey'] or '**not in references.bib**'} |")
    lines += ["", f"Mapped: {len(refs) - len(report['bib_unmapped'])} of {len(refs)}.", ""]
    if report["cited_not_in_refs"]:
        lines += ["Cited in the text but missing from the reference list:", ""] + \
                 [f"- {c}" for c in report["cited_not_in_refs"]] + [""]
    report["_report_lines"] = lines


def finish_citation_report(refs, report: dict):
    """Append what natbib changes relative to the manuscript and write the file."""
    lines = report.pop("_report_lines")
    changed = report.get("natbib_changed") or {}
    if changed:
        lines += ["## Author names natbib prints differently from the manuscript text", "",
                  "natbib prints one surname, two surnames, or the first surname plus",
                  "\"et al.\", from the author list in references.bib.", "",
                  "| Manuscript text | natbib prints |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in sorted(changed.items())] + [""]
    diffs = []
    bib = report["bib"]
    for d in refs:
        if not d["bibkey"]:
            continue
        e = bib[d["bibkey"]]
        if e["year"] != d["year"]:
            diffs.append(f"| {d['surname']} ({d['year']}) | year in references.bib: {e['year']} |")
        n = len(e["authors"])
        listed = d["entry"].split("(")[0]
        if n > 1 and "et al" not in listed and listed.count(",") < 2 * n - 2:
            diffs.append(f"| {d['surname']} ({d['year']}) | {n} authors in references.bib, "
                         f"reference list gives: {listed.strip()} |")
    if diffs:
        lines += ["## Reference list entries that differ from references.bib", "",
                  "| Reference | Difference |", "|---|---|"] + diffs + [""]
    (HERE / "citation_report.md").write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------- checks

def check_numbering(report: dict, body: str):
    issues = []
    for kind in ("tables", "figures"):
        nums = report[kind]
        if nums != list(range(1, len(nums) + 1)):
            issues.append(f"{kind} captions are numbered {nums}, LaTeX numbers them 1..{len(nums)}")
    flat = CODE_SPAN.sub(" ", body)
    for word, kind in (("Table", "tables"), ("Figure", "figures")):
        for n in sorted({int(x) for x in re.findall(rf"\b{word}s? (\d+)", flat)}):
            if n not in report[kind]:
                issues.append(f"text refers to {word} {n}, which has no caption")
    report["numbering"] = issues


def unknown_chars(text: str):
    return sorted({c for c in text if ord(c) > 127 and c not in SAFE_NON_ASCII})


def latex_warnings(log: str):
    w = {"overfull": [], "underfull": 0, "missing": [], "undefined": [], "other": []}
    for m in re.finditer(r"Overfull \\hbox \(([\d.]+)pt too wide\)[^\n]*lines? (\d+)", log):
        w["overfull"].append((float(m.group(1)), int(m.group(2))))
    w["underfull"] = len(re.findall(r"Underfull \\hbox", log))
    w["missing"] = re.findall(r"File `([^']+)' not found", log)
    w["undefined"] = re.findall(r"(?:Reference|Citation) `([^']+)' .*undefined", log)
    w["other"] = sorted(set(re.findall(r"LaTeX Warning: ([^\n]+)", log))
                        - {x for x in re.findall(r"LaTeX Warning: ([^\n]+)", log)
                           if "Rerun" in x or "undefined" in x})
    return w


def pdflatex(tex: Path, runs: int = 2):
    exe = find_tool("pdflatex")
    for _ in range(runs):
        p = run([exe, "-interaction=nonstopmode", "-halt-on-error", tex.name],
                cwd=tex.parent, check=False)
        if p.returncode != 0:
            sys.stderr.write(p.stdout[-3000:])
            raise SystemExit(f"pdflatex failed on {tex}")
    log = tex.with_suffix(".log").read_text(encoding="latin-1")
    pages = re.search(r"Output written on .*?\((\d+) pages?", log)
    return int(pages.group(1)) if pages else 0, log


def compile_tex(tex: Path, use_bibtex: bool, report: dict):
    """pdflatex twice, or pdflatex, bibtex, pdflatex twice."""
    if not use_bibtex:
        return pdflatex(tex)
    pdflatex(tex, runs=1)
    p = run([find_tool("bibtex"), tex.stem], cwd=tex.parent, check=False)
    blg = tex.with_suffix(".blg")
    text = blg.read_text(encoding="latin-1") if blg.exists() else p.stdout
    report["bibtex_warnings"] = re.findall(r"^Warning--(.+)$", text, re.M)
    if p.returncode > 1:  # 1 means warnings only
        sys.stderr.write(p.stdout[-3000:])
        raise SystemExit("bibtex failed")
    return pdflatex(tex)


# --------------------------------------------------------------------- main

def build(draft: bool, make_pdf: bool, bibmode: str = "natbib", supplement: bool = True):
    text = MANUSCRIPT.read_text(encoding="utf-8")
    draftnote, title, abstract, body, refs_md = split_manuscript(text)
    report = {"missing_figures": []}
    supp = SUPPLEMENT.read_text(encoding="utf-8") if supplement and SUPPLEMENT.exists() else ""
    report["supplement"] = bool(supp)
    SEP = "\n\n<!-- supplement -->\n\n"

    refs = parse_references(refs_md)
    citation_report(body + SEP + supp, refs, report)
    if bibmode == "natbib" and report["bib_unmapped"]:
        print("natbib needs every reference in references.bib; missing: "
              + ", ".join(report["bib_unmapped"]) + ". Falling back to option (a).")
        bibmode = "plain"
    report["bibmode"] = bibmode
    if bibmode == "natbib":
        body, supp = natbib_body(body + SEP + supp, refs, report["bib"], report).split(SEP)
        write_main_bib(refs, report["bib"], HERE / "main.bib")
        bibliography = "```{=latex}\n\\bibliographystyle{plainnat}\n\\bibliography{main}\n```\n"
    else:
        bibliography = bibliography_markdown(refs)
    finish_citation_report(refs, report)
    body = restructure(body, report)
    check_numbering(report, body)
    body = rewrite_math(body)
    appendix = ""
    if supp:
        appendix = rewrite_math(supplement_markdown(supp, report))
        check_supplement(report, body)
    report["unknown_chars"] = unknown_chars(body + appendix + abstract + title)

    def yaml_block(s):
        return "|\n" + "\n".join("  " + l for l in s.splitlines())
    meta = ["---", f"title: {yaml_block(rewrite_math(title))}",
            f"abstract: {yaml_block(rewrite_math(abstract))}"]
    if draft and draftnote:
        meta.append(f"draftnote: {yaml_block(rewrite_math(draftnote))}")
    if bibmode == "natbib":
        meta.append("natbib: true")
    meta.append("---")
    md = "\n".join(meta) + "\n\n" + body + "\n\n" + bibliography + "\n\n" + appendix

    src = HERE / "main.md"
    src.write_text(md, encoding="utf-8")
    run([find_tool("pandoc"), str(src), "-f", "markdown-implicit_figures+implicit_figures",
         "-t", "latex", "--template", str(HERE / "template.tex"),
         "--lua-filter", str(HERE / "filters.lua"), "--shift-heading-level-by=-1",
         "--wrap=preserve", "-o", str(HERE / "main.tex")])
    src.unlink()
    if bibmode != "natbib":
        for stale in ("main.bib", "main.bbl", "main.blg"):
            (HERE / stale).unlink(missing_ok=True)

    pages, warnings = 0, None
    if make_pdf:
        pages, log = compile_tex(HERE / "main.tex", bibmode == "natbib", report)
        warnings = latex_warnings(log)
        bundle_arxiv(report)

    print_report(report, pages, warnings, len(refs))


def bundle_arxiv(report: dict):
    if ARXIV.exists():
        shutil.rmtree(ARXIV)
    (ARXIV / "figures").mkdir(parents=True)
    tex = (HERE / "main.tex").read_text(encoding="utf-8")
    shutil.copy(HERE / "main.tex", ARXIV / "main.tex")
    if report.get("bibmode") == "natbib":
        shutil.copy(HERE / "main.bbl", ARXIV / "main.bbl")  # arXiv does not run bibtex
    for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        shutil.copy(FIGDIR / name, ARXIV / "figures" / name)
    pages, log = pdflatex(ARXIV / "main.tex")
    report["arxiv_pages"] = pages
    report["arxiv_missing"] = re.findall(r"File `([^']+)' not found", log)
    for f in ARXIV.iterdir():
        if f.suffix in (".aux", ".log", ".out", ".pdf", ".blg"):
            f.unlink()
    zip_base = HERE / "arxiv"
    shutil.make_archive(str(zip_base), "zip", ARXIV)


def print_report(r: dict, pages: int, w, nrefs: int):
    print("\n=== build report ===")
    print(f"main.tex: {HERE / 'main.tex'}")
    if pages:
        print(f"main.pdf: {pages} pages")
    print(f"tables captioned: {r['tables']}, figures captioned: {r['figures']}")
    print("supplement: " + (f"appended after the references, tables "
                            f"{', '.join(f'S{n}' for n in r.get('supp_tables', []))}"
                            if r.get("supplement") else "not included"))
    print("numbering: " + ("ok" if not r["numbering"] else "; ".join(r["numbering"])))
    print("missing figures: " + (", ".join(r["missing_figures"]) or "none"))
    if r["unknown_chars"]:
        print("non-ASCII characters not mapped (check rendering): " + " ".join(r["unknown_chars"]))
    if w:
        over = sorted(w["overfull"], reverse=True)
        print(f"overfull hboxes: {len(over)}" +
              (f" (largest {over[0][0]:.1f}pt at main.tex line {over[0][1]}; "
               f"> 5pt: {sum(1 for o in over if o[0] > 5)})" if over else ""))
        print(f"underfull hboxes: {w['underfull']}")
        print("files not found: " + (", ".join(w["missing"]) or "none"))
        print("undefined references: " + (", ".join(w["undefined"]) or "none"))
        for o in w["other"]:
            print(f"LaTeX warning: {o}")
    print(f"references: {nrefs}")
    print("cited in text, not in reference list: " + (", ".join(r["cited_not_in_refs"]) or "none"))
    print("in reference list, not found cited in text: " + (", ".join(r["refs_not_cited"]) or "none"))
    if r["bibmode"] == "natbib":
        print("bibliography: natbib + plainnat from references.bib (option b); "
              f"bibtex warnings: {len(r.get('bibtex_warnings', []))}")
        for w_ in r.get("bibtex_warnings", []):
            print(f"  bibtex: {w_}")
        if r.get("natbib_changed"):
            print(f"author names printed differently from the text: {len(r['natbib_changed'])} "
                  "(see citation_report.md)")
    else:
        print(f"bibliography: reference list as written (option a); not in references.bib: "
              f"{len(r['bib_unmapped'])} (see citation_report.md)")
    if "arxiv_pages" in r:
        print(f"arxiv/: compiles on its own, {r['arxiv_pages']} pages; missing files: "
              + (", ".join(r["arxiv_missing"]) or "none") + "; zipped to arxiv.zip")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--draft", action="store_true", help="include the DRAFT note")
    ap.add_argument("--no-pdf", action="store_true", help="write main.tex only")
    ap.add_argument("--bib", choices=["natbib", "plain"], default="natbib",
                    help="natbib from references.bib (option b, default) or the "
                         "manuscript's list as written (option a)")
    ap.add_argument("--no-supplementary", action="store_true",
                    help="leave out paper/supplementary.md (appended by default)")
    a = ap.parse_args()
    build(a.draft, not a.no_pdf, a.bib, not a.no_supplementary)


if __name__ == "__main__":
    main()

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


def widen_table(par: str) -> tuple[str, bool]:
    """Proportional column widths for tables pandoc would wrap anyway."""
    rows = par.strip().splitlines()
    if max(len(r) for r in rows) <= 72:
        return par, False
    data = [cells(r) for i, r in enumerate(rows) if i != 1]
    ncol = len(data[0])
    lens = []
    for j in range(ncol):
        col = [len(re.sub(r"[*`]", "", d[j])) for d in data if j < len(d)]
        # square root damps the long prose columns so short label columns
        # (author names) still get enough room to avoid one-word lines
        lens.append(max(6, sum(col) / len(col) * 0.5 + max(col) * 0.5) ** 0.5)
    total = sum(lens)
    sep = "|" + "|".join("-" * max(3, round(60 * l / total)) for l in lens) + "|"
    rows[1] = sep
    return "\n".join(rows), True


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
            table, wide = widen_table(p)
            block = table
            if pending_caption:
                block += f"\n\n: {pending_caption}"
                pending_caption = None
            if wide:
                block = "```{=latex}\n\\begingroup\\small\n```\n\n" + block + \
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


NAME = r"(?:(?:[Vv]an den|[Vv]an|[Vv]on|[Dd]e|Del|del|Le)\s)?[A-Z][\w'’-]+"
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


def bib_keys():
    if not BIBFILE.exists():
        return {}
    text = BIBFILE.read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(r"@\w+\{([^,]+),(.*?)(?=\n@|\Z)", text, re.S):
        key, rest = m.group(1).strip(), m.group(2)
        a = re.search(r"author\s*=\s*\{(.+?)\}\s*,\s*\w+\s*=", rest, re.S)
        y = re.search(r"year\s*=\s*\{?(\d{4})", rest)
        doi = re.search(r"DOI\s*=\s*\{([^}]+)\}", rest, re.I)
        surname = a.group(1).split(" and ")[0].split(",")[0].strip() if a else ""
        out[key] = {"surname": surname, "year": y.group(1) if y else "",
                    "doi": doi.group(1).lower() if doi else ""}
    return out


def citation_report(body: str, refs, report: dict):
    cited = text_citations(body)
    ref_index = {(strip_accents(d["surname"]).lower(), d["year"]): d for d in refs}
    report["cited_not_in_refs"] = sorted(v for k, v in cited.items() if k not in ref_index)
    cited_keys = set(cited)
    report["refs_not_cited"] = sorted(
        f"{d['surname']} ({d['year']})" for k, d in ref_index.items() if k not in cited_keys)

    bib = bib_keys()
    lines = ["# Citation report (generated by build.py)", "",
             "Option (a) is used for the PDF: the reference list in manuscript.md is",
             "typeset as `thebibliography`. This report shows how far option (b), natbib",
             "with `paper/literature/references.bib`, would get: each reference entry is",
             "matched to a BibTeX key by DOI, then by first author surname and year.",
             "Nothing is added to references.bib here; missing entries are listed only.", "",
             "| Reference | Cited in text | references.bib key |", "|---|---|---|"]
    unmapped = []
    for d in refs:
        doi = re.search(r"doi:(10\.\S+?)(?=[.,;]?(\s|$))", d["entry"])
        doi = doi.group(1).lower() if doi else ""
        key = next((k for k, b in bib.items() if doi and b["doi"] == doi), None)
        if key is None:
            key = next((k for k, b in bib.items()
                        if strip_accents(b["surname"]).lower() == strip_accents(d["surname"]).lower()
                        and b["year"] == d["year"]), None)
        incited = (strip_accents(d["surname"]).lower(), d["year"]) in cited_keys
        lines.append(f"| {d['surname']} ({d['year']}) | {'yes' if incited else '**no**'} | "
                     f"{key or '**not in references.bib**'} |")
        if key is None:
            unmapped.append(f"{d['surname']} ({d['year']})")
    lines += ["", f"Mapped: {len(refs) - len(unmapped)} of {len(refs)}; "
              f"not in references.bib: {len(unmapped)}.", ""]
    if report["cited_not_in_refs"]:
        lines += ["Cited in the text but missing from the reference list:", ""] + \
                 [f"- {c}" for c in report["cited_not_in_refs"]] + [""]
    (HERE / "citation_report.md").write_text("\n".join(lines), encoding="utf-8")
    report["bib_unmapped"] = unmapped


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


# --------------------------------------------------------------------- main

def build(draft: bool, make_pdf: bool):
    text = MANUSCRIPT.read_text(encoding="utf-8")
    draftnote, title, abstract, body, refs_md = split_manuscript(text)
    report = {"missing_figures": []}

    refs = parse_references(refs_md)
    citation_report(body, refs, report)
    body = restructure(body, report)
    check_numbering(report, body)
    body = rewrite_math(body)
    report["unknown_chars"] = unknown_chars(body + abstract + title)

    def yaml_block(s):
        return "|\n" + "\n".join("  " + l for l in s.splitlines())
    meta = ["---", f"title: {yaml_block(rewrite_math(title))}",
            f"abstract: {yaml_block(rewrite_math(abstract))}"]
    if draft and draftnote:
        meta.append(f"draftnote: {yaml_block(rewrite_math(draftnote))}")
    meta.append("---")
    md = "\n".join(meta) + "\n\n" + body + "\n\n" + bibliography_markdown(refs)

    src = HERE / "main.md"
    src.write_text(md, encoding="utf-8")
    run([find_tool("pandoc"), str(src), "-f", "markdown-implicit_figures+implicit_figures",
         "-t", "latex", "--template", str(HERE / "template.tex"),
         "--lua-filter", str(HERE / "filters.lua"), "--shift-heading-level-by=-1",
         "--wrap=preserve", "-o", str(HERE / "main.tex")])
    src.unlink()

    pages, warnings = 0, None
    if make_pdf:
        pages, log = pdflatex(HERE / "main.tex")
        warnings = latex_warnings(log)
        bundle_arxiv(report)

    print_report(report, pages, warnings, len(refs))


def bundle_arxiv(report: dict):
    if ARXIV.exists():
        shutil.rmtree(ARXIV)
    (ARXIV / "figures").mkdir(parents=True)
    tex = (HERE / "main.tex").read_text(encoding="utf-8")
    shutil.copy(HERE / "main.tex", ARXIV / "main.tex")
    for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        shutil.copy(FIGDIR / name, ARXIV / "figures" / name)
    pages, log = pdflatex(ARXIV / "main.tex")
    report["arxiv_pages"] = pages
    report["arxiv_missing"] = re.findall(r"File `([^']+)' not found", log)
    for f in ARXIV.iterdir():
        if f.suffix in (".aux", ".log", ".out", ".pdf"):
            f.unlink()
    zip_base = HERE / "arxiv"
    shutil.make_archive(str(zip_base), "zip", ARXIV)


def print_report(r: dict, pages: int, w, nrefs: int):
    print("\n=== build report ===")
    print(f"main.tex: {HERE / 'main.tex'}")
    if pages:
        print(f"main.pdf: {pages} pages")
    print(f"tables captioned: {r['tables']}, figures captioned: {r['figures']}")
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
    print(f"option (b): references not in references.bib: {len(r['bib_unmapped'])} "
          f"(see citation_report.md)")
    if "arxiv_pages" in r:
        print(f"arxiv/: compiles on its own, {r['arxiv_pages']} pages; missing files: "
              + (", ".join(r["arxiv_missing"]) or "none") + "; zipped to arxiv.zip")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--draft", action="store_true", help="include the DRAFT note")
    ap.add_argument("--no-pdf", action="store_true", help="write main.tex only")
    a = ap.parse_args()
    build(a.draft, not a.no_pdf)


if __name__ == "__main__":
    main()

"""Systematic search bookkeeping: merge, deduplicate, verify, sample, PRISMA.

Protocol: docs/14_SISTEMATIK_TARAMA_PROTOKOLU.md. Results: docs/15_LITERATUR_SONUCLARI.md.

Database exports live in paper/literature/exports/ and are named
<database>_<part>_<YYYY-MM-DD>.<ext>, part A = the protocol's main search, B1-B4 = the
complementary searches. Recognised formats:

    scopus_*.csv     Scopus CSV export (citation information + abstract)
    wos_*.txt        Web of Science "Tab delimited file" (Full Record)
    wos_*.ris, *.ris RIS
    ieee_*.csv       IEEE Xplore CSV export
    *.bib            BibTeX
    arxiv_*.xml      arXiv API Atom response

Licensing: Scopus, Web of Science and IEEE exports carry publisher abstracts, which
their terms do not allow to redistribute in a public repository. The full exports
therefore stay local (exports/.gitignore); `strip` writes <export>_meta.csv with the
citation fields only, and those are committed. merge reads the full export when it is
present and the _meta file otherwise. records.csv keeps abstracts only where arXiv
provides them (arXiv metadata is CC0); all abstracts go to abstracts_<part>.csv, which
stays local (.gitignore).

Commands (run from the repository root):

    python paper/literature/screen.py strip              # full exports -> *_meta.csv
    python paper/literature/screen.py merge --part A     # -> records.csv (keeps decisions)
    python paper/literature/screen.py verify --part A    # Crossref / arXiv check of every id
    python paper/literature/screen.py sample --part A    # random subset for the second pass
    python paper/literature/screen.py agreement --part A # agreement of the two passes
    python paper/literature/screen.py prisma             # counts + prisma.png

Deduplication: records with the same DOI are one record; records without a matching DOI
are merged when their normalised titles (lower case, letters and digits only) are equal.
An arXiv preprint and its published version are therefore merged when the arXiv entry
names the DOI or the titles agree. Every record keeps the list of databases it came from.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPORTS = HERE / "exports"
SCREEN_COLS = ["stage1", "stage1_reason", "stage1_note"]
RECORD_COLS = ["record_id", "sources", "title", "authors", "year", "venue", "doi",
               "arxiv_id", "doc_type", "abstract"]
VERIFY_COLS = ["verified", "verify_note"]
ATOM = {"a": "http://www.w3.org/2005/Atom", "o": "http://a9.com/-/spec/opensearch/1.1/",
        "x": "http://arxiv.org/schemas/atom"}
UA = "eeg-fusion-benchmark literature screen (mailto:hasanberkberber@hotmail.com)"

csv.field_size_limit(sys.maxsize)


def records_path(part: str) -> Path:
    return HERE / ("records.csv" if part == "A" else f"records_{part}.csv")


# --- Parsing -------------------------------------------------------------------------------

def clean(s) -> str:
    return " ".join(str(s or "").split())


def norm_doi(d: str) -> str:
    d = clean(d).lower()
    d = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", d)
    return d.rstrip(".")


def norm_title(t: str) -> str:
    t = unicodedata.normalize("NFKD", t or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", t.lower())


def parse_scopus_csv(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            out.append({"title": r.get("Title"), "authors": r.get("Authors"),
                        "year": r.get("Year"), "venue": r.get("Source title"),
                        "doi": r.get("DOI"), "doc_type": r.get("Document Type"),
                        "abstract": r.get("Abstract")})
    return out


def parse_ieee_csv(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            out.append({"title": r.get("Document Title"), "authors": r.get("Authors"),
                        "year": r.get("Publication Year"),
                        "venue": r.get("Publication Title"), "doi": r.get("DOI"),
                        "doc_type": r.get("Document Identifier"),
                        "abstract": r.get("Abstract")})
    return out


def parse_wos_txt(path: Path) -> list[dict]:
    """Web of Science tab-delimited export (UTF-8 or UTF-16)."""
    raw = path.read_bytes()
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else \
        raw.decode("utf-8-sig")
    out = []
    for r in csv.DictReader(text.splitlines(), delimiter="\t"):
        out.append({"title": r.get("TI"), "authors": r.get("AU"), "year": r.get("PY"),
                    "venue": r.get("SO"), "doi": r.get("DI"), "doc_type": r.get("DT"),
                    "abstract": r.get("AB")})
    return out


def parse_ris(path: Path) -> list[dict]:
    out, cur = [], None
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        m = re.match(r"^([A-Z][A-Z0-9])  - ?(.*)$", line)
        if not m:
            continue
        tag, val = m.group(1), m.group(2).strip()
        if tag == "TY":
            cur = {"doc_type": val, "authors": []}
        elif cur is None:
            continue
        elif tag == "ER":
            cur["authors"] = "; ".join(cur["authors"])
            out.append(cur)
            cur = None
        elif tag in ("AU", "A1"):
            cur["authors"].append(val)
        elif tag in ("TI", "T1") and "title" not in cur:
            cur["title"] = val
        elif tag in ("PY", "Y1") and "year" not in cur:
            cur["year"] = val[:4]
        elif tag in ("T2", "JO", "JF", "SO") and "venue" not in cur:
            cur["venue"] = val
        elif tag == "DO":
            cur["doi"] = val
        elif tag in ("AB", "N2") and "abstract" not in cur:
            cur["abstract"] = val
    return out


def parse_bibtex(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    out = []
    for m in re.finditer(r"@(\w+)\s*\{[^,]*,(.*?)\n\}", text, re.S):
        body = m.group(2)
        f = {k.lower(): v for k, v in re.findall(
            r"(\w+)\s*=\s*[\{\"](.*?)[\}\"]\s*,?\s*\n", body, re.S)}
        out.append({"title": f.get("title", "").replace("{", "").replace("}", ""),
                    "authors": f.get("author", "").replace(" and ", "; "),
                    "year": f.get("year"), "venue": f.get("journal") or f.get("booktitle"),
                    "doi": f.get("doi"), "doc_type": m.group(1),
                    "abstract": f.get("abstract")})
    return out


def parse_arxiv(path: Path) -> list[dict]:
    root = ET.parse(path).getroot()
    out = []
    for e in root.findall("a:entry", ATOM):
        aid = e.find("a:id", ATOM).text.split("/abs/")[-1]
        doi = e.find("x:doi", ATOM)
        jref = e.find("x:journal_ref", ATOM)
        out.append({"title": e.find("a:title", ATOM).text,
                    "authors": "; ".join(a.find("a:name", ATOM).text
                                         for a in e.findall("a:author", ATOM)),
                    "year": e.find("a:published", ATOM).text[:4],
                    "venue": "arXiv (preprint)" + (f"; {clean(jref.text)}" if jref is not None else ""),
                    "doi": doi.text if doi is not None else "",
                    "arxiv_id": re.sub(r"v\d+$", "", aid), "doc_type": "preprint",
                    "abstract": e.find("a:summary", ATOM).text})
    return out


def parse_file(path: Path) -> list[dict]:
    name = path.name.lower()
    if path.stem.endswith("_meta"):
        return parse_meta(path)
    if name.endswith(".xml") and name.startswith("arxiv"):
        return parse_arxiv(path)
    if name.endswith(".csv") and name.startswith("scopus"):
        return parse_scopus_csv(path)
    if name.endswith(".csv") and name.startswith("ieee"):
        return parse_ieee_csv(path)
    if name.endswith(".txt") and name.startswith("wos"):
        return parse_wos_txt(path)
    if name.endswith(".ris"):
        return parse_ris(path)
    if name.endswith(".bib"):
        return parse_bibtex(path)
    raise ValueError(f"unknown export format: {path.name}")


LICENSED = ("scopus", "wos", "ieee")
META_FIELDS = ["title", "authors", "year", "venue", "doi", "doc_type"]


def export_files(part: str) -> list[Path]:
    """Export files of a part; a _meta file is used only when its full export is absent."""
    files = sorted(p for p in EXPORTS.iterdir()
                   if re.match(rf"^[a-z]+_{part}(-[a-z]+)?_\d{{4}}-\d{{2}}-\d{{2}}", p.name, re.I))
    full = {p.name for p in files if not p.stem.endswith("_meta")}
    return [p for p in files if not p.stem.endswith("_meta")
            or not any(f.startswith(p.stem[:-5] + ".") for f in full)]


def strip() -> None:
    """Write <export>_meta.csv (citation fields, no abstracts) for every licensed export."""
    for p in sorted(EXPORTS.iterdir()):
        db = p.name.split("_")[0].lower()
        if db not in LICENSED or p.stem.endswith("_meta") or p.name.startswith("."):
            continue
        rows = [{k: clean(r.get(k)) for k in META_FIELDS} for r in parse_file(p)]
        out = p.with_name(p.stem + "_meta.csv")
        write_csv(out, rows, META_FIELDS)
        print(f"{p.name} -> {out.name} ({len(rows)} records, abstracts removed)")


def parse_meta(path: Path) -> list[dict]:
    return read_csv(path)


# --- Merge and deduplicate ----------------------------------------------------------------------

def merge(part: str) -> list[dict]:
    """Merge all exports of a part into records_<part>.csv; screening decisions already
    in the file are kept (matched by DOI, arXiv id or normalised title)."""
    raw = []
    per_file = {}
    for f in export_files(part):
        db = f.name.split("_")[0].lower()
        rs = parse_file(f)
        per_file[f.name] = len(rs)
        for r in rs:
            r = {k: clean(v) for k, v in r.items()}
            r["doi"] = norm_doi(r.get("doi", ""))
            r["source"] = db
            raw.append(r)

    groups: list[dict] = []
    by_doi, by_title, by_arxiv = {}, {}, {}
    for r in raw:
        nt = norm_title(r.get("title", ""))
        g = (by_doi.get(r["doi"]) if r["doi"] else None) or \
            (by_arxiv.get(r.get("arxiv_id")) if r.get("arxiv_id") else None) or \
            (by_title.get(nt) if nt else None)
        if g is None:
            g = {"sources": set(), "items": []}
            groups.append(g)
        g["sources"].add(r["source"])
        g["items"].append(r)
        if r["doi"]:
            by_doi[r["doi"]] = g
        if r.get("arxiv_id"):
            by_arxiv[r["arxiv_id"]] = g
        if nt:
            by_title[nt] = g

    order = ["scopus", "wos", "ieee", "arxiv"]
    records = []
    for g in groups:
        items = sorted(g["items"], key=lambda r: order.index(r["source"])
                       if r["source"] in order else 9)
        best = {}
        for k in RECORD_COLS:
            best[k] = next((it.get(k) for it in items if it.get(k)), "")
        # public file: only arXiv's (CC0) abstract; every abstract goes to the local file
        best["_abstract_any"] = best["abstract"]
        best["abstract"] = next((it.get("abstract") for it in items
                                 if it["source"] == "arxiv" and it.get("abstract")), "")
        best["sources"] = ";".join(s for s in order if s in g["sources"]) + \
            "".join(f";{s}" for s in sorted(g["sources"]) if s not in order)
        records.append(best)
    records.sort(key=lambda r: (-int(r["year"] or 0), norm_title(r["title"])))

    old = {}
    path = records_path(part)
    if path.exists():
        for r in read_csv(path):
            for key in (r.get("doi"), r.get("arxiv_id"), norm_title(r.get("title", ""))):
                if key:
                    old[key] = r
    for i, r in enumerate(records, 1):
        r["record_id"] = f"{part}{i:04d}"
        prev = next((old[k] for k in (r["doi"], r.get("arxiv_id"), norm_title(r["title"]))
                     if k and k in old), None)
        for c in SCREEN_COLS + VERIFY_COLS:
            r[c] = prev.get(c, "") if prev else ""
    write_csv(path, records, RECORD_COLS + VERIFY_COLS + SCREEN_COLS)
    write_csv(HERE / f"abstracts_{part}.csv",
              [{"record_id": r["record_id"], "abstract": r["_abstract_any"]} for r in records],
              ["record_id", "abstract"])

    # unique records per database (a database searched with two strings, as arXiv in
    # part A, is counted once per record), the PRISMA "identified" numbers
    per_db = {}
    for g in groups:
        for db in g["sources"]:
            per_db[db] = per_db.get(db, 0) + 1
    n_raw = len(raw)
    print(f"part {part}: {n_raw} records from {len(per_file)} exports -> "
          f"{len(records)} after deduplication")
    for f, n in per_file.items():
        print(f"  {f}: {n}")
    by_src = {}
    for r in records:
        by_src[r["sources"]] = by_src.get(r["sources"], 0) + 1
    print("  unique records by source combination:",
          ", ".join(f"{k} {v}" for k, v in sorted(by_src.items(), key=lambda x: -x[1])))
    (HERE / f"merge_log_{part}.json").write_text(json.dumps(
        {"exports": per_file, "per_database_unique": per_db, "records_raw": n_raw,
         "records_unique": len(records)},
        indent=2))
    return records


# --- Verification -----------------------------------------------------------------------------------

def _get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def crossref(doi: str) -> dict | None:
    try:
        return _get_json("https://api.crossref.org/works/" + urllib.parse.quote(doi))["message"]
    except Exception:
        return None


def arxiv_meta(aid: str) -> dict | None:
    url = "https://export.arxiv.org/api/query?id_list=" + urllib.parse.quote(aid)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as r:
            root = ET.fromstring(r.read())
        e = root.find("a:entry", ATOM)
        if e is None or e.find("a:title", ATOM) is None:
            return None
        return {"title": clean(e.find("a:title", ATOM).text),
                "year": e.find("a:published", ATOM).text[:4],
                "authors": [a.find("a:name", ATOM).text for a in e.findall("a:author", ATOM)]}
    except Exception:
        return None


def _surname_match(listed: str, found: list[str]) -> bool:
    if not listed or not found:
        return True
    first = re.split(r"[;,]", listed)[0].strip().split()
    found_s = " ".join(found).lower()
    return any(len(w) > 2 and w.lower().strip(".") in found_s for w in first)


def check_record(r: dict) -> tuple[str, str]:
    """('yes'|'no'|'', note): title, first author, year and venue against Crossref (by
    DOI) or arXiv (by id)."""
    if r.get("doi"):
        m = crossref(r["doi"])
        if m is None:
            return "no", "DOI not found in Crossref"
        title = clean((m.get("title") or [""])[0])
        year = str((m.get("issued", {}).get("date-parts") or [[None]])[0][0] or "")
        venue = clean((m.get("container-title") or [""])[0])
        authors = [f"{a.get('given', '')} {a.get('family', '')}" for a in m.get("author", [])]
        notes = []
        if norm_title(title) != norm_title(r["title"]):
            notes.append(f"title differs: Crossref '{title[:80]}'")
        if r.get("year") and year and abs(int(r["year"]) - int(year)) > 1:
            notes.append(f"year {r['year']} vs Crossref {year}")
        if not _surname_match(r.get("authors", ""), authors):
            notes.append("first author not in Crossref author list")
        if venue and r.get("venue") and not r["venue"].startswith("arXiv") and \
                norm_title(venue)[:12] != norm_title(r["venue"])[:12]:
            notes.append(f"venue '{r['venue'][:40]}' vs Crossref '{venue[:40]}'")
        return ("yes" if not notes else "check"), "; ".join(notes) or f"Crossref: {venue} {year}"
    if r.get("arxiv_id"):
        m = arxiv_meta(r["arxiv_id"])
        if m is None:
            return "no", "arXiv id not found"
        ok = norm_title(m["title"]) == norm_title(r["title"])
        return ("yes" if ok else "check"), "arXiv" + ("" if ok else f": title '{m['title'][:80]}'")
    return "", "no DOI or arXiv id"


def verify(part: str, only_missing: bool = True) -> None:
    path = records_path(part)
    rows = read_csv(path)
    todo = [r for r in rows if not (only_missing and r.get("verified"))]
    print(f"verifying {len(todo)} of {len(rows)} records")
    for i, r in enumerate(todo, 1):
        r["verified"], r["verify_note"] = check_record(r)
        if i % 25 == 0:
            print(f"  {i}/{len(todo)}")
            write_csv(path, rows, list(rows[0].keys()))
        time.sleep(0.2 if r.get("doi") else 3.0)        # arXiv asks for 3 s between calls
    write_csv(path, rows, list(rows[0].keys()))
    c = {}
    for r in rows:
        c[r["verified"] or "none"] = c.get(r["verified"] or "none", 0) + 1
    print("verification:", c)


# --- Second pass ----------------------------------------------------------------------------------

def load_abstracts(part: str) -> dict[str, str]:
    p = HERE / f"abstracts_{part}.csv"
    return {r["record_id"]: r["abstract"] for r in read_csv(p)} if p.exists() else {}


def sample(part: str, frac: float = 0.2, seed: int = 20261003) -> Path:
    """Random subset for the independent second pass, without the first-pass decisions.
    The file holds abstracts and stays local (.gitignore)."""
    import random
    rows = read_csv(records_path(part))
    ab = load_abstracts(part)
    for r in rows:
        r["abstract"] = ab.get(r["record_id"]) or r["abstract"]
    rng = random.Random(seed)
    k = max(1, round(frac * len(rows)))
    sub = sorted(rng.sample(rows, k), key=lambda r: r["record_id"])
    out = HERE / f"second_pass_{part}.csv"
    write_csv(out, [{c: r[c] for c in ("record_id", "title", "abstract", "year", "venue")}
                    | {"stage1_second": "", "stage1_second_reason": ""} for r in sub],
              ["record_id", "title", "abstract", "year", "venue", "stage1_second",
               "stage1_second_reason"])
    print(f"{k} of {len(rows)} records ({100 * k / len(rows):.1f} %), seed {seed} -> {out}")
    return out


def agreement(part: str) -> dict:
    first = {r["record_id"]: r for r in read_csv(records_path(part))}
    second = read_csv(HERE / f"second_pass_{part}.csv")
    pairs = [(first[r["record_id"]]["stage1"], r["stage1_second"]) for r in second
             if r["stage1_second"] and first[r["record_id"]]["stage1"]]
    labels = sorted({a for p in pairs for a in p})
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pe = sum((sum(a == l for a, _ in pairs) / n) * (sum(b == l for _, b in pairs) / n)
             for l in labels)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    # include versus not (exclude); "unclear" counts as include, as both go to full text
    bin_pairs = [(a != "exclude", b != "exclude") for a, b in pairs]
    pob = sum(a == b for a, b in bin_pairs) / n
    disagree = [(r["record_id"], first[r["record_id"]]["stage1"], r["stage1_second"])
                for r in second if r["stage1_second"] != first[r["record_id"]]["stage1"]]
    res = {"n": n, "agreement": po, "kappa": kappa, "agreement_to_full_text": pob,
           "disagreements": disagree}
    print(json.dumps(res, indent=2))
    (HERE / f"agreement_{part}.json").write_text(json.dumps(res, indent=2))
    return res


# --- PRISMA ---------------------------------------------------------------------------------------

def prisma() -> dict:
    """PRISMA counts from records.csv and extraction.csv, and paper/literature/prisma.png."""
    log = json.loads((HERE / "merge_log_A.json").read_text())
    rows = read_csv(records_path("A"))
    ext = read_csv(HERE / "extraction.csv") if (HERE / "extraction.csv").exists() else []
    # raw counts per database; a database searched with two strings (arXiv, part A) is
    # counted once per record, so the overlap between its strings is not "duplicates"
    multi = {}
    for f in log["exports"]:
        multi.setdefault(f.split("_")[0], []).append(f)
    per_db = {db: (log["per_database_unique"][db] if len({f.split("_")[1] for f in fs}) > 1
                   else sum(log["exports"][f] for f in fs)) for db, fs in multi.items()}
    stage1 = {}
    reasons = {}
    for r in rows:
        stage1[r["stage1"] or "unscreened"] = stage1.get(r["stage1"] or "unscreened", 0) + 1
        if r["stage1"] == "exclude":
            reasons[r["stage1_reason"]] = reasons.get(r["stage1_reason"], 0) + 1
    search_ext = [e for e in ext if e.get("origin", "search") == "search"]
    ft = [e for e in search_ext if e.get("full_text_read") == "yes"]
    ft_reasons = {}
    for e in ft:
        if e["stage2"] == "exclude":
            ft_reasons[e["stage2_reason"]] = ft_reasons.get(e["stage2_reason"], 0) + 1
    counts = {
        "identified": per_db, "identified_total": sum(per_db.values()),
        "duplicates_removed": sum(per_db.values()) - log["records_unique"],
        "screened": len(rows), "stage1": stage1, "stage1_exclusion_reasons": reasons,
        "sought_full_text": len(search_ext),
        "full_text_assessed": len(ft),
        "full_text_excluded": sum(e["stage2"] == "exclude" for e in ft),
        "full_text_exclusion_reasons": ft_reasons,
        "included_full_text_verified": sum(e["stage2"] == "include" for e in ft),
        "awaiting_full_text": sum(e["stage2"] == "pending" for e in search_ext),
        "review_bucket": reasons.get("3", 0),
        "other_sources": sum(e.get("origin") == "previous" for e in ext),
        "other_sources_pending_or_included": sum(e.get("origin") == "previous" and
                                                 e["stage2"] in ("include", "pending")
                                                 for e in ext),
    }
    (HERE / "prisma_counts.json").write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts, indent=2))
    _draw_prisma(counts, HERE / "prisma.png")
    return counts


REASON_TEXT = {"1": "not EEG seizure detection/prediction",
               "2": "no DL fusion of two representations/modalities",
               "3": "review only", "4": "full text not accessible"}


def _draw_prisma(c: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10, 11))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eef3fa"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc,
                                    ec="#33415c", lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, wrap=True)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", color="#33415c"))

    def reasons(d):
        return "\n".join(f"  {REASON_TEXT.get(k, k)}: {v}" for k, v in sorted(d.items()))

    dbs = c["identified"]
    names = {"scopus": "Scopus", "wos": "Web of Science", "ieee": "IEEE Xplore",
             "arxiv": "arXiv (preprints)"}
    ident = "\n".join(f"{names.get(k, k)}: n = {v}" for k, v in dbs.items())
    for y, lab in ((10.2, "Identification"), (7.4, "Screening"), (4.6, "Eligibility"),
                   (1.4, "Included")):
        ax.text(0.1, y, lab, rotation=90, fontsize=10, weight="bold", va="center")
    box(0.6, 9.4, 4.6, 1.8, f"Records identified\n{ident}\ntotal n = {c['identified_total']}")
    box(6.0, 9.4, 3.6, 1.8, f"Duplicates removed\nn = {c['duplicates_removed']}", "#f7f0e8")
    arrow(5.2, 10.3, 6.0, 10.3)
    box(0.6, 6.8, 4.6, 1.3, f"Records screened (title and abstract)\nn = {c['screened']}")
    arrow(2.9, 9.4, 2.9, 8.1)
    s1 = c["stage1"]
    box(6.0, 6.3, 3.6, 2.3, f"Records excluded\nn = {s1.get('exclude', 0)}\n"
        + reasons(c["stage1_exclusion_reasons"]), "#f7f0e8")
    arrow(5.2, 7.45, 6.0, 7.45)
    box(0.6, 4.0, 4.6, 1.3, f"Reports sought for full-text assessment\nn = {c['sought_full_text']}"
        f"\nfull text assessed so far: n = {c['full_text_assessed']}")
    arrow(2.9, 6.8, 2.9, 5.3)
    box(6.0, 3.5, 3.6, 2.3, f"Full texts excluded\nn = {c['full_text_excluded']}\n"
        + reasons(c["full_text_exclusion_reasons"]), "#f7f0e8")
    arrow(5.2, 4.65, 6.0, 4.65)
    box(0.6, 0.5, 4.6, 1.9, f"Included, full text verified: n = {c['included_full_text_verified']}\n"
        f"Included on title/abstract, full text pending: n = {c['awaiting_full_text']}\n"
        f"Other sources (scoping review, Table 1): n = {c['other_sources']}\n"
        f"Review bucket (context only): n = {c['review_bucket']}", "#e8f4ea")
    arrow(2.9, 4.0, 2.9, 2.4)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"figure: {out}")


# --- CSV helpers ----------------------------------------------------------------------------------

def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["strip", "merge", "verify", "sample", "agreement",
                                        "prisma"])
    ap.add_argument("--part", default="A", help="A, B1, B2, B3 or B4")
    ap.add_argument("--all", action="store_true", help="verify: recheck verified records")
    ap.add_argument("--frac", type=float, default=0.2, help="sample: share of records")
    args = ap.parse_args()
    if args.command == "strip":
        strip()
    elif args.command == "merge":
        merge(args.part)
    elif args.command == "verify":
        verify(args.part, only_missing=not args.all)
    elif args.command == "sample":
        sample(args.part, args.frac)
    elif args.command == "agreement":
        agreement(args.part)
    else:
        prisma()


if __name__ == "__main__":
    main()

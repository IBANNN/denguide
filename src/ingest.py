"""
Step 1 of the indexing pipeline: PDF -> clean paragraphs -> chunks.

Main functions
    extract_paragraphs(pdf_path)  read the CPG and return clean paragraphs with metadata
    make_chunks(paragraphs)       pack paragraphs into retrieval chunks
    run_ingest()                  do both and save .jsonl files

Cleaning rules (see notebook Section 2.2)
    1. Keep only the main body pages (CONFIG["keep_pdf_pages"]).
    2. Read the printed page label at the bottom of each page and drop it from the text.
    3. Drop superscript citation numbers (e.g. "...in 2021.1"); keep I² for heterogeneity.
    4. Drop bracketed citation lists such as [14, 16, 18].
    5. Drop GRADE certainty symbols; the certainty word (Low, Very low) is kept.
    6. Rebuild tables as "Header: value" sentences and carry headers across page breaks.
       The Summary of Recommendations table is skipped (duplicate of each question's statement).
    7. Skip every REFERENCES section.
    8. Track chapter, guideline question and section headings for metadata.
"""
import json
import re
from collections import Counter

import pymupdf  # PyMuPDF

from .config import CONFIG, PDF_PATH, PARAGRAPHS_PATH, CHUNKS_PATH, ensure_dirs

# ---------------------------------------------------------------- patterns
SUPERSCRIPT_FLAG = 1                                   # PyMuPDF span flag bit for superscript
CITATION_SUP = re.compile(r"^[\d,\-–\s]+$")            # superscript text made of citation numbers
BRACKET_CITATION = re.compile(r"\[\s*\d+(?:\s*[-–,]\s*\d+)*\s*[\]}]")
GRADE_SYMBOLS = re.compile(r"[⨁◯⊕○]+")
PAGE_LABEL = re.compile(r"^(?:[ivxlc]+|\d{1,3})$", re.IGNORECASE)
QUESTION_HEADING = re.compile(r"^QUESTION\s+(\d+[AB]?)\s*:\s*(.*)$", re.IGNORECASE)
BULLET = re.compile(r"^(?:[•▪●◦]|o\s|\d+\.\s|[a-h]\.\s)")

CHAPTERS = [
    "Disclaimer and Contact Information", "Executive Summary", "Summary of Recommendations",
    "Introduction", "CPG Development Methodology", "Recommendations and Evidence Summaries",
    "Applicability Issues", "Monitoring and Evaluation", "Research Implications/Gaps",
]
SUBSECTIONS = [
    # Sections inside each guideline question
    "KEY FINDINGS", "BACKGROUND", "BENEFITS AND HARMS", "CERTAINTY OF EVIDENCE",
    "RECOMMENDATIONS FROM OTHER GROUPS", "COST IMPLICATION",
    "CONSIDERATIONS DURING THE CONSENSUS PANEL MEETING",
    "ETHICAL, SOCIAL AND HEALTH SYSTEMS IMPACT", "ONGOING STUDIES AND RESEARCH GAPS",
    "REFERENCES",
    # Sections inside Introduction and Methodology
    "Background", "Objectives", "Scope and Purpose", "Target Population", "Intended Users",
    "Key Clinical Issues and Questions", "Guideline Preparation", "Evidence Synthesis",
    "Formulating Recommendations", "Guideline Dissemination", "Guideline Monitoring and Evaluation",
    "External Review", "Guideline Updating", "Editorial Independence",
]
# The Summary of Recommendations table repeats the recommendation statements found at the
# start of each question section, and its row layout does not extract cleanly.
SKIP_TABLE_CHAPTERS = {"Summary of Recommendations"}
_CHAPTER_KEYS = {c.lower(): c for c in CHAPTERS}
_SUBSECTION_KEYS = {s.lower(): s for s in SUBSECTIONS}


# ---------------------------------------------------------------- text helpers
def clean_text(text):
    """Remove citation brackets and GRADE symbols, then normalise spaces."""
    text = BRACKET_CITATION.sub("", text)
    text = GRADE_SYMBOLS.sub("", text)
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" +([,.;:)])", r"\1", text)
    return text.strip()


def spans_to_text(spans):
    """Join the spans of one PDF line, dropping superscript citation numbers."""
    out = ""
    for span in spans:
        t = span["text"]
        is_sup = span["flags"] & SUPERSCRIPT_FLAG
        if is_sup and t.strip() and CITATION_SUP.match(t.strip()):
            # I2 (heterogeneity) is the only numeric superscript worth keeping.
            if t.strip() == "2" and out.rstrip().endswith("I"):
                out = out.rstrip() + "²"
            continue
        out += t
    return out


def join_lines(lines):
    """Join PDF lines into one paragraph. Bullets start a new line; hyphen breaks are rejoined."""
    text = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not text:
            text = line
        elif BULLET.match(line):
            text += "\n" + line
        elif text.endswith("-") and (line[0].islower() or line[0].isdigit()):
            text += line                      # "in-" + "hospital" -> "in-hospital"
        else:
            text += " " + line
    return clean_text(text)


# ---------------------------------------------------------------- tables
def _cell_text(page, bbox):
    """Text inside one table cell. Reading by position keeps decimals and commas in place."""
    if bbox is None:
        return ""
    blocks = page.get_text("dict", clip=bbox)["blocks"]
    text = " ".join(spans_to_text(l["spans"]).strip()
                    for b in blocks if b["type"] == 0 for l in b["lines"])
    return " ".join(text.split())


def table_to_lines(page, table, carried_header=None):
    """
    Convert a PyMuPDF table into readable lines of the form "Header: value; Header: value".
    - Header cells wrapped over several rows are joined.
    - Rows whose first cell is empty continue the row above (text wrapped inside a cell).
    - carried_header: header of the same table from the previous page (table split by a page break).
    Returns (lines, header).
    """
    rows = [[_cell_text(page, bb) for bb in row.cells] for row in table.rows]
    keep = [j for j in range(len(rows[0])) if any(r[j] for r in rows)]   # drop empty columns
    rows = [[r[j] for j in keep] for r in rows]
    if not rows or not rows[0]:
        return [], None

    if carried_header and len(carried_header) == len(rows[0]):
        header, body = list(carried_header), rows
    else:
        header, body = rows[0], rows[1:]
        while body and not body[0][0]:                      # header text wrapped onto more rows
            header = [(a + " " + b).strip() for a, b in zip(header, body[0])]
            body = body[1:]
        header = [h or f"Column {i + 1}" for i, h in enumerate(header)]

    merged = []
    for row in body:
        if merged and not row[0]:
            merged[-1] = [(a + " " + b).strip() for a, b in zip(merged[-1], row)]
        else:
            merged.append(row)

    lines = []
    for row in merged:
        pairs = [f"{h}: {v}" for h, v in zip(header, row)
                 if v and h.lower() not in ("references", "reference")]
        if pairs:
            lines.append(clean_text("; ".join(pairs)))
    return lines, header


def _inside(bbox, rect, margin=2):
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return rect[0] - margin <= cx <= rect[2] + margin and rect[1] - margin <= cy <= rect[3] + margin


# ---------------------------------------------------------------- page reading
def read_page(page, carried_header=None):
    """
    Return (page_label, items, last_table_header) for one PDF page.
    items = list of (y_position, kind, content) sorted top to bottom,
    kind is "text" (content = list of lines, "" marks a paragraph break) or "table" (list of row lines).
    """
    tables = [t for t in page.find_tables().tables if t.row_count >= 2 and t.col_count >= 2]
    label, items = None, []

    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:                      # skip images
            continue
        lines = [spans_to_text(l["spans"]).strip() for l in block["lines"]]
        flat = " ".join(l for l in lines if l).strip()
        if not flat:
            continue
        if PAGE_LABEL.match(flat) and block["bbox"][1] > 700:   # printed page number at the bottom
            label = flat
            continue
        if any(_inside(block["bbox"], t.bbox) for t in tables):
            continue                                 # table text is handled below
        items.append((block["bbox"][1], "text", lines + [""]))

    last_header = None
    for t in tables:
        # A table that starts at the very top of the page may continue from the previous page.
        carry = carried_header if t.bbox[1] < 130 else None
        lines, header = table_to_lines(page, t, carry)
        if lines:
            items.append((t.bbox[1], "table", lines))
        last_header = header
    items.sort(key=lambda it: it[0])
    return label, items, last_header


# ---------------------------------------------------------------- structure tracking
def extract_paragraphs(pdf_path=PDF_PATH, keep_pages=None):
    """Read the PDF and return a list of paragraph dicts with chapter/question/section metadata."""
    keep_pages = keep_pages or CONFIG["keep_pdf_pages"]
    doc = pymupdf.open(pdf_path)
    state = {"chapter": None, "question": None, "question_title": None,
             "section": None, "skip": False}
    paragraphs, carried_header = [], None

    def add(text, page_label, pdf_page, kind="text"):
        text = text.strip()
        if text and not state["skip"]:
            paragraphs.append({
                "text": text, "kind": kind, "page": page_label, "pdf_page": pdf_page,
                "chapter": state["chapter"], "question": state["question"],
                "question_title": state["question_title"], "section": state["section"],
            })

    for pdf_page in keep_pages:
        page = doc[pdf_page - 1]
        label, items, carried_header = read_page(page, carried_header)
        label = label or str(pdf_page)

        for _, kind, content in items:
            if kind == "table":
                if state["chapter"] in SKIP_TABLE_CHAPTERS:
                    continue      # same statements appear as clean text in each question section
                add("[Table] " + "\n".join(content), label, pdf_page, kind="table")
                continue

            buffer, i = [], 0
            while i < len(content):
                line = content[i].strip()
                key = clean_text(line).rstrip(":").lower()
                q_match = QUESTION_HEADING.match(line)

                if not line:                                     # paragraph break
                    add(join_lines(buffer), label, pdf_page); buffer = []
                elif key in _CHAPTER_KEYS:                       # new chapter
                    add(join_lines(buffer), label, pdf_page); buffer = []
                    state.update(chapter=_CHAPTER_KEYS[key], question=None,
                                 question_title=None, section=None, skip=False)
                elif q_match:                                    # new guideline question
                    add(join_lines(buffer), label, pdf_page); buffer = []
                    title = q_match.group(2)
                    while not title.rstrip().endswith("?") and i + 1 < len(content) and content[i + 1].strip():
                        i += 1
                        title += " " + content[i].strip()
                    state.update(chapter="Recommendations and Evidence Summaries",
                                 question=q_match.group(1).upper(),
                                 question_title=clean_text(title), section="RECOMMENDATION", skip=False)
                elif key in _SUBSECTION_KEYS:                    # new section
                    add(join_lines(buffer), label, pdf_page); buffer = []
                    section = _SUBSECTION_KEYS[key]
                    state["section"] = section.upper() if state["question"] else section
                    state["skip"] = state["section"] == "REFERENCES"
                else:
                    buffer.append(line)
                i += 1
            add(join_lines(buffer), label, pdf_page)
    return paragraphs


# ---------------------------------------------------------------- chunking
def section_header(p):
    """Short context line placed at the top of every chunk (helps retrieval and citation)."""
    if p["question"]:
        return f"Question {p['question']}: {p['question_title']} | Section: {p['section']}"
    parts = [p["chapter"] or "Front matter"]
    if p["section"]:
        parts.append(p["section"])
    return " | Section: ".join(parts)


def _split_long(text, max_words):
    """Split one very long paragraph at sentence ends so no piece exceeds max_words."""
    sentences = re.split(r"(?<=[.;])\s+", text)
    pieces, current = [], []
    for s in sentences:
        if current and len(" ".join(current + [s]).split()) > max_words:
            pieces.append(" ".join(current)); current = []
        current.append(s)
    if current:
        pieces.append(" ".join(current))
    return pieces


def make_chunks(paragraphs, max_words=None, overlap_words=None, min_words=None):
    """
    Pack consecutive paragraphs of the same section into chunks of up to max_words.
    A new chunk starts whenever chapter, question or section changes.
    The last overlap_words of a chunk are repeated at the start of the next chunk in the same section.
    """
    max_words = max_words or CONFIG["chunk_words"]
    overlap_words = CONFIG["overlap_words"] if overlap_words is None else overlap_words
    min_words = min_words or CONFIG["min_chunk_words"]

    # 1. group paragraphs by section (excluded sections are dropped here)
    paragraphs = [p for p in paragraphs if p["section"] not in CONFIG["exclude_sections"]]
    groups, current, key = [], [], None
    for p in paragraphs:
        k = (p["chapter"], p["question"], p["section"])
        if k != key and current:
            groups.append(current); current = []
        key = k
        current.append(p)
    if current:
        groups.append(current)

    # 2. pack each group
    chunks = []
    for group in groups:
        header = section_header(group[0])
        units = []                                   # (text, page)
        for p in group:
            for piece in _split_long(p["text"], max_words):
                units.append((piece, p["page"]))

        body, pages, overlap = [], [], ""
        def flush():
            nonlocal body, pages, overlap
            text = "\n".join(body).strip()
            if len(text.split()) >= min_words:
                chunks.append(_new_chunk(len(chunks), header, overlap, text, pages, group[0]))
                words = text.split()
                overlap = " ".join(words[-overlap_words:]) if overlap_words else ""
            body, pages = [], []

        for text, page in units:
            if body and len(" ".join(body + [text]).split()) > max_words:
                flush()
            body.append(text)
            if page not in pages:
                pages.append(page)
        flush()
    return chunks


def _new_chunk(n, header, overlap, text, pages, meta):
    full = header + "\n" + (("... " + overlap + "\n") if overlap else "") + text
    return {
        "chunk_id": f"chunk_{n:04d}",
        "text": full,
        "pages": ",".join(pages),
        "page_start": pages[0],
        "page_end": pages[-1],
        "chapter": meta["chapter"] or "",
        "question": meta["question"] or "",
        "question_title": meta["question_title"] or "",
        "section": meta["section"] or "",
        "n_words": len(full.split()),
    }


# ---------------------------------------------------------------- save / load
def save_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_ingest(pdf_path=PDF_PATH):
    """Full ingest: extract, chunk, save, and return a small statistics dict."""
    ensure_dirs()
    paragraphs = extract_paragraphs(pdf_path)
    chunks = make_chunks(paragraphs)
    save_jsonl(paragraphs, PARAGRAPHS_PATH)
    save_jsonl(chunks, CHUNKS_PATH)
    lengths = [c["n_words"] for c in chunks]
    stats = {
        "pages_kept": len(CONFIG["keep_pdf_pages"]),
        "paragraphs": len(paragraphs),
        "tables": sum(p["kind"] == "table" for p in paragraphs),
        "characters_after_cleaning": sum(len(p["text"]) for p in paragraphs),
        "chunks": len(chunks),
        "mean_words_per_chunk": round(sum(lengths) / len(lengths), 1),
        "max_words_per_chunk": max(lengths),
        "chunks_per_question": dict(sorted(Counter(c["question"] or "-" for c in chunks).items())),
    }
    return paragraphs, chunks, stats

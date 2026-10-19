"""Turn each corpus format into an ordered list of Blocks.

Every parser keeps tables as row lists (rendered as Markdown), records the
nearest section heading for each block, and drops the per-document metadata
table (document ID, version, ...), because that information comes from
manifest.json instead.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from .models import Block
from .ocr import ocr_pdf_page

_WS = re.compile(r"\s+")


def clean(text: str) -> str:
    return _WS.sub(" ", text.replace(" ", " ")).strip()


def table_to_markdown(rows: list[list[str]]) -> str:
    rows = [[clean(c or "") for c in row] for row in rows if any((c or "").strip() for c in row)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def is_metadata_table(rows: list[list[str]]) -> bool:
    first_col = " ".join((r[0] or "") for r in rows if r)
    return "document id" in first_col.lower()


def _table_block(rows: list[list[str]], heading: str | None, page: int | None) -> Block | None:
    rows = [[clean(c or "") for c in row] for row in rows]
    rows = [r for r in rows if any(r)]
    if not rows or is_metadata_table(rows):
        return None
    return Block(kind="table", text=table_to_markdown(rows), heading=heading, page=page, rows=rows)


# --------------------------------------------------------------------------- PDF


def parse_pdf(path: Path) -> tuple[list[Block], bool, int]:
    import pdfplumber

    blocks: list[Block] = []
    ocr_pages = 0
    has_headings = False
    with pdfplumber.open(str(path)) as pdf:
        sizes: Counter = Counter()
        for page in pdf.pages:
            for ch in page.chars:
                sizes[round(ch["size"], 1)] += 1
        body_size = sizes.most_common(1)[0][0] if sizes else 10.0
        title_size = max(sizes) if sizes else 0.0

        heading: str | None = None
        for page_no, page in enumerate(pdf.pages, start=1):
            if not page.chars or not (page.extract_text() or "").strip():
                # Scanned page: no text layer, so read it with OCR.
                ocr_pages += 1
                for para in ocr_pdf_page(path, page_no - 1):
                    blocks.append(Block(kind="text", text=clean(para), heading=heading, page=page_no, ocr=True))
                continue

            tables = page.find_tables()
            bboxes = [t.bbox for t in tables]
            items: list[tuple[float, str, object]] = []
            for t in tables:
                items.append((t.bbox[1], "table", t.extract()))
            for line in page.extract_text_lines(return_chars=True, strip=True):
                mid = (line["top"] + line["bottom"]) / 2
                if any(b[1] <= mid <= b[3] and b[0] <= line["x0"] <= b[2] for b in bboxes):
                    continue
                chars = [c for c in line["chars"] if c["text"].strip()]
                if not chars:
                    continue
                size = round(sorted(c["size"] for c in chars)[len(chars) // 2], 1)
                bold = sum("bold" in c["fontname"].lower() for c in chars) > len(chars) / 2
                small = size < body_size - 0.4
                in_margin = line["top"] < page.height * 0.09 or line["bottom"] > page.height * 0.93
                if small and in_margin:
                    continue  # running header / footer / page number
                items.append((line["top"], "line", (line["text"], size, bold, line["top"], line["bottom"])))
            items.sort(key=lambda it: it[0])

            para: list[str] = []
            prev_bottom = None

            def flush() -> None:
                nonlocal para
                text = clean(" ".join(para))
                if text and text != "•":
                    blocks.append(Block(kind="text", text=text, heading=heading, page=page_no))
                para = []

            for _top, kind, payload in items:
                if kind == "table":
                    flush()
                    block = _table_block(payload, heading, page_no)  # type: ignore[arg-type]
                    if block:
                        blocks.append(block)
                    prev_bottom = None
                    continue
                text, size, bold, top, bottom = payload  # type: ignore[misc]
                if size >= title_size - 0.1 and size > body_size + 0.4:
                    flush()  # document title: the manifest already has it
                    prev_bottom = bottom
                    continue
                if bold and size > body_size + 0.4:
                    flush()
                    heading = clean(text)
                    has_headings = True
                    prev_bottom = bottom
                    continue
                starts_bullet = text.lstrip().startswith("•")
                big_gap = prev_bottom is not None and top - prev_bottom > 0.6 * size
                if (starts_bullet or big_gap) and para != ["•"]:
                    flush()
                para.append(text)
                prev_bottom = bottom
            flush()
    return blocks, has_headings, ocr_pages


# -------------------------------------------------------------------------- DOCX


def parse_docx(path: Path) -> tuple[list[Block], bool, int]:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(str(path))
    blocks: list[Block] = []
    heading: str | None = None
    has_headings = False
    for element in document.element.body.iterchildren():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(element, document)
            text = clean(para.text)
            if not text:
                continue
            style = (para.style.name if para.style is not None else "") or ""
            if style.lower() == "title":
                continue
            if style.lower().startswith("heading"):
                heading = text
                has_headings = True
                continue
            if "list" in style.lower():
                text = "- " + text
            blocks.append(Block(kind="text", text=text, heading=heading))
        elif tag == "tbl":
            table = Table(element, document)
            rows = []
            for row in table.rows:
                cells = []
                for cell in row.cells:
                    cells.append(cell.text)
                rows.append(cells)
            block = _table_block(rows, heading, None)
            if block:
                blocks.append(block)
    return blocks, has_headings, 0


# -------------------------------------------------------------------------- HTML

_BLOCK_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table", "pre", "blockquote", "dt", "dd"]


def parse_html(path: Path) -> tuple[list[Block], bool, int]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "lxml")
    for tag in soup(["script", "style", "nav", "footer", "head", "noscript"]):
        tag.decompose()
    body = soup.body or soup
    blocks: list[Block] = []
    heading: str | None = None
    has_headings = False
    for el in body.find_all(_BLOCK_TAGS):
        if el.find_parent(["table", "li"]) is not None and el.name != "table":
            continue
        if el.name == "table" and el.find_parent("table") is not None:
            continue
        if el.name == "h1":
            continue  # document title
        if el.name in ("h2", "h3", "h4", "h5", "h6"):
            text = clean(el.get_text(" "))
            if text:
                heading = text
                has_headings = True
            continue
        if el.name == "table":
            rows = [[c.get_text(" ") for c in tr.find_all(["th", "td"])] for tr in el.find_all("tr")]
            block = _table_block(rows, heading, None)
            if block:
                blocks.append(block)
            continue
        text = clean(el.get_text(" "))
        if not text:
            continue
        if el.name == "li":
            parent = el.find_parent(["ol", "ul"])
            if parent is not None and parent.name == "ol":
                index = len(el.find_previous_siblings("li")) + 1
                text = f"{index}. {text}"
            else:
                text = "- " + text
        blocks.append(Block(kind="text", text=text, heading=heading))
    return blocks, has_headings, 0


# ---------------------------------------------------------------------- Markdown

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def parse_markdown(path: Path) -> tuple[list[Block], bool, int]:
    blocks: list[Block] = []
    heading: str | None = None
    has_headings = False
    para: list[str] = []
    table: list[list[str]] = []

    def flush_para() -> None:
        nonlocal para
        text = clean(" ".join(para))
        if text:
            blocks.append(Block(kind="text", text=text, heading=heading))
        para = []

    def flush_table() -> None:
        nonlocal table
        if table:
            block = _table_block(table, heading, None)
            if block:
                blocks.append(block)
        table = []

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("|"):
            flush_para()
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue  # separator row
            table.append(cells)
            continue
        flush_table()
        match = _MD_HEADING.match(line)
        if match:
            flush_para()
            if len(match.group(1)) == 1:
                continue  # document title
            heading = clean(match.group(2))
            has_headings = True
            continue
        if not line:
            flush_para()
            continue
        if re.match(r"^([-*+]|\d+[.)])\s+", line):
            flush_para()
            blocks.append(Block(kind="text", text=clean(line), heading=heading))
            continue
        para.append(line)
    flush_para()
    flush_table()
    return blocks, has_headings, 0


PARSERS = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".html": parse_html,
    ".htm": parse_html,
    ".md": parse_markdown,
    ".markdown": parse_markdown,
    ".txt": parse_markdown,
}


def parse_file(path: Path) -> tuple[list[Block], bool, int]:
    parser = PARSERS.get(path.suffix.lower())
    if parser is None:
        raise ValueError(f"unsupported file type: {path.suffix}")
    return parser(path)

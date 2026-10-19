"""OCR for pages without a text layer (RapidOCR: ONNX, CPU-only, deterministic).

RapidOCR reads characters accurately but sometimes drops the spaces between
words ("Corehoursof10:00"). `repair_spacing` fixes that afterwards, using a
general English word splitter plus the vocabulary of the rest of the corpus
(so names such as the company name are not split apart).
"""
from __future__ import annotations

import re
import threading
from pathlib import Path
from statistics import median

import numpy as np

OCR_DPI = 200

_engine = None
_engine_lock = threading.Lock()


def _get_engine():
    global _engine
    with _engine_lock:
        if _engine is None:
            from rapidocr_onnxruntime import RapidOCR

            _engine = RapidOCR()
        return _engine


def ocr_pdf_page(pdf_path: Path, page_index: int) -> list[str]:
    """Return the page's text as paragraphs, in reading order."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        image = pdf[page_index].render(scale=OCR_DPI / 72).to_pil()
    finally:
        pdf.close()
    result, _ = _get_engine()(np.array(image))
    if not result:
        return []
    boxes = []
    for box, text, _conf in result:
        text = text.strip()
        if text:
            boxes.append((min(p[1] for p in box), max(p[1] for p in box), min(p[0] for p in box), text))
    boxes.sort(key=lambda b: b[0])

    # Group boxes into visual rows (vertical overlap), then read each row left to right.
    rows: list[list[tuple]] = []
    for box in boxes:
        centre = (box[0] + box[1]) / 2
        if rows and rows[-1][0][0] <= centre <= max(b[1] for b in rows[-1]):
            rows[-1].append(box)
        else:
            rows.append([box])
    line_h = median(b[1] - b[0] for b in boxes)
    paragraphs: list[list[str]] = []
    prev_bottom = None
    for row in rows:
        row.sort(key=lambda b: b[2])
        top = min(b[0] for b in row)
        text = " ".join(b[3] for b in row)
        if prev_bottom is None or top - prev_bottom > 0.9 * line_h:
            paragraphs.append([text])
        else:
            paragraphs[-1].append(text)
        prev_bottom = max(b[1] for b in row)
    return [" ".join(p) for p in paragraphs]


_RUNS = re.compile(r"[A-Za-z]+|\d+|[^A-Za-z\d]+")


def _split_letters(run: str, vocab: set[str]) -> list[str]:
    if len(run) < 7 or run.lower() in vocab:
        return [run]
    import wordninja

    pieces = wordninja.split(run)
    if "".join(pieces) != run:
        return [run]
    merged: list[str] = []
    for piece in pieces:
        # Re-join pieces that the corpus knows as one word (e.g. proper nouns).
        if merged and (merged[-1] + piece).lower() in vocab and not (
            merged[-1].lower() in vocab and piece.lower() in vocab
        ):
            merged[-1] += piece
        else:
            merged.append(piece)
    return merged


def repair_spacing(text: str, vocab: set[str]) -> str:
    text = re.sub(r"([,;:)])(?=[A-Za-z])", r"\1 ", text)
    text = re.sub(r"(?<=[a-z])\.(?=[A-Z])", ". ", text)  # "office.This", not "veltrix.example"
    text = re.sub(r"(?<=[A-Za-z])\(", " (", text)
    out_tokens = []
    for token in text.split():
        parts: list[str] = []
        runs = _RUNS.findall(token)
        for i, run in enumerate(runs):
            pieces = _split_letters(run, vocab) if run.isalpha() else [run]
            if parts and i > 0:
                prev = runs[i - 1]
                # Space between a word and a number ("of10" -> "of 10"), not "L6".
                boundary = (prev.isalpha() and run.isdigit() and len(prev) > 1) or (
                    prev[-1].isdigit() and run.isalpha() and len(run) > 1
                )
                if boundary:
                    parts.append(" ")
            parts.append(" ".join(pieces))
        out_tokens.append("".join(parts))
    return " ".join(out_tokens)

# ============================================================
# UTILIDADES PARA PDF
# ============================================================

import re
from pathlib import Path
import fitz


def extract_pdf_text_with_pages(pdf_path):
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))

    pages = []

    for i, page in enumerate(doc, start=1):
        text = page.get_text("text")
        pages.append({
            "page": i,
            "text": text
        })

    doc.close()
    return pages


def join_pages(pages):
    parts = []

    for page in pages:
        parts.append(
            f"\n\n===== PAGE {page['page']} =====\n\n{page['text']}"
        )

    return "\n".join(parts)


def clean_text(text):
    text = str(text or "")
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def is_valid_pdf(pdf_path):
    pdf_path = Path(pdf_path)

    try:
        doc = fitz.open(str(pdf_path))
        page_count = len(doc)
        doc.close()
        return page_count > 0
    except Exception:
        return False

"""Rough page-count estimate for the Status Report, so the 5-page body limit can be checked
without opening Word. Approximate: it wraps text by average character width rather than by the
real font metrics, so treat it as +/- half a page.

Usage: python reports/estimate_pages.py [path.docx]
"""
import math
import sys
from pathlib import Path

import docx
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "reports" / "Deliverable3_Status" / \
    "NetSentries_Deliverable3_Interim_Status_Report.docx"
FIGURE_PTS = 170          # a 6.3in-wide figure at the aspect ratios used here


def estimate(path):
    d = docx.Document(path)
    sec = d.sections[0]
    tw = (sec.page_width - sec.left_margin - sec.right_margin) / 914400
    th = (sec.page_height - sec.top_margin - sec.bottom_margin) / 914400
    page_pts = th * 72

    def para_pts(p):
        size = 9
        for r in p.runs:
            if r.font.size:
                size = r.font.size.pt
                break
        if p.style.name.startswith("Heading"):
            size = 14
        cpl = max(10, int(tw * 72 / (size * 0.47)))
        lines = max(1, math.ceil(len(p.text) / cpl)) if p.text.strip() else 1
        after = p.paragraph_format.space_after.pt if p.paragraph_format.space_after else 4
        return lines * size * 1.15 + after + (8 if p.style.name.startswith("Heading") else 0)

    def table_pts(tb):
        total = 0
        for row in tb.rows:
            h = 0
            for cell in row.cells:
                w = cell.width / 914400 if cell.width else tw / len(row.cells)
                w -= 0.16          # Word's default left+right cell margins
                cpl = max(6, int(w * 72 / (8 * 0.47)))
                h = max(h, math.ceil(max(1, len(cell.text)) / cpl) * 8 * 1.22 + 2.4)
            total += h
        return total

    paras = {p._p: p for p in d.paragraphs}
    tbls = {t._tbl: t for t in d.tables}
    pts, marks = 0, {}
    for el in d.element.body.iterchildren():
        if el.tag == qn("w:p"):
            p = paras.get(el)
            if p is None:
                continue
            if p.text.strip():
                marks.setdefault(p.text.strip()[:30], pts / page_pts + 1)
            pts += para_pts(p)
            if p._p.findall(".//" + qn("w:drawing")):
                pts += FIGURE_PTS
        elif el.tag == qn("w:tbl"):
            pts += table_pts(tbls[el])
    return marks, pts / page_pts


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    marks, total = estimate(path)
    for key in ["1. Updated", "2. Model Dev", "3. Preliminary", "4. Error", "5. General",
                "6. Adversarial", "7. Responsible", "8. Monitoring", "9. Final-Phase",
                "References", "Appendix A"]:
        for text, page in marks.items():
            if text.startswith(key):
                print(f"{key:16s} starts ~page {page:.1f}")
                break
    refs = next((v for k, v in marks.items() if k.startswith("References")), total)
    body = refs - 1                 # References starting on page N means the body filled N-1
    print(f"\nBody (through Section 9): ~{body:.1f} pages   [limit: 5]")
    print(f"Whole document:           ~{total:.1f} pages")
    print("OK" if body <= 5.0 else "OVER THE LIMIT - trim the body")


if __name__ == "__main__":
    main()

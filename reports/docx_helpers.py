"""Small python-docx helpers for filling the course template in place.

The template supplies the styles, the section headings and the rubric tables. These helpers
only replace the bracketed prompts with real content and resize the tables, so the submitted
document keeps the required structure exactly.
"""
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Inches, RGBColor
from docx.text.paragraph import Paragraph

# The brief caps the body at five pages excluding references and appendices, so the body runs
# dense and the supporting detail lives in the appendices.
BODY_SIZE = Pt(9)
TABLE_SIZE = Pt(8)
INK = RGBColor(0x11, 0x18, 0x27)
MUTED = RGBColor(0x66, 0x70, 0x85)


def _clear(paragraph):
    for run in list(paragraph.runs):
        run._element.getparent().remove(run._element)


def set_paragraph(paragraph, text, style="Normal", size=BODY_SIZE, italic=False,
                  bold=False, color=INK, align=None, space_after=Pt(4)):
    """Replace a paragraph's text, keeping it in place in the document flow."""
    _clear(paragraph)
    try:
        paragraph.style = paragraph.part.document.styles[style]
    except KeyError:
        pass
    run = paragraph.add_run(text)
    run.font.size = size
    run.italic = italic
    run.bold = bold
    run.font.color.rgb = color
    paragraph.paragraph_format.space_after = space_after
    # The template's Normal style uses 1.25 line spacing, which is generous for a report
    # capped at five pages; 1.15 keeps it readable and reclaims roughly a fifth of a page.
    if not style.startswith("Heading"):
        paragraph.paragraph_format.line_spacing = 1.15
    if align is not None:
        paragraph.alignment = align
    return paragraph


def paragraph_after(anchor, text, style="Normal", size=BODY_SIZE, italic=False,
                    bold=False, color=INK, space_after=Pt(4), align=None):
    """Insert a new paragraph immediately after `anchor` and return it."""
    new_p = OxmlElement("w:p")
    anchor._p.addnext(new_p)
    para = Paragraph(new_p, anchor._parent)
    return set_paragraph(para, text, style=style, size=size, italic=italic, bold=bold,
                         color=color, space_after=space_after, align=align)


def write_paragraphs(anchor, blocks, **kw):
    """Replace `anchor` with the first block, then chain the rest after it."""
    first, *rest = blocks
    cur = set_paragraph(anchor, first, **kw)
    for b in rest:
        cur = paragraph_after(cur, b, **kw)
    return cur


def bullets_after(anchor, items, size=BODY_SIZE):
    cur = anchor
    for it in items:
        cur = paragraph_after(cur, it, style="List Bullet", size=size, space_after=Pt(2))
    return cur


def write_cell(cell, text, bold=False, size=TABLE_SIZE, italic=False, color=INK):
    cell.text = ""
    para = cell.paragraphs[0]
    para.paragraph_format.space_before = Pt(0.8)
    para.paragraph_format.space_after = Pt(0.8)
    run = para.add_run("" if text is None else str(text))
    run.font.size = size
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = color
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    return cell


def _shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def fill_table(table, rows, header=True, size=TABLE_SIZE, widths=None, header_fill="DCE6F1"):
    """Write `rows` (list of lists, first row the header) into an existing template table."""
    n_cols = len(table.columns)
    while len(table.rows) > len(rows):
        table._tbl.remove(table.rows[-1]._tr)
    while len(table.rows) < len(rows):
        table.add_row()
    for i, row in enumerate(rows):
        vals = list(row)[:n_cols] + [""] * max(0, n_cols - len(row))
        for jx, val in enumerate(vals):
            is_head = header and i == 0
            write_cell(table.cell(i, jx), val, bold=is_head, size=size)
            if is_head and header_fill:
                _shade(table.cell(i, jx), header_fill)
    if widths:
        for jx, wdt in enumerate(widths):
            for row in table.rows:
                row.cells[jx].width = Inches(wdt)
    return table


def delete_table(table):
    table._element.getparent().remove(table._element)


def delete_paragraph(paragraph):
    paragraph._p.getparent().remove(paragraph._p)


def put_figure(table, image_path, width_inches=6.4):
    """Replace a 1x1 placeholder table with the figure it was holding a space for."""
    parent = table._element.getparent()
    new_p = OxmlElement("w:p")
    parent.insert(list(parent).index(table._element), new_p)
    # The paragraph needs the table's own parent: add_picture resolves the image part
    # by walking up to the document, and a None parent breaks that chain.
    para = Paragraph(new_p, table._parent)
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.space_after = Pt(2)
    para.add_run().add_picture(str(image_path), width=Inches(width_inches))
    delete_table(table)
    return para


def figure_after(anchor_el, parent, image_path, cap_text, width_inches=6.2):
    """Insert a centred figure plus its caption immediately after a table or paragraph."""
    pic_p = OxmlElement("w:p")
    anchor_el.addnext(pic_p)
    para = Paragraph(pic_p, parent)
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.space_before = Pt(5)
    para.paragraph_format.space_after = Pt(1)
    para.add_run().add_picture(str(image_path), width=Inches(width_inches))
    cap_p = OxmlElement("w:p")
    pic_p.addnext(cap_p)
    return caption(Paragraph(cap_p, parent), cap_text)


def caption(paragraph, text):
    return set_paragraph(paragraph, text, size=Pt(8.5), italic=True, color=MUTED,
                         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=Pt(10))


def find_paragraph(doc, prefix):
    for p in doc.paragraphs:
        if p.text.strip().startswith(prefix):
            return p
    raise LookupError(f"no paragraph starting with {prefix!r}")


def appendix_heading(doc, text):
    p = doc.add_paragraph()
    return set_paragraph(p, text, style="Heading 1", size=Pt(13), space_after=Pt(4))


def appendix_figure(doc, image_path, cap, width_inches=6.4):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(image_path), width=Inches(width_inches))
    c = doc.add_paragraph()
    caption(c, cap)
    return c


def page_break(doc):
    from docx.enum.text import WD_BREAK
    p = doc.add_paragraph()
    p.add_run().add_break(WD_BREAK.PAGE)
    return p

"""Build manuscript.docx from content.py.

content.py -> manuscript.fodt (LibreOffice flat ODT: StarMath formulas, embedded figures, tables)
-> manuscript.docx (LibreOffice export; formulas become native Word equations).

    python3 manuscript/build.py            # writes manuscript.docx in the project root
    python3 manuscript/build.py jse        # ASCE J. Surv. Eng. version from content_jse.py -> manuscript_JSE.docx

Block types in content.py:
  ('h1'|'h2', text)                   headings
  ('p', text)                         paragraph; $...$ inline StarMath, **bold** run-in heads
  ('eq'|'eq:label', starmath)         numbered display equation; {eq:label} in text -> number
  ('fig:label', path, caption, cm)    figure (PNG) with caption; {fig:label} -> number
  ('tab:label', caption, header, rows) table with caption above; {tab:label} -> number
  ('ref', text)                       reference (*italic*), sorted alphabetically
  ('title'|'center', text)            centred title / title-page line
  ('break',)                          page break

In the ASCE layout (jse) the text is double spaced with line and page numbers; with DEFER the tables,
the list of figure captions and the figures follow the references, each on its own page.
"""
import base64
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ASCE = len(sys.argv) > 1 and sys.argv[1] == 'jse'
DEFER = False   # True: tables and figures after the references; False: at their place in the text
if ASCE:
    from content_jse import BLOCKS, REFERENCES  # noqa: E402
else:
    from content import BLOCKS, REFERENCES      # noqa: E402

HEAD = '''<?xml version="1.0" encoding="UTF-8"?>
<office:document xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
 xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
 xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
 xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0"
 xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0"
 xmlns:math="http://www.w3.org/1998/Math/MathML"
 xmlns:xlink="http://www.w3.org/1999/xlink"
 office:version="1.3" office:mimetype="application/vnd.oasis.opendocument.text">
<office:font-face-decls>
 <style:font-face style:name="Times New Roman" svg:font-family="'Times New Roman'" style:font-family-generic="roman"/>
</office:font-face-decls>
<office:styles>
 <style:default-style style:family="paragraph">
  <style:paragraph-properties fo:line-height="150%"/>
  <style:text-properties style:font-name="Times New Roman" fo:font-size="12pt" fo:language="en" fo:country="GB"/>
 </style:default-style>
 <style:style style:name="Standard" style:family="paragraph">
  <style:text-properties style:font-name="Times New Roman" fo:font-size="12pt"/>
 </style:style>
 <style:style style:name="Body" style:display-name="Body Text" style:family="paragraph" style:parent-style-name="Standard">
  <style:paragraph-properties fo:text-align="justify" fo:text-indent="1cm" fo:margin-top="0cm" fo:margin-bottom="0.2cm" fo:line-height="150%"/>
 </style:style>
 <style:style style:name="Heading_20_1" style:display-name="Heading 1" style:family="paragraph" style:parent-style-name="Standard" style:default-outline-level="1">
  <style:paragraph-properties fo:margin-top="0.6cm" fo:margin-bottom="0.3cm" fo:keep-with-next="always"/>
  <style:text-properties style:font-name="Times New Roman" fo:font-size="14pt" fo:font-weight="bold"/>
 </style:style>
 <style:style style:name="Heading_20_2" style:display-name="Heading 2" style:family="paragraph" style:parent-style-name="Standard" style:default-outline-level="2">
  <style:paragraph-properties fo:margin-top="0.4cm" fo:margin-bottom="0.2cm" fo:keep-with-next="always"/>
  <style:text-properties style:font-name="Times New Roman" fo:font-size="12pt" fo:font-weight="bold"/>
 </style:style>
 <style:style style:name="Equation" style:family="paragraph" style:parent-style-name="Standard">
  <style:paragraph-properties fo:margin-top="0.15cm" fo:margin-bottom="0.25cm" fo:line-height="100%">
   <style:tab-stops>
    <style:tab-stop style:position="8cm" style:type="center"/>
    <style:tab-stop style:position="16cm" style:type="right"/>
   </style:tab-stops>
  </style:paragraph-properties>
 </style:style>
 <style:style style:name="Figure" style:family="paragraph" style:parent-style-name="Standard">
  <style:paragraph-properties fo:text-align="center" fo:margin-top="0.3cm" fo:margin-bottom="0.1cm" fo:line-height="100%" fo:keep-with-next="always"/>
 </style:style>
 <style:style style:name="Caption" style:family="paragraph" style:parent-style-name="Standard">
  <style:paragraph-properties fo:text-align="justify" fo:margin-top="0.1cm" fo:margin-bottom="0.35cm" fo:line-height="120%"/>
  <style:text-properties fo:font-size="10pt"/>
 </style:style>
 <style:style style:name="TableCaption" style:family="paragraph" style:parent-style-name="Caption">
  <style:paragraph-properties fo:keep-with-next="always" fo:margin-top="0.3cm"/>
 </style:style>
 <style:style style:name="Cell" style:family="paragraph" style:parent-style-name="Standard">
  <style:paragraph-properties fo:text-align="center" fo:line-height="110%"/>
  <style:text-properties fo:font-size="8pt"/>
 </style:style>
 <style:style style:name="CellLeft" style:family="paragraph" style:parent-style-name="Cell">
  <style:paragraph-properties fo:text-align="start"/>
 </style:style>
 <style:style style:name="Reference" style:family="paragraph" style:parent-style-name="Standard">
  <style:paragraph-properties fo:text-align="justify" fo:margin-left="1cm" fo:text-indent="-1cm" fo:margin-bottom="0.12cm" fo:line-height="130%"/>
  <style:text-properties fo:font-size="11pt"/>
 </style:style>
 <style:style style:name="Title" style:family="paragraph" style:parent-style-name="Standard">
  <style:paragraph-properties fo:text-align="center" fo:margin-top="1cm" fo:margin-bottom="0.6cm" fo:line-height="150%"/>
  <style:text-properties fo:font-size="16pt" fo:font-weight="bold"/>
 </style:style>
 <style:style style:name="Center" style:family="paragraph" style:parent-style-name="Standard">
  <style:paragraph-properties fo:text-align="center" fo:margin-bottom="0.2cm" fo:line-height="150%"/>
 </style:style>
 <style:style style:name="Footer" style:family="paragraph" style:parent-style-name="Standard">
  <style:paragraph-properties fo:text-align="center"/>
 </style:style>
 <style:style style:name="Break" style:family="paragraph" style:parent-style-name="Standard">
  <style:paragraph-properties fo:break-before="page"/>
 </style:style>
 <style:style style:name="Bold" style:family="text"><style:text-properties fo:font-weight="bold"/></style:style>
 <style:style style:name="Italic" style:family="text"><style:text-properties fo:font-style="italic"/></style:style>
</office:styles>
<office:automatic-styles>
 <style:page-layout style:name="pm1">
  <style:page-layout-properties fo:page-width="21cm" fo:page-height="29.7cm" fo:margin-top="2.5cm" fo:margin-bottom="2.5cm" fo:margin-left="2.5cm" fo:margin-right="2.5cm"/>
 </style:page-layout>
 <style:style style:name="Tbl" style:family="table"><style:table-properties style:width="16cm" table:align="center"/></style:style>
 <style:style style:name="TblHead" style:family="table-cell">
  <style:table-cell-properties fo:padding="0.05cm" fo:border-top="0.75pt solid #000000" fo:border-bottom="0.5pt solid #000000" fo:border-left="none" fo:border-right="none"/>
 </style:style>
 <style:style style:name="TblCell" style:family="table-cell">
  <style:table-cell-properties fo:padding="0.05cm" fo:border="none"/>
 </style:style>
 <style:style style:name="TblLast" style:family="table-cell">
  <style:table-cell-properties fo:padding="0.05cm" fo:border-bottom="0.75pt solid #000000" fo:border-top="none" fo:border-left="none" fo:border-right="none"/>
 </style:style>
</office:automatic-styles>
<office:master-styles><style:master-page style:name="Standard" style:page-layout-name="pm1"/></office:master-styles>
<office:body><office:text>
'''
TAIL = '</office:text></office:body></office:document>\n'

counter = {'f': 0}
labels = {}


def formula(src, block=False):
    counter['f'] += 1
    width = '14cm' if block else '1.5cm'
    display = ' display="block"' if block else ''
    return (f'<draw:frame draw:name="Formula{counter["f"]}" text:anchor-type="as-char" svg:width="{width}" svg:height="0.6cm">'
            f'<draw:object><math:math{display}><math:semantics><math:mrow/>'
            f'<math:annotation encoding="StarMath 5.0">{escape(src)}</math:annotation>'
            '</math:semantics></math:math></draw:object></draw:frame>')


def resolve(text):
    return re.sub(r'\{(eq|fig|tab):(\w+)\}', lambda m: str(labels[m.group(1)+':'+m.group(2)]), text)


def inline(text, italic_marks=False):
    """Text with $math$, **bold** and (references) *italic* spans -> ODF inline XML."""
    text = resolve(text)
    out = []
    for part in re.split(r'(\$[^$]+\$|\*\*[^*]+\*\*)', text):
        if not part:
            continue
        if part.startswith('$'):
            out.append(formula(part[1:-1]))
        elif part.startswith('**'):
            out.append(f'<text:span text:style-name="Bold">{escape(part[2:-2])}</text:span>')
        elif italic_marks:
            for piece in re.split(r'(\*[^*]+\*)', part):
                if piece.startswith('*') and piece.endswith('*') and len(piece) > 1:
                    out.append(f'<text:span text:style-name="Italic">{escape(piece[1:-1])}</text:span>')
                else:
                    out.append(escape(piece))
        else:
            out.append(escape(part))
    return ''.join(out)


def figure(path, caption, width_cm, number):
    from PIL import Image
    data = Path(path).read_bytes()
    with Image.open(path) as image:
        height_cm = width_cm*image.height/image.width
    picture = (f'<draw:frame draw:name="Figure{number}" text:anchor-type="as-char" svg:width="{width_cm}cm" '
               f'svg:height="{height_cm:.2f}cm"><draw:image><office:binary-data>{base64.b64encode(data).decode()}'
               '</office:binary-data></draw:image></draw:frame>')
    return (f'<text:p text:style-name="Figure">{picture}</text:p>'
            f'<text:p text:style-name="Caption"><text:span text:style-name="Bold">Fig. {number}</text:span> {inline(caption)}</text:p>')


COLUMN_STYLES = []


def table(caption, header, rows, number):
    columns = len(header)
    # Column widths follow the longest entry of each column (header words may wrap), within limits.
    longest = [max([max(len(w) for w in str(header[k]).split())]+[len(str(r[k])) for r in rows]) for k in range(columns)]
    weights = [min(max(n, 4), 40) for n in longest]
    widths = [w/sum(weights) for w in weights]
    for k, w in enumerate(widths):
        COLUMN_STYLES.append(f'<style:style style:name="T{number}C{k}" style:family="table-column">'
                             f'<style:table-column-properties style:column-width="{16*w:.2f}cm"/></style:style>')
    cells = lambda row, style, last=False: ''.join(
        f'<table:table-cell table:style-name="{"TblLast" if last else style}" office:value-type="string">'
        f'<text:p text:style-name="{"CellLeft" if k == 0 else "Cell"}">{inline(str(v))}</text:p></table:table-cell>'
        for k, v in enumerate(row))
    body = [f'<table:table-row>{cells(header, "TblHead")}</table:table-row>']
    body += [f'<table:table-row>{cells(row, "TblCell", k == len(rows)-1)}</table:table-row>' for k, row in enumerate(rows)]
    return (f'<text:p text:style-name="TableCaption"><text:span text:style-name="Bold">Table {number}</text:span> {inline(caption)}</text:p>'
            f'<table:table table:name="Table{number}" table:style-name="Tbl">'
            + ''.join(f'<table:table-column table:style-name="T{number}C{k}"/>' for k in range(columns))
            + f'{"".join(body)}</table:table>')


def sort_key(block):
    return unicodedata.normalize('NFD', block[1]).encode('ascii', 'ignore').decode().lower()


def asce_layout(head):
    """Double spacing, line numbers and page numbers (ASCE manuscript format)."""
    head = head.replace('fo:text-align="justify" fo:text-indent="1cm" fo:margin-top="0cm" fo:margin-bottom="0.2cm" fo:line-height="150%"',
                        'fo:text-align="start" fo:text-indent="1cm" fo:margin-top="0cm" fo:margin-bottom="0cm" fo:line-height="200%"')
    head = head.replace('fo:margin-bottom="0.12cm" fo:line-height="130%"', 'fo:margin-bottom="0cm" fo:line-height="200%"')
    head = head.replace('</office:styles>', '<text:linenumbering-configuration text:number-lines="true" text:increment="1" '
                        'text:restart-on-page="false" style:num-format="1" text:offset="0.6cm" text:number-position="left"/>'
                        '</office:styles>')
    head = head.replace('<style:master-page style:name="Standard" style:page-layout-name="pm1"/>',
                        '<style:master-page style:name="Standard" style:page-layout-name="pm1"><style:footer>'
                        '<text:p text:style-name="Footer"><text:page-number text:select-page="current"/></text:p>'
                        '</style:footer></style:master-page>')
    return head.replace('<style:page-layout-properties fo:page-width="21cm"', '<style:page-layout-properties fo:page-width="21.59cm" fo:page-height="27.94cm"'
                        ).replace('fo:page-height="29.7cm" ', '')


def build():
    count = {'eq': 0, 'fig': 0, 'tab': 0}
    for block in BLOCKS:                       # first pass: numbers of labelled items
        kind = block[0].split(':')[0]
        if kind in count:
            count[kind] += 1
            if ':' in block[0]:
                labels[block[0]] = count[kind]
    body, count, tables, figures = [], {'eq': 0, 'fig': 0, 'tab': 0}, [], []
    for block in BLOCKS:
        kind = block[0].split(':')[0]
        if kind == 'h1':
            body.append(f'<text:h text:style-name="Heading_20_1" text:outline-level="1">{escape(block[1])}</text:h>')
        elif kind == 'h2':
            body.append(f'<text:h text:style-name="Heading_20_2" text:outline-level="2">{escape(block[1])}</text:h>')
        elif kind == 'title':
            body.append(f'<text:p text:style-name="Title">{inline(block[1])}</text:p>')
        elif kind == 'center':
            body.append(f'<text:p text:style-name="Center">{inline(block[1])}</text:p>')
        elif kind == 'break':
            body.append('<text:p text:style-name="Break"/>')
        elif kind == 'p':
            body.append(f'<text:p text:style-name="Body">{inline(block[1])}</text:p>')
        elif kind == 'eq':
            count['eq'] += 1
            body.append(f'<text:p text:style-name="Equation"><text:tab/>{formula(block[1], True)}<text:tab/>({count["eq"]})</text:p>')
        elif kind == 'fig':
            count['fig'] += 1
            (figures if DEFER else body).append((block, count['fig']) if DEFER else figure(block[1], block[2], block[3], count['fig']))
        elif kind == 'tab':
            count['tab'] += 1
            (tables if DEFER else body).append(table(block[1], block[2], block[3], count['tab']))
    refs = [b for b in REFERENCES if b[0] == 'ref']
    body.append('<text:h text:style-name="Heading_20_1" text:outline-level="1">References</text:h>')
    for _, text in sorted(refs, key=sort_key):
        body.append(f'<text:p text:style-name="Reference">{inline(text, italic_marks=True)}</text:p>')
    for t in tables:                           # ASCE: each table on its own page after the references
        body += ['<text:p text:style-name="Break"/>', t]
    if figures:
        body += ['<text:p text:style-name="Break"/>',
                 '<text:h text:style-name="Heading_20_1" text:outline-level="1">List of Figure Captions</text:h>']
        body += [f'<text:p text:style-name="Reference"><text:span text:style-name="Bold">Fig. {n}.</text:span> {inline(b[2])}</text:p>'
                 for b, n in figures]
        for b, n in figures:
            body += ['<text:p text:style-name="Break"/>', figure(b[1], b[2], b[3], n)]
    head = asce_layout(HEAD) if ASCE else HEAD
    return head.replace('</office:automatic-styles>', ''.join(COLUMN_STYLES)+'</office:automatic-styles>')+'\n'.join(body)+TAIL, count, len(refs)


if __name__ == '__main__':
    xml, count, references = build()
    work = HERE/'build'
    work.mkdir(exist_ok=True)
    (work/'manuscript.fodt').write_text(xml, encoding='utf-8')
    subprocess.run(['soffice', '--headless', '--convert-to', 'docx:MS Word 2007 XML', '--outdir', str(work),
                    str(work/'manuscript.fodt')], check=True, capture_output=True)
    target = HERE.parent/('manuscript_JSE.docx' if ASCE else 'manuscript.docx')
    target.write_bytes((work/'manuscript.docx').read_bytes())
    print(f'{count["eq"]} equations, {count["fig"]} figures, {count["tab"]} tables, {references} references -> {target}')

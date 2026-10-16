"""
docx_to_text.py -- Minimal, dependency-free .docx -> plain text extractor.

A .docx is a zip archive; its text lives in word/document.xml as a sequence
of <w:t> runs. This avoids installing python-docx just to read a few
read-only report files referenced by the audit.

Usage:
    python scripts/docx_to_text.py "path/to/file.docx"
"""
import re
import sys
import zipfile


def docx_to_text(path: str) -> str:
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8")

    def unescape(s):
        return (s.replace("&amp;", "&").replace("&lt;", "<")
                 .replace("&gt;", ">").replace("&quot;", '"').replace("&apos;", "'"))

    # Single left-to-right scan preserving document order: emit run text as
    # found, a tab at the end of each table cell, a newline at the end of
    # each paragraph/table row.
    out = []
    # NOTE: the tag-name part must be anchored to exactly "t" (</w:t>, <w:t>,
    # <w:t ...>) -- a loose "<w:t[^>]*>" also matches <w:tcPr>, <w:tcBorders>,
    # <w:tcW>, <w:tcMar>, etc. since they all start with "w:t" too.
    for m in re.finditer(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>|</w:tc>|</w:tr>|</w:p>", xml, flags=re.DOTALL):
        token = m.group(0)
        if token == "</w:tc>":
            out.append("\t")
        elif token == "</w:tr>":
            out.append("\n")
        elif token == "</w:p>":
            out.append("\n")
        else:
            out.append(unescape(m.group(1)))
    return "".join(out)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(f"\n{'='*100}\n{p}\n{'='*100}")
        try:
            print(docx_to_text(p))
        except Exception as e:
            print(f"COULD NOT OPEN: {e}")

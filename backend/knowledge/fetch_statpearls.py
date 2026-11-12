"""StatPearls (NCBI Bookshelf open-access subset; CC BY-NC-ND 4.0) -> section chunks.
Download once (2 GB):  https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz  into backend/data/knowledge/raw/
From backend/: python -m knowledge.fetch_statpearls

Reads the archive as a stream (no images extracted). Archived (outdated) chapters and non-clinical sections are left out.
Text is kept word for word (the licence forbids altering it); we only split long sections at sentence ends.
"""

import re
import tarfile
import xml.etree.ElementTree as ET

from knowledge.common import RAW, chunk, clean, write_chunks

ARCHIVE = RAW / "statpearls_NBK430685.tar.gz"
LICENCE = "CC BY-NC-ND 4.0, StatPearls Publishing (NCBI Bookshelf)"
SKIP_SECTIONS = {"review questions", "continuing education activity", "references", "personnel", "equipment",
                 "enhancing healthcare team outcomes", "nursing, allied health, and interprofessional team interventions"}


def _text(el) -> str:
    out = []
    for node in el.iter():
        if node.tag in ("ref-list", "fig", "table-wrap", "media"):
            continue
        if node.tag in ("p", "list-item", "title") and node is not el:
            out.append(clean("".join(node.itertext())))
    return " ".join(t for t in out if t)


def parse(xml: str, article: str) -> list[dict]:
    xml = re.sub(r"<!DOCTYPE[^>]*>", "", xml)
    root = ET.fromstring(xml)
    title_el = root.find(".//book-part-meta/title-group/title")
    if title_el is None:
        title_el = root.find(".//title-group/title")
    title = clean("".join(title_el.itertext())) if title_el is not None else ""
    if not title or "archived" in title.lower():
        return []
    body = root.find(".//body")
    if body is None:
        return []
    url = f"https://www.ncbi.nlm.nih.gov/books/n/statpearls/{article}/"
    chunks = []
    for sec in body.findall("sec"):
        t = sec.find("title")
        name = clean("".join(t.itertext())) if t is not None else ""
        if not name or name.lower() in SKIP_SECTIONS:
            continue
        text = _text(sec)
        if text:
            chunks += chunk("StatPearls", title, name, url, LICENCE, text, article=article)
    return chunks


def main() -> None:
    chunks, chapters, skipped = [], 0, 0
    with tarfile.open(ARCHIVE, "r|gz") as tar:
        for m in tar:
            if not (m.isfile() and m.name.endswith(".nxml")):
                continue
            article = m.name.rsplit("/", 1)[-1][:-5]
            try:
                got = parse(tar.extractfile(m).read().decode("utf-8", "replace"), article)
            except ET.ParseError:
                got = []
            if got:
                chunks += got
                chapters += 1
            else:
                skipped += 1
    n = write_chunks("statpearls", chunks)
    print(f"StatPearls chapters used {chapters}, skipped (archived/empty) {skipped}, chunks {n}")


if __name__ == "__main__":
    main()

"""MedlinePlus health topics (US National Library of Medicine; public domain) -> chunks with plain-language summaries.
From backend/: python -m knowledge.fetch_medlineplus
"""

import html
import re
import urllib.request
import xml.etree.ElementTree as ET

from knowledge.common import RAW, chunk, clean, write_chunks

INDEX = "https://medlineplus.gov/xml.html"
LICENCE = "Public domain (MedlinePlus, US National Library of Medicine)"


def latest_url() -> str:
    page = urllib.request.urlopen(INDEX, timeout=60).read().decode("utf-8", "replace")
    found = sorted(set(re.findall(r"https://medlineplus\.gov/xml/mplus_topics_\d{4}-\d{2}-\d{2}\.xml", page)))
    if not found:
        raise SystemExit("No MedlinePlus topics file found on " + INDEX)
    return found[-1]


def main() -> None:
    url = latest_url()
    path = RAW / url.rsplit("/", 1)[-1]
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    root = ET.parse(path).getroot()
    chunks = []
    for topic in root.findall("health-topic"):
        if topic.get("language") != "English":
            continue
        summary = topic.findtext("full-summary") or ""
        text = clean(html.unescape(re.sub(r"<[^>]+>", " ", summary)))
        if text:
            chunks += chunk("MedlinePlus", topic.get("title", ""), "Summary", topic.get("url", ""), LICENCE, text, keep_short=True)
    n = write_chunks("medlineplus", chunks)
    print(f"MedlinePlus topics file {path.name}: chunks {n}")


if __name__ == "__main__":
    main()

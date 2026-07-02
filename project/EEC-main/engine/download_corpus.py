"""Download ~50 public-domain books from Project Gutenberg, strip the
header/footer boilerplate, and combine into a single cleaned corpus."""
import os
import re
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "raw")
OUT = os.path.join(HERE, "corpus.txt")

# A spread of genres/authors. Extra IDs included to absorb failed downloads;
# we stop once we have 50 successful books.
BOOK_IDS = [
    1342, 11, 84, 1661, 2701, 98, 1400, 64317, 174, 345,
    1080, 2542, 4300, 2600, 5200, 1232, 16328, 25344, 158, 215,
    120, 76, 74, 1184, 2554, 28054, 600, 1260, 768, 766,
    1399, 100, 135, 6130, 1727, 3207, 996, 829, 2814, 1497,
    30254, 19942, 2680, 1998, 4363, 209, 271, 161, 141, 105,
    33, 36, 35, 1228, 730, 521, 20203, 1023, 932, 902,
]

HEADER_RE = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
                       re.IGNORECASE | re.DOTALL)
FOOTER_RE = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK",
                       re.IGNORECASE)


def candidate_urls(bid):
    return [
        f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt",
        f"https://www.gutenberg.org/files/{bid}/{bid}-0.txt",
        f"https://www.gutenberg.org/files/{bid}/{bid}.txt",
    ]


def fetch(bid):
    for url in candidate_urls(bid):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read().decode("utf-8", errors="ignore")
            if len(data) > 5000:
                return data
        except Exception:
            continue
    return None


def strip_boilerplate(text):
    m = HEADER_RE.search(text)
    if m:
        text = text[m.end():]
    m = FOOTER_RE.search(text)
    if m:
        text = text[:m.start()]
    return text.strip()


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    books = []
    for bid in BOOK_IDS:
        if len(books) >= 50:
            break
        cache = os.path.join(RAW_DIR, f"{bid}.txt")
        if os.path.exists(cache):
            with open(cache, encoding="utf-8") as f:
                cleaned = f.read()
            books.append(cleaned)
            print(f"  cached  {bid:>6}  ({len(cleaned):>8} chars)")
            continue
        raw = fetch(bid)
        if raw is None:
            print(f"  FAIL    {bid:>6}")
            continue
        cleaned = strip_boilerplate(raw)
        if len(cleaned) < 2000:
            print(f"  thin    {bid:>6}  skipped")
            continue
        with open(cache, "w", encoding="utf-8") as f:
            f.write(cleaned)
        books.append(cleaned)
        print(f"  ok      {bid:>6}  ({len(cleaned):>8} chars)")
        time.sleep(1.0)

    combined = "\n\n".join(books)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(combined)
    print(f"\nDownloaded {len(books)} books -> {OUT}  ({len(combined):,} chars)")


if __name__ == "__main__":
    main()

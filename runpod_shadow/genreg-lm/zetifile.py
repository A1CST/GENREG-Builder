"""ZetiFile — I2's encyclopedia, served from a LOCAL Wikipedia dump.

The corpus lives on THIS machine only (corpora/wikipedia/, gitignored) and is
never pushed to the network: the code ships to every node, but a node without
the dump simply reports ZetiFile as unavailable and serves no articles.

Storage model — Wikipedia "multistream" dump, no extraction ever:
  enwiki-latest-pages-articles-multistream.xml.bz2   the articles (~22GB)
  enwiki-latest-pages-articles-multistream-index.txt.bz2  offset:pageid:title
The multistream file is a concatenation of independent bz2 streams, each
holding <=100 pages. An article read = seek(offset) -> decompress ONE stream
(~1MB) -> pick the page out of the XML fragment. Random access in ~10ms with
zero disk overhead beyond the dump itself.

The title index is loaded once into SQLite (zetifile_index.sqlite): exact,
prefix, and FTS5 word search over ~7M titles.

  python zetifile.py build     # build the SQLite index from the index file
  python zetifile.py info      # availability + article count
  python zetifile.py search q  # try a search
  python zetifile.py get TITLE # fetch + strip one article
"""

import bz2
import hashlib
import html
import json
import os
import re
import sqlite3
import sys
import threading

ROOT = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_DIRS = [
    os.environ.get("I2_ZETIFILE_DIR") or "",
    os.path.join(ROOT, "corpora", "wikipedia"),
    r"C:\Users\paytonm\Documents\GENREG\corpora\wikipedia",   # dev machine home
]
ZETI_DIR = next((d for d in _CANDIDATE_DIRS if d and os.path.isdir(d)),
                os.path.join(ROOT, "corpora", "wikipedia"))
DUMP = os.path.join(ZETI_DIR, "enwiki-latest-pages-articles-multistream.xml.bz2")
INDEX_BZ2 = os.path.join(ZETI_DIR, "enwiki-latest-pages-articles-multistream-index.txt.bz2")
DB_PATH = os.path.join(ZETI_DIR, "zetifile_index.sqlite")

# non-article namespaces to skip when indexing
_SKIP_NS = {
    "talk", "user", "user talk", "wikipedia", "wikipedia talk", "file",
    "file talk", "mediawiki", "mediawiki talk", "template", "template talk",
    "help", "help talk", "category", "category talk", "portal", "portal talk",
    "draft", "draft talk", "timedtext", "timedtext talk", "module",
    "module talk", "book", "book talk", "gadget", "gadget definition",
}

_local = threading.local()


def _db():
    if getattr(_local, "db", None) is None:
        _local.db = sqlite3.connect(DB_PATH)
    return _local.db


def available():
    return os.path.exists(DB_PATH) and os.path.exists(DUMP)


def info():
    out = {"available": available(), "dump": os.path.exists(DUMP),
           "index": os.path.exists(DB_PATH), "articles": 0,
           "dump_bytes": os.path.getsize(DUMP) if os.path.exists(DUMP) else 0}
    if out["index"]:
        try:
            row = _db().execute("SELECT v FROM meta WHERE k='articles'").fetchone()
            out["articles"] = int(row[0]) if row else 0
        except sqlite3.Error:
            pass
    return out


# --------------------------------------------------------------------------
# Index build: stream the bz2 index file into SQLite (one-time, ~minutes)
# --------------------------------------------------------------------------
def build_index(progress=None):
    if not os.path.exists(INDEX_BZ2):
        raise FileNotFoundError(f"no index file at {INDEX_BZ2}")
    tmp = DB_PATH + ".building"
    if os.path.exists(tmp):
        os.remove(tmp)
    db = sqlite3.connect(tmp)
    db.executescript(
        "PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;"
        "CREATE TABLE a(t TEXT, tl TEXT, o INTEGER);"
        "CREATE TABLE meta(k TEXT PRIMARY KEY, v TEXT);"
        "CREATE VIRTUAL TABLE a_fts USING fts5(t, content='a', content_rowid='rowid');")
    n = 0
    batch = []
    with bz2.open(INDEX_BZ2, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            off, _, rest = line.partition(":")
            _pid, _, title = rest.partition(":")
            title = title.rstrip("\n")
            if not title:
                continue
            ns = title.split(":", 1)[0].lower() if ":" in title else ""
            if ns in _SKIP_NS:
                continue
            batch.append((title, title.lower(), int(off)))
            if len(batch) >= 50000:
                db.executemany("INSERT INTO a VALUES(?,?,?)", batch)
                n += len(batch)
                batch.clear()
                if progress:
                    progress(n)
    if batch:
        db.executemany("INSERT INTO a VALUES(?,?,?)", batch)
        n += len(batch)
    db.execute("CREATE INDEX a_tl ON a(tl)")
    db.execute("INSERT INTO a_fts(a_fts) VALUES('rebuild')")
    db.execute("INSERT INTO meta VALUES('articles', ?)", (str(n),))
    db.commit()
    db.close()
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    os.rename(tmp, DB_PATH)
    return n


# --------------------------------------------------------------------------
# Search: exact title -> prefix -> FTS words, deduped, in that order
# --------------------------------------------------------------------------
def slug_for(title):
    """Deterministic I2 page-name segment for an article title. Readable
    prefix + 6 hex of the title hash (titles are unicode, page names are
    [a-z0-9-]). 'Albert Einstein' -> 'albert-einstein-156ab1'."""
    base = re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", title.lower())).strip("-")[:48].strip("-")
    if not base or not re.match(r"^[a-z0-9]", base):
        base = "a" + base
    return base + "-" + hashlib.sha256(title.encode("utf-8")).hexdigest()[:6]


def _ensure_slugs():
    """One-time backfill: slug -> title for every indexed title, so a page
    name like zetifile/albert-einstein-156ab1 resolves back to its article."""
    db = _db()
    try:
        db.execute("SELECT 1 FROM slugs LIMIT 1")
        return
    except sqlite3.OperationalError:
        pass
    db.execute("CREATE TABLE slugs(s TEXT PRIMARY KEY, t TEXT)")
    cur = db.execute("SELECT t FROM a")
    batch = []
    while True:
        rows = cur.fetchmany(50000)
        if not rows:
            break
        batch = [(slug_for(t), t) for (t,) in rows]
        db.executemany("INSERT OR IGNORE INTO slugs VALUES(?,?)", batch)
    db.commit()


def title_for_slug(slug):
    try:
        row = _db().execute("SELECT t FROM slugs WHERE s = ?", (slug,)).fetchone()
    except sqlite3.OperationalError:
        return None
    return row[0] if row else None


def search(query, limit=8):
    if not available():
        return []
    q = (query or "").strip()
    if not q:
        return []
    db = _db()
    seen, out = set(), []

    def take(rows, why):
        for t, in rows:
            if t.lower() not in seen and len(out) < limit:
                seen.add(t.lower())
                out.append({"title": t, "slug": slug_for(t), "match": why})

    take(db.execute("SELECT t FROM a WHERE tl = ? LIMIT 1", (q.lower(),)), "exact")
    take(db.execute("SELECT t FROM a WHERE tl LIKE ? ORDER BY length(t) LIMIT ?",
                    (q.lower().replace("%", "").replace("_", " ") + "%", limit)), "prefix")
    if len(out) < limit:
        try:
            fts_q = " ".join('"' + w.replace('"', "") + '"'
                             for w in re.findall(r"\w+", q))
            if fts_q:
                take(db.execute(
                    "SELECT t FROM a WHERE rowid IN "
                    "(SELECT rowid FROM a_fts WHERE a_fts MATCH ? ORDER BY rank LIMIT ?)",
                    (fts_q, limit)), "words")
        except sqlite3.OperationalError:
            pass
    return out[:limit]


# --------------------------------------------------------------------------
# Article fetch: seek into the multistream dump, decompress one block
# --------------------------------------------------------------------------
def _stream_at(offset):
    dec = bz2.BZ2Decompressor()
    parts = []
    with open(DUMP, "rb") as fh:
        fh.seek(offset)
        while not dec.eof:
            chunk = fh.read(262144)
            if not chunk:
                break
            parts.append(dec.decompress(chunk))
    return b"".join(parts).decode("utf-8", "replace")


def _raw_wikitext(title, _depth=0):
    row = _db().execute("SELECT t, o FROM a WHERE tl = ? LIMIT 1",
                        (title.lower().replace("_", " "),)).fetchone()
    if not row:
        return None, None
    real_title, offset = row
    try:
        frag = _stream_at(offset)
    except OSError:
        return real_title, None
    m = re.search(r"<page>\s*<title>" + re.escape(html.escape(real_title, quote=False))
                  + r"</title>.*?</page>", frag, re.S)
    if not m:
        return real_title, None
    tm = re.search(r"<text[^>]*>(.*?)</text>", m.group(0), re.S)
    if not tm:
        return real_title, None
    text = html.unescape(tm.group(1))
    rm = re.match(r"#REDIRECT\s*\[\[([^\]|#]+)", text, re.I)
    if rm and _depth < 3:
        return _raw_wikitext(rm.group(1).strip(), _depth + 1)
    return real_title, text


def _strip_wikitext(w):
    w = re.sub(r"<!--.*?-->", "", w, flags=re.S)
    w = re.sub(r"<ref[^>/]*/\s*>", "", w)
    w = re.sub(r"<ref[^>]*>.*?</ref>", "", w, flags=re.S)
    prev = None                                  # nested {{templates}}
    while prev != w:
        prev = w
        w = re.sub(r"\{\{[^{}]*\}\}", "", w)
    prev = None                                  # nested {| tables |}
    while prev != w:
        prev = w
        w = re.sub(r"\{\|(?:[^{|]|\|(?!\})|\{(?!\|))*\|\}", "", w, flags=re.S)
    prev = None                                  # [[File:...]] with nested links
    while prev != w:
        prev = w
        w = re.sub(r"\[\[(?:File|Image|Category)\s*:[^\[\]]*(?:\[\[[^\[\]]*\]\][^\[\]]*)*\]\]",
                   "", w, flags=re.I)
    w = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", w)   # [[target|label]] -> label
    w = re.sub(r"\[\[([^\]]*)\]\]", r"\1", w)            # [[target]] -> target
    w = re.sub(r"\[https?://[^\s\]]*\s*([^\]]*)\]", r"\1", w)
    w = w.replace("'''", "").replace("''", "")
    w = re.sub(r"<[^>]+>", "", w)
    return w


def article(title, max_blocks=400):
    """{title, blocks:[{t:'h', l:2..4, x} | {t:'p', x} | {t:'li', x}]} or None."""
    if not available():
        return None
    real_title, wikitext = _raw_wikitext(title)
    if wikitext is None:
        if not real_title:
            return None
        err = "article body not available"
        row = _db().execute("SELECT o FROM a WHERE tl = ? LIMIT 1",
                            (real_title.lower(),)).fetchone()
        have = os.path.getsize(DUMP) if os.path.exists(DUMP) else 0
        if row and row[0] >= have:
            err = (f"this article sits at byte {row[0]:,} of the corpus but only "
                   f"{have:,} bytes have been downloaded so far — it will appear "
                   f"when the download reaches it")
        return {"title": real_title, "blocks": [], "error": err}
    text = _strip_wikitext(wikitext)
    blocks, para = [], []

    def flush():
        if para:
            p = " ".join(para).strip()
            if len(p) > 2:
                blocks.append({"t": "p", "x": p})
            para.clear()

    for line in text.split("\n"):
        line = line.strip()
        if len(blocks) >= max_blocks:
            break
        hm = re.match(r"^(={2,4})\s*(.+?)\s*=+$", line)
        if hm:
            flush()
            blocks.append({"t": "h", "l": len(hm.group(1)), "x": hm.group(2)})
        elif line.startswith(("*", "#", ";", ":")):
            flush()
            item = line.lstrip("*#;: ").strip()
            if len(item) > 2:
                blocks.append({"t": "li", "x": item})
        elif not line:
            flush()
        else:
            para.append(line)
    flush()
    # drop leading junk before the first real paragraph
    while blocks and blocks[0]["t"] != "p":
        blocks.pop(0)
    return {"title": real_title, "blocks": blocks}


def page_text(title):
    """Render an article as I2 page markup ('# heading', '* bullet', plain
    paragraphs) ready for latentization. Returns (real_title, text) or
    (real_title_or_None, None) when the body is unavailable."""
    a = article(title)
    if not a:
        return None, None
    if a.get("error") and not a.get("blocks"):
        return a.get("title"), None
    lines = ["# " + a["title"], ""]
    for b in a["blocks"]:
        if b["t"] == "h":
            lines.append("# " + b["x"])
            lines.append("")
        elif b["t"] == "li":
            lines.append("* " + b["x"])
        else:
            lines.append(b["x"])
            lines.append("")
    lines.append("")
    lines.append("From ZetiFile, the latent encyclopedia. This article was encoded "
                 "into the latent space the first time it was read; what you decoded "
                 "is the latent, not the source corpus.")
    return a["title"], "\n".join(lines)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "info"
    if cmd == "slugs":
        _ensure_slugs()
        print("slug table ready:",
              _db().execute("SELECT COUNT(*) FROM slugs").fetchone()[0], "slugs")
    elif cmd == "build":
        n = build_index(progress=lambda k: print(f"\r{k:,} titles", end=""))
        print(f"\nindexed {n:,} article titles -> {DB_PATH}")
    elif cmd == "search":
        print(json.dumps(search(" ".join(sys.argv[2:])), indent=2))
    elif cmd == "get":
        a = article(" ".join(sys.argv[2:]))
        if not a:
            print("not found")
        else:
            print(a["title"])
            for b in a["blocks"][:30]:
                line = (("#" * b.get("l", 0) + " " + b["x"]) if b["t"] == "h"
                        else ("- " + b["x"] if b["t"] == "li" else b["x"][:200]))
                print(line.encode(sys.stdout.encoding or "utf-8", "replace")
                      .decode(sys.stdout.encoding or "utf-8"))
    else:
        print(json.dumps(info(), indent=2))

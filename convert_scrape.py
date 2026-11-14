#!/usr/bin/env python3
"""
Convert the scraper output (dataset.json.gz: one JSON list, one record per book) into the
data-contract files that extract_features.py reads:

    books.parquet           book_id, title, author, author_birth, author_death, genre,
                            downloads, n_words, is_anthology, text_path  (+ a few extra columns)
    texts/<book_id>.txt     the book text (gutenberg_text_content)

This is a STOP-GAP so P3 can work before P1's cleaned files exist. P1 owns the real genre
mapping, anthology flag and download counts -- see the NOTES printed at the end.

Usage:
    python convert_scrape.py --input dataset.json.gz --out-dir data
    python convert_scrape.py --input dataset.json.gz --out-dir data --limit 500   # dev kit
"""
import argparse
import gzip
import json
import re
from pathlib import Path

import pandas as pd

# ---- placeholder genre mapping (first match wins; most specific first) -------------------
GENRE_RULES = [
    ("Science Fiction", ["science fiction"]),
    ("Detective / Mystery", ["detective", "mystery", "mysteries"]),
    ("Historical Fiction", ["historical fiction"]),
    ("Love Stories", ["love stories", "love story"]),
    ("Adventure", ["adventure"]),
    ("Poetry", ["poetry", "poems"]),
    ("Short Stories", ["short stories"]),
    ("Fiction", ["fiction"]),           # any remaining "... -- Fiction" subject
]
ANTHOLOGY_TITLE = re.compile(
    r"complete (works|poems|stories|novels)|collected (works|poems|stories)|"
    r"the works of|anthology|selected (works|poems|stories)|\bvolumes? \d+.*\bworks\b", re.I)

_YEARS = re.compile(r",?\s*(?:ca\.\s*)?(\d{3,4})?\??\s*-\s*(?:ca\.\s*)?(\d{3,4})?\??\s*$")


def parse_author(raw):
    """'Still, Henry, 1920-' -> ('Still, Henry', 1920, None)"""
    if not raw:
        return None, None, None
    m = _YEARS.search(raw)
    if m and (m.group(1) or m.group(2)):
        name = raw[: m.start()].strip().rstrip(",")
        b = int(m.group(1)) if m.group(1) else None
        d = int(m.group(2)) if m.group(2) else None
        return name, b, d
    return raw.strip(), None, None


def parse_downloads(raw):
    """'359 downloads in the last 30 days.' -> 359  (NOTE: this is a 30-day count!)"""
    if not raw:
        return None
    m = re.search(r"([\d,]+)\s+downloads?", str(raw))
    return int(m.group(1).replace(",", "")) if m else None


def map_genre(subjects):
    text = " | ".join(s.lower() for s in (subjects or []))
    for genre, keys in GENRE_RULES:
        if any(k in text for k in keys):
            return genre
    return None


def iter_records(path):
    """Stream the JSON list if ijson is installed (low memory), otherwise load it at once."""
    opener = gzip.open if str(path).endswith((".gz", ".tmp")) else open
    try:
        import ijson
        with opener(path, "rb") as fh:
            yield from ijson.items(fh, "item", use_float=True)
        return
    except ImportError:
        pass
    with opener(path, "rt", encoding="utf-8") as fh:
        yield from json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--limit", type=int, default=0, help="stop after N usable books")
    ap.add_argument("--english-only", action="store_true", default=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "texts").mkdir(parents=True, exist_ok=True)

    rows, seen, skipped = [], set(), {"no_text": 0, "not_english": 0, "duplicate": 0, "no_id": 0}
    for rec in iter_records(args.input):
        bid = rec.get("gutenberg_id")
        if bid is None:
            skipped["no_id"] += 1
            continue
        if bid in seen:
            skipped["duplicate"] += 1
            continue
        text = rec.get("gutenberg_text_content")
        if not text or not text.strip():
            skipped["no_text"] += 1
            continue
        lang = rec.get("gutenberg_Language") or ""
        if args.english_only and lang and "english" not in lang.lower():
            skipped["not_english"] += 1
            continue
        seen.add(bid)

        name, birth, death = parse_author(rec.get("gutenberg_Author"))
        title = rec.get("gutenberg_Title") or rec.get("goodreads_title")
        subjects = rec.get("gutenberg_subjects") or []
        path = out / "texts" / f"{bid}.txt"
        path.write_text(text, encoding="utf-8")
        rows.append({
            "book_id": int(bid),
            "title": title,
            "author": name,
            "author_birth": birth,
            "author_death": death,
            "genre": map_genre(subjects),                       # placeholder, P1 owns this
            "downloads": parse_downloads(rec.get("gutenberg_Downloads")),   # 30-day count!
            "n_words": len(text.split()),
            "is_anthology": bool((title and ANTHOLOGY_TITLE.search(title)) or name == "Various"),
            "text_path": str(Path("texts") / f"{bid}.txt"),
            # extras (not in the contract, harmless):
            "subjects": "; ".join(subjects),
            "goodreads_rating": rec.get("goodreads_rating"),
            "goodreads_num_ratings": rec.get("goodreads_num_ratings"),
            "release_date": rec.get("gutenberg_Release Date"),
        })
        if len(rows) % 500 == 0:
            print(f"{len(rows)} books written", flush=True)
        if args.limit and len(rows) >= args.limit:
            break

    df = pd.DataFrame(rows).sort_values("book_id").reset_index(drop=True)
    df.to_parquet(out / "books.parquet", index=False)
    print(f"\nwrote {len(df)} books to {out / 'books.parquet'}  (skipped: {skipped})")
    print("books per placeholder genre:")
    print(df["genre"].fillna("<none>").value_counts().to_string())
    print("\nNOTES\n"
          " * `downloads` is parsed from 'N downloads in the last 30 days' -> a 30-day count,\n"
          "   NOT the all-time total the plan/paper use. Agree with the team what to use.\n"
          " * genre / is_anthology are simple placeholder rules; P1 should replace them.\n"
          " * goodreads_* fields can be matched to the wrong book (see 'Slow Burn'); do not\n"
          "   rely on them without checking.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import gzip
import json
import re
from pathlib import Path

import pandas as pd

GENRE_RULES = [
    ("Science Fiction", ["science fiction"]),
    ("Detective / Mystery", ["detective", "mystery", "mysteries"]),
    ("Historical Fiction", ["historical fiction"]),
    ("Love Stories", ["love stories", "love story"]),
    ("Adventure", ["adventure"]),
    ("Poetry", ["poetry", "poems"]),
    ("Short Stories", ["short stories"]),
    ("Fiction", ["fiction"]),
]
ANTHOLOGY_TITLE = re.compile(
    r"complete (works|poems|stories|novels)|collected (works|poems|stories)|"
    r"the works of|anthology|selected (works|poems|stories)|\bvolumes? \d+.*\bworks\b", re.I)

_YEARS = re.compile(r",?\s*(?:ca\.\s*)?(\d{3,4})?\??\s*-\s*(?:ca\.\s*)?(\d{3,4})?\??\s*$")


def parse_author(raw):
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
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return int(raw)
    m = re.search(r"([\d,]+)", str(raw))
    return int(m.group(1).replace(",", "")) if m else None


def map_genre(subjects):
    text = " | ".join(s.lower() for s in (subjects or []))
    for genre, keys in GENRE_RULES:
        if any(k in text for k in keys):
            return genre
    return None


def iter_records(path):
    """Stream JSONL (.jsonl / .jsonl.gz) or JSON list (.json / .json.gz)."""
    p_str = str(path)
    opener = gzip.open if p_str.endswith((".gz", ".tmp")) else open
    if ".jsonl" in p_str:
        with opener(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
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
        bid = int(bid)
        if bid in seen:
            skipped["duplicate"] += 1
            continue
        text = rec.get("gutenberg_text_content")
        if not text or not text.strip():
            skipped["no_text"] += 1
            continue
        lang = str(rec.get("gutenberg_Language") or "").strip().lower()
        if args.english_only and lang and lang not in ("en", "english") and "english" not in lang:
            skipped["not_english"] += 1
            continue
        seen.add(bid)

        name, birth, death = parse_author(rec.get("gutenberg_Author"))
        title = rec.get("gutenberg_Title") or rec.get("title")
        subjects = rec.get("gutenberg_subjects") or []
        if isinstance(subjects, str):
            subjects = [subjects]

        genre = rec.get("main_category")
        if not genre or str(genre).lower() in ("uncategorized", "none", "nan"):
            genre = map_genre(subjects)

        downloads = parse_downloads(rec.get("work_downloads_sum"))
        if downloads is None:
            downloads = parse_downloads(rec.get("gutenberg_Downloads"))

        path = out / "texts" / f"{bid}.txt"
        path.write_text(text, encoding="utf-8")
        rows.append({
            "book_id": bid,
            "title": title,
            "author": name,
            "author_birth": birth,
            "author_death": death,
            "genre": genre,
            "downloads": downloads,
            "n_words": len(text.split()),
            "is_anthology": bool((title and ANTHOLOGY_TITLE.search(title)) or name == "Various"),
            "text_path": str(Path("texts") / f"{bid}.txt"),
            # Extra P1/P2 columns preserved for P4 & P5:
            "work_id": rec.get("work_id"),
            "canonical_book_id": rec.get("canonical_book_id"),
            "work_downloads_sum": rec.get("work_downloads_sum"),
            "popularity_score": rec.get("popularity_score"),
            "success_category": rec.get("success_category"),
            "edition_count": rec.get("edition_count"),
            "subjects": "; ".join(str(s) for s in subjects),
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
    print("books per genre:")
    print(df["genre"].fillna("<none>").value_counts().to_string())


if __name__ == "__main__":
    main()
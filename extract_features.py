#!/usr/bin/env python3
"""
P3 -- Feature extraction (02807 project, Gutenberg popularity)

Reads the data contract files written by P1:
    books.parquet            book_id, title, author, ..., genre, n_words, is_anthology, text_path
    texts/<book_id>.txt      cleaned text (header/footer removed)

Writes the files P4 reads:
    features_style.parquet   book_id + POS, dependency, readability, sentence statistics
    features_bigrams.npz     scipy CSR matrix of RAW bigram counts (rows = features_style rows)
    vocab.json               list of bigram strings; column j of the matrix = vocab[j]
    feature_schema.json      documented feature schema (groups, definitions)
    features_timing.json     seconds per book (for the Nov 16 checkpoint)

Two subcommands, so it fits an LSF job array:

    extract  -- process ONE shard of books, write checkpointed part files
    merge    -- combine all shards into the final contract files (cheap, run once)

Sampling modes (--mode):
    spread (default)  1000 evenly spaced sentences across the whole body of the book
    first             first 1000 sentences (robustness check from the paper's setup)

Only the standard library, numpy, pandas, pyarrow, scipy and spaCy are needed.
"""
import argparse
import json
import pickle
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
N_SENTENCES = 1000          # sentences per book (plan: 1,000 evenly spaced)
MIN_SENT_WORDS = 3          # sentences shorter than this are skipped when sampling
MAX_SENT_WORDS = 120        # very long "sentences" are usually broken text / lists

# Penn Treebank tags (same tagset as the Stanford tagger used by Ashok et al. 2013).
PENN_TAGS = [
    "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNP",
    "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO",
    "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB",
]
# Dependency labels that mark a subordinate / embedded clause (for "complex sentence")
SUBORD_DEPS = {"advcl", "ccomp", "relcl", "acl", "csubj", "csubjpass"}
# Dependency labels that mark clause coordination / juxtaposition
COORD_CLAUSE_DEPS = {"parataxis"}

ABBREV = {
    "mr", "mrs", "ms", "dr", "st", "sr", "jr", "prof", "capt", "col", "gen", "lt", "sgt",
    "rev", "hon", "messrs", "mme", "mlle", "vs", "etc", "viz", "no", "vol", "ch", "fig",
    "i.e", "e.g", "mt", "ft", "co", "inc", "ltd", "mss", "ms.", "cf", "pp", "ed",
}
_BOUNDARY = re.compile(r"""([.!?]+["'\u201d\u2019)\]]*)\s+(?=["'\u201c\u2018(\[]?[A-Z0-9])""")
_PARA_SPLIT = re.compile(r"\n\s*\n")
_WS = re.compile(r"\s+")
_WORD = re.compile(r"[A-Za-z][A-Za-z'\u2019-]*")
_VOWEL_GROUP = re.compile(r"[aeiouy]+")


_SCENE_BREAK = re.compile(r"^[ \t]*([*\u2022\-=~#][ \t]*){3,}$", re.M)   # "*   *   *   *"
_BRACKET_NOTE = re.compile(r"\[[^\]]{0,600}\]")                         # [Illustration], [Transcriber's Note: ...]


def clean_text(text):
    """Light normalisation of Gutenberg markup (P1 already removes the header/footer)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _BRACKET_NOTE.sub(" ", text)
    text = _SCENE_BREAK.sub("", text)
    text = text.replace("_", "")                    # _italic_ markers
    return text


# ----------------------------------------------------------------------------
# Text -> sentences (cheap, no spaCy: parsing a whole book would be wasteful)
# ----------------------------------------------------------------------------
def split_sentences(text):
    """Paragraph-aware rule-based sentence splitter.

    Gutenberg texts are hard-wrapped, so lines inside a paragraph are joined first,
    otherwise every line break would look like a sentence boundary.
    """
    sentences = []
    for para in _PARA_SPLIT.split(text):
        para = _WS.sub(" ", para).strip()
        if not para:
            continue
        start = 0
        for m in _BOUNDARY.finditer(para):
            end = m.end(1)
            chunk = para[start:end]
            # do not split after abbreviations / single-letter initials ("Mr. Smith", "J. K.")
            last_word = chunk.rstrip(".!?\"'\u201d\u2019)] ").rsplit(" ", 1)[-1].lower()
            terminal = m.group(1)[0]
            if terminal == "." and (last_word in ABBREV or (len(last_word) == 1 and last_word.isalpha())):
                continue
            sentences.append(chunk.strip())
            start = m.end()
        tail = para[start:].strip()
        if tail:
            sentences.append(tail)
    return sentences


def eligible_sentences(sentences):
    out = []
    for s in sentences:
        n = len(_WORD.findall(s))
        if MIN_SENT_WORDS <= n <= MAX_SENT_WORDS:
            out.append(s)
    return out


def pick_sentences(sentences, n, mode):
    if len(sentences) <= n:
        return sentences
    if mode == "first":
        return sentences[:n]
    idx = np.linspace(0, len(sentences) - 1, n).round().astype(int)  # evenly spaced
    return [sentences[i] for i in idx]


# ----------------------------------------------------------------------------
# Readability helpers
# ----------------------------------------------------------------------------
def count_syllables(word):
    w = word.lower().strip("'-\u2019")
    if not w:
        return 0
    n = len(_VOWEL_GROUP.findall(w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(n, 1)


# ----------------------------------------------------------------------------
# Per-book feature extraction
# ----------------------------------------------------------------------------
def book_features(nlp, dep_labels, text, mode):
    """Return (feature_dict, bigram_Counter) for one book, or None if the book is unusable."""
    all_sents = split_sentences(clean_text(text))
    elig = eligible_sentences(all_sents)
    if len(elig) < 20:
        return None
    sample = pick_sentences(elig, N_SENTENCES, mode)

    pos_counts = Counter()
    dep_counts = Counter()
    bigrams = Counter()
    n_tokens = 0                      # all spaCy tokens incl. punctuation
    n_words = 0                       # alphabetic tokens
    n_syll = 0
    n_complex = 0                     # >=3 syllables (for FOG)
    word_chars = 0
    sent_lens = []
    n_simple = n_subord = n_coord_clause = n_quote = n_question = n_excl = 0
    n_commas = n_semicolons = n_colons = n_dashes = 0
    lower_words = []                  # for type-token ratio

    for doc in nlp.pipe(sample, batch_size=256):
        toks = [t for t in doc if not t.is_space]
        if not toks:
            continue
        n_tokens += len(toks)
        sent_words = 0
        alpha_seq = []
        has_subord = has_coord = has_verbal_conj = False
        for t in toks:
            pos_counts[t.tag_] += 1
            dep_counts[t.dep_] += 1
            if t.dep_ in SUBORD_DEPS:
                has_subord = True
            elif t.dep_ in COORD_CLAUSE_DEPS:
                has_coord = True
            elif t.dep_ == "conj" and t.pos_ in ("VERB", "AUX"):
                has_verbal_conj = True
            if t.is_alpha:
                sent_words += 1
                syl = count_syllables(t.text)
                n_syll += syl
                if syl >= 3:
                    n_complex += 1
                word_chars += len(t.text)
                lw = t.lower_
                alpha_seq.append(lw)
                lower_words.append(lw)
            elif t.text == ",":
                n_commas += 1
            elif t.text == ";":
                n_semicolons += 1
            elif t.text == ":":
                n_colons += 1
            elif t.text in ("--", "\u2014", "\u2013"):
                n_dashes += 1
        n_words += sent_words
        sent_lens.append(sent_words)
        for a, b in zip(alpha_seq, alpha_seq[1:]):
            bigrams[a + " " + b] += 1
        n_subord += has_subord
        n_coord_clause += has_coord or has_verbal_conj
        n_simple += not (has_subord or has_coord or has_verbal_conj)
        txt = doc.text
        n_quote += ('"' in txt) or ("\u201c" in txt) or ("\u201d" in txt)
        n_question += txt.rstrip().endswith("?")
        n_excl += txt.rstrip().endswith("!")

    n_sent = len(sent_lens)
    if n_sent < 20 or n_words == 0:
        return None
    sl = np.asarray(sent_lens, dtype=float)
    wps = n_words / n_sent
    spw = n_syll / n_words

    f = {}
    # --- POS (Penn tag share of all tokens; fixed column set) ---
    for tag in PENN_TAGS:
        f[f"pos_{tag}"] = pos_counts.get(tag, 0) / n_tokens
    # --- dependency label share of all tokens (labels from the model, fixed set) ---
    for lab in dep_labels:
        f[f"dep_{lab}"] = dep_counts.get(lab, 0) / n_tokens
    # --- readability (computed on the sampled sentences) ---
    f["read_flesch_ease"] = 206.835 - 1.015 * wps - 84.6 * spw
    f["read_fog"] = 0.4 * (wps + 100.0 * n_complex / n_words)
    f["read_fk_grade"] = 0.39 * wps + 11.8 * spw - 15.59
    f["read_syll_per_word"] = spw
    f["read_frac_complex_words"] = n_complex / n_words
    # --- sentence statistics ---
    f["sent_mean_len"] = float(sl.mean())
    f["sent_std_len"] = float(sl.std())
    f["sent_median_len"] = float(np.median(sl))
    f["sent_p90_len"] = float(np.percentile(sl, 90))
    f["sent_frac_short"] = float((sl < 6).mean())
    f["sent_frac_long"] = float((sl > 30).mean())
    f["sent_frac_simple"] = n_simple / n_sent
    f["sent_frac_subordinate"] = n_subord / n_sent
    f["sent_frac_coord_clause"] = n_coord_clause / n_sent
    f["sent_frac_quoted"] = n_quote / n_sent
    f["sent_frac_question"] = n_question / n_sent
    f["sent_frac_exclaim"] = n_excl / n_sent
    f["sent_commas_per_sent"] = n_commas / n_sent
    f["sent_semicolons_per_1k_tok"] = 1000.0 * n_semicolons / n_tokens
    f["sent_colons_per_1k_tok"] = 1000.0 * n_colons / n_tokens
    f["sent_dashes_per_1k_tok"] = 1000.0 * n_dashes / n_tokens
    f["lex_mean_word_len"] = word_chars / n_words
    ttr_words = lower_words[:10000]    # fixed window so TTR is comparable across books
    f["lex_ttr_10k"] = len(set(ttr_words)) / len(ttr_words)
    # --- bookkeeping (not "style", but useful for P4 sanity checks) ---
    f["meta_n_sentences_total"] = len(all_sents)
    f["meta_n_sentences_eligible"] = len(elig)
    f["meta_n_sentences_used"] = n_sent
    f["meta_n_tokens_used"] = n_tokens
    return f, bigrams


# ----------------------------------------------------------------------------
# Loading books
# ----------------------------------------------------------------------------
def load_book_table(args):
    books = pd.read_parquet(args.books)
    if args.ids_file:
        ids = set(pd.read_csv(args.ids_file, header=None)[0].astype(str))
        books = books[books["book_id"].astype(str).isin(ids)]
    elif not args.all_books:
        # modelling subset: has a genre, not an anthology (plan: model on deduplicated prose
        # works that have a genre; P1 applies the length filter in books.parquet / n_words).
        keep = books["genre"].notna()
        if "is_anthology" in books.columns:
            keep &= ~books["is_anthology"].fillna(False).astype(bool)
        if args.min_words and "n_words" in books.columns:
            keep &= books["n_words"].fillna(0) >= args.min_words
        books = books[keep]
    books = books.sort_values("book_id").reset_index(drop=True)
    if args.limit:
        books = books.iloc[: args.limit]
    return books


def resolve_text_path(row, texts_dir):
    p = row.get("text_path", None)
    if isinstance(p, str) and p:
        cand = Path(p)
        if cand.exists():
            return cand
        cand = Path(texts_dir) / Path(p).name
        if cand.exists():
            return cand
    return Path(texts_dir) / f"{row['book_id']}.txt"


# ----------------------------------------------------------------------------
# extract subcommand
# ----------------------------------------------------------------------------
def cmd_extract(args):
    import spacy

    nlp = spacy.load(args.model, disable=["ner", "lemmatizer"])   # NER off (plan: fallback)
    dep_labels = sorted(nlp.get_pipe("parser").labels)

    books = load_book_table(args)
    n_shards, shard = args.n_shards, args.shard           # shard is 1-based (LSB_JOBINDEX)
    if not 1 <= shard <= n_shards:
        sys.exit(f"--shard must be in 1..{n_shards}")
    books = books.iloc[(shard - 1)::n_shards]              # round-robin: balanced load

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"shard_{shard:04d}"

    done = set()
    next_part = 0
    for p in sorted(out.glob(f"{tag}_part_*.pkl")):         # resume after a killed job
        with open(p, "rb") as fh:
            done.update(str(r["book_id"]) for r in pickle.load(fh))
        next_part = max(next_part, int(p.stem.rsplit("_", 1)[-1]) + 1)
    failed_path = out / f"{tag}_failed.tsv"

    todo = books[~books["book_id"].astype(str).isin(done)]
    print(f"[{tag}] {len(books)} books in shard, {len(done)} already done, {len(todo)} to do",
          flush=True)

    buf, t_shard = [], time.time()
    failed = []

    def flush():
        nonlocal buf, next_part
        if buf:
            tmp = out / f"{tag}_part_{next_part:04d}.pkl.tmp"
            with open(tmp, "wb") as fh:
                pickle.dump(buf, fh, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.rename(out / f"{tag}_part_{next_part:04d}.pkl")   # atomic
            next_part += 1
            buf = []

    for i, (_, row) in enumerate(todo.iterrows(), 1):
        t0 = time.time()
        path = resolve_text_path(row, args.texts_dir)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            res = book_features(nlp, dep_labels, text, args.mode)
        except Exception as e:                                # keep the array job alive
            failed.append((row["book_id"], f"error: {e!r}"))
            continue
        if res is None:
            failed.append((row["book_id"], "too few usable sentences"))
            continue
        feats, bigr = res
        buf.append({"book_id": row["book_id"], "features": feats, "bigrams": dict(bigr),
                    "seconds": time.time() - t0, "dep_labels": dep_labels})
        if len(buf) >= args.checkpoint_every:
            flush()
        if i % 25 == 0:
            el = time.time() - t_shard
            print(f"[{tag}] {i}/{len(todo)}  {el / i:.2f} s/book", flush=True)
    flush()
    if failed:
        with open(failed_path, "a") as fh:
            for bid, why in failed:
                fh.write(f"{bid}\t{why}\n")
    print(f"[{tag}] finished in {time.time() - t_shard:.0f}s, {len(failed)} failed", flush=True)


# ----------------------------------------------------------------------------
# merge subcommand
# ----------------------------------------------------------------------------
def cmd_merge(args):
    import scipy.sparse as sp

    d = Path(args.out_dir)
    records = {}
    for p in sorted(d.glob("shard_*_part_*.pkl")):
        with open(p, "rb") as fh:
            for r in pickle.load(fh):
                records[str(r["book_id"])] = r
    if not records:
        sys.exit(f"no part files found in {d}")
    ids = sorted(records, key=lambda x: (len(x), x))       # stable, numeric-like ordering
    # keep the original dtype of book_id (int if P1 used ints)
    orig_ids = [records[i]["book_id"] for i in ids]
    print(f"merging {len(ids)} books", flush=True)

    # ---- style table ----
    rows = [records[i]["features"] for i in ids]
    style = pd.DataFrame(rows).fillna(0.0)
    style.insert(0, "book_id", orig_ids)
    # fixed, documented column order: pos, dep, read, sent, lex, meta
    order = ["book_id"] + [c for pre in ("pos_", "dep_", "read_", "sent_", "lex_", "meta_")
                           for c in style.columns if c.startswith(pre)]
    style = style[order]
    final = Path(args.final_dir) if args.final_dir else d.parent
    final.mkdir(parents=True, exist_ok=True)
    style.to_parquet(final / "features_style.parquet", index=False)

    # ---- bigram vocabulary: document frequency over the books being modelled ----
    # Label-free (no popularity information), so this is not leakage; P4 still fits TF-IDF in CV.
    df = Counter()
    for i in ids:
        df.update(records[i]["bigrams"].keys())
    kept = [(b, c) for b, c in df.items() if c >= args.min_df]
    kept.sort(key=lambda bc: (-bc[1], bc[0]))
    kept = kept[: args.max_features]
    vocab = sorted(b for b, _ in kept)                      # alphabetical -> reproducible columns
    col = {b: j for j, b in enumerate(vocab)}
    indptr, indices, data = [0], [], []
    for i in ids:
        items = sorted((col[b], c) for b, c in records[i]["bigrams"].items() if b in col)
        indices.extend(j for j, _ in items)
        data.extend(c for _, c in items)
        indptr.append(len(indices))
    X = sp.csr_matrix((np.asarray(data, dtype=np.int32), np.asarray(indices, dtype=np.int32),
                       np.asarray(indptr, dtype=np.int64)), shape=(len(ids), len(vocab)))

    sp.save_npz(final / "features_bigrams.npz", X)
    (final / "vocab.json").write_text(json.dumps(vocab))

    # ---- schema + timing ----
    secs = np.array([records[i]["seconds"] for i in ids])
    dep_labels = records[ids[0]]["dep_labels"]
    schema = {
        "row_order": "features_style.parquet rows == features_bigrams.npz rows, sorted by book_id",
        "n_books": len(ids),
        "sampling": {"mode": args.mode_note, "sentences_per_book": N_SENTENCES,
                     "eligible_sentence_words": [MIN_SENT_WORDS, MAX_SENT_WORDS]},
        "groups": {
            "pos_<TAG>": "Penn Treebank tag share of all tokens (incl. punctuation), spaCy tag_",
            "dep_<label>": "dependency label share of all tokens; labels: " + ", ".join(dep_labels),
            "read_*": "Flesch reading ease, Gunning FOG, Flesch-Kincaid grade, syllables/word, "
                      "share of >=3-syllable words (on sampled sentences)",
            "sent_*": "sentence length mean/std/median/p90, share short(<6)/long(>30), share of "
                      "simple / subordinate / coordinated-clause / quoted / question / exclamation "
                      "sentences, punctuation rates",
            "lex_*": "mean word length, type-token ratio over first 10k sampled words",
            "meta_*": "bookkeeping counts (NOT style features; exclude from the SVM)",
        },
        "bigrams": {"counts": "raw int32 counts of lower-cased alphabetic word bigrams "
                              "within sampled sentences",
                    "min_df_books": args.min_df, "max_features": args.max_features,
                    "n_features": len(vocab)},
    }
    (final / "feature_schema.json").write_text(json.dumps(schema, indent=2))
    timing = {"n_books": len(ids), "mean_s_per_book": float(secs.mean()),
              "median_s_per_book": float(np.median(secs)), "p95_s_per_book": float(np.percentile(secs, 95)),
              "total_cpu_hours": float(secs.sum() / 3600)}
    (final / "features_timing.json").write_text(json.dumps(timing, indent=2))

    failed = []
    for p in d.glob("shard_*_failed.tsv"):
        failed += p.read_text().splitlines()
    (final / "features_failed.tsv").write_text("\n".join(failed))
    print(json.dumps(timing, indent=2))
    print(f"{len(failed)} books failed (see features_failed.tsv); "
          f"{len(style.columns) - 1} style columns, {len(vocab)} bigram features", flush=True)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="process one shard of books")
    e.add_argument("--books", default="books.parquet")
    e.add_argument("--texts-dir", default="texts")
    e.add_argument("--out-dir", default="features_parts", help="where part files go")
    e.add_argument("--mode", choices=["spread", "first"], default="spread")
    e.add_argument("--shard", type=int, default=1, help="1-based; use $LSB_JOBINDEX")
    e.add_argument("--n-shards", type=int, default=1, help="size of the job array")
    e.add_argument("--all-books", action="store_true", help="skip modelling-subset filter")
    e.add_argument("--min-words", type=int, default=0, help="optional extra n_words filter")
    e.add_argument("--ids-file", help="one book_id per line; overrides the subset filter")
    e.add_argument("--limit", type=int, default=0, help="only first N books (sample/test mode)")
    e.add_argument("--checkpoint-every", type=int, default=100)
    e.add_argument("--model", default="en_core_web_sm")
    e.set_defaults(fn=cmd_extract)

    m = sub.add_parser("merge", help="combine all shards into the contract files")
    m.add_argument("--out-dir", default="features_parts", help="directory with part files")
    m.add_argument("--final-dir", default=None, help="where to write contract files "
                                                     "(default: parent of --out-dir)")
    m.add_argument("--min-df", type=int, default=5, help="bigram must occur in >= this many books")
    m.add_argument("--max-features", type=int, default=100000)
    m.add_argument("--mode-note", default="spread", help="recorded in feature_schema.json")
    m.set_defaults(fn=cmd_merge)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

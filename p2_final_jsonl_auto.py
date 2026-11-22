"""DTU 02807 P2 - FULLY AUTOMATIC deduplication for categorized Gutenberg data.

Uses full-text 5-word shingles with MinHash/LSH AND 3-character title/author
shingles. Automatically decides MERGE or KEEP_SEPARATE for every candidate;
no interactive approval, hand-labeling, or manual-review queue.
By default writes a compact audit of only automatic merges, not a review list.

Default: python p2_final_jsonl_auto.py --workers 4
Input: final_data_categorized.jsonl.gz (streamed JSONL; large datasets supported)
Optional: --mode study or --mode benchmark for experimental reports.

IMPORTANT: Automatic deduplication can produce both false merges and missed
matches, especially for translations, anthologies, and alternate titles.
Keep an auditable pair_decisions.csv; do not claim verified ground truth.
The new category field is preserved in unique_works.csv; cross-category
merges are flagged instead of silently resolving genre conflicts.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import re
import tempfile
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import ExitStack
from functools import lru_cache
from itertools import combinations, product
from pathlib import Path
import numpy as np

PRIME = 4294967291
NUM_HASHES = 100
SEED = 42
CONFIGS = (10,20,25,50)
def create_shingles(text: str, k: int = 5) -> set[str]:
    """Return unique word k-grams after simple lowercasing/tokenization."""
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
    return {" ".join(words[i:i+k]) for i in range(len(words) - k + 1)}

def hash_coefficients(num_hashes: int = 100, seed: int = 42) -> list[tuple[int, int]]:
    """Make a reproducible family of universal-hash approximations."""
    rng = random.Random(seed)
    return [(rng.randrange(1, 2**31), rng.randrange(0, PRIME))
            for _ in range(num_hashes)]

def stable_shingle_hash(shingle: str) -> int:
    """Deterministic integer for a shingle; never use Python's random hash()."""
    digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % PRIME

def estimated_jaccard(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> float:
    """Fraction of signature entries that agree."""
    if len(sig_a) != len(sig_b) or not sig_a:
        raise ValueError("Both signatures must have the same nonzero length")
    return sum(x == y for x, y in zip(sig_a, sig_b)) / len(sig_a)

def lsh_candidates(signatures: dict, bands: int = 20) -> set[tuple]:
    """Return candidate ID pairs that share all rows in at least one band."""
    if not signatures:
        return set()

    length = len(next(iter(signatures.values())))
    if length % bands != 0:
        raise ValueError("Signature length must be divisible by number of bands")
    rows = length // bands
    buckets = defaultdict(list)

    for book_id, signature in signatures.items():
        if len(signature) != length:
            raise ValueError("All signatures must have the same length")
        for band in range(bands):
            start = band * rows
            # Band index is part of the key to prevent cross-band collisions.
            buckets[(band, signature[start:start + rows])].append(book_id)

    pairs = set()
    for same_bucket in buckets.values():
        for i in range(len(same_bucket)):
            for j in range(i + 1, len(same_bucket)):
                pairs.add(tuple(sorted((same_bucket[i], same_bucket[j]), key=str)))
    return pairs

def normalize_text(value):
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    value = unicodedata.normalize("NFKD", str(value or ""))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())

def normalize_author(value):
    """Reduce catalog variations such as 'Austen, Jane, 1775-1817'."""
    value = str(value or "")
    value = re.sub(r"\([^)]*\)", "", value)  # parenthetical name expansions
    value = re.sub(r"\b\d{4}\s*[-–]\s*\d{0,4}\b", "", value)
    pieces = [p.strip() for p in value.split(",") if p.strip()]
    # Catalog often stores 'Surname, Given'; move surname to the end.
    # If there are more components, leave the order unchanged rather than guessing.
    if len(pieces) == 2:
        value = f"{pieces[1]} {pieces[0]}"
    else:
        value = " ".join(pieces) if pieces else value
    return normalize_text(value)

def character_shingles(text, k=3):
    """Unique overlapping character k-grams; returns a set."""
    normalized = normalize_text(text)
    if not normalized:
        return set()
    if len(normalized) < k:
        return {normalized}
    return {normalized[i:i+k] for i in range(len(normalized) - k + 1)}

def jaccard_similarity(a, b):
    """Exact Jaccard for sets (no MinHash necessary for short metadata)."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def title_containment(title_a, title_b):
    """Check if one sufficiently informative whole title appears in the other."""
    ta, tb = normalize_text(title_a), normalize_text(title_b)
    if not ta or not tb:
        return False
    short, long = sorted((ta, tb), key=len)
    return len(short) >= 8 and f" {short} " in f" {long} "

def pair_key(a, b):
    return tuple(sorted((str(a), str(b))))

def downloads(value):
    m = re.match(r'^\s*([0-9][0-9,]*)\s*(?:downloads?\b|$)', str(value), re.I)
    return int(m.group(1).replace(',', '')) if m else None

def id_order(x):
    return (0,int(x)) if x.isdigit() else (1,x)

def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def exact_all_pairs_indexed(books, shard_count: int, work_dir: Path, progress_every=50):
    """Exact Jaccard for every possible pair with bounded RAM.

    The shard assignment only decides which temporary file a shingle is placed
    in. The keys used for equality are the ACTUAL shingle strings, so no hash
    collisions can alter the similarity results.
    """
    n = len(books)
    totals = np.zeros(n, dtype=np.int64)
    intersection = np.zeros((n, n), dtype=np.int64)
    files = [work_dir / f"shingles_{i:03}.txt" for i in range(shard_count)]

    print(f"Building exact index for {n} full-length books, {shard_count} disk shards...", flush=True)
    t0 = time.perf_counter()
    with ExitStack() as stack:
        handles = [stack.enter_context(p.open("w", encoding="utf-8", buffering=262144)) for p in files]
        for index, book in enumerate(books):
            shingles = create_shingles(text_of(book), k=5)
            totals[index] = len(shingles)
            buffers = [[] for _ in range(shard_count)]
            for shingle in shingles:
                bucket = hash(shingle) % shard_count
                buffers[bucket].append(f"{index}\t{shingle}\n")
            for h, chunk in zip(handles, buffers):
                if chunk:
                    h.write("".join(chunk))
            del shingles, buffers
            if (index + 1) % progress_every == 0 or index + 1 == n:
                print(f"  Indexed {index + 1}/{n} books", flush=True)
    build_time = time.perf_counter() - t0

    print("Counting shared 5-word shingles across books...", flush=True)
    t1 = time.perf_counter()
    for shard, path in enumerate(files, 1):
        posting_masks = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                pos = line.index("\t")
                idx = int(line[:pos])
                shingle = line[pos + 1:-1]
                posting_masks[shingle] = posting_masks.get(shingle, 0) | (1 << idx)

        # Many shingles occur in the same subset of books. Combine those masks
        # so intersections can be updated in larger integer steps.
        repeated_masks = Counter(
            mask for mask in posting_masks.values() if mask.bit_count() > 1
        )
        del posting_masks
        for mask, shared_shingle_count in repeated_masks.items():
            ids = []
            while mask:
                lowest_bit = mask & -mask
                ids.append(lowest_bit.bit_length() - 1)
                mask ^= lowest_bit
            for i, j in combinations(ids, 2):
                intersection[i, j] += shared_shingle_count

        # Release disk space as each shard finishes.
        path.unlink()
        if shard % max(1, shard_count // 8) == 0 or shard == shard_count:
            print(f"  Counted {shard}/{shard_count} shards", flush=True)

    count_time = time.perf_counter() - t1
    upper_i, upper_j = np.triu_indices(n, k=1)
    common = intersection[upper_i, upper_j]
    union = totals[upper_i] + totals[upper_j] - common
    scores = np.divide(
        common.astype(np.float64), union,
        out=np.zeros(len(common), dtype=np.float64), where=union > 0
    )
    print(f"Exact all-pairs analysis: {len(scores):,} scores in {build_time+count_time:.2f}s")
    return upper_i, upper_j, scores, totals, {
        "build_shingle_index_s": round(build_time, 3),
        "count_shared_shingles_s": round(count_time, 3),
        "total_exact_all_pairs_s": round(build_time + count_time, 3),
    }

# ---------------------------------------------------------------------------
# 1. Inputs, deterministic MinHash, multiprocessing and signature caching
# ---------------------------------------------------------------------------
COEFF = hash_coefficients(NUM_HASHES, SEED)
A = np.array([a for a,b in COEFF], dtype=np.uint64)[:,None]
B = np.array([b for a,b in COEFF], dtype=np.uint64)[:,None]


def text_of(book):
    if 'text' in book:
        return book['text']
    return Path(book['text_path']).read_text(encoding='utf-8-sig', errors='replace')


def iter_original_records(path):
    """Stream Gutenberg books from JSONL.gz (one JSON object per line).

    Also accepts .json.gz containing a JSON array, but requires ijson to stream
    that format without loading thousands of books into memory.
    """
    opener = gzip.open if str(path).endswith('.gz') else open
    # Detect the actual structure rather than relying exclusively on filename.
    with opener(path, 'rt', encoding='utf-8-sig') as probe:
        first = ''
        while not first:
            line = probe.readline()
            if not line:
                return
            first = line.lstrip()[:1]
    if first == '[':
        try:
            import ijson
        except ImportError as exc:
            raise ImportError(
                'Input appears to be a JSON array. Install ijson: python -m pip install ijson'
            ) from exc
        with opener(path, 'rb') as stream:
            yield from ijson.items(stream, 'item')
        return
    with opener(path, 'rt', encoding='utf-8-sig') as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f'Line {lineno:,} is not valid JSONL; expected one JSON object per line. '
                    'If your input is a JSON array, rename is not necessary; the detector '
                    'should handle it automatically.'
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f'Line {lineno:,} is not a book JSON object')
            yield record


def load_books(path, limit=None, stream_dir=None):
    """Keep small metadata in RAM and full book texts on disk for 10k+ books."""
    books = {}
    if path.suffix == '.parquet':
        import pandas as pd
        frame = pd.read_parquet(path)
        if not {'book_id','title','author','text_path'}.issubset(frame.columns):
            raise ValueError('Parquet requires book_id, title, author, text_path columns')
        for r in frame.to_dict('records'):
            bid = str(r['book_id'])
            if bid in books:
                raise ValueError(f'Duplicate ID: {bid}')
            text_path = Path(str(r['text_path']))
            if not text_path.is_absolute():
                text_path = path.parent / text_path
            books[bid] = dict(book_id=bid, title=str(r['title']) if pd.notna(r['title']) else '',
                author=str(r['author']) if pd.notna(r['author']) else '',
                text_path=str(text_path), downloads=downloads(r.get('downloads')),
                category=str(r.get('category') or 'Uncategorized'),
                is_anthology=bool(pd.notna(r.get('is_anthology')) and r.get('is_anthology') in (True,1,'true','True')))
            if limit and len(books) >= limit:
                break
    else:
        if stream_dir is None:
            raise ValueError('Stream directory is required for compressed book datasets')
        # Stat-based cache location. Different versions of the dataset won't share texts.
        stat = path.stat()
        stamp = hashlib.sha256(f'{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}'.encode()).hexdigest()[:24]
        stream_dir = stream_dir / stamp
        stream_dir.mkdir(parents=True, exist_ok=True)
        for record in iter_original_records(path):
            if record.get('gutenberg_id') is None:
                continue
            bid = str(record['gutenberg_id'])
            if bid in books:
                raise ValueError(f'Duplicate Gutenberg ID in input: {bid}')
            raw_text = record.get('gutenberg_text_content_clean') or record.get('gutenberg_text_content') or ''
            raw_text = raw_text if isinstance(raw_text, str) else ''
            dest = stream_dir / (hashlib.sha256(bid.encode()).hexdigest() + '.txt')
            if not dest.exists() or dest.stat().st_size == 0:
                tmp = dest.with_suffix('.tmp')
                tmp.write_text(raw_text, encoding='utf-8')
                tmp.replace(dest)
            books[bid] = dict(
                book_id=bid,
                title=str(record.get('gutenberg_Title') or record.get('title') or ''),
                author=str(record.get('gutenberg_Author') or record.get('author') or ''),
                downloads=downloads(record.get('gutenberg_Downloads', record.get('downloads'))),
                category=str(record.get('category') or record.get('gutenberg_main_category') or 'Uncategorized'),
                is_anthology=record.get('is_anthology') in (True,1,'true','True'),
                text_path=str(dest),
            )
            if len(books) % 1000 == 0:
                print(f'  Read {len(books):,} books', flush=True)
            if limit and len(books) >= limit:
                break
    if not books:
        raise ValueError('No Gutenberg books loaded. Check JSONL keys and compression.')
    return books


def signature(task):
    bid,text,k=task
    shingles=create_shingles(text,k)
    if not shingles: return bid,None
    vals=np.fromiter((stable_shingle_hash(s) for s in shingles),dtype=np.uint64,count=len(shingles))
    del shingles
    minima=np.full(NUM_HASHES,PRIME,dtype=np.uint64)
    for start in range(0,len(vals),8192):
        np.minimum(minima,((A*vals[None,start:start+8192]+B)%PRIME).min(axis=1),out=minima)
    return bid,tuple(map(int,minima))


def compute_signatures(books,k,workers):
    tasks=iter((i,text_of(b),k) for i,b in books.items())
    result={}; done_count=0
    def consume(item):
        nonlocal done_count
        i,sig=item
        if sig is not None: result[i]=sig
        done_count+=1
        if done_count%100==0 or done_count==len(books):
            print(f'  k={k}: {done_count}/{len(books)} signatures',flush=True)
    if workers==1:
        for item in map(signature,tasks): consume(item)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending=set()
            def submit():
                try: task=next(tasks)
                except StopIteration: return
                pending.add(pool.submit(signature,task))
            for _ in range(2*workers): submit()
            while pending:
                done,pending=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    consume(future.result()); submit()
    return result


def identity(path,books,k):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    # Manifest mode also detects changed text-file sizes and timestamps.
    for i,b in books.items():
        h.update(i.encode()+b'\0')
        if 'text_path' in b:
            st=Path(b['text_path']).stat()
            h.update(f'{st.st_size}:{st.st_mtime_ns}'.encode())
    return dict(version='p2-automatic-categorized-v2',digest=h.hexdigest(),k=k,hashes=NUM_HASHES,seed=SEED)


def atomic_json(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    tmp.replace(path)


def cached_signatures(books,args,k):
    ident=identity(args.dataset,books,k)
    base=args.output/'cache'; base.mkdir(parents=True,exist_ok=True)
    def part(idx): return base/f"{ident['digest'][:20]}_k{k}_n{args.shards}_part{idx}.json"
    def load(idx):
        obj=json.loads(part(idx).read_text(encoding='utf-8'))
        if obj['identity']!=ident: raise ValueError('Signature cache identity mismatch')
        if obj['shard']!=idx or obj['shards']!=args.shards: raise ValueError('Wrong shard')
        return {i:tuple(s) for i,s in obj['signatures'].items()}
    if args.mode=='signatures' or args.shards==1:
        idx=args.shard_index if args.mode=='signatures' else 0
        if part(idx).exists() and not args.refresh_cache:
            print(f'Cache hit: {part(idx).name}'); return load(idx)
        subset={i:b for j,(i,b) in enumerate(books.items()) if j%args.shards==idx}
        result=compute_signatures(subset,k,args.workers)
        atomic_json(part(idx),dict(identity=ident,shard=idx,shards=args.shards,signatures=result))
        return result
    result={}
    for idx in range(args.shards):
        if not part(idx).exists(): raise FileNotFoundError(f'Missing signature shard {part(idx)}; run --mode signatures first')
        data=load(idx)
        if result.keys() & data.keys(): raise ValueError('Duplicate IDs across shards')
        result.update(data)
    print(f'Combined {args.shards} signature shards: {len(result)} books')
    return result

# ---------------------------------------------------------------------------
# 2. Metadata and text candidate retrieval; reviewed edition grouping
# ---------------------------------------------------------------------------
def metadata_pairs(books,tk,ak,tt,at):
    eligible={i:b for i,b in books.items() if normalize_text(b['title']) and normalize_author(b['author']) not in UNKNOWN_AUTHORS and normalize_author(b['author'])}
    ts={i:character_shingles(b['title'],tk) for i,b in eligible.items()}
    au={i:character_shingles(normalize_author(b['author']),ak) for i,b in eligible.items()}
    # Inverted author index: at a positive author threshold every match must
    # share an author shingle. Verify exact Jaccard, no author-pair approximation.
    index=defaultdict(list); title_index=defaultdict(list); pairs=set(); seen=set()
    for i in eligible:
        neighbors=set().union(*(index[s] for s in au[i])) if au[i] else set()
        if at==0: neighbors=set(indexed for indexed in seen)
        if tt>0 and tk<=8:
            neighbors &= set().union(*(title_index[s] for s in ts[i])) if ts[i] else set()
        for j in neighbors:
            if jaccard_similarity(au[i],au[j])>=at and (title_containment(eligible[i]['title'],eligible[j]['title']) or jaccard_similarity(ts[i],ts[j])>=tt):
                pairs.add(pair_key(i,j))
        for s in au[i]: index[s].append(i)
        for s in ts[i]: title_index[s].append(i)
        seen.add(i)
    return pairs


GROUP_FIELDS=['book_id','work_id','canonical_book_id']
WORK_FIELDS=['work_id','canonical_book_id','title','author','category','categories_in_group','category_conflict','edition_count','edition_ids','downloads_sum','downloads_known_sum','missing_download_counts']


def pair_details(books,signatures,pairs,text_pairs,meta_pairs,args):
    @lru_cache(maxsize=6)
    def sh(i): return create_shingles(text_of(books[i]),args.word_size)
    rows=[]
    for a,b in sorted(pairs):
        x,y=books[a],books[b]
        sa,sb=sh(a),sh(b)
        score=jaccard_similarity(sa,sb) if sa and sb else None
        rows.append(dict(book_id_a=a,title_a=x['title'],author_a=x['author'],book_id_b=b,title_b=y['title'],author_b=y['author'],
            source='+'.join(name for name,ps in [('text_lsh',text_pairs),('metadata',meta_pairs)] if (a,b) in ps) or 'metadata',
            full_text_jaccard=score,minhash_similarity=estimated_jaccard(signatures[a],signatures[b]) if a in signatures and b in signatures else None,
            title_jaccard=jaccard_similarity(character_shingles(x['title'],args.title_size),character_shingles(y['title'],args.title_size)),
            author_jaccard=jaccard_similarity(character_shingles(normalize_author(x['author']),args.author_size),character_shingles(normalize_author(y['author']),args.author_size)),
            title_contained=int(title_containment(x['title'],y['title'])),high_text_similarity=int(score is not None and score>=args.threshold)))
    return rows


GENERIC_TITLES={'poems','poetry','stories','short stories','letters','essays','works','selected works','collected works','novels','tales','the works','a novel','the novel','the book','memoirs','diary','the diary','journal','the journal','sermons','plays','volume','part','unknown','untitled'}
COLLECTION_RE=re.compile(r'\b(?:antholog(?:y|ies)|collected|collection|complete works|selected works|miscellany|omnibus|vol(?:ume)?\.?\s+(?:[0-9]+|[ivxlcdm]+)|part\s+(?:[0-9]+|[ivxlcdm]+)|book\s+(?:[0-9]+|[ivxlcdm]+)|from\s+|[0-9]+\s+stories)\b',re.I)
UNKNOWN_AUTHORS={'unknown','anonymous','various','various authors','unidentified','anon'}


def explicit_alias(a,b):
    """One title matches a complete parenthetical alternate title, not a fragment."""
    for short,long in [(a,b),(b,a)]:
        short=normalize_text(short)
        if len(short)<8 or short in GENERIC_TITLES: continue
        for alias in re.findall(r'\(([^()]*)\)',str(long)):
            alias=normalize_text(alias)
            if alias.startswith(('from ','volume ','part ','book ')): continue
            if alias==short or (alias.startswith('the ') and alias[4:]==short) or (short.startswith('the ') and short[4:]==alias): return True
    return False


def title_token_similarity(a, b):
    """Token overlap for distinguishing minor punctuation changes from series titles."""
    a, b = set(normalize_text(a).split()), set(normalize_text(b).split())
    return len(a & b) / len(a | b) if a and b else 0.0


def automatic_decision(row, books, args):
    """Decide automatically and conservatively: MERGE or KEEP_SEPARATE.

    A low text score is not a rejection when metadata provides exact author and
    complete alternate-title evidence: translated editions can share few words.
    """
    a, b = books[row['book_id_a']], books[row['book_id_b']]
    aa, ab = normalize_author(a['author']), normalize_author(b['author'])
    ta, tb = normalize_text(a['title']), normalize_text(b['title'])
    author_similarity = float(row['author_jaccard'])
    same_author = bool(aa and aa == ab and aa not in UNKNOWN_AUTHORS)
    author_compatible = same_author or (aa and ab and author_similarity >= .95 and aa not in UNKNOWN_AUTHORS and ab not in UNKNOWN_AUTHORS)
    title_sim = float(row['title_jaccard'])
    text_sim = row['full_text_jaccard']
    exact_title = bool(ta and ta == tb and len(ta) >= 10 and ta not in GENERIC_TITLES)
    alias = explicit_alias(a['title'], b['title'])
    # Common collections and anthology extracts must not be silently joined.
    collection = (a['is_anthology'] or b['is_anthology'] or
                  any(COLLECTION_RE.search(x['title']) for x in (a, b)))
    if collection:
        if text_sim is not None and text_sim >= .98 and author_compatible and exact_title:
            return 'merge', 'Almost identical text, author and title despite collection indicator'
        return 'keep_separate', 'Collection, excerpt, part or volume safeguard'
    if text_sim is not None and text_sim >= .98 and (author_compatible or title_sim >= .90):
        return 'merge', 'Near-identical full text plus supporting metadata'
    if author_compatible and text_sim is not None and text_sim >= args.auto_text_threshold and (title_sim >= .50 or exact_title or alias or row['title_contained']):
        return 'merge', 'High full-text Jaccard with compatible author and title'
    if same_author and exact_title:
        return 'merge', 'Exact distinctive title and author (translations included)'
    if same_author and alias:
        return 'merge', 'Explicit alternate title and same author (translations included)'
    if same_author and min(len(ta),len(tb)) >= 16 and title_sim >= .92 and title_token_similarity(ta,tb) >= .85:
        return 'merge', 'Near-identical title and same author'
    return 'keep_separate', 'Insufficient evidence of same literary work'


def group_works(books, rows, signatures, args):
    """Union automatic matches without allowing unsupported transitive chains."""
    parent={i:i for i in books}
    members={i:{i} for i in books}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def merge(a,b):
        a,b=find(a),find(b)
        if a!=b:
            lo,hi=sorted((a,b),key=id_order)
            parent[hi]=lo
            members[lo].update(members.pop(hi))
    for r in rows:
        r['decision'], r['decision_reason'] = automatic_decision(r,books,args)
    positive={pair_key(r['book_id_a'],r['book_id_b']) for r in rows if r['decision']=='merge'}
    # For a multi-edition cluster, every cross-pair needs supporting evidence.
    # Without this check, a chain of merely similar books could collapse many works.
    for r in sorted((r for r in rows if r['decision']=='merge'),
                    key=lambda r:(-(r['full_text_jaccard'] or 0),r['book_id_a'],r['book_id_b'])):
        a,b=find(r['book_id_a']),find(r['book_id_b'])
        if a==b: continue
        if all(pair_key(x,y) in positive for x in members[a] for y in members[b]):
            merge(a,b)
        else:
            r['decision']='keep_separate'
            r['decision_reason']='Conservative group-consistency safeguard'
    groups=defaultdict(list)
    for i in books:
        groups[find(i)].append(i)
    membership=[]
    works=[]
    for wid,group in sorted(groups.items(),key=lambda p:id_order(p[0])):
        group.sort(key=id_order)
        canonical=next((i for i in group if i in signatures and not books[i]['is_anthology']),group[0])
        for i in group:
            membership.append(dict(book_id=i,work_id=wid,canonical_book_id=canonical))
        counts=[books[i]['downloads'] for i in group]
        known=[x for x in counts if x is not None]
        categories=sorted({books[i].get('category','Uncategorized') for i in group})
        works.append(dict(work_id=wid,canonical_book_id=canonical,title=books[canonical]['title'],author=books[canonical]['author'],category=books[canonical].get('category','Uncategorized'),categories_in_group=';'.join(categories),category_conflict=int(len(categories)>1),edition_count=len(group),edition_ids=';'.join(group),
                          downloads_sum=sum(known) if len(known)==len(counts) else None,
                          downloads_known_sum=sum(known) if known else None,
                          missing_download_counts=len(counts)-len(known)))
    return membership,works


AUTO_DECISION_FIELDS=['book_id_a','title_a','author_a','book_id_b','title_b','author_b',
                      'source','full_text_jaccard','minhash_similarity','title_jaccard',
                      'author_jaccard','title_contained','high_text_similarity','decision','decision_reason']


def export_groups(books, signatures, rows, text_pairs, meta_pairs, args):
    membership, works=group_works(books,rows,signatures,args)
    audit=[{k:r.get(k,'') for k in AUTO_DECISION_FIELDS} for r in rows]
    if args.audit:
        write_csv(args.output/'pair_decisions.csv', AUTO_DECISION_FIELDS, audit)
    write_csv(args.output/'automatic_matches.csv',AUTO_DECISION_FIELDS,[r for r in audit if r['decision']=='merge'])
    write_csv(args.output/'edition_groups.csv',GROUP_FIELDS,membership)
    write_csv(args.output/'unique_works.csv',WORK_FIELDS,works)
    parquet=False
    try:
        import pandas as pd
        frame=pd.DataFrame(membership)
        if all(i.isdigit() for i in books):
            frame=frame.astype({c:'int64' for c in GROUP_FIELDS})
        frame.to_parquet(args.output/'edition_groups.parquet',index=False)
        parquet=True
    except ImportError:
        print('Parquet not written; install pandas and pyarrow if P1 needs it.')
    multi=[w for w in works if w['edition_count']>1]
    stats=dict(original_records=len(books),unique_works=len(works),editions_removed=len(books)-len(works),
               multi_edition_groups=len(multi),full_text_candidates=len(text_pairs),
               metadata_candidates=len(meta_pairs),union_candidates=len(text_pairs|meta_pairs),
               automatic_merged_pairs=sum(r['decision']=='merge' for r in audit),
               automatically_kept_separate=sum(r['decision']=='keep_separate' for r in audit),
               possible_comparisons=len(books)*(len(books)-1)//2,
               missing_download_counts=sum(b['downloads'] is None for b in books.values()),
               category_distribution=dict(Counter(b.get('category','Uncategorized') for b in books.values())),
               cross_category_merged_groups=sum(bool(w['category_conflict']) for w in works),
               total_known_downloads_before=sum(b['downloads'] or 0 for b in books.values()),
               total_known_downloads_after=sum(w['downloads_known_sum'] or 0 for w in works),
               parquet_written=parquet,
               parameters=dict(word_size=args.word_size,title_size=args.title_size,
                               author_size=args.author_size,minhash_values=NUM_HASHES,
                               bands=args.bands,rows=NUM_HASHES//args.bands,
                               title_threshold=args.title_threshold,author_threshold=args.author_threshold,
                               auto_text_threshold=args.auto_text_threshold),
               caveat='All decisions are automatic predictions, NOT verified ground truth. Conservative rules keep uncertain pairs separate. Summed downloads refer to records, not unique readers.')
    assert stats['total_known_downloads_before']==stats['total_known_downloads_after']
    atomic_json(args.output/'summary.json',stats)
    print(json.dumps(stats,indent=2))
    return stats

# 3. Evaluation (text similarity and human same-work labels are separate).
def metrics(candidates,positive,negative):
    tp=len(candidates&positive); fp=len(candidates&negative); fn=len(positive-candidates); tn=len(negative-candidates)
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,f1=2*tp/(2*tp+fp+fn) if tp+fp and tp+fn else None)


def study(books,args,labels):
    if args.shards!=1: raise ValueError('Run study locally with --shards 1')
    rng=random.Random(SEED); ids=list(books); rng.shuffle(ids)
    copies={}; expected=[]; chosen=0
    for i in ids:
        txt=text_of(books[i]); lines=txt.splitlines(keepends=True)
        if len(lines)<100 or len(txt)<10000: continue
        copies[i]={'text':txt}; order=list(range(len(lines))); rng.shuffle(order)
        for pct in (10,20,30):
            removed=set(order[:round(len(lines)*pct/100)]); sid=f'synthetic:{i}:{pct}'
            copies[sid]={'text':''.join(line for j,line in enumerate(lines) if j not in removed)}
            expected.append((i,sid,pct))
        chosen+=1
        if chosen>=args.synthetic_books: break
    pos={p for p,v in labels.items() if v[0]=='1'}; neg={p for p,v in labels.items() if v[0]=='0'}
    text_rows=[]; detail=[]; meta_rows=[]
    for k in sorted(set(args.word_sizes+[args.word_size])):
        sig=cached_signatures(books,args,k)
        synth=compute_signatures(copies,k,args.workers) if copies else {}
        for b in CONFIGS:
            c=lsh_candidates(sig,b); hits=0
            for i,j,pct in expected:
                found=bool(i in synth and j in synth and lsh_candidates({i:synth[i],j:synth[j]},b))
                hits+=found
                detail.append(dict(word_size=k,bands=b,book_id=i,deleted_lines_pct=pct,detected=int(found)))
            text_rows.append(dict(word_size=k,bands=b,rows=100//b,midpoint=(1-0.5**(1/b))**(b/100),original_candidates=len(c),synthetic_found=hits,synthetic_total=len(expected),synthetic_recall=hits/len(expected) if expected else None,**metrics(c,pos,neg)))
            print(f'k={k}, bands={b}: synthetic recall {hits}/{len(expected)}, original candidates {len(c)}',flush=True)
    for t,a in product(args.title_sizes,args.author_sizes):
        c=metadata_pairs(books,t,a,args.title_threshold,args.author_threshold)
        meta_rows.append(dict(title_size=t,author_size=a,candidates=len(c),**metrics(c,pos,neg)))
    write_csv(args.output/'text_parameter_study.csv',list(text_rows[0]),text_rows)
    write_csv(args.output/'metadata_parameter_study.csv',list(meta_rows[0]),meta_rows)
    write_csv(args.output/'synthetic_pairs.csv',['word_size','bands','book_id','deleted_lines_pct','detected'],detail)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        x=np.linspace(0,1,501)
        for b in CONFIGS: plt.plot(x,1-(1-x**(100//b))**b,label=f'{b} bands x {100//b} rows')
        plt.xlabel('Jaccard similarity'); plt.ylabel('Candidate probability'); plt.legend(); plt.tight_layout(); plt.savefig(args.output/'lsh_s_curves.png',dpi=180); plt.close()
    except ImportError: print('Install matplotlib to generate the S-curve figure.')
    print('Synthetic recall is not real-edition recall. Reviewed pairs are not a random corpus sample.')


def benchmark(books,signatures,args):
    if args.word_size!=5: raise ValueError('Benchmark uses 5-word shingles')
    ids=[i for i in books if i in signatures]
    if len(ids)>2000: ids=random.Random(SEED).sample(ids,2000)
    if len(ids)<2: raise ValueError('Need two usable books')
    subset=[books[i] for i in ids]
    with tempfile.TemporaryDirectory(prefix='exact_',dir=args.output) as temp:
        ii,jj,scores,totals,timing=exact_all_pairs_indexed(subset,args.disk_shards,Path(temp))
    sample=[create_shingles(text_of(b),5) for b in subset[:10]]
    for i,j in combinations(range(len(sample)),2):
        idx=i*(2*len(ids)-i-1)//2+j-i-1
        assert abs(jaccard_similarity(sample[i],sample[j])-scores[idx])<1e-12
    truth={pair_key(ids[int(ii[x])],ids[int(jj[x])]) for x in np.flatnonzero(scores>=args.threshold)}
    all_sig={i:signatures[i] for i in ids}; rows=[]
    for b in CONFIGS:
        t=time.perf_counter(); c=lsh_candidates(all_sig,b); elapsed=time.perf_counter()-t
        tp=len(c&truth); fp=len(c-truth); fn=len(truth-c)
        rows.append(dict(books=len(ids),bands=b,rows=100//b,possible_pairs=len(scores),threshold=args.threshold,exact_positive_pairs=len(truth),lsh_candidates=len(c),tp=tp,fp=fp,fn=fn,precision=tp/len(c) if c else None,recall=tp/len(truth) if truth else None,indexed_exact_seconds=timing['total_exact_all_pairs_s'],lsh_banding_seconds=elapsed,comparison_reduction=1-len(c)/len(scores)))
    write_csv(args.output/'bruteforce_comparison.csv',list(rows[0]),rows)
    def describe(x):
        a,b=ids[int(ii[x])],ids[int(jj[x])]
        return dict(book_id_a=a,title_a=books[a]['title'],book_id_b=b,title_b=books[b]['title'],exact_jaccard=float(scores[x]))
    fields=['book_id_a','title_a','book_id_b','title_b','exact_jaccard']
    write_csv(args.output/'most_similar_pairs.csv',fields,[describe(x) for x in np.argsort(-scores)[:50]])
    write_csv(args.output/'pairs_above_threshold.csv',fields,[describe(x) for x in np.flatnonzero(scores>=args.threshold)])
    atomic_json(args.output/'benchmark_summary.json',dict(books=len(ids),possible_pairs=len(scores),max_jaccard=float(max(scores)),exact_positive_pairs=len(truth),exact_timing=timing,naive_validation_pairs=len(sample)*(len(sample)-1)//2,note='Exact all-pairs index vs LSH candidate retrieval; banding time excludes signature construction and verification. No positives means undefined recall.'))
    print(f'Exact comparison: {len(ids)} books, {len(scores)} pairs, highest Jaccard={max(scores):.5f}')




def export_categorized_works_json(books, args):
    """Export a streaming JSONL.gz dataset with ONE record per inferred work.

    Keeps every original field, including category and the canonical text.
    Original Gutenberg download counts are unchanged. Use work_downloads_sum
    as the work-level label input (can be null if a count was missing).
    """
    if args.dataset.suffix == '.parquet':
        print('Skipping JSONL export for Parquet input')
        return
    with (args.output / 'edition_groups.csv').open(encoding='utf-8-sig', newline='') as f:
        groups = list(csv.DictReader(f))
    with (args.output / 'unique_works.csv').open(encoding='utf-8-sig', newline='') as f:
        works = {r['work_id']: r for r in csv.DictReader(f)}
    canonical = {g['canonical_book_id']: works[g['work_id']] for g in groups}
    destination = args.output / 'deduplicated_final_data_categorized.jsonl.gz'
    count = 0
    with gzip.open(destination, 'wt', encoding='utf-8', compresslevel=5) as out:
        for record in iter_original_records(args.dataset):
            book_id = str(record.get('gutenberg_id', ''))
            work = canonical.get(book_id)
            if work is None:
                continue
            record['work_id'] = work['work_id']
            record['canonical_book_id'] = work['canonical_book_id']
            record['edition_count'] = int(work['edition_count'])
            record['edition_ids'] = work['edition_ids'].split(';')
            record['work_downloads_sum'] = int(work['downloads_sum']) if work['downloads_sum'] else None
            record['work_downloads_known_sum'] = int(work['downloads_known_sum']) if work['downloads_known_sum'] else None
            record['categories_in_group'] = work['categories_in_group'].split(';')
            record['category_conflict'] = bool(int(work['category_conflict']))
            out.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
            count += 1
            if count % 1000 == 0:
                print(f'  Exported {count:,} unique work records', flush=True)
    if count != len(works):
        raise AssertionError(f'Expected {len(works):,} works, wrote {count:,}')
    print(f'Exported {count:,} unique works to {destination}', flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    root=Path(__file__).resolve().parent
    p.add_argument('--mode',choices=['run','study','benchmark','signatures'],default='run')
    p.add_argument('--dataset',type=Path,default=root/'final_data_categorized.jsonl.gz',help='Default: final_data_categorized.jsonl.gz (streamed line by line)')
    p.add_argument('--output',type=Path,default=root/'p2_final_data_results')
    p.add_argument('--workers',type=int,default=min(4,os.cpu_count() or 1))
    p.add_argument('--word-size',type=int,default=5); p.add_argument('--title-size',type=int,default=3); p.add_argument('--author-size',type=int,default=3)
    p.add_argument('--bands',type=int,default=25); p.add_argument('--title-threshold',type=float,default=.30); p.add_argument('--author-threshold',type=float,default=.90)
    p.add_argument('--threshold',type=float,default=.80,help='Exact text verification flag / benchmark cutoff; never an automatic merge')
    p.add_argument('--auto-text-threshold',type=float,default=.80)
    p.add_argument('--no-export-json',action='store_true',help='Skip deduplicated JSON.gz export (default: export automatically)')
    p.add_argument('--audit',action='store_true',help='Optional: write automatic merge/keep decisions for auditing (no manual review)')
    p.add_argument('--stream-json',action='store_true',help='Kept for compatibility: JSONL input automatically streams to disk')
    p.add_argument('--limit',type=int,help='First N records; automatically separate output directory')
    p.add_argument('--refresh-cache',action='store_true'); p.add_argument('--word-sizes',nargs='+',type=int,default=[3,5,7]); p.add_argument('--title-sizes',nargs='+',type=int,default=[2,3,4,5]); p.add_argument('--author-sizes',nargs='+',type=int,default=[2,3,4,5]); p.add_argument('--synthetic-books',type=int,default=8)
    p.add_argument('--shards',type=int,default=1,help='Number of distributed signature partitions'); p.add_argument('--shard-index',type=int,default=0,help='Zero-based index for --mode signatures'); p.add_argument('--disk-shards',type=int,default=64)
    args=p.parse_args()
    if min([args.workers,args.word_size,args.title_size,args.author_size,args.shards,args.disk_shards,args.bands]+args.word_sizes+args.title_sizes+args.author_sizes)<1 or 100%args.bands or not 0<=args.shard_index<args.shards or args.synthetic_books<1 or args.disk_shards>128: p.error('Invalid settings; bands must divide 100; disk shards 1–128')
    if any(not 0<=v<=1 for v in [args.threshold,args.author_threshold,args.title_threshold,args.auto_text_threshold]): p.error('Thresholds must be in [0,1]')
    if args.limit is not None:
        if args.limit<2: p.error('Limit must be >=2')
        args.output=args.output/f'first_{args.limit}'
    args.output.mkdir(parents=True,exist_ok=True)
    if args.mode=='run':
        for stale in ['manual_review_needed.csv','review_candidates.csv']:
            (args.output/stale).unlink(missing_ok=True)
        if not args.audit:
            (args.output/'pair_decisions.csv').unlink(missing_ok=True)
    if not args.dataset.exists():
        p.error(f'Input file not found: {args.dataset}. Put it next to this script or pass --dataset PATH.')
    # JSONL is streamed by default; no --stream-json option is required.
    books=load_books(args.dataset,args.limit,args.output/'text_cache')
    print(f'Loaded {len(books):,} book records from {args.dataset}',flush=True)
    if args.mode=='study': study(books,args,{}); return
    t=time.perf_counter(); signatures=cached_signatures(books,args,args.word_size); print(f'Signature stage: {time.perf_counter()-t:.2f}s (may include cache reuse)')
    if args.mode=='signatures': return
    if args.mode=='benchmark': benchmark(books,signatures,args); return
    text_pairs=lsh_candidates(signatures,args.bands)
    meta_pairs=metadata_pairs(books,args.title_size,args.author_size,args.title_threshold,args.author_threshold)
    pairs=text_pairs|meta_pairs
    rows=pair_details(books,signatures,pairs,text_pairs,meta_pairs,args)
    export_groups(books,signatures,rows,text_pairs,meta_pairs,args)
    if not args.no_export_json and args.mode=='run':
        export_categorized_works_json(books,args)
    print('Automatic deduplication complete. Output folder:',args.output)

if __name__=='__main__':
    main()

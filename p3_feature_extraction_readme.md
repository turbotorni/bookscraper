# P3 — Stylometric & Bigram Feature Extraction

This folder contains the feature extraction pipeline for **Part P3** of the 02807 Project Gutenberg Popularity project. It converts cleaned book texts into stylometric tables and sparse bigram count matrices for **P4** (Linear SVM) and **P5** (Frequent Itemsets).

---

## 1. Repository Files

* **`extract_features.py`** *(Main deliverable)*: Extracts stylometric features (POS tags, dependency labels, readability scores, sentence structure, lexical diversity) and word bigram counts from 1,000 evenly spaced sentences per book using `spaCy`. Supports parallel execution via LSF job arrays (`extract`) and consolidates shards into final contract files (`merge`).
* **`convert_scrape.py`** *(Temporary stopgap)*: Converts the raw scraper dump (`dataset.json.gz`) into the `books.parquet` and `texts/<book_id>.txt` format expected by `extract_features.py`. Use this only for local testing before P1's official cleaned dataset is ready.

---

## 2. Setup & Requirements

Install dependencies (preferably inside a virtual environment or conda env):

```bash
pip install numpy pandas pyarrow scipy spacy
python -m spacy download en_core_web_sm
```

---

## 3. How to Run

### Step A (Optional / Local Testing Only): Convert Raw Scraper Output
If running on the raw `dataset.json.gz` scrape instead of P1's cleaned files:

```bash
# Convert a 500-book sample for local testing
python convert_scrape.py --input dataset.json.gz --out-dir data --limit 500
```

### Step B: Extract Features (`extract`)
Runs on books that have a `genre`, are not anthologies (`is_anthology == False`), and pass the word-length filter.

**Single-machine / local sample run:**
```bash
python extract_features.py extract \
  --books data/books.parquet \
  --texts-dir data/texts \
  --out-dir out/parts \
  --min-words 5000
```

**DTU HPC LSF Job Array run (full corpus):**
Split the dataset across `N` parallel workers (e.g., 20 shards) using `$LSB_JOBINDEX`:
```bash
python extract_features.py extract \
  --books /path/to/p1/books.parquet \
  --texts-dir /path/to/p1/texts \
  --out-dir out/parts \
  --min-words 5000 \
  --n-shards 20 \
  --shard $LSB_JOBINDEX
```
*(Optional robustness check: add `--mode first` to sample the first 1,000 sentences instead of evenly spaced `spread` sampling.)*

### Step C: Merge Shards into Final Contract Files (`merge`)
Once all `extract` shards finish, run `merge` once to build the global bigram vocabulary and output the final files:

```bash
python extract_features.py merge \
  --out-dir out/parts \
  --final-dir out \
  --min-df 5
```

---

## 4. Output Files (Data Contract for P4 & P5)

Running `merge` produces the following files inside `out/`:

| File | Format | Description |
| :--- | :--- | :--- |
| **`features_style.parquet`** | Parquet | `book_id` + **108 stylometric & bookkeeping columns** (`pos_*`, `dep_*`, `read_*`, `sent_*`, `lex_*`, `meta_*`). **Note for P4:** exclude `meta_*` columns when training the SVM. |
| **`features_bigrams.npz`** | SciPy CSR | Sparse `int32` matrix of raw lowercased word bigram counts, aligned row-for-row with `features_style.parquet`. (P4 applies TF-IDF inside CV folds). |
| **`vocab.json`** | JSON list | Alphabetically sorted list of bigram strings; column `j` in `features_bigrams.npz` corresponds to `vocab[j]`. |
| **`feature_schema.json`** | JSON | Full documentation of feature groups, sampling parameters, and bigram thresholds. |
| **`features_timing.json`** | JSON | Processing speed benchmarks (mean/median/p95 seconds per book and total CPU hours). |
| **`features_failed.tsv`** | TSV | Log of any skipped `book_id`s (e.g., fewer than 20 eligible sentences) and failure reasons. |

---

## 5. Sample Run Output (500-Book Dev Test)

Running the pipeline on a 500-book limit from `dataset.json.gz`:

1. **`convert_scrape.py`** extracted **476 English books** into `data/books.parquet` and `data/texts/*.txt` (201 had no placeholder genre, 275 had genres).
2. **`extract_features.py extract`** filtered to **236 qualifying books** (`genre` present, non-anthology, `>= 5,000` words) and processed them in **447 seconds** (~7.5 minutes) with **0 failures**.
3. **`extract_features.py merge`** produced `features_style.parquet` (`236 × 109`) and `features_bigrams.npz` (`236 × 44,897` bigrams at `min_df=5`):

```json
{
  "n_books": 236,
  "mean_s_per_book": 1.89,
  "median_s_per_book": 1.81,
  "p95_s_per_book": 2.74,
  "total_cpu_hours": 0.124
}
```
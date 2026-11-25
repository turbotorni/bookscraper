#!/bin/bash
#BSUB -J p3_extract[1-20]
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 04:00
#BSUB -o logs/p3_%J_%I.out
#BSUB -e logs/p3_%J_%I.err

cd ~/Desktop/bookscraper
source ~/Desktop/.venv/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python extract_features.py extract \
  --books data/books.parquet \
  --texts-dir data/texts \
  --out-dir out/parts \
  --min-words 5000 \
  --n-shards 20 \
  --shard $LSB_JOBINDEX
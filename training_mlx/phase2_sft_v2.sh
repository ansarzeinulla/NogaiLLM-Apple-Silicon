#!/bin/bash
# phase2_sft_v2.sh
#
# NogaiLLM - Phase 2 SFT on the clean, leakage-free split from build_sft_clean.py
# (sft_v2/train.jsonl, valid.jsonl, test.jsonl: 501/62/62 pairs, both directions).
# Same hyperparameters as the published v1 run; about 2 epochs of the new train split.
# Trains three seeds so results can be reported as mean +/- spread, then reports
# test loss/perplexity with mlx-lm --test.
#
# Usage: bash phase2_sft_v2.sh [path/to/sft_v2]
set -euo pipefail
DATA="${1:-../data_engineering/phase2_sft_alignment/sft_v2}"
ROWS=$(wc -l < "$DATA/train.jsonl")
ITERS=$(( ROWS * 2 ))   # batch 1, 2 epochs
mkdir -p adapters

for SEED in 0 1 2; do
  mlx_lm.lora \
      --model local_qwen_1.5B_Nogai_Base \
      --data "$DATA" \
      --train --test \
      --iters "$ITERS" \
      --batch-size 1 \
      --learning-rate 2e-5 \
      --num-layers 16 \
      --max-seq-length 512 \
      --seed "$SEED" \
      --adapter-path "adapters/sft_v2_seed$SEED" 2>&1 | tee "adapters/sft_v2_seed$SEED.log"
done

#!/bin/bash
# phase1_cpt_lora.sh
#
# NogaiLLM - Phase 1: continued pre-training (CPT) with LoRA on Nogai-Unified-Corpus-v1.
# Reproduces the published adapter ansarzeinulla/Qwen2.5-1.5B-Nogai-LoRA. Every setting
# below is copied from its adapter_config.json (configs/phase1_published_adapter_config.json):
#   batch 2, 2500 iterations, learning rate 2e-4, max 512 tokens, seed 0,
#   LoRA rank 8, scale 20, dropout 0, on all linear layers (q/k/v/o/gate/up/down)
#   of the top 16 transformer blocks (layers 12-27), which is mlx-lm's default.
#
# The base is an MLX copy of Qwen2.5-1.5B-Instruct. Without -q it is not quantized,
# i.e. this is LoRA, not QLoRA. If you create the base with -q (4-bit), it is QLoRA.
set -euo pipefail

if [ ! -d local_qwen_1.5B ]; then
  mlx_lm.convert --hf-path Qwen/Qwen2.5-1.5B-Instruct --mlx-path local_qwen_1.5B
fi

# --data must contain train.jsonl and valid.jsonl ({"text": ...} rows).
mlx_lm.lora \
    --model local_qwen_1.5B \
    --data ../data_engineering/phase1_corpus/ \
    --train \
    --iters 2500 \
    --batch-size 2 \
    --learning-rate 2e-4 \
    --num-layers 16 \
    --max-seq-length 512 \
    --steps-per-eval 200 \
    --save-every 500 \
    --seed 0 \
    --adapter-path adapters/qwen_1.5b_nogai_phase1

echo "Phase 1 done: adapters/qwen_1.5b_nogai_phase1"

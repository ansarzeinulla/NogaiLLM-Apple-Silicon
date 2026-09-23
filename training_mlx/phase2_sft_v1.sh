#!/bin/bash
# phase2_sft_v1.sh
#
# NogaiLLM - Phase 2: supervised fine-tuning (SFT), as used for the published adapter
# ansarzeinulla/Qwen2.5-1.5B-Nogai-SFT-Experimental. Settings copied from its
# adapter_config.json (configs/phase2_published_adapter_config.json):
#   batch 1, 2400 iterations, learning rate 2e-5, max 512 tokens, seed 0,
#   LoRA rank 8, scale 20 on all linear layers of the top 16 blocks.
# The published run was resumed from a saved adapter (resume_adapter_file) after a
# Metal crash; clear_cache_threshold was 0 (not used) and gradient checkpointing was off.
#
# Known problem with the v1 data (Nogai-Russian-SFT-Biblical-v1): rows are duplicated up
# to 4x and every validation example also occurs in train, so its validation loss does
# not measure generalisation. Use phase2_sft_v2.sh with the clean split for any result.
#
# Phase 2 must run on the fused Phase 1 model, not on the raw base.
set -euo pipefail

mlx_lm.lora \
    --model local_qwen_1.5B_Nogai_Base \
    --data ../data_engineering/phase2_sft_alignment/ \
    --train \
    --iters 2400 \
    --batch-size 1 \
    --learning-rate 2e-5 \
    --num-layers 16 \
    --max-seq-length 512 \
    --seed 0 \
    --adapter-path adapters/qwen_1.5b_nogai_phase2_sft_v1

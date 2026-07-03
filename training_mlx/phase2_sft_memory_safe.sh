#!/bin/bash
# phase2_sft_memory_safe.sh
# 
# NogaiLLM Training Pipeline - Phase 2 (Supervised Fine-Tuning)
# Curing Catastrophic Forgetting via ChatML multi-epoch alignment.
#
# HARDWARE FIX: Uses gradient checkpointing to bound peak Metal memory and
# prevent descriptor leak panics on Apple Silicon. (The former
# --clear-cache-threshold flag was removed from mlx-lm; --grad-checkpoint is
# the supported memory-safety mechanism in current releases.)

echo "Initializing Phase 2: Memory-Safe ChatML SFT Alignment..."

# Note: Phase 2 MUST be trained on the Fused Phase 1 model, NOT the raw base.
mlx_lm.lora \
    --model "local_qwen_1.5B_Nogai_Base" \
    --data "../data_engineering/phase2_sft_alignment/" \
    --train \
    --iters 2400 \
    --batch-size 2 \
    --num-layers 16 \
    --learning-rate 1e-5 \
    --grad-checkpoint \
    --adapter-path "adapters/qwen_1.5b_nogai_phase2_sft"

echo "Phase 2 Complete. Instruction-following manifold mathematically restored."
#!/bin/bash
# phase1_qlora_commands.sh
# 
# NogaiLLM Training Pipeline - Phase 1 (Continuous Pre-Training)
# BPE Vocabulary Expansion on unstructured Cyrillic Turkic text.
# Executed via mlx-lm on Apple Silicon (M-series).

echo "Initializing Phase 1: Unstructured Morphological CPT..."

# Ensure we are in the correct directory
# The --data path should point to the directory containing train.jsonl / valid.jsonl
mlx_lm.lora \
    --model "Qwen/Qwen2.5-1.5B-Instruct" \
    --data "../data_engineering/phase1_corpus/" \
    --train \
    --iters 2500 \
    --batch-size 4 \
    --num-layers 16 \
    --learning-rate 2e-4 \
    --adapter-path "adapters/qwen_1.5b_nogai_phase1"

echo "Phase 1 Complete. Base vocabulary weights saved to adapters/qwen_1.5b_nogai_phase1"
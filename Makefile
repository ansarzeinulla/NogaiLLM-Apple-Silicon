# NogaiLLM Pipeline Automation
# Usage:
#   make data     - Run the end-to-end data engineering pipeline
#   make train    - Execute Phase 1 and Phase 2 training
#   make evaluate - Run the ablation studies

.PHONY: data train evaluate clean

data:
	@echo "=> Executing Phase 1 Corpus Engineering..."
	python data_engineering/phase1_corpus/03_cross_lingual_filter.py --input raw.txt --output clean.txt
	@echo "=> Executing Phase 2 SFT Alignment..."
	python data_engineering/phase2_sft_alignment/build_sft_train.py --input chunked.jsonl

train:
	@echo "=> Initiating MLX Apple Silicon Training Pipeline..."
	bash training_mlx/phase1_qlora_commands.sh
	python training_mlx/fuse_lora_weights.py
	bash training_mlx/phase2_sft_memory_safe.sh

evaluate:
	@echo "=> Running Typographic Density Ablation..."
	python evaluation/typographic_density_metric.py --dir evaluation/results/

clean:
	@echo "=> Cleaning up cache and safely purging temp files..."
	rm -rf __pycache__ .pytest_cache
	find . -type d -name "*.egg-info" -exec rm -rf {} +
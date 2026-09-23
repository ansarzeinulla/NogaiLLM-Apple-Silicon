# NogaiLLM Pipeline Automation
# Usage:
#   make data     - Run the end-to-end data engineering pipeline
#                   (requires PDF_DIR with raw PDFs and IBT_DIR with nog_*.txt / ru_*.txt pairs)
#   make train    - Execute Phase 1 and Phase 2 training
#   make evaluate - Run the ablation studies
#
# Override inputs, e.g.:  make data PDF_DIR=~/corpora/pdfs IBT_DIR=~/corpora/ibt

PDF_DIR ?= data_engineering/phase1_corpus/pdfs
IBT_DIR ?= data_engineering/phase2_sft_alignment/ibt
P1      := data_engineering/phase1_corpus
P2      := data_engineering/phase2_sft_alignment

.PHONY: data train evaluate clean

data:
	@echo "=> Executing Phase 1 Corpus Engineering..."
	python $(P1)/01_pdf_multiprocess_extractor.py --input_dir $(PDF_DIR) --output_file $(P1)/combined.txt
	python $(P1)/03_cross_lingual_filter.py --input $(P1)/combined.txt --output $(P1)/filtered.txt
	python $(P1)/04_morphological_reconstructor.py --input $(P1)/filtered.txt --output $(P1)/pristine.txt
	python $(P1)/05_chatml_jsonl_compiler.py --input $(P1)/pristine.txt --train_out $(P1)/train.jsonl --valid_out $(P1)/valid.jsonl
	@echo "=> Executing Phase 2 SFT Alignment..."
	python $(P2)/extract_ibt_verses.py --source_dir $(IBT_DIR) --output $(P2)/raw_parallel_blocks.jsonl
	python $(P2)/analyze_sentence_alignment.py --input $(P2)/raw_parallel_blocks.jsonl --output $(P2)/aligned_sentences.jsonl
	python $(P2)/proportional_chunker.py --input $(P2)/aligned_sentences.jsonl --output $(P2)/chunked_parallel.jsonl
	python $(P2)/build_sft_clean.py --chunks $(P2)/chunked_parallel.jsonl --cpt-corpus $(P1) --out-dir $(P2)/sft_v2

train:
	@echo "=> Initiating MLX Apple Silicon Training Pipeline..."
	cd training_mlx && bash phase1_cpt_lora.sh
	cd training_mlx && python fuse_lora_weights.py
	cd training_mlx && bash phase2_sft_v2.sh ../$(P2)/sft_v2

evaluate:
	mkdir -p evaluation/results
	@echo "=> Bits per byte on held-out corpus text (base vs Phase 1)..."
	python evaluation/eval_bits_per_byte.py --model Qwen/Qwen2.5-1.5B-Instruct --data $(P1)/valid.jsonl --out evaluation/results/bpb_base.json
	python evaluation/eval_bits_per_byte.py --model Qwen/Qwen2.5-1.5B-Instruct --adapter training_mlx/adapters/qwen_1.5b_nogai_phase1 --data $(P1)/valid.jsonl --out evaluation/results/bpb_phase1.json
	@echo "=> chrF++/BLEU on the held-out test pairs (Phase 1 base + Phase 2 SFT)..."
	python evaluation/eval_translation_chrf.py --model training_mlx/local_qwen_1.5B_Nogai_Base --adapter training_mlx/adapters/sft_v2_seed0 --test $(P2)/sft_v2/test.jsonl --out evaluation/results/chrf_sft_v2_seed0.json --outputs-txt evaluation/results/results_sft_v2_seed0.txt
	@echo "=> Typographic Density against the human reference..."
	python evaluation/typographic_density_metric.py --dir evaluation/results/ --reference $(P1)/valid.jsonl

clean:
	@echo "=> Cleaning up cache and safely purging temp files..."
	rm -rf __pycache__ .pytest_cache
	find . -type d -name "*.egg-info" -exec rm -rf {} +

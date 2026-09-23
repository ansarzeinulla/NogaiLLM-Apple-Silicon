# NogaiLLM: Adapting Qwen2.5 to Nogai with LoRA on Apple Silicon

[![CI/CD Pipeline](https://github.com/ansarzeinulla/NogaiLLM-Apple-Silicon/actions/workflows/ml_integrity.yml/badge.svg)](https://github.com/ansarzeinulla/NogaiLLM-Apple-Silicon/actions)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Demo-Hugging%20Face%20Space-ffcc00.svg)](https://huggingface.co/spaces/ansarzeinulla/NogaiLLM-Zero-Resource)
[![Models & Datasets](https://img.shields.io/badge/🤗%20HF%20Collection-Models%20%26%20Data-blue.svg)](https://huggingface.co/ansarzeinulla)
[![Framework](https://img.shields.io/badge/Framework-Apple%20MLX-000000.svg?logo=apple&logoColor=white)](https://github.com/ml-explore/mlx)

Code, data pipeline and evaluation for adapting an instruction-tuned LLM (Qwen2.5-1.5B-Instruct) to **Nogai**, a Kipchak Turkic language written in Cyrillic, with almost no digital resources. Everything runs on a single Apple M2 Pro (16 GB) with `mlx-lm`.

Two stages:

1. **Phase 1, continued pre-training (CPT):** LoRA on raw Nogai text (*Nogai-Unified-Corpus-v1*). The model learns Nogai spelling and morphology but stops following chat instructions.
2. **Phase 2, supervised fine-tuning (SFT):** LoRA on Russian↔Nogai translation pairs from human Bible translations, on top of the fused Phase 1 model, to bring instruction following back.

> **Status (September 2026).** The first SFT dataset (v1) has duplicated rows and its validation rows also appear in its training rows, so the v1 validation/test numbers are not a measure of generalisation. This repo now includes a clean split (`build_sft_clean.py`), a bits-per-byte evaluation that is comparable across tokenizers, and chrF++/BLEU against human references. Results will be updated after re-running them.

---

## 🗂️ Repository

```text
NogaiLLM-Apple-Silicon/
├── data_engineering/
│   ├── phase1_corpus/            # PDF/Wiki extraction, Russian filtering, JSONL compiler
│   └── phase2_sft_alignment/     # IBT verse extraction, chunking, SFT builders (v1, clean v2)
├── training_mlx/
│   ├── phase1_cpt_lora.sh        # reproduces the published Phase 1 adapter
│   ├── fuse_lora_weights.py      # fuses Phase 1 LoRA into the base model
│   ├── phase2_sft_v1.sh          # the published v1 SFT run (for the record)
│   ├── phase2_sft_v2.sh          # SFT on the clean split, 3 seeds, with test loss
│   └── configs/                  # adapter_config.json of both published adapters
├── evaluation/
│   ├── eval_bits_per_byte.py     # Phase 1: bits per UTF-8 byte (tokenizer-independent)
│   ├── eval_translation_chrf.py  # Phase 2: chrF++ / BLEU against human translations
│   ├── typographic_density_metric.py
│   ├── evaluate_phase1_few_shot.py
│   └── evaluate_phase2_sft_chatml.py
├── demo/app.py                   # Gradio demo (Hugging Face Space)
└── Makefile                      # make data / make train / make evaluate
```

## 🛠️ Data

| Dataset | Content | Size |
|---|---|---|
| [Nogai-Unified-Corpus-v1](https://huggingface.co/datasets/ansarzeinulla/Nogai-Unified-Corpus-v1) | Newspapers *Шоьл тавысы* (Dagestan) and *Ногай давысы* (Karachay-Cherkessia), IBT Bible translations, Wikimedia Incubator (Wp/nog) | 163,531 rows (155,354 train / 8,177 valid), 2.35 M words, 9.8 M Qwen2.5 tokens |
| [Nogai-Russian-SFT-Biblical-v1](https://huggingface.co/datasets/ansarzeinulla/Nogai-Russian-SFT-Biblical-v1) | Russian–Nogai Bible passages (IBT), both translation directions | 4,310 rows = 650 unique pairs; see known issues |
| [Nogai-Russian-SFT-Biblical-v2](https://huggingface.co/datasets/ansarzeinulla/Nogai-Russian-SFT-Biblical-v2) | The same pairs, deduplicated and split for evaluation | 625 pairs = 1,250 rows: 507 / 60 / 58 pairs (train / valid / test) |

Corpus checks (September 2026): no exact duplicate rows; no validation row appears verbatim in train; about 0.4% of rows look Russian (Russian stop-word filtering removes most but not all of it).

**SFT data, clean v2:** `build_sft_clean.py` removes duplicates, drops 25 pairs whose lengths show the Russian and Nogai sides are different passages, and splits **by pair** before creating the two directions: **507 / 60 / 58** pairs (train / valid / test), zero overlap. Because one verse can appear in two differently aligned pairs, any validation or test pair that shares a Russian or Nogai sentence with train (or, for test, with validation) is moved into train: 6 pairs. No test pair contains a sentence that occurs in the Phase 1 corpus.

```bash
python data_engineering/phase2_sft_alignment/build_sft_clean.py --from-hf \
    --cpt-corpus data_engineering/phase1_corpus --out-dir data_engineering/phase2_sft_alignment/sft_v2
```

## 💻 Training (Apple M2 Pro, 16 GB, mlx-lm)

Settings of the published adapters, copied from their `adapter_config.json` (`training_mlx/configs/`):

| | Phase 1 (CPT) | Phase 2 (SFT v1) |
|---|---|---|
| Base | MLX copy of Qwen2.5-1.5B-Instruct (`local_qwen_1.5B`) | fused Phase 1 model |
| LoRA | rank 8, scale 20, dropout 0; q/k/v/o/gate/up/down in layers 12–27 (16 blocks, 5.28 M parameters) | same |
| Batch / iterations / learning rate | 2 / 2,500 / 2e-4 | 1 / 2,400 / 2e-5 |
| Max sequence length | 512 | 512 |

Long SFT runs crashed with a Metal `Internal Error`. The published SFT run was resumed from its last saved adapter (`resume_adapter_file` in its config). `clear_cache_threshold` was 0 and gradient checkpointing was off. Peak unified memory during SFT: 5.73 GB.

```bash
make data      # needs the raw PDFs and IBT texts (see Makefile)
make train
make evaluate
```

## 📊 Evaluation

- **Phase 1:** bits per UTF-8 byte on the corpus validation split. Per-token perplexity is also printed but can't be compared between Qwen and Llama, whose tokenizers split Nogai very differently.
- **Phase 2:** chrF++ and BLEU against the human IBT translations of the held-out test pairs, both directions, greedy decoding.
- **Typographic Density (TD):** Nogai digraphs (аь, оь, уь, нъ) per 100 **words**. Human Nogai text scores **22.8**, so TD is reported as the distance from that value (`--reference`).

Results from the first version (per-token perplexity 191.13 → 12.14 on Phase 1 for Qwen2.5-1.5B; Llama-3.2-1B emitting only end-of-sequence after CPT) are being re-measured with the methods above.

## 🚀 Demo

The [Space](https://huggingface.co/spaces/ansarzeinulla/NogaiLLM-Zero-Resource) converts both MLX adapters to PEFT, merges Phase 1 into Qwen2.5-1.5B-Instruct and applies Phase 2 on top (CPU, float32). It is an experimental baseline: trained only on Bible text, it often answers modern-domain inputs with religious vocabulary.

```bash
pip install -r requirements.txt
python demo/app.py
```

## ⚠️ Known issues

- **SFT v1 data:** 4,310 rows but 1,300 unique; rows repeated up to 4×; all 200 unique validation rows also appear in train. The v1 validation/test loss is therefore optimistic. Use the clean v2 split.
- **Proportional chunking** pairs blocks by position, not by verse number, so a few pairs are different passages (25 of 650 fail a length check). Verse-level alignment would remove this.
- **Domain:** SFT data is Bible text only.

## ⚖️ License

Code: Apache-2.0. Datasets and adapters on Hugging Face: CC BY-NC 4.0.

## 📄 Citation

```bibtex
@misc{zeinulla2026nogaillm,
  title  = {NogaiLLM: Parameter-Efficient Continued Pre-Training and Catastrophic Forgetting in Zero-Resource Turkic Languages},
  author = {Zeinulla, Ansar},
  year   = {2026},
  note   = {Manuscript under revision. Code: https://github.com/ansarzeinulla/NogaiLLM-Apple-Silicon}
}
```

<div align="center">
  <i>Ansar Zeinulla · Nazarbayev University</i>
</div>

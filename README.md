# ⚡️ NogaiLLM: Curing Catastrophic Forgetting in Zero-Resource Turkic NLP (Apple Silicon Native)

[![Framework](https://img.shields.io/badge/Framework-Apple%20MLX-000000.svg?logo=apple&logoColor=white)](https://github.com/ml-explore/mlx)
[![Paper](https://img.shields.io/badge/Paper-ACM%20TALLIP-B31B1B.svg)](#)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces%20Live-ffcc00.svg?logo=huggingface)](#)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#)

An end-to-end, reproducible machine learning pipeline engineered to adapt frontier Large Language Models (LLMs) to **mathematically isolated, zero-resource languages** (specifically, the endangered Nogai Cyrillic language). 

This repository contains the entire systems and data engineering framework used to execute **Parameter-Efficient Continuous Pre-Training (CPT)** and **Supervised Fine-Tuning (SFT)** entirely natively on Apple Silicon consumer hardware.

---

## 🔬 Scientific Overview & Architectural Findings

Expanding dense Byte-Pair Encoding (BPE) vocabularies to accommodate zero-resource agglutinative languages natively triggers two severe architectural failures, which this framework specifically resolves:

1. **The Llama Latin-Centric Bottleneck:** Standard Western architectures (e.g., Llama-3.2) exhibit severe out-of-distribution (OOD) decay when forced to process isolated Cyrillic morphologies, frequently collapsing into autoregressive hallucination loops. Our ablation studies demonstrate that Eastern-optimized BPEs (Qwen-2.5) successfully map local Turkic morphology.
2. **Catastrophic Forgetting of the Instruction Manifold:** When subjected to unstructured Continuous Pre-Training (Phase 1), the model's multi-head attention successfully learns the new vocabulary (dropping Perplexity from 191.13 to **12.14**), but its predictive routing overwrites the base model's instruction-following capabilities. It defaults to unconditional text generation.

**NogaiLLM solves this via a Two-Phase Recovery Pipeline:**
*   **Phase 1 (CPT):** Unstructured morphological acquisition via the rigorously isolated *Nogai Unified Corpus v1*.
*   **Phase 2 (SFT):** Restoring the native `ChatML` multi-head attention routing using custom Proportional Sentence-Boundary Chunking and Bidirectional Tensor Alignment.

---

## 🗂️ Repository Architecture

```text
NogaiLLM-Apple-Silicon/
│
├── data_engineering/
│   ├── phase1_corpus/          # Cross-lingual filtering, OCR repair, and isolation
│   └── phase2_sft_alignment/   # Proportional chunking & Bidirectional tensor alignment
│
├── training_mlx/               # Native Apple Silicon Hardware Optimizations
│   ├── phase1_qlora_commands.sh
│   ├── phase2_sft_memory_safe.sh
│   └── fuse_lora_weights.py    # Fuses Phase 1 LoRA matrices to base MLP
│
├── evaluation/                 # Architectural Ablation & Benchmarking
│   ├── evaluate_phase1_few_shot.py
│   ├── evaluate_phase2_sft_chatml.py
│   └── typographic_density_metric.py
│
├── demo/                       # HF Spaces Deployment Engine
│   └── app.py                  # Dynamic MLX-to-PEFT safe-tensor conversion
│
└── requirements.txt
```

---

## 🛠️ Data Engineering Pipeline

A zero-resource model is only as stable as its cross-lingual boundaries. The `data_engineering/` suite bypasses standard MT contamination through rigorous syntactic preservation:

*   **Proportional Sentence Chunking (`proportional_chunker.py`):** Slavic and Turkic languages lack 1:1 syntactic mapping. Standard extraction scripts that truncate sequences by token count slice agglutinative words in half, destroying attention mappings. We implemented an algorithmic chunker that proportionally groups sequences to ensure every training sample contains grammatically intact thought blocks.
*   **Bidirectional Tensor Alignment (`build_sft_train.py`):** To symmetrically stabilize the LLM's cross-attention mechanisms during Phase 2 SFT, prompts are mathematically mirrored (50% Russian $\rightarrow$ Nogai, 50% Nogai $\rightarrow$ Russian).

---

## 💻 Hardware & Systems Optimization (Apple MLX)

Training multi-epoch LLM adapters natively on macOS Metal backend introduces severe hardware bottlenecks. 

### Resolving the Metal Descriptor Leak
Extended backpropagation triggers a memory-leak panic (`0000000e:Internal Error`) where the macOS Metal driver accumulates active buffer handles faster than Python runtime garbage collection. In `phase2_sft_memory_safe.sh`, we explicitly resolve this by injecting `--clear-cache-threshold 0.7` into the MLX allocator, dynamically flushing command buffers and bottlenecking peak unified memory to a mathematically stable **5.734 GB**.

### Dynamic MLX-to-PEFT Conversion (`app.py`)
Apple MLX utilizes a proprietary `adapters.safetensors` structure that is incompatible with standard Hugging Face HuggingFace/Transformers deployments. Our Gradio application (`demo/app.py`) features a dynamic, on-the-fly tensor conversion engine that mathematically casts MLX low-rank matrices (`.lora_a` / `.lora_b`) back into standard PyTorch `PEFT` float32 structures for stable, cross-platform CPU deployment.

---

## 📊 Evaluation & Ablation Metrics

To objectively measure morphological acquisition vs. mode collapse, we engineered the **Nogai Typographic Density Metric** (`evaluation/typographic_density_metric.py`). This script measures the exact frequency of structurally accurate native digraphs (e.g., `нъ`, `аь`, `оь`, `уь`) against hallucination loops.

| Model Configuration | Base Scale | Perplexity ($\downarrow$) | Nogai Typographic Density ($\uparrow$) | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Qwen-2.5-1.5B (Phase 1 CPT)** | 1.5B | **12.14** | **15.05%** | ✅ Perfect Morphological Mapping |
| Llama-3.2-1B (Phase 1 CPT) | 1.0B | 245.82 | 0.00% | ❌ OOD Mode Collapse |

---

## 🚀 Quick Start: Running the Demo locally

To test the structurally recovered (Phase 2) conversational model via the Gradio UI:

```bash
# 1. Clone repository
git clone https://github.com/yourusername/NogaiLLM-Apple-Silicon.git
cd NogaiLLM-Apple-Silicon

# 2. Install deployment dependencies
pip install -r requirements.txt

# 3. Launch the MLX-to-PEFT conversion and UI engine
cd demo
python app.py
```
*Note: The demo automatically downloads the Phase 2 SFT adapter from Hugging Face and executes the tensor translation natively.*

---

## 📚 Datasets & Model Adapters

The fully curated datasets and pre-trained adapter weights are permanently hosted on Hugging Face for academic reproducibility:

1.  **Phase 1 Corpus:** [Nogai Unified Corpus v1](https://huggingface.co/datasets/ansarzeinulla/Nogai-Unified-Corpus-v1)
2.  **Phase 2 Corpus:** [Nogai-Russian SFT Biblical Corpus v1](https://huggingface.co/datasets/ansarzeinulla/Nogai-Russian-SFT-Biblical-v1)
3.  **Phase 1 CPT Adapter:** [Qwen2.5-1.5B-Nogai-LoRA](https://huggingface.co/ansarzeinulla/Qwen2.5-1.5B-Nogai-LoRA)
4.  **Phase 2 SFT Adapter:** [Qwen2.5-1.5B-Nogai-SFT-Experimental](https://huggingface.co/ansarzeinulla/Qwen2.5-1.5B-Nogai-SFT-Experimental)

---

## 📄 Citation & Academic Context

If you utilize this infrastructure, the chunking algorithms, or the memory-safe MLX training commands in your own zero-resource NLP research, please cite our core paper:

```bibtex
@article{zeinulla2026nogaillm,
  title={NogaiLLM: Parameter-Efficient Continuous Pre-Training and Architectures of Catastrophic Forgetting in Zero-Resource Turkic Languages},
  author={Zeinulla, Ansar},
  journal={arXiv preprint arXiv:2607.xxxxx},
  year={2026},
  publisher={Nazarbayev University / ACM TALLIP}
}
```

<div align="center">
  <i>Engineered and maintained by Ansar Zeinulla • Nazarbayev University</i>
</div>
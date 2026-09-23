"""
eval_bits_per_byte.py

NogaiLLM Evaluation Suite - Phase 1 (tokenizer-independent)
Per-token perplexity cannot be compared across models with different tokenizers:
Llama splits Nogai text into more, smaller tokens than Qwen, so its per-token loss
is not on the same scale. Bits per UTF-8 byte is: every model is charged for
predicting the same bytes.

For each held-out text, the model scores every token after the first, given the
previous tokens (the first token has no context and is not scored for any model).

  BPB = (sum of negative log-likelihoods in bits) / (UTF-8 bytes of the scored text)

Example:
  python eval_bits_per_byte.py --model Qwen/Qwen2.5-1.5B-Instruct \
      --data ../data_engineering/phase1_corpus/valid.jsonl --limit 2000
  python eval_bits_per_byte.py --model Qwen/Qwen2.5-1.5B-Instruct \
      --adapter ansarzeinulla/Qwen2.5-1.5B-Nogai-LoRA --data ... --limit 2000
"""

import argparse
import json
import math
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load


def load_texts(path, limit):
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                texts.append(json.loads(line)["text"])
            if limit and len(texts) >= limit:
                break
    return texts


def resolve_adapter(adapter):
    """Accepts a local folder or a Hugging Face repo id with adapters.safetensors."""
    if adapter is None:
        return None
    from pathlib import Path

    if Path(adapter).exists():
        return adapter
    from huggingface_hub import snapshot_download

    return snapshot_download(adapter, allow_patterns=["adapter_config.json", "adapters.safetensors"])


def main():
    parser = argparse.ArgumentParser(description="Bits per byte on held-out Nogai text")
    parser.add_argument("--model", required=True, help="HF repo id or local MLX model folder")
    parser.add_argument("--adapter", default=None, help="MLX LoRA adapter (folder or HF repo id)")
    parser.add_argument("--data", required=True, help="JSONL with a 'text' field (use valid.jsonl)")
    parser.add_argument("--limit", type=int, default=2000, help="number of texts (0 = all)")
    parser.add_argument("--max-tokens", type=int, default=1024, help="truncate longer texts")
    parser.add_argument("--out", default=None, help="optional JSON file for the result")
    args = parser.parse_args()

    model, tokenizer = load(args.model, adapter_path=resolve_adapter(args.adapter))
    model.eval()
    texts = load_texts(args.data, args.limit)

    total_nats = 0.0
    total_bytes = 0
    total_tokens = 0
    start = time.time()
    for i, text in enumerate(texts, 1):
        ids = tokenizer.encode(text)[: args.max_tokens]
        if len(ids) < 2:
            continue
        tokens = mx.array(ids)[None]
        logits = model(tokens[:, :-1])
        nll = nn.losses.cross_entropy(logits, tokens[:, 1:], reduction="sum")
        total_nats += nll.item()
        total_tokens += len(ids) - 1
        # Bytes of the scored part: everything after the first token.
        first = tokenizer.decode(ids[:1])
        scored = tokenizer.decode(ids)[len(first):]
        total_bytes += len(scored.encode("utf-8"))
        if i % 200 == 0:
            print(f"  {i}/{len(texts)} texts, running BPB {total_nats / math.log(2) / total_bytes:.4f}",
                  flush=True)

    result = {
        "model": args.model,
        "adapter": args.adapter,
        "data": args.data,
        "texts": len(texts),
        "scored_tokens": total_tokens,
        "scored_bytes": total_bytes,
        "bits_per_byte": total_nats / math.log(2) / total_bytes,
        "per_token_perplexity": math.exp(total_nats / total_tokens),
        "tokens_per_byte": total_tokens / total_bytes,
        "seconds": round(time.time() - start, 1),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()

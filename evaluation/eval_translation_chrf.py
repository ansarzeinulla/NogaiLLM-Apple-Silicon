"""
eval_translation_chrf.py

NogaiLLM Evaluation Suite - Phase 2 (translation quality against human references)
Translates the held-out test pairs built by build_sft_clean.py in both directions
and scores them with chrF++ and BLEU (sacrebleu) against the human IBT translations.
Decoding is greedy (temperature 0), so a run is repeatable.

The test pairs are never seen in SFT training and none of their Nogai sentences
occur in the Phase 1 corpus, so the scores measure generalisation, not recall.

Example (model = the fused Phase 1 base, adapter = the Phase 2 SFT adapter):
  python eval_translation_chrf.py --model local_qwen_1.5B_Nogai_Base \
      --adapter adapters/sft_v2 --test sft_v2/test.jsonl --out results_sft_v2.json
Baselines: run the same command without --adapter, and with --model set to
Qwen/Qwen2.5-1.5B-Instruct.
"""

import argparse
import json
import time

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from sacrebleu.metrics import BLEU, CHRF


def load_test(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                messages = json.loads(line)["messages"]
                direction = "ru->nog" if "ногайский" in messages[1]["content"].split(":", 1)[0] else "nog->ru"
                rows.append((direction, messages[:2], messages[2]["content"]))
    return rows


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
    parser = argparse.ArgumentParser(description="chrF++/BLEU on held-out Russian-Nogai pairs")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--test", required=True, help="test.jsonl from build_sft_clean.py")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0, help="0 = all rows")
    parser.add_argument("--out", default=None)
    parser.add_argument("--outputs-txt", default=None,
                        help="also write every Russian->Nogai output in the format read by "
                             "typographic_density_metric.py (name it results_<model>.txt)")
    args = parser.parse_args()

    model, tokenizer = load(args.model, adapter_path=resolve_adapter(args.adapter))
    sampler = make_sampler(temp=0.0)
    rows = load_test(args.test)
    if args.limit:
        rows = rows[: args.limit]

    outputs = {"ru->nog": [], "nog->ru": []}
    refs = {"ru->nog": [], "nog->ru": []}
    samples = []
    start = time.time()
    for i, (direction, messages, reference) in enumerate(rows, 1):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        hypothesis = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens,
                              sampler=sampler, verbose=False).strip()
        outputs[direction].append(hypothesis)
        refs[direction].append(reference)
        if len(samples) < 6:
            samples.append({"direction": direction, "reference": reference, "model": hypothesis})
        if i % 20 == 0:
            print(f"  {i}/{len(rows)}", flush=True)

    if args.outputs_txt:
        with open(args.outputs_txt, "w", encoding="utf-8") as f:
            for i, (hyp, ref) in enumerate(zip(outputs["ru->nog"], refs["ru->nog"]), 1):
                f.write(f"--- Sequence {i} ---\nTRUE:  {ref}\nMODEL: {' '.join(hyp.split())}\n\n")

    chrf, bleu = CHRF(word_order=2), BLEU()
    scores = {}
    for direction in outputs:
        if outputs[direction]:
            scores[direction] = {
                "n": len(outputs[direction]),
                "chrF++": round(chrf.corpus_score(outputs[direction], [refs[direction]]).score, 2),
                "BLEU": round(bleu.corpus_score(outputs[direction], [refs[direction]]).score, 2),
            }
    result = {"model": args.model, "adapter": args.adapter, "test": args.test, "scores": scores,
              "seconds": round(time.time() - start, 1), "samples": samples}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

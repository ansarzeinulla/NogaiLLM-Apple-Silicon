"""
build_sft_clean.py

NogaiLLM Data Engineering Pipeline - Phase 2 (clean rebuild)
Rebuilds the Russian-Nogai SFT dataset so that evaluation numbers mean something:

  1. removes exact duplicate pairs;
  2. drops pairs whose Nogai/Russian length ratio is outside [--min-ratio, --max-ratio]
     (proportional chunking pairs blocks by position, so some pairs are different verses);
  3. splits by PAIR into train / valid / test before creating the two translation
     directions, so the reverse of a validation or test example is never in train;
  4. keeps the test split free of pairs whose Nogai sentences occur verbatim in the
     Phase 1 CPT corpus (the model has already read those sentences);
  5. mirrors each split (Russian->Nogai and Nogai->Russian) and writes ChatML JSONL.

Input: either the parallel chunks from proportional_chunker.py (--chunks, JSONL with
"russian"/"nogai") or the published v1 dataset (--from-hf), which contains the same
650 unique pairs.

Example:
  python build_sft_clean.py --from-hf --cpt-corpus ../phase1_corpus --out-dir sft_v2
"""

import argparse
import json
import random
import re
from pathlib import Path

SYSTEM_PROMPT = "You are a highly accurate bilingual translator for Russian and Nogai."
V1_REPO = "ansarzeinulla/Nogai-Russian-SFT-Biblical-v1"


def chatml(source_text, target_text, target_lang):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Переведи этот текст на {target_lang} язык: {source_text}"},
            {"role": "assistant", "content": target_text},
        ]
    }


def pairs_from_chunks(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                pairs.append((row["russian"].strip(), row["nogai"].strip()))
    return pairs


def pairs_from_hf():
    from huggingface_hub import hf_hub_download

    pairs = []
    for name in ("train.jsonl", "valid.jsonl"):
        path = hf_hub_download(V1_REPO, name, repo_type="dataset")
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                messages = json.loads(line)["messages"]
                prompt, answer = messages[1]["content"], messages[2]["content"].strip()
                head, source = prompt.split(":", 1)
                source = source.strip()
                if "ногайский" in head:
                    pairs.append((source, answer))  # Russian -> Nogai
                else:
                    pairs.append((answer, source))  # Nogai -> Russian
    return pairs


def load_cpt_text(corpus_dir):
    text = []
    for name in ("train.jsonl", "valid.jsonl"):
        path = Path(corpus_dir) / name
        if path.exists():
            with open(path, encoding="utf-8") as f:
                text.extend(json.loads(line)["text"].lower() for line in f if line.strip())
    return " \n ".join(text)


def seen_in_cpt(nogai, cpt_text, min_chars=40):
    for sentence in re.split(r"(?<=[.!?])\s+", nogai):
        sentence = re.sub(r"\s+", " ", sentence.strip().lower())
        if len(sentence) >= min_chars and sentence in cpt_text:
            return True
    return False


def write_split(pairs, path):
    with open(path, "w", encoding="utf-8") as f:
        for russian, nogai in pairs:
            f.write(json.dumps(chatml(russian, nogai, "ногайский"), ensure_ascii=False) + "\n")
            f.write(json.dumps(chatml(nogai, russian, "русский"), ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Clean, leakage-free SFT dataset builder")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--chunks", help="chunked_parallel.jsonl from proportional_chunker.py")
    source.add_argument("--from-hf", action="store_true", help=f"rebuild from {V1_REPO}")
    parser.add_argument("--cpt-corpus", help="dir with the Phase 1 train.jsonl/valid.jsonl")
    parser.add_argument("--out-dir", default="sft_v2")
    parser.add_argument("--valid-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--min-ratio", type=float, default=0.5)
    parser.add_argument("--max-ratio", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw = pairs_from_hf() if args.from_hf else pairs_from_chunks(args.chunks)
    pairs = sorted(set(raw))  # exact dedupe, deterministic order
    aligned = [p for p in pairs if args.min_ratio <= len(p[1]) / max(1, len(p[0])) <= args.max_ratio]

    cpt_text = load_cpt_text(args.cpt_corpus) if args.cpt_corpus else ""
    unseen = [p for p in aligned if not (cpt_text and seen_in_cpt(p[1], cpt_text))]
    seen = [p for p in aligned if p not in set(unseen)]

    rng = random.Random(args.seed)
    rng.shuffle(unseen)
    rng.shuffle(seen)
    n_test = round(len(aligned) * args.test_frac)
    n_valid = round(len(aligned) * args.valid_frac)
    if n_test > len(unseen):
        raise SystemExit(f"only {len(unseen)} CPT-unseen pairs, need {n_test} for test")
    test = unseen[:n_test]
    rest = unseen[n_test:] + seen
    rng.shuffle(rest)
    valid, train = rest[:n_valid], rest[n_valid:]

    assert not (set(train) & set(valid)) and not (set(train) & set(test)) and not (set(valid) & set(test))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("valid", valid), ("test", test)):
        write_split(split, out / f"{name}.jsonl")

    stats = {
        "input_rows": len(raw),
        "unique_pairs": len(pairs),
        "dropped_length_ratio": len(pairs) - len(aligned),
        "pairs_seen_in_cpt_corpus": len(seen),
        "train_pairs": len(train),
        "valid_pairs": len(valid),
        "test_pairs": len(test),
        "rows_per_pair": 2,
        "seed": args.seed,
        "length_ratio_bounds": [args.min_ratio, args.max_ratio],
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

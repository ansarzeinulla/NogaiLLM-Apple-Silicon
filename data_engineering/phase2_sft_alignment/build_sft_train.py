"""
build_sft_train.py

NogaiLLM Data Engineering Pipeline - Phase 2 (Script 4)
Compiles the final Supervised Fine-Tuning (SFT) dataset. Applies Bidirectional 
Tensor Alignment (50% Ru->Nog, 50% Nog->Ru) to symmetrically stabilize multi-head 
attention routing. Formats output natively to the ChatML standard.
"""

import json
import random
import logging
import argparse
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are a highly accurate bilingual translator for Russian and Nogai."

def build_chatml(source_text: str, target_text: str, target_lang: str) -> Dict:
    """Constructs the exact strict dictionary schema required by Qwen/Llama SFT."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Переведи этот текст на {target_lang} язык: {source_text}"},
            {"role": "assistant", "content": target_text}
        ]
    }

def compile_dataset(input_file: str, train_file: str, valid_file: str, seed: int = 42) -> None:
    in_path = Path(input_file)
    if not in_path.exists():
        logger.error("Chunked parallel file missing.")
        return

    logger.info("Initializing Bidirectional Tensor Alignment...")
    
    chatml_records = []
    
    with open(in_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            ru_text = data["russian"]
            nog_text = data["nogai"]
            
            # Symmetrical embedding alignment: 1 Forward, 1 Backward
            chatml_records.append(build_chatml(ru_text, nog_text, "ногайский"))
            chatml_records.append(build_chatml(nog_text, ru_text, "русский"))

    # Cryptographic shuffle for gradient descent variance
    random.seed(seed)
    random.shuffle(chatml_records)
    
    # 81/19 Split defined in Model Card
    split_idx = int(len(chatml_records) * 0.81)
    train_data = chatml_records[:split_idx]
    valid_data = chatml_records[split_idx:]

    with open(train_file, 'w', encoding='utf-8') as f:
        for record in train_data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    with open(valid_file, 'w', encoding='utf-8') as f:
        for record in valid_data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    logger.info("=== SFT DATASET COMPILATION SUCCESSFUL ===")
    logger.info(f"Total ChatML Sequences: {len(chatml_records)}")
    logger.info(f"Train Split (81%):  {len(train_data)} sequences -> {train_file}")
    logger.info(f"Valid Split (19%):  {len(valid_data)} sequences -> {valid_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bidirectional SFT ChatML Compiler")
    parser.add_argument("--input", type=str, required=True, help="Chunked parallel JSONL")
    parser.add_argument("--train_out", type=str, default="nogai_sft_train.jsonl", help="Train split output")
    parser.add_argument("--valid_out", type=str, default="nogai_sft_valid.jsonl", help="Validation split output")
    parser.add_argument("--seed", type=int, default=42, help="Randomization seed")
    args = parser.parse_args()
    
    compile_dataset(args.input, args.train_out, args.valid_out, args.seed)
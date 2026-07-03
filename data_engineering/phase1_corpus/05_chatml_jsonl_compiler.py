"""
05_jsonl_dataset_compiler.py

NogaiLLM Data Engineering Pipeline - Phase 5
Normalizes casing, enforces strict O(1) set deduplication, and compiles the 
final mathematically isolated text into reproducible JSONL formats natively 
optimized for Hugging Face `datasets` and MLX Continuous Pre-Training.
"""

import json
import random
import logging
import argparse
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize_capitalization_smart(sentence: str) -> str:
    """
    Applies deterministic Title-Casing to the first word of boundaries and 
    maps mid-sentence all-caps artifacts to title/lower casing.
    """
    words = sentence.split()
    if not words: return ""
    
    normalized_words = []
    for i, word in enumerate(words):
        if i == 0:
            normalized_words.append(word.capitalize())
        else:
            if any(char.isupper() for char in word):
                normalized_words.append(word.capitalize())
            else:
                normalized_words.append(word.lower())
                
    return " ".join(normalized_words)

def stream_to_jsonl(data_list: List[str], filepath: Path):
    """Safely streams serialized JSON schemas to disk without memory bloat."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data_list:
            # ensure_ascii=False ensures Cyrillic renders natively in the file
            json_record = json.dumps({"text": item}, ensure_ascii=False)
            f.write(json_record + '\n')

def compile_dataset(input_file: str, train_file: str, valid_file: str, seed: int = 42):
    in_path = Path(input_file)
    if not in_path.exists():
        logger.error(f"Input file {in_path} not found.")
        return

    logger.info(f"Loading pristine corpus from {in_path.name}...")
    
    seen_sentences = set()
    data_units = []
    duplicate_count = 0

    with open(in_path, 'r', encoding='utf-8') as f:
        for line in f:
            cleaned_line = line.strip()
            if cleaned_line:
                normalized_line = normalize_capitalization_smart(cleaned_line)
                if normalized_line:
                    # Strict cryptographic hashing (O(1)) deduplication
                    if normalized_line not in seen_sentences:
                        seen_sentences.add(normalized_line)
                        data_units.append(normalized_line)
                    else:
                        duplicate_count += 1

    # Ensure academic reproducibility
    logger.info(f"Applying uniform batch shuffling (Seed: {seed})...")
    random.seed(seed)
    random.shuffle(data_units)

    # Calculate deterministic split points (95% Train / 5% Validation)
    total_units = len(data_units)
    train_split_idx = int(total_units * 0.95)

    train_data = data_units[:train_split_idx]
    valid_data = data_units[train_split_idx:]

    logger.info("Compiling and serializing to Hugging Face JSONL schema...")
    stream_to_jsonl(train_data, Path(train_file))
    stream_to_jsonl(valid_data, Path(valid_file))

    logger.info("=== DATASET COMPILATION SUCCESSFUL ===")
    logger.info(f"Duplicates Purged:  {duplicate_count}")
    logger.info(f"Total Unique Pairs: {total_units}")
    logger.info(f"Train Split (95%):  {len(train_data)} sequences -> {train_file}")
    logger.info(f"Valid Split (5%):   {len(valid_data)} sequences -> {valid_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hugging Face JSONL Compiler")
    parser.add_argument("--input", type=str, default="PRISTINE_NOGAI_CORPUS.txt", help="Path to normalized corpus")
    parser.add_argument("--train_out", type=str, default="train.jsonl", help="Output train JSONL")
    parser.add_argument("--valid_out", type=str, default="valid.jsonl", help="Output validation JSONL")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    compile_dataset(args.input, args.train_out, args.valid_out, args.seed)
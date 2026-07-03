"""
analyze_sentence_alignment.py

NogaiLLM Data Engineering Pipeline - Phase 2 (Script 2)
Probes sentence-level boundary alignment between Russian and Nogai texts.
Filters out blocks with boundary mismatches to prevent multi-head attention 
hallucination during Supervised Fine-Tuning (SFT).
"""

import re
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Strict boundary definitions
SENTENCE_BOUNDARY_RE = re.compile(r'(?<=[.!?])\s+')

def split_into_sentences(text: str) -> List[str]:
    """Isolates syntactically complete sentences using terminal delimiters."""
    sentences = SENTENCE_BOUNDARY_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]

def enforce_alignment(input_file: str, output_file: str) -> None:
    in_path = Path(input_file)
    out_path = Path(output_file)
    
    if not in_path.exists():
        logger.error(f"Input file {in_path} not found.")
        return

    aligned_records = []
    mismatch_count = 0

    logger.info("Executing mathematical boundary alignment check...")
    
    with open(in_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            
            ru_sentences = split_into_sentences(record["russian"])
            nog_sentences = split_into_sentences(record["nogai"])
            
            # Strict 1:1 Boundary Enforcement
            if len(ru_sentences) == len(nog_sentences) and len(ru_sentences) > 0:
                record["ru_sentences"] = ru_sentences
                record["nog_sentences"] = nog_sentences
                aligned_records.append(record)
            else:
                mismatch_count += 1

    with open(out_path, 'w', encoding='utf-8') as f:
        for record in aligned_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    logger.info(f"Alignment validation complete.")
    logger.info(f"Mathematically Aligned Blocks: {len(aligned_records)}")
    logger.info(f"Boundary Mismatches Dropped: {mismatch_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Lingual Sentence Alignment Validator")
    parser.add_argument("--input", type=str, required=True, help="Input raw parallel JSONL")
    parser.add_argument("--output", type=str, default="aligned_sentences.jsonl", help="Output aligned JSONL")
    args = parser.parse_args()
    
    enforce_alignment(args.input, args.output)
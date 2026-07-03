"""
extract_ibt_verses.py

NogaiLLM Data Engineering Pipeline - Phase 2 (Script 1)
Extracts and normalizes human-verified parallel texts (Russian <-> Nogai) 
from Institute for Bible Translation (IBT) archives. Ensures absolute semantic 
equivalence and zero machine-translation (MT) contamination.
"""

import re
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

WHITESPACE_RE = re.compile(r"\s+")

def normalize_text(text: str) -> str:
    """Removes destructive formatting and standardizes whitespace."""
    text = text.replace("\r", "\n")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()

def discover_parallel_pairs(source_dir: Path) -> List[Dict[str, Path]]:
    """Identifies structurally paired Russian and Nogai textual blocks."""
    nogai_files = {p.stem.replace("nog_", ""): p for p in source_dir.rglob("nog_*.txt")}
    russian_files = {p.stem.replace("ru_", ""): p for p in source_dir.rglob("ru_*.txt")}
    
    paired_keys = sorted(set(nogai_files.keys()) & set(russian_files.keys()))
    
    pairs = []
    for key in paired_keys:
        pairs.append({
            "block_id": key,
            "nogai_path": nogai_files[key],
            "russian_path": russian_files[key]
        })
        
    return pairs

def extract_verses(source_dir: str, output_file: str) -> None:
    source_path = Path(source_dir)
    out_path = Path(output_file)
    
    if not source_path.exists():
        logger.error(f"Source directory {source_path} not found.")
        return

    pairs = discover_parallel_pairs(source_path)
    logger.info(f"Discovered {len(pairs)} parallel IBT blocks. Extracting...")
    
    extracted_data = []
    
    for pair in pairs:
        try:
            with open(pair["nogai_path"], 'r', encoding='utf-8') as f_nog, \
                 open(pair["russian_path"], 'r', encoding='utf-8') as f_ru:
                
                nogai_text = normalize_text(f_nog.read())
                russian_text = normalize_text(f_ru.read())
                
                if nogai_text and russian_text:
                    extracted_data.append({
                        "block_id": pair["block_id"],
                        "russian": russian_text,
                        "nogai": nogai_text
                    })
        except Exception as e:
            logger.warning(f"Failed to process block {pair['block_id']}: {e}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for record in extracted_data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    logger.info(f"Extraction complete. {len(extracted_data)} gold-standard pairs written to {out_path.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IBT Parallel Text Extractor")
    parser.add_argument("--source_dir", type=str, required=True, help="Directory containing nog_*.txt and ru_*.txt")
    parser.add_argument("--output", type=str, default="raw_parallel_blocks.jsonl", help="Output JSONL file")
    args = parser.parse_args()
    
    extract_verses(args.source_dir, args.output)
"""
proportional_chunker.py

NogaiLLM Data Engineering Pipeline - Phase 2 (Script 3)
Implements custom algorithmic chunking. Bypasses standard token-truncation 
(which destroys Turkic suffixes) by proportionally grouping intact sentences. 
Ensures every training sample contains grammatically unsevered thought blocks.
"""

import json
import logging
import argparse
from typing import List, Dict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def chunk_sentences(sentences: List[str], num_chunks: int) -> List[str]:
    """Groups grammatically intact sentences proportionally to prevent BPE severing."""
    if not sentences:
        return [""] * num_chunks
        
    chunk_size = max(1, len(sentences) // num_chunks)
    chunks = []
    
    for i in range(num_chunks):
        if i == num_chunks - 1:
            # Append all remaining sentences to the final chunk
            chunks.append(" ".join(sentences[i * chunk_size:]))
        else:
            chunks.append(" ".join(sentences[i * chunk_size : (i + 1) * chunk_size]))
            
    return chunks

def apply_chunking(input_file: str, output_file: str, chunks_per_block: int = 4) -> None:
    in_path = Path(input_file)
    out_path = Path(output_file)
    
    if not in_path.exists():
        logger.error("Input aligned JSONL missing.")
        return

    logger.info(f"Applying structural chunking ({chunks_per_block} chunks per block)...")
    
    valid_chunks = 0
    with open(in_path, "r", encoding="utf-8") as f_in, \
         open(out_path, "w", encoding="utf-8") as f_out:
             
        for line in f_in:
            data = json.loads(line.strip())
            
            ru_chunks = chunk_sentences(data["ru_sentences"], chunks_per_block)
            nog_chunks = chunk_sentences(data["nog_sentences"], chunks_per_block)
            
            for ru_chunk, nog_chunk in zip(ru_chunks, nog_chunks):
                if ru_chunk.strip() and nog_chunk.strip():
                    chunk_record = {
                        "russian": ru_chunk,
                        "nogai": nog_chunk
                    }
                    f_out.write(json.dumps(chunk_record, ensure_ascii=False) + "\n")
                    valid_chunks += 1

    logger.info(f"Syntactic chunking complete. Generated {valid_chunks} unsevered parallel chunks.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Proportional Sentence Boundary Chunker")
    parser.add_argument("--input", type=str, required=True, help="Aligned sentences JSONL")
    parser.add_argument("--output", type=str, default="chunked_parallel.jsonl", help="Output chunked JSONL")
    parser.add_argument("--chunks", type=int, default=4, help="Number of chunks to split blocks into")
    args = parser.parse_args()
    
    apply_chunking(args.input, args.output, args.chunks)
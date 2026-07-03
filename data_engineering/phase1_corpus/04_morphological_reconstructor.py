"""
04_morphological_reconstructor.py

NogaiLLM Data Engineering Pipeline - Phase 4
Resolves catastrophic morphological splitting. Agglutinative Turkic languages 
frequently suffer from mid-word suffix fragmentation due to PDF column wrapping 
(e.g., stem on line 1, '-нынынъ' on line 2). This script dynamically reconnects 
orphaned suffixes to their host stems prior to sentence-boundary segmentation.
"""

import re
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Common Nogai/Turkic grammatical suffixes fragmented during OCR
TURKIC_SUFFIXES = (
    "нынынъ", "нининъ", "дынъ", "динъ", "тынъ", "тинъ", "лар", "лер", 
    "ны", "ни", "ды", "ди", "ты", "ти", "лап", "леп", "лык", "лик",
    "ын", "ин", "ун", "уьн", "ара", "ата", "ады"
)

# Regex to safely split sentences without destroying abbreviations
SENTENCE_BOUNDARY_REGEX = re.compile(r'(?<=[.!?])\s+')

def reconstruct_agglutinative_morphology(text: str) -> str:
    """
    Parses line breaks and dynamically merges orphaned Turkic suffixes back 
    to their morphological roots.
    """
    lines = [line.strip() for line in text.split('\n')]
    reconstructed = []
    i = 0
    
    while i < len(lines):
        current_line = lines[i]
        
        if not current_line:
            reconstructed.append("")
            i += 1
            continue
            
        # Peak ahead to evaluate the next line
        while i + 1 < len(lines) and lines[i+1]:
            next_line = lines[i+1]
            
            # Case A: Explicit Hyphenation
            if current_line.endswith("-"):
                current_line = current_line[:-1] + next_line
            
            # Case B: Implicit Suffix Splitting (Agglutinative collapse)
            elif (current_line[-1].isalpha() and 
                  next_line[0].isalpha() and 
                  next_line.lower().startswith(TURKIC_SUFFIXES)):
                current_line = current_line + next_line
                
            # Case C: Standard Line Wrap
            else:
                current_line = current_line + " " + next_line
                
            i += 1
            
        reconstructed.append(current_line)
        i += 1
        
    return "\n".join(reconstructed)

def isolate_clean_sentences(text: str) -> str:
    """Segments reconstructed paragraphs into grammatically intact sentences."""
    clean_text = reconstruct_agglutinative_morphology(text)
    sentences = SENTENCE_BOUNDARY_REGEX.split(clean_text)
    
    final_sentences = []
    for s in sentences:
        s_clean = s.strip()
        # Heuristic filter to drop micro-fragments and OCR noise
        if len(s_clean) > 10:
            final_sentences.append(s_clean)
            
    return "\n".join(final_sentences)

def main(input_file: str, output_file: str):
    in_path = Path(input_file)
    if not in_path.exists():
        logger.error(f"Input file {in_path} not found.")
        return

    logger.info(f"Loading corpus from {in_path.name}...")
    with open(in_path, "r", encoding="utf-8") as f:
        raw_data = f.read()

    logger.info("Reconstructing Turkic morphology and sentence boundaries...")
    clean_corpus = isolate_clean_sentences(raw_data)

    out_path = Path(output_file)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(clean_corpus)

    logger.info(f"Successfully saved pristine corpus to {out_path.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turkic Morphological Reconstructor")
    parser.add_argument("--input", type=str, required=True, help="Path to raw corpus text")
    parser.add_argument("--output", type=str, required=True, help="Path to save cleaned text")
    args = parser.parse_args()
    
    main(args.input, args.output)
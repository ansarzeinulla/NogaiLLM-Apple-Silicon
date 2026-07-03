"""
03_cross_lingual_filter.py

NogaiLLM Data Engineering Pipeline - Phase 3
Performs a two-pass statistical frequency analysis and cross-lingual contamination purge.
Detects and removes high-resource Slavic (Russian) structural leakage and OCR loop 
artifacts to mathematically isolate the true Nogai Cyrillic vocabulary distribution.
"""

import re
import nltk
import logging
import argparse
import collections
from pathlib import Path
from nltk.tokenize import sent_tokenize
from typing import Set

# Ensure NLTK punkt is available without spamming logs
nltk.download('punkt', quiet=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# High-frequency Slavic administrative markers indicating domain collapse
RUSSIAN_INDICATORS: Set[str] = {
    "что", "это", "для", "как", "федерации", "федерация", "области", 
    "государственной", "законодательства", "республики", "постановление", 
    "администрации", "района", "соответствии", "президента"
}

# Unique Turkic Cyrillic morphological anchors
NOGAI_INDICATORS: Set[str] = {"аь", "оь", "уь", "нъ", "гъ", "жъ"}

# Compiled regex filters for OCR structural noise
REGEX_ONE_LETTER_DOT = re.compile(r'\b[^\W\d_]\.[^\W\d_]+', re.UNICODE)
REGEX_YEAR_END = re.compile(r'20\d\d\s+йыл\s*$', re.IGNORECASE)
REGEX_START_DIGIT = re.compile(r'^\d\.\d\d')
REGEX_SINGLE_LETTER_SPACE = re.compile(r'(?<=\s)[^\W\d_](?=\s)', re.UNICODE)

def is_cross_lingual_contamination(text: str) -> bool:
    """
    Evaluates token distribution to detect high-resource Russian overlap.
    If Slavic legal/admin words exist without Turkic morphological anchors, 
    the sequence is flagged as contamination.
    """
    text_lower = text.lower()
    has_russian = any(word in text_lower for word in RUSSIAN_INDICATORS)
    has_nogai = any(char in text_lower for char in NOGAI_INDICATORS)
    
    return has_russian and not has_nogai

def build_frequency_distributions(filepath: Path) -> tuple[collections.Counter, collections.Counter]:
    """Pass 1: Builds O(1) lookup tables for repetitive artifact detection."""
    line_counts = collections.Counter()
    sentence_counts = collections.Counter()
    
    with open(filepath, 'r', encoding='utf-8') as infile:
        for line in infile:
            clean_line = line.strip()
            if not clean_line: continue
            
            line_counts[clean_line] += 1
            for s in sent_tokenize(clean_line):
                sentence_counts[s.strip()] += 1
                
    return line_counts, sentence_counts

def filter_corpus(input_path: Path, output_path: Path):
    """Pass 2: Applies structural, statistical, and cross-lingual filters."""
    logger.info("Pass 1: Building global frequency distributions...")
    line_counts, sentence_counts = build_frequency_distributions(input_path)
    
    logger.info("Pass 2: Executing deterministic cross-lingual purge...")
    kept_count, purged_count = 0, 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            raw_line = line.strip()
            if not raw_line: continue

            # 1. Statistical Artifact Filter (>5 repetitions = hallucination loop risk)
            if line_counts[raw_line] > 5:
                purged_count += 1
                continue

            # 2. Structural OCR Filters
            if REGEX_START_DIGIT.match(raw_line) or \
               REGEX_ONE_LETTER_DOT.search(raw_line) or \
               REGEX_YEAR_END.search(raw_line):
                purged_count += 1
                continue

            # 3. Cross-Lingual Contamination Filter
            if is_cross_lingual_contamination(raw_line):
                purged_count += 1
                continue

            # 4. Granular Sentence-Level Processing
            valid_sentences = []
            for s in sent_tokenize(raw_line):
                s_clean = s.strip()
                if sentence_counts[s_clean] > 5: continue
                
                # Cleanup isolated characters left by OCR
                s_cleaned_letters = REGEX_SINGLE_LETTER_SPACE.sub('', s_clean)
                s_cleaned_letters = re.sub(r'\s+', ' ', s_cleaned_letters).strip()

                # Minimum semantic density check (must be > 3 words)
                if len(s_cleaned_letters.split()) > 3:
                    valid_sentences.append(s_cleaned_letters)

            # Reconstruct and verify final density
            if len(valid_sentences) < 3:
                continue

            final_line = " ".join(valid_sentences)
            digit_ratio = sum(char.isdigit() for char in final_line) / len(final_line) if final_line else 1.0
            
            if digit_ratio < 0.20:
                outfile.write(final_line + '\n')
                kept_count += 1
            else:
                purged_count += 1

    logger.info(f"Purge complete. Retained: {kept_count} lines | Purged: {purged_count} lines.")
    logger.info(f"Isolated dataset saved to: {output_path.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical Cross-Lingual Purge")
    parser.add_argument("--input", type=str, required=True, help="Input raw corpus")
    parser.add_argument("--output", type=str, required=True, help="Output isolated corpus")
    args = parser.parse_args()
    
    filter_corpus(Path(args.input), Path(args.output))
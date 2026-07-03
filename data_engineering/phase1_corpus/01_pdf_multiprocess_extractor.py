"""
01_pdf_multiprocess_extractor.py

NogaiLLM Data Engineering Pipeline - Phase 1
Extracts, repairs, and sanitizes text from raw PDF newspaper archives utilizing 
multiprocessing and dynamic font-size heuristics. Includes byte-level OCR repair 
for degraded Cyrillic encodings (cp1251 fallback).
"""

import os
import re
import fitz  # PyMuPDF
import logging
import argparse
import collections
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Dict, List

# Configure production logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Compile regex for performance
MASTHEAD_REGEX = re.compile(r'№\s*\d+\s*\(\d+\)[^\n]*\d{4}\s*йыл[^\n]*', re.IGNORECASE)
LATIN_REGEX = re.compile(r'[a-zA-Z]')
MULTI_SPACE_REGEX = re.compile(r' +')

def fix_cyrillic_ocr(text: str) -> str:
    """
    Attempts byte-level recovery of degraded Cyrillic OCR characters 
    by falling back to the cp1251 encoding table.
    """
    fixed_text = []
    for char in text:
        char_code = ord(char)
        if 128 <= char_code <= 255:
            try:
                fixed_text.append(bytes([char_code]).decode('cp1251'))
            except UnicodeError:
                fixed_text.append(char)
        else:
            fixed_text.append(char)
    return "".join(fixed_text)

def enforce_smart_casing(word: str) -> str:
    """Enforces deterministic capitalization based on the first alphabetic character."""
    first_letter = next((c for c in word if c.isalpha()), None)
    if first_letter:
        return word.upper() if first_letter.isupper() else word.lower()
    return word

def apply_structural_filters(text: str) -> str:
    """Applies cross-lingual and formatting filters to isolate standard Nogai text."""
    text = MASTHEAD_REGEX.sub('', text)
    text = LATIN_REGEX.sub('', text)  # Purge raw Latin artifacts
    
    # Standardize dashes and typography
    replacements = {
        '—': '-', '–': '-', '―': '-', '…': '...', '·': '*', '•': '*', '°': '*',
        '−': '-', '‐': '-', '@': '', '›': '', '~': '=', 'є': '', '№': '',
        '[': '(', '}': ')', '●': '*', '©': '*', '®': '*', '«': '"', '»': '"'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
        
    # Unicode canonicalization
    text = text.replace('\u0301', '').replace('\u0618', "'").replace('\u00AD', '')
    
    cleaned_words = [enforce_smart_casing(w) for w in text.split(' ')]
    return ' '.join(cleaned_words)

def process_single_pdf(pdf_path: Path) -> Tuple[str, collections.Counter]:
    """
    Worker function: Dynamically detects the body-text font size to exclude 
    headers/footers, then extracts and normalizes the text block.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Failed to open {pdf_path}: {e}")
        return "", collections.Counter()

    # Pass 1: Heuristic Font-Size Profiling
    font_sizes = collections.defaultdict(int)
    for page in doc:
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") == 0:  # Text blocks only
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        font_sizes[round(span["size"])] += len(span["text"].strip())
    
    if not font_sizes:
        doc.close()
        return "", collections.Counter()
    
    body_size = max(font_sizes, key=font_sizes.get)
    extracted_text = []

    # Pass 2: Contextual Extraction
    for page in doc:
        for block in page.get_text("dict", sort=True).get("blocks", []):
            if block.get("type") != 0: continue
            
            paragraph_text = []
            for line in block.get("lines", []):
                line_text = ""
                for span in line.get("spans", []):
                    if round(span["size"]) <= body_size + 2:
                        line_text += fix_cyrillic_ocr(span["text"])
                
                line_text = line_text.strip()
                if not line_text: continue
                
                if line_text.endswith('-'):
                    paragraph_text.append(line_text[:-1])
                else:
                    paragraph_text.append(line_text + " ")
            
            raw_para = "".join(paragraph_text).strip()
            raw_para = MULTI_SPACE_REGEX.sub(' ', raw_para)
            clean_para = apply_structural_filters(raw_para)
            
            if clean_para:
                extracted_text.append(clean_para)
                
    doc.close()
    logger.info(f"Successfully processed: {pdf_path.name}")
    
    final_text = "\n".join(extracted_text)
    return final_text, collections.Counter(final_text)

def main(input_dir: str, output_file: str):
    pdf_dir = Path(input_dir)
    if not pdf_dir.exists():
        logger.error(f"Directory {pdf_dir} not found.")
        return

    pdf_files = sorted(pdf_dir.rglob("*.pdf"))
    if not pdf_files:
        logger.error("No PDFs found.")
        return

    logger.info(f"Initializing multiprocessing pool for {len(pdf_files)} documents...")
    pdf_texts: Dict[Path, str] = {}
    global_char_counter = collections.Counter()

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_single_pdf, p): p for p in pdf_files}
        for future in as_completed(futures):
            pdf_path = futures[future]
            try:
                text, local_counter = future.result()
                pdf_texts[pdf_path] = text
                global_char_counter.update(local_counter)
            except Exception as e:
                logger.error(f"Exception processing {pdf_path.name}: {e}")

    # Aggregation mapping
    with open(output_file, "w", encoding="utf-8") as f:
        for p in pdf_files:
            if text := pdf_texts.get(p):
                f.write(text + "\n\n")
                
    logger.info(f"Pipeline complete. Output written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multiprocessing PDF Extractor for Zero-Resource NLP")
    parser.add_argument("--input_dir", type=str, default="pdfs", help="Directory containing raw PDFs")
    parser.add_argument("--output_file", type=str, default="combined.txt", help="Output text file")
    args = parser.parse_args()
    
    main(args.input_dir, args.output_file)
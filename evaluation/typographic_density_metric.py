"""
typographic_density_metric.py

NogaiLLM Evaluation Suite - Typographic Density (TD)
TD = 100 * (occurrences of the Nogai digraphs аь, оь, уь, нъ) / (number of words).

The denominator is WORDS, not characters. Human Nogai text scores about 22.8 on
this scale (Nogai-Unified-Corpus-v1: 22.74 on valid, 22.88 on train), so a model's
TD is only meaningful next to that reference: well below it means the output is
drifting to Russian, far above it means a repetition loop. Use --reference to
print the reference and each file's distance from it.
"""

import json
import re
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Nogai-specific digraphs
NOGAI_DIGRAPHS_RE = re.compile(r'(аь|оь|уь|нъ|Аь|Оь|Уь|Нъ)')
MODEL_OUTPUT_RE = re.compile(r"MODEL:\s*(.*)")

def density_of_text(text: str) -> float:
    """TD of a string: Nogai digraphs per 100 words."""
    words = text.split()
    if not words:
        return 0.0
    return 100 * len(NOGAI_DIGRAPHS_RE.findall(text)) / len(words)


def reference_density(jsonl_path: Path) -> float:
    """TD of human Nogai text, e.g. the corpus valid.jsonl."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        text = " ".join(json.loads(line)["text"] for line in f if line.strip())
    return round(density_of_text(text), 2)


def calculate_density(filepath: Path) -> float:
    """TD of the MODEL: lines in an evaluation output file (digraphs per 100 words)."""
    if not filepath.exists():
        return 0.0
        
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        
    model_outputs = MODEL_OUTPUT_RE.findall(content)
    combined_text = " ".join(model_outputs)
    
    if not combined_text.strip():
        return 0.0 # Strict penalty for Mode Collapse (e.g., Llama outputting blank spaces)
        
    words = combined_text.split()
    total_words = len(words)
    if total_words == 0:
        return 0.0
        
    nogai_markers = NOGAI_DIGRAPHS_RE.findall(combined_text)
    density = (len(nogai_markers) / total_words) * 100
    
    return round(density, 2)

def execute_ablation_study(results_dir: str, reference_path: str = None):
    target_dir = Path(results_dir)
    if not target_dir.exists():
        logger.error(f"Directory {target_dir} not found.")
        return

    result_files = sorted(target_dir.glob("results_*.txt"))
    if not result_files:
        logger.warning("No result files found for ablation analysis.")
        return

    logger.info("="*50)
    logger.info(" NOGAI TYPOGRAPHIC DENSITY ABLATION RESULTS")
    logger.info("="*50)
    
    reference = reference_density(Path(reference_path)) if reference_path else None
    if reference is not None:
        logger.info(f"{'human reference (' + Path(reference_path).name + ')':40} : {reference:>6}")
    for file_path in result_files:
        density = calculate_density(file_path)
        line = f"{file_path.name.ljust(40)} : {density:>6}"
        if reference is not None:
            line += f"   |TD - reference| = {abs(density - reference):.2f}"
        logger.info(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Typographic Density Ablation Calculator")
    parser.add_argument("--dir", type=str, default=".", help="Directory containing result txt files")
    parser.add_argument("--reference", type=str, default=None,
                        help="JSONL of human Nogai text (e.g. phase1 valid.jsonl) for the reference TD")
    args = parser.parse_args()

    execute_ablation_study(args.dir, args.reference)
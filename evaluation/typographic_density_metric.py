"""
typographic_density_metric.py

NogaiLLM Evaluation Suite - Architectural Ablation Metric
Calculates the Morphological Typographic Density of generated sequences.
Measures the frequency of native Nogai Cyrillic digraphs (аь, оь, уь, нъ)
to objectively quantify structural transfer and out-of-distribution (OOD) decay.
"""

import re
import os
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Core Cyrillic Turkic mathematical markers
NOGAI_DIGRAPHS_RE = re.compile(r'(аь|оь|уь|нъ|Аь|Оь|Уь|Нъ)')
MODEL_OUTPUT_RE = re.compile(r"MODEL:\s*(.*)")

def calculate_density(filepath: Path) -> float:
    """Calculates the percentage of text composed of Nogai-specific morphology."""
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

def execute_ablation_study(results_dir: str):
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
    
    for file_path in result_files:
        density = calculate_density(file_path)
        # Formats the output perfectly for academic markdown tables
        logger.info(f"{file_path.name.ljust(40)} : {density:>5}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Typographic Density Ablation Calculator")
    parser.add_argument("--dir", type=str, default=".", help="Directory containing result txt files")
    args = parser.parse_args()
    
    execute_ablation_study(args.dir)
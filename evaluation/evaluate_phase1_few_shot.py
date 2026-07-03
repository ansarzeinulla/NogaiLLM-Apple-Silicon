"""
evaluate_phase1_few_shot.py

NogaiLLM Evaluation Suite - Phase 1
Evaluates the raw morphological acquisition of the Continuous Pre-Trained (CPT) 
adapter. Bypasses standard instruction templates (ChatML) to prevent the model 
from collapsing due to the Catastrophic Forgetting induced during Phase 1.
"""

import logging
import argparse
from pathlib import Path
from mlx_lm import load, generate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_few_shot(model_path: str, adapter_path: str, source_file: str, target_file: str, output_file: str):
    logger.info(f"Loading Base Architecture: {model_path}")
    if adapter_path:
        logger.info(f"Injecting Phase 1 LoRA Weights: {adapter_path}")
        model, tokenizer = load(model_path, adapter_path=adapter_path)
    else:
        model, tokenizer = load(model_path)

    with open(source_file, "r", encoding="utf-8") as f:
        source_sentences = [line.strip() for line in f if line.strip()]

    with open(target_file, "r", encoding="utf-8") as f:
        target_sentences = [line.strip() for line in f if line.strip()]

    if len(source_sentences) != len(target_sentences):
        logger.warning("Source and Target files have mismatched sequence counts.")

    logger.info(f"Initiating Unconditional Few-Shot Evaluation on {len(source_sentences)} sequences...")

    with open(output_file, "w", encoding="utf-8") as out:
        for i, source_text in enumerate(source_sentences):
            
            # Structurally rigid Few-Shot prompt mapping without ChatML
            prompt = (
                "English: The boy is reading a book.\n"
                "Nogai: Яс китап окуйды.\n"
                "English: The weather is good today.\n"
                "Nogai: Буьгуьн куьн аьруьв.\n"
                f"English: {source_text}\n"
                "Nogai:"
            )
            
            response = generate(
                model, 
                tokenizer, 
                prompt=prompt, 
                max_tokens=30,  # Strict limit to prevent autoregressive looping
                verbose=False
            )
            
            # Isolate the predicted semantic block
            translation = response.strip().split('\n')[0]

            out.write(f"--- Sequence {i+1} ---\n")
            out.write(f"SRC:   {source_text}\n")
            out.write(f"TRUE:  {target_sentences[i] if i < len(target_sentences) else 'N/A'}\n")
            out.write(f"MODEL: {translation}\n\n")
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i+1}/{len(source_sentences)} sequences.")

    logger.info(f"Phase 1 Evaluation complete. Metrics saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 Unconditional Few-Shot Evaluator")
    parser.add_argument("--model", type=str, required=True, help="Path to base model")
    parser.add_argument("--adapter", type=str, default=None, help="Path to Phase 1 LoRA adapter")
    parser.add_argument("--source", type=str, default="en.txt", help="Source language text file")
    parser.add_argument("--target", type=str, default="nog.txt", help="Target language ground truth")
    parser.add_argument("--output", type=str, default="results_phase1_few_shot.txt", help="Output file")
    args = parser.parse_args()
    
    evaluate_few_shot(args.model, args.adapter, args.source, args.target, args.output)
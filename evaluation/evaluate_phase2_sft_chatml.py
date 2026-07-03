"""
evaluate_phase2_sft_chatml.py

NogaiLLM Evaluation Suite - Phase 2
Validates the restoration of prompt-adherence following Phase 2 Supervised 
Fine-Tuning (SFT). Tests the model's ability to process ChatML structural 
vectors and generate morphologically accurate zero-shot translations.
"""

import logging
import argparse
from pathlib import Path
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_sft_chatml(model_path: str, adapter_path: str, input_file: str, output_file: str, temp: float):
    logger.info(f"Loading Fused Base: {model_path}")
    logger.info(f"Injecting Phase 2 SFT Weights: {adapter_path}")
    model, tokenizer = load(model_path, adapter_path=adapter_path)

    in_path = Path(input_file)
    if not in_path.exists():
        logger.error(f"Evaluation dataset {in_path} not found.")
        return

    with open(in_path, "r", encoding="utf-8") as f:
        source_sentences = [line.strip() for line in f if line.strip()]

    logger.info(f"Initiating ChatML Zero-Shot Evaluation on {len(source_sentences)} sequences...")
    
    # MLX architectural requirement for deterministic/probabilistic generation
    sampler = make_sampler(temp=temp)

    with open(output_file, "w", encoding="utf-8") as out:
        for i, text in enumerate(source_sentences):
            
            messages = [
                {"role": "system", "content": "You are a highly accurate bilingual translator for Russian and Nogai."},
                {"role": "user", "content": f"Переведи этот текст на русский язык: {text}"}
            ]
            
            # Map semantic inputs to latent structural tokens
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            response = generate(
                model, 
                tokenizer, 
                prompt=prompt, 
                max_tokens=128, 
                sampler=sampler,
                verbose=False
            )
            
            # Isolate completion before EOS or hallucinatory trailing
            translation = response.strip().split('\n')[0]

            out.write(f"--- Sequence {i+1} ---\n")
            out.write(f"SRC:   {text}\n")
            out.write(f"MODEL: {translation}\n\n")
            
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i+1}/{len(source_sentences)} sequences.")

    logger.info(f"Phase 2 SFT Evaluation complete. Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 ChatML SFT Evaluator")
    parser.add_argument("--model", type=str, required=True, help="Fused base model path")
    parser.add_argument("--adapter", type=str, required=True, help="Phase 2 SFT adapter path")
    parser.add_argument("--input", type=str, default="nog.txt", help="Input text for translation")
    parser.add_argument("--output", type=str, default="results_phase2_sft.txt", help="Output file")
    parser.add_argument("--temp", type=float, default=0.3, help="Sampling temperature")
    args = parser.parse_args()
    
    evaluate_sft_chatml(args.model, args.adapter, args.input, args.output, args.temp)
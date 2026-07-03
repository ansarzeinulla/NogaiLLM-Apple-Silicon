import os
import json
import torch
import gradio as gr
from pathlib import Path
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from safetensors.torch import load_file, save_file
from threading import Thread
from transformers import TextIteratorStreamer

# --- 1. CONFIGURATION ---
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_ID = "ansarzeinulla/Qwen2.5-1.5B-Nogai-SFT-Experimental"
CONVERTED_DIR = Path("./converted_peft")


# --- 2. AUTOMATED MLX-TO-PEFT CONVERTER ---
def convert_mlx_to_peft():
    
    if (CONVERTED_DIR / "adapter_config.json").exists() and (CONVERTED_DIR / "adapter_model.safetensors").exists():
        return

    CONVERTED_DIR.mkdir(parents=True, exist_ok=True)
    mlx_config_path = hf_hub_download(repo_id=ADAPTER_ID, filename="adapter_config.json")
    mlx_weights_path = hf_hub_download(repo_id=ADAPTER_ID, filename="adapters.safetensors")

    with open(mlx_config_path, "r") as f:
        mlx_config = json.load(f)

    lora_params = mlx_config.get("lora_parameters", {})
    peft_config = {
        "base_model_name_or_path": BASE_MODEL_ID,
        "lora_alpha": lora_params.get("alpha", 32),
        "lora_dropout": lora_params.get("dropout", 0.0),
        "peft_type": "LORA",
        "r": lora_params.get("rank", 16),
        "target_modules": ["q_proj", "v_proj"],
        "task_type": "CAUSAL_LM",
    }

    with open(CONVERTED_DIR / "adapter_config.json", "w") as f:
        json.dump(peft_config, f, indent=2)

    mlx_weights = load_file(mlx_weights_path)
    peft_weights = {}

    for mlx_key, tensor in mlx_weights.items():
        peft_key = mlx_key.replace("model.layers", "base_model.model.model.layers")
        
        # 1. Cast adapter weights to float32 to prevent CPU matmul dtype crashes
        tensor = tensor.to(torch.float32)

        # 2. Safely capture both standard mlx-lm key structures (.lora_a and .lora_a.weight)
        if ".lora_a" in peft_key:
            peft_key = peft_key.replace(".lora_a.weight", ".lora_A.default.weight")
            peft_key = peft_key.replace(".lora_a", ".lora_A.default.weight")
            tensor = tensor.t()
            
        elif ".lora_b" in peft_key:
            peft_key = peft_key.replace(".lora_b.weight", ".lora_B.default.weight")
            peft_key = peft_key.replace(".lora_b", ".lora_B.default.weight")
            tensor = tensor.t()
            
        peft_weights[peft_key] = tensor.contiguous()

    save_file(peft_weights, CONVERTED_DIR / "adapter_model.safetensors")

# --- 3. SYSTEM LAUNCH ---
convert_mlx_to_peft()
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
# Hardcoded to CPU to ensure stability on Free Tier
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float32, device_map="cpu")
model = PeftModel.from_pretrained(base_model, str(CONVERTED_DIR))
model.eval()

# --- 4. INFERENCE ENGINE ---
def generate_translation(message, history):
    messages = [{"role": "system", "content": "You are a highly accurate bilingual translator for Russian and Nogai."}]
    if history:
        for interaction in history:
            # Assuming interaction is a list/tuple of [user_msg, bot_msg]
            if isinstance(interaction, (list, tuple)) and len(interaction) >= 2:
                messages.append({"role": "user", "content": interaction[0]})
                messages.append({"role": "assistant", "content": interaction[1]})
    messages.append({"role": "user", "content": message})
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([prompt], return_tensors="pt")
    
    # Increased timeout to 120s for CPU inference
    streamer = TextIteratorStreamer(tokenizer, timeout=120.0, skip_prompt=True, skip_special_tokens=True)
    
    def run_generation():
        try:
            model.generate(**model_inputs, streamer=streamer, max_new_tokens=128, temperature=0.3, do_sample=True)
        except Exception as e:
            print(f"Generation error: {e}")

    t = Thread(target=run_generation)
    t.start()
    
    partial_message = ""
    for new_token in streamer:
        partial_message += new_token
        yield partial_message

# --- 5. GRADIO UI ---
with gr.Blocks() as demo:
    gr.HTML("<h1 style='text-align: center;'>⚡️ NogaiLLM: Zero-Resource Turkic NLP</h1>")
    chatbot = gr.ChatInterface(
        fn=generate_translation,
        examples=["Переведи с русского языка на ногайский язык: В начале сотворил Бог небо и землю.", "Переведи с ногайского языка на русский язык: Бизикилер, яшавлары пайдасыз болмасын."],
        cache_examples=False
    )
    
demo.launch()
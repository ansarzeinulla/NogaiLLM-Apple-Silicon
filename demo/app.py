import json
import re
from pathlib import Path
from threading import Thread

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# --- 1. CONFIGURATION ---
# The Phase 2 SFT adapter was trained on top of the Phase 1 CPT model, so both
# adapters are needed: Phase 1 is merged into the base first, then Phase 2 is applied.
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
PHASE1_ADAPTER_ID = "ansarzeinulla/Qwen2.5-1.5B-Nogai-LoRA"
PHASE2_ADAPTER_ID = "ansarzeinulla/Qwen2.5-1.5B-Nogai-SFT-Experimental"
SYSTEM_PROMPT = "You are a highly accurate bilingual translator for Russian and Nogai."


# --- 2. MLX-TO-PEFT CONVERTER ---
def convert_mlx_to_peft(adapter_id: str, out_dir: Path) -> Path:
    """Converts an mlx-lm LoRA adapter (adapters.safetensors) into a PEFT adapter.

    Target modules and layers are read from the adapter's own tensor names, so every
    trained matrix is loaded (the published adapters train q/k/v/o/gate/up/down in
    layers 12-27, not only q_proj/v_proj).
    """
    if (out_dir / "adapter_config.json").exists() and (out_dir / "adapter_model.safetensors").exists():
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(hf_hub_download(repo_id=adapter_id, filename="adapter_config.json")) as f:
        mlx_config = json.load(f)
    mlx_weights = load_file(hf_hub_download(repo_id=adapter_id, filename="adapters.safetensors"))

    lora = mlx_config.get("lora_parameters", {})
    rank = lora.get("rank", 8)
    # mlx-lm applies delta = scale * (x @ A @ B); PEFT uses lora_alpha / r, so alpha = scale * r.
    scale = lora.get("scale", 20.0)
    layers = sorted({int(m.group(1)) for k in mlx_weights if (m := re.search(r"layers\.(\d+)\.", k))})
    modules = sorted({k.split(".lora_")[0].rsplit(".", 1)[-1] for k in mlx_weights})

    peft_config = {
        "base_model_name_or_path": BASE_MODEL_ID,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": rank,
        "lora_alpha": scale * rank,
        "lora_dropout": lora.get("dropout", 0.0),
        "target_modules": modules,
        "layers_to_transform": layers,
    }
    with open(out_dir / "adapter_config.json", "w") as f:
        json.dump(peft_config, f, indent=2)

    peft_weights = {}
    for mlx_key, tensor in mlx_weights.items():
        # mlx stores lora_a as (in, r) and lora_b as (r, out); PEFT expects (r, in) and (out, r).
        peft_key = "base_model.model." + mlx_key
        peft_key = peft_key.replace(".lora_a", ".lora_A.weight").replace(".lora_b", ".lora_B.weight")
        peft_weights[peft_key] = tensor.to(torch.float32).t().contiguous()
    save_file(peft_weights, out_dir / "adapter_model.safetensors")
    return out_dir


# --- 3. SYSTEM LAUNCH ---
phase1_dir = convert_mlx_to_peft(PHASE1_ADAPTER_ID, Path("./converted_phase1"))
phase2_dir = convert_mlx_to_peft(PHASE2_ADAPTER_ID, Path("./converted_phase2"))

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
# CPU-only for the free Space tier.
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float32, device_map="cpu")
model = PeftModel.from_pretrained(base_model, str(phase1_dir)).merge_and_unload()  # Phase 1 CPT
model = PeftModel.from_pretrained(model, str(phase2_dir))  # Phase 2 SFT on top
model.eval()


# --- 4. INFERENCE ENGINE ---
def generate_translation(message, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for interaction in history:
            # gradio >= 5 passes history as ChatML-style dicts; older versions
            # pass [user_msg, bot_msg] pairs. Support both.
            if isinstance(interaction, dict) and interaction.get("role") in ("user", "assistant"):
                content = interaction.get("content")
                if isinstance(content, str) and content:
                    messages.append({"role": interaction["role"], "content": content})
            elif isinstance(interaction, (list, tuple)) and len(interaction) >= 2:
                messages.append({"role": "user", "content": interaction[0]})
                messages.append({"role": "assistant", "content": interaction[1]})
    messages.append({"role": "user", "content": message})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([prompt], return_tensors="pt")

    # Generous timeout for CPU inference.
    streamer = TextIteratorStreamer(tokenizer, timeout=120.0, skip_prompt=True, skip_special_tokens=True)

    def run_generation():
        try:
            model.generate(**model_inputs, streamer=streamer, max_new_tokens=128, temperature=0.3,
                           do_sample=True)
        except Exception as e:
            print(f"Generation error: {e}")

    Thread(target=run_generation).start()

    partial_message = ""
    for new_token in streamer:
        partial_message += new_token
        yield partial_message


# --- 5. GRADIO UI ---
with gr.Blocks() as demo:
    gr.HTML("<h1 style='text-align: center;'>NogaiLLM: Russian ↔ Nogai (experimental)</h1>")
    gr.ChatInterface(
        fn=generate_translation,
        # Same instruction format as the SFT training data.
        examples=[
            "Переведи этот текст на ногайский язык: В начале сотворил Бог небо и землю.",
            "Переведи этот текст на русский язык: Бизикилер, яшавлары пайдасыз болмасын.",
        ],
        cache_examples=False,
    )

demo.launch()

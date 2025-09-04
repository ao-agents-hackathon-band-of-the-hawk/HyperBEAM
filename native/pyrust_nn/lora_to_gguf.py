from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess
import os
import tempfile
import logging

logger = logging.getLogger(__name__)

def lora_to_gguf(params):
    # --- START OF FIX ---
    # Get the project root from the parameters provided by Rust. Default to '.' for standalone runs.
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_script_dir, "llama.cpp", "convert_lora_to_gguf.py")
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Conversion script not found at expected path: {script_path}")
    # --- END OF FIX ---

    model_name = params.get("model_name", "Qwen/Qwen3-1.7B")
    adapter_path = os.path.abspath(params["adapter_path"])
    gguf_precision = params.get("gguf_precision", "f16")
    gguf_output_path_lora = params.get("gguf_output_path_lora", os.path.join(adapter_path, "lora.gguf"))
    
    output_dir = os.path.dirname(gguf_output_path_lora)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with tempfile.TemporaryDirectory() as temp_base_dir:
        logger.info(f"Downloading base model: {model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model.save_pretrained(temp_base_dir)
        tokenizer.save_pretrained(temp_base_dir)
        logger.info("Base model downloaded and saved to temp dir.")
        
        try:
            # Use the full script_path variable
            subprocess.run([
                "python", script_path,
                adapter_path,
                "--base", temp_base_dir,
                "--outfile", gguf_output_path_lora,
                "--outtype", gguf_precision
            ], check=True)
            logger.info(f"Converted LoRA to GGUF at {gguf_output_path_lora}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"GGUF conversion failed with exit code {e.returncode}: {e.stderr or e.stdout}")
    
    return gguf_output_path_lora
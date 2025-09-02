from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess
import os
import tempfile
import logging


logger = logging.getLogger(__name__)

def lora_to_gguf(params):
    """Merges LoRA adapter and converts to GGUF."""
    if "adapter_path" not in params:
        raise ValueError("Adapter path not provided.")
    
    model_name = params.get("model_name", "Qwen/Qwen1.5-1.8B-Chat")
    adapter_path = os.path.abspath(params["adapter_path"])
    gguf_precision = params.get("gguf_precision", "f16")
    gguf_output_path_lora = params.get("gguf_output_path_lora", os.path.join(adapter_path, "lora.gguf"))
    
    if not os.path.isdir(adapter_path):
        raise ValueError(f"Adapter directory '{adapter_path}' does not exist.")
        
    # --- START OF FIX ---
    # Ensure the output directory for the GGUF file exists before running the subprocess.
    output_dir = os.path.dirname(gguf_output_path_lora)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Ensured output directory exists: {output_dir}")
    # --- END OF FIX ---
    
    with tempfile.TemporaryDirectory() as temp_base_dir:
        logger.info(f"Downloading base model: {model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model.save_pretrained(temp_base_dir)
        tokenizer.save_pretrained(temp_base_dir)
        logger.info("Base model downloaded and saved to temp dir.")
        
        try:
            subprocess.run([
                "python", "llama.cpp/convert_lora_to_gguf.py",
                adapter_path,
                "--base", temp_base_dir,
                "--outfile", gguf_output_path_lora,
                "--outtype", gguf_precision
            ], check=True)
            logger.info(f"Converted LoRA to GGUF at {gguf_output_path_lora}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"GGUF conversion failed with exit code {e.returncode}: {e.stderr or e.stdout}")
        except Exception as e:
            raise RuntimeError(f"Error in GGUF conversion: {str(e)}")
    
    return gguf_output_path_lora

if __name__ == "__main__":
    sample_params = {"adapter_path": "models/lora_adapter", "gguf_precision": "q8_0"}
    lora_to_gguf(sample_params)
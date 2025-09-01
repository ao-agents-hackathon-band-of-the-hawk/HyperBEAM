from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess  # Assuming llama.cpp convert tool
import os
import tempfile
import logging

# --- START: CORRECTED LOGGING SETUP ---
# 1. Get the root logger.
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# 2. Clear any existing handlers.
if root_logger.hasHandlers():
    root_logger.handlers.clear()

# 3. Create a formatter.
formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')

# 4. Create and add the file handler.
file_handler = logging.FileHandler('activity.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

# 5. Create and add the console handler.
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

# 6. Get the logger for this module.
logger = logging.getLogger(__name__)
# --- END: CORRECTED LOGGING SETUP ---

def lora_to_gguf(params):
    """Merges LoRA adapter and converts to GGUF."""
    if "adapter_path" not in params:
        raise ValueError("Adapter path not provided. Please specify 'adapter_path' in params.")
    
    model_name = params.get("model_name", "Qwen/Qwen1.5-1.8B-Chat")
    adapter_path = os.path.abspath(params["adapter_path"])
    gguf_precision = params.get("gguf_precision", "f16")
    if gguf_precision not in ["f32", "f16", "bf16", "q8_0", "auto"]:
        raise ValueError(f"Invalid gguf_precision '{gguf_precision}'. Supported values are: f32, f16, bf16, q8_0, auto")
    gguf_output_path_lora = os.path.join(adapter_path, params.get("gguf_output_path_lora", "lora.gguf"))
    
    if not os.path.isdir(adapter_path):
        raise ValueError(f"Adapter directory '{adapter_path}' does not exist. Ensure LoRA fine-tuning has been run first and the path is correct.")
    
    # Create a temporary directory for the base model
    with tempfile.TemporaryDirectory() as temp_base_dir:
        # Download base model and tokenizer to temp dir
        logger.info("Downloading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model.save_pretrained(temp_base_dir)
        tokenizer.save_pretrained(temp_base_dir)
        logger.info("Base model downloaded and saved to temp dir.")
        
        # Convert to GGUF using the local base model path
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
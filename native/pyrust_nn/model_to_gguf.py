from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess
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

def model_to_gguf(params):
    """Converts full model to GGUF."""
    if "model_name" not in params:
        raise ValueError("Model name or path not provided. Please specify 'model_name' in params.")
    
    model_name = params["model_name"]
    gguf_output_path_model = params.get("gguf_output_path_model", "models/model.gguf")
    gguf_precision = params.get("gguf_precision", "f16")
    if gguf_precision not in ["f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"]:
        raise ValueError(f"Invalid gguf_precision '{gguf_precision}'. Supported values are: f32, f16, bf16, q8_0, tq1_0, tq2_0, auto")
    # Ensure output directory exists if dirname is not empty
    dirname = os.path.dirname(gguf_output_path_model)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    
    model_path = model_name
    is_local = model_name.startswith("models/")
    
    if is_local:
        if not os.path.isdir(model_path):
            raise ValueError(f"Local model directory '{model_path}' does not exist.")
        logger.info(f"Using local model from {model_path}")
    else:
        # Create a temporary directory for the model if downloading
        model_path = tempfile.mkdtemp()
        logger.info("Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        logger.info("Model downloaded and saved to temp dir.")
    
    # Convert to GGUF using the model path (local or temp)
    try:
        subprocess.run([
            "python", "llama.cpp/convert_hf_to_gguf.py",
            model_path,
            "--outfile", gguf_output_path_model,
            "--outtype", gguf_precision
        ], check=True)
        logger.info(f"Converted model to GGUF at {gguf_output_path_model}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"GGUF conversion failed with exit code {e.returncode}: {e.stderr or e.stdout}")
    except Exception as e:
        raise RuntimeError(f"Error in GGUF conversion: {str(e)}")
    finally:
        # Clean up temp dir if it was used
        if not is_local:
            import shutil
            shutil.rmtree(model_path)
    
    return gguf_output_path_model

if __name__ == "__main__":
    sample_params = {"model_name": "Qwen/Qwen1.5-1.8B-Chat", "gguf_output_path_model": "models/gguf/model.gguf", "gguf_precision": "tq1_0"}
    model_to_gguf(sample_params)
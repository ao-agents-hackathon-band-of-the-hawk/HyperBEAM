from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess
import os
import tempfile
import logging
import shutil

logger = logging.getLogger(__name__)

def model_to_gguf(params):
    """Converts full model to GGUF."""
    if "model_name" not in params:
        raise ValueError("Model name or path not provided.")
    
    model_name = params["model_name"]
    gguf_output_path_model = params.get("gguf_output_path_model", "models/model.gguf")
    gguf_precision = params.get("gguf_precision", "f16")
    
    # --- START OF FIX ---
    # Ensure the output directory for the GGUF file exists.
    output_dir = os.path.dirname(gguf_output_path_model)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Ensured output directory exists: {output_dir}")
    # --- END OF FIX ---
    
    model_path = model_name
    # A more robust check for local vs. Hub model
    is_local = os.path.isdir(model_name)
    temp_dir = None
    
    try:
        if is_local:
            logger.info(f"Using local model from {model_path}")
        else:
            # Create a temporary directory for the model if downloading
            temp_dir = tempfile.mkdtemp()
            model_path = temp_dir
            logger.info(f"Downloading model '{model_name}' to temporary directory...")
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            logger.info("Model downloaded successfully.")
    
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
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
    
    return gguf_output_path_model

if __name__ == "__main__":
    sample_params = {"model_name": "Qwen/Qwen1.5-1.8B-Chat", "gguf_output_path_model": "models/gguf/model.gguf"}
    model_to_gguf(sample_params)
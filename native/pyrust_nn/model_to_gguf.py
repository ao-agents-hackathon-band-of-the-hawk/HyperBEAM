from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess
import os
import tempfile
import logging
import shutil

logger = logging.getLogger(__name__)

def model_to_gguf(params):
    # --- START OF FIX ---
    # Get the project root from the parameters provided by Rust.
    project_root = params.get("project_root", ".")
    script_path = os.path.join(project_root, "llama.cpp", "convert_hf_to_gguf.py")
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Conversion script not found at expected path: {script_path}")
    # --- END OF FIX ---

    model_name = params["model_name"]
    gguf_output_path_model = params.get("gguf_output_path_model", "models/model.gguf")
    gguf_precision = params.get("gguf_precision", "f16")
    
    output_dir = os.path.dirname(gguf_output_path_model)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    model_path = model_name
    is_local = os.path.isdir(model_name)
    temp_dir = None
    
    try:
        if not is_local:
            temp_dir = tempfile.mkdtemp()
            model_path = temp_dir
            logger.info(f"Downloading model '{model_name}'...")
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
        
        # Use the full script_path variable
        subprocess.run([
            "python", script_path,
            model_path,
            "--outfile", gguf_output_path_model,
            "--outtype", gguf_precision
        ], check=True)
        logger.info(f"Converted model to GGUF at {gguf_output_path_model}")
    
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    return gguf_output_path_model
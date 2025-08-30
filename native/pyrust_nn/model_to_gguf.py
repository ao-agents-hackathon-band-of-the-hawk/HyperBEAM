from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess
import os
import tempfile

def model_to_gguf(params):
    """Converts full model to GGUF."""
    if "model_name" not in params:
        raise ValueError("Model name or path not provided. Please specify 'model_name' in params.")
    
    model_name = params["model_name"]
    gguf_output_path_model = params.get("gguf_output_path_model", "models/model.gguf")
    
    # Ensure output directory exists if dirname is not empty
    dirname = os.path.dirname(gguf_output_path_model)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    
    # Create a temporary directory for the model
    with tempfile.TemporaryDirectory() as temp_model_dir:
        # Download model and tokenizer to temp dir
        print("Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.save_pretrained(temp_model_dir)
        tokenizer.save_pretrained(temp_model_dir)
        print("Model downloaded and saved to temp dir.")
        
        # Convert to GGUF using the local model path
        try:
            subprocess.run([
                "python", "llama.cpp/convert_hf_to_gguf.py",
                temp_model_dir,
                "--outfile", gguf_output_path_model,
                "--outtype", "f16"
            ], check=True)
            print(f"Converted model to GGUF at {gguf_output_path_model}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"GGUF conversion failed with exit code {e.returncode}: {e.stderr or e.stdout}")
        except Exception as e:
            raise RuntimeError(f"Error in GGUF conversion: {str(e)}")
    
    return gguf_output_path_model

if __name__ == "__main__":
    sample_params = {"model_name": "Qwen/Qwen3-0.6B"}
    model_to_gguf(sample_params)
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess  # Assuming llama.cpp convert tool
import os
import tempfile

def lora_to_gguf(params):
    """Merges LoRA adapter and converts to GGUF."""
    if "adapter_path" not in params:
        raise ValueError("Adapter path not provided. Please specify 'adapter_path' in params.")
    
    model_name = params.get("model_name", "Qwen/Qwen1.5-1.8B-Chat")
    adapter_path = os.path.abspath(params["adapter_path"])
    gguf_output_path_lora = os.path.join(adapter_path, params.get("gguf_output_path_lora", "lora.gguf"))
    
    if not os.path.isdir(adapter_path):
        raise ValueError(f"Adapter directory '{adapter_path}' does not exist. Ensure LoRA fine-tuning has been run first and the path is correct.")
    
    # Create a temporary directory for the base model
    with tempfile.TemporaryDirectory() as temp_base_dir:
        # Download base model and tokenizer to temp dir
        print("Downloading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model.save_pretrained(temp_base_dir)
        tokenizer.save_pretrained(temp_base_dir)
        print("Base model downloaded and saved to temp dir.")
        
        # Convert to GGUF using the local base model path
        try:
            subprocess.run([
                "python", "llama.cpp/convert_lora_to_gguf.py",
                adapter_path,
                "--base", temp_base_dir,
                "--outfile", gguf_output_path_lora,
                "--outtype", "f16"
            ], check=True)
            print(f"Converted LoRA to GGUF at {gguf_output_path_lora}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"GGUF conversion failed with exit code {e.returncode}: {e.stderr or e.stdout}")
        except Exception as e:
            raise RuntimeError(f"Error in GGUF conversion: {str(e)}")
    
    return gguf_output_path_lora

if __name__ == "__main__":
    sample_params = {"adapter_path": "models/lora_adapter"}
    lora_to_gguf(sample_params)
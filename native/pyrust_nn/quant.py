import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gc
import os

def get_model_memory_footprint(model):
    mem = sum(p.numel() * p.element_size() for p in model.parameters())
    return mem / (1024 * 1024)

def quantize_model(params):
    """Quantizes model to specified precision and saves it locally."""
    model_name = params.get("model_name", "Qwen/Qwen3-0.6B")
    precision = params.get("precision", "8-bit")
    prompt = params.get("prompt", "The future of artificial intelligence is")
    quant_output_dir = params.get("quant_output_dir", f"models/quant/{precision}")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    
    quantization_config = None
    torch_dtype = torch.float32
    
    if precision == "16-bit":
        torch_dtype = torch.bfloat16
    elif precision == "8-bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        torch_dtype = None
    elif precision == "4-bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
        )
        torch_dtype = None
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, quantization_config=quantization_config, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    mem_footprint = get_model_memory_footprint(model)
    print(f"Model memory footprint ({precision}): {mem_footprint:.2f} MB")
    
    # Inference example
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Save the quantized model and tokenizer
    os.makedirs(quant_output_dir, exist_ok=True)
    model.save_pretrained(quant_output_dir)
    tokenizer.save_pretrained(quant_output_dir)
    print(f"Quantized model saved to {quant_output_dir}")
    
    gc.collect()
    torch.cuda.empty_cache()
    return {"generated_text": generated_text, "memory_mb": mem_footprint, "saved_dir": quant_output_dir}

if __name__ == "__main__":
    sample_params = {"precision": "8-bit"}
    print(quantize_model(sample_params))
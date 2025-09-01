import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_inference(params):
    """Runs inference on a model."""
    model_name = params.get("model_name", "Qwen/Qwen1.5-1.8B-Chat")
    prompt = params.get("prompt", "Give me a short introduction to large language model.")
    max_new_tokens = params.get("max_new_tokens", 512)
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded.")
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    print("Model loaded.")
    
    try:
        model = torch.compile(model)
        print("Model compiled successfully.")
    except Exception as e:
        print(f"Model compilation failed with error: {e}")
        print("Proceeding without compilation.")


    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    
    print("Thinking content:", thinking_content)
    print("Content:", content)
    
    return {"thinking_content": thinking_content, "content": content}

if __name__ == "__main__":
    sample_params = {"prompt": "Test prompt"}
    print(run_inference(sample_params))
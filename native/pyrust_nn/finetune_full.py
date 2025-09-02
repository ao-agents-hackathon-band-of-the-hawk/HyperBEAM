# finetune_full.py (updated with ETA callback)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer , AutoConfig, TrainerCallback
from datasets import  Dataset
import json
import gc
import os
from collections import OrderedDict
from safetensors.torch import load_file
import logging


logger = logging.getLogger(__name__)

class ETACallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and 'loss' in logs:  # Log at each logging step
            if state.global_step > 0 and state.max_steps > 0:
                logger.info(f"Step {state.global_step}/{state.max_steps} ")

# Modified to accept the tokenizer as an argument
def data_loader(dataset_path, tokenizer, sample_start=0, max_length=512):
    """Loads dataset from text or JSON file, starting from sample_start."""
  
    with open(dataset_path, 'r') as f:
        data = json.load(f)[sample_start:]
        dataset = data  # For Qwen format, load as list of {"messages": [...]}
    
    # This nested function now has access to the 'tokenizer' passed to data_loader
    def tokenize_function(examples):
        if "messages" in examples:  # Handle chat format for JSON
            # Format messages using chat template to get string
            formatted_chat = tokenizer.apply_chat_template(examples["messages"], tokenize=False, add_generation_prompt=False)
            tokenized = tokenizer(formatted_chat, truncation=True, padding="max_length", max_length=max_length)
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            # Mask labels for non-assistant parts (set to -100 except for assistant responses)
            # To do this accurately, tokenize parts separately
            input_ids = []
            labels = []
            for msg in examples["messages"]:
                msg_text = f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
                msg_tokens = tokenizer.encode(msg_text, add_special_tokens=False)
                input_ids.extend(msg_tokens)
                if msg["role"] == "assistant":
                    labels.extend(msg_tokens)
                else:
                    labels.extend([-100] * len(msg_tokens))
            
            # Truncate and pad
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]
            padding_length = max_length - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * padding_length
            labels += [-100] * padding_length
            
            tokenized["input_ids"] = input_ids
            tokenized["labels"] = labels
            tokenized["attention_mask"] = [1 if tid != tokenizer.pad_token_id else 0 for tid in input_ids]
        else:  # Handle text format
            tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)
            tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    ds = Dataset.from_list(dataset)
    return ds.map(tokenize_function, batched=False, remove_columns=ds.column_names)

def fine_tune_full(params):
    """Full fine-tuning function."""
    # The 'global tokenizer' line is no longer needed
    model_name =  "Qwen/Qwen1.5-1.8B-Chat"
    dataset_path = "data.json"
    output_dir = params.get("output_dir", "models/finetuned")
    num_epochs = params.get("num_epochs", 1)
    batch_size = params.get("batch_size", 1)
    learning_rate = params.get("learning_rate", 2e-5)
    sample_start = params.get("sample_start", 0)
    max_length = params.get("max_length", 512)
    gradient_checkpointing = params.get("gradient_checkpointing", True)
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for fine-tuning.")
    
    # --- START: ADDED CODE ---
    # 1. Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 2. Set the padding token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # --- END: ADDED CODE ---

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, use_cache=False)
    
    # --- START: MODIFIED CODE ---
    # 3. Pass the tokenizer to the data_loader function
    tokenized_dataset = data_loader(dataset_path, tokenizer, sample_start, max_length)
    # --- END: MODIFIED CODE ---
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=batch_size,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=200,
        gradient_checkpointing=gradient_checkpointing,
        save_total_limit=2,
        label_names=["labels"],
        report_to="none",
        remove_unused_columns=False, 
    )
    
    torch.set_float32_matmul_precision('high')
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset, tokenizer=tokenizer, callbacks=[ETACallback()])
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"Finetuned model is saved at {output_dir}")
    return output_dir

if __name__ == "__main__":
    sample_params = {"dataset_path":"data.json", "output_dir": "models/finetuned"}
    fine_tune_full(sample_params)
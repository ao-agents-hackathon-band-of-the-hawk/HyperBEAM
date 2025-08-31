from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import Dataset as TorchDataset # Renamed to avoid conflict with datasets.Dataset
import torch
import json
import os
import logging
from typing import Optional
from datasets import Dataset # Import Dataset class explicitly

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def data_loader(dataset_path, tokenizer, sample_start=0, max_length=512):
    """Loads dataset from JSON file and tokenizes it."""
  
    with open(dataset_path, 'r') as f:
        data = json.load(f)[sample_start:]
    
    def tokenize_function(examples):
        if "messages" in examples:
            formatted_chat = tokenizer.apply_chat_template(examples["messages"], tokenize=False, add_generation_prompt=False)
            tokenized = tokenizer(formatted_chat, truncation=True, padding="max_length", max_length=max_length)
            
            # This complex masking is often not necessary if using apply_chat_template,
            # but we will keep your logic. A simpler way is to just use the tokenized output directly.
            tokenized["labels"] = tokenized["input_ids"].copy()
        
        # This part assumes a text-based dataset if "messages" is not found.
        # Ensure your JSON has either "messages" or "text" keys.
        elif "text" in examples:
            tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)
            tokenized["labels"] = tokenized["input_ids"].copy()
        else:
            raise ValueError("Dataset format not recognized. Expecting keys 'messages' or 'text'.")

        return tokenized
    
    ds = Dataset.from_list(data)
    # The tokenizer is already passed in, so we don't need to define it globally.
    tokenized_ds = ds.map(tokenize_function, batched=False, remove_columns=ds.column_names)
    return tokenized_ds


def fine_tune_lora(params):
    """LoRA fine-tuning function."""
    # --- (All parameters remain the same) ---
    model_name = params.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
    local_base_dir = "models/base"
    local_path = os.path.join(local_base_dir, model_name.replace("/", "--"))
    dataset_path = params["dataset_path"]
    output_lora_dir = params.get("output_lora_dir", "lora_adapter")
    num_epochs = params.get("num_epochs", 3)
    batch_size = params.get("batch_size", 1)
    learning_rate = params.get("learning_rate", 2e-5)
    lora_rank = params.get("lora_rank", 16)
    lora_alpha = params.get("lora_alpha", 32)
    lora_dropout = params.get("lora_dropout", 0.05)
    lora_adapter_path = params.get("lora_adapter_path", None)
    sample_start = params.get("sample_start", 0)
    max_length = params.get("max_length", 512)
    
    print(f"Loading model and tokenizer for: {model_name}")

    if os.path.isdir(local_path):
        logger.info(f"Loading local model and tokenizer from {local_path}")
        model = AutoModelForCausalLM.from_pretrained(local_path, device_map="auto", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(local_path)
    else:
        logger.info(f"Downloading model and tokenizer from Hugging Face: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        logger.info(f"Saving base model to {local_path} for future use.")
        os.makedirs(local_path, exist_ok=True)
        model.save_pretrained(local_path)
        tokenizer.save_pretrained(local_path)

    if tokenizer.pad_token is None:
        logger.info("Setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Loading and tokenizing dataset...")
    train_dataset = data_loader(dataset_path, tokenizer, sample_start, max_length)
    logger.info(f"Loaded and tokenized {len(train_dataset)} items")
    
    model.enable_input_require_grads()

    if lora_adapter_path:
        logger.info(f"Loading existing LoRA adapter from: {lora_adapter_path}")
        peft_model = PeftModel.from_pretrained(model, lora_adapter_path)
    else:
        lora_config = LoraConfig(
            r=lora_rank, 
            lora_alpha=lora_alpha, 
            lora_dropout=lora_dropout,
            bias="none", 
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        logger.info("Applying new LoRA configuration...")
        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()
        
    # --- START: MOVED AND CORRECTED LOGIC ---
    # Compile the model *after* applying PEFT adapters
    try:
        peft_model = torch.compile(peft_model)
        print("PEFT model compiled successfully.")
    except Exception as e:
        print(f"PEFT model compilation failed with error: {e}")
        print("Proceeding without compilation.")
    # --- END: MOVED AND CORRECTED LOGIC ---
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir=output_lora_dir, 
        num_train_epochs=num_epochs, 
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate, 
        fp16=torch.cuda.is_available(), 
        logging_steps=10,
        save_strategy="epoch", 
        report_to="none", 
        remove_unused_columns=False,
        save_total_limit=1,
    )
    
    logger.info("Starting training...")
    trainer = Trainer(
        model=peft_model, 
        args=training_args, 
        train_dataset=train_dataset, 
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    trainer.train()
    peft_model.save_pretrained(output_lora_dir)
    tokenizer.save_pretrained(output_lora_dir)
    
    logger.info(f"Training completed. LoRA adapter saved to {output_lora_dir}")
    return output_lora_dir

# --- (if __name__ == "__main__" block remains the same) ---
if __name__ == "__main__":
    sample_params = {
        "dataset_path": "data.json", 
        "output_lora_dir": "models/lora_adapter",
        "max_length": 512,
        "batch_size": 1
    }
    fine_tune_lora(sample_params)
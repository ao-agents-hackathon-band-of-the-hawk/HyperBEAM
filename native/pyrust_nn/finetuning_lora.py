from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import Dataset # Renamed to avoid conflict with datasets.Dataset
import torch
import json
import os
import logging
from datasets import Dataset 


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
    model_name = params.get("model_name", "Qwen/Qwen1.5-1.8B-Chat")
    # --- START: CORRECTED LOGIC ---
    # Define the path for an OPTIONAL local cache. The script will no longer create this directory.
    local_base_dir = "models/base"
    local_path = os.path.join(local_base_dir, model_name.replace("/", "--"))
    # --- END: CORRECTED LOGIC ---
    dataset_path = params["dataset_path"]
    output_lora_dir = params.get("output_lora_dir", "lora_adapter")
    num_epochs = params.get("num_epochs", 3)
    batch_size = params.get("batch_size", 1)
    learning_rate = params.get("learning_rate", 2e-5)
    lora_rank = params.get("lora_rank", 16)
    lora_alpha = params.get("lora_alpha", 32)
    lora_dropout = params.get("lora_dropout", 0.05)
    lora_adapter_path = params.get("checkpoint_lora", None)
    sample_start = params.get("sample_start", 0)
    max_length = params.get("max_length", 512)
    
    logger.info(f"Loading model and tokenizer for: {model_name}")

    # --- START: MODIFIED MODEL LOADING ---
    # This logic now checks for an optional local directory.
    # If it doesn't exist, it loads from the Hub without saving a new copy.
    if os.path.isdir(local_path):
        logger.info(f"Loading local model and tokenizer from {local_path}")
        model = AutoModelForCausalLM.from_pretrained(local_path, device_map="auto", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(local_path)
    else:
        logger.info(f"Local model not found. Downloading from Hugging Face Hub: {model_name}")
        # Load directly from the Hub. This will use the central Hugging Face cache
        # without creating the 'models/base' directory in your project.
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    # --- END: MODIFIED MODEL LOADING ---

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
        peft_model.logger.info_trainable_parameters()
        
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    torch.set_float32_matmul_precision('high')
    training_args = TrainingArguments(
        output_dir=output_lora_dir, 
        num_train_epochs=num_epochs, 
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate, 
        bf16=torch.cuda.is_available(), 
        logging_steps=10,
        save_strategy="epoch", 
        report_to="none", 
        remove_unused_columns=False,
        save_total_limit=1,
        label_names=["labels"]
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
    logger.info(f"Training completed. LoRA adapter saved to {output_lora_dir}")
    return output_lora_dir
    
   

# --- (if __name__ == "__main__" block remains the same) ---
if __name__ == "__main__":
    sample_params = {
        "dataset_path": "data.json", 
        "output_lora_dir": "models/lora_adapter",
        "max_length": 512,
        "num_epochs": 5,
        "batch_size": 1
    }
    fine_tune_lora(sample_params)
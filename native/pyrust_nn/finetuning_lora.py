from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments
)
# Import Trainer separately to handle potential issues
try:
    from transformers import Trainer
except ImportError:
    print("Warning: Could not import Trainer directly")
    from transformers.trainer import Trainer

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
import torch
import json

class SimpleDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        }

def data_loader(dataset_path, dataset_type="text", text_column="text", sample_start=0, max_length=512):
    """Loads dataset from text or JSON file, starting from sample_start."""
    if dataset_type == "text":
        with open(dataset_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[sample_start:]
        texts = [line.strip() for line in lines if line.strip()]
    elif dataset_type == "json":
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)[sample_start:]
        texts = [item[text_column] for item in data if text_column in item]
    else:
        raise ValueError("Unsupported dataset_type.")
    return texts

def fine_tune_lora(params):
    """LoRA fine-tuning function."""
    model_name = params.get("model_name", "Qwen/Qwen3-0.6B")
    dataset_path = params["dataset_path"]
    dataset_type = params.get("dataset_type", "text")
    text_column = params.get("text_column", "text")
    output_lora_dir = params.get("output_lora_dir", "lora_adapter")
    num_epochs = params.get("num_epochs", 3)
    batch_size = params.get("batch_size", 4)
    learning_rate = params.get("learning_rate", 2e-5)
    lora_rank = params.get("lora_rank", 16)
    lora_alpha = params.get("lora_alpha", 32)
    lora_dropout = params.get("lora_dropout", 0.05)
    checkpoint = params.get("checkpoint", None)
    sample_start = params.get("sample_start", 0)
    max_length = params.get("max_length", 128)
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, 
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    print("Loading dataset...")
    texts = data_loader(dataset_path, dataset_type, text_column, sample_start, max_length)
    print(f"Loaded {len(texts)} texts")
    
    train_dataset = SimpleDataset(texts, tokenizer, max_length)
    
    # Enable input gradients
    model.enable_input_require_grads()
    
    # Prepare model for training (only if using quantization)
    if hasattr(model, 'is_loaded_in_8bit') or hasattr(model, 'is_loaded_in_4bit'):
        model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_rank, 
        lora_alpha=lora_alpha, 
        lora_dropout=lora_dropout,
        bias="none", 
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]  # Simplified target modules
    )
    
    print("Applying LoRA configuration...")
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    
    training_args = TrainingArguments(
        output_dir=output_lora_dir, 
        num_train_epochs=num_epochs, 
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate, 
        fp16=torch.cuda.is_available(), 
        logging_steps=10,
        save_strategy="epoch", 
        report_to="none", 
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    
    print("Starting training...")
    trainer = Trainer(
        model=peft_model, 
        args=training_args, 
        train_dataset=train_dataset, 
        tokenizer=tokenizer
    )
    
    trainer.train(resume_from_checkpoint=checkpoint)
    peft_model.save_pretrained(output_lora_dir)
    tokenizer.save_pretrained(output_lora_dir)
    
    print(f"Training completed. Model saved to {output_lora_dir}")
    return output_lora_dir

if __name__ == "__main__":
    sample_params = {"dataset_path": "data.txt", "output_lora_dir": "test_lora"}
    fine_tune_lora(sample_params)
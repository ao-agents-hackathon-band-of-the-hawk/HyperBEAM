from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import Dataset
import torch
import json
import os
import logging
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CustomTrainer(Trainer):
    pass

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info(f"TextDataset initialized with {len(texts)} texts")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        if idx == 0:
            logger.debug(f"Processing first text sample: {text[:100]}...")
        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        }

class ChatDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info(f"ChatDataset initialized with {len(data)} conversations")
        if data:
            logger.info(f"Sample conversation (first item): {json.dumps(data[0], indent=2)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        messages = self.data[idx]["messages"]
        input_ids_list = []
        labels_list = []
        full_text = ""
        for i, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                text = f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                text = f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                text = f"<|im_start|>assistant\n{content}<|im_end|>\n"
            else:
                logger.warning(f"Unknown role '{role}' in message at index {i} for item {idx}")
                text = ""
            full_text += text
            msg_tokens = self.tokenizer.encode(text, add_special_tokens=False)
            input_ids_list.extend(msg_tokens)
            if role == "assistant":
                labels_list.extend(msg_tokens)
            else:
                labels_list.extend([-100] * len(msg_tokens))
        
        if len(input_ids_list) > self.max_length:
            logger.info(f"Truncating item {idx} from {len(input_ids_list)} to {self.max_length} tokens")
            input_ids_list = input_ids_list[:self.max_length]
            labels_list = labels_list[:self.max_length]
        
        padding_length = self.max_length - len(input_ids_list)
        input_ids_list += [self.tokenizer.pad_token_id] * padding_length
        labels_list += [-100] * padding_length
        attention_mask = [1 if tid != self.tokenizer.pad_token_id else 0 for tid in input_ids_list]
        
        if idx == 0:
            logger.debug(f"Processed first chat sample (full formatted text): {full_text[:200]}...")
        
        return {
            "input_ids": torch.tensor(input_ids_list),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels_list)
        }

def data_loader(dataset_path, dataset_type="text", text_column="text", sample_start=0, max_length=512):
    logger.info(f"Loading data from {dataset_path} (type: {dataset_type}, start from sample {sample_start})")
    if dataset_type == "text":
        with open(dataset_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[sample_start:]
        texts = [line.strip() for line in lines if line.strip()]
        logger.info(f"Loaded {len(texts)} text lines")
        if texts:
            logger.info(f"Sample text: {texts[0][:100]}...")
        return texts
    elif dataset_type == "json":
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)[sample_start:]
        logger.info(f"Loaded {len(data)} JSON items")
        if data:
            logger.info(f"Sample JSON item: {json.dumps(data[0], indent=2)}")
        return data
    else:
        raise ValueError("Unsupported dataset_type.")

def fine_tune_lora(params):
    """LoRA fine-tuning function."""
    model_name = params.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
    local_base_dir = "models/base"
    local_path = os.path.join(local_base_dir, model_name.replace("/", "--"))
    dataset_path = params["dataset_path"]
    dataset_type = params.get("dataset_type", "text")
    text_column = params.get("text_column", "text")
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

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load base model without quantization
    if os.path.isdir(local_path):
        logger.info(f"Loading local model from {local_path}")
        model = AutoModelForCausalLM.from_pretrained(
            local_path, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
    else:
        logger.info(f"Downloading model from Hugging Face: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        os.makedirs(local_path, exist_ok=True)
        tokenizer.save_pretrained(local_path)
        model.save_pretrained(local_path)
        logger.info(f"Model saved to {local_path} for future use.")

    if tokenizer.pad_token is None:
        logger.info("Setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Loading dataset...")
    data = data_loader(dataset_path, dataset_type, text_column, sample_start, max_length)
    logger.info(f"Loaded {len(data)} items")
    
    if dataset_type == "json" and all("messages" in item for item in data):
        train_dataset = ChatDataset(data, tokenizer, max_length)
    else:
        train_dataset = TextDataset(data, tokenizer, max_length)
    
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
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
        logger.info("Applying new LoRA configuration...")
        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
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
        save_total_limit=1,
        label_names=["labels"],
    )
    
    logger.info("Starting training...")
    trainer = CustomTrainer(
        model=peft_model, 
        args=training_args, 
        train_dataset=train_dataset, 
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    
    trainer.train()
    peft_model.save_pretrained(output_lora_dir)
    tokenizer.save_pretrained(output_lora_dir)
    
    logger.info(f"Training completed. Model saved to {output_lora_dir}")
    return output_lora_dir

if __name__ == "__main__":
    sample_params = {
        "dataset_path": "data.json", 
        "dataset_type": "json",
        "output_lora_dir": "models/lora_adapter",
        "max_length": 512,
        "batch_size": 1
        # To train further on existing adapter, add: "lora_adapter_path": "path/to/existing/lora_adapter"
    }
    fine_tune_lora(sample_params)
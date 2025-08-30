import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import json
import gc

def data_loader(dataset_path, dataset_type="text", text_column="text", sample_start=0, max_length=512):
    """Loads dataset from text or JSON file, starting from sample_start."""
    if dataset_type == "text":
        with open(dataset_path, 'r') as f:
            lines = f.readlines()[sample_start:]
        dataset = [{"text": line.strip()} for line in lines if line.strip()]
    elif dataset_type == "json":
        with open(dataset_path, 'r') as f:
            data = json.load(f)[sample_start:]
        dataset = data  # For Qwen format, load as list of {"messages": [...]}
    else:
        raise ValueError("Unsupported dataset_type. Use 'text' or 'json'.")
    
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
            tokenized = tokenizer(examples[text_column], truncation=True, padding="max_length", max_length=max_length)
            tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    from datasets import Dataset
    ds = Dataset.from_list(dataset)
    return ds.map(tokenize_function, batched=False, remove_columns=ds.column_names)  # Use batched=False for per-example processing

def fine_tune_full(params):
    """Full fine-tuning function."""
    global tokenizer
    model_name = params.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
    dataset_path = params["dataset_path"]
    dataset_type = params.get("dataset_type", "text")
    text_column = params.get("text_column", "text")
    output_dir = params.get("output_dir", "models/finetuned")
    num_epochs = params.get("num_epochs", 1)
    batch_size = params.get("batch_size", 1)  # Reduced to 1 to lower memory usage
    learning_rate = params.get("learning_rate", 2e-5)
    sample_start = params.get("sample_start", 0)
    max_length = params.get("max_length", 512)
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for fine-tuning.")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float32)
    
    tokenized_dataset = data_loader(dataset_path, dataset_type, text_column, sample_start, max_length)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=256,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=200,
        gradient_checkpointing=True,
        fp16=True,
        save_total_limit=2,
        report_to="none",
    )
    
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset, tokenizer=tokenizer)
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    return output_dir

if __name__ == "__main__":
    # Test with sample params
    sample_params = {"dataset_path":"data.json", "output_dir": "models/finetuned"}
    fine_tune_full(sample_params)
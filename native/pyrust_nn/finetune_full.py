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
        dataset = [{text_column: item[text_column]} for item in data if text_column in item]
    else:
        raise ValueError("Unsupported dataset_type. Use 'text' or 'json'.")
    
    def tokenize_function(examples):
        tokenized = tokenizer(examples[text_column], truncation=True, padding="max_length", max_length=max_length)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    from datasets import Dataset
    ds = Dataset.from_list(dataset)
    return ds.map(tokenize_function, batched=True, remove_columns=[text_column])

def fine_tune_full(params):
    """Full fine-tuning function."""
    global tokenizer
    model_name = params.get("model_name", "Qwen/Qwen3-0.6B")
    dataset_path = params["dataset_path"]
    dataset_type = params.get("dataset_type", "text")
    text_column = params.get("text_column", "text")
    output_dir = params.get("output_dir", "./models/finetuned-full")
    num_epochs = params.get("num_epochs", 1)
    batch_size = params.get("batch_size", 2)
    learning_rate = params.get("learning_rate", 2e-5)
    checkpoint = params.get("checkpoint", None)
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
        gradient_accumulation_steps=4,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
        resume_from_checkpoint=checkpoint
    )
    
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset, tokenizer=tokenizer)
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    return output_dir

if __name__ == "__main__":
    # Test with sample params
    sample_params = {"dataset_path":"data.txt", "output_dir": "./test_finetune"}
    fine_tune_full(sample_params)
// src/main.rs

use pyrust_nn::*; // Import all public functions and structs from our lib
use chrono::Local;
use std::fs;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    // === 1. SETUP ===
    // Generate a single, unique session ID for this entire run.
    let session_id = Local::now().format("%Y%m%d-%H%M%S").to_string();
    println!("{}\n--- Starting Pipeline Run ---\nSession ID: {}\n{}", "=".repeat(60), session_id, "=".repeat(60));

    // Define the base directory for all outputs for this session.
    let session_path = PathBuf::from("runs").join(&session_id);

    // Define all the nested subdirectories for each step.
    let full_finetune_path = session_path.join("finetune");
    let lora_finetune_path = session_path.join("lora");
    // let quantize_path = session_path.join("quantize");
    // let gguf_path = session_path.join("gguf");
    
    // Create the top-level session directory. Sub-functions will create the rest.
    fs::create_dir_all(&session_path)?;

    // === 2. DEFINE PARAMETERS ===
    let base_model_id = "Qwen/Qwen1.5-0.5B-Chat";
    let dataset_file = "data.json";

    let finetune_params = FinetuneParams {
        dataset_path: dataset_file.to_string(),
        num_epochs: Some(1),
    };

    // === 3. EXECUTE WORKFLOW ===
    
    // Step A: Full fine-tuning
    let finetuned_model_path = finetune_full(base_model_id, &finetune_params, &full_finetune_path)?;
    println!("\n✅ Full fine-tuning complete. Model saved in: {:?}", finetuned_model_path);

    // Step B: LoRA fine-tuning
    let lora_adapter_path = finetune_lora(base_model_id, &finetune_params, &lora_finetune_path)?;
    println!("\n✅ LoRA fine-tuning complete. Adapter saved in: {:?}", lora_adapter_path);
    
    // ... you would add subsequent steps here, like quantization and GGUF conversion,
    // passing the output path from the previous step as input to the next.
    
    println!("\n{}\n--- Pipeline Run {} Complete! ---\n{}", "=".repeat(60), session_id, "=".repeat(60));
    Ok(())
}
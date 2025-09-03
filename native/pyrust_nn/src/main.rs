// src/main.rs

use pyrust_nn::*; // Import all public functions and structs from lib.rs
use std::env;
use std::fs::{self, File};
use std::path::PathBuf;
use simplelog::{Config, LevelFilter, WriteLogger, CombinedLogger, TermLogger, TerminalMode};
use anyhow::Result;

fn main() -> Result<()> {
    // === 1. SETUP SESSION AND CENTRALIZED LOGGER (ONCE!) ===
    let session_id = "RandomSession";
    let session_path = PathBuf::from("runs").join(&session_id);
    fs::create_dir_all(&session_path)?;

    // Get the project root directory. This is crucial context for GGUF conversion.
    let project_root = env::current_dir()?;

    // Initialize the global logger ONCE to a single pipeline.log file for the entire run.
    let log_path = session_path.join("pipeline.log");
    CombinedLogger::init(vec![
        // The console logger remains clean at the INFO level.
        TermLogger::new(LevelFilter::Info, Config::default(), TerminalMode::Mixed, simplelog::ColorChoice::Auto),
        // The file logger now captures EVERYTHING from both Rust and Python at the DEBUG level.
        WriteLogger::new(LevelFilter::Debug, Config::default(), File::create(&log_path)?),
    ]).expect("Failed to initialize the global logger");
    
    log::info!("--- Starting New Pipeline Run ---");
    log::info!("--- Session ID: {} ---", session_id);
    log::info!("--- All artifacts will be saved under: {:?} ---", &session_path);
    log::info!("--- Project Root: {:?} ---", &project_root);
    
    // === 2. DEFINE PARAMETERS ===
    let base_model_id = "Qwen/Qwen1.5-1.8B-Chat";
    let dataset_file = "data.json";

    let finetune_full_params = FinetuneFullParams {
        dataset_path: dataset_file.to_string(),
        num_epochs: Some(1),
        batch_size: Some(1),
        learning_rate: Some(0.00002),
    };

    let finetune_lora_params = FinetuneLoraParams {
        dataset_path: dataset_file.to_string(),
        num_epochs: Some(50),
        batch_size: Some(1),
        learning_rate: Some(0.00002),
        lora_rank: Some(8),
        lora_alpha: Some(32),
        lora_dropout: Some(0.05),
        checkpoint_lora: "".to_string(),
    };
    
    let gguf_params = ConvertToGgufParams {
        gguf_precision: "q8_0".to_string(),
    };

    // === 3. EXECUTE THE WORKFLOW ===

    // Step 1: LoRA Fine-tuning
    let lora_adapter_path = match finetune_lora(&session_id, base_model_id, &finetune_lora_params) {
        Ok(path) => {
            log::info!("\n✅ LoRA fine-tuning complete. Adapter saved in: {}", path);
            path
        }
        Err(e) => {
            log::error!("\n❌ LoRA fine-tuning failed: {}", e);
            return Err(e); // Stop the pipeline on critical failure
        }
    };

    // Step 2: Convert the LoRA adapter to GGUF
    match convert_lora_to_gguf(&session_id, &project_root, base_model_id, &lora_adapter_path, &gguf_params) {
        Ok(path) => log::info!("\n✅ LoRA to GGUF conversion complete. File at: {}", path),
        Err(e) => log::error!("\n❌ LoRA to GGUF conversion failed: {}", e),
    }

    log::info!("\n{}\n", "-".repeat(60)); // Visual separator

    // Step 3: Full Fine-tuning
    let finetuned_model_path = match finetune_full(&session_id, base_model_id, &finetune_full_params) {
        Ok(path) => {
            log::info!("\n✅ Full fine-tuning complete. Model saved in: {}", path);
            path
        }
        Err(e) => {
            log::error!("\n❌ Full fine-tuning failed: {}", e);
            return Err(e);
        }
    };
    
    // Step 4: Convert the fully fine-tuned model to GGUF
    match convert_model_to_gguf(&session_id, &project_root, &finetuned_model_path, &gguf_params) {
        Ok(path) => log::info!("\n✅ Full model to GGUF conversion complete. File at: {}", path),
        Err(e) => log::error!("\n❌ Full model to GGUF conversion failed: {}", e),
    }

    // Step 5: Final Inference Test
    match run_inference(&session_id, &finetuned_model_path, "What is the capital of Taiwan?") {
        Ok(result) => log::info!("\n✅ Final inference complete. Result: '{}'", result.content),
        Err(e) => log::error!("\n❌ Final inference failed: {}", e),
    }

    log::info!("\n--- Pipeline Run {} Complete! ---", session_id);
    Ok(())
}
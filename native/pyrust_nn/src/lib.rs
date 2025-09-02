// src/lib.rs

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::Bound; // Correct top-level import for Bound
use serde::Serialize;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use anyhow::{Result, anyhow};

// --- Public Parameter Structs for the API ---

#[derive(Clone, Debug, Serialize)]
pub struct FinetuneFullParams {
    pub dataset_path: String,
    pub num_epochs: Option<u32>,
    pub batch_size: Option<u32>,
    pub learning_rate: Option<f32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FinetuneLoraParams {
    pub dataset_path: String,
    pub num_epochs: Option<u32>,
    pub batch_size: Option<u32>,
    pub learning_rate: Option<f32>,
    pub lora_rank: Option<u32>,
    pub lora_alpha: Option<u32>,
    pub lora_dropout: Option<f32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct QuantizeParams {
    pub precision: String, // e.g., "8-bit", "4-bit"
}

#[derive(Clone, Debug, Serialize)]
pub struct ConvertToGgufParams {
    pub gguf_precision: String, // e.g., "q8_0", "f16"
}

// --- Public Result Structs ---

#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct QuantizationResult {
    pub saved_dir: String,
}


// --- Internal Helpers ---

#[derive(Serialize)]
struct RunSummary<T: Serialize> {
    parameters: T,
    status: String,
    output_path: Option<String>,
    error_message: Option<String>,
}

/// Writes the final JSON summary for a step.
fn write_summary<T: Serialize>(step_path: &Path, parameters: T, result: &Result<String>) -> Result<()> {
    let (status, output_path, error_message) = match result {
        Ok(path) => ("Success".to_string(), Some(path.clone()), None),
        Err(e) => ("Failure".to_string(), None, Some(e.to_string())),
    };
    let summary = RunSummary { parameters, status, output_path, error_message };
    let summary_path = step_path.join("summary.json");
    fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;
    Ok(())
}

/// A single, safe function to run a Python callback with the environment set up.
fn with_python_env<F, T>(callback: F) -> Result<T>
where
    F: FnOnce(Python) -> Result<T, PyErr>,
    T: Send,
{
    Python::with_gil(|py| {
        // Forcefully configure Python's root logger to be fully verbose.
        let logging = py.import("logging")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("level", logging.getattr("DEBUG")?)?;
        kwargs.set_item("force", true)?; 
        logging.call_method("basicConfig", (), Some(kwargs))?;

        // Correct, lifetime-safe pattern for sys.path modification
        let sys = py.import("sys")?;
        let path_attr = sys.getattr("path")?;
        let path: &Bound<PyList> = path_attr.downcast()?;
        let current_dir = env::current_dir()?.to_str().unwrap().to_string();
        if !path.contains(&current_dir)? {
            path.insert(0, &current_dir)?;
        }
        callback(py)
    }).map_err(anyhow::Error::from)
}

// --- High-Level Public API Functions ---

pub fn finetune_full(session_id: &str, model_id: &str, params: &FinetuneFullParams) -> Result<String> {
    let step_path = PathBuf::from("runs").join(session_id).join("finetune_full");
    fs::create_dir_all(&step_path)?;
    log::info!("--- Starting: Full Fine-tuning ---");
    
    let result = with_python_env(|py| {
        let func = py.import("finetune_full")?.getattr("fine_tune_full")?;
        let py_params = PyDict::new(py);
        py_params.set_item("model_name", model_id)?;
        py_params.set_item("dataset_path", &params.dataset_path)?;
        py_params.set_item("output_dir", step_path.to_str())?;
        if let Some(val) = params.num_epochs { py_params.set_item("num_epochs", val)?; }
        if let Some(val) = params.batch_size { py_params.set_item("batch_size", val)?; }
        if let Some(val) = params.learning_rate { py_params.set_item("learning_rate", val)?; }
        func.call1((py_params,))?.extract()
    });

    write_summary(&step_path, params.clone(), &result)?;
    log::info!("--- Finished: Full Fine-tuning ---");
    result
}

pub fn finetune_lora(session_id: &str, model_id: &str, params: &FinetuneLoraParams) -> Result<String> {
    let step_path = PathBuf::from("runs").join(session_id).join("finetune_lora");
    fs::create_dir_all(&step_path)?;
    log::info!("--- Starting: LoRA Fine-tuning ---");
    
    let result = with_python_env(|py| {
        let func = py.import("finetuning_lora")?.getattr("fine_tune_lora")?;
        let py_params = PyDict::new(py);
        py_params.set_item("model_name", model_id)?;
        py_params.set_item("dataset_path", &params.dataset_path)?;
        py_params.set_item("output_lora_dir", step_path.to_str())?;
        if let Some(val) = params.num_epochs { py_params.set_item("num_epochs", val)?; }
        if let Some(val) = params.batch_size { py_params.set_item("batch_size", val)?; }
        if let Some(val) = params.learning_rate { py_params.set_item("learning_rate", val)?; }
        if let Some(val) = params.lora_rank { py_params.set_item("lora_rank", val)?; }
        if let Some(val) = params.lora_alpha { py_params.set_item("lora_alpha", val)?; }
        if let Some(val) = params.lora_dropout { py_params.set_item("lora_dropout", val)?; }
        func.call1((py_params,))?.extract()
    });

    write_summary(&step_path, params.clone(), &result)?;
    log::info!("--- Finished: LoRA Fine-tuning ---");
    result
}

pub fn quantize(session_id: &str, model_path: &str, params: &QuantizeParams) -> Result<QuantizationResult> {
    let step_path = PathBuf::from("runs").join(session_id).join("quantize");
    fs::create_dir_all(&step_path)?;
    log::info!("--- Starting: Quantization ---");
    
    let result = with_python_env(|py| {
        let func = py.import("quant")?.getattr("quantize_model")?;
        let py_params = PyDict::new(py);
        py_params.set_item("model_name", model_path)?;
        py_params.set_item("precision", &params.precision)?;
        py_params.set_item("quant_output_dir", step_path.to_str())?;
        
        let result_obj = func.call1((py_params,))?;
        let result_dict: &Bound<PyDict> = result_obj.downcast()?;
        
        let saved_dir: String = result_dict.get_item("saved_dir")?.unwrap().extract()?;
        Ok(QuantizationResult { saved_dir })
    });

    let summary_result = result.as_ref().map(|r| r.saved_dir.clone()).map_err(|e| anyhow!(e.to_string()));
    write_summary(&step_path, params.clone(), &summary_result)?;
    log::info!("--- Finished: Quantization ---");
    result
}

pub fn convert_model_to_gguf(session_id: &str, project_root: &Path, model_path: &str, params: &ConvertToGgufParams) -> Result<String> {
    let step_path = PathBuf::from("runs").join(session_id).join("model_to_gguf");
    fs::create_dir_all(&step_path)?;
    log::info!("--- Starting: Model to GGUF Conversion ---");
    
    let output_file = step_path.join("model.gguf");

    let result = with_python_env(|py| {
        let func = py.import("model_to_gguf")?.getattr("model_to_gguf")?;
        let py_params = PyDict::new(py);
        py_params.set_item("project_root", project_root.to_str())?;
        py_params.set_item("model_name", model_path)?;
        py_params.set_item("gguf_output_path_model", output_file.to_str())?;
        py_params.set_item("gguf_precision", &params.gguf_precision)?;
        func.call1((py_params,))?.extract()
    });

    write_summary(&step_path, params.clone(), &result)?;
    log::info!("--- Finished: Model to GGUF Conversion ---");
    result
}

pub fn convert_lora_to_gguf(session_id: &str, project_root: &Path, base_model_id: &str, adapter_path: &str, params: &ConvertToGgufParams) -> Result<String> {
    let step_path = PathBuf::from("runs").join(session_id).join("lora_to_gguf");
    fs::create_dir_all(&step_path)?;
    log::info!("--- Starting: LoRA to GGUF Conversion ---");

    let output_file = step_path.join("lora_adapter.gguf");

    let result = with_python_env(|py| {
        let func = py.import("lora_to_gguf")?.getattr("lora_to_gguf")?;
        let py_params = PyDict::new(py);
        py_params.set_item("project_root", project_root.to_str())?;
        py_params.set_item("model_name", base_model_id)?;
        py_params.set_item("adapter_path", adapter_path)?;
        py_params.set_item("gguf_output_path_lora", output_file.to_str())?;
        py_params.set_item("gguf_precision", &params.gguf_precision)?;
        func.call1((py_params,))?.extract()
    });

    write_summary(&step_path, params.clone(), &result)?;
    log::info!("--- Finished: LoRA to GGUF Conversion ---");
    result
}

pub fn run_inference(session_id: &str, model_path: &str, prompt: &str) -> Result<InferenceResult> {
    let step_path = PathBuf::from("runs").join(session_id).join("inference");
    fs::create_dir_all(&step_path)?;
    log::info!("--- Starting: Inference ---");
    
    let result = with_python_env(|py| {
        let func = py.import("inference")?.getattr("run_inference")?;
        let py_params = PyDict::new(py);
        py_params.set_item("model_name", model_path)?;
        py_params.set_item("prompt", prompt)?;
        
        let result_obj = func.call1((py_params,))?;
        let result_dict: &Bound<PyDict> = result_obj.downcast()?;
        
        let content: String = result_dict.get_item("content")?.unwrap().extract()?;
        Ok(InferenceResult { content })
    });
    
    // Summary is not strictly needed for simple inference, but we log the completion.
    let summary_result = result.as_ref().map(|r| r.content.clone()).map_err(|e| anyhow!(e.to_string()));
    write_summary(&step_path, prompt.to_string(), &summary_result)?;
    log::info!("--- Finished: Inference ---");
    result
}
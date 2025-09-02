// src/lib.rs

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::Bound;
use serde::Serialize;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use anyhow::{Result, anyhow};

// --- Public Parameter & Result Structs ---

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
    pub checkpoint_lora: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct QuantizeParams {
    pub precision: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ConvertToGgufParams {
    pub gguf_precision: String,
}

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

fn with_python_env<F, T>(session_id: &str, callback: F) -> Result<T>
where
    F: FnOnce(Python) -> Result<T, PyErr>,
    T: Send,
{
    let log_path = PathBuf::from("runs").join(session_id).join("pipeline.log");

    Python::with_gil(|py| {
        // --- The Python Logging Bridge ---
        let logging = py.import("logging")?;
        let formatter = logging.call_method1("Formatter", ("%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",))?;
        let file_handler = logging.call_method1("FileHandler", (log_path.to_str().unwrap(),))?;
        file_handler.call_method1("setLevel", (logging.getattr("DEBUG")?,))?;
        file_handler.call_method1("setFormatter", (formatter,))?;
        let root_logger = logging.call_method0("getLogger")?;
        root_logger.call_method1("setLevel", (logging.getattr("DEBUG")?,))?;
        root_logger.setattr("handlers", PyList::empty(py))?; // Clear existing handlers
        root_logger.call_method1("addHandler", (file_handler,))?;

        // --- The sys.path logic ---
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
    
    let result = with_python_env(session_id, |py| {
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
    
    let result = with_python_env(session_id, |py| {
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
        if let Some(val) = Some(&params.checkpoint_lora) { py_params.set_item("checkpoint_lora", val)?; }   
        func.call1((py_params,))?.extract()
    });

    write_summary(&step_path, params.clone(), &result)?;
    log::info!("--- Finished: LoRA Fine-tuning ---");
    result
}

pub fn convert_model_to_gguf(session_id: &str, project_root: &Path, model_path: &str, params: &ConvertToGgufParams) -> Result<String> {
    let step_path = PathBuf::from("runs").join(session_id).join("model_to_gguf");
    fs::create_dir_all(&step_path)?;
    log::info!("--- Starting: Model to GGUF Conversion ---");
    
    let output_file = step_path.join("model.gguf");

    let result = with_python_env(session_id, |py| {
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

    let result = with_python_env(session_id, |py| {
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
    
    let result = with_python_env(session_id, |py| {
        let func = py.import("inference")?.getattr("run_inference")?;
        let py_params = PyDict::new(py);
        py_params.set_item("model_name", model_path)?;
        py_params.set_item("prompt", prompt)?;
        
        let result_obj = func.call1((py_params,))?;
        let result_dict: &Bound<PyDict> = result_obj.downcast()?;
        
        let content: String = result_dict.get_item("content")?.unwrap().extract()?;
        Ok(InferenceResult { content })
    });
    
    let summary_result = result.as_ref().map(|r| r.content.clone()).map_err(|e| anyhow!(e.to_string()));
    write_summary(&step_path, prompt.to_string(), &summary_result)?;
    log::info!("--- Finished: Inference ---");
    result
}

pub fn get_status(session_id: &str, last_n: Option<usize>) -> Result<String> {
    let log_path = PathBuf::from("runs").join(session_id).join("pipeline.log");
    let content = fs::read_to_string(&log_path)?;
    if let Some(n) = last_n {
        let lines: Vec<&str> = content.lines().collect();
        let start = lines.len().saturating_sub(n);
        Ok(lines[start..].join("\n"))
    } else {
        Ok(content)
    }
}
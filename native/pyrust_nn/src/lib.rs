// src/lib.rs

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::Bound; // Corrected import for Bound
use serde::Serialize;
use simplelog::{Config, LevelFilter, WriteLogger, CombinedLogger, TermLogger, TerminalMode};
use std::env;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use anyhow::Result;

// --- Public Parameter Structs for the API ---

#[derive(Default, Debug, Serialize)]
pub struct FinetuneParams {
    pub dataset_path: String,
    pub num_epochs: Option<u32>,
}

#[derive(Default, Debug, Serialize)]
pub struct QuantizeParams {
    pub precision: String,
}

#[derive(Default, Debug, Serialize)]
pub struct GgufParams {
    pub gguf_precision: String,
}

// --- Internal Helper for Session Management ---

#[derive(Serialize)]
struct RunSummary<T: Serialize> {
    step_name: String,
    parameters: T,
    status: String,
    output_path: Option<String>,
    error_message: Option<String>,
}

/// Sets up logging for a specific step within a session.
fn setup_step_logging(step_path: &Path) -> Result<()> {
    fs::create_dir_all(step_path)?;
    let log_path = step_path.join("run.log");
    CombinedLogger::init(vec![
        TermLogger::new(LevelFilter::Info, Config::default(), TerminalMode::Mixed, simplelog::ColorChoice::Auto),
        WriteLogger::new(LevelFilter::Info, Config::default(), File::create(&log_path)?),
    ]).unwrap(); // In a real app, handle this Result better. For this tool, panic is ok.
    Ok(())
}

/// Writes a final JSON summary for a step.
fn write_summary<T: Serialize>(
    step_path: &Path,
    step_name: &str,
    parameters: T,
    result: &Result<PathBuf>,
) -> Result<()> {
    let (status, output_path, error_message) = match result {
        Ok(path) => ("Success".to_string(), Some(path.to_string_lossy().to_string()), None),
        Err(e) => ("Failure".to_string(), None, Some(e.to_string())),
    };
    let summary = RunSummary {
        step_name: step_name.to_string(),
        parameters,
        status,
        output_path,
        error_message,
    };
    let summary_path = step_path.join("summary.json");
    fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;
    Ok(())
}

// --- Private Module for Python Runner Implementation ---
mod python_runners {
    use super::*;

    /// Sets up the Python environment by adding the current directory to sys.path.
    fn setup_python_env(py: Python) -> PyResult<()> {
        let sys = py.import("sys")?;

        // --- THIS IS THE FIX for E0716 ---
        // 1. Store the result of `getattr` in a variable to extend its lifetime.
        let path_attr = sys.getattr("path")?;
        // 2. Now `.downcast()` can safely borrow from `path_attr`.
        let path: &Bound<PyList> = path_attr.downcast()?;
        
        let current_dir = env::current_dir()?.to_str().unwrap().to_string();
        if !path.contains(&current_dir)? {
             path.insert(0, &current_dir)?;
        }
        Ok(())
    }

    /// Generic function to call a Python script with parameters.
    fn call_python_script(
        module_name: &str, 
        func_name: &str, 
        py_params: &PyDict
    ) -> PyResult<String> {
        Python::with_gil(|py| {
            setup_python_env(py)?;
            let module = py.import(module_name)?;
            let func = module.getattr(func_name)?;
            // --- THIS IS THE FIX for E0283 ---
            // We explicitly tell extract what type we expect.
            func.call1((py_params,))?.extract::<String>()
        })
    }
    
    // All the public-facing functions now delegate to this internal module.
    
    pub fn finetune_full(model_id: &str, params: &FinetuneParams, output_path: &Path) -> Result<String> {
        let py_params = Python::with_gil(|py| {
            let p = PyDict::new(py);
            p.set_item("model_name", model_id).unwrap();
            p.set_item("dataset_path", &params.dataset_path).unwrap();
            p.set_item("output_dir", output_path.to_str()).unwrap();
            if let Some(epochs) = params.num_epochs { p.set_item("num_epochs", epochs).unwrap(); }
            p.to_object(py)
        });
        
        Python::with_gil(|py| {
            call_python_script("finetune_full", "fine_tune_full", py_params.downcast(py)?)
        }).map_err(anyhow::Error::from)
    }

    pub fn finetune_lora(model_id: &str, params: &FinetuneParams, output_path: &Path) -> Result<String> {
        let py_params = Python::with_gil(|py| {
            let p = PyDict::new(py);
            p.set_item("model_name", model_id).unwrap();
            p.set_item("dataset_path", &params.dataset_path).unwrap();
            p.set_item("output_lora_dir", output_path.to_str()).unwrap();
            if let Some(epochs) = params.num_epochs { p.set_item("num_epochs", epochs).unwrap(); }
            p.to_object(py)
        });

        Python::with_gil(|py| {
            call_python_script("finetuning_lora", "fine_tune_lora", py_params.downcast(py)?)
        }).map_err(anyhow::Error::from)
    }
}

// --- Main Public API Functions ---

pub fn finetune_full(
    model_id: &str,
    params: &FinetuneParams,
    output_path: &Path,
) -> Result<PathBuf> {
    setup_step_logging(output_path)?;
    log::info!("--- Starting: Full Fine-tuning ---");

    let result = python_runners::finetune_full(model_id, params, output_path);
    let final_path = PathBuf::from(result.as_deref().unwrap_or_default());

    write_summary(output_path, "finetune_full", params, &result.map(|_| final_path.clone()))?;
    log::info!("--- Finished: Full Fine-tuning ---");
    Ok(final_path)
}

pub fn finetune_lora(
    model_id: &str,
    params: &FinetuneParams,
    output_path: &Path,
) -> Result<PathBuf> {
    setup_step_logging(output_path)?;
    log::info!("--- Starting: LoRA Fine-tuning ---");

    let result = python_runners::finetune_lora(model_id, params, output_path);
    let final_path = PathBuf::from(result.as_deref().unwrap_or_default());

    write_summary(output_path, "finetune_lora", params, &result.map(|_| final_path.clone()))?;
    log::info!("--- Finished: LoRA Fine-tuning ---");
    Ok(final_path)
}
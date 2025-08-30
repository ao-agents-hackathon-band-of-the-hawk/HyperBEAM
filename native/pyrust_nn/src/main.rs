use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule, PyList};
use pyo3::conversion::IntoPyObjectExt;
use std::fs::File;
use std::io::Read;
use serde_json::Value;
use pyo3::PyObject;

fn value_to_py(py: Python, value: &Value) -> PyResult<PyObject> {
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(b) => (*b).into_py_any(py),
        Value::Number(n) if n.is_i64() => n.as_i64().unwrap().into_py_any(py),
        Value::Number(n) if n.is_u64() => n.as_u64().unwrap().into_py_any(py),
        Value::Number(n) if n.is_f64() => n.as_f64().unwrap().into_py_any(py),
        Value::String(s) => s.as_str().into_py_any(py),
        _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Unsupported JSON type")),
    }
}

fn main() -> PyResult<()> {
    // Parse params.json
    let mut file = File::open("params.json").expect("Failed to open params.json");
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("Failed to read params.json");
    let params: Value = serde_json::from_str(&contents).expect("Invalid JSON");

    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        // Add current directory to sys.path to fix ModuleNotFoundError
        let sys = py.import("sys")?;
        let path: Bound<'_, PyList> = sys.getattr("path")?.downcast::<PyList>()?.clone();
        let current_dir = std::env::current_dir()?.to_string_lossy().to_string();
        path.insert(0, current_dir)?;

        // Create PyDict from params, converting types properly
        let py_params = PyDict::new(py);
        if let Value::Object(obj) = &params {
            for (k, v) in obj {
                py_params.set_item(k, value_to_py(py, v)?)?;
            }
        }

        // Example: Call fine_tune_full
        let finetune_mod = PyModule::import(py, "finetune_full")?;
        let fine_tune_func = finetune_mod.getattr("fine_tune_full")?;
        //let result = fine_tune_func.call1((py_params.clone(),))?;
        //println!("Fine-tune full result: {:?}", result); 

        // Similarly for other functions...
        let lora_mod = PyModule::import(py, "finetuning_lora")?;
        let lora_func = lora_mod.getattr("fine_tune_lora")?;
        //lora_func.call1((py_params.clone(),))?;

        // Inference
        let inf_mod = PyModule::import(py, "inference")?;
        let inf_func = inf_mod.getattr("run_inference")?;
        //inf_func.call1((py_params.clone(),))?;

        // Quantize
        let quant_mod = PyModule::import(py, "quant")?;
        let quant_func = quant_mod.getattr("quantize_model")?;
        //quant_func.call1((py_params.clone(),))?;

        // LoRA to GGUF
        let lora_gguf_mod = PyModule::import(py, "lora_to_gguf")?;
        let lora_gguf_func = lora_gguf_mod.getattr("lora_to_gguf")?;
        lora_gguf_func.call1((py_params.clone(),))?;

        // Model to GGUF
        let model_gguf_mod = PyModule::import(py, "model_to_gguf")?;
        let model_gguf_func = model_gguf_mod.getattr("model_to_gguf")?;
        model_gguf_func.call1((py_params.clone(),))?;

        Ok(())
    })
}
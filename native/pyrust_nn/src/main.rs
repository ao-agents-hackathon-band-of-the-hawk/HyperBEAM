use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule, PyList};
use std::fs::File;
use std::io::Read;
use serde_json::Value;
use pyo3::PyObject;
use simplelog::{WriteLogger, LevelFilter, Config};
use log::{info, error};

/// Convert `serde_json::Value` to owned `PyObject`.
///
/// This implementation uses Python's builtins constructors (`bool`, `int`,
/// `float`, `str`) via `builtins` to create owned Python objects reliably
/// across PyO3 versions. For arrays/objects we still build `PyList`/`PyDict`.
fn value_to_py(py: Python, value: &Value) -> PyResult<PyObject> {
    // Import builtins once per call (cheap) to construct primitives in a
    // version-agnostic way.
    let builtins = py.import("builtins")?;

    match value {
        Value::Null => Ok(py.None().into()),

        Value::Bool(b) => Ok(builtins.getattr("bool")?.call1(( *b, ))?.into()),

        Value::Number(n) if n.is_i64() => {
            let i = n.as_i64().unwrap();
            Ok(builtins.getattr("int")?.call1(( i, ))?.into())
        }

        Value::Number(n) if n.is_u64() => {
            let u = n.as_u64().unwrap();
            // int() accepts large integers in Python
            Ok(builtins.getattr("int")?.call1(( u, ))?.into())
        }

        Value::Number(n) if n.is_f64() => {
            let f = n.as_f64().unwrap();
            Ok(builtins.getattr("float")?.call1(( f, ))?.into())
        }

        Value::String(s) => Ok(builtins.getattr("str")?.call1(( s.as_str(), ))?.into()),

        // arrays -> PyList
        Value::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                list.append(value_to_py(py, item)?)?;
            }
            Ok(list.into())
        }

        // objects -> PyDict
        Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k, value_to_py(py, v)?)?;
            }
            Ok(dict.into())
        }

        _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Unsupported JSON type")),
    }
}

fn main() -> PyResult<()> {
    // Initialize the logger to write to "activity.log"
    let _ = WriteLogger::init(
        LevelFilter::Info,
        Config::default(),
        File::create("activity.log").expect("Failed to create log file")
    );

    info!("Logger initialized. Starting application.");

    // Parse params.json
    info!("Reading parameters from params.json.");
    let mut file = File::open("params.json").expect("Failed to open params.json");
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("Failed to read params.json");
    let params: Value = serde_json::from_str(&contents).expect("Invalid JSON");
    info!("Parameters loaded successfully.");

    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        // --- THIS FIXES THE LIFETIME ERROR (E0716) ---
        let sys = py.import("sys")?;
        // Create a longer-lived binding for the 'path' attribute
        let path_attr = sys.getattr("path")?;
        // Now downcast from the binding, not the temporary value
        let path = path_attr.downcast::<PyList>()?;
        
        let current_dir = std::env::current_dir()?.to_string_lossy().to_string();
        path.insert(0, &current_dir)?;
        info!("Python environment prepared and current directory added to sys.path.");

        // Create PyDict from params
        let py_params = PyDict::new(py);
        if let Value::Object(obj) = &params {
            for (k, v) in obj {
                py_params.set_item(k, value_to_py(py, v)?)?;
            }
        }
        info!("Parameters converted for Python.");

        // Call fine_tune_full
        info!("Attempting to call 'fine_tune_full'.");
        match PyModule::import(py, "finetune_full")?.getattr("fine_tune_full")?.call1((py_params.clone(),)) {
            Ok(result) => info!("'fine_tune_full' executed successfully. Result: {:?}", result),
            Err(e) => error!("'fine_tune_full' failed: {:?}", e),
        }

        // Call fine_tune_lora
        info!("Attempting to call 'fine_tune_lora'.");
        match PyModule::import(py, "finetuning_lora")?.getattr("fine_tune_lora")?.call1((py_params.clone(),)) {
            Ok(_) => info!("'fine_tune_lora' executed successfully."),
            Err(e) => error!("'fine_tune_lora' failed: {:?}", e),
        }
        
        // Call inference
        info!("Attempting to call 'run_inference'.");
        match PyModule::import(py, "inference")?.getattr("run_inference")?.call1((py_params.clone(),)) {
            Ok(_) => info!("'run_inference' executed successfully."),
            Err(e) => error!("'run_inference' failed: {:?}", e),
        }
        
        // Call quantize
        info!("Attempting to call 'quantize_model'.");
        match PyModule::import(py, "quant")?.getattr("quantize_model")?.call1((py_params.clone(),)) {
            Ok(_) => info!("'quantize_model' executed successfully."),
            Err(e) => error!("'quantize_model' failed: {:?}", e),
        }

        // Call LoRA to GGUF
        info!("Attempting to call 'lora_to_gguf'.");
        match PyModule::import(py, "lora_to_gguf")?.getattr("lora_to_gguf")?.call1((py_params.clone(),)) {
            Ok(_) => info!("'lora_to_gguf' executed successfully."),
            Err(e) => error!("'lora_to_gguf' failed: {:?}", e),
        }

        // Call Model to GGUF
        info!("Attempting to call 'model_to_gguf'.");
        match PyModule::import(py, "model_to_gguf")?.getattr("model_to_gguf")?.call1((py_params.clone(),)) {
            Ok(_) => info!("'model_to_gguf' executed successfully."),
            Err(e) => error!("'model_to_gguf' failed: {:?}", e),
        }

        info!("All Python scripts have been called. Application finished.");
        Ok(())
    })
}

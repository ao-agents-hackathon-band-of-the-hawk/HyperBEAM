use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule, PyList};
use std::fs::File;
use std::io::Read;
use serde_json::Value;
use pyo3::PyObject;

fn value_to_py(py: Python, value: &Value) -> PyResult<PyObject> {
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
            Ok(builtins.getattr("int")?.call1(( u, ))?.into())
        }

        Value::Number(n) if n.is_f64() => {
            let f = n.as_f64().unwrap();
            Ok(builtins.getattr("float")?.call1(( f, ))?.into())
        }

        Value::String(s) => Ok(builtins.getattr("str")?.call1(( s.as_str(), ))?.into()),

        Value::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                list.append(value_to_py(py, item)?)?;
            }
            Ok(list.into())
        }

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
    let mut file = File::open("params.json").expect("Failed to open params.json");
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("Failed to read params.json");
    let params: Value = serde_json::from_str(&contents).expect("Invalid JSON");

    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let sys = py.import("sys")?;
        let path_attr = sys.getattr("path")?;
        let path = path_attr.downcast::<PyList>()?;
        
        let current_dir = std::env::current_dir()?.to_string_lossy().to_string();
        path.insert(0, &current_dir)?;

        let py_params = PyDict::new(py);
        if let Value::Object(obj) = &params {
            for (k, v) in obj {
                py_params.set_item(k, value_to_py(py, v)?)?;
            }
        }

        PyModule::import(py, "finetune_full")?.getattr("fine_tune_full")?.call1((py_params.clone(),))?;

        PyModule::import(py, "finetuning_lora")?.getattr("fine_tune_lora")?.call1((py_params.clone(),))?;
        
        PyModule::import(py, "inference")?.getattr("run_inference")?.call1((py_params.clone(),))?;
        
        PyModule::import(py, "quant")?.getattr("quantize_model")?.call1((py_params.clone(),))?;

        PyModule::import(py, "lora_to_gguf")?.getattr("lora_to_gguf")?.call1((py_params.clone(),))?;

        PyModule::import(py, "model_to_gguf")?.getattr("model_to_gguf")?.call1((py_params.clone(),))?;

        Ok(())
    })
}
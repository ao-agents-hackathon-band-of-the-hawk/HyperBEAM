#[cfg(test)]
mod tests {
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyModule, PyList};
    use std::collections::HashMap;

    fn setup_params() -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("model_name".to_string(), "Qwen/Qwen3-0.6B".to_string());
        params.insert("dataset_path".to_string(), "/content/data.txt".to_string());
        params.insert("prompt".to_string(), "Test prompt".to_string());
        params.insert("precision".to_string(), "8-bit".to_string());
        params
    }

    

    #[test]
    fn test_fine_tune_lora() -> PyResult<()> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            // Add current directory to sys.path
            let sys = py.import("sys")?;
            let path: Bound<'_, PyList> = sys.getattr("path")?.downcast::<PyList>()?.clone();
            let current_dir = std::env::current_dir()?.to_string_lossy().to_string();
            path.insert(0, current_dir)?;

            let py_params = PyDict::new(py);
            let params = setup_params();
            for (k, v) in params {
                py_params.set_item(k, v)?;
            }
            let mod_ = PyModule::import(py, "finetuning_lora")?;
            let func = mod_.getattr("fine_tune_lora")?;
            func.call1((py_params,))?;
            Ok(())
        })
    }

    #[test]
    fn test_inference() -> PyResult<()> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            // Add current directory to sys.path
            let sys = py.import("sys")?;
            let path: Bound<'_, PyList> = sys.getattr("path")?.downcast::<PyList>()?.clone();
            let current_dir = std::env::current_dir()?.to_string_lossy().to_string();
            path.insert(0, current_dir)?;

            let py_params = PyDict::new(py);
            let params = setup_params();
            for (k, v) in params {
                py_params.set_item(k, v)?;
            }
            let mod_ = PyModule::import(py, "inference")?;
            let func = mod_.getattr("run_inference")?;
            func.call1((py_params,))?;
            Ok(())
        })
    }

    #[test]
    fn test_quantize() -> PyResult<()> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            // Add current directory to sys.path
            let sys = py.import("sys")?;
            let path: Bound<'_, PyList> = sys.getattr("path")?.downcast::<PyList>()?.clone();
            let current_dir = std::env::current_dir()?.to_string_lossy().to_string();
            path.insert(0, current_dir)?;

            let py_params = PyDict::new(py);
            let params = setup_params();
            for (k, v) in params {
                py_params.set_item(k, v)?;
            }
            let mod_ = PyModule::import(py, "quant")?;
            let func = mod_.getattr("quantize_model")?;
            func.call1((py_params,))?;
            Ok(())
        })
    }

    #[test]
    fn test_lora_to_gguf() -> PyResult<()> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            // Add current directory to sys.path
            let sys = py.import("sys")?;
            let path: Bound<'_, PyList> = sys.getattr("path")?.downcast::<PyList>()?.clone();
            let current_dir = std::env::current_dir()?.to_string_lossy().to_string();
            path.insert(0, current_dir)?;

            let py_params = PyDict::new(py);
            let params = setup_params();
            for (k, v) in params {
                py_params.set_item(k, v)?;
            }
            let mod_ = PyModule::import(py, "lora_to_gguf")?;
            let func = mod_.getattr("lora_to_gguf")?;
            func.call1((py_params,))?;
            Ok(())
        })
    }

    #[test]
    fn test_model_to_gguf() -> PyResult<()> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            // Add current directory to sys.path
            let sys = py.import("sys")?;
            let path: Bound<'_, PyList> = sys.getattr("path")?.downcast::<PyList>()?.clone();
            let current_dir = std::env::current_dir()?.to_string_lossy().to_string();
            path.insert(0, current_dir)?;

            let py_params = PyDict::new(py);
            let params = setup_params();
            for (k, v) in params {
                py_params.set_item(k, v)?;
            }
            let mod_ = PyModule::import(py, "model_to_gguf")?;
            let func = mod_.getattr("model_to_gguf")?;
            func.call1((py_params,))?;
            Ok(())
        })
    }
}
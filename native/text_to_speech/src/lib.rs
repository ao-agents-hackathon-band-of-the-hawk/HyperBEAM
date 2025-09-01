use lazy_static::lazy_static;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rustler::{types::binary::OwnedBinary, Encoder, Env, Error as RustlerError, NifResult, Term};
use std::env;

mod atoms {
    rustler::atoms! {
        ok,
        error,
    }
}

// A struct to hold the Python objects, which can be safely stored in a static variable.
struct TTSGenerator {
    generator: Py<PyAny>,
    sample_rate: u32,
}

unsafe impl Send for TTSGenerator {}
unsafe impl Sync for TTSGenerator {}

lazy_static! {
    static ref TTS_MODEL: TTSGenerator = {
        eprintln!("Loading CSM text-to-speech model...");
        Python::with_gil(|py| {
            let sys = py.import("sys").expect("Failed to import sys");
            let path = sys.getattr("path").expect("Failed to get sys.path");
            let manifest_dir = env!("CARGO_MANIFEST_DIR");
            path.call_method1("insert", (0, manifest_dir))
                .expect("Failed to add to sys.path");

            let generator_module = py
                .import("generator")
                .expect("Failed to import generator module");

            let torch = py.import("torch").unwrap();
            let device = if torch
                .getattr("cuda")
                .unwrap()
                .call_method0("is_available")
                .unwrap()
                .is_truthy()
                .unwrap()
            {
                "cuda"
            } else {
                "cpu"
            };
            eprintln!("Using device: {}", device);

            let generator_instance: Py<PyAny> = generator_module
                .getattr("load_csm_1b")
                .unwrap()
                .call1((device,))
                .unwrap()
                .into();

            let sample_rate: u32 = generator_instance
                .getattr(py,"sample_rate")
                .unwrap()
                .extract(py)
                .unwrap();

            eprintln!("CSM model loaded successfully.");

            TTSGenerator {
                generator: generator_instance,
                sample_rate,
            }
        })
    };
}

#[rustler::nif(schedule = "DirtyCpu")]
fn generate_audio<'a>(env: Env<'a>, text: String, speaker: u32) -> NifResult<Term<'a>> {
    let result = std::panic::catch_unwind(|| {
        Python::with_gil(|py| -> PyResult<Vec<u8>> {
            let context = PyList::empty(py);
            let kwargs = PyDict::new(py);
            kwargs.set_item("context", context)?;
            kwargs.set_item("max_audio_length_ms", 10000)?;

            let audio_tensor =
                TTS_MODEL
                    .generator
                    .call_method(py, "generate", (text, speaker), Some(&kwargs))?;

            let torchaudio = py.import("torchaudio")?;
            let io = py.import("io")?;
            let buffer = io.call_method0("BytesIO")?;

            let save_kwargs = PyDict::new(py);
            save_kwargs.set_item("format", "wav")?;

            let unsqueezed_tensor = audio_tensor.call_method1(py, "unsqueeze", (0,))?;
            let cpu_tensor = unsqueezed_tensor.call_method0(py, "cpu")?;
            
            // THE FIX: Pass `buffer` by reference (`&buffer`) so it is not moved.
            torchaudio.call_method(
                "save",
                (
                    &buffer, // Pass as a reference to avoid moving the value
                    cpu_tensor,
                    TTS_MODEL.sample_rate,
                ),
                Some(&save_kwargs),
            )?;

            buffer.call_method1("seek", (0,))?;
            let audio_bytes: Vec<u8> = buffer.call_method0("read")?.extract()?;

            Ok(audio_bytes)
        })
    });

    match result {
        Ok(Ok(audio_bytes)) => {
            let mut binary = OwnedBinary::new(audio_bytes.len()).unwrap();
            binary.as_mut_slice().copy_from_slice(&audio_bytes);
            Ok((atoms::ok(), binary.release(env)).encode(env))
        }
        Ok(Err(e)) => {
            let err_msg = format!("Python error: {}", e);
            eprintln!("{}", err_msg);
            Ok((atoms::error(), err_msg).encode(env))
        }
        Err(panic) => {
            let panic_msg = if let Some(s) = panic.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = panic.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Unknown panic".to_string()
            };
            eprintln!("Panic during audio generation: {}", panic_msg);
            Err(RustlerError::Term(Box::new(panic_msg)))
        }
    }
}

fn load(_env: Env, _info: Term) -> bool {
    lazy_static::initialize(&TTS_MODEL);
    true
}

// FINAL FIX: Remove the deprecated explicit function list to clear the warning.
rustler::init!("dev_text_to_speech_nif", load = load);
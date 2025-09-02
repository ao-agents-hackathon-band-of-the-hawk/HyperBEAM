use lazy_static::lazy_static;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rustler::{Encoder, Env, Error as RustlerError, NifResult, Term, types::binary::OwnedBinary};
use serde_json;
use std::env;
use std::fs;
use std::path::Path;
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
    segment_class: Py<PyAny>,
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
                .getattr(py, "sample_rate")
                .unwrap()
                .extract(py)
                .unwrap();

            let segment_class = generator_module.getattr("Segment").unwrap().into();

            eprintln!("CSM model loaded successfully.");

            TTSGenerator {
                generator: generator_instance,
                sample_rate,
                segment_class,
            }
        })
    };
}
#[rustler::nif(schedule = "DirtyCpu")]
fn generate_audio<'a>(
    env: Env<'a>,
    text: String,
    speaker: u32,
    session_id: String,
) -> NifResult<Term<'a>> {
    let result = std::panic::catch_unwind(|| {
        Python::with_gil(|py| -> PyResult<Vec<u8>> {
            let context = PyList::empty(py);
            if !session_id.is_empty() {
                let sessions_dir = Path::new("sessions").join(&session_id);
                if sessions_dir.exists() {
                    let user_audios_path = sessions_dir.join("user-audios");
                    let response_audios_path = sessions_dir.join("response-audios");

                    let load_segments = |py: Python, dir: &Path, speaker_id: u32| -> PyResult<()> {
                        let transcript_path = dir.join("string-list.json");
                        if transcript_path.exists() {
                            let transcripts_str = fs::read_to_string(transcript_path)?;
                            let transcripts: Vec<String> =
                                serde_json::from_str(&transcripts_str).unwrap_or_default();

                            for (i, transcript) in transcripts.iter().enumerate() {
                                let audio_path = dir.join(format!("{}.wav", i));
                                if audio_path.exists() {
                                    let torchaudio = py.import("torchaudio")?;
                                    let audio_tensor_tuple = torchaudio
                                        .call_method1("load", (audio_path.to_str().unwrap(),))?;
                                    let audio_tensor = audio_tensor_tuple.get_item(0)?;

                                    // FIX: Call methods on Bound types without passing `py`.
                                    let final_audio_tensor = if audio_tensor
                                        .call_method0("dim")?
                                        .extract::<usize>()?
                                        > 1
                                    {
                                        audio_tensor.call_method1("mean", (0,))?
                                    } else {
                                        audio_tensor
                                    };

                                    let seg_kwargs = PyDict::new(py);
                                    seg_kwargs.set_item("text", transcript)?;
                                    seg_kwargs.set_item("speaker", speaker_id)?;
                                    seg_kwargs.set_item("audio", final_audio_tensor)?;

                                    // FIX: Pass the PyDict by reference.
                                    let segment =
                                        TTS_MODEL.segment_class.call(py, (), Some(&seg_kwargs))?;
                                    context.append(segment)?;
                                }
                            }
                        }
                        Ok(())
                    };

                    load_segments(py, &user_audios_path, 0)?;
                    load_segments(py, &response_audios_path, 1)?;
                }
            }

            let kwargs = PyDict::new(py);
            kwargs.set_item("context", context)?;
            kwargs.set_item("max_audio_length_ms", 90_000)?;

            // FIX: Pass kwargs by reference
            let audio_tensor =
                TTS_MODEL
                    .generator
                    .call_method(py, "generate", (text, speaker), Some(&kwargs))?;

            let torchaudio = py.import("torchaudio")?;
            let io = py.import("io")?;
            // FIX: Call method on Bound type without `py`.
            let buffer = io.call_method0("BytesIO")?;
            let save_kwargs = PyDict::new(py);
            save_kwargs.set_item("format", "wav")?;

            // FIX: When calling a method on a `Py<PyAny>`, you *do* need `py`.
            let unsqueezed_tensor = audio_tensor.call_method1(py, "unsqueeze", (0,))?;
            let cpu_tensor = unsqueezed_tensor.call_method0(py, "cpu")?;

            // FIX: The `torchaudio.save` method is on a module, so it's a Bound. Pass kwargs by reference.
            torchaudio.call_method(
                "save",
                (&buffer, cpu_tensor, TTS_MODEL.sample_rate),
                Some(&save_kwargs),
            )?;

            // FIX: Call methods on the `buffer` (a Bound type) without `py`.
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
// FIX: Remove the deprecated explicit function list.
rustler::init!("dev_text_to_speech_nif", load = load);

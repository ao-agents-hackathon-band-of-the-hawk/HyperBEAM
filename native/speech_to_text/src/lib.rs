pub mod config;
pub mod pyscripts;

use config::*;
use pyo3::{prelude::*, types::PyModule};
use pyscripts::*;
use std::{error::Error, fmt::Debug, i32};

use lazy_static::lazy_static;
use rustler::{Encoder, Env, Error as RustlerError, NifResult, Term};

mod atoms {
    rustler::atoms! {
        ok,
        error,
    }
}
#[derive(Clone, Debug)]
pub struct WhisperModel {
    module: Py<PyModule>,
    model: Py<pyo3::PyAny>,
    pub config: WhisperConfig,
}

#[derive(Clone, Debug)]
pub struct Segment {
    pub id: i32,
    pub seek: i32,
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub temperature: f32,
    pub avg_logprob: f32,
    pub compression_ratio: f32,
    pub no_speech_prob: f32,
}

#[derive(Clone)]
pub struct Segments(String, pub Vec<Segment>);

impl ToString for Segments {
    fn to_string(&self) -> String {
        self.0.clone()
    }
}

impl Debug for Segments {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

lazy_static! {
    static ref MODEL: WhisperModel = {
        // Optional: Add logging to confirm one-time load
        eprintln!("Loading Whisper model...");
        let config = WhisperConfig::default();
        WhisperModel::new(
            "base.en".to_string(),
            "cpu".to_string(),
            "int8".to_string(),
            config,
        ).expect("Failed to load Whisper model")
    };
}

impl Default for WhisperModel {
    fn default() -> Self {
        return Self::new(
            "base.en".to_string(),
            "cpu".to_string(),
            "int8".to_string(),
            WhisperConfig::default(),
        )
        .unwrap();
    }
}

impl WhisperModel {
    pub fn new(
        model: String,
        device: String,
        compute_type: String,
        config: WhisperConfig,
    ) -> Result<Self, Box<dyn Error>> {
        let m = Python::with_gil(|py| {
            let activators =
                PyModule::from_code_bound(py, &get_script(), "whisper.py", "Whisper").unwrap();
            let args = (model, device, compute_type);
            let model = activators
                .getattr("new_model")
                .unwrap()
                .call1(args)
                .unwrap()
                .unbind();
            return Self {
                module: activators.unbind(),
                model,
                config,
            };
        });

        return Ok(m);
    }

    fn convert<T: ToString>(value: Option<T>) -> String {
        match value {
            Some(x) => x.to_string(),
            None => "None".to_string(),
        }
    }

    pub fn transcribe(&self, path: String) -> Result<Segments, Box<dyn Error>> {
        let segments = Python::with_gil(|py| {
            let vad = (
                self.config.vad.active,
                self.config.vad.threshold,
                self.config.vad.min_speech_duration,
                Self::convert(self.config.vad.max_speech_duration),
                self.config.vad.min_silence_duration,
                self.config.vad.padding_duration,
            );

            let args = (
                self.model.clone(),
                path,
                Self::convert(self.config.starting_prompt.clone()),
                Self::convert(self.config.prefix.clone()),
                Self::convert(self.config.language.clone()),
                self.config.beam_size,
                self.config.best_of,
                self.config.patience,
                self.config.length_penalty,
                Self::convert(self.config.chunk_length.clone()),
                vad,
            );

            let pysegments = self
                .module
                .getattr(py, "transcribe_audio")
                .unwrap()
                .call1(py, args)?
                .extract::<Vec<(i32, i32, f32, f32, String, f32, f32, f32, f32)>>(py)?;
            let mut segments = Vec::new();

            for segment in pysegments {
                segments.push(Segment {
                    id: segment.0,
                    seek: segment.1,
                    start: segment.2,
                    end: segment.3,
                    text: segment.4,
                    temperature: segment.5,
                    avg_logprob: segment.6,
                    compression_ratio: segment.7,
                    no_speech_prob: segment.8,
                });
            }

            return Ok::<Vec<Segment>, Box<dyn Error>>(segments);
        })?;

        let mut text = String::new();

        for segment in &segments {
            text.push_str(&segment.text);
        }

        return Ok(Segments(text, segments));
    }
}

#[rustler::nif(schedule = "DirtyCpu")]
fn transcribe_audio<'a>(env: Env<'a>, path: String) -> NifResult<Term<'a>> {
    // Use catch_unwind to prevent panics from crashing the BEAM VM
    match std::panic::catch_unwind(|| MODEL.transcribe(path)) {
        Ok(Ok(transcript)) => {
            // Return the transcribed text
            Ok((atoms::ok(), transcript.to_string()).encode(env))
        }
        Ok(Err(e)) => {
            // Log the error to terminal
            eprintln!("Transcription error: {:?}", e);
            // Return error tuple
            Ok((atoms::error(), e.to_string()).encode(env))
        }
        Err(panic) => {
            // Handle panic: log and return error
            let panic_msg = match panic.downcast::<String>() {
                Ok(msg) => *msg,
                Err(_) => "Unknown panic".to_string(),
            };
            eprintln!("Panic during transcription: {}", panic_msg);
            Err(RustlerError::Term(Box::new(panic_msg)))
        }
    }
}

fn load(_env: Env, _info: Term) -> bool {
    // Force lazy initialization of the model during NIF load
    let _ = &*MODEL;
    true
}

rustler::init!("dev_speech_to_text_nif", load = load);

#[test]
fn create_test() {
    // Create a model that uses CUDA instead of using the default
    let fw = WhisperModel::new(
        "base.en".to_string(),
        "cuda".to_string(),
        "float16".to_string(), // float16 is generally better for CUDA performance
        WhisperConfig::default(),
    )
    .unwrap();

    let trans = fw.transcribe(get_path("./man.mp3".to_string())).unwrap();
    println!("{}", trans.to_string());
    assert!(!trans.0.is_empty());
}

pub fn get_path(path: String) -> String {
    let mut new_path = env!("CARGO_MANIFEST_DIR").to_string();
    new_path.push_str(&format!("/src/{}", path));
    return new_path;
}

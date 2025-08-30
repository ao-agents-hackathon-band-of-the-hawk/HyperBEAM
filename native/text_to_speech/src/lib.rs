use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::env;

#[derive(Clone)]
pub struct Segment {
    pub speaker: u32,
    pub text: String,
    pub audio: Vec<f32>,
}

pub fn generate_speech(
    text: String,
    speaker: u32,
    context: Vec<Segment>,
    device: Option<String>,
    filename: Option<String>,
) -> PyResult<Vec<f32>> {
    Python::with_gil(|py| {
        let sys = py.import("sys")?;
        let path = sys.getattr("path")?;
        let dir = env::current_dir()?.to_str().unwrap().to_string();
        path.call_method1("insert", (0, dir))?;
        let generator_mod = py.import("generator")?;
        let load_csm_1b = generator_mod.getattr("load_csm_1b")?;
        let device_str = device.unwrap_or("cuda".to_string());
        let generator = load_csm_1b.call1((device_str,))?;
        let segment_class = generator_mod.getattr("Segment")?;
        let torch = py.import("torch")?;
        let context_py = PyList::empty(py);
        for seg in context {
            let audio_list = PyList::new(py, seg.audio.iter().cloned())?;
            let dtype = torch.getattr("float32")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("dtype", dtype)?;
            let tensor = torch
                .getattr("tensor")?
                .call((audio_list,), Some(&kwargs))?;
            let py_seg = segment_class.call1((seg.speaker as i64, seg.text, tensor))?;
            context_py.append(py_seg)?;
        }
        let audio =
            generator.call_method1("generate", (text, speaker as i64, context_py, 10_000))?;
        if let Some(file) = filename {
            let torchaudio = py.import("torchaudio")?;
            let sample_rate = generator.getattr("sample_rate")?;
            let audio_unsqueeze = audio.call_method1("unsqueeze", (0,))?;
            let cpu_audio = audio_unsqueeze.call_method0("cpu")?;
            torchaudio.call_method1("save", (file, cpu_audio, sample_rate))?;
        }
        let audio_list = audio.call_method0("tolist")?;
        let vec: Vec<f32> = audio_list.extract()?;
        Ok(vec)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_speech() {
        let result = generate_speech(
            "Hello from Sesame. I'm here to suck you dry".to_string(),
            0,
            vec![],
            Some("cpu".to_string()),
            Some("test_audio.wav".to_string()),
        );
        match &result {
            Ok(audio) => assert!(!audio.is_empty()),
            Err(e) => panic!("Error: {:?}", e),
        }
    }
}

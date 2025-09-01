use pyo3::prelude::*;
use pyo3::types::PyList;
use std::env;

#[derive(Clone)]
pub struct Segment {
    pub speaker: u32,
    pub text: String,
    pub audio_path: String,
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
        let torchaudio = py.import("torchaudio")?;
        let sample_rate: i32 = generator.getattr("sample_rate")?.extract()?;
        let context_py = PyList::empty(py);
        for seg in context {
            let audio_path = seg.audio_path.clone();
            let loaded = torchaudio.getattr("load")?.call1((audio_path,))?;
            let mut audio_tensor = loaded.get_item(0)?;
            let orig_sample_rate: i32 = loaded.get_item(1)?.extract()?;
            audio_tensor = audio_tensor.call_method1("mean", (0,))?;
            let functional = torchaudio.getattr("functional")?;
            let resample = functional.getattr("resample")?;
            audio_tensor = resample.call((audio_tensor, orig_sample_rate, sample_rate), None)?;
            let py_seg = segment_class.call((seg.speaker as i64, seg.text, audio_tensor), None)?;
            context_py.append(py_seg)?;
        }
        let audio =
            generator.call_method("generate", (text, speaker as i64, context_py, 10_000), None)?;
        if let Some(file) = filename {
            let audio_unsqueeze = audio.call_method1("unsqueeze", (0,))?;
            let cpu_audio = audio_unsqueeze.call_method0("cpu")?;
            torchaudio.call_method("save", (file, cpu_audio, sample_rate), None)?;
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
    fn test_generate_speech_no_context() {
        let result = generate_speech(
            "Hello from Sesame.".to_string(),
            0,
            vec![],
            Some("cuda".to_string()),
            Some("test_audio.wav".to_string()),
        );
        match &result {
            Ok(audio) => assert!(!audio.is_empty()),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_generate_speech_with_context() {
        let context = vec![Segment {
            speaker: 0,
            text: "In a 1997 AI class at UT Austin, a neural net playing infinite board tic-tac-toe found an unbeatable strategy. Choose moves billions of squares away, causing your opponents to run out of memory and crash.".to_string(),
            audio_path: "utterance_0.mp3".to_string(),
        }];
        let result = generate_speech(
            "Hello Am I audible, I wanted you guys to tell me if my voice changes alot, okay thanks!".to_string(),
            0,
            context,
            Some("cuda".to_string()),
            Some("test_audio_with_context.wav".to_string()),
        );
        match &result {
            Ok(audio) => assert!(!audio.is_empty()),
            Err(e) => panic!("Error: {:?}", e),
        }
    }
}

// src/lib.rs (updated with into_py and remove unused PyNone)

use pyo3::prelude::*;
use pyo3::types::PyString;
use std::env;
use std::process::Command;
use std::error::Error;

#[derive(Clone)]
pub struct Segment {
    pub speaker: u32,
    pub text: String,
    pub audio_path: String,
}

pub fn generate_speech(
    text: String,
    output: String,
    speaker: u32,
    device: Option<String>,
) -> PyResult<()> {
    Python::with_gil(|py| {
        let sys = py.import("sys")?;
        let path = sys.getattr("path")?;
        let dir = env::current_dir()?.to_str().unwrap().to_string();
        path.call_method1("insert", (0, dir))?;
        
        let shell_mod = py.import("shell")?;
        let generate_audio = shell_mod.getattr("generate_audio")?;
        
        let device_py = match device {
            Some(d) => PyString::new(py, &d).into(),
            None => py.None(),
        };
        
        generate_audio.call((text, output, speaker as i64, device_py), None)?;
        
        Ok(())
    })
}

/// Runs the shell.py script as an external process, handling stdout, stderr, and errors.
/// This function spawns a Python process to execute shell.py with the provided arguments.
/// The audio file is saved by the Python script.
pub fn run_shell(
    text: &str,
    output: &str,
    speaker: u32,
    device: &str,
) -> Result<String, Box<dyn Error>> {
    // Assume shell.py is in the current directory or provide full path if needed
    let shell_path = "shell.py";

    let cmd_output = Command::new("python3")
        .arg(shell_path)
        .arg("--text")
        .arg(text)
        .arg("--output")
        .arg(output)
        .arg("--speaker")
        .arg(speaker.to_string())
        .arg("--device")
        .arg(device)
        .output()?;

    // Route stderr to console if there's an error
    if !cmd_output.status.success() {
        let stderr = String::from_utf8_lossy(&cmd_output.stderr);
        eprintln!("Error from shell.py: {}", stderr);
        return Err(format!("shell.py failed with status: {}. STDERR: {}", cmd_output.status, stderr).into());
    }

    // Route stdout to console
    let stdout = String::from_utf8_lossy(&cmd_output.stdout);
    println!("Output from shell.py: {}", stdout);

    // Return the stdout as result for further processing if needed
    Ok(stdout.to_string())
}

#[cfg(test)]
mod tests;
// src/tests.rs (unchanged, as it matches the signature)

use super::*;
use std::error::Error;
use std::path::Path;

#[test]
fn test_generate_speech_no_context() -> Result<(), Box<dyn Error>> {
    let result = generate_speech(
        "Hello from Sesame.".to_string(),
        "test_audio.wav".to_string(),
        0,
        Some("cuda".to_string()),
    );
    match &result {
        Ok(_) => assert!(Path::new("test_audio.wav").exists()),
        Err(e) => panic!("Error: {:?}", e),
    }
    Ok(())
}

#[test]
fn test_generate_speech_with_context() -> Result<(), Box<dyn Error>> {
    // Note: Context is no longer used in the new implementation, as generate_audio doesn't support it.
    // This test is adapted to match the new signature without context.
    let result = generate_speech(
        "Hello Am I audible, I wanted you guys to tell me if my voice changes alot, okay thanks!".to_string(),
        "test_audio_with_context.wav".to_string(),
        0,
        Some("cuda".to_string()),
    );
    match &result {
        Ok(_) => assert!(Path::new("test_audio_with_context.wav").exists()),
        Err(e) => panic!("Error: {:?}", e),
    }
    Ok(())
}

#[test]
fn test_run_shell() -> Result<(), Box<dyn Error>> {
    let output_file = "shell-test.wav";
    let result = run_shell(
        "Hello from shell.py.",
        output_file,
        0,
        "cuda",
    );

    match result {
        Ok(output) => {
            assert!(output.contains("Audio saved to"));
            // Check if file was saved
            assert!(Path::new(output_file).exists());
        }
        Err(e) => panic!("Error: {:?}", e),
    }

    Ok(())
}
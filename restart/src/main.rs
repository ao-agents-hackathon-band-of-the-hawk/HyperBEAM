use std::env;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <directory> <command>", args[0]);
        std::process::exit(1);
    }
    let dir = &args[1];
    let command = &args[2];
    let dir_path = Path::new(dir);

    // Split the command string into program and arguments
    let mut command_parts = command.split_whitespace();
    let program = command_parts.next().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "Command cannot be empty")
    })?;
    let command_args: Vec<&str> = command_parts.collect();

    let mut counter = 0;
    loop {
        let log_filename = if counter == 0 {
            "rebar3.log".to_string()
        } else {
            format!("rebar3-reload-{}.log", counter)
        };
        let log_path = dir_path.join(&log_filename);
        let file = File::create(&log_path)?;

        let mut child = Command::new(program)
            .args(&command_args)
            .current_dir(dir_path)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        // Get handles to stdout and stderr
        let mut stdout = child.stdout.take().expect("Failed to capture stdout");
        let mut stderr = child.stderr.take().expect("Failed to capture stderr");

        // Create a thread to handle stdout
        let stdout_file = file.try_clone()?;
        let stdout_thread = std::thread::spawn(move || {
            let mut buffer = [0; 1024];
            loop {
                match stdout.read(&mut buffer) {
                    Ok(0) => break, // EOF
                    Ok(n) => {
                        let slice = &buffer[..n];
                        // Print to console
                        io::stdout().write_all(slice)?;
                        io::stdout().flush()?;
                        // Write to file
                        stdout_file.try_clone()?.write_all(slice)?;
                    }
                    Err(e) => {
                        eprintln!("Error reading stdout: {}", e);
                        break;
                    }
                }
            }
            Ok::<(), io::Error>(())
        });

        // Create a thread to handle stderr
        let stderr_file = file;
        let stderr_thread = std::thread::spawn(move || {
            let mut buffer = [0; 1024];
            loop {
                match stderr.read(&mut buffer) {
                    Ok(0) => break, // EOF
                    Ok(n) => {
                        let slice = &buffer[..n];
                        // Print to console
                        io::stderr().write_all(slice)?;
                        io::stderr().flush()?;
                        // Write to file
                        stderr_file.try_clone()?.write_all(slice)?;
                    }
                    Err(e) => {
                        eprintln!("Error reading stderr: {}", e);
                        break;
                    }
                }
            }
            Ok::<(), io::Error>(())
        });

        // Wait for the child process to exit
        let status = child.wait()?;

        // Wait for the output threads to finish
        stdout_thread.join().expect("Stdout thread panicked")?;
        stderr_thread.join().expect("Stderr thread panicked")?;

        if status.success() {
            break;
        } else {
            counter += 1;
        }
    }
    Ok(())
}
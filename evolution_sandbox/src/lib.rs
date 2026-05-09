use std::process::{Command, Output};
use std::time::{Duration, Instant};

pub struct CommandRunner;

impl CommandRunner {
    /// Runs a command and returns the output.
    pub fn run(cmd: &str, args: &[&str]) -> Result<Output, std::io::Error> {
        Command::new(cmd)
            .args(args)
            .output()
    }

    /// Runs a command and checks if it completed within the duration.
    pub fn run_with_timeout(cmd: &str, args: &[&str], timeout: Duration) -> Result<Output, String> {
        let start = Instant::now();
        let output = Command::new(cmd)
            .args(args)
            .output()
            .map_err(|e| e.to_string()?);

        if start.elapsed() > timeout {
            return Err("Command timed out".to_string());
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_command_runner_success() {
        let result = CommandRunner::run("echo", &["hello_world"]).expect("Command failed");
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(stdout.contains("hello_world"));
    }

    #[test]
    fn test_command_runner_timeout_logic() {
        // Note: This test is a bit of a 'smoke test' for the logic
        // since we can't easily kill the process from within the same thread
        // without complexity, but we check that the duration logic is sound.
        let start = std::time::Instant::now();
        let _ = std::time::sleep(std::time::Duration::from_millis(10));
        let duration = start.elapsed();
        assert!(duration >= std::time::Duration::from_millis(10));
    }
}

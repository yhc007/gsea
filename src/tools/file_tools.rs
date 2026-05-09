use super::Tool;
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::{json, Value};

// ─── ReadFile ───────────────────────────────────────────────────

pub struct ReadFile;

#[async_trait]
impl Tool for ReadFile {
    fn name(&self) -> &str {
        "read_file"
    }
    fn description(&self) -> &str {
        "Read the contents of a file at the given path."
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                }
            },
            "required": ["path"]
        })
    }
    async fn execute(&self, params: Value) -> Result<Value> {
        let path = params["path"]
            .as_str()
            .context("missing 'path' parameter")?;
        let content = tokio::fs::read_to_string(path)
            .await
            .with_context(|| format!("failed to read {}", path))?;
        Ok(json!({ "content": content, "path": path, "bytes": content.len() }))
    }
}

// ─── WriteFile ──────────────────────────────────────────────────

pub struct WriteFile;

#[async_trait]
impl Tool for WriteFile {
    fn name(&self) -> &str {
        "write_file"
    }
    fn description(&self) -> &str {
        "Write content to a file. Creates parent directories if needed."
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write"
                }
            },
            "required": ["path", "content"]
        })
    }
    async fn execute(&self, params: Value) -> Result<Value> {
        let path = params["path"]
            .as_str()
            .context("missing 'path' parameter")?;
        let content = params["content"]
            .as_str()
            .context("missing 'content' parameter")?;

        if let Some(parent) = std::path::Path::new(path).parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(path, content).await?;
        Ok(json!({ "path": path, "bytes": content.len() }))
    }
}

// ─── RunShell ───────────────────────────────────────────────────

pub struct RunShell;

#[async_trait]
impl Tool for RunShell {
    fn name(&self) -> &str {
        "run_shell"
    }
    fn description(&self) -> &str {
        "Execute a shell command and return its stdout/stderr."
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute"
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 30)"
                }
            },
            "required": ["command"]
        })
    }
    async fn execute(&self, params: Value) -> Result<Value> {
        let cmd = params["command"]
            .as_str()
            .context("missing 'command' parameter")?;
        let timeout = params["timeout_secs"].as_i64().unwrap_or(30);

        let output = tokio::time::timeout(
            std::time::Duration::from_secs(timeout as u64),
            tokio::process::Command::new("sh")
                .arg("-c")
                .arg(cmd)
                .output(),
        )
        .await
        .context("command timed out")??;

        Ok(json!({
            "stdout": String::from_utf8_lossy(&output.stdout).to_string(),
            "stderr": String::from_utf8_lossy(&output.stderr).to_string(),
            "exit_code": output.status.code(),
            "success": output.status.success(),
        }))
    }
}

impl GitCommit {
    /// Resolve the working directory. If "." (default), use the binary's parent.
    fn resolve_project_dir(dir: &str) -> std::path::PathBuf {
        if dir == "." {
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().and_then(|p| p.parent()).map(|p| p.to_path_buf()))
                .unwrap_or_else(|| std::path::PathBuf::from("."))
        } else {
            std::path::PathBuf::from(dir)
        }
    }
}

// ─── CargoBuild ─────────────────────────────────────────────────

pub struct CargoBuild;

#[async_trait]
impl Tool for CargoBuild {
    fn name(&self) -> &str {
        "cargo_build"
    }
    fn description(&self) -> &str {
        "Run 'cargo build' in the project directory and report compilation result."
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "project_dir": {
                    "type": "string",
                    "description": "Path to the Cargo project directory (default: current)"
                },
                "release": {
                    "type": "boolean",
                    "description": "Build in release mode"
                }
            }
        })
    }
    async fn execute(&self, params: Value) -> Result<Value> {
        let project_dir = params["project_dir"]
            .as_str()
            .unwrap_or(".")
            .to_string();
        let release = params["release"].as_bool().unwrap_or(false);

        let mut cmd = tokio::process::Command::new("cargo");
        cmd.arg("build").current_dir(&project_dir);
        if release {
            cmd.arg("--release");
        }

        let output = cmd.output().await?;
        let success = output.status.success();
        Ok(json!({
            "stdout": String::from_utf8_lossy(&output.stdout).to_string(),
            "stderr": String::from_utf8_lossy(&output.stderr).to_string(),
            "exit_code": output.status.code(),
            "success": success,
        }))
    }
}

// ─── CargoTest ──────────────────────────────────────────────────

pub struct CargoTest;

#[async_trait]
impl Tool for CargoTest {
    fn name(&self) -> &str {
        "cargo_test"
    }
    fn description(&self) -> &str {
        "Run 'cargo test' in the project directory and report results."
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "project_dir": {
                    "type": "string",
                    "description": "Path to the Cargo project directory (default: current)"
                },
                "test_name": {
                    "type": "string",
                    "description": "Optional: run a specific test"
                }
            }
        })
    }
    async fn execute(&self, params: Value) -> Result<Value> {
        let project_dir = params["project_dir"]
            .as_str()
            .unwrap_or(".")
            .to_string();

        let mut cmd = tokio::process::Command::new("cargo");
        cmd.arg("test").current_dir(&project_dir);
        if let Some(test_name) = params["test_name"].as_str() {
            cmd.arg(test_name);
        }

        let output = cmd.output().await?;
        Ok(json!({
            "stdout": String::from_utf8_lossy(&output.stdout).to_string(),
            "stderr": String::from_utf8_lossy(&output.stderr).to_string(),
            "exit_code": output.status.code(),
            "success": output.status.success(),
        }))
    }
}

// ─── GitCommit ──────────────────────────────────────────────────

pub struct GitCommit;

#[async_trait]
impl Tool for GitCommit {
    fn name(&self) -> &str {
        "git_commit"
    }
    fn description(&self) -> &str {
        "Stage all changes and create a git commit with the given message."
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Commit message"
                },
                "project_dir": {
                    "type": "string",
                    "description": "Path to the git project directory (default: .)"
                }
            },
            "required": ["message"]
        })
    }
    async fn execute(&self, params: Value) -> Result<Value> {
        let msg = params["message"]
            .as_str()
            .context("missing 'message' parameter")?;

        let project_dir = Self::resolve_project_dir(
            params["project_dir"].as_str().unwrap_or(".")
        );

        let add_output = tokio::process::Command::new("git")
            .args(["add", "-A"])
            .current_dir(&project_dir)
            .output()
            .await?;

        let commit_output = tokio::process::Command::new("git")
            .args(["commit", "-m", msg])
            .current_dir(&project_dir)
            .output()
            .await?;

        let diff_output = tokio::process::Command::new("git")
            .args(["diff", "--stat", "HEAD~1..HEAD", "--"])
            .current_dir(&project_dir)
            .output()
            .await;

        Ok(json!({
            "add_stdout": String::from_utf8_lossy(&add_output.stdout).to_string(),
            "commit_stdout": String::from_utf8_lossy(&commit_output.stdout).to_string(),
            "commit_stderr": String::from_utf8_lossy(&commit_output.stderr).to_string(),
            "commit_success": commit_output.status.success(),
            "diff_stat": diff_output.map(|o| String::from_utf8_lossy(&o.stdout).to_string()).unwrap_or_default(),
        }))
    }
}

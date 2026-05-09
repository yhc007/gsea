use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;

use super::{Tool, ToolRegistry};
use crate::memory_brain::Brain;
use anyhow::{Context, Result};

/// Dynamically loads a skill from Brain, compiles it with a wrapper, and runs it.
/// Skills are simple pure functions stored in procedural memory.
pub struct CallSkill {
    brain: Arc<std::sync::Mutex<Brain>>,
}

impl CallSkill {
    pub fn new(brain: Arc<std::sync::Mutex<Brain>>) -> Self {
        Self { brain }
    }

    /// Generate a wrapper main.rs that calls the skill function with stdin args.
    fn generate_wrapper(skill_code: &str) -> String {
        format!(
            r#"// Auto-generated wrapper for skill function
// Compile: rustc -o /tmp/skill_exec /tmp/skill_wrapper.rs

use std::io::Read;

fn main() {{
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).expect("stdin read");
    let input = input.trim();

    // Try to parse as JSON array first, else as space-separated numbers or raw string
    let result = if input.starts_with('[') {{
        // JSON array: parse as Vec<i32>
        if let Ok(nums) = serde_json::from_str::<Vec<i32>>(input) {{
            let res = execute(&nums);
            serde_json::to_string(&res).unwrap()
        }} else {{
            "error: expected JSON array of integers".to_string()
        }}
    }} else if input.contains(' ') || input.contains('\n') {{
        // Space/newline separated numbers
        let nums: Vec<i32> = input.split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        let res = execute(&nums);
        res.to_string()
    }} else {{
        // Single value: try number first, then treat as raw string
        if let Ok(n) = input.parse::<i32>() {{
            let res = execute(&[n]);
            res.to_string()
        }} else {{
            execute_str(input)
        }}
    }};

    println!("{{}}", result);
}}

// ─── Injected skill code ──────────────────────────────────────
{}
// ─── End skill code ───────────────────────────────────────────

// Generic dispatch: override these if needed in your skill code
fn execute(nums: &[i32]) -> bool {{
    // Default: call a function matching the pattern
    is_all_even(nums)
}}

fn execute_str(_input: &str) -> String {{
    "string input not supported by this skill".to_string()
}}
"#,
            skill_code
        )
    }
}

#[async_trait]
impl Tool for CallSkill {
    fn name(&self) -> &str {
        "call_skill"
    }
    fn description(&self) -> &str {
        "Execute a previously learned Rust skill function dynamically. Provide the skill name and a JSON array of arguments."
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Name of the skill to execute (e.g. 'is_all_even')"
                },
                "args": {
                    "type": "string",
                    "description": "Arguments for the skill: JSON array for structured input, or space-separated values"
                }
            },
            "required": ["skill_name", "args"]
        })
    }
    async fn execute(&self, params: Value) -> Result<Value> {
        let skill_name = params["skill_name"]
            .as_str()
            .context("missing 'skill_name' parameter")?;
        let args = params["args"]
            .as_str()
            .context("missing 'args' parameter")?;

        // 1. Look up skill code from Brain
        let code = {
            let brain = self.brain.lock().unwrap();
            brain.get_skill_code(skill_name)
        };
        let code = code.context(format!("skill '{}' not found in Brain", skill_name))?;

        // 2. Generate wrapper
        let wrapper = Self::generate_wrapper(&code);

        // 3. Write to temp files and compile
        let tmp_dir = std::env::temp_dir().join(format!("gsea_skill_{}", skill_name));
        tokio::fs::create_dir_all(&tmp_dir).await?;
        let rs_path = tmp_dir.join("main.rs");
        let bin_path = tmp_dir.join("skill_bin");
        tokio::fs::write(&rs_path, &wrapper).await?;

        // Check if rustc is available
        let rustc_check = tokio::process::Command::new("which")
            .arg("rustc")
            .output()
            .await;
        if rustc_check.map_or(true, |o| !o.status.success()) {
            return Ok(json!({
                "error": "rustc not found on PATH. Cannot compile skill dynamically."
            }));
        }

        // 4. Compile
        let compile_output = tokio::process::Command::new("rustc")
            .arg(&rs_path)
            .arg("-o")
            .arg(&bin_path)
            .output()
            .await
            .context("rustc compilation failed")?;

        if !compile_output.status.success() {
            let stderr = String::from_utf8_lossy(&compile_output.stderr);
            return Ok(json!({
                "error": format!("compilation failed:\n{}", stderr)
            }));
        }

        // 5. Run with args as stdin
        let args_owned = args.to_string();
        let bin_path_clone = bin_path.clone();
        let run_output = tokio::task::spawn_blocking(move || {
            let mut child = std::process::Command::new(&bin_path_clone)
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()?;
            use std::io::Write;
            if let Some(ref mut stdin) = child.stdin {
                write!(stdin, "{}", args_owned)?;
            }
            child.wait_with_output()
        })
        .await
        .context("spawn_blocking panicked")?
        .context("skill execution failed")?;

        let stdout = String::from_utf8_lossy(&run_output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&run_output.stderr).trim().to_string();

        // 6. Clean up temp files (async, don't block on it)
        let _ = tokio::fs::remove_dir_all(&tmp_dir).await;

        Ok(json!({
            "skill": skill_name,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": run_output.status.code(),
            "success": run_output.status.success(),
        }))
    }
}

// ─── DynamicSkillTool ──────────────────────────────────────────

/// A tool dynamically created from a stored skill.
/// Each stored skill becomes its own callable tool.
pub struct DynamicSkillTool {
    brain: Arc<std::sync::Mutex<Brain>>,
    skill_name: String,
    skill_desc: String,
}

impl DynamicSkillTool {
    pub fn new(brain: Arc<std::sync::Mutex<Brain>>, skill_name: &str, skill_desc: &str) -> Self {
        Self {
            brain,
            skill_name: skill_name.to_string(),
            skill_desc: skill_desc.to_string(),
        }
    }
}

#[async_trait]
impl Tool for DynamicSkillTool {
    fn name(&self) -> &str {
        &self.skill_name
    }
    fn description(&self) -> &str {
        &self.skill_desc
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "args": {
                    "type": "string",
                    "description": "Arguments: JSON array (e.g. [1,2,3]) or space-separated values"
                }
            },
            "required": ["args"]
        })
    }
    async fn execute(&self, params: Value) -> Result<Value> {
        // Delegate to CallSkill's logic with our fixed skill_name
        let args = params["args"]
            .as_str()
            .context("missing 'args' parameter")?;

        let code = {
            let brain = self.brain.lock().unwrap();
            brain.get_skill_code(&self.skill_name)
        };
        let code = code.context(format!("skill '{}' not found in Brain", self.skill_name))?;

        let wrapper = CallSkill::generate_wrapper(&code);
        let tmp_dir = std::env::temp_dir().join(format!("gsea_skill_{}", self.skill_name));
        tokio::fs::create_dir_all(&tmp_dir).await?;
        let rs_path = tmp_dir.join("main.rs");
        let bin_path = tmp_dir.join("skill_bin");
        tokio::fs::write(&rs_path, &wrapper).await?;

        let compile_output = tokio::process::Command::new("rustc")
            .arg(&rs_path).arg("-o").arg(&bin_path)
            .output().await.context("rustc compilation failed")?;
        if !compile_output.status.success() {
            let stderr = String::from_utf8_lossy(&compile_output.stderr);
            return Ok(json!({ "error": format!("compilation failed:\n{}", stderr) }));
        }

        let args_owned = args.to_string();
        let bin_path_clone = bin_path.clone();
        let run_output = tokio::task::spawn_blocking(move || {
            let mut child = std::process::Command::new(&bin_path_clone)
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()?;
            use std::io::Write;
            if let Some(ref mut stdin) = child.stdin {
                write!(stdin, "{}", args_owned)?;
            }
            child.wait_with_output()
        }).await.context("spawn_blocking panicked")?.context("skill execution failed")?;

        let stdout = String::from_utf8_lossy(&run_output.stdout).trim().to_string();
        let _ = tokio::fs::remove_dir_all(&tmp_dir).await;

        Ok(json!({
            "skill": self.skill_name,
            "stdout": stdout,
            "success": run_output.status.success(),
        }))
    }
}

/// Register all stored skills as dynamic tools in the ToolRegistry.
pub fn register_skills(
    registry: &mut ToolRegistry,
    brain: &Arc<std::sync::Mutex<Brain>>,
) {
    let skills = brain.lock().unwrap().list_skills();
    for (name, desc) in skills {
        let tool = Box::new(DynamicSkillTool::new(brain.clone(), &name, &desc));
        registry.register(tool);
        tracing::info!("Registered dynamic tool: {}", name);
    }
}

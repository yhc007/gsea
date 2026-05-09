use anyhow::Result;
use std::sync::Arc;

use crate::agent::Agent;
use crate::memory_brain::Brain;

/// The evolution engine periodically triggers self-review and self-improvement.
pub struct EvolutionEngine {
    pub brain: Arc<std::sync::Mutex<Brain>>,
    reflection_interval: u64,
    episode_count: u64,
}

impl EvolutionEngine {
    pub fn new(brain: Arc<std::sync::Mutex<Brain>>, reflection_interval: u64) -> Self {
        Self {
            brain,
            reflection_interval,
            episode_count: 0,
        }
    }

    /// Called after each user episode. At the configured interval, triggers
    /// a full evolution cycle: review → propose → extract → build → commit.
    pub async fn after_episode(&mut self, agent: &mut Agent) -> Result<Option<String>> {
        self.episode_count += 1;

        if self.episode_count % self.reflection_interval == 0 {
            tracing::info!(
                "Evolution cycle triggered after {} episodes",
                self.episode_count
            );

            let result = self.run_evolution_cycle(agent).await?;
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    // ─── Self-Evolution Cycle ──────────────────────────────────

    /// 1. Gemma proposes code → 2. Extract → 3. Save & Build → 4. Store & Commit
    async fn run_evolution_cycle(&self, agent: &mut Agent) -> Result<String> {
        let summary = self.brain.lock().unwrap().generate_context_summary();
        self.brain.lock().unwrap().record_reflection("evolution_cycle", "Starting self-evolution cycle")?;

        // Step 1: Ask Gemma for a utility function
        let code_prompt = format!(
            r#"Suggest ONE small, standalone Rust utility function that could be useful.

Rules:
- The function must be pure (no external crate dependencies)
- Max 20 lines of code
- Include a doc comment explaining what it does
- Format exactly like this:

```rust
/// Description of the utility
pub fn my_utility(param: Type) -> ReturnType {{
    // implementation
}}
```

Current system state:
{}"#,
            summary
        );

        let response = match tokio::time::timeout(
            std::time::Duration::from_secs(45),
            agent.process_message(&code_prompt),
        )
        .await
        {
            Ok(Ok(text)) => text,
            Ok(Err(e)) => return Ok(format!("Evolution error: {}", e)),
            Err(_) => return Ok("Evolution timed out".to_string()),
        };

        // Step 2: Extract Rust code block
        let code = match self.extract_rust_code(&response) {
            Some(c) => c,
            None => {
                let msg = "No Rust code block found in response".to_string();
                self.brain.lock().unwrap().record_reflection("evolution_result", &msg)?;
                return Ok(msg);
            }
        };

        // Step 3: Extract function name
        let fn_name = self.extract_fn_name(&code).unwrap_or("unnamed_utility");

        // Step 4: Save to file
        let file_path = format!("skills/{}.rs", fn_name);
        match tokio::fs::write(&file_path, &code).await {
            Ok(_) => tracing::info!("Saved skill to {}", file_path),
            Err(e) => {
                let msg = format!("Failed to save {}: {}", file_path, e);
                self.brain.lock().unwrap().record_reflection("evolution_result", &msg)?;
                return Ok(msg);
            }
        }

        // Step 5: Store the skill in Brain + filesystem + git
        let description = self.extract_description(&code).unwrap_or(fn_name);
        {
            let brain = self.brain.lock().unwrap();
            brain.store_skill(fn_name, description, &code)?;
        }
        tracing::info!("Skill '{}' stored in Brain and saved to {}", fn_name, file_path);

        // Git commit
        self.git_commit(fn_name).await;

        Ok(format!(
            "✅ Skill '{}' created and committed.\n```rust\n{}\n```",
            fn_name, code
        ))
    }

    // ─── Helpers ────────────────────────────────────────────────

    fn extract_rust_code(&self, text: &str) -> Option<String> {
        if let Some(start) = text.find("```rust") {
            let after = &text[start + 7..];
            if let Some(end) = after.find("```") {
                return Some(after[..end].trim().to_string());
            }
        }
        if let Some(start) = text.find("```") {
            let after = &text[start + 3..];
            if let Some(end) = after.find("```") {
                let code = after[..end].trim();
                if code.contains("fn ") {
                    return Some(code.to_string());
                }
            }
        }
        None
    }

    fn extract_fn_name<'a>(&self, code: &'a str) -> Option<&'a str> {
        for line in code.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("pub fn ") {
                if let Some(name_end) = rest.find('(') {
                    return Some(&rest[..name_end]);
                }
            } else if let Some(rest) = trimmed.strip_prefix("fn ") {
                if let Some(name_end) = rest.find('(') {
                    return Some(&rest[..name_end]);
                }
            }
        }
        None
    }

    fn extract_description<'a>(&self, code: &'a str) -> Option<&'a str> {
        for line in code.lines() {
            let trimmed = line.trim();
            if let Some(desc) = trimmed.strip_prefix("///") {
                return Some(desc.trim());
            }
        }
        None
    }

    async fn build_project(&self) -> String {
        let output = tokio::process::Command::new("cargo")
            .args(["build", "--message-format=json"])
            .current_dir(std::env::current_dir().unwrap_or_default())
            .output()
            .await;

        match output {
            Ok(out) => {
                format!(
                    "success: {} | stdout: {} | stderr: {}",
                    out.status.success(),
                    String::from_utf8_lossy(&out.stdout).lines().last().unwrap_or(""),
                    String::from_utf8_lossy(&out.stderr).lines().last().unwrap_or(""),
                )
            }
            Err(e) => format!("cargo error: {}", e),
        }
    }

    async fn git_commit(&self, skill_name: &str) {
        let msg = format!("gsea: auto-learned skill '{}'", skill_name);
        let _ = tokio::process::Command::new("git")
            .args(["add", "-A"])
            .current_dir(std::env::current_dir().unwrap_or_default())
            .output()
            .await;
        let _ = tokio::process::Command::new("git")
            .args(["commit", "-m", &msg])
            .current_dir(std::env::current_dir().unwrap_or_default())
            .output()
            .await;
    }
}

use anyhow::Result;
use std::sync::Arc;

use crate::agent::Agent;
use crate::memory_brain::Brain;
use crate::tools::{ToolRegistry, skill_tools};

/// The evolution engine periodically triggers self-review and self-improvement.
pub struct EvolutionEngine {
    pub brain: Arc<std::sync::Mutex<Brain>>,
    pub registry: Arc<std::sync::Mutex<ToolRegistry>>,
    reflection_interval: u64,
    episode_count: u64,
}

impl EvolutionEngine {
    pub fn new(
        brain: Arc<std::sync::Mutex<Brain>>,
        registry: Arc<std::sync::Mutex<ToolRegistry>>,
        reflection_interval: u64,
    ) -> Self {
        Self {
            brain,
            registry,
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
            std::time::Duration::from_secs(30),
            agent.process_message_fast(&code_prompt),
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

        // Step 4: Save to skills/ directory (always)
        let file_path = format!("skills/{}.rs", fn_name);
        let _ = tokio::fs::write(&file_path, &code).await;

        // Step 4b: Promote to src/tools/skills/ as a real module + verify build
        let promoted_path = format!("src/tools/skills/{}.rs", fn_name);
        let mod_path = "src/tools/skills/mod.rs";
        let promoted = tokio::fs::write(&promoted_path, &code).await.is_ok()
            && self.add_skill_module(mod_path, fn_name).await
            && self.build_project().await;

        if promoted {
            tracing::info!("Skill '{}' promoted and compiled into project", fn_name);
        } else {
            // Rollback on build failure
            let _ = tokio::fs::remove_file(&promoted_path).await;
            let _ = self.remove_skill_module(mod_path, fn_name).await;
            tracing::warn!("Build failed for '{}', rolled back", fn_name);
        }

        // Step 5: Store the skill in Brain + filesystem + git
        let description = self.extract_description(&code).unwrap_or(fn_name);
        {
            let brain = self.brain.lock().unwrap();
            brain.store_skill(fn_name, description, &code)?;
        }

        // Register as dynamic tool in the shared registry
        {
            let mut reg = self.registry.lock().unwrap();
            let dyn_tool = Box::new(skill_tools::DynamicSkillTool::new(
                self.brain.clone(), fn_name, description,
            ));
            reg.register_by_name(fn_name, dyn_tool);
            tracing::info!("Registered '{}' as a dynamic tool", fn_name);
        }

        tracing::info!("Skill '{}' stored in Brain and saved to {}", fn_name, file_path);

        // Git commit
        self.git_commit(fn_name).await;

        Ok(format!(
            "✅ Skill '{}' created, committed, and registered as a tool.\n```rust\n{}\n```",
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

    /// Add a `pub mod {name};` line to the skills module.
    async fn add_skill_module(&self, mod_path: &str, name: &str) -> bool {
        let line = format!("pub mod {};\n", name);
        match tokio::fs::read_to_string(mod_path).await {
            Ok(mut content) => {
                if !content.contains(&line) {
                    content.push_str(&line);
                    tokio::fs::write(mod_path, &content).await.is_ok()
                } else {
                    true // already present
                }
            }
            Err(_) => false,
        }
    }

    /// Remove a `pub mod {name};` line from the skills module (for rollback).
    async fn remove_skill_module(&self, mod_path: &str, name: &str) -> bool {
        let search = format!("pub mod {};\n", name);
        match tokio::fs::read_to_string(mod_path).await {
            Ok(content) => {
                let new_content = content.replace(&search, "");
                tokio::fs::write(mod_path, &new_content).await.is_ok()
            }
            Err(_) => false,
        }
    }

    /// Run cargo build to verify the skill compiles.
    async fn build_project(&self) -> bool {
        let output = tokio::process::Command::new("cargo")
            .args(["build", "--message-format=short"])
            .current_dir(std::env::current_dir().unwrap_or_default())
            .output()
            .await;
        matches!(output, Ok(o) if o.status.success())
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

#[cfg(test)]
mod tests {
    use super::EvolutionEngine;
    use crate::memory_brain::Brain;
    use std::sync::{Arc, Mutex};

    // Helper: create a minimal EvolutionEngine for testing
    fn test_engine() -> EvolutionEngine {
        let brain = Arc::new(Mutex::new(Brain::new("/tmp/gsea_test_memory").unwrap()));
        let registry = Arc::new(Mutex::new(crate::tools::ToolRegistry::new()));
        EvolutionEngine::new(brain, registry, 5)
    }

    #[test]
    fn test_extract_rust_code_with_markers() {
        let engine = test_engine();
        let text = "Here's the code:\n```rust\npub fn foo() -> bool { true }\n```\nEnd.";
        let code = engine.extract_rust_code(text);
        assert_eq!(code, Some("pub fn foo() -> bool { true }".to_string()));
    }

    #[test]
    fn test_extract_rust_code_without_lang() {
        let engine = test_engine();
        let text = "Code:\n```\npub fn bar() {}\n```\nDone.";
        let code = engine.extract_rust_code(text);
        assert_eq!(code, Some("pub fn bar() {}".to_string()));
    }

    #[test]
    fn test_extract_rust_code_none() {
        let engine = test_engine();
        assert!(engine.extract_rust_code("no code here").is_none());
    }

    #[test]
    fn test_extract_fn_name_pub() {
        let engine = test_engine();
        let code = "pub fn is_all_even(nums: &[i32]) -> bool { true }";
        assert_eq!(engine.extract_fn_name(code), Some("is_all_even"));
    }

    #[test]
    fn test_extract_fn_name_private() {
        let engine = test_engine();
        let code = "fn helper(x: i32) -> i32 { x }";
        assert_eq!(engine.extract_fn_name(code), Some("helper"));
    }

    #[test]
    fn test_extract_fn_name_none() {
        let engine = test_engine();
        assert!(engine.extract_fn_name("struct Foo;").is_none());
    }

    #[test]
    fn test_extract_description() {
        let engine = test_engine();
        let code = "/// Check if all numbers are even\npub fn is_all_even() -> bool { true }";
        assert_eq!(engine.extract_description(code), Some("Check if all numbers are even"));
    }
}

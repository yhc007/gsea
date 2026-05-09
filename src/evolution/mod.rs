use anyhow::Result;
use std::sync::Arc;

use crate::agent::Agent;
use crate::memory_brain::MemoryBrain;

/// The evolution engine periodically triggers reflection and
/// applies self-improvements to the agent's capabilities.
pub struct EvolutionEngine {
    pub brain: Arc<MemoryBrain>,
    /// How many episodes between automatic reflection cycles.
    reflection_interval: u64,
    episode_count: u64,
}

impl EvolutionEngine {
    pub fn new(brain: Arc<MemoryBrain>, reflection_interval: u64) -> Self {
        Self {
            brain,
            reflection_interval,
            episode_count: 0,
        }
    }

    /// Called after each episode. Triggers reflection at the configured interval.
    pub async fn after_episode(&mut self, agent: &mut Agent) -> Result<Option<String>> {
        self.episode_count += 1;

        if self.episode_count % self.reflection_interval == 0 {
            tracing::info!(
                "Triggering reflection cycle after {} episodes",
                self.episode_count
            );

            let reflection = agent.run_reflection_cycle().await?;

            // Process any action plans from the reflection
            if let Some(action) = self.extract_action_plan(&reflection) {
                tracing::info!("Evolution action plan: {}", action);
                // In Phase 4, this will trigger code generation + testing
            }

            Ok(Some(reflection))
        } else {
            Ok(None)
        }
    }

    /// Simple action plan extraction — will be enhanced in Phase 4.
    fn extract_action_plan(&self, reflection: &str) -> Option<String> {
        // Look for lines after "Action Plan:" or similar markers
        for line in reflection.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("Action:") || trimmed.starts_with("TODO:") {
                return Some(trimmed.to_string());
            }
        }
        None
    }
}

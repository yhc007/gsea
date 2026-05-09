use anyhow::Result;
use serde_json::Value;
use std::collections::HashMap;

pub mod file_tools;
pub mod memory_tools;

/// A tool that the agent can invoke. Each tool has a name, description,
/// and an `execute` method that takes JSON parameters and returns JSON.
#[async_trait::async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> Value;

    async fn execute(&self, params: Value) -> Result<Value>;
}

/// Registry that holds all available tools.
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        let name = tool.name().to_string();
        self.tools.insert(name, tool);
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|b| b.as_ref())
    }

    /// Get tool specifications formatted for Ollama's tool-use API.
    pub fn tool_specs(&self) -> Vec<crate::llm::ToolSpec> {
        self.tools
            .values()
            .map(|tool| crate::llm::ToolSpec {
                tool_type: "function".to_string(),
                function: crate::llm::ToolFunction {
                    name: tool.name().to_string(),
                    description: tool.description().to_string(),
                    parameters: tool.parameters(),
                },
            })
            .collect()
    }

    pub fn list_tools(&self) -> Vec<&dyn Tool> {
        self.tools.values().map(|b| b.as_ref()).collect()
    }

    pub fn tool_description_text(&self) -> String {
        let mut lines = Vec::new();
        lines.push("Available tools:".to_string());
        for tool in self.tools.values() {
            lines.push(format!(
                "  - {}: {}",
                tool.name(),
                tool.description()
            ));
        }
        lines.join("\n")
    }
}

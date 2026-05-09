use async_trait::async_trait;
use serde_json::{json, Value};

use super::Tool;
use crate::memory_brain::Brain;
use anyhow::{Context, Result};
use std::sync::Arc;

// ─── MemoryStore ────────────────────────────────────────────────

pub struct MemoryStore {
    brain: Arc<std::sync::Mutex<Brain>>,
}

impl MemoryStore {
    pub fn new(brain: Arc<std::sync::Mutex<Brain>>) -> Self {
        Self { brain }
    }
}

#[async_trait]
impl Tool for MemoryStore {
    fn name(&self) -> &str {
        "memory_store"
    }
    fn description(&self) -> &str {
        "Store a piece of information in long-term memory for future recall."
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content to remember"
                },
                "content_type": {
                    "type": "string",
                    "description": "Type: 'note', 'code_pattern', 'learning', 'decision'"
                },
                "importance": {
                    "type": "number",
                    "description": "Importance score 0.0-1.0 (default: 0.5)"
                }
            },
            "required": ["content"]
        })
    }
    async fn execute(&self, params: Value) -> Result<Value> {
        let content = params["content"]
            .as_str()
            .context("missing 'content' parameter")?;

        let mut brain = self.brain.lock().unwrap();
        let mtype = crate::memory_brain::MemoryType::Semantic;
        brain.process(content, Some(mtype))?;

        Ok(json!({ "stored": true }))
    }
}

// ─── MemoryRecall ───────────────────────────────────────────────

pub struct MemoryRecall {
    brain: Arc<std::sync::Mutex<Brain>>,
}

impl MemoryRecall {
    pub fn new(brain: Arc<std::sync::Mutex<Brain>>) -> Self {
        Self { brain }
    }
}

#[async_trait]
impl Tool for MemoryRecall {
    fn name(&self) -> &str {
        "memory_recall"
    }
    fn description(&self) -> &str {
        "Search for relevant memories by keyword. Returns matching entries."
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to find relevant memories"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results (default: 10)"
                }
            },
            "required": ["query"]
        })
    }
    async fn execute(&self, params: Value) -> Result<Value> {
        let query = params["query"].as_str().context("missing 'query' parameter")?;
        let limit = params["limit"].as_i64().unwrap_or(10) as usize;

        let brain = self.brain.lock().unwrap();
        let results = brain.recall(query, limit);
        let items: Vec<Value> = results
            .into_iter()
            .map(|item| {
                json!({
                    "id": item.id.to_string(),
                    "content": item.content,
                    "type": item.memory_type.to_string(),
                    "strength": item.strength,
                })
            })
            .collect();

        Ok(json!({ "results": items, "count": items.len() }))
    }
}

// ─── MemoryStats ────────────────────────────────────────────────

pub struct MemoryStats {
    brain: Arc<std::sync::Mutex<Brain>>,
}

impl MemoryStats {
    pub fn new(brain: Arc<std::sync::Mutex<Brain>>) -> Self {
        Self { brain }
    }
}

#[async_trait]
impl Tool for MemoryStats {
    fn name(&self) -> &str {
        "memory_stats"
    }
    fn description(&self) -> &str {
        "Get statistics about the Brain — memory count by type."
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {}
        })
    }
    async fn execute(&self, _params: Value) -> Result<Value> {
        let brain = self.brain.lock().unwrap();
        let stats = brain.stats();
        let summary = brain.generate_context_summary();

        Ok(json!({
            "memory_stats": stats,
            "summary": summary,
        }))
    }
}

// ─── Reflect ────────────────────────────────────────────────────

pub struct Reflect {
    brain: Arc<std::sync::Mutex<Brain>>,
}

impl Reflect {
    pub fn new(brain: Arc<std::sync::Mutex<Brain>>) -> Self {
        Self { brain }
    }
}

#[async_trait]
impl Tool for Reflect {
    fn name(&self) -> &str {
        "reflect"
    }
    fn description(&self) -> &str {
        "Record a reflection about the system's performance, a problem, or an improvement idea."
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "observation": {
                    "type": "string",
                    "description": "What did you observe or notice?"
                },
                "hypothesis": {
                    "type": "string",
                    "description": "What do you think is the cause or opportunity?"
                },
                "action_plan": {
                    "type": "string",
                    "description": "What action should be taken?"
                }
            },
            "required": ["observation"]
        })
    }
    async fn execute(&self, params: Value) -> Result<Value> {
        let observation = params["observation"]
            .as_str()
            .context("missing 'observation' parameter")?;

        let mut brain = self.brain.lock().unwrap();
        brain.record_reflection("agent_insight", observation)?;

        Ok(json!({
            "recorded": true,
            "message": "Reflection recorded. It will be reviewed in the next evolution cycle."
        }))
    }
}

use async_trait::async_trait;
use serde_json::{json, Value};

use super::Tool;
use crate::llm::embedding::EmbeddingEngine;
use crate::memory_brain::MemoryBrain;
use anyhow::{Context, Result};
use std::sync::Arc;

// ─── MemoryStore ────────────────────────────────────────────────

pub struct MemoryStore {
    brain: Arc<MemoryBrain>,
    embedder: Arc<dyn EmbeddingEngine>,
}

impl MemoryStore {
    pub fn new(brain: Arc<MemoryBrain>, embedder: Arc<dyn EmbeddingEngine>) -> Self {
        Self { brain, embedder }
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
        let content_type = params["content_type"].as_str().unwrap_or("note");
        let importance = params["importance"].as_f64().unwrap_or(0.5);

        // Try to store with embedding for semantic search
        let id = match self.embedder.embed(content).await {
            Ok(emb) => self.brain.store_with_embedding(content, content_type, &emb, importance)?,
            Err(_) => self.brain.store_memory(content, content_type, None, None, importance)?,
        };

        Ok(json!({ "memory_id": id, "stored": true }))
    }
}

// ─── MemoryRecall ───────────────────────────────────────────────

pub struct MemoryRecall {
    brain: Arc<MemoryBrain>,
}

impl MemoryRecall {
    pub fn new(brain: Arc<MemoryBrain>) -> Self {
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

        let results = self.brain.search_memories(query, limit)?;
        let items: Vec<Value> = results
            .into_iter()
            .map(|(id, content, content_type, importance)| {
                json!({
                    "id": id,
                    "content": content,
                    "type": content_type,
                    "importance": importance,
                })
            })
            .collect();

        Ok(json!({ "results": items, "count": items.len() }))
    }
}

// ─── MemoryStats ────────────────────────────────────────────────

pub struct MemoryStats {
    brain: Arc<MemoryBrain>,
}

impl MemoryStats {
    pub fn new(brain: Arc<MemoryBrain>) -> Self {
        Self { brain }
    }
}

#[async_trait]
impl Tool for MemoryStats {
    fn name(&self) -> &str {
        "memory_stats"
    }
    fn description(&self) -> &str {
        "Get statistics about the MemoryBrain — memory count, skill count, reflection status."
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {}
        })
    }
    async fn execute(&self, _params: Value) -> Result<Value> {
        let stats = self.brain.get_memory_stats()?;
        let skills = self.brain.list_skills()?;
        let pending = self.brain.pending_reflections()?;
        let summary = self.brain.generate_context_summary()?;

        Ok(json!({
            "memory_stats": stats,
            "skill_count": skills.len(),
            "pending_reflections": pending.len(),
            "summary": summary,
        }))
    }
}

// ─── Reflect ────────────────────────────────────────────────────

pub struct Reflect {
    brain: Arc<MemoryBrain>,
}

impl Reflect {
    pub fn new(brain: Arc<MemoryBrain>) -> Self {
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
        let hypothesis = params["hypothesis"].as_str();
        let action_plan = params["action_plan"].as_str();

        let id = self
            .brain
            .record_reflection("agent_insight", observation, hypothesis, action_plan)?;

        Ok(json!({
            "reflection_id": id,
            "recorded": true,
            "message": "Reflection recorded. It will be reviewed in the next evolution cycle."
        }))
    }
}

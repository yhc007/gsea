//! Procedural memory — patterns and habits (code patterns, skills)
use anyhow::Result;
use crate::memory_brain::{storage::Storage, types::*};

pub struct ProceduralMemory {
    storage: Storage,
}

impl ProceduralMemory {
    pub fn new(db_path: &str) -> Result<Self> {
        Ok(Self {
            storage: Storage::new(db_path, "procedural")?,
        })
    }

    pub fn store(&self, item: MemoryItem) -> Result<()> {
        self.storage.insert(item)?;
        Ok(())
    }

    pub fn search_by_embedding(&self, query_emb: &[f32], limit: usize, min_score: f64) -> Vec<(MemoryItem, f64)> {
        self.storage.search_by_embedding(query_emb, limit, MemoryType::Procedural, min_score)
    }

    pub fn search(&self, query: &str, limit: usize) -> Vec<MemoryItem> {
        self.storage.search(query, limit, MemoryType::Procedural)
    }

    pub fn decay(&self, factor: f32) -> Result<()> {
        self.storage.apply_decay(factor, MemoryType::Procedural)?;
        Ok(())
    }

    pub fn count(&self) -> i64 {
        self.storage.count(MemoryType::Procedural)
    }

    // ─── Skills (typed procedural memories) ─────────────────────

    /// Store a learned skill with its Rust source code.
    pub fn store_skill(&self, name: &str, description: &str, code: &str) -> Result<()> {
        let content = format!("SKILL:{}|{}\n```rust\n{}\n```", name, description, code);
        let item = MemoryItem::new(&content, MemoryType::Procedural)
            .with_tags(vec!["skill".to_string(), name.to_string()]);
        self.storage.insert(item)?;
        Ok(())
    }

    /// List all stored skills.
    pub fn list_skills(&self) -> Vec<(String, String)> {
        let results = self.storage.search("SKILL:", 50, MemoryType::Procedural);
        let mut skills = Vec::new();
        for item in results {
            if let Some(rest) = item.content.strip_prefix("SKILL:") {
                if let Some(pipe_pos) = rest.find('|') {
                    let name = &rest[..pipe_pos];
                    let desc_rest = &rest[pipe_pos + 1..];
                    if let Some(_code_start) = desc_rest.find("```rust\n") {
                        let desc = &desc_rest[.._code_start];
                        skills.push((name.to_string(), desc.to_string()));
                    } else {
                        skills.push((name.to_string(), desc_rest.to_string()));
                    }
                }
            }
        }
        skills
    }

    /// Get the Rust code for a named skill.
    pub fn get_skill_code(&self, name: &str) -> Option<String> {
        let query = format!("SKILL:{}", name);
        let results = self.storage.search(&query, 1, MemoryType::Procedural);
        if let Some(item) = results.into_iter().next() {
            if let Some(code_start) = item.content.find("```rust\n") {
                let after = &item.content[code_start + 8..];
                if let Some(code_end) = after.find("```") {
                    return Some(after[..code_end].to_string());
                }
            }
        }
        None
    }
}

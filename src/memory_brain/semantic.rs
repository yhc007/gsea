//! Semantic memory — facts and concepts
use crate::memory_brain::{storage::Storage, types::*};

pub struct SemanticMemory {
    storage: Storage,
}

impl SemanticMemory {
    pub fn new(db_path: &str) -> Result<Self, anyhow::Error> {
        Ok(Self {
            storage: Storage::new(db_path, "semantic")?,
        })
    }

    pub fn delete(&self, id: &str) -> Result<(), anyhow::Error> {
        self.storage.delete(id)?;
        Ok(())
    }

    pub fn store(&self, item: MemoryItem) -> Result<(), anyhow::Error> {
        self.storage.insert(item)?;
        Ok(())
    }

    pub fn search_by_embedding(&self, query_emb: &[f32], limit: usize, min_score: f64) -> Vec<(MemoryItem, f64)> {
        self.storage.search_by_embedding(query_emb, limit, MemoryType::Semantic, min_score)
    }

    pub fn search(&self, query: &str, limit: usize) -> Vec<MemoryItem> {
        self.storage.search(query, limit, MemoryType::Semantic)
    }

    pub fn decay(&self, factor: f32) -> Result<(), anyhow::Error> {
        self.storage.apply_decay(factor, MemoryType::Semantic)?;
        Ok(())
    }

    pub fn count(&self) -> i64 {
        self.storage.count(MemoryType::Semantic)
    }
}

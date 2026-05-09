//! MemoryBrain — Human brain-inspired memory system
//! Port of yhc007/memory-brain API design with SQLite backend.
//! 
//! ## Memory Types
//! - Working: short-term, 7±2 items (Miller's Law)
//! - Episodic: "when did what" - events
//! - Semantic: facts and concepts  
//! - Procedural: patterns and skills
//!
//! ## Brain-like Features
//! - Forgetting curve (Ebbinghaus): unused memories fade
//! - Memory consolidation: important working → long-term
//! - Associative recall: related memories linked
//! - Emotional weighting: strong events remembered longer

pub mod working;
pub mod episodic;
pub mod semantic;
pub mod procedural;
pub mod consolidate;
pub mod forgetting;
pub mod types;
pub mod storage;

pub use types::*;
pub use working::WorkingMemory;
pub use episodic::EpisodicMemory;
pub use semantic::SemanticMemory;
pub use procedural::ProceduralMemory;
pub use consolidate::Consolidator;
pub use forgetting::ForgettingCurve;

/// The unified Brain — coordinates all memory systems.
/// Port of yhc007/memory-brain's Brain struct.
pub struct Brain {
    pub working: WorkingMemory,
    pub episodic: EpisodicMemory,
    pub semantic: SemanticMemory,
    pub procedural: ProceduralMemory,
    consolidator: Consolidator,
    forgetting: ForgettingCurve,
}

impl Brain {
    pub fn new(storage_path: &str) -> Result<Self, anyhow::Error> {
        Ok(Self {
            working: WorkingMemory::new(7), // Miller's magic number
            episodic: EpisodicMemory::new(storage_path)?,
            semantic: SemanticMemory::new(storage_path)?,
            procedural: ProceduralMemory::new(storage_path)?,
            consolidator: Consolidator::new(),
            forgetting: ForgettingCurve::new(),
        })
    }

    /// Process new input: embed, classify, store in working, consolidate to long-term.
    pub fn process(&mut self, input: &str, memory_type: Option<MemoryType>) -> Result<UuidValue, anyhow::Error> {
        let mtype = memory_type.unwrap_or(MemoryType::Semantic);
        let mut item = MemoryItem::new(input, mtype);
        
        // Classify by content
        item.memory_type = self.consolidator.classify(&item);
        
        // Add to working memory
        let id = item.id;
        self.working.push(item.clone());
        
        // Consolidate to long-term
        self.consolidate_memory(item)?;
        
        Ok(id)
    }

    /// Search across all memory types for the given query.
    pub fn recall(&self, query: &str, limit: usize) -> Vec<MemoryItem> {
        let mut results = Vec::new();
        
        // Check working memory first (fastest)
        results.extend(self.working.search(query));
        
        // Then long-term memories
        results.extend(self.episodic.search(query, limit));
        results.extend(self.semantic.search(query, limit));
        results.extend(self.procedural.search(query, limit));
        
        results.truncate(limit);
        results
    }

    /// Store to memory and consolidate (internal helper).
    fn consolidate_memory(&mut self, item: MemoryItem) -> Result<(), anyhow::Error> {
        match item.memory_type {
            MemoryType::Episodic => self.episodic.store(item)?,
            MemoryType::Semantic => self.semantic.store(item)?,
            MemoryType::Procedural => self.procedural.store(item)?,
            MemoryType::Working => { /* stays in working memory only */ }
        }
        Ok(())
    }

    /// Apply forgetting curve decay to all long-term memories.
    pub fn apply_forgetting(&mut self) -> Result<(), anyhow::Error> {
        let decay_factor = self.forgetting.decay_factor();
        self.episodic.decay(decay_factor)?;
        self.semantic.decay(decay_factor)?;
        self.procedural.decay(decay_factor)?;
        Ok(())
    }

    /// Record a reflection observation.
    pub fn record_reflection(&mut self, trigger: &str, observation: &str) -> Result<(), anyhow::Error> {
        let content = format!("[{}] {}", trigger, observation);
        self.process(&content, Some(MemoryType::Semantic))?;
        Ok(())
    }

    /// Generate a text summary of system state.
    pub fn generate_context_summary(&self) -> String {
        let s = self.stats();
        format!(
            "System Summary — working: {}, episodic: {}, semantic: {}, procedural: {}",
            s["working"].as_i64().unwrap_or(0),
            s["episodic"].as_i64().unwrap_or(0),
            s["semantic"].as_i64().unwrap_or(0),
            s["procedural"].as_i64().unwrap_or(0),
        )
    }

    // ─── Skills ─────────────────────────────────────────────────

    /// Store a Rust skill (code pattern, utility) in procedural memory.
    pub fn store_skill(&self, name: &str, description: &str, code: &str) -> Result<(), anyhow::Error> {
        self.procedural.store_skill(name, description, code)?;
        Ok(())
    }

    /// List all stored skills as (name, description) pairs.
    pub fn list_skills(&self) -> Vec<(String, String)> {
        self.procedural.list_skills()
    }

    /// Get Rust source code for a named skill.
    pub fn get_skill_code(&self, name: &str) -> Option<String> {
        self.procedural.get_skill_code(name)
    }

    /// Learn: store a new piece of information as a semantic memory.
    /// Returns the memory ID for later reference.
    pub fn learn(&self, content: &str) -> Result<UuidValue, anyhow::Error> {
        let item = MemoryItem::new(content, MemoryType::Semantic);
        let id = item.id;
        self.semantic.store(item)?;
        Ok(id)
    }

    /// Forget: delete a memory by its UUID from all storage types.
    pub fn forget(&self, id: &str) -> Result<(), anyhow::Error> {
        let _ = self.episodic.delete(id);
        let _ = self.semantic.delete(id);
        let _ = self.procedural.delete(id);
        Ok(())
    }

    /// Search across all memory types by embedding similarity.
    pub fn recall_by_similarity(&self, query_emb: &[f32], limit: usize, min_score: f64) -> Vec<(MemoryItem, f64)> {
        let mut results = Vec::new();
        results.extend(self.episodic.search_by_embedding(query_emb, limit, min_score));
        results.extend(self.semantic.search_by_embedding(query_emb, limit, min_score));
        results.extend(self.procedural.search_by_embedding(query_emb, limit, min_score));
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    /// Get statistics about all memory systems.
    pub fn stats(&self) -> serde_json::Value {
        serde_json::json!({
            "working": self.working.len(),
            "episodic": self.episodic.count(),
            "semantic": self.semantic.count(),
            "procedural": self.procedural.count(),
        })
    }
}

//! Memory consolidation — short-term → long-term transfer logic
use crate::memory_brain::types::*;

pub struct Consolidator;

impl Consolidator {
    pub fn new() -> Self {
        Self
    }

    /// Classify the memory type based on content heuristics.
    pub fn classify(&self, item: &MemoryItem) -> MemoryType {
        let content = item.content.to_lowercase();

        // Code patterns → procedural
        if content.contains("fn ") || content.contains("impl ") || content.contains("struct ")
            || content.contains("let ") || content.contains("pub ") || content.contains(".rs")
        {
            return MemoryType::Procedural;
        }

        // Time references → episodic
        if content.contains("today") || content.contains("yesterday") || content.contains("ago")
            || content.contains("this morning") || content.contains("just now")
            || content.contains("earlier")
        {
            return MemoryType::Episodic;
        }

        // Default → semantic (facts and concepts)
        MemoryType::Semantic
    }
}

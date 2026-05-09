//! Memory types — ported from yhc007/memory-brain
pub use uuid::Uuid;
use serde::{Deserialize, Serialize};
use std::fmt;

pub type UuidValue = Uuid;

/// Type of memory storage
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MemoryType {
    Working,
    Episodic,
    Semantic,
    Procedural,
}

impl fmt::Display for MemoryType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryType::Working => write!(f, "working"),
            MemoryType::Episodic => write!(f, "episodic"),
            MemoryType::Semantic => write!(f, "semantic"),
            MemoryType::Procedural => write!(f, "procedural"),
        }
    }
}

/// Emotional valence affects memory strength
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Emotion {
    Neutral,
    Positive,
    Negative,
    Surprise,
}

/// A single memory item — central data type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryItem {
    pub id: Uuid,
    pub content: String,
    pub context: Option<String>,
    pub memory_type: MemoryType,
    pub emotion: Emotion,
    pub created_at: String,
    pub last_accessed: String,
    pub access_count: u32,
    pub strength: f32,
    pub embedding: Option<Vec<f32>>,
    pub associations: Vec<Uuid>,
    pub tags: Vec<String>,
}

impl MemoryItem {
    pub fn new(content: &str, memory_type: MemoryType) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        Self {
            id: Uuid::new_v4(),
            content: content.to_string(),
            context: None,
            memory_type,
            emotion: Emotion::Neutral,
            created_at: now.clone(),
            last_accessed: now,
            access_count: 1,
            strength: 1.0,
            embedding: None,
            associations: Vec::new(),
            tags: Vec::new(),
        }
    }

    pub fn relevance_score(&self) -> f32 {
        self.strength * 0.7 + (self.access_count as f32).min(10.0) / 10.0 * 0.3
    }

    pub fn access(&mut self) {
        self.last_accessed = chrono::Utc::now().to_rfc3339();
        self.access_count += 1;
        self.strength = (self.strength + 0.1).min(1.0);
    }

    pub fn decay(&mut self, factor: f32) {
        self.strength *= factor;
    }

    pub fn is_forgotten(&self) -> bool {
        self.strength < 0.1
    }

    pub fn with_context(mut self, context: &str) -> Self {
        self.context = Some(context.to_string());
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    pub fn with_emotion(mut self, emotion: Emotion) -> Self {
        if !matches!(emotion, Emotion::Neutral) {
            self.strength = (self.strength * 1.5).min(1.0);
        }
        self.emotion = emotion;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_item_creation() {
        let item = MemoryItem::new("test content", MemoryType::Semantic);
        assert_eq!(item.content, "test content");
        assert_eq!(item.memory_type, MemoryType::Semantic);
        assert_eq!(item.strength, 1.0);
    }

    #[test]
    fn test_memory_item_type_display() {
        assert_eq!(MemoryType::Working.to_string(), "working");
        assert_eq!(MemoryType::Episodic.to_string(), "episodic");
        assert_eq!(MemoryType::Semantic.to_string(), "semantic");
        assert_eq!(MemoryType::Procedural.to_string(), "procedural");
    }

    #[test]
    fn test_relevance_score() {
        let item = MemoryItem::new("test", MemoryType::Semantic);
        let score = item.relevance_score();
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_emotional_memory_stronger() {
        let mut neutral = MemoryItem::new("neutral", MemoryType::Semantic);
        neutral.strength = 0.5;
        let emotional = MemoryItem::new("emotional", MemoryType::Semantic)
            .with_emotion(Emotion::Positive);
        // emotional starts at 1.0, gets boosted to min(1.5, 1.0) = 1.0
        // neutral is 0.5 → emotional > neutral
        assert!(emotional.strength > neutral.strength);
    }
}

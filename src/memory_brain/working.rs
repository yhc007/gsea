//! Working memory — short-term, limited capacity (7±2 items)
use crate::memory_brain::types::*;

pub struct WorkingMemory {
    items: Vec<MemoryItem>,
    capacity: usize,
}

impl WorkingMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            items: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Add item to working memory. Evicts lowest-relevance item if full.
    pub fn push(&mut self, item: MemoryItem) {
        if self.items.len() >= self.capacity {
            // Find and remove the item with lowest relevance score
            if let Some(idx) = self
                .items
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.relevance_score().partial_cmp(&b.relevance_score()).unwrap())
                .map(|(i, _)| i)
            {
                self.items.remove(idx);
            }
        }
        self.items.push(item);
    }

    /// Search working memory by keyword.
    pub fn search(&self, query: &str) -> Vec<MemoryItem> {
        let q = query.to_lowercase();
        self.items
            .iter()
            .filter(|item| item.content.to_lowercase().contains(&q))
            .cloned()
            .collect()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn clear(&mut self) {
        self.items.clear();
    }
}

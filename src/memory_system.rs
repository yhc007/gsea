//! MemorySystem — thin wrapper around the CLS MemoryGuardian.
//!
//! Connects the `memory-actor` crate into the GSEA pekko run-mode.
//! All heavy lifting is done by `MemoryGuardian`; this module provides
//! a convenient, ergonomic API surface for the rest of the binary.

use memory_actor::{
    MemoryGuardian, MemorySystemConfig, MemorySystemStats,
    MemoryContext, RecallResult,
};
use uuid::Uuid;

/// Wrapper around [`MemoryGuardian`] exposed to the GSEA actor system.
pub struct MemorySystem {
    pub guardian: MemoryGuardian,
}

impl MemorySystem {
    /// Create a new `MemorySystem` with default CLS configuration.
    pub fn new() -> Self {
        Self {
            guardian: MemoryGuardian::new(MemorySystemConfig::default()),
        }
    }

    /// Store `content` as a new episodic memory and return its UUID.
    pub fn store(&mut self, content: &str) -> Uuid {
        self.guardian.store(
            content.to_string(),
            MemoryContext {
                source: "pekko-repl".to_string(),
                ..Default::default()
            },
        )
    }

    /// Recall the top-`k` memories semantically closest to `query`.
    pub fn recall(&mut self, query: &str, k: usize) -> Vec<RecallResult> {
        self.guardian.recall(query, k)
    }

    /// Run one round of dream (offline) consolidation.
    pub fn dream(&mut self) {
        self.guardian.start_dream();
    }

    /// Return a snapshot of system-wide statistics.
    pub fn stats(&self) -> MemorySystemStats {
        self.guardian.stats()
    }
}

impl Default for MemorySystem {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_returns_uuid() {
        let mut ms = MemorySystem::new();
        let id = ms.store("hello world");
        // A freshly generated UUID is not nil.
        assert_ne!(id, Uuid::nil());
    }

    #[test]
    fn test_recall_finds_stored_memory() {
        let mut ms = MemorySystem::new();
        let id = ms.store("Rust is a systems programming language");
        let results = ms.recall("systems language", 5);
        assert!(!results.is_empty(), "should recall at least one memory");
        assert_eq!(results[0].memory.id, id);
    }

    #[test]
    fn test_recall_empty_store_returns_empty() {
        let mut ms = MemorySystem::new();
        let results = ms.recall("anything", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_dream_runs_without_panic() {
        let mut ms = MemorySystem::new();
        ms.store("memory alpha");
        ms.store("memory beta");
        ms.dream(); // must not panic
        let stats = ms.stats();
        assert!(stats.dream_last_run.is_some(), "dream should have recorded a timestamp");
    }

    #[test]
    fn test_stats_reflect_stored_memories() {
        let mut ms = MemorySystem::new();
        assert_eq!(ms.stats().total_memories, 0);
        ms.store("first");
        ms.store("second");
        assert_eq!(ms.stats().total_memories, 2);
    }

    #[test]
    fn test_stats_subsystems_active() {
        let ms = MemorySystem::new();
        let stats = ms.stats();
        assert!(stats.hippocampus_active);
        assert!(stats.neocortex_active);
    }
}

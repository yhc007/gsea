//! Pekko adapter — bridges GSEA's Brain (SQLite) to pekko-agent-core memory traits.
//!
//! # Structs
//! - [`BrainShortTermMemory`] — implements [`ShortTermMemory`], tracks per-session
//!   conversation windows in an in-memory HashMap and also persists messages to
//!   Brain's episodic store for long-term retention.
//! - [`BrainLongTermMemory`] — implements [`LongTermMemory`], delegates to
//!   Brain::learn / Brain::recall / Brain::forget.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use uuid::Uuid;

use pekko_agent_core::{
    memory::{LongTermMemory, MemoryDocument, SearchResult, ShortTermMemory},
    error::MemoryError,
    message::{Message, MessageRole},
};

use crate::memory_brain::{Brain, MemoryItem, MemoryType};

// ─── helpers ────────────────────────────────────────────────────────────────

fn brain_err(e: anyhow::Error) -> MemoryError {
    MemoryError::StorageError(e.to_string())
}

/// Serialise a `Message` into a compact string that can be stored inside Brain.
/// Format: `[<role>] <content>`
fn message_to_content(msg: &Message) -> String {
    format!("[{}] {}", msg.role, msg.content)
}

/// Deserialise the compact string back to a `Message`.
/// If the prefix cannot be parsed the whole string is treated as assistant content.
fn content_to_message(s: &str) -> Message {
    if let Some(rest) = s.strip_prefix("[user] ") {
        return Message {
            role: MessageRole::User,
            content: rest.to_string(),
            timestamp: chrono::Utc::now(),
        };
    }
    if let Some(rest) = s.strip_prefix("[assistant] ") {
        return Message {
            role: MessageRole::Assistant,
            content: rest.to_string(),
            timestamp: chrono::Utc::now(),
        };
    }
    if let Some(rest) = s.strip_prefix("[system] ") {
        return Message {
            role: MessageRole::System,
            content: rest.to_string(),
            timestamp: chrono::Utc::now(),
        };
    }
    if let Some(rest) = s.strip_prefix("[tool] ") {
        return Message {
            role: MessageRole::Tool,
            content: rest.to_string(),
            timestamp: chrono::Utc::now(),
        };
    }
    Message {
        role: MessageRole::Assistant,
        content: s.to_string(),
        timestamp: chrono::Utc::now(),
    }
}

// ─── BrainShortTermMemory ───────────────────────────────────────────────────

/// Wraps [`Brain`] to implement pekko-agent-core's [`ShortTermMemory`] trait.
///
/// Conversation windows are kept in an in-memory `HashMap<Uuid, Vec<Message>>`
/// protected by a `Mutex` so the struct is `Send + Sync`.  Each appended
/// message is *also* persisted to Brain's episodic memory so it survives
/// process restarts via long-term recall.
pub struct BrainShortTermMemory {
    brain: Arc<Mutex<Brain>>,
    /// In-memory window: session_id → ordered list of messages
    conversations: Mutex<HashMap<Uuid, Vec<Message>>>,
}

impl BrainShortTermMemory {
    pub fn new(brain: Arc<Mutex<Brain>>) -> Self {
        Self {
            brain,
            conversations: Mutex::new(HashMap::new()),
        }
    }

    /// Convenience constructor: creates its own `Brain` from `storage_path`.
    pub fn from_path(storage_path: &str) -> Result<Self, anyhow::Error> {
        let brain = Brain::new(storage_path)?;
        Ok(Self::new(Arc::new(Mutex::new(brain))))
    }
}

#[async_trait]
impl ShortTermMemory for BrainShortTermMemory {
    async fn get_conversation(&self, session_id: &Uuid) -> Result<Vec<Message>, MemoryError> {
        let guard = self.conversations.lock().unwrap();
        Ok(guard.get(session_id).cloned().unwrap_or_default())
    }

    async fn append_message(&self, session_id: &Uuid, msg: Message) -> Result<(), MemoryError> {
        // 1. Push to in-memory window
        {
            let mut guard = self.conversations.lock().unwrap();
            guard.entry(*session_id).or_default().push(msg.clone());
        }

        // 2. Persist to Brain episodic memory (best-effort; failure is not fatal)
        let content = message_to_content(&msg);
        let context = format!("session:{}", session_id);
        let item = MemoryItem::new(&content, MemoryType::Episodic)
            .with_context(&context);

        let brain = self.brain.lock().unwrap();
        // Use episodic.store directly to avoid re-classifying as Semantic
        let _ = brain.episodic.store(item);

        Ok(())
    }

    async fn summarize(&self, session_id: &Uuid) -> Result<String, MemoryError> {
        let guard = self.conversations.lock().unwrap();
        let messages = guard.get(session_id).cloned().unwrap_or_default();

        if messages.is_empty() {
            return Ok(String::from("No conversation history."));
        }

        let summary = messages
            .iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n");

        Ok(format!(
            "Conversation summary ({} messages):\n{}",
            messages.len(),
            summary
        ))
    }

    async fn clear(&self, session_id: &Uuid) -> Result<(), MemoryError> {
        let mut guard = self.conversations.lock().unwrap();
        guard.remove(session_id);
        Ok(())
    }
}

// ─── BrainLongTermMemory ───────────────────────────────────────────────────

/// Wraps [`Brain`] to implement pekko-agent-core's [`LongTermMemory`] trait.
///
/// - `store` → [`Brain::learn`] (stores as semantic memory, returns UUID string)
/// - `search` → [`Brain::recall`] (keyword search across all memory types)
/// - `delete` → [`Brain::forget`] (removes from all stores)
pub struct BrainLongTermMemory {
    brain: Arc<Mutex<Brain>>,
}

impl BrainLongTermMemory {
    pub fn new(brain: Arc<Mutex<Brain>>) -> Self {
        Self { brain }
    }

    /// Convenience constructor: creates its own `Brain` from `storage_path`.
    pub fn from_path(storage_path: &str) -> Result<Self, anyhow::Error> {
        let brain = Brain::new(storage_path)?;
        Ok(Self::new(Arc::new(Mutex::new(brain))))
    }
}

#[async_trait]
impl LongTermMemory for BrainLongTermMemory {
    async fn store(&self, doc: MemoryDocument) -> Result<String, MemoryError> {
        // Encode source + agent_id + metadata into the content stored in Brain
        let content = format!(
            "[source:{}][agent:{}] {}",
            doc.source, doc.agent_id, doc.content
        );
        let brain = self.brain.lock().unwrap();

        // Use the caller-supplied id if non-empty, otherwise let Brain generate one
        if doc.id.is_empty() {
            let uuid = brain.learn(&content).map_err(brain_err)?;
            Ok(uuid.to_string())
        } else {
            // Store with the given id: build a MemoryItem directly
            let mut item = MemoryItem::new(&content, MemoryType::Semantic);
            // Parse and reuse the caller-supplied id when it's a valid UUID
            if let Ok(parsed) = uuid::Uuid::parse_str(&doc.id) {
                item.id = parsed;
            }
            brain.semantic.store(item).map_err(brain_err)?;
            Ok(doc.id)
        }
    }

    async fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>, MemoryError> {
        let brain = self.brain.lock().unwrap();
        let items = brain.recall(query, top_k);

        let results = items
            .into_iter()
            .map(|item| {
                // Extract source from encoded prefix if present
                let source = extract_source(&item.content)
                    .unwrap_or_else(|| "brain".to_string());
                SearchResult {
                    id: item.id.to_string(),
                    score: item.relevance_score(),
                    content: strip_prefixes(&item.content),
                    source,
                }
            })
            .collect();

        Ok(results)
    }

    async fn delete(&self, doc_id: &str) -> Result<(), MemoryError> {
        let brain = self.brain.lock().unwrap();
        brain.forget(doc_id).map_err(brain_err)
    }
}

// ─── prefix helpers ─────────────────────────────────────────────────────────

/// Extract the `[source:…]` tag from an encoded content string.
fn extract_source(content: &str) -> Option<String> {
    let start = content.find("[source:")?;
    let rest = &content[start + 8..];
    let end = rest.find(']')?;
    Some(rest[..end].to_string())
}

/// Strip `[source:…][agent:…]` prefixes that were added during storage.
fn strip_prefixes(content: &str) -> String {
    let mut s = content;
    while s.starts_with('[') {
        if let Some(end) = s.find(']') {
            s = s[end + 1..].trim_start();
        } else {
            break;
        }
    }
    s.to_string()
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn make_brain(dir: &str) -> Arc<Mutex<Brain>> {
        Arc::new(Mutex::new(Brain::new(dir).expect("Brain::new")))
    }

    // ── ShortTermMemory ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn brain_short_term_append_and_get() {
        let dir = tempdir().unwrap();
        let brain = make_brain(dir.path().to_str().unwrap());
        let mem = BrainShortTermMemory::new(brain);

        let sid = Uuid::new_v4();
        mem.append_message(&sid, Message::user("hello")).await.unwrap();
        mem.append_message(&sid, Message::assistant("hi there")).await.unwrap();

        let conv = mem.get_conversation(&sid).await.unwrap();
        assert_eq!(conv.len(), 2);
        assert!(matches!(conv[0].role, MessageRole::User));
        assert_eq!(conv[0].content, "hello");
        assert!(matches!(conv[1].role, MessageRole::Assistant));
    }

    #[tokio::test]
    async fn brain_short_term_clear() {
        let dir = tempdir().unwrap();
        let brain = make_brain(dir.path().to_str().unwrap());
        let mem = BrainShortTermMemory::new(brain);

        let sid = Uuid::new_v4();
        mem.append_message(&sid, Message::user("test")).await.unwrap();
        assert_eq!(mem.get_conversation(&sid).await.unwrap().len(), 1);

        mem.clear(&sid).await.unwrap();
        assert_eq!(mem.get_conversation(&sid).await.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn brain_short_term_empty_session_returns_empty_vec() {
        let dir = tempdir().unwrap();
        let brain = make_brain(dir.path().to_str().unwrap());
        let mem = BrainShortTermMemory::new(brain);

        let sid = Uuid::new_v4();
        let conv = mem.get_conversation(&sid).await.unwrap();
        assert!(conv.is_empty());
    }

    #[tokio::test]
    async fn brain_short_term_summarize() {
        let dir = tempdir().unwrap();
        let brain = make_brain(dir.path().to_str().unwrap());
        let mem = BrainShortTermMemory::new(brain);

        let sid = Uuid::new_v4();
        mem.append_message(&sid, Message::user("what is rust?")).await.unwrap();
        mem.append_message(&sid, Message::assistant("a systems language")).await.unwrap();

        let summary = mem.summarize(&sid).await.unwrap();
        assert!(summary.contains("2 messages"));
        assert!(summary.contains("what is rust?"));
    }

    #[tokio::test]
    async fn brain_short_term_sessions_are_independent() {
        let dir = tempdir().unwrap();
        let brain = make_brain(dir.path().to_str().unwrap());
        let mem = BrainShortTermMemory::new(brain);

        let sid1 = Uuid::new_v4();
        let sid2 = Uuid::new_v4();

        mem.append_message(&sid1, Message::user("session 1")).await.unwrap();
        mem.append_message(&sid2, Message::user("session 2")).await.unwrap();
        mem.append_message(&sid2, Message::user("session 2 again")).await.unwrap();

        assert_eq!(mem.get_conversation(&sid1).await.unwrap().len(), 1);
        assert_eq!(mem.get_conversation(&sid2).await.unwrap().len(), 2);
    }

    // ── LongTermMemory ───────────────────────────────────────────────────────

    fn make_doc(content: &str) -> MemoryDocument {
        MemoryDocument {
            id: String::new(),
            content: content.to_string(),
            source: "test".to_string(),
            agent_id: "agent-1".to_string(),
            metadata: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn brain_long_term_store_returns_id() {
        let dir = tempdir().unwrap();
        let brain = make_brain(dir.path().to_str().unwrap());
        let mem = BrainLongTermMemory::new(brain);

        let id = mem.store(make_doc("Rust is fast")).await.unwrap();
        assert!(!id.is_empty());
        // Should be a valid UUID
        Uuid::parse_str(&id).expect("store() should return a UUID string");
    }

    #[tokio::test]
    async fn brain_long_term_store_with_explicit_id() {
        let dir = tempdir().unwrap();
        let brain = make_brain(dir.path().to_str().unwrap());
        let mem = BrainLongTermMemory::new(brain);

        let explicit_id = Uuid::new_v4().to_string();
        let doc = MemoryDocument {
            id: explicit_id.clone(),
            content: "explicit id test".to_string(),
            source: "test".to_string(),
            agent_id: "agent-1".to_string(),
            metadata: HashMap::new(),
        };
        let returned_id = mem.store(doc).await.unwrap();
        assert_eq!(returned_id, explicit_id);
    }

    #[tokio::test]
    async fn brain_long_term_search() {
        let dir = tempdir().unwrap();
        let brain = make_brain(dir.path().to_str().unwrap());
        let mem = BrainLongTermMemory::new(brain);

        mem.store(make_doc("memory palace technique for learning")).await.unwrap();
        mem.store(make_doc("spaced repetition for memorization")).await.unwrap();
        mem.store(make_doc("unrelated document about cats")).await.unwrap();

        let results = mem.search("memory", 10).await.unwrap();
        assert!(!results.is_empty(), "should find memory-related documents");
        // All results should have a positive score
        for r in &results {
            assert!(r.score >= 0.0);
        }
    }

    #[tokio::test]
    async fn brain_long_term_delete() {
        let dir = tempdir().unwrap();
        let brain = make_brain(dir.path().to_str().unwrap());
        let mem = BrainLongTermMemory::new(brain);

        let id = mem.store(make_doc("to be deleted")).await.unwrap();
        // Verify it's findable
        let before = mem.search("deleted", 10).await.unwrap();
        assert!(!before.is_empty());

        mem.delete(&id).await.unwrap();

        // After deletion it should no longer appear
        let after = mem.search("deleted", 10).await.unwrap();
        assert!(after.iter().all(|r| r.id != id));
    }

    // ── helper unit tests ────────────────────────────────────────────────────

    #[test]
    fn brain_message_round_trip() {
        for (role_str, role) in [
            ("user", MessageRole::User),
            ("assistant", MessageRole::Assistant),
            ("system", MessageRole::System),
            ("tool", MessageRole::Tool),
        ] {
            let msg = Message {
                role,
                content: "hello".to_string(),
                timestamp: chrono::Utc::now(),
            };
            let encoded = message_to_content(&msg);
            assert_eq!(encoded, format!("[{}] hello", role_str));
            let decoded = content_to_message(&encoded);
            assert_eq!(decoded.content, "hello");
        }
    }

    #[test]
    fn brain_strip_prefixes_works() {
        let raw = "[source:test][agent:a1] actual content here";
        assert_eq!(strip_prefixes(raw), "actual content here");
    }

    #[test]
    fn brain_extract_source_works() {
        let raw = "[source:wikipedia][agent:a1] some fact";
        assert_eq!(extract_source(raw), Some("wikipedia".to_string()));
        assert_eq!(extract_source("no prefixes here"), None);
    }
}

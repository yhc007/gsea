use anyhow::Result;
use rusqlite::Connection;
use std::path::PathBuf;
use std::sync::Mutex;

use crate::llm::embedding::{blob_to_vec, vec_to_blob, cosine_similarity};

/// MemoryBrain — persistent memory system with multiple tiers.
pub struct MemoryBrain {
    pub conn: Mutex<Connection>,
    db_path: PathBuf,
}

impl MemoryBrain {
    /// Open (or create) the SQLite database at the given path.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path: PathBuf = path.into();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(&path)?;
        let brain = Self {
            conn: Mutex::new(conn),
            db_path: path,
        };
        brain.initialize_schema()?;
        Ok(brain)
    }

    fn initialize_schema(&self) -> Result<()> {
        let guard = self.conn.lock().unwrap();
        guard.execute_batch(
            "
            -- Episodic memory: raw interaction logs
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                session_id TEXT NOT NULL,
                input TEXT NOT NULL,
                response TEXT,
                tool_calls TEXT,     -- JSON array
                outcome TEXT,        -- 'success', 'failure', 'partial'
                tokens_used INTEGER DEFAULT 0,
                duration_ms INTEGER DEFAULT 0,
                tags TEXT            -- JSON array
            );

            -- Semantic memory: vector+text knowledge store
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding BLOB,          -- F32 vector bytes
                content TEXT NOT NULL,
                content_type TEXT NOT NULL DEFAULT 'note',
                tags TEXT,
                source TEXT,             -- origin reference
                importance REAL DEFAULT 0.5,
                created TEXT NOT NULL DEFAULT (datetime('now')),
                last_accessed TEXT,
                access_count INTEGER DEFAULT 0
            );

            -- Skills: reusable capabilities the agent has learned
            CREATE TABLE IF NOT EXISTS skills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                code TEXT,               -- Rust source
                test_code TEXT,
                dependencies TEXT,        -- JSON of crate deps
                success_rate REAL DEFAULT 1.0,
                use_count INTEGER DEFAULT 0,
                version INTEGER DEFAULT 1,
                created TEXT NOT NULL DEFAULT (datetime('now')),
                updated TEXT
            );

            -- Reflection journal
            CREATE TABLE IF NOT EXISTS reflections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                trigger TEXT,
                observation TEXT NOT NULL,
                hypothesis TEXT,
                action_plan TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                result TEXT,
                related_episode_id INTEGER REFERENCES episodes(id)
            );

            -- Vector index metadata
            CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(content_type);
            CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
            CREATE INDEX IF NOT EXISTS idx_skills_name ON skills(name);
            CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
            ",
        )?;
        Ok(())
    }

    // ─── Episodic ──────────────────────────────────────────────

    pub fn record_episode(
        &self,
        session_id: &str,
        input: &str,
        response: Option<&str>,
        tool_calls: Option<&str>,
        outcome: &str,
        tokens_used: i64,
        duration_ms: i64,
        tags: Option<&str>,
    ) -> Result<i64> {
        let guard = self.conn.lock().unwrap();
        guard.execute(
            "INSERT INTO episodes (session_id, input, response, tool_calls, outcome, tokens_used, duration_ms, tags)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![session_id, input, response, tool_calls, outcome, tokens_used, duration_ms, tags],
        )?;
        Ok(guard.last_insert_rowid())
    }

    pub fn recent_episodes(&self, limit: usize) -> Result<Vec<(i64, String, String)>> {
        let guard = self.conn.lock().unwrap();
        let mut stmt = guard.prepare(
            "SELECT id, input, outcome FROM episodes ORDER BY id DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(rusqlite::params![limit as i64], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    // ─── Semantic Memory ────────────────────────────────────────

    pub fn store_memory(
        &self,
        content: &str,
        content_type: &str,
        tags: Option<&str>,
        source: Option<&str>,
        importance: f64,
    ) -> Result<i64> {
        let guard = self.conn.lock().unwrap();
        guard.execute(
            "INSERT INTO memories (content, content_type, tags, source, importance)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![content, content_type, tags, source, importance],
        )?;
        Ok(guard.last_insert_rowid())
    }

    pub fn search_memories(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<(i64, String, String, f64)>> {
        let pattern = format!("%{}%", query);
        let guard = self.conn.lock().unwrap();
        let mut stmt = guard.prepare(
            "SELECT id, content, content_type, importance
             FROM memories
             WHERE content LIKE ?1
             ORDER BY importance DESC, last_accessed DESC
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(rusqlite::params![pattern, limit as i64], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn get_memory_stats(&self) -> Result<serde_json::Value> {
        let guard = self.conn.lock().unwrap();
        let total: i64 = guard.query_row(
            "SELECT COUNT(*) FROM memories", [], |row| row.get(0))?;
        let mut stmt = guard.prepare(
            "SELECT content_type, COUNT(*) FROM memories GROUP BY content_type",
        )?;
        let rows = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
        let by_type: Vec<(String, i64)> = rows.collect::<Result<Vec<_>, _>>().map_err(anyhow::Error::from)?;

        Ok(serde_json::json!({
            "total_memories": total,
            "by_type": by_type,
        }))
    }

    // ─── Skills ────────────────────────────────────────────────

    pub fn store_skill(
        &self,
        name: &str,
        description: &str,
        code: &str,
        dependencies: Option<&str>,
    ) -> Result<i64> {
        let guard = self.conn.lock().unwrap();
        guard.execute(
            "INSERT INTO skills (name, description, code, dependencies)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(name) DO UPDATE SET
                description = ?2, code = ?3, dependencies = ?4,
                version = version + 1, updated = datetime('now')",
            rusqlite::params![name, description, code, dependencies],
        )?;
        Ok(guard.last_insert_rowid())
    }

    pub fn list_skills(&self) -> Result<Vec<(String, String, f64, i64)>> {
        let guard = self.conn.lock().unwrap();
        let mut stmt = guard.prepare(
            "SELECT name, description, success_rate, use_count FROM skills ORDER BY use_count DESC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    // ─── Reflections ────────────────────────────────────────────

    pub fn record_reflection(
        &self,
        trigger: &str,
        observation: &str,
        hypothesis: Option<&str>,
        action_plan: Option<&str>,
    ) -> Result<i64> {
        let guard = self.conn.lock().unwrap();
        guard.execute(
            "INSERT INTO reflections (trigger, observation, hypothesis, action_plan)
             VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![trigger, observation, hypothesis, action_plan],
        )?;
        Ok(guard.last_insert_rowid())
    }

    pub fn pending_reflections(&self) -> Result<Vec<(i64, String, String)>> {
        let guard = self.conn.lock().unwrap();
        let mut stmt = guard.prepare(
            "SELECT id, observation, action_plan FROM reflections WHERE status = 'pending' ORDER BY id",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn update_reflection_status(
        &self,
        id: i64,
        status: &str,
        result: Option<&str>,
    ) -> Result<()> {
        let guard = self.conn.lock().unwrap();
        guard.execute(
            "UPDATE reflections SET status = ?1, result = COALESCE(?2, result) WHERE id = ?3",
            rusqlite::params![status, result, id],
        )?;
        Ok(())
    }

    pub fn generate_context_summary(&self) -> Result<String> {
        let guard = self.conn.lock().unwrap();
        let episode_count: i64 = guard.query_row(
            "SELECT COUNT(*) FROM episodes WHERE timestamp > datetime('now', '-1 hour')",
            [],
            |row| row.get(0),
        )?;
        let success_rate: f64 = guard.query_row(
            "SELECT CASE WHEN COUNT(*) > 0
                THEN CAST(SUM(CASE WHEN outcome='success' THEN 1 ELSE 0 END) AS REAL) / COUNT(*)
                ELSE 0 END
             FROM episodes WHERE timestamp > datetime('now', '-1 hour')",
            [],
            |row| row.get(0),
        )?;
        let memory_count: i64 =
            guard.query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))?;
        let skill_count: i64 =
            guard.query_row("SELECT COUNT(*) FROM skills", [], |row| row.get(0))?;
        let pending_reflections: i64 = guard.query_row(
            "SELECT COUNT(*) FROM reflections WHERE status = 'pending'",
            [],
            |row| row.get(0),
        )?;

        Ok(format!(
            "📊 System Summary:\n\
             - Episodes (last hour): {}\n\
             - Success rate (last hour): {:.1}%\n\
             - Total memories: {}\n\
             - Learned skills: {}\n\
             - Pending reflections: {}",
            episode_count, success_rate * 100.0, memory_count, skill_count, pending_reflections
        ))
    }

    // ─── Embedding-based Semantic Search ────────────────────────

    /// Store a memory together with its embedding vector.
    pub fn store_with_embedding(
        &self,
        content: &str,
        content_type: &str,
        embedding: &[f32],
        importance: f64,
    ) -> Result<i64> {
        let blob = vec_to_blob(embedding);
        let guard = self.conn.lock().unwrap();
        guard.execute(
            "INSERT INTO memories (embedding, content, content_type, importance)
             VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![blob, content, content_type, importance],
        )?;
        Ok(guard.last_insert_rowid())
    }

    /// Search memories by embedding similarity. Returns (id, content, content_type, score).
    pub fn search_by_similarity(
        &self,
        query_embedding: &[f32],
        limit: usize,
        min_score: f64,
    ) -> Result<Vec<(i64, String, String, f64)>> {
        let guard = self.conn.lock().unwrap();
        let mut stmt = guard.prepare(
            "SELECT id, content, content_type, embedding FROM memories WHERE embedding IS NOT NULL",
        )?;
        let rows = stmt.query_map([], |row| {
            let id: i64 = row.get(0)?;
            let content: String = row.get(1)?;
            let content_type: String = row.get(2)?;
            let blob: Vec<u8> = row.get(3)?;
            let embedding = blob_to_vec(&blob);
            Ok((id, content, content_type, embedding))
        })?;

        let mut scored: Vec<(i64, String, String, f64)> = rows
            .filter_map(|r| r.ok())
            .map(|(id, content, content_type, emb)| {
                let score = cosine_similarity(query_embedding, &emb);
                (id, content, content_type, score)
            })
            .filter(|(_, _, _, score)| *score >= min_score)
            .collect();

        scored.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        Ok(scored)
    }
}

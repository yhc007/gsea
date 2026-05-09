//! Persistent storage for long-term memories using SQLite.
use anyhow::Result;
use crate::memory_brain::types::*;
use rusqlite::Connection;
use std::path::PathBuf;
use std::sync::Mutex;

// ─── Embedding vector helpers ───────────────────────────────────

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| *x as f64 * *y as f64).sum();
    let norm_a: f64 = a.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { return 0.0; }
    dot / (norm_a * norm_b)
}

fn vec_to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn blob_to_vec(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

pub struct Storage {
    conn: Mutex<Connection>,
    table: String,
}

impl Storage {
    pub fn new(db_dir: &str, table_name: &str) -> Result<Self, anyhow::Error> {
        let path = PathBuf::from(db_dir);
        std::fs::create_dir_all(&path)?;
        let db_path = path.join("memory_brain.db");
        let conn = Connection::open(&db_path)?;

        let storage = Self {
            conn: Mutex::new(conn),
            table: table_name.to_string(),
        };
        storage.initialize_schema()?;
        Ok(storage)
    }

    fn initialize_schema(&self) -> Result<(), anyhow::Error> {
        let sql = format!(
            "CREATE TABLE IF NOT EXISTS {} (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                context TEXT,
                memory_type TEXT NOT NULL,
                emotion TEXT NOT NULL DEFAULT 'Neutral',
                created_at TEXT NOT NULL,
                last_accessed TEXT NOT NULL,
                access_count INTEGER DEFAULT 1,
                strength REAL DEFAULT 1.0,
                embedding BLOB,
                associations TEXT DEFAULT '[]',
                tags TEXT DEFAULT '[]'
            )",
            self.table
        );
        let guard = self.conn.lock().unwrap();
        guard.execute_batch(&sql)?;
        Ok(())
    }

    pub fn insert(&self, item: MemoryItem) -> Result<(), anyhow::Error> {
        let embedding_blob = item.embedding.as_ref().map(|v| {
            v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>()
        });
        let associations_json = serde_json::to_string(&item.associations)?;
        let tags_json = serde_json::to_string(&item.tags)?;
        let mtype = item.memory_type.to_string();

        let sql = format!(
            "INSERT OR REPLACE INTO {} (id, content, context, memory_type, emotion, created_at, last_accessed, access_count, strength, embedding, associations, tags)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            self.table
        );
        let guard = self.conn.lock().unwrap();
        guard.execute(
            &sql,
            rusqlite::params![
                item.id.to_string(),
                item.content,
                item.context,
                mtype,
                format!("{:?}", item.emotion),
                item.created_at,
                item.last_accessed,
                item.access_count,
                item.strength,
                embedding_blob,
                associations_json,
                tags_json,
            ],
        )?;
        Ok(())
    }

    pub fn search(&self, query: &str, limit: usize, memory_type: MemoryType) -> Vec<MemoryItem> {
        let pattern = format!("%{}%", query);
        let mtype = memory_type.to_string();
        let sql = format!(
            "SELECT id, content, context, emotion, created_at, last_accessed, access_count, strength, associations, tags
             FROM {} WHERE memory_type = ?1 AND content LIKE ?2
             ORDER BY strength DESC, last_accessed DESC LIMIT ?3",
            self.table
        );

        let guard = self.conn.lock().unwrap();
        let mut stmt = match guard.prepare(&sql) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        let rows = match stmt.query_map(rusqlite::params![mtype, pattern, limit as i64], |row| {
            let assoc_str: String = row.get(8)?;
            let tags_str: String = row.get(9)?;
            Ok(MemoryItem {
                id: row.get::<_, String>(0).and_then(|s| Uuid::parse_str(&s).map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))).unwrap_or_default(),
                content: row.get(1)?,
                context: row.get(2)?,
                memory_type: memory_type.clone(),
                emotion: serde_json::from_str::<Emotion>(&format!("\"{}\"", row.get::<_, String>(3).unwrap_or_default())).unwrap_or(Emotion::Neutral),
                created_at: row.get(4)?,
                last_accessed: row.get(5)?,
                access_count: row.get(6)?,
                strength: row.get(7)?,
                embedding: None,
                associations: serde_json::from_str(&assoc_str).unwrap_or_default(),
                tags: serde_json::from_str(&tags_str).unwrap_or_default(),
            })
        }) {
            Ok(r) => r.filter_map(|r| r.ok()).collect(),
            Err(_) => Vec::new(),
        };

        rows
    }

    pub fn apply_decay(&self, factor: f32, memory_type: MemoryType) -> Result<(), anyhow::Error> {
        let mtype = memory_type.to_string();
        let sql = format!(
            "UPDATE {} SET strength = strength * ?1, last_accessed = datetime('now') WHERE memory_type = ?2 AND strength > 0.1",
            self.table
        );
        let guard = self.conn.lock().unwrap();
        guard.execute(&sql, rusqlite::params![factor, mtype])?;
        Ok(())
    }

    /// Delete a memory by its UUID string.
    pub fn delete(&self, id: &str) -> Result<(), anyhow::Error> {
        let sql = format!("DELETE FROM {} WHERE id = ?1", self.table);
        let guard = self.conn.lock().unwrap();
        guard.execute(&sql, rusqlite::params![id])?;
        Ok(())
    }

    /// Search memories by embedding similarity (cosine).
    /// Returns (MemoryItem, score) sorted by descending similarity.
    pub fn search_by_embedding(
        &self,
        query_emb: &[f32],
        limit: usize,
        memory_type: MemoryType,
        min_score: f64,
    ) -> Vec<(MemoryItem, f64)> {
        let mtype = memory_type.to_string();
        let sql = format!(
            "SELECT id, content, context, emotion, created_at, last_accessed, access_count, strength, embedding, associations, tags
             FROM {} WHERE memory_type = ?1 AND embedding IS NOT NULL",
            self.table
        );

        let guard = self.conn.lock().unwrap();
        let mut stmt = match guard.prepare(&sql) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        let rows: Vec<(MemoryItem, f64)> = match stmt.query_map(rusqlite::params![mtype], |row| {
            let assoc_str: String = row.get(9)?;
            let tags_str: String = row.get(10)?;
            let emb_blob: Vec<u8> = row.get(8)?;
            let emb = blob_to_vec(&emb_blob);

            let item = MemoryItem {
                id: row.get::<_, String>(0).and_then(|s| Uuid::parse_str(&s).map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))).unwrap_or_default(),
                content: row.get(1)?,
                context: row.get(2)?,
                memory_type: memory_type.clone(),
                emotion: serde_json::from_str::<Emotion>(&format!("\"{}\"", row.get::<_, String>(3).unwrap_or_default())).unwrap_or(Emotion::Neutral),
                created_at: row.get(4)?,
                last_accessed: row.get(5)?,
                access_count: row.get(6)?,
                strength: row.get(7)?,
                embedding: Some(emb.clone()),
                associations: serde_json::from_str(&assoc_str).unwrap_or_default(),
                tags: serde_json::from_str(&tags_str).unwrap_or_default(),
            };
            let score = cosine_similarity(query_emb, &emb);
            Ok((item, score))
        }) {
            Ok(r) => r.filter_map(|r| r.ok()).filter(|(_, s)| *s >= min_score).collect(),
            Err(_) => Vec::new(),
        };

        let mut scored = rows;
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        scored
    }

    pub fn count(&self, memory_type: MemoryType) -> i64 {
        let mtype = memory_type.to_string();
        let sql = format!("SELECT COUNT(*) FROM {} WHERE memory_type = ?1", self.table);
        let guard = self.conn.lock().unwrap();
        guard.query_row(&sql, rusqlite::params![mtype], |row| row.get(0)).unwrap_or(0)
    }
}

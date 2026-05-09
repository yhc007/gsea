use anyhow::Result;

/// A simple cosine similarity calculator & embedding engine trait.
#[async_trait::async_trait]
pub trait EmbeddingEngine: Send + Sync {
    /// Generate a vector embedding for the given text.
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
}

/// Embedding engine backed by Ollama's `/api/embeddings` endpoint.
pub struct OllamaEmbedder {
    client: crate::llm::OllamaClient,
}

impl OllamaEmbedder {
    pub fn new(ollama_base_url: &str, model: &str) -> Self {
        Self {
            client: crate::llm::OllamaClient::new(ollama_base_url, model),
        }
    }
}

#[async_trait::async_trait]
impl EmbeddingEngine for OllamaEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.client.embed(text).await
    }
}

/// Compute cosine similarity between two f32 vectors.
/// Returns a value in [-1.0, 1.0] (higher = more similar).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| *x as f64 * *y as f64).sum();
    let norm_a: f64 = a.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Serialize a vector of f32 to bytes for SQLite BLOB storage.
pub fn vec_to_blob(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| x.to_le_bytes())
        .collect()
}

/// Deserialize a vector of f32 from SQLite BLOB bytes.
pub fn blob_to_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

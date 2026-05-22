pub mod embedding;

use anyhow::Result;
use futures::StreamExt;
use pekko_actor::{CircuitBreaker, CircuitBreakerError, CircuitBreakerState};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;

/// Ollama API client for interacting with local Gemma models.
///
/// Includes an optional `CircuitBreaker` that protects HTTP calls to the
/// Ollama server.  When the circuit is open (too many consecutive failures
/// or timeouts) the client returns `Err` immediately rather than waiting.
#[derive(Clone)]
pub struct OllamaClient {
    base_url: String,
    model: String,
    client: reqwest::Client,
    /// Circuit breaker protecting the Ollama HTTP endpoint.
    cb: CircuitBreaker,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    stream: bool,
    tools: Option<Vec<ToolSpec>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatResponse {
    pub message: Message,
    pub done: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolSpec {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ToolFunction,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

impl OllamaClient {
    pub fn new(base_url: &str, model: &str) -> Self {
        let cb = CircuitBreaker::builder()
            .max_failures(3)
            .call_timeout(Duration::from_secs(120))
            .reset_timeout(Duration::from_secs(30))
            .max_half_open_calls(1)
            .exponential_backoff(2.0)
            .build();

        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            client: reqwest::Client::new(),
            cb,
        }
    }

    /// Change the model at runtime (used by GUI model switcher).
    pub fn set_model(&mut self, model: &str) {
        self.model = model.to_string();
    }

    pub fn model_name(&self) -> &str {
        &self.model
    }

    /// Check whether the circuit breaker is currently open.
    pub fn is_circuit_open(&self) -> bool {
        self.cb.state() == CircuitBreakerState::Open
    }

    /// Get circuit breaker stats for diagnostics.
    pub fn circuit_stats(&self) -> pekko_actor::CircuitBreakerStats {
        self.cb.stats()
    }

    /// Send a chat completion request (non-streaming), protected by the
    /// circuit breaker.
    pub async fn chat(&self, messages: Vec<Message>) -> Result<Message> {
        let base_url = self.base_url.clone();
        let client = self.client.clone();
        let body = ChatRequest {
            model: self.model.clone(),
            messages,
            stream: false,
            tools: None,
        };

        let result = self.cb.call(|| {
            let client = client.clone();
            let url = format!("{}/api/chat", base_url);
            let body_json = serde_json::to_value(&body).unwrap();
            async move {
                let resp = client
                    .post(&url)
                    .json(&body_json)
                    .send()
                    .await
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
                let chat_resp: ChatResponse = resp
                    .json()
                    .await
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
                Ok::<Message, anyhow::Error>(chat_resp.message)
            }
        }).await;

        match result {
            Ok(msg) => Ok(msg),
            Err(CircuitBreakerError::Open) => {
                Err(anyhow::anyhow!("circuit_open: Ollama circuit breaker is open (model: {})", self.model))
            }
            Err(CircuitBreakerError::Timeout) => {
                Err(anyhow::anyhow!("circuit_timeout: Ollama request timed out (model: {})", self.model))
            }
            Err(CircuitBreakerError::CallFailed(e)) => Err(e),
        }
    }

    /// Send a chat request with tool definitions (function calling),
    /// protected by the circuit breaker.
    pub async fn chat_with_tools(
        &self,
        messages: Vec<Message>,
        tools: Vec<ToolSpec>,
    ) -> Result<ChatResponse> {
        let base_url = self.base_url.clone();
        let client = self.client.clone();
        let body = ChatRequest {
            model: self.model.clone(),
            messages,
            stream: false,
            tools: Some(tools),
        };

        let result = self.cb.call(|| {
            let client = client.clone();
            let url = format!("{}/api/chat", base_url);
            let body_json = serde_json::to_value(&body).unwrap();
            async move {
                let resp = client
                    .post(&url)
                    .json(&body_json)
                    .send()
                    .await
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
                let chat_resp: ChatResponse = resp
                    .json()
                    .await
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
                Ok::<ChatResponse, anyhow::Error>(chat_resp)
            }
        }).await;

        match result {
            Ok(resp) => Ok(resp),
            Err(CircuitBreakerError::Open) => {
                Err(anyhow::anyhow!("circuit_open: Ollama circuit breaker is open (model: {})", self.model))
            }
            Err(CircuitBreakerError::Timeout) => {
                Err(anyhow::anyhow!("circuit_timeout: Ollama request timed out (model: {})", self.model))
            }
            Err(CircuitBreakerError::CallFailed(e)) => Err(e),
        }
    }

    /// Send a streaming chat request. Returns an mpsc Receiver that yields
    /// content chunks as they arrive. The final empty string signals completion.
    ///
    /// Note: the initial HTTP connection is protected by the circuit breaker.
    /// Once the stream is established, individual chunk failures do not trip it.
    pub async fn chat_stream(
        &self,
        messages: Vec<Message>,
    ) -> Result<tokio::sync::mpsc::Receiver<String>> {
        let base_url = self.base_url.clone();
        let client = self.client.clone();
        let body = ChatRequest {
            model: self.model.clone(),
            messages,
            stream: true,
            tools: None,
        };

        // Protect the initial connection with the circuit breaker
        let resp = self.cb.call(|| {
            let client = client.clone();
            let url = format!("{}/api/chat", base_url);
            let body_json = serde_json::to_value(&body).unwrap();
            async move {
                let resp = client
                    .post(&url)
                    .json(&body_json)
                    .send()
                    .await
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
                Ok::<reqwest::Response, anyhow::Error>(resp)
            }
        }).await;

        let resp = match resp {
            Ok(r) => r,
            Err(CircuitBreakerError::Open) => {
                return Err(anyhow::anyhow!("circuit_open: Ollama circuit breaker is open (model: {})", self.model));
            }
            Err(CircuitBreakerError::Timeout) => {
                return Err(anyhow::anyhow!("circuit_timeout: Ollama request timed out (model: {})", self.model));
            }
            Err(CircuitBreakerError::CallFailed(e)) => return Err(e),
        };

        let (tx, rx) = tokio::sync::mpsc::channel::<String>(64);

        tokio::spawn(async move {
            let mut stream = resp.bytes_stream();
            let mut buf = String::new();

            while let Some(chunk) = stream.next().await {
                let bytes = match chunk {
                    Ok(b) => b,
                    Err(_) => break,
                };
                buf.push_str(&String::from_utf8_lossy(&bytes));

                // Process complete lines (newline-delimited JSON)
                while let Some(newline_pos) = buf.find('\n') {
                    let line = buf[..newline_pos].trim().to_string();
                    buf = buf[newline_pos + 1..].to_string();

                    if line.is_empty() {
                        continue;
                    }

                    if let Ok(partial) = serde_json::from_str::<ChatResponse>(&line) {
                        if !partial.message.content.is_empty() {
                            if tx.send(partial.message.content).await.is_err() {
                                return; // receiver dropped
                            }
                        }
                        if partial.done {
                            return;
                        }
                    }
                }
            }
        });

        Ok(rx)
    }

    /// Generate embeddings using Ollama's embedding endpoint.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        #[derive(Serialize)]
        struct EmbedRequest {
            model: String,
            prompt: String,
        }

        #[derive(Deserialize)]
        struct EmbedResponse {
            embedding: Vec<f32>,
        }

        let body = EmbedRequest {
            model: self.model.clone(),
            prompt: text.to_string(),
        };

        let resp = self
            .client
            .post(format!("{}/api/embeddings", self.base_url))
            .json(&body)
            .send()
            .await?;

        let embed_resp: EmbedResponse = resp.json().await?;
        Ok(embed_resp.embedding)
    }
}

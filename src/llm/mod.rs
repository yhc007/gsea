pub mod embedding;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Ollama API client for interacting with local Gemma models.
#[derive(Clone)]
pub struct OllamaClient {
    base_url: String,
    model: String,
    client: reqwest::Client,
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

#[derive(Debug, Serialize)]
pub struct ToolSpec {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ToolFunction,
}

#[derive(Debug, Serialize)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

impl OllamaClient {
    pub fn new(base_url: &str, model: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Send a chat completion request (non-streaming).
    pub async fn chat(&self, messages: Vec<Message>) -> Result<Message> {
        let body = ChatRequest {
            model: self.model.clone(),
            messages,
            stream: false,
            tools: None,
        };

        let resp = self
            .client
            .post(format!("{}/api/chat", self.base_url))
            .json(&body)
            .send()
            .await?;

        let chat_resp: ChatResponse = resp.json().await?;
        Ok(chat_resp.message)
    }

    /// Send a chat request with tool definitions (function calling).
    pub async fn chat_with_tools(
        &self,
        messages: Vec<Message>,
        tools: Vec<ToolSpec>,
    ) -> Result<ChatResponse> {
        let body = ChatRequest {
            model: self.model.clone(),
            messages,
            stream: false,
            tools: Some(tools),
        };

        let resp = self
            .client
            .post(format!("{}/api/chat", self.base_url))
            .json(&body)
            .send()
            .await?;

        let chat_resp: ChatResponse = resp.json().await?;
        Ok(chat_resp)
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

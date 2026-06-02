//! Web API server for GSEA — REST + WebSocket endpoints.
//! Allows remote access to the agent system via HTTP.
//!
//! Architecture: Agent is !Send (rusqlite::Connection), so we use a channel-based
//! pattern. The agent lives on a dedicated thread with its own tokio runtime.
//! HTTP handlers send requests via channels and await responses.
//! WebSocket connections stream tokens in real-time via a broadcast channel.

use std::sync::Arc;

use axum::{
    Router,
    extract::{State, WebSocketUpgrade},
    extract::ws::{Message as WsMessage, WebSocket},
    response::{Json, IntoResponse},
    routing::{get, post},
};
use futures::SinkExt;
use futures::StreamExt;
use tower_http::cors::CorsLayer;

use crate::agent::Agent;
use crate::memory_brain::Brain;
use crate::tools::ToolRegistry;

/// Request to the agent worker thread (batch response).
struct AgentRequest {
    message: String,
    reply: tokio::sync::oneshot::Sender<Result<String, String>>,
}

/// Request to the agent worker thread (streaming response).
struct StreamRequest {
    message: String,
    chunk_tx: tokio::sync::mpsc::Sender<Result<String, String>>,
}

/// Shared state for the web server.
#[derive(Clone)]
struct AppState {
    agent_tx: tokio::sync::mpsc::Sender<AgentRequest>,
    stream_tx: tokio::sync::mpsc::Sender<StreamRequest>,
    brain: Arc<std::sync::Mutex<Brain>>,
    registry: Arc<std::sync::Mutex<ToolRegistry>>,
    model_name: String,
}

/// Start the web server on the given port.
pub async fn run_web_server(
    agent: Agent,
    brain: Arc<std::sync::Mutex<Brain>>,
    registry: Arc<std::sync::Mutex<ToolRegistry>>,
    model_name: &str,
    port: u16,
) -> anyhow::Result<()> {
    // Channel for batch chat requests
    let (agent_tx, agent_rx) = tokio::sync::mpsc::channel::<AgentRequest>(32);
    // Channel for streaming requests (WebSocket)
    let (stream_tx, stream_rx) = tokio::sync::mpsc::channel::<StreamRequest>(32);

    // Spawn agent worker on a dedicated thread (Agent is !Send due to rusqlite)
    let mut agent = agent;
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().expect("agent worker runtime");
        rt.block_on(async move {
            let mut batch_rx = agent_rx;
            let mut stream_rx = stream_rx;
            loop {
                tokio::select! {
                    Some(req) = batch_rx.recv() => {
                        let result = match agent.process_message(&req.message).await {
                            Ok(response) => Ok(response),
                            Err(e) => Err(e.to_string()),
                        };
                        let _ = req.reply.send(result);
                    }
                    Some(req) = stream_rx.recv() => {
                        match agent.process_message_stream(&req.message).await {
                            Ok(mut chunk_rx) => {
                                while let Some(chunk) = chunk_rx.recv().await {
                                    if req.chunk_tx.send(Ok(chunk)).await.is_err() {
                                        break; // client disconnected
                                    }
                                }
                                // Signal end of stream
                                let _ = req.chunk_tx.send(Err("__done__".to_string())).await;
                            }
                            Err(e) => {
                                let _ = req.chunk_tx.send(Err(e.to_string())).await;
                            }
                        }
                    }
                    else => break,
                }
            }
        });
    });

    let state = AppState {
        agent_tx,
        stream_tx,
        brain,
        registry,
        model_name: model_name.to_string(),
    };

    let app = Router::new()
        .route("/api/health", get(health))
        .route("/api/chat", post(chat))
        .route("/api/ws", get(ws_handler))
        .route("/api/stats", get(stats))
        .route("/api/brain", get(brain_stats))
        .route("/api/tools", get(list_tools))
        .route("/api/skills", get(list_skills))
        .route("/api/memory", post(store_memory))
        .route("/api/recall", post(recall))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    println!("GSEA Web API starting on http://{}", addr);
    println!("  Endpoints:");
    println!("    GET  /api/health  — health check");
    println!("    POST /api/chat    — send message ({{\"message\": \"...\"}})");
    println!("    GET  /api/ws      — WebSocket streaming chat");
    println!("    GET  /api/stats   — brain + tool stats");
    println!("    GET  /api/brain   — brain memory stats");
    println!("    GET  /api/tools   — list tools");
    println!("    GET  /api/skills  — list skills");
    println!("    POST /api/memory  — store memory ({{\"content\": \"...\"}})");
    println!("    POST /api/recall  — search memory ({{\"query\": \"...\", \"limit\": 5}})");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

// ─── Handlers ──────────────────────────────────────────────────

async fn health(State(state): State<AppState>) -> Json<serde_json::Value> {
    let brain = state.brain.lock().unwrap();
    let tool_count = state.registry.lock().unwrap().list_tools().len();
    Json(serde_json::json!({
        "status": "ok",
        "model": state.model_name,
        "brain": brain.stats(),
        "tools": tool_count,
    }))
}

#[derive(serde::Deserialize)]
struct ChatRequest {
    message: String,
}

async fn chat(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Json<serde_json::Value> {
    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
    let send_result = state.agent_tx.send(AgentRequest {
        message: req.message,
        reply: reply_tx,
    }).await;

    if send_result.is_err() {
        return Json(serde_json::json!({ "error": "agent worker unavailable" }));
    }

    match reply_rx.await {
        Ok(Ok(response)) => Json(serde_json::json!({
            "response": response,
            "model": state.model_name,
        })),
        Ok(Err(e)) => Json(serde_json::json!({ "error": e })),
        Err(_) => Json(serde_json::json!({ "error": "agent worker dropped" })),
    }
}

async fn stats(State(state): State<AppState>) -> Json<serde_json::Value> {
    let brain = state.brain.lock().unwrap();
    let tool_count = state.registry.lock().unwrap().list_tools().len();
    Json(serde_json::json!({
        "brain": brain.stats(),
        "tools": tool_count,
        "model": state.model_name,
    }))
}

async fn brain_stats(State(state): State<AppState>) -> Json<serde_json::Value> {
    let brain = state.brain.lock().unwrap();
    Json(brain.stats())
}

async fn list_tools(State(state): State<AppState>) -> Json<serde_json::Value> {
    let tools = state.registry.lock().unwrap();
    let names: Vec<serde_json::Value> = tools.list_tools().iter().map(|t| {
        serde_json::json!({
            "name": t.name(),
            "description": t.description(),
        })
    }).collect();
    Json(serde_json::json!({ "tools": names }))
}

async fn list_skills(State(state): State<AppState>) -> Json<serde_json::Value> {
    let brain = state.brain.lock().unwrap();
    let skills = brain.list_skills();
    let items: Vec<serde_json::Value> = skills.iter().map(|(name, desc)| {
        serde_json::json!({ "name": name, "description": desc })
    }).collect();
    Json(serde_json::json!({ "skills": items }))
}

#[derive(serde::Deserialize)]
struct MemoryRequest {
    content: String,
    #[serde(default)]
    memory_type: Option<String>,
}

async fn store_memory(
    State(state): State<AppState>,
    Json(req): Json<MemoryRequest>,
) -> Json<serde_json::Value> {
    let mut brain = state.brain.lock().unwrap();
    let mtype = match req.memory_type.as_deref() {
        Some("episodic") => crate::memory_brain::MemoryType::Episodic,
        Some("procedural") => crate::memory_brain::MemoryType::Procedural,
        _ => crate::memory_brain::MemoryType::Semantic,
    };
    match brain.process(&req.content, Some(mtype)) {
        Ok(id) => Json(serde_json::json!({ "id": id.to_string(), "status": "stored" })),
        Err(e) => Json(serde_json::json!({ "error": e.to_string() })),
    }
}

#[derive(serde::Deserialize)]
struct RecallRequest {
    query: String,
    #[serde(default = "default_limit")]
    limit: usize,
}

fn default_limit() -> usize { 5 }

async fn recall(
    State(state): State<AppState>,
    Json(req): Json<RecallRequest>,
) -> Json<serde_json::Value> {
    let brain = state.brain.lock().unwrap();
    let results = brain.recall(&req.query, req.limit);
    let items: Vec<serde_json::Value> = results.iter().map(|item| {
        serde_json::json!({
            "id": item.id.to_string(),
            "content": item.content,
            "type": format!("{:?}", item.memory_type),
            "created_at": item.created_at,
        })
    }).collect();
    Json(serde_json::json!({ "results": items, "count": items.len() }))
}

// ─── WebSocket Handler ─────────────────────────────────────────

/// WebSocket upgrade handler for real-time streaming chat.
///
/// Protocol:
///   Client sends: `{"message": "..."}`
///   Server streams: `{"type": "chunk", "data": "..."}` per token
///   Server sends:   `{"type": "done"}` when complete
///   Server sends:   `{"type": "error", "data": "..."}` on error
async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws(socket, state))
}

async fn handle_ws(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();

    // Send welcome message
    let welcome = serde_json::json!({
        "type": "connected",
        "model": state.model_name,
    });
    let _ = sender.send(WsMessage::Text(welcome.to_string().into())).await;

    while let Some(Ok(msg)) = receiver.next().await {
        let text = match msg {
            WsMessage::Text(t) => t.to_string(),
            WsMessage::Close(_) => break,
            _ => continue,
        };

        // Parse incoming message
        let parsed: serde_json::Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(_) => {
                let err = serde_json::json!({"type": "error", "data": "invalid JSON"});
                let _ = sender.send(WsMessage::Text(err.to_string().into())).await;
                continue;
            }
        };

        let message = match parsed.get("message").and_then(|m| m.as_str()) {
            Some(m) => m.to_string(),
            None => {
                let err = serde_json::json!({"type": "error", "data": "missing 'message' field"});
                let _ = sender.send(WsMessage::Text(err.to_string().into())).await;
                continue;
            }
        };

        // Create a channel for streaming chunks from the agent worker
        let (chunk_tx, mut chunk_rx) = tokio::sync::mpsc::channel::<Result<String, String>>(64);

        // Send streaming request to agent worker
        if state.stream_tx.send(StreamRequest {
            message,
            chunk_tx,
        }).await.is_err() {
            let err = serde_json::json!({"type": "error", "data": "agent worker unavailable"});
            let _ = sender.send(WsMessage::Text(err.to_string().into())).await;
            continue;
        }

        // Forward chunks to WebSocket
        while let Some(result) = chunk_rx.recv().await {
            match result {
                Ok(chunk) => {
                    let msg = serde_json::json!({"type": "chunk", "data": chunk});
                    if sender.send(WsMessage::Text(msg.to_string().into())).await.is_err() {
                        break; // client disconnected
                    }
                }
                Err(e) if e == "__done__" => {
                    let done = serde_json::json!({"type": "done"});
                    let _ = sender.send(WsMessage::Text(done.to_string().into())).await;
                    break;
                }
                Err(e) => {
                    let err = serde_json::json!({"type": "error", "data": e});
                    let _ = sender.send(WsMessage::Text(err.to_string().into())).await;
                    break;
                }
            }
        }
    }
}

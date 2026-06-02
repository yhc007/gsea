//! MCP (Model Context Protocol) server — exposes GSEA tools, memory, and prompts
//! to MCP clients (Claude Desktop, Cursor, Zed, etc.) via stdio transport.
//!
//! Protocol: JSON-RPC 2.0 over stdin/stdout
//! Spec: https://spec.modelcontextprotocol.io/

use std::sync::Arc;

use anyhow::Result;
use serde_json::Value;

use crate::agent::Agent;
use crate::llm::OllamaClient;
use crate::memory_brain::Brain;
use crate::tools::ToolRegistry;

/// Tool call timeout in seconds.
const TOOL_TIMEOUT_SECS: u64 = 60;

/// Interval for resource change checks (seconds).
const RESOURCE_POLL_SECS: u64 = 5;

/// Tracks resource subscriptions from MCP clients.
struct ResourceSubscriptions {
    /// Set of subscribed resource URIs.
    subscribed: std::collections::HashSet<String>,
    /// Cached snapshots for change detection (uri -> last known value).
    snapshots: std::collections::HashMap<String, String>,
}

impl ResourceSubscriptions {
    fn new() -> Self {
        Self {
            subscribed: std::collections::HashSet::new(),
            snapshots: std::collections::HashMap::new(),
        }
    }

    fn subscribe(&mut self, uri: &str) {
        self.subscribed.insert(uri.to_string());
    }

    fn unsubscribe(&mut self, uri: &str) {
        self.subscribed.remove(uri);
        self.snapshots.remove(uri);
    }

    fn is_subscribed(&self, uri: &str) -> bool {
        self.subscribed.contains(uri)
    }

    fn subscribed_uris(&self) -> Vec<String> {
        self.subscribed.iter().cloned().collect()
    }

    /// Check if a resource has changed since last snapshot. Returns true if changed.
    fn check_changed(&mut self, uri: &str, current_value: &str) -> bool {
        let changed = self.snapshots.get(uri).map_or(true, |prev| prev != current_value);
        if changed {
            self.snapshots.insert(uri.to_string(), current_value.to_string());
        }
        changed
    }
}

/// Run the MCP server: reads JSON-RPC requests from stdin, writes responses to stdout.
pub async fn run_mcp_server(
    tools: Arc<std::sync::Mutex<ToolRegistry>>,
    brain: Arc<std::sync::Mutex<Brain>>,
    agent: Option<Arc<tokio::sync::Mutex<Agent>>>,
    llm: Option<OllamaClient>,
) -> Result<()> {
    let mut line_buf = String::new();
    let mut stdin = tokio::io::BufReader::new(tokio::io::stdin());
    let subs = Arc::new(std::sync::Mutex::new(ResourceSubscriptions::new()));

    // Log server startup
    log_notification("info", "GSEA MCP server starting").await;

    // Spawn resource change monitor
    {
        let brain_clone = brain.clone();
        let subs_clone = subs.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(RESOURCE_POLL_SECS)).await;
                check_resource_changes(&brain_clone, &subs_clone).await;
            }
        });
    }

    loop {
        line_buf.clear();
        let n = tokio::io::AsyncBufReadExt::read_line(&mut stdin, &mut line_buf).await?;
        if n == 0 {
            break; // EOF
        }

        let line = line_buf.trim();
        if line.is_empty() {
            continue;
        }

        let request: RpcRequest = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => {
                let err = rpc_error(None, -32700, &format!("Parse error: {}", e));
                let _ = writeln_json(&err).await;
                continue;
            }
        };

        // Handle notifications (no response expected)
        if request.id.is_none() {
            handle_notification(&request).await;
            continue;
        }

        let response = handle_request(&request, &tools, &brain, &agent, &llm, &subs).await;
        let _ = writeln_json(&response).await;
    }

    Ok(())
}

// ─── JSON-RPC types ────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct RpcRequest {
    #[serde(default)]
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Option<Value>,
}

#[derive(serde::Serialize)]
struct RpcResponse {
    jsonrpc: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<RpcError>,
}

#[derive(serde::Serialize)]
struct RpcError {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

/// JSON-RPC notification (no id field).
#[derive(serde::Serialize)]
struct RpcNotification {
    jsonrpc: &'static str,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
}

fn rpc_result(id: Option<Value>, result: Value) -> RpcResponse {
    RpcResponse {
        jsonrpc: "2.0",
        id,
        result: Some(result),
        error: None,
    }
}

fn rpc_error(id: Option<Value>, code: i32, message: &str) -> RpcResponse {
    RpcResponse {
        jsonrpc: "2.0",
        id,
        result: None,
        error: Some(RpcError {
            code,
            message: message.to_string(),
            data: None,
        }),
    }
}

fn rpc_error_with_data(id: Option<Value>, code: i32, message: &str, data: Value) -> RpcResponse {
    RpcResponse {
        jsonrpc: "2.0",
        id,
        result: None,
        error: Some(RpcError {
            code,
            message: message.to_string(),
            data: Some(data),
        }),
    }
}

async fn writeln_json(response: &RpcResponse) -> Result<()> {
    let json = serde_json::to_string(response)?;
    use tokio::io::AsyncWriteExt;
    let mut stdout = tokio::io::stdout();
    stdout.write_all(json.as_bytes()).await?;
    stdout.write_all(b"\n").await?;
    stdout.flush().await?;
    Ok(())
}

/// Send a JSON-RPC notification to the client (no id, no response expected).
async fn send_notification(method: &str, params: Value) {
    let notif = RpcNotification {
        jsonrpc: "2.0",
        method: method.to_string(),
        params: Some(params),
    };
    if let Ok(json) = serde_json::to_string(&notif) {
        use tokio::io::AsyncWriteExt;
        let mut stdout = tokio::io::stdout();
        let _ = stdout.write_all(json.as_bytes()).await;
        let _ = stdout.write_all(b"\n").await;
        let _ = stdout.flush().await;
    }
}

/// Send a logging notification to the client.
async fn log_notification(level: &str, message: &str) {
    send_notification("notifications/message", serde_json::json!({
        "level": level,
        "logger": "gsea",
        "data": message
    })).await;
}

// ─── Notification handlers ────────────────────────────────────

async fn handle_notification(req: &RpcRequest) {
    match req.method.as_str() {
        "notifications/initialized" => {
            tracing::info!("MCP client initialized");
            log_notification("info", "Client connected and initialized").await;
        }
        "notifications/cancelled" => {
            if let Some(params) = &req.params {
                let request_id = params.get("requestId").cloned().unwrap_or(Value::Null);
                tracing::info!("Client cancelled request: {:?}", request_id);
            }
        }
        _ => {
            tracing::debug!("Unhandled notification: {}", req.method);
        }
    }
}

// ─── Request handlers ──────────────────────────────────────────

async fn handle_request(
    req: &RpcRequest,
    tools: &Arc<std::sync::Mutex<ToolRegistry>>,
    brain: &Arc<std::sync::Mutex<Brain>>,
    agent: &Option<Arc<tokio::sync::Mutex<Agent>>>,
    llm: &Option<OllamaClient>,
    subs: &Arc<std::sync::Mutex<ResourceSubscriptions>>,
) -> RpcResponse {
    match req.method.as_str() {
        "initialize" => handle_initialize(req),
        "initialized" => rpc_result(req.id.clone(), serde_json::json!({})),
        "tools/list" => handle_tools_list(req, tools),
        "tools/call" => handle_tools_call(req, tools, brain, agent, llm).await,
        "resources/list" => handle_resources_list(req, brain),
        "resources/read" => handle_resources_read(req, brain),
        "resources/subscribe" => handle_resources_subscribe(req, subs),
        "resources/unsubscribe" => handle_resources_unsubscribe(req, subs),
        "resources/templates/list" => handle_resource_templates(req),
        "prompts/list" => handle_prompts_list(req),
        "prompts/get" => handle_prompts_get(req, brain),
        "logging/setLevel" => handle_set_log_level(req),
        "ping" => rpc_result(req.id.clone(), serde_json::json!({})),
        "shutdown" | "exit" => {
            let _ = writeln_json(&rpc_result(req.id.clone(), serde_json::json!({}))).await;
            std::process::exit(0);
        }
        _ => rpc_error(req.id.clone(), -32601, &format!("Method not found: {}", req.method)),
    }
}

fn handle_initialize(req: &RpcRequest) -> RpcResponse {
    rpc_result(req.id.clone(), serde_json::json!({
        "protocolVersion": "2024-11-05",
        "serverInfo": {
            "name": "gsea",
            "version": env!("CARGO_PKG_VERSION")
        },
        "capabilities": {
            "tools": { "listChanged": true },
            "resources": { "subscribe": true, "listChanged": true },
            "prompts": { "listChanged": true },
            "logging": {}
        }
    }))
}

fn handle_set_log_level(req: &RpcRequest) -> RpcResponse {
    // Accept the log level but we use tracing internally
    rpc_result(req.id.clone(), serde_json::json!({}))
}

// ─── Tools ─────────────────────────────────────────────────────

fn handle_tools_list(req: &RpcRequest, tools: &Arc<std::sync::Mutex<ToolRegistry>>) -> RpcResponse {
    let reg = tools.lock().unwrap();
    let mut tool_list: Vec<Value> = reg
        .tool_specs()
        .into_iter()
        .map(|spec| {
            serde_json::json!({
                "name": spec.function.name,
                "description": spec.function.description,
                "inputSchema": spec.function.parameters,
            })
        })
        .collect();

    // Add the built-in chat tool (requires agent)
    tool_list.push(serde_json::json!({
        "name": "chat",
        "description": "Send a message to the GSEA agent and get a response. Use this for complex reasoning, code generation, or any task requiring the full agent loop.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to send to the agent"
                }
            },
            "required": ["message"]
        }
    }));

    // Add the query tool (simple LLM call without agent loop)
    tool_list.push(serde_json::json!({
        "name": "query",
        "description": "Send a simple query to the LLM without the full agent loop. Faster but no tool use or memory recall.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The prompt to send"
                },
                "system": {
                    "type": "string",
                    "description": "Optional system prompt override"
                }
            },
            "required": ["prompt"]
        }
    }));

    // Add multi-agent tools
    tool_list.push(serde_json::json!({
        "name": "agent_health",
        "description": "Check the health status of all GSEA agents. Returns status (healthy/degraded/unhealthy) for each agent.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        }
    }));

    tool_list.push(serde_json::json!({
        "name": "brain_search",
        "description": "Search the GSEA MemoryBrain for stored knowledge, agent results, and learned skills.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for memory recall"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 10)"
                }
            },
            "required": ["query"]
        }
    }));

    tool_list.push(serde_json::json!({
        "name": "store_memory",
        "description": "Store a piece of information in the GSEA MemoryBrain for later recall.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content to store"
                },
                "memory_type": {
                    "type": "string",
                    "description": "Type of memory: semantic, episodic, or procedural",
                    "enum": ["semantic", "episodic", "procedural"]
                }
            },
            "required": ["content"]
        }
    }));

    rpc_result(req.id.clone(), serde_json::json!({ "tools": tool_list }))
}

async fn handle_tools_call(
    req: &RpcRequest,
    tools: &Arc<std::sync::Mutex<ToolRegistry>>,
    brain: &Arc<std::sync::Mutex<Brain>>,
    agent: &Option<Arc<tokio::sync::Mutex<Agent>>>,
    llm: &Option<OllamaClient>,
) -> RpcResponse {
    let params = match &req.params {
        Some(p) => p,
        None => return rpc_error(req.id.clone(), -32602, "Missing params"),
    };

    let tool_name = match params.get("name").and_then(|v| v.as_str()) {
        Some(n) => n,
        None => return rpc_error(req.id.clone(), -32602, "Missing tool name"),
    };

    let arguments = params.get("arguments").cloned().unwrap_or(serde_json::json!({}));

    // Built-in tools: chat, query, and multi-agent
    match tool_name {
        "chat" => return handle_chat_tool(req, &arguments, agent).await,
        "query" => return handle_query_tool(req, &arguments, llm).await,
        "agent_health" => return handle_agent_health_tool(req, brain),
        "brain_search" => return handle_brain_search_tool(req, &arguments, brain),
        "store_memory" => return handle_store_memory_tool(req, &arguments, brain),
        _ => {}
    }

    // Execute registry tool with timeout
    let result = {
        let reg = tools.lock().unwrap();
        match reg.get(tool_name) {
            Some(tool) => {
                log_notification("info", &format!("Executing tool: {}", tool_name)).await;

                let timeout = tokio::time::timeout(
                    std::time::Duration::from_secs(TOOL_TIMEOUT_SECS),
                    tool.execute(arguments),
                ).await;

                match timeout {
                    Ok(Ok(val)) => tool_result(req.id.clone(), &val, false),
                    Ok(Err(e)) => tool_result_error(req.id.clone(), &format!("Tool error: {}", e)),
                    Err(_) => tool_result_error(
                        req.id.clone(),
                        &format!("Tool '{}' timed out after {}s", tool_name, TOOL_TIMEOUT_SECS),
                    ),
                }
            }
            None => rpc_error_with_data(
                req.id.clone(),
                -32602,
                &format!("Unknown tool: {}", tool_name),
                serde_json::json!({ "available_tools": list_tool_names(&reg) }),
            ),
        }
    };

    result
}

/// Handle the built-in `chat` tool — sends a message through the full agent loop.
async fn handle_chat_tool(
    req: &RpcRequest,
    arguments: &Value,
    agent: &Option<Arc<tokio::sync::Mutex<Agent>>>,
) -> RpcResponse {
    let message = match arguments.get("message").and_then(|v| v.as_str()) {
        Some(m) => m,
        None => return rpc_error(req.id.clone(), -32602, "Missing 'message' argument"),
    };

    let agent = match agent {
        Some(a) => a,
        None => return tool_result_error(
            req.id.clone(),
            "Agent not available in MCP server mode. Start with --interactive or pekko mode to enable chat.",
        ),
    };

    log_notification("info", &format!("Chat: {}", &message[..message.len().min(100)])).await;

    let timeout = tokio::time::timeout(
        std::time::Duration::from_secs(120),
        async {
            let mut ag = agent.lock().await;
            ag.process_message(message).await
        },
    ).await;

    match timeout {
        Ok(Ok(response)) => tool_result(req.id.clone(), &serde_json::json!(response), false),
        Ok(Err(e)) => tool_result_error(req.id.clone(), &format!("Agent error: {}", e)),
        Err(_) => tool_result_error(req.id.clone(), "Agent chat timed out after 120s"),
    }
}

/// Handle the built-in `query` tool — simple LLM call without agent loop.
async fn handle_query_tool(
    req: &RpcRequest,
    arguments: &Value,
    llm: &Option<OllamaClient>,
) -> RpcResponse {
    let prompt = match arguments.get("prompt").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => return rpc_error(req.id.clone(), -32602, "Missing 'prompt' argument"),
    };

    let llm = match llm {
        Some(l) => l,
        None => return tool_result_error(req.id.clone(), "LLM not available"),
    };

    let system = arguments.get("system").and_then(|v| v.as_str()).unwrap_or(
        "You are GSEA, a self-evolving Rust engineering agent. Respond concisely and accurately."
    );

    let messages = vec![
        crate::llm::Message { role: "system".into(), content: system.to_string() },
        crate::llm::Message { role: "user".into(), content: prompt.to_string() },
    ];

    let timeout = tokio::time::timeout(
        std::time::Duration::from_secs(60),
        llm.chat(messages),
    ).await;

    match timeout {
        Ok(Ok(response)) => tool_result(req.id.clone(), &serde_json::json!(response.content), false),
        Ok(Err(e)) => tool_result_error(req.id.clone(), &format!("LLM error: {}", e)),
        Err(_) => tool_result_error(req.id.clone(), "LLM query timed out after 60s"),
    }
}

/// Handle the `agent_health` tool — returns brain stats as a health proxy.
fn handle_agent_health_tool(
    req: &RpcRequest,
    brain: &Arc<std::sync::Mutex<Brain>>,
) -> RpcResponse {
    let b = brain.lock().unwrap();
    let stats = b.stats();
    let skills = b.list_skills();
    let recent = b.recall("", 5);

    let result = serde_json::json!({
        "brain_stats": stats,
        "skills_count": skills.len(),
        "recent_memories": recent.len(),
        "status": "operational",
    });

    tool_result(req.id.clone(), &result, false)
}

/// Handle the `brain_search` tool — search memories.
fn handle_brain_search_tool(
    req: &RpcRequest,
    arguments: &Value,
    brain: &Arc<std::sync::Mutex<Brain>>,
) -> RpcResponse {
    let query = match arguments.get("query").and_then(|v| v.as_str()) {
        Some(q) => q,
        None => return rpc_error(req.id.clone(), -32602, "Missing 'query' argument"),
    };
    let limit = arguments.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

    let b = brain.lock().unwrap();
    let results = b.recall(query, limit);
    let items: Vec<Value> = results.iter().map(|item| {
        serde_json::json!({
            "id": item.id.to_string(),
            "content": item.content,
            "type": format!("{}", item.memory_type),
            "strength": item.strength,
            "created_at": item.created_at.to_string(),
        })
    }).collect();

    tool_result(req.id.clone(), &serde_json::json!({
        "results": items,
        "count": items.len(),
        "query": query,
    }), false)
}

/// Handle the `store_memory` tool — store content in brain.
fn handle_store_memory_tool(
    req: &RpcRequest,
    arguments: &Value,
    brain: &Arc<std::sync::Mutex<Brain>>,
) -> RpcResponse {
    let content = match arguments.get("content").and_then(|v| v.as_str()) {
        Some(c) => c,
        None => return rpc_error(req.id.clone(), -32602, "Missing 'content' argument"),
    };

    let mtype = match arguments.get("memory_type").and_then(|v| v.as_str()) {
        Some("episodic") => crate::memory_brain::MemoryType::Episodic,
        Some("procedural") => crate::memory_brain::MemoryType::Procedural,
        _ => crate::memory_brain::MemoryType::Semantic,
    };

    let mut b = brain.lock().unwrap();
    match b.process(content, Some(mtype)) {
        Ok(id) => tool_result(req.id.clone(), &serde_json::json!({
            "id": id.to_string(),
            "status": "stored",
        }), false),
        Err(e) => tool_result_error(req.id.clone(), &format!("Store failed: {}", e)),
    }
}

/// Build a successful tool result with proper MCP content format.
fn tool_result(id: Option<Value>, value: &Value, is_error: bool) -> RpcResponse {
    let text = match value {
        Value::String(s) => s.clone(),
        _ => serde_json::to_string_pretty(value).unwrap_or_default(),
    };
    rpc_result(id, serde_json::json!({
        "content": [{ "type": "text", "text": text }],
        "isError": is_error
    }))
}

/// Build a tool error result (isError: true).
fn tool_result_error(id: Option<Value>, message: &str) -> RpcResponse {
    rpc_result(id, serde_json::json!({
        "content": [{ "type": "text", "text": message }],
        "isError": true
    }))
}

/// List tool names for error diagnostics.
fn list_tool_names(reg: &ToolRegistry) -> Vec<String> {
    reg.list_tools().iter().map(|t| t.name().to_string()).collect()
}

// ─── Resources ─────────────────────────────────────────────────

fn handle_resources_list(req: &RpcRequest, brain: &Arc<std::sync::Mutex<Brain>>) -> RpcResponse {
    let _b = brain.lock().unwrap();
    rpc_result(req.id.clone(), serde_json::json!({
        "resources": [
            {
                "uri": "memory://brain/stats",
                "name": "Brain Statistics",
                "description": "Memory counts by type (episodic, semantic, procedural, reflection)",
                "mimeType": "application/json"
            },
            {
                "uri": "memory://skills",
                "name": "Learned Skills",
                "description": "All Rust skills stored in procedural memory",
                "mimeType": "text/plain"
            },
            {
                "uri": "memory://brain/recent",
                "name": "Recent Memories",
                "description": "Last 10 episodic memories",
                "mimeType": "application/json"
            }
        ]
    }))
}

fn handle_resource_templates(req: &RpcRequest) -> RpcResponse {
    rpc_result(req.id.clone(), serde_json::json!({
        "resourceTemplates": [
            {
                "uriTemplate": "memory://brain/search/{query}",
                "name": "Search Memories",
                "description": "Search brain memories by keyword",
                "mimeType": "application/json"
            },
            {
                "uriTemplate": "memory://agent/{name}/stats",
                "name": "Agent Stats",
                "description": "Get performance stats for a specific agent",
                "mimeType": "application/json"
            },
            {
                "uriTemplate": "memory://agent/{name}/skills",
                "name": "Agent Skills",
                "description": "List skills learned by a specific agent",
                "mimeType": "application/json"
            },
            {
                "uriTemplate": "memory://brain/type/{memory_type}",
                "name": "Memories by Type",
                "description": "List memories of a specific type (episodic, semantic, procedural)",
                "mimeType": "application/json"
            }
        ]
    }))
}

fn handle_resources_read(req: &RpcRequest, brain: &Arc<std::sync::Mutex<Brain>>) -> RpcResponse {
    let params = match &req.params {
        Some(p) => p,
        None => return rpc_error(req.id.clone(), -32602, "Missing params"),
    };

    let uri = match params.get("uri").and_then(|v| v.as_str()) {
        Some(u) => u,
        None => return rpc_error(req.id.clone(), -32602, "Missing uri"),
    };

    let b = brain.lock().unwrap();
    match uri {
        "memory://brain/stats" => {
            let stats = b.stats();
            rpc_result(req.id.clone(), serde_json::json!({
                "contents": [{
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": serde_json::to_string_pretty(&stats).unwrap_or_default()
                }]
            }))
        }
        "memory://skills" => {
            let skills = b.list_skills();
            let text: String = skills.iter()
                .map(|(name, desc)| format!("- {}: {}", name, desc))
                .collect::<Vec<_>>()
                .join("\n");
            rpc_result(req.id.clone(), serde_json::json!({
                "contents": [{
                    "uri": uri,
                    "mimeType": "text/plain",
                    "text": if text.is_empty() { "No skills learned yet.".to_string() } else { text }
                }]
            }))
        }
        "memory://brain/recent" => {
            let recent = b.recall("", 10);
            let items: Vec<Value> = recent.iter().map(|item| {
                serde_json::json!({
                    "id": item.id,
                    "type": format!("{}", item.memory_type),
                    "content": item.content.chars().take(200).collect::<String>(),
                    "strength": item.strength,
                    "created": item.created_at.to_string()
                })
            }).collect();
            rpc_result(req.id.clone(), serde_json::json!({
                "contents": [{
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": serde_json::to_string_pretty(&items).unwrap_or_default()
                }]
            }))
        }
        _ if uri.starts_with("memory://brain/search/") => {
            let query = uri.trim_start_matches("memory://brain/search/");
            let query = urlencoding::decode(query).unwrap_or_else(|_| query.into());
            let results = b.recall(&query, 10);
            let items: Vec<Value> = results.iter().map(|item| {
                serde_json::json!({
                    "id": item.id,
                    "type": format!("{}", item.memory_type),
                    "content": item.content.chars().take(300).collect::<String>(),
                    "strength": item.strength,
                })
            }).collect();
            rpc_result(req.id.clone(), serde_json::json!({
                "contents": [{
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": serde_json::to_string_pretty(&items).unwrap_or_default()
                }]
            }))
        }
        _ if uri.starts_with("memory://agent/") && uri.ends_with("/stats") => {
            let name = uri.trim_start_matches("memory://agent/").trim_end_matches("/stats");
            // Return agent results from brain as stats proxy
            let results = b.recall_agent_results(name, 5);
            let items: Vec<Value> = results.iter().map(|item| {
                serde_json::json!({
                    "id": item.id,
                    "content": item.content.chars().take(200).collect::<String>(),
                    "strength": item.strength,
                })
            }).collect();
            rpc_result(req.id.clone(), serde_json::json!({
                "contents": [{
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": serde_json::to_string_pretty(&serde_json::json!({
                        "agent": name,
                        "result_count": items.len(),
                        "results": items,
                    })).unwrap_or_default()
                }]
            }))
        }
        _ if uri.starts_with("memory://agent/") && uri.ends_with("/skills") => {
            let name = uri.trim_start_matches("memory://agent/").trim_end_matches("/skills");
            let skills = b.list_skills();
            let agent_skills: Vec<_> = skills.iter()
                .filter(|(_, desc)| desc.to_lowercase().contains(&name.to_lowercase()))
                .collect();
            let text = if agent_skills.is_empty() {
                format!("No skills specifically for agent '{}'.\nAll skills:\n{}", name,
                    skills.iter().map(|(n, d)| format!("- {}: {}", n, d)).collect::<Vec<_>>().join("\n"))
            } else {
                agent_skills.iter().map(|(n, d)| format!("- {}: {}", n, d)).collect::<Vec<_>>().join("\n")
            };
            rpc_result(req.id.clone(), serde_json::json!({
                "contents": [{
                    "uri": uri,
                    "mimeType": "text/plain",
                    "text": text
                }]
            }))
        }
        _ if uri.starts_with("memory://brain/type/") => {
            let memory_type = uri.trim_start_matches("memory://brain/type/");
            let results = b.recall(memory_type, 20);
            let items: Vec<Value> = results.iter()
                .filter(|item| format!("{}", item.memory_type).to_lowercase() == memory_type.to_lowercase())
                .take(10)
                .map(|item| {
                    serde_json::json!({
                        "id": item.id,
                        "type": format!("{}", item.memory_type),
                        "content": item.content.chars().take(200).collect::<String>(),
                        "strength": item.strength,
                    })
                }).collect();
            rpc_result(req.id.clone(), serde_json::json!({
                "contents": [{
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": serde_json::to_string_pretty(&items).unwrap_or_default()
                }]
            }))
        }
        _ => rpc_error(req.id.clone(), -32602, &format!("Resource not found: {}", uri)),
    }
}

// ─── Prompts ───────────────────────────────────────────────────

fn handle_prompts_list(req: &RpcRequest) -> RpcResponse {
    rpc_result(req.id.clone(), serde_json::json!({
        "prompts": [
            {
                "name": "recall",
                "description": "Search memory for relevant information",
                "arguments": [
                    { "name": "query", "description": "Search query", "required": true }
                ]
            },
            {
                "name": "code_review",
                "description": "Review code changes against a git ref",
                "arguments": [
                    { "name": "rev", "description": "Git ref (default: HEAD~1)", "required": false }
                ]
            },
            {
                "name": "reflect",
                "description": "Run a self-evolution reflection cycle to identify improvements",
                "arguments": []
            },
            {
                "name": "debug_session",
                "description": "Start a debugging session with recent errors and context from memory",
                "arguments": [
                    { "name": "component", "description": "Component or module to focus on", "required": false }
                ]
            },
            {
                "name": "architecture_review",
                "description": "Review system architecture using skills and patterns from memory",
                "arguments": [
                    { "name": "focus", "description": "Area to focus on (e.g. 'performance', 'security')", "required": false }
                ]
            },
            {
                "name": "daily_summary",
                "description": "Generate a summary of recent agent activity and learnings",
                "arguments": []
            }
        ]
    }))
}

fn handle_prompts_get(
    req: &RpcRequest,
    brain: &Arc<std::sync::Mutex<Brain>>,
) -> RpcResponse {
    let params = match &req.params {
        Some(p) => p,
        None => return rpc_error(req.id.clone(), -32602, "Missing params"),
    };

    let name = match params.get("name").and_then(|v| v.as_str()) {
        Some(n) => n,
        None => return rpc_error(req.id.clone(), -32602, "Missing prompt name"),
    };

    let args = params.get("arguments").cloned().unwrap_or(serde_json::json!({}));

    match name {
        "recall" => {
            let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("general");
            rpc_result(req.id.clone(), serde_json::json!({
                "messages": [{
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": format!("Search your MemoryBrain for information about: {}. If found, summarize what you know.", query)
                    }
                }]
            }))
        }
        "code_review" => {
            let rev = args.get("rev").and_then(|v| v.as_str()).unwrap_or("HEAD~1");
            rpc_result(req.id.clone(), serde_json::json!({
                "messages": [{
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": format!("Review the git diff against {} and provide: 1) Summary 2) Issues 3) Suggestions", rev)
                    }
                }]
            }))
        }
        "reflect" => {
            rpc_result(req.id.clone(), serde_json::json!({
                "messages": [{
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": "Run a self-evolution reflection cycle. Review recent activity, identify patterns (good and bad), suggest improvements, and save any useful skills or learnings."
                    }
                }]
            }))
        }
        "debug_session" => {
            let component = args.get("component").and_then(|v| v.as_str()).unwrap_or("");
            // Pull recent memories for error context
            let recent_context = if let Ok(brain) = brain.lock() {
                let query = if component.is_empty() { "error failure bug" } else { component };
                let memories = brain.recall(query, 5);
                if !memories.is_empty() {
                    let ctx = memories.iter()
                        .map(|m| {
                            let c = &m.content;
                            format!("- {}", &c[..c.len().min(200)])
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    format!("\n\nRelevant memories from brain:\n{}", ctx)
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            let focus = if component.is_empty() {
                String::new()
            } else {
                format!(" Focus on the '{}' component.", component)
            };

            rpc_result(req.id.clone(), serde_json::json!({
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": format!(
                                "Start a debugging session.{}{}\n\n\
                                 Steps:\n\
                                 1. Identify the error or unexpected behavior\n\
                                 2. Check recent changes and memory for related patterns\n\
                                 3. Form a hypothesis\n\
                                 4. Suggest a fix with test coverage",
                                focus, recent_context
                            )
                        }
                    }
                ]
            }))
        }
        "architecture_review" => {
            let focus = args.get("focus").and_then(|v| v.as_str()).unwrap_or("general");
            // Pull skills and patterns from brain
            let skills_context = if let Ok(brain) = brain.lock() {
                let skills = brain.list_skills();
                if skills.is_empty() {
                    String::new()
                } else {
                    let list = skills.iter()
                        .take(10)
                        .map(|(name, desc)| format!("- {}: {}", name, desc))
                        .collect::<Vec<_>>()
                        .join("\n");
                    format!("\n\nKnown skills/patterns:\n{}", list)
                }
            } else {
                String::new()
            };

            rpc_result(req.id.clone(), serde_json::json!({
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": format!(
                                "Conduct an architecture review with focus on: {}.{}\n\n\
                                 Evaluate:\n\
                                 1. Component boundaries and responsibilities\n\
                                 2. Data flow and coupling\n\
                                 3. Error handling and resilience patterns\n\
                                 4. Scalability considerations\n\
                                 5. Recommendations for improvement",
                                focus, skills_context
                            )
                        }
                    }
                ]
            }))
        }
        "daily_summary" => {
            // Pull recent activity from brain
            let activity_context = if let Ok(brain) = brain.lock() {
                let memories = brain.recall("recent activity summary", 10);
                if !memories.is_empty() {
                    let ctx = memories.iter()
                        .map(|m| {
                            let c = &m.content;
                            format!("- {}", &c[..c.len().min(150)])
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    format!("\n\nRecent activity from memory:\n{}", ctx)
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            rpc_result(req.id.clone(), serde_json::json!({
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": format!(
                                "Generate a daily summary report.{}\n\n\
                                 Include:\n\
                                 1. Tasks completed and key outcomes\n\
                                 2. Skills learned or improved\n\
                                 3. Notable patterns or recurring issues\n\
                                 4. Recommendations for tomorrow",
                                activity_context
                            )
                        }
                    }
                ]
            }))
        }
        _ => rpc_error(req.id.clone(), -32602, &format!("Prompt not found: {}", name)),
    }
}

// ─── Resource Subscriptions ───────────────────────────────────

fn handle_resources_subscribe(
    req: &RpcRequest,
    subs: &Arc<std::sync::Mutex<ResourceSubscriptions>>,
) -> RpcResponse {
    let params = match &req.params {
        Some(p) => p,
        None => return rpc_error(req.id.clone(), -32602, "Missing params"),
    };

    let uri = match params.get("uri").and_then(|v| v.as_str()) {
        Some(u) => u,
        None => return rpc_error(req.id.clone(), -32602, "Missing uri"),
    };

    // Validate it's a known resource URI
    let valid_uris = ["memory://brain/stats", "memory://skills", "memory://brain/recent"];
    let is_valid = valid_uris.contains(&uri) || uri.starts_with("memory://brain/search/");

    if !is_valid {
        return rpc_error(req.id.clone(), -32602, &format!("Unknown resource: {}", uri));
    }

    subs.lock().unwrap().subscribe(uri);
    tracing::info!("Resource subscribed: {}", uri);
    rpc_result(req.id.clone(), serde_json::json!({}))
}

fn handle_resources_unsubscribe(
    req: &RpcRequest,
    subs: &Arc<std::sync::Mutex<ResourceSubscriptions>>,
) -> RpcResponse {
    let params = match &req.params {
        Some(p) => p,
        None => return rpc_error(req.id.clone(), -32602, "Missing params"),
    };

    let uri = match params.get("uri").and_then(|v| v.as_str()) {
        Some(u) => u,
        None => return rpc_error(req.id.clone(), -32602, "Missing uri"),
    };

    subs.lock().unwrap().unsubscribe(uri);
    tracing::info!("Resource unsubscribed: {}", uri);
    rpc_result(req.id.clone(), serde_json::json!({}))
}

/// Periodically check subscribed resources for changes and send notifications.
async fn check_resource_changes(
    brain: &Arc<std::sync::Mutex<Brain>>,
    subs: &Arc<std::sync::Mutex<ResourceSubscriptions>>,
) {
    let uris = subs.lock().unwrap().subscribed_uris();
    if uris.is_empty() {
        return;
    }

    // Collect changed URIs while holding locks, then notify without locks
    let changed_uris: Vec<String> = {
        let b = brain.lock().unwrap();
        let mut changed = Vec::new();

        for uri in &uris {
            let current_value = match uri.as_str() {
                "memory://brain/stats" => {
                    serde_json::to_string(&b.stats()).unwrap_or_default()
                }
                "memory://skills" => {
                    let skills = b.list_skills();
                    skills.iter().map(|(n, d)| format!("{}:{}", n, d)).collect::<Vec<_>>().join("|")
                }
                "memory://brain/recent" => {
                    let recent = b.recall("", 10);
                    recent.iter().map(|m| m.id.to_string()).collect::<Vec<_>>().join(",")
                }
                _ => continue,
            };

            if subs.lock().unwrap().check_changed(uri, &current_value) {
                changed.push(uri.clone());
            }
        }
        changed
    }; // brain lock dropped here

    // Send notifications without holding any locks
    for uri in &changed_uris {
        send_notification("notifications/resources/updated", serde_json::json!({
            "uri": uri,
        })).await;
    }
}

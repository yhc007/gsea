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

/// Run the MCP server: reads JSON-RPC requests from stdin, writes responses to stdout.
pub async fn run_mcp_server(
    tools: Arc<std::sync::Mutex<ToolRegistry>>,
    brain: Arc<std::sync::Mutex<Brain>>,
    agent: Option<Arc<tokio::sync::Mutex<Agent>>>,
    llm: Option<OllamaClient>,
) -> Result<()> {
    let mut line_buf = String::new();
    let mut stdin = tokio::io::BufReader::new(tokio::io::stdin());

    // Log server startup
    log_notification("info", "GSEA MCP server starting").await;

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

        let response = handle_request(&request, &tools, &brain, &agent, &llm).await;
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
) -> RpcResponse {
    match req.method.as_str() {
        "initialize" => handle_initialize(req),
        "initialized" => rpc_result(req.id.clone(), serde_json::json!({})),
        "tools/list" => handle_tools_list(req, tools),
        "tools/call" => handle_tools_call(req, tools, brain, agent, llm).await,
        "resources/list" => handle_resources_list(req, brain),
        "resources/read" => handle_resources_read(req, brain),
        "resources/templates/list" => handle_resource_templates(req),
        "prompts/list" => handle_prompts_list(req),
        "prompts/get" => handle_prompts_get(req),
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

    rpc_result(req.id.clone(), serde_json::json!({ "tools": tool_list }))
}

async fn handle_tools_call(
    req: &RpcRequest,
    tools: &Arc<std::sync::Mutex<ToolRegistry>>,
    _brain: &Arc<std::sync::Mutex<Brain>>,
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

    // Built-in tools: chat and query
    match tool_name {
        "chat" => return handle_chat_tool(req, &arguments, agent).await,
        "query" => return handle_query_tool(req, &arguments, llm).await,
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
            }
        ]
    }))
}

fn handle_prompts_get(req: &RpcRequest) -> RpcResponse {
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
        _ => rpc_error(req.id.clone(), -32602, &format!("Prompt not found: {}", name)),
    }
}

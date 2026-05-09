//! MCP (Model Context Protocol) server — exposes GSEA tools, memory, and prompts
//! to MCP clients (Claude Desktop, Cursor, Zed, etc.) via stdio transport.
//!
//! Protocol: JSON-RPC 2.0 over stdin/stdout
//! Spec: https://spec.modelcontextprotocol.io/

use std::sync::Arc;

use anyhow::Result;
use serde_json::Value;

use crate::memory_brain::Brain;
use crate::tools::ToolRegistry;

/// Run the MCP server: reads JSON-RPC requests from stdin, writes responses to stdout.
pub async fn run_mcp_server(
    tools: Arc<std::sync::Mutex<ToolRegistry>>,
    brain: Arc<std::sync::Mutex<Brain>>,
) -> Result<()> {
    let mut line_buf = String::new();
    let mut stdin = tokio::io::BufReader::new(tokio::io::stdin());

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

        let response = handle_request(&request, &tools, &brain).await;
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

async fn writeln_json(response: &RpcResponse) -> Result<()> {
    let json = serde_json::to_string(response)?;
    use tokio::io::AsyncWriteExt;
    let mut stdout = tokio::io::stdout();
    stdout.write_all(json.as_bytes()).await?;
    stdout.write_all(b"\n").await?;
    stdout.flush().await?;
    Ok(())
}

// ─── Request handlers ──────────────────────────────────────────

async fn handle_request(
    req: &RpcRequest,
    tools: &Arc<std::sync::Mutex<ToolRegistry>>,
    brain: &Arc<std::sync::Mutex<Brain>>,
) -> RpcResponse {
    match req.method.as_str() {
        "initialize" => handle_initialize(req),
        "initialized" => rpc_result(req.id.clone(), serde_json::json!({})),
        "tools/list" => handle_tools_list(req, tools),
        "tools/call" => handle_tools_call(req, tools, brain).await,
        "resources/list" => handle_resources_list(req, brain),
        "resources/read" => handle_resources_read(req, brain),
        "prompts/list" => handle_prompts_list(req),
        "prompts/get" => handle_prompts_get(req),
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
            "tools": {},
            "resources": {},
            "prompts": {}
        }
    }))
}

// ─── Tools ─────────────────────────────────────────────────────

fn handle_tools_list(req: &RpcRequest, tools: &Arc<std::sync::Mutex<ToolRegistry>>) -> RpcResponse {
    let reg = tools.lock().unwrap();
    let tool_list: Vec<Value> = reg
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

    rpc_result(req.id.clone(), serde_json::json!({ "tools": tool_list }))
}

async fn handle_tools_call(
    req: &RpcRequest,
    tools: &Arc<std::sync::Mutex<ToolRegistry>>,
    _brain: &Arc<std::sync::Mutex<Brain>>,
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

    // Execute the tool while holding the registry lock.
    // This is safe because the async execution takes &self (shared ref).
    let result = {
        let reg = tools.lock().unwrap();
        match reg.get(tool_name) {
            Some(tool) => {
                let output = tool.execute(arguments).await;
                match output {
                    Ok(val) => rpc_result(req.id.clone(), serde_json::json!({
                        "content": [{ "type": "text", "text": serde_json::to_string_pretty(&val).unwrap_or_default() }]
                    })),
                    Err(e) => rpc_error(req.id.clone(), -32603, &format!("Tool error: {}", e)),
                }
            }
            None => rpc_error(req.id.clone(), -32602, &format!("Unknown tool: {}", tool_name)),
        }
    };

    result
}

// ─── Resources ─────────────────────────────────────────────────

fn handle_resources_list(req: &RpcRequest, brain: &Arc<std::sync::Mutex<Brain>>) -> RpcResponse {
    let _b = brain.lock().unwrap();
    rpc_result(req.id.clone(), serde_json::json!({
        "resources": [
            {
                "uri": "memory://brain/stats",
                "name": "Brain Statistics",
                "description": "Memory counts by type",
                "mimeType": "application/json"
            },
            {
                "uri": "memory://skills",
                "name": "Learned Skills",
                "description": "All Rust skills stored in procedural memory",
                "mimeType": "text/plain"
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

    match name {
        "recall" => rpc_result(req.id.clone(), serde_json::json!({
            "messages": [{
                "role": "user",
                "content": "Search your MemoryBrain for information about: {{query}}. If found, summarize what you know."
            }]
        })),
        "code_review" => rpc_result(req.id.clone(), serde_json::json!({
            "messages": [{
                "role": "user",
                "content": "Review the git diff against {{rev}} and provide: 1) Summary 2) Issues 3) Suggestions"
            }]
        })),
        _ => rpc_error(req.id.clone(), -32602, &format!("Prompt not found: {}", name)),
    }
}

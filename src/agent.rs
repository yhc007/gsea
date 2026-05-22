use anyhow::Result;
use serde_json::Value;
use std::sync::Arc;


use crate::llm::{embedding::EmbeddingEngine, Message, OllamaClient};
use crate::memory_brain::Brain;
use crate::tools::{ToolRegistry, skill_tools};

/// The core agent loop. Manages conversation with Gemma, tool execution,
/// and memory logging.
pub struct Agent {
    llm: OllamaClient,
    fast_llm: OllamaClient,
    brain: Arc<std::sync::Mutex<Brain>>,
    pub tools: Arc<std::sync::Mutex<ToolRegistry>>,
    embedder: Arc<dyn EmbeddingEngine>,
    session_id: String,
    /// User-facing conversation history (uses llm)
    messages: Vec<Message>,
    /// Fast model conversation history (uses fast_llm, for evolution)
    fast_messages: Vec<Message>,
}

impl Agent {
    pub fn new(
        llm: OllamaClient,
        fast_llm: OllamaClient,
        brain: Arc<std::sync::Mutex<Brain>>,
        tools: Arc<std::sync::Mutex<ToolRegistry>>,
        embedder: Arc<dyn EmbeddingEngine>,
    ) -> Self {
        let session_id = uuid::Uuid::new_v4().to_string();

        // Register any stored skills as dynamic tools
        {
            let mut reg = tools.lock().unwrap();
            skill_tools::register_skills(&mut reg, &brain);
        }

        let system_prompt = {
            let b = brain.lock().unwrap();
            let reg = tools.lock().unwrap();
            Self::build_system_prompt(&reg, &b)
        };

        let fast_system = String::from(
            "You are GSEA's fast assistant for evolution and utility tasks. Respond concisely."
        );

        Self {
            llm,
            fast_llm,
            brain,
            tools,
            embedder,
            session_id,
            messages: vec![Message {
                role: "system".to_string(),
                content: system_prompt,
            }],
            fast_messages: vec![Message {
                role: "system".to_string(),
                content: fast_system,
            }],
        }
    }

    /// Save the current conversation history to a JSON file.
    pub fn save_session(&self, path: &str) -> Result<()> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(&self.messages)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load conversation history from a JSON file (replaces current messages).
    pub fn load_session(&mut self, path: &str) -> Result<()> {
        let json = std::fs::read_to_string(path)?;
        let messages: Vec<Message> = serde_json::from_str(&json)?;
        self.messages = messages;
        Ok(())
    }

    /// Process a message using the fast model (qwen3:8b).
    /// Used by EvolutionEngine for self-review and code generation.
    pub async fn process_message_fast(&mut self, prompt: &str) -> Result<String> {
        self.fast_messages.push(Message {
            role: "user".to_string(),
            content: prompt.to_string(),
        });

        let response = self
            .fast_llm
            .chat(self.fast_messages.clone())
            .await?;

        self.fast_messages.push(response.clone());
        Ok(response.content)
    }

    fn build_system_prompt(tools: &ToolRegistry, brain: &Brain) -> String {
        let tools_text = tools.tool_description_text();

        let skills = brain.list_skills();
        let skills_text = if skills.is_empty() {
            String::new()
        } else {
            let mut s = String::from("\n\n### Previously Learned Skills\n");
            s.push_str("You have learned the following skills from past evolution cycles. Reference them when relevant:\n");
            for (name, desc) in &skills {
                s.push_str(&format!("- **{}**: {}\n", name, desc));
                if let Some(code) = brain.get_skill_code(name) {
                    s.push_str(&format!("  ```rust\n  {}\n  ```\n", code));
                }
            }
            s
        };

        format!(
            r#"You are GSEA — a self-evolving Rust engineering agent powered by a local LLM.

You have access to a MemoryBrain that stores your experiences, learnings, and skills.
Use it actively:
- When you learn something useful, save it as a memory.
- When you face a problem, recall relevant past experiences.
- When you notice a pattern or improvement opportunity, record a reflection.

Your ultimate goal is to improve your own capabilities over time by:
1. Writing and testing Rust code
2. Saving useful patterns as skills
3. Reflecting on what works and what doesn't
4. Generating and applying improvements to your own codebase

{}{}

When you want to use a tool, respond with a JSON tool call in this format:
```json
{{"tool": "tool_name", "params": {{"key": "value"}}}}
```

The system will execute it and return the result. You can chain multiple tool calls.
When you're done, provide a final response to the user."#,
            tools_text, skills_text
        )
    }

    /// Heuristically select model based on prompt complexity.
    /// Returns true if the main (complex/capable) model should be used.
    fn needs_complex_model(input: &str) -> bool {
        let trimmed = input.trim();

        // Short, simple greetings → fast model
        if trimmed.len() < 80 && !trimmed.contains("```") {
            let simple_words = ["hi", "hello", "hey", "ok", "okay", "yes", "no", "thanks",
                                "bye", "goodbye", "exit", "quit", "help", "/help", "/tools",
                                "/stats", "/reflect"];
            let lower = trimmed.to_lowercase();
            if simple_words.iter().any(|w| lower == *w || lower.starts_with(w)) {
                return false;
            }
        }

        // Contains code blocks or technical keywords → complex model
        let complex_patterns = [
            "```", "fn ", "impl ", "struct ", "enum ", "trait ",
            "rust", "compile", "refactor", "optimize", "review",
            "debug", "analyze", "memory", "async", "unsafe",
            "cargo", "build", "test", "benchmark",
            // Goal/plan decomposition needs the main model
            "goal:", "decompose", "sub-task", "plan",
            // Multi-step reasoning
            "implement", "design", "architect", "integrate",
            "error", "fix", "bug", "issue", "problem",
        ];
        let lower = trimmed.to_lowercase();
        if complex_patterns.iter().any(|p| lower.contains(p)) {
            return true;
        }

        // Longer technical requests → complex model
        if trimmed.len() > 200 {
            return true;
        }

        // Default to fast model for simple conversational queries
        false
    }

    /// Override the system prompt (used for specialized agent roles).
    pub fn set_system_prompt(&mut self, prompt: &str) {
        if let Some(first) = self.messages.first_mut() {
            if first.role == "system" {
                first.content = prompt.to_string();
            }
        }
    }

    /// Check whether an error is a circuit breaker error (open or timeout).
    fn is_circuit_error(e: &anyhow::Error) -> bool {
        let msg = e.to_string();
        msg.starts_with("circuit_open:") || msg.starts_with("circuit_timeout:")
    }

    /// Get a reference to the main LLM client (for GUI model switching).
    pub fn llm(&mut self) -> &mut OllamaClient {
        &mut self.llm
    }

    /// Get a reference to the fast LLM client.
    pub fn fast_llm(&mut self) -> &mut OllamaClient {
        &mut self.fast_llm
    }

    /// Process a message with streaming output. Returns an mpsc Receiver that
    /// yields content chunks as they arrive. Memory recall and tool execution
    /// happen normally; only the final LLM response is streamed.
    ///
    /// If the LLM response contains a tool call, streaming stops and the tool
    /// chain executes non-streaming. The final tool result is sent as a single chunk.
    pub async fn process_message_stream(
        &mut self,
        user_input: &str,
    ) -> Result<tokio::sync::mpsc::Receiver<String>> {
        // 1. Memory recall (same as process_message)
        let memory_context = match self.embedder.embed(user_input).await {
            Ok(query_emb) => {
                let brain = self.brain.lock().unwrap();
                let results = brain.recall_by_similarity(&query_emb, 5, 0.35);
                if !results.is_empty() {
                    let lines: Vec<String> = results
                        .iter()
                        .map(|(item, score)| {
                            format!(
                                "[{}] (sim: {:.2}) {}",
                                item.memory_type, score,
                                item.content.chars().take(200).collect::<String>()
                            )
                        })
                        .collect();
                    format!("\nRelevant memories:\n{}\n", lines.join("\n"))
                } else {
                    String::new()
                }
            }
            Err(_) => {
                let brain = self.brain.lock().unwrap();
                let results = brain.recall(user_input, 5);
                if !results.is_empty() {
                    let lines: Vec<String> = results
                        .iter()
                        .map(|item| {
                            format!(
                                "[{}] (strength: {:.2}) {}",
                                item.memory_type, item.strength,
                                item.content.chars().take(200).collect::<String>()
                            )
                        })
                        .collect();
                    format!("\nRelevant memories:\n{}\n", lines.join("\n"))
                } else {
                    String::new()
                }
            }
        };

        // 2. Build augmented prompt
        let augmented_input = if memory_context.is_empty() {
            user_input.to_string()
        } else {
            format!("{}\n\n---\nContext from MemoryBrain:\n{}", user_input, memory_context)
        };

        self.messages.push(Message {
            role: "user".to_string(),
            content: augmented_input,
        });

        // 3. Stream the LLM response (with circuit breaker fallback)
        let use_main = Self::needs_complex_model(user_input);
        let mut stream_rx = if use_main {
            match self.llm.chat_stream(self.messages.clone()).await {
                Ok(rx) => rx,
                Err(e) if Self::is_circuit_error(&e) => {
                    tracing::warn!("Main model circuit open for stream, falling back: {}", e);
                    self.fast_llm.chat_stream(self.messages.clone()).await?
                }
                Err(e) => return Err(e),
            }
        } else {
            self.fast_llm.chat_stream(self.messages.clone()).await?
        };

        let (tx, rx) = tokio::sync::mpsc::channel::<String>(64);
        let messages = self.messages.clone();
        let brain = self.brain.clone();
        let embedder = self.embedder.clone();
        let user_input_owned = user_input.to_string();

        // Collect the full response while forwarding chunks
        let messages_ref = Arc::clone(&Arc::new(std::sync::Mutex::new(messages)));

        tokio::spawn(async move {
            let mut full_response = String::new();

            while let Some(chunk) = stream_rx.recv().await {
                full_response.push_str(&chunk);
                if tx.send(chunk).await.is_err() {
                    return; // receiver dropped
                }
            }

            // Store conversation in messages
            {
                let mut msgs = messages_ref.lock().unwrap();
                msgs.push(Message {
                    role: "assistant".to_string(),
                    content: full_response.clone(),
                });
            }

            // Store in memory
            let truncated: String = full_response.chars().take(300).collect();
            let content = format!("User: {}\nAssistant: {}", user_input_owned, truncated);
            if let Ok(emb) = embedder.embed(&content).await {
                let mut item = crate::memory_brain::MemoryItem::new(
                    &content,
                    crate::memory_brain::MemoryType::Episodic,
                );
                item.embedding = Some(emb);
                let brain = brain.lock().unwrap();
                let _ = brain.episodic.store(item);
            } else {
                let mut brain = brain.lock().unwrap();
                let _ = brain.process(&content, Some(crate::memory_brain::MemoryType::Episodic));
            }
        });

        Ok(rx)
    }

    /// Send a chat request using the appropriate model based on prompt complexity.
    /// Logs which model was selected.
    async fn chat_with_selected_model(&self, messages: Vec<Message>) -> Result<Message> {
        let last_user = messages.iter().rev().find(|m| m.role == "user");
        let use_main = last_user.map(|m| Self::needs_complex_model(&m.content)).unwrap_or(true);

        if use_main {
            tracing::debug!("Using main model for response");
            self.llm.chat(messages).await
        } else {
            tracing::debug!("Using fast model for response");
            self.fast_llm.chat(messages).await
        }
    }

    /// Process a single user message — the core agent loop.
    pub async fn process_message(&mut self, user_input: &str) -> Result<String> {
        // 1. Recall relevant memories (embedding-based, with keyword fallback)
        let memory_context = match self.embedder.embed(user_input).await {
            Ok(query_emb) => {
                let brain = self.brain.lock().unwrap();
                let results = brain.recall_by_similarity(&query_emb, 5, 0.35);
                if !results.is_empty() {
                    let lines: Vec<String> = results
                        .iter()
                        .map(|(item, score)| {
                            format!(
                                "[{}] (sim: {:.2}) {}",
                                item.memory_type, score,
                                item.content.chars().take(200).collect::<String>()
                            )
                        })
                        .collect();
                    format!("\nRelevant memories:\n{}\n", lines.join("\n"))
                } else {
                    String::new()
                }
            }
            Err(_) => {
                // Fallback: keyword search
                let brain = self.brain.lock().unwrap();
                let results = brain.recall(user_input, 5);
                if !results.is_empty() {
                    let lines: Vec<String> = results
                        .iter()
                        .map(|item| {
                            format!(
                                "[{}] (strength: {:.2}) {}",
                                item.memory_type, item.strength,
                                item.content.chars().take(200).collect::<String>()
                            )
                        })
                        .collect();
                    format!("\nRelevant memories:\n{}\n", lines.join("\n"))
                } else {
                    String::new()
                }
            }
        };

        // 2. Build the augmented prompt
        let augmented_input = if memory_context.is_empty() {
            user_input.to_string()
        } else {
            format!("{}\n\n---\nContext from MemoryBrain:\n{}", user_input, memory_context)
        };

        self.messages.push(Message {
            role: "user".to_string(),
            content: augmented_input,
        });

        // 3. Send to appropriate model and get response (with circuit breaker fallback)
        let tool_specs = self.tools.lock().unwrap().tool_specs();
        let use_main = Self::needs_complex_model(user_input);
        let response = if use_main {
            tracing::debug!("Using main model ({}) for: {:.60}", self.llm.model_name(), user_input);
            match self.llm.chat_with_tools(self.messages.clone(), tool_specs.clone()).await {
                Ok(resp) => resp,
                Err(e) if Self::is_circuit_error(&e) => {
                    tracing::warn!("Main model circuit open, falling back to fast model: {}", e);
                    self.fast_llm.chat_with_tools(self.messages.clone(), tool_specs).await?
                }
                Err(e) => return Err(e),
            }
        } else {
            tracing::info!("Using fast model ({}) for: {:.60}", self.fast_llm.model_name(), user_input);
            self.fast_llm.chat_with_tools(self.messages.clone(), tool_specs).await?
        };

        let response_content = response.message.content.clone();
        self.messages.push(Message {
            role: "assistant".to_string(),
            content: response_content.clone(),
        });

        // 4. Check for tool calls
        let final_output = if let Some(tool_call) = Self::parse_tool_call(&response_content) {
            self.execute_tool_chain(tool_call).await?
        } else {
            response_content
        };

        // 5. Store in memory (with embedding for future similarity search)
        {
            let truncated: String = final_output.chars().take(300).collect();
            let content = format!("User: {}\nAssistant: {}", user_input, truncated);
            if let Ok(emb) = self.embedder.embed(&content).await {
                let mut item = crate::memory_brain::MemoryItem::new(&content, crate::memory_brain::MemoryType::Episodic);
                item.embedding = Some(emb);
                let brain = self.brain.lock().unwrap();
                brain.episodic.store(item)?;
            } else {
                let mut brain = self.brain.lock().unwrap();
                brain.process(&content, Some(crate::memory_brain::MemoryType::Episodic))?;
            }
        }

        Ok(final_output)
    }

    /// Execute a tool call (and possible chain).
    async fn execute_tool_chain(&mut self, first_call: ToolCall) -> Result<String> {
        let mut current_tool = first_call;
        loop {
            // Execute the tool
            let result = {
                let tools = self.tools.lock().unwrap();
                match tools.get(&current_tool.name) {
                    Some(tool) => tool.execute(current_tool.params.clone()).await,
                    None => Ok(serde_json::json!({
                        "error": format!("Unknown tool: {}", current_tool.name)
                    })),
                }
            };

            let result_json = match result {
                Ok(v) => v,
                Err(e) => serde_json::json!({"error": e.to_string()}),
            };

            // Add result to message history
            self.messages.push(Message {
                role: "user".to_string(),
                content: format!(
                    "Tool '{}' result:\n```json\n{}\n```",
                    current_tool.name,
                    serde_json::to_string_pretty(&result_json)?
                ),
            });

            // Get next response from Gemma
            let tool_specs = self.tools.lock().unwrap().tool_specs();
            let response = self
                .llm
                .chat_with_tools(self.messages.clone(), tool_specs)
                .await?;

            let response_content = response.message.content.clone();
            self.messages.push(Message {
                role: "assistant".to_string(),
                content: response_content.clone(),
            });

            // Check if there's another tool call or final answer
            match Self::parse_tool_call(&response_content) {
                Some(next_tool) => {
                    current_tool = next_tool;
                    continue;
                }
                None => {
                    return Ok(response_content);
                }
            }
        }
    }

    /// Parse a JSON tool call from the model's response.
    fn parse_tool_call(content: &str) -> Option<ToolCall> {
        // Look for ```json ... ``` blocks containing tool calls
        if let Some(json_start) = content.find("```json") {
            let rest = &content[json_start + 7..];
            if let Some(json_end) = rest.find("```") {
                let json_str = rest[..json_end].trim();
                if let Ok(val) = serde_json::from_str::<Value>(json_str) {
                    if let (Some(name), Some(params)) = (
                        val.get("tool").and_then(|v| v.as_str()),
                        val.get("params").and_then(|v| v.as_object()),
                    ) {
                        return Some(ToolCall {
                            name: name.to_string(),
                            params: serde_json::Value::Object(params.clone()),
                        });
                    }
                }
            }
        }
        None
    }

    /// Run a reflection cycle: ask Gemma to review recent activity and generate improvements.
    pub async fn run_reflection_cycle(&mut self) -> Result<String> {
        let summary = {
            let brain = self.brain.lock().unwrap();
            brain.generate_context_summary()
        };

        let reflection_prompt = format!(
            r#"Self-Reflection Cycle

{}

Review the recent episode history and your current capabilities.

Consider:
1. What patterns are repeating — both good and bad?
2. Are there any tools you're missing that would help?
3. What's the single most impactful improvement you could make?
4. Is there a Rust skill or code pattern you should save?

Write a brief reflection and then one specific action plan.
If you want to save a skill, create a new tool, or modify your code,
describe the exact code changes needed."#,
            summary
        );

        self.messages.push(Message {
            role: "user".to_string(),
            content: reflection_prompt,
        });

        let response = self.llm.chat(self.messages.clone()).await?;
        self.messages.push(response.clone());

        // Record the reflection
        {
            let mut brain = self.brain.lock().unwrap();
            brain.record_reflection("scheduled_reflection", &response.content)?;
        }

        Ok(response.content)
    }
}

struct ToolCall {
    name: String,
    params: Value,
}

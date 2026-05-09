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
    brain: Arc<std::sync::Mutex<Brain>>,
    pub tools: Arc<std::sync::Mutex<ToolRegistry>>,
    embedder: Arc<dyn EmbeddingEngine>,
    session_id: String,
    messages: Vec<Message>,
}

impl Agent {
    pub fn new(
        llm: OllamaClient,
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

        Self {
            llm,
            brain,
            tools,
            embedder,
            session_id,
            messages: vec![Message {
                role: "system".to_string(),
                content: system_prompt,
            }],
        }
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
                                &item.content[..item.content.len().min(200)]
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
                                &item.content[..item.content.len().min(200)]
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

        // 3. Send to Gemma and get response
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

        // 4. Check for tool calls
        let final_output = if let Some(tool_call) = Self::parse_tool_call(&response_content) {
            self.execute_tool_chain(tool_call).await?
        } else {
            response_content
        };

        // 5. Store in memory (with embedding for future similarity search)
        {
            let content = format!("User: {}\nAssistant: {}", user_input, &final_output[..final_output.len().min(300)]);
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

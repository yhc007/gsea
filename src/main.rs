// Suppress dead_code warnings for intentionally retained future-use API surface
#![allow(dead_code)]

mod agent;
mod evolution;
mod gui;
mod llm;
mod mcp_server;
mod memory_brain;
mod memory_system;
mod pekko_agent;
mod tools;

use std::io::IsTerminal;
use std::sync::Arc;

use anyhow::Result;
use clap::Parser;

use agent::Agent;
use evolution::EvolutionEngine;
use llm::{
    embedding::{EmbeddingEngine, OllamaEmbedder},
    OllamaClient,
};
use memory_brain::Brain;
use pekko_agent::GseaPekkoAgent;
use pekko_actor::{ActorRef, ActorSystem};
use pekko_agent_core::{AgentInfo, AgentMessage, AgentStatus, AgentTask, TaskPriority, UserQuery};
use pekko_agent_orchestrator::{OrchestratorActor, OrchestratorMessage, Workflow, WorkflowStep};
use tools::{
    file_tools,
    memory_tools,
    skill_tools,
    ToolRegistry,
};

#[derive(Parser)]
#[command(name = "gsea", version, about = "Gemma Self-Evolving Agent")]
struct Cli {
    /// Main model
    #[arg(short, long, default_value = "gemma4:26b")]
    model: String,

    /// Ollama base URL
    #[arg(short, long, default_value = "http://localhost:11434")]
    ollama_url: String,

    /// Path to the MemoryBrain SQLite database
    #[arg(short = 'd', long, default_value = "memory")]
    db_path: String,

    /// Fast model for evolution cycles and simple tasks
    #[arg(long, default_value = "qwen3:8b")]
    fast_model: String,

    /// Embedding model for semantic memory search
    #[arg(short = 'e', long, default_value = "nomic-embed-text")]
    embed_model: String,

    /// Interval for automatic reflection cycles (number of episodes)
    #[arg(short, long, default_value_t = 5)]
    reflect_interval: u64,

    /// Run in interactive mode
    #[arg(short, long)]
    interactive: bool,

    /// Resume from a saved session file
    #[arg(long)]
    resume: Option<String>,

    /// Save session to this path on exit (interactive mode only)
    #[arg(long, default_value = "sessions/latest.json")]
    session_out: String,

    /// One-shot prompt (non-interactive)
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    prompt: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let cli = Cli::parse();

    // Initialize Brain (memory-brain)
    let brain = Arc::new(std::sync::Mutex::new(Brain::new(&cli.db_path)?));
    tracing::info!("Brain initialized at {}", cli.db_path);

    // Initialize Ollama clients (main + fast)
    let llm = OllamaClient::new(&cli.ollama_url, &cli.model);
    let fast_llm = OllamaClient::new(&cli.ollama_url, &cli.fast_model);
    tracing::info!("Main model: {}, Fast model: {}", cli.model, cli.fast_model);

    // Build tool registry early (needed by serve-mcp and review)
    let registry = Arc::new(std::sync::Mutex::new(ToolRegistry::new()));
    {
        let mut reg = registry.lock().unwrap();
        reg.register(Box::new(file_tools::ReadFile));
        reg.register(Box::new(file_tools::WriteFile));
        reg.register(Box::new(file_tools::RunShell));
        reg.register(Box::new(file_tools::CargoBuild));
        reg.register(Box::new(file_tools::CargoTest));
        reg.register(Box::new(file_tools::GitCommit));
        reg.register(Box::new(memory_tools::MemoryStore::new(brain.clone())));
        reg.register(Box::new(memory_tools::MemoryRecall::new(brain.clone())));
        reg.register(Box::new(memory_tools::MemoryStats::new(brain.clone())));
        reg.register(Box::new(memory_tools::Reflect::new(brain.clone())));
        reg.register(Box::new(skill_tools::CallSkill::new(brain.clone())));
    }

    // Check for subcommands that don't need an agent
    let first_arg = cli.prompt.first().map(|s| s.as_str());
    if first_arg == Some("review") {
        let rev = cli.prompt.get(1).cloned().unwrap_or_else(|| "HEAD~1".to_string());
        return run_review(&llm, &rev).await;
    }
    if first_arg == Some("serve-mcp") || first_arg == Some("server-mcp") {
        // Create LLM client for MCP query tool
        let mcp_llm = OllamaClient::new(&cli.ollama_url, &cli.model);
        return mcp_server::run_mcp_server(registry, brain, None, Some(mcp_llm)).await;
    }
    // Initialize embedding engine
    let embedder: Arc<dyn EmbeddingEngine> = Arc::new(OllamaEmbedder::new(
        &cli.ollama_url,
        &cli.embed_model,
    ));
    tracing::info!("Embedding engine initialized with model: {}", cli.embed_model);

    tracing::info!(
        "GSEA initialized with {} tools (startup)",
        registry.lock().unwrap().list_tools().len()
    );

    // Create agent (with fast model for evolution)
    let mut agent = Agent::new(llm, fast_llm, brain.clone(), registry.clone(), embedder);

    // Resume session if requested
    if let Some(ref session_path) = cli.resume {
        match agent.load_session(session_path) {
            Ok(_) => tracing::info!("Resumed session from {}", session_path),
            Err(e) => tracing::warn!("Could not load session {}: {}", session_path, e),
        }
    }

    // Create evolution engine (uses its own fast LLM client)
    let fast_llm_evo = OllamaClient::new(&cli.ollama_url, &cli.fast_model);
    let mut evolution = EvolutionEngine::new(brain.clone(), registry.clone(), fast_llm_evo, cli.reflect_interval);

    // Run mode
    if first_arg == Some("gui") {
        run_gui(agent, brain, registry, &cli.model).await?;
    } else if first_arg == Some("pekko") {
        run_pekko(agent, &mut evolution).await?;
    } else if cli.interactive {
        let session_path = cli.session_out.clone();
        run_interactive(&mut agent, &mut evolution).await?;
        // Save session on exit
        if let Err(e) = agent.save_session(&session_path) {
            tracing::warn!("Failed to save session: {}", e);
        } else {
            tracing::info!("Session saved to {}", session_path);
        }
    } else if !cli.prompt.is_empty() {
        let prompt = cli.prompt.join(" ");
        run_one_shot(&mut agent, &mut evolution, &prompt).await?;
    } else {
        // Read from stdin if available, otherwise show help
        if std::io::stdin().is_terminal() {
            println!("GSEA — Gemma Self-Evolving Agent");
            println!("Usage: gsea [OPTIONS] <PROMPT>");
            println!("       gsea --interactive");
            println!();
            let _ = Cli::parse_from(&["gsea", "--help"]);
        } else {
            use std::io::Read;
            let mut input = String::new();
            std::io::stdin().read_to_string(&mut input)?;
            run_one_shot(&mut agent, &mut evolution, input.trim()).await?;
        }
    }

    Ok(())
}

// ─── Code Review ──────────────────────────────────────────────

/// Review a git diff. Usage: gsea review [<git-ref>]
async fn run_review(llm: &OllamaClient, rev: &str) -> Result<()> {
    // Get the git diff
    let output = tokio::process::Command::new("git")
        .args(["diff", rev])
        .output()
        .await?;

    let diff = String::from_utf8_lossy(&output.stdout);
    if diff.trim().is_empty() {
        println!("No diff found against {}", rev);
        return Ok(());
    }

    let prompt = format!(
        r#"You are a senior Rust code reviewer. Review the following git diff and provide:

1. **Summary**: What does this change do in one sentence?
2. **Issues**: Any bugs, safety concerns, or style problems? Be specific.
3. **Suggestions**: Concrete code improvement suggestions.

```diff
{}
```"#,
        diff
    );

    let messages = vec![
        crate::llm::Message {
            role: "system".to_string(),
            content: "You are a concise, senior Rust code reviewer.".to_string(),
        },
        crate::llm::Message {
            role: "user".to_string(),
            content: prompt,
        },
    ];

    let response = llm.chat(messages).await?;
    println!("{}", "-".repeat(50));
    println!("📋 Code Review (diff against {})", rev);
    println!("{}", "-".repeat(50));
    println!("{}", response.content);
    Ok(())
}

// ─── GUI Mode ─────────────────────────────────────────────────

async fn run_gui(
    agent: Agent,
    brain: Arc<std::sync::Mutex<Brain>>,
    registry: Arc<std::sync::Mutex<ToolRegistry>>,
    model: &str,
) -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([960.0, 640.0])
            .with_title("GSEA — Gemma Self-Evolving Agent"),
        ..Default::default()
    };

    let app = gui::GseaGui::new(Some(agent), brain, registry, model);
    if let Err(e) = eframe::run_native(
        "GSEA",
        options,
        Box::new(|_cc| Ok(Box::new(app))),
    ) {
        eprintln!("GUI error: {}", e);
    }
    Ok(())
}

async fn run_one_shot(
    agent: &mut Agent,
    evolution: &mut EvolutionEngine,
    prompt: &str,
) -> Result<()> {
    tracing::info!("Processing one-shot prompt: {:.100}...", prompt);

    let response = agent.process_message(prompt).await?;
    println!("{}", response);

    evolution.after_episode().await?;
    Ok(())
}

async fn run_interactive(
    agent: &mut Agent,
    evolution: &mut EvolutionEngine,
) -> Result<()> {
    println!("GSEA Interactive Mode");
    println!("  Type /help for available commands");
    println!("{}", "─".repeat(50));

    let mut rl = rustyline::DefaultEditor::new()?;

    loop {
        let readline = rl.readline(">> ");
        match readline {
            Ok(line) => {
                let line = line.trim().to_string();

                match line.as_str() {
                    "exit" | "quit" => {
                        println!("Goodbye!");
                        break;
                    }
                    "/reflect" => {
                        println!("Running reflection cycle...");
                        let reflection = agent.run_reflection_cycle().await?;
                        println!("{}", reflection);
                        continue;
                    }
                    "/stats" => {
                        let brain = evolution.brain.lock().unwrap();
                        let stats = brain.stats();
                        println!("{}", serde_json::to_string_pretty(&stats)?);
                        continue;
                    }
                    "/help" => {
                        println!("Commands:");
                        println!("  /learn <text>   Store information in long-term memory");
                        println!("  /forget <id>    Delete a memory by its UUID");
                        println!("  /tools          List all registered tools");
                        println!("  /stats          Show memory statistics");
                        println!("  /reflect        Run a self-evolution reflection cycle");
                        println!("  exit, quit      Exit");
                        continue;
                    }
                    "/tools" => {
                        let reg = agent.tools.lock().unwrap();
                        println!("Registered tools ({}):", reg.list_tools().len());
                        println!("{}", reg.tool_description_text());
                        continue;
                    }
                    s if s.starts_with("/forget ") => {
                        let id = s.trim_start_matches("/forget ").trim();
                        let brain = evolution.brain.lock().unwrap();
                        match brain.forget(id) {
                            Ok(_) => println!("Forgotten: {}", id),
                            Err(e) => println!("Error: {}", e),
                        }
                        continue;
                    }
                    s if s.starts_with("/learn ") => {
                        let content = s.trim_start_matches("/learn ").trim();
                        let brain = evolution.brain.lock().unwrap();
                        match brain.learn(content) {
                            Ok(id) => println!("✅ Learned (id: {})", id),
                            Err(e) => println!("Error: {}", e),
                        }
                        continue;
                    }
                    "" => continue,
                    _ => {}
                }

                rl.add_history_entry(&line)?;

                // Stream response token by token
                print!("\n");
                match agent.process_message_stream(&line).await {
                    Ok(mut rx) => {
                        while let Some(chunk) = rx.recv().await {
                            print!("{}", chunk);
                            use std::io::Write;
                            let _ = std::io::stdout().flush();
                        }
                        println!("\n");
                    }
                    Err(e) => {
                        // Fallback to non-streaming on error
                        eprintln!("Stream error ({}), using non-streaming...", e);
                        let response = agent.process_message(&line).await?;
                        println!("{}", response);
                        println!();
                    }
                }

                evolution.after_episode().await?;
            }
            Err(rustyline::error::ReadlineError::Interrupted)
            | Err(rustyline::error::ReadlineError::Eof) => {
                break;
            }
            Err(e) => {
                eprintln!("Input error: {}", e);
                break;
            }
        }
    }

    Ok(())
}

// ─── Pekko Actor Mode ──────────────────────────────────────────────────────

/// Role-specific system prompts for specialized agents.
fn role_system_prompt(role: &str) -> &'static str {
    match role {
        "coder" => "\
You are GSEA-Coder, a specialized Rust code writer within the GSEA multi-agent system.\n\
Your job is to write clean, idiomatic, well-tested Rust code.\n\
Focus on: implementation, code generation, refactoring, and writing tests.\n\
Always follow Rust best practices: error handling with Result/anyhow, proper lifetimes, no unwrap in production code.\n\
When given a task, produce complete, compilable code. Use the available tools (read_file, write_file, cargo_build, cargo_test).",

        "reviewer" => "\
You are GSEA-Reviewer, a specialized code review agent within the GSEA multi-agent system.\n\
Your job is to review code for correctness, safety, performance, and idiomatic Rust style.\n\
Focus on: finding bugs, unsafe patterns, missing error handling, performance issues, and style problems.\n\
Be specific and actionable in your feedback. Reference exact lines and suggest concrete fixes.\n\
Use available tools (read_file, run_shell with 'git diff') to examine code.",

        "tester" => "\
You are GSEA-Tester, a specialized testing agent within the GSEA multi-agent system.\n\
Your job is to write comprehensive tests, run test suites, and verify code correctness.\n\
Focus on: unit tests, integration tests, edge cases, error paths, and property-based testing.\n\
Use cargo_test to run tests and verify they pass. Report test coverage and any failures clearly.",

        _ => "\
You are GSEA — a self-evolving Rust engineering agent powered by a local LLM.\n\
You are the coordinator in a multi-agent system with specialized agents:\n\
  - gsea-coder: writes Rust code, implements features\n\
  - gsea-reviewer: reviews code for bugs, safety, style\n\
  - gsea-tester: writes and runs tests\n\n\
When a task is better handled by a specialist, delegate using this format:\n\
@delegate(agent-id, task description)\n\
Examples:\n\
  @delegate(gsea-coder, implement a retry utility with exponential backoff)\n\
  @delegate(gsea-reviewer, review the retry utility in src/retry.rs)\n\
  @delegate(gsea-tester, write tests for the retry module)\n\n\
Only delegate when the task clearly fits a specialist. For general questions, answer directly.",
    }
}

/// Register built-in workflow templates in the Brain if not already present.
fn register_builtin_workflow_templates(brain: &Brain) {
    let templates = vec![
        (
            "pr-review",
            "Code review pipeline: coder implements → reviewer reviews → tester verifies",
            r#"[
                {"step_id":"code","agent":"gsea-coder","prompt_template":"Implement the following:\n\nTask: {task}"},
                {"step_id":"review","agent":"gsea-reviewer","prompt_template":"Review the code below:\n\nOriginal task: {task}\n\n--- Coder's output ---\n{prev_output}\n--- End ---\n\nCheck for bugs, safety issues, performance, and style."},
                {"step_id":"test","agent":"gsea-tester","prompt_template":"Write tests for the implementation:\n\nOriginal task: {task}\n\n--- Implementation ---\n{all_outputs}\n--- End ---\n\nCover happy path, edge cases, and error paths."}
            ]"#,
        ),
        (
            "refactor",
            "Refactoring pipeline: reviewer analyzes → coder refactors → tester validates",
            r#"[
                {"step_id":"analyze","agent":"gsea-reviewer","prompt_template":"Analyze the following code for refactoring opportunities:\n\nTarget: {task}\n\nIdentify: code smells, duplication, complexity, naming issues. Be specific with line references."},
                {"step_id":"refactor","agent":"gsea-coder","prompt_template":"Refactor based on this analysis:\n\nOriginal target: {task}\n\n--- Analysis ---\n{prev_output}\n--- End ---\n\nApply the suggested improvements. Show the refactored code."},
                {"step_id":"verify","agent":"gsea-tester","prompt_template":"Verify the refactoring is correct:\n\nOriginal target: {task}\n\n--- Changes ---\n{all_outputs}\n--- End ---\n\nRun existing tests and write new ones if needed. Confirm behavior is preserved."}
            ]"#,
        ),
        (
            "bugfix",
            "Bug fix pipeline: reviewer diagnoses → coder fixes → tester validates",
            r#"[
                {"step_id":"diagnose","agent":"gsea-reviewer","prompt_template":"Diagnose this bug:\n\nBug report: {task}\n\nAnalyze root cause, identify affected code, and suggest a fix approach."},
                {"step_id":"fix","agent":"gsea-coder","prompt_template":"Fix this bug based on the diagnosis:\n\nBug report: {task}\n\n--- Diagnosis ---\n{prev_output}\n--- End ---\n\nImplement the fix. Show the changed code."},
                {"step_id":"validate","agent":"gsea-tester","prompt_template":"Validate the bug fix:\n\nBug report: {task}\n\n--- Fix details ---\n{all_outputs}\n--- End ---\n\nWrite a regression test that reproduces the bug and confirms the fix works."}
            ]"#,
        ),
    ];

    for (name, description, steps_json) in templates {
        // Only insert if not already stored
        if brain.get_workflow_template(name).is_none() {
            if let Err(e) = brain.store_workflow_template(name, description, steps_json) {
                tracing::warn!("Failed to register template '{}': {}", name, e);
            }
        }
    }
}

/// Start an `ActorSystem` with multiple specialized agents and an orchestrator.
///
/// Spawns: gsea-main (coordinator), gsea-coder, gsea-reviewer, gsea-tester
/// Plus an OrchestratorActor for task management.
///
/// Extra REPL commands:
///   `/agents`                     — list registered agents and their status
///   `/delegate <agent> <task>`    — delegate a task to a specific agent
///   `/workflow <task>`            — run coder→reviewer→tester pipeline
///   `/workflow list`              — list available workflow templates
///   `/workflow run <name> <task>` — run a named workflow template
///   `/workflow save <name> <desc> <json>` — save a custom workflow template
///   `/history [query]`            — recall past agent results
///   `/memory <text>`              — store `<text>` as a CLS episodic memory
///   `/dream`                      — run one offline consolidation pass
async fn run_pekko(agent: Agent, evolution: &mut EvolutionEngine) -> Result<()> {
    println!("GSEA Pekko Mode — Multi-Agent ActorSystem starting…");
    println!("  Type 'exit' or 'quit' to stop");
    println!("  /agents                      — list agents");
    println!("  /delegate <agent> <task>     — delegate to agent");
    println!("  /workflow <task>             — run coder→reviewer→tester pipeline");
    println!("  /workflow list               — list workflow templates");
    println!("  /workflow run <name> <task>  — run a named template");
    println!("  /history [query]             — recall past agent results");
    println!("  /memory <text>               — store a CLS memory");
    println!("  /dream                       — run dream consolidation");
    println!("{}", "─".repeat(50));

    // Boot the ActorSystem.
    let system = ActorSystem::new("gsea");

    // ─── Spawn the Orchestrator ─────────────────────────────────
    let orchestrator = OrchestratorActor::new();
    let orch_ref = system.spawn(orchestrator, "orchestrator").await?;
    tracing::info!("OrchestratorActor spawned");

    // ─── Spawn the main agent ───────────────────────────────────
    let main_agent = GseaPekkoAgent::new("gsea-main", agent);
    let main_ref = system.spawn(main_agent, "gsea-main").await?;
    tracing::info!(actor = %main_ref.name(), "Main agent spawned");

    // Register main agent with orchestrator
    orch_ref.tell(OrchestratorMessage::RegisterAgent(AgentInfo {
        agent_id: "gsea-main".to_string(),
        agent_type: "coordinator".to_string(),
        description: "Main GSEA coordinator agent".to_string(),
        capabilities: vec!["reasoning".into(), "tools".into(), "memory".into()],
        status: AgentStatus::Available,
    })).await?;

    // ─── Spawn specialized agents ───────────────────────────────
    // Each specialized agent shares the same Brain and ToolRegistry but
    // has a role-specific system prompt.
    let brain_ref = evolution.brain.clone();
    let registry_ref = {
        // Get registry from the main agent's tools
        // We need to create new Agent instances for each role
        let b = brain_ref.clone();
        let reg = Arc::new(std::sync::Mutex::new(ToolRegistry::new()));
        {
            let mut r = reg.lock().unwrap();
            r.register(Box::new(file_tools::ReadFile));
            r.register(Box::new(file_tools::WriteFile));
            r.register(Box::new(file_tools::RunShell));
            r.register(Box::new(file_tools::CargoBuild));
            r.register(Box::new(file_tools::CargoTest));
            r.register(Box::new(file_tools::GitCommit));
            r.register(Box::new(memory_tools::MemoryStore::new(b.clone())));
            r.register(Box::new(memory_tools::MemoryRecall::new(b.clone())));
            r.register(Box::new(memory_tools::MemoryStats::new(b.clone())));
            r.register(Box::new(memory_tools::Reflect::new(b.clone())));
            r.register(Box::new(skill_tools::CallSkill::new(b)));
        }
        reg
    };

    // Store agent refs for delegation and shared agent arcs for streaming
    let mut agent_refs: std::collections::HashMap<String, ActorRef<AgentMessage>> = std::collections::HashMap::new();
    agent_refs.insert("gsea-main".to_string(), main_ref.clone());

    // Shared agent Arcs — allows streaming access to agents outside the actor system
    let mut agent_arcs: std::collections::HashMap<String, SharedAgent> = std::collections::HashMap::new();

    let roles = [("gsea-coder", "coder"), ("gsea-reviewer", "reviewer"), ("gsea-tester", "tester")];
    let cli = Cli::parse();

    for (agent_id, role) in &roles {
        let llm = OllamaClient::new(&cli.ollama_url, &cli.model);
        let fast_llm = OllamaClient::new(&cli.ollama_url, &cli.fast_model);
        let embedder: Arc<dyn llm::embedding::EmbeddingEngine> = Arc::new(
            llm::embedding::OllamaEmbedder::new(&cli.ollama_url, &cli.embed_model)
        );
        let mut role_agent = Agent::new(
            llm, fast_llm, brain_ref.clone(), registry_ref.clone(), embedder,
        );
        role_agent.set_system_prompt(role_system_prompt(role));

        // Snapshot tool definitions before moving agent into the Arc
        let tool_defs: Vec<pekko_agent_core::ToolDefinition> = {
            let registry = role_agent.tools.lock().expect("tool registry lock poisoned");
            registry
                .list_tools()
                .into_iter()
                .map(|t| pekko_agent_core::ToolDefinition {
                    name: t.name().to_string(),
                    description: t.description().to_string(),
                    input_schema: t.parameters(),
                    required_permissions: vec![],
                    timeout_ms: 30_000,
                    idempotent: false,
                })
                .collect()
        };

        // Create shared Arc so we can stream from the agent outside the actor system
        let agent_arc: SharedAgent = Arc::new(tokio::sync::Mutex::new(Some(role_agent)));
        agent_arcs.insert(agent_id.to_string(), agent_arc.clone());

        let pekko_agent = GseaPekkoAgent::new_with_shared_agent(agent_id, role, agent_arc, tool_defs);
        let actor_ref = system.spawn(pekko_agent, agent_id).await?;
        tracing::info!(actor = %actor_ref.name(), role = %role, "Specialized agent spawned");

        orch_ref.tell(OrchestratorMessage::RegisterAgent(AgentInfo {
            agent_id: agent_id.to_string(),
            agent_type: role.to_string(),
            description: format!("GSEA {} agent", role),
            capabilities: match *role {
                "coder" => vec!["code_generation".into(), "refactoring".into(), "testing".into()],
                "reviewer" => vec!["code_review".into(), "analysis".into()],
                "tester" => vec!["testing".into(), "verification".into()],
                _ => vec![],
            },
            status: AgentStatus::Available,
        })).await?;

        agent_refs.insert(agent_id.to_string(), actor_ref);
    }

    // Register built-in workflow templates
    {
        let brain = brain_ref.lock().unwrap();
        register_builtin_workflow_templates(&brain);
    }

    println!("Agents online: gsea-main, gsea-coder, gsea-reviewer, gsea-tester");

    // Boot the CLS MemorySystem and print initial stats.
    let mut mem = memory_system::MemorySystem::new();
    {
        let s = mem.stats();
        println!(
            "CLS MemorySystem ready — {} memories, {} concepts",
            s.total_memories, s.total_concepts
        );
    }
    println!("{}", "─".repeat(50));

    // REPL loop.
    let mut rl = rustyline::DefaultEditor::new()?;
    loop {
        let readline = rl.readline("pekko>> ");
        match readline {
            Ok(line) => {
                let line = line.trim().to_string();
                match line.as_str() {
                    "" => continue,
                    "exit" | "quit" => {
                        println!("Goodbye!");
                        break;
                    }
                    "/agents" => {
                        println!("Registered agents:");
                        for (id, _) in &agent_refs {
                            let role = match id.as_str() {
                                "gsea-main" => "coordinator",
                                "gsea-coder" => "coder",
                                "gsea-reviewer" => "reviewer",
                                "gsea-tester" => "tester",
                                _ => "unknown",
                            };
                            println!("  {} [{}]", id, role);
                        }
                        continue;
                    }
                    "/dream" => {
                        println!("Running dream consolidation…");
                        mem.dream();
                        let s = mem.stats();
                        println!(
                            "Done — {} memories, {} concepts, last_run: {:?}",
                            s.total_memories,
                            s.total_concepts,
                            s.dream_last_run
                        );
                        continue;
                    }
                    cmd if cmd.starts_with("/delegate ") => {
                        let rest = cmd.trim_start_matches("/delegate ").trim();
                        let parts: Vec<&str> = rest.splitn(2, ' ').collect();
                        if parts.len() < 2 {
                            println!("Usage: /delegate <agent-id> <task description>");
                            println!("  Agents: gsea-coder, gsea-reviewer, gsea-tester");
                            continue;
                        }
                        let target = parts[0];
                        let task_desc = parts[1];

                        if let Some(target_ref) = agent_refs.get(target) {
                            println!("Delegating to {}…", target);

                            // Submit task to orchestrator for tracking
                            let task_id = uuid::Uuid::new_v4();
                            orch_ref.tell(OrchestratorMessage::SubmitTask(AgentTask {
                                task_id,
                                description: task_desc.to_string(),
                                input: serde_json::json!({ "prompt": task_desc }),
                                priority: TaskPriority::Normal,
                                timeout_ms: 120_000,
                            })).await?;

                            // Send directly to target agent via ask()
                            let query = make_query(task_desc);
                            let response = target_ref.ask(
                                |reply_tx| AgentMessage::QueryWithReply(query, reply_tx),
                                std::time::Duration::from_secs(120),
                            ).await;

                            match response {
                                Ok(Ok(text)) => {
                                    println!("\n[{}] {}", target, text);
                                    println!();
                                    // Mark task complete
                                    orch_ref.tell(OrchestratorMessage::CompleteTask {
                                        task_id,
                                        result: serde_json::json!({ "response": &text[..text.len().min(500)] }),
                                    }).await?;
                                    // Store result in Brain for cross-session memory sharing
                                    if let Ok(mut brain) = brain_ref.lock() {
                                        let _ = brain.store_agent_result(
                                            target, task_desc, &text[..text.len().min(2000)], None,
                                        );
                                    }
                                }
                                Ok(Err(e)) => {
                                    eprintln!("[{}] Error: {}", target, e);
                                    orch_ref.tell(OrchestratorMessage::FailTask {
                                        task_id,
                                        error: e.clone(),
                                    }).await?;
                                }
                                Err(e) => {
                                    eprintln!("[{}] Communication error: {}", target, e);
                                }
                            }
                        } else {
                            println!("Unknown agent: {}", target);
                            println!("Available: {}", agent_refs.keys().cloned().collect::<Vec<_>>().join(", "));
                        }
                        continue;
                    }
                    cmd if cmd.starts_with("/workflow") => {
                        let rest = cmd.trim_start_matches("/workflow").trim();
                        if rest.is_empty() {
                            println!("Usage:");
                            println!("  /workflow <task>             — run default pipeline (coder→reviewer→tester)");
                            println!("  /workflow list               — list available templates");
                            println!("  /workflow run <name> <task>  — run a named template");
                            println!("  /workflow save <name> <desc> — save current pipeline as template");
                            continue;
                        }

                        if rest == "list" {
                            let brain = brain_ref.lock().unwrap();
                            let templates = brain.list_workflow_templates();
                            if templates.is_empty() {
                                println!("No workflow templates found.");
                            } else {
                                println!("Workflow templates:");
                                for (name, desc) in &templates {
                                    println!("  {} — {}", name, desc);
                                }
                            }
                            continue;
                        }

                        if rest.starts_with("run ") {
                            let run_rest = rest.trim_start_matches("run ").trim();
                            let parts: Vec<&str> = run_rest.splitn(2, ' ').collect();
                            if parts.len() < 2 {
                                println!("Usage: /workflow run <template-name> <task description>");
                                continue;
                            }
                            let template_name = parts[0];
                            let task_desc = parts[1];

                            let template = {
                                let brain = brain_ref.lock().unwrap();
                                brain.get_workflow_template(template_name)
                            };

                            if let Some((_desc, steps_json)) = template {
                                run_template_workflow(
                                    template_name, task_desc, &steps_json,
                                    &agent_arcs, &orch_ref, &brain_ref,
                                ).await?;
                            } else {
                                println!("Unknown template: {}", template_name);
                                let brain = brain_ref.lock().unwrap();
                                let templates = brain.list_workflow_templates();
                                if !templates.is_empty() {
                                    println!("Available: {}", templates.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", "));
                                }
                            }
                            continue;
                        }

                        if rest.starts_with("save ") {
                            let save_rest = rest.trim_start_matches("save ").trim();
                            let parts: Vec<&str> = save_rest.splitn(2, ' ').collect();
                            if parts.len() < 2 {
                                println!("Usage: /workflow save <name> <description>");
                                println!("  Saves the default coder→reviewer→tester pipeline as a named template.");
                                continue;
                            }
                            let name = parts[0];
                            let desc = parts[1];
                            let default_steps = r#"[
                                {"step_id":"code","agent":"gsea-coder","prompt_template":"Implement the following:\n\nTask: {task}"},
                                {"step_id":"review","agent":"gsea-reviewer","prompt_template":"Review the code below:\n\nOriginal task: {task}\n\n--- Coder's output ---\n{prev_output}\n--- End ---"},
                                {"step_id":"test","agent":"gsea-tester","prompt_template":"Write tests:\n\nOriginal task: {task}\n\n--- Implementation ---\n{all_outputs}\n--- End ---"}
                            ]"#;
                            let brain = brain_ref.lock().unwrap();
                            match brain.store_workflow_template(name, desc, default_steps) {
                                Ok(_) => println!("Template '{}' saved.", name),
                                Err(e) => eprintln!("Failed to save template: {}", e),
                            }
                            continue;
                        }

                        // Default: run the built-in coder→reviewer→tester pipeline
                        run_workflow(rest, &agent_arcs, &orch_ref, &brain_ref).await?;
                        continue;
                    }
                    cmd if cmd.starts_with("/history") => {
                        let query = cmd.trim_start_matches("/history").trim();
                        let brain = brain_ref.lock().unwrap();
                        let results = if query.is_empty() {
                            brain.recall_agent_results("", 10)
                        } else {
                            brain.recall_agent_results(query, 10)
                        };
                        if results.is_empty() {
                            println!("No agent results found.");
                        } else {
                            println!("Past agent results ({}):", results.len());
                            for item in &results {
                                // Parse AGENT_RESULT:agent|workflow|task header
                                let first_line = item.content.lines().next().unwrap_or("");
                                if let Some(rest) = first_line.strip_prefix("AGENT_RESULT:") {
                                    let parts: Vec<&str> = rest.splitn(3, '|').collect();
                                    if parts.len() == 3 {
                                        println!("  [{}] {} — {}", parts[0], parts[2], item.created_at);
                                        let body: String = item.content.lines().skip(1).take(3).collect::<Vec<_>>().join("\n");
                                        if !body.is_empty() {
                                            let preview: String = body.chars().take(200).collect();
                                            println!("    {}", preview);
                                        }
                                    }
                                }
                                println!();
                            }
                        }
                        continue;
                    }
                    cmd if cmd.starts_with("/memory ") => {
                        let content = cmd.trim_start_matches("/memory ").trim();
                        if content.is_empty() {
                            println!("Usage: /memory <text>");
                        } else {
                            let id = mem.store(content);
                            println!("Memory stored — id: {id}");
                            let s = mem.stats();
                            println!("  total memories: {}", s.total_memories);
                        }
                        continue;
                    }
                    _ => {}
                }

                rl.add_history_entry(&line)?;

                // Default: send to main agent via ask()
                let query = make_query(&line);
                let response = main_ref.ask(
                    |reply_tx| AgentMessage::QueryWithReply(query, reply_tx),
                    std::time::Duration::from_secs(120),
                ).await;

                match response {
                    Ok(Ok(text)) => {
                        // Check for auto-delegation in the response
                        if let Some((target, task)) = parse_delegation(&text) {
                            if let Some(target_ref) = agent_refs.get(&target) {
                                println!("\n[main → {}] Auto-delegating: {}", target, task);

                                let del_query = make_query(&task);
                                let del_response = target_ref.ask(
                                    |reply_tx| AgentMessage::QueryWithReply(del_query, reply_tx),
                                    std::time::Duration::from_secs(120),
                                ).await;

                                match del_response {
                                    Ok(Ok(del_text)) => {
                                        println!("\n[{}] {}", target, del_text);
                                        println!();
                                        // Store auto-delegated result in Brain
                                        if let Ok(mut brain) = brain_ref.lock() {
                                            let _ = brain.store_agent_result(
                                                &target, &task, &del_text[..del_text.len().min(2000)], None,
                                            );
                                        }
                                    }
                                    Ok(Err(e)) => eprintln!("[{}] Error: {}", target, e),
                                    Err(e) => eprintln!("[{}] Communication error: {}", target, e),
                                }
                            } else {
                                // Unknown agent, print the original response
                                println!("\n{}", text);
                                println!();
                            }
                        } else {
                            println!("\n{}", text);
                            println!();
                        }
                    }
                    Ok(Err(e)) => {
                        eprintln!("Agent error: {}", e);
                    }
                    Err(e) => {
                        eprintln!("Communication error: {}", e);
                    }
                }

                // Trigger evolution cycle if interval reached
                if let Ok(Some(evo_result)) = evolution.after_episode().await {
                    println!("\n🧬 Evolution: {}", evo_result);
                }
            }
            Err(rustyline::error::ReadlineError::Interrupted)
            | Err(rustyline::error::ReadlineError::Eof) => {
                break;
            }
            Err(e) => {
                eprintln!("Input error: {}", e);
                break;
            }
        }
    }

    Ok(())
}

/// Parse a delegation directive from the main agent's response.
///
/// Looks for `@delegate(agent-id, task description)` anywhere in the text.
/// Returns `Some((agent_id, task))` if found, `None` otherwise.
fn parse_delegation(text: &str) -> Option<(String, String)> {
    let marker = "@delegate(";
    let start = text.find(marker)?;
    let rest = &text[start + marker.len()..];
    let end = rest.find(')')?;
    let inner = &rest[..end];

    let comma = inner.find(',')?;
    let agent_id = inner[..comma].trim().to_string();
    let task = inner[comma + 1..].trim().to_string();

    if agent_id.is_empty() || task.is_empty() {
        return None;
    }

    Some((agent_id, task))
}

/// Build a `UserQuery` from raw input text.
fn make_query(input: &str) -> UserQuery {
    UserQuery {
        session_id: uuid::Uuid::new_v4(),
        content: input.to_string(),
        context: pekko_agent_core::ConversationContext {
            messages: vec![],
            metadata: std::collections::HashMap::new(),
        },
        auth: pekko_agent_core::AuthContext {
            user_id: "repl".to_string(),
            tenant_id: "local".to_string(),
            roles: vec!["user".to_string()],
        },
    }
}

/// Type alias for shared agent arcs used by streaming workflows.
type SharedAgent = Arc<tokio::sync::Mutex<Option<Agent>>>;

/// Run a streaming workflow step: temporarily borrow the agent from its shared
/// Arc, call `process_message_stream`, print chunks in real-time, and return
/// the full collected response.
///
/// Falls back to `process_message` (non-streaming) if the stream fails.
async fn run_streaming_step(
    agent_arc: &SharedAgent,
    prompt: &str,
) -> Result<String> {
    // Take the agent out of the Arc temporarily
    let mut agent = {
        let mut guard = agent_arc.lock().await;
        guard.take().ok_or_else(|| anyhow::anyhow!("agent slot is empty (busy?)"))?
    };

    // Try streaming first, fall back to non-streaming
    let result = match agent.process_message_stream(prompt).await {
        Ok(mut rx) => {
            let mut full_response = String::new();
            while let Some(chunk) = rx.recv().await {
                print!("{}", chunk);
                use std::io::Write;
                let _ = std::io::stdout().flush();
                full_response.push_str(&chunk);
            }
            println!(); // newline after stream completes
            Ok(full_response)
        }
        Err(stream_err) => {
            eprintln!("(stream failed: {}, using non-streaming…)", stream_err);
            agent.process_message(prompt).await
        }
    };

    // Restore the agent to its slot
    {
        let mut guard = agent_arc.lock().await;
        *guard = Some(agent);
    }

    result
}

/// Execute a coder → reviewer → tester workflow pipeline with streaming output.
///
/// 1. **Coder** receives the task and produces code/implementation
/// 2. **Reviewer** receives the coder's output and reviews it
/// 3. **Tester** receives both outputs and writes/runs tests
///
/// Each step streams tokens in real-time as they arrive from the LLM.
/// Each step's full output is passed as context to the next step.
/// The workflow is tracked via the OrchestratorActor.
async fn run_workflow(
    task_desc: &str,
    agent_arcs: &std::collections::HashMap<String, SharedAgent>,
    orch_ref: &ActorRef<OrchestratorMessage>,
    brain_ref: &Arc<std::sync::Mutex<Brain>>,
) -> Result<()> {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Workflow: coder → reviewer → tester (streaming)");
    println!("Task: {}", task_desc);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Build workflow definition for orchestrator tracking
    let mut workflow = Workflow::new(
        format!("pipeline: {}", &task_desc[..task_desc.len().min(40)]),
        task_desc.to_string(),
    );
    workflow.add_step(WorkflowStep {
        step_id: "code".into(),
        agent_type: "coder".into(),
        action: "implement".into(),
        input_mapping: std::collections::HashMap::new(),
        output_key: "code_output".into(),
        depends_on: vec![],
        timeout_ms: 120_000,
    });
    workflow.add_step(WorkflowStep {
        step_id: "review".into(),
        agent_type: "reviewer".into(),
        action: "review".into(),
        input_mapping: [("code".into(), "code_output".into())].into_iter().collect(),
        output_key: "review_output".into(),
        depends_on: vec!["code".into()],
        timeout_ms: 120_000,
    });
    workflow.add_step(WorkflowStep {
        step_id: "test".into(),
        agent_type: "tester".into(),
        action: "test".into(),
        input_mapping: [("code".into(), "code_output".into()), ("review".into(), "review_output".into())].into_iter().collect(),
        output_key: "test_output".into(),
        depends_on: vec!["code".into(), "review".into()],
        timeout_ms: 120_000,
    });

    let workflow_id = workflow.id;
    orch_ref.tell(OrchestratorMessage::CreateWorkflow(workflow)).await?;

    // Pipeline steps: (agent_id, step_label)
    let steps: Vec<(&str, &str)> = vec![
        ("gsea-coder", "Step 1/3: Coder"),
        ("gsea-reviewer", "Step 2/3: Reviewer"),
        ("gsea-tester", "Step 3/3: Tester"),
    ];

    let mut step_outputs: Vec<String> = Vec::new();
    let mut failed = false;

    for (i, (agent_id, label)) in steps.iter().enumerate() {
        println!("\n── {} ──────────────────────────────────", label);

        let agent_arc = match agent_arcs.get(*agent_id) {
            Some(a) => a,
            None => {
                eprintln!("Agent {} not found, aborting workflow", agent_id);
                failed = true;
                break;
            }
        };

        // Build the prompt with context from previous steps
        let prompt = match i {
            0 => format!(
                "Task: {}\n\n\
                 Write the implementation for this task. \
                 Produce complete, compilable Rust code. \
                 Explain your approach briefly, then show the code.",
                task_desc
            ),
            1 => format!(
                "Original task: {}\n\n\
                 --- Coder's output ---\n{}\n\
                 --- End of coder's output ---\n\n\
                 Review the code above for:\n\
                 1. Correctness and potential bugs\n\
                 2. Safety issues (unwrap, unsafe, panics)\n\
                 3. Performance concerns\n\
                 4. Idiomatic Rust style\n\
                 Be specific and actionable.",
                task_desc, step_outputs[0]
            ),
            2 => format!(
                "Original task: {}\n\n\
                 --- Coder's output ---\n{}\n\
                 --- Reviewer's feedback ---\n{}\n\
                 --- End ---\n\n\
                 Write comprehensive tests for the implementation above.\n\
                 Cover: happy path, edge cases, error paths.\n\
                 Show the test code and describe what each test verifies.",
                task_desc, step_outputs[0], step_outputs[1]
            ),
            _ => unreachable!(),
        };

        // Submit task to orchestrator
        let task_id = uuid::Uuid::new_v4();
        orch_ref.tell(OrchestratorMessage::SubmitTask(AgentTask {
            task_id,
            description: format!("{}: {}", label, &task_desc[..task_desc.len().min(60)]),
            input: serde_json::json!({ "prompt": &prompt[..prompt.len().min(200)] }),
            priority: TaskPriority::Normal,
            timeout_ms: 120_000,
        })).await?;

        // Stream the response from the agent
        match run_streaming_step(agent_arc, &prompt).await {
            Ok(text) => {
                orch_ref.tell(OrchestratorMessage::CompleteTask {
                    task_id,
                    result: serde_json::json!({ "status": "ok" }),
                }).await?;

                // Store step result in Brain for cross-session memory sharing
                if let Ok(mut brain) = brain_ref.lock() {
                    let _ = brain.store_agent_result(
                        agent_id,
                        &format!("[{}] {}", label, task_desc),
                        &text[..text.len().min(2000)],
                        Some(&workflow_id.to_string()),
                    );
                }

                step_outputs.push(text);
            }
            Err(e) => {
                eprintln!("[{}] Error: {}", agent_id, e);
                orch_ref.tell(OrchestratorMessage::FailTask {
                    task_id,
                    error: e.to_string(),
                }).await?;
                failed = true;
                break;
            }
        }
    }

    // Summary
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    if failed {
        println!("Workflow FAILED (completed {}/{} steps)", step_outputs.len(), steps.len());
        println!("Workflow ID: {}", workflow_id);
    } else {
        println!("Workflow COMPLETED ({}/{} steps)", step_outputs.len(), steps.len());
        println!("Workflow ID: {}", workflow_id);
        println!("  coder:    {} chars", step_outputs.get(0).map(|s| s.len()).unwrap_or(0));
        println!("  reviewer: {} chars", step_outputs.get(1).map(|s| s.len()).unwrap_or(0));
        println!("  tester:   {} chars", step_outputs.get(2).map(|s| s.len()).unwrap_or(0));
    }
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    Ok(())
}

/// Execute a named workflow template loaded from Brain, with streaming output.
///
/// Template steps are stored as JSON:
/// ```json
/// [
///   {"step_id": "code", "agent": "gsea-coder", "prompt_template": "...{task}...{prev_output}...{all_outputs}..."},
///   ...
/// ]
/// ```
///
/// Placeholder variables in `prompt_template`:
///   - `{task}` — the user's task description
///   - `{prev_output}` — the previous step's output
///   - `{all_outputs}` — all previous steps' outputs concatenated
async fn run_template_workflow(
    template_name: &str,
    task_desc: &str,
    steps_json: &str,
    agent_arcs: &std::collections::HashMap<String, SharedAgent>,
    orch_ref: &ActorRef<OrchestratorMessage>,
    brain_ref: &Arc<std::sync::Mutex<Brain>>,
) -> Result<()> {
    #[derive(serde::Deserialize)]
    struct TemplateStep {
        step_id: String,
        agent: String,
        prompt_template: String,
    }

    let steps: Vec<TemplateStep> = serde_json::from_str(steps_json)
        .map_err(|e| anyhow::anyhow!("Invalid template JSON: {}", e))?;

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Template: {} ({} steps, streaming)", template_name, steps.len());
    println!("Task: {}", task_desc);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let workflow_id = uuid::Uuid::new_v4();

    let mut step_outputs: Vec<String> = Vec::new();
    let mut failed = false;
    let total = steps.len();

    for (i, step) in steps.iter().enumerate() {
        let label = format!("Step {}/{}: {} ({})", i + 1, total, step.step_id, step.agent);
        println!("\n── {} ──────────────────────────────────", label);

        let agent_arc = match agent_arcs.get(&step.agent) {
            Some(a) => a,
            None => {
                eprintln!("Agent {} not found, aborting workflow", step.agent);
                failed = true;
                break;
            }
        };

        // Build prompt from template
        let prev_output = step_outputs.last().cloned().unwrap_or_default();
        let all_outputs = step_outputs.iter().enumerate()
            .map(|(j, o)| format!("--- Step {} output ---\n{}", j + 1, o))
            .collect::<Vec<_>>()
            .join("\n\n");

        let prompt = step.prompt_template
            .replace("{task}", task_desc)
            .replace("{prev_output}", &prev_output)
            .replace("{all_outputs}", &all_outputs);

        // Submit task to orchestrator
        let task_id = uuid::Uuid::new_v4();
        orch_ref.tell(OrchestratorMessage::SubmitTask(AgentTask {
            task_id,
            description: format!("{}: {}", label, &task_desc[..task_desc.len().min(60)]),
            input: serde_json::json!({ "prompt": &prompt[..prompt.len().min(200)] }),
            priority: TaskPriority::Normal,
            timeout_ms: 120_000,
        })).await?;

        // Stream the response
        match run_streaming_step(agent_arc, &prompt).await {
            Ok(text) => {
                orch_ref.tell(OrchestratorMessage::CompleteTask {
                    task_id,
                    result: serde_json::json!({ "status": "ok" }),
                }).await?;

                // Store in Brain for memory sharing
                if let Ok(mut brain) = brain_ref.lock() {
                    let _ = brain.store_agent_result(
                        &step.agent,
                        &format!("[{}/{}] {}", template_name, step.step_id, task_desc),
                        &text[..text.len().min(2000)],
                        Some(&workflow_id.to_string()),
                    );
                }

                step_outputs.push(text);
            }
            Err(e) => {
                eprintln!("[{}] Error: {}", step.agent, e);
                orch_ref.tell(OrchestratorMessage::FailTask {
                    task_id,
                    error: e.to_string(),
                }).await?;
                failed = true;
                break;
            }
        }
    }

    // Summary
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    if failed {
        println!("Template '{}' FAILED ({}/{} steps)", template_name, step_outputs.len(), total);
    } else {
        println!("Template '{}' COMPLETED ({}/{} steps)", template_name, step_outputs.len(), total);
        for (i, step) in steps.iter().enumerate() {
            println!("  {}: {} chars", step.step_id, step_outputs.get(i).map(|s| s.len()).unwrap_or(0));
        }
    }
    println!("Workflow ID: {}", workflow_id);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    Ok(())
}

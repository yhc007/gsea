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
    println!("  /workflow parallel <task>    — run coder+reviewer in parallel, then tester");
    println!("  /goal <description>           — autonomous: decompose→plan→execute");
    println!("  /evolve [target]             — self-evolution: analyze→improve→verify");
    println!("  /evolve loop <N>             — run N evolution iterations");
    println!("  /broadcast <msg>             — broadcast message to all agents");
    println!("  /events                      — show recent agent events");
    println!("  /recall <query>              — semantic search across all memories");
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
                    cmd if cmd.starts_with("/goal ") => {
                        let goal = cmd.trim_start_matches("/goal ").trim();
                        if goal.is_empty() {
                            println!("Usage: /goal <high-level goal description>");
                            continue;
                        }
                        run_autonomous_goal(
                            goal, &agent_arcs, &orch_ref, &brain_ref, &cli,
                        ).await?;
                        continue;
                    }
                    cmd if cmd.starts_with("/evolve") => {
                        let rest = cmd.trim_start_matches("/evolve").trim();
                        if rest.starts_with("loop ") {
                            let n_str = rest.trim_start_matches("loop ").trim();
                            let n: usize = n_str.parse().unwrap_or(3);
                            run_evolve_loop(n, &agent_arcs, &orch_ref, &brain_ref, evolution).await?;
                        } else {
                            run_evolve(rest, &agent_arcs, &orch_ref, &brain_ref, evolution).await?;
                        }
                        continue;
                    }
                    cmd if cmd.starts_with("/broadcast ") => {
                        let msg = cmd.trim_start_matches("/broadcast ").trim();
                        if msg.is_empty() {
                            println!("Usage: /broadcast <message>");
                            continue;
                        }
                        println!("Broadcasting to all agents…");
                        for (id, agent_ref) in &agent_refs {
                            if id == "gsea-main" { continue; }
                            let query = make_query(&format!(
                                "[BROADCAST from coordinator] {}", msg
                            ));
                            let _ = agent_ref.tell(AgentMessage::Query(query)).await;
                            // Also publish as event
                            let envelope = pekko_agent_events::AgentEventEnvelope::new(
                                "gsea-main",
                                "agent.broadcast",
                                "default",
                                uuid::Uuid::new_v4(),
                                serde_json::json!({
                                    "from": "gsea-main",
                                    "to": id,
                                    "message": msg,
                                }),
                            );
                            if let Some(arc) = agent_arcs.get(id) {
                                // Get event publisher from the pekko agent
                                // For now, log the broadcast
                                let _ = envelope; // event logged via agent's own publisher
                                let _ = arc; // suppress unused
                            }
                            println!("  → {} ✓", id);
                        }
                        // Store broadcast in Brain
                        if let Ok(mut brain) = brain_ref.lock() {
                            let _ = brain.store_agent_result(
                                "gsea-main", &format!("[Broadcast] {}", msg),
                                msg, None,
                            );
                        }
                        println!("Broadcast sent to {} agents", agent_refs.len() - 1);
                        continue;
                    }
                    "/events" => {
                        // Show recent agent results from Brain as an event log
                        let brain = brain_ref.lock().unwrap();
                        let results = brain.recall_agent_results("", 20);
                        if results.is_empty() {
                            println!("No agent events recorded.");
                        } else {
                            println!("Recent agent events ({}):", results.len());
                            for item in &results {
                                let first_line = item.content.lines().next().unwrap_or("");
                                if let Some(rest) = first_line.strip_prefix("AGENT_RESULT:") {
                                    let parts: Vec<&str> = rest.splitn(3, '|').collect();
                                    if parts.len() == 3 {
                                        println!("  {} │ {} │ {}", item.created_at, parts[0], parts[2]);
                                    }
                                }
                            }
                        }
                        continue;
                    }
                    cmd if cmd.starts_with("/recall ") => {
                        let query = cmd.trim_start_matches("/recall ").trim();
                        if query.is_empty() {
                            println!("Usage: /recall <query>");
                            continue;
                        }
                        // Semantic search using embeddings
                        run_semantic_recall(query, &brain_ref, &cli).await;
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

                        if rest.starts_with("parallel ") {
                            let task = rest.trim_start_matches("parallel ").trim();
                            run_parallel_workflow(task, &agent_arcs, &orch_ref, &brain_ref).await?;
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
                            // Store in CLS MemorySystem
                            let id = mem.store(content);
                            println!("Memory stored — CLS id: {id}");
                            // Also store in Brain with embedding for semantic search
                            match store_memory_with_embedding(
                                content, memory_brain::MemoryType::Semantic,
                                &brain_ref, &cli,
                            ).await {
                                Ok(brain_id) => println!("  Brain id: {} (with embedding)", brain_id),
                                Err(e) => eprintln!("  Brain store failed: {}", e),
                            }
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

/// Run a ReAct (Reason-Act-Observe) loop: send prompt → get response →
/// verify with `cargo build`/`cargo test` → if failure, feed error back
/// to the agent and repeat.
///
/// Returns the final successful response or the last response after
/// exhausting `max_iterations`.
async fn run_react_step(
    agent_arc: &SharedAgent,
    initial_prompt: &str,
    verify_cmd: &str,
    max_iterations: usize,
) -> Result<String> {
    let mut prompt = initial_prompt.to_string();
    let mut last_response = String::new();

    for iteration in 1..=max_iterations {
        if iteration > 1 {
            println!("\n  ↻ ReAct iteration {}/{}", iteration, max_iterations);
        }

        // Get agent response (streaming)
        last_response = run_streaming_step(agent_arc, &prompt).await?;

        // Skip verification if no verify command
        if verify_cmd.is_empty() {
            return Ok(last_response);
        }

        // Run verification command
        println!("\n  ⚙ Verifying: {}…", verify_cmd);
        let verify_output = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(verify_cmd)
            .output()
            .await;

        match verify_output {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);

                if output.status.success() {
                    println!("  ✓ Verification passed");
                    return Ok(last_response);
                }

                // Verification failed — build the retry prompt
                let error_output = if !stderr.is_empty() {
                    stderr.to_string()
                } else {
                    stdout.to_string()
                };

                // Truncate error output for the prompt
                let error_truncated: String = error_output.chars().take(3000).collect();

                println!("  ✗ Verification failed (iteration {}/{})", iteration, max_iterations);
                if iteration < max_iterations {
                    prompt = format!(
                        "The previous code has errors. Fix them.\n\n\
                         --- Verification command: {} ---\n\
                         --- Error output ---\n{}\n\
                         --- End ---\n\n\
                         Fix the issues and provide the corrected code. \
                         Show only the changed parts.",
                        verify_cmd, error_truncated
                    );
                }
            }
            Err(e) => {
                eprintln!("  ⚠ Could not run verification: {}", e);
                return Ok(last_response);
            }
        }
    }

    println!("  ⚠ Max iterations ({}) reached, returning last response", max_iterations);
    Ok(last_response)
}

/// Analyze a reviewer's response to determine severity.
///
/// Returns `true` if the review indicates critical issues that require
/// the coder to fix before proceeding.
///
/// Heuristic: looks for keywords indicating blocking problems.
fn review_has_critical_issues(review_text: &str) -> bool {
    let lower = review_text.to_lowercase();
    let critical_keywords = [
        "critical", "severe", "blocking", "must fix", "incorrect",
        "bug", "panic", "undefined behavior", "unsafe", "security",
        "memory leak", "data race", "deadlock", "crash",
        "will not compile", "does not compile", "compilation error",
        "fails to build", "broken",
    ];
    let positive_keywords = [
        "no issues", "looks good", "lgtm", "approved",
        "no critical", "no major", "well-written", "clean",
        "no bugs", "correct",
    ];

    // If positive signals dominate, no critical issues
    let positive_count = positive_keywords.iter().filter(|k| lower.contains(*k)).count();
    if positive_count >= 2 {
        return false;
    }

    // Check for critical signals
    let critical_count = critical_keywords.iter().filter(|k| lower.contains(*k)).count();
    critical_count >= 2
}

/// Execute a coder → reviewer → tester workflow pipeline with:
/// - **Streaming output** for real-time token display
/// - **ReAct loop** on the coder step: auto-verify with `cargo build` and
///   feed errors back for up to 3 self-correction iterations
/// - **Conditional branching**: if the reviewer finds critical issues, the
///   coder is asked to fix them and the reviewer re-reviews (up to 2 rounds)
///
/// The workflow is tracked via the OrchestratorActor.
async fn run_workflow(
    task_desc: &str,
    agent_arcs: &std::collections::HashMap<String, SharedAgent>,
    orch_ref: &ActorRef<OrchestratorMessage>,
    brain_ref: &Arc<std::sync::Mutex<Brain>>,
) -> Result<()> {
    const MAX_REACT_ITERATIONS: usize = 3;
    const MAX_REVIEW_ROUNDS: usize = 2;

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Workflow: coder → reviewer → tester (streaming + ReAct)");
    println!("Task: {}", task_desc);
    println!("  ReAct: up to {} coder iterations with cargo build", MAX_REACT_ITERATIONS);
    println!("  Review: up to {} coder↔reviewer rounds", MAX_REVIEW_ROUNDS);
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

    let coder_arc = agent_arcs.get("gsea-coder");
    let reviewer_arc = agent_arcs.get("gsea-reviewer");
    let tester_arc = agent_arcs.get("gsea-tester");

    if coder_arc.is_none() || reviewer_arc.is_none() || tester_arc.is_none() {
        eprintln!("Missing required agents, aborting workflow");
        return Ok(());
    }
    let coder_arc = coder_arc.unwrap();
    let reviewer_arc = reviewer_arc.unwrap();
    let tester_arc = tester_arc.unwrap();

    // ── Step 1: Coder with ReAct loop ──────────────────────────────
    println!("\n── Step 1/3: Coder (ReAct) ──────────────────────────────────");

    let coder_prompt = format!(
        "Task: {}\n\n\
         Write the implementation for this task. \
         Produce complete, compilable Rust code. \
         Explain your approach briefly, then show the code.",
        task_desc
    );

    let task_id = uuid::Uuid::new_v4();
    orch_ref.tell(OrchestratorMessage::SubmitTask(AgentTask {
        task_id,
        description: format!("Coder (ReAct): {}", &task_desc[..task_desc.len().min(60)]),
        input: serde_json::json!({ "prompt": &coder_prompt[..coder_prompt.len().min(200)] }),
        priority: TaskPriority::Normal,
        timeout_ms: 120_000,
    })).await?;

    let coder_output = match run_react_step(
        coder_arc, &coder_prompt, "cargo build 2>&1", MAX_REACT_ITERATIONS,
    ).await {
        Ok(text) => {
            orch_ref.tell(OrchestratorMessage::CompleteTask {
                task_id,
                result: serde_json::json!({ "status": "ok" }),
            }).await?;
            if let Ok(mut brain) = brain_ref.lock() {
                let _ = brain.store_agent_result(
                    "gsea-coder", &format!("[Coder/ReAct] {}", task_desc),
                    &text[..text.len().min(2000)], Some(&workflow_id.to_string()),
                );
            }
            text
        }
        Err(e) => {
            eprintln!("[gsea-coder] Error: {}", e);
            orch_ref.tell(OrchestratorMessage::FailTask {
                task_id, error: e.to_string(),
            }).await?;
            println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("Workflow FAILED at coder step");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            return Ok(());
        }
    };

    // ── Step 2: Reviewer with conditional branching ──────────────────
    let mut current_code = coder_output.clone();
    let mut review_output = String::new();

    for review_round in 1..=MAX_REVIEW_ROUNDS {
        let round_label = if review_round == 1 {
            "Step 2/3: Reviewer".to_string()
        } else {
            format!("Step 2/3: Reviewer (round {})", review_round)
        };
        println!("\n── {} ──────────────────────────────────", round_label);

        let review_prompt = format!(
            "Original task: {}\n\n\
             --- Coder's output ---\n{}\n\
             --- End of coder's output ---\n\n\
             Review the code above for:\n\
             1. Correctness and potential bugs\n\
             2. Safety issues (unwrap, unsafe, panics)\n\
             3. Performance concerns\n\
             4. Idiomatic Rust style\n\n\
             Start your review with a severity assessment:\n\
             - If there are critical/blocking issues, say \"CRITICAL\" and list them.\n\
             - If the code looks good, say \"APPROVED\" or \"no critical issues\".\n\
             Be specific and actionable.",
            task_desc, current_code
        );

        let task_id = uuid::Uuid::new_v4();
        orch_ref.tell(OrchestratorMessage::SubmitTask(AgentTask {
            task_id,
            description: format!("{}: {}", round_label, &task_desc[..task_desc.len().min(60)]),
            input: serde_json::json!({ "prompt": &review_prompt[..review_prompt.len().min(200)] }),
            priority: TaskPriority::Normal,
            timeout_ms: 120_000,
        })).await?;

        match run_streaming_step(reviewer_arc, &review_prompt).await {
            Ok(text) => {
                orch_ref.tell(OrchestratorMessage::CompleteTask {
                    task_id, result: serde_json::json!({ "status": "ok" }),
                }).await?;
                if let Ok(mut brain) = brain_ref.lock() {
                    let _ = brain.store_agent_result(
                        "gsea-reviewer", &format!("[Reviewer/round {}] {}", review_round, task_desc),
                        &text[..text.len().min(2000)], Some(&workflow_id.to_string()),
                    );
                }
                review_output = text;
            }
            Err(e) => {
                eprintln!("[gsea-reviewer] Error: {}", e);
                orch_ref.tell(OrchestratorMessage::FailTask {
                    task_id, error: e.to_string(),
                }).await?;
                println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                println!("Workflow FAILED at reviewer step");
                println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                return Ok(());
            }
        }

        // Check if review found critical issues
        if review_round < MAX_REVIEW_ROUNDS && review_has_critical_issues(&review_output) {
            println!("\n  ⚠ Critical issues found — sending back to coder for fixes");

            let fix_prompt = format!(
                "The reviewer found critical issues in your code. Fix them.\n\n\
                 --- Original task ---\n{}\n\n\
                 --- Your previous code ---\n{}\n\n\
                 --- Reviewer's feedback ---\n{}\n\
                 --- End ---\n\n\
                 Fix all critical issues mentioned in the review.\n\
                 Show the corrected code.",
                task_desc, current_code, review_output
            );

            println!("\n── Coder fix (round {}) ──────────────────────────────────", review_round);
            match run_react_step(
                coder_arc, &fix_prompt, "cargo build 2>&1", MAX_REACT_ITERATIONS,
            ).await {
                Ok(fixed_code) => {
                    if let Ok(mut brain) = brain_ref.lock() {
                        let _ = brain.store_agent_result(
                            "gsea-coder", &format!("[Coder/fix round {}] {}", review_round, task_desc),
                            &fixed_code[..fixed_code.len().min(2000)], Some(&workflow_id.to_string()),
                        );
                    }
                    current_code = fixed_code;
                    // Continue to next review round
                }
                Err(e) => {
                    eprintln!("[gsea-coder] Fix error: {}", e);
                    break; // proceed with current code
                }
            }
        } else {
            // Review passed or max rounds reached
            if review_round > 1 {
                println!("\n  ✓ Review passed after {} rounds", review_round);
            }
            break;
        }
    }

    // ── Step 3: Tester ──────────────────────────────────────────────
    println!("\n── Step 3/3: Tester ──────────────────────────────────");

    let test_prompt = format!(
        "Original task: {}\n\n\
         --- Coder's output ---\n{}\n\
         --- Reviewer's feedback ---\n{}\n\
         --- End ---\n\n\
         Write comprehensive tests for the implementation above.\n\
         Cover: happy path, edge cases, error paths.\n\
         Show the test code and describe what each test verifies.",
        task_desc, current_code, review_output
    );

    let task_id = uuid::Uuid::new_v4();
    orch_ref.tell(OrchestratorMessage::SubmitTask(AgentTask {
        task_id,
        description: format!("Tester: {}", &task_desc[..task_desc.len().min(60)]),
        input: serde_json::json!({ "prompt": &test_prompt[..test_prompt.len().min(200)] }),
        priority: TaskPriority::Normal,
        timeout_ms: 120_000,
    })).await?;

    // Tester also uses ReAct to verify tests pass
    let tester_output = match run_react_step(
        tester_arc, &test_prompt, "cargo test 2>&1", MAX_REACT_ITERATIONS,
    ).await {
        Ok(text) => {
            orch_ref.tell(OrchestratorMessage::CompleteTask {
                task_id, result: serde_json::json!({ "status": "ok" }),
            }).await?;
            if let Ok(mut brain) = brain_ref.lock() {
                let _ = brain.store_agent_result(
                    "gsea-tester", &format!("[Tester/ReAct] {}", task_desc),
                    &text[..text.len().min(2000)], Some(&workflow_id.to_string()),
                );
            }
            text
        }
        Err(e) => {
            eprintln!("[gsea-tester] Error: {}", e);
            orch_ref.tell(OrchestratorMessage::FailTask {
                task_id, error: e.to_string(),
            }).await?;
            String::from("(tester failed)")
        }
    };

    // Summary
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Workflow COMPLETED (3/3 steps)");
    println!("Workflow ID: {}", workflow_id);
    println!("  coder:    {} chars", current_code.len());
    println!("  reviewer: {} chars", review_output.len());
    println!("  tester:   {} chars", tester_output.len());
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

// ─── Self-Evolution (/evolve) ─────────────────────────────────────────

/// Self-evolution pipeline: the agent analyzes its own code, proposes
/// improvements, implements them via the coder agent with ReAct loop
/// verification, and gets a review before committing.
///
/// Flow: analyze → propose → coder(ReAct) → reviewer → commit
///
/// If `target` is empty, the agent picks what to improve. If given,
/// it focuses on the specified file or module.
async fn run_evolve(
    target: &str,
    agent_arcs: &std::collections::HashMap<String, SharedAgent>,
    orch_ref: &ActorRef<OrchestratorMessage>,
    brain_ref: &Arc<std::sync::Mutex<Brain>>,
    evolution: &mut EvolutionEngine,
) -> Result<()> {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Self-Evolution Pipeline");
    if target.is_empty() {
        println!("Target: (auto-detect)");
    } else {
        println!("Target: {}", target);
    }
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let coder_arc = agent_arcs.get("gsea-coder");
    let reviewer_arc = agent_arcs.get("gsea-reviewer");
    if coder_arc.is_none() || reviewer_arc.is_none() {
        eprintln!("Missing coder or reviewer agent");
        return Ok(());
    }
    let coder_arc = coder_arc.unwrap();
    let reviewer_arc = reviewer_arc.unwrap();

    // ── Phase 1: Analyze ──────────────────────────────────────────
    println!("\n── Phase 1/4: Analyze ──────────────────────────────────");

    // Gather context: memory stats, recent skills, project structure
    let context = {
        let brain = brain_ref.lock().unwrap();
        let stats = brain.stats();
        let skills = brain.list_skills();
        let skill_list: String = skills.iter()
            .take(10)
            .map(|(n, d)| format!("  - {}: {}", n, d))
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "Memory: {}\nLearned skills ({}):\n{}",
            stats, skills.len(), skill_list
        )
    };

    // Read target source or list src/ files for auto-detection
    let source_context = if !target.is_empty() {
        match tokio::fs::read_to_string(target).await {
            Ok(content) => {
                let truncated: String = content.chars().take(4000).collect();
                format!("Target file: {}\n```rust\n{}\n```", target, truncated)
            }
            Err(_) => format!("Target: {} (could not read file, treating as module name)", target),
        }
    } else {
        // List src/ files for auto-detection
        let output = tokio::process::Command::new("find")
            .args(["src", "-name", "*.rs", "-type", "f"])
            .output()
            .await;
        match output {
            Ok(o) => {
                let files = String::from_utf8_lossy(&o.stdout);
                format!("Source files:\n{}", files)
            }
            Err(_) => "Source files: (could not list)".to_string(),
        }
    };

    let analyze_prompt = format!(
        "You are GSEA's self-evolution system. Analyze the codebase and suggest ONE concrete improvement.\n\n\
         {}\n\n\
         {}\n\n\
         Pick ONE improvement that:\n\
         1. Adds a useful utility function, improves error handling, or refactors for clarity\n\
         2. Is small (under 50 lines of changes)\n\
         3. Will compile and pass tests\n\n\
         Respond with:\n\
         - **Target file**: exact path\n\
         - **Change type**: new function / refactor / fix\n\
         - **Description**: what and why (1-2 sentences)\n\
         - **Specific changes**: describe exactly what to add/modify",
        context, source_context
    );

    let analysis = run_streaming_step(reviewer_arc, &analyze_prompt).await?;

    // ── Phase 2: Implement ────────────────────────────────────────
    println!("\n── Phase 2/4: Implement (ReAct) ──────────────────────────────────");

    let implement_prompt = format!(
        "Implement the following improvement to the GSEA codebase.\n\n\
         --- Analysis ---\n{}\n--- End ---\n\n\
         Rules:\n\
         - Write complete, compilable Rust code\n\
         - Use the write_file tool to save your changes\n\
         - Include unit tests if adding new functions\n\
         - Keep changes minimal and focused\n\
         - Use anyhow::Result for error handling\n\n\
         Implement the change now.",
        analysis
    );

    let task_id = uuid::Uuid::new_v4();
    orch_ref.tell(OrchestratorMessage::SubmitTask(AgentTask {
        task_id,
        description: format!("Evolve: {}", if target.is_empty() { "auto" } else { target }),
        input: serde_json::json!({ "type": "evolution" }),
        priority: TaskPriority::Normal,
        timeout_ms: 180_000,
    })).await?;

    let implementation = match run_react_step(
        coder_arc, &implement_prompt, "cargo build 2>&1", 3,
    ).await {
        Ok(text) => {
            orch_ref.tell(OrchestratorMessage::CompleteTask {
                task_id,
                result: serde_json::json!({ "status": "ok" }),
            }).await?;
            text
        }
        Err(e) => {
            eprintln!("Implementation failed: {}", e);
            orch_ref.tell(OrchestratorMessage::FailTask {
                task_id, error: e.to_string(),
            }).await?;
            println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("Evolution FAILED at implementation phase");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            return Ok(());
        }
    };

    // ── Phase 3: Review ───────────────────────────────────────────
    println!("\n── Phase 3/4: Review ──────────────────────────────────");

    let review_prompt = format!(
        "Review this self-evolution change for correctness and safety.\n\n\
         --- Original analysis ---\n{}\n\n\
         --- Implementation ---\n{}\n--- End ---\n\n\
         Check:\n\
         1. Does the implementation match the analysis?\n\
         2. Any bugs, panics, or unsafe patterns?\n\
         3. Will this break existing functionality?\n\n\
         Say APPROVED if safe, or list specific issues.",
        analysis, implementation
    );

    let review = run_streaming_step(reviewer_arc, &review_prompt).await?;

    // ── Phase 4: Commit (if approved) ─────────────────────────────
    println!("\n── Phase 4/4: Commit ──────────────────────────────────");

    if review_has_critical_issues(&review) {
        println!("  ⚠ Review found critical issues — skipping commit");
        println!("  Run /evolve again to retry with a different improvement");
        if let Ok(mut brain) = brain_ref.lock() {
            let _ = brain.store_agent_result(
                "evolution", "self-evolution (rejected)",
                &format!("Analysis:\n{}\n\nReview:\n{}", &analysis[..analysis.len().min(500)], &review[..review.len().min(500)]),
                None,
            );
        }
    } else {
        println!("  ✓ Review passed — committing");

        // Run tests before committing
        println!("  ⚙ Running cargo test…");
        let test_output = tokio::process::Command::new("cargo")
            .args(["test", "--", "--quiet"])
            .output()
            .await;

        let tests_pass = matches!(&test_output, Ok(o) if o.status.success());
        if tests_pass {
            println!("  ✓ Tests passed");

            // Git commit
            evolution.git_commit("self-evolution").await;
            println!("  ✓ Committed");

            // Store in Brain
            if let Ok(mut brain) = brain_ref.lock() {
                let _ = brain.store_agent_result(
                    "evolution", "self-evolution (committed)",
                    &format!("Analysis:\n{}\n\nImplementation:\n{}", &analysis[..analysis.len().min(1000)], &implementation[..implementation.len().min(1000)]),
                    None,
                );
                let _ = brain.record_reflection("evolution_success", &analysis[..analysis.len().min(200)]);
            }
        } else {
            println!("  ✗ Tests failed — skipping commit");
            if let Ok(o) = &test_output {
                let stderr = String::from_utf8_lossy(&o.stderr);
                let preview: String = stderr.chars().take(500).collect();
                if !preview.is_empty() {
                    println!("  {}", preview);
                }
            }
        }
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Self-Evolution pipeline complete");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    Ok(())
}

// ─── Parallel Workflow ────────────────────────────────────────────────

/// Execute a parallel workflow: coder and reviewer run simultaneously
/// on the same task, then tester runs with both outputs.
///
/// This leverages tokio::spawn to run independent steps concurrently,
/// reducing total wall-clock time compared to the sequential pipeline.
///
/// Flow:
///   ┌── coder ──┐
///   │           ├──→ tester
///   └── reviewer┘
async fn run_parallel_workflow(
    task_desc: &str,
    agent_arcs: &std::collections::HashMap<String, SharedAgent>,
    orch_ref: &ActorRef<OrchestratorMessage>,
    brain_ref: &Arc<std::sync::Mutex<Brain>>,
) -> Result<()> {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Parallel Workflow: (coder ∥ reviewer) → tester");
    println!("Task: {}", task_desc);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let coder_arc = agent_arcs.get("gsea-coder");
    let reviewer_arc = agent_arcs.get("gsea-reviewer");
    let tester_arc = agent_arcs.get("gsea-tester");

    if coder_arc.is_none() || reviewer_arc.is_none() || tester_arc.is_none() {
        eprintln!("Missing required agents");
        return Ok(());
    }
    let coder_arc = coder_arc.unwrap().clone();
    let reviewer_arc = reviewer_arc.unwrap().clone();
    let tester_arc = tester_arc.unwrap();

    let workflow_id = uuid::Uuid::new_v4();

    // ── Phase 1: Coder + Reviewer in parallel ─────────────────────
    println!("\n── Phase 1/2: Coder ∥ Reviewer (parallel) ──────────────────────");

    let coder_prompt = format!(
        "Task: {}\n\n\
         Write the implementation for this task. \
         Produce complete, compilable Rust code. \
         Explain your approach briefly, then show the code.",
        task_desc
    );

    let reviewer_prompt = format!(
        "Task: {}\n\n\
         Analyze this task from a reviewer's perspective BEFORE seeing any code.\n\
         Identify:\n\
         1. Potential pitfalls and edge cases\n\
         2. Rust safety considerations\n\
         3. Suggested approach and patterns\n\
         4. What tests should cover\n\n\
         This pre-review will be combined with the coder's output.",
        task_desc
    );

    // Submit tasks to orchestrator
    let coder_task_id = uuid::Uuid::new_v4();
    let reviewer_task_id = uuid::Uuid::new_v4();
    orch_ref.tell(OrchestratorMessage::SubmitTask(AgentTask {
        task_id: coder_task_id,
        description: format!("Parallel coder: {}", &task_desc[..task_desc.len().min(60)]),
        input: serde_json::json!({ "type": "parallel_code" }),
        priority: TaskPriority::Normal,
        timeout_ms: 120_000,
    })).await?;
    orch_ref.tell(OrchestratorMessage::SubmitTask(AgentTask {
        task_id: reviewer_task_id,
        description: format!("Parallel reviewer: {}", &task_desc[..task_desc.len().min(60)]),
        input: serde_json::json!({ "type": "parallel_review" }),
        priority: TaskPriority::Normal,
        timeout_ms: 120_000,
    })).await?;

    // Run both tasks concurrently using tokio::join!
    // (Not tokio::spawn — agent futures hold std::sync::MutexGuard across await points)
    println!("  [coder + reviewer] Starting in parallel…");
    let (coder_result, reviewer_result) = tokio::join!(
        run_streaming_step(&coder_arc, &coder_prompt),
        run_streaming_step(&reviewer_arc, &reviewer_prompt),
    );
    println!("  [coder + reviewer] Both done");

    let coder_output = match coder_result {
        Ok(text) => {
            orch_ref.tell(OrchestratorMessage::CompleteTask {
                task_id: coder_task_id,
                result: serde_json::json!({ "status": "ok" }),
            }).await?;
            if let Ok(mut brain) = brain_ref.lock() {
                let _ = brain.store_agent_result(
                    "gsea-coder", &format!("[Parallel/coder] {}", task_desc),
                    &text[..text.len().min(2000)], Some(&workflow_id.to_string()),
                );
            }
            text
        }
        Err(e) => {
            eprintln!("[coder] Error: {}", e);
            orch_ref.tell(OrchestratorMessage::FailTask {
                task_id: coder_task_id, error: e.to_string(),
            }).await?;
            println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("Parallel workflow FAILED at coder");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            return Ok(());
        }
    };

    let reviewer_output = match reviewer_result {
        Ok(text) => {
            orch_ref.tell(OrchestratorMessage::CompleteTask {
                task_id: reviewer_task_id,
                result: serde_json::json!({ "status": "ok" }),
            }).await?;
            if let Ok(mut brain) = brain_ref.lock() {
                let _ = brain.store_agent_result(
                    "gsea-reviewer", &format!("[Parallel/reviewer] {}", task_desc),
                    &text[..text.len().min(2000)], Some(&workflow_id.to_string()),
                );
            }
            text
        }
        Err(e) => {
            eprintln!("[reviewer] Error: {}", e);
            orch_ref.tell(OrchestratorMessage::FailTask {
                task_id: reviewer_task_id, error: e.to_string(),
            }).await?;
            format!("(reviewer failed: {})", e)
        }
    };

    // ── Phase 2: Tester with combined context ─────────────────────
    println!("\n── Phase 2/2: Tester ──────────────────────────────────");

    let test_prompt = format!(
        "Original task: {}\n\n\
         --- Coder's implementation ---\n{}\n\
         --- End coder ---\n\n\
         --- Reviewer's pre-analysis ---\n{}\n\
         --- End reviewer ---\n\n\
         Write comprehensive tests combining the coder's implementation\n\
         with the reviewer's insights. Cover:\n\
         - Happy path\n\
         - Edge cases identified by the reviewer\n\
         - Error paths\n\
         Show the test code.",
        task_desc, coder_output, reviewer_output
    );

    let test_task_id = uuid::Uuid::new_v4();
    orch_ref.tell(OrchestratorMessage::SubmitTask(AgentTask {
        task_id: test_task_id,
        description: format!("Parallel tester: {}", &task_desc[..task_desc.len().min(60)]),
        input: serde_json::json!({ "type": "parallel_test" }),
        priority: TaskPriority::Normal,
        timeout_ms: 120_000,
    })).await?;

    let tester_output = match run_react_step(
        tester_arc, &test_prompt, "cargo test 2>&1", 3,
    ).await {
        Ok(text) => {
            orch_ref.tell(OrchestratorMessage::CompleteTask {
                task_id: test_task_id,
                result: serde_json::json!({ "status": "ok" }),
            }).await?;
            if let Ok(mut brain) = brain_ref.lock() {
                let _ = brain.store_agent_result(
                    "gsea-tester", &format!("[Parallel/tester] {}", task_desc),
                    &text[..text.len().min(2000)], Some(&workflow_id.to_string()),
                );
            }
            text
        }
        Err(e) => {
            eprintln!("[tester] Error: {}", e);
            orch_ref.tell(OrchestratorMessage::FailTask {
                task_id: test_task_id, error: e.to_string(),
            }).await?;
            String::from("(tester failed)")
        }
    };

    // Summary
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Parallel Workflow COMPLETED");
    println!("Workflow ID: {}", workflow_id);
    println!("  coder:    {} chars (parallel)", coder_output.len());
    println!("  reviewer: {} chars (parallel)", reviewer_output.len());
    println!("  tester:   {} chars (sequential)", tester_output.len());
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    Ok(())
}

// ─── Semantic Recall ──────────────────────────────────────────────────

/// Search memories using embedding similarity. Embeds the query via
/// Ollama, then runs cosine similarity against all stored memory embeddings.
async fn run_semantic_recall(
    query: &str,
    brain_ref: &Arc<std::sync::Mutex<Brain>>,
    cli: &Cli,
) {
    println!("Embedding query…");
    let embedder = OllamaEmbedder::new(&cli.ollama_url, &cli.embed_model);
    match embedder.embed(query).await {
        Ok(query_emb) => {
            let brain = brain_ref.lock().unwrap();
            let results = brain.recall_by_similarity(&query_emb, 10, 0.3);
            if results.is_empty() {
                println!("No semantically similar memories found.");
                // Fallback to keyword search
                let keyword_results = brain.recall(query, 5);
                if !keyword_results.is_empty() {
                    println!("Keyword matches ({}):", keyword_results.len());
                    for item in &keyword_results {
                        let preview: String = item.content.chars().take(120).collect();
                        println!("  [{}] {}", item.memory_type, preview);
                    }
                }
            } else {
                println!("Semantic matches ({}):", results.len());
                for (item, score) in &results {
                    let preview: String = item.content.chars().take(120).collect();
                    println!("  [{:.2}] [{}] {}", score, item.memory_type, preview);
                }
            }
        }
        Err(e) => {
            eprintln!("Embedding failed: {}. Falling back to keyword search.", e);
            let brain = brain_ref.lock().unwrap();
            let results = brain.recall(query, 10);
            if results.is_empty() {
                println!("No matches found.");
            } else {
                println!("Keyword matches ({}):", results.len());
                for item in &results {
                    let preview: String = item.content.chars().take(120).collect();
                    println!("  [{}] {}", item.memory_type, preview);
                }
            }
        }
    }
}

// ─── Evolve Loop ──────────────────────────────────────────────────────

/// Run N iterations of self-evolution, feeding each iteration's results
/// into the next as context. Stops early if an iteration fails.
async fn run_evolve_loop(
    iterations: usize,
    agent_arcs: &std::collections::HashMap<String, SharedAgent>,
    orch_ref: &ActorRef<OrchestratorMessage>,
    brain_ref: &Arc<std::sync::Mutex<Brain>>,
    evolution: &mut EvolutionEngine,
) -> Result<()> {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Self-Evolution Loop — {} iterations", iterations);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let mut successes = 0;
    let mut failures = 0;

    for i in 1..=iterations {
        println!("\n╔══════════════════════════════════════════════════╗");
        println!("║ Evolution iteration {}/{}", i, iterations);
        println!("╚══════════════════════════════════════════════════╝");

        // Build a target hint from previous iteration context
        let target = if i > 1 {
            // Let the system pick based on accumulated context
            ""
        } else {
            ""
        };

        match run_evolve(target, agent_arcs, orch_ref, brain_ref, evolution).await {
            Ok(()) => {
                successes += 1;
                println!("\n  ✓ Iteration {}/{} complete", i, iterations);
            }
            Err(e) => {
                failures += 1;
                eprintln!("\n  ✗ Iteration {}/{} failed: {}", i, iterations, e);
            }
        }

        // Brief pause between iterations to avoid overwhelming Ollama
        if i < iterations {
            println!("\n  ⏳ Pausing before next iteration…");
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        }
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Evolution Loop COMPLETED");
    println!("  {} iterations: {} succeeded, {} failed", iterations, successes, failures);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    Ok(())
}

// ─── Embedding on Store ──────────────────────────────────────────────

/// Store a memory with an embedding vector for semantic search.
/// This wraps Brain.process() with an additional embedding step.
async fn store_memory_with_embedding(
    content: &str,
    memory_type: memory_brain::MemoryType,
    brain_ref: &Arc<std::sync::Mutex<Brain>>,
    cli: &Cli,
) -> Result<memory_brain::UuidValue> {
    let embedder = OllamaEmbedder::new(&cli.ollama_url, &cli.embed_model);

    // Try to embed, fall back to plain storage
    let mut item = memory_brain::MemoryItem::new(content, memory_type.clone());

    if let Ok(emb) = embedder.embed(content).await {
        item.embedding = Some(emb);
    }

    let id = item.id;
    let brain = brain_ref.lock().unwrap();
    brain.consolidate_memory_public(item)?;
    Ok(id)
}

// ─── Autonomous Goal ──────────────────────────────────────────────────

/// Autonomous goal execution: the coordinator agent decomposes a high-level
/// goal into sub-tasks, then executes them one by one using the appropriate
/// specialist agent with the appropriate model.
///
/// Flow:
///   1. Coordinator analyzes goal → generates task plan (JSON)
///   2. For each sub-task: pick agent + model tier → execute with ReAct
///   3. Store results in Brain for future reference
async fn run_autonomous_goal(
    goal: &str,
    agent_arcs: &std::collections::HashMap<String, SharedAgent>,
    orch_ref: &ActorRef<OrchestratorMessage>,
    brain_ref: &Arc<std::sync::Mutex<Brain>>,
    _cli: &Cli,
) -> Result<()> {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Autonomous Goal Execution");
    println!("Goal: {}", goal);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let main_arc = agent_arcs.get("gsea-main");
    if main_arc.is_none() {
        eprintln!("Main agent not available");
        return Ok(());
    }
    let main_arc = main_arc.unwrap();

    // ── Phase 1: Decompose goal into sub-tasks ────────────────────
    println!("\n── Phase 1: Decomposing goal into sub-tasks ──────────────────");

    // Gather context from memory
    let memory_context = {
        let brain = brain_ref.lock().unwrap();
        let results = brain.recall(goal, 5);
        if results.is_empty() {
            String::new()
        } else {
            let lines: Vec<String> = results.iter()
                .map(|item| format!("  - [{}] {}", item.memory_type, item.content.chars().take(100).collect::<String>()))
                .collect();
            format!("\nRelevant past context:\n{}\n", lines.join("\n"))
        }
    };

    let decompose_prompt = format!(
        r#"You are a task planner. Decompose this goal into concrete sub-tasks.
{memory_context}
Goal: {goal}

Respond with a JSON array of sub-tasks. Each sub-task has:
- "task": description of what to do
- "agent": which agent should handle it ("gsea-coder", "gsea-reviewer", or "gsea-tester")
- "complexity": "simple", "moderate", or "complex"
- "verify_cmd": shell command to verify (empty string if none)

Example:
```json
[
  {{"task": "Write a retry utility function with exponential backoff", "agent": "gsea-coder", "complexity": "moderate", "verify_cmd": "cargo build 2>&1"}},
  {{"task": "Review the retry utility for edge cases", "agent": "gsea-reviewer", "complexity": "simple", "verify_cmd": ""}},
  {{"task": "Write tests for the retry utility", "agent": "gsea-tester", "complexity": "moderate", "verify_cmd": "cargo test 2>&1"}}
]
```

Keep it to 2-5 sub-tasks. Be specific and actionable."#,
    );

    let plan_response = run_streaming_step(main_arc, &decompose_prompt).await?;

    // Parse the task plan from the response
    let tasks = parse_task_plan(&plan_response);
    if tasks.is_empty() {
        println!("  Could not parse task plan from response. Aborting.");
        return Ok(());
    }

    println!("\n  Parsed {} sub-tasks:", tasks.len());
    for (i, task) in tasks.iter().enumerate() {
        let model_tier = select_model_tier(&task.complexity);
        println!("  {}. [{}] [{}] {}", i + 1, task.agent, model_tier, task.task);
    }

    let workflow_id = uuid::Uuid::new_v4();

    // ── Phase 2: Execute sub-tasks ────────────────────────────────
    let mut step_outputs: Vec<String> = Vec::new();
    let total = tasks.len();

    for (i, task) in tasks.iter().enumerate() {
        let model_tier = select_model_tier(&task.complexity);
        println!("\n── Sub-task {}/{}: {} [{}] ──────────────────────────────",
            i + 1, total, task.agent, model_tier);
        println!("  {}", task.task);

        let agent_arc = match agent_arcs.get(&task.agent) {
            Some(a) => a,
            None => {
                eprintln!("  Agent {} not found, skipping", task.agent);
                step_outputs.push(format!("(skipped: agent {} not found)", task.agent));
                continue;
            }
        };

        // Build prompt with context from previous steps
        let prev_context = if step_outputs.is_empty() {
            String::new()
        } else {
            let ctx: Vec<String> = step_outputs.iter().enumerate()
                .map(|(j, o)| format!("--- Step {} output ---\n{}", j + 1,
                    o.chars().take(1500).collect::<String>()))
                .collect();
            format!("\nPrevious step results:\n{}\n", ctx.join("\n\n"))
        };

        let task_prompt = format!(
            "Goal: {}\n\nYour specific task: {}\n{}\n\
             Execute this task. Be thorough but concise.",
            goal, task.task, prev_context
        );

        // Submit to orchestrator
        let task_id = uuid::Uuid::new_v4();
        orch_ref.tell(OrchestratorMessage::SubmitTask(AgentTask {
            task_id,
            description: format!("Goal/{}: {}", i + 1, &task.task[..task.task.len().min(60)]),
            input: serde_json::json!({
                "goal": goal,
                "sub_task": task.task,
                "model_tier": model_tier,
            }),
            priority: TaskPriority::Normal,
            timeout_ms: 180_000,
        })).await?;

        // Execute with ReAct if verify command is present
        let max_react = match task.complexity.as_str() {
            "complex" => 3,
            "moderate" => 2,
            _ => 1,
        };

        let output = match run_react_step(
            agent_arc, &task_prompt, &task.verify_cmd, max_react,
        ).await {
            Ok(text) => {
                orch_ref.tell(OrchestratorMessage::CompleteTask {
                    task_id,
                    result: serde_json::json!({ "status": "ok", "model_tier": model_tier }),
                }).await?;
                // Store in Brain
                if let Ok(mut brain) = brain_ref.lock() {
                    let _ = brain.store_agent_result(
                        &task.agent,
                        &format!("[Goal/step {}] {}", i + 1, task.task),
                        &text[..text.len().min(2000)],
                        Some(&workflow_id.to_string()),
                    );
                }
                text
            }
            Err(e) => {
                eprintln!("  Sub-task failed: {}", e);
                orch_ref.tell(OrchestratorMessage::FailTask {
                    task_id, error: e.to_string(),
                }).await?;
                format!("(failed: {})", e)
            }
        };

        step_outputs.push(output);
    }

    // ── Summary ───────────────────────────────────────────────────
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Autonomous Goal COMPLETED");
    println!("Goal: {}", goal);
    println!("Workflow ID: {}", workflow_id);
    for (i, task) in tasks.iter().enumerate() {
        let output_len = step_outputs.get(i).map(|s| s.len()).unwrap_or(0);
        let model_tier = select_model_tier(&task.complexity);
        println!("  {}. [{}] [{}] {} chars", i + 1, task.agent, model_tier, output_len);
    }
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Store the complete goal result in Brain
    if let Ok(mut brain) = brain_ref.lock() {
        let summary: String = step_outputs.iter().enumerate()
            .map(|(i, o)| format!("Step {}: {} chars", i + 1, o.len()))
            .collect::<Vec<_>>()
            .join(", ");
        let _ = brain.store_agent_result(
            "gsea-main",
            &format!("[Goal] {}", goal),
            &format!("{} sub-tasks completed: {}", tasks.len(), summary),
            Some(&workflow_id.to_string()),
        );
    }

    Ok(())
}

/// A parsed sub-task from the coordinator's task plan.
struct SubTask {
    task: String,
    agent: String,
    complexity: String,
    verify_cmd: String,
}

/// Parse a JSON task plan from the coordinator's response.
/// Extracts the JSON array from markdown code blocks if present.
fn parse_task_plan(response: &str) -> Vec<SubTask> {
    // Try to extract JSON from code block
    let json_str = if let Some(start) = response.find("```json") {
        let after = &response[start + 7..];
        if let Some(end) = after.find("```") {
            &after[..end]
        } else {
            after
        }
    } else if let Some(start) = response.find('[') {
        // Try to find raw JSON array
        if let Some(end) = response.rfind(']') {
            &response[start..=end]
        } else {
            return Vec::new();
        }
    } else {
        return Vec::new();
    };

    // Parse JSON
    let parsed: Result<Vec<serde_json::Value>, _> = serde_json::from_str(json_str.trim());
    match parsed {
        Ok(items) => {
            items.iter().filter_map(|item| {
                let task = item.get("task")?.as_str()?.to_string();
                let agent = item.get("agent")?.as_str()?.to_string();
                let complexity = item.get("complexity")
                    .and_then(|v| v.as_str())
                    .unwrap_or("moderate")
                    .to_string();
                let verify_cmd = item.get("verify_cmd")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                Some(SubTask { task, agent, complexity, verify_cmd })
            }).collect()
        }
        Err(e) => {
            eprintln!("  JSON parse error: {}", e);
            Vec::new()
        }
    }
}

// ─── Multi-Model Strategy ─────────────────────────────────────────────

/// Select model tier based on task complexity.
/// Returns a label ("fast" or "main") indicating which model to prefer.
///
/// This is used at the workflow/goal level to inform the user and for
/// future model routing. The actual model selection happens inside Agent
/// via `needs_complex_model()` which examines the prompt content.
fn select_model_tier(complexity: &str) -> &'static str {
    match complexity {
        "simple" => "fast",
        "complex" => "main",
        _ => "main", // default to main for moderate and unknown
    }
}

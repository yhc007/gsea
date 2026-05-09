mod agent;
mod evolution;
mod llm;
mod memory_brain;
mod tools;

use std::sync::Arc;

use anyhow::Result;
use clap::Parser;

use agent::Agent;
use evolution::EvolutionEngine;
use llm::{
    embedding::{EmbeddingEngine, OllamaEmbedder},
    OllamaClient,
};
use memory_brain::MemoryBrain;
use tools::{
    file_tools,
    memory_tools,
    ToolRegistry,
};

#[derive(Parser)]
#[command(name = "gsea", version, about = "Gemma Self-Evolving Agent")]
struct Cli {
    /// The Gemma model to use (must be available in Ollama)
    #[arg(short, long, default_value = "gemma4:26b")]
    model: String,

    /// Ollama base URL
    #[arg(short, long, default_value = "http://localhost:11434")]
    ollama_url: String,

    /// Path to the MemoryBrain SQLite database
    #[arg(short = 'd', long, default_value = "memory/gsea.db")]
    db_path: String,

    /// Embedding model for semantic memory search
    #[arg(short = 'e', long, default_value = "nomic-embed-text")]
    embed_model: String,

    /// Interval for automatic reflection cycles (number of episodes)
    #[arg(short, long, default_value_t = 5)]
    reflect_interval: u64,

    /// Run in interactive mode
    #[arg(short, long)]
    interactive: bool,

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

    // Initialize MemoryBrain
    let brain = Arc::new(MemoryBrain::open(&cli.db_path)?);
    tracing::info!("MemoryBrain initialized at {}", cli.db_path);

    // Initialize Ollama client
    let llm = OllamaClient::new(&cli.ollama_url, &cli.model);

    // Initialize embedding engine
    let embedder: Arc<dyn EmbeddingEngine> = Arc::new(OllamaEmbedder::new(
        &cli.ollama_url,
        &cli.embed_model,
    ));
    tracing::info!("Embedding engine initialized with model: {}", cli.embed_model);

    // Build tool registry
    let mut registry = ToolRegistry::new();
    registry.register(Box::new(file_tools::ReadFile));
    registry.register(Box::new(file_tools::WriteFile));
    registry.register(Box::new(file_tools::RunShell));
    registry.register(Box::new(file_tools::CargoBuild));
    registry.register(Box::new(file_tools::CargoTest));
    registry.register(Box::new(memory_tools::MemoryStore::new(brain.clone(), embedder.clone())));
    registry.register(Box::new(memory_tools::MemoryRecall::new(brain.clone())));
    registry.register(Box::new(memory_tools::MemoryStats::new(brain.clone())));
    registry.register(Box::new(memory_tools::Reflect::new(brain.clone())));

    tracing::info!(
        "GSEA initialized with {} tools",
        registry.list_tools().len()
    );

    // Create agent
    let mut agent = Agent::new(llm, brain.clone(), registry, embedder);

    // Create evolution engine
    let mut evolution = EvolutionEngine::new(brain.clone(), cli.reflect_interval);

    // Run mode
    if cli.interactive {
        run_interactive(&mut agent, &mut evolution).await?;
    } else if !cli.prompt.is_empty() {
        let prompt = cli.prompt.join(" ");
        run_one_shot(&mut agent, &mut evolution, &prompt).await?;
    } else {
        // Read from stdin if available, otherwise show help
        if atty::is(atty::Stream::Stdin) {
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

async fn run_one_shot(
    agent: &mut Agent,
    evolution: &mut EvolutionEngine,
    prompt: &str,
) -> Result<()> {
    tracing::info!("Processing one-shot prompt: {:.100}...", prompt);

    let response = agent.process_message(prompt).await?;
    println!("{}", response);

    evolution.after_episode(agent).await?;
    Ok(())
}

async fn run_interactive(
    agent: &mut Agent,
    evolution: &mut EvolutionEngine,
) -> Result<()> {
    println!("GSEA Interactive Mode (type 'exit' to quit, '/reflect' for manual reflection)");
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
                        let stats = evolution.brain.get_memory_stats()?;
                        println!("{}", serde_json::to_string_pretty(&stats)?);
                        continue;
                    }
                    "" => continue,
                    _ => {}
                }

                rl.add_history_entry(&line)?;

                let response = agent.process_message(&line).await?;
                println!("\n{}", response);
                println!();

                evolution.after_episode(agent).await?;
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

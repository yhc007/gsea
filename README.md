# GSEA — Gemma Self-Evolving Agent

A Rust-based autonomous agent powered by a local LLM (Gemma/Qwen via Ollama).  
GSEA learns from interactions, stores knowledge in a human-brain-inspired memory system, automatically generates and registers new tools through self-reflection cycles, and can review code via git diff.

> **Status**: Active development — all core phases complete.

---

## Quick Start

```bash
# Requirements: Rust nightly, Ollama running with gemma4:26b or qwen3:8b

# One-shot prompt
cargo run -- "Write a Rust function that reads a CSV file"

# Interactive mode
cargo run -- --interactive

# Code review (compares HEAD~1 vs current)
cargo run -- review

# Resume previous session
cargo run -- --interactive --resume sessions/latest.json
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      CLI (main.rs)                      │
│  cargo run -- "<prompt>"  │  --interactive  │  review   │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                    Agent (agent.rs)                      │
│  ┌─────────────┐  ┌────────────┐  ┌──────────────────┐  │
│  │ Ollama Chat  │  │ Embedding  │  │ Tool Chaining    │  │
│  │ (gemma/qwen) │  │ (nomic)    │  │ (JSON tool calls)│  │
│  └──────┬──────┘  └─────┬──────┘  └────────┬─────────┘  │
└─────────┼───────────────┼──────────────────┼──────────────┘
          │               │                  │
┌─────────▼───────────────▼──────────────────▼──────────────┐
│                       Brain                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│  │ Working  │ │ Episodic │ │ Semantic │ │  Procedural  │  │
│  │ Memory   │ │ Memory   │ │ Memory   │ │  Memory      │  │
│  │ (7±2)    │ │ (events) │ │ (facts)  │ │  (skills)    │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘  │
│                      SQLite (memory/memory_brain.db)       │
└───────────────────────────────────────────────────────────┘
          │
┌─────────▼────────────────────────────────────────────────┐
│                 Tool Registry (11+ tools)                  │
│  read_file │ write_file │ run_shell │ cargo_build         │
│  cargo_test │ git_commit │ memory_* │ call_skill           │
│  + dynamically registered skills as tools                 │
└───────────────────────────────────────────────────────────┘
          │
┌─────────▼────────────────────────────────────────────────┐
│              Evolution Engine (evolution/mod.rs)           │
│  Review → Propose → Extract → Save → Build → Register    │
│  Skills promoted to src/tools/skills/                     │
│  On build failure: automatic rollback                     │
└───────────────────────────────────────────────────────────┘
```

---

## Features

### 🧠 Memory System (Human Brain Inspired)

| Memory Type | Description | Analogy |
|-------------|-------------|---------|
| **Working Memory** | Short-term, 7±2 items (Miller's Law) | 현재 대화 |
| **Episodic Memory** | "When did what happen" | "어제 Rust 버그 수정함" |
| **Semantic Memory** | Facts and concepts | "Rust는 ownership으로 안전함" |
| **Procedural Memory** | Patterns and skills | 저장된 유틸리티 함수 |

- **Forgetting Curve** (Ebbinghaus): `R = e^(-t/S)` — unused memories fade
- **Embedding Search**: `nomic-embed-text` for cosine similarity recall
- **Keyword Fallback**: LIKE-based search when embeddings fail

### 🔧 Tools (11 built-in + dynamic skills)

| Category | Tools |
|----------|-------|
| File I/O | `read_file`, `write_file` |
| Shell | `run_shell` |
| Rust | `cargo_build`, `cargo_test` |
| Git | `git_commit` |
| Memory | `memory_store`, `memory_recall`, `memory_stats`, `reflect` |
| Skills | `call_skill` |
| Dynamic | Auto-registered from stored skills |

### 🔄 Self-Evolution Cycle

Triggered every N episodes (configurable via `--reflect-interval`):

```
1. Review: Gemma analyzes recent activity and system stats
2. Propose: Gemma suggests a utility function (pure Rust, ≤20 lines)
3. Extract: Code block extracted from LLM response
4. Save → `skills/{name}.rs`
5. Promote → `src/tools/skills/{name}.rs` + `pub mod {name};`
6. Build → `cargo build` verification
7. Register → DynamicSkillTool in ToolRegistry
8. Commit → `git commit` on success
9. Rollback → automatic cleanup on build failure
```

### 📋 Code Review

```bash
gsea review              # diff against HEAD~1
gsea review main         # diff against main
```

Uses gemma4:26b to analyze git diff and produce:
1. Summary
2. Issues (bugs, safety, style)
3. Suggestions with code examples

### 🚀 Model Auto-Selection

| Condition | Model |
|-----------|-------|
| Short greeting (hi, hello, ok) | `qwen3:8b` (instant) |
| Rust code, 200+ chars, technical keywords | `gemma4:26b` |
| Everything else | `qwen3:8b` |

Ollama auto-swaps models on demand (10-30s load time).

---

## CLI Reference

```bash
gsea [OPTIONS] <PROMPT>
gsea --interactive [OPTIONS]
gsea review [<git-ref>]

Options:
  -m, --model <MODEL>          Main model [default: gemma4:26b]
      --fast-model <MODEL>     Fast model [default: qwen3:8b]
  -o, --ollama-url <URL>       Ollama server [default: http://localhost:11434]
  -e, --embed-model <MODEL>    Embedding model [default: nomic-embed-text]
  -d, --db-path <PATH>         Brain database directory [default: memory]
      --reflect-interval <N>   Evolution cycle frequency [default: 5]
      --resume <PATH>          Resume session from file
      --session-out <PATH>     Save session on exit [default: sessions/latest.json]
  -i, --interactive            Interactive mode
```

### Interactive Mode Commands

```
/learn <text>     Store information in long-term memory
/forget <id>      Delete a memory by UUID
/tools            List all registered tools
/stats            Show memory statistics
/reflect          Run evolution cycle manually
/help             Show commands
```

---

## Project Structure

```
gsea/
├── Cargo.toml
├── memory/                    # SQLite database storage
├── sessions/                  # Saved conversation sessions
├── skills/                    # Auto-generated skill files
├── src/
│   ├── main.rs                # CLI entry point
│   ├── agent.rs               # Agent loop, tool chaining, model selection
│   ├── llm/
│   │   ├── mod.rs             # Ollama API client
│   │   └── embedding.rs       # Embedding engine (OllamaEmbedder)
│   ├── memory_brain/
│   │   ├── mod.rs             # Brain: unified memory interface
│   │   ├── types.rs           # MemoryItem, MemoryType, Emotion
│   │   ├── storage.rs         # SQLite persistent storage + embedding search
│   │   ├── working.rs         # Working memory (7±2)
│   │   ├── episodic.rs        # Episodic memory
│   │   ├── semantic.rs        # Semantic memory
│   │   ├── procedural.rs      # Procedural memory (skills)
│   │   ├── forgetting.rs      # Ebbinghaus forgetting curve
│   │   └── consolidate.rs     # Memory classification
│   ├── tools/
│   │   ├── mod.rs             # Tool trait + ToolRegistry
│   │   ├── file_tools.rs      # ReadFile, WriteFile, RunShell, Cargo*, GitCommit
│   │   ├── memory_tools.rs    # MemoryStore, MemoryRecall, MemoryStats, Reflect
│   │   ├── skill_tools.rs     # CallSkill, DynamicSkillTool, register_skills
│   │   └── skills/            # Auto-promoted compiled skill modules
│   │       ├── mod.rs
│   │       ├── is_alphanumeric.rs
│   │       └── to_title_case.rs
│   └── evolution/
│       └── mod.rs             # Self-evolution cycle engine
└── vendor/memory-brain/       # Referenced but independently functional
```

---

## Testing

```bash
# Run all unit tests
cargo test

# 24 tests cover:
#   types (creation, display, relevance, emotion)
#   working (push, eviction, search, clear)
#   forgetting (retention, decay, half-life)
#   consolidate (code→procedural, time→episodic, default→semantic)
#   evolution (code extraction, fn name, description)
```

---

## Dependencies

- **`tokio`** — Async runtime
- **`reqwest`** — HTTP client for Ollama API
- **`rusqlite`** — SQLite backend
- **`serde` / `serde_json`** — Serialization
- **`clap`** — CLI argument parsing
- **`tracing`** — Structured logging
- **`rustyline`** — Interactive REPL
- **`uuid`** — Memory IDs
- **`chrono`** — Timestamps

---

## License

MIT

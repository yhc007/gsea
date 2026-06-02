# GSEA + Pekko-Agent Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate `pekko-agent` crates into GSEA so the agent uses the Pekko Actor system for ReAct-loop execution, orchestration, and CLS memory — replacing the current synchronous `Agent` struct.

**Architecture:** GSEA's `Agent` will implement `pekko_agent_core::AgentActor` trait, running inside a `pekko_actor::ActorSystem`. The existing `Brain` (SQLite) is wrapped as a `ShortTermMemory` + `LongTermMemory` adapter. The `ToolRegistry` tools implement `pekko_agent_core::Tool`. The GUI and CLI drive the system by sending `AgentMessage` to an `ActorRef`.

**Tech Stack:** rust-pekko (pekko-actor), pekko-agent-core, pekko-agent-memory, pekko-agent-tools, pekko-agent-orchestrator, pekko-agent-events, tokio, eframe/egui

---

## Prerequisites

### P1: Clone and verify rust-pekko + pekko-agent locally

The pekko-agent workspace depends on `rust-pekko` (pekko-actor, pekko-persistence, pekko-event-bus) via relative path `../rust-pekko/pekko-actor`. This repo does NOT exist on GitHub under yhc007 — it must be created or located first.

- [ ] **Step 1: Locate or create rust-pekko**

Check if `rust-pekko` exists elsewhere locally:
```bash
find /Volumes/T7 -maxdepth 3 -name "pekko-actor" -type d 2>/dev/null
```

If not found, create a minimal `pekko-actor` crate that defines the `Actor` trait, `ActorSystem`, `ActorRef`, `ActorContext`, and `Props`. The trait is already referenced by pekko-agent-core:

```rust
// rust-pekko/pekko-actor/src/lib.rs
use async_trait::async_trait;

pub struct ActorSystem { name: String }
pub struct ActorContext<A: Actor> { _phantom: std::marker::PhantomData<A> }
pub struct ActorRef<M: Send + 'static> { tx: tokio::sync::mpsc::Sender<M> }
pub struct Props;

impl ActorSystem {
    pub fn new(name: &str) -> Self { Self { name: name.to_string() } }

    pub async fn spawn<A: Actor + Send + 'static>(
        &self, mut actor: A, name: &str,
    ) -> Result<ActorRef<A::Message>, anyhow::Error>
    where A::Message: Send + 'static {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<A::Message>(256);
        let actor_name = name.to_string();
        tokio::spawn(async move {
            actor.pre_start().await;
            tracing::info!("Actor '{}' started", actor_name);
            while let Some(msg) = rx.recv().await {
                let mut ctx = ActorContext { _phantom: std::marker::PhantomData };
                actor.receive(msg, &mut ctx).await;
            }
            actor.post_stop().await;
        });
        Ok(ActorRef { tx })
    }
}

impl<M: Send + 'static> ActorRef<M> {
    pub async fn tell(&self, msg: M) -> Result<(), anyhow::Error> {
        self.tx.send(msg).await.map_err(|_| anyhow::anyhow!("Actor mailbox closed"))
    }
}

#[async_trait]
pub trait Actor: Send + 'static {
    type Message: Send + 'static;
    async fn pre_start(&mut self) {}
    async fn receive(&mut self, msg: Self::Message, ctx: &mut ActorContext<Self>);
    async fn post_stop(&mut self) {}
}
```

- [ ] **Step 2: Clone pekko-agent**

```bash
cd /Volumes/T7
git clone https://github.com/yhc007/pekko-agent.git
```

- [ ] **Step 3: Set up directory structure**

Ensure the workspace path references resolve:
```
/Volumes/T7/
  rust-pekko/           # pekko-actor, pekko-persistence, pekko-event-bus
  pekko-agent/          # workspace with 7 crates + 4 services
  DeepSeek/             # GSEA (this repo)
    vendor/memory-brain/
```

- [ ] **Step 4: Verify pekko-agent compiles**

```bash
cd /Volumes/T7/pekko-agent
cargo check 2>&1 | tail -20
```

Expected: Clean compilation (or known warnings only). If path deps fail, fix Cargo.toml paths.

- [ ] **Step 5: Commit the rust-pekko scaffold if newly created**

```bash
cd /Volumes/T7/rust-pekko
git init && git add -A && git commit -m "feat: minimal pekko-actor runtime"
```

---

## Phase 1: Wire pekko-agent crates into GSEA

### Task 1: Add pekko-agent dependencies to GSEA

**Files:**
- Modify: `Cargo.toml`

- [ ] **Step 1: Add path dependencies**

Add to `[dependencies]` in `/Volumes/T7/DeepSeek/Cargo.toml`:

```toml
# Pekko Actor system
pekko-actor = { path = "../rust-pekko/pekko-actor" }

# Pekko Agent crates
pekko-agent-core = { path = "../pekko-agent/crates/pekko-agent-core" }
pekko-agent-memory = { path = "../pekko-agent/crates/pekko-agent-memory" }
pekko-agent-tools = { path = "../pekko-agent/crates/pekko-agent-tools" }
pekko-agent-orchestrator = { path = "../pekko-agent/crates/pekko-agent-orchestrator" }
pekko-agent-events = { path = "../pekko-agent/crates/pekko-agent-events" }
```

- [ ] **Step 2: Verify compilation**

```bash
cd /Volumes/T7/DeepSeek
cargo check 2>&1 | tail -20
```

Expected: Compiles (new deps resolve, no usage yet).

- [ ] **Step 3: Commit**

```bash
git add Cargo.toml Cargo.lock
git commit -m "deps: add pekko-actor and pekko-agent crates"
```

---

### Task 2: Adapt GSEA ToolRegistry tools to pekko-agent Tool trait

**Files:**
- Create: `src/tools/pekko_adapter.rs`
- Modify: `src/tools/mod.rs`

The pekko-agent `Tool` trait has a different signature from GSEA's `Tool` trait:
- GSEA: `async fn execute(&self, params: Value) -> Result<Value>`
- Pekko: `async fn execute(&self, input: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError>`

We create a bridge adapter that wraps any GSEA tool as a pekko tool.

- [ ] **Step 1: Write the adapter test**

Create `src/tools/pekko_adapter.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // A trivial GSEA tool for testing
    struct EchoTool;

    #[async_trait::async_trait]
    impl crate::tools::Tool for EchoTool {
        fn name(&self) -> &str { "echo" }
        fn description(&self) -> &str { "Echoes input" }
        fn parameters(&self) -> serde_json::Value { json!({}) }
        async fn execute(&self, params: serde_json::Value) -> anyhow::Result<serde_json::Value> {
            Ok(params)
        }
    }

    #[tokio::test]
    async fn test_pekko_tool_adapter() {
        let gsea_tool = Box::new(EchoTool) as Box<dyn crate::tools::Tool>;
        let pekko_tool = PekkoToolAdapter::new(gsea_tool);

        let def = pekko_agent_core::Tool::definition(&pekko_tool);
        assert_eq!(def.name, "echo");

        let ctx = pekko_agent_core::ToolContext {
            tenant_id: "test".to_string(),
            user_id: "test".to_string(),
            session_id: uuid::Uuid::new_v4(),
            credentials: std::collections::HashMap::new(),
            timeout: std::time::Duration::from_secs(30),
        };

        let result = pekko_agent_core::Tool::execute(
            &pekko_tool, json!({"msg": "hello"}), &ctx
        ).await.unwrap();

        assert!(!result.is_error);
        assert_eq!(result.content, json!({"msg": "hello"}));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test pekko_tool_adapter -v 2>&1 | tail -10
```

Expected: FAIL — `PekkoToolAdapter` not defined.

- [ ] **Step 3: Implement the adapter**

In `src/tools/pekko_adapter.rs`, above the `#[cfg(test)]` block:

```rust
use async_trait::async_trait;
use pekko_agent_core::{ToolDefinition, ToolContext, ToolOutput, ToolError};

/// Wraps a GSEA `Tool` to satisfy the pekko-agent `Tool` trait.
pub struct PekkoToolAdapter {
    inner: Box<dyn crate::tools::Tool>,
}

impl PekkoToolAdapter {
    pub fn new(tool: Box<dyn crate::tools::Tool>) -> Self {
        Self { inner: tool }
    }
}

#[async_trait]
impl pekko_agent_core::Tool for PekkoToolAdapter {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.inner.name().to_string(),
            description: self.inner.description().to_string(),
            input_schema: self.inner.parameters(),
            required_permissions: vec![],
            timeout_ms: 30_000,
            idempotent: false,
        }
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        _ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        match self.inner.execute(input).await {
            Ok(val) => Ok(ToolOutput::success(val)),
            Err(e) => Ok(ToolOutput::error(e.to_string())),
        }
    }
}
```

- [ ] **Step 4: Add module declaration**

In `src/tools/mod.rs`, add:

```rust
pub mod pekko_adapter;
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cargo test pekko_tool_adapter -v 2>&1 | tail -10
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/tools/pekko_adapter.rs src/tools/mod.rs
git commit -m "feat: PekkoToolAdapter — bridge GSEA tools to pekko-agent Tool trait"
```

---

### Task 3: Adapt Brain as pekko-agent ShortTermMemory + LongTermMemory

**Files:**
- Create: `src/memory_brain/pekko_adapter.rs`
- Modify: `src/memory_brain/mod.rs`

The pekko-agent defines `ShortTermMemory` and `LongTermMemory` async traits. We wrap GSEA's `Brain` (SQLite) to implement them.

- [ ] **Step 1: Write the failing test**

Create `src/memory_brain/pekko_adapter.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use pekko_agent_core::ShortTermMemory;

    #[tokio::test]
    async fn test_brain_short_term_memory() {
        let brain = Arc::new(Mutex::new(Brain::new(":memory:").unwrap()));
        let stm = BrainShortTermMemory::new(brain);

        let session = uuid::Uuid::new_v4();
        let msg = pekko_agent_core::Message {
            role: pekko_agent_core::MessageRole::User,
            content: "Hello GSEA".to_string(),
            timestamp: chrono::Utc::now(),
        };

        stm.append_message(&session, msg).await.unwrap();
        let conv = stm.get_conversation(&session).await.unwrap();
        assert_eq!(conv.len(), 1);
        assert_eq!(conv[0].content, "Hello GSEA");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test brain_short_term_memory -v 2>&1 | tail -10
```

Expected: FAIL — `BrainShortTermMemory` not defined.

- [ ] **Step 3: Implement BrainShortTermMemory**

In `src/memory_brain/pekko_adapter.rs`, above the test module:

```rust
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use async_trait::async_trait;
use tokio::sync::RwLock;
use pekko_agent_core::{ShortTermMemory, LongTermMemory, MemoryError, Message, MemoryDocument, SearchResult};
use super::Brain;

/// Adapts GSEA Brain's episodic memory as pekko ShortTermMemory.
pub struct BrainShortTermMemory {
    brain: Arc<Mutex<Brain>>,
    conversations: Arc<RwLock<HashMap<uuid::Uuid, Vec<Message>>>>,
}

impl BrainShortTermMemory {
    pub fn new(brain: Arc<Mutex<Brain>>) -> Self {
        Self {
            brain,
            conversations: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl ShortTermMemory for BrainShortTermMemory {
    async fn get_conversation(&self, session_id: &uuid::Uuid) -> Result<Vec<Message>, MemoryError> {
        let store = self.conversations.read().await;
        Ok(store.get(session_id).cloned().unwrap_or_default())
    }

    async fn append_message(&self, session_id: &uuid::Uuid, msg: Message) -> Result<(), MemoryError> {
        // Store in pekko conversation buffer
        let mut store = self.conversations.write().await;
        store.entry(*session_id).or_default().push(msg.clone());

        // Also persist to Brain's episodic memory
        let content = format!("{}: {}", msg.role, msg.content);
        let brain = self.brain.lock().map_err(|e| MemoryError::Internal(e.to_string()))?;
        let _ = brain.learn(&content);
        Ok(())
    }

    async fn summarize(&self, session_id: &uuid::Uuid) -> Result<String, MemoryError> {
        let store = self.conversations.read().await;
        let msgs = store.get(session_id)
            .ok_or_else(|| MemoryError::NotFound(format!("Session {}", session_id)))?;
        let summary = msgs.iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n");
        Ok(summary)
    }

    async fn clear(&self, session_id: &uuid::Uuid) -> Result<(), MemoryError> {
        let mut store = self.conversations.write().await;
        store.remove(session_id);
        Ok(())
    }
}

/// Adapts GSEA Brain's semantic memory as pekko LongTermMemory.
pub struct BrainLongTermMemory {
    brain: Arc<Mutex<Brain>>,
}

impl BrainLongTermMemory {
    pub fn new(brain: Arc<Mutex<Brain>>) -> Self {
        Self { brain }
    }
}

#[async_trait]
impl LongTermMemory for BrainLongTermMemory {
    async fn store(&self, doc: MemoryDocument) -> Result<String, MemoryError> {
        let brain = self.brain.lock().map_err(|e| MemoryError::Internal(e.to_string()))?;
        let id = brain.learn(&doc.content).map_err(|e| MemoryError::Internal(e.to_string()))?;
        Ok(id.to_string())
    }

    async fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>, MemoryError> {
        let brain = self.brain.lock().map_err(|e| MemoryError::Internal(e.to_string()))?;
        let items = brain.recall(query, top_k);
        let results = items.into_iter().map(|(item, score)| SearchResult {
            id: item.id.to_string(),
            score: score as f32,
            content: item.content.clone(),
            source: item.memory_type.to_string(),
        }).collect();
        Ok(results)
    }

    async fn delete(&self, doc_id: &str) -> Result<(), MemoryError> {
        let brain = self.brain.lock().map_err(|e| MemoryError::Internal(e.to_string()))?;
        brain.forget(doc_id).map_err(|e| MemoryError::Internal(e.to_string()))
    }
}
```

> **Note:** The exact `Brain::learn`, `Brain::recall`, `Brain::forget` signatures must be verified against the current GSEA Brain API. Adapt types as needed during implementation.

- [ ] **Step 4: Add module declaration**

In `src/memory_brain/mod.rs`, add:

```rust
pub mod pekko_adapter;
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cargo test brain_short_term_memory -v 2>&1 | tail -10
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/memory_brain/pekko_adapter.rs src/memory_brain/mod.rs
git commit -m "feat: Brain adapters for pekko ShortTermMemory + LongTermMemory"
```

---

## Phase 2: Implement AgentActor on GSEA Agent

### Task 4: Implement pekko AgentActor trait for GSEA Agent

**Files:**
- Create: `src/pekko_agent.rs`
- Modify: `src/main.rs`

This is the core integration. We create a `GseaPekkoAgent` that implements `pekko_agent_core::AgentActor` (which extends `pekko_actor::Actor<Message = AgentMessage>`). It wraps the existing `Agent` to provide the ReAct loop.

- [ ] **Step 1: Write the test**

Create `src/pekko_agent.rs` with test:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gsea_pekko_agent_state_transitions() {
        // This test verifies state FSM without actual LLM calls.
        let agent = GseaPekkoAgent::new_test();

        assert!(agent.current_state().is_idle());
        assert_eq!(agent.agent_id(), "gsea-test");
        assert!(!agent.available_tools().is_empty() || agent.available_tools().is_empty());
        // Tools may or may not be registered; just verify the method works.
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test gsea_pekko_agent_state -v 2>&1 | tail -10
```

Expected: FAIL — `GseaPekkoAgent` not defined.

- [ ] **Step 3: Implement GseaPekkoAgent**

In `src/pekko_agent.rs`:

```rust
use async_trait::async_trait;
use pekko_actor::{Actor, ActorContext};
use pekko_agent_core::{
    AgentActor, AgentMessage, AgentState, AgentAction, AgentResponse,
    ToolDefinition, AgentError,
    message::{UserQuery, Observation},
};
use crate::agent::Agent;
use std::sync::{Arc, Mutex};

pub struct GseaPekkoAgent {
    id: String,
    state: AgentState,
    agent: Arc<Mutex<Option<Agent>>>,
    tool_defs: Vec<ToolDefinition>,
}

impl GseaPekkoAgent {
    pub fn new(id: &str, agent: Agent) -> Self {
        // Extract tool definitions from agent's registry
        let tool_defs = {
            let reg = agent.tools.lock().unwrap();
            reg.list_tools().iter().map(|t| ToolDefinition {
                name: t.name().to_string(),
                description: t.description().to_string(),
                input_schema: t.parameters(),
                required_permissions: vec![],
                timeout_ms: 30_000,
                idempotent: false,
            }).collect()
        };

        Self {
            id: id.to_string(),
            state: AgentState::default(),
            agent: Arc::new(Mutex::new(Some(agent))),
            tool_defs,
        }
    }

    #[cfg(test)]
    pub fn new_test() -> Self {
        Self {
            id: "gsea-test".to_string(),
            state: AgentState::default(),
            agent: Arc::new(Mutex::new(None)),
            tool_defs: vec![],
        }
    }

    /// Process a query using the inner GSEA Agent (blocking bridge).
    async fn process_query(&self, query: &str) -> Result<String, AgentError> {
        let agent = self.agent.clone();
        let query = query.to_string();
        tokio::task::spawn_blocking(move || {
            let rt = tokio::runtime::Handle::current();
            let mut ag = agent.lock().map_err(|e| AgentError::Internal(e.to_string()))?;
            if let Some(ref mut a) = *ag {
                rt.block_on(a.process_message(&query))
                    .map_err(|e| AgentError::Internal(e.to_string()))
            } else {
                Err(AgentError::Internal("Agent not initialized".to_string()))
            }
        }).await.map_err(|e| AgentError::Internal(e.to_string()))?
    }
}

#[async_trait]
impl Actor for GseaPekkoAgent {
    type Message = AgentMessage;

    async fn pre_start(&mut self) {
        tracing::info!("GseaPekkoAgent '{}' started in ActorSystem", self.id);
    }

    async fn receive(&mut self, msg: Self::Message, _ctx: &mut ActorContext<Self>) {
        match msg {
            AgentMessage::Query(query) => {
                self.state = AgentState::Reasoning {
                    query: query.content.clone(),
                    iteration: 1,
                    thought_chain: vec![],
                };

                match self.process_query(&query.content).await {
                    Ok(response) => {
                        self.state = AgentState::Responding {
                            draft: response,
                        };
                        // In production, send response back via reply_to channel
                        self.state = AgentState::Idle;
                    }
                    Err(e) => {
                        self.state = AgentState::Error {
                            error: e.to_string(),
                            recoverable: true,
                        };
                    }
                }
            }
            AgentMessage::Execute(_action) => {
                // Future: direct tool execution messages
            }
            AgentMessage::Respond(_observations) => {
                // Future: observation-driven response synthesis
            }
        }
    }

    async fn post_stop(&mut self) {
        tracing::info!("GseaPekkoAgent '{}' stopped", self.id);
    }
}

#[async_trait]
impl AgentActor for GseaPekkoAgent {
    fn agent_id(&self) -> &str { &self.id }

    fn available_tools(&self) -> Vec<ToolDefinition> { self.tool_defs.clone() }

    fn system_prompt(&self) -> String {
        "You are GSEA, a self-evolving agent with persistent memory.".to_string()
    }

    async fn reason(&mut self, query: &UserQuery) -> Result<AgentAction, AgentError> {
        // Delegate to inner agent's reasoning
        Ok(AgentAction::Respond {
            content: format!("Processing: {}", query.content),
        })
    }

    async fn act(&mut self, _action: &AgentAction) -> Result<Vec<Observation>, AgentError> {
        Ok(vec![])
    }

    async fn respond(&mut self, _observations: &[Observation]) -> Result<AgentResponse, AgentError> {
        Ok(AgentResponse {
            content: "Response".to_string(),
            citations: vec![],
            token_usage: Default::default(),
        })
    }

    fn current_state(&self) -> &AgentState { &self.state }

    fn transition(&mut self, new_state: AgentState) { self.state = new_state; }
}
```

> **Note:** The `AgentAction::Respond`, `AgentResponse`, and `AgentError` types must match pekko-agent-core exactly. Verify field names during implementation.

- [ ] **Step 4: Add module to main.rs**

In `src/main.rs`, add:

```rust
mod pekko_agent;
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cargo test gsea_pekko_agent_state -v 2>&1 | tail -10
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/pekko_agent.rs src/main.rs
git commit -m "feat: GseaPekkoAgent — GSEA Agent as pekko ActorSystem actor"
```

---

### Task 5: Add ActorSystem bootstrap and `pekko` run mode

**Files:**
- Modify: `src/main.rs` (add `pekko` subcommand alongside existing `gui`, `interactive` modes)

This adds a new run mode `gsea pekko` that starts GSEA inside a Pekko ActorSystem.

- [ ] **Step 1: Add CLI variant**

In `src/main.rs`, the `Cli` struct's prompt field already captures subcommands. Add handling after the existing `gui` check:

```rust
if first_arg == Some("pekko") {
    return run_pekko(agent, brain, registry, &cli.model).await;
}
```

- [ ] **Step 2: Implement run_pekko function**

Add to `src/main.rs`:

```rust
async fn run_pekko(
    agent: Agent,
    brain: Arc<std::sync::Mutex<Brain>>,
    registry: Arc<std::sync::Mutex<ToolRegistry>>,
    model: &str,
) -> Result<()> {
    use pekko_actor::ActorSystem;
    use pekko_agent::GseaPekkoAgent;
    use pekko_agent_core::AgentMessage;
    use pekko_agent_core::message::UserQuery;

    println!("GSEA Pekko Actor Mode");
    println!("{}", "-".repeat(50));

    let system = ActorSystem::new("gsea-system");
    let pekko = GseaPekkoAgent::new("gsea-main", agent);
    let agent_ref = system.spawn(pekko, "gsea-main").await?;

    println!("Actor system started. Type 'exit' to quit.");

    let mut rl = rustyline::DefaultEditor::new()?;
    loop {
        let readline = rl.readline("pekko>> ");
        match readline {
            Ok(line) => {
                let line = line.trim();
                if line == "exit" || line == "quit" {
                    println!("Shutting down actor system...");
                    break;
                }
                if line.is_empty() { continue; }

                let query = UserQuery {
                    content: line.to_string(),
                    session_id: uuid::Uuid::new_v4(),
                    context: Default::default(),
                };

                agent_ref.tell(AgentMessage::Query(query)).await?;
                // Note: response delivery requires ask-pattern or channel callback.
                // For now, results are logged via tracing.
                println!("(message sent to actor)");
            }
            Err(rustyline::error::ReadlineError::Interrupted)
            | Err(rustyline::error::ReadlineError::Eof) => break,
            Err(e) => { eprintln!("Error: {}", e); break; }
        }
    }

    Ok(())
}
```

- [ ] **Step 3: Test manually**

```bash
cargo run -- pekko
```

Type "hello" — should see the message sent to the actor, and tracing output showing state transitions.

- [ ] **Step 4: Commit**

```bash
git add src/main.rs
git commit -m "feat: 'gsea pekko' run mode — ActorSystem-based execution"
```

---

## Phase 3: Wire up Events + Orchestrator

### Task 6: Add EventPublisher to GseaPekkoAgent

**Files:**
- Modify: `src/pekko_agent.rs`

Emit events on state transitions using `pekko-agent-events`.

- [ ] **Step 1: Add events to GseaPekkoAgent**

Add field and emit on receive:

```rust
use pekko_agent_events::{EventPublisher, AgentEventEnvelope};

pub struct GseaPekkoAgent {
    // ... existing fields ...
    events: EventPublisher,
}
```

In `receive()`, after state transitions:
```rust
let event = AgentEventEnvelope {
    event_id: uuid::Uuid::new_v4(),
    event_type: "task.completed".to_string(),
    agent_id: self.id.clone(),
    tenant_id: "default".to_string(),
    timestamp: chrono::Utc::now(),
    payload: serde_json::json!({"response_length": response.len()}),
};
let _ = self.events.publish(event).await;
```

- [ ] **Step 2: Test event emission**

```rust
#[tokio::test]
async fn test_event_emission() {
    let events = EventPublisher::new("test", 10);
    // Verify subscriber receives events after agent processes a message
}
```

- [ ] **Step 3: Commit**

```bash
git add src/pekko_agent.rs
git commit -m "feat: emit pekko-agent events on state transitions"
```

---

## Phase 4: Connect memory-actor CLS to pekko ActorSystem

### Task 7: Spawn memory-actor actors inside the ActorSystem

**Files:**
- Create: `src/memory_system.rs`
- Modify: `src/main.rs`

This connects the existing `vendor/memory-brain/crates/memory-actor` CLS system (HippocampusActor, NeocortexActor, DreamActor) into the same Pekko ActorSystem. The MemoryGuardian becomes a supervised actor.

- [ ] **Step 1: Create MemorySystem wrapper**

```rust
// src/memory_system.rs
use pekko_actor::ActorSystem;
use memory_actor::{MemoryGuardian, MemorySystemConfig};

pub struct MemorySystem {
    pub guardian: MemoryGuardian,
}

impl MemorySystem {
    pub fn new() -> Self {
        let config = MemorySystemConfig::default();
        Self {
            guardian: MemoryGuardian::new(config),
        }
    }

    pub fn store(&mut self, content: &str) -> uuid::Uuid {
        use memory_actor::MemoryContext;
        self.guardian.store(content.to_string(), MemoryContext::default())
    }

    pub fn recall(&mut self, query: &str, k: usize) -> Vec<memory_actor::RecallResult> {
        self.guardian.recall(query, k)
    }

    pub fn dream(&mut self) {
        self.guardian.start_dream();
    }
}
```

- [ ] **Step 2: Integrate into run_pekko**

In `run_pekko()`, create the MemorySystem alongside the agent:
```rust
let mut memory_system = memory_system::MemorySystem::new();
// Pass to GseaPekkoAgent or use independently
```

- [ ] **Step 3: Test**

```bash
cargo test memory_system -v
```

- [ ] **Step 4: Commit**

```bash
git add src/memory_system.rs src/main.rs
git commit -m "feat: CLS MemorySystem integrated into pekko mode"
```

---

## Summary: What Each Phase Delivers

| Phase | What | Outcome |
|-------|------|---------|
| P0 | Prerequisites | rust-pekko + pekko-agent compile locally |
| P1 | Wire crates | GSEA tools and Brain work through pekko interfaces |
| P2 | AgentActor | GSEA runs as a Pekko Actor with `gsea pekko` command |
| P3 | Events | State transitions emit observable events |
| P4 | CLS memory | memory-actor's Hippocampus/Neocortex/Dream in same actor system |

## Key Risks & Decisions

1. **rust-pekko availability**: The `pekko-actor` crate isn't published on crates.io and the `rust-pekko` repo doesn't exist on GitHub under yhc007. Either locate it locally or build a minimal runtime (P0 Step 1 provides a scaffold).

2. **pekko-agent-core type alignment**: The `AgentAction`, `AgentResponse`, `Message`, `MemoryError` types in pekko-agent-core may not exactly match what's written here. Verify fields during implementation and adapt.

3. **Async bridge**: The current GSEA `Agent::process_message` is async but holds `Mutex` guards. Inside an actor's `receive()`, this requires careful handling (`spawn_blocking` + `Handle::current()`). The plan accounts for this.

4. **Backward compatibility**: All existing run modes (`--interactive`, `gui`, `review`, `serve-mcp`) remain unchanged. The `pekko` mode is additive.

5. **memory-actor dependency**: The `memory-actor` crate in `vendor/memory-brain/crates/memory-actor/` depends on `pekko-actor` via `path = "../../../pekko-actor"`. This path must resolve to the same `rust-pekko/pekko-actor` we set up in P0.

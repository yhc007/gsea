/// GseaPekkoAgent — wraps the GSEA `Agent` behind the `pekko_agent_core::AgentActor` trait.
///
/// Architecture
/// ────────────
/// * `GseaPekkoAgent` is a `pekko_actor::Actor<Message = AgentMessage>` so it
///   can be spawned inside a `pekko_actor::ActorSystem`.
/// * It also implements `pekko_agent_core::AgentActor`, exposing the ReAct-loop
///   interface (reason / act / respond) on top of GSEA's single
///   `Agent::process_message` entry point.
/// * The inner `Agent` is wrapped in `Arc<tokio::sync::Mutex<Option<…>>>` so we
///   can take a mutable reference for `process_message` while the outer struct
///   remains `Send + Sync`.
///
/// ## Send-safety note
///
/// GSEA's `Agent::execute_tool_chain` holds a `std::sync::MutexGuard<ToolRegistry>`
/// across an `.await` point, making the future `!Send`.  Because
/// `pekko_agent_core::AgentActor` requires `+ Send` futures (via `#[async_trait]`)
/// we delegate `process_message` to a dedicated `tokio::spawn` task: we move the
/// Agent into the task, run the call, then return both the result and the Agent
/// through a one-shot channel.  This keeps our own futures Send while the inner
/// `!Send` work executes safely on a single thread pool worker.
use std::sync::Arc;

use async_trait::async_trait;
use pekko_agent_core::{
    AgentAction, AgentError, AgentMessage, AgentResponse, AgentState, Observation,
    ToolDefinition, UserQuery,
};
use pekko_actor::{Actor, ActorContext};
use pekko_agent_events::{
    AgentEventEnvelope,
    EventPublisher,
    schema::event_types,
};
use tokio::sync::Mutex;
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::agent::Agent;

// ─── GseaPekkoAgent ──────────────────────────────────────────────────────────

/// Actor that wraps GSEA's `Agent` and satisfies `pekko_agent_core::AgentActor`.
pub struct GseaPekkoAgent {
    /// Unique actor/agent identifier (e.g. `"gsea-main"`).
    pub id: String,

    /// Role of this agent (e.g. "main", "coder", "reviewer", "tester").
    pub role: String,

    /// Current FSM state.
    pub state: AgentState,

    /// The inner GSEA agent.
    ///
    /// Wrapped in `Arc<Mutex<Option<_>>>` so `run_inner` can temporarily move
    /// the `Agent` into a `tokio::spawn` task (to satisfy `Send` bounds), then
    /// place it back once the call completes.
    pub agent: Arc<Mutex<Option<Agent>>>,

    /// Cached `ToolDefinition`s built from the agent's `ToolRegistry` at
    /// construction time.
    pub tool_defs: Vec<ToolDefinition>,

    /// Event publisher — emits `AgentEventEnvelope` messages on every state
    /// transition so downstream consumers can observe the agent's FSM.
    pub events: EventPublisher,

    /// Optional monitoring tracker — updated on every state transition.
    pub monitor: Option<monitoring_bridge::AgentTracker>,
}

// ─── Constructor ─────────────────────────────────────────────────────────────

impl GseaPekkoAgent {
    /// Create a new `GseaPekkoAgent`, snapshotting `ToolDefinition`s from the
    /// inner `Agent`'s `ToolRegistry`.
    pub fn new(id: &str, agent: Agent) -> Self {
        Self::new_with_role(id, "main", agent)
    }

    /// Create a new `GseaPekkoAgent` with a specific role.
    pub fn new_with_role(id: &str, role: &str, agent: Agent) -> Self {
        // Snapshot tool definitions before moving agent into the Arc.
        let tool_defs: Vec<ToolDefinition> = {
            let registry = agent.tools.lock().expect("tool registry lock poisoned");
            registry
                .list_tools()
                .into_iter()
                .map(|t| ToolDefinition {
                    name: t.name().to_string(),
                    description: t.description().to_string(),
                    input_schema: t.parameters(),
                    required_permissions: vec![],
                    timeout_ms: 30_000,
                    idempotent: false,
                })
                .collect()
        };
        let agent_arc = Arc::new(Mutex::new(Some(agent)));
        Self::new_with_shared_agent(id, role, agent_arc, tool_defs)
    }

    /// Create a new `GseaPekkoAgent` with a pre-built shared agent Arc and
    /// pre-computed tool definitions.
    ///
    /// This allows external code (e.g. streaming workflows) to hold a clone
    /// of the `Arc<Mutex<Option<Agent>>>` and temporarily borrow the agent
    /// for streaming calls, while the actor system owns the `GseaPekkoAgent`.
    ///
    /// Tool definitions must be extracted before calling this (to avoid
    /// blocking inside a tokio runtime).
    pub fn new_with_shared_agent(
        id: &str,
        role: &str,
        agent_arc: Arc<Mutex<Option<Agent>>>,
        tool_defs: Vec<ToolDefinition>,
    ) -> Self {
        Self {
            id: id.to_string(),
            role: role.to_string(),
            state: AgentState::Idle,
            agent: agent_arc,
            tool_defs,
            events: EventPublisher::new("agent-events", 100),
            monitor: None,
        }
    }

    pub fn with_monitor(mut self, tracker: monitoring_bridge::AgentTracker) -> Self {
        self.monitor = Some(tracker);
        self
    }

    // ── Send-safe inner dispatch ───────────────────────────────────────────────
    //
    // GSEA's `process_message` holds a `std::sync::MutexGuard<ToolRegistry>`
    // across `.await`, making its future `!Send`.  We bridge this by moving the
    // `Agent` out of our `Mutex<Option<Agent>>` into a `tokio::spawn` task (which
    // is `Send + 'static`), running the call there, and returning both the result
    // and the agent through a one-shot channel.

    /// Run `Agent::process_message(input)`, restoring the agent to its slot
    /// afterwards.
    ///
    /// ## Thread safety
    ///
    /// GSEA's `process_message` holds a `std::sync::MutexGuard<ToolRegistry>`
    /// across an `.await` point, making its future `!Send`.  Because
    /// `AgentActor` (via `#[async_trait]`) requires `Send` futures we use
    /// `tokio::task::block_in_place` + `Handle::block_on` to drive the inner
    /// `!Send` future on the current thread, which is moved out of the async
    /// worker pool for the duration of the call.
    ///
    /// Errors:
    /// * `AgentError::Internal` if the inner agent slot is empty.
    /// * `AgentError::Internal` wrapping any `anyhow::Error` from `process_message`.
    async fn run_inner(&self, input: &str) -> Result<String, AgentError> {
        // Take the agent out of its Option slot while we hold the mutex.
        let mut agent = {
            let mut guard = self.agent.lock().await;
            guard.take().ok_or_else(|| {
                AgentError::Internal(anyhow::anyhow!("inner agent slot is empty"))
            })?
        };
        // The mutex is released here so other callers can queue.

        let input_owned = input.to_string();

        // `block_in_place` moves the current thread out of the async worker pool
        // so it is safe to call `Handle::block_on` with a `!Send` future.
        // We return both the result and the agent so we can restore the slot.
        let (result, returned_agent) = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async move {
                let r = agent.process_message(&input_owned).await;
                (r, agent)
            })
        });

        // Restore the agent to its slot.
        {
            let mut guard = self.agent.lock().await;
            *guard = Some(returned_agent);
        }

        result.map_err(AgentError::Internal)
    }

    // ── Internal state mutator ────────────────────────────────────────────────
    //
    // `AgentActor::transition` is defined on the trait and satisfies the same
    // role, but because `Actor::receive` cannot call trait methods on `self`
    // without an explicit `use pekko_agent_core::AgentActor`, we also expose
    // the raw mutation here so `receive` can drive FSM changes without a
    // circular import concern.

    fn set_state(&mut self, new_state: AgentState) {
        info!(
            agent_id = %self.id,
            from = ?self.state,
            to   = ?new_state,
            "State transition"
        );
        if let Some(ref tracker) = self.monitor {
            let label = match &new_state {
                AgentState::Idle => monitoring_core::AgentStateLabel::Idle,
                AgentState::Reasoning { .. } => monitoring_core::AgentStateLabel::Reasoning,
                AgentState::Acting { .. } => monitoring_core::AgentStateLabel::Acting,
                AgentState::Observing { .. } => monitoring_core::AgentStateLabel::Observing,
                AgentState::Responding { .. } => monitoring_core::AgentStateLabel::Responding,
                AgentState::Error { .. } => monitoring_core::AgentStateLabel::Error,
            };
            tracker.update_state(&self.id, label);
        }
        self.state = new_state;
    }

    /// Emit a `state.changed` event for the current FSM state.
    ///
    /// Failures are logged but not propagated — event publishing is
    /// best-effort and must not abort the actor's message loop.
    async fn emit_state_event(&self, correlation_id: Uuid) {
        let state_label = match &self.state {
            AgentState::Idle => "idle",
            AgentState::Reasoning { .. } => "reasoning",
            AgentState::Acting { .. } => "acting",
            AgentState::Observing { .. } => "observing",
            AgentState::Responding { .. } => "responding",
            AgentState::Error { .. } => "error",
        };

        let envelope = AgentEventEnvelope::new(
            &self.id,
            event_types::STATE_CHANGED,
            "default",
            correlation_id,
            serde_json::json!({ "state": state_label, "agent_id": self.id }),
        );

        if let Err(e) = self.events.publish(envelope).await {
            warn!(agent_id = %self.id, error = %e, "Failed to publish state-change event");
        }
    }
}

// ─── pekko_actor::Actor impl ──────────────────────────────────────────────────
//
// `pekko_actor::Actor` uses RPIT (`impl Future`) for its methods — no
// `#[async_trait]` needed or wanted here.

impl Actor for GseaPekkoAgent {
    type Message = AgentMessage;

    async fn pre_start(&mut self) {
        info!(agent_id = %self.id, "GseaPekkoAgent starting up");
    }

    async fn receive(&mut self, msg: AgentMessage, ctx: &mut ActorContext<Self>) {
        match msg {
            AgentMessage::Query(query) => {
                info!(
                    agent_id = %self.id,
                    actor    = %ctx.name(),
                    session  = %query.session_id,
                    "Received Query"
                );
                if let Some(ref m) = self.monitor { m.record_received(&self.id); }

                let correlation_id = query.session_id;
                self.set_state(AgentState::Reasoning {
                    query: query.content.clone(),
                    iteration: 1,
                    thought_chain: vec![],
                });
                self.emit_state_event(correlation_id).await;

                match self.run_inner(&query.content).await {
                    Ok(reply) => {
                        info!(
                            agent_id = %self.id,
                            chars    = reply.len(),
                            "Query processed successfully"
                        );
                        if let Some(ref m) = self.monitor { m.record_sent(&self.id); }
                        self.set_state(AgentState::Idle);
                        self.emit_state_event(correlation_id).await;
                    }
                    Err(e) => {
                        error!(agent_id = %self.id, error = %e, "Error processing query");
                        if let Some(ref m) = self.monitor { m.record_error(&self.id); }
                        self.set_state(AgentState::Error {
                            error: e.to_string(),
                            recoverable: true,
                        });
                        self.emit_state_event(correlation_id).await;
                    }
                }
            }

            AgentMessage::QueryWithReply(query, reply_tx) => {
                info!(
                    agent_id = %self.id,
                    actor    = %ctx.name(),
                    session  = %query.session_id,
                    "Received QueryWithReply"
                );
                if let Some(ref m) = self.monitor { m.record_received(&self.id); }

                let correlation_id = query.session_id;
                self.set_state(AgentState::Reasoning {
                    query: query.content.clone(),
                    iteration: 1,
                    thought_chain: vec![],
                });
                self.emit_state_event(correlation_id).await;

                match self.run_inner(&query.content).await {
                    Ok(reply) => {
                        info!(
                            agent_id = %self.id,
                            chars    = reply.len(),
                            "QueryWithReply processed successfully"
                        );
                        if let Some(ref m) = self.monitor { m.record_sent(&self.id); }
                        self.set_state(AgentState::Idle);
                        self.emit_state_event(correlation_id).await;
                        let _ = reply_tx.send(Ok(reply));
                    }
                    Err(e) => {
                        error!(agent_id = %self.id, error = %e, "Error processing query");
                        if let Some(ref m) = self.monitor { m.record_error(&self.id); }
                        self.set_state(AgentState::Error {
                            error: e.to_string(),
                            recoverable: true,
                        });
                        self.emit_state_event(correlation_id).await;
                        let _ = reply_tx.send(Err(e.to_string()));
                    }
                }
            }

            AgentMessage::Execute(action) => {
                info!(
                    agent_id = %self.id,
                    actor    = %ctx.name(),
                    "Received Execute"
                );

                let correlation_id = Uuid::new_v4();
                let prompt = match &action {
                    AgentAction::UseTool(calls) => {
                        let names: Vec<_> = calls.iter().map(|c| c.name.as_str()).collect();
                        format!("Execute tools: {}", names.join(", "))
                    }
                    AgentAction::Respond(text) => text.clone(),
                    AgentAction::DelegateToAgent { target_agent, task } => {
                        format!("Delegate to {}: {}", target_agent, task.description)
                    }
                };

                self.set_state(AgentState::Acting {
                    tool_calls: vec![],
                    pending: 0,
                });
                self.emit_state_event(correlation_id).await;

                match self.run_inner(&prompt).await {
                    Ok(_) => {
                        self.set_state(AgentState::Idle);
                        self.emit_state_event(correlation_id).await;
                    }
                    Err(e) => {
                        error!(agent_id = %self.id, error = %e, "Error executing action");
                        self.set_state(AgentState::Error {
                            error: e.to_string(),
                            recoverable: true,
                        });
                        self.emit_state_event(correlation_id).await;
                    }
                }
            }

            AgentMessage::Respond(observations) => {
                info!(
                    agent_id = %self.id,
                    actor  = %ctx.name(),
                    count  = observations.len(),
                    "Received Respond with observations"
                );

                let correlation_id = Uuid::new_v4();
                let obs_text = observations
                    .iter()
                    .map(|o| {
                        format!(
                            "[{}] {}: {}",
                            o.tool_name,
                            if o.is_error { "ERROR" } else { "OK" },
                            serde_json::to_string(&o.result).unwrap_or_default()
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                let prompt = format!(
                    "Synthesise a final response from these tool observations:\n{}",
                    obs_text
                );

                self.set_state(AgentState::Responding {
                    draft: obs_text.clone(),
                });
                self.emit_state_event(correlation_id).await;

                match self.run_inner(&prompt).await {
                    Ok(_) => {
                        self.set_state(AgentState::Idle);
                        self.emit_state_event(correlation_id).await;
                    }
                    Err(e) => {
                        error!(agent_id = %self.id, error = %e, "Error synthesising response");
                        self.set_state(AgentState::Error {
                            error: e.to_string(),
                            recoverable: true,
                        });
                        self.emit_state_event(correlation_id).await;
                    }
                }
            }
        }
    }

    async fn post_stop(&mut self) {
        info!(agent_id = %self.id, "GseaPekkoAgent shutting down");
    }
}

// ─── pekko_agent_core::AgentActor impl ───────────────────────────────────────

#[async_trait]
impl pekko_agent_core::AgentActor for GseaPekkoAgent {
    // ── Identity & metadata ───────────────────────────────────────────────────

    fn agent_id(&self) -> &str {
        &self.id
    }

    fn available_tools(&self) -> Vec<ToolDefinition> {
        self.tool_defs.clone()
    }

    fn system_prompt(&self) -> String {
        String::from(
            "You are GSEA — a self-evolving Rust engineering agent powered by a local LLM.\n\
             You have access to a MemoryBrain that stores your experiences, learnings, and skills.\n\
             Your ultimate goal is to improve your own capabilities over time by:\n\
             1. Writing and testing Rust code\n\
             2. Saving useful patterns as skills\n\
             3. Reflecting on what works and what doesn't\n\
             4. Generating and applying improvements to your own codebase",
        )
    }

    // ── ReAct loop ────────────────────────────────────────────────────────────

    /// Reasoning step: forward the user query to the inner agent and wrap the
    /// reply as `AgentAction::Respond`.
    ///
    /// Future iterations can parse structured tool-call JSON from the LLM reply
    /// and return `AgentAction::UseTool` instead.
    async fn reason(&mut self, query: &UserQuery) -> Result<AgentAction, AgentError> {
        let reply = self.run_inner(&query.content).await?;
        Ok(AgentAction::Respond(reply))
    }

    /// Action step: execute tool calls from the chosen `AgentAction`.
    ///
    /// For `AgentAction::Respond` (the common initial case) there are no tool
    /// calls — return an empty observation list so the FSM advances to
    /// `Responding` immediately.
    async fn act(&mut self, action: &AgentAction) -> Result<Vec<Observation>, AgentError> {
        match action {
            AgentAction::Respond(_) => {
                // Nothing to execute; the response content is already in `action`.
                Ok(vec![])
            }
            AgentAction::UseTool(calls) => {
                let mut observations = Vec::with_capacity(calls.len());
                for call in calls {
                    let prompt = format!(
                        "Execute tool '{}' with params: {}",
                        call.name,
                        serde_json::to_string(&call.input).unwrap_or_default()
                    );

                    let start = std::time::Instant::now();
                    let result = self.run_inner(&prompt).await;
                    let duration_ms = start.elapsed().as_millis() as u64;

                    let (result_value, is_error) = match result {
                        Ok(text) => (serde_json::json!({ "output": text }), false),
                        Err(e) => (serde_json::json!({ "error": e.to_string() }), true),
                    };

                    observations.push(Observation {
                        tool_call_id: call.id.clone(),
                        tool_name: call.name.clone(),
                        result: result_value,
                        is_error,
                        duration_ms,
                    });
                }
                Ok(observations)
            }
            AgentAction::DelegateToAgent { target_agent, task } => {
                warn!(
                    agent_id = %self.id,
                    target   = %target_agent,
                    "Delegation not yet implemented — returning pending observation"
                );
                Ok(vec![Observation {
                    tool_call_id: task.task_id.to_string(),
                    tool_name: format!("delegate:{}", target_agent),
                    result: serde_json::json!({ "status": "delegation_pending" }),
                    is_error: false,
                    duration_ms: 0,
                }])
            }
        }
    }

    /// Response step: synthesise a final `AgentResponse` from observations.
    ///
    /// When `observations` is empty (i.e. `reason` returned `Respond` directly)
    /// we use a placeholder; a future iteration will thread the reply content
    /// through a dedicated field on the action enum.
    async fn respond(
        &mut self,
        observations: &[Observation],
    ) -> Result<AgentResponse, AgentError> {
        let content = if observations.is_empty() {
            // The final reply was already produced by `reason` — return a
            // minimal acknowledgement until we wire up a reply channel.
            "Response generated.".to_string()
        } else {
            let obs_text = observations
                .iter()
                .map(|o| {
                    format!(
                        "[{}] {}: {}",
                        o.tool_name,
                        if o.is_error { "ERROR" } else { "OK" },
                        serde_json::to_string(&o.result).unwrap_or_default()
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");

            let prompt = format!(
                "Synthesise a clear, concise final answer from these tool results:\n{}",
                obs_text
            );
            self.run_inner(&prompt).await?
        };

        Ok(AgentResponse {
            content,
            citations: vec![],
            suggested_actions: vec![],
            token_usage: pekko_agent_core::TokenUsage::default(),
        })
    }

    // ── FSM ───────────────────────────────────────────────────────────────────

    fn current_state(&self) -> &AgentState {
        &self.state
    }

    fn transition(&mut self, new_state: AgentState) {
        self.set_state(new_state);
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use pekko_agent_core::{AgentActor, AgentState};

    // ── Stub constructor ──────────────────────────────────────────────────────
    //
    // We cannot construct a real `Agent` in unit tests (requires Ollama + SQLite),
    // so we test FSM and metadata logic in isolation by building the struct
    // directly with an empty agent slot.

    fn make_stub(id: &str) -> GseaPekkoAgent {
        GseaPekkoAgent {
            id: id.to_string(),
            role: "main".to_string(),
            state: AgentState::Idle,
            agent: Arc::new(Mutex::new(None)), // intentionally empty
            tool_defs: vec![],
            events: pekko_agent_events::EventPublisher::new("agent-events", 100),
            monitor: None,
        }
    }

    // ── Identity & metadata ───────────────────────────────────────────────────

    #[test]
    fn agent_id_matches_constructor_arg() {
        let agent = make_stub("gsea-main");
        assert_eq!(agent.agent_id(), "gsea-main");
    }

    #[test]
    fn available_tools_empty_on_stub() {
        let agent = make_stub("gsea-main");
        assert!(agent.available_tools().is_empty());
    }

    #[test]
    fn system_prompt_is_non_empty_and_mentions_gsea() {
        let agent = make_stub("gsea-main");
        let prompt = agent.system_prompt();
        assert!(!prompt.is_empty());
        assert!(prompt.contains("GSEA"));
    }

    #[test]
    fn max_iterations_defaults_to_ten() {
        let agent = make_stub("gsea-main");
        assert_eq!(agent.max_iterations(), 10);
    }

    // ── FSM: initial state ────────────────────────────────────────────────────

    #[test]
    fn initial_state_is_idle() {
        let agent = make_stub("fsm-test");
        assert!(agent.current_state().is_idle());
        assert!(!agent.current_state().is_busy());
    }

    // ── FSM: individual transitions ───────────────────────────────────────────

    #[test]
    fn transition_idle_to_reasoning() {
        let mut agent = make_stub("fsm-test");
        agent.transition(AgentState::Reasoning {
            query: "hello".to_string(),
            iteration: 1,
            thought_chain: vec![],
        });
        assert!(agent.current_state().is_busy());
        assert!(!agent.current_state().is_idle());
    }

    #[test]
    fn transition_idle_to_acting_to_idle() {
        let mut agent = make_stub("fsm-test");

        agent.transition(AgentState::Acting {
            tool_calls: vec![],
            pending: 0,
        });
        assert!(agent.current_state().is_busy());

        agent.transition(AgentState::Idle);
        assert!(agent.current_state().is_idle());
    }

    #[test]
    fn transition_to_observing() {
        let mut agent = make_stub("fsm-test");
        agent.transition(AgentState::Observing {
            observations: vec![],
            needs_more: false,
        });
        assert!(agent.current_state().is_busy());
    }

    #[test]
    fn transition_to_responding() {
        let mut agent = make_stub("fsm-test");
        agent.transition(AgentState::Responding {
            draft: "draft text".to_string(),
        });
        assert!(agent.current_state().is_busy());
    }

    #[test]
    fn transition_to_error_and_recover() {
        let mut agent = make_stub("fsm-test");

        agent.transition(AgentState::Error {
            error: "something went wrong".to_string(),
            recoverable: true,
        });
        // Error is neither idle nor busy per AgentState::is_busy definition.
        assert!(!agent.current_state().is_idle());
        assert!(!agent.current_state().is_busy());

        // Recover by transitioning back to Idle.
        agent.transition(AgentState::Idle);
        assert!(agent.current_state().is_idle());
    }

    // ── FSM: full ReAct walkthrough ───────────────────────────────────────────

    #[test]
    fn full_react_state_walkthrough() {
        let mut agent = make_stub("fsm-walkthrough");

        // Idle
        assert!(agent.current_state().is_idle());

        // → Reasoning
        agent.transition(AgentState::Reasoning {
            query: "test query".to_string(),
            iteration: 1,
            thought_chain: vec!["initial thought".to_string()],
        });
        assert!(agent.current_state().is_busy());

        // → Acting
        agent.transition(AgentState::Acting {
            tool_calls: vec![],
            pending: 0,
        });
        assert!(agent.current_state().is_busy());

        // → Observing
        agent.transition(AgentState::Observing {
            observations: vec![],
            needs_more: false,
        });
        assert!(agent.current_state().is_busy());

        // → Responding
        agent.transition(AgentState::Responding {
            draft: "draft response".to_string(),
        });
        assert!(agent.current_state().is_busy());

        // → Idle
        agent.transition(AgentState::Idle);
        assert!(agent.current_state().is_idle());
    }

    // ── Async: run_inner errors when slot is empty ────────────────────────────

    #[tokio::test]
    async fn run_inner_errors_when_agent_slot_is_empty() {
        let agent = make_stub("gsea-main");
        let result = agent.run_inner("hello").await;
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        // The error should mention that the slot is empty.
        assert!(msg.contains("empty") || msg.contains("Internal"));
    }

    // ── Async: act with Respond action returns empty observations ─────────────

    #[tokio::test]
    async fn act_respond_action_produces_no_observations() {
        let mut agent = make_stub("gsea-main");
        let action = AgentAction::Respond("hello world".to_string());
        let obs = agent.act(&action).await.unwrap();
        assert!(obs.is_empty());
    }

    // ── Async: act with Delegate action returns pending observation ───────────

    #[tokio::test]
    async fn act_delegate_action_produces_pending_observation() {
        use pekko_agent_core::{AgentTask, TaskPriority};
        use uuid::Uuid;

        let mut agent = make_stub("gsea-main");
        let action = AgentAction::DelegateToAgent {
            target_agent: "another-agent".to_string(),
            task: AgentTask {
                task_id: Uuid::new_v4(),
                description: "do something".to_string(),
                input: serde_json::json!({}),
                priority: TaskPriority::Normal,
                timeout_ms: 5000,
            },
        };

        let obs = agent.act(&action).await.unwrap();
        assert_eq!(obs.len(), 1);
        assert!(!obs[0].is_error);
        assert!(obs[0].tool_name.starts_with("delegate:"));
        assert_eq!(
            obs[0].result["status"].as_str().unwrap(),
            "delegation_pending"
        );
    }

    // ── Async: respond with empty observations returns placeholder ────────────

    #[tokio::test]
    async fn respond_with_empty_observations_uses_placeholder() {
        let mut agent = make_stub("gsea-main");
        let response = agent.respond(&[]).await.unwrap();
        assert!(!response.content.is_empty());
        assert!(response.citations.is_empty());
    }

    // ── EventPublisher: events are emitted on state transitions ──────────────

    #[tokio::test]
    async fn emit_state_event_publishes_to_bus() {
        use pekko_event_bus::EventBusHandle;

        let agent = make_stub("event-test");

        // Capture the bus handle before emitting so we can read back messages.
        let bus_handle: EventBusHandle = agent.events.bus_handle();

        // Emit a `state.changed` event directly.
        agent.emit_state_event(uuid::Uuid::new_v4()).await;

        // The bus should now have exactly one message on the "agent-events" topic.
        let messages = bus_handle
            .read_all("agent-events")
            .expect("read_all should succeed");
        assert_eq!(messages.len(), 1, "expected exactly one published event");

        // Deserialise and verify the envelope fields.
        let envelope: pekko_agent_events::AgentEventEnvelope =
            serde_json::from_slice(&messages[0]).expect("envelope must deserialise");
        assert_eq!(envelope.source_service, "event-test");
        assert_eq!(envelope.event_type, pekko_agent_events::schema::event_types::STATE_CHANGED);
    }

    #[tokio::test]
    async fn transition_emits_state_event_via_helper() {
        let mut agent = make_stub("event-test-2");
        let bus_handle = agent.events.bus_handle();

        // Transition to Reasoning and emit.
        agent.set_state(AgentState::Reasoning {
            query: "test".to_string(),
            iteration: 1,
            thought_chain: vec![],
        });
        agent.emit_state_event(uuid::Uuid::new_v4()).await;

        // Transition back to Idle and emit.
        agent.set_state(AgentState::Idle);
        agent.emit_state_event(uuid::Uuid::new_v4()).await;

        let messages = bus_handle.read_all("agent-events").expect("read_all");
        assert_eq!(messages.len(), 2, "expected two events (reasoning + idle)");

        // First event should say reasoning.
        let first: pekko_agent_events::AgentEventEnvelope =
            serde_json::from_slice(&messages[0]).unwrap();
        assert_eq!(first.payload["state"].as_str().unwrap(), "reasoning");

        // Second event should say idle.
        let second: pekko_agent_events::AgentEventEnvelope =
            serde_json::from_slice(&messages[1]).unwrap();
        assert_eq!(second.payload["state"].as_str().unwrap(), "idle");
    }
}

# Core Agent Monitoring System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `core-agent-monitoring`, a standalone Rust WASM-based real-time monitoring dashboard for pekko-agent ecosystems — visualizing agent messages, actor states, circuit breakers, event-bus traffic, and system health.

**Architecture:** The system is a **3-layer design**: (1) a `monitoring-core` library crate that defines a `MetricsCollector` trait + ring-buffer storage for metrics snapshots, (2) a `monitoring-exporter` crate that exposes an HTTP/WebSocket API (axum) for pushing live snapshots to the browser, (3) a `monitoring-ui` crate compiled to WASM via `egui`/`eframe` that renders dashboards in the browser. The core library is injected into the pekko-agent process via a thin integration crate (`monitoring-bridge`) that hooks into existing extension points (EventBus subscribers, CircuitBreaker stats, ToolRegistry stats, Orchestrator callbacks).

**Tech Stack:**
- Rust (workspace with 4 crates)
- `eframe` + `egui` (WASM UI — same stack GSEA already uses)
- `axum` + `tokio-tungstenite` (WebSocket metrics server)
- `serde` + `serde_json` (serialization)
- `pekko-actor`, `pekko-event-bus`, `pekko-agent-core`, `pekko-agent-events` (monitoring targets)
- `wasm-bindgen` + `trunk` (WASM build tooling)

---

## Scope & Sub-Systems

This plan covers 4 crates in a single Cargo workspace:

| Crate | Type | Purpose |
|-------|------|---------|
| `monitoring-core` | lib | MetricsCollector trait, snapshot data model, ring-buffer store |
| `monitoring-bridge` | lib | Hooks into pekko ecosystem, collects metrics |
| `monitoring-exporter` | bin/lib | axum HTTP + WebSocket server for metric export |
| `monitoring-ui` | bin (WASM) | egui dashboard compiled to WASM |

---

## File Structure

```
core-agent-monitoring/
├── Cargo.toml                          # workspace root
├── README.md
├── crates/
│   ├── monitoring-core/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  # re-exports
│   │       ├── snapshot.rs             # MetricsSnapshot, AgentSnapshot, etc.
│   │       ├── collector.rs            # MetricsCollector trait
│   │       ├── store.rs               # RingBufferStore (time-series in memory)
│   │       └── types.rs               # shared enums (AgentStateLabel, CBState, etc.)
│   │
│   ├── monitoring-bridge/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  # re-exports + PekkoMonitor builder
│   │       ├── actor_hooks.rs          # agent message + state tracking
│   │       ├── eventbus_hooks.rs       # subscribe to event-bus topics
│   │       ├── circuit_breaker_hooks.rs # poll CircuitBreaker::stats()
│   │       ├── tool_hooks.rs           # poll ToolRegistry::get_all_stats()
│   │       └── orchestrator_hooks.rs   # task queue + active task tracking
│   │
│   ├── monitoring-exporter/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── main.rs                 # axum server entry (standalone mode)
│   │       ├── lib.rs                  # ExporterServer (embedded mode)
│   │       ├── routes.rs               # REST endpoints: /api/snapshot, /api/agents, etc.
│   │       └── ws.rs                   # WebSocket push: /ws/live
│   │
│   └── monitoring-ui/
│       ├── Cargo.toml
│       ├── index.html                  # trunk entry point
│       └── src/
│           ├── main.rs                 # eframe::WebRunner entry
│           ├── app.rs                  # MonitoringApp struct + eframe::App impl
│           ├── panels/
│           │   ├── mod.rs
│           │   ├── overview.rs         # system overview panel (agent count, health)
│           │   ├── agents.rs           # per-agent detail (state FSM, messages)
│           │   ├── event_bus.rs        # topic traffic, partition distribution
│           │   ├── circuit_breakers.rs # CB state gauges, failure timeline
│           │   ├── tools.rs            # tool call stats, latency histograms
│           │   └── tasks.rs            # orchestrator task queue + timeline
│           └── ws_client.rs            # WebSocket client (WASM-compatible)
│
├── examples/
│   └── demo_metrics.rs                 # generates fake metrics for UI development
│
└── trunk.toml                          # trunk config for WASM build
```

---

## Task 1: Workspace Setup & Core Types

**Files:**
- Create: `core-agent-monitoring/Cargo.toml`
- Create: `core-agent-monitoring/crates/monitoring-core/Cargo.toml`
- Create: `core-agent-monitoring/crates/monitoring-core/src/lib.rs`
- Create: `core-agent-monitoring/crates/monitoring-core/src/types.rs`
- Test: inline `#[cfg(test)]` in `types.rs`

- [ ] **Step 1: Create workspace directory and root Cargo.toml**

```bash
mkdir -p /Volumes/T7/core-agent-monitoring/crates
```

```toml
# /Volumes/T7/core-agent-monitoring/Cargo.toml
[workspace]
resolver = "2"
members = [
    "crates/monitoring-core",
]

[workspace.dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
uuid = { version = "1", features = ["v4", "serde"] }
chrono = { version = "0.4", features = ["serde"] }
anyhow = "1"
```

- [ ] **Step 2: Create monitoring-core crate**

```toml
# /Volumes/T7/core-agent-monitoring/crates/monitoring-core/Cargo.toml
[package]
name = "monitoring-core"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { workspace = true }
serde_json = { workspace = true }
uuid = { workspace = true }
chrono = { workspace = true }
```

- [ ] **Step 3: Write types.rs — shared enums matching pekko types**

```rust
// crates/monitoring-core/src/types.rs
use serde::{Deserialize, Serialize};

/// Mirrors pekko_agent_core::AgentState but without internal data.
/// Used for monitoring display only.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum AgentStateLabel {
    Idle,
    Reasoning,
    Acting,
    Observing,
    Responding,
    Error,
}

impl std::fmt::Display for AgentStateLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Idle => write!(f, "Idle"),
            Self::Reasoning => write!(f, "Reasoning"),
            Self::Acting => write!(f, "Acting"),
            Self::Observing => write!(f, "Observing"),
            Self::Responding => write!(f, "Responding"),
            Self::Error => write!(f, "Error"),
        }
    }
}

/// Mirrors pekko_actor::CircuitBreakerState
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum CBState {
    Closed,
    Open,
    HalfOpen,
}

impl std::fmt::Display for CBState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Closed => write!(f, "Closed"),
            Self::Open => write!(f, "Open"),
            Self::HalfOpen => write!(f, "HalfOpen"),
        }
    }
}

/// Severity level for system alerts
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Info,
    Warning,
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_state_label_display() {
        assert_eq!(AgentStateLabel::Idle.to_string(), "Idle");
        assert_eq!(AgentStateLabel::Reasoning.to_string(), "Reasoning");
        assert_eq!(AgentStateLabel::Error.to_string(), "Error");
    }

    #[test]
    fn test_cb_state_display() {
        assert_eq!(CBState::Closed.to_string(), "Closed");
        assert_eq!(CBState::Open.to_string(), "Open");
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Critical);
    }

    #[test]
    fn test_types_serialize_roundtrip() {
        let state = AgentStateLabel::Acting;
        let json = serde_json::to_string(&state).unwrap();
        let back: AgentStateLabel = serde_json::from_str(&json).unwrap();
        assert_eq!(back, AgentStateLabel::Acting);
    }
}
```

- [ ] **Step 4: Write lib.rs**

```rust
// crates/monitoring-core/src/lib.rs
pub mod types;

pub use types::*;
```

- [ ] **Step 5: Verify it compiles and tests pass**

```bash
cd /Volumes/T7/core-agent-monitoring && cargo test -p monitoring-core
```
Expected: 4 tests pass.

- [ ] **Step 6: Initialize git and commit**

```bash
cd /Volumes/T7/core-agent-monitoring
git init
git add -A
git commit -m "feat: workspace setup + monitoring-core types"
```

---

## Task 2: MetricsSnapshot Data Model

**Files:**
- Create: `crates/monitoring-core/src/snapshot.rs`
- Modify: `crates/monitoring-core/src/lib.rs`
- Test: inline `#[cfg(test)]` in `snapshot.rs`

- [ ] **Step 1: Write the snapshot test**

```rust
// Bottom of crates/monitoring-core/src/snapshot.rs
#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_empty_snapshot() {
        let snap = MetricsSnapshot::new();
        assert!(snap.agents.is_empty());
        assert!(snap.circuit_breakers.is_empty());
        assert!(snap.event_bus_topics.is_empty());
        assert!(snap.tool_stats.is_empty());
    }

    #[test]
    fn test_snapshot_with_agent() {
        let mut snap = MetricsSnapshot::new();
        snap.agents.push(AgentSnapshot {
            agent_id: "coder-1".to_string(),
            agent_type: "coder".to_string(),
            state: AgentStateLabel::Idle,
            messages_received: 42,
            messages_sent: 38,
            errors: 2,
            last_activity: Utc::now(),
            current_task: None,
        });
        assert_eq!(snap.agents.len(), 1);
        assert_eq!(snap.agents[0].agent_id, "coder-1");
    }

    #[test]
    fn test_snapshot_serialization() {
        let snap = MetricsSnapshot::new();
        let json = serde_json::to_string(&snap).unwrap();
        let back: MetricsSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(back.agents.len(), 0);
    }

    #[test]
    fn test_cb_snapshot() {
        let cb = CircuitBreakerSnapshot {
            name: "ollama".to_string(),
            state: CBState::Closed,
            failure_count: 0,
            success_count: 50,
            total_calls: 50,
        };
        assert_eq!(cb.failure_rate(), 0.0);
    }

    #[test]
    fn test_cb_failure_rate() {
        let cb = CircuitBreakerSnapshot {
            name: "ollama".to_string(),
            state: CBState::Open,
            failure_count: 3,
            success_count: 7,
            total_calls: 10,
        };
        assert!((cb.failure_rate() - 30.0).abs() < 0.01);
    }
}
```

- [ ] **Step 2: Run test — verify it fails**

```bash
cargo test -p monitoring-core
```
Expected: FAIL (MetricsSnapshot not defined)

- [ ] **Step 3: Implement snapshot.rs**

```rust
// crates/monitoring-core/src/snapshot.rs
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use crate::types::{AgentStateLabel, CBState, Severity};

/// A point-in-time snapshot of the entire pekko-agent system.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub timestamp: DateTime<Utc>,
    pub agents: Vec<AgentSnapshot>,
    pub circuit_breakers: Vec<CircuitBreakerSnapshot>,
    pub event_bus_topics: Vec<TopicSnapshot>,
    pub tool_stats: Vec<ToolSnapshot>,
    pub tasks: TaskQueueSnapshot,
    pub alerts: Vec<AlertSnapshot>,
}

impl MetricsSnapshot {
    pub fn new() -> Self {
        Self {
            timestamp: Utc::now(),
            agents: Vec::new(),
            circuit_breakers: Vec::new(),
            event_bus_topics: Vec::new(),
            tool_stats: Vec::new(),
            tasks: TaskQueueSnapshot::default(),
            alerts: Vec::new(),
        }
    }
}

/// Per-agent snapshot
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentSnapshot {
    pub agent_id: String,
    pub agent_type: String,
    pub state: AgentStateLabel,
    pub messages_received: u64,
    pub messages_sent: u64,
    pub errors: u64,
    pub last_activity: DateTime<Utc>,
    pub current_task: Option<String>,
}

/// Circuit breaker snapshot
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitBreakerSnapshot {
    pub name: String,
    pub state: CBState,
    pub failure_count: u64,
    pub success_count: u64,
    pub total_calls: u64,
}

impl CircuitBreakerSnapshot {
    pub fn failure_rate(&self) -> f64 {
        if self.total_calls == 0 { 0.0 }
        else { (self.failure_count as f64 / self.total_calls as f64) * 100.0 }
    }
}

/// Event bus topic snapshot
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TopicSnapshot {
    pub topic: String,
    pub subscriber_count: usize,
    pub messages_published: u64,
    pub partition_count: u32,
    pub messages_per_partition: Vec<u64>,
}

/// Tool execution stats snapshot
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolSnapshot {
    pub name: String,
    pub call_count: u64,
    pub success_count: u64,
    pub failure_count: u64,
    pub avg_duration_ms: f64,
    pub total_duration_ms: u64,
}

/// Orchestrator task queue snapshot
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TaskQueueSnapshot {
    pub queued: usize,
    pub running: usize,
    pub completed: u64,
    pub failed: u64,
    pub active_tasks: Vec<ActiveTaskSnapshot>,
}

/// Individual active task
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActiveTaskSnapshot {
    pub task_id: String,
    pub assigned_agent: String,
    pub description: String,
    pub started_at: DateTime<Utc>,
    pub status: String,
}

/// System alert
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlertSnapshot {
    pub severity: Severity,
    pub source: String,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}
```

- [ ] **Step 4: Update lib.rs**

```rust
// crates/monitoring-core/src/lib.rs
pub mod types;
pub mod snapshot;

pub use types::*;
pub use snapshot::*;
```

- [ ] **Step 5: Run tests**

```bash
cargo test -p monitoring-core
```
Expected: 9 tests pass (4 from types + 5 from snapshot).

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: MetricsSnapshot data model with agent, CB, topic, tool snapshots"
```

---

## Task 3: MetricsCollector Trait + RingBufferStore

**Files:**
- Create: `crates/monitoring-core/src/collector.rs`
- Create: `crates/monitoring-core/src/store.rs`
- Modify: `crates/monitoring-core/src/lib.rs`
- Test: inline `#[cfg(test)]`

- [ ] **Step 1: Write collector.rs**

```rust
// crates/monitoring-core/src/collector.rs

use crate::snapshot::MetricsSnapshot;

/// Trait for components that collect metrics from the pekko ecosystem.
///
/// The monitoring bridge implements this — polling various pekko components
/// and assembling a MetricsSnapshot.
pub trait MetricsCollector: Send + Sync {
    /// Collect a point-in-time snapshot of all system metrics.
    fn collect(&self) -> MetricsSnapshot;
}
```

- [ ] **Step 2: Write store tests**

```rust
// Bottom of crates/monitoring-core/src/store.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_push_and_latest() {
        let mut store = RingBufferStore::new(5);
        let snap = MetricsSnapshot::new();
        store.push(snap.clone());
        assert_eq!(store.len(), 1);
        assert!(store.latest().is_some());
    }

    #[test]
    fn test_store_capacity() {
        let mut store = RingBufferStore::new(3);
        for _ in 0..5 {
            store.push(MetricsSnapshot::new());
        }
        assert_eq!(store.len(), 3); // oldest evicted
    }

    #[test]
    fn test_store_recent() {
        let mut store = RingBufferStore::new(10);
        for _ in 0..7 {
            store.push(MetricsSnapshot::new());
        }
        let recent = store.recent(3);
        assert_eq!(recent.len(), 3);
    }

    #[test]
    fn test_store_all() {
        let mut store = RingBufferStore::new(10);
        store.push(MetricsSnapshot::new());
        store.push(MetricsSnapshot::new());
        let all = store.all();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_store_empty() {
        let store = RingBufferStore::new(5);
        assert!(store.latest().is_none());
        assert_eq!(store.len(), 0);
        assert!(store.recent(3).is_empty());
    }
}
```

- [ ] **Step 3: Implement store.rs**

```rust
// crates/monitoring-core/src/store.rs
use std::collections::VecDeque;
use crate::snapshot::MetricsSnapshot;

/// Fixed-capacity ring buffer for time-series snapshots.
///
/// Keeps the last `capacity` snapshots in memory for the dashboard
/// to render time-series charts (e.g. failure rate over time).
pub struct RingBufferStore {
    buffer: VecDeque<MetricsSnapshot>,
    capacity: usize,
}

impl RingBufferStore {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, snapshot: MetricsSnapshot) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(snapshot);
    }

    pub fn latest(&self) -> Option<&MetricsSnapshot> {
        self.buffer.back()
    }

    pub fn recent(&self, n: usize) -> Vec<&MetricsSnapshot> {
        self.buffer.iter().rev().take(n).collect()
    }

    pub fn all(&self) -> Vec<&MetricsSnapshot> {
        self.buffer.iter().collect()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}
```

- [ ] **Step 4: Update lib.rs**

```rust
// crates/monitoring-core/src/lib.rs
pub mod types;
pub mod snapshot;
pub mod collector;
pub mod store;

pub use types::*;
pub use snapshot::*;
pub use collector::MetricsCollector;
pub use store::RingBufferStore;
```

- [ ] **Step 5: Run tests**

```bash
cargo test -p monitoring-core
```
Expected: 14 tests pass.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: MetricsCollector trait + RingBufferStore time-series storage"
```

---

## Task 4: Monitoring Bridge — EventBus + CircuitBreaker Hooks

**Files:**
- Create: `crates/monitoring-bridge/Cargo.toml`
- Create: `crates/monitoring-bridge/src/lib.rs`
- Create: `crates/monitoring-bridge/src/eventbus_hooks.rs`
- Create: `crates/monitoring-bridge/src/circuit_breaker_hooks.rs`
- Modify: `Cargo.toml` (workspace members)
- Test: inline `#[cfg(test)]`

- [ ] **Step 1: Add monitoring-bridge to workspace**

```toml
# Append to workspace members in root Cargo.toml
members = [
    "crates/monitoring-core",
    "crates/monitoring-bridge",
]
```

- [ ] **Step 2: Create monitoring-bridge Cargo.toml**

```toml
# crates/monitoring-bridge/Cargo.toml
[package]
name = "monitoring-bridge"
version = "0.1.0"
edition = "2021"

[dependencies]
monitoring-core = { path = "../monitoring-core" }
serde = { workspace = true }
serde_json = { workspace = true }
uuid = { workspace = true }
chrono = { workspace = true }
anyhow = { workspace = true }

# pekko ecosystem (path deps — same machine)
pekko-actor = { path = "/Volumes/T7/rust-pekko/pekko-actor" }
pekko-event-bus = { path = "/Volumes/T7/rust-pekko/pekko-event-bus" }
pekko-agent-core = { path = "/Volumes/T7/pekko-agent/crates/pekko-agent-core" }
pekko-agent-events = { path = "/Volumes/T7/pekko-agent/crates/pekko-agent-events" }
pekko-agent-tools = { path = "/Volumes/T7/pekko-agent/crates/pekko-agent-tools" }
pekko-agent-orchestrator = { path = "/Volumes/T7/pekko-agent/crates/pekko-agent-orchestrator" }

[dev-dependencies]
tokio = { version = "1", features = ["full"] }
```

- [ ] **Step 3: Write eventbus_hooks.rs**

```rust
// crates/monitoring-bridge/src/eventbus_hooks.rs
use monitoring_core::snapshot::TopicSnapshot;
use pekko_event_bus::EventBusHandle;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

/// Tracks event-bus activity by subscribing to topics.
pub struct EventBusMonitor {
    bus: EventBusHandle,
    topics: Vec<String>,
    counters: Arc<Mutex<HashMap<String, u64>>>,
}

impl EventBusMonitor {
    pub fn new(bus: EventBusHandle, topics: Vec<String>) -> Self {
        let counters = Arc::new(Mutex::new(HashMap::new()));
        for t in &topics {
            counters.lock().unwrap().insert(t.clone(), 0);
        }
        Self { bus, topics, counters }
    }

    /// Subscribe a counting listener to each topic.
    /// Call this once during initialization.
    pub fn start_counting(&self) {
        for topic in &self.topics {
            let counters = self.counters.clone();
            let topic_name = topic.clone();
            let _ = self.bus.subscribe(topic, move |_topic, _payload| {
                if let Ok(mut c) = counters.lock() {
                    *c.entry(topic_name.clone()).or_insert(0) += 1;
                }
            });
        }
    }

    /// Snapshot current topic metrics.
    pub fn snapshot(&self) -> Vec<TopicSnapshot> {
        let counts = self.counters.lock().unwrap();
        self.topics.iter().map(|topic| {
            let sub_count = self.bus.subscriber_count(topic).unwrap_or(0);
            let published = counts.get(topic).copied().unwrap_or(0);
            TopicSnapshot {
                topic: topic.clone(),
                subscriber_count: sub_count,
                messages_published: published,
                partition_count: 0,
                messages_per_partition: Vec::new(),
            }
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pekko_event_bus::{EventBus, EventBusHandle, partition_strategy::PartitionKey};
    use pekko_event_bus::bus_config::TopicConfig;

    fn make_bus(topic: &str) -> EventBusHandle {
        let bus = EventBus::builder()
            .add_topic(TopicConfig::new(topic).partitions(1))
            .build()
            .unwrap();
        EventBusHandle::new(bus)
    }

    #[test]
    fn test_eventbus_monitor_empty() {
        let handle = make_bus("test-topic");
        let monitor = EventBusMonitor::new(handle, vec!["test-topic".to_string()]);
        let snaps = monitor.snapshot();
        assert_eq!(snaps.len(), 1);
        assert_eq!(snaps[0].messages_published, 0);
    }

    #[test]
    fn test_eventbus_monitor_counts() {
        let handle = make_bus("events");
        let monitor = EventBusMonitor::new(handle.clone(), vec!["events".to_string()]);
        monitor.start_counting();

        handle.publish("events", &PartitionKey::new("k"), b"msg1".to_vec()).unwrap();
        handle.publish("events", &PartitionKey::new("k"), b"msg2".to_vec()).unwrap();

        let snaps = monitor.snapshot();
        assert_eq!(snaps[0].messages_published, 2);
    }

    #[test]
    fn test_eventbus_monitor_subscriber_count() {
        let handle = make_bus("events");
        let monitor = EventBusMonitor::new(handle.clone(), vec!["events".to_string()]);
        monitor.start_counting();

        let snaps = monitor.snapshot();
        // start_counting adds 1 subscriber
        assert!(snaps[0].subscriber_count >= 1);
    }
}
```

- [ ] **Step 4: Write circuit_breaker_hooks.rs**

```rust
// crates/monitoring-bridge/src/circuit_breaker_hooks.rs
use monitoring_core::snapshot::CircuitBreakerSnapshot;
use monitoring_core::types::CBState;
use pekko_actor::{CircuitBreaker, CircuitBreakerState};

/// Wraps a named CircuitBreaker for monitoring.
pub struct CBMonitor {
    name: String,
    cb: CircuitBreaker,
}

impl CBMonitor {
    pub fn new(name: impl Into<String>, cb: CircuitBreaker) -> Self {
        Self { name: name.into(), cb }
    }

    pub fn snapshot(&self) -> CircuitBreakerSnapshot {
        let stats = self.cb.stats();
        CircuitBreakerSnapshot {
            name: self.name.clone(),
            state: match stats.state {
                CircuitBreakerState::Closed => CBState::Closed,
                CircuitBreakerState::Open => CBState::Open,
                CircuitBreakerState::HalfOpen => CBState::HalfOpen,
            },
            failure_count: stats.failure_count,
            success_count: stats.success_count,
            total_calls: stats.total_calls,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn make_cb() -> CircuitBreaker {
        CircuitBreaker::builder()
            .max_failures(3)
            .call_timeout(Duration::from_secs(5))
            .reset_timeout(Duration::from_secs(10))
            .build()
    }

    #[test]
    fn test_cb_monitor_initial_state() {
        let cb = make_cb();
        let monitor = CBMonitor::new("ollama", cb);
        let snap = monitor.snapshot();
        assert_eq!(snap.name, "ollama");
        assert_eq!(snap.state, CBState::Closed);
        assert_eq!(snap.total_calls, 0);
    }

    #[test]
    fn test_cb_monitor_failure_rate_zero() {
        let cb = make_cb();
        let monitor = CBMonitor::new("llm", cb);
        let snap = monitor.snapshot();
        assert_eq!(snap.failure_rate(), 0.0);
    }
}
```

- [ ] **Step 5: Write lib.rs for monitoring-bridge**

```rust
// crates/monitoring-bridge/src/lib.rs
pub mod eventbus_hooks;
pub mod circuit_breaker_hooks;

pub use eventbus_hooks::EventBusMonitor;
pub use circuit_breaker_hooks::CBMonitor;
```

- [ ] **Step 6: Run tests**

```bash
cargo test -p monitoring-bridge
```
Expected: 5 tests pass.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: monitoring-bridge with EventBus + CircuitBreaker hooks"
```

---

## Task 5: Monitoring Bridge — Tool + Orchestrator Hooks + PekkoMonitor

**Files:**
- Create: `crates/monitoring-bridge/src/tool_hooks.rs`
- Create: `crates/monitoring-bridge/src/orchestrator_hooks.rs`
- Create: `crates/monitoring-bridge/src/actor_hooks.rs`
- Modify: `crates/monitoring-bridge/src/lib.rs`
- Test: inline `#[cfg(test)]`

- [ ] **Step 1: Write tool_hooks.rs**

```rust
// crates/monitoring-bridge/src/tool_hooks.rs
use monitoring_core::snapshot::ToolSnapshot;
use std::collections::HashMap;

/// Converts pekko ToolStats into monitoring ToolSnapshot.
/// We accept a HashMap rather than depending on ToolRegistry directly
/// to keep the coupling loose (ToolRegistry is &mut self, not easily shared).
pub fn snapshot_tools(
    stats: &HashMap<String, pekko_agent_tools::registry::ToolStats>,
) -> Vec<ToolSnapshot> {
    stats.iter().map(|(name, s)| {
        ToolSnapshot {
            name: name.clone(),
            call_count: s.call_count,
            success_count: s.success_count,
            failure_count: s.failure_count,
            avg_duration_ms: s.avg_duration_ms(),
            total_duration_ms: s.total_duration_ms,
        }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pekko_agent_tools::registry::ToolStats;

    #[test]
    fn test_snapshot_tools_empty() {
        let stats: HashMap<String, ToolStats> = HashMap::new();
        let snaps = snapshot_tools(&stats);
        assert!(snaps.is_empty());
    }

    #[test]
    fn test_snapshot_tools_conversion() {
        let mut stats = HashMap::new();
        stats.insert("web_search".to_string(), ToolStats {
            call_count: 10,
            success_count: 8,
            failure_count: 2,
            total_duration_ms: 5000,
        });
        let snaps = snapshot_tools(&stats);
        assert_eq!(snaps.len(), 1);
        assert_eq!(snaps[0].name, "web_search");
        assert_eq!(snaps[0].call_count, 10);
        assert!((snaps[0].avg_duration_ms - 500.0).abs() < 0.01);
    }
}
```

- [ ] **Step 2: Write actor_hooks.rs**

```rust
// crates/monitoring-bridge/src/actor_hooks.rs
use monitoring_core::snapshot::AgentSnapshot;
use monitoring_core::types::AgentStateLabel;
use chrono::Utc;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Tracks per-agent message counts and state.
///
/// The host application calls `record_message()` and `update_state()`
/// when agent activity occurs.
#[derive(Clone)]
pub struct AgentTracker {
    inner: Arc<Mutex<HashMap<String, AgentRecord>>>,
}

struct AgentRecord {
    agent_type: String,
    state: AgentStateLabel,
    messages_received: u64,
    messages_sent: u64,
    errors: u64,
    last_activity: chrono::DateTime<Utc>,
    current_task: Option<String>,
}

impl AgentTracker {
    pub fn new() -> Self {
        Self { inner: Arc::new(Mutex::new(HashMap::new())) }
    }

    pub fn register(&self, agent_id: &str, agent_type: &str) {
        let mut map = self.inner.lock().unwrap();
        map.insert(agent_id.to_string(), AgentRecord {
            agent_type: agent_type.to_string(),
            state: AgentStateLabel::Idle,
            messages_received: 0,
            messages_sent: 0,
            errors: 0,
            last_activity: Utc::now(),
            current_task: None,
        });
    }

    pub fn record_received(&self, agent_id: &str) {
        if let Ok(mut map) = self.inner.lock() {
            if let Some(r) = map.get_mut(agent_id) {
                r.messages_received += 1;
                r.last_activity = Utc::now();
            }
        }
    }

    pub fn record_sent(&self, agent_id: &str) {
        if let Ok(mut map) = self.inner.lock() {
            if let Some(r) = map.get_mut(agent_id) {
                r.messages_sent += 1;
            }
        }
    }

    pub fn record_error(&self, agent_id: &str) {
        if let Ok(mut map) = self.inner.lock() {
            if let Some(r) = map.get_mut(agent_id) {
                r.errors += 1;
            }
        }
    }

    pub fn update_state(&self, agent_id: &str, state: AgentStateLabel) {
        if let Ok(mut map) = self.inner.lock() {
            if let Some(r) = map.get_mut(agent_id) {
                r.state = state;
                r.last_activity = Utc::now();
            }
        }
    }

    pub fn snapshot(&self) -> Vec<AgentSnapshot> {
        let map = self.inner.lock().unwrap();
        map.iter().map(|(id, r)| AgentSnapshot {
            agent_id: id.clone(),
            agent_type: r.agent_type.clone(),
            state: r.state.clone(),
            messages_received: r.messages_received,
            messages_sent: r.messages_sent,
            errors: r.errors,
            last_activity: r.last_activity,
            current_task: r.current_task.clone(),
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_tracker_register() {
        let tracker = AgentTracker::new();
        tracker.register("agent-1", "coder");
        let snaps = tracker.snapshot();
        assert_eq!(snaps.len(), 1);
        assert_eq!(snaps[0].state, AgentStateLabel::Idle);
    }

    #[test]
    fn test_agent_tracker_message_counts() {
        let tracker = AgentTracker::new();
        tracker.register("a1", "coder");
        tracker.record_received("a1");
        tracker.record_received("a1");
        tracker.record_sent("a1");
        let snaps = tracker.snapshot();
        assert_eq!(snaps[0].messages_received, 2);
        assert_eq!(snaps[0].messages_sent, 1);
    }

    #[test]
    fn test_agent_tracker_state_update() {
        let tracker = AgentTracker::new();
        tracker.register("a1", "reviewer");
        tracker.update_state("a1", AgentStateLabel::Reasoning);
        let snaps = tracker.snapshot();
        assert_eq!(snaps[0].state, AgentStateLabel::Reasoning);
    }

    #[test]
    fn test_agent_tracker_error_count() {
        let tracker = AgentTracker::new();
        tracker.register("a1", "tester");
        tracker.record_error("a1");
        tracker.record_error("a1");
        let snaps = tracker.snapshot();
        assert_eq!(snaps[0].errors, 2);
    }
}
```

- [ ] **Step 3: Write orchestrator_hooks.rs**

```rust
// crates/monitoring-bridge/src/orchestrator_hooks.rs
use monitoring_core::snapshot::{TaskQueueSnapshot, ActiveTaskSnapshot};
use chrono::Utc;
use std::sync::{Arc, Mutex};

/// Tracks orchestrator task metrics.
///
/// The host application calls `task_queued()`, `task_started()`,
/// `task_completed()`, `task_failed()` as orchestrator events occur.
#[derive(Clone)]
pub struct TaskTracker {
    inner: Arc<Mutex<TaskState>>,
}

struct TaskState {
    queued: usize,
    running: usize,
    completed: u64,
    failed: u64,
    active: Vec<ActiveTaskSnapshot>,
}

impl TaskTracker {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(TaskState {
                queued: 0, running: 0, completed: 0, failed: 0, active: Vec::new(),
            })),
        }
    }

    pub fn task_queued(&self) {
        self.inner.lock().unwrap().queued += 1;
    }

    pub fn task_started(&self, task_id: &str, agent: &str, desc: &str) {
        let mut s = self.inner.lock().unwrap();
        s.queued = s.queued.saturating_sub(1);
        s.running += 1;
        s.active.push(ActiveTaskSnapshot {
            task_id: task_id.to_string(),
            assigned_agent: agent.to_string(),
            description: desc.to_string(),
            started_at: Utc::now(),
            status: "running".to_string(),
        });
    }

    pub fn task_completed(&self, task_id: &str) {
        let mut s = self.inner.lock().unwrap();
        s.running = s.running.saturating_sub(1);
        s.completed += 1;
        s.active.retain(|t| t.task_id != task_id);
    }

    pub fn task_failed(&self, task_id: &str) {
        let mut s = self.inner.lock().unwrap();
        s.running = s.running.saturating_sub(1);
        s.failed += 1;
        s.active.retain(|t| t.task_id != task_id);
    }

    pub fn snapshot(&self) -> TaskQueueSnapshot {
        let s = self.inner.lock().unwrap();
        TaskQueueSnapshot {
            queued: s.queued,
            running: s.running,
            completed: s.completed,
            failed: s.failed,
            active_tasks: s.active.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_tracker_lifecycle() {
        let tracker = TaskTracker::new();
        tracker.task_queued();
        tracker.task_queued();
        assert_eq!(tracker.snapshot().queued, 2);

        tracker.task_started("t1", "agent-1", "build");
        assert_eq!(tracker.snapshot().queued, 1);
        assert_eq!(tracker.snapshot().running, 1);

        tracker.task_completed("t1");
        assert_eq!(tracker.snapshot().running, 0);
        assert_eq!(tracker.snapshot().completed, 1);
    }

    #[test]
    fn test_task_tracker_failure() {
        let tracker = TaskTracker::new();
        tracker.task_queued();
        tracker.task_started("t1", "a1", "test");
        tracker.task_failed("t1");
        assert_eq!(tracker.snapshot().failed, 1);
        assert_eq!(tracker.snapshot().running, 0);
    }

    #[test]
    fn test_task_tracker_active_list() {
        let tracker = TaskTracker::new();
        tracker.task_queued();
        tracker.task_started("t1", "a1", "compile");
        let snap = tracker.snapshot();
        assert_eq!(snap.active_tasks.len(), 1);
        assert_eq!(snap.active_tasks[0].assigned_agent, "a1");
    }
}
```

- [ ] **Step 4: Update lib.rs — add PekkoMonitor facade**

```rust
// crates/monitoring-bridge/src/lib.rs
pub mod eventbus_hooks;
pub mod circuit_breaker_hooks;
pub mod tool_hooks;
pub mod actor_hooks;
pub mod orchestrator_hooks;

pub use eventbus_hooks::EventBusMonitor;
pub use circuit_breaker_hooks::CBMonitor;
pub use actor_hooks::AgentTracker;
pub use orchestrator_hooks::TaskTracker;

use monitoring_core::{MetricsCollector, MetricsSnapshot};
use std::collections::HashMap;

/// Top-level facade that assembles a full MetricsSnapshot
/// from all registered monitors.
pub struct PekkoMonitor {
    pub agents: AgentTracker,
    pub tasks: TaskTracker,
    pub event_bus: Option<EventBusMonitor>,
    pub circuit_breakers: Vec<CBMonitor>,
    tool_stats_fn: Option<Box<dyn Fn() -> HashMap<String, pekko_agent_tools::registry::ToolStats> + Send + Sync>>,
}

impl PekkoMonitor {
    pub fn new() -> Self {
        Self {
            agents: AgentTracker::new(),
            tasks: TaskTracker::new(),
            event_bus: None,
            circuit_breakers: Vec::new(),
            tool_stats_fn: None,
        }
    }

    pub fn with_event_bus(mut self, monitor: EventBusMonitor) -> Self {
        self.event_bus = Some(monitor);
        self
    }

    pub fn add_circuit_breaker(&mut self, cb: CBMonitor) {
        self.circuit_breakers.push(cb);
    }

    pub fn set_tool_stats_provider<F>(&mut self, f: F)
    where F: Fn() -> HashMap<String, pekko_agent_tools::registry::ToolStats> + Send + Sync + 'static
    {
        self.tool_stats_fn = Some(Box::new(f));
    }
}

impl MetricsCollector for PekkoMonitor {
    fn collect(&self) -> MetricsSnapshot {
        let mut snap = MetricsSnapshot::new();

        snap.agents = self.agents.snapshot();

        snap.circuit_breakers = self.circuit_breakers.iter()
            .map(|cb| cb.snapshot())
            .collect();

        if let Some(ref eb) = self.event_bus {
            snap.event_bus_topics = eb.snapshot();
        }

        if let Some(ref f) = self.tool_stats_fn {
            snap.tool_stats = tool_hooks::snapshot_tools(&f());
        }

        snap.tasks = self.tasks.snapshot();

        snap
    }
}
```

- [ ] **Step 5: Run all tests**

```bash
cargo test --workspace
```
Expected: all tests pass across monitoring-core (14) and monitoring-bridge (~14).

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: tool + orchestrator + actor hooks + PekkoMonitor facade"
```

---

## Task 6: WebSocket Metrics Exporter

**Files:**
- Create: `crates/monitoring-exporter/Cargo.toml`
- Create: `crates/monitoring-exporter/src/lib.rs`
- Create: `crates/monitoring-exporter/src/routes.rs`
- Create: `crates/monitoring-exporter/src/ws.rs`
- Create: `crates/monitoring-exporter/src/main.rs`
- Modify: root `Cargo.toml` (workspace members)
- Test: inline `#[cfg(test)]`

- [ ] **Step 1: Add to workspace**

Append `"crates/monitoring-exporter"` to `[workspace] members`.

- [ ] **Step 2: Create Cargo.toml**

```toml
# crates/monitoring-exporter/Cargo.toml
[package]
name = "monitoring-exporter"
version = "0.1.0"
edition = "2021"

[dependencies]
monitoring-core = { path = "../monitoring-core" }
axum = { version = "0.8", features = ["ws"] }
tokio = { version = "1", features = ["full"] }
tower-http = { version = "0.6", features = ["cors"] }
serde = { workspace = true }
serde_json = { workspace = true }
anyhow = { workspace = true }
futures = "0.3"

[dev-dependencies]
reqwest = { version = "0.12", features = ["json"] }
tokio = { version = "1", features = ["full", "test-util"] }
```

- [ ] **Step 3: Write routes.rs**

```rust
// crates/monitoring-exporter/src/routes.rs
use axum::{extract::State, response::Json};
use monitoring_core::{MetricsSnapshot, RingBufferStore};
use std::sync::{Arc, Mutex};

pub type SharedStore = Arc<Mutex<RingBufferStore>>;

pub async fn get_latest(State(store): State<SharedStore>) -> Json<Option<MetricsSnapshot>> {
    let store = store.lock().unwrap();
    Json(store.latest().cloned())
}

pub async fn get_history(State(store): State<SharedStore>) -> Json<Vec<MetricsSnapshot>> {
    let store = store.lock().unwrap();
    Json(store.all().into_iter().cloned().collect())
}

pub async fn get_health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_store_empty() {
        let store: SharedStore = Arc::new(Mutex::new(RingBufferStore::new(10)));
        let s = store.lock().unwrap();
        assert!(s.latest().is_none());
    }

    #[test]
    fn test_shared_store_with_snapshot() {
        let store: SharedStore = Arc::new(Mutex::new(RingBufferStore::new(10)));
        store.lock().unwrap().push(MetricsSnapshot::new());
        let s = store.lock().unwrap();
        assert!(s.latest().is_some());
    }
}
```

- [ ] **Step 4: Write ws.rs**

```rust
// crates/monitoring-exporter/src/ws.rs
use axum::{
    extract::{State, ws::{Message, WebSocket, WebSocketUpgrade}},
    response::Response,
};
use crate::routes::SharedStore;
use std::time::Duration;

pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(store): State<SharedStore>,
) -> Response {
    ws.on_upgrade(move |socket| handle_socket(socket, store))
}

async fn handle_socket(mut socket: WebSocket, store: SharedStore) {
    loop {
        let snapshot = {
            let s = store.lock().unwrap();
            s.latest().cloned()
        };

        if let Some(snap) = snapshot {
            if let Ok(json) = serde_json::to_string(&snap) {
                if socket.send(Message::Text(json.into())).await.is_err() {
                    return; // client disconnected
                }
            }
        }

        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}
```

- [ ] **Step 5: Write lib.rs**

```rust
// crates/monitoring-exporter/src/lib.rs
pub mod routes;
pub mod ws;

use axum::Router;
use axum::routing::get;
use tower_http::cors::{CorsLayer, Any};
use routes::SharedStore;
use monitoring_core::RingBufferStore;
use std::sync::{Arc, Mutex};

pub struct ExporterServer {
    store: SharedStore,
}

impl ExporterServer {
    pub fn new(capacity: usize) -> Self {
        Self {
            store: Arc::new(Mutex::new(RingBufferStore::new(capacity))),
        }
    }

    pub fn store(&self) -> SharedStore {
        self.store.clone()
    }

    pub fn router(&self) -> Router {
        let cors = CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any);

        Router::new()
            .route("/api/health", get(routes::get_health))
            .route("/api/snapshot", get(routes::get_latest))
            .route("/api/history", get(routes::get_history))
            .route("/ws/live", get(ws::ws_handler))
            .layer(cors)
            .with_state(self.store.clone())
    }
}
```

- [ ] **Step 6: Write main.rs**

```rust
// crates/monitoring-exporter/src/main.rs
use monitoring_exporter::ExporterServer;
use monitoring_core::MetricsSnapshot;
use std::time::Duration;

#[tokio::main]
async fn main() {
    let server = ExporterServer::new(300); // 5 minutes at 1/sec
    let store = server.store();

    // Demo: push empty snapshots every second
    tokio::spawn(async move {
        loop {
            store.lock().unwrap().push(MetricsSnapshot::new());
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    });

    let listener = tokio::net::TcpListener::bind("0.0.0.0:9100").await.unwrap();
    println!("Monitoring exporter listening on http://0.0.0.0:9100");
    axum::serve(listener, server.router()).await.unwrap();
}
```

- [ ] **Step 7: Run tests**

```bash
cargo test --workspace
```
Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "feat: monitoring-exporter with REST + WebSocket metrics endpoints"
```

---

## Task 7: WASM Dashboard UI — Scaffold + Overview Panel

**Files:**
- Create: `crates/monitoring-ui/Cargo.toml`
- Create: `crates/monitoring-ui/index.html`
- Create: `crates/monitoring-ui/src/main.rs`
- Create: `crates/monitoring-ui/src/app.rs`
- Create: `crates/monitoring-ui/src/ws_client.rs`
- Create: `crates/monitoring-ui/src/panels/mod.rs`
- Create: `crates/monitoring-ui/src/panels/overview.rs`
- Create: `trunk.toml`
- Modify: root `Cargo.toml`

- [ ] **Step 1: Add to workspace**

Append `"crates/monitoring-ui"` to workspace members.

- [ ] **Step 2: Create Cargo.toml**

```toml
# crates/monitoring-ui/Cargo.toml
[package]
name = "monitoring-ui"
version = "0.1.0"
edition = "2021"

[dependencies]
monitoring-core = { path = "../monitoring-core" }
eframe = { version = "0.31", default-features = false, features = ["glow", "wasm_screen_reader"] }
serde = { workspace = true }
serde_json = { workspace = true }
chrono = { workspace = true }
```

- [ ] **Step 3: Create index.html**

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Pekko Agent Monitor</title>
    <link data-trunk rel="rust" data-wasm-opt="z" />
    <style>
        html, body { margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background: #1a1a2e; }
        canvas { width: 100% !important; height: 100% !important; }
    </style>
</head>
<body></body>
</html>
```

- [ ] **Step 4: Write ws_client.rs (WASM WebSocket stub)**

```rust
// crates/monitoring-ui/src/ws_client.rs
use monitoring_core::MetricsSnapshot;
use std::sync::{Arc, Mutex};

/// WebSocket client for receiving live metrics.
/// In WASM, this will use web_sys::WebSocket.
/// For now, stores the latest snapshot received.
pub struct WsClient {
    latest: Arc<Mutex<Option<MetricsSnapshot>>>,
    url: String,
    connected: bool,
}

impl WsClient {
    pub fn new(url: &str) -> Self {
        Self {
            latest: Arc::new(Mutex::new(None)),
            url: url.to_string(),
            connected: false,
        }
    }

    pub fn latest(&self) -> Option<MetricsSnapshot> {
        self.latest.lock().ok().and_then(|g| g.clone())
    }

    pub fn is_connected(&self) -> bool {
        self.connected
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    /// Inject a snapshot (used for testing or manual feed).
    pub fn inject(&self, snapshot: MetricsSnapshot) {
        if let Ok(mut g) = self.latest.lock() {
            *g = Some(snapshot);
        }
    }
}
```

- [ ] **Step 5: Write panels/overview.rs**

```rust
// crates/monitoring-ui/src/panels/overview.rs
use eframe::egui;
use monitoring_core::MetricsSnapshot;

pub fn draw(ui: &mut egui::Ui, snapshot: &Option<MetricsSnapshot>) {
    ui.heading("System Overview");
    ui.separator();

    match snapshot {
        None => {
            ui.colored_label(egui::Color32::YELLOW, "Waiting for metrics...");
        }
        Some(snap) => {
            ui.horizontal(|ui| {
                stat_card(ui, "Agents", &snap.agents.len().to_string(), egui::Color32::LIGHT_BLUE);
                stat_card(ui, "Active Tasks", &snap.tasks.running.to_string(), egui::Color32::LIGHT_GREEN);
                stat_card(ui, "Queued", &snap.tasks.queued.to_string(), egui::Color32::YELLOW);
                stat_card(ui, "Alerts", &snap.alerts.len().to_string(),
                    if snap.alerts.is_empty() { egui::Color32::GRAY } else { egui::Color32::RED });
            });

            ui.add_space(10.0);
            ui.heading("Agents");
            egui::Grid::new("agents_grid").striped(true).show(ui, |ui| {
                ui.strong("ID");
                ui.strong("Type");
                ui.strong("State");
                ui.strong("Msgs In");
                ui.strong("Msgs Out");
                ui.strong("Errors");
                ui.end_row();

                for agent in &snap.agents {
                    ui.label(&agent.agent_id);
                    ui.label(&agent.agent_type);
                    let state_color = match agent.state {
                        monitoring_core::AgentStateLabel::Idle => egui::Color32::GRAY,
                        monitoring_core::AgentStateLabel::Error => egui::Color32::RED,
                        _ => egui::Color32::LIGHT_GREEN,
                    };
                    ui.colored_label(state_color, agent.state.to_string());
                    ui.label(agent.messages_received.to_string());
                    ui.label(agent.messages_sent.to_string());
                    let err_color = if agent.errors > 0 { egui::Color32::RED } else { egui::Color32::GRAY };
                    ui.colored_label(err_color, agent.errors.to_string());
                    ui.end_row();
                }
            });

            ui.add_space(10.0);
            ui.heading("Circuit Breakers");
            for cb in &snap.circuit_breakers {
                let color = match cb.state {
                    monitoring_core::CBState::Closed => egui::Color32::GREEN,
                    monitoring_core::CBState::Open => egui::Color32::RED,
                    monitoring_core::CBState::HalfOpen => egui::Color32::YELLOW,
                };
                ui.horizontal(|ui| {
                    ui.colored_label(color, format!("● {}", cb.name));
                    ui.label(format!("{} — fail rate: {:.1}% ({} calls)",
                        cb.state, cb.failure_rate(), cb.total_calls));
                });
            }
        }
    }
}

fn stat_card(ui: &mut egui::Ui, label: &str, value: &str, color: egui::Color32) {
    egui::Frame::none()
        .inner_margin(egui::Margin::same(12))
        .rounding(egui::Rounding::same(8))
        .fill(egui::Color32::from_rgb(30, 30, 50))
        .show(ui, |ui| {
            ui.vertical(|ui| {
                ui.colored_label(color, egui::RichText::new(value).size(28.0).strong());
                ui.label(label);
            });
        });
}
```

- [ ] **Step 6: Write panels/mod.rs**

```rust
// crates/monitoring-ui/src/panels/mod.rs
pub mod overview;
```

- [ ] **Step 7: Write app.rs**

```rust
// crates/monitoring-ui/src/app.rs
use eframe::egui;
use crate::ws_client::WsClient;
use crate::panels;

pub struct MonitoringApp {
    ws: WsClient,
}

impl MonitoringApp {
    pub fn new(ws_url: &str) -> Self {
        Self {
            ws: WsClient::new(ws_url),
        }
    }
}

impl eframe::App for MonitoringApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let snapshot = self.ws.latest();

        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("🔍 Pekko Agent Monitor");
                ui.separator();
                let status = if self.ws.is_connected() { "● Connected" } else { "○ Disconnected" };
                let color = if self.ws.is_connected() { egui::Color32::GREEN } else { egui::Color32::RED };
                ui.colored_label(color, status);
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            panels::overview::draw(ui, &snapshot);
        });

        // Request repaint for live updates
        ctx.request_repaint_after(std::time::Duration::from_secs(1));
    }
}
```

- [ ] **Step 8: Write main.rs**

```rust
// crates/monitoring-ui/src/main.rs
mod app;
mod ws_client;
mod panels;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_title("Pekko Agent Monitor"),
        ..Default::default()
    };

    eframe::run_native(
        "pekko-monitor",
        options,
        Box::new(|_cc| Ok(Box::new(app::MonitoringApp::new("ws://localhost:9100/ws/live")))),
    )
}
```

- [ ] **Step 9: Create trunk.toml**

```toml
# /Volumes/T7/core-agent-monitoring/trunk.toml
[build]
target = "crates/monitoring-ui/index.html"
dist = "dist"
```

- [ ] **Step 10: Verify native build compiles**

```bash
cargo build -p monitoring-ui
```
Expected: compiles (native mode, no WASM yet).

- [ ] **Step 11: Commit**

```bash
git add -A && git commit -m "feat: monitoring-ui scaffold with overview panel (egui/eframe)"
```

---

## Task 8: Additional Dashboard Panels

**Files:**
- Create: `crates/monitoring-ui/src/panels/event_bus.rs`
- Create: `crates/monitoring-ui/src/panels/circuit_breakers.rs`
- Create: `crates/monitoring-ui/src/panels/tools.rs`
- Create: `crates/monitoring-ui/src/panels/tasks.rs`
- Create: `crates/monitoring-ui/src/panels/agents.rs`
- Modify: `crates/monitoring-ui/src/panels/mod.rs`
- Modify: `crates/monitoring-ui/src/app.rs`

- [ ] **Step 1: Write event_bus.rs**

```rust
// crates/monitoring-ui/src/panels/event_bus.rs
use eframe::egui;
use monitoring_core::MetricsSnapshot;

pub fn draw(ui: &mut egui::Ui, snapshot: &Option<MetricsSnapshot>) {
    ui.heading("Event Bus Traffic");
    ui.separator();

    let snap = match snapshot { Some(s) => s, None => { ui.label("No data"); return; } };

    if snap.event_bus_topics.is_empty() {
        ui.label("No topics registered.");
        return;
    }

    egui::Grid::new("eventbus_grid").striped(true).show(ui, |ui| {
        ui.strong("Topic");
        ui.strong("Subscribers");
        ui.strong("Messages");
        ui.strong("Partitions");
        ui.end_row();

        for topic in &snap.event_bus_topics {
            ui.label(&topic.topic);
            ui.label(topic.subscriber_count.to_string());
            ui.label(topic.messages_published.to_string());
            ui.label(topic.partition_count.to_string());
            ui.end_row();
        }
    });
}
```

- [ ] **Step 2: Write circuit_breakers.rs**

```rust
// crates/monitoring-ui/src/panels/circuit_breakers.rs
use eframe::egui;
use monitoring_core::MetricsSnapshot;

pub fn draw(ui: &mut egui::Ui, snapshot: &Option<MetricsSnapshot>) {
    ui.heading("Circuit Breakers");
    ui.separator();

    let snap = match snapshot { Some(s) => s, None => { ui.label("No data"); return; } };

    for cb in &snap.circuit_breakers {
        let (color, icon) = match cb.state {
            monitoring_core::CBState::Closed => (egui::Color32::GREEN, "✓"),
            monitoring_core::CBState::Open => (egui::Color32::RED, "✗"),
            monitoring_core::CBState::HalfOpen => (egui::Color32::YELLOW, "◐"),
        };

        egui::Frame::none()
            .inner_margin(egui::Margin::same(10))
            .rounding(egui::Rounding::same(6))
            .fill(egui::Color32::from_rgb(30, 30, 50))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.colored_label(color, egui::RichText::new(format!("{} {}", icon, cb.name)).size(18.0).strong());
                    ui.label(format!("State: {}", cb.state));
                });
                ui.horizontal(|ui| {
                    ui.label(format!("Total: {} | Success: {} | Failures: {} | Fail Rate: {:.1}%",
                        cb.total_calls, cb.success_count, cb.failure_count, cb.failure_rate()));
                });
            });
        ui.add_space(4.0);
    }
}
```

- [ ] **Step 3: Write tools.rs**

```rust
// crates/monitoring-ui/src/panels/tools.rs
use eframe::egui;
use monitoring_core::MetricsSnapshot;

pub fn draw(ui: &mut egui::Ui, snapshot: &Option<MetricsSnapshot>) {
    ui.heading("Tool Execution Stats");
    ui.separator();

    let snap = match snapshot { Some(s) => s, None => { ui.label("No data"); return; } };

    if snap.tool_stats.is_empty() {
        ui.label("No tool executions recorded.");
        return;
    }

    egui::Grid::new("tools_grid").striped(true).show(ui, |ui| {
        ui.strong("Tool");
        ui.strong("Calls");
        ui.strong("Success");
        ui.strong("Failures");
        ui.strong("Avg (ms)");
        ui.strong("Total (ms)");
        ui.end_row();

        for tool in &snap.tool_stats {
            ui.label(&tool.name);
            ui.label(tool.call_count.to_string());
            ui.colored_label(egui::Color32::GREEN, tool.success_count.to_string());
            let fc = if tool.failure_count > 0 { egui::Color32::RED } else { egui::Color32::GRAY };
            ui.colored_label(fc, tool.failure_count.to_string());
            ui.label(format!("{:.1}", tool.avg_duration_ms));
            ui.label(tool.total_duration_ms.to_string());
            ui.end_row();
        }
    });
}
```

- [ ] **Step 4: Write tasks.rs**

```rust
// crates/monitoring-ui/src/panels/tasks.rs
use eframe::egui;
use monitoring_core::MetricsSnapshot;

pub fn draw(ui: &mut egui::Ui, snapshot: &Option<MetricsSnapshot>) {
    ui.heading("Orchestrator Tasks");
    ui.separator();

    let snap = match snapshot { Some(s) => s, None => { ui.label("No data"); return; } };

    ui.horizontal(|ui| {
        ui.label(format!("Queued: {}", snap.tasks.queued));
        ui.separator();
        ui.colored_label(egui::Color32::LIGHT_GREEN, format!("Running: {}", snap.tasks.running));
        ui.separator();
        ui.label(format!("Completed: {}", snap.tasks.completed));
        ui.separator();
        let fc = if snap.tasks.failed > 0 { egui::Color32::RED } else { egui::Color32::GRAY };
        ui.colored_label(fc, format!("Failed: {}", snap.tasks.failed));
    });

    if !snap.tasks.active_tasks.is_empty() {
        ui.add_space(8.0);
        ui.strong("Active Tasks");
        egui::Grid::new("tasks_active_grid").striped(true).show(ui, |ui| {
            ui.strong("Task ID");
            ui.strong("Agent");
            ui.strong("Description");
            ui.strong("Status");
            ui.end_row();

            for task in &snap.tasks.active_tasks {
                ui.label(&task.task_id);
                ui.label(&task.assigned_agent);
                ui.label(&task.description);
                ui.label(&task.status);
                ui.end_row();
            }
        });
    }
}
```

- [ ] **Step 5: Write agents.rs**

```rust
// crates/monitoring-ui/src/panels/agents.rs
use eframe::egui;
use monitoring_core::MetricsSnapshot;

pub fn draw(ui: &mut egui::Ui, snapshot: &Option<MetricsSnapshot>) {
    ui.heading("Agent Detail");
    ui.separator();

    let snap = match snapshot { Some(s) => s, None => { ui.label("No data"); return; } };

    for agent in &snap.agents {
        let state_color = match agent.state {
            monitoring_core::AgentStateLabel::Idle => egui::Color32::GRAY,
            monitoring_core::AgentStateLabel::Reasoning => egui::Color32::LIGHT_BLUE,
            monitoring_core::AgentStateLabel::Acting => egui::Color32::LIGHT_GREEN,
            monitoring_core::AgentStateLabel::Observing => egui::Color32::YELLOW,
            monitoring_core::AgentStateLabel::Responding => egui::Color32::from_rgb(180, 120, 255),
            monitoring_core::AgentStateLabel::Error => egui::Color32::RED,
        };

        egui::Frame::none()
            .inner_margin(egui::Margin::same(10))
            .rounding(egui::Rounding::same(6))
            .fill(egui::Color32::from_rgb(25, 25, 45))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.colored_label(state_color, egui::RichText::new(format!("● {}", agent.agent_id)).size(16.0).strong());
                    ui.label(format!("[{}]", agent.agent_type));
                    ui.colored_label(state_color, agent.state.to_string());
                });
                ui.horizontal(|ui| {
                    ui.label(format!("In: {} | Out: {} | Errors: {}", agent.messages_received, agent.messages_sent, agent.errors));
                    if let Some(ref task) = agent.current_task {
                        ui.separator();
                        ui.label(format!("Task: {}", task));
                    }
                });
            });
        ui.add_space(4.0);
    }
}
```

- [ ] **Step 6: Update panels/mod.rs**

```rust
// crates/monitoring-ui/src/panels/mod.rs
pub mod overview;
pub mod event_bus;
pub mod circuit_breakers;
pub mod tools;
pub mod tasks;
pub mod agents;
```

- [ ] **Step 7: Update app.rs with tab navigation**

```rust
// crates/monitoring-ui/src/app.rs
use eframe::egui;
use crate::ws_client::WsClient;
use crate::panels;

#[derive(PartialEq)]
enum Tab {
    Overview,
    Agents,
    EventBus,
    CircuitBreakers,
    Tools,
    Tasks,
}

pub struct MonitoringApp {
    ws: WsClient,
    active_tab: Tab,
}

impl MonitoringApp {
    pub fn new(ws_url: &str) -> Self {
        Self {
            ws: WsClient::new(ws_url),
            active_tab: Tab::Overview,
        }
    }
}

impl eframe::App for MonitoringApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let snapshot = self.ws.latest();

        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Pekko Agent Monitor");
                ui.separator();
                let status = if self.ws.is_connected() { "Connected" } else { "Disconnected" };
                let color = if self.ws.is_connected() { egui::Color32::GREEN } else { egui::Color32::RED };
                ui.colored_label(color, status);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if let Some(ref snap) = snapshot {
                        ui.label(format!("{}", snap.timestamp.format("%H:%M:%S")));
                    }
                });
            });
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.active_tab, Tab::Overview, "Overview");
                ui.selectable_value(&mut self.active_tab, Tab::Agents, "Agents");
                ui.selectable_value(&mut self.active_tab, Tab::EventBus, "Event Bus");
                ui.selectable_value(&mut self.active_tab, Tab::CircuitBreakers, "Circuit Breakers");
                ui.selectable_value(&mut self.active_tab, Tab::Tools, "Tools");
                ui.selectable_value(&mut self.active_tab, Tab::Tasks, "Tasks");
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                match self.active_tab {
                    Tab::Overview => panels::overview::draw(ui, &snapshot),
                    Tab::Agents => panels::agents::draw(ui, &snapshot),
                    Tab::EventBus => panels::event_bus::draw(ui, &snapshot),
                    Tab::CircuitBreakers => panels::circuit_breakers::draw(ui, &snapshot),
                    Tab::Tools => panels::tools::draw(ui, &snapshot),
                    Tab::Tasks => panels::tasks::draw(ui, &snapshot),
                }
            });
        });

        ctx.request_repaint_after(std::time::Duration::from_secs(1));
    }
}
```

- [ ] **Step 8: Build**

```bash
cargo build -p monitoring-ui
```
Expected: compiles.

- [ ] **Step 9: Commit**

```bash
git add -A && git commit -m "feat: dashboard panels — agents, event bus, circuit breakers, tools, tasks"
```

---

## Task 9: Demo Metrics Generator + Integration Test

**Files:**
- Create: `examples/demo_metrics.rs`
- Modify: root `Cargo.toml`
- Test: manual visual verification

- [ ] **Step 1: Write demo_metrics.rs**

```rust
// examples/demo_metrics.rs
//! Generates fake metrics and pushes to the exporter store.
//! Run with: cargo run --example demo_metrics
//! Then open the monitoring-ui to see live data.

use monitoring_core::*;
use monitoring_exporter::ExporterServer;
use chrono::Utc;
use std::time::Duration;

#[tokio::main]
async fn main() {
    let server = ExporterServer::new(300);
    let store = server.store();

    let store_clone = store.clone();
    tokio::spawn(async move {
        let mut tick = 0u64;
        loop {
            let mut snap = MetricsSnapshot::new();

            // Fake agents
            snap.agents.push(AgentSnapshot {
                agent_id: "coder-1".to_string(),
                agent_type: "coder".to_string(),
                state: if tick % 5 == 0 { AgentStateLabel::Reasoning } else { AgentStateLabel::Idle },
                messages_received: tick * 3,
                messages_sent: tick * 2,
                errors: tick / 20,
                last_activity: Utc::now(),
                current_task: if tick % 5 == 0 { Some("build-v2".to_string()) } else { None },
            });
            snap.agents.push(AgentSnapshot {
                agent_id: "reviewer-1".to_string(),
                agent_type: "reviewer".to_string(),
                state: AgentStateLabel::Idle,
                messages_received: tick * 2,
                messages_sent: tick,
                errors: 0,
                last_activity: Utc::now(),
                current_task: None,
            });

            // Fake circuit breakers
            snap.circuit_breakers.push(CircuitBreakerSnapshot {
                name: "ollama-llm".to_string(),
                state: if tick % 30 < 5 { CBState::Open } else { CBState::Closed },
                failure_count: tick / 10,
                success_count: tick * 9 / 10,
                total_calls: tick,
            });

            // Fake event bus
            snap.event_bus_topics.push(TopicSnapshot {
                topic: "agent-events".to_string(),
                subscriber_count: 3,
                messages_published: tick * 5,
                partition_count: 4,
                messages_per_partition: vec![tick, tick + 1, tick, tick + 2],
            });

            // Fake tools
            snap.tool_stats.push(ToolSnapshot {
                name: "web_search".to_string(),
                call_count: tick / 2,
                success_count: tick / 2 - tick / 20,
                failure_count: tick / 20,
                avg_duration_ms: 250.0 + (tick % 10) as f64 * 10.0,
                total_duration_ms: tick * 250,
            });

            // Fake tasks
            snap.tasks.queued = (tick % 5) as usize;
            snap.tasks.running = (tick % 3) as usize;
            snap.tasks.completed = tick / 3;
            snap.tasks.failed = tick / 50;

            store_clone.lock().unwrap().push(snap);
            tick += 1;
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    });

    let listener = tokio::net::TcpListener::bind("0.0.0.0:9100").await.unwrap();
    println!("Demo exporter on http://localhost:9100");
    println!("  GET  /api/snapshot  — latest snapshot");
    println!("  GET  /api/history   — all snapshots");
    println!("  WS   /ws/live       — live WebSocket stream");
    println!("\nRun monitoring-ui to see the dashboard.");
    axum::serve(listener, server.router()).await.unwrap();
}
```

- [ ] **Step 2: Add to root Cargo.toml**

```toml
# After [workspace] section
[[example]]
name = "demo_metrics"
path = "examples/demo_metrics.rs"

[dependencies]
monitoring-core = { path = "crates/monitoring-core" }
monitoring-exporter = { path = "crates/monitoring-exporter" }
tokio = { version = "1", features = ["full"] }
axum = "0.8"
chrono = { version = "0.4", features = ["serde"] }
```

- [ ] **Step 3: Run demo + UI and verify**

Terminal 1:
```bash
cargo run --example demo_metrics
```

Terminal 2:
```bash
cargo run -p monitoring-ui
```

Expected: UI shows live-updating data — agents table, circuit breaker gauge, tool stats.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: demo metrics generator + integration with UI"
```

---

## Task 10: WASM Build Configuration

**Files:**
- Modify: `crates/monitoring-ui/Cargo.toml` (add wasm target features)
- Modify: `crates/monitoring-ui/src/main.rs` (conditional wasm entry)
- Create: `.cargo/config.toml` (optional, WASM-specific)

- [ ] **Step 1: Update monitoring-ui Cargo.toml for WASM**

```toml
# crates/monitoring-ui/Cargo.toml
[package]
name = "monitoring-ui"
version = "0.1.0"
edition = "2021"

[dependencies]
monitoring-core = { path = "../monitoring-core" }
eframe = { version = "0.31", default-features = false, features = ["glow", "wasm_screen_reader"] }
serde = { workspace = true }
serde_json = { workspace = true }
chrono = { workspace = true }

[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen-futures = "0.4"
web-sys = { version = "0.3", features = ["console"] }
```

- [ ] **Step 2: Update main.rs for WASM entry point**

```rust
// crates/monitoring-ui/src/main.rs
mod app;
mod ws_client;
mod panels;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_title("Pekko Agent Monitor"),
        ..Default::default()
    };
    eframe::run_native(
        "pekko-monitor",
        options,
        Box::new(|_cc| Ok(Box::new(app::MonitoringApp::new("ws://localhost:9100/ws/live")))),
    )
}

#[cfg(target_arch = "wasm32")]
fn main() {
    use eframe::wasm_bindgen::JsCast as _;
    let web_options = eframe::WebOptions::default();
    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window().unwrap().document().unwrap();
        let canvas = document
            .get_element_by_id("the_canvas_id")
            .unwrap()
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap();

        eframe::WebRunner::new()
            .start(
                canvas,
                web_options,
                Box::new(|_cc| Ok(Box::new(app::MonitoringApp::new("ws://localhost:9100/ws/live")))),
            )
            .await
            .expect("failed to start eframe");
    });
}
```

- [ ] **Step 3: Install trunk (if not present) and test WASM build**

```bash
cargo install trunk
rustup target add wasm32-unknown-unknown
cd /Volumes/T7/core-agent-monitoring && trunk build
```
Expected: builds to `dist/` directory.

- [ ] **Step 4: Test WASM serve**

```bash
trunk serve --open
```
Expected: browser opens with the monitoring dashboard.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: WASM build support via trunk"
```

---

## Summary

| Task | Crate | What it delivers | Est. tests |
|------|-------|-----------------|-----------|
| 1 | monitoring-core | Workspace + shared types | 4 |
| 2 | monitoring-core | MetricsSnapshot data model | 5 |
| 3 | monitoring-core | MetricsCollector trait + RingBufferStore | 5 |
| 4 | monitoring-bridge | EventBus + CircuitBreaker hooks | 5 |
| 5 | monitoring-bridge | Tool + Orchestrator + Agent hooks + PekkoMonitor | 9 |
| 6 | monitoring-exporter | REST + WebSocket server | 2 |
| 7 | monitoring-ui | Scaffold + Overview panel | — (visual) |
| 8 | monitoring-ui | All dashboard panels + tab nav | — (visual) |
| 9 | examples | Demo metrics generator | — (manual) |
| 10 | monitoring-ui | WASM build configuration | — (build) |

**Total: ~30 automated tests + visual verification**

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────┐
│                  pekko-agent process                 │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ Agents   │  │ Orch-    │  │ EventBus         │   │
│  │ (Actor-  │  │ estrator │  │ (Topics/         │   │
│  │  System) │  │ (Actor)  │  │  Partitions)     │   │
│  └────┬─────┘  └────┬─────┘  └───────┬──────────┘   │
│       │              │                │              │
│  ┌────▼──────────────▼────────────────▼──────────┐   │
│  │           monitoring-bridge                   │   │
│  │  PekkoMonitor (MetricsCollector impl)         │   │
│  │  ┌AgentTracker┐ ┌TaskTracker┐ ┌CBMonitor┐    │   │
│  │  └────────────┘ └──────────┘ └─────────┘     │   │
│  └─────────────────────┬─────────────────────────┘   │
│                        │ collect() → MetricsSnapshot │
│  ┌─────────────────────▼─────────────────────────┐   │
│  │         monitoring-exporter                   │   │
│  │  axum server :9100                            │   │
│  │  GET /api/snapshot  ← REST                    │   │
│  │  WS  /ws/live       ← WebSocket push (1/sec) │   │
│  └─────────────────────┬─────────────────────────┘   │
│                        │                             │
└────────────────────────┼─────────────────────────────┘
                         │ WebSocket JSON
                         ▼
              ┌──────────────────────┐
              │   monitoring-ui      │
              │   (WASM / eframe)    │
              │                      │
              │  ┌─────┐ ┌────────┐  │
              │  │Over-│ │Agents  │  │
              │  │view │ │Detail  │  │
              │  ├─────┤ ├────────┤  │
              │  │Event│ │Circuit │  │
              │  │Bus  │ │Breaker │  │
              │  ├─────┤ ├────────┤  │
              │  │Tools│ │Tasks   │  │
              │  └─────┘ └────────┘  │
              └──────────────────────┘
              runs in browser (WASM)
```

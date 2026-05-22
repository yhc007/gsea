use async_trait::async_trait;
use pekko_agent_core::{ToolContext, ToolDefinition, ToolError, ToolOutput};
use serde_json::Value;

/// Wraps a GSEA `Tool` so it can be used anywhere a `pekko_agent_core::Tool`
/// is expected.
pub struct PekkoToolAdapter {
    inner: Box<dyn crate::tools::Tool>,
}

impl PekkoToolAdapter {
    pub fn new(inner: Box<dyn crate::tools::Tool>) -> Self {
        Self { inner }
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

    async fn execute(&self, input: Value, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        match self.inner.execute(input).await {
            Ok(value) => Ok(ToolOutput::success(value)),
            Err(err) => Ok(ToolOutput::error(err.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use pekko_agent_core::Tool as PekkoTool;
    use serde_json::{json, Value};
    use std::collections::HashMap;
    use std::time::Duration;
    use uuid::Uuid;

    // ── Minimal stub GSEA tool ────────────────────────────────────────────────

    struct EchoTool;

    #[async_trait::async_trait]
    impl crate::tools::Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "Echoes its input back as output"
        }
        fn parameters(&self) -> Value {
            json!({
                "type": "object",
                "properties": {
                    "message": { "type": "string" }
                },
                "required": ["message"]
            })
        }
        async fn execute(&self, params: Value) -> Result<Value> {
            Ok(params)
        }
    }

    struct FailingTool;

    #[async_trait::async_trait]
    impl crate::tools::Tool for FailingTool {
        fn name(&self) -> &str {
            "fail"
        }
        fn description(&self) -> &str {
            "Always fails"
        }
        fn parameters(&self) -> Value {
            json!({ "type": "object", "properties": {} })
        }
        async fn execute(&self, _params: Value) -> Result<Value> {
            Err(anyhow::anyhow!("intentional failure"))
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn dummy_ctx() -> ToolContext {
        ToolContext {
            tenant_id: "test-tenant".to_string(),
            user_id: "test-user".to_string(),
            session_id: Uuid::new_v4(),
            credentials: HashMap::new(),
            timeout: Duration::from_secs(5),
        }
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    #[test]
    fn pekko_tool_definition_maps_correctly() {
        let adapter = PekkoToolAdapter::new(Box::new(EchoTool));
        let def = PekkoTool::definition(&adapter);

        assert_eq!(def.name, "echo");
        assert_eq!(def.description, "Echoes its input back as output");
        assert_eq!(def.input_schema["type"], "object");
        assert_eq!(def.timeout_ms, 30_000);
        assert!(!def.idempotent);
        assert!(def.required_permissions.is_empty());
    }

    #[tokio::test]
    async fn pekko_tool_success_wraps_value() {
        let adapter = PekkoToolAdapter::new(Box::new(EchoTool));
        let ctx = dummy_ctx();
        let input = json!({ "message": "hello" });

        let output = PekkoTool::execute(&adapter, input.clone(), &ctx).await.unwrap();

        assert!(!output.is_error);
        assert_eq!(output.content, input);
    }

    #[tokio::test]
    async fn pekko_tool_inner_error_becomes_error_output() {
        let adapter = PekkoToolAdapter::new(Box::new(FailingTool));
        let ctx = dummy_ctx();

        let output = PekkoTool::execute(&adapter, json!({}), &ctx)
            .await
            .unwrap();

        assert!(output.is_error);
        assert!(output.content["error"]
            .as_str()
            .unwrap()
            .contains("intentional failure"));
    }
}

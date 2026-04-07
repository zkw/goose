use crate::agents::{
    mcp_client::{Error, McpClientTrait},
    platform_extensions::PlatformExtensionContext,
    tool_execution::ToolCallContext,
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use rmcp::model::{
    CallToolResult, Content, Implementation, InitializeResult, JsonObject, ListToolsResult,
    ServerCapabilities,
};
use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::info;

use super::engine::{EngineHandle, SessionId};
use super::formats::{
    format_delegate_error, format_dispatch_message, format_hint, format_submit_report_busy,
    format_tool_not_found, DELEGATE_LOG_PREFIX, ERROR_EMPTY_INSTRUCTIONS, ERROR_PARENT_SESSION,
    ERROR_SUBAGENT_CANNOT_DELEGATE,
};
use super::tools::{DELEGATE_TOOL, REPORT_TOOL};
use super::worker::SubagentRunParams;

#[derive(Deserialize)]
struct DelegateArgs {
    instructions: Option<String>,
}

pub struct BetterSummonClient {
    ctx: PlatformExtensionContext,
    info: InitializeResult,
    tk: CancellationToken,
    engine: EngineHandle,
}

impl Drop for BetterSummonClient {
    fn drop(&mut self) {
        self.tk.cancel();
    }
}

impl BetterSummonClient {
    pub fn new(ctx: PlatformExtensionContext, engine: EngineHandle) -> Result<Self> {
        Ok(Self {
            ctx,
            info: InitializeResult::new(ServerCapabilities::builder().enable_tools().build())
                .with_server_info(Implementation::new(
                    "goose-better-summon",
                    env!("CARGO_PKG_VERSION"),
                ))
                .with_instructions(
                    "Highly efficient background task delegation with real-time supervision and consolidated reporting."
                        .to_string(),
                ),
            tk: CancellationToken::new(),
            engine,
        })
    }

    async fn handle_delegate(
        &self,
        session_id: &str,
        instructions: &str,
    ) -> anyhow::Result<CallToolResult> {
        if instructions.is_empty() {
            anyhow::bail!(ERROR_EMPTY_INSTRUCTIONS);
        }
        info!(
            "{} {}",
            DELEGATE_LOG_PREFIX,
            instructions.lines().next().unwrap_or("")
        );

        let ps = self
            .ctx
            .session_manager
            .get_session(session_id, false)
            .await
            .context(ERROR_PARENT_SESSION)?;
        if ps.session_type == crate::session::session_manager::SessionType::SubAgent {
            anyhow::bail!(ERROR_SUBAGENT_CANNOT_DELEGATE);
        }

        let sid = format!("{:04X}", rand::random::<u16>());
        let run_params = SubagentRunParams::from_context(
            &self.ctx,
            &ps,
            instructions,
            sid.clone(),
            self.engine.clone(),
            self.tk.child_token(),
        )
        .await?;

        let idle_count = self
            .engine
            .query_status(SessionId(Arc::from(session_id)))
            .await
            .idle_count
            .saturating_sub(1);
        let _ = self.engine.dispatch_task(run_params);
        Ok(CallToolResult::success(vec![Content::text(
            format_dispatch_message(&sid, idle_count),
        )]))
    }

    async fn is_subagent(&self, id: &str) -> bool {
        self.ctx
            .session_manager
            .get_session(id, false)
            .await
            .is_ok_and(|s| s.session_type == crate::session::session_manager::SessionType::SubAgent)
    }
}

#[async_trait]
impl McpClientTrait for BetterSummonClient {
    async fn list_tools(
        &self,
        id: &str,
        _: Option<String>,
        _: CancellationToken,
    ) -> Result<ListToolsResult, Error> {
        Ok(ListToolsResult::with_all_items(
            if self.is_subagent(id).await {
                vec![REPORT_TOOL.clone()]
            } else {
                vec![REPORT_TOOL.clone(), DELEGATE_TOOL.clone()]
            },
        ))
    }

    async fn call_tool(
        &self,
        ctx: &ToolCallContext,
        name: &str,
        args: Option<JsonObject>,
        _: CancellationToken,
    ) -> Result<CallToolResult, Error> {
        match name {
            "delegate" => {
                let a = args.unwrap_or_default();
                let delegate_instructions =
                    serde_json::from_value::<DelegateArgs>(Value::Object(a))
                        .ok()
                        .and_then(|d| d.instructions);
                let inst = delegate_instructions.as_deref().unwrap_or("");
                match self.handle_delegate(&ctx.session_id, inst).await {
                    Ok(r) => Ok(r),
                    Err(e) => Ok(CallToolResult::error(vec![Content::text(
                        format_delegate_error(&e.to_string()),
                    )])),
                }
            }
            "submit_task_report" => {
                let status = self
                    .engine
                    .query_status(SessionId(Arc::from(ctx.session_id.as_str())))
                    .await;
                if status.running_tasks > 0 || status.pending_tasks > 0 {
                    let message =
                        format_submit_report_busy(status.running_tasks, status.pending_tasks);
                    Ok(CallToolResult::error(vec![Content::text(message)]))
                } else {
                    Ok(CallToolResult::success(vec![Content::text(
                        "报告提交成功，任务流结束。",
                    )]))
                }
            }
            _ => Ok(CallToolResult::error(vec![Content::text(
                format_tool_not_found(name),
            )])),
        }
    }

    async fn get_moim(&self, id: &str) -> Option<String> {
        let mut hint = format_hint(self.is_subagent(id).await);
        let status = self.engine.query_status(SessionId(Arc::from(id))).await;
        if !status.reports.is_empty() {
            let reports_block =
                super::formats::render_report_prompt(&status.task_ids, 0, &status.reports);
            hint.push_str("\n\n");
            hint.push_str(&reports_block);
        }
        Some(hint)
    }

    fn get_info(&self) -> Option<&InitializeResult> {
        Some(&self.info)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::platform_extensions::better_summon::engine::{
        self, BgEv, EngineCommand, SessionId, TaskId,
    };
    use crate::agents::tool_execution::ToolCallContext;
    use crate::session::SessionManager;
    use rmcp::model::RawContent;
    use std::sync::Arc;
    use tokio_util::sync::CancellationToken;

    fn create_test_context() -> PlatformExtensionContext {
        PlatformExtensionContext {
            extension_manager: None,
            session_manager: Arc::new(SessionManager::instance()),
            session: None,
        }
    }

    #[tokio::test]
    async fn test_submit_task_report_returns_friendly_error_when_tasks_are_running() {
        let ctx = create_test_context();
        let engine = engine::get_engine_handle();
        let client = BetterSummonClient::new(ctx, engine.clone()).unwrap();
        let session_id = "test-report-session";

        let _ = engine.send_cmd(EngineCommand::InjectEvent {
            session_id: SessionId(Arc::from(session_id.to_string())),
            event: BgEv::Spawned(TaskId("work-1".into())),
        });

        let call_ctx = ToolCallContext::new(session_id.to_string(), None, None);
        let result = client
            .call_tool(
                &call_ctx,
                "submit_task_report",
                None,
                CancellationToken::new(),
            )
            .await
            .unwrap();

        assert_eq!(result.is_error, Some(true));
        let text = match result.content.first().unwrap().raw {
            RawContent::Text(ref text) => text,
            _ => panic!("Expected text content"),
        };
        assert!(text.text.contains("目前无法提交最终报告"));
        assert!(text.text.contains("直接输出普通文本"));
        assert!(text.text.contains("自动发送新消息唤醒你"));
    }

    #[tokio::test]
    async fn test_submit_task_report_succeeds_when_no_background_tasks_remain() {
        let ctx = create_test_context();
        let engine = engine::get_engine_handle();
        let client = BetterSummonClient::new(ctx, engine.clone()).unwrap();
        let session_id = "test-report-session-complete";

        let call_ctx = ToolCallContext::new(session_id.to_string(), None, None);
        let result = client
            .call_tool(
                &call_ctx,
                "submit_task_report",
                None,
                CancellationToken::new(),
            )
            .await
            .unwrap();

        assert_eq!(result.is_error, Some(false));
        let text = match result.content.first().unwrap().raw {
            RawContent::Text(ref text) => text,
            _ => panic!("Expected text content"),
        };
        assert!(text.text.contains("报告提交成功"));
    }
}

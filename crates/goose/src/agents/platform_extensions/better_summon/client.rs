use crate::{
    agents::{mcp_client::{Error, McpClientTrait}, platform_extensions::PlatformExtensionContext, tool_execution::ToolCallContext, AgentConfig, GoosePlatform},
    config::{Config, GooseMode},
    recipe::{local_recipes::load_local_recipe_file, Recipe},
    session::extension_data::EnabledExtensionsState,
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use rmcp::model::{CallToolResult, Content, Implementation, InitializeResult, JsonObject, ListToolsResult, ServerCapabilities};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::info;

use super::{actor, subagent::SubagentRunParams, ENGINEER_HINT, ARCHITECT_HINT, COMMON_HINT};

pub struct BetterSummonClient {
    ctx: PlatformExtensionContext,
    info: InitializeResult,
    tk: CancellationToken,
}

impl Drop for BetterSummonClient {
    fn drop(&mut self) { self.tk.cancel(); }
}

impl BetterSummonClient {
    pub fn new(ctx: PlatformExtensionContext) -> Result<Self> {
        Ok(Self { ctx, info: InitializeResult::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("goose-better-summon", env!("CARGO_PKG_VERSION"))), tk: CancellationToken::new() })
    }

    async fn handle_delegate(&self, session_id: &str, instructions: &str) -> anyhow::Result<CallToolResult> {
        if instructions.is_empty() { anyhow::bail!("Instructions cannot be empty"); }
        info!("Delegating task: {}", instructions.split('\n').next().unwrap_or(""));
        let ps = self.ctx.session_manager.get_session(session_id, false).await.context("Failed to retrieve parent session")?;
        if ps.session_type == crate::session::session_manager::SessionType::SubAgent { anyhow::bail!("Subagents cannot delegate"); }
        let sid = format!("{:04X}", rand::random::<u16>());
        let mut rec = Recipe::from_content(instructions).unwrap_or_else(|_| Recipe::builder().title("Ad-hoc Task").description(instructions).prompt(instructions).build().unwrap());
        if !instructions.contains('\n') && instructions.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-' || c == '.') {
            if let Ok(f) = load_local_recipe_file(ps.working_dir.join(instructions).to_str().unwrap_or("")) {
                if let Ok(r) = Recipe::from_content(&f.content) { rec = r; }
            }
        }
        rec.instructions = Some(rec.instructions.unwrap_or_default());
        let p_name = rec.settings.as_ref().and_then(|s| s.goose_provider.clone()).or(ps.provider_name.clone()).unwrap_or_else(|| Config::global().get_param("GOOSE_PROVIDER").unwrap());
        let m_name = rec.settings.as_ref().and_then(|s| s.goose_model.clone()).or(ps.model_config.as_ref().map(|c| c.model_name.clone())).unwrap_or_else(|| Config::global().get_param("GOOSE_MODEL").unwrap());
        let mut m_cfg = crate::model::ModelConfig::new(&m_name)?;
        if let Some(t) = rec.settings.as_ref().and_then(|s| s.temperature) { m_cfg = m_cfg.with_temperature(Some(t)); }
        let provider = crate::providers::create(&p_name, m_cfg, vec![]).await?;
        let exts = EnabledExtensionsState::extensions_or_default(Some(&ps.extension_data), Config::global());
        let cfg = AgentConfig::new(self.ctx.session_manager.clone(), crate::config::permission::PermissionManager::instance(), None, GooseMode::Auto, true, GoosePlatform::GooseCli);
        let ssess = self.ctx.session_manager.create_session(ps.working_dir.clone(), format!("ENGINEER-{}", sid), crate::session::session_manager::SessionType::SubAgent, GooseMode::Auto).await.context("Failed to create sub-session")?;
        let _ = actor::dispatch_task(SubagentRunParams { config: cfg, recipe: rec, provider, extensions: exts, sub_id: sid.clone(), sess_id: ssess.id, p_sess_id: Arc::from(session_id), token: Some(self.tk.child_token()) });
        Ok(CallToolResult::success(vec![Content::text(format!("Engineer {} dispatched to background queue.", sid))]))
    }

    async fn is_subagent(&self, id: &str) -> bool {
        self.ctx.session_manager.get_session(id, false).await.is_ok_and(|s| s.session_type == crate::session::session_manager::SessionType::SubAgent)
    }
}

#[async_trait]
impl McpClientTrait for BetterSummonClient {
    async fn list_tools(&self, id: &str, _: Option<String>, _: CancellationToken) -> Result<ListToolsResult, Error> {
        Ok(ListToolsResult::with_all_items(if self.is_subagent(id).await { vec![super::REPORT_TOOL.clone()] } else { vec![super::REPORT_TOOL.clone(), super::DELEGATE_TOOL.clone()] }))
    }

    async fn call_tool(&self, ctx: &ToolCallContext, name: &str, args: Option<JsonObject>, _: CancellationToken) -> Result<CallToolResult, Error> {
        match name {
            "delegate" => {
                let a = args.unwrap_or_default();
                let inst = a.get("instructions").and_then(|v| v.as_str()).unwrap_or("");
                match self.handle_delegate(&ctx.session_id, inst).await {
                    Ok(r) => Ok(r),
                    Err(e) => Ok(CallToolResult::error(vec![Content::text(format!("Error: {}", e))])),
                }
            }
            "submit_task_report" => Ok(CallToolResult::success(vec![Content::text("Report submitted. Current task ending.")])),
            _ => Ok(CallToolResult::error(vec![Content::text(format!("Tool {} not found", name))])),
        }
    }

    async fn get_moim(&self, id: &str) -> Option<String> {
        let hint = if self.is_subagent(id).await { ENGINEER_HINT } else { ARCHITECT_HINT };
        Some(format!("{}{}\n*作为工程师，你必须调用 submit_task_report 工具来结束当前任务。*", hint, COMMON_HINT))
    }
    fn get_info(&self) -> Option<&InitializeResult> { Some(&self.info) }
}

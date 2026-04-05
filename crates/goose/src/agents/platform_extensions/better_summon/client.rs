use crate::{
    agents::{
        mcp_client::{Error, McpClientTrait},
        platform_extensions::PlatformExtensionContext,
        tool_execution::ToolCallContext,
        AgentConfig, GoosePlatform,
    },
    config::{Config, GooseMode},
    recipe::{local_recipes::load_local_recipe_file, Recipe},
    session::extension_data::EnabledExtensionsState,
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use rmcp::model::{
    CallToolResult, Content, Implementation, InitializeResult, JsonObject, ListToolsResult,
    ServerCapabilities,
};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::info;

use super::engine::{dispatch_task, route_event, BgEv};
use super::formats::{
    format_delegate_error, format_dispatch_message, format_hint, format_tool_not_found,
    DELEGATE_LOG_PREFIX, ERROR_CREATE_SUBSESSION, ERROR_EMPTY_INSTRUCTIONS, ERROR_PARENT_SESSION,
    ERROR_SUBAGENT_CANNOT_DELEGATE,
};
use super::tools::{DELEGATE_TOOL, REPORT_TOOL};
use super::worker::SubagentRunParams;

pub struct BetterSummonClient {
    ctx: PlatformExtensionContext,
    info: InitializeResult,
    tk: CancellationToken,
}

impl Drop for BetterSummonClient {
    fn drop(&mut self) {
        self.tk.cancel();
    }
}

impl BetterSummonClient {
    pub fn new(ctx: PlatformExtensionContext) -> Result<Self> {
        Ok(Self {
            ctx,
            info: InitializeResult::new(ServerCapabilities::builder().enable_tools().build())
                .with_server_info(Implementation::new(
                    "goose-better-summon",
                    env!("CARGO_PKG_VERSION"),
                )),
            tk: CancellationToken::new(),
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
            instructions.split('\n').next().unwrap_or("")
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
        let rec = if let Ok(mut r) = Recipe::from_content(instructions) {
            r.instructions = Some(r.instructions.unwrap_or_default());
            r
        } else {
            let mut found_rec = None;
            if !instructions.contains('\n')
                && instructions
                    .chars()
                    .all(|c| c.is_alphanumeric() || c == '_' || c == '-' || c == '.')
            {
                // Try standard recipe search
                if let Ok(f) = load_local_recipe_file(instructions) {
                    found_rec = Recipe::from_content(&f.content).ok();
                }
                // Try parent working dir + extensions
                if found_rec.is_none() {
                    for ext in crate::recipe::RECIPE_FILE_EXTENSIONS {
                        let path = ps.working_dir.join(format!("{}.{}", instructions, ext));
                        if let Ok(r) = Recipe::from_file_path(&path) {
                            found_rec = Some(r);
                            break;
                        }
                    }
                }
            }

            if let Some(mut r) = found_rec {
                r.instructions = Some(r.instructions.unwrap_or_default());
                r
            } else {
                Recipe::builder()
                    .title(format!("TASK-{}", sid))
                    .description(instructions)
                    .prompt(instructions)
                    .instructions(instructions) // Give it some default instructions if it's just a text prompt
                    .build()
                    .unwrap()
            }
        };

        let p_name = rec
            .settings
            .as_ref()
            .and_then(|s| s.goose_provider.clone())
            .or(ps.provider_name.clone())
            .unwrap_or_else(|| Config::global().get_param("GOOSE_PROVIDER").unwrap());
        let m_name = rec
            .settings
            .as_ref()
            .and_then(|s| s.goose_model.clone())
            .or(ps.model_config.as_ref().map(|c| c.model_name.clone()))
            .unwrap_or_else(|| Config::global().get_param("GOOSE_MODEL").unwrap());
        let mut m_cfg = if ps
            .model_config
            .as_ref()
            .is_some_and(|c| c.model_name == m_name)
        {
            ps.model_config.as_ref().unwrap().clone()
        } else {
            crate::model::ModelConfig::new(&m_name)?
        };
        if let Some(t) = rec.settings.as_ref().and_then(|s| s.temperature) {
            m_cfg = m_cfg.with_temperature(Some(t));
        }
        let provider = crate::providers::create(&p_name, m_cfg, vec![]).await?;
        let mut exts = EnabledExtensionsState::extensions_or_default(
            Some(&ps.extension_data),
            Config::global(),
        );
        if let Some(recipe_exts) = rec.extensions.as_ref() {
            for rext in recipe_exts {
                if !exts.iter().any(|e| e.name() == rext.name()) {
                    exts.push(rext.clone());
                }
            }
        }
        let cfg = AgentConfig::new(
            self.ctx.session_manager.clone(),
            crate::config::permission::PermissionManager::instance(),
            None,
            GooseMode::Auto,
            true,
            GoosePlatform::GooseCli,
        );
        let ssess = self
            .ctx
            .session_manager
            .create_session(
                ps.working_dir.clone(),
                format!("ENGINEER-{}", sid),
                crate::session::session_manager::SessionType::SubAgent,
                GooseMode::Auto,
            )
            .await
            .context(ERROR_CREATE_SUBSESSION)?;
        let p_sess_arc = Arc::from(session_id);
        let _ = dispatch_task(SubagentRunParams {
            config: cfg,
            recipe: rec,
            provider,
            extensions: exts,
            sub_id: sid.clone(),
            sess_id: ssess.id,
            p_sess_id: Arc::clone(&p_sess_arc),
            token: Some(self.tk.child_token()),
        });
        route_event(p_sess_arc, BgEv::Spawned(sid.clone()));
        Ok(CallToolResult::success(vec![Content::text(
            format_dispatch_message(&sid),
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
                let inst = a.get("instructions").and_then(|v| v.as_str()).unwrap_or("");
                match self.handle_delegate(&ctx.session_id, inst).await {
                    Ok(r) => Ok(r),
                    Err(e) => Ok(CallToolResult::error(vec![Content::text(
                        format_delegate_error(&e.to_string()),
                    )])),
                }
            }
            "submit_task_report" => Ok(CallToolResult::success(vec![Content::text("")])),
            _ => Ok(CallToolResult::error(vec![Content::text(
                format_tool_not_found(name),
            )])),
        }
    }

    async fn get_moim(&self, id: &str) -> Option<String> {
        let mut hint = format_hint(self.is_subagent(id).await);
        let (reports, task_ids) = super::engine::take_reports(id).await;
        if !reports.is_empty() {
            let reports_block = super::formats::render_report_prompt(&task_ids, 0, &reports);
            hint.push_str("\n\n");
            hint.push_str(&reports_block);
        }
        Some(hint)
    }

    fn get_info(&self) -> Option<&InitializeResult> {
        Some(&self.info)
    }
}

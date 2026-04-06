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

use super::engine::{dispatch_task, idle_engineer_count, route_event, BgEv};
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
    const HARD_CODED_RECIPE_NAME: &'static str = "developer";
    pub fn new(ctx: PlatformExtensionContext) -> Result<Self> {
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
        let rec = self.resolve_recipe(instructions, &ps.working_dir, &sid);
        let run_params = self.build_run_params(&ps, rec, sid.clone()).await?;

        let idle_count = idle_engineer_count().saturating_sub(1);
        let _ = dispatch_task(run_params);

        let p_sess_arc = Arc::from(session_id);
        route_event(p_sess_arc, BgEv::Spawned(sid.clone()));
        Ok(CallToolResult::success(vec![Content::text(
            format_dispatch_message(&sid, idle_count),
        )]))
    }

    fn resolve_recipe(
        &self,
        instructions: &str,
        _working_dir: &std::path::Path,
        _sid: &str,
    ) -> Recipe {
        let recipe_name = Config::global()
            .get_param::<String>("GOOSE_BETTER_SUMMON_DEFAULT_RECIPE")
            .unwrap_or_else(|_| Self::HARD_CODED_RECIPE_NAME.to_string());
        let f = load_local_recipe_file(&recipe_name).expect("default recipe load failed");
        let mut recipe = Recipe::from_content(&f.content).expect("default recipe must be valid");
        recipe.instructions = Some(recipe.instructions.unwrap_or_default());
        if instructions != recipe_name {
            recipe.prompt = Some(instructions.to_string());
        }
        recipe
    }

    async fn build_run_params(
        &self,
        ps: &crate::session::Session,
        rec: Recipe,
        sid: String,
    ) -> anyhow::Result<SubagentRunParams> {
        let resolve_param =
            |recipe_val: Option<String>, session_val: Option<String>, env_key: &'static str| {
                recipe_val
                    .or(session_val)
                    .unwrap_or_else(|| Config::global().get_param(env_key).unwrap())
            };

        let p_name = resolve_param(
            rec.settings.as_ref().and_then(|s| s.goose_provider.clone()),
            ps.provider_name.clone(),
            "GOOSE_PROVIDER",
        );

        let m_name = resolve_param(
            rec.settings.as_ref().and_then(|s| s.goose_model.clone()),
            ps.model_config.as_ref().map(|c| c.model_name.clone()),
            "GOOSE_MODEL",
        );

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
            let existing_names: std::collections::HashSet<String> = exts
                .iter()
                .map(|existing| existing.name().to_string())
                .collect();
            exts.extend(
                recipe_exts
                    .iter()
                    .cloned()
                    .filter(|rext| !existing_names.contains(&rext.name().to_string())),
            );
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

        Ok(SubagentRunParams {
            config: cfg,
            recipe: rec,
            provider,
            extensions: exts,
            sub_id: sid,
            sess_id: ssess.id,
            p_sess_id: Arc::from(ps.id.as_str()),
            token: Some(self.tk.child_token()),
        })
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

#[cfg(test)]
mod tests {
    use super::BetterSummonClient;
    use crate::agents::platform_extensions::PlatformExtensionContext;
    use crate::session::SessionManager;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Arc;
    use tempfile::TempDir;

    fn make_client() -> BetterSummonClient {
        let ctx = PlatformExtensionContext {
            extension_manager: None,
            session_manager: Arc::new(SessionManager::new(std::env::temp_dir())),
            session: None,
        };

        BetterSummonClient::new(ctx).expect("failed to create client")
    }

    #[test]
    fn resolve_recipe_prefers_developer_recipe_when_present() {
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let recipe_path = temp_dir.path().join("developer.yaml");
        fs::write(
            &recipe_path,
            r#"version: "1.0.0"
title: Developer Recipe
description: Developer wrapper
instructions: Use developer tools to complete tasks.
"#,
        )
        .expect("failed to write developer recipe");

        let current_dir = std::env::current_dir().expect("failed to get cwd");
        std::env::set_current_dir(temp_dir.path()).expect("failed to set cwd");

        let client = make_client();
        let recipe = client.resolve_recipe("perform delegated work", temp_dir.path(), "ABCD");

        assert_eq!(recipe.title, "Developer Recipe");
        assert_eq!(recipe.description, "Developer wrapper");
        assert_eq!(
            recipe.instructions.as_deref(),
            Some("Use developer tools to complete tasks.")
        );
        assert_eq!(recipe.prompt.as_deref(), Some("perform delegated work"));

        std::env::set_current_dir(current_dir).expect("failed to restore cwd");
    }

    #[test]
    #[should_panic(expected = "developer recipe load failed")]
    fn resolve_recipe_panics_when_developer_recipe_missing() {
        let client = make_client();
        let _ = client.resolve_recipe("just do something", PathBuf::from(".").as_path(), "ABCD");
    }
}

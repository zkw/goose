use crate::agents::extension::PlatformExtensionContext;
use crate::agents::mcp_client::{Error, McpClientTrait};
pub mod actor;
pub mod agent;
pub mod subagent;
use self::subagent::{
    run_subagent_task, SubagentRunParams, TaskConfig, DEFAULT_SUBAGENT_MAX_TURNS,
};
use crate::agents::tool_execution::ToolCallContext;
use crate::config::{Config, GooseMode};
use crate::conversation::message::Message;
use crate::recipe::local_recipes::load_local_recipe_file;
use crate::recipe::Recipe;
use crate::session::extension_data::EnabledExtensionsState;
use crate::session::session_manager::SessionType;
use anyhow::{Context, Result};
use async_trait::async_trait;
use dashmap::DashMap;
use rmcp::model::{
    CallToolResult, Content, Implementation, InitializeResult, JsonObject, ListToolsResult,
    ServerCapabilities, Tool,
};
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;
use tracing::info;

const ARCHITECT_HINT: &str = include_str!("architect_hint.md");
const ENGINEER_HINT: &str = include_str!("engineer_hint.md");
const COMMON_HINT: &str = include_str!("common_hint.md");

pub static EXTENSION_NAME: &str = "better_summon";

/// BetterSummonClient provides parallel background task delegation capabilities.
/// It implements McpClientTrait to expose `delegate` and `submit_task_report` tools.
pub struct BetterSummonClient {
    context: PlatformExtensionContext,
    info: InitializeResult,
    /// Maps sub-session IDs to their parent session IDs for event routing
    parent_registry: Arc<DashMap<String, String>>,
    /// Global limit on concurrent background tasks
    task_semaphore: Arc<Semaphore>,
    session_cancel_token: CancellationToken,
}

impl Drop for BetterSummonClient {
    fn drop(&mut self) {
        self.session_cancel_token.cancel();
    }
}

impl BetterSummonClient {
    pub fn new(context: PlatformExtensionContext) -> Result<Self> {
        let max_tasks = Config::global()
            .get_param::<usize>("GOOSE_MAX_BACKGROUND_TASKS")
            .unwrap_or(50);
        let info = InitializeResult::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new(
                "goose-better-summon",
                env!("CARGO_PKG_VERSION"),
            ));

        Ok(Self {
            context,
            info,
            parent_registry: Arc::new(DashMap::new()),
            task_semaphore: Arc::new(Semaphore::new(max_tasks)),
            session_cancel_token: CancellationToken::new(),
        })
    }

    /// Creates the 'delegate' tool which allows the Architect to spawn background subagents.
    fn create_delegate_tool(&self) -> Tool {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "instructions": {
                    "type": "string",
                    "description": "Specific task instructions for the background engineer"
                }
            },
            "required": ["instructions"]
        });

        Tool::new(
            "delegate",
            "Dispatches an engineer to execute tasks in the background. Results are automatically reported upon completion.",
            schema.as_object().unwrap().clone(),
        )
    }

    /// Creates the 'submit_task_report' tool used by subagents to end their task and report back.
    fn create_submit_task_report_tool(&self) -> Tool {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "task_report": {
                    "type": "string",
                    "description": "The final comprehensive report to be delivered to the Architect"
                }
            },
            "required": ["task_report"]
        });

        Tool::new(
            "submit_task_report",
            "Submits the final task report and terminates the current subagent session.",
            schema.as_object().unwrap().clone(),
        )
    }

    /// Handles the delegation logic: session creation, RAII guard setup, and background task spawning.
    async fn handle_delegate(
        &self,
        session_id: &str,
        instructions: &str,
    ) -> Result<CallToolResult, Error> {
        if instructions.is_empty() {
            anyhow::bail!("Instructions cannot be empty");
        }
        info!(
            "Delegating task: {}",
            instructions.split('\n').next().unwrap_or(""),
        );

        let parent_session = self
            .context
            .session_manager
            .get_session(session_id, false)
            .await
            .context("Failed to retrieve parent session")?;

        if parent_session.session_type == SessionType::SubAgent {
            anyhow::bail!("Subagents cannot delegats tasks further");
        }

        let permit = match self.task_semaphore.clone().try_acquire_owned() {
            Ok(permit) => permit,
            Err(_) => anyhow::bail!("Maximum concurrent background task limit reached"),
        };

        let id_raw = uuid::Uuid::new_v4().simple().to_string();
        let task_id = id_raw.get(..8).unwrap_or(&id_raw).to_uppercase();

        let working_dir = parent_session.working_dir.clone();
        let recipe = self.build_recipe(instructions);
        let provider = self.resolve_provider(&recipe, &parent_session).await?;

        let extensions = if let Some(recipe_extensions) = &recipe.extensions {
            recipe_extensions.clone()
        } else {
            EnabledExtensionsState::extensions_or_default(
                Some(&parent_session.extension_data),
                Config::global(),
            )
        };

        let task_config = TaskConfig::new(provider, session_id, &working_dir, extensions)
            .with_subagent_id(task_id.clone());

        let sub_session = self
            .context
            .session_manager
            .create_session(
                working_dir,
                format!("Engineer-{}", &task_id),
                SessionType::SubAgent,
                GooseMode::Auto,
            )
            .await
            .context("Failed to create sub-session")?;

        let agent_config = crate::agents::AgentConfig::new(
            self.context.session_manager.clone(),
            crate::config::permission::PermissionManager::instance(),
            None,
            GooseMode::Auto,
            true,
            crate::agents::GoosePlatform::GooseCli,
        );


        self.parent_registry
            .insert(sub_session.id.clone(), session_id.to_string());

        // RAII Guard that increments active task count and keeps the parent session alive
        let guard = actor::TaskGuard::new(session_id.to_string());

        let main_session_id = session_id.to_string();
        let sub_session_id = sub_session.id.clone();
        let task_id_bg = task_id.clone();
        let task_semaphore = self.task_semaphore.clone();

        let mut recipe_with_context = recipe;
        recipe_with_context.instructions =
            Some(recipe_with_context.instructions.unwrap_or_default());

        let (notif_tx, mut notif_rx) = tokio::sync::mpsc::unbounded_channel();
        let session_for_notifs = main_session_id.clone();
        // Relay subagent notifications (e.g. tool usage) to the parent "Thinking" UI
        tokio::spawn(async move {
            while let Some(notif) = notif_rx.recv().await {
                actor::deliver_event(
                    &session_for_notifs,
                    actor::BackgroundEvent::McpNotification(notif),
                );
            }
        });

        let session_cancel_token = self.session_cancel_token.clone();
        tokio::spawn(async move {
            let _permit = permit;
            let _guard = guard;

            info!("Engineer task {} started", task_id_bg);

            let result = run_subagent_task(SubagentRunParams {
                config: agent_config,
                recipe: recipe_with_context,
                task_config,
                return_last_only: true,
                session_id: sub_session_id.clone(),
                cancellation_token: Some(session_cancel_token.child_token()),
                on_message: None,
                notification_tx: Some(notif_tx),
            })
            .await;

            let idle = task_semaphore.available_permits();

            let quoted = match result {
                Ok(text) if text.is_empty() => "No report provided.".to_string(),
                Ok(text) => text,
                Err(e) => format!("Execution failed: {}", e),
            }
            .lines()
            .map(|line| {
                if line.is_empty() {
                    ">".to_string()
                } else {
                    format!("> {}", line)
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

            // Final delivery: deliver report to parent Architect
            actor::deliver_event(
                &main_session_id,
                actor::BackgroundEvent::TaskComplete(quoted, task_id_bg.clone(), idle),
            );
            info!("Engineer task {} finished", task_id_bg);
        });

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Engineer {} has been dispatched to the background.",
            task_id
        ))]))
    }

    /// Parses instructions to see if they refer to a local recipe file or just raw text.
    fn build_recipe(&self, user_task: &str) -> Recipe {
        let default_yaml = format!(
            r#"
version: "1.0.0"
title: "工程师"
description: "后台任务"
settings:
  max_turns: {}
"#,
            DEFAULT_SUBAGENT_MAX_TURNS
        );

        let mut recipe = load_local_recipe_file("developer")
            .ok()
            .and_then(|file| Recipe::from_content(&file.content).ok())
            .unwrap_or_else(|| Recipe::from_content(&default_yaml).unwrap());

        recipe.prompt = Some(user_task.to_string());

        if recipe.response.is_none() {
            recipe.response = Some(crate::recipe::Response {
                json_schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "task_report": {
                            "type": "string",
                            "description": "最终发给架构师的执行报告。请务必详细包含执行的过程、查看的源码或者验证的结果。"
                        }
                    },
                    "required": ["task_report"]
                })),
            });
        }

        recipe
    }

    /// Resolve a model provider for the subagent, prioritizing recipe instructions.
    async fn resolve_provider(
        &self,
        recipe: &Recipe,
        session: &crate::session::Session,
    ) -> anyhow::Result<Arc<dyn crate::providers::base::Provider>> {
        let settings = recipe.settings.as_ref();

        let provider_name = settings
            .and_then(|s| s.goose_provider.clone())
            .or_else(|| {
                Config::global()
                    .get_param::<String>("GOOSE_SUBAGENT_PROVIDER")
                    .ok()
            })
            .or_else(|| session.provider_name.clone())
            .context("未配置提供者")?;

        let mut model_config = match &session.model_config {
            Some(cfg) => cfg.clone(),
            None => crate::model::ModelConfig::new("default")
                .map(|c| c.with_canonical_limits(&provider_name))
                .context("无法创建模型配置")?,
        };

        if let Some(model) = settings.and_then(|s| s.goose_model.clone()).or_else(|| {
            Config::global()
                .get_param::<String>("GOOSE_SUBAGENT_MODEL")
                .ok()
        }) {
            model_config.model_name = model;
        }

        if let Some(temp) = settings.and_then(|s| s.temperature) {
            model_config = model_config.with_temperature(Some(temp));
        }

        crate::providers::create(&provider_name, model_config, Vec::new())
            .await
            .context("无法初始化提供者实例")
    }

    async fn is_subagent(&self, session_id: &str) -> bool {
        self.context
            .session_manager
            .get_session(session_id, false)
            .await
            .map(|s| s.session_type == SessionType::SubAgent)
            .unwrap_or(false)
    }
}

#[async_trait]
impl McpClientTrait for BetterSummonClient {
    async fn list_tools(
        &self,
        session_id: &str,
        _next_cursor: Option<String>,
        _cancel_token: CancellationToken,
    ) -> Result<ListToolsResult, Error> {
        let is_subagent = self.is_subagent(session_id).await;

        let mut tools = vec![
            self.create_submit_task_report_tool(),
        ];

        if !is_subagent {
            tools.push(self.create_delegate_tool());
        }

        Ok(ListToolsResult {
            tools,
            next_cursor: None,
            meta: None,
        })
    }

    async fn call_tool(
        &self,
        ctx: &ToolCallContext,
        name: &str,
        arguments: Option<JsonObject>,
        _cancel_token: CancellationToken,
    ) -> Result<CallToolResult, Error> {
        #[allow(clippy::result_large_err)]
        fn parse_tool_args<T: serde::de::DeserializeOwned>(
            arguments: Option<JsonObject>,
        ) -> Result<T, CallToolResult> {
            serde_json::from_value(serde_json::Value::Object(arguments.unwrap_or_default()))
                .map_err(|e| CallToolResult::error(vec![Content::text(format!("参数错误: {}", e))]))
        }

        match name {
            "delegate" => {
                #[derive(serde::Deserialize)]
                struct DelegateArgs {
                    instructions: String,
                }
                let args: DelegateArgs = match parse_tool_args(arguments) {
                    Ok(a) => a,
                    Err(e) => return Ok(e),
                };
                match self
                    .handle_delegate(
                        &ctx.session_id,
                        &args.instructions,
                    )
                    .await
                {
                    Ok(result) => Ok(result),
                    Err(error) => Ok(CallToolResult::error(vec![Content::text(format!(
                        "错误: {}",
                        error
                    ))])),
                }
            }

            "submit_task_report" => {
                #[derive(serde::Deserialize)]
                struct SubmitArgs {
                    task_report: String,
                }
                let args: SubmitArgs = match parse_tool_args(arguments) {
                    Ok(a) => a,
                    Err(e) => return Ok(e),
                };

                let agent_id = self
                    .id_registry
                    .get(&ctx.session_id)
                    .map(|r| r.value().to_string())
                    .unwrap_or_else(|| "未知工程师".to_string());
                let target_session_id = &ctx.session_id;

                // Immediate delivery to the parent session for real-time stream multiplexing.
                // The task_id is used by the Architect to identify which background task has reported.
                if let Some(parent_id) = self.parent_registry.get(target_session_id) {
                    actor::deliver_event(
                        parent_id.value(),
                        actor::BackgroundEvent::TaskComplete(
                            args.task_report.clone(),
                            agent_id,
                            self.task_semaphore.available_permits(),
                        ),
                    );
                }

                Ok(CallToolResult::success(vec![Content::text(
                    "任务报告已提交，任务结束。".to_string(),
                )]))
            }
            _ => Ok(CallToolResult::error(vec![Content::text(format!(
                "未知工具: {}",
                name
            ))])),
        }
    }

    async fn get_moim(&self, session_id: &str) -> Option<String> {
        let role_hint = if self.is_subagent(session_id).await {
            ENGINEER_HINT
        } else {
            ARCHITECT_HINT
        };
        let mut hint = format!("{}{}", role_hint, COMMON_HINT);
        hint.push_str("\n\n*注意：如果你是一个后台工程师，必须且只能调用 submit_task_report 工具来结束当前任务。*");
        Some(hint)
    }

    fn get_info(&self) -> Option<&InitializeResult> {
        Some(&self.info)
    }
}

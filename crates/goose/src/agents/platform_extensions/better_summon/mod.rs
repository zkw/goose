use crate::agents::mcp_client::{Error, McpClientTrait};
use crate::agents::platform_extensions::PlatformExtensionContext;
use crate::agents::tool_execution::ToolCallContext;
use crate::config::{Config, GooseMode};
use crate::recipe::local_recipes::load_local_recipe_file;
use crate::recipe::Recipe;
use crate::session::extension_data::EnabledExtensionsState;
use anyhow::{Context, Result};
use once_cell::sync::Lazy;
use rmcp::model::{CallToolResult, Content, InitializeResult, JsonObject, ListToolsResult, Tool};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::info;

pub mod actor;
pub mod agent;
pub mod subagent;
use crate::session::session_manager::SessionType;
use async_trait::async_trait;
use rmcp::model::{Implementation, ServerCapabilities};

const ARCHITECT_HINT: &str = include_str!("architect_hint.md");
const ENGINEER_HINT: &str = include_str!("engineer_hint.md");
const COMMON_HINT: &str = include_str!("common_hint.md");

pub static EXTENSION_NAME: &str = "better_summon";

pub struct BetterSummonClient {
    context: PlatformExtensionContext,
    info: InitializeResult,
    session_cancel_token: CancellationToken,
}

impl Drop for BetterSummonClient {
    fn drop(&mut self) {
        self.session_cancel_token.cancel();
    }
}

static DELEGATE_TOOL: Lazy<Tool> = Lazy::new(|| {
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
});

static SUBMIT_REPORT_TOOL: Lazy<Tool> = Lazy::new(|| {
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
});

impl BetterSummonClient {
    pub fn new(context: PlatformExtensionContext) -> Result<Self> {
        let info = InitializeResult::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new(
                "goose-better-summon",
                env!("CARGO_PKG_VERSION"),
            ));

        Ok(Self {
            context,
            info,
            session_cancel_token: CancellationToken::new(),
        })
    }

    async fn handle_delegate(
        &self,
        session_id: &str,
        instructions: &str,
    ) -> anyhow::Result<CallToolResult> {
        if instructions.is_empty() {
            anyhow::bail!("Instructions cannot be empty");
        }
        info!(
            "Delegating task: {}",
            instructions.split('\n').next().unwrap_or("")
        );

        let parent_session = self
            .context
            .session_manager
            .get_session(session_id, false)
            .await
            .context("Failed to retrieve parent session")?;
        if parent_session.session_type == SessionType::SubAgent {
            anyhow::bail!("Subagents cannot delegates tasks further");
        }

        let task_id = format!("{:04X}", rand::random::<u16>());
        let agent_name = format!("ENGINEER-{}", &task_id);

        let mut recipe = self.build_recipe(instructions)?;
        recipe.instructions = Some(recipe.instructions.unwrap_or_default());
        let provider = self.resolve_provider(&recipe, &parent_session).await?;

        let extensions = EnabledExtensionsState::extensions_or_default(
            Some(&parent_session.extension_data),
            Config::global(),
        );
        let agent_config = crate::agents::AgentConfig::new(
            self.context.session_manager.clone(),
            crate::config::permission::PermissionManager::instance(),
            None,
            GooseMode::Auto,
            true,
            crate::agents::GoosePlatform::GooseCli,
        );

        let sub_session = self
            .context
            .session_manager
            .create_session(
                parent_session.working_dir.clone(),
                agent_name.clone(),
                SessionType::SubAgent,
                GooseMode::Auto,
            )
            .await
            .context("Failed to create sub-session")?;

        let run_params = subagent::SubagentRunParams {
            config: agent_config,
            recipe,
            provider,
            extensions,
            parent_working_dir: parent_session.working_dir.clone(),
            subagent_id: task_id.clone(),
            session_id: sub_session.id.clone(),
            parent_session_id: Arc::from(session_id),
            cancellation_token: Some(self.session_cancel_token.child_token()),
        };

        if let Err(e) = actor::dispatch_task(run_params) {
            anyhow::bail!("Failed to dispatch task: {}", e);
        }

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Engineer {} dispatched to background queue.",
            task_id
        ))]))
    }

    fn build_recipe(&self, instructions: &str) -> anyhow::Result<Recipe> {
        if !instructions.contains('\n')
            && instructions
                .chars()
                .all(|c| c.is_alphanumeric() || c == '_' || c == '-' || c == '.')
        {
            let working_dir = self
                .context
                .session
                .as_ref()
                .map(|s| s.working_dir.clone())
                .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
            let path = working_dir.join(instructions);
            if let Ok(recipe_file) = load_local_recipe_file(path.to_str().unwrap_or("")) {
                if let Ok(recipe) = Recipe::from_content(&recipe_file.content) {
                    return Ok(recipe);
                }
            }
        }
        Recipe::from_content(instructions).or_else(|_| {
            Recipe::builder()
                .title("Ad-hoc Task")
                .description(instructions)
                .instructions(instructions)
                .build()
                .map_err(|e| anyhow::anyhow!("Recipe construction failed: {}", e))
        })
    }

    async fn resolve_provider(
        &self,
        recipe: &Recipe,
        parent_session: &crate::session::Session,
    ) -> Result<Arc<dyn crate::providers::base::Provider>> {
        let provider_name = recipe
            .settings
            .as_ref()
            .and_then(|s| s.goose_provider.clone())
            .or_else(|| parent_session.provider_name.clone())
            .unwrap_or_else(|| Config::global().get_param("GOOSE_PROVIDER").unwrap());

        let model = recipe
            .settings
            .as_ref()
            .and_then(|s| s.goose_model.clone())
            .or_else(|| {
                parent_session
                    .model_config
                    .as_ref()
                    .map(|c| c.model_name.clone())
            })
            .unwrap_or_else(|| Config::global().get_param("GOOSE_MODEL").unwrap());

        let mut model_config = crate::model::ModelConfig::new(&model)?;
        if let Some(settings) = &recipe.settings {
            if let Some(t) = settings.temperature {
                model_config = model_config.with_temperature(Some(t));
            }
        }

        crate::providers::create(&provider_name, model_config, Vec::new()).await
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
        let mut tools = vec![SUBMIT_REPORT_TOOL.clone()];
        if !is_subagent {
            tools.push(DELEGATE_TOOL.clone());
        }
        Ok(ListToolsResult::with_all_items(tools))
    }

    async fn call_tool(
        &self,
        ctx: &ToolCallContext,
        name: &str,
        arguments: Option<JsonObject>,
        _cancel_token: CancellationToken,
    ) -> Result<CallToolResult, Error> {
        match name {
            "delegate" => {
                let args: serde_json::Value =
                    serde_json::Value::Object(arguments.unwrap_or_default());
                let instructions = args
                    .get("instructions")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                match self.handle_delegate(&ctx.session_id, instructions).await {
                    Ok(res) => Ok(res),
                    Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                        "Error: {}",
                        e
                    ))])),
                }
            }
            "submit_task_report" => Ok(CallToolResult::success(vec![Content::text(
                "Report submitted. Current task ending.",
            )])),
            _ => Ok(CallToolResult::error(vec![Content::text(format!(
                "Tool {} not found",
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

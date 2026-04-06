use crate::agents::platform_extensions::better_summon::formats::ERROR_CREATE_SUBSESSION;
use crate::agents::platform_extensions::better_summon::utils::MessageExt;
use crate::{
    agents::{
        platform_extensions::PlatformExtensionContext, Agent, AgentConfig, AgentEvent,
        GoosePlatform, SessionConfig,
    },
    config::{Config, GooseMode},
    conversation::{message::Message, Conversation},
    recipe::{local_recipes::load_local_recipe_file, Recipe},
    session::extension_data::EnabledExtensionsState,
};
use anyhow::{Context, Result};
use futures::{future::pending, stream::BoxStream, StreamExt};
use rmcp::model::{
    LoggingLevel, LoggingMessageNotificationParam, Notification, ServerNotification,
};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::info;

use super::engine::{BgEv, EngineCommand};
use super::formats::MSG_MISSING_REPORT_AGENT;

pub struct SubagentRunParams {
    pub config: AgentConfig,
    pub recipe: Recipe,
    pub provider: Arc<dyn crate::providers::base::Provider>,
    pub extensions: Vec<crate::agents::extension::ExtensionConfig>,
    pub sub_id: String,
    pub sess_id: String,
    pub event_tx: Option<mpsc::Sender<EngineCommand>>,
    pub token: Option<CancellationToken>,
}

impl SubagentRunParams {
    pub async fn from_context(
        ctx: &PlatformExtensionContext,
        ps: &crate::session::Session,
        instructions: &str,
        sid: String,
        token: CancellationToken,
    ) -> anyhow::Result<Self> {
        let recipe = Self::resolve_recipe(instructions);
        let model_cfg = Self::negotiate_model_config(ps, &recipe)?;
        let provider = Self::build_provider(ps, &recipe, model_cfg.clone()).await?;
        let extensions = Self::merge_extensions(ps, &recipe);
        let config = Self::build_agent_config(ctx);
        let sess_id = Self::create_sub_session(ctx, ps, &sid).await?;

        Ok(Self {
            config,
            recipe,
            provider,
            extensions,
            sub_id: sid,
            sess_id,
            event_tx: None,
            token: Some(token),
        })
    }

    fn negotiate_model_config(
        ps: &crate::session::Session,
        recipe: &Recipe,
    ) -> anyhow::Result<crate::model::ModelConfig> {
        let model_name = Self::resolve_param(
            recipe.settings.as_ref().and_then(|s| s.goose_model.clone()),
            ps.model_config.as_ref().map(|c| c.model_name.clone()),
            "GOOSE_MODEL",
        )?;

        let mut model_cfg = if ps
            .model_config
            .as_ref()
            .is_some_and(|c| c.model_name == model_name)
        {
            ps.model_config.as_ref().unwrap().clone()
        } else {
            crate::model::ModelConfig::new(&model_name)?
        };

        if let Some(temperature) = recipe.settings.as_ref().and_then(|s| s.temperature) {
            model_cfg = model_cfg.with_temperature(Some(temperature));
        }

        Ok(model_cfg)
    }

    async fn build_provider(
        ps: &crate::session::Session,
        recipe: &Recipe,
        model_cfg: crate::model::ModelConfig,
    ) -> anyhow::Result<Arc<dyn crate::providers::base::Provider>> {
        let provider_name = Self::resolve_param(
            recipe
                .settings
                .as_ref()
                .and_then(|s| s.goose_provider.clone()),
            ps.provider_name.clone(),
            "GOOSE_PROVIDER",
        )?;

        crate::providers::create(&provider_name, model_cfg, vec![]).await
    }

    fn merge_extensions(
        ps: &crate::session::Session,
        recipe: &Recipe,
    ) -> Vec<crate::agents::extension::ExtensionConfig> {
        let mut extensions = EnabledExtensionsState::extensions_or_default(
            Some(&ps.extension_data),
            Config::global(),
        );

        if let Some(recipe_exts) = recipe.extensions.as_ref() {
            let existing_names: Vec<String> =
                extensions.iter().map(|existing| existing.name()).collect();
            extensions.extend(recipe_exts.iter().cloned().filter(|recipe_ext| {
                !existing_names
                    .iter()
                    .any(|existing| existing == &recipe_ext.name())
            }));
        }

        extensions
    }

    fn build_agent_config(ctx: &PlatformExtensionContext) -> AgentConfig {
        AgentConfig::new(
            ctx.session_manager.clone(),
            crate::config::permission::PermissionManager::instance(),
            None,
            GooseMode::Auto,
            true,
            GoosePlatform::GooseCli,
        )
    }

    async fn create_sub_session(
        ctx: &PlatformExtensionContext,
        ps: &crate::session::Session,
        sid: &str,
    ) -> anyhow::Result<String> {
        let ssess = ctx
            .session_manager
            .create_session(
                ps.working_dir.clone(),
                format!("ENGINEER-{}", sid),
                crate::session::session_manager::SessionType::SubAgent,
                GooseMode::Auto,
            )
            .await
            .context(ERROR_CREATE_SUBSESSION)?;

        Ok(ssess.id)
    }

    fn resolve_recipe(instructions: &str) -> Recipe {
        let recipe_name = Config::global()
            .get_param::<String>("GOOSE_BETTER_SUMMON_DEFAULT_RECIPE")
            .unwrap_or_else(|_| "developer".to_string());
        let f = load_local_recipe_file(&recipe_name).expect("default recipe load failed");
        let mut recipe = Recipe::from_content(&f.content).expect("default recipe must be valid");
        recipe.instructions = Some(recipe.instructions.unwrap_or_default());
        if instructions != recipe_name {
            recipe.prompt = Some(instructions.to_string());
        }
        recipe
    }

    fn resolve_param(
        recipe_val: Option<String>,
        session_val: Option<String>,
        env_key: &'static str,
    ) -> anyhow::Result<String> {
        if let Some(value) = recipe_val.or(session_val) {
            Ok(value)
        } else {
            Config::global().get_param(env_key).map_err(Into::into)
        }
    }
}

pub const SUBAGENT_TOOL_REQ_TYPE: &str = "subagent_tool_request";
const DEFAULT_MAX_TURNS: usize = 1000;

pub async fn run_subagent_task(params: SubagentRunParams) -> Result<String> {
    info!("Subagent {} starting", params.sess_id);
    let (_conv, rep) = run(params).await?;
    Ok(rep.unwrap_or_default())
}

async fn run(p: SubagentRunParams) -> Result<(Conversation, Option<String>)> {
    let SubagentRunParams {
        config,
        recipe,
        provider,
        extensions,
        sub_id,
        sess_id,
        event_tx,
        token,
    } = p;
    let ag = Arc::new(Agent::with_config(config));
    ag.update_provider(provider, &sess_id).await?;
    for ext in &extensions {
        let _ = ag.add_extension(ext.clone(), &sess_id).await;
    }
    ag.apply_recipe_components(recipe.response, true).await;
    ag.extend_system_prompt(
        "subagent_system".to_string(),
        recipe.instructions.unwrap_or_default(),
    )
    .await;

    let scfg = SessionConfig {
        id: sess_id.clone(),
        schedule_id: None,
        max_turns: recipe
            .settings
            .as_ref()
            .and_then(|s| s.max_turns)
            .map(|t| t as u32)
            .or_else(|| {
                Some(
                    Config::global()
                        .get_param("GOOSE_SUBAGENT_MAX_TURNS")
                        .unwrap_or(DEFAULT_MAX_TURNS) as u32,
                )
            }),
        retry_config: recipe.retry,
    };

    let mut current_msg = Message::user().with_text(recipe.prompt.unwrap_or_else(String::new));

    loop {
        match process_single_turn(
            &ag,
            current_msg.clone(),
            &sess_id,
            &scfg,
            token.clone(),
            event_tx.as_ref(),
            &sub_id,
        )
        .await
        {
            Ok(Some(report)) => return Ok((Conversation::new_unvalidated(vec![]), Some(report))),
            Ok(None) => {
                current_msg = Message::user()
                    .with_text(MSG_MISSING_REPORT_AGENT)
                    .with_generated_id()
                    .agent_only();
            }
            Err(e) => {
                if token.as_ref().map(|t| t.is_cancelled()).unwrap_or(false) {
                    return Err(e);
                }
                tracing::warn!("Subagent stream interrupted: {}", e);
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                current_msg = Message::user()
                    .with_text("There was a server error. Please retry and continue your work.")
                    .with_visibility(false, false)
                    .with_generated_id();
            }
        }
    }
}

async fn process_single_turn(
    ag: &Agent,
    current_msg: Message,
    sess_id: &str,
    scfg: &SessionConfig,
    token: Option<CancellationToken>,
    event_tx: Option<&mpsc::Sender<EngineCommand>>,
    sub_id: &str,
) -> Result<Option<String>> {
    let mut stream = create_reply_stream(
        ag,
        current_msg.clone(),
        sess_id,
        scfg.clone(),
        token.clone(),
    )
    .await?;
    let mut rep_found = None;

    let mut stream_handler = Box::pin(async move {
        while let Some(ev) = stream.next().await {
            match ev {
                Ok(AgentEvent::Message(msg)) => {
                    if rep_found.is_none() {
                        rep_found = super::tools::extract_report(&msg);
                    }
                    if let Some(n) = create_tool_notification(&msg, sub_id) {
                        if let Some(event_tx) = event_tx {
                            let session_id = Arc::from(sess_id);
                            let _ = event_tx.try_send(EngineCommand::RouteEvent {
                                session_id,
                                event: BgEv::Mcp(n),
                            });
                        }
                    }
                    if rep_found.is_some() {
                        return Ok(rep_found);
                    }
                }
                Ok(AgentEvent::HistoryReplaced(_)) => {}
                Ok(_) => {}
                Err(e) => return Err(e),
            }
        }
        Ok(rep_found)
    });

    tokio::select! {
        _ = async {
            if let Some(cancel_token) = &token {
                cancel_token.cancelled().await;
            } else {
                pending::<()>().await;
            }
        } => Err(anyhow::anyhow!("Cancelled")),
        result = &mut stream_handler => result,
    }
}

async fn create_reply_stream<'a>(
    ag: &'a Agent,
    current_msg: Message,
    sess_id: &str,
    scfg: SessionConfig,
    token: Option<CancellationToken>,
) -> Result<BoxStream<'a, anyhow::Result<AgentEvent>>> {
    loop {
        match crate::session_context::with_session_id(
            Some(sess_id.to_string()),
            ag.reply(current_msg.clone(), scfg.clone(), token.clone()),
        )
        .await
        {
            Ok(stream) => {
                return Ok(stream.map(|item| item.map_err(Into::into)).boxed());
            }
            Err(e) => {
                tracing::warn!("Subagent stream creation failed: {}, retrying", e);
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            }
        }
    }
}

pub fn create_tool_notification(msg: &Message, subagent_id: &str) -> Option<ServerNotification> {
    let (_, call) = msg.first_tool_request()?;
    Some(ServerNotification::LoggingMessageNotification(
        Notification::new(
            LoggingMessageNotificationParam::new(
                LoggingLevel::Info,
                serde_json::json!({
                    "type": SUBAGENT_TOOL_REQ_TYPE,
                    "subagent_id": subagent_id,
                    "tool_call": { "name": call.name, "arguments": call.arguments }
                }),
            )
            .with_logger(format!("sub:{}", subagent_id)),
        ),
    ))
}

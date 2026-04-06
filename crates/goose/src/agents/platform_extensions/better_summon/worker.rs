use crate::agents::platform_extensions::better_summon::utils::MessageExt;
use crate::{
    agents::{Agent, AgentConfig, AgentEvent, SessionConfig},
    config::Config,
    conversation::{message::Message, Conversation},
    recipe::Recipe,
};
use anyhow::Result;
use futures::{future::pending, StreamExt};
use rmcp::model::{
    LoggingLevel, LoggingMessageNotificationParam, Notification, ServerNotification,
};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::info;

use super::engine::{route_event, BgEv};
use super::formats::MSG_MISSING_REPORT_AGENT;

pub struct SubagentRunParams {
    pub config: AgentConfig,
    pub recipe: Recipe,
    pub provider: Arc<dyn crate::providers::base::Provider>,
    pub extensions: Vec<crate::agents::extension::ExtensionConfig>,
    pub sub_id: String,
    pub sess_id: String,
    pub p_sess_id: Arc<str>,
    pub token: Option<CancellationToken>,
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
        p_sess_id,
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
    let mut rep_found = None;
    let cancel_token = token.clone();

    tokio::select! {
        _ = async {
            if let Some(cancel_token) = &cancel_token {
                cancel_token.cancelled().await;
            } else {
                pending::<()>().await;
            }
        } => return Err(anyhow::anyhow!("Cancelled")),
        result = async move {
            loop {
                let mut stream = loop {
                    match crate::session_context::with_session_id(
                        Some(sess_id.clone()),
                        ag.reply(current_msg.clone(), scfg.clone(), token.clone()),
                    )
                    .await
                    {
                        Ok(s) => break s,
                        Err(e) => {
                            tracing::warn!("Subagent stream creation failed: {}, retrying", e);
                            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                        }
                    }
                };

                let mut stream_errored = false;

                while let Some(ev) = stream.next().await {
                    match ev {
                        Ok(AgentEvent::Message(msg)) => {
                            if rep_found.is_none() {
                                rep_found = super::tools::extract_report(&msg);
                            }
                            if let Some(n) = create_tool_notification(&msg, &sub_id) {
                                route_event(Arc::clone(&p_sess_id), BgEv::Mcp(n));
                            }
                            if rep_found.is_some() {
                                break;
                            }
                        }
                        Ok(AgentEvent::HistoryReplaced(_)) => {}
                        Ok(_) => {}
                        Err(e) => {
                            tracing::warn!("Subagent stream interrupted: {}", e);
                            stream_errored = true;
                            break;
                        }
                    }
                }

                if rep_found.is_some() {
                    break;
                }

                if stream_errored {
                    current_msg = Message::user()
                        .with_text("There was a server error. Please retry and continue your work.")
                        .with_visibility(false, false)
                        .with_generated_id();
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                } else {
                    current_msg = Message::user()
                        .with_text(MSG_MISSING_REPORT_AGENT)
                        .with_generated_id()
                        .agent_only();
                }
            }

            Ok((Conversation::new_unvalidated(vec![]), rep_found))
        } => result,
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

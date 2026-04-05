use crate::{
    agents::{Agent, AgentConfig, AgentEvent, SessionConfig},
    config::Config,
    conversation::{
        message::{Message, MessageContent},
        Conversation,
    },
    recipe::Recipe,
};
use anyhow::Result;
use futures::StreamExt;
use rmcp::model::{
    LoggingLevel, LoggingMessageNotificationParam, Notification, ServerNotification,
};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::info;

use super::engine::{route_event, BgEv};
use super::formats::{format_stream_terminated, NO_TEXT_CONTENT};

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
    let (conv, rep) = run(params).await?;
    if let Some(r) = rep {
        return Ok(r);
    }
    Ok(conv
        .messages()
        .last()
        .and_then(|m| {
            m.content.iter().find_map(|c| {
                if let MessageContent::Text(t) = c {
                    Some(t.text.clone())
                } else {
                    None
                }
            })
        })
        .unwrap_or_else(|| NO_TEXT_CONTENT.to_string()))
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

    let mut conv = Conversation::new_unvalidated(vec![
        Message::user().with_text(recipe.prompt.unwrap_or_else(String::new))
    ]);
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

    let mut stream = crate::session_context::with_session_id(
        Some(sess_id.clone()),
        ag.reply(conv.messages().last().unwrap().clone(), scfg, token),
    )
    .await?;
    let mut rep_found = None;

    while let Some(ev) = stream.next().await {
        match ev {
            Ok(AgentEvent::Message(msg)) => {
                if rep_found.is_none() {
                    rep_found = msg.content.iter().find_map(|c| {
                        let req = match c {
                            MessageContent::ToolRequest(r) => r,
                            _ => return None,
                        };
                        let call = req.tool_call.as_ref().ok()?;
                        (call.name == "submit_task_report")
                            .then(|| {
                                call.arguments
                                    .as_ref()?
                                    .get("task_report")?
                                    .as_str()
                                    .map(String::from)
                            })
                            .flatten()
                    });
                }
                if let Some(n) = create_tool_notification(&msg, &sub_id) {
                    route_event(Arc::clone(&p_sess_id), BgEv::Mcp(n));
                }
                conv.push(msg);
            }
            Ok(AgentEvent::HistoryReplaced(u)) => conv = u,
            Ok(_) => {}
            Err(e) => {
                tracing::warn!("Subagent stream interrupted: {}", e);
                conv.push(
                    Message::user()
                        .with_text(format_stream_terminated(&e.to_string()))
                        .with_visibility(false, false),
                );
                break;
            }
        }
    }
    Ok((conv, rep_found))
}

pub fn create_tool_notification(msg: &Message, subagent_id: &str) -> Option<ServerNotification> {
    let call = msg.content.iter().find_map(|c| {
        if let MessageContent::ToolRequest(req) = c {
            req.tool_call.as_ref().ok()
        } else {
            None
        }
    })?;
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

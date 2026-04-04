use crate::{
    agents::{Agent, AgentConfig, AgentEvent, SessionConfig},
    config::Config,
    conversation::{
        message::{Message, MessageContent},
        Conversation,
    },
    recipe::Recipe,
};
use anyhow::{anyhow, Result};
use futures::StreamExt;
use rmcp::model::{
    ErrorCode, ErrorData, LoggingLevel, LoggingMessageNotificationParam, Notification,
    ServerNotification,
};
use std::path::PathBuf;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::info;

pub struct SubagentRunParams {
    pub config: AgentConfig,
    pub recipe: Recipe,
    pub provider: Arc<dyn crate::providers::base::Provider>,
    pub extensions: Vec<crate::agents::extension::ExtensionConfig>,
    pub parent_working_dir: PathBuf,
    pub subagent_id: String,
    pub session_id: String,
    pub parent_session_id: Arc<str>,
    pub cancellation_token: Option<CancellationToken>,
}

pub const SUBAGENT_TOOL_REQ_TYPE: &str = "subagent_tool_request";
pub const DEFAULT_SUBAGENT_MAX_TURNS: usize = 1000;

pub async fn run_subagent_task(params: SubagentRunParams) -> Result<String, anyhow::Error> {
    info!("Subagent task starting in session {}", params.session_id);
    let (messages, task_report) = get_agent_messages(params).await.map_err(|e| {
        ErrorData::new(
            ErrorCode::INTERNAL_ERROR,
            format!("Subagent execution failed: {}", e),
            None,
        )
    })?;

    if let Some(output) = task_report {
        return Ok(output);
    }

    Ok(extract_response_text(&messages))
}

fn extract_response_text(messages: &Conversation) -> String {
    messages
        .messages()
        .last()
        .and_then(|m| m.content.iter().find_map(|c| {
            if let MessageContent::Text(t) = c { Some(t.text.clone()) } else { None }
        }))
        .unwrap_or_else(|| "No text content available".into())
}

async fn get_agent_messages(params: SubagentRunParams) -> Result<(Conversation, Option<String>)> {
    let SubagentRunParams {
        config,
        recipe,
        provider,
        extensions,
        session_id,
        parent_session_id,
        cancellation_token,
        subagent_id,
        ..
    } = params;

    let system_instructions = recipe.instructions.clone().unwrap_or_default();
    let user_task = recipe
        .prompt
        .clone()
        .unwrap_or_else(|| "Begin.".to_string());

    let agent = Arc::new(Agent::with_config(config));
    agent.update_provider(provider, &session_id).await?;

    for extension in &extensions {
        let _ = agent.add_extension(extension.clone(), &session_id).await;
    }

    agent
        .extend_system_prompt("subagent_system".to_string(), system_instructions)
        .await;

    let mut conversation =
        Conversation::new_unvalidated(vec![Message::user().with_text(&user_task)]);
    let session_config = SessionConfig {
        id: session_id.clone(),
        schedule_id: None,
        max_turns: Some(
            Config::global()
                .get_param::<usize>("GOOSE_SUBAGENT_MAX_TURNS")
                .unwrap_or(DEFAULT_SUBAGENT_MAX_TURNS) as u32,
        ),
        retry_config: recipe.retry,
    };

    let current_user_message = Message::user().with_text(user_task);

    let mut stream = crate::session_context::with_session_id(Some(session_id.clone()), async {
        agent
            .reply(current_user_message, session_config, cancellation_token)
            .await
    })
    .await
    .map_err(|e| anyhow!("Stream error: {}", e))?;

    let mut task_report_found = None;

    while let Some(message_result) = stream.next().await {
        match message_result {
            Ok(AgentEvent::Message(msg)) => {
                if task_report_found.is_none() {
                    task_report_found = msg.content.iter().find_map(|c| {
                        let req = match c { MessageContent::ToolRequest(r) => r, _ => return None };
                        let call = req.tool_call.as_ref().ok()?;
                        (call.name == "submit_task_report").then(|| {
                            call.arguments.as_ref()?.get("task_report")?.as_str().map(String::from)
                        }).flatten()
                    });
                }

                if let Some(n) = create_tool_notification(&msg, &subagent_id) {
                    super::actor::route_event(Arc::clone(&parent_session_id), super::actor::BgEvent::McpNotification(n));
                }
                conversation.push(msg);
            }
            Ok(AgentEvent::HistoryReplaced(updated)) => conversation = updated,
            Ok(_) => {}
            Err(e) => {
                tracing::warn!("Subagent stream interrupted: {}", e);
                let error_msg = Message::user()
                    .with_text(format!(
                        "System Error: Stream abruptly terminated due to {}. Output may be incomplete.",
                        e
                    ))
                    .with_visibility(false, false);
                conversation.push(error_msg);
                break;
            }
        }
    }

    Ok((conversation, task_report_found))
}

pub fn create_tool_notification(msg: &Message, subagent_id: &str) -> Option<ServerNotification> {
    let call = msg.content.iter().find_map(|c| {
        if let MessageContent::ToolRequest(req) = c { req.tool_call.as_ref().ok() } else { None }
    })?;

    Some(ServerNotification::LoggingMessageNotification(
        Notification::new(
            LoggingMessageNotificationParam::new(LoggingLevel::Info, serde_json::json!({
                "type": SUBAGENT_TOOL_REQ_TYPE,
                "subagent_id": subagent_id,
                "tool_call": { "name": call.name, "arguments": call.arguments }
            })).with_logger(format!("subagent:{}", subagent_id)),
        ),
    ))
}

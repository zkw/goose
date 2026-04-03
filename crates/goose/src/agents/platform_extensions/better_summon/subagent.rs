use crate::{
    agents::extension::ExtensionConfig,
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
use serde::Serialize;
use std::fmt;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

pub type OnMessageCallback = Arc<dyn Fn(&Message) + Send + Sync>;

type AgentMessagesFuture =
    Pin<Box<dyn Future<Output = Result<(Conversation, Option<String>)>> + Send>>;

pub struct SubagentRunParams {
    pub config: AgentConfig,
    pub recipe: Recipe,
    pub task_config: TaskConfig,
    pub return_last_only: bool,
    pub session_id: String,
    pub cancellation_token: Option<CancellationToken>,
    pub on_message: Option<OnMessageCallback>,
    pub notification_tx: Option<tokio::sync::mpsc::UnboundedSender<ServerNotification>>,
}

/// Executes a subagent task and returns the final report or the last message text.
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

    Ok(extract_response_text(&messages, true))
}

fn extract_response_text(messages: &Conversation, return_last_only: bool) -> String {
    if return_last_only {
        messages
            .messages()
            .last()
            .and_then(|message| {
                message.content.iter().find_map(|content| match content {
                    crate::conversation::message::MessageContent::Text(text_content) => {
                        Some(text_content.text.clone())
                    }
                    _ => None,
                })
            })
            .unwrap_or_else(|| String::from("No text content available"))
    } else {
        messages
            .iter()
            .flat_map(|message| {
                message.content.iter().filter_map(|content| match content {
                    crate::conversation::message::MessageContent::Text(text_content) => {
                        Some(text_content.text.clone())
                    }
                    crate::conversation::message::MessageContent::ToolResponse(tool_res) => {
                        if let Ok(res) = &tool_res.tool_result {
                            let texts: Vec<String> = res.content.iter().filter_map(|c| {
                                if let rmcp::model::RawContent::Text(t) = &c.raw {
                                    Some(t.text.clone())
                                } else { None }
                            }).collect();
                            (!texts.is_empty()).then(|| format!("Tool result: {}", texts.join("\n")))
                        } else { None }
                    }
                    _ => None,
                })
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

pub const SUBAGENT_TOOL_REQUEST_TYPE: &str = "subagent_tool_request";

/// Manages the full lifecycle of a subagent session.
/// Implements a "Tag-and-Restart" loop where the LLM is prompted to continue or finish
/// until it explicitly calls `submit_task_report`.
fn get_agent_messages(params: SubagentRunParams) -> AgentMessagesFuture {
    Box::pin(async move {
        let (config, recipe, task_config, session_id, cancellation_token, on_message, notification_tx) = (
            params.config, params.recipe, params.task_config, params.session_id,
            params.cancellation_token, params.on_message, params.notification_tx
        );

        let system_instructions = recipe.instructions.clone().unwrap_or_default();
        let user_task = recipe.prompt.clone().unwrap_or_else(|| "Begin.".to_string());

        let agent = Arc::new(Agent::with_config(config));
        agent.update_provider(task_config.provider.clone(), &session_id).await?;

        for extension in &task_config.extensions {
            let _ = agent.add_extension(extension.clone(), &session_id).await;
        }

        // Subagent receives the raw Architect instructions as its system prompt.
        agent.extend_system_prompt("subagent_system".to_string(), system_instructions).await;

        let mut conversation = Conversation::new_unvalidated(vec![Message::user().with_text(&user_task)]);
        let session_config = SessionConfig {
            id: session_id.clone(),
            schedule_id: None,
            max_turns: task_config.max_turns.map(|v| v as u32),
            retry_config: recipe.retry,
        };

        let mut current_user_message = Message::user().with_text(user_task);

        loop {
            let mut stream = crate::session_context::with_session_id(Some(session_id.clone()), async {
                super::agent::BetterAgent::new(&agent)
                    .reply(current_user_message.clone(), session_config.clone(), cancellation_token.clone())
                    .await
            }).await.map_err(|e| anyhow!("Stream error: {}", e))?;

            let mut task_report_found = None;

            while let Some(message_result) = stream.next().await {
                match message_result {
                    Ok(AgentEvent::Message(msg)) => {
                        for content in &msg.content {
                            // Intercept the submission tool call to extract the final report
                            if let MessageContent::ToolRequest(tools) = content {
                                if let Ok(call) = &tools.tool_call {
                                    if call.name == "submit_task_report" {
                                        if let Some(report) = call.arguments.as_ref().and_then(|a| a.get("task_report")).and_then(|v| v.as_str()) {
                                            task_report_found = Some(report.to_string());
                                        }
                                    }
                                }
                            }

                            // Relay tool usage as thinking messages back to the Architect's session
                            if let Some(ref tx) = notification_tx {
                                if let Some(notif) = create_tool_notification(content, &task_config.subagent_id) {
                                    let _ = tx.send(notif);
                                }
                            }
                        }
                        if let Some(ref callback) = on_message { callback(&msg); }
                        conversation.push(msg);
                    }
                    Ok(AgentEvent::HistoryReplaced(updated)) => conversation = updated,
                    Ok(_) => {}
                    Err(e) => {
                        tracing::warn!("Subagent stream interrupted: {}", e);
                        break;
                    }
                }
            }

            if let Some(report) = task_report_found {
                return Ok((conversation, Some(report)));
            }

            // Nag the LLM if it stopped talking without submitting a report
            current_user_message = Message::user().with_text(
                "Continue if unfinished; call `submit_task_report` explicitly if task is complete."
            );
            conversation.push(current_user_message.clone());
        }
    })
}

pub fn create_tool_notification(content: &MessageContent, subagent_id: &str) -> Option<ServerNotification> {
    let MessageContent::ToolRequest(req) = content else { return None; };
    let tool_call = req.tool_call.as_ref().ok()?;

    Some(ServerNotification::LoggingMessageNotification(
        Notification::new(
            LoggingMessageNotificationParam::new(
                LoggingLevel::Info,
                serde_json::json!({
                    "type": SUBAGENT_TOOL_REQUEST_TYPE,
                    "subagent_id": subagent_id,
                    "tool_call": {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments
                    }
                }),
            )
            .with_logger(format!("subagent:{}", subagent_id)),
        )
    ))
}

#[derive(Clone)]
pub struct TaskConfig {
    pub provider: Arc<dyn crate::providers::base::Provider>,
    pub parent_session_id: String,
    pub parent_working_dir: PathBuf,
    pub extensions: Vec<ExtensionConfig>,
    pub max_turns: Option<usize>,
    pub subagent_id: String,
}

impl fmt::Debug for TaskConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TaskConfig")
            .field("parent_session_id", &self.parent_session_id)
            .field("max_turns", &self.max_turns)
            .field("subagent_id", &self.subagent_id)
            .finish()
    }
}

pub const DEFAULT_SUBAGENT_MAX_TURNS: usize = 1000;

impl TaskConfig {
    pub fn new(
        provider: Arc<dyn crate::providers::base::Provider>,
        parent_session_id: &str,
        parent_working_dir: &Path,
        extensions: Vec<ExtensionConfig>,
    ) -> Self {
        Self {
            provider,
            parent_session_id: parent_session_id.to_owned(),
            parent_working_dir: parent_working_dir.to_owned(),
            extensions,
            subagent_id: String::new(),
            max_turns: Some(
                Config::global()
                    .get_param::<usize>("GOOSE_SUBAGENT_MAX_TURNS")
                    .unwrap_or(DEFAULT_SUBAGENT_MAX_TURNS),
            ),
        }
    }

    pub fn with_subagent_id(mut self, subagent_id: String) -> Self {
        self.subagent_id = subagent_id;
        self
    }

    pub fn with_max_turns(mut self, max_turns: Option<usize>) -> Self {
        if let Some(turns) = max_turns {
            self.max_turns = Some(turns);
        }
        self
    }
}

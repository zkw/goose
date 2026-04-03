use crate::{
    agents::extension::ExtensionConfig,
    agents::{Agent, AgentConfig, AgentEvent, SessionConfig},
    config::Config,
    conversation::{
        message::{Message, MessageContent},
        Conversation,
    },
    prompt_template::render_template,
    providers::base::Provider,
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

#[derive(Serialize)]
pub struct SubagentPromptContext {
    pub max_turns: usize,
    pub subagent_id: String,
    pub task_instructions: String,
    pub tool_count: usize,
    pub available_tools: String,
}

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

pub async fn run_subagent_task(params: SubagentRunParams) -> Result<String, anyhow::Error> {
    info!("Subagent task starting in session {}", params.session_id);
    let return_last_only = params.return_last_only;
    let (messages, final_output) = get_agent_messages(params).await.map_err(|e| {
        ErrorData::new(
            ErrorCode::INTERNAL_ERROR,
            format!("Failed to execute task: {}", e),
            None,
        )
    })?;

    if let Some(output) = final_output {
        return Ok(output);
    }

    Ok(extract_response_text(&messages, return_last_only))
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
            .unwrap_or_else(|| String::from("No text content in last message"))
    } else {
        let all_text_content: Vec<String> = messages
            .iter()
            .flat_map(|message| {
                message.content.iter().filter_map(|content| match content {
                    crate::conversation::message::MessageContent::Text(text_content) => {
                        Some(text_content.text.clone())
                    }
                    crate::conversation::message::MessageContent::ToolResponse(tool_response) => {
                        if let Ok(result) = &tool_response.tool_result {
                            let texts: Vec<String> = result
                                .content
                                .iter()
                                .filter_map(|content| {
                                    if let rmcp::model::RawContent::Text(raw_text_content) =
                                        &content.raw
                                    {
                                        Some(raw_text_content.text.clone())
                                    } else {
                                        None
                                    }
                                })
                                .collect();
                            if !texts.is_empty() {
                                Some(format!("Tool result: {}", texts.join("\n")))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    _ => None,
                })
            })
            .collect();

        all_text_content.join("\n")
    }
}

pub const SUBAGENT_TOOL_REQUEST_TYPE: &str = "subagent_tool_request";

fn get_agent_messages(params: SubagentRunParams) -> AgentMessagesFuture {
    Box::pin(async move {
        let SubagentRunParams {
            config,
            recipe,
            task_config,
            session_id,
            cancellation_token,
            on_message,
            notification_tx,
            ..
        } = params;

        let system_instructions = recipe.instructions.clone().unwrap_or_default();
        let user_task = recipe
            .prompt
            .clone()
            .unwrap_or_else(|| "Begin.".to_string());

        let agent = Arc::new(Agent::with_config(config));

        agent
            .update_provider(task_config.provider.clone(), &session_id)
            .await
            .map_err(|e| anyhow!("Failed to set provider on sub agent: {}", e))?;

        for extension in &task_config.extensions {
            if let Err(e) = agent.add_extension(extension.clone(), &session_id).await {
                debug!(
                    "Failed to add extension '{}' to subagent: {}",
                    extension.name(),
                    e
                );
            }
        }

        let subagent_prompt =
            build_subagent_prompt(&agent, &task_config, &session_id, system_instructions).await?;
        agent
            .extend_system_prompt("subagent_system".to_string(), subagent_prompt)
            .await;

        let mut conversation =
            Conversation::new_unvalidated(vec![Message::user().with_text(user_task.clone())]);

        if let Some(activities) = recipe.activities {
            for activity in activities {
                info!("Recipe activity: {}", activity);
            }
        }
        let session_config = SessionConfig {
            id: session_id.clone(),
            schedule_id: None,
            max_turns: task_config.max_turns.map(|v| v as u32),
            retry_config: recipe.retry,
        };

        let mut current_user_message = Message::user().with_text(user_task);

        loop {
            let mut stream =
                crate::session_context::with_session_id(Some(session_id.clone()), async {
                    let better_agent = super::agent::BetterAgent::new(&agent);
                    better_agent
                        .reply(
                            current_user_message.clone(),
                            session_config.clone(),
                            cancellation_token.clone(),
                        )
                        .await
                })
                .await
                .map_err(|e| anyhow!("Failed to get reply from agent: {}", e))?;

            let mut final_report_found = None;

            while let Some(message_result) = stream.next().await {
                match message_result {
                    Ok(AgentEvent::Message(msg)) => {
                        for content in &msg.content {
                            if let MessageContent::ToolRequest(tools) = content {
                                if let Ok(tool_call) = &tools.tool_call {
                                    if tool_call.name == "submit_task_report" {
                                        if let Some(args) = &tool_call.arguments {
                                            if let Some(report) =
                                                args.get("final_report").and_then(|v| v.as_str())
                                            {
                                                final_report_found = Some(report.to_string());
                                            }
                                        }
                                    }
                                }
                            }

                            if let Some(ref tx) = notification_tx {
                                if let Some(notif) =
                                    create_tool_notification(content, &task_config.subagent_id)
                                {
                                    if tx.send(notif).is_err() {
                                        debug!(
                                            "Notification receiver dropped for subagent {}",
                                            task_config.subagent_id
                                        );
                                    }
                                }
                            }
                        }

                        if let Some(ref callback) = on_message {
                            callback(&msg);
                        }

                        conversation.push(msg);
                    }
                    Ok(AgentEvent::McpNotification(_)) => {}
                    Ok(AgentEvent::HistoryReplaced(updated_conversation)) => {
                        conversation = updated_conversation;
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Stream interrupted for subagent: {}. Returning partial conversation.",
                            e
                        );
                        break;
                    }
                }
            }

            if let Some(report) = final_report_found {
                return Ok((conversation, Some(report)));
            }

            // LLM finished without calling submit_task_report
            current_user_message = Message::user().with_text(
                "如果你需要继续工作，可以直接继续；如果你已经完成了任务，请调用 `submit_task_report` 结束。"
            );
            conversation.push(current_user_message.clone());
        }
    })
}

async fn build_subagent_prompt(
    agent: &Agent,
    task_config: &TaskConfig,
    session_id: &str,
    system_instructions: String,
) -> Result<String> {
    let tools: Vec<_> = agent.list_tools(session_id, None).await;

    render_template(
        "subagent_system.md",
        &SubagentPromptContext {
            max_turns: task_config
                .max_turns
                .expect("TaskConfig always sets max_turns"),
            subagent_id: task_config.subagent_id.clone(),
            task_instructions: system_instructions,
            tool_count: tools.len(),
            available_tools: tools
                .iter()
                .map(|t| t.name.to_string())
                .collect::<Vec<_>>()
                .join(", "),
        },
    )
    .map_err(|e| anyhow!("Failed to render subagent system prompt: {}", e))
}

pub fn create_tool_notification(
    content: &MessageContent,
    subagent_id: &str,
) -> Option<ServerNotification> {
    if let MessageContent::ToolRequest(req) = content {
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
            ),
        ))
    } else {
        None
    }
}

#[derive(Clone)]
pub struct TaskConfig {
    pub provider: Arc<dyn Provider>,
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
            .field("parent_working_dir", &self.parent_working_dir)
            .field("max_turns", &self.max_turns)
            .field("extensions", &self.extensions)
            .field("subagent_id", &self.subagent_id)
            .finish()
    }
}

pub const DEFAULT_SUBAGENT_MAX_TURNS: usize = 10;

impl TaskConfig {
    pub fn new(
        provider: Arc<dyn Provider>,
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

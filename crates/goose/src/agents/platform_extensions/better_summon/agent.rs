use crate::agents::platform_extensions::better_summon::actor;
use crate::agents::types::SessionConfig;
use crate::conversation::message::Message;
use anyhow::Result;
use futures::stream::BoxStream;
use futures::StreamExt;
use tokio_util::sync::CancellationToken;

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct SubagentToolRequestNotification {
    #[serde(rename = "type")]
    pub msg_type: String,
    pub subagent_id: String,
    pub tool_call: crate::conversation::message::ToolRequest,
}

use crate::agents::agent::{Agent, AgentEvent};
pub use rmcp::model::ServerNotification;

pub struct BetterAgent<'a> {
    pub core: &'a Agent,
}

impl<'a> BetterAgent<'a> {
    pub fn new(core: &'a Agent) -> Self {
        Self { core }
    }

    pub async fn reply(
        self,
        user_message: Message,
        session_config: SessionConfig,
        cancel_token: Option<CancellationToken>,
    ) -> Result<BoxStream<'a, Result<AgentEvent>>> {
        let session_id = session_config.id.clone();
        let session_manager = self.core.config.session_manager.clone();
        
        let cancel_clone = cancel_token.clone();
        let inner_stream = self.core.reply(
            user_message,
            session_config,
            cancel_clone,
        ).await?;
        
        Ok(Self::wrap_stream(
            session_manager,
            session_id,
            inner_stream,
            cancel_token,
        ))
    }

    pub fn wrap_stream(
        session_manager: std::sync::Arc<crate::session::SessionManager>,
        session_id: String,
        mut inner_stream: BoxStream<'a, Result<AgentEvent>>,
        cancel_token: Option<CancellationToken>,
    ) -> BoxStream<'a, Result<AgentEvent>> {
        let mut bg_rx = actor::subscribe(&session_id);

        Box::pin(async_stream::try_stream! {
            let mut event_queue_active = true;
            let mut inner_active = true;

            loop {
                // If there are background tasks running, keep event queue active
                let active_tasks = actor::active_tasks(&session_id);
                if active_tasks == 0 {
                    event_queue_active = false;
                } else {
                    event_queue_active = true;
                }

                if !event_queue_active && !inner_active {
                    break;
                }

                tokio::select! {
                    ev_res = bg_rx.recv(), if event_queue_active => {
                        match ev_res {
                            Some(ev) => {
                                match ev {
                                    actor::BackgroundEvent::Message(msg) => {
                                        if let Err(e) = session_manager.add_message(&session_id, &msg).await {
                                            tracing::warn!("Failed to save background message to session: {}", e);
                                        }
                                        let yield_it = msg.metadata.user_visible
                                            && (!msg.as_concat_text().is_empty() || msg.is_tool_call());
                                        if yield_it {
                                            yield AgentEvent::Message(msg);
                                        }
                                    }
                                    actor::BackgroundEvent::McpNotification(notif) => {
                                        if let Some(msg) = Self::as_thinking_message(&notif) {
                                            yield AgentEvent::Message(msg);
                                        }
                                    }
                                    actor::BackgroundEvent::TaskComplete(report) => {
                                        if let Some(token) = &cancel_token {
                                            token.cancel();
                                        }
                                        yield AgentEvent::Message(Message::assistant().with_text(report));
                                        break;
                                    }
                                }
                            }
                            None => event_queue_active = false,
                        }
                    }
                    next_res = inner_stream.next(), if inner_active => {
                        match next_res {
                            Some(Ok(event)) => {
                                yield event;
                            }
                            Some(Err(e)) => {
                                yield Err(e)?;
                            }
                            None => {
                                inner_active = false;
                            }
                        }
                    }
                }
            }
        })
    }

    pub(crate) fn as_thinking_message(notif: &ServerNotification) -> Option<Message> {
        let ServerNotification::LoggingMessageNotification(log) = notif else {
            return None;
        };
        let parsed: SubagentToolRequestNotification =
            serde_json::from_value(log.params.data.clone()).unwrap_or_else(|_| {
                // Return a dummy to avoid bubbling Error if not parsable
                SubagentToolRequestNotification {
                    msg_type: "unknown".to_string(),
                    subagent_id: "unknown".to_string(),
                    tool_call: crate::conversation::message::ToolRequest {
                        id: "unknown".to_string(),
                        tool_call: Err(rmcp::model::ErrorData::new(rmcp::model::ErrorCode::INTERNAL_ERROR, "unknown".to_string(), None)),
                        metadata: None,
                        tool_meta: None,
                    },
                }
            });
            
        if parsed.msg_type != super::subagent::SUBAGENT_TOOL_REQUEST_TYPE {
            return None;
        }

        let subagent_id = &parsed.subagent_id;
        let tool_call = parsed.tool_call.tool_call.as_ref().ok()?;
        let tool_name = &tool_call.name;
        let arguments_opt = &tool_call.arguments;
        let arguments = arguments_opt.as_ref();
        let short_id = subagent_id.rsplit('_').next().unwrap_or(subagent_id);
        let tool_name_short = tool_name.split("__").last().unwrap_or(tool_name);
        let get_arg = |k: &str| arguments.and_then(|m| m.get(k)).and_then(|v: &serde_json::Value| v.as_str());

        let detail = get_arg("command")
            .or_else(|| get_arg("code"))
            .or_else(|| {
                ["path", "TargetFile", "AbsolutePath", "TargetDir"]
                    .iter()
                    .find_map(|&k| get_arg(k))
            })
            .map(|s: &str| s.replace('\n', " ").trim().to_string())
            .unwrap_or_else(|| {
                if arguments.map(|m| m.is_empty()).unwrap_or(true) {
                    "working...".to_string()
                } else {
                    let raw = serde_json::to_string(arguments_opt).unwrap_or_default();
                    let end = raw
                        .char_indices()
                        .nth(500)
                        .map(|(i, _)| i)
                        .unwrap_or(raw.len());
                    raw[..end].replace('\n', " ")
                }
            });

        const MAX_THINKING_SUMMARY_CHARS: usize = 157;
        let mut summary = format!("{}: {}", tool_name_short, detail);
        if let Some((cut, _)) = summary.char_indices().nth(MAX_THINKING_SUMMARY_CHARS) {
            summary.truncate(cut);
            summary.push_str("...");
        }

        Some(
            Message::assistant()
                .with_system_notification(
                    crate::conversation::message::SystemNotificationType::ThinkingMessage,
                    format!("工程师[{}] {}", short_id, summary),
                )
                .with_metadata(crate::conversation::message::MessageMetadata::user_only().with_user_invisible()),
        )
    }
}

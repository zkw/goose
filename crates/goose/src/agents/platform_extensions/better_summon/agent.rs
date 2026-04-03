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

/// BetterAgent is a non-intrusive wrapper around the core Agent.
/// It uses a parasitic AOP-style approach to intercept session streams and
/// multiplex background events (subagent reports and logs) into the active conversation.
pub struct BetterAgent<'a> {
    pub core: &'a Agent,
}

impl<'a> BetterAgent<'a> {
    pub fn new(core: &'a Agent) -> Self {
        Self { core }
    }

    /// Primary entry point that wraps the core agent's reply stream.
    pub async fn reply(
        self,
        user_message: Message,
        session_config: SessionConfig,
        cancel_token: Option<CancellationToken>,
    ) -> Result<BoxStream<'a, Result<AgentEvent>>> {
        let session_id = session_config.id.clone();
        let session_manager = self.core.config.session_manager.clone();

        let cancel_clone = cancel_token.clone();
        let inner_stream = self
            .core
            .reply(user_message, session_config, cancel_clone)
            .await?;

        Ok(Self::wrap_stream(
            session_manager,
            session_id,
            inner_stream,
            cancel_token,
        ))
    }

    /// Multiplexes background events from subagents into the core agent's response stream.
    /// background reports are merged into structured "Tag-and-Restart" messages that
    /// provide context to the Architect and progress logs to the User.
    pub fn wrap_stream(
        session_manager: std::sync::Arc<crate::session::SessionManager>,
        session_id: String,
        mut inner_stream: BoxStream<'a, Result<AgentEvent>>,
        cancel_token: Option<CancellationToken>,
    ) -> BoxStream<'a, Result<AgentEvent>> {
        let mut bg_rx = actor::subscribe(&session_id);

        Box::pin(async_stream::try_stream! {
            let mut inner_active = true;

            loop {
                let event_queue_active = actor::has_active_tasks(&session_id);

                // Terminate the stream only when both the core generation and background tasks are done
                if !event_queue_active && !inner_active {
                    break;
                }

                tokio::select! {
                    // Handle events from background subagents
                    ev_res = bg_rx.recv(), if event_queue_active => {
                        if let Some(ev) = ev_res {
                            match ev {
                                actor::BackgroundEvent::Message(msg) => {
                                    if let Err(e) = session_manager.add_message(&session_id, &msg).await {
                                        tracing::warn!("Failed to save background message: {}", e);
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
                                actor::BackgroundEvent::TaskComplete(report, agent_id, idle) => {
                                    let mut reports = vec![(report, agent_id, idle)];
                                    // Aggressive merging of concurrent reports to reduce conversation clutter
                                    while let Ok(next_ev) = bg_rx.try_recv() {
                                        match next_ev {
                                            actor::BackgroundEvent::TaskComplete(r, id, i) => reports.push((r, id, i)),
                                            actor::BackgroundEvent::Message(msg) => {
                                                if let Err(e) = session_manager.add_message(&session_id, &msg).await {
                                                    tracing::warn!("Failed to save background message: {}", e);
                                                }
                                                yield AgentEvent::Message(msg);
                                            }
                                            actor::BackgroundEvent::McpNotification(notif) => {
                                                if let Some(msg) = Self::as_thinking_message(&notif) {
                                                    yield AgentEvent::Message(msg);
                                                }
                                            }
                                        }
                                    }

                                    let report_ui_tmpl = include_str!("report_ui.md");
                                    let report_prompt_tmpl = include_str!("report_prompt.md");

                                    let mut logs = Vec::new();
                                    let mut ids = Vec::new();
                                    let mut results = Vec::new();
                                    let mut last_idle = 0;

                                    for (r, id, idle_count) in reports {
                                        let quoted = r.lines().map(|l| if l.is_empty() { ">".to_string() } else { format!("> {}", l) }).collect::<Vec<_>>().join("\n");
                                        logs.push(report_ui_tmpl.replace("{TASK_ID}", &id).replace("{RESULT}", &quoted));
                                        ids.push(id);
                                        results.push(format!("### 工程师 {} 的报告 ###\n{}", ids.last().unwrap(), quoted));
                                        last_idle = idle_count;
                                    }

                                    let combined_log = logs.join("\n\n---\n\n");
                                    let combined_result = results.join("\n\n");
                                    let combined_ids = ids.join(", ");

                                    // Display logs to user (agent-invisible)
                                    let log_msg = Message::assistant()
                                        .with_text(combined_log)
                                        .with_generated_id()
                                        .user_only();

                                    // Inject report into Architect's context to trigger the next reasoning cycle
                                    let trigger_text = report_prompt_tmpl
                                        .replace("{TASK_ID}", &combined_ids)
                                        .replace("{IDLE}", &last_idle.to_string())
                                        .replace("{RESULT}", &combined_result);

                                    let trigger_msg = Message::user()
                                        .with_text(trigger_text)
                                        .with_generated_id()
                                        .agent_only();

                                    if let Err(e) = session_manager.add_message(&session_id, &log_msg).await {
                                        tracing::warn!("Failed to save background log: {}", e);
                                    }
                                    yield AgentEvent::Message(log_msg);

                                    if let Err(e) = session_manager.add_message(&session_id, &trigger_msg).await {
                                        tracing::warn!("Failed to save background report trigger: {}", e);
                                    }
                                    yield AgentEvent::Message(trigger_msg);

                                    // If we were idling (waiting for report), cancel the silence and restart Architect
                                    if !inner_active {
                                        if let Some(token) = &cancel_token {
                                            token.cancel();
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    // Pass-through events from the core agent
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

    /// Formats a subagent's tool request notification into a thinking message for the parent UI.
    pub(crate) fn as_thinking_message(notif: &ServerNotification) -> Option<Message> {
        let ServerNotification::LoggingMessageNotification(log) = notif else {
            return None;
        };
        let parsed: SubagentToolRequestNotification =
            serde_json::from_value(log.params.data.clone()).unwrap_or_else(|_| {
                SubagentToolRequestNotification {
                    msg_type: "unknown".to_string(),
                    subagent_id: "unknown".to_string(),
                    tool_call: crate::conversation::message::ToolRequest {
                        id: "unknown".to_string(),
                        tool_call: Err(rmcp::model::ErrorData::new(
                            rmcp::model::ErrorCode::INTERNAL_ERROR,
                            "unknown".to_string(),
                            None,
                        )),
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
        let get_arg = |k: &str| {
            arguments
                .and_then(|m| m.get(k))
                .and_then(|v: &serde_json::Value| v.as_str())
        };

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
                    raw.get(..end).unwrap_or(&raw).replace('\n', " ")
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
                .with_metadata(
                    crate::conversation::message::MessageMetadata::user_only()
                        .with_user_invisible(),
                ),
        )
    }
}

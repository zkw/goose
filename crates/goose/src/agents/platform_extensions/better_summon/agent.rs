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

    /// Logical multi-turn wrapper that ensures the agent (main or subagent)
    /// provides a final report via `submit_task_report` before finishing.
    pub async fn reply(
        self,
        user_message: Message,
        session_config: SessionConfig,
        cancel_token: Option<CancellationToken>,
    ) -> Result<BoxStream<'a, Result<AgentEvent>>> {
        let session_id = session_config.id.clone();
        let session_manager = self.core.config.session_manager.clone();
        let agent = self.core;

        Ok(Box::pin(async_stream::try_stream! {
            let mut current_user_message = user_message;

            loop {
                let mut inner_stream = agent
                    .reply(current_user_message.clone(), session_config.clone(), cancel_token.clone())
                    .await?;

                let mut report_found = false;
                let mut inner_active = true;
                let mut bg_rx = actor::subscribe(&session_id);

                loop {
                    let event_queue_active = actor::has_active_tasks(&session_id);

                    if !event_queue_active && !inner_active {
                        break;
                    }

                    tokio::select! {
                        ev_res = bg_rx.recv(), if event_queue_active => {
                            if let Some(ev) = ev_res {
                                match ev {
                                    actor::BackgroundEvent::Message(msg) => {
                                        let _ = session_manager.add_message(&session_id, &msg).await;
                                        if msg.metadata.user_visible && (!msg.as_concat_text().is_empty() || msg.is_tool_call()) {
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
                                        while let Ok(next_ev) = bg_rx.try_recv() {
                                            match next_ev {
                                                actor::BackgroundEvent::TaskComplete(r, id, i) => reports.push((r, id, i)),
                                                actor::BackgroundEvent::Message(msg) => {
                                                    let _ = session_manager.add_message(&session_id, &msg).await;
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

                                        let log_msg = Message::assistant().with_text(logs.join("\n\n---\n\n")).with_generated_id().user_only();
                                        let trigger_msg = Message::user().with_text(
                                            report_prompt_tmpl.replace("{TASK_ID}", &ids.join(", "))
                                                .replace("{IDLE}", &last_idle.to_string())
                                                .replace("{RESULT}", &results.join("\n\n"))
                                        ).with_generated_id().agent_only();

                                        let _ = session_manager.add_message(&session_id, &log_msg).await;
                                        yield AgentEvent::Message(log_msg);
                                        let _ = session_manager.add_message(&session_id, &trigger_msg).await;
                                        yield AgentEvent::Message(trigger_msg);

                                        if !inner_active {
                                            if let Some(token) = &cancel_token { token.cancel(); }
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        next_res = inner_stream.next(), if inner_active => {
                            match next_res {
                                Some(Ok(event)) => {
                                    if let AgentEvent::Message(ref msg) = event {
                                        for content in &msg.content {
                                            if let crate::conversation::message::MessageContent::ToolRequest(tools) = content {
                                                if let Ok(call) = &tools.tool_call {
                                                    if call.name == "submit_task_report" {
                                                        report_found = true;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    yield event;
                                }
                                Some(Err(e)) => { yield Err(e)?; }
                                None => { inner_active = false; }
                            }
                        }
                    }
                }

                if report_found {
                    break;
                }

                // Tag-and-Restart: Force the agent (Main or Subagent) to submit a report
                current_user_message = Message::user().with_text(
                    "You must finish the task by calling `submit_task_report` with a detailed report. Direct closure is not permitted."
                );
                let _ = session_manager.add_message(&session_id, &current_user_message).await;
                yield AgentEvent::Message(current_user_message.clone());
            }
        }))
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

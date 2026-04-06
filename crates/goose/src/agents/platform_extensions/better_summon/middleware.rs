use crate::agents::platform_extensions::better_summon::utils::MessageExt;
use crate::{
    agents::{agent::Agent, types::SessionConfig, AgentEvent},
    conversation::message::{Message, MessageContent, SystemNotificationType},
    session::session_manager::SessionType,
};
use anyhow::Result;
use async_stream::stream;
use futures::{stream::BoxStream, StreamExt};
use std::{collections::VecDeque, sync::Arc};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::engine;
use super::formats::{
    build_thinking_message, render_no_report_ui, render_report_ui, THINKING_WORKING,
};

struct Guard(Arc<str>);
impl Drop for Guard {
    fn drop(&mut self) {
        engine::unbind_session(Arc::clone(&self.0));
    }
}

pub struct BetterAgent;

impl BetterAgent {
    pub async fn try_wrap<'a>(
        ag: &'a Agent,
        scfg: SessionConfig,
        inner: BoxStream<'a, Result<AgentEvent>>,
        tk: Option<CancellationToken>,
    ) -> BoxStream<'a, Result<AgentEvent>> {
        let extensions = ag.get_extension_configs().await;
        if !extensions.iter().any(|e| e.name() == super::EXTENSION_NAME) {
            return inner;
        }
        Self::wrap(ag, scfg, inner, tk)
    }

    fn wrap<'a>(
        ag: &'a Agent,
        scfg: SessionConfig,
        inner: BoxStream<'a, Result<AgentEvent>>,
        tk: Option<CancellationToken>,
    ) -> BoxStream<'a, Result<AgentEvent>> {
        let id_arc: Arc<str> = Arc::from(scfg.id.as_str());
        let (tx, mut rx) = mpsc::channel(32);
        let bound = engine::bind_session(Arc::clone(&id_arc), tx);

        let retry_msg = Message::user()
            .with_text("There was a server error. Please retry and continue your work.")
            .with_visibility(false, false)
            .with_generated_id();
        let recovery_config = scfg.clone();
        let retry_token = tk.clone();

        let mut current = inner;
        let mut ui_buffer = VecDeque::new();
        let mut is_safe = true;

        let pipeline = stream! {
            if !bound.await {
                while let Some(e) = current.next().await {
                    yield e;
                }
                return;
            }

            let _guard = Guard(Arc::clone(&id_arc));
            let id_str: &str = &id_arc;
            let is_sub = ag.config.session_manager.get_session(id_str, false)
                .await.is_ok_and(|s| s.session_type == SessionType::SubAgent);

            loop {
                tokio::select! {
                    maybe_item = current.next() => {
                        match maybe_item {
                            Some(Ok(event)) => {
                                let event = Self::normalize_agent_event(event, is_sub);
                                if let AgentEvent::Message(msg) = &event {
                                    is_safe = Self::is_safe_phase(msg);
                                }
                                if is_safe {
                                    while let Some(buffered) = ui_buffer.pop_front() {
                                        yield Ok(AgentEvent::Message(buffered));
                                    }
                                }
                                yield Ok(event);
                            }
                            Some(Err(error)) => {
                                tracing::warn!(%error, "Reply stream interrupted: initiating recovery");
                                loop {
                                    match ag.reply(retry_msg.clone(), recovery_config.clone(), retry_token.clone()).await {
                                        Ok(next_stream) => {
                                            current = next_stream;
                                            break;
                                        }
                                        Err(e) => {
                                            tracing::warn!(%e, "Reply recovery failed, retrying in 2s");
                                            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                                        }
                                    }
                                }
                            }
                            None => break,
                        }
                    }
                    bg_event = rx.recv(), if !rx.is_closed() => match bg_event {
                        Some(ev) => {
                            if let Some(msg) = Self::bg_ev_to_message(ev) {
                                if is_safe {
                                    while let Some(buffered) = ui_buffer.pop_front() {
                                        yield Ok(AgentEvent::Message(buffered));
                                    }
                                    yield Ok(AgentEvent::Message(msg));
                                } else {
                                    ui_buffer.push_back(msg);
                                }
                            }
                        }
                        None => {}
                    },
                }
            }

            while let Some(buffered) = ui_buffer.pop_front() {
                yield Ok(AgentEvent::Message(buffered));
            }
        };

        Box::pin(pipeline)
    }

    fn normalize_agent_event(event: AgentEvent, is_sub: bool) -> AgentEvent {
        match event {
            AgentEvent::Message(msg) => {
                AgentEvent::Message(Self::normalize_agent_message(msg, is_sub))
            }
            other => other,
        }
    }

    fn normalize_agent_message(mut msg: Message, is_sub: bool) -> Message {
        if !is_sub {
            if let Some((_, call)) = msg.tool_request("submit_task_report") {
                if let Some(text) = call
                    .arguments
                    .as_ref()
                    .and_then(|a| a.get("task_report"))
                    .and_then(|v| v.as_str())
                    .map(String::from)
                {
                    if let Some(replaced) =
                        msg.with_tool_request_replaced_by_text("submit_task_report", text)
                    {
                        msg = replaced;
                    }
                }
            }
        }
        msg
    }

    fn bg_ev_to_message(ev: engine::BgEv) -> Option<Message> {
        match ev {
            engine::BgEv::ToolCall {
                subagent_id,
                tool_name,
                tool_args,
            } => Self::as_thinking(&subagent_id, &tool_name, tool_args),
            engine::BgEv::Done(rep, aid) => Some(
                Message::assistant()
                    .with_text(render_report_ui(&aid, rep.trim_end()))
                    .with_generated_id()
                    .user_only(),
            ),
            engine::BgEv::NoReport(aid) => Some(
                Message::assistant()
                    .with_text(render_no_report_ui(&aid))
                    .with_generated_id()
                    .user_only(),
            ),
            _ => None,
        }
    }

    fn as_thinking(
        subagent_id: &str,
        tool_name: &str,
        args: Option<rmcp::model::JsonObject>,
    ) -> Option<Message> {
        let detail = ["command", "code", "path", "target_file"]
            .iter()
            .find_map(|k| args.as_ref().and_then(|m| m.get(*k)).and_then(|v| v.as_str()))
            .map(|s| s.replace('\n', " ").trim().to_string())
            .unwrap_or_else(|| THINKING_WORKING.to_string());
        let name = tool_name.split("__").last().unwrap_or(tool_name);
        let sm = if detail.len() > 150 {
            format!("{}...", detail.chars().take(147).collect::<String>())
        } else {
            detail
        };

        Some(
            Message::assistant()
                .with_content(MessageContent::system_notification(
                    SystemNotificationType::ThinkingMessage,
                    build_thinking_message(subagent_id, name, &sm),
                ))
                .with_generated_id()
                .with_visibility(false, false),
        )
    }

    fn is_safe_phase(msg: &Message) -> bool {
        if msg.content.is_empty() {
            return true;
        }
        match msg.content.last() {
            Some(MessageContent::ToolRequest(_)) => false,
            Some(MessageContent::ToolResponse(_)) => true,
            Some(MessageContent::Text(_)) | Some(MessageContent::SystemNotification(_)) => false,
            _ => false,
        }
    }
}

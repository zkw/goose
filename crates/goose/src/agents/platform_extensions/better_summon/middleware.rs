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

use super::engine::{self, BgEv, EngineHandle, SessionId};
use super::formats::{
    build_thinking_message, render_no_report_ui, render_report_prompt, render_report_ui,
};

struct Guard {
    handle: EngineHandle,
    session_id: SessionId,
}

impl Drop for Guard {
    fn drop(&mut self) {
        self.handle.unsubscribe(self.session_id.clone());
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
        let sess_id = SessionId(id_arc.clone());
        let (tx, mut rx) = mpsc::unbounded_channel();
        let engine = engine::get_engine_handle();
        let bound = engine.subscribe(sess_id.clone(), tx);

        let retry_msg = Message::user()
            .with_text("There was a server error. Please retry and continue your work.")
            .with_visibility(false, false)
            .with_generated_id();
        let recovery_config = scfg.clone();
        let retry_token = tk.clone();

        let mut current: Option<BoxStream<'a, Result<AgentEvent>>> = Some(inner);
        let mut ui_buffer = VecDeque::new();
        let mut is_safe = true;
        let mut unprompted_reports: Vec<(String, String)> = Vec::new();
        let mut has_tool_call = false;

        let pipeline = stream! {
            if bound.is_err() {
                while let Some(e) = current.as_mut().unwrap().next().await {
                    yield e;
                }
                return;
            }

            let _guard = Guard {
                handle: engine.clone(),
                session_id: sess_id.clone(),
            };
            let id_str: &str = &id_arc;
            let is_sub = ag.config.session_manager.get_session(id_str, false)
                .await.is_ok_and(|s| s.session_type == SessionType::SubAgent);

            loop {
                tokio::select! {
                    maybe_item = async {
                        if let Some(c) = current.as_mut() {
                            c.next().await
                        } else {
                            futures::future::pending().await
                        }
                    } => {
                        match maybe_item {
                            Some(Ok(event)) => {
                                let event = Self::normalize_agent_event(event, is_sub);
                                if let AgentEvent::Message(msg) = &event {
                                    is_safe = Self::is_safe_phase(msg);
                                    if msg.has_tool_request() {
                                        has_tool_call = true;
                                    }
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
                                            current = Some(next_stream);
                                            break;
                                        }
                                        Err(e) => {
                                            tracing::warn!(%e, "Reply recovery failed, retrying in 2s");
                                            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                                        }
                                    }
                                }
                            }
                            None => {
                                current = None;
                            }
                        }
                    }
                    bg_event = rx.recv(), if !rx.is_closed() => match bg_event {
                        Some(ev) => {
                            match &ev {
                                BgEv::Done(task_id, rep) => {
                                    unprompted_reports.push((task_id.0.clone(), rep.clone()));
                                }
                                _ => {}
                            }
                            if let Some(msg) = Self::bg_ev_to_message(ev) {
                                if is_safe || current.is_none() {
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

                if current.is_none() {
                    while let Some(buffered) = ui_buffer.pop_front() {
                        yield Ok(AgentEvent::Message(buffered));
                    }

                    if has_tool_call {
                        break;
                    }

                    if !is_sub {
                        let status = engine.query_status(sess_id.clone()).await;
                        let total_active = status.running_tasks + status.pending_tasks;

                        if !is_sub && !unprompted_reports.is_empty() {
                            let (tids, reps): (Vec<_>, Vec<_>) = unprompted_reports.drain(..).unzip();
                            let prompt = render_report_prompt(&tids, status.idle_count, &reps);
                            let tm = Message::user().with_text(prompt).with_generated_id().agent_only();

                            let mut retry_success = false;
                            loop {
                                if retry_token.as_ref().is_some_and(|t| t.is_cancelled()) {
                                    break;
                                }
                                match ag.reply(tm.clone(), recovery_config.clone(), retry_token.clone()).await {
                                    Ok(next_stream) => {
                                        current = Some(next_stream);
                                        retry_success = true;
                                        break;
                                    }
                                    Err(e) => {
                                        tracing::warn!("Agent report reply error: {}, retrying", e);
                                        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                                    }
                                }
                            }
                            if retry_success {
                                continue;
                            }
                        }

                        if total_active > 0 {
                            continue; // Keep the door open (Wait)
                        }
                    }

                    break;
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

    fn bg_ev_to_message(ev: BgEv) -> Option<Message> {
        match ev {
            BgEv::Thinking {
                task_id,
                tool_name,
                detail,
            } => Self::as_thinking(&task_id.0, &tool_name, detail),
            BgEv::Done(task_id, rep) => Some(
                Message::assistant()
                    .with_text(render_report_ui(&task_id.0, rep.trim_end()))
                    .with_generated_id()
                    .user_only(),
            ),
            BgEv::NoReport(task_id) => Some(
                Message::assistant()
                    .with_text(render_no_report_ui(&task_id.0))
                    .with_generated_id()
                    .user_only(),
            ),
            _ => None,
        }
    }

    fn as_thinking(subagent_id: &str, tool_name: &str, detail: String) -> Option<Message> {
        let detail = detail.trim().to_string();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::platform_extensions::better_summon::engine::{
        EngineCommand, SessionId, TaskId,
    };
    use crate::agents::AgentConfig;
    use crate::agents::GoosePlatform;
    use crate::config::permission::PermissionManager;
    use crate::config::GooseMode;
    use crate::conversation::message::Message;
    use crate::session::SessionManager;
    use futures::StreamExt;
    use rmcp::model::CallToolRequestParams;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::time::sleep;
    use tokio_util::sync::CancellationToken;

    #[tokio::test]
    async fn test_door_keeping_mechanism() {
        let session_manager = Arc::new(SessionManager::new(std::env::temp_dir()));
        let config = AgentConfig::new(
            session_manager,
            PermissionManager::instance(),
            None,
            GooseMode::Auto,
            false,
            GoosePlatform::GooseCli,
        );
        let agent = Agent::with_config(config);

        let scfg = SessionConfig {
            id: "test_door_keeping".to_string(),
            schedule_id: None,
            max_turns: None,
            retry_config: None,
        };

        let inner = async_stream::stream! {
            yield Ok(AgentEvent::Message(Message::user().with_text("ping").with_generated_id()));
            sleep(Duration::from_millis(100)).await;
        }
        .boxed();

        let token = tokio_util::sync::CancellationToken::new();
        let mut wrapped = BetterAgent::wrap(&agent, scfg.clone(), inner, Some(token.clone()));

        let engine = engine::get_engine_handle();
        let session_id = SessionId(Arc::from("test_door_keeping"));

        let engine_clone = engine.clone();
        let session_id_clone = session_id.clone();
        tokio::spawn(async move {
            sleep(Duration::from_millis(50)).await;

            let _ = engine_clone.send_cmd(EngineCommand::InjectEvent {
                session_id: session_id_clone.clone(),
                event: BgEv::Spawned(TaskId("test_task".into())),
            });

            sleep(Duration::from_millis(150)).await;

            let _ = engine_clone.send_cmd(EngineCommand::InjectEvent {
                session_id: session_id_clone,
                event: BgEv::Done(TaskId("test_task".into()), "Success".into()),
            });
        });

        let mut got_ping = false;
        let mut got_done = false;

        while let Some(event) = wrapped.next().await {
            if let Ok(AgentEvent::Message(m)) = event {
                if m.as_concat_text().contains("ping") {
                    got_ping = true;
                }
                if m.as_concat_text().contains("Success") {
                    got_done = true;
                    token.cancel();
                }
            }
        }

        assert!(got_ping, "Should process normal inner stream output.");
        assert!(
            got_done,
            "Should yield the Done report because the door was kept open."
        );
    }

    #[tokio::test]
    async fn test_breaks_on_tool_call_when_tasks_running() {
        let session_manager = Arc::new(SessionManager::new(std::env::temp_dir()));
        let config = AgentConfig::new(
            session_manager,
            PermissionManager::instance(),
            None,
            GooseMode::Auto,
            false,
            GoosePlatform::GooseCli,
        );
        let agent = Agent::with_config(config);

        let scfg = SessionConfig {
            id: "test_breaks_on_tool_call".to_string(),
            schedule_id: None,
            max_turns: None,
            retry_config: None,
        };

        let inner = async_stream::stream! {
            yield Ok(AgentEvent::Message(
                Message::assistant()
                    .with_tool_request("call_1", Ok(CallToolRequestParams::new("delegate")))
                    .with_generated_id(),
            ));
        }
        .boxed();

        let token = CancellationToken::new();
        let mut wrapped = BetterAgent::wrap(&agent, scfg.clone(), inner, Some(token.clone()));

        let engine = engine::get_engine_handle();
        let session_id = SessionId(Arc::from("test_breaks_on_tool_call"));

        let engine_clone = engine.clone();
        let session_id_clone = session_id.clone();
        tokio::spawn(async move {
            sleep(Duration::from_millis(50)).await;

            let _ = engine_clone.send_cmd(EngineCommand::InjectEvent {
                session_id: session_id_clone.clone(),
                event: BgEv::Spawned(TaskId("test_bg_task".into())),
            });
        });

        let mut got_tool_call = false;

        let result = tokio::time::timeout(Duration::from_secs(2), async {
            while let Some(event) = wrapped.next().await {
                if let Ok(AgentEvent::Message(m)) = event {
                    if m.has_tool_request() {
                        got_tool_call = true;
                    }
                }
            }
        })
        .await;

        assert!(
            result.is_ok(),
            "Stream hung waiting for tasks instead of breaking out for the tool call!"
        );
        assert!(got_tool_call, "Did not yield the tool call message.");
    }
}

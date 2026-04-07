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
    MSG_MISSING_REPORT_AGENT,
};
use super::retry;

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
        let mut last_tool_request_name: Option<String> = None;

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
                    _ = async {
                        if let Some(token) = retry_token.as_ref() {
                            token.cancelled().await;
                        } else {
                            futures::future::pending::<()>().await;
                        }
                    } => {
                        break;
                    }
                    maybe_item = async {
                        if let Some(c) = current.as_mut() {
                            c.next().await
                        } else {
                            futures::future::pending().await
                        }
                    } => {
                        match maybe_item {
                            Some(Ok(event)) => {
                                if let AgentEvent::Message(msg) = &event {
                                    if let Some((_, call)) = msg.first_tool_request() {
                                        last_tool_request_name = Some(call.name.to_string());
                                    } else {
                                        last_tool_request_name = None;
                                    }
                                }
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
                                if let Some(next_stream) = retry::retry_until_cancelled(
                                    || ag.reply(retry_msg.clone(), recovery_config.clone(), retry_token.clone()),
                                    |duration| tokio::time::sleep(duration),
                                    |e| tracing::warn!(%e, "Reply recovery failed, retrying in 2s"),
                                    retry_token.as_ref(),
                                )
                                .await
                                {
                                    current = Some(next_stream);
                                }
                            }
                            None => {
                                current = None;
                            }
                        }
                    }
                    bg_event = rx.recv(), if !rx.is_closed() => match bg_event {
                        Some(ev) => {
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

                    let (tids, reps, idle_count, running_tasks) = engine.take_reports(sess_id.clone()).await;
                    let status = engine.query_status(sess_id.clone()).await;
                    let total_active = running_tasks + status.pending_tasks;

                    if total_active > 0 {
                        continue;
                    }

                    if !is_sub {
                        if last_tool_request_name.as_deref() == Some("submit_task_report") {
                            break;
                        }

                        let prompt = if !reps.is_empty() {
                            let tid_strs: Vec<String> = tids.into_iter().map(|t| t.0).collect();
                            render_report_prompt(&tid_strs, idle_count, &reps)
                        } else {
                            MSG_MISSING_REPORT_AGENT.to_string()
                        };

                        let tm = Message::user()
                            .with_text(prompt)
                            .with_generated_id()
                            .agent_only();

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
    use crate::model::ModelConfig;
    use crate::providers::base::{stream_from_single_message, Provider, ProviderUsage, MessageStream, Usage};
    use crate::providers::errors::ProviderError;
    use crate::session::SessionManager;
    use futures::StreamExt;
    use rmcp::model::CallToolRequestParams;
    use std::{path::PathBuf, sync::Arc};
    use tokio_util::sync::CancellationToken;

    struct MockProvider;

    #[async_trait::async_trait]
    impl Provider for MockProvider {
        fn get_name(&self) -> &str {
            "mock-provider"
        }

        async fn stream(
            &self,
            _model_config: &ModelConfig,
            _session_id: &str,
            _system: &str,
            _messages: &[Message],
            _tools: &[rmcp::model::Tool],
        ) -> Result<MessageStream, ProviderError> {
            let message = Message::assistant()
                .with_text("OK")
                .with_generated_id();
            let usage = ProviderUsage::new("mock-model".into(), Usage::default());
            Ok(stream_from_single_message(message, usage))
        }

        fn get_model_config(&self) -> ModelConfig {
            ModelConfig::new("mock-model").unwrap()
        }
    }

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
        *agent.provider.lock().await = Some(Arc::new(MockProvider));

        let session = agent
            .config
            .session_manager
            .create_session(
                PathBuf::from(std::env::temp_dir()),
                "test_door_keeping".to_string(),
                SessionType::Hidden,
                GooseMode::Auto,
            )
            .await
            .expect("Failed to create session");

        let scfg = SessionConfig {
            id: session.id.clone(),
            schedule_id: None,
            max_turns: None,
            retry_config: None,
        };

        let inner = async_stream::stream! {
            yield Ok(AgentEvent::Message(Message::user().with_text("ping").with_generated_id()));
        }
        .boxed();

        let token = tokio_util::sync::CancellationToken::new();
        let mut wrapped = BetterAgent::wrap(&agent, scfg.clone(), inner, Some(token.clone()));

        let engine = engine::get_engine_handle();
        let session_id = SessionId(Arc::from(session.id.as_str()));

        let engine_clone = engine.clone();
        let session_id_clone = session_id.clone();

        let mut got_ping = false;
        let mut got_done = false;

        if let Some(Ok(AgentEvent::Message(m))) = wrapped.next().await {
            if m.as_concat_text().contains("ping") {
                got_ping = true;
            }
            let _ = engine_clone.send_cmd(EngineCommand::InjectEvent {
                session_id: session_id_clone.clone(),
                event: BgEv::Spawned(TaskId("test_task".into())),
            });

            let mut next_fut = wrapped.next();
            tokio::select! {
                maybe = &mut next_fut => {
                    panic!("Expected wrapped to wait for background tasks, got {:?}", maybe);
                }
                _ = tokio::task::yield_now() => {}
            }

            let _ = engine_clone.send_cmd(EngineCommand::InjectEvent {
                session_id: session_id_clone,
                event: BgEv::Done(TaskId("test_task".into()), "Success".into()),
            });

            while let Some(event) = wrapped.next().await {
                if let Ok(AgentEvent::Message(m)) = event {
                    if m.as_concat_text().contains("Success") {
                        got_done = true;
                        token.cancel();
                        break;
                    }
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
    async fn test_keeps_door_open_when_final_message_is_text_even_if_earlier_tool_call() {
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
        *agent.provider.lock().await = Some(Arc::new(MockProvider));

        let session = agent
            .config
            .session_manager
            .create_session(
                PathBuf::from(std::env::temp_dir()),
                "test_keep_open_after_tool_then_text".to_string(),
                SessionType::Hidden,
                GooseMode::Auto,
            )
            .await
            .expect("Failed to create session");

        let scfg = SessionConfig {
            id: session.id.clone(),
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
            yield Ok(AgentEvent::Message(Message::assistant().with_text("final").with_generated_id()));
        }
        .boxed();

        let token = tokio_util::sync::CancellationToken::new();
        let mut wrapped = BetterAgent::wrap(&agent, scfg.clone(), inner, Some(token.clone()));

        let engine = engine::get_engine_handle();
        let session_id = SessionId(Arc::from(session.id.as_str()));

        let engine_clone = engine.clone();
        let session_id_clone = session_id.clone();

        let mut got_final = false;
        let mut got_done = false;

        // consume two messages (tool call and final text)
        if let Some(Ok(AgentEvent::Message(m1))) = wrapped.next().await {
            if m1.has_tool_request() {
                // first message is tool request
            }
            if let Some(Ok(AgentEvent::Message(m2))) = wrapped.next().await {
                if m2.as_concat_text().contains("final") {
                    got_final = true;
                }

                let _ = engine_clone.send_cmd(EngineCommand::InjectEvent {
                    session_id: session_id_clone.clone(),
                    event: BgEv::Spawned(TaskId("test_task".into())),
                });

                let mut next_fut = wrapped.next();
                tokio::select! {
                    maybe = &mut next_fut => {
                        panic!("Expected wrapped to wait for background tasks, got {:?}", maybe);
                    }
                    _ = tokio::task::yield_now() => {}
                }

                let _ = engine_clone.send_cmd(EngineCommand::InjectEvent {
                    session_id: session_id_clone,
                    event: BgEv::Done(TaskId("test_task".into()), "Success".into()),
                });

                while let Some(event) = wrapped.next().await {
                    if let Ok(AgentEvent::Message(m)) = event {
                        if m.as_concat_text().contains("Success") {
                            got_done = true;
                            token.cancel();
                            break;
                        }
                    }
                }
            }
        }

        assert!(got_final, "Should process final text message.");
        assert!(got_done, "Should yield the Done report because the door was kept open.");
    }

    #[tokio::test]
    async fn test_prompts_for_report_when_final_message_is_tool_call() {
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
        *agent.provider.lock().await = Some(Arc::new(MockProvider));

        let session = agent
            .config
            .session_manager
            .create_session(
                PathBuf::from(std::env::temp_dir()),
                "test_prompt_for_report_on_final_tool_call".to_string(),
                SessionType::Hidden,
                GooseMode::Auto,
            )
            .await
            .expect("Failed to create session");

        let scfg = SessionConfig {
            id: session.id.clone(),
            schedule_id: None,
            max_turns: None,
            retry_config: None,
        };

        let inner = async_stream::stream! {
            yield Ok(AgentEvent::Message(
                Message::assistant()
                    .with_tool_request(
                        "submit_task_report",
                        Ok(CallToolRequestParams::new("submit_task_report").with_arguments(serde_json::json!({"task_report": "final report"}).as_object().unwrap().clone())),
                    )
                    .with_generated_id(),
            ));
        }
        .boxed();

        let token = CancellationToken::new();
        let mut wrapped = BetterAgent::wrap(&agent, scfg.clone(), inner, Some(token.clone()));

        let first = wrapped.next().await;
        assert!(matches!(
            first,
            Some(Ok(AgentEvent::Message(m))) if !m.has_tool_request() && m.as_concat_text().contains("final report")
        ));

        token.cancel();
        let _ = wrapped.next().await;
    }

    #[tokio::test]
    async fn test_keeps_door_open_when_tool_call_and_tasks_running() {
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
        *agent.provider.lock().await = Some(Arc::new(MockProvider));

        let session = agent
            .config
            .session_manager
            .create_session(
                PathBuf::from(std::env::temp_dir()),
                "test_keeps_door_open_on_tool_call".to_string(),
                SessionType::Hidden,
                GooseMode::Auto,
            )
            .await
            .expect("Failed to create session");

        let scfg = SessionConfig {
            id: session.id.clone(),
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
        let session_id = SessionId(Arc::from(session.id.as_str()));

        let engine_clone = engine.clone();
        let session_id_clone = session_id.clone();

        let mut got_tool_call = false;
        let mut got_done = false;

        if let Some(Ok(AgentEvent::Message(m))) = wrapped.next().await {
            if m.has_tool_request() {
                got_tool_call = true;
            }

            let _ = engine_clone.send_cmd(EngineCommand::InjectEvent {
                session_id: session_id_clone.clone(),
                event: BgEv::Spawned(TaskId("test_bg_task".into())),
            });

            let mut next_fut = wrapped.next();
            tokio::select! {
                maybe = &mut next_fut => {
                    panic!("Expected wrapped to wait while tasks are active, got {:?}", maybe);
                }
                _ = tokio::task::yield_now() => {}
            }

            let _ = engine_clone.send_cmd(EngineCommand::InjectEvent {
                session_id: session_id_clone,
                event: BgEv::Done(TaskId("test_bg_task".into()), "Success".into()),
            });

            while let Some(event) = wrapped.next().await {
                if let Ok(AgentEvent::Message(m)) = event {
                    if m.as_concat_text().contains("Success") {
                        got_done = true;
                        token.cancel();
                        break;
                    }
                }
            }
        }

        assert!(got_tool_call, "Did not yield the tool call message.");
        assert!(got_done, "Should keep the door open until the background task finished.");
    }

    #[tokio::test]
    async fn test_missed_events_are_recovered_after_tool_call_gap() {
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

        let session = agent
            .config
            .session_manager
            .create_session(
                PathBuf::from(std::env::temp_dir()),
                "test_missed_events_gap".to_string(),
                SessionType::Hidden,
                GooseMode::Auto,
            )
            .await
            .expect("Failed to create session");

        let scfg = SessionConfig {
            id: session.id.clone(),
            schedule_id: None,
            max_turns: None,
            retry_config: None,
        };

        let token1 = CancellationToken::new();
        let token2 = CancellationToken::new();
        let engine = engine::get_engine_handle();
        let session_id = SessionId(Arc::from(session.id.as_str()));

        struct MockProvider;

        #[async_trait::async_trait]
        impl Provider for MockProvider {
            fn get_name(&self) -> &str {
                "mock-provider"
            }

            async fn stream(
                &self,
                _model_config: &ModelConfig,
                _session_id: &str,
                _system: &str,
                _messages: &[Message],
                _tools: &[rmcp::model::Tool],
            ) -> Result<MessageStream, ProviderError> {
                let message = Message::assistant()
                    .with_text("Report recovered")
                    .with_generated_id();
                let usage = ProviderUsage::new("mock-model".into(), Usage::default());
                Ok(stream_from_single_message(message, usage))
            }

            fn get_model_config(&self) -> ModelConfig {
                ModelConfig::new("mock-model").unwrap()
            }
        }

        *agent.provider.lock().await = Some(Arc::new(MockProvider));

        let inner1 = async_stream::stream! {
            yield Ok(AgentEvent::Message(
                Message::assistant()
                    .with_tool_request("call_1", Ok(CallToolRequestParams::new("delegate")))
                    .with_generated_id(),
            ));
        }
        .boxed();

        let mut wrapped1 = BetterAgent::wrap(&agent, scfg.clone(), inner1, Some(token1.clone()));
        let first = wrapped1.next().await;
        assert!(matches!(
            first,
            Some(Ok(AgentEvent::Message(m))) if m.has_tool_request()
        ));
        token1.cancel();
        drop(wrapped1);

        tokio::task::yield_now().await;

        let _ = engine.send_cmd(EngineCommand::InjectEvent {
            session_id: session_id.clone(),
            event: BgEv::Spawned(TaskId("gap_task".into())),
        });
        let _ = engine.send_cmd(EngineCommand::InjectEvent {
            session_id: session_id.clone(),
            event: BgEv::Done(TaskId("gap_task".into()), "Gap Success".into()),
        });

        let inner2 = async_stream::stream! {
            yield Ok(AgentEvent::Message(Message::user().with_text("tool result").with_generated_id()));
        }
        .boxed();

        let mut wrapped2 = BetterAgent::wrap(&agent, scfg.clone(), inner2, Some(token2.clone()));

        let second = wrapped2.next().await;
        assert!(matches!(
            second,
            Some(Ok(AgentEvent::Message(m))) if m.has_tool_request() == false
        ));

        let third = wrapped2.next().await;
        assert!(matches!(
            third,
            Some(Ok(AgentEvent::Message(ref m))) if m.as_concat_text().contains("Gap Success")
        ));

        token2.cancel();

        assert!(
            third.is_some(),
            "The background event generated during the stream gap was LOST!"
        );
    }

    #[tokio::test]
    async fn test_report_reply_retries_indefinitely_until_cancelled() {
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
        *agent.provider.lock().await = Some(Arc::new(MockProvider));

        let session = agent
            .config
            .session_manager
            .create_session(
                PathBuf::from(std::env::temp_dir()),
                "test_report_reply_retries_forever".to_string(),
                SessionType::Hidden,
                GooseMode::Auto,
            )
            .await
            .expect("Failed to create session");

        let scfg = SessionConfig {
            id: session.id.clone(),
            schedule_id: None,
            max_turns: None,
            retry_config: None,
        };

        let token = CancellationToken::new();
        let engine = engine::get_engine_handle();
        let session_id = SessionId(Arc::from(session.id.as_str()));

        tokio::spawn(async move {
            let _ = engine.send_cmd(EngineCommand::InjectEvent {
                session_id: session_id.clone(),
                event: BgEv::Done(TaskId("bg_task".into()), "Success".into()),
            });
        });

        let inner = async_stream::stream! {
            yield Ok(AgentEvent::Message(
                Message::assistant().with_text("done").with_generated_id(),
            ));
        }
        .boxed();

        let mut wrapped = BetterAgent::wrap(&agent, scfg.clone(), inner, Some(token.clone()));

        // Consume the assistant response and the buffered report event.
        assert!(wrapped.next().await.is_some(), "Expected an initial assistant response");
        assert!(wrapped.next().await.is_some(), "Expected the buffered report event");

        let mut next_fut = wrapped.next();
        tokio::select! {
            maybe = &mut next_fut => {
                panic!("Stream ended before cancellation: {:#?}", maybe);
            }
            _ = tokio::task::yield_now() => {}
        }

        token.cancel();
        let _ = next_fut.await;
        while let Some(_) = wrapped.next().await {}
    }
}

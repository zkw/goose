use crate::agents::platform_extensions::better_summon::utils::MessageExt;
use crate::{
    agents::{agent::Agent, types::SessionConfig, AgentEvent},
    conversation::message::{Message, MessageContent, SystemNotificationType},
    session::session_manager::SessionType,
};
use anyhow::Result;
use futures::{pin_mut, stream::BoxStream, StreamExt};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::engine;
use super::formats::{
    build_thinking_message, render_no_report_ui, render_report_ui, THINKING_WORKING,
};
use super::stream_ext::{StreamRecoveryExt, StreamUiExt};
use super::worker::SUBAGENT_TOOL_REQ_TYPE;
use rmcp::model::ServerNotification;

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
        let (tx, rx) = mpsc::unbounded_channel();
        let rrx = engine::bind_session(Arc::clone(&id_arc), tx);

        let recovery_config = scfg.clone();
        let retry_msg = Message::user()
            .with_text("There was a server error. Please retry and continue your work.")
            .with_visibility(false, false)
            .with_generated_id();

        let pipeline = async_stream::stream! {
            if !rrx.await {
                let mut s = inner;
                while let Some(e) = s.next().await {
                    yield e;
                }
                return;
            }

            let _guard = Guard(Arc::clone(&id_arc));
            let id_str: &str = &id_arc;
            let is_sub = ag.config.session_manager.get_session(id_str, false)
                .await.is_ok_and(|s| s.session_type == SessionType::SubAgent);

            let retry_token = tk.clone();
            let recovered = inner.robust_retry(move || {
                let retry_msg = retry_msg.clone();
                let scfg = recovery_config.clone();
                let tk = retry_token.clone();
                async move { ag.reply(retry_msg, scfg, tk).await }
            });

            let mapped = recovered.map(move |result| {
                if let Ok(AgentEvent::Message(msg)) = result {
                    let msg = msg;
                    if let Some((_, call)) = msg.tool_request("submit_task_report") {
                        if !is_sub {
                            if let Some(text) = call
                                .arguments
                                .as_ref()
                                .and_then(|a| a.get("task_report"))
                                .and_then(|v| v.as_str())
                                .map(String::from)
                            {
                                if let Some(replaced) = msg.with_tool_request_replaced_by_text(
                                    "submit_task_report",
                                    text,
                                ) {
                                    return Ok(AgentEvent::Message(replaced));
                                }
                            }
                        }
                    }
                    Ok(AgentEvent::Message(msg))
                } else {
                    result
                }
            });

            let safe_stream = mapped.interleave_ui_safely(rx, Self::bg_ev_to_message);
            pin_mut!(safe_stream);
            while let Some(item) = safe_stream.next().await {
                yield item;
            }
        };

        Box::pin(pipeline)
    }

    fn bg_ev_to_message(ev: engine::BgEv) -> Option<Message> {
        match ev {
            engine::BgEv::Mcp(n) => Self::as_thinking(&n),
            engine::BgEv::Done(rep, aid, _) => Some(
                Message::assistant()
                    .with_text(render_report_ui(&aid, rep.trim_end()))
                    .with_generated_id()
                    .user_only(),
            ),
            engine::BgEv::NoReport(aid, _) => Some(
                Message::assistant()
                    .with_text(render_no_report_ui(&aid))
                    .with_generated_id()
                    .user_only(),
            ),
            _ => None,
        }
    }

    fn as_thinking(n: &ServerNotification) -> Option<Message> {
        let ServerNotification::LoggingMessageNotification(log) = n else {
            return None;
        };
        let data = &log.params.data;
        if data.get("type").and_then(|v| v.as_str()) != Some(SUBAGENT_TOOL_REQ_TYPE) {
            return None;
        }
        let sid = data.get("subagent_id")?.as_str()?;
        let call = data.get("tool_call")?;
        let cname = call.get("name")?.as_str()?;
        let args = call.get("arguments");
        let detail = ["command", "code", "path", "target_file"]
            .iter()
            .find_map(|k| args.and_then(|m| m.get(*k)).and_then(|v| v.as_str()))
            .map(|s| s.replace('\n', " ").trim().to_string())
            .unwrap_or_else(|| THINKING_WORKING.to_string());
        let name = cname.split("__").last().unwrap_or(cname);
        let sm = if detail.len() > 150 {
            format!("{}...", detail.chars().take(147).collect::<String>())
        } else {
            detail
        };

        Some(
            Message::assistant()
                .with_content(MessageContent::system_notification(
                    SystemNotificationType::ThinkingMessage,
                    build_thinking_message(sid, name, &sm),
                ))
                .with_generated_id()
                .with_visibility(false, false),
        )
    }
}

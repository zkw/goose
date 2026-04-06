use crate::{
    agents::{agent::Agent, types::SessionConfig, AgentEvent},
    conversation::message::{Message, MessageContent, SystemNotificationType},
    session::session_manager::SessionType,
};
use anyhow::Result;
use futures::{
    future::pending,
    stream::{BoxStream, StreamExt},
};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::engine::{self, BgEv};
use super::formats::{
    build_thinking_message, render_no_report_ui, render_report_prompt, render_report_ui,
    MSG_MISSING_REPORT_AGENT, MSG_MISSING_REPORT_USER, THINKING_WORKING,
};
use super::worker::SUBAGENT_TOOL_REQ_TYPE;
use crate::agents::platform_extensions::better_summon::utils::MessageExt;
use rmcp::model::ServerNotification;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamPhase {
    Idle,
    GeneratingText,
    TextCompleted,
    ExecutingTool,
    ToolCompleted,
}

struct StreamReducer {
    is_sub: bool,
    tasks: usize,
    has_rep: bool,
    rep_id: Option<String>,
    over: Option<Message>,
    ui: Vec<Message>, // UI 积压队列
    idle_count: usize,
    has_shown_reps: bool,
    last_msg_has_tool_call: bool,
    phase: StreamPhase,
    last_phase_update: Instant,
}

impl StreamReducer {
    fn new(is_sub: bool) -> Self {
        let now = Instant::now();
        Self {
            is_sub,
            tasks: 0,
            has_rep: false,
            rep_id: None,
            over: None,
            ui: Vec::new(),
            idle_count: 0,
            has_shown_reps: false,
            last_msg_has_tool_call: false,
            phase: StreamPhase::Idle,
            last_phase_update: now,
        }
    }

    fn add_ui(&mut self, text: String) {
        self.ui.push(
            Message::assistant()
                .with_text(text)
                .with_generated_id()
                .user_only(),
        );
    }

    // 后台事件不再直接返回 AgentEvent，而是放入积压队列
    fn handle_bg_event(&mut self, ev: BgEv) {
        match ev {
            BgEv::Spawned(_) => {
                self.tasks += 1;
            }
            BgEv::Mcp(n) => {
                if let Some(msg) = BetterAgent::as_thinking(&n) {
                    self.ui.push(msg);
                }
            }
            BgEv::Done(rep, aid, idle) => {
                self.tasks = self.tasks.saturating_sub(1);
                self.idle_count = idle;
                self.add_ui(render_report_ui(&aid, rep.trim_end()));
            }
            BgEv::NoReport(aid, idle) => {
                self.tasks = self.tasks.saturating_sub(1);
                self.idle_count = idle;
                self.add_ui(render_no_report_ui(&aid));
            }
        }
    }

    // 只负责提取并分发纯 UI 通知
    async fn drain_ui(&mut self, ag: &Agent, id_str: &str) -> Vec<AgentEvent> {
        let mut events = Vec::new();
        for m in self.ui.drain(..) {
            let _ = ag.config.session_manager.add_message(id_str, &m).await;
            events.push(AgentEvent::Message(m));
        }
        events
    }

    fn update_phase_based_on_message(&mut self, message: &Message) {
        if message.content.is_empty() {
            self.phase = StreamPhase::Idle;
            self.last_phase_update = Instant::now();
            return;
        }

        self.phase = match message.content.last() {
            Some(MessageContent::ToolRequest(_)) => StreamPhase::ExecutingTool,
            Some(MessageContent::ToolResponse(_)) => StreamPhase::ToolCompleted,
            Some(MessageContent::Text(_)) | Some(MessageContent::SystemNotification(_)) => {
                StreamPhase::GeneratingText
            }
            _ => StreamPhase::GeneratingText,
        };
        self.last_phase_update = Instant::now();
    }

    fn is_safe_to_push_ui(&self) -> bool {
        matches!(
            self.phase,
            StreamPhase::Idle | StreamPhase::TextCompleted | StreamPhase::ToolCompleted
        )
    }

    fn mark_text_completed_if_stalled(&mut self) {
        if self.phase == StreamPhase::GeneratingText
            && self.last_phase_update.elapsed() >= Duration::from_millis(500)
        {
            self.phase = StreamPhase::TextCompleted;
            self.last_phase_update = Instant::now();
        }
    }

    fn reset_after_reply(&mut self) {
        self.last_msg_has_tool_call = false;
    }

    fn reduce_stream_event(&mut self, event: AgentEvent) -> (Option<AgentEvent>, bool) {
        if let AgentEvent::Message(msg) = event {
            self.update_phase_based_on_message(&msg);
            self.last_msg_has_tool_call = msg.has_tool_request();

            if let Some((id, call)) = msg.tool_request("submit_task_report") {
                self.has_rep = true;
                self.rep_id = Some(id.to_string());
                let rep_text = call
                    .arguments
                    .as_ref()
                    .and_then(|a| a.get("task_report"))
                    .and_then(|v| v.as_str())
                    .map(String::from);

                if !self.is_sub {
                    if let Some(text) = rep_text.clone() {
                        if let Some(replaced) =
                            msg.with_tool_request_replaced_by_text("submit_task_report", text)
                        {
                            self.over = Some(replaced.clone());
                            return (Some(AgentEvent::Message(replaced)), false);
                        }
                    }
                }

                if let Some(rep_id) = self.rep_id.as_deref() {
                    if msg.tool_response(rep_id).is_some() {
                        return (None, true);
                    }
                }
            }

            return (Some(AgentEvent::Message(msg)), false);
        }

        (Some(event), false)
    }
}

async fn with_infinite_retry<F, Fut, T>(
    tk: Option<&CancellationToken>,
    context_name: &'static str,
    mut operation: F,
) -> anyhow::Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<T>>,
{
    let mut retry_count = 0;

    loop {
        let result = tokio::select! {
            _ = async {
                if let Some(token) = tk {
                    token.cancelled().await;
                } else {
                    pending::<()>().await;
                }
            } => return Err(anyhow::anyhow!("Cancelled during {}", context_name)),
            output = operation() => output,
        };

        match result {
            Ok(value) => return Ok(value),
            Err(error) => {
                retry_count += 1;
                tracing::warn!(%context_name, retry_count, error = %error, "retrying after failure");
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }
    }
}

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
        let (tx, mut rx) = mpsc::unbounded_channel();
        let rrx = engine::bind_session(Arc::clone(&id_arc), tx);

        Box::pin(async_stream::stream! {
            if !rrx.await {
                let mut s = inner;
                while let Some(e) = s.next().await { yield e; }
                return;
            }

            let _g = Guard(Arc::clone(&id_arc));
            let mut cur = Some(inner);
            let id_str: &str = &id_arc;
            let is_sub = ag.config.session_manager.get_session(id_str, false)
                .await.is_ok_and(|s| s.session_type == SessionType::SubAgent);

            let mut ctx = StreamReducer::new(is_sub);
            let mut retry_count = 0;
            let retry_msg = Message::user()
                .with_text("There was a server error. Please retry and continue your work.")
                .with_visibility(false, false)
                .with_generated_id();

            loop {
                let maybe_event = tokio::select! {
                    biased;
                    Some(ev) = rx.recv() => {
                        ctx.handle_bg_event(ev);
                        None
                    },
                    res = async { cur.as_mut().unwrap().next().await }, if cur.is_some() => {
                        match res {
                            Some(result) => Some(result),
                            None => { cur = None; None }
                        }
                    },
                    _ = tokio::time::sleep(Duration::from_millis(500)) => {
                        ctx.mark_text_completed_if_stalled();
                        None
                    }
                };

                // 处理主模型吐出的事件
                if let Some(res) = maybe_event {
                    match res {
                        Ok(ev) => {
                            let (maybe_event, fused) = ctx.reduce_stream_event(ev);
                            if fused {
                                cur = None;
                            }
                            if let Some(event) = maybe_event {
                                yield Ok(event);
                            }
                        }
                        Err(e) => {
                            retry_count += 1;
                            tracing::warn!("Agent stream error: {}, retrying {}", e, retry_count);

                            match with_infinite_retry(tk.as_ref(), "Agent Reply Recovery", || async {
                                ag.reply(retry_msg.clone(), scfg.clone(), tk.clone()).await
                            })
                            .await
                            {
                                Ok(s) => {
                                    ctx.reset_after_reply();
                                    cur = Some(s);
                                    continue;
                                }
                                Err(_) => {
                                    yield Err(e);
                                    break;
                                }
                            }
                        }
                    }
                }

                // 【核心机制】只有当前处于安全间隙，才把积压的 UI 通知刷出去
                if ctx.is_safe_to_push_ui() && !ctx.ui.is_empty() {
                    for ui_event in ctx.drain_ui(ag, &id_str).await {
                        yield Ok(ui_event);
                    }
                }

                if cur.is_none() {
                    if let Some(m) = ctx.over.take() {
                        let _ = ag.config.session_manager.add_message(&id_str, &m).await;
                    }

                    // 回合结束，强制清空所有的 UI 通知积压
                    for ui_event in ctx.drain_ui(ag, &id_str).await {
                        yield Ok(ui_event);
                    }

                    if ctx.last_msg_has_tool_call {
                        continue;
                    }

                    // 把真实的报告提取出来喂给 AI，并隐匿处理防止泄漏
                    let (reps, tids) = engine::take_reports(Arc::clone(&id_arc)).await;
                    if !tids.is_empty() && !ctx.is_sub {
                        ctx.has_shown_reps = true;
                        let prompt = render_report_prompt(&tids, ctx.idle_count, &reps);
                        let tm = Message::user().with_text(prompt).with_generated_id().agent_only();

                        match with_infinite_retry(tk.as_ref(), "Agent Report Reply", || async {
                            ag.reply(tm.clone(), scfg.clone(), tk.clone()).await
                        })
                        .await
                        {
                            Ok(s) => {
                                ctx.reset_after_reply();
                                cur = Some(s);
                                continue;
                            }
                            Err(_) => {
                                yield Err(anyhow::anyhow!("Cancelled while generating report"));
                                break;
                            }
                        }
                    }

                    if ctx.tasks == 0 {
                        if !ctx.has_rep && (ctx.is_sub || ctx.has_shown_reps) {
                             let lm = Message::assistant().with_text(MSG_MISSING_REPORT_USER).with_generated_id().agent_only();
                             let tm = Message::user().with_text(MSG_MISSING_REPORT_AGENT).with_generated_id().agent_only();
                             let _ = ag.config.session_manager.add_message(&id_str, &lm).await;
                             yield Ok(AgentEvent::Message(lm));

                             match with_infinite_retry(tk.as_ref(), "Agent Missing Report Reply", || async {
                                 ag.reply(tm.clone(), scfg.clone(), tk.clone()).await
                             })
                             .await
                             {
                                 Ok(s) => {
                                     ctx.reset_after_reply();
                                     cur = Some(s);
                                     continue;
                                 }
                                 Err(_) => {
                                     yield Err(anyhow::anyhow!("Cancelled while triggering missing report"));
                                     break;
                                 }
                             }
                        } else {
                             break;
                        }
                    }
                }
            }
        })
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

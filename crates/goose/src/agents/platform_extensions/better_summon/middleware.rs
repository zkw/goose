use crate::{
    agents::{agent::Agent, types::SessionConfig, AgentEvent},
    conversation::message::{Message, MessageContent, SystemNotificationType},
    session::session_manager::SessionType,
};
use anyhow::Result;
use futures::stream::{BoxStream, StreamExt};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::engine::{self, BgEv};
use super::formats::{
    build_thinking_message, render_no_report_ui, render_report_prompt, render_report_ui,
    MSG_MISSING_REPORT_AGENT, MSG_MISSING_REPORT_USER, THINKING_WORKING,
};
use super::worker::SUBAGENT_TOOL_REQ_TYPE;
use rmcp::model::ServerNotification;

struct Ctx {
    is_sub: bool,
    tasks: usize,
    has_rep: bool,
    rep_id: Option<String>,
    over: Option<Message>,
    ui: Vec<Message>,
    reps_pushed: usize,
    idle_count: usize,
    has_shown_reps: bool,
    wait_for_thinking_end: bool,
    prompted: bool,
    last_msg_has_tool_call: bool,
}

impl Ctx {
    fn new(is_sub: bool) -> Self {
        Self {
            is_sub,
            tasks: 0,
            has_rep: false,
            rep_id: None,
            over: None,
            ui: Vec::new(),
            reps_pushed: 0,
            idle_count: 0,
            has_shown_reps: false,
            wait_for_thinking_end: false,
            prompted: false,
            last_msg_has_tool_call: false,
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

    fn handle_bg_event(&mut self, ev: BgEv) -> Option<AgentEvent> {
        match ev {
            BgEv::Spawned(_) => {
                self.tasks += 1;
                None
            }
            BgEv::Mcp(n) => BetterAgent::as_thinking(&n).map(AgentEvent::Message),
            BgEv::Done(rep, aid, idle) => {
                self.tasks = self.tasks.saturating_sub(1);
                self.idle_count = idle;
                self.add_ui(render_report_ui(&aid, rep.trim_end()));
                None
            }
            BgEv::NoReport(aid, idle) => {
                self.tasks = self.tasks.saturating_sub(1);
                self.idle_count = idle;
                self.add_ui(render_no_report_ui(&aid));
                None
            }
        }
    }

    async fn drain_ui_and_reports(
        &mut self,
        ag: &Agent,
        id_str: &str,
        peek: bool,
    ) -> Vec<AgentEvent> {
        let mut events = Vec::new();
        let mut has_new_ui = false;

        for m in self.ui.drain(..) {
            has_new_ui = true;
            let _ = ag.config.session_manager.add_message(id_str, &m).await;
            events.push(AgentEvent::Message(m));
        }

        let (reps, tids) = if peek {
            engine::peek_reports(id_str).await
        } else {
            engine::take_reports(id_str).await
        };

        if !reps.is_empty() && (has_new_ui || reps.len() > self.reps_pushed || !peek) {
            let prompt = render_report_prompt(&tids, self.idle_count, &reps);
            let lm = Message::assistant()
                .with_text(prompt.clone())
                .with_generated_id()
                .user_only();
            let _ = ag.config.session_manager.add_message(id_str, &lm).await;
            events.push(AgentEvent::Message(lm));

            self.reps_pushed = if peek { reps.len() } else { 0 };
            self.has_shown_reps = true;
        }
        events
    }

    fn reduce_stream_event(&mut self, ev: &mut AgentEvent) -> (bool, bool) {
        let mut has_gap = false;
        let mut fused = false;
        let mut msg_is_thinking = false;

        if let AgentEvent::Message(msg) = ev {
            msg_is_thinking = msg.content.iter().any(|c| {
                matches!(c, MessageContent::SystemNotification(n)
                    if n.notification_type == SystemNotificationType::ThinkingMessage)
            });
            if msg_is_thinking {
                self.wait_for_thinking_end = true;
            }

            self.last_msg_has_tool_call = msg
                .content
                .iter()
                .any(|c| matches!(c, MessageContent::ToolRequest(_)));

            let mut rep_text = None;
            for c in &msg.content {
                if let MessageContent::ToolRequest(req) = c {
                    has_gap = true;
                    if req
                        .tool_call
                        .as_ref()
                        .is_ok_and(|t| t.name == "submit_task_report")
                    {
                        self.has_rep = true;
                        self.rep_id = Some(req.id.clone());
                        rep_text = req
                            .tool_call
                            .as_ref()
                            .unwrap()
                            .arguments
                            .as_ref()
                            .and_then(|a| a.get("task_report"))
                            .and_then(|v| v.as_str())
                            .map(String::from);
                    }
                } else if let MessageContent::ToolResponse(r) = c {
                    has_gap = true;
                    if self.rep_id.as_ref() == Some(&r.id) {
                        fused = true;
                    }
                }
            }

            if !self.is_sub && rep_text.is_some() {
                if let Some(c) = msg.content.iter_mut().find(|c| {
                    matches!(c, MessageContent::ToolRequest(req)
                        if req.tool_call.as_ref().is_ok_and(|t| t.name == "submit_task_report"))
                }) {
                    *c = MessageContent::text(rep_text.unwrap());
                    self.over = Some(msg.clone());
                }
            }
        }

        let thinking_ended = self.wait_for_thinking_end && !msg_is_thinking;
        if thinking_ended {
            self.wait_for_thinking_end = false;
        }

        (has_gap || thinking_ended, fused)
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
            if !rrx.await.unwrap_or(false) {
                let mut s = inner;
                while let Some(e) = s.next().await { yield e; }
                return;
            }

            let _g = Guard(Arc::clone(&id_arc));
            let mut cur = Some(inner);
            let id_str = id_arc.to_string();
            let is_sub = ag.config.session_manager.get_session(&id_str, false)
                .await.is_ok_and(|s| s.session_type == SessionType::SubAgent);

            let mut ctx = Ctx::new(is_sub);

            let mut retry_count = 0;
            const MAX_RETRIES: usize = 5;

            loop {
                let maybe_event = tokio::select! {
                    biased;
                    Some(ev) = rx.recv() => ctx.handle_bg_event(ev).map(Ok),
                    res = async { cur.as_mut().unwrap().next().await }, if cur.is_some() => {
                        match res {
                            Some(result) => Some(result),
                            None => { cur = None; None }
                        }
                    }
                };

                if let Some(res) = maybe_event {
                    match res {
                        Ok(mut ev) => {
                            let (trigger_ui, fused) = ctx.reduce_stream_event(&mut ev);

                            if fused {
                                cur = None;
                            } else {
                                yield Ok(ev);
                            }

                            if trigger_ui && !ctx.wait_for_thinking_end {
                                for ui_event in ctx.drain_ui_and_reports(ag, &id_str, true).await {
                                    yield Ok(ui_event);
                                }
                            }
                        }
                        Err(e) => {
                            if retry_count < MAX_RETRIES {
                                retry_count += 1;
                                tracing::warn!("Agent stream error: {}, retrying {}/{}", e, retry_count, MAX_RETRIES);

                                let retry_msg = Message::user()
                                    .with_text("There was a server error. Please retry and continue your work.")
                                    .with_visibility(false, false)
                                    .with_generated_id();

                                let mut retry_success = false;
                                while retry_count <= MAX_RETRIES {
                                    match ag.reply(retry_msg.clone(), scfg.clone(), tk.clone()).await {
                                        Ok(s) => {
                                            ctx.last_msg_has_tool_call = false;
                                            cur = Some(s);
                                            retry_success = true;
                                            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                                            break;
                                        },
                                        Err(e2) => {
                                            if retry_count < MAX_RETRIES {
                                                retry_count += 1;
                                                tracing::warn!("Agent reply error: {}, retrying {}/{}", e2, retry_count, MAX_RETRIES);
                                                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                                            } else {
                                                break;
                                            }
                                        }
                                    }
                                }

                                if retry_success {
                                    continue;
                                } else {
                                    yield Err(e);
                                    break;
                                }
                            } else {
                                yield Err(e);
                                break;
                            }
                        }
                    }
                }

                if cur.is_none() {
                    if let Some(m) = ctx.over.take() {
                        let _ = ag.config.session_manager.add_message(&id_str, &m).await;
                    }

                    for ui_event in ctx.drain_ui_and_reports(ag, &id_str, false).await {
                        yield Ok(ui_event);
                    }

                    if ctx.last_msg_has_tool_call { continue; }

                    let (_, tids) = engine::take_reports(&id_str).await;
                    if !tids.is_empty() && !ctx.is_sub {
                        let prompt = render_report_prompt(&tids, ctx.idle_count, &[]);
                        let tm = Message::user().with_text(prompt).with_generated_id().agent_only();

                        {
                            let mut retry_count = 0;
                            let mut retry_success = false;
                            while retry_count <= MAX_RETRIES {
                                match ag.reply(tm.clone(), scfg.clone(), tk.clone()).await {
                                    Ok(s) => {
                                        ctx.last_msg_has_tool_call = false;
                                        cur = Some(s);
                                        retry_success = true;
                                        break;
                                    },
                                    Err(e) => {
                                        if retry_count < MAX_RETRIES {
                                            retry_count += 1;
                                            tracing::warn!("Agent report reply error: {}, retrying {}/{}", e, retry_count, MAX_RETRIES);
                                            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                                        } else {
                                            break;
                                        }
                                    }
                                }
                            }
                            if retry_success {
                                continue;
                            } else {
                                yield Err(anyhow::anyhow!("Failed to generate report after retries"));
                                break;
                            }
                        }
                    }

                    if ctx.tasks == 0 {
                        if !ctx.has_rep && !ctx.prompted && (ctx.is_sub || ctx.has_shown_reps) {
                             ctx.prompted = true;
                             let lm = Message::assistant().with_text(MSG_MISSING_REPORT_USER).with_generated_id().agent_only();
                             let tm = Message::user().with_text(MSG_MISSING_REPORT_AGENT).with_generated_id().agent_only();
                             let _ = ag.config.session_manager.add_message(&id_str, &lm).await;
                             yield Ok(AgentEvent::Message(lm));

                             {
                                 let mut retry_count = 0;
                                 let mut retry_success = false;
                                 while retry_count <= MAX_RETRIES {
                                     match ag.reply(tm.clone(), scfg.clone(), tk.clone()).await {
                                         Ok(s) => {
                                             ctx.last_msg_has_tool_call = false;
                                             cur = Some(s);
                                             retry_success = true;
                                             break;
                                         },
                                         Err(e) => {
                                             if retry_count < MAX_RETRIES {
                                                 retry_count += 1;
                                                 tracing::warn!("Agent missing report reply error: {}, retrying {}/{}", e, retry_count, MAX_RETRIES);
                                                 tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                                             } else {
                                                 break;
                                             }
                                         }
                                     }
                                 }
                                 if !retry_success {
                                     yield Err(anyhow::anyhow!("Failed to trigger missing report after retries"));
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

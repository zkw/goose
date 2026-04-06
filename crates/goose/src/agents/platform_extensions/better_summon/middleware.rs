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
    fn reduce(&mut self, ev: &AgentEvent) {
        if let AgentEvent::Message(msg) = ev {
            self.last_msg_has_tool_call = msg
                .content
                .iter()
                .any(|c| matches!(c, MessageContent::ToolRequest(_)));
            for c in &msg.content {
                if let MessageContent::ToolRequest(req) = c {
                    if req
                        .tool_call
                        .as_ref()
                        .is_ok_and(|t| t.name == "submit_task_report")
                    {
                        self.has_rep = true;
                        self.rep_id = Some(req.id.clone());
                    }
                }
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
    pub fn wrap<'a>(
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
                while let Some(e) = s.next().await {
                    yield e;
                }
                return;
            }
            let _g = Guard(Arc::clone(&id_arc));
            let mut cur = Some(inner);
            let id_str = id_arc.to_string();
            let is_sub = ag
                .config
                .session_manager
                .get_session(&id_str, false)
                .await
                .is_ok_and(|s| s.session_type == SessionType::SubAgent);
            let mut ctx = Ctx {
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
            };

            loop {
                let maybe_event = tokio::select! {
                    biased;
                    Some(ev) = rx.recv() => {
                        match ev {
                            BgEv::Spawned(_sid) => {
                                ctx.tasks += 1;
                                None
                            }
                            BgEv::Mcp(n) => {
                                if let Some(msg) = Self::as_thinking(&n) {
                                    yield Ok(AgentEvent::Message(msg));
                                }
                                None
                            }
                            BgEv::Done(rep, aid, idle) => {
                                ctx.tasks = ctx.tasks.saturating_sub(1);
                                ctx.idle_count = idle;
                                let q = rep.trim_end();
                                let tm = Message::assistant()
                                    .with_text(render_report_ui(&aid, q))
                                    .with_generated_id()
                                    .user_only();
                                ctx.ui.push(tm);
                                None
                            }
                            BgEv::NoReport(aid, idle) => {
                                ctx.tasks = ctx.tasks.saturating_sub(1);
                                ctx.idle_count = idle;
                                let tm = Message::assistant()
                                    .with_text(render_no_report_ui(&aid))
                                    .with_generated_id()
                                    .user_only();
                                ctx.ui.push(tm);
                                None
                            }
                        }
                    }
                    res = async { cur.as_mut().unwrap().next().await }, if cur.is_some() => {
                        match res {
                            Some(result) => Some(result),
                            None => {
                                cur = None;
                                None
                            }
                        }
                    }
                };

                if let Some(res) = maybe_event {
                    match res {
                        Ok(mut ev) => {
                            ctx.reduce(&ev);
                            let mut fused = false;
                            let mut has_gap = false;
                            let mut msg_is_thinking = false;

                            if let AgentEvent::Message(msg) = &mut ev {
                                msg_is_thinking = msg
                                    .content
                                    .iter()
                                    .any(|c| matches!(c, MessageContent::SystemNotification(n) if n.notification_type == SystemNotificationType::ThinkingMessage));
                                if msg_is_thinking {
                                    ctx.wait_for_thinking_end = true;
                                }

                                let mut rep_text = None;
                                for c in &msg.content {
                                    if let MessageContent::ToolRequest(req) = c {
                                        has_gap = true;
                                        if req
                                            .tool_call
                                            .as_ref()
                                            .is_ok_and(|t| t.name == "submit_task_report")
                                        {
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
                                        if ctx.rep_id.as_ref() == Some(&r.id) {
                                            fused = true;
                                        }
                                    }
                                }
                                if !ctx.is_sub {
                                    if let Some(t) = rep_text {
                                        if let Some(c) = msg.content.iter_mut().find(|c| {
                                            matches!(c, MessageContent::ToolRequest(req) if
                                                req.tool_call.as_ref().is_ok_and(|t| t.name == "submit_task_report"))
                                        }) {
                                            *c = MessageContent::text(t.clone());
                                            ctx.over = Some(msg.clone());
                                        }
                                    }
                                }
                            }

                            let should_drain_ui = has_gap && !ctx.wait_for_thinking_end;
                            let thinking_ended = ctx.wait_for_thinking_end && !msg_is_thinking;
                            if thinking_ended {
                                ctx.wait_for_thinking_end = false;
                                let mut has_ui = false;
                                for m in ctx.ui.drain(..) {
                                    has_ui = true;
                                    let _ = ag.config.session_manager.add_message(&id_str, &m).await;
                                    yield Ok(AgentEvent::Message(m));
                                }
                                let (reps, tids) = engine::peek_reports(&id_str).await;
                                if has_ui || reps.len() > ctx.reps_pushed {
                                    let prompt = render_report_prompt(&tids, ctx.idle_count, &reps);
                                    let lm = Message::assistant()
                                        .with_text(prompt)
                                        .with_generated_id()
                                        .user_only();
                                    let _ = ag.config.session_manager.add_message(&id_str, &lm).await;
                                    yield Ok(AgentEvent::Message(lm));
                                    ctx.reps_pushed = reps.len();
                                    ctx.has_shown_reps = true;
                                }
                            }

                            if fused {
                                cur = None;
                            } else {
                                yield Ok(ev);
                            }

                            if should_drain_ui {
                                let mut has_ui = false;
                                for m in ctx.ui.drain(..) {
                                    has_ui = true;
                                    let _ = ag.config.session_manager.add_message(&id_str, &m).await;
                                    yield Ok(AgentEvent::Message(m));
                                }
                                let (reps, tids) = engine::peek_reports(&id_str).await;
                                if has_ui || reps.len() > ctx.reps_pushed {
                                    let prompt = render_report_prompt(&tids, ctx.idle_count, &reps);
                                    let lm = Message::assistant()
                                        .with_text(prompt)
                                        .with_generated_id()
                                        .user_only();
                                    let _ = ag.config.session_manager.add_message(&id_str, &lm).await;
                                    yield Ok(AgentEvent::Message(lm));
                                    ctx.reps_pushed = reps.len();
                                    ctx.has_shown_reps = true;
                                }
                            }
                        }
                        Err(e) => {
                            yield Err(e);
                            break;
                        }
                    }
                }

                if cur.is_none() {
                    if let Some(m) = ctx.over.take() {
                        let _ = ag.config.session_manager.add_message(&id_str, &m).await;
                    }
                    for m in ctx.ui.drain(..) {
                        let _ = ag.config.session_manager.add_message(&id_str, &m).await;
                        yield Ok(AgentEvent::Message(m));
                    }
                    if ctx.last_msg_has_tool_call {
                        continue;
                    }
                    let (reps, tids) = engine::take_reports(&id_str).await;
                    if !reps.is_empty() {
                        let prompt = render_report_prompt(&tids, ctx.idle_count, &reps);
                        ctx.reps_pushed = 0; // Reset as we've consumed them
                        if !ctx.is_sub {
                            // Dual-track trigger: UI sees Assistant, LLM sees User.
                            let lm = Message::assistant()
                                .with_text(prompt.clone())
                                .with_generated_id()
                                .user_only();
                            let tm = Message::user()
                                .with_text(prompt)
                                .with_generated_id()
                                .agent_only();

                            let _ = ag.config.session_manager.add_message(&id_str, &lm).await;
                            yield Ok(AgentEvent::Message(lm));
                            ctx.has_shown_reps = true;

                            match ag.reply(tm, scfg.clone(), tk.clone()).await {
                                Ok(s) => {
                                    ctx.last_msg_has_tool_call = false;
                                    cur = Some(s);
                                    continue;
                                },
                                Err(e) => {
                                    yield Err(e);
                                    break;
                                }
                            }
                        }
                    }
                    if ctx.tasks == 0 {
                        // All background tasks finished.
                        if !ctx.has_rep && !ctx.prompted && (ctx.is_sub || ctx.has_shown_reps) {
                             // Must report before finishing turn.
                             ctx.prompted = true;
                             let lm = Message::assistant()
                                 .with_text(MSG_MISSING_REPORT_USER)
                                 .with_generated_id()
                                 .agent_only();
                             let tm = Message::user()
                                 .with_text(MSG_MISSING_REPORT_AGENT)
                                 .with_generated_id()
                                 .agent_only();
                             let _ = ag.config.session_manager.add_message(&id_str, &lm).await;
                             yield Ok(AgentEvent::Message(lm));
                             match ag.reply(tm, scfg.clone(), tk.clone()).await {
                                 Ok(s) => {
                                     ctx.last_msg_has_tool_call = false;
                                     cur = Some(s);
                                 },
                                 Err(e) => {
                                     yield Err(e);
                                     break;
                                 }
                             }
                        } else {
                             // Parent or reported subagent: done.
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

use crate::{
    agents::{agent::Agent, types::SessionConfig, AgentEvent},
    conversation::message::{Message, MessageContent, SystemNotificationType},
    session::session_manager::SessionType,
};
use anyhow::Result;
use futures::stream::{BoxStream, StreamExt};
use std::{collections::HashSet, sync::Arc};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::{actor, BgEv, REPORT_UI, REPORT_PROMPT};
use rmcp::model::ServerNotification;

#[derive(PartialEq, Eq)]
enum Phase { Normal, Action, Review, Missing }

struct Ctx {
    phase: Phase,
    has_rep: bool,
    is_sub: bool,
    tasks: usize,
    task_ids: Vec<String>,
    idle_count: usize,
    reports: Vec<String>,
    ui: Vec<Message>,
    prompted: bool,
    rep_id: Option<String>,
    seen: HashSet<String>,
    over: Option<Message>,
}

impl Ctx {
    fn reduce(&mut self, ev: &AgentEvent) {
        if let AgentEvent::Message(msg) = ev {
            if !msg.is_tool_call() { return; }
            self.phase = Phase::Action;
            for c in &msg.content {
                if let MessageContent::ToolRequest(req) = c {
                    if req.tool_call.as_ref().is_ok_and(|t| t.name == "submit_task_report") {
                        self.has_rep = true;
                        self.phase = Phase::Normal;
                        self.rep_id = Some(req.id.clone());
                    }
                    if req.tool_call.as_ref().is_ok_and(|t| t.name == "delegate") && self.seen.insert(req.id.clone()) {
                        self.tasks += 1;
                    }
                }
            }
        }
    }
}

struct Guard(Arc<str>);
impl Drop for Guard { fn drop(&mut self) { actor::unbind_session(Arc::clone(&self.0)); } }

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
        let rrx = actor::bind_session(Arc::clone(&id_arc), tx);

        Box::pin(async_stream::stream! {
            if !rrx.await.unwrap_or(false) {
                let mut s = inner;
                while let Some(e) = s.next().await { yield e; }
                return;
            }
            let _g = Guard(Arc::clone(&id_arc));
            let mut cur = Some(inner);
            let id_str = id_arc.to_string();
            let is_sub = ag.config.session_manager.get_session(&id_str, false).await
                .is_ok_and(|s| s.session_type == SessionType::SubAgent);
            let mut ctx = Ctx {
                phase: Phase::Normal, has_rep: false, is_sub, tasks: 0,
                task_ids: Vec::new(), idle_count: 0, reports: Vec::new(),
                ui: Vec::new(), prompted: false, rep_id: None, seen: HashSet::new(), over: None,
            };

            loop {
                tokio::select! {
                    biased;
                    Some(ev) = rx.recv() => {
                        if matches!(ev, BgEv::Done(..)) && ctx.phase != Phase::Action { ctx.phase = Phase::Review; }
                        match ev {
                            BgEv::Mcp(n) => {
                                if let Some(msg) = Self::as_thinking(&n) {
                                    if cur.is_some() { ctx.ui.push(msg); } else { yield Ok(AgentEvent::Message(msg)); }
                                }
                            }
                            BgEv::Done(rep, aid, idle) => {
                                ctx.tasks = ctx.tasks.saturating_sub(1);
                                let q = rep.lines().fold(String::new(), |mut acc, l| {
                                    use std::fmt::Write; let _ = writeln!(acc, "> {}", l.trim_end()); acc
                                });
                                let q = q.trim_end();
                                ctx.ui.push(Message::assistant().with_text(REPORT_UI.replace("{TASK_ID}", &aid).replace("{RESULT}", q)).with_generated_id().user_only());
                                ctx.task_ids.push(aid);
                                ctx.idle_count = idle;
                                ctx.reports.push(q.to_string());
                            }
                        }
                    }
                    res = async { cur.as_mut().unwrap().next().await }, if cur.is_some() => match res {
                        Some(Ok(mut ev)) => {
                            ctx.reduce(&ev);
                            let mut fused = false;
                            if let AgentEvent::Message(msg) = &mut ev {
                                let mut rep_text = None;
                                for c in &msg.content {
                                    if let MessageContent::ToolRequest(req) = c {
                                        if req.tool_call.as_ref().is_ok_and(|t| t.name == "submit_task_report") {
                                            rep_text = req.tool_call.as_ref().unwrap().arguments.as_ref()
                                                .and_then(|a| a.get("task_report")).and_then(|v| v.as_str()).map(String::from);
                                        }
                                    } else if let MessageContent::ToolResponse(r) = c {
                                        if ctx.rep_id.as_ref() == Some(&r.id) { fused = true; }
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
                            if fused { cur = None; } else { yield Ok(ev); }
                        }
                        Some(Err(e)) => { yield Err(e); break; }
                        None => cur = None,
                    }
                }

                if cur.is_none() {
                    if let Some(m) = ctx.over.take() { let _ = ag.config.session_manager.add_message(&id_str, &m).await; }
                    for m in ctx.ui.drain(..) { let _ = ag.config.session_manager.add_message(&id_str, &m).await; yield Ok(AgentEvent::Message(m)); }
                    if !ctx.reports.is_empty() {
                        let prompt = REPORT_PROMPT
                            .replace("{TASK_ID}", &ctx.task_ids.join(", "))
                            .replace("{IDLE}", &ctx.idle_count.to_string())
                            .replace("{RESULT}", &ctx.reports.iter().map(|r| format!("> {}", r.replace('\n', "\n> "))).collect::<Vec<_>>().join("\n\n"));
                        let tm = Message::user().with_text(prompt).with_generated_id().agent_only();
                        ctx.task_ids.clear(); ctx.reports.clear();
                        match ag.reply(tm, scfg.clone(), tk.clone()).await {
                            Ok(s) => { cur = Some(s); ctx.phase = Phase::Normal; }
                            Err(e) => { yield Err(e); break; }
                        }
                    } else if ctx.tasks == 0 {
                        if ctx.is_sub && !ctx.has_rep && !ctx.prompted {
                            if ctx.phase == Phase::Missing { break; }
                            ctx.prompted = true; ctx.phase = Phase::Missing;
                            let lm = Message::assistant().with_text("System: A final summary report is required before ending the session. Waiting for `submit_task_report`...").with_generated_id().user_only();
                            let tm = Message::user().with_text("System: A final summary report is required before ending this session. Please call `submit_task_report` now.").with_generated_id().agent_only();
                            let _ = ag.config.session_manager.add_message(&id_str, &lm).await;
                            yield Ok(AgentEvent::Message(lm)); yield Ok(AgentEvent::Message(tm.clone()));
                            match ag.reply(tm, scfg.clone(), tk.clone()).await {
                                Ok(s) => { cur = Some(s); ctx.phase = Phase::Normal; }
                                Err(e) => { yield Err(e); break; }
                            }
                        } else if !ctx.is_sub && ctx.phase == Phase::Review {
                            let tm = Message::user().with_text("System: Status update received. Proceed.").with_generated_id().agent_only();
                            yield Ok(AgentEvent::Message(tm.clone()));
                            match ag.reply(tm, scfg.clone(), tk.clone()).await {
                                Ok(s) => { cur = Some(s); ctx.phase = Phase::Normal; }
                                Err(e) => { yield Err(e); break; }
                            }
                        } else { break; }
                    }
                }
            }
        })
    }

    fn as_thinking(n: &ServerNotification) -> Option<Message> {
        let ServerNotification::LoggingMessageNotification(log) = n else { return None };
        let data = &log.params.data;
        if data.get("type").and_then(|v| v.as_str()) != Some(super::subagent::SUBAGENT_TOOL_REQ_TYPE) { return None; }
        let sid = data.get("subagent_id")?.as_str()?;
        let call = data.get("tool_call")?;
        let cname = call.get("name")?.as_str()?;
        let args = call.get("arguments");
        let detail = ["command", "code", "path", "target_file"].iter()
            .find_map(|k| args.and_then(|m| m.get(*k)).and_then(|v| v.as_str()))
            .map(|s| s.replace('\n', " ").trim().to_string())
            .unwrap_or_else(|| "working...".into());
        let name = cname.split("__").last().unwrap_or(cname);
        let sm = format!("{}: {}", name, if detail.len() > 150 { format!("{}...", detail.chars().take(147).collect::<String>()) } else { detail });
        Some(Message::assistant().with_content(MessageContent::system_notification(
            SystemNotificationType::ThinkingMessage, format!("工程师[{}] {}", sid, sm)
        )).with_generated_id().with_visibility(false, false))
    }
}

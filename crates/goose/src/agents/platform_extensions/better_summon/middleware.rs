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
    ui: Vec<Message>, // UI 积压队列
    idle_count: usize,
    has_shown_reps: bool,
    last_msg_has_tool_call: bool,
    is_safe_to_push_ui: bool, // 核心状态：跟踪主模型是否处于安全间隙
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
            idle_count: 0,
            has_shown_reps: false,
            last_msg_has_tool_call: false,
            is_safe_to_push_ui: true, // 初始状态安全
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

    fn reduce_stream_event(&mut self, ev: &mut AgentEvent) -> bool {
        let mut fused = false;

        if let AgentEvent::Message(msg) = ev {
            // 精准探测安全间隙：如果最后一个块是文字或思考特效，说明正在吐字（不安全）；如果是工具调用/响应，则是安全间隙。
            self.is_safe_to_push_ui = match msg.content.last() {
                Some(MessageContent::ToolRequest(_)) | Some(MessageContent::ToolResponse(_)) => {
                    true
                }
                Some(MessageContent::Text(_)) | Some(MessageContent::SystemNotification(_)) => {
                    false
                }
                Some(_) => false,
                None => true,
            };

            self.last_msg_has_tool_call = msg
                .content
                .iter()
                .any(|c| matches!(c, MessageContent::ToolRequest(_)));

            let mut rep_text = None;
            for c in &msg.content {
                if let MessageContent::ToolRequest(req) = c {
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
        fused
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
                    }
                };

                // 处理主模型吐出的事件
                if let Some(res) = maybe_event {
                    match res {
                        Ok(mut ev) => {
                            let fused = ctx.reduce_stream_event(&mut ev);
                            if fused {
                                cur = None;
                            } else {
                                yield Ok(ev);
                            }
                        }
                        Err(e) => {
                            retry_count += 1;
                            tracing::warn!("Agent stream error: {}, retrying {}", e, retry_count);

                            let retry_msg = Message::user()
                                .with_text("There was a server error. Please retry and continue your work.")
                                .with_visibility(false, false)
                                .with_generated_id();

                            let mut retry_success = false;
                            loop {
                                if tk.as_ref().is_some_and(|t| t.is_cancelled()) {
                                    break;
                                }

                                match ag.reply(retry_msg.clone(), scfg.clone(), tk.clone()).await {
                                    Ok(s) => {
                                        ctx.last_msg_has_tool_call = false;
                                        cur = Some(s);
                                        retry_success = true;
                                        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                                        break;
                                    },
                                    Err(e2) => {
                                        retry_count += 1;
                                        tracing::warn!("Agent reply error: {}, retrying {}", e2, retry_count);
                                        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                                    }
                                }
                            }

                            if retry_success {
                                continue;
                            } else {
                                yield Err(e);
                                break;
                            }
                        }
                    }
                }

                // 【核心机制】只有当前处于安全间隙，才把积压的 UI 通知刷出去
                if ctx.is_safe_to_push_ui && !ctx.ui.is_empty() {
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

                    if ctx.last_msg_has_tool_call { continue; }

                    // 把真实的报告提取出来喂给 AI，并隐匿处理防止泄漏
                    let (reps, tids) = engine::take_reports(&id_str).await;
                    if !tids.is_empty() && !ctx.is_sub {
                        ctx.has_shown_reps = true;
                        let prompt = render_report_prompt(&tids, ctx.idle_count, &reps);
                        let tm = Message::user().with_text(prompt).with_generated_id().agent_only();

                        {
                            let mut retry_count = 0;
                            let mut retry_success = false;
                            loop {
                                if tk.as_ref().is_some_and(|t| t.is_cancelled()) {
                                    break;
                                }
                                match ag.reply(tm.clone(), scfg.clone(), tk.clone()).await {
                                    Ok(s) => {
                                        ctx.last_msg_has_tool_call = false;
                                        cur = Some(s);
                                        retry_success = true;
                                        break;
                                    },
                                    Err(e) => {
                                        retry_count += 1;
                                        tracing::warn!("Agent report reply error: {}, retrying {}", e, retry_count);
                                        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                                    }
                                }
                            }
                            if retry_success {
                                continue;
                            } else {
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

                             {
                                 let mut retry_count = 0;
                                 let mut retry_success = false;
                                 loop {
                                     if tk.as_ref().is_some_and(|t| t.is_cancelled()) {
                                         break;
                                     }
                                     match ag.reply(tm.clone(), scfg.clone(), tk.clone()).await {
                                         Ok(s) => {
                                             ctx.last_msg_has_tool_call = false;
                                             cur = Some(s);
                                             retry_success = true;
                                             break;
                                         },
                                         Err(e) => {
                                             retry_count += 1;
                                             tracing::warn!("Agent missing report reply error: {}, retrying {}", e, retry_count);
                                             tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                                         }
                                     }
                                 }
                                 if retry_success {
                                     continue;
                                 } else {
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

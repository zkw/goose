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

use super::actor::{self, BgEvent};
use rmcp::model::ServerNotification;

#[derive(PartialEq, Eq)]
pub enum AgentPhase {
    Normal,
    ActionExecuting, // 当前正在调用工具，处于原子执行期
    ReviewRequired,  // 收到后台报告，等待架构师审阅
    ReportMissing,   // 后台工程师结束会话前缺失 report
}

// 定义严格的状态上下文
struct StreamContext {
    phase: AgentPhase,
    has_submitted_report: bool,
    is_subagent: bool,
    pending_tasks: usize,              // 核心：记录仍在运行的后台工程师数量
    pending_prompts: String,           // LLM Prompt 暂存池（流耗尽后统一下发）
    pending_ui_messages: Vec<Message>, // UI 消息缓冲池（流耗尽后统一 yield）
}

impl StreamContext {
    // 纯状态转移函数
    fn reduce(&mut self, event: &AgentEvent) {
        if let AgentEvent::Message(msg) = event {
            if msg.is_tool_call() {
                self.phase = AgentPhase::ActionExecuting;

                // 状态机流转：识别工程师提交报告
                if msg.content.iter().any(|c| {
                    matches!(c, MessageContent::ToolRequest(req) if req.tool_call.as_ref().is_ok_and(|t| t.name == "submit_task_report"))
                }) {
                    self.has_submitted_report = true;
                    self.phase = AgentPhase::Normal;
                }

                // 状态机流转：识别架构师派发任务
                if msg.content.iter().any(|c| {
                    matches!(c, MessageContent::ToolRequest(req) if req.tool_call.as_ref().is_ok_and(|t| t.name == "delegate"))
                }) {
                    self.pending_tasks += 1;
                }
            }
        }
    }
}

struct SessionBindGuard(Arc<str>);

impl Drop for SessionBindGuard {
    fn drop(&mut self) {
        actor::unbind_session(Arc::clone(&self.0));
    }
}

pub struct BetterAgent;

impl BetterAgent {
    pub fn wrap<'a>(
        agent: &'a Agent,
        session_config: SessionConfig,
        inner_stream: BoxStream<'a, Result<AgentEvent>>,
        cancel_token: Option<CancellationToken>,
    ) -> BoxStream<'a, Result<AgentEvent>> {
        let (session_tx, mut session_rx) = mpsc::unbounded_channel::<BgEvent>();
        let session_id_arc: Arc<str> = Arc::from(session_config.id.as_str());
        actor::bind_session(Arc::clone(&session_id_arc), session_tx);
        let guard = SessionBindGuard(Arc::clone(&session_id_arc));

        Box::pin(async_stream::stream! {
            let _guard = guard;
            // 核心 1：将 Stream 包装入 Option，使其生命周期数据化
            let mut current_stream = Some(inner_stream);
            let session_id_str = session_id_arc.to_string();
            let mut ctx = StreamContext {
                phase: AgentPhase::Normal,
                has_submitted_report: false,
                is_subagent: agent.config.session_manager
                    .get_session(&session_id_str, false).await
                    .map(|s| s.session_type == SessionType::SubAgent).unwrap_or(false),
                pending_tasks: 0,
                pending_prompts: String::new(),
                pending_ui_messages: Vec::new(),
            };

            loop {
                tokio::select! {
                    biased;

                    // 分支 A：永远监听后台 Actor 事件
                    Some(bg_ev) = session_rx.recv() => {
                        // 处理后台任务完成事件，更新 phase
                        if matches!(bg_ev, BgEvent::TaskComplete(..)) && ctx.phase != AgentPhase::ActionExecuting {
                            ctx.phase = AgentPhase::ReviewRequired;
                        }

                        match bg_ev {
                            BgEvent::McpNotification(notif) => {
                                // 将后台 MCP 通知转译为 Thinking 消息，保持 UI 加载动画与终端日志跳动
                                if let Some(msg) = Self::as_thinking_message(&notif) {
                                    yield Ok(AgentEvent::Message(msg));
                                }
                            }
                            BgEvent::TaskComplete(report, agent_id, idle) => {
                                ctx.pending_tasks = ctx.pending_tasks.saturating_sub(1);
                                if ctx.phase != AgentPhase::ActionExecuting {
                                    ctx.phase = AgentPhase::ReviewRequired;
                                }

                                let mut quoted = String::with_capacity(report.len() + report.lines().count() * 3);
                                for line in report.lines() {
                                    use std::fmt::Write;
                                    if line.is_empty() {
                                        let _ = writeln!(quoted, ">");
                                    } else {
                                        let _ = writeln!(quoted, "> {}", line);
                                    }
                                }
                                let quoted = quoted.trim_end();

                                // 1. UI 消息入缓冲池（不打断正在生成的流）
                                let ui_text = format!("▶ **工程师 {}** 已完成任务\n\n{}", agent_id, quoted);
                                let log_msg = Message::assistant().with_text(ui_text).with_generated_id().user_only();
                                ctx.pending_ui_messages.push(log_msg);

                                // 2. Prompt 入暂存区（等流耗尽后统一唤醒 LLM）
                                let prompt_text = format!(
                                    "工程师：{}\n闲置数：{}\n报告内容：\n{}\n\n",
                                    agent_id, idle, quoted
                                );
                                ctx.pending_prompts.push_str(&prompt_text);
                            }
                        }
                    }

                    // 分支 B：核心 2 -> 仅当 current_stream 为 Some 时才参与 select 竞态
                    res = async { current_stream.as_mut().unwrap().next().await }, if current_stream.is_some() => {
                        match res {
                            Some(Ok(event)) => {
                                ctx.reduce(&event);
                                yield Ok(event);
                            }
                            Some(Err(e)) => {
                                yield Err(e);
                                break;
                            }
                            // 核心 3：流自然结束，安全卸载，下次 select 自动忽略此分支
                            None => {
                                current_stream = None;
                            }
                        }
                    }
                }

                // 核心控制流：仅在 LLM 本地输出耗尽时，安全下发所有副作用
                if current_stream.is_none() {
                    // 步骤一：先排空 UI 缓冲池（此时没有任何活跃流，前端能完美渲染多个独立气泡）
                    if !ctx.pending_ui_messages.is_empty() {
                        let messages: Vec<_> = ctx.pending_ui_messages.drain(..).collect();
                        for msg in messages {
                            let _ = agent.config.session_manager.add_message(&session_id_str, &msg).await;
                            yield Ok(AgentEvent::Message(msg));
                        }
                    }

                    // 步骤二：消费 Prompt 暂存区并唤醒 LLM
                    if !ctx.pending_prompts.is_empty() {
                        let trigger_text = format!(
                            "<system_instruction>\n以下后台任务已返回结果：\n\n{}请评估并决定下一步。\n</system_instruction>",
                            ctx.pending_prompts
                        );
                        ctx.pending_prompts.clear();

                        let trigger_msg = Message::user()
                            .with_text(trigger_text)
                            .with_generated_id()
                            .agent_only();
                        let _ = agent.config.session_manager.add_message(&session_id_str, &trigger_msg).await;

                        match agent.reply(trigger_msg, session_config.clone(), cancel_token.clone()).await {
                            Ok(new_stream) => {
                                current_stream = Some(new_stream);
                                ctx.phase = AgentPhase::Normal;
                            }
                            Err(e) => {
                                yield Err(e);
                                break;
                            }
                        }
                    } else if ctx.pending_tasks == 0 {
                        // 步骤三：兜底判定（既无报告也无任务时的退出/续写）
                        if let Some((log_msg, trigger_msg)) =
                            Self::determine_continuation(ctx.is_subagent, ctx.has_submitted_report, &mut ctx.phase)
                        {
                            if let Some(log) = log_msg {
                                let _ = agent.config.session_manager.add_message(&session_id_str, &log).await;
                                yield Ok(AgentEvent::Message(log));
                            }
                            let _ = agent.config.session_manager.add_message(&session_id_str, &trigger_msg).await;
                            yield Ok(AgentEvent::Message(trigger_msg.clone()));

                            match agent.reply(trigger_msg, session_config.clone(), cancel_token.clone()).await {
                                Ok(new_stream) => {
                                    current_stream = Some(new_stream);
                                    ctx.phase = AgentPhase::Normal;
                                }
                                Err(e) => {
                                    yield Err(e);
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

    fn determine_continuation(
        is_subagent: bool,
        has_submitted_report: bool,
        phase: &mut AgentPhase,
    ) -> Option<(Option<Message>, Message)> {
        // 规则1：后台工程师严禁未提交报告就退出
        if is_subagent && !has_submitted_report {
            let log = Message::assistant()
                .with_text("System: A final summary report is required before ending the session. Waiting for `submit_task_report`...")
                .with_generated_id()
                .user_only();
            let trigger = Message::user()
                .with_text("System: A final summary report is required before ending this session. Please call `submit_task_report` now.")
                .with_generated_id()
                .agent_only();
            *phase = AgentPhase::ReportMissing;
            return Some((Some(log), trigger));
        }

        // 规则2：架构师收到新报告
        if !is_subagent && *phase == AgentPhase::ReviewRequired {
            let update = Message::user()
                .with_text("System: Background status update received. Please analyze and proceed.")
                .with_generated_id()
                .agent_only();
            return Some((None, update));
        }

        None
    }

    pub(crate) fn as_thinking_message(notif: &ServerNotification) -> Option<Message> {
        let ServerNotification::LoggingMessageNotification(log) = notif else {
            return None;
        };
        let data = &log.params.data;

        if data.get("type").and_then(|v| v.as_str())
            != Some(super::subagent::SUBAGENT_TOOL_REQ_TYPE)
        {
            return None;
        }

        let short_id = data.get("subagent_id").and_then(|v| v.as_str())?;

        let call = data.get("tool_call")?;
        let call_name = call.get("name").and_then(|v| v.as_str())?;
        let args = call.get("arguments");

        let get_arg = |k: &str| args.and_then(|m| m.get(k)).and_then(|v| v.as_str());
        let detail = ["command", "code", "path", "target_file"]
            .iter()
            .find_map(|&k| get_arg(k))
            .map(|s| s.replace('\n', " ").trim().to_string())
            .unwrap_or_else(|| "working...".into());

        let short_name = call_name.split("__").last().unwrap_or(call_name);
        let mut summary = format!("{}: {}", short_name, detail);

        if summary.len() > 150 {
            summary.truncate(147);
            summary.push_str("...");
        }

        Some(
            Message::assistant()
                .with_content(MessageContent::system_notification(
                    SystemNotificationType::ThinkingMessage,
                    format!("工程师[{}] {}", short_id, summary),
                ))
                .with_generated_id()
                .with_visibility(false, false),
        )
    }
}

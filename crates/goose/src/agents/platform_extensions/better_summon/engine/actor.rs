use crate::agents::platform_extensions::better_summon::engine::events::BgEv;
use crate::agents::platform_extensions::better_summon::formats::{
    format_execution_error, ERROR_SCHEDULER_OFFLINE,
};
use crate::agents::platform_extensions::better_summon::worker::{
    run_subagent_task, SubagentRunParams,
};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, OnceLock};
use tokio::sync::{mpsc, oneshot};

const ENGINE_COMMAND_CAPACITY: usize = 128;

pub struct SessionStatus {
    pub idle_count: usize,
    pub reports: Vec<String>,
    pub task_ids: Vec<String>,
}

pub enum EngineCommand {
    SessionBind {
        session_id: Arc<str>,
        sender: mpsc::Sender<BgEv>,
        reply: oneshot::Sender<bool>,
    },
    SessionUnbind {
        session_id: Arc<str>,
    },
    QueryStatus {
        session_id: Arc<str>,
        reply: oneshot::Sender<SessionStatus>,
    },
    DispatchTask {
        params: SubagentRunParams,
    },
    WorkerProgress {
        session_id: Arc<str>,
        subagent_id: String,
        tool_name: String,
        tool_args: Option<rmcp::model::JsonObject>,
    },
    WorkerFinished {
        session_id: Arc<str>,
        task_id: String,
        report: Option<String>,
    },
}

#[derive(Clone)]
pub struct EngineHandle {
    tx: mpsc::Sender<EngineCommand>,
}

impl EngineHandle {
    pub fn spawn(limit: usize) -> Self {
        let (tx, mut rx) = mpsc::channel(ENGINE_COMMAND_CAPACITY);
        let actor_tx = tx.clone();

        tokio::spawn(async move {
            let mut state = ActorState::new(limit);
            while let Some(cmd) = rx.recv().await {
                match cmd {
                    EngineCommand::SessionBind {
                        session_id,
                        sender,
                        reply,
                    } => {
                        let session = state.sessions.entry(Arc::clone(&session_id)).or_default();
                        let is_new = session.downstream_tx.is_none();
                        session.downstream_tx = Some(sender);
                        let _ = reply.send(is_new);
                    }
                    EngineCommand::SessionUnbind { session_id } => {
                        if let Some(session) = state.sessions.get_mut(&session_id) {
                            session.downstream_tx = None;
                        }
                    }
                    EngineCommand::QueryStatus { session_id, reply } => {
                        let status = state
                            .sessions
                            .get(&session_id)
                            .map(|session| SessionStatus {
                                idle_count: state.available_permits,
                                reports: session.reports.clone(),
                                task_ids: session.completed_tasks.clone(),
                            })
                            .unwrap_or_else(|| SessionStatus {
                                idle_count: state.available_permits,
                                reports: Vec::new(),
                                task_ids: Vec::new(),
                            });
                        let _ = reply.send(status);
                    }
                    EngineCommand::DispatchTask { params } => {
                        if state.available_permits == 0 {
                            state.pending.push_back(params);
                        } else {
                            state.spawn_task(params, actor_tx.clone());
                        }
                    }
                    EngineCommand::WorkerProgress {
                        session_id,
                        subagent_id,
                        tool_name,
                        tool_args,
                    } => {
                        state.notify_downstream(
                            &session_id,
                            BgEv::ToolCall {
                                subagent_id,
                                tool_name,
                                tool_args,
                            },
                        );
                    }
                    EngineCommand::WorkerFinished {
                        session_id,
                        task_id,
                        report,
                    } => {
                        state.available_permits = state.available_permits.saturating_add(1);
                        if let Some(session) = state.sessions.get_mut(&session_id) {
                            session.running_tasks = session.running_tasks.saturating_sub(1);
                            let event = if let Some(report) = report.clone() {
                                session.reports.push(report.clone());
                                session.completed_tasks.push(task_id.clone());
                                BgEv::Done(report, task_id)
                            } else {
                                BgEv::NoReport(task_id)
                            };
                            state.notify_downstream(&session_id, event);
                        }
                        state.try_dispatch_pending(&actor_tx);
                    }
                }
            }
        });

        Self { tx }
    }

    pub fn try_send(
        &self,
        cmd: EngineCommand,
    ) -> Result<(), mpsc::error::TrySendError<EngineCommand>> {
        self.tx.try_send(cmd)
    }
}

#[derive(Default)]
struct SessionContext {
    downstream_tx: Option<mpsc::Sender<BgEv>>,
    reports: Vec<String>,
    completed_tasks: Vec<String>,
    running_tasks: usize,
}

struct ActorState {
    sessions: HashMap<Arc<str>, SessionContext>,
    available_permits: usize,
    pending: VecDeque<SubagentRunParams>,
}

impl ActorState {
    fn new(limit: usize) -> Self {
        Self {
            sessions: HashMap::new(),
            available_permits: limit,
            pending: VecDeque::new(),
        }
    }

    fn spawn_task(&mut self, params: SubagentRunParams, actor_tx: mpsc::Sender<EngineCommand>) {
        self.available_permits = self.available_permits.saturating_sub(1);
        let session_id = Arc::from(params.sess_id.as_str());
        let task_id = format!("ENGINEER-{}", params.sub_id);

        let session = self.sessions.entry(Arc::clone(&session_id)).or_default();
        session.running_tasks += 1;
        session.downstream_tx.as_ref().map(|tx| {
            let _ = tx.try_send(BgEv::Spawned(task_id.clone()));
        });

        let token = params.token.clone();
        tokio::spawn(async move {
            let report = tokio::select! {
                _ = async {
                    if let Some(t) = &token {
                        t.cancelled().await;
                    } else {
                        std::future::pending::<()>().await;
                    }
                } => None,
                r = run_subagent_task(params) => match r {
                    Ok(rep) if rep.is_empty() => None,
                    Ok(rep) => Some(rep),
                    Err(e) => Some(format_execution_error(&e.to_string())),
                },
            };

            let _ = actor_tx.try_send(EngineCommand::WorkerFinished {
                session_id,
                task_id,
                report,
            });
        });
    }

    fn try_dispatch_pending(&mut self, actor_tx: &mpsc::Sender<EngineCommand>) {
        while self.available_permits > 0 {
            if let Some(params) = self.pending.pop_front() {
                self.spawn_task(params, actor_tx.clone());
            } else {
                break;
            }
        }
    }

    fn notify_downstream(&self, session_id: &Arc<str>, event: BgEv) {
        if let Some(session) = self.sessions.get(session_id) {
            if let Some(tx) = &session.downstream_tx {
                let _ = tx.try_send(event);
            }
        }
    }
}

static GLOBAL_ENGINE_HANDLE: OnceLock<EngineHandle> = OnceLock::new();

fn engine_limit() -> usize {
    crate::config::Config::global()
        .get_param::<usize>("GOOSE_MAX_BACKGROUND_TASKS")
        .unwrap_or(50)
}

pub fn get_engine_handle() -> EngineHandle {
    GLOBAL_ENGINE_HANDLE
        .get_or_init(|| EngineHandle::spawn(engine_limit()))
        .clone()
}

pub async fn fetch_status(session_id: Arc<str>) -> SessionStatus {
    let (reply_tx, reply_rx) = oneshot::channel();
    let handle = get_engine_handle();
    let _ = handle
        .tx
        .send(EngineCommand::QueryStatus {
            session_id,
            reply: reply_tx,
        })
        .await;
    reply_rx.await.unwrap_or_else(|_| SessionStatus {
        idle_count: 0,
        reports: vec![],
        task_ids: vec![],
    })
}

pub fn dispatch_task(params: SubagentRunParams) -> Result<(), &'static str> {
    let handle = get_engine_handle();
    handle
        .try_send(EngineCommand::DispatchTask { params })
        .map_err(|_| ERROR_SCHEDULER_OFFLINE)
}

pub async fn bind_session(session_id: Arc<str>, sender: mpsc::Sender<BgEv>) -> bool {
    let (reply_tx, reply_rx) = oneshot::channel();
    let handle = get_engine_handle();
    let _ = handle
        .tx
        .send(EngineCommand::SessionBind {
            session_id,
            sender,
            reply: reply_tx,
        })
        .await;
    reply_rx.await.unwrap_or(false)
}

pub fn unbind_session(session_id: Arc<str>) {
    let handle = get_engine_handle();
    let _ = handle.try_send(EngineCommand::SessionUnbind { session_id });
}

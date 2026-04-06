use crate::agents::platform_extensions::better_summon::engine::events::BgEv;
use crate::agents::platform_extensions::better_summon::formats::{
    format_execution_error, ERROR_SCHEDULER_OFFLINE,
};
use crate::agents::platform_extensions::better_summon::worker::{run_subagent_task, SubagentRunParams};
use std::collections::{hash_map::Entry, HashMap, VecDeque};
use std::sync::{Arc, OnceLock};
use tokio::sync::{mpsc, oneshot};

const ENGINE_COMMAND_CAPACITY: usize = 128;

pub struct SessionStatus {
    pub idle_count: usize,
    pub reports: Vec<String>,
    pub task_ids: Vec<String>,
}

pub enum EngineCommand {
    BindSession {
        session_id: Arc<str>,
        sender: mpsc::Sender<BgEv>,
        reply: oneshot::Sender<bool>,
    },
    UnbindSession {
        session_id: Arc<str>,
    },
    RouteEvent {
        session_id: Arc<str>,
        event: BgEv,
    },
    FetchStatus {
        session_id: Arc<str>,
        reply: oneshot::Sender<SessionStatus>,
    },
    DispatchTask {
        params: SubagentRunParams,
    },
    TaskFinished {
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
                    EngineCommand::BindSession {
                        session_id,
                        sender,
                        reply,
                    } => {
                        let is_new = match state.sessions.entry(session_id) {
                            Entry::Vacant(v) => {
                                v.insert(sender);
                                true
                            }
                            Entry::Occupied(_) => false,
                        };
                        let _ = reply.send(is_new);
                    }
                    EngineCommand::UnbindSession { session_id } => {
                        state.sessions.remove(&session_id);
                    }
                    EngineCommand::RouteEvent { session_id, event } => {
                        if let BgEv::Done(report, task_id) = &event {
                            state
                                .reports
                                .entry(Arc::clone(&session_id))
                                .or_default()
                                .push(report.clone());
                            state
                                .task_ids
                                .entry(Arc::clone(&session_id))
                                .or_default()
                                .push(task_id.clone());
                        }

                        if let Some(tx) = state.sessions.get(&session_id) {
                            let _ = tx.try_send(event);
                        }
                    }
                    EngineCommand::FetchStatus { session_id, reply } => {
                        let reports = state.reports.remove(&session_id).unwrap_or_default();
                        let task_ids = state.task_ids.remove(&session_id).unwrap_or_default();
                        let _ = reply.send(SessionStatus {
                            idle_count: state.available_permits,
                            reports,
                            task_ids,
                        });
                    }
                    EngineCommand::DispatchTask { params } => {
                        state.spawn_task(params, actor_tx.clone());
                    }
                    EngineCommand::TaskFinished {
                        session_id,
                        task_id,
                        report,
                    } => {
                        state.available_permits = state.available_permits.saturating_add(1);

                        let event = if let Some(report) = report.clone() {
                            state
                                .reports
                                .entry(Arc::clone(&session_id))
                                .or_default()
                                .push(report.clone());
                            state
                                .task_ids
                                .entry(Arc::clone(&session_id))
                                .or_default()
                                .push(task_id.clone());
                            BgEv::Done(report, task_id)
                        } else {
                            BgEv::NoReport(task_id)
                        };

                        if let Some(tx) = state.sessions.get(&session_id) {
                            let _ = tx.try_send(event.clone());
                        }

                        state.try_dispatch_pending(&actor_tx);
                    }
                }
            }
        });

        Self { tx }
    }

    pub fn try_send(&self, cmd: EngineCommand) -> Result<(), mpsc::error::TrySendError<EngineCommand>> {
        self.tx.try_send(cmd)
    }
}

struct ActorState {
    sessions: HashMap<Arc<str>, mpsc::Sender<BgEv>>,
    reports: HashMap<Arc<str>, Vec<String>>,
    task_ids: HashMap<Arc<str>, Vec<String>>,
    available_permits: usize,
    pending: VecDeque<SubagentRunParams>,
}

impl ActorState {
    fn new(limit: usize) -> Self {
        Self {
            sessions: HashMap::new(),
            reports: HashMap::new(),
            task_ids: HashMap::new(),
            available_permits: limit,
            pending: VecDeque::new(),
        }
    }

    fn spawn_task(&mut self, mut params: SubagentRunParams, actor_tx: mpsc::Sender<EngineCommand>) {
        if self.available_permits == 0 {
            self.pending.push_back(params);
            return;
        }

        self.available_permits -= 1;
        params.event_tx = Some(actor_tx.clone());
        let session_id = Arc::from(params.sess_id.as_str());
        let task_id = format!("ENGINEER-{}", params.sub_id);
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

            let _ = actor_tx.try_send(EngineCommand::TaskFinished {
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
        .send(EngineCommand::FetchStatus {
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
        .send(EngineCommand::BindSession {
            session_id,
            sender,
            reply: reply_tx,
        })
        .await;
    reply_rx.await.unwrap_or(false)
}

pub fn unbind_session(session_id: Arc<str>) {
    let handle = get_engine_handle();
    let _ = handle.try_send(EngineCommand::UnbindSession { session_id });
}

pub fn route_event(session_id: Arc<str>, event: BgEv) {
    let handle = get_engine_handle();
    let _ = handle.try_send(EngineCommand::RouteEvent { session_id, event });
}

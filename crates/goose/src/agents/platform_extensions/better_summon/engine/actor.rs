use crate::agents::platform_extensions::better_summon::engine::events::{BgEv, SessionId, TaskId};
use crate::agents::platform_extensions::better_summon::formats::{
    format_execution_error, ERROR_SCHEDULER_OFFLINE,
};
use crate::agents::platform_extensions::better_summon::worker::{
    run_subagent_task, SubagentRunParams,
};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

const ENGINE_COMMAND_CAPACITY: usize = 128;

pub struct SessionStatus {
    pub idle_count: usize,
    pub reports: Vec<String>,
    pub task_ids: Vec<String>,
}

pub enum EngineCommand {
    Subscribe {
        session_id: SessionId,
        sender: mpsc::Sender<BgEv>,
    },
    Unsubscribe {
        session_id: SessionId,
    },
    QueryStatus {
        session_id: SessionId,
        reply: oneshot::Sender<SessionStatus>,
    },
    DispatchTask {
        params: SubagentRunParams,
    },
    WorkerProgress {
        session_id: SessionId,
        task_id: TaskId,
        tool_name: String,
        detail: String,
    },
    WorkerFinished {
        session_id: SessionId,
        task_id: TaskId,
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
                    EngineCommand::Subscribe { session_id, sender } => {
                        let session = state.sessions.entry(session_id.clone()).or_default();
                        session.downstream_tx = Some(sender);
                    }
                    EngineCommand::Unsubscribe { session_id } => {
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
                                task_ids: session
                                    .completed_tasks
                                    .iter()
                                    .map(|task_id| task_id.0.clone())
                                    .collect(),
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
                        task_id,
                        tool_name,
                        detail,
                    } => {
                        state.notify(
                            &session_id,
                            BgEv::Thinking {
                                task_id,
                                tool_name,
                                detail,
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
                                BgEv::Done(task_id, report)
                            } else {
                                BgEv::NoReport(task_id)
                            };
                            state.notify(&session_id, event);
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

    pub fn dispatch_task(&self, params: SubagentRunParams) -> Result<(), &'static str> {
        self.try_send(EngineCommand::DispatchTask { params })
            .map_err(|_| ERROR_SCHEDULER_OFFLINE)
    }

    pub async fn query_status(&self, session_id: SessionId) -> SessionStatus {
        let (reply_tx, reply_rx) = oneshot::channel();
        let _ = self
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

    pub async fn subscribe(
        self,
        session_id: SessionId,
        sender: mpsc::Sender<BgEv>,
    ) -> Result<(), mpsc::error::SendError<EngineCommand>> {
        self.tx
            .send(EngineCommand::Subscribe { session_id, sender })
            .await
    }

    pub fn unsubscribe(&self, session_id: SessionId) {
        let _ = self.try_send(EngineCommand::Unsubscribe { session_id });
    }
}

#[derive(Default)]
struct SessionContext {
    downstream_tx: Option<mpsc::Sender<BgEv>>,
    reports: Vec<String>,
    completed_tasks: Vec<TaskId>,
    running_tasks: usize,
}

struct ActorState {
    sessions: HashMap<SessionId, SessionContext>,
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
        let session_id = SessionId(Arc::from(params.sess_id.as_str()));
        let task_id = TaskId(format!("ENGINEER-{}", params.sub_id));

        let session = self.sessions.entry(session_id.clone()).or_default();
        session.running_tasks += 1;
        if let Some(tx) = session.downstream_tx.as_ref() {
            let _ = tx.try_send(BgEv::Spawned(task_id.clone()));
        }

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

    fn notify(&self, session_id: &SessionId, event: BgEv) {
        if let Some(session) = self.sessions.get(session_id) {
            if let Some(tx) = &session.downstream_tx {
                let _ = tx.try_send(event);
            }
        }
    }
}

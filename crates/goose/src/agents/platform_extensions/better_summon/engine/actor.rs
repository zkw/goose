use crate::agents::platform_extensions::better_summon::engine::events::BgEv;
use crate::agents::platform_extensions::better_summon::formats::{
    format_execution_error, ERROR_SCHEDULER_OFFLINE,
};
use crate::agents::platform_extensions::better_summon::worker::{
    run_subagent_task, SubagentRunParams,
};
use once_cell::sync::Lazy;
use std::collections::{hash_map::Entry, HashMap, VecDeque};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use tokio::sync::{mpsc, oneshot};

enum SupervisorCommand {
    BindSession {
        session_id: Arc<str>,
        sender: mpsc::UnboundedSender<BgEv>,
        reply: oneshot::Sender<bool>,
    },
    UnbindSession {
        session_id: Arc<str>,
    },
    RouteEvent {
        session_id: Arc<str>,
        event: BgEv,
    },
    TakeReports {
        session_id: Arc<str>,
        reply: oneshot::Sender<(Vec<String>, Vec<String>)>,
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

struct ActorState {
    sessions: HashMap<Arc<str>, mpsc::UnboundedSender<BgEv>>,
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

    fn spawn_task(
        &mut self,
        params: SubagentRunParams,
        actor_tx: mpsc::UnboundedSender<SupervisorCommand>,
    ) {
        if self.available_permits == 0 {
            self.pending.push_back(params);
            return;
        }

        self.available_permits -= 1;
        AVAILABLE_PERMITS.store(self.available_permits, Ordering::Relaxed);

        let actor_tx = actor_tx.clone();
        let session_id = Arc::clone(&params.p_sess_id);
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

            let _ = actor_tx.send(SupervisorCommand::TaskFinished {
                session_id,
                task_id,
                report,
            });
        });
    }

    fn try_dispatch_pending(&mut self, actor_tx: &mpsc::UnboundedSender<SupervisorCommand>) {
        while self.available_permits > 0 {
            if let Some(params) = self.pending.pop_front() {
                self.spawn_task(params, actor_tx.clone());
            } else {
                break;
            }
        }
    }
}

static AVAILABLE_PERMITS: Lazy<AtomicUsize> = Lazy::new(|| {
    let limit = crate::config::Config::global()
        .get_param::<usize>("GOOSE_MAX_BACKGROUND_TASKS")
        .unwrap_or(50);
    AtomicUsize::new(limit)
});

static ACTOR_SENDER: Lazy<mpsc::UnboundedSender<SupervisorCommand>> = Lazy::new(|| {
    let (tx, mut rx) = mpsc::unbounded_channel();
    let actor_tx = tx.clone();
    let limit = AVAILABLE_PERMITS.load(Ordering::Relaxed);

    tokio::spawn(async move {
        let mut state = ActorState::new(limit);

        while let Some(cmd) = rx.recv().await {
            match cmd {
                SupervisorCommand::BindSession {
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
                SupervisorCommand::UnbindSession { session_id } => {
                    state.sessions.remove(&session_id);
                }
                SupervisorCommand::RouteEvent { session_id, event } => {
                    if let BgEv::Done(report, task_id, _) = &event {
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
                        let _ = tx.send(event);
                    }
                }
                SupervisorCommand::TakeReports { session_id, reply } => {
                    let reports = state.reports.remove(&session_id).unwrap_or_default();
                    let task_ids = state.task_ids.remove(&session_id).unwrap_or_default();
                    let _ = reply.send((reports, task_ids));
                }
                SupervisorCommand::DispatchTask { params } => {
                    state.spawn_task(params, actor_tx.clone());
                }
                SupervisorCommand::TaskFinished {
                    session_id,
                    task_id,
                    report,
                } => {
                    state.available_permits = state.available_permits.saturating_add(1);
                    AVAILABLE_PERMITS.store(state.available_permits, Ordering::Relaxed);

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
                        BgEv::Done(report, task_id, state.available_permits)
                    } else {
                        BgEv::NoReport(task_id, state.available_permits)
                    };

                    if let Some(tx) = state.sessions.get(&session_id) {
                        let _ = tx.send(event);
                    }

                    state.try_dispatch_pending(&actor_tx);
                }
            }
        }
    });

    tx
});

pub fn idle_engineer_count() -> usize {
    AVAILABLE_PERMITS.load(Ordering::Relaxed)
}

pub fn dispatch_task(params: SubagentRunParams) -> Result<(), &'static str> {
    ACTOR_SENDER
        .send(SupervisorCommand::DispatchTask { params })
        .map_err(|_| ERROR_SCHEDULER_OFFLINE)
}

pub async fn bind_session(session_id: Arc<str>, sender: mpsc::UnboundedSender<BgEv>) -> bool {
    let (reply_tx, reply_rx) = oneshot::channel();
    let _ = ACTOR_SENDER.send(SupervisorCommand::BindSession {
        session_id,
        sender,
        reply: reply_tx,
    });
    reply_rx.await.unwrap_or(false)
}

pub fn unbind_session(session_id: Arc<str>) {
    let _ = ACTOR_SENDER.send(SupervisorCommand::UnbindSession { session_id });
}

pub fn route_event(session_id: Arc<str>, event: BgEv) {
    let _ = ACTOR_SENDER.send(SupervisorCommand::RouteEvent { session_id, event });
}

pub async fn take_reports(session_id: &str) -> (Vec<String>, Vec<String>) {
    let (reply_tx, reply_rx) = oneshot::channel();
    let _ = ACTOR_SENDER.send(SupervisorCommand::TakeReports {
        session_id: Arc::from(session_id),
        reply: reply_tx,
    });
    reply_rx.await.unwrap_or_default()
}

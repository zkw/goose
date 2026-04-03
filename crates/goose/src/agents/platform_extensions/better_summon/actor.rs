use crate::conversation::message::Message;
use dashmap::DashMap;
use once_cell::sync::Lazy;
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, watch};
use tracing::warn;

/// Centralized state for background task events and lifecycle monitoring.
/// This enables the AOP-style multiplexing by providing channels for subagents
/// to push notifications and reports to their parent session.
#[derive(Clone)]
pub enum BackgroundEvent {
    Message(Message),
    McpNotification(rmcp::model::ServerNotification),
    TaskComplete(String, String, usize), // (report, agent_id, idle_task_permits)
}

pub struct SessionState {
    /// Channel for background events to be consumed by the stream multiplexer
    pub tx: mpsc::Sender<BackgroundEvent>,
    pub rx: Mutex<Option<mpsc::Receiver<BackgroundEvent>>>,
    /// Watcher to track the number of active tasks in this session
    pub tasks_tx: watch::Sender<usize>,
    pub tasks_rx: watch::Receiver<usize>,
}

static SESSIONS: Lazy<DashMap<String, Arc<SessionState>>> = Lazy::new(DashMap::new);

/// Get the current number of active background tasks for a session
pub fn active_tasks(session_id: &str) -> usize {
    SESSIONS
        .get(session_id)
        .map(|s| *s.tasks_rx.borrow())
        .unwrap_or(0)
}

/// Atomically cleans up the session state if no tasks are active and the receiver has been returned.
fn try_cleanup_session(session_id: &str) {
    let state = SESSIONS.get(session_id).map(|r| r.value().clone());
    let Some(state) = state else {
        return;
    };
    let rx_idle = state.rx.lock().unwrap_or_else(|e| e.into_inner()).is_some();
    if !rx_idle {
        return;
    }
    // Only remove if task count remains zero to avoid racing with new TaskGuards
    SESSIONS.remove_if(session_id, |_, s| *s.tasks_rx.borrow() == 0);
}

/// A RAII guard that holds the receiver for a session's background event channel.
/// Ensures the receiver is safely returned to the central registry upon drop.
pub struct ReceiverGuard {
    session_id: String,
    pub rx: Option<mpsc::Receiver<BackgroundEvent>>,
}

impl ReceiverGuard {
    pub async fn recv(&mut self) -> Option<BackgroundEvent> {
        self.rx.as_mut()?.recv().await
    }

    pub fn try_recv(&mut self) -> Result<BackgroundEvent, mpsc::error::TryRecvError> {
        self.rx
            .as_mut()
            .ok_or(mpsc::error::TryRecvError::Disconnected)?
            .try_recv()
    }
}

impl Drop for ReceiverGuard {
    fn drop(&mut self) {
        if let Some(rx) = self.rx.take() {
            let state = SESSIONS.get(&self.session_id).map(|r| r.value().clone());
            if let Some(state) = state {
                *state.rx.lock().unwrap_or_else(|e| e.into_inner()) = Some(rx);
            }
            try_cleanup_session(&self.session_id);
        }
    }
}

/// Subscribe to a session's background event stream. Retrieves or initializes the session context.
pub fn subscribe(session_id: &str) -> ReceiverGuard {
    let state = SESSIONS
        .entry(session_id.to_string())
        .or_insert_with(|| {
            let (tx, rx) = mpsc::channel(1024);
            let (tasks_tx, tasks_rx) = watch::channel(0usize);
            Arc::new(SessionState {
                tx,
                rx: Mutex::new(Some(rx)),
                tasks_tx,
                tasks_rx,
            })
        })
        .value()
        .clone();
    let rx = state.rx.lock().unwrap_or_else(|e| e.into_inner()).take();
    ReceiverGuard {
        session_id: session_id.to_string(),
        rx,
    }
}

/// A RAII guard to track background task lifecycle within a session.
pub struct TaskGuard(String);
impl TaskGuard {
    pub fn new(session_id: String) -> Self {
        SESSIONS
            .entry(session_id.clone())
            .or_insert_with(|| {
                let (tx, rx) = mpsc::channel(1024);
                let (tasks_tx, tasks_rx) = watch::channel(0);
                Arc::new(SessionState {
                    tx,
                    rx: Mutex::new(Some(rx)),
                    tasks_tx,
                    tasks_rx,
                })
            })
            .value()
            .tasks_tx
            .send_modify(|c| *c += 1);
        Self(session_id)
    }
}

impl Drop for TaskGuard {
    fn drop(&mut self) {
        if let Some(state) = SESSIONS.get(&self.0) {
            state.tasks_tx.send_modify(|c| *c = c.saturating_sub(1));
        }
        try_cleanup_session(&self.0);
    }
}

/// Push a background event to a target session if it has active listeners.
pub fn deliver_event(session_id: &str, event: BackgroundEvent) {
    if let Some(state) = SESSIONS.get(session_id) {
        if let Err(e) = state.tx.try_send(event) {
            warn!("Background event dropped for session {}: {}", session_id, e);
        }
    }
}

/// Returns true if there are active background tasks for the session.
pub fn has_active_tasks(session_id: &str) -> bool {
    SESSIONS
        .get(session_id)
        .map(|s| *s.tasks_rx.borrow() > 0)
        .unwrap_or(false)
}

/// Get a watcher for the session's active task count.
pub fn get_task_watcher(session_id: &str) -> Option<watch::Receiver<usize>> {
    SESSIONS.get(session_id).map(|s| s.tasks_rx.clone())
}

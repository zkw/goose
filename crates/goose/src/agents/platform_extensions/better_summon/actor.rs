use dashmap::DashMap;
use once_cell::sync::Lazy;
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, watch};
use crate::conversation::message::Message;

#[derive(Clone)]
pub enum BackgroundEvent {
    Message(Message),
    McpNotification(rmcp::model::ServerNotification),
}

pub struct SessionState {
    pub tx: mpsc::UnboundedSender<BackgroundEvent>,
    pub rx: Mutex<Option<mpsc::UnboundedReceiver<BackgroundEvent>>>,
    pub tasks_tx: watch::Sender<usize>,
    pub tasks_rx: watch::Receiver<usize>,
}

static SESSIONS: Lazy<DashMap<String, Arc<SessionState>>> = Lazy::new(DashMap::new);

fn get_state(session_id: &str) -> Arc<SessionState> {
    SESSIONS.entry(session_id.to_string()).or_insert_with(|| {
        let (tx, rx) = mpsc::unbounded_channel();
        let (tasks_tx, tasks_rx) = watch::channel(0);
        Arc::new(SessionState {
            tx,
            rx: Mutex::new(Some(rx)),
            tasks_tx,
            tasks_rx,
        })
    }).clone()
}

pub struct ReceiverGuard {
    session_id: String,
    pub rx: Option<mpsc::UnboundedReceiver<BackgroundEvent>>,
}

impl ReceiverGuard {
    pub async fn recv(&mut self) -> Option<BackgroundEvent> {
        if let Some(r) = self.rx.as_mut() { r.recv().await } else { None }
    }
    
    pub fn try_recv(&mut self) -> Result<BackgroundEvent, mpsc::error::TryRecvError> {
        if let Some(r) = self.rx.as_mut() {
            r.try_recv()
        } else {
            Err(mpsc::error::TryRecvError::Disconnected)
        }
    }
}

impl Drop for ReceiverGuard {
    fn drop(&mut self) {
        if let Some(rx) = self.rx.take() {
            if let Some(state) = SESSIONS.get(&self.session_id) {
                *state.rx.lock().unwrap() = Some(rx);
            }
        }
    }
}

pub fn subscribe(session_id: &str) -> ReceiverGuard {
    let rx = get_state(session_id).rx.lock().unwrap().take();
    ReceiverGuard { 
        session_id: session_id.to_string(), 
        rx 
    }
}

pub struct TaskGuard(String);
impl TaskGuard {
    pub fn new(session_id: String) -> Self {
        let state = get_state(&session_id);
        let current = *state.tasks_rx.borrow();
        let _ = state.tasks_tx.send(current + 1);
        Self(session_id)
    }
}
impl Drop for TaskGuard {
    fn drop(&mut self) {
        let mut maybe_remove = false;
        if let Some(state) = SESSIONS.get(&self.0) {
            let next_count = state.tasks_rx.borrow().saturating_sub(1);
            let _ = state.tasks_tx.send(next_count);
            // 记录潜在的清理意图
            maybe_remove = next_count == 0;
        }

        if maybe_remove {
            // 利用 remove_if 实现原子化条件删除，彻底消灭 TOCTOU 竞态
            SESSIONS.remove_if(&self.0, |_, state| {
                *state.tasks_rx.borrow() == 0 && state.rx.lock().unwrap().is_some()
            });
        }
    }
}

pub fn deliver_event(session_id: &str, event: BackgroundEvent) {
    let _ = get_state(session_id).tx.send(event);
}

pub fn is_door_held(session_id: &str) -> bool {
    SESSIONS.get(session_id).map(|s| *s.tasks_rx.borrow() > 0).unwrap_or(false)
}

pub fn get_task_watcher(session_id: &str) -> Option<watch::Receiver<usize>> {
    SESSIONS.get(session_id).map(|s| s.tasks_rx.clone())
}

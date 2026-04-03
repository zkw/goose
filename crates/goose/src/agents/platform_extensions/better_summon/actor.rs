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

/// 原子化清理逻辑：只有当后台任务数归零、接收端已安全归还，且 Channel 内没有任何余温消息时，才正式移除。
/// 修复：加入 is_empty() 检测，防止 final_report 等阶段性汇报在主循环 Idle 期间由于状态清理而丢失。
fn try_cleanup_session(session_id: &str) {
    SESSIONS.remove_if(session_id, |_key, state| {
        let tasks_zero = *state.tasks_rx.borrow() == 0;
        let rx_idle_and_empty = state.rx.lock().unwrap().as_ref()
            .is_some_and(|r| r.is_empty());
        tasks_zero && rx_idle_and_empty
    });
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
            try_cleanup_session(&self.session_id);
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
        state.tasks_tx.send_modify(|c| *c += 1);
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

pub fn deliver_event(session_id: &str, event: BackgroundEvent) {
    let _ = get_state(session_id).tx.send(event);
}

pub fn is_door_held(session_id: &str) -> bool {
    SESSIONS.get(session_id).map(|s| *s.tasks_rx.borrow() > 0).unwrap_or(false)
}

pub fn get_task_watcher(session_id: &str) -> Option<watch::Receiver<usize>> {
    SESSIONS.get(session_id).map(|s| s.tasks_rx.clone())
}

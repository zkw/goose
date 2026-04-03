use dashmap::DashMap;
use once_cell::sync::Lazy;
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, watch};
use crate::conversation::message::Message;
use tracing::warn;

#[derive(Clone)]
pub enum BackgroundEvent {
    Message(Message),
    McpNotification(rmcp::model::ServerNotification),
}

pub struct SessionState {
    pub tx: mpsc::Sender<BackgroundEvent>,
    pub rx: Mutex<Option<mpsc::Receiver<BackgroundEvent>>>,
    pub tasks_tx: watch::Sender<usize>,
    pub tasks_rx: watch::Receiver<usize>,
}

static SESSIONS: Lazy<DashMap<String, Arc<SessionState>>> = Lazy::new(DashMap::new);

/// 原子化清理逻辑：只有当后台任务数归零、接收端已安全归还，才移除会话。
/// 修复：防御性处理锁毒化。
fn try_cleanup_session(session_id: &str) {
    SESSIONS.remove_if(session_id, |_key, state| {
        let tasks_zero = *state.tasks_rx.borrow() == 0;
        let rx_idle = state.rx.lock().ok().and_then(|l| {
            l.as_ref().map(|_| true) // 仅检查是否归还，不在此处读 is_empty 以防锁竞争
        }).unwrap_or(false);
        tasks_zero && rx_idle
    });
}

pub struct ReceiverGuard {
    session_id: String,
    pub rx: Option<mpsc::Receiver<BackgroundEvent>>,
}

impl ReceiverGuard {
    pub async fn recv(&mut self) -> Option<BackgroundEvent> {
        if let Some(r) = self.rx.as_mut() { r.recv().await } else { None }
    }
}

impl Drop for ReceiverGuard {
    fn drop(&mut self) {
        if let Some(rx) = self.rx.take() {
            if let Some(state) = SESSIONS.get(&self.session_id) {
                // 修复：防御 Mutex 毒化，不调用 unwrap()
                if let Ok(mut lock) = state.rx.lock() {
                    *lock = Some(rx);
                }
            }
            try_cleanup_session(&self.session_id);
        }
    }
}

pub fn subscribe(session_id: &str) -> ReceiverGuard {
    let rx = SESSIONS.get(session_id)
        .and_then(|s| s.rx.lock().ok()?.take());
    
    ReceiverGuard { 
        session_id: session_id.to_string(), 
        rx 
    }
}

pub struct TaskGuard(String);
impl TaskGuard {
    pub fn new(session_id: String) -> Self {
        SESSIONS.entry(session_id.clone())
            .or_insert_with(|| {
                // KISS: 使用有界通道防止 OOM (1024 条足以应对绝大多数爆发输出)
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
            .tasks_tx.send_modify(|c| *c += 1);
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
    if let Some(state) = SESSIONS.get(session_id) {
        let tx = state.tx.clone();
        tokio::spawn(async move {
            // 如果通道满，记录警告而不是永久阻塞后台任务，实现一定程度的自护
            if let Err(e) = tx.try_send(event) {
                warn!("Background event dropped for session {}: {}", session_id, e);
            }
        });
    }
}

pub fn is_door_held(session_id: &str) -> bool {
    SESSIONS.get(session_id).map(|s| *s.tasks_rx.borrow() > 0).unwrap_or(false)
}

pub fn get_task_watcher(session_id: &str) -> Option<watch::Receiver<usize>> {
    SESSIONS.get(session_id).map(|s| s.tasks_rx.clone())
}

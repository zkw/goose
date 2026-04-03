use dashmap::DashMap;
use once_cell::sync::Lazy;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use crate::conversation::message::Message;

#[derive(Clone)]
pub enum BackgroundEvent {
    Message(Message),
    McpNotification(rmcp::model::ServerNotification),
}

pub struct SessionState {
    pub tx: mpsc::UnboundedSender<BackgroundEvent>, // UnboundedSender 原生支持多线程并发发送，无需 Mutex
    pub rx: Mutex<Option<mpsc::UnboundedReceiver<BackgroundEvent>>>, // 暂时寄存 rx，直到架构师来“借走”它
    pub counts: Mutex<(i32, i32)>, 
}

static SESSIONS: Lazy<DashMap<String, Arc<SessionState>>> = Lazy::new(DashMap::new);

/// DRY: 内部统一的状态初始化入口
fn get_or_create_state(session_id: &str) -> Arc<SessionState> {
    SESSIONS.entry(session_id.to_string()).or_insert_with(|| {
        let (tx, rx) = mpsc::unbounded_channel();
        Arc::new(SessionState {
            tx,
            rx: Mutex::new(Some(rx)),
            counts: Mutex::new((0, 0)),
        })
    }).clone()
}

/// 接收端守卫：借走 rx 监听，并在 Drop 时安全归还，确保消息在空档期不丢失
pub struct ReceiverGuard {
    session_id: String,
    pub rx: Option<mpsc::UnboundedReceiver<BackgroundEvent>>,
}

impl ReceiverGuard {
    pub async fn recv(&mut self) -> Option<BackgroundEvent> {
        if let Some(r) = self.rx.as_mut() {
            r.recv().await
        } else {
            None
        }
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
                let mut slot = state.rx.lock().unwrap();
                *slot = Some(rx); // 归还接收端
            }
        }
    }
}

pub fn subscribe(session_id: &str) -> ReceiverGuard {
    let state = get_or_create_state(session_id);
    let mut rx_slot = state.rx.lock().unwrap();
    let rx = rx_slot.take().expect("并发安全错误：严禁在未归还前并行 subscribe 同一个 Session");
    ReceiverGuard { 
        session_id: session_id.to_string(), 
        rx: Some(rx) 
    }
}

pub fn update_session(session_id: &str, task_delta: i32, inbox_delta: i32) {
    let state = get_or_create_state(session_id);
    let mut counts = state.counts.lock().unwrap();
    counts.0 += task_delta;
    counts.1 += inbox_delta;

    if counts.0 == 0 && counts.1 == 0 {
        // 在持有 counts 锁的情况下提前 drop，确保 remove 时无新接入
        drop(counts);
        SESSIONS.remove(session_id);
    }
}

pub struct TaskGuard { session_id: String }
impl TaskGuard {
    pub fn new(session_id: String) -> Self {
        update_session(&session_id, 1, 0);
        Self { session_id }
    }
}
impl Drop for TaskGuard {
    fn drop(&mut self) { update_session(&self.session_id, -1, 0); }
}

pub struct InboxGuard { session_id: String }
impl InboxGuard {
    pub fn new(session_id: String) -> Self {
        update_session(&session_id, 0, 1);
        Self { session_id }
    }
}
impl Drop for InboxGuard {
    fn drop(&mut self) { update_session(&self.session_id, 0, -1); }
}

pub fn deliver_event(session_id: &str, event: BackgroundEvent) {
    let state = get_or_create_state(session_id);
    let _ = state.tx.send(event); // 无锁原子发送，极致吞吐
}

pub fn is_door_held(session_id: &str) -> bool {
    SESSIONS.get(session_id).map(|s| s.counts.lock().unwrap().0 > 0).unwrap_or(false)
}

use crate::conversation::message::Message;
use dashmap::DashMap;
use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;

#[derive(Clone)]
pub enum BackgroundEvent {
    Message(Message),
    McpNotification(rmcp::model::ServerNotification),
}

pub struct SessionState {
    tx: mpsc::UnboundedSender<BackgroundEvent>,
    counts: AtomicU64,
}

static SESSIONS: Lazy<DashMap<String, Arc<SessionState>>> = Lazy::new(DashMap::new);

/// 获取或创建 Session 状态（内部使用）
fn get_or_init(session_id: &str) -> Arc<SessionState> {
    SESSIONS.entry(session_id.to_string()).or_insert_with(|| {
        let (tx, _) = mpsc::unbounded_channel(); // 这里的 Receiver 将被 subscribe 接管
        Arc::new(SessionState { tx, counts: AtomicU64::new(0) })
    }).clone()
}

/// 在 Agent 开启回复流时调用，获取一个接收端并激活通道
pub fn subscribe(session_id: &str) -> mpsc::UnboundedReceiver<BackgroundEvent> {
    let (tx, rx) = mpsc::unbounded_channel();
    SESSIONS.insert(session_id.to_string(), Arc::new(SessionState {
        tx,
        counts: AtomicU64::new(0),
    }));
    rx
}

pub fn update_session(session_id: &str, guard_delta: i32, inbox_delta: i32) {
    if let Some(s) = SESSIONS.get(session_id) {
        let delta = ((guard_delta as i64) << 32) | (inbox_delta as i64 & 0xFFFFFFFF);
        let new_counts = s.counts.fetch_add(delta as u64, Ordering::SeqCst).wrapping_add(delta as u64);
        
        // 如果计数归零且不再有活跃任务，将 Sender 从全局 Map 移除
        // 这会导致 subscribe 拿到的 Receiver 返回 None
        if new_counts == 0 {
            drop(s);
            SESSIONS.remove(session_id);
        }
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
    if let Some(s) = SESSIONS.get(session_id) {
        let _ = s.tx.send(event);
    }
}

pub fn is_door_held(session_id: &str) -> bool {
    SESSIONS.get(session_id).map(|s| (s.counts.load(Ordering::Acquire) >> 32) > 0).unwrap_or(false)
}

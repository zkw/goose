use crate::conversation::message::Message;
use dashmap::DashMap;
use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;

#[derive(Clone)]
pub enum BackgroundEvent {
    Message(Message),
    McpNotification(rmcp::model::ServerNotification),
}

pub struct SessionState {
    tx: mpsc::UnboundedSender<BackgroundEvent>,
    active_tasks: AtomicI32,
    active_inboxes: AtomicI32,
}

static SESSIONS: Lazy<DashMap<String, Arc<SessionState>>> = Lazy::new(DashMap::new);

/// 在 Agent 开启回复流时调用，获取一个接收端并激活通道
pub fn subscribe(session_id: &str) -> mpsc::UnboundedReceiver<BackgroundEvent> {
    let (tx, rx) = mpsc::unbounded_channel();
    SESSIONS.insert(session_id.to_string(), Arc::new(SessionState {
        tx,
        active_tasks: AtomicI32::new(0),
        active_inboxes: AtomicI32::new(0),
    }));
    rx
}

pub fn update_session(session_id: &str, guard_delta: i32, inbox_delta: i32) {
    if let Some(s) = SESSIONS.get(session_id) {
        let tasks = s.active_tasks.fetch_add(guard_delta, Ordering::SeqCst) + guard_delta;
        let inboxes = s.active_inboxes.fetch_add(inbox_delta, Ordering::SeqCst) + inbox_delta;
        
        // 只有当任务和入站消息全部归零时，才彻底从全局映射中移除 Sender，触发 Receiver::recv 返回 None
        if tasks == 0 && inboxes == 0 {
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
    SESSIONS.get(session_id).map(|s| s.active_tasks.load(Ordering::Acquire) > 0).unwrap_or(false)
}

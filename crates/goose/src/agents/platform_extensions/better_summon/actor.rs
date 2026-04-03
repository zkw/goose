use crate::conversation::message::Message;
use dashmap::DashMap;
use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

#[derive(Clone)]
pub enum BackgroundEvent {
    Message(Message),
    McpNotification(rmcp::model::ServerNotification),
}

pub struct SessionState {
    // 使用 Mutex 包装 tx，允许在保留计数器的前提下，无缝切换接收端
    pub tx: Mutex<mpsc::UnboundedSender<BackgroundEvent>>,
    pub active_tasks: AtomicI32,
    pub active_inboxes: AtomicI32,
}

static SESSIONS: Lazy<DashMap<String, Arc<SessionState>>> = Lazy::new(DashMap::new);

/// 核心逻辑：确保 Session 状态实体存在（DRY）
fn ensure_session(session_id: &str) -> Arc<SessionState> {
    SESSIONS
        .entry(session_id.to_string())
        .or_insert_with(|| {
            let (tx, _) = mpsc::unbounded_channel();
            Arc::new(SessionState {
                tx: Mutex::new(tx),
                active_tasks: AtomicI32::new(0),
                active_inboxes: AtomicI32::new(0),
            })
        })
        .value()
        .clone()
}

/// 在 Agent 开启回复流时调用，获取一个接收端并激活/更新通道
pub fn subscribe(session_id: &str) -> mpsc::UnboundedReceiver<BackgroundEvent> {
    let (tx, rx) = mpsc::unbounded_channel();
    let s = ensure_session(session_id);
    
    // 仅替换发送端，保留原有的 active_tasks 和 active_inboxes 计数
    let mut guard = s.tx.lock().unwrap();
    *guard = tx;
    
    rx
}

pub fn update_session(session_id: &str, guard_delta: i32, inbox_delta: i32) {
    let s = ensure_session(session_id);
    let tasks = s.active_tasks.fetch_add(guard_delta, Ordering::SeqCst) + guard_delta;
    let inboxes = s.active_inboxes.fetch_add(inbox_delta, Ordering::SeqCst) + inbox_delta;

    // 只有当所有活动任务和入站消息全部归零时，才彻底移除 Session 实体
    if tasks == 0 && inboxes == 0 {
        // 由于 DashMap 可能在 fetch_add 和 remove 之间发生并发插入，
        // 这里依赖 Entry 锁或简单的直接 remove 是安全的，因为 ensure_session 会在需要时重新创建
        SESSIONS.remove(session_id);
    }
}

pub struct TaskGuard {
    session_id: String,
}
impl TaskGuard {
    pub fn new(session_id: String) -> Self {
        update_session(&session_id, 1, 0);
        Self { session_id }
    }
}
impl Drop for TaskGuard {
    fn drop(&mut self) {
        update_session(&self.session_id, -1, 0);
    }
}

pub struct InboxGuard {
    session_id: String,
}
impl InboxGuard {
    pub fn new(session_id: String) -> Self {
        update_session(&session_id, 0, 1);
        Self { session_id }
    }
}
impl Drop for InboxGuard {
    fn drop(&mut self) {
        update_session(&self.session_id, 0, -1);
    }
}

pub fn deliver_event(session_id: &str, event: BackgroundEvent) {
    if let Some(s) = SESSIONS.get(session_id) {
        let tx = s.tx.lock().unwrap();
        let _ = tx.send(event);
    }
}

pub fn is_door_held(session_id: &str) -> bool {
    SESSIONS
        .get(session_id)
        .map(|s| s.active_tasks.load(Ordering::Acquire) > 0)
        .unwrap_or(false)
}

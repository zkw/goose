use crate::conversation::message::Message;
use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use once_cell::sync::Lazy;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

#[derive(Clone)]
pub enum BackgroundEvent {
    Message(Message),
    McpNotification(rmcp::model::ServerNotification),
}

pub struct SessionState {
    pub tx: Mutex<mpsc::UnboundedSender<BackgroundEvent>>,
    // (active_tasks, active_inboxes)
    pub counts: Mutex<(i32, i32)>,
}

static SESSIONS: Lazy<DashMap<String, Arc<SessionState>>> = Lazy::new(DashMap::new);

/// 在 Agent 开启回复流时调用，获取一个接收端
pub fn subscribe(session_id: &str) -> mpsc::UnboundedReceiver<BackgroundEvent> {
    let (tx, rx) = mpsc::unbounded_channel();
    match SESSIONS.entry(session_id.to_string()) {
        Entry::Occupied(o) => {
            let mut s_tx = o.get().tx.lock().unwrap();
            *s_tx = tx;
        }
        Entry::Vacant(v) => {
            v.insert(Arc::new(SessionState {
                tx: Mutex::new(tx),
                counts: Mutex::new((0, 0)),
            }));
        }
    }
    rx
}

pub fn update_session(session_id: &str, task_delta: i32, inbox_delta: i32) {
    match SESSIONS.entry(session_id.to_string()) {
        Entry::Occupied(mut o) => {
            let mut counts = o.get().counts.lock().unwrap();
            counts.0 += task_delta;
            counts.1 += inbox_delta;
            if counts.0 == 0 && counts.1 == 0 {
                drop(counts); // 释放锁后原子移除
                o.remove();
            }
        }
        Entry::Vacant(v) => {
            // 如果是在订阅前创建 Guard，则初始化计数
            if task_delta > 0 || inbox_delta > 0 {
                let (tx, _) = mpsc::unbounded_channel();
                v.insert(Arc::new(SessionState {
                    tx: Mutex::new(tx),
                    counts: Mutex::new((task_delta, inbox_delta)),
                }));
            }
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
        let tx = s.tx.lock().unwrap();
        let _ = tx.send(event);
    }
}

pub fn is_door_held(session_id: &str) -> bool {
    SESSIONS.get(session_id).map(|s| s.counts.lock().unwrap().0 > 0).unwrap_or(false)
}

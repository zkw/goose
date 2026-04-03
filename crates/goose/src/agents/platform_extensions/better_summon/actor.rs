use crate::conversation::message::Message;
use dashmap::DashMap;
use once_cell::sync::Lazy;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;
use tokio::sync::{oneshot, Mutex};

#[derive(Clone)]
pub enum BackgroundEvent {
    Message(Message),
    McpNotification(rmcp::model::ServerNotification),
}

pub struct SessionState {
    pub guards: AtomicI32,
    pub inboxes: AtomicI32,
    pub events: Mutex<VecDeque<BackgroundEvent>>,
    pub waiters: Mutex<VecDeque<oneshot::Sender<Option<BackgroundEvent>>>>,
}

static SESSIONS: Lazy<DashMap<String, Arc<SessionState>>> = Lazy::new(DashMap::new);

fn get_or_create_session(session_id: &str) -> Arc<SessionState> {
    SESSIONS
        .entry(session_id.to_string())
        .or_insert_with(|| {
            Arc::new(SessionState {
                guards: AtomicI32::new(0),
                inboxes: AtomicI32::new(0),
                events: Mutex::new(VecDeque::new()),
                waiters: Mutex::new(VecDeque::new()),
            })
        })
        .clone()
}

pub fn update_session(session_id: &str, guard_delta: i32, inbox_delta: i32) {
    let s = get_or_create_session(session_id);
    s.guards.fetch_add(guard_delta, Ordering::Relaxed);
    s.inboxes.fetch_add(inbox_delta, Ordering::Relaxed);

    if s.guards.load(Ordering::Acquire) <= 0 && s.inboxes.load(Ordering::Acquire) <= 0 {
        let sid = session_id.to_string();
        let state = s.clone();
        tokio::spawn(async move {
            let mut waiters = state.waiters.lock().await;
            for w in waiters.drain(..) {
                let _ = w.send(None);
            }
            if state.events.lock().await.is_empty() {
                SESSIONS.remove(&sid);
            }
        });
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
    let s = get_or_create_session(session_id);
    let state = s.clone();
    tokio::spawn(async move {
        let mut waiters = state.waiters.lock().await;
        // Clean up dropped waiters (fix the oneshot leak)
        waiters.retain(|w| !w.is_closed());

        while let Some(w) = waiters.pop_front() {
            if w.send(Some(event.clone())).is_ok() {
                return;
            }
        }
        state.events.lock().await.push_back(event);
    });
}

pub async fn try_wait_event(session_id: &str) -> Option<BackgroundEvent> {
    let s = get_or_create_session(session_id);
    let event = s.events.lock().await.pop_front();
    event
}

pub async fn wait_event(session_id: &str) -> Option<BackgroundEvent> {
    let s = get_or_create_session(session_id);
    {
        let mut events = s.events.lock().await;
        if let Some(ev) = events.pop_front() {
            return Some(ev);
        }
    }

    if s.guards.load(Ordering::Acquire) <= 0 && s.inboxes.load(Ordering::Acquire) <= 0 {
        return None;
    }

    let (tx, rx) = oneshot::channel();
    s.waiters.lock().await.push_back(tx);
    rx.await.unwrap_or(None)
}

pub async fn is_door_held(session_id: &str) -> bool {
    SESSIONS
        .get(session_id)
        .map(|s| s.guards.load(Ordering::Relaxed) > 0)
        .unwrap_or(false)
}

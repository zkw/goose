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
    pub tx: mpsc::UnboundedSender<BackgroundEvent>,
    pub rx: Mutex<Option<mpsc::UnboundedReceiver<BackgroundEvent>>>,
    pub tasks: Mutex<usize>,
}

static SESSIONS: Lazy<DashMap<String, Arc<SessionState>>> = Lazy::new(DashMap::new);

fn get_state(session_id: &str) -> Arc<SessionState> {
    SESSIONS.entry(session_id.to_string()).or_insert_with(|| {
        let (tx, rx) = mpsc::unbounded_channel();
        Arc::new(SessionState {
            tx,
            rx: Mutex::new(Some(rx)),
            tasks: Mutex::new(0),
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
    let rx = get_state(session_id).rx.lock().unwrap().take()
        .expect("并发安全错误：严禁在未归还前并行 subscribe 同一个 Session");
    ReceiverGuard { session_id: session_id.to_string(), rx: Some(rx) }
}

pub struct TaskGuard(String);
impl TaskGuard {
    pub fn new(session_id: String) -> Self {
        *get_state(&session_id).tasks.lock().unwrap() += 1;
        Self(session_id)
    }
}
impl Drop for TaskGuard {
    fn drop(&mut self) {
        if let Some(state) = SESSIONS.get(&self.0) {
            *state.tasks.lock().unwrap() -= 1;
        }
    }
}

pub fn deliver_event(session_id: &str, event: BackgroundEvent) {
    let _ = get_state(session_id).tx.send(event);
}

pub fn is_door_held(session_id: &str) -> bool {
    SESSIONS.get(session_id).map(|s| *s.tasks.lock().unwrap() > 0).unwrap_or(false)
}

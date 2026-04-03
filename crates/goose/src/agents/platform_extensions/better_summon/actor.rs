use crate::conversation::message::Message;
use std::collections::{HashMap, VecDeque};
use std::sync::OnceLock;
use tokio::sync::{mpsc, oneshot};

#[derive(Clone)]
pub enum BackgroundEvent {
    Message(Message),
    McpNotification(rmcp::model::ServerNotification),
}

pub enum ActorCmd {
    Update {
        session_id: String,
        guard_delta: i32,
        inbox_delta: i32,
    },
    DeliverEvent {
        session_id: String,
        event: BackgroundEvent,
    },
    WaitEvent {
        session_id: String,
        block: bool,
        reply: oneshot::Sender<Option<BackgroundEvent>>,
    },
    IsDoorHeld {
        session_id: String,
        reply: oneshot::Sender<bool>,
    },
}

static ACTOR_TX: OnceLock<mpsc::UnboundedSender<ActorCmd>> = OnceLock::new();

pub fn get_actor_tx() -> mpsc::UnboundedSender<ActorCmd> {
    ACTOR_TX
        .get_or_init(|| {
            let (tx, rx) = mpsc::unbounded_channel();
            tokio::spawn(actor_loop(rx));
            tx
        })
        .clone()
}

struct SessionData {
    guards: usize,
    inboxes: usize,
    events: VecDeque<BackgroundEvent>,
    waiters: VecDeque<oneshot::Sender<Option<BackgroundEvent>>>,
}

async fn actor_loop(mut rx: mpsc::UnboundedReceiver<ActorCmd>) {
    let mut sessions: HashMap<String, SessionData> = HashMap::new();

    while let Some(cmd) = rx.recv().await {
        match cmd {
            ActorCmd::Update {
                session_id,
                guard_delta,
                inbox_delta,
            } => {
                let s = sessions.entry(session_id.clone()).or_insert_with(|| SessionData {
                    guards: 0,
                    inboxes: 0,
                    events: VecDeque::new(),
                    waiters: VecDeque::new(),
                });
                s.guards = (s.guards as i32 + guard_delta).max(0) as usize;
                s.inboxes = (s.inboxes as i32 + inbox_delta).max(0) as usize;

                if s.guards == 0 && s.inboxes == 0 {
                    for w in s.waiters.drain(..) {
                        let _ = w.send(None);
                    }
                    if s.events.is_empty() {
                        sessions.remove(&session_id);
                    }
                }
            }
            ActorCmd::DeliverEvent { session_id, event } => {
                let s = sessions.entry(session_id).or_insert_with(|| SessionData {
                    guards: 0,
                    inboxes: 0,
                    events: VecDeque::new(),
                    waiters: VecDeque::new(),
                });
                if let Some(w) = s.waiters.pop_front() {
                    if w.send(Some(event.clone())).is_err() {
                        s.events.push_front(event);
                    }
                } else {
                    s.events.push_back(event);
                }
            }
            ActorCmd::WaitEvent {
                session_id,
                block,
                reply,
            } => {
                if let Some(s) = sessions.get_mut(&session_id) {
                    if let Some(ev) = s.events.pop_front() {
                        let _ = reply.send(Some(ev));
                        if s.guards == 0 && s.inboxes == 0 && s.events.is_empty() && s.waiters.is_empty() {
                            sessions.remove(&session_id);
                        }
                    } else if block && (s.guards > 0 || s.inboxes > 0) {
                        s.waiters.push_back(reply);
                    } else {
                        let _ = reply.send(None);
                    }
                } else {
                    let _ = reply.send(None);
                }
            }
            ActorCmd::IsDoorHeld { session_id, reply } => {
                let held = sessions
                    .get(&session_id)
                    .map(|s| s.guards > 0)
                    .unwrap_or(false);
                let _ = reply.send(held);
            }
        }
    }
}

pub struct TaskGuard {
    session_id: String,
}

impl TaskGuard {
    pub fn new(session_id: String) -> Self {
        let _ = get_actor_tx().send(ActorCmd::Update {
            session_id: session_id.clone(),
            guard_delta: 1,
            inbox_delta: 0,
        });
        Self { session_id }
    }
}

impl Drop for TaskGuard {
    fn drop(&mut self) {
        let _ = get_actor_tx().send(ActorCmd::Update {
            session_id: self.session_id.clone(),
            guard_delta: -1,
            inbox_delta: 0,
        });
    }
}

pub struct InboxGuard {
    session_id: String,
}

impl InboxGuard {
    pub fn new(session_id: String) -> Self {
        let _ = get_actor_tx().send(ActorCmd::Update {
            session_id: session_id.clone(),
            guard_delta: 0,
            inbox_delta: 1,
        });
        Self { session_id }
    }
}

impl Drop for InboxGuard {
    fn drop(&mut self) {
        let _ = get_actor_tx().send(ActorCmd::Update {
            session_id: self.session_id.clone(),
            guard_delta: 0,
            inbox_delta: -1,
        });
    }
}

pub fn deliver_event(session_id: &str, event: BackgroundEvent) {
    let _ = get_actor_tx().send(ActorCmd::DeliverEvent {
        session_id: session_id.to_string(),
        event,
    });
}

async fn wait_event_internal(session_id: &str, block: bool) -> Option<BackgroundEvent> {
    let (tx, rx) = oneshot::channel();
    let _ = get_actor_tx().send(ActorCmd::WaitEvent {
        session_id: session_id.to_string(),
        block,
        reply: tx,
    });
    rx.await.unwrap_or(None)
}

pub async fn try_wait_event(session_id: &str) -> Option<BackgroundEvent> {
    wait_event_internal(session_id, false).await
}

pub async fn wait_event(session_id: &str) -> Option<BackgroundEvent> {
    wait_event_internal(session_id, true).await
}

pub async fn is_door_held(session_id: &str) -> bool {
    let (tx, rx) = oneshot::channel::<bool>();
    let _ = get_actor_tx().send(ActorCmd::IsDoorHeld {
        session_id: session_id.to_string(),
        reply: tx,
    });
    rx.await.unwrap_or(false)
}

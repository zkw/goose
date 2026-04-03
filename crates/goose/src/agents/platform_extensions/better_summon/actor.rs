use crate::conversation::message::Message;
use std::collections::{HashMap, VecDeque};
use std::sync::OnceLock;
use tokio::sync::{mpsc, oneshot};

pub enum ActorCmd {
    AddGuard {
        session_id: String,
    },
    DropGuard {
        session_id: String,
    },
    AddInbox {
        session_id: String,
    },
    DropInbox {
        session_id: String,
    },
    DeliverMessage {
        session_id: String,
        message: Message,
    },
    TryWaitMessage {
        session_id: String,
        reply: oneshot::Sender<Option<Message>>,
    },
    WaitMessage {
        session_id: String,
        reply: oneshot::Sender<Option<Message>>,
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
    messages: VecDeque<Message>,
    waiters: VecDeque<oneshot::Sender<Option<Message>>>,
}

async fn actor_loop(mut rx: mpsc::UnboundedReceiver<ActorCmd>) {
    let mut sessions: HashMap<String, SessionData> = HashMap::new();

    while let Some(cmd) = rx.recv().await {
        match cmd {
            ActorCmd::AddGuard { session_id } => {
                let s = sessions.entry(session_id).or_insert_with(|| SessionData {
                    guards: 0,
                    inboxes: 0,
                    messages: VecDeque::new(),
                    waiters: VecDeque::new(),
                });
                s.guards += 1;
            }
            ActorCmd::DropGuard { session_id } => {
                if let Some(s) = sessions.get_mut(&session_id) {
                    s.guards = s.guards.saturating_sub(1);
                    if s.guards == 0 && s.inboxes == 0 {
                        for w in s.waiters.drain(..) {
                            let _ = w.send(None);
                        }
                    }
                    if s.guards == 0
                        && s.inboxes == 0
                        && s.messages.is_empty()
                        && s.waiters.is_empty()
                    {
                        sessions.remove(&session_id);
                    }
                }
            }
            ActorCmd::AddInbox { session_id } => {
                let s = sessions.entry(session_id).or_insert_with(|| SessionData {
                    guards: 0,
                    inboxes: 0,
                    messages: VecDeque::new(),
                    waiters: VecDeque::new(),
                });
                s.inboxes += 1;
            }
            ActorCmd::DropInbox { session_id } => {
                if let Some(s) = sessions.get_mut(&session_id) {
                    s.inboxes = s.inboxes.saturating_sub(1);
                    if s.guards == 0 && s.inboxes == 0 {
                        for w in s.waiters.drain(..) {
                            let _ = w.send(None);
                        }
                    }
                    if s.guards == 0
                        && s.inboxes == 0
                        && s.messages.is_empty()
                        && s.waiters.is_empty()
                    {
                        sessions.remove(&session_id);
                    }
                }
            }
            ActorCmd::DeliverMessage {
                session_id,
                message,
            } => {
                let s = sessions.entry(session_id).or_insert_with(|| SessionData {
                    guards: 0,
                    inboxes: 0,
                    messages: VecDeque::new(),
                    waiters: VecDeque::new(),
                });
                if let Some(w) = s.waiters.pop_front() {
                    let _ = w.send(Some(message));
                } else {
                    s.messages.push_back(message);
                }
            }
            ActorCmd::TryWaitMessage { session_id, reply } => {
                let msg = if let Some(s) = sessions.get_mut(&session_id) {
                    s.messages.pop_front()
                } else {
                    None
                };
                let _ = reply.send(msg);
            }
            ActorCmd::WaitMessage { session_id, reply } => {
                if let Some(s) = sessions.get_mut(&session_id) {
                    if let Some(msg) = s.messages.pop_front() {
                        let _ = reply.send(Some(msg));
                    } else if s.guards > 0 || s.inboxes > 0 {
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
        let _ = get_actor_tx().send(ActorCmd::AddGuard {
            session_id: session_id.clone(),
        });
        Self { session_id }
    }
}

impl Drop for TaskGuard {
    fn drop(&mut self) {
        let _ = get_actor_tx().send(ActorCmd::DropGuard {
            session_id: self.session_id.clone(),
        });
    }
}

pub struct InboxGuard {
    session_id: String,
}

impl InboxGuard {
    pub fn new(session_id: String) -> Self {
        let _ = get_actor_tx().send(ActorCmd::AddInbox {
            session_id: session_id.clone(),
        });
        Self { session_id }
    }
}

impl Drop for InboxGuard {
    fn drop(&mut self) {
        let _ = get_actor_tx().send(ActorCmd::DropInbox {
            session_id: self.session_id.clone(),
        });
    }
}

pub fn deliver_message(session_id: &str, message: Message) {
    let _ = get_actor_tx().send(ActorCmd::DeliverMessage {
        session_id: session_id.to_string(),
        message,
    });
}

pub async fn try_wait_message(session_id: &str) -> Option<Message> {
    let (tx, rx) = oneshot::channel::<Option<Message>>();
    let _ = get_actor_tx().send(ActorCmd::TryWaitMessage {
        session_id: session_id.to_string(),
        reply: tx,
    });
    rx.await.unwrap_or(None)
}

pub async fn wait_message(session_id: &str) -> Option<Message> {
    let (tx, rx) = oneshot::channel::<Option<Message>>();
    let _ = get_actor_tx().send(ActorCmd::WaitMessage {
        session_id: session_id.to_string(),
        reply: tx,
    });
    rx.await.unwrap_or(None)
}

pub async fn is_door_held(session_id: &str) -> bool {
    let (tx, rx) = oneshot::channel::<bool>();
    let _ = get_actor_tx().send(ActorCmd::IsDoorHeld {
        session_id: session_id.to_string(),
        reply: tx,
    });
    rx.await.unwrap_or(false)
}

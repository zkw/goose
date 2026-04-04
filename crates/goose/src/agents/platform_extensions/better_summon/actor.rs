use super::subagent::{run_subagent_task, SubagentRunParams};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, Semaphore};

#[derive(Clone)]
pub enum BgEvent {
    McpNotification(rmcp::model::ServerNotification),
    TaskComplete(String, String, usize), // (report, agent_id, idle_count)
}

pub enum RouterMsg {
    Bind(Arc<str>, mpsc::UnboundedSender<BgEvent>),
    Unbind(Arc<str>),
    Route(Arc<str>, BgEvent),
}

pub static ROUTER: Lazy<mpsc::UnboundedSender<RouterMsg>> = Lazy::new(|| {
    let (tx, rx) = mpsc::unbounded_channel();
    tokio::spawn(router_loop(rx));
    tx
});

async fn router_loop(mut rx: mpsc::UnboundedReceiver<RouterMsg>) {
    let mut sessions: HashMap<Arc<str>, _> = HashMap::new();
    while let Some(msg) = rx.recv().await {
        match msg {
            RouterMsg::Bind(id, tx) => {
                sessions.insert(id, tx);
            }
            RouterMsg::Unbind(id) => {
                sessions.remove(&id);
            }
            RouterMsg::Route(id, ev) => {
                if let Some(tx) = sessions.get(&id) {
                    let _ = tx.send(ev);
                }
            }
        }
    }
}

pub static SCHEDULER: Lazy<mpsc::Sender<SubagentRunParams>> = Lazy::new(|| {
    let (tx, rx) = mpsc::channel(1000);
    let limit = crate::config::Config::global()
        .get_param::<usize>("GOOSE_MAX_BACKGROUND_TASKS")
        .unwrap_or(50);
    tokio::spawn(scheduler_actor(rx, limit));
    tx
});

async fn scheduler_actor(mut rx: mpsc::Receiver<SubagentRunParams>, limit: usize) {
    let semaphore = Arc::new(Semaphore::new(limit));
    while let Some(params) = rx.recv().await {
        let permit = semaphore
            .clone()
            .acquire_owned()
            .await
            .expect("semaphore closed");
        tokio::spawn(async move {
            let _permit = permit;
            let task_name = format!("ENGINEER-{}", params.subagent_id);
            let session_id = Arc::clone(&params.parent_session_id);
            let cancel_token = params.cancellation_token.clone();
            let result = tokio::select! {
                _ = async {
                    match cancel_token { Some(t) => t.cancelled().await, None => std::future::pending().await }
                } => return,
                res = run_subagent_task(params) => res,
            };
            let report = match result {
                Ok(text) if text.is_empty() => "No report provided.".to_string(),
                Ok(text) => text,
                Err(e) => format!("Execution failed: {}", e),
            };
            route_event(session_id, BgEvent::TaskComplete(report, task_name, 0));
        });
    }
}

pub fn bind_session(id: Arc<str>, tx: mpsc::UnboundedSender<BgEvent>) {
    let _ = ROUTER.send(RouterMsg::Bind(id, tx));
}
pub fn unbind_session(id: Arc<str>) {
    let _ = ROUTER.send(RouterMsg::Unbind(id));
}
pub fn route_event(id: Arc<str>, ev: BgEvent) {
    let _ = ROUTER.send(RouterMsg::Route(id, ev));
}

pub fn dispatch_task(params: SubagentRunParams) -> Result<(), &'static str> {
    SCHEDULER
        .try_send(params)
        .map_err(|_| "scheduler offline or queue full (1000+)")
}

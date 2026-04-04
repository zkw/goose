use super::subagent::{run_subagent_task, SubagentRunParams};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, Semaphore};

#[derive(Clone)]
pub enum BgEv {
    Mcp(rmcp::model::ServerNotification),
    Done(String, String, usize),
}

enum RMsg {
    Bind(Arc<str>, mpsc::UnboundedSender<BgEv>, oneshot::Sender<bool>),
    Unbind(Arc<str>),
    Route(Arc<str>, BgEv),
}

static ROUTER: Lazy<mpsc::UnboundedSender<RMsg>> = Lazy::new(|| {
    let (tx, mut rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        let mut sessions: HashMap<Arc<str>, mpsc::UnboundedSender<BgEv>> = HashMap::new();
        while let Some(msg) = rx.recv().await {
            match msg {
                RMsg::Bind(id, tx, rtx) => { let _ = rtx.send(sessions.insert(id, tx).is_none()); }
                RMsg::Unbind(id) => { sessions.remove(&id); }
                RMsg::Route(id, ev) => if let Some(tx) = sessions.get(&id) { let _ = tx.send(ev); }
            }
        }
    });
    tx
});

static SCHEDULER: Lazy<mpsc::Sender<SubagentRunParams>> = Lazy::new(|| {
    let (tx, mut rx) = mpsc::channel::<SubagentRunParams>(1000);
    let limit = crate::config::Config::global()
        .get_param::<usize>("GOOSE_MAX_BACKGROUND_TASKS")
        .unwrap_or(50);
    tokio::spawn(async move {
        let sem = Arc::new(Semaphore::new(limit));
        while let Some(params) = rx.recv().await {
            let Ok(permit) = sem.clone().acquire_owned().await else { continue };
            let sc = Arc::clone(&sem);
            tokio::spawn(async move {
                let _permit = permit;
                let name = format!("ENGINEER-{}", params.sub_id);
                let sid = Arc::clone(&params.p_sess_id);
                let ct = params.token.clone();
                let result = tokio::select! {
                    _ = async { match &ct { Some(t) => t.cancelled().await, None => std::future::pending().await } } => return,
                    r = run_subagent_task(params) => r,
                };
                let report = match &result {
                    Ok(t) if t.is_empty() => "No report provided.".to_string(),
                    Ok(t) => t.clone(),
                    Err(e) => format!("Execution failed: {}", e),
                };
                drop(_permit);
                let _ = ROUTER.send(RMsg::Route(sid, BgEv::Done(report, name, sc.available_permits())));
            });
        }
    });
    tx
});

pub fn bind_session(id: Arc<str>, tx: mpsc::UnboundedSender<BgEv>) -> oneshot::Receiver<bool> {
    let (rtx, rrx) = oneshot::channel();
    let _ = ROUTER.send(RMsg::Bind(id, tx, rtx));
    rrx
}

pub fn unbind_session(id: Arc<str>) {
    let _ = ROUTER.send(RMsg::Unbind(id));
}

pub fn route_event(id: Arc<str>, ev: BgEv) {
    let _ = ROUTER.send(RMsg::Route(id, ev));
}

pub fn dispatch_task(params: SubagentRunParams) -> Result<(), &'static str> {
    SCHEDULER.try_send(params).map_err(|_| "scheduler offline")
}

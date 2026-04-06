use once_cell::sync::Lazy;
use std::sync::Arc;
use tokio::sync::{mpsc, Semaphore};

use super::events::BgEv;
use super::router::route_event;
use crate::agents::platform_extensions::better_summon::formats::{
    format_execution_error, ERROR_SCHEDULER_OFFLINE,
};
use crate::agents::platform_extensions::better_summon::worker::run_subagent_task;
use crate::agents::platform_extensions::better_summon::worker::SubagentRunParams;

static SCHEDULER_SEMAPHORE: Lazy<Arc<Semaphore>> = Lazy::new(|| {
    let limit = crate::config::Config::global()
        .get_param::<usize>("GOOSE_MAX_BACKGROUND_TASKS")
        .unwrap_or(50);
    Arc::new(Semaphore::new(limit))
});

#[allow(dead_code)]
static SCHEDULER: Lazy<mpsc::Sender<SubagentRunParams>> = Lazy::new(|| {
    let (tx, mut rx) = mpsc::channel::<SubagentRunParams>(1000);
    let sem = Arc::clone(&SCHEDULER_SEMAPHORE);
    tokio::spawn(async move {
        while let Some(params) = rx.recv().await {
            let permit = match sem.clone().acquire_owned().await {
                Ok(p) => p,
                Err(_) => continue,
            };
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
                drop(_permit);
                match result {
                    Ok(t) if t.is_empty() => {
                        route_event(sid, BgEv::NoReport(name, sc.available_permits()));
                    }
                    Ok(t) => {
                        route_event(sid, BgEv::Done(t, name, sc.available_permits()));
                    }
                    Err(e) => {
                        let report = format_execution_error(&e.to_string());
                        route_event(sid, BgEv::Done(report, name, sc.available_permits()));
                    }
                }
            });
        }
    });
    tx
});

pub fn idle_engineer_count() -> usize {
    SCHEDULER_SEMAPHORE.available_permits()
}

#[allow(dead_code)]
pub fn dispatch_task(params: SubagentRunParams) -> Result<(), &'static str> {
    SCHEDULER
        .try_send(params)
        .map_err(|_| ERROR_SCHEDULER_OFFLINE)
}

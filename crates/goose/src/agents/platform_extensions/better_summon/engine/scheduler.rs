use once_cell::sync::Lazy;
use std::sync::Arc;
use tokio::sync::{mpsc, Semaphore};

use super::events::BgEv;
use super::router::route_event;
use crate::agents::platform_extensions::better_summon::templates::{
    ERROR_EXECUTION_FAILED, ERROR_NO_REPORT, ERROR_SCHEDULER_OFFLINE,
};
use crate::agents::platform_extensions::better_summon::worker::run_subagent_task;
use crate::agents::platform_extensions::better_summon::worker::SubagentRunParams;

#[allow(dead_code)]
static SCHEDULER: Lazy<mpsc::Sender<SubagentRunParams>> = Lazy::new(|| {
    let (tx, mut rx) = mpsc::channel::<SubagentRunParams>(1000);
    let limit = crate::config::Config::global()
        .get_param::<usize>("GOOSE_MAX_BACKGROUND_TASKS")
        .unwrap_or(50);
    tokio::spawn(async move {
        let sem = Arc::new(Semaphore::new(limit));
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
                let report = match &result {
                    Ok(t) if t.is_empty() => ERROR_NO_REPORT.to_string(),
                    Ok(t) => t.clone(),
                    Err(e) => ERROR_EXECUTION_FAILED.replace("{}", &e.to_string()),
                };
                drop(_permit);
                route_event(sid, BgEv::Done(report, name, sc.available_permits()));
            });
        }
    });
    tx
});

#[allow(dead_code)]
pub fn dispatch_task(params: SubagentRunParams) -> Result<(), &'static str> {
    SCHEDULER.try_send(params).map_err(|_| ERROR_SCHEDULER_OFFLINE)
}

use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

pub(crate) struct PendingReports {
    pub(crate) reports: HashMap<String, Vec<String>>,
    pub(crate) task_ids: HashMap<String, Vec<String>>,
}

static REPORTS: Lazy<Arc<Mutex<PendingReports>>> = Lazy::new(|| {
    Arc::new(Mutex::new(PendingReports {
        reports: HashMap::new(),
        task_ids: HashMap::new(),
    }))
});

pub async fn push_report(session_id: &str, task_id: &str, report: &str) {
    let mut lock = REPORTS.lock().await;
    lock.reports
        .entry(session_id.to_string())
        .or_default()
        .push(report.to_string());
    lock.task_ids
        .entry(session_id.to_string())
        .or_default()
        .push(task_id.to_string());
}

pub async fn take_reports(session_id: &str) -> (Vec<String>, Vec<String>) {
    let mut lock = REPORTS.lock().await;
    let reports = lock.reports.remove(session_id).unwrap_or_default();
    let task_ids = lock.task_ids.remove(session_id).unwrap_or_default();
    (reports, task_ids)
}

pub async fn peek_reports(session_id: &str) -> (Vec<String>, Vec<String>) {
    let lock = REPORTS.lock().await;
    let reports = lock.reports.get(session_id).cloned().unwrap_or_default();
    let task_ids = lock.task_ids.get(session_id).cloned().unwrap_or_default();
    (reports, task_ids)
}

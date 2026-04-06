mod actor;
mod events;

use std::sync::{Arc, OnceLock};
use tokio::sync::mpsc;

static ENGINE_HANDLE: OnceLock<actor::EngineHandle> = OnceLock::new();

pub use actor::{EngineCommand, EngineHandle, SessionStatus};
pub use events::BgEv;

pub fn get_engine_handle() -> EngineHandle {
    ENGINE_HANDLE
        .get_or_init(|| EngineHandle::spawn(2))
        .clone()
}

pub async fn bind_session(session_id: Arc<str>, sender: mpsc::Sender<BgEv>) -> bool {
    get_engine_handle().subscribe(session_id, sender).await
}

pub fn unbind_session(session_id: Arc<str>) {
    if let Some(handle) = ENGINE_HANDLE.get() {
        handle.unsubscribe(session_id);
    }
}

pub fn dispatch_task(
    params: crate::agents::platform_extensions::better_summon::worker::SubagentRunParams,
) -> Result<(), &'static str> {
    get_engine_handle().dispatch_task(params)
}

pub async fn fetch_status(session_id: Arc<str>) -> SessionStatus {
    get_engine_handle().query_status(session_id).await
}

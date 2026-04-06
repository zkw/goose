mod actor;
mod events;

use std::sync::OnceLock;

static ENGINE_HANDLE: OnceLock<actor::EngineHandle> = OnceLock::new();

pub use actor::{EngineCommand, EngineHandle, SessionStatus};
pub use events::{BgEv, SessionId, TaskId};

pub fn get_engine_handle() -> EngineHandle {
    ENGINE_HANDLE
        .get_or_init(|| EngineHandle::spawn(2))
        .clone()
}

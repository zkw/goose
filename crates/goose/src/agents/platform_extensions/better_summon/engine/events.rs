use std::sync::Arc;

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct SessionId(pub Arc<str>);

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct TaskId(pub String);

#[derive(Clone)]
pub enum BgEv {
    Thinking {
        task_id: TaskId,
        tool_name: String,
        detail: String,
    },
    Spawned(TaskId),
    Done(TaskId, String),
    NoReport(TaskId),
}

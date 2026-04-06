#[derive(Clone)]
pub enum BgEv {
    ToolCall {
        subagent_id: String,
        tool_name: String,
        detail: String,
    },
    Spawned(String),
    Done(String, String),
    NoReport(String),
}

use rmcp::model::JsonObject;

#[derive(Clone)]
pub enum BgEv {
    ToolCall {
        subagent_id: String,
        tool_name: String,
        tool_args: Option<JsonObject>,
    },
    Spawned(String),
    Done(String, String),
    NoReport(String),
}

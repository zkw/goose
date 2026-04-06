use crate::agents::platform_extensions::better_summon::utils::MessageExt;
use crate::conversation::message::Message;
use once_cell::sync::Lazy;
use rmcp::model::Tool;

pub static REPORT_TOOL: Lazy<Tool> = Lazy::new(|| {
    Tool::new(
        "submit_task_report",
        "Submit report.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "task_report": {"type": "string"}
            },
            "required": ["task_report"]
        })
        .as_object()
        .unwrap()
        .clone(),
    )
});

pub fn extract_report(message: &Message) -> Option<String> {
    let (_, call) = message.tool_request("submit_task_report")?;
    call.arguments
        .as_ref()?
        .get("task_report")?
        .as_str()
        .map(String::from)
}

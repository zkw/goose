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

use once_cell::sync::Lazy;
use rmcp::model::Tool;

pub static DELEGATE_TOOL: Lazy<Tool> = Lazy::new(|| {
    Tool::new(
        "delegate",
        "Dispatch engineer.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "instructions": {"type": "string"}
            },
            "required": ["instructions"]
        })
        .as_object()
        .unwrap()
        .clone(),
    )
});

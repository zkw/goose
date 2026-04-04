pub const ARCHITECT_HINT: &str = include_str!("architect_hint.md");
pub const COMMON_HINT: &str = include_str!("common_hint.md");
pub const ENGINEER_HINT: &str = include_str!("engineer_hint.md");
pub const EXTENSION_NAME: &str = "better_summon";
pub const REPORT_UI: &str = include_str!("report_ui.md");
pub const REPORT_PROMPT: &str = include_str!("report_prompt.md");

use once_cell::sync::Lazy;
use rmcp::model::Tool;

pub static DELEGATE_TOOL: Lazy<Tool> = Lazy::new(|| Tool::new("delegate", "Dispatch engineer.", serde_json::json!({"type": "object", "properties": {"instructions": {"type": "string"}}, "required": ["instructions"]}).as_object().unwrap().clone()));
pub static REPORT_TOOL: Lazy<Tool> = Lazy::new(|| Tool::new("submit_task_report", "Submit report.", serde_json::json!({"type": "object", "properties": {"task_report": {"type": "string"}}, "required": ["task_report"]}).as_object().unwrap().clone()));

pub mod actor;
pub mod agent;
pub mod client;
pub mod subagent;

pub use actor::BgEv;
pub use client::BetterSummonClient;

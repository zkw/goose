pub const ARCHITECT_HINT: &str = include_str!("architect_hint.md");
pub const COMMON_HINT: &str = include_str!("common_hint.md");
pub const ENGINEER_HINT: &str = include_str!("engineer_hint.md");
const REPORT_UI: &str = include_str!("report_ui.md");
const REPORT_PROMPT: &str = include_str!("report_prompt.md");

// User-facing messages
pub const MSG_MISSING_REPORT_USER: &str = "System: A final summary report is required before ending the session. Waiting for `submit_task_report`...";
pub const MSG_MISSING_REPORT_AGENT: &str = "System: A final summary report is required before ending this session. Please call `submit_task_report` now.";
pub const MSG_REVIEW_PROCEED: &str = "System: Status update received. Proceed.";

// LLM prompt builders
pub fn render_report_ui(task_id: &str, result: &str) -> String {
    REPORT_UI
        .replace("{TASK_ID}", task_id)
        .replace("{RESULT}", result)
}

pub fn render_report_prompt(task_ids: &[String], idle: usize, reports: &[String]) -> String {
    let report_block = reports
        .iter()
        .map(|r| format!("> {}", r.replace('\n', "\n> ")))
        .collect::<Vec<_>>()
        .join("\n\n");

    REPORT_PROMPT
        .replace("{TASK_ID}", &task_ids.join(", "))
        .replace("{IDLE}", &idle.to_string())
        .replace("{RESULT}", &report_block)
}

pub fn build_subagent_hint(_session_id: &str, hint: &str) -> String {
    format!(
        "{}{}\n*作为工程师，你必须调用 submit_task_report 工具来结束当前任务。*",
        hint, COMMON_HINT
    )
}

// Thinking message formatting
pub const THINKING_WORKING: &str = "working...";
pub fn build_thinking_message(subagent_id: &str, tool_name: &str, detail: &str) -> String {
    format!("工程师[{}] {}: {}", subagent_id, tool_name, detail)
}

// Subagent defaults
pub const DEFAULT_PROMPT: &str = "Begin.";
pub const NO_TEXT_CONTENT: &str = "No text content";
pub const ERROR_STREAM_TERMINATED: &str = "System Error: Stream terminated due to {}.";

// Client messages
pub const DELEGATE_LOG_PREFIX: &str = "Delegating task:";
pub const ERROR_TOOL_NOT_FOUND: &str = "Tool {} not found";
pub const ERROR_DELEGATE: &str = "Error: {}";
pub const ERROR_EMPTY_INSTRUCTIONS: &str = "Instructions cannot be empty";
pub const ERROR_SUBAGENT_CANNOT_DELEGATE: &str = "Subagents cannot delegate";
pub const ERROR_PARENT_SESSION: &str = "Failed to retrieve parent session";
pub const ERROR_CREATE_SUBSESSION: &str = "Failed to create sub-session";
pub const MSG_REPORT_SUBMITTED: &str = "Report submitted. Current task ending.";
pub const DISPATCH_LOG: &str = "Engineer {} dispatched to background queue.";
pub const TITLE_ADHOC_TASK: &str = "Ad-hoc Task";

// Scheduler messages
pub const ERROR_NO_REPORT: &str = "No report provided.";
pub const ERROR_EXECUTION_FAILED: &str = "Execution failed: {}";
pub const ERROR_SCHEDULER_OFFLINE: &str = "scheduler offline";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_report_ui() {
        let result = render_report_ui("ENGINEER-001", "Task completed\nwith multiple lines");
        assert!(result.contains("ENGINEER-001"));
        assert!(result.contains("Task completed\nwith multiple lines"));
    }

    #[test]
    fn test_render_report_prompt() {
        let task_ids = vec!["E1".to_string(), "E2".to_string()];
        let reports = vec!["Report 1\nLine 2".to_string(), "Report 2".to_string()];
        let rendered = render_report_prompt(&task_ids, 3, &reports);

        assert!(rendered.contains("E1, E2"));
        assert!(rendered.contains("3"));
        assert!(rendered.contains("> Report 1"));
    }

    #[test]
    fn test_build_subagent_hint() {
        let hint = build_subagent_hint("test-session", "Test hint");
        assert!(hint.contains("Test hint"));
        assert!(hint.contains("submit_task_report"));
    }

    #[test]
    fn test_build_thinking_message() {
        let msg = build_thinking_message("001", "bash", "running ls");
        assert!(msg.contains("工程师[001]"));
        assert!(msg.contains("bash: running ls"));
    }
}

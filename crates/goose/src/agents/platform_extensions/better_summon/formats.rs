pub const ARCHITECT_PROMPT: &str = r#"
你是系统架构师，通过并行派发任务来完成用户交代的工作。

- 维持尽可能多的后台任务并行，不超过系统上限。
- 每次收到执行结果后，判断是否派发新任务填充空余并发位。
- 调查和收集信息要适可而止，训练在未知环境中快速开工的能力。
- 尽量让工程师在调查的同时就做些事情，不要纯调查任务。
- 不要擅自拓展做与用户布置无关的任务。
- 任务派发后无需等待，可以立刻结束回合；工程师完成后系统会自动将结果作为新消息触发你。"#;

pub const ENGINEER_PROMPT: &str = r#"
你是并发工程师之一，专注执行被分配的单一任务，你的同伴正在并行处理其他任务。

- 静态推演优先，不要全量动态测试，除非最终交付且确实不可替代。
- 遇到与其他工程师冲突或被阻塞时，立刻切换至其他不冲突的工作。
- 禁用高危命令：严禁 git restore/stash/pkill 等命令，防止回滚他人代码。"#;

pub const COMMON_PROMPT: &str = r#"
请务必遵守以下原则。

- 必须通过调用 submit_task_report 工具提交最终报告，不得直接说话结束任务。
- 报告须总结执行的操作，你的困惑，以及最终结论和下一步的建议。
- 尽量在一次回复中并行调用多个工具，降低延迟。
- 极简原则：DRY & KISS，拒绝过度设计。
- 语言双轨制：中文思考/交互，纯英文写代码。
- 除非极度隐晦，否则零注释；变量命名清晰，一行一件事。"#;

pub const MSG_MISSING_REPORT_USER: &str = "正在等待提交最终报告...";
pub const MSG_MISSING_REPORT_AGENT: &str = "结束任务需要立刻提交最终报告。";
pub const THINKING_WORKING: &str = "处理中...";
pub const NO_TEXT_CONTENT: &str = "[无内容]";
pub const DELEGATE_LOG_PREFIX: &str = "任务委派：";
pub const ERROR_EMPTY_INSTRUCTIONS: &str = "指令不能为空";
pub const ERROR_SUBAGENT_CANNOT_DELEGATE: &str = "子代理无法委派任务";
pub const ERROR_PARENT_SESSION: &str = "无法检索父会话";
pub const ERROR_CREATE_SUBSESSION: &str = "无法创建子会话";
pub const ERROR_NO_REPORT: &str = "[未提供报告]";
pub const ERROR_SCHEDULER_OFFLINE: &str = "调度器离线";

pub fn build_thinking_message(subagent_id: &str, tool_name: &str, detail: &str) -> String {
    format!("工程师 [{}] {}: {}", subagent_id, tool_name, detail)
}

pub fn render_report_ui(task_id: &str, result: &str) -> String {
    format!("▶ **工程师 {}** 已完成任务：\n\n{}", task_id, result)
}

pub fn render_report_prompt(task_ids: &[String], idle: usize, reports: &[String]) -> String {
    let report_block = reports
        .iter()
        .map(|r| format!("> {}", r.replace('\n', "\n> ")))
        .collect::<Vec<_>>()
        .join("\n\n");
    let task_list = task_ids.join(", ");
    format!(
        "以下工程师已完成任务并提交报告：{}。共有 {} 名工程师闲置。\n\n{}",
        task_list, idle, report_block
    )
}

pub fn format_dispatch_message(engineer_id: &str) -> String {
    format!("已派遣工程师 {}", engineer_id)
}

pub fn format_tool_not_found(tool_name: &str) -> String {
    format!("未找到工具 {}", tool_name)
}

pub fn format_delegate_error(error: &str) -> String {
    format!("错误：{}", error)
}

pub fn format_execution_error(error: &str) -> String {
    format!("执行失败：{}", error)
}

pub fn format_stream_terminated(reason: &str) -> String {
    format!("流式错误终止，原因是{}", reason)
}

pub fn format_hint(is_subagent: bool) -> String {
    format!(
        "{}{}",
        if is_subagent {
            ENGINEER_PROMPT
        } else {
            ARCHITECT_PROMPT
        },
        COMMON_PROMPT
    )
}

use crate::agents::extension::PlatformExtensionContext;
use crate::agents::mcp_client::{Error, McpClientTrait};
use crate::agents::subagent_handler::{run_subagent_task, SubagentRunParams};
use crate::agents::subagent_task_config::{TaskConfig, DEFAULT_SUBAGENT_MAX_TURNS};
use crate::agents::tool_execution::ToolCallContext;
use crate::config::{Config, GooseMode};
use crate::conversation::message::Message;
use crate::recipe::local_recipes::load_local_recipe_file;
use crate::recipe::Recipe;
use crate::session::extension_data::EnabledExtensionsState;
pub mod actor;
use crate::session::session_manager::SessionType;
use anyhow::Result;
use async_trait::async_trait;
use rmcp::model::{
    CallToolResult, Content, Implementation, InitializeResult, JsonObject, ListToolsResult,
    ServerCapabilities, Tool,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;
use tracing::info;

const REPORT_PROMPT: &str = include_str!("report_prompt.md");
const REPORT_UI: &str = include_str!("report_ui.md");
const ARCHITECT_HINT: &str = include_str!("architect_hint.md");
const ENGINEER_HINT: &str = include_str!("engineer_hint.md");
const COMMON_HINT: &str = include_str!("common_hint.md");

pub static EXTENSION_NAME: &str = "better_summon";

fn session_short_id(session_id: &str) -> String {
    session_id
        .chars()
        .filter(|c| c.is_alphanumeric())
        .take(8)
        .collect()
}

pub fn main_agent_final_output_response() -> crate::recipe::Response {
    crate::recipe::Response {
        json_schema: Some(serde_json::json!({
            "type": "object",
            "properties": {
                "final_report": {
                    "type": "string",
                    "description": "最终汇报给用户的完整执行报告。"
                }
            },
            "required": ["final_report"]
        })),
    }
}
pub struct BetterSummonClient {
    context: PlatformExtensionContext,
    info: InitializeResult,
    task_registry: Arc<Mutex<HashMap<String, String>>>,
    session_to_id: Arc<Mutex<HashMap<String, String>>>,
    task_semaphore: Arc<Semaphore>,
    session_cancel_token: CancellationToken,
}

impl Drop for BetterSummonClient {
    fn drop(&mut self) {
        self.session_cancel_token.cancel();
    }
}

impl BetterSummonClient {
    pub fn new(context: PlatformExtensionContext) -> Result<Self> {
        let max_tasks = Config::global()
            .get_param::<usize>("GOOSE_MAX_BACKGROUND_TASKS")
            .unwrap_or(50);
        let info = InitializeResult::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new(
                "goose-better-summon",
                env!("CARGO_PKG_VERSION"),
            ));

        Ok(Self {
            context,
            info,
            task_registry: Arc::new(Mutex::new(HashMap::new())),
            session_to_id: Arc::new(Mutex::new(HashMap::new())),
            task_semaphore: Arc::new(Semaphore::new(max_tasks)),
            session_cancel_token: CancellationToken::new(),
        })
    }

    fn register_agent(&self, short_id: &str, session_id: &str) {
        self.task_registry
            .lock()
            .unwrap()
            .insert(short_id.to_string(), session_id.to_string());
        self.session_to_id
            .lock()
            .unwrap()
            .insert(session_id.to_string(), short_id.to_string());
    }

    fn resolve_agent(&self, short_id: &str) -> Option<String> {
        self.task_registry.lock().unwrap().get(short_id).cloned()
    }

    fn create_delegate_tool(&self) -> Tool {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "instructions": {
                    "type": "string",
                    "description": "具体任务指令"
                },
                "expected_turns": {
                    "type": "integer",
                    "description": "预期该工程师需要多少回合完成任务。这是项目管理评估，和工作量挂钩：简单任务 50~200，中等 200~600，复杂 600+。工程师将在 25%/50%/75%/100%/125%... 里程碑节点收到系统通知并被要求主动汇报进度。"
                }
            },
            "required": ["instructions", "expected_turns"]
        });

        Tool::new(
            "delegate",
            "派遣工程师到后台执行任务，结果自动展示。",
            schema.as_object().unwrap().clone(),
        )
    }

    fn create_send_message_tool(&self) -> Tool {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "目标工程师的 8 位 ID（delegate 启动时返回的 ID，或系统注入的架构师 ID）"
                },
                "message": {
                    "type": "string",
                    "description": "要发送的消息内容"
                }
            },
            "required": ["agent_id", "message"]
        });

        Tool::new(
            "send_message",
            "向任意运行中的工程师或架构师发送实时消息。可向工程师传达最新指示，也可向架构师主动汇报进度。目标会在下次工具调用或中断点立即收到。",
            schema.as_object().unwrap().clone(),
        )
    }

    async fn handle_delegate(
        &self,
        session_id: &str,
        instructions: &str,
        expected_turns: u32,
    ) -> Result<CallToolResult, String> {
        if instructions.is_empty() {
            return Err("指令不能为空".to_string());
        }
        info!(
            "派遣工程师任务: {}, 预期回合: {}",
            instructions.split('\n').next().unwrap_or(""),
            expected_turns
        );

        let parent_session = self
            .context
            .session_manager
            .get_session(session_id, false)
            .await
            .map_err(|e| format!("无法获取父会话: {}", e))?;

        if parent_session.session_type == SessionType::SubAgent {
            return Err("工程师无法再次派发任务".to_string());
        }

        let parent_short_id = session_short_id(session_id);
        // Only register the parent's short_id → session_id mapping (for engineers to find the architect)
        self.task_registry
            .lock()
            .unwrap()
            .insert(parent_short_id.clone(), session_id.to_string());

        let permit = match self.task_semaphore.clone().try_acquire_owned() {
            Ok(permit) => permit,
            Err(_) => return Err("已达到工程师并发上限".to_string()),
        };

        let task_id = uuid::Uuid::new_v4().simple().to_string()[..8].to_string();

        let working_dir = parent_session.working_dir.clone();
        let recipe = self.build_recipe(instructions);
        let provider = self.resolve_provider(&recipe, &parent_session).await?;

        let extensions = if let Some(recipe_extensions) = &recipe.extensions {
            recipe_extensions.clone()
        } else {
            EnabledExtensionsState::extensions_or_default(
                Some(&parent_session.extension_data),
                Config::global(),
            )
        };

        let task_config = TaskConfig::new(provider, session_id, &working_dir, extensions)
            .with_max_turns(Some(expected_turns as usize))
            .with_subagent_id(task_id.clone());

        let sub_session = self
            .context
            .session_manager
            .create_session(
                working_dir,
                format!("后台-{}", &task_id),
                SessionType::SubAgent,
                GooseMode::Auto,
            )
            .await
            .map_err(|e| format!("无法创建会话: {}", e))?;

        let agent_config = crate::agents::AgentConfig::new(
            self.context.session_manager.clone(),
            crate::config::permission::PermissionManager::instance(),
            None,
            GooseMode::Auto,
            true,
            crate::agents::GoosePlatform::GooseCli,
        );

        // Register the sub-session so send_message can reach it
        self.register_agent(&task_id, &sub_session.id);

        // inbox_guard keeps the sub-session's channel alive so deliver_message
        // (from send_message tool) can reach the sub-agent via
        // try_wait_for_background_task inside reply_internal.
        // Using add_inbox (not add_background_task) so is_door_held stays false
        // for the sub-agent's own session — otherwise reply_internal deadlocks
        // by waiting for itself after every LLM turn.
        let inbox_guard = actor::InboxGuard::new(sub_session.id.clone());

        // Guard for the parent session so the main agent waits for this task
        let guard = actor::TaskGuard::new(session_id.to_string());

        let main_session_id = session_id.to_string();
        let sub_session_id = sub_session.id.clone();
        let task_id_bg = task_id.clone();
        let task_semaphore = self.task_semaphore.clone();

        let mut recipe_with_context = recipe;
        let context_note = format!(
            "\n\n[系统] 你的任务 ID: {}，架构师 ID: {}，预期回合数: {}。可用 send_message 工具向任意 ID 发消息。",
            task_id, parent_short_id, expected_turns
        );
        recipe_with_context.instructions =
            Some(recipe_with_context.instructions.unwrap_or_default() + context_note.as_str());

        let (notif_tx, mut notif_rx) = tokio::sync::mpsc::unbounded_channel();
        let session_for_notifs = main_session_id.clone();
        tokio::spawn(async move {
            while let Some(notif) = notif_rx.recv().await {
                actor::deliver_event(
                    &session_for_notifs,
                    actor::BackgroundEvent::McpNotification(notif),
                );
            }
        });

        let task_registry = self.task_registry.clone();
        let session_to_id = self.session_to_id.clone();

        struct AgentRegistryGuard {
            task_id: String,
            session_id: String,
            task_registry: Arc<Mutex<HashMap<String, String>>>,
            session_to_id: Arc<Mutex<HashMap<String, String>>>,
        }
        impl Drop for AgentRegistryGuard {
            fn drop(&mut self) {
                self.task_registry.lock().unwrap().remove(&self.task_id);
                self.session_to_id.lock().unwrap().remove(&self.session_id);
            }
        }

        let session_cancel_token = self.session_cancel_token.clone();
        tokio::spawn(async move {
            let _registry_guard = AgentRegistryGuard {
                task_id: task_id_bg.clone(),
                session_id: sub_session_id.clone(),
                task_registry: task_registry.clone(),
                session_to_id: session_to_id.clone(),
            };
            let _permit = permit;
            let _guard = guard;
            let _inbox_guard = inbox_guard;

            info!("工程师任务 {} 开始执行", task_id_bg);

            let result = run_subagent_task(SubagentRunParams {
                config: agent_config,
                recipe: recipe_with_context,
                task_config,
                return_last_only: true,
                session_id: sub_session_id.clone(),
                cancellation_token: Some(session_cancel_token.child_token()),
                on_message: None,
                notification_tx: Some(notif_tx),
            })
            .await;

            let idle = task_semaphore.available_permits();

            let quoted = match result {
                Ok(text) => {
                    if text.is_empty() {
                        "未提供最终文本输出。".to_string()
                    } else {
                        serde_json::from_str::<serde_json::Value>(&text)
                            .ok()
                            .and_then(|json| {
                                json.get("final_report")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string())
                            })
                            .unwrap_or(text)
                    }
                }
                Err(e) => format!("执行失败: {}", e),
            }
            .lines()
            .map(|line| {
                if line.is_empty() {
                    ">".to_string()
                } else {
                    format!("> {}", line)
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

            let display = REPORT_UI
                .replace("{TASK_ID}", &task_id_bg)
                .replace("{RESULT}", &quoted);
            let assistant_log_msg = Message::assistant()
                .with_text(display)
                .with_generated_id()
                .user_only();

            let trigger_msg = Message::user()
                .with_text(
                    REPORT_PROMPT
                        .replace("{TASK_ID}", &task_id_bg)
                        .replace("{IDLE}", &idle.to_string())
                        .replace("{RESULT}", &quoted),
                )
                .with_generated_id()
                .agent_only();

            actor::deliver_event(
                &main_session_id,
                actor::BackgroundEvent::Message(assistant_log_msg),
            );
            actor::deliver_event(
                &main_session_id,
                actor::BackgroundEvent::Message(trigger_msg),
            );
            info!("工程师任务 {} 执行完毕并已汇报", task_id_bg);
        });

        let idle = self.task_semaphore.available_permits();
        Ok(CallToolResult::success(vec![Content::text(format!(
            "工程师 {} 已启动，还有 {} 名工程师闲置。",
            task_id, idle
        ))]))
    }

    fn build_recipe(&self, user_task: &str) -> Recipe {
        let default_yaml = format!(
            r#"
version: "1.0.0"
title: "工程师"
description: "后台任务"
settings:
  max_turns: {}
"#,
            DEFAULT_SUBAGENT_MAX_TURNS
        );

        let mut recipe = load_local_recipe_file("developer")
            .ok()
            .and_then(|file| Recipe::from_content(&file.content).ok())
            .unwrap_or_else(|| Recipe::from_content(&default_yaml).unwrap());

        recipe.prompt = Some(user_task.to_string());

        if recipe.response.is_none() {
            recipe.response = Some(crate::recipe::Response {
                json_schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "final_report": {
                            "type": "string",
                            "description": "最终发给架构师的执行报告。请务必详细包含执行的过程、查看的源码或者验证的结果。"
                        }
                    },
                    "required": ["final_report"]
                })),
            });
        }

        recipe
    }

    async fn resolve_provider(
        &self,
        recipe: &Recipe,
        session: &crate::session::Session,
    ) -> Result<Arc<dyn crate::providers::base::Provider>, String> {
        let settings = recipe.settings.as_ref();

        let provider_name = settings
            .and_then(|s| s.goose_provider.clone())
            .or_else(|| {
                Config::global()
                    .get_param::<String>("GOOSE_SUBAGENT_PROVIDER")
                    .ok()
            })
            .or_else(|| session.provider_name.clone())
            .ok_or_else(|| "未配置 Provider".to_string())?;

        let mut model_config = match &session.model_config {
            Some(cfg) => cfg.clone(),
            None => crate::model::ModelConfig::new("default")
                .map(|c| c.with_canonical_limits(&provider_name))
                .map_err(|e| format!("无法创建模型配置: {}", e))?,
        };

        if let Some(model) = settings.and_then(|s| s.goose_model.clone()).or_else(|| {
            Config::global()
                .get_param::<String>("GOOSE_SUBAGENT_MODEL")
                .ok()
        }) {
            model_config.model_name = model;
        }

        if let Some(temp) = settings.and_then(|s| s.temperature) {
            model_config = model_config.with_temperature(Some(temp));
        }

        crate::providers::create(&provider_name, model_config, Vec::new())
            .await
            .map_err(|e| format!("无法创建 Provider: {}", e))
    }

    async fn is_subagent(&self, session_id: &str) -> bool {
        self.context
            .session_manager
            .get_session(session_id, false)
            .await
            .map(|s| s.session_type == SessionType::SubAgent)
            .unwrap_or(false)
    }
}

#[async_trait]
impl McpClientTrait for BetterSummonClient {
    async fn list_tools(
        &self,
        session_id: &str,
        _next_cursor: Option<String>,
        _cancel_token: CancellationToken,
    ) -> Result<ListToolsResult, Error> {
        let is_subagent = self.is_subagent(session_id).await;

        let mut tools = vec![self.create_send_message_tool()];

        if !is_subagent {
            tools.push(self.create_delegate_tool());
        }

        Ok(ListToolsResult {
            tools,
            next_cursor: None,
            meta: None,
        })
    }

    async fn call_tool(
        &self,
        ctx: &ToolCallContext,
        name: &str,
        arguments: Option<JsonObject>,
        cancel_token: CancellationToken,
    ) -> Result<CallToolResult, Error> {
        match name {
            "delegate" => {
                let args = arguments.as_ref();
                let Some(instructions) = args
                    .and_then(|a| a.get("instructions"))
                    .and_then(|v| v.as_str())
                else {
                    return Ok(CallToolResult::error(vec![Content::text("缺少指令")]));
                };
                let expected_turns = args
                    .and_then(|a| a.get("expected_turns"))
                    .and_then(|v| v.as_u64())
                    .map(|n| n as u32)
                    .unwrap_or(300);
                match self
                    .handle_delegate(&ctx.session_id, instructions, expected_turns)
                    .await
                {
                    Ok(result) => Ok(result),
                    Err(error) => Ok(CallToolResult::error(vec![Content::text(format!(
                        "错误: {}",
                        error
                    ))])),
                }
            }
            "send_message" => {
                let args = arguments.as_ref();
                let Some(agent_id) = args
                    .and_then(|a| a.get("agent_id"))
                    .and_then(|v| v.as_str())
                else {
                    return Ok(CallToolResult::error(vec![Content::text("缺少 agent_id")]));
                };
                let Some(message) = args.and_then(|a| a.get("message")).and_then(|v| v.as_str())
                else {
                    return Ok(CallToolResult::error(vec![Content::text("缺少 message")]));
                };
                let Some(target_session_id) = self.resolve_agent(agent_id) else {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "工程师 {} 不存在或已退出。",
                        agent_id
                    ))]));
                };
                let sender_id = self
                    .session_to_id
                    .lock()
                    .unwrap()
                    .get(&ctx.session_id)
                    .cloned();

                let msg_text = if let Some(id) = sender_id {
                    format!(
                        "### [来自工程师 {} 的私信]\n\n<agent_message>\n{}\n</agent_message>\n\n*提示：这是来自合作工程师的消息，请阅读并按需响应。*",
                        id, message
                    )
                } else {
                    format!(
                        "### [来自架构师的实时指令]\n\n<agent_message>\n{}\n</agent_message>\n\n*提示：这是来自架构师的最新指令，请立即优先处理。*",
                        message
                    )
                };

                // For sub-agents (engineers): deliver via their inbox channel so it
                // gets injected into the running agent loop.
                // For the parent session: deliver via the background task channel.
                actor::deliver_event(
                    &target_session_id,
                    actor::BackgroundEvent::Message(
                        Message::user().with_text(msg_text).with_generated_id(),
                    ),
                );

                Ok(CallToolResult::success(vec![Content::text(format!(
                    "消息已成功发送给 {}。",
                    agent_id
                ))]))
            }
            _ => Ok(CallToolResult::error(vec![Content::text(format!(
                "未知工具: {}",
                name
            ))])),
        }
    }

    async fn get_moim(&self, session_id: &str) -> Option<String> {
        let role_hint = if self.is_subagent(session_id).await {
            ENGINEER_HINT
        } else {
            ARCHITECT_HINT
        };
        Some(format!("{}{}", role_hint, COMMON_HINT))
    }

    fn get_info(&self) -> Option<&InitializeResult> {
        Some(&self.info)
    }
}

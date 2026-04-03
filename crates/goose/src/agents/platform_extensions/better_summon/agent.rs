use crate::agents::types::SessionConfig;
use crate::conversation::Conversation;
use crate::session::Session;
use anyhow::{Result, anyhow};
use futures::stream::{self, BoxStream, StreamExt};
use futures::TryStreamExt;
use tokio_util::sync::CancellationToken;
use tracing::{warn, instrument, error};
use tracing_futures::Instrument;
use crate::agents::platform_extensions::better_summon::actor;
use crate::conversation::message::{
    Message, MessageContent, SystemNotificationType, ProviderMetadata as MessageProviderMetadata,
};
pub use rmcp::model::Content;
use crate::config::{Config, GooseMode};
use std::collections::HashMap;
use uuid::Uuid;
use crate::utils::is_token_cancelled;
use crate::session::session_manager::SessionManager;
use crate::agents::platform_extensions::MANAGE_EXTENSIONS_TOOL_NAME_COMPLETE;
use crate::agents::agent::{Agent, AgentEvent, DEFAULT_MAX_TURNS, ToolStreamItem};
use crate::agents::tool_execution::CHAT_MODE_TOOL_SKIPPED_RESPONSE;
use rmcp::model::{Role, CallToolResult};
use crate::providers::errors::ProviderError;
use crate::permission::permission_judge::PermissionCheckResult;

pub struct BetterAgent {
    pub core: Agent,
}

impl BetterAgent {
    pub fn new(core: Agent) -> Self {
        Self { core }
    }

    #[instrument(
        skip(self, user_message, session_config, cancel_token),
        fields(user_message, trace_input, session.id = %session_config.id)
    )]
    pub async fn reply(
        self,
        user_message: Message,
        session_config: SessionConfig,
        cancel_token: Option<CancellationToken>,
    ) -> Result<BoxStream<'static, Result<AgentEvent>>> {
        let session_manager = self.core.config.session_manager.clone();
        let message_text = user_message.as_concat_text();

        let command_result = self.core.execute_command(&message_text, &session_config.id).await;

        match command_result {
            Err(e) => {
                let error_message = Message::assistant()
                    .with_text(e.to_string())
                    .with_visibility(true, false);
                return Ok(Box::pin(stream::once(async move {
                    Ok(AgentEvent::Message(error_message))
                })));
            }
            Ok(Some(response)) if response.role == Role::Assistant => {
                session_manager.add_message(&session_config.id, &user_message.clone().with_visibility(true, false)).await?;
                session_manager.add_message(&session_config.id, &response.clone().with_visibility(true, false)).await?;
                return Ok(Box::pin(async_stream::try_stream! {
                    yield AgentEvent::Message(user_message);
                    yield AgentEvent::Message(response);
                }));
            }
            Ok(Some(resolved_message)) => {
                session_manager.add_message(&session_config.id, &user_message.clone().with_visibility(true, false)).await?;
                session_manager.add_message(&session_config.id, &resolved_message.clone().with_visibility(false, true)).await?;
            }
            Ok(None) => {
                session_manager.add_message(&session_config.id, &user_message).await?;
            }
        }

        let session = session_manager.get_session(&session_config.id, true).await?;
        let conversation = session.conversation.clone().ok_or_else(|| anyhow!("Session has no conversation"))?;

        let stream = self.reply_internal(conversation, session_config, session, cancel_token).await?;
        Ok(Box::pin(stream))
    }

    pub async fn reply_internal(
        self,
        conversation: Conversation,
        session_config: SessionConfig,
        session: Session,
        cancel_token: Option<CancellationToken>,
    ) -> Result<BoxStream<'static, Result<AgentEvent>>> {
        let context = self.core.prepare_reply_context(&session.id, conversation, session.working_dir.as_path()).await?;
        let crate::agents::agent::ReplyContext {
            mut conversation,
            mut tools,
            mut toolshim_tools,
            mut system_prompt,
            tool_call_cut_off,
            goose_mode,
            initial_messages,
        } = context;

        self.core.reset_retry_attempts().await;

        let provider = self.core.provider().await?;
        let session_manager = self.core.config.session_manager.clone();
        let session_id = session_config.id.clone();
        let mut bg_rx = actor::subscribe(&session_id);

        if !self.core.config.disable_session_naming {
            let manager_for_spawn = session_manager.clone();
            let session_id_for_naming = session_id.clone();
            let provider_for_naming = provider.clone();
            tokio::spawn(async move {
                if let Err(e) = manager_for_spawn
                    .maybe_update_name(&session_id_for_naming, provider_for_naming)
                    .await
                {
                    warn!("Failed to generate session description: {}", e);
                }
            });
        }

        let pre_turn_tool_count = conversation
            .messages()
            .iter()
            .flat_map(|m| m.content.iter())
            .filter(|c| matches!(c, MessageContent::ToolRequest(_)))
            .count();

        let working_dir = session.working_dir.clone();
        let reply_stream_span = tracing::info_span!("better_summon_reply_stream", session.id = %session_config.id);

        let inner: BoxStream<'_, Result<AgentEvent>> = Box::pin(async_stream::try_stream! {
            let mut _last_assistant_text = String::new();
            let mut compaction_attempts = 0;

            macro_rules! try_yield_bg_events {
                ($ev:expr, $got_msg:expr) => {
                    let (yield_msg, visible) = self.core.handle_background_event(
                        $ev, &session_id, &session_manager, &mut conversation
                    ).await;
                    if visible { $got_msg = true; }
                    if let Some(e) = yield_msg {
                        yield e;
                    }
                };
            }

            let mut transient_retry_count = 0u32;
            let mut turns_taken = 0u32;
            let max_turns = session_config.max_turns.unwrap_or_else(|| {
                Config::global()
                    .get_param::<u32>("GOOSE_MAX_TURNS")
                    .unwrap_or(DEFAULT_MAX_TURNS)
            });
            let mut reached_max_turns = false;

            loop {
                let mut got_agent_message = false;
                if is_token_cancelled(&cancel_token) { break; }

                let mut exit_chat = false;
                {
                    let mut guard = self.core.final_output_tool.lock().await;
                    if let Some(output) = guard.as_mut().and_then(|fot| fot.final_output.take()) {
                        yield AgentEvent::Message(Message::assistant().with_text(output));
                        exit_chat = true;
                    }
                }

                if !exit_chat {
                    if reached_max_turns {
                        exit_chat = true;
                    } else {
                        turns_taken += 1;
                        if turns_taken > max_turns {
                            reached_max_turns = true;
                            yield AgentEvent::Message(
                                Message::assistant().with_text(
                                    "I've reached the maximum number of actions I can do without user input. Would you like me to continue?"
                                )
                            );
                            exit_chat = true;
                        }
                    }
                }

                if !exit_chat {
                    {
                        let mut any_pending = false;
                        while let Ok(ev) = bg_rx.try_recv() {
                            try_yield_bg_events!(ev, any_pending);
                        }
                        if any_pending { got_agent_message = true; }
                    }

                    let provider = self.core.provider().await?;
                    let conversation_with_moim = crate::agents::moim::inject_moim(
                        &session_id,
                        conversation.clone(),
                        &self.core.extension_manager,
                        &working_dir,
                    ).await;

                    let mut stream = {
                        let mut stream_fut = std::pin::pin!(Agent::stream_response_from_provider(
                            provider,
                            &session_id,
                            &system_prompt,
                            conversation_with_moim.messages(),
                            &tools,
                            &toolshim_tools,
                        ));
                        let mut event_queue_active = true;
                        loop {
                            tokio::select! {
                                stream_res = &mut stream_fut => {
                                    break stream_res;
                                }
                                ev_res = bg_rx.recv(), if event_queue_active => {
                                    match ev_res {
                                        Some(ev) => { 
                                            try_yield_bg_events!(ev, got_agent_message); 
                                        }
                                        None => event_queue_active = false,
                                    }
                                }
                            };
                        }
                    }?;

                    let current_turn_tool_count = conversation.messages().iter()
                        .flat_map(|m| m.content.iter())
                        .filter(|c| matches!(c, MessageContent::ToolRequest(_)))
                        .count()
                        .saturating_sub(pre_turn_tool_count);

                    let tool_pair_summarization_task = crate::context_mgmt::maybe_summarize_tool_pairs(
                        self.core.provider().await?,
                        session_id.clone(),
                        conversation.clone(),
                        tool_call_cut_off,
                        current_turn_tool_count,
                    );

                    let mut no_tools_called = true;
                    let mut messages_to_add = Conversation::default();
                    let mut tools_updated = false;
                    let mut did_recovery_compact_this_iteration = false;
                    let mut did_transient_retry_this_iteration = false;
                    let mut surfaced_thinking_in_turn = false;

                    loop {
                        let next = match stream.next().await {
                            Some(n) => n,
                            None => break,
                        };

                        if is_token_cancelled(&cancel_token) { break; }
                        
                        if exit_chat && !got_agent_message {
                            if !actor::is_door_held(&session_id) { break; }
                        }

                        match next {
                            Ok((response, usage)) => {
                                compaction_attempts = 0;
                                if let Some(ref usage) = usage {
                                    self.core.update_session_metrics(&session_id, session_config.schedule_id.clone(), usage, false).await?;
                                }

                                if let Some(response) = response {
                                    let (frontend_requests, remaining_requests, filtered_response) = self.core
                                        .categorize_tool_requests(
                                            &response,
                                            &tools,
                                            surfaced_thinking_in_turn,
                                        )
                                        .await;

                                    surfaced_thinking_in_turn |= filtered_response.content.iter().any(
                                        |content| {
                                            matches!(
                                                content,
                                                MessageContent::Thinking(_)
                                                    | MessageContent::RedactedThinking(_)
                                            )
                                        },
                                    );

                                    yield AgentEvent::Message(filtered_response.clone());
                                    tokio::task::yield_now().await;

                                    let num_tool_requests = frontend_requests.len() + remaining_requests.len();
                                    if num_tool_requests == 0 {
                                        let text = filtered_response.as_concat_text();
                                        if !text.is_empty() {
                                            _last_assistant_text = text;
                                        }
                                        messages_to_add.push(response);
                                        continue;
                                    }

                                    let mut request_to_response_map = HashMap::new();
                                    let mut request_metadata: HashMap<String, Option<MessageProviderMetadata>> = HashMap::new();
                                    for request in frontend_requests.iter().chain(remaining_requests.iter()) {
                                        request_to_response_map.insert(request.id.clone(), Message::user().with_generated_id());
                                        request_metadata.insert(request.id.clone(), request.metadata.clone());
                                    }

                                    for request in frontend_requests.iter() {
                                        let response_msg = request_to_response_map.get_mut(&request.id)
                                            .ok_or_else(|| anyhow!("missing response entry for request {}", request.id))?;
                                        let mut frontend_tool_stream = self.core.handle_frontend_tool_request(
                                            request,
                                            response_msg,
                                        );

                                        while let Some(msg) = frontend_tool_stream.try_next().await? {
                                            yield AgentEvent::Message(msg);
                                        }
                                    }

                                    if goose_mode == GooseMode::Chat {
                                        for request in remaining_requests.iter() {
                                            if let Some(response) = request_to_response_map.get_mut(&request.id) {
                                                response.add_tool_response_with_metadata(
                                                    request.id.clone(),
                                                    Ok(CallToolResult::success(vec![Content::text(CHAT_MODE_TOOL_SKIPPED_RESPONSE)])),
                                                    request.metadata.as_ref(),
                                                );
                                            }
                                        }
                                    } else {
                                        let inspection_results = self.core.tool_inspection_manager
                                            .inspect_tools(
                                                &session_id,
                                                &remaining_requests,
                                                conversation.messages(),
                                                goose_mode,
                                            )
                                            .await?;

                                        let permission_check_result = self.core.tool_inspection_manager
                                            .process_inspection_results_with_permission_inspector(
                                                &remaining_requests,
                                                &inspection_results,
                                            )
                                            .unwrap_or_else(|| {
                                                let mut result = PermissionCheckResult {
                                                    approved: vec![],
                                                    needs_approval: vec![],
                                                    denied: vec![],
                                                };
                                                result.needs_approval.extend(remaining_requests.iter().cloned());
                                                result
                                            });

                                        let mut enable_extension_request_ids = vec![];
                                        for request in &remaining_requests {
                                            if let Ok(tool_call) = &request.tool_call {
                                                if tool_call.name == MANAGE_EXTENSIONS_TOOL_NAME_COMPLETE {
                                                    enable_extension_request_ids.push(request.id.clone());
                                                }
                                            }
                                        }

                                        let mut tool_futures = self.core.handle_approved_and_denied_tools(
                                            &permission_check_result,
                                            &mut request_to_response_map,
                                            cancel_token.clone(),
                                            &session,
                                        ).await?;

                                        {
                                            let mut tool_approval_stream = self.core.handle_approval_tool_requests(
                                                &permission_check_result.needs_approval,
                                                &mut tool_futures,
                                                &mut request_to_response_map,
                                                cancel_token.clone(),
                                                &session,
                                                &inspection_results,
                                            );

                                            while let Some(msg) = tool_approval_stream.try_next().await? {
                                                yield AgentEvent::Message(msg);
                                            }
                                        }

                                        let with_id = tool_futures
                                            .into_iter()
                                            .map(|(request_id, stream)| {
                                                stream.map(move |item| (request_id.clone(), item))
                                            })
                                            .collect::<Vec<_>>();

                                        let mut combined = stream::select_all(with_id);
                                        let mut all_install_successful = true;

                                        loop {
                                            if is_token_cancelled(&cancel_token) { break; }

                                            for msg in self.core.drain_elicitation_messages(&session_id).await {
                                                yield AgentEvent::Message(msg);
                                            }

                                            tokio::select! {
                                                biased;
                                                tool_item = combined.next() => {
                                                    match tool_item {
                                                        Some((request_id, item)) => {
                                                            match item {
                                                                ToolStreamItem::Result(output) => {
                                                                    if enable_extension_request_ids.contains(&request_id) && output.is_err() {
                                                                        all_install_successful = false;
                                                                    }
                                                                    if let Some(response) = request_to_response_map.get_mut(&request_id) {
                                                                        let metadata: Option<&MessageProviderMetadata> = request_metadata.get(&request_id).and_then(|m| m.as_ref());
                                                                        response.add_tool_response_with_metadata(request_id, output, metadata);
                                                                    }
                                                                }
                                                                ToolStreamItem::Message(msg) => {
                                                                    yield AgentEvent::McpNotification((request_id, msg));
                                                                }
                                                            }
                                                        }
                                                        None => break,
                                                    }
                                                }
                                                _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {}
                                            }
                                        }

                                        for msg in self.core.drain_elicitation_messages(&session_id).await {
                                            yield AgentEvent::Message(msg);
                                        }

                                        let mut any_visible = false;
                                        while let Ok(ev) = bg_rx.try_recv() {
                                            try_yield_bg_events!(ev, any_visible);
                                        }
                                        if any_visible { got_agent_message = true; }

                                        if all_install_successful && !enable_extension_request_ids.is_empty() {
                                            if let Err(e) = self.core.save_extension_state(&session_config).await {
                                                warn!("Failed to save extension state: {}", e);
                                            }
                                            tools_updated = true;
                                        }
                                    }

                                    let thinking_content: Vec<MessageContent> = response.content.iter()
                                        .filter(|c| matches!(c, MessageContent::Thinking(_)))
                                        .cloned()
                                        .collect();
                                    
                                    for request in frontend_requests.iter().chain(remaining_requests.iter()) {
                                        if request.tool_call.is_ok() {
                                            let mut request_msg = Message::assistant().with_id(format!("msg_{}", Uuid::new_v4()));
                                            for rc in &thinking_content { request_msg = request_msg.with_content(rc.clone()); }
                                            request_msg = request_msg.with_tool_request_with_metadata(
                                                    request.id.clone(),
                                                    request.tool_call.clone(),
                                                    request.metadata.as_ref(),
                                                    request.tool_meta.clone(),
                                                );
                                            messages_to_add.push(request_msg);
                                            let final_response = request_to_response_map.remove(&request.id).unwrap_or_else(|| Message::user().with_generated_id());
                                            yield AgentEvent::Message(final_response.clone());
                                            messages_to_add.push(final_response);
                                        }
                                    }
                                    no_tools_called = false;
                                }
                            }
                            Err(ref _provider_err @ ProviderError::ContextLengthExceeded(_)) => {
                                compaction_attempts += 1;
                                if compaction_attempts >= 2 {
                                    yield AgentEvent::Message(Message::assistant().with_system_notification(SystemNotificationType::InlineMessage, "Unable to continue: Context limit still exceeded after compaction."));
                                    exit_chat = true; break;
                                }
                                yield AgentEvent::Message(Message::assistant().with_system_notification(SystemNotificationType::InlineMessage, "Context limit reached. Compacting..."));
                                match crate::context_mgmt::compact_messages(self.core.provider().await?.as_ref(), &session_id, &conversation, false).await {
                                    Ok((compacted_conversation, usage)) => {
                                        session_manager.replace_conversation(&session_id, &compacted_conversation).await?;
                                        self.core.update_session_metrics(&session_id, session_config.schedule_id.clone(), &usage, true).await?;
                                        conversation = compacted_conversation;
                                        did_recovery_compact_this_iteration = true;
                                        yield AgentEvent::HistoryReplaced(conversation.clone());
                                        break;
                                    }
                                    Err(e) => { error!("Compaction failed: {}", e); exit_chat = true; break; }
                                }
                            }
                            Err(ref provider_err) => {
                                if provider_err.is_retryable_stream_error() {
                                    transient_retry_count += 1;
                                    if transient_retry_count <= 3 {
                                        yield AgentEvent::Message(Message::assistant().with_system_notification(SystemNotificationType::InlineMessage, format!("遇到流错误，正在重试... (第 {} 次)", transient_retry_count)));
                                        did_transient_retry_this_iteration = true;
                                        turns_taken = turns_taken.saturating_sub(1);
                                        messages_to_add.clear();
                                        break;
                                    }
                                }
                                yield AgentEvent::Message(Message::assistant().with_text(format!("Ran into this error: {provider_err}.\nPlease retry.")));
                                exit_chat = true; break;
                            }
                        }
                    }

                    if !did_transient_retry_this_iteration { transient_retry_count = 0; }
                    if tools_updated || self.core.prompt_manager.lock().await.load_subdirectory_hints(&working_dir) {
                        let updated = self.core.prepare_tools_and_prompt(&session_id, &session.working_dir).await?;
                        tools = updated.0; toolshim_tools = updated.1; system_prompt = updated.2;
                    }

                    if no_tools_called {
                        let final_output = { let guard = self.core.final_output_tool.lock().await; guard.as_ref().map(|fot| fot.final_output.clone()) };
                        match final_output {
                            Some(Some(output)) => {
                                let message = Message::assistant().with_text(output);
                                messages_to_add.push(message.clone());
                                yield AgentEvent::Message(message);
                                exit_chat = true;
                            }
                            None if did_recovery_compact_this_iteration || did_transient_retry_this_iteration => {}
                            _ => {
                                match self.core.handle_retry_logic(&mut conversation, &session_config, &initial_messages).await {
                                    Ok(true) => {
                                        messages_to_add = Conversation::default();
                                        session_manager.replace_conversation(&session_id, &conversation).await?;
                                        yield AgentEvent::HistoryReplaced(conversation.clone());
                                    }
                                    Ok(false) => { exit_chat = true; }
                                    Err(e) => { error!("Retry logic failed: {}", e); exit_chat = true; }
                                }
                            }
                            _ => { exit_chat = true; }
                        }
                    }

                    if is_token_cancelled(&cancel_token) { tool_pair_summarization_task.abort(); }
                    if let Ok(summaries) = tool_pair_summarization_task.await {
                        let mut updated_messages = conversation.messages().clone();
                        for (summary_msg, tool_id) in summaries {
                            let matching: Vec<&mut Message> = updated_messages.iter_mut().filter(|msg| {
                                msg.id.is_some() && msg.content.iter().any(|c| match c {
                                    MessageContent::ToolRequest(req) => req.id == tool_id,
                                    MessageContent::ToolResponse(resp) => resp.id == tool_id,
                                    _ => false,
                                })
                            }).collect();
                            if matching.len() == 2 {
                                for msg in matching {
                                    let id = msg.id.as_ref().unwrap();
                                    msg.metadata = msg.metadata.with_agent_invisible();
                                    SessionManager::update_message_metadata(&session_id, id, |metadata| metadata.with_agent_invisible()).await?;
                                }
                                messages_to_add.push(summary_msg);
                            }
                        }
                        conversation = Conversation::new_unvalidated(updated_messages);
                    }

                    if !exit_chat {
                        for msg in &messages_to_add { session_manager.add_message(&session_id, msg).await?; }
                        conversation.extend(messages_to_add);
                    }
                }

                if exit_chat {
                    let mut any_agent_visible = false;
                    let mut task_watcher = actor::get_task_watcher(&session_id);
                    if let Some(mut w) = task_watcher {
                        if actor::is_door_held(&session_id) {
                            yield AgentEvent::Message(Message::assistant().with_system_notification(SystemNotificationType::ThinkingMessage, "Waiting for background tasks..."));
                        }
                        let mut event_queue_active = true;
                        loop {
                            if !actor::is_door_held(&session_id) {
                                while let Ok(ev) = bg_rx.try_recv() {
                                    try_yield_bg_events!(ev, any_agent_visible);
                                }
                                break;
                            }
                            tokio::select! {
                                ev_res = bg_rx.recv(), if event_queue_active => {
                                    match ev_res {
                                        Some(ev) => {
                                            try_yield_bg_events!(ev, any_agent_visible);
                                            if any_agent_visible { break; }
                                        }
                                        None => event_queue_active = false,
                                    }
                                }
                                _ = async { loop { if *w.borrow() == 0 { break; } if w.changed().await.is_err() { break; } } } => { continue; }
                                _ = async { if let Some(token) = &cancel_token { token.cancelled().await; } else { futures::future::pending::<()>().await; } } => { break; }
                            }
                        }
                    }
                    if any_agent_visible { continue; }
                    break;
                }
                tokio::task::yield_now().await;
            }
        }.instrument(reply_stream_span));

        Ok(inner)
    }
}

use crate::conversation::message::{Message, MessageContent, ToolResponse};
use rmcp::model::CallToolRequestParams;

pub trait MessageExt {
    fn first_tool_request(&self) -> Option<(&str, &CallToolRequestParams)>;
    fn tool_request(&self, tool_name: &str) -> Option<(&str, &CallToolRequestParams)>;
    fn tool_response(&self, tool_id: &str) -> Option<&ToolResponse>;
    fn has_tool_request(&self) -> bool;
    fn replace_tool_request_with_text(&mut self, tool_name: &str, text: String) -> bool;
    fn with_tool_request_replaced_by_text(&self, tool_name: &str, text: String) -> Option<Message>;
}

impl MessageExt for Message {
    fn first_tool_request(&self) -> Option<(&str, &CallToolRequestParams)> {
        self.content.iter().find_map(|content| {
            if let MessageContent::ToolRequest(req) = content {
                req.tool_call
                    .as_ref()
                    .ok()
                    .map(|call| (req.id.as_str(), call))
            } else {
                None
            }
        })
    }

    fn tool_request(&self, tool_name: &str) -> Option<(&str, &CallToolRequestParams)> {
        self.content.iter().find_map(|content| {
            if let MessageContent::ToolRequest(req) = content {
                let call = req.tool_call.as_ref().ok()?;
                if call.name == tool_name {
                    return Some((req.id.as_str(), call));
                }
            }
            None
        })
    }

    fn tool_response(&self, tool_id: &str) -> Option<&ToolResponse> {
        self.content.iter().find_map(|content| {
            if let MessageContent::ToolResponse(resp) = content {
                if resp.id == tool_id {
                    return Some(resp);
                }
            }
            None
        })
    }

    fn has_tool_request(&self) -> bool {
        self.content
            .iter()
            .any(|content| matches!(content, MessageContent::ToolRequest(_)))
    }

    fn replace_tool_request_with_text(&mut self, tool_name: &str, text: String) -> bool {
        if let Some(content) = self.content.iter_mut().find(|content| {
            matches!(content, MessageContent::ToolRequest(req)
                if req.tool_call.as_ref().ok().map_or(false, |call| call.name == tool_name))
        }) {
            *content = MessageContent::text(text);
            true
        } else {
            false
        }
    }

    fn with_tool_request_replaced_by_text(&self, tool_name: &str, text: String) -> Option<Message> {
        let mut message = self.clone();
        if message.replace_tool_request_with_text(tool_name, text) {
            Some(message)
        } else {
            None
        }
    }
}

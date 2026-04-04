use rmcp::model::ServerNotification;

#[derive(Clone)]
pub enum BgEv {
    Mcp(ServerNotification),
    Done(String, String, usize),
}

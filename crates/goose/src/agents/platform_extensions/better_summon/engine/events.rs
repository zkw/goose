use rmcp::model::ServerNotification;

#[derive(Clone)]
pub enum BgEv {
    Mcp(ServerNotification),
    Spawned(String),
    Done(String, String, usize),
}

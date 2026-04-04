use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

use super::events::BgEv;
use once_cell::sync::Lazy;

pub(super) enum RMsg {
    Bind(Arc<str>, mpsc::UnboundedSender<BgEv>, oneshot::Sender<bool>),
    Unbind(Arc<str>),
    Route(Arc<str>, BgEv),
}

pub(super) static ROUTER: Lazy<mpsc::UnboundedSender<RMsg>> = Lazy::new(|| {
    let (tx, mut rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        let mut sessions: HashMap<Arc<str>, mpsc::UnboundedSender<BgEv>> = HashMap::new();
        while let Some(msg) = rx.recv().await {
            match msg {
                RMsg::Bind(id, tx, rtx) => {
                    let _ = rtx.send(sessions.insert(id, tx).is_none());
                }
                RMsg::Unbind(id) => {
                    sessions.remove(&id);
                }
                RMsg::Route(id, ev) => {
                    if let Some(tx) = sessions.get(&id) {
                        let _ = tx.send(ev);
                    }
                }
            }
        }
    });
    tx
});

pub fn bind_session(id: Arc<str>, tx: mpsc::UnboundedSender<BgEv>) -> oneshot::Receiver<bool> {
    let (rtx, rrx) = oneshot::channel();
    let _ = ROUTER.send(RMsg::Bind(id, tx, rtx));
    rrx
}

pub fn unbind_session(id: Arc<str>) {
    let _ = ROUTER.send(RMsg::Unbind(id));
}

pub fn route_event(id: Arc<str>, ev: BgEv) {
    let _ = ROUTER.send(RMsg::Route(id, ev));
}

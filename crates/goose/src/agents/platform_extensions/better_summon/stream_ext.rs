use crate::{
    agents::{platform_extensions::better_summon::engine::BgEv, AgentEvent},
    conversation::message::{Message, MessageContent},
};
use anyhow::Result;
use async_stream::stream;
use futures::{Stream, StreamExt};
use std::{collections::VecDeque, future::Future, time::Duration};
use tokio::sync::mpsc;

fn is_safe_phase(msg: &Message) -> bool {
    if msg.content.is_empty() {
        return true;
    }
    match msg.content.last() {
        Some(MessageContent::ToolRequest(_)) => false,
        Some(MessageContent::ToolResponse(_)) => true,
        Some(MessageContent::Text(_)) | Some(MessageContent::SystemNotification(_)) => false,
        _ => false,
    }
}

pub trait StreamRecoveryExt<T, E>: Stream<Item = Result<T, E>> + Sized + Unpin {
    fn robust_retry<F, Fut>(self, mut recover_factory: F) -> impl Stream<Item = Result<T, E>>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<Self, E>>,
        E: std::fmt::Display,
    {
        stream! {
            let mut current = self;
            loop {
                let mut errored = false;

                while let Some(res) = current.next().await {
                    match res {
                        Ok(item) => yield Ok(item),
                        Err(error) => {
                            tracing::warn!(%error, "Stream output interrupted: initiating recovery");
                            errored = true;
                            break;
                        }
                    }
                }

                if !errored {
                    break;
                }

                let mut attempts = 0;
                loop {
                    attempts += 1;
                    match recover_factory().await {
                        Ok(new_stream) => {
                            tracing::info!(attempt = attempts, "Stream recovered successfully");
                            current = new_stream;
                            break;
                        }
                        Err(error) => {
                            tracing::error!(attempt = attempts, %error, "Stream recovery failed, retrying in 2s");
                            tokio::time::sleep(Duration::from_secs(2)).await;
                        }
                    }
                }
            }
        }
    }
}

impl<S, T, E> StreamRecoveryExt<T, E> for S
where
    S: Stream<Item = Result<T, E>> + Sized + Unpin,
    E: std::fmt::Display,
{
}

pub trait StreamUiExt<E>: Stream<Item = Result<AgentEvent, E>> + Sized {
    fn interleave_ui_safely(
        self,
        mut ui_rx: mpsc::UnboundedReceiver<BgEv>,
        bg_ev_to_message: impl Fn(BgEv) -> Option<Message> + 'static,
    ) -> impl Stream<Item = Result<AgentEvent, E>> {
        stream! {
            let mut ui_buffer = VecDeque::new();
            let mut is_safe = true;
            let bg_ev_to_message = Box::new(bg_ev_to_message);
            let mut source = Box::pin(self);

            loop {
                let mut source_ref = source.as_mut();
                tokio::select! {
                    biased;
                    Some(ev) = ui_rx.recv() => {
                        if let Some(msg) = bg_ev_to_message(ev) {
                            ui_buffer.push_back(msg);
                        }
                    }
                    res = source_ref.next() => {
                        match res {
                            Some(Ok(AgentEvent::Message(msg))) => {
                                is_safe = is_safe_phase(&msg);
                                yield Ok(AgentEvent::Message(msg));
                            }
                            Some(other) => yield other,
                            None => break,
                        }
                    }
                    _ = tokio::time::sleep(Duration::from_millis(500)) => {
                        is_safe = true;
                    }
                }

                if is_safe {
                    while let Some(msg) = ui_buffer.pop_front() {
                        yield Ok(AgentEvent::Message(msg));
                    }
                }
            }

            while let Some(msg) = ui_buffer.pop_front() {
                yield Ok(AgentEvent::Message(msg));
            }
        }
    }
}

impl<S, E> StreamUiExt<E> for S where S: Stream<Item = Result<AgentEvent, E>> + Sized {}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::{stream, StreamExt};

    #[tokio::test]
    async fn robust_retry_recovers_after_error() {
        let signals = vec![Ok(1), Err("broken")];
        let inner = stream::iter(signals);
        let recovered = inner.robust_retry(|| async { Ok(stream::iter(vec![Ok(2), Ok(3)])) });

        let values: Vec<_> = recovered.collect().await;
        assert_eq!(values, vec![Ok(1), Ok(2), Ok(3)]);
    }
}

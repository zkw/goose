use std::{future::Future, time::Duration};
use tokio_util::sync::CancellationToken;

/// Retries the provided async operation forever until it succeeds or the cancellation token is triggered.
///
/// The caller provides a sleep function so time can be injected for tests.
pub async fn retry_until_cancelled<Op, Fut, T, E, SleepFn, SleepFut, OnError>(
    mut operation: Op,
    mut sleep: SleepFn,
    mut on_error: OnError,
    cancel_token: Option<&CancellationToken>,
) -> Option<T>
where
    Op: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    SleepFn: FnMut(Duration) -> SleepFut,
    SleepFut: Future<Output = ()>,
    OnError: FnMut(&E),
{
    loop {
        if cancel_token
            .as_ref()
            .is_some_and(|token| token.is_cancelled())
        {
            return None;
        }

        match operation().await {
            Ok(result) => return Some(result),
            Err(error) => {
                on_error(&error);
                sleep(Duration::from_secs(1)).await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_retry_until_cancelled_retries_at_least_100_times() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let failures = Arc::new(AtomicUsize::new(0));

        let operation_attempts = attempts.clone();
        let service = move || {
            let operation_attempts = operation_attempts.clone();
            async move {
                let current = operation_attempts.fetch_add(1, Ordering::SeqCst);
                if current < 100 {
                    Err::<(), _>(())
                } else {
                    Ok(())
                }
            }
        };

        let sleep_calls = Arc::new(AtomicUsize::new(0));
        let sleep_calls_clone = sleep_calls.clone();
        let sleep_fn = move |_duration: Duration| {
            let sleep_calls_clone = sleep_calls_clone.clone();
            async move {
                sleep_calls_clone.fetch_add(1, Ordering::SeqCst);
            }
        };

        let failures_clone = failures.clone();
        let error_logger = move |_err: &()| {
            failures_clone.fetch_add(1, Ordering::SeqCst);
        };

        let result = retry_until_cancelled(service, sleep_fn, error_logger, None).await;

        assert!(result.is_some(), "Expected operation to eventually succeed");
        assert!(
            attempts.load(Ordering::SeqCst) >= 101,
            "Expected at least 101 operation attempts, got {}",
            attempts.load(Ordering::SeqCst)
        );
        assert!(
            failures.load(Ordering::SeqCst) >= 100,
            "Expected at least 100 failures before success, got {}",
            failures.load(Ordering::SeqCst)
        );
        assert_eq!(
            sleep_calls.load(Ordering::SeqCst),
            100,
            "Expected a sleep call after each failure"
        );
    }
}

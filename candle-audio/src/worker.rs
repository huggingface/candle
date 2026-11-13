use crate::*;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TryRecvError, TrySendError},
    Arc, Mutex,
};
use std::thread::{self, JoinHandle};
use std::time::Duration;

#[derive(Debug)]
pub enum WorkerCommand {
    Audio(AudioBuffer),
    Text(String),
    Commit,
    /// Cancel the current turn, or acknowledge the same already-rolled-back
    /// interrupt while idle. New input or cancel_current starts a new intent.
    Cancel,
    ResetConversation,
    Close,
}
#[derive(Debug)]
pub enum WorkerEvent {
    Event(AudioEvent),
    CommandCompleted { id: u64 },
    Error { id: u64, error: Error },
}
#[derive(Clone, Debug)]
pub struct WorkerOptions {
    pub command_capacity: usize,
    pub event_capacity: usize,
    pub poll_interval: Duration,
}
impl Default for WorkerOptions {
    fn default() -> Self {
        Self {
            command_capacity: 8,
            event_capacity: 16,
            poll_interval: Duration::from_millis(2),
        }
    }
}
struct StampedEvent {
    generation: u64,
    event: WorkerEvent,
}

/// An inference thread with bounded commands/events. Stepwise models are polled
/// between commands, so incoming chunks and generated output can interleave.
/// Blocking model calls remain blocking; cancellation requires their cooperation.
/// Dropping cancels and joins the owned thread. No inference runs on an audio callback.
pub struct RealtimeWorker {
    commands: SyncSender<(u64, u64, WorkerCommand)>,
    events: Receiver<StampedEvent>,
    stop: Arc<AtomicBool>,
    generation: Arc<AtomicU64>,
    cancellation: Arc<Mutex<CancellationToken>>,
    next_id: AtomicU64,
    thread: Option<JoinHandle<()>>,
    limits: SessionLimits,
    input: Option<AudioFormat>,
}
impl RealtimeWorker {
    pub fn spawn<M: RealtimeModel + Send + 'static>(
        model: M,
        limits: SessionLimits,
        options: WorkerOptions,
    ) -> Result<Self> {
        if options.command_capacity == 0
            || options.event_capacity == 0
            || options.poll_interval.is_zero()
        {
            return Err(Error::InvalidInput(
                "worker capacities and poll interval must be nonzero".into(),
            ));
        }
        let mut session = RealtimeSession::with_limits(model, limits)?;
        let limits = session.limits().clone();
        let input = session.capabilities().input_format;
        let (commands, rx) =
            mpsc::sync_channel::<(u64, u64, WorkerCommand)>(options.command_capacity);
        let (tx, events) = mpsc::sync_channel(options.event_capacity);
        let stop = Arc::new(AtomicBool::new(false));
        let generation = Arc::new(AtomicU64::new(0));
        let cancellation = Arc::new(Mutex::new(session.cancellation_token()));
        let (thread_stop, thread_generation, thread_cancel) =
            (stop.clone(), generation.clone(), cancellation.clone());
        let thread = thread::Builder::new()
            .name("candle-audio-inference".into())
            .spawn(move || {
                // A queued Cancel often acknowledges cancel_current after the
                // running action or idle timeout has already rolled its turn back.
                let mut cancelled_epoch=None;
                while !thread_stop.load(Ordering::Acquire) {
                    let operational =
                        matches!(session.state(), SessionState::Ready | SessionState::Active);
                    let command = rx.recv_timeout(if session.output_pending() && operational {
                        options.poll_interval
                    } else {
                        Duration::from_millis(20)
                    });
                    // Cancellation may advance the generation while recv_timeout
                    // waits. Snapshot after receiving, so a new-generation command
                    // is never compared with the generation from before that wait.
                    let generation = thread_generation.load(Ordering::Acquire);
                    let cancellation_observed=std::cell::Cell::new(None);
                    let event_generation=command.as_ref().map(|(_,generation,_)|*generation).unwrap_or(generation);
                    // cancel_current is also useful while idle: finish its
                    // rollback before executing fresh input, without discarding
                    // that new command. Explicit Cancel/Reset handle themselves.
                    let needs_recovery=session.cancellation_token().is_cancelled() && operational
                        && matches!(&command,Ok((_,command_generation,WorkerCommand::Audio(_)|WorkerCommand::Text(_)|WorkerCommand::Commit)) if *command_generation==generation);
                    let recovery=if needs_recovery {
                        session.cancel(&mut |event| {
                            let epoch=thread_generation.load(Ordering::Acquire);
                            let result=send_event(&tx,StampedEvent{generation:epoch,event:WorkerEvent::Event(event)},&thread_stop,None);
                            if result.is_ok() {cancellation_observed.set(Some(epoch));}
                            result
                        })
                    } else {Ok(())};
                    if session.state()==SessionState::Ready {
                        if let Some(epoch)=cancellation_observed.take() {cancelled_epoch=Some(epoch);}
                    }
                    let token = session.cancellation_token();
                    *thread_cancel.lock().unwrap_or_else(|e|e.into_inner())=token.clone();
                    if thread_generation.load(Ordering::Acquire)!=generation {token.cancel();}
                    let mut sink = |event: AudioEvent| -> Result<()> {
                        let lifecycle = matches!(
                            event,
                            AudioEvent::Cancelled { .. } | AudioEvent::TurnCompleted { .. }
                        );
                        // Cancellation acknowledges the current interrupt even
                        // when the cancelled inference started in an older
                        // generation. Its data/completion events stay stale.
                        let lifecycle_generation=if matches!(&event,AudioEvent::Cancelled{..}) {thread_generation.load(Ordering::Acquire)} else {generation};
                        let is_cancelled=matches!(&event,AudioEvent::Cancelled{..});
                        let result=send_event(
                            &tx,
                            StampedEvent {
                                generation:lifecycle_generation,
                                event: WorkerEvent::Event(event),
                            },
                            &thread_stop,
                            if lifecycle { None } else { Some(&token) },
                        );
                        if is_cancelled && result.is_ok() {cancellation_observed.set(Some(lifecycle_generation));}
                        result
                    };
                    let (id, result) = match command {
                        Ok((id, command_generation, _)) if command_generation != generation => {
                            (id, Err(Error::Cancelled))
                        }
                        Ok((id, _, command)) => {
                            let result = recovery.and_then(|()| match command {
                                WorkerCommand::Audio(audio) => {
                                    cancelled_epoch=None;
                                    session.push_audio(&audio, &mut sink)
                                }
                                WorkerCommand::Text(text) => {cancelled_epoch=None;session.push_text(&text, &mut sink)},
                                WorkerCommand::Commit => {cancelled_epoch=None;session.commit(&mut sink)},
                                WorkerCommand::Cancel if cancelled_epoch==Some(generation) && session.state()==SessionState::Ready=>Ok(()),
                                WorkerCommand::Cancel => session.cancel(&mut sink),
                                WorkerCommand::ResetConversation => {
                                    let result=session.reset_conversation();
                                    if result.is_ok() && token.is_cancelled() {cancelled_epoch=Some(generation);}
                                    result
                                },
                                WorkerCommand::Close => {
                                    thread_stop.store(true, Ordering::Release);
                                    Ok(())
                                }
                            });
                            let result = result.and_then(|()| {
                                if thread_stop.load(Ordering::Acquire) {Ok(())}
                                else if session.output_pending() {
                                    session.poll_output(&mut sink).map(|_| ())
                                } else {
                                    Ok(())
                                }
                            });
                            (id, result)
                        }
                        Err(RecvTimeoutError::Disconnected) => break,
                        Err(RecvTimeoutError::Timeout) => {
                            let result = if token.is_cancelled() && operational {
                                session.cancel(&mut sink)
                            } else if session.output_pending() && operational {
                                session.poll_output(&mut sink).map(|_| ())
                            } else {
                                Ok(())
                            };
                            if session.state()==SessionState::Ready {
                                if let Some(epoch)=cancellation_observed.get() {cancelled_epoch=Some(epoch);}
                            }
                            *thread_cancel.lock().unwrap_or_else(|e| e.into_inner()) =
                                session.cancellation_token();
                            if let Err(error) = result {
                                let _ = send_event(
                                    &tx,
                                    StampedEvent {
                                        generation:if session.state()==SessionState::Faulted {thread_generation.load(Ordering::Acquire)} else {generation},
                                        event: WorkerEvent::Error { id: 0, error },
                                    },
                                    &thread_stop,
                                    None,
                                );
                            }
                            continue;
                        }
                    };
                    if session.state()==SessionState::Ready {
                        if let Some(epoch)=cancellation_observed.get() {cancelled_epoch=Some(epoch);}
                    }
                    *thread_cancel.lock().unwrap_or_else(|e| e.into_inner()) =
                        session.cancellation_token();
                    // A rollback/model failure leaves the worker faulted for the
                    // current epoch. It must remain visible even if cancellation
                    // invalidated the command that first encountered that failure.
                    let persistent_fault=session.state()==SessionState::Faulted
                        && matches!(&result,Err(error) if !matches!(error,Error::Cancelled));
                    let event = match result {
                        Ok(()) => WorkerEvent::CommandCompleted { id },
                        Err(error) => WorkerEvent::Error { id, error },
                    };
                    if send_event(
                        &tx,
                        StampedEvent {
                            generation: if persistent_fault {thread_generation.load(Ordering::Acquire)} else {event_generation},
                            event,
                        },
                        &thread_stop,
                        None,
                    )
                    .is_err()
                    {
                        break;
                    }
                }
            })?;
        Ok(Self {
            commands,
            events,
            stop,
            generation,
            cancellation,
            next_id: AtomicU64::new(1),
            thread: Some(thread),
            limits,
            input,
        })
    }
    pub fn try_send(&self, command: WorkerCommand) -> Result<u64> {
        match &command {
            WorkerCommand::Audio(audio) => {
                if Some(audio.format()) != self.input {
                    return Err(Error::InvalidInput("worker input format mismatch".into()));
                }
                if audio.frames() > self.limits.max_chunk_frames {
                    return Err(Error::LimitExceeded(
                        "worker audio chunk exceeds budget".into(),
                    ));
                }
            }
            WorkerCommand::Text(text) if text.len() > self.limits.max_text_bytes_per_message => {
                return Err(Error::LimitExceeded(
                    "worker text message exceeds budget".into(),
                ))
            }
            _ => {}
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.commands
            .try_send((id, self.generation.load(Ordering::Acquire), command))
            .map_err(|error| match error {
                TrySendError::Full(_) => {
                    Error::LimitExceeded("inference command queue is full".into())
                }
                TrySendError::Disconnected(_) => {
                    Error::InvalidState("inference worker is closed".into())
                }
            })?;
        Ok(id)
    }
    /// Interrupt running work and invalidate queued output/commands from that generation.
    /// Fresh input is accepted after rollback without an additional Cancel command.
    /// A queued Cancel for this same interrupt is an idempotent acknowledgement
    /// while idle; new input or another interrupt ends that acknowledgement window.
    pub fn cancel_current(&self) {
        self.generation.fetch_add(1, Ordering::AcqRel);
        self.cancellation
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .cancel();
    }
    pub fn recv_timeout(
        &self,
        timeout: Duration,
    ) -> std::result::Result<WorkerEvent, RecvTimeoutError> {
        let started = std::time::Instant::now();
        loop {
            let remaining = timeout.saturating_sub(started.elapsed());
            let event = self.events.recv_timeout(remaining)?;
            if event.generation == self.generation.load(Ordering::Acquire) {
                return Ok(event.event);
            }
        }
    }
    pub fn try_recv(&self) -> std::result::Result<WorkerEvent, TryRecvError> {
        loop {
            let event = self.events.try_recv()?;
            if event.generation == self.generation.load(Ordering::Acquire) {
                return Ok(event.event);
            }
        }
    }
    pub fn close(&mut self) -> Result<()> {
        self.stop.store(true, Ordering::Release);
        self.cancellation
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .cancel();
        if let Some(thread) = self.thread.take() {
            thread
                .join()
                .map_err(|_| Error::InvalidState("inference worker panicked".into()))?;
        }
        Ok(())
    }
}
impl Drop for RealtimeWorker {
    fn drop(&mut self) {
        let _ = self.close();
    }
}
fn send_event(
    tx: &SyncSender<StampedEvent>,
    mut event: StampedEvent,
    stop: &AtomicBool,
    cancel: Option<&CancellationToken>,
) -> Result<()> {
    loop {
        if stop.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        if let Some(cancel) = cancel {
            cancel.check()?;
        }
        match tx.try_send(event) {
            Ok(()) => return Ok(()),
            Err(TrySendError::Full(value)) => {
                event = value;
                thread::sleep(Duration::from_millis(1));
            }
            Err(TrySendError::Disconnected(_)) => {
                return Err(Error::InvalidState("event consumer disconnected".into()))
            }
        }
    }
}

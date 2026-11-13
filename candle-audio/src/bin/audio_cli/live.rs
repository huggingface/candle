//! Manual-turn native audio frontend. Hardware is opened only after Enter.
use anyhow::{bail, ensure, Result};
use candle_audio::{
    live::{self, DeviceSelector, Direction, LiveCapture, LiveOptions, LivePlayback},
    AudioBuffer, AudioConverter, AudioEvent, AudioFormat, CancellationToken, ChannelPolicy, Error,
    FinishReason, RealtimeModel, RealtimeWorker, SessionLimits, WorkerCommand, WorkerEvent,
    WorkerOptions,
};
use std::{
    collections::VecDeque,
    io::{self, Read},
    sync::mpsc::{self, Receiver, TryRecvError},
    thread,
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Control {
    Toggle,
    Cancel,
    Reset,
    Quit,
    Help,
}
fn control(line: &[u8]) -> Control {
    match line.trim_ascii() {
        b"" => Control::Toggle,
        b"c" | b"C" => Control::Cancel,
        b"r" | b"R" => Control::Reset,
        b"q" | b"Q" => Control::Quit,
        _ => Control::Help,
    }
}
fn controls() -> Receiver<Control> {
    let (tx, rx) = mpsc::sync_channel(8);
    thread::spawn(move || {
        let input = io::stdin();
        let mut input = input.lock();
        let mut line = [0u8; 16];
        let mut length = 0;
        let mut overflow = false;
        let mut byte = [0u8; 1];
        loop {
            match input.read(&mut byte) {
                Ok(0) | Err(_) => {
                    let _ = tx.send(Control::Quit);
                    break;
                }
                Ok(_) => {
                    if byte[0] == b'\n' {
                        let command = if overflow {
                            Control::Help
                        } else {
                            control(&line[..length])
                        };
                        if tx.send(command).is_err() {
                            break;
                        }
                        length = 0;
                        overflow = false;
                    } else if length < line.len() {
                        line[length] = byte[0];
                        length += 1;
                    } else {
                        overflow = true;
                    }
                }
            }
        }
    });
    rx
}
fn policy(source: AudioFormat, target: AudioFormat) -> Result<ChannelPolicy> {
    if source.channels() == target.channels() {
        Ok(ChannelPolicy::Preserve)
    } else if target.channels() == 1 {
        Ok(ChannelPolicy::Average)
    } else if source.channels() == 1 && target.channels() == 2 {
        Ok(ChannelPolicy::Map(vec![0, 0]))
    } else {
        bail!(
            "explicit channel mapping required for {}→{} channels",
            source.channels(),
            target.channels()
        )
    }
}
#[derive(Clone)]
enum Command {
    Audio(AudioBuffer),
    Text(String),
    Commit,
    Cancel,
    Reset,
}
impl Command {
    fn worker(&self) -> WorkerCommand {
        match self {
            Self::Audio(audio) => WorkerCommand::Audio(audio.clone()),
            Self::Text(text) => WorkerCommand::Text(text.clone()),
            Self::Commit => WorkerCommand::Commit,
            Self::Cancel => WorkerCommand::Cancel,
            Self::Reset => WorkerCommand::ResetConversation,
        }
    }
}
struct PendingCommands {
    commands: VecDeque<Command>,
    audio_frames: usize,
    max_frames: usize,
}
impl PendingCommands {
    fn new(max_frames: usize) -> Self {
        Self {
            commands: VecDeque::new(),
            audio_frames: 0,
            max_frames,
        }
    }
    fn push(&mut self, command: Command) -> Result<()> {
        let frames = match &command {
            Command::Audio(audio) => audio.frames(),
            _ => 0,
        };
        ensure!(self.commands.len()<32 && self.audio_frames.checked_add(frames).is_some_and(|n|n<=self.max_frames),"inference backpressure exceeded bounded input queue; aborting this turn without reporting a complete result");
        self.audio_frames += frames;
        self.commands.push_back(command);
        Ok(())
    }
    fn audio(&mut self, audio: AudioBuffer) -> Result<()> {
        if !audio.is_empty() {
            self.push(Command::Audio(audio))?;
        }
        Ok(())
    }
    fn pop(&mut self) {
        if let Some(Command::Audio(audio)) = self.commands.pop_front() {
            self.audio_frames -= audio.frames();
        }
    }
    fn clear(&mut self) {
        self.commands.clear();
        self.audio_frames = 0;
    }
}
fn capture_integrity(metrics: live::LiveMetrics) -> Result<()> {
    ensure!(metrics.overrun_frames == 0,
        "capture lost {} frames under backpressure; aborting the turn instead of returning an incomplete result", metrics.overrun_frames);
    ensure!(
        metrics.backend_xruns == 0,
        "capture backend reported {} xruns; audio may be missing, so this turn is aborted",
        metrics.backend_xruns
    );
    Ok(())
}
struct PendingPcm {
    audio: AudioBuffer,
    offset: usize,
}
impl PendingPcm {
    fn chunk(&self, max_frames: usize) -> Result<Option<AudioBuffer>> {
        let frames = (self.audio.frames() - self.offset).min(max_frames);
        if frames == 0 {
            return Ok(None);
        }
        let channels = self.audio.format().channels() as usize;
        Ok(Some(AudioBuffer::new(
            self.audio.format(),
            self.audio.samples()[self.offset * channels..(self.offset + frames) * channels]
                .to_vec(),
        )?))
    }
    fn complete(&self) -> bool {
        self.offset == self.audio.frames()
    }
}
struct Output {
    playback: LivePlayback,
    converter: AudioConverter,
    pending: Option<PendingPcm>,
    started: bool,
}
impl Output {
    fn push(&mut self, audio: &AudioBuffer) -> Result<()> {
        ensure!(
            self.pending.is_none(),
            "internal output backpressure violation"
        );
        let audio = self.converter.push(audio)?;
        if !audio.is_empty() {
            self.pending = Some(PendingPcm { audio, offset: 0 });
        }
        Ok(())
    }
    fn flush(&mut self) -> Result<()> {
        let audio = self.converter.finish()?;
        ensure!(
            self.pending.is_none(),
            "internal output tail backpressure violation"
        );
        if !audio.is_empty() {
            self.pending = Some(PendingPcm { audio, offset: 0 });
        }
        Ok(())
    }
    fn pump(&mut self) -> Result<()> {
        if let Some(pending) = &mut self.pending {
            let metrics = self.playback.metrics();
            let available = metrics
                .capacity_frames
                .saturating_sub(metrics.queued_frames);
            if let Some(chunk) = pending.chunk(available)? {
                pending.offset += self.playback.write(&chunk)?;
                if !self.started {
                    self.playback.start()?;
                    self.started = true;
                }
            }
            if pending.complete() {
                self.pending = None;
            }
        }
        Ok(())
    }
    fn drained(&self) -> bool {
        self.pending.is_none() && self.playback.metrics().queued_frames == 0
    }
    fn stop(&mut self) {
        self.playback.stop();
        self.pending = None;
        self.converter.reset();
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Idle,
    Capturing,
    Draining,
    Responding,
    Recovering,
}
struct Run {
    worker: RealtimeWorker,
    commands: PendingCommands,
    capture: Option<LiveCapture>,
    input_converter: Option<AudioConverter>,
    output: Option<Output>,
    phase: Phase,
    waiting: Option<u64>,
    model_done: bool,
    input: AudioFormat,
    output_format: Option<AudioFormat>,
    max_seconds: f64,
    prompt: Option<String>,
    input_limit: usize,
    source_frames: usize,
    source_limit: usize,
    converted_frames: usize,
    started: Option<Instant>,
    cancel: CancellationToken,
}
impl Run {
    fn start(
        &mut self,
        input_selector: &DeviceSelector,
        output_selector: &DeviceSelector,
    ) -> Result<()> {
        ensure!(
            self.phase == Phase::Idle,
            "a turn is still active; use c to interrupt or wait for completion"
        );
        // Resolving metadata and opening streams occurs only after explicit Enter.
        let native = live::resolve_format(input_selector, Direction::Input, Some(self.input))?;
        let capture = LiveCapture::open(
            input_selector,
            LiveOptions::new(native),
            self.cancel.clone(),
        )?;
        let input_converter =
            AudioConverter::with_format(native, self.input, policy(native, self.input)?)?;
        let output = if let Some(format) = self.output_format {
            let target = live::resolve_format(output_selector, Direction::Output, Some(format))?;
            let playback = LivePlayback::open(
                output_selector,
                LiveOptions::new(target),
                self.cancel.clone(),
            )?;
            eprintln!(
                "Playback device {}: {:?}, {}",
                playback.info().device.name,
                target,
                playback.info().sample_format
            );
            Some(Output {
                playback,
                converter: AudioConverter::with_format(format, target, policy(format, target)?)?,
                pending: None,
                started: false,
            })
        } else {
            None
        };
        self.source_frames = 0;
        self.converted_frames = 0;
        self.source_limit = (self.input_limit as f64 / self.input.sample_rate() as f64
            * native.sample_rate() as f64)
            .floor() as usize;
        eprintln!(
            "Capture device {}: {:?}, {} → model {:?}; Enter stops the turn",
            capture.info().device.name,
            native,
            capture.info().sample_format,
            self.input
        );
        if let Some(prompt) = &self.prompt {
            self.commands.push(Command::Text(prompt.clone()))?;
        }
        capture.start()?;
        self.capture = Some(capture);
        self.input_converter = Some(input_converter);
        self.output = output;
        self.phase = Phase::Capturing;
        self.started = Some(Instant::now());
        self.model_done = false;
        Ok(())
    }
    fn stop_capture(&mut self) {
        if let Some(capture) = &mut self.capture {
            capture.stop();
        }
        self.phase = Phase::Draining;
    }
    fn capture(&mut self) -> Result<()> {
        if !matches!(self.phase, Phase::Capturing | Phase::Draining) {
            return Ok(());
        }
        let capture = self.capture.as_ref().unwrap();
        capture_integrity(capture.metrics())?;
        if let Some(error) = capture.take_error() {
            eprintln!("Capture backend: {error}");
        }
        self.cancel.check()?;
        if self.phase == Phase::Capturing
            && self
                .started
                .is_some_and(|t| t.elapsed().as_secs_f64() >= self.max_seconds)
        {
            eprintln!("Manual turn reached its time budget; stopping capture");
            self.stop_capture();
        }
        // Leave enough room for this block and a resampler tail. When inference
        // is slower, let the bounded native queue apply explicit pressure.
        if self.commands.audio_frames >= self.commands.max_frames / 2 {
            return Ok(());
        }
        let capture = self.capture.as_ref().unwrap();
        let native = capture.info().format;
        let block = (native.sample_rate() as usize / 50)
            .max(1)
            .min(capture.metrics().capacity_frames);
        let audio = capture.read(block)?;
        if let Some(audio) = audio {
            let remaining = self.source_limit.saturating_sub(self.source_frames);
            let keep = audio.frames().min(remaining);
            let audio = if keep < audio.frames() {
                AudioBuffer::new(
                    native,
                    audio.samples()[..keep * native.channels() as usize].to_vec(),
                )?
            } else {
                audio
            };
            self.source_frames += audio.frames();
            let audio = self.input_converter.as_mut().unwrap().push(&audio)?;
            self.converted_frames += audio.frames();
            ensure!(
                self.converted_frames <= self.input_limit,
                "capture conversion exceeded model context budget"
            );
            self.commands.audio(audio)?;
            if self.source_frames >= self.source_limit {
                eprintln!("Input context budget reached; committing this bounded turn");
                self.stop_capture();
                self.capture.as_ref().unwrap().clear();
            }
        } else if self.phase == Phase::Draining {
            let tail = self.input_converter.as_mut().unwrap().finish()?;
            self.converted_frames += tail.frames();
            ensure!(
                self.converted_frames <= self.input_limit,
                "resampler tail exceeded model context budget"
            );
            self.commands.audio(tail)?;
            self.report_capture();
            self.capture = None;
            self.input_converter = None;
            if self.converted_frames == 0 {
                eprintln!("No audio captured; no model turn was committed");
                if let Some(output) = &mut self.output {
                    output.stop();
                }
                self.output = None;
                self.phase = Phase::Idle;
            } else {
                self.commands.push(Command::Commit)?;
                self.phase = Phase::Responding;
            }
        }
        Ok(())
    }
    fn report_capture(&self) {
        if let Some(capture) = &self.capture {
            eprintln!(
                "Capture metrics: {}",
                serde_json::to_string(&capture.metrics()).unwrap_or_default()
            );
        }
    }
    fn interrupt(&mut self, reset: bool) -> Result<()> {
        self.report_capture();
        if let Some(capture) = &mut self.capture {
            capture.stop();
            capture.clear();
        }
        self.capture = None;
        self.input_converter = None;
        if let Some(output) = &mut self.output {
            eprintln!(
                "Playback metrics: {}",
                serde_json::to_string(&output.playback.metrics())?
            );
            output.stop();
        }
        self.output = None;
        self.commands.clear();
        self.worker.cancel_current();
        self.commands.push(if reset {
            Command::Reset
        } else {
            Command::Cancel
        })?;
        self.phase = Phase::Recovering;
        self.waiting = None;
        self.model_done = false;
        eprintln!(
            "{} requested; waiting for inference cleanup",
            if reset {
                "Conversation reset"
            } else {
                "Cancellation"
            }
        );
        Ok(())
    }
    fn send(&mut self) -> Result<()> {
        for _ in 0..4 {
            let Some(command) = self.commands.commands.front() else {
                break;
            };
            match self.worker.try_send(command.worker()) {
                Ok(id) => {
                    if matches!(command, Command::Cancel | Command::Reset) {
                        self.waiting = Some(id);
                    }
                    self.commands.pop();
                }
                Err(Error::LimitExceeded(_)) => break,
                Err(error) => return Err(error.into()),
            }
        }
        Ok(())
    }
    fn events(&mut self) -> Result<()> {
        for _ in 0..16 {
            if self
                .output
                .as_ref()
                .is_some_and(|output| output.pending.is_some())
            {
                break;
            }
            let event = match self.worker.try_recv() {
                Ok(event) => event,
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => bail!("inference worker disconnected"),
            };
            match event {
                WorkerEvent::CommandCompleted { id } => {
                    if self.waiting == Some(id) {
                        self.waiting = None;
                        self.phase = Phase::Idle;
                        eprintln!("Ready; Enter starts a new turn");
                    }
                }
                WorkerEvent::Error {
                    error: Error::Cancelled,
                    ..
                } if self.phase == Phase::Recovering => {}
                WorkerEvent::Error { id, error } => bail!("inference command {id}: {error}"),
                WorkerEvent::Event(event) => match event {
                    AudioEvent::Audio { audio, .. } | AudioEvent::AudioChunk { audio, .. } => {
                        if self.phase != Phase::Recovering {
                            if let Some(output) = &mut self.output {
                                output.push(&audio)?;
                            } else {
                                bail!("model emitted audio without an advertised output format");
                            }
                        }
                    }
                    AudioEvent::Transcript { transcript, .. }
                    | AudioEvent::TranscriptSegment { transcript, .. } => {
                        println!("{}", transcript.text)
                    }
                    AudioEvent::Text { text, .. } => {
                        use std::io::Write;
                        print!("{text}");
                        io::stdout().flush()?;
                    }
                    AudioEvent::TurnCompleted { finish_reason, .. } => {
                        eprintln!("Turn finish_reason={finish_reason:?}");
                        if matches!(finish_reason, FinishReason::Length | FinishReason::Deadline) {
                            eprintln!("Response stopped at its budget; the received output may be incomplete");
                        }
                        self.model_done = true;
                        if let Some(output) = &mut self.output {
                            output.flush()?;
                        }
                        println!();
                    }
                    AudioEvent::Cancelled { .. } if self.phase != Phase::Recovering => {
                        self.interrupt(false)?;
                    }
                    _ => {}
                },
            }
        }
        Ok(())
    }
    fn finish_output(&mut self) -> Result<()> {
        if let Some(output) = &mut self.output {
            if let Some(error) = output.playback.take_error() {
                eprintln!("Playback backend: {error}");
            }
            self.cancel.check()?;
            output.pump()?;
        }
        if self.phase == Phase::Responding
            && self.model_done
            && self.output.as_ref().is_none_or(Output::drained)
        {
            if let Some(output) = &mut self.output {
                eprintln!(
                    "Playback metrics: {}",
                    serde_json::to_string(&output.playback.metrics())?
                );
                output.stop();
            }
            self.output = None;
            self.phase = Phase::Idle;
            eprintln!("Turn complete; Enter starts another turn");
        }
        Ok(())
    }
    fn loop_run(
        &mut self,
        controls: Receiver<Control>,
        input_selector: &DeviceSelector,
        output_selector: &DeviceSelector,
    ) -> Result<()> {
        loop {
            if self.cancel.is_cancelled() {
                break;
            }
            self.cancel.check()?;
            for _ in 0..8 {
                match controls.try_recv() {
                    Ok(Control::Quit) | Err(TryRecvError::Disconnected) => return Ok(()),
                    Err(TryRecvError::Empty) => break,
                    Ok(Control::Cancel) => self.interrupt(false)?,
                    Ok(Control::Reset) => self.interrupt(true)?,
                    Ok(Control::Toggle) => match self.phase {
                        Phase::Idle => self.start(input_selector, output_selector)?,
                        Phase::Capturing => self.stop_capture(),
                        _ => eprintln!("Turn is finishing; c cancels, r resets, q exits"),
                    },
                    Ok(Control::Help) => {
                        eprintln!("Enter starts/stops capture; c cancels, r resets, q exits")
                    }
                }
            }
            self.finish_output()?;
            self.events()?;
            self.send()?;
            self.capture()?;
            thread::sleep(Duration::from_millis(5));
        }
        Ok(())
    }
}
/// Native live I/O is opt-in at the command and feature level. Device discovery
/// and stream opening are deferred until the user presses Enter.
pub fn run(
    model: Box<dyn RealtimeModel + Send>,
    input_selector: DeviceSelector,
    output_selector: DeviceSelector,
    max_seconds: f64,
    prompt: Option<String>,
    cancel: CancellationToken,
) -> Result<()> {
    ensure!(
        max_seconds.is_finite() && max_seconds > 0. && max_seconds <= 3600.,
        "max-seconds must be finite and in (0,3600]"
    );
    let caps = model.capabilities();
    ensure!(
        prompt.is_none() || caps.text_input,
        "selected realtime model does not accept text context"
    );
    let input = caps.input_format.ok_or_else(|| {
        anyhow::anyhow!("selected realtime adapter does not accept microphone audio")
    })?;
    let input_limit = ((max_seconds * input.sample_rate() as f64).floor() as usize)
        .min(caps.max_input_frames_per_turn.unwrap_or(usize::MAX));
    ensure!(input_limit > 0, "model input budget is empty");
    let limits = SessionLimits {
        max_chunk_frames: input.sample_rate() as usize,
        max_turn_frames: input_limit,
        max_input_messages_per_turn: 8192,
        max_output_frames_per_turn: caps
            .output_format
            .map_or(1, |format| format.sample_rate() as usize * 60),
        max_output_text_bytes_per_turn: 65536,
        max_events_per_turn: 8192,
        ..Default::default()
    };
    eprintln!("Manual turns, model input {:?}, output {:?}, input mode {:?}, output mode {:?}.\nEnter starts/stops capture; c cancels output, r resets conversation, q exits. Hardware remains closed until Enter.",input,caps.output_format,caps.streaming,caps.output_streaming);
    let worker = RealtimeWorker::spawn(
        model,
        limits,
        WorkerOptions {
            command_capacity: 8,
            event_capacity: 4,
            ..Default::default()
        },
    )?;
    let mut run = Run {
        worker,
        commands: PendingCommands::new(input.sample_rate() as usize / 2 + 2048),
        capture: None,
        input_converter: None,
        output: None,
        phase: Phase::Idle,
        waiting: None,
        model_done: false,
        input,
        output_format: caps.output_format,
        max_seconds,
        prompt,
        input_limit,
        source_frames: 0,
        source_limit: 0,
        converted_frames: 0,
        started: None,
        cancel,
    };
    let result = run.loop_run(controls(), &input_selector, &output_selector);
    let mut hardware_failed = false;
    run.report_capture();
    if let Some(capture) = &mut run.capture {
        hardware_failed |= capture.metrics().fatal_errors > 0;
        while let Some(error) = capture.take_error() {
            eprintln!("Capture backend: {error}");
        }
        capture.stop();
    }
    if let Some(output) = &mut run.output {
        hardware_failed |= output.playback.metrics().fatal_errors > 0;
        while let Some(error) = output.playback.take_error() {
            eprintln!("Playback backend: {error}");
        }
        eprintln!(
            "Playback metrics: {}",
            serde_json::to_string(&output.playback.metrics())?
        );
        output.stop();
    }
    run.worker.close()?;
    if hardware_failed && result.is_ok() {
        bail!("native audio device failed; active turn cancelled");
    }
    result
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn manual_controls_are_explicit_and_bounded() {
        assert_eq!(control(b"\r"), Control::Toggle);
        assert_eq!(control(b" c "), Control::Cancel);
        assert_eq!(control(b"R"), Control::Reset);
        assert_eq!(control(b"q"), Control::Quit);
        assert_eq!(control(b"record this text"), Control::Help);
    }
    #[test]
    fn input_backpressure_retains_commands_and_bounds_audio() {
        let format = AudioFormat::mono(16000).unwrap();
        let mut queue = PendingCommands::new(4);
        queue
            .audio(AudioBuffer::new(format, vec![0.; 3]).unwrap())
            .unwrap();
        assert!(queue
            .audio(AudioBuffer::new(format, vec![0.; 2]).unwrap())
            .is_err());
        assert_eq!(queue.audio_frames, 3);
        assert_eq!(queue.commands.len(), 1);
        queue.pop();
        assert_eq!(queue.audio_frames, 0);
        queue.push(Command::Commit).unwrap();
        assert!(matches!(queue.commands.front(), Some(Command::Commit)));
        queue.clear();
        assert!(queue.commands.is_empty());
    }
    #[test]
    fn playback_retry_preserves_exact_interleaved_suffix() {
        let format = AudioFormat::new(48000, 2).unwrap();
        let mut pending = PendingPcm {
            audio: AudioBuffer::new(format, vec![1., 2., 3., 4., 5., 6.]).unwrap(),
            offset: 0,
        };
        assert_eq!(pending.chunk(1).unwrap().unwrap().samples(), [1., 2.]);
        pending.offset = 1;
        assert_eq!(
            pending.chunk(2).unwrap().unwrap().samples(),
            [3., 4., 5., 6.]
        );
        pending.offset = 3;
        assert!(pending.complete());
        assert!(pending.chunk(1).unwrap().is_none());
    }
    #[test]
    fn capture_loss_never_reaches_successful_commit() {
        let mut metrics = live::LiveMetrics::default();
        assert!(capture_integrity(metrics).is_ok());
        metrics.overrun_frames = 1;
        assert!(capture_integrity(metrics).is_err());
        metrics.overrun_frames = 0;
        metrics.backend_xruns = 1;
        assert!(capture_integrity(metrics).is_err());
        metrics.backend_xruns = 0;
        metrics.underrun_frames = 100;
        assert!(capture_integrity(metrics).is_ok());
    }
}

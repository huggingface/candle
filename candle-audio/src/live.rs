//! Optional native capture and playback. Streams are opened in a paused state;
//! only an explicit `start` begins capture/playback. Convert/resample PCM on the
//! application thread, never inside the device callbacks.
//!
//! Callbacks copy complete frames through a bounded lock-free queue, with no
//! allocation, locks, inference, logging or waits in the data callback. Capture
//! drops newest frames on overflow; playback writes silence on underrun. The
//! queue holds at most two seconds, supports up to 32 channels, and is allocated
//! before opening the stream. Cancellation suppresses subsequent callback data;
//! it cannot retract samples already submitted to the hardware.
use crate::*;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Device, FromSample, Sample, SampleFormat, SizedSample, Stream, StreamConfig,
};
use crossbeam_queue::ArrayQueue;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

const MAX_CHANNELS: usize = 32;
type Frame = [f32; MAX_CHANNELS];
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    Input,
    Output,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceSelector {
    #[default]
    Default,
    Id(String),
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub id: String,
    pub name: String,
    pub is_default: bool,
    pub default_format: Option<AudioFormat>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamInfo {
    pub device: DeviceInfo,
    pub format: AudioFormat,
    pub sample_format: String,
}
#[derive(Clone, Debug)]
pub struct LiveOptions {
    pub format: AudioFormat,
    pub capacity_frames: usize,
}
impl LiveOptions {
    pub fn new(format: AudioFormat) -> Self {
        Self {
            format,
            capacity_frames: (format.sample_rate() as usize / 4).max(1),
        }
    }
    fn validate(&self) -> Result<()> {
        if self.format.channels() as usize > MAX_CHANNELS
            || self.capacity_frames == 0
            || self.capacity_frames > self.format.sample_rate() as usize * 2
        {
            return Err(Error::InvalidInput(
                "live audio requires 1–32 channels and a positive queue of at most two seconds"
                    .into(),
            ));
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct LiveMetrics {
    pub input_frames: u64,
    pub output_frames: u64,
    /// Capture frames dropped, or playback frames refused by `write`.
    pub overrun_frames: u64,
    pub underrun_frames: u64,
    pub invalid_samples: u64,
    pub backend_errors: u64,
    pub fatal_errors: u64,
    pub backend_xruns: u64,
    pub device_changes: u64,
    pub queued_frames: usize,
    pub capacity_frames: usize,
    pub disconnected: bool,
    pub cancelled: bool,
}
#[derive(Default)]
struct Counters {
    input: AtomicU64,
    output: AtomicU64,
    overrun: AtomicU64,
    underrun: AtomicU64,
    invalid: AtomicU64,
    errors: AtomicU64,
    fatal: AtomicU64,
    xruns: AtomicU64,
    changes: AtomicU64,
    disconnected: AtomicBool,
}
struct Queue {
    frames: ArrayQueue<Frame>,
    errors: ArrayQueue<String>,
    format: AudioFormat,
    cancel: CancellationToken,
    active: AtomicBool,
    metrics: Counters,
}
impl Queue {
    fn new(options: &LiveOptions, cancel: CancellationToken) -> Result<Arc<Self>> {
        options.validate()?;
        Ok(Arc::new(Self {
            frames: ArrayQueue::new(options.capacity_frames),
            errors: ArrayQueue::new(8),
            format: options.format,
            cancel,
            active: AtomicBool::new(false),
            metrics: Counters::default(),
        }))
    }
    fn snapshot(&self) -> LiveMetrics {
        let m = &self.metrics;
        LiveMetrics {
            input_frames: m.input.load(Ordering::Relaxed),
            output_frames: m.output.load(Ordering::Relaxed),
            overrun_frames: m.overrun.load(Ordering::Relaxed),
            underrun_frames: m.underrun.load(Ordering::Relaxed),
            invalid_samples: m.invalid.load(Ordering::Relaxed),
            backend_errors: m.errors.load(Ordering::Relaxed),
            fatal_errors: m.fatal.load(Ordering::Relaxed),
            backend_xruns: m.xruns.load(Ordering::Relaxed),
            device_changes: m.changes.load(Ordering::Relaxed),
            queued_frames: self.frames.len(),
            capacity_frames: self.frames.capacity(),
            disconnected: m.disconnected.load(Ordering::Relaxed),
            cancelled: self.cancel.is_cancelled(),
        }
    }
    fn clear(&self) {
        for _ in 0..self.frames.capacity() {
            if self.frames.pop().is_none() {
                break;
            }
        }
    }
    fn input<T: Sample>(&self, data: &[T])
    where
        f32: FromSample<T>,
    {
        if self.cancel.is_cancelled() || !self.active.load(Ordering::Acquire) {
            return;
        }
        let channels = self.format.channels() as usize;
        for samples in data.chunks_exact(channels) {
            let mut frame = [0f32; MAX_CHANNELS];
            for (dst, src) in frame.iter_mut().zip(samples) {
                let v = f32::from_sample(*src);
                *dst = if v.is_finite() {
                    v
                } else {
                    self.metrics.invalid.fetch_add(1, Ordering::Relaxed);
                    0.
                };
            }
            if self.frames.push(frame).is_err() {
                self.metrics.overrun.fetch_add(1, Ordering::Relaxed);
            } else {
                self.metrics.input.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    fn output<T: Sample + FromSample<f32>>(&self, data: &mut [T]) {
        if self.cancel.is_cancelled() || !self.active.load(Ordering::Acquire) {
            data.fill(T::from_sample(0.));
            return;
        }
        let channels = self.format.channels() as usize;
        for samples in data.chunks_exact_mut(channels) {
            let frame = match self.frames.pop() {
                Some(frame) => {
                    self.metrics.output.fetch_add(1, Ordering::Relaxed);
                    frame
                }
                None => {
                    self.metrics.underrun.fetch_add(1, Ordering::Relaxed);
                    [0.; MAX_CHANNELS]
                }
            };
            for (dst, src) in samples.iter_mut().zip(frame) {
                *dst = T::from_sample(src.clamp(-1., 1.));
            }
        }
        // Backends normally supply whole frames; sanitize any incomplete tail.
        let whole_samples = data.len() / channels * channels;
        for value in &mut data[whole_samples..] {
            *value = T::from_sample(0.);
        }
    }
    fn backend_error(&self, error: cpal::Error) {
        self.metrics.errors.fetch_add(1, Ordering::Relaxed);
        match error.kind() {
            cpal::ErrorKind::DeviceChanged => {
                self.metrics.changes.fetch_add(1, Ordering::Relaxed);
            }
            cpal::ErrorKind::Xrun => {
                self.metrics.xruns.fetch_add(1, Ordering::Relaxed);
            }
            cpal::ErrorKind::RealtimeDenied => {}
            cpal::ErrorKind::DeviceNotAvailable | cpal::ErrorKind::HostUnavailable => {
                self.metrics.fatal.fetch_add(1, Ordering::Relaxed);
                self.metrics.disconnected.store(true, Ordering::Release);
                self.cancel.cancel();
            }
            _ => {
                self.metrics.fatal.fetch_add(1, Ordering::Relaxed);
                self.cancel.cancel();
            }
        }
        // This is the backend's error callback, separate from its audio callback.
        let mut message = error.to_string();
        if message.len() > 1024 {
            let mut end = 1024;
            while !message.is_char_boundary(end) {
                end -= 1;
            }
            message.truncate(end);
        }
        let _ = self.errors.push(message);
    }
    fn read(&self, max_frames: usize) -> Result<Option<AudioBuffer>> {
        self.cancel.check()?;
        if max_frames == 0 || max_frames > self.frames.capacity() {
            return Err(Error::InvalidInput(
                "capture read size must fit the configured queue".into(),
            ));
        }
        let mut samples = Vec::with_capacity(max_frames * self.format.channels() as usize);
        for _ in 0..max_frames {
            match self.frames.pop() {
                Some(frame) => samples.extend_from_slice(&frame[..self.format.channels() as usize]),
                None => break,
            }
        }
        if samples.is_empty() {
            Ok(None)
        } else {
            AudioBuffer::new(self.format, samples).map(Some)
        }
    }
    fn write(&self, audio: &AudioBuffer) -> Result<usize> {
        self.cancel.check()?;
        if audio.format() != self.format {
            return Err(Error::InvalidInput(
                "playback format mismatch; convert PCM outside the callback".into(),
            ));
        }
        let channels = self.format.channels() as usize;
        let mut accepted = 0;
        for samples in audio.samples().chunks_exact(channels) {
            let mut frame = [0.; MAX_CHANNELS];
            frame[..channels].copy_from_slice(samples);
            if self.frames.push(frame).is_err() {
                break;
            }
            accepted += 1;
        }
        self.metrics
            .overrun
            .fetch_add((audio.frames() - accepted) as u64, Ordering::Relaxed);
        Ok(accepted)
    }
}
fn device_error(error: impl std::fmt::Display) -> Error {
    Error::InvalidState(format!("native audio device: {error}"))
}
fn default_device(host: &cpal::Host, direction: Direction) -> Option<Device> {
    match direction {
        Direction::Input => host.default_input_device(),
        Direction::Output => host.default_output_device(),
    }
}
fn default_config(
    device: &Device,
    direction: Direction,
) -> std::result::Result<cpal::SupportedStreamConfig, cpal::Error> {
    match direction {
        Direction::Input => device.default_input_config(),
        Direction::Output => device.default_output_config(),
    }
}
fn device_info(device: &Device, is_default: bool, direction: Direction) -> Result<DeviceInfo> {
    Ok(DeviceInfo {
        id: device.id().map_err(device_error)?.to_string(),
        name: device.to_string(),
        is_default,
        default_format: default_config(device, direction)
            .ok()
            .and_then(|config| AudioFormat::new(config.sample_rate(), config.channels()).ok()),
    })
}
/// Only queries device metadata; never opens a stream or starts recording.
pub fn enumerate_devices(direction: Direction) -> Result<Vec<DeviceInfo>> {
    let host = cpal::default_host();
    let default = default_device(&host, direction);
    host.devices()
        .map_err(device_error)?
        .filter(|device| match direction {
            Direction::Input => device.supports_input(),
            Direction::Output => device.supports_output(),
        })
        .map(|device| device_info(&device, default.as_ref() == Some(&device), direction))
        .collect()
}
fn select(selector: &DeviceSelector, direction: Direction) -> Result<Device> {
    let host = cpal::default_host();
    match selector {
        DeviceSelector::Default => default_device(&host, direction),
        DeviceSelector::Id(id) => {
            host.device_by_id(&id.parse::<cpal::DeviceId>().map_err(device_error)?)
        }
    }
    .ok_or_else(|| Error::Unsupported("selected native audio device is unavailable".into()))
}
fn supported(
    device: &Device,
    direction: Direction,
) -> Result<Vec<cpal::SupportedStreamConfigRange>> {
    match direction {
        Direction::Input => device.supported_input_configs().map(|v| v.collect()),
        Direction::Output => device.supported_output_configs().map(|v| v.collect()),
    }
    .map_err(device_error)
}
fn config(
    device: &Device,
    direction: Direction,
    format: AudioFormat,
) -> Result<cpal::SupportedStreamConfig> {
    let mut candidates = supported(device, direction)?;
    candidates.sort_by_key(|c| match c.sample_format() {
        SampleFormat::F32 => 0,
        SampleFormat::I16 => 1,
        SampleFormat::U16 => 2,
        _ => 3,
    });
    candidates.into_iter().find(|c|c.channels()==format.channels() && c.min_sample_rate()<=format.sample_rate() && c.max_sample_rate()>=format.sample_rate()).map(|c|c.with_sample_rate(format.sample_rate())).ok_or_else(||Error::Unsupported("device does not support requested PCM format; call resolve_format and convert PCM explicitly".into()))
}
/// Prefer a requested format when the device supports it, otherwise return its
/// native default. The caller must convert PCM to/from the returned format.
pub fn resolve_format(
    selector: &DeviceSelector,
    direction: Direction,
    requested: Option<AudioFormat>,
) -> Result<AudioFormat> {
    let device = select(selector, direction)?;
    if let Some(format) = requested {
        if config(&device, direction, format).is_ok() {
            return Ok(format);
        }
    }
    let cfg = default_config(&device, direction).map_err(device_error)?;
    AudioFormat::new(cfg.sample_rate(), cfg.channels())
}
fn input_stream<T: SizedSample>(
    device: &Device,
    config: StreamConfig,
    queue: &Arc<Queue>,
) -> Result<Stream>
where
    f32: FromSample<T>,
{
    let data = queue.clone();
    let errors = queue.clone();
    device
        .build_input_stream(
            config,
            move |samples: &[T], _| data.input(samples),
            move |error| errors.backend_error(error),
            None,
        )
        .map_err(device_error)
}
fn output_stream<T: SizedSample + FromSample<f32>>(
    device: &Device,
    config: StreamConfig,
    queue: &Arc<Queue>,
) -> Result<Stream> {
    let data = queue.clone();
    let errors = queue.clone();
    device
        .build_output_stream(
            config,
            move |samples: &mut [T], _| data.output(samples),
            move |error| errors.backend_error(error),
            None,
        )
        .map_err(device_error)
}
fn open(
    selector: &DeviceSelector,
    options: LiveOptions,
    cancel: CancellationToken,
    direction: Direction,
) -> Result<(Stream, Arc<Queue>, StreamInfo)> {
    let queue = Queue::new(&options, cancel)?;
    queue.cancel.check()?;
    let device = select(selector, direction)?;
    let config = config(&device, direction, options.format)?;
    let sample_format = config.sample_format();
    let info = StreamInfo {
        device: device_info(
            &device,
            matches!(selector, DeviceSelector::Default),
            direction,
        )?,
        format: options.format,
        sample_format: sample_format.to_string(),
    };
    macro_rules! stream {
        ($ty:ty) => {
            match direction {
                Direction::Input => input_stream::<$ty>(&device, config.into(), &queue),
                Direction::Output => output_stream::<$ty>(&device, config.into(), &queue),
            }
        };
    }
    let stream = match sample_format {
        SampleFormat::F32 => stream!(f32),
        SampleFormat::F64 => stream!(f64),
        SampleFormat::I8 => stream!(i8),
        SampleFormat::I16 => stream!(i16),
        SampleFormat::I24 => stream!(cpal::I24),
        SampleFormat::I32 => stream!(i32),
        SampleFormat::I64 => stream!(i64),
        SampleFormat::U8 => stream!(u8),
        SampleFormat::U16 => stream!(u16),
        SampleFormat::U32 => stream!(u32),
        SampleFormat::U64 => stream!(u64),
        other => {
            return Err(Error::Unsupported(format!(
                "unsupported native audio sample type {other}"
            )))
        }
    }?;
    Ok((stream, queue, info))
}
/// Opens paused. `start` explicitly enables microphone capture.
pub struct LiveCapture {
    stream: Option<Stream>,
    queue: Arc<Queue>,
    info: StreamInfo,
}
impl LiveCapture {
    pub fn open(
        selector: &DeviceSelector,
        options: LiveOptions,
        cancel: CancellationToken,
    ) -> Result<Self> {
        let (stream, queue, info) = open(selector, options, cancel, Direction::Input)?;
        Ok(Self {
            stream: Some(stream),
            queue,
            info,
        })
    }
    pub fn start(&self) -> Result<()> {
        self.queue.cancel.check()?;
        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| Error::InvalidState("capture stream is stopped".into()))?;
        self.queue.active.store(true, Ordering::Release);
        if let Err(error) = stream.play() {
            self.queue.active.store(false, Ordering::Release);
            return Err(device_error(error));
        }
        Ok(())
    }
    pub fn read(&self, max_frames: usize) -> Result<Option<AudioBuffer>> {
        self.queue.read(max_frames)
    }
    pub fn metrics(&self) -> LiveMetrics {
        self.queue.snapshot()
    }
    pub fn info(&self) -> &StreamInfo {
        &self.info
    }
    pub fn take_error(&self) -> Option<String> {
        self.queue.errors.pop()
    }
    pub fn clear(&self) {
        self.queue.clear();
    }
    /// Stop hardware capture; queued frames remain available to drain with read.
    pub fn stop(&mut self) {
        self.queue.active.store(false, Ordering::Release);
        self.stream.take();
    }
}
impl Drop for LiveCapture {
    fn drop(&mut self) {
        self.stop();
        self.queue.clear();
    }
}
/// Opens paused. Prebuffer PCM with `write` before calling `start`.
pub struct LivePlayback {
    stream: Option<Stream>,
    queue: Arc<Queue>,
    info: StreamInfo,
}
impl LivePlayback {
    pub fn open(
        selector: &DeviceSelector,
        options: LiveOptions,
        cancel: CancellationToken,
    ) -> Result<Self> {
        let (stream, queue, info) = open(selector, options, cancel, Direction::Output)?;
        Ok(Self {
            stream: Some(stream),
            queue,
            info,
        })
    }
    pub fn start(&self) -> Result<()> {
        self.queue.cancel.check()?;
        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| Error::InvalidState("playback stream is stopped".into()))?;
        self.queue.active.store(true, Ordering::Release);
        if let Err(error) = stream.play() {
            self.queue.active.store(false, Ordering::Release);
            return Err(device_error(error));
        }
        Ok(())
    }
    /// Returns the number of whole frames accepted without waiting. Retain and
    /// retry any unaccepted suffix on the application thread when appropriate.
    pub fn write(&self, audio: &AudioBuffer) -> Result<usize> {
        if self.stream.is_none() {
            return Err(Error::InvalidState("playback stream is stopped".into()));
        }
        self.queue.write(audio)
    }
    pub fn metrics(&self) -> LiveMetrics {
        self.queue.snapshot()
    }
    pub fn info(&self) -> &StreamInfo {
        &self.info
    }
    pub fn take_error(&self) -> Option<String> {
        self.queue.errors.pop()
    }
    pub fn clear(&self) {
        self.queue.clear();
    }
    pub fn stop(&mut self) {
        self.queue.active.store(false, Ordering::Release);
        self.stream.take();
        self.queue.clear();
    }
}
impl Drop for LivePlayback {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn queue(channels: u16, capacity: usize) -> Arc<Queue> {
        let q = Queue::new(
            &LiveOptions {
                format: AudioFormat::new(48000, channels).unwrap(),
                capacity_frames: capacity,
            },
            CancellationToken::default(),
        )
        .unwrap();
        q.active.store(true, Ordering::Release);
        q
    }
    #[test]
    fn capture_drops_whole_frames_and_preserves_channel_order() {
        let q = queue(2, 2);
        q.input(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6]);
        assert_eq!(q.snapshot().overrun_frames, 1);
        let pcm = q.read(2).unwrap().unwrap();
        assert_eq!(pcm.samples(), [0.1, 0.2, 0.3, 0.4]);
        assert_eq!(q.snapshot().input_frames, 2);
        assert!(q.read(2).unwrap().is_none());
    }
    #[test]
    fn playback_is_bounded_fills_silence_and_clears_cancelled_audio() {
        let q = queue(2, 2);
        let pcm = AudioBuffer::new(q.format, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
        assert_eq!(q.write(&pcm).unwrap(), 2);
        let mut output = [9f32; 6];
        q.output(&mut output);
        assert_eq!(output, [0.1, 0.2, 0.3, 0.4, 0., 0.]);
        assert_eq!(q.snapshot().overrun_frames, 1);
        assert_eq!(q.snapshot().underrun_frames, 1);
        assert_eq!(q.snapshot().output_frames, 2);
        q.write(&pcm).unwrap();
        q.cancel.cancel();
        q.output(&mut output);
        assert_eq!(output, [0.; 6]);
        assert!(matches!(q.read(1), Err(Error::Cancelled)));
        assert!(matches!(q.write(&pcm), Err(Error::Cancelled)));
        q.clear();
        assert_eq!(q.snapshot().queued_frames, 0);
    }
    #[test]
    fn capture_sanitizes_nonfinite_and_converts_integer_pcm() {
        let q = queue(1, 4);
        q.input(&[f32::NAN, f32::INFINITY]);
        assert_eq!(q.read(2).unwrap().unwrap().samples(), [0., 0.]);
        assert_eq!(q.snapshot().invalid_samples, 2);
        q.input(&[i16::MIN, i16::MAX]);
        let values = q.read(2).unwrap().unwrap();
        assert_eq!(values.samples()[0], -1.);
        assert!(values.samples()[1] > 0.99);
    }
    #[test]
    fn native_queues_reject_unbounded_memory_and_format_mismatch() {
        let mut options = LiveOptions::new(AudioFormat::mono(48000).unwrap());
        options.capacity_frames = 96001;
        assert!(Queue::new(&options, CancellationToken::default()).is_err());
        let q = queue(1, 2);
        assert!(q
            .write(&AudioBuffer::new(AudioFormat::mono(16000).unwrap(), vec![0.]).unwrap())
            .is_err());
        assert!(q.read(0).is_err());
        assert!(q.read(3).is_err());
    }
    #[test]
    fn paused_callbacks_neither_capture_nor_consume() {
        let q = queue(1, 2);
        q.active.store(false, Ordering::Release);
        q.input(&[0.5f32]);
        assert_eq!(q.snapshot().queued_frames, 0);
        q.write(&AudioBuffer::new(q.format, vec![0.5]).unwrap())
            .unwrap();
        let mut out = [1f32];
        q.output(&mut out);
        assert_eq!(out, [0.]);
        assert_eq!(q.snapshot().queued_frames, 1);
        q.active.store(true, Ordering::Release);
        q.output(&mut out);
        assert_eq!(out, [0.5]);
    }
    #[test]
    fn backend_xruns_are_nonfatal_and_disconnect_cancels() {
        let q = queue(1, 2);
        q.backend_error(cpal::Error::new(cpal::ErrorKind::Xrun));
        q.backend_error(cpal::Error::new(cpal::ErrorKind::DeviceChanged));
        assert_eq!(q.snapshot().backend_xruns, 1);
        assert_eq!(q.snapshot().device_changes, 1);
        assert!(!q.cancel.is_cancelled());
        q.backend_error(cpal::Error::new(cpal::ErrorKind::DeviceNotAvailable));
        assert!(q.snapshot().disconnected);
        assert!(q.cancel.is_cancelled());
        assert_eq!(q.snapshot().backend_errors, 3);
        for _ in 0..20 {
            q.backend_error(cpal::Error::new(cpal::ErrorKind::Xrun));
        }
        assert_eq!(q.errors.len(), 8);
    }
}

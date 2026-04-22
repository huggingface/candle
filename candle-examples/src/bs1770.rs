// Copied from https://github.com/ruuda/bs1770/blob/master/src/lib.rs
// BS1770 -- Loudness analysis library conforming to ITU-R BS.1770
// Copyright 2020 Ruud van Asseldonk

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// A copy of the License has been included in the root of the repository.

//! Loudness analysis conforming to [ITU-R BS.1770-4][bs17704].
//!
//! This library offers the building blocks to perform BS.1770 loudness
//! measurements, but you need to put the pieces together yourself.
//!
//! [bs17704]: https://www.itu.int/rec/R-REC-BS.1770-4-201510-I/en
//!
//! # Stereo integrated loudness example
//!
//! ```ignore
//! # fn load_stereo_audio() -> [Vec<i16>; 2] {
//! #     [vec![0; 48_000], vec![0; 48_000]]
//! # }
//! #
//! let sample_rate_hz = 44_100;
//! let bits_per_sample = 16;
//! let channel_samples: [Vec<i16>; 2] = load_stereo_audio();
//!
//! // When converting integer samples to float, note that the maximum amplitude
//! // is `1 << (bits_per_sample - 1)`, one bit is the sign bit.
//! let normalizer = 1.0 / (1_u64 << (bits_per_sample - 1)) as f32;
//!
//! let channel_power: Vec<_> = channel_samples.iter().map(|samples| {
//!     let mut meter = bs1770::ChannelLoudnessMeter::new(sample_rate_hz);
//!     meter.push(samples.iter().map(|&s| s as f32 * normalizer));
//!     meter.into_100ms_windows()
//! }).collect();
//!
//! let stereo_power = bs1770::reduce_stereo(
//!     channel_power[0].as_ref(),
//!     channel_power[1].as_ref(),
//! );
//!
//! let gated_power = bs1770::gated_mean(
//!     stereo_power.as_ref()
//! ).unwrap_or(bs1770::Power(0.0));
//! println!("Integrated loudness: {:.1} LUFS", gated_power.loudness_lkfs());
//! ```

use std::f32;

/// Coefficients for a 2nd-degree infinite impulse response filter.
///
/// Coefficient a0 is implicitly 1.0.
#[derive(Clone)]
struct Filter {
    a1: f32,
    a2: f32,
    b0: f32,
    b1: f32,
    b2: f32,

    // The past two input and output samples.
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl Filter {
    /// Stage 1 of th BS.1770-4 pre-filter.
    pub fn high_shelf(sample_rate_hz: f32) -> Filter {
        // Coefficients taken from https://github.com/csteinmetz1/pyloudnorm/blob/
        // 6baa64d59b7794bc812e124438692e7fd2e65c0c/pyloudnorm/meter.py#L135-L136.
        let gain_db = 3.999_843_8;
        let q = 0.707_175_25;
        let center_hz = 1_681.974_5;

        // Formula taken from https://github.com/csteinmetz1/pyloudnorm/blob/
        // 6baa64d59b7794bc812e124438692e7fd2e65c0c/pyloudnorm/iirfilter.py#L134-L143.
        let k = (f32::consts::PI * center_hz / sample_rate_hz).tan();
        let vh = 10.0_f32.powf(gain_db / 20.0);
        let vb = vh.powf(0.499_666_78);
        let a0 = 1.0 + k / q + k * k;
        Filter {
            b0: (vh + vb * k / q + k * k) / a0,
            b1: 2.0 * (k * k - vh) / a0,
            b2: (vh - vb * k / q + k * k) / a0,
            a1: 2.0 * (k * k - 1.0) / a0,
            a2: (1.0 - k / q + k * k) / a0,

            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Stage 2 of th BS.1770-4 pre-filter.
    pub fn high_pass(sample_rate_hz: f32) -> Filter {
        // Coefficients taken from https://github.com/csteinmetz1/pyloudnorm/blob/
        // 6baa64d59b7794bc812e124438692e7fd2e65c0c/pyloudnorm/meter.py#L135-L136.
        let q = 0.500_327_05;
        let center_hz = 38.135_47;

        // Formula taken from https://github.com/csteinmetz1/pyloudnorm/blob/
        // 6baa64d59b7794bc812e124438692e7fd2e65c0c/pyloudnorm/iirfilter.py#L145-L151
        let k = (f32::consts::PI * center_hz / sample_rate_hz).tan();
        Filter {
            a1: 2.0 * (k * k - 1.0) / (1.0 + k / q + k * k),
            a2: (1.0 - k / q + k * k) / (1.0 + k / q + k * k),
            b0: 1.0,
            b1: -2.0,
            b2: 1.0,

            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Feed the next input sample, get the next output sample.
    #[inline(always)]
    pub fn apply(&mut self, x0: f32) -> f32 {
        let y0 = 0.0 + self.b0 * x0 + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;

        self.x2 = self.x1;
        self.x1 = x0;
        self.y2 = self.y1;
        self.y1 = y0;

        y0
    }
}

/// Compensated sum, for summing many values of different orders of magnitude
/// accurately.
#[derive(Copy, Clone, PartialEq)]
struct Sum {
    sum: f32,
    residue: f32,
}

impl Sum {
    #[inline(always)]
    fn zero() -> Sum {
        Sum {
            sum: 0.0,
            residue: 0.0,
        }
    }

    #[inline(always)]
    fn add(&mut self, x: f32) {
        let sum = self.sum + (self.residue + x);
        self.residue = (self.residue + x) - (sum - self.sum);
        self.sum = sum;
    }
}

/// The mean of the squares of the K-weighted samples in a window of time.
///
/// K-weighted power is equivalent to K-weighted loudness, the only difference
/// is one of scale: power is quadratic in sample amplitudes, whereas loudness
/// units are logarithmic. `loudness_lkfs` and `from_lkfs` convert between power,
/// and K-weighted Loudness Units relative to nominal Full Scale (LKFS).
///
/// The term “LKFS” (Loudness Units, K-Weighted, relative to nominal Full Scale)
/// is used in BS.1770-4 to emphasize K-weighting, but the term is otherwise
/// interchangeable with the more widespread term “LUFS” (Loudness Units,
/// relative to Full Scale). Loudness units are related to decibels in the
/// following sense: boosting a signal that has a loudness of
/// -<var>L<sub>K</sub></var> LUFS by <var>L<sub>K</sub></var> dB (by
/// multiplying the amplitude by 10<sup><var>L<sub>K</sub></var>/20</sup>) will
/// bring the loudness to 0 LUFS.
///
/// K-weighting refers to a high-shelf and high-pass filter that model the
/// effect that humans perceive a certain amount of power in low frequencies to
/// be less loud than the same amount of power in higher frequencies. In this
/// library the `Power` type is used exclusively to refer to power after applying K-weighting.
///
/// The nominal “full scale” is the range [-1.0, 1.0]. Because the power is the
/// mean square of the samples, if no input samples exceeded the full scale, the
/// power will be in the range [0.0, 1.0]. However, the power delivered by
/// multiple channels, which is a weighted sum over individual channel powers,
/// can exceed this range, because the weighted sum is not normalized.
#[derive(Copy, Clone, PartialEq, PartialOrd)]
pub struct Power(pub f32);

impl Power {
    /// Convert Loudness Units relative to Full Scale into a squared sample amplitude.
    ///
    /// This is the inverse of `loudness_lkfs`.
    pub fn from_lkfs(lkfs: f32) -> Power {
        // The inverse of the formula below.
        Power(10.0_f32.powf((lkfs + 0.691) * 0.1))
    }

    /// Return the loudness of this window in Loudness Units, K-weighted, relative to Full Scale.
    ///
    /// This is the inverse of `from_lkfs`.
    pub fn loudness_lkfs(&self) -> f32 {
        // Equation 2 (p.5) of BS.1770-4.
        -0.691 + 10.0 * self.0.log10()
    }
}

/// A `T` value for non-overlapping windows of audio, 100ms in length.
///
/// The `ChannelLoudnessMeter` applies K-weighting and then produces the power
/// for non-overlapping windows of 100ms duration.
///
/// These non-overlapping 100ms windows can later be combined into overlapping
/// windows of 400ms, spaced 100ms apart, to compute instantaneous loudness or
/// to perform a gated measurement, or they can be combined into even larger
/// windows for a momentary loudness measurement.
#[derive(Copy, Clone, Debug)]
pub struct Windows100ms<T> {
    pub inner: T,
}

impl<T> Windows100ms<T> {
    /// Wrap a new empty vector.
    pub fn new() -> Windows100ms<Vec<T>> {
        Windows100ms { inner: Vec::new() }
    }

    /// Apply `as_ref` to the inner value.
    pub fn as_ref(&self) -> Windows100ms<&[Power]>
    where
        T: AsRef<[Power]>,
    {
        Windows100ms {
            inner: self.inner.as_ref(),
        }
    }

    /// Apply `as_mut` to the inner value.
    pub fn as_mut(&mut self) -> Windows100ms<&mut [Power]>
    where
        T: AsMut<[Power]>,
    {
        Windows100ms {
            inner: self.inner.as_mut(),
        }
    }

    #[allow(clippy::len_without_is_empty)]
    /// Apply `len` to the inner value.
    pub fn len(&self) -> usize
    where
        T: AsRef<[Power]>,
    {
        self.inner.as_ref().len()
    }
}

/// Measures K-weighted power of non-overlapping 100ms windows of a single channel of audio.
///
/// # Output
///
/// The output of the meter is an intermediate result in the form of power for
/// 100ms non-overlapping windows. The windows need to be processed further to
/// get one of the instantaneous, momentary, and integrated loudness
/// measurements defined in BS.1770.
///
/// The windows can also be inspected directly; the data is meaningful
/// on its own (the K-weighted power delivered in that window of time), but it
/// is not something that BS.1770 defines a term for.
///
/// # Multichannel audio
///
/// To perform a loudness measurement of multichannel audio, construct a
/// `ChannelLoudnessMeter` per channel, and later combine the measured power
/// with e.g. `reduce_stereo`.
///
/// # Instantaneous loudness
///
/// The instantaneous loudness is the power over a 400ms window, so you can
/// average four 100ms windows. No special functionality is implemented to help
/// with that at this time. ([Pull requests would be accepted.][contribute])
///
/// # Momentary loudness
///
/// The momentary loudness is the power over a 3-second window, so you can
/// average thirty 100ms windows. No special functionality is implemented to
/// help with that at this time. ([Pull requests would be accepted.][contribute])
///
/// # Integrated loudness
///
/// Use `gated_mean` to perform an integrated loudness measurement:
///
/// ```ignore
/// # use std::iter;
/// # use bs1770::{ChannelLoudnessMeter, gated_mean};
/// # let sample_rate_hz = 44_100;
/// # let samples_per_100ms = sample_rate_hz / 10;
/// # let mut meter = ChannelLoudnessMeter::new(sample_rate_hz);
/// # meter.push((0..44_100).map(|i| (i as f32 * 0.01).sin()));
/// let integrated_loudness_lkfs = gated_mean(meter.as_100ms_windows())
///     .unwrap_or(bs1770::Power(0.0))
///     .loudness_lkfs();
/// ```
///
/// [contribute]: https://github.com/ruuda/bs1770/blob/master/CONTRIBUTING.md
#[derive(Clone)]
pub struct ChannelLoudnessMeter {
    /// The number of samples that fit in 100ms of audio.
    samples_per_100ms: u32,

    /// Stage 1 filter (head effects, high shelf).
    filter_stage1: Filter,

    /// Stage 2 filter (high-pass).
    filter_stage2: Filter,

    /// Sum of the squares over non-overlapping windows of 100ms.
    windows: Windows100ms<Vec<Power>>,

    /// The number of samples in the current unfinished window.
    count: u32,

    /// The sum of the squares of the samples in the current unfinished window.
    square_sum: Sum,
}

impl ChannelLoudnessMeter {
    /// Construct a new loudness meter for the given sample rate.
    pub fn new(sample_rate_hz: u32) -> ChannelLoudnessMeter {
        ChannelLoudnessMeter {
            samples_per_100ms: sample_rate_hz / 10,
            filter_stage1: Filter::high_shelf(sample_rate_hz as f32),
            filter_stage2: Filter::high_pass(sample_rate_hz as f32),
            windows: Windows100ms::new(),
            count: 0,
            square_sum: Sum::zero(),
        }
    }

    /// Feed input samples for loudness analysis.
    ///
    /// # Full scale
    ///
    /// Full scale for the input samples is the interval [-1.0, 1.0]. If your
    /// input consists of signed integer samples, you can convert as follows:
    ///
    /// ```ignore
    /// # let mut meter = bs1770::ChannelLoudnessMeter::new(44_100);
    /// # let bits_per_sample = 16_usize;
    /// # let samples = &[0_i16];
    /// // Note that the maximum amplitude is `1 << (bits_per_sample - 1)`,
    /// // one bit is the sign bit.
    /// let normalizer = 1.0 / (1_u64 << (bits_per_sample - 1)) as f32;
    /// meter.push(samples.iter().map(|&s| s as f32 * normalizer));
    /// ```
    ///
    /// # Repeated calls
    ///
    /// You can call `push` multiple times to feed multiple batches of samples.
    /// This is equivalent to feeding a single chained iterator. The leftover of
    /// samples that did not fill a full 100ms window is not discarded:
    ///
    /// ```ignore
    /// # use std::iter;
    /// # use bs1770::ChannelLoudnessMeter;
    /// let sample_rate_hz = 44_100;
    /// let samples_per_100ms = sample_rate_hz / 10;
    /// let mut meter = ChannelLoudnessMeter::new(sample_rate_hz);
    ///
    /// meter.push(iter::repeat(0.0).take(samples_per_100ms as usize - 1));
    /// assert_eq!(meter.as_100ms_windows().len(), 0);
    ///
    /// meter.push(iter::once(0.0));
    /// assert_eq!(meter.as_100ms_windows().len(), 1);
    /// ```
    pub fn push<I: Iterator<Item = f32>>(&mut self, samples: I) {
        let normalizer = 1.0 / self.samples_per_100ms as f32;

        // LLVM, if you could go ahead and inline those apply calls, and then
        // unroll and vectorize the loop, that'd be terrific.
        for x in samples {
            let y = self.filter_stage1.apply(x);
            let z = self.filter_stage2.apply(y);

            self.square_sum.add(z * z);
            self.count += 1;

            // TODO: Should this branch be marked cold?
            if self.count == self.samples_per_100ms {
                let mean_squares = Power(self.square_sum.sum * normalizer);
                self.windows.inner.push(mean_squares);
                // We intentionally do not reset the residue. That way, leftover
                // energy from this window is not lost, so for the file overall,
                // the sum remains more accurate.
                self.square_sum.sum = 0.0;
                self.count = 0;
            }
        }
    }

    /// Return a reference to the 100ms windows analyzed so far.
    pub fn as_100ms_windows(&self) -> Windows100ms<&[Power]> {
        self.windows.as_ref()
    }

    /// Return all 100ms windows analyzed so far.
    pub fn into_100ms_windows(self) -> Windows100ms<Vec<Power>> {
        self.windows
    }
}

/// Combine power for multiple channels by taking a weighted sum.
///
/// Note that BS.1770-4 defines power for a multi-channel signal as a weighted
/// sum over channels which is not normalized. This means that a stereo signal
/// is inherently louder than a mono signal. For a mono signal played back on
/// stereo speakers, you should therefore still apply `reduce_stereo`, passing
/// in the same signal for both channels.
pub fn reduce_stereo(
    left: Windows100ms<&[Power]>,
    right: Windows100ms<&[Power]>,
) -> Windows100ms<Vec<Power>> {
    assert_eq!(
        left.len(),
        right.len(),
        "Channels must have the same length."
    );
    let mut result = Vec::with_capacity(left.len());
    for (l, r) in left.inner.iter().zip(right.inner) {
        result.push(Power(l.0 + r.0));
    }
    Windows100ms { inner: result }
}

/// In-place version of `reduce_stereo` that stores the result in the former left channel.
pub fn reduce_stereo_in_place(left: Windows100ms<&mut [Power]>, right: Windows100ms<&[Power]>) {
    assert_eq!(
        left.len(),
        right.len(),
        "Channels must have the same length."
    );
    for (l, r) in left.inner.iter_mut().zip(right.inner) {
        l.0 += r.0;
    }
}

/// Perform gating and averaging for a BS.1770-4 integrated loudness measurement.
///
/// The integrated loudness measurement is not just the average power over the
/// entire signal. BS.1770-4 defines two stages of gating that exclude
/// parts of the signal, to ensure that silent parts do not contribute to the
/// loudness measurement. This function performs that gating, and returns the
/// average power over the windows that were not excluded.
///
/// The result of this function is the integrated loudness measurement.
///
/// When no signal remains after applying the gate, this function returns
/// `None`. In particular, this happens when all of the signal is softer than
/// -70 LKFS, including a signal that consists of pure silence.
pub fn gated_mean(windows_100ms: Windows100ms<&[Power]>) -> Option<Power> {
    let mut gating_blocks = Vec::with_capacity(windows_100ms.len());

    // Stage 1: an absolute threshold of -70 LKFS. (Equation 6, p.6.)
    let absolute_threshold = Power::from_lkfs(-70.0);

    // Iterate over all 400ms windows.
    for window in windows_100ms.inner.windows(4) {
        // Note that the sum over channels has already been performed at this point.
        let gating_block_power = Power(0.25 * window.iter().map(|mean| mean.0).sum::<f32>());

        if gating_block_power > absolute_threshold {
            gating_blocks.push(gating_block_power);
        }
    }

    if gating_blocks.is_empty() {
        return None;
    }

    // Compute the loudness after applying the absolute gate, in order to
    // determine the threshold for the relative gate.
    let mut sum_power = Sum::zero();
    for &gating_block_power in &gating_blocks {
        sum_power.add(gating_block_power.0);
    }
    let absolute_gated_power = Power(sum_power.sum / (gating_blocks.len() as f32));

    // Stage 2: Apply the relative gate.
    let relative_threshold = Power::from_lkfs(absolute_gated_power.loudness_lkfs() - 10.0);
    let mut sum_power = Sum::zero();
    let mut n_blocks = 0_usize;
    for &gating_block_power in &gating_blocks {
        if gating_block_power > relative_threshold {
            sum_power.add(gating_block_power.0);
            n_blocks += 1;
        }
    }

    if n_blocks == 0 {
        return None;
    }

    let relative_gated_power = Power(sum_power.sum / n_blocks as f32);
    Some(relative_gated_power)
}

//! Mel-spectrogram front-end, equivalent to `torchaudio.transforms.MelSpectrogram`
//! with `power=1.0`, `center=true`, reflect padding and `win_length == n_fft`.
//!
//! The Hann window and the (HTK, un-normalized) mel filterbank are not computed
//! here: MuScriptor checkpoints ship both as buffers, and they are loaded from
//! the weights so the numerics match the reference exactly. The FFT and the
//! filterbank application accumulate in f64 (the cost is negligible and it
//! keeps the roundoff well below the reference implementation's own f32
//! noise floor).

/// In-place iterative radix-2 Cooley-Tukey FFT.
fn fft_inplace(re: &mut [f64], im: &mut [f64], twiddle_re: &[f64], twiddle_im: &[f64]) {
    let n = re.len();
    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 0..n {
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
    }
    let mut len = 2;
    while len <= n {
        let step = n / len;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let w_re = twiddle_re[k * step];
                let w_im = twiddle_im[k * step];
                let a = start + k;
                let b = start + k + len / 2;
                let t_re = re[b] * w_re - im[b] * w_im;
                let t_im = re[b] * w_im + im[b] * w_re;
                re[b] = re[a] - t_re;
                im[b] = im[a] - t_im;
                re[a] += t_re;
                im[a] += t_im;
            }
        }
        len <<= 1;
    }
}

#[derive(Debug, Clone)]
pub struct MelSpectrogram {
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    window: Vec<f64>,
    /// Mel filterbank, transposed to `[n_mels, n_fft / 2 + 1]` row-major so
    /// each mel bin reduces over a contiguous frequency slice.
    fb_t: Vec<f64>,
    twiddle_re: Vec<f64>,
    twiddle_im: Vec<f64>,
}

impl MelSpectrogram {
    /// `fb` is the `[n_fft / 2 + 1, n_mels]` row-major filterbank as stored
    /// in the checkpoint.
    pub fn new(
        n_fft: usize,
        hop_length: usize,
        n_mels: usize,
        window: Vec<f32>,
        fb: Vec<f32>,
    ) -> Self {
        assert!(n_fft.is_power_of_two(), "n_fft must be a power of two");
        assert_eq!(window.len(), n_fft);
        let n_freqs = n_fft / 2 + 1;
        assert_eq!(fb.len(), n_freqs * n_mels);
        let mut fb_t = vec![0f64; fb.len()];
        for f in 0..n_freqs {
            for m in 0..n_mels {
                fb_t[m * n_freqs + f] = fb[f * n_mels + m] as f64;
            }
        }
        let (twiddle_re, twiddle_im) = (0..n_fft / 2)
            .map(|k| {
                let angle = -2. * std::f64::consts::PI * k as f64 / n_fft as f64;
                (angle.cos(), angle.sin())
            })
            .unzip();
        Self {
            n_fft,
            hop_length,
            n_mels,
            window: window.into_iter().map(|w| w as f64).collect(),
            fb_t,
            twiddle_re,
            twiddle_im,
        }
    }

    pub fn hop_length(&self) -> usize {
        self.hop_length
    }

    /// Magnitude mel spectrogram of `samples`; returns `(mel, n_frames)` with
    /// `mel` in `[n_frames, n_mels]` row-major order (natural log is applied
    /// by the caller).
    pub fn compute(&self, samples: &[f32]) -> (Vec<f32>, usize) {
        let pad = self.n_fft / 2;
        assert!(
            samples.len() > pad,
            "audio segment too short for reflect padding"
        );
        // Center the frames: reflect-pad by n_fft / 2 on both sides.
        let padded_len = samples.len() + 2 * pad;
        let mut padded = Vec::with_capacity(padded_len);
        padded.extend((0..pad).map(|i| samples[pad - i] as f64));
        padded.extend(samples.iter().map(|&v| v as f64));
        padded.extend((0..pad).map(|i| samples[samples.len() - 2 - i] as f64));

        let n_frames = 1 + (padded_len - self.n_fft) / self.hop_length;
        let n_freqs = self.n_fft / 2 + 1;
        let mut mel = vec![0f32; n_frames * self.n_mels];
        let mut re = vec![0f64; self.n_fft];
        let mut im = vec![0f64; self.n_fft];
        let mut mag = vec![0f64; n_freqs];
        for frame in 0..n_frames {
            let start = frame * self.hop_length;
            for i in 0..self.n_fft {
                re[i] = padded[start + i] * self.window[i];
                im[i] = 0.;
            }
            fft_inplace(&mut re, &mut im, &self.twiddle_re, &self.twiddle_im);
            for f in 0..n_freqs {
                mag[f] = (re[f] * re[f] + im[f] * im[f]).sqrt();
            }
            let out = &mut mel[frame * self.n_mels..(frame + 1) * self.n_mels];
            for (m, o) in out.iter_mut().enumerate() {
                let fb_row = &self.fb_t[m * n_freqs..(m + 1) * n_freqs];
                // The filters are triangular: only a contiguous band is
                // non-zero, but the full dot product is cheap enough.
                *o = fb_row
                    .iter()
                    .zip(mag.iter())
                    .map(|(&w, &v)| w * v)
                    .sum::<f64>() as f32;
            }
        }
        (mel, n_frames)
    }
}

//! Logit Processing and Sampling
//!
//! Functionality for modeling sampling strategies and logits processing in text
//! generation with support for temperature-based sampling, top-k filtering,
//! nucleus sampling (top-p), and combinations thereof.
use candle::{DType, Error, Result, Tensor};
use rand::{distr::Distribution, Rng, SeedableRng};

#[derive(Clone, PartialEq, Debug)]
pub enum Sampling {
    ArgMax,
    All { temperature: f64 },
    TopK { k: usize, temperature: f64 },
    TopP { p: f64, temperature: f64 },
    TopKThenTopP { k: usize, p: f64, temperature: f64 },
    // Note that the rng is not used for the Gumbel-Softmax sampling.
    GumbelSoftmax { temperature: f64 },
}

/// Builds the (unnormalized) cumulative distribution of `prs` into `cdf`.
///
/// The normalization constant cancels in a uniform-draw-over-the-CDF sample,
/// so it is not applied (unlike `rand::distr::weighted::WeightedIndex`, which
/// divides every entry by the total as an f64). `cdf[i]` holds
/// `prs[0] + ... + prs[i]`, and the returned value is `cdf[n - 1]` (the
/// total). `cdf` must have the same length as `prs` and is reused across
/// tokens by the caller, so no per-token allocation is needed.
fn build_cdf(prs: &[f32], cdf: &mut [f32]) -> f32 {
    debug_assert_eq!(prs.len(), cdf.len());
    let mut sum = 0.0f32;
    for (i, &p) in prs.iter().enumerate() {
        sum += p;
        cdf[i] = sum;
    }
    sum
}

pub struct LogitsProcessor {
    rng: rand::rngs::StdRng,
    sampling: Sampling,
    // Reused CDF buffer for multinomial sampling: grown to the vocabulary
    // size on first use and kept across tokens, so no per-token allocation
    // is needed for the (unnormalized) cumulative distribution.
    cdf: Vec<f32>,
}

impl LogitsProcessor {
    pub fn from_sampling(seed: u64, sampling: Sampling) -> Self {
        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        Self {
            rng,
            sampling,
            cdf: Vec::new(),
        }
    }

    pub fn new(seed: u64, temperature: Option<f64>, top_p: Option<f64>) -> Self {
        // `manual_filter` suggests `.filter(|&v| v >= 1e-7)`, but `f64` is only `PartialOrd`:
        // that drops `NaN` where the comparison below keeps it. Left as-is on purpose.
        #[allow(clippy::manual_filter)]
        let temperature = temperature.and_then(|v| if v < 1e-7 { None } else { Some(v) });
        let sampling = match temperature {
            None => Sampling::ArgMax,
            Some(temperature) => match top_p {
                None => Sampling::All { temperature },
                Some(p) => Sampling::TopP { p, temperature },
            },
        };
        Self::from_sampling(seed, sampling)
    }

    fn sample_argmax(&mut self, logits: Tensor) -> Result<u32> {
        logits.argmax(candle::D::Minus1)?.to_scalar::<u32>()
    }

    fn sample_gumbel_softmax(&mut self, logits: &Tensor, temperature: f64) -> Result<u32> {
        let sampled = candle_nn::sampling::gumbel_softmax(logits, temperature, candle::D::Minus1)?;
        sampled.to_scalar::<u32>()
    }

    fn sample_multinomial(&mut self, prs: &Vec<f32>) -> Result<u32> {
        let n = prs.len();
        if n == 0 {
            return Err(Error::Msg(
                "sample_multinomial: empty distribution".to_string(),
            ));
        }
        // Build the unnormalized CDF (no per-entry division, no per-token
        // allocation) and binary-search the uniform draw against it — the
        // same distribution as WeightedIndex::sample, minus the
        // normalization pass and the 1 MiB allocation per token.
        self.cdf.resize(n, 0.0);
        let total = build_cdf(prs, &mut self.cdf);
        if !total.is_finite() || total <= 0.0 {
            // Degenerate weights (NaN/negative/non-finite total): keep the
            // historical error behavior of WeightedIndex::new.
            let distr = rand::distr::weighted::WeightedIndex::new(prs).map_err(Error::wrap)?;
            return Ok(distr.sample(&mut self.rng) as u32);
        }
        let u: f64 = self.rng.random::<f64>() * total as f64;
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.cdf[mid] as f64 > u {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        Ok(lo as u32)
    }

    /// top-p sampling (or "nucleus sampling") samples from the smallest set of tokens that exceed
    /// probability top_p. This way we never sample tokens that are very low probabilities and are
    /// less likely to get "off the rails".
    fn sample_topp(&mut self, prs: &mut Vec<f32>, top_p: f32) -> Result<u32> {
        let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();

        // Sort by descending probability.
        argsort_indices.sort_by(|&i, &j| prs[j].total_cmp(&prs[i]));

        // Clamp smaller probabilities to zero.
        let mut cumsum = 0.;
        for index in &argsort_indices {
            if cumsum >= top_p {
                prs[*index] = 0.0;
            } else {
                cumsum += prs[*index];
            }
        }
        // Sample with clamped probabilities.
        self.sample_multinomial(prs)
    }

    // top-k sampling samples from the k tokens with the largest probabilities.
    fn sample_topk(&mut self, prs: &mut Vec<f32>, top_k: usize) -> Result<u32> {
        if top_k >= prs.len() {
            self.sample_multinomial(prs)
        } else {
            let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();
            let (indices, _, _) =
                argsort_indices.select_nth_unstable_by(top_k, |&i, &j| prs[j].total_cmp(&prs[i]));
            let prs = indices.iter().map(|&i| prs[i]).collect::<Vec<_>>();
            let index = self.sample_multinomial(&prs)?;
            Ok(indices[index as usize] as u32)
        }
    }

    // top-k sampling samples from the k tokens with the largest probabilities.
    // then top-p sampling.
    fn sample_topk_topp(&mut self, prs: &mut Vec<f32>, top_k: usize, top_p: f32) -> Result<u32> {
        if top_k >= prs.len() {
            self.sample_topp(prs, top_p)
        } else {
            let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();
            let (indices, _, _) =
                argsort_indices.select_nth_unstable_by(top_k, |&i, &j| prs[j].total_cmp(&prs[i]));
            let mut prs = indices.iter().map(|&i| prs[i]).collect::<Vec<_>>();
            let sum_p = prs.iter().sum::<f32>();
            let index = if top_p <= 0.0 || top_p >= sum_p {
                self.sample_multinomial(&prs)?
            } else {
                self.sample_topp(&mut prs, top_p)?
            };
            Ok(indices[index as usize] as u32)
        }
    }

    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        self.sample_f(logits, |_| {})
    }

    pub fn sample_f(&mut self, logits: &Tensor, f: impl FnOnce(&mut [f32])) -> Result<u32> {
        let logits = logits.to_dtype(DType::F32)?;
        let prs = |temperature: f64| -> Result<Vec<f32>> {
            let logits = (&logits / temperature)?;
            let prs = candle_nn::ops::softmax_last_dim(&logits)?;
            let mut prs = prs.to_vec1()?;
            f(&mut prs);
            Ok(prs)
        };

        let next_token = match &self.sampling {
            Sampling::ArgMax => self.sample_argmax(logits)?,
            Sampling::GumbelSoftmax { temperature } => {
                self.sample_gumbel_softmax(&logits, *temperature)?
            }
            Sampling::All { temperature } => {
                let prs = prs(*temperature)?;
                self.sample_multinomial(&prs)?
            }
            Sampling::TopP { p, temperature } => {
                let mut prs = prs(*temperature)?;
                if *p <= 0.0 || *p >= 1.0 {
                    // simply sample from the predicted probability distribution
                    self.sample_multinomial(&prs)?
                } else {
                    // top-p (nucleus) sampling, clamping the least likely tokens to zero
                    self.sample_topp(&mut prs, *p as f32)?
                }
            }
            Sampling::TopK { k, temperature } => {
                let mut prs = prs(*temperature)?;
                self.sample_topk(&mut prs, *k)?
            }
            Sampling::TopKThenTopP { k, p, temperature } => {
                let mut prs = prs(*temperature)?;
                self.sample_topk_topp(&mut prs, *k, *p as f32)?
            }
        };
        Ok(next_token)
    }
}

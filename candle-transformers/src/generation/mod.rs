use candle::{DType, Error, Result, Tensor, D};
use rand::{distributions::Distribution, SeedableRng};

pub struct LogitsProcessor {
    rng: rand::rngs::StdRng,
    temperature: Option<f64>,
    top_p: Option<f64>,
}

impl LogitsProcessor {
    pub fn new(seed: u64, temperature: Option<f64>, top_p: Option<f64>) -> Self {
        Self {
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            temperature,
            top_p,
        }
    }

    fn sample_argmax(&mut self, logits: Tensor) -> Result<u32> {
        let logits_v: Vec<f32> = logits.to_vec1()?;
        let next_token = logits_v
            .iter()
            .enumerate()
            .max_by(|(_, u), (_, v)| u.total_cmp(v))
            .map(|(i, _)| i as u32)
            .unwrap();
        Ok(next_token)
    }

    fn sample_multi(&mut self, prs: &Vec<f32>) -> Result<u32> {
        let distr = rand::distributions::WeightedIndex::new(prs).map_err(Error::wrap)?;
        let next_token = distr.sample(&mut self.rng) as u32;
        Ok(next_token)
    }

    fn sample_topp(&mut self, prs: &mut Vec<f32>, top_p: f32) -> Result<u32> {
        // top-p sampling (or "nucleus sampling") samples from the smallest set of
        // tokens that exceed probability top_p. This way we never sample tokens that
        // have very low probabilities and are less likely to go "off the rails".
        let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();
        
        // Sort by descending probability.
        argsort_indices.sort_by(|&i, &j| prs[j].partial_cmp(&prs[i]).unwrap());
        
        // Clamp smaller probabilities to zero.
        let mut cumsum = 0.;
        for i in 0..argsort_indices.len() {
            let index = argsort_indices[i];
            if cumsum >= top_p {
                prs[index] = 0.0;
            } else {
                cumsum += prs[index];
            }
        }

        // Sample with clamped probabilities.
        let next_token = self.sample_multi(prs)?;
        Ok(next_token)
    }

    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        let logits = logits.to_dtype(DType::F32)?;
        let temperature = self.temperature.unwrap_or(0.);
        let top_p = self.top_p.unwrap_or(1.);
        let next_token = if temperature == 0. {
            self.sample_argmax(logits)?
        } else {
            let logits = &(&logits / temperature)?;
            let prs = candle_nn::ops::softmax(logits, D::Minus1)?;
            let mut prs: Vec<f32> = prs.to_vec1()?;
            if top_p <= 0.0 || top_p >= 1.0 {
                // simply sample from the predicted probability distribution
                self.sample_multi(&prs)?
            } else {
                // top-p (nucleus) sampling, clamping the least likely tokens to zero
                self.sample_topp(&mut prs, top_p as f32)?
            }
        };
        Ok(next_token)
    }
}

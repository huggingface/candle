use candle::{DType, Error, Result, Tensor, D};
use rand::{distributions::Distribution, SeedableRng};

pub struct LogitsProcessor {
    rng: rand::rngs::StdRng,
    temperature: Option<f64>,
}

impl LogitsProcessor {
    pub fn new(seed: u64, temperature: Option<f64>) -> Self {
        Self {
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            temperature,
        }
    }

    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        let logits = logits.to_dtype(DType::F32)?;
        let temperature = self.temperature.unwrap_or(0.);
        let next_token = if temperature > 0. {
            let prs = candle_nn::ops::softmax(&(&logits / temperature)?, D::Minus1)?;
            let prs: Vec<f32> = prs.to_vec1()?;
            let distr = rand::distributions::WeightedIndex::new(prs).map_err(Error::wrap)?;
            distr.sample(&mut self.rng) as u32
        } else {
            let logits_v: Vec<f32> = logits.to_vec1()?;
            logits_v
                .iter()
                .enumerate()
                .max_by(|(_, u), (_, v)| u.total_cmp(v))
                .map(|(i, _)| i as u32)
                .unwrap()
        };
        Ok(next_token)
    }
}

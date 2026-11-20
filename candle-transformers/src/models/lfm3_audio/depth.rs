//! Native MaskGIT depth prediction and codebook feedback embeddings.
use super::{
    super::lfm3::{Linear, Stack, Weights},
    Config,
};
use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::RmsNorm;
use rand::{rngs::StdRng, seq::SliceRandom, Rng};
#[derive(Clone)]
pub struct DepthDecoder {
    projection: Linear,
    mask: Tensor,
    stack: Stack,
    embeddings: Vec<Tensor>,
    norms: Vec<RmsNorm>,
    feedback: Tensor,
    books: usize,
    hidden: usize,
    vocab: usize,
    steps: usize,
}
impl DepthDecoder {
    pub fn new(cfg: &Config, w: &Weights<'_>) -> Result<Self> {
        let h = cfg.depth_config.hidden_size;
        let books = cfg.num_codebooks;
        let vocab = cfg.codebook_size;
        let tables = w.dense(
            (books, vocab, h),
            "depth_embeddings.weight",
            "a.gen.code.embd.weight",
        )?;
        let norms = w.dense(
            (books, h),
            "depth_embeddings.norm.weight",
            "a.gen.code.head_norm.weight",
        )?;
        let mut embeddings = Vec::new();
        let mut ns = Vec::new();
        for i in 0..books {
            embeddings.push(tables.i(i)?.contiguous()?);
            ns.push(RmsNorm::new(
                norms.i(i)?.contiguous()?,
                cfg.depth_config.norm_eps,
            ));
        }
        Ok(Self {
            projection: w.linear(
                cfg.text_config.hidden_size,
                books * h,
                "depth_linear",
                "a.gen.code.proj_in",
                true,
            )?,
            mask: w.dense(h, "mask_embedding", "a.gen.code.mask_embd")?,
            stack: Stack::new(
                &cfg.depth_config,
                w,
                "depthformer.model",
                "a.gen.code.",
                false,
                false,
                true,
            )?,
            embeddings,
            norms: ns,
            feedback: w.dense(
                (books * vocab, cfg.text_config.hidden_size),
                "codebook_embedding.weight",
                "a.gen.code.out_embd.weight",
            )?,
            books,
            hidden: h,
            vocab,
            steps: cfg.maskgit_steps,
        })
    }
    pub fn feedback(&self, codes: &[u32]) -> Result<Tensor> {
        if codes.len() != self.books {
            candle::bail!("wrong LFM3 codebook count")
        }
        let ids = codes
            .iter()
            .enumerate()
            .map(|(i, &c)| c + i as u32 * self.vocab as u32)
            .collect::<Vec<_>>();
        self.feedback
            .index_select(&Tensor::new(ids.as_slice(), self.feedback.device())?, 0)?
            .sum(0)?
            .reshape((1, 1, ()))
    }
    fn logits(&mut self, hidden: &Tensor, codes: &[u32], masked: &[bool]) -> Result<Vec<Vec<f32>>> {
        let embeds = (0..self.books)
            .map(|i| {
                if masked[i] {
                    Ok(self.mask.clone())
                } else {
                    self.embeddings[i].i(codes[i] as usize)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let embeds = Tensor::stack(&embeds, 0)?.unsqueeze(0)?;
        let projected = self
            .projection
            .forward(&hidden.reshape((1, ()))?)?
            .reshape((1, self.books, self.hidden))?;
        let states = self.stack.forward(&(projected + embeds)?)?;
        let logits = (0..self.books)
            .map(|i| {
                self.norms[i]
                    .forward(&states.i((0, i))?.unsqueeze(0)?)?
                    .matmul(&self.embeddings[i].t()?)
            })
            .collect::<Result<Vec<_>>>()?;
        Tensor::cat(&logits, 0)?
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()
    }
    pub fn sample(
        &mut self,
        hidden: &Tensor,
        history: &[Vec<u32>],
        rng: &mut StdRng,
        temperature: f64,
        top_k: usize,
    ) -> Result<Vec<u32>> {
        if self.steps == 0
            || self.steps > self.books
            || !temperature.is_finite()
            || temperature < 0.
        {
            candle::bail!("invalid LFM3 MaskGIT sampling settings")
        }
        let mut order = (0..self.books).collect::<Vec<_>>();
        order.shuffle(rng);
        let mut codes = vec![0; self.books];
        let mut masked = vec![true; self.books];
        let per = self.books / self.steps;
        let mut offset = 0;
        for step in 0..self.steps {
            let logits = self.logits(hidden, &codes, &masked)?;
            let count = if step + 1 == self.steps {
                self.books - offset
            } else {
                per
            };
            for &i in &order[offset..offset + count] {
                let mut scores = logits[i].clone();
                let mut seen = vec![false; scores.len()];
                for frame in history.iter().rev().take(4096) {
                    let id = frame[i] as usize;
                    if !seen[id] {
                        scores[id] = if scores[id] < 0. {
                            scores[id] * 1.1
                        } else {
                            scores[id] / 1.1
                        };
                        seen[id] = true;
                    }
                }
                let mut ranked = scores
                    .into_iter()
                    .enumerate()
                    .filter(|(_, v)| v.is_finite())
                    .collect::<Vec<_>>();
                ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
                if ranked.is_empty() {
                    candle::bail!("LFM3 audio logits have no finite values")
                };
                if top_k > 0 {
                    ranked.truncate(top_k);
                }
                let token = if temperature == 0. {
                    ranked[0].0
                } else {
                    let max = ranked[0].1 as f64;
                    let probs = ranked
                        .iter()
                        .map(|(_, v)| ((*v as f64 - max) / temperature).exp())
                        .collect::<Vec<_>>();
                    let mut draw = rng.random::<f64>() * probs.iter().sum::<f64>();
                    let mut selected = ranked.last().unwrap().0;
                    for ((id, _), p) in ranked.iter().zip(probs) {
                        draw -= p;
                        if draw <= 0. {
                            selected = *id;
                            break;
                        }
                    }
                    selected
                };
                codes[i] = token as u32;
                masked[i] = false;
            }
            offset += count;
        }
        Ok(codes)
    }
}

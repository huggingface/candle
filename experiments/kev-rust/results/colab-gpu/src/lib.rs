#[cfg(feature = "cuda")]
mod cuda;
mod delta;
pub mod qwen3_5;

use anyhow::{bail, ensure, Context, Result};
use candle::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{Linear, VarBuilder};
use regex::Regex;
use serde_json::{json, Map, Value};
use std::{path::Path, time::Instant};
use tokenizers::Tokenizer;

const SPECIAL: [&str; 5] = [
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
];

pub struct Runner {
    device: Device,
    model: qwen3_5::Model,
    q: Linear,
    k: Linear,
    scale: f64,
    tokenizer: Tokenizer,
    delimiter: [u32; 5],
    escape: Regex,
    pub metadata: Value,
}

struct Branch {
    id: String,
    kind: String,
    keys: Vec<String>,
    options: Vec<String>,
    ids: Vec<u32>,
    ends: Vec<usize>,
}

fn python_number(n: &serde_json::Number) -> String {
    let literal = n.to_string();
    // JSON integers remain exact even when they exceed u64. Python also keeps them as ints.
    if !literal.contains(['.', 'e', 'E']) {
        return literal;
    }
    let value: f64 = literal.parse().expect("valid JSON float");
    if value.is_infinite() {
        return if value.is_sign_negative() {
            "-inf"
        } else {
            "inf"
        }
        .into();
    }
    let scientific = format!("{value:e}");
    let (mantissa, exponent) = scientific.split_once('e').expect("scientific float");
    let exponent: i32 = exponent.parse().expect("float exponent");
    // Python str(float) uses scientific notation below 1e-4 and at/above 1e16,
    // with a signed exponent containing at least two digits.
    if !(-4..16).contains(&exponent) {
        format!("{mantissa}e{exponent:+03}")
    } else {
        let plain = value.to_string();
        if plain.contains('.') {
            plain
        } else {
            format!("{plain}.0")
        }
    }
}

/// Match kev.api.render, including labelled fields and Python boolean spelling.
pub fn render(v: &Value, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    match v {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::Number(n) => python_number(n),
        Value::Array(xs) => xs
            .iter()
            .map(|x| format!("{pad}- {}", render(x, indent + 1).trim_start()))
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Object(xs) => xs
            .iter()
            .map(|(k, x)| match x {
                Value::Array(_) | Value::Object(_) => {
                    format!("{pad}{k}:\n{}", render(x, indent + 1))
                }
                _ => format!("{pad}{k}: {}", render(x, 0)),
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn option(name: &str, desc: &Value) -> String {
    if desc.is_null() || desc.as_str() == Some("") {
        name.into()
    } else {
        format!("{name}: {}", render(desc, 0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn structured_state_matches_python_rendering() {
        let state = json!({"ticket":{"message":"damaged","paid":true},"items":["vase",2]});
        assert_eq!(
            render(&state, 0),
            "ticket:\n  message: damaged\n  paid: True\nitems:\n  - vase\n  - 2"
        );
        assert_eq!(option("no", &Value::Null), "no");
        assert_eq!(option("yes", &json!({"a":false})), "yes: a: False");
    }

    #[test]
    fn numeric_state_preserves_python_float_and_integer_text() {
        let state: Value =
            serde_json::from_str(r#"[1e-7,0.0001,1e16,1.0,-0.0,123456789012345678901234567890]"#)
                .unwrap();
        assert_eq!(
            render(&state, 0),
            "- 1e-07\n- 0.0001\n- 1e+16\n- 1.0\n- -0.0\n- 123456789012345678901234567890"
        );
    }
}

impl Runner {
    /// Shared prefix with independent batched questions on the selected device.
    pub fn infer(&self, req: &Value) -> Result<Value> {
        self.predict_inner(req, true, true)
    }
    pub fn load(dir: &Path, fused: bool) -> Result<Self> {
        Self::load_on_device(dir, fused, Device::Cpu)
    }

    pub fn load_on_device(dir: &Path, fused: bool, dev: Device) -> Result<Self> {
        ensure!(
            dev.is_cpu() || dev.is_cuda(),
            "only CPU and CUDA are supported"
        );
        #[cfg(feature = "cuda")]
        if dev.is_cuda() {
            candle::cuda_backend::set_gemm_reduced_precision_f32(false);
        }
        let cfg: qwen3_5::Config =
            serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        ensure!(
            cfg.text_config.layer_types.len() == cfg.text_config.num_hidden_layers,
            "layer count mismatch"
        );
        let metadata: Value = serde_json::from_slice(&std::fs::read(dir.join("metadata.json"))?)?;
        ensure!(
            metadata["dtype"] == "float32",
            "only fp32 exports supported"
        );
        let temperature = metadata["temperature"]
            .as_f64()
            .context("temperature missing")?;
        ensure!(
            temperature.is_finite() && temperature > 0.,
            "invalid temperature"
        );
        let dp = metadata["head_dim"].as_u64().context("head_dim missing")? as usize;
        // Files are immutable while mapped. No Python or torch runtime is used here.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &dev)?
        };
        let mut model = qwen3_5::Model::new(&cfg, vb)?;
        model.set_fused(fused);
        let head = unsafe {
            VarBuilder::from_mmaped_safetensors(&[dir.join("head.safetensors")], DType::F32, &dev)?
        };
        let q = candle_nn::linear(cfg.text_config.hidden_size, dp, head.pp("q"))?;
        let k = candle_nn::linear(cfg.text_config.hidden_size, dp, head.pp("k"))?;
        let tokenizer =
            Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut delimiter = [0; 5];
        for (i, name) in SPECIAL.iter().enumerate() {
            delimiter[i] = tokenizer
                .token_to_id(name)
                .with_context(|| format!("missing {name}"))?;
        }
        dev.synchronize()?;
        Ok(Self {
            device: dev,
            model,
            q,
            k,
            scale: 1. / (dp as f64).sqrt() / temperature,
            tokenizer,
            delimiter,
            escape: Regex::new(r"<\|([A-Za-z0-9_]+)\|>")?,
            metadata,
        })
    }

    fn tokens(&self, text: &str) -> Result<Vec<u32>> {
        let text = self.escape.replace_all(text, "<¦${1}¦>");
        Ok(self
            .tokenizer
            .encode(text.as_ref(), false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids()
            .to_vec())
    }

    fn encode(&self, req: &Value) -> Result<(Vec<u32>, Vec<Branch>)> {
        ensure!(req.get("state").is_some(), "state is required");
        let mut state = vec![self.delimiter[0]];
        state.extend(self.tokens(&render(&req["state"], 0))?);
        ensure!(state.len() <= 65536, "state exceeds 65536 tokens");
        let qs = req["questions"]
            .as_object()
            .context("questions must be an object")?;
        ensure!(!qs.is_empty(), "at least one question is required");
        let mut branches = vec![];
        for (id, q) in qs {
            let kind = q["type"].as_str().context("question type is required")?;
            let (keys, options): (Vec<_>, Vec<_>) = match kind {
                "choice" => {
                    let c = q["criteria"]
                        .as_object()
                        .context("choice criteria must be an object")?;
                    (
                        c.keys().cloned().collect(),
                        c.iter().map(|(k, v)| option(k, v)).collect(),
                    )
                }
                "score" => {
                    let c = q["criteria"]
                        .as_array()
                        .context("score criteria must be an array")?;
                    (
                        (0..c.len()).map(|i| i.to_string()).collect(),
                        c.iter().map(|v| render(v, 0)).collect(),
                    )
                }
                "noul" => {
                    ensure!(
                        q["criteria"].is_null() || q["criteria"].is_object(),
                        "noul criteria must be an object"
                    );
                    (
                        vec!["false".into(), "true".into()],
                        vec![
                            option("no", &q["criteria"]["false"]),
                            option("yes", &q["criteria"]["true"]),
                        ],
                    )
                }
                _ => bail!("unsupported question type: {kind}"),
            };
            ensure!(
                (1..=255).contains(&keys.len()),
                "criteria must have 1..255 options"
            );
            let mut ids = vec![self.delimiter[1]];
            ids.extend(self.tokens(&render(&q["instructions"], 0))?);
            let mut ends = vec![];
            for opt in &options {
                ids.push(self.delimiter[2]);
                ids.extend(self.tokens(opt)?);
                ids.push(self.delimiter[3]);
                ends.push(ids.len() - 1);
            }
            ids.push(self.delimiter[4]);
            ensure!(
                state.len() + ids.len() <= 73728,
                "question row exceeds 73728 tokens"
            );
            branches.push(Branch {
                id: id.clone(),
                kind: kind.into(),
                keys,
                options,
                ids,
                ends,
            });
        }
        Ok((state, branches))
    }

    /// `prefix=true` computes the shared state once and forks immutable tensor caches per question.
    /// `prefix=false` is the full-row correctness/performance baseline.
    pub fn predict(&self, req: &Value, prefix: bool) -> Result<Value> {
        self.predict_inner(req, prefix, false)
    }

    pub fn predict_batched(&self, req: &Value, prefix: bool) -> Result<Value> {
        self.predict_inner(req, prefix, true)
    }

    fn hidden_rows(
        &self,
        state: &[u32],
        branches: &[Branch],
        prefix: bool,
        batch: bool,
    ) -> Result<(Vec<Tensor>, usize)> {
        let dev = &self.device;
        let mut shared = self.model.clone();
        let mut computed = 0;
        if prefix {
            let mut offset = 0;
            for part in state.chunks(256) {
                let _ = shared.forward(&Tensor::new(part, &dev)?.unsqueeze(0)?, offset)?;
                offset += part.len();
            }
            computed += state.len();
        }
        let longest = branches.iter().map(|b| b.ids.len()).max().unwrap() + state.len();
        let count = if batch {
            (16384 / longest).clamp(1, 8)
        } else {
            1
        };
        let mut hidden = vec![];
        for group in branches.chunks(count) {
            let mut model = if prefix {
                shared.clone()
            } else {
                self.model.clone()
            };
            if prefix && group.len() > 1 {
                model.repeat_cache(group.len())?;
            }
            let offset = if prefix { state.len() } else { 0 };
            let row_len = group.iter().map(|b| b.ids.len()).max().unwrap()
                + if prefix { 0 } else { state.len() };
            let mut ids = vec![0u32; group.len() * row_len];
            for (row, br) in group.iter().enumerate() {
                let start = row * row_len;
                if prefix {
                    ids[start..start + br.ids.len()].copy_from_slice(&br.ids);
                } else {
                    ids[start..start + state.len()].copy_from_slice(state);
                    ids[start + state.len()..start + state.len() + br.ids.len()]
                        .copy_from_slice(&br.ids);
                }
            }
            // Padding is strictly after each real row. Causality prevents padded
            // tokens from influencing real readouts; padded final caches are discarded.
            let h = model.forward(
                &Tensor::from_vec(ids, (group.len(), row_len), &dev)?,
                offset,
            )?;
            let base = if prefix { 0 } else { state.len() };
            for (row, br) in group.iter().enumerate() {
                let mut markers = br
                    .ends
                    .iter()
                    .map(|i| (i + base) as u32)
                    .collect::<Vec<_>>();
                markers.push((base + br.ids.len() - 1) as u32);
                hidden.push(h.i(row)?.index_select(&Tensor::new(markers, &dev)?, 0)?);
            }
            computed += group.len() * row_len;
        }
        Ok((hidden, computed))
    }

    fn predict_inner(&self, req: &Value, prefix: bool, batch: bool) -> Result<Value> {
        let started = Instant::now();
        let (state, branches) = self.encode(req)?;
        let tokenization_ms = started.elapsed().as_secs_f64() * 1000.;
        self.device.synchronize()?;
        let infer = Instant::now();
        let (hidden, computed_tokens) = self.hidden_rows(&state, &branches, prefix, batch)?;
        let mut answers = Map::new();
        let mut logits = Map::new();
        let mut encodings = Map::new();
        for (br, h) in branches.iter().zip(hidden) {
            let decide = self
                .q
                .forward(&h.i(br.ends.len())?.unsqueeze(0)?)?
                .squeeze(0)?;
            let opts = self.k.forward(&h.narrow(0, 0, br.ends.len())?)?;
            let z = (opts.matmul(&decide.unsqueeze(1)?)?.squeeze(1)? * self.scale)?;
            let ps = candle_nn::ops::softmax_last_dim(&z)?.to_vec1::<f32>()?;
            ensure!(ps.iter().all(|p| p.is_finite()), "non-finite probabilities");
            // Keep the first option in an exact tie, as Python does.
            let mut winner = 0;
            for i in 1..ps.len() {
                if ps[i] > ps[winner] {
                    winner = i;
                }
            }
            let distribution: Map<_, _> = br
                .keys
                .iter()
                .zip(&ps)
                .map(|(k, p)| (k.clone(), json!(p)))
                .collect();
            let answer = match br.kind.as_str() {
                "choice" => {
                    json!({"type":"choice", "choice":br.keys[winner], "confidence": if ps.len()==1 {1.} else {(ps[winner] as f64 - 1./ps.len() as f64)/(1.-1./ps.len() as f64)}, "probabilities":distribution})
                }
                "noul" => json!({"type":"noul", "noul":ps[1]}),
                "score" => {
                    let n = ps.len() as f64;
                    let uniform = (0..ps.len())
                        .map(|i| (i as f64 - (n - 1.) / 2.).abs())
                        .sum::<f64>()
                        / n;
                    let deviation = ps
                        .iter()
                        .enumerate()
                        .map(|(i, p)| *p as f64 * (i as f64 - winner as f64).abs())
                        .sum::<f64>();
                    let legend: Map<_, _> = br
                        .keys
                        .iter()
                        .cloned()
                        .zip(br.options.iter().map(|s| json!(s)))
                        .collect();
                    json!({"type":"score", "score":ps.iter().enumerate().map(|(i,p)|i as f64 * *p as f64).sum::<f64>(), "confidence":if ps.len()==1 {1.} else {(1.-deviation/uniform).max(0.)}, "legend":legend,"probabilities":distribution})
                }
                _ => unreachable!(),
            };
            answers.insert(br.id.clone(), answer);
            logits.insert(
                br.id.clone(),
                json!(br
                    .keys
                    .iter()
                    .zip(z.to_vec1::<f32>()?)
                    .map(|(k, z)| (k.clone(), json!(z)))
                    .collect::<Map<_, _>>()),
            );
            encodings.insert(br.id.clone(), json!({"ids":br.ids,"ends":br.ends}));
        }
        self.device.synchronize()?;
        Ok(
            json!({"model":"kev-0.8b", "answers":answers,"logits":logits,
            "latency_ms":infer.elapsed().as_secs_f64()*1000.,"tokenization_ms":tokenization_ms,
            "computed_tokens":computed_tokens,"state_tokens":state.len(),
            "input_tokens":state.len()+branches.iter().map(|b|b.ids.len()).sum::<usize>(),
            "encoding":{"state":state,"branches":encodings},"mode":if prefix {"prefix"} else {"rows"},"batched":batch,
            "device":if self.device.is_cuda() {"cuda"} else {"cpu"},"dtype":"float32"}),
        )
    }
}

use candle::{DType, Result, Tensor, D};
use candle_nn::{Embedding, Linear, Module, VarBuilder};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct PredictNetworkArgs {
    pub pred_hidden: usize,
    pub pred_rnn_layers: usize,
    #[serde(default)]
    pub rnn_hidden_size: Option<i64>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct JointNetworkArgs {
    pub joint_hidden: usize,
    pub activation: String,
    pub encoder_hidden: usize,
    pub pred_hidden: usize,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct PredictArgs {
    pub blank_as_pad: bool,
    pub vocab_size: usize,
    pub prednet: PredictNetworkArgs,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct JointArgs {
    pub num_classes: usize,
    pub vocabulary: Vec<String>,
    pub jointnet: JointNetworkArgs,
    #[serde(default)]
    pub num_extra_outputs: usize,
}

#[derive(Debug, Clone)]
struct LstmLayer {
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
}

impl LstmLayer {
    fn load(
        input_size: usize,
        hidden_size: usize,
        layer_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        fn try_get_weight(
            vb: &VarBuilder,
            name: &str,
            out_dim: usize,
            in_dim: usize,
        ) -> Option<Tensor> {
            if let Ok(w) = vb.get((out_dim, in_dim), name) {
                return Some(w);
            }
            if let Ok(w) = vb.get((in_dim, out_dim), name) {
                return w.t().ok();
            }
            None
        }

        fn load_weight(
            vb_layer: &VarBuilder,
            vb_root: &VarBuilder,
            vb_nemo: &VarBuilder,
            base: &str,
            layer_idx: usize,
            out_dim: usize,
            in_dim: usize,
        ) -> Result<Tensor> {
            let layer_name = format!("{base}_l{layer_idx}");
            let nemo_name = match base {
                "weight_ih" => Some("Wx"),
                "weight_hh" => Some("Wh"),
                _ => None,
            };
            let candidates = [
                (vb_layer, base),
                (vb_layer, layer_name.as_str()),
                (vb_root, layer_name.as_str()),
                (vb_root, base),
            ];
            for (vb, name) in candidates {
                if let Some(w) = try_get_weight(vb, name, out_dim, in_dim) {
                    return Ok(w);
                }
            }
            if let Some(name) = nemo_name {
                if let Some(w) = try_get_weight(vb_nemo, name, out_dim, in_dim) {
                    return Ok(w);
                }
            }
            Err(candle::Error::Msg(format!(
                "missing lstm weight {base} (layer {layer_idx})"
            )))
        }

        fn load_bias(
            vb_layer: &VarBuilder,
            vb_root: &VarBuilder,
            vb_nemo: &VarBuilder,
            base: &str,
            layer_idx: usize,
            size: usize,
            alt: Option<&str>,
        ) -> Result<Tensor> {
            let layer_name = format!("{base}_l{layer_idx}");
            let candidates = [
                (vb_layer, base),
                (vb_layer, layer_name.as_str()),
                (vb_root, layer_name.as_str()),
                (vb_root, base),
            ];
            for (vb, name) in candidates {
                if let Ok(bias) = vb.get((size,), name) {
                    return Ok(bias);
                }
            }
            if let Some(name) = alt {
                if let Ok(bias) = vb_nemo.get((size,), name) {
                    return Ok(bias);
                }
            }
            Tensor::zeros((size,), vb_root.dtype(), vb_root.device())
        }

        let vb_layer = vb.pp(format!("layer_{layer_idx}"));
        let vb_nemo = vb.pp("lstm").pp(format!("{layer_idx}"));
        let w_ih = load_weight(
            &vb_layer,
            &vb,
            &vb_nemo,
            "weight_ih",
            layer_idx,
            4 * hidden_size,
            input_size,
        )?;
        let w_hh = load_weight(
            &vb_layer,
            &vb,
            &vb_nemo,
            "weight_hh",
            layer_idx,
            4 * hidden_size,
            hidden_size,
        )?;
        let b_ih = load_bias(
            &vb_layer,
            &vb,
            &vb_nemo,
            "bias_ih",
            layer_idx,
            4 * hidden_size,
            Some("bias"),
        )?;
        let b_hh = load_bias(
            &vb_layer,
            &vb,
            &vb_nemo,
            "bias_hh",
            layer_idx,
            4 * hidden_size,
            None,
        )?;
        Ok(Self {
            w_ih,
            w_hh,
            b_ih,
            b_hh,
        })
    }

    fn forward(&self, x: &Tensor, h: &Tensor, c: &Tensor) -> Result<(Tensor, Tensor)> {
        let gates = x.matmul(&self.w_ih.t()?)?.broadcast_add(&self.b_ih)?;
        let gates = gates
            .broadcast_add(&h.matmul(&self.w_hh.t()?)?)?
            .broadcast_add(&self.b_hh)?;
        let chunks = gates.chunk(4, D::Minus1)?;
        let i = candle_nn::ops::sigmoid(&chunks[0])?;
        let f = candle_nn::ops::sigmoid(&chunks[1])?;
        let g = chunks[2].tanh()?;
        let o = candle_nn::ops::sigmoid(&chunks[3])?;
        let c_next = ((&f * c)? + (&i * &g)?)?;
        let h_next = (&o * c_next.tanh()?)?;
        Ok((h_next, c_next))
    }
}

#[derive(Debug, Clone)]
pub struct Lstm {
    layers: Vec<LstmLayer>,
    hidden_size: usize,
    num_layers: usize,
    batch_first: bool,
}

impl Lstm {
    pub fn load(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        batch_first: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let in_size = if i == 0 { input_size } else { hidden_size };
            layers.push(LstmLayer::load(
                in_size,
                hidden_size,
                i,
                vb.clone(),
            )?);
        }
        Ok(Self {
            layers,
            hidden_size,
            num_layers,
            batch_first,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        h_c: Option<(Tensor, Tensor)>,
    ) -> Result<(Tensor, (Tensor, Tensor))> {
        let (b, t, _) = x.dims3()?;
        let device = x.device();
        let dtype = x.dtype();

        let mut h_layers: Vec<Tensor> = Vec::with_capacity(self.num_layers);
        let mut c_layers: Vec<Tensor> = Vec::with_capacity(self.num_layers);

        if let Some((h, c)) = h_c {
            for i in 0..self.num_layers {
                h_layers.push(h.narrow(0, i, 1)?.squeeze(0)?);
                c_layers.push(c.narrow(0, i, 1)?.squeeze(0)?);
            }
        } else {
            for _ in 0..self.num_layers {
                h_layers.push(Tensor::zeros((b, self.hidden_size), dtype, device)?);
                c_layers.push(Tensor::zeros((b, self.hidden_size), dtype, device)?);
            }
        }

        let mut outputs = x.clone();
        for layer_idx in 0..self.num_layers {
            let layer = &self.layers[layer_idx];
            let mut h_t = h_layers[layer_idx].clone();
            let mut c_t = c_layers[layer_idx].clone();

            let mut layer_outputs = Vec::with_capacity(t);
            for time_idx in 0..t {
                let x_t = outputs.narrow(1, time_idx, 1)?.squeeze(1)?;
                let (h_next, c_next) = layer.forward(&x_t, &h_t, &c_t)?;
                h_t = h_next;
                c_t = c_next;
                layer_outputs.push(h_t.clone());
            }
            outputs = Tensor::stack(&layer_outputs, 1)?;
            h_layers[layer_idx] = h_t;
            c_layers[layer_idx] = c_t;
        }

        let h = Tensor::stack(&h_layers, 0)?;
        let c = Tensor::stack(&c_layers, 0)?;
        Ok((outputs, (h, c)))
    }
}

#[derive(Debug, Clone)]
pub struct PredictNetwork {
    pred_hidden: usize,
    embed: Embedding,
    dec_rnn: Lstm,
}

impl PredictNetwork {
    pub fn load(args: &PredictArgs, vb: VarBuilder) -> Result<Self> {
        let vocab = if args.blank_as_pad {
            args.vocab_size + 1
        } else {
            args.vocab_size
        };
        let pred_hidden = args.prednet.pred_hidden;
        let embed = candle_nn::embedding(vocab, pred_hidden, vb.pp("prediction").pp("embed"))?;
        let hidden_size = args
            .prednet
            .rnn_hidden_size
            .and_then(|v| if v > 0 { Some(v as usize) } else { None })
            .unwrap_or(pred_hidden);
        let dec_rnn = Lstm::load(
            pred_hidden,
            hidden_size,
            args.prednet.pred_rnn_layers,
            true,
            vb.pp("prediction").pp("dec_rnn"),
        )?;
        Ok(Self {
            pred_hidden,
            embed,
            dec_rnn,
        })
    }

    pub fn forward(
        &self,
        y: Option<&Tensor>,
        h_c: Option<(Tensor, Tensor)>,
    ) -> Result<(Tensor, (Tensor, Tensor))> {
        let device = if let Some(y) = y {
            y.device()
        } else if let Some((ref h, _)) = h_c {
            h.device()
        } else {
            self.embed.embeddings().device()
        };
        let embedded = if let Some(y) = y {
            self.embed.forward(y)?
        } else {
            let batch = if let Some((ref h, _)) = h_c {
                h.dims3()?.1
            } else {
                1
            };
            Tensor::zeros((batch, 1, self.pred_hidden), DType::F32, device)?
        };
        self.dec_rnn.forward(&embedded, h_c)
    }
}

#[derive(Debug, Clone)]
pub struct JointNetwork {
    pub num_classes: usize,
    pred: Linear,
    enc: Linear,
    activation: String,
    out: Linear,
}

impl JointNetwork {
    pub fn load(args: &JointArgs, vb: VarBuilder) -> Result<Self> {
        let num_classes = args.num_classes + 1 + args.num_extra_outputs;
        let pred = candle_nn::linear(
            args.jointnet.pred_hidden,
            args.jointnet.joint_hidden,
            vb.pp("pred"),
        )?;
        let enc = candle_nn::linear(
            args.jointnet.encoder_hidden,
            args.jointnet.joint_hidden,
            vb.pp("enc"),
        )?;
        let out = candle_nn::linear(
            args.jointnet.joint_hidden,
            num_classes,
            vb.pp("joint_net").pp(2),
        )?;
        Ok(Self {
            num_classes,
            pred,
            enc,
            activation: args.jointnet.activation.clone(),
            out,
        })
    }

    pub fn forward(&self, enc: &Tensor, pred: &Tensor) -> Result<Tensor> {
        let enc = self.enc.forward(enc)?;
        let pred = self.pred.forward(pred)?;
        let enc = enc.unsqueeze(2)?;
        let pred = pred.unsqueeze(1)?;
        let mut x = enc.broadcast_add(&pred)?;
        x = match self.activation.as_str() {
            "relu" => x.relu()?,
            "sigmoid" => candle_nn::ops::sigmoid(&x)?,
            _ => x.tanh()?,
        };
        self.out.forward(&x)
    }
}

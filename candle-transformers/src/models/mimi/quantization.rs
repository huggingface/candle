// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use candle::{IndexOp, Layout, Result, Shape, Tensor, D};
use candle_nn::{linear, Linear, VarBuilder};

struct CodebookEncode;

impl candle::CustomOp2 for CodebookEncode {
    fn name(&self) -> &'static str {
        "cb"
    }

    fn cpu_fwd(
        &self,
        lhs_storage: &candle::CpuStorage,
        lhs_layout: &Layout,
        rhs_storage: &candle::CpuStorage,
        rhs_layout: &Layout,
    ) -> Result<(candle::CpuStorage, Shape)> {
        use rayon::prelude::*;

        let (lhs_dim1, lhs_dim2) = lhs_layout.shape().dims2()?;
        let (rhs_dim1, rhs_dim2) = rhs_layout.shape().dims2()?;
        if lhs_dim2 != rhs_dim2 {
            candle::bail!("CodebookEncode, mismatch on last dim, {lhs_layout:?} {rhs_layout:?}");
        }
        if lhs_dim2 == 0 {
            candle::bail!("CodebookEncode, empty last dim {lhs_layout:?}")
        }
        let lhs = match lhs_layout.contiguous_offsets() {
            None => candle::bail!("CodebookEncode, lhs has to be contiguous, got {lhs_layout:?}"),
            Some((o1, o2)) => {
                let slice = lhs_storage.as_slice::<f32>()?;
                &slice[o1..o2]
            }
        };
        let rhs = match rhs_layout.contiguous_offsets() {
            None => candle::bail!("CodebookEncode, rhs has to be contiguous, got {rhs_layout:?}"),
            Some((o1, o2)) => {
                let slice = rhs_storage.as_slice::<f32>()?;
                &slice[o1..o2]
            }
        };
        let dst = (0..lhs_dim1)
            .into_par_iter()
            .map(|idx1| {
                let mut where_min = 0;
                let mut min_dist = f32::INFINITY;
                let lhs = &lhs[idx1 * lhs_dim2..(idx1 + 1) * lhs_dim2];
                for idx2 in 0..rhs_dim1 {
                    let rhs = &rhs[idx2 * rhs_dim2..(idx2 + 1) * rhs_dim2];
                    let mut dist = 0f32;
                    for (a, b) in lhs.iter().zip(rhs.iter()) {
                        dist += (a - b) * (a - b)
                    }
                    if dist < min_dist {
                        min_dist = dist;
                        where_min = idx2;
                    }
                }
                where_min as u32
            })
            .collect();
        let storage = candle::WithDType::to_cpu_storage_owned(dst);
        Ok((storage, (lhs_dim1,).into()))
    }
}

#[allow(unused)]
#[derive(Debug, Clone)]
pub struct EuclideanCodebook {
    initialized: Tensor,
    cluster_usage: Tensor,
    embedding_sum: Tensor,
    embedding: Tensor,
    c2: Tensor,
    epsilon: f64,
    dim: usize,
    span_encode: tracing::Span,
    span_decode: tracing::Span,
}

impl EuclideanCodebook {
    pub fn new(dim: usize, codebook_size: usize, vb: VarBuilder) -> Result<Self> {
        let epsilon = 1e-5;
        let initialized = vb.get(1, "initialized")?;
        let cluster_usage = vb.get(codebook_size, "cluster_usage")?;
        let embedding_sum = vb.get((codebook_size, dim), "embed_sum")?;
        let embedding = {
            let cluster_usage = cluster_usage.maximum(epsilon)?.unsqueeze(1)?;
            embedding_sum.broadcast_div(&cluster_usage)?
        };
        let c2 = ((&embedding * &embedding)?.sum(D::Minus1)? / 2.0)?;
        Ok(Self {
            initialized,
            cluster_usage,
            embedding_sum,
            embedding,
            c2,
            epsilon,
            dim,
            span_encode: tracing::span!(tracing::Level::TRACE, "euclidean-encode"),
            span_decode: tracing::span!(tracing::Level::TRACE, "euclidean-encode"),
        })
    }

    pub fn encode_very_slow(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span_encode.enter();
        let mut target_shape = xs.dims().to_vec();
        target_shape.pop();
        let xs = xs.flatten_to(D::Minus2)?;
        let _ = xs.dims2()?;
        // TODO: avoid repeating this.
        let cluster_usage = self.cluster_usage.maximum(self.epsilon)?.unsqueeze(1)?;
        let embedding = self.embedding_sum.broadcast_div(&cluster_usage)?;
        // Manual cdist implementation.
        let diff = xs.unsqueeze(1)?.broadcast_sub(&embedding.unsqueeze(0)?)?;
        let dists = diff.sqr()?.sum(D::Minus1)?;
        let codes = dists.argmin(D::Minus1)?;
        codes.reshape(target_shape)
    }

    pub fn encode_slow(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span_encode.enter();
        let mut target_shape = xs.dims().to_vec();
        target_shape.pop();
        let xs = xs.flatten_to(D::Minus2)?;
        let _ = xs.dims2()?;
        let dot_prod = xs.matmul(&self.embedding.t()?)?;
        let codes = self.c2.broadcast_sub(&dot_prod)?.argmin(D::Minus1)?;
        codes.reshape(target_shape)
    }

    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span_encode.enter();
        let mut target_shape = xs.dims().to_vec();
        target_shape.pop();
        let xs = xs.flatten_to(D::Minus2)?;
        let _ = xs.dims2()?;
        let codes = Tensor::apply_op2(&xs, &self.embedding, CodebookEncode)?;
        codes.reshape(target_shape)
    }

    pub fn decode(&self, indexes: &Tensor) -> Result<Tensor> {
        let _enter = self.span_decode.enter();
        // let ys = candle_nn::Embedding::new(self.embedding.clone(), self.dim).forward(xs)?;
        let mut final_dims = indexes.dims().to_vec();
        final_dims.push(self.dim);
        let indexes = indexes.flatten_all()?;
        let values = self.embedding.index_select(&indexes, 0)?;
        let values = values.reshape(final_dims)?;
        Ok(values)
    }
}

#[allow(unused)]
#[derive(Debug, Clone)]
pub struct VectorQuantization {
    project_in: Option<Linear>,
    project_out: Option<Linear>,
    codebook: EuclideanCodebook,
}

impl VectorQuantization {
    pub fn new(
        dim: usize,
        codebook_size: usize,
        codebook_dim: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let codebook_dim = codebook_dim.unwrap_or(dim);
        let (project_in, project_out) = if codebook_dim == dim {
            (None, None)
        } else {
            let p_in = linear(dim, codebook_dim, vb.pp("project_in"))?;
            let p_out = linear(codebook_dim, dim, vb.pp("project_out"))?;
            (Some(p_in), Some(p_out))
        };
        let codebook = EuclideanCodebook::new(codebook_dim, codebook_size, vb.pp("codebook"))?;
        Ok(Self {
            project_in,
            project_out,
            codebook,
        })
    }

    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.t()?.apply(&self.project_in.as_ref())?;
        self.codebook.encode_slow(&xs)
    }

    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let quantized = self.codebook.decode(codes)?;
        let quantized = match &self.project_out {
            None => quantized,
            Some(p) => quantized.apply(p)?,
        };
        quantized.t()
    }
}

#[derive(Debug, Clone)]
pub struct ResidualVectorQuantization {
    layers: Vec<VectorQuantization>,
}

impl ResidualVectorQuantization {
    pub fn new(
        n_q: usize,
        dim: usize,
        codebook_size: usize,
        codebook_dim: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb = vb.pp("layers");
        let mut layers = Vec::with_capacity(n_q);
        for i in 0..n_q {
            let layer = VectorQuantization::new(dim, codebook_size, codebook_dim, vb.pp(i))?;
            layers.push(layer)
        }
        Ok(Self { layers })
    }

    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        let mut codes = Vec::with_capacity(self.layers.len());
        let mut residual = xs.clone();
        for layer in self.layers.iter() {
            let indices = layer.encode(&residual)?;
            let quantized = layer.decode(&indices)?;
            residual = (residual - quantized)?;
            codes.push(indices)
        }
        Tensor::stack(&codes, 0)
    }

    pub fn decode(&self, xs: &Tensor) -> Result<Tensor> {
        if self.layers.is_empty() {
            candle::bail!("empty layers in ResidualVectorQuantization")
        }
        if self.layers.len() != xs.dim(0)? {
            candle::bail!(
                "mismatch between the number of layers {} and the code shape {:?}",
                self.layers.len(),
                xs.shape()
            )
        }
        let mut quantized = self.layers[0].decode(&xs.i(0)?)?;
        for (i, layer) in self.layers.iter().enumerate().skip(1) {
            let xs = xs.i(i)?;
            quantized = (quantized + layer.decode(&xs))?
        }
        Ok(quantized)
    }
}

#[allow(unused)]
#[derive(Debug, Clone)]
pub struct ResidualVectorQuantizer {
    vq: ResidualVectorQuantization,
    input_proj: Option<candle_nn::Conv1d>,
    output_proj: Option<candle_nn::Conv1d>,
}

impl ResidualVectorQuantizer {
    pub fn new(
        dim: usize,
        input_dim: Option<usize>,
        output_dim: Option<usize>,
        n_q: usize,
        bins: usize,
        force_projection: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let input_dim = input_dim.unwrap_or(dim);
        let output_dim = output_dim.unwrap_or(dim);

        let input_proj = if input_dim == dim && !force_projection {
            None
        } else {
            let c = candle_nn::conv1d_no_bias(
                input_dim,
                dim,
                1,
                Default::default(),
                vb.pp("input_proj"),
            )?;
            Some(c)
        };
        let output_proj = if output_dim == dim && !force_projection {
            None
        } else {
            let c = candle_nn::conv1d_no_bias(
                dim,
                output_dim,
                1,
                Default::default(),
                vb.pp("output_proj"),
            )?;
            Some(c)
        };

        let vq = ResidualVectorQuantization::new(
            n_q, dim, /* codebook_size */ bins, /* codebook_dim */ None, vb,
        )?;
        Ok(Self {
            vq,
            input_proj,
            output_proj,
        })
    }

    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        let codes = self.vq.encode(&xs.apply(&self.input_proj.as_ref())?)?;
        codes.transpose(0, 1)
    }

    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        // codes is [B, K, T], with T frames, K nb of codebooks, vq.decode expects [K, B, T].
        let codes = codes.transpose(0, 1)?;
        let quantized = self.vq.decode(&codes)?;
        match &self.output_proj {
            None => Ok(quantized),
            Some(p) => quantized.apply(p),
        }
    }
}

// we do not use any codebook_offset at the moment. When reconstructing the codes, we could just
// concatenate the indexes.
#[derive(Debug, Clone)]
pub struct SplitResidualVectorQuantizer {
    rvq_first: ResidualVectorQuantizer,
    rvq_rest: ResidualVectorQuantizer,
    n_q: usize,
    span_encode: tracing::Span,
    span_decode: tracing::Span,
}

impl SplitResidualVectorQuantizer {
    pub fn new(
        dim: usize,
        input_dim: Option<usize>,
        output_dim: Option<usize>,
        n_q: usize,
        bins: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let rvq_first = ResidualVectorQuantizer::new(
            dim,
            input_dim,
            output_dim,
            1,
            bins,
            true,
            vb.pp("semantic_residual_vector_quantizer"),
        )?;
        let rvq_rest = ResidualVectorQuantizer::new(
            dim,
            input_dim,
            output_dim,
            n_q - 1,
            bins,
            true,
            vb.pp("acoustic_residual_vector_quantizer"),
        )?;
        let span_encode = tracing::span!(tracing::Level::TRACE, "split-rvq-encode");
        let span_decode = tracing::span!(tracing::Level::TRACE, "split-rvq-decode");
        Ok(Self {
            rvq_first,
            rvq_rest,
            n_q,
            span_encode,
            span_decode,
        })
    }

    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span_encode.enter();
        let codes = self.rvq_first.encode(xs)?;
        if self.n_q > 1 {
            // We encode xs again here rather than the residual. The decomposition is not
            // hierarchical but rather having semantic tokens for rvq_first and the acoustic tokens
            // for rvq_rest.
            let rest_codes = self.rvq_rest.encode(xs)?;
            Tensor::cat(&[codes, rest_codes], 1)
        } else {
            Ok(codes)
        }
    }

    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        // codes is [B, K, T], with T frames, K nb of codebooks.
        let _enter = self.span_decode.enter();
        let quantized = self.rvq_first.decode(&codes.i((.., ..1))?)?;
        let quantized = if self.n_q > 1 {
            (quantized + self.rvq_rest.decode(&codes.i((.., 1..))?))?
        } else {
            quantized
        };
        Ok(quantized)
    }
}

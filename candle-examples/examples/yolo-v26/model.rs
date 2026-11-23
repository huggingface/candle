// YOLO26 model implementation for candle.
//
// Architecture: Ultralytics YOLO26 with C3k2, C2PSA (attention), and NMS-free detection.
// Reference: https://docs.ultralytics.com/models/yolo26
// Weight keys follow ultralytics naming: model.{layer_idx}.{submodule}.weight

use candle::{Device, Result, Tensor};
use candle_nn::{batch_norm, conv2d, conv2d_no_bias, Conv2d, Conv2dConfig, Module, VarBuilder};

// ---------------------------------------------------------------------------
// Model size scaling
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Multiples {
    depth: f64,
    width: f64,
    max_ch: usize,
}

impl Multiples {
    pub fn n() -> Self {
        Self {
            depth: 0.50,
            width: 0.25,
            max_ch: 1024,
        }
    }
    pub fn s() -> Self {
        Self {
            depth: 0.50,
            width: 0.50,
            max_ch: 1024,
        }
    }
    pub fn m() -> Self {
        Self {
            depth: 0.50,
            width: 1.00,
            max_ch: 512,
        }
    }
    pub fn l() -> Self {
        Self {
            depth: 1.00,
            width: 1.00,
            max_ch: 512,
        }
    }
    pub fn x() -> Self {
        Self {
            depth: 1.00,
            width: 1.50,
            max_ch: 512,
        }
    }

    /// Whether ALL C3k2 blocks use C3k branches.
    /// Ultralytics parse_model: `if scale in "mlx": args[3] = True`.
    pub fn c3k_all(&self) -> bool {
        *self == Self::m() || *self == Self::l() || *self == Self::x()
    }
}

/// Compute scaled channel count: `make_divisible(min(base, max_ch) * width, 8)`.
/// Matches ultralytics parse_model: clamp to max_ch first, then scale, then align to 8.
fn ch(base: usize, m: Multiples) -> usize {
    let raw: f64 = base.min(m.max_ch) as f64 * m.width;
    ((raw / 8.0).ceil() as usize) * 8
}

/// Compute scaled depth (repeat count): max(round(base * depth), 1).
fn depth(base: usize, m: Multiples) -> usize {
    (base as f64 * m.depth).round().max(1.0) as usize
}

// ---------------------------------------------------------------------------
// ConvBlock: Conv2d + BatchNorm (fused via absorb_bn) + optional SiLU
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct ConvBlock {
    conv: Conv2d,
    is_activated: bool,
    span: tracing::Span,
}

impl ConvBlock {
    fn load(
        vb: VarBuilder,
        c_in: usize,
        c_out: usize,
        k: usize,
        stride: usize,
        groups: usize,
        is_activated: bool,
    ) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: k / 2,
            stride,
            groups,
            ..Default::default()
        };
        let bn = batch_norm(c_out, 1e-3, vb.pp("bn"))?;
        let conv = conv2d_no_bias(c_in, c_out, k, cfg, vb.pp("conv"))?.absorb_bn(&bn)?;
        Ok(Self {
            conv,
            is_activated,
            span: tracing::span!(tracing::Level::TRACE, "conv-block"),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x = self.conv.forward(x)?;
        if self.is_activated {
            candle_nn::ops::silu(&x)
        } else {
            Ok(x)
        }
    }
}

// ---------------------------------------------------------------------------
// Bottleneck: cv1(c1→c_hidden) + cv2(c_hidden→c2) with optional residual
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Bottleneck {
    cv1: ConvBlock,
    cv2: ConvBlock,
    residual: bool,
    span: tracing::Span,
}

impl Bottleneck {
    fn load(
        vb: VarBuilder,
        c_in: usize,
        c_out: usize,
        shortcut: bool,
        k: (usize, usize),
        expansion: f64,
    ) -> Result<Self> {
        let c_hidden = (c_out as f64 * expansion) as usize;
        let cv1 = ConvBlock::load(vb.pp("cv1"), c_in, c_hidden, k.0, 1, 1, true)?;
        let cv2 = ConvBlock::load(vb.pp("cv2"), c_hidden, c_out, k.1, 1, 1, true)?;
        let residual = shortcut && c_in == c_out;
        Ok(Self {
            cv1,
            cv2,
            residual,
            span: tracing::span!(tracing::Level::TRACE, "bottleneck"),
        })
    }
}

impl Module for Bottleneck {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let y = self.cv2.forward(&self.cv1.forward(x)?)?;
        if self.residual {
            x + y
        } else {
            Ok(y)
        }
    }
}

// ---------------------------------------------------------------------------
// C3k: C3 variant with k=3 bottleneck kernels
// cv1→split→Sequential(Bottleneck(e=1.0,k=3))→cat→cv3
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct C3k {
    cv1: ConvBlock,
    cv2: ConvBlock,
    cv3: ConvBlock,
    m: Vec<Bottleneck>,
    span: tracing::Span,
}

impl C3k {
    fn load(vb: VarBuilder, c_in: usize, c_out: usize, n: usize, shortcut: bool) -> Result<Self> {
        let c_hidden = c_out / 2;
        let cv1 = ConvBlock::load(vb.pp("cv1"), c_in, c_hidden, 1, 1, 1, true)?;
        let cv2 = ConvBlock::load(vb.pp("cv2"), c_in, c_hidden, 1, 1, 1, true)?;
        let cv3 = ConvBlock::load(vb.pp("cv3"), 2 * c_hidden, c_out, 1, 1, 1, true)?;
        let mut m = Vec::with_capacity(n);
        for i in 0..n {
            m.push(Bottleneck::load(
                vb.pp("m").pp(i.to_string()),
                c_hidden,
                c_hidden,
                shortcut,
                (3, 3),
                1.0,
            )?);
        }
        Ok(Self {
            cv1,
            cv2,
            cv3,
            m,
            span: tracing::span!(tracing::Level::TRACE, "c3k"),
        })
    }
}

impl Module for C3k {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let mut a = self.cv1.forward(x)?;
        for bn in &self.m {
            a = bn.forward(&a)?;
        }
        let b = self.cv2.forward(x)?;
        self.cv3.forward(&Tensor::cat(&[&a, &b], 1)?)
    }
}

// ---------------------------------------------------------------------------
// Attention: Multi-head self-attention via 1×1 Conv QKV + DWConv PE
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Attention {
    qkv: ConvBlock,
    proj: ConvBlock,
    pe: ConvBlock,
    num_heads: usize,
    key_dim: usize,
    head_dim: usize,
    scale: f64,
    span: tracing::Span,
}

impl Attention {
    fn load(vb: VarBuilder, dim: usize, num_heads: usize) -> Result<Self> {
        let head_dim = dim / num_heads;
        let key_dim = head_dim / 2; // attn_ratio=0.5
        let nh_kd = num_heads * key_dim;
        let h = dim + nh_kd * 2;
        let qkv = ConvBlock::load(vb.pp("qkv"), dim, h, 1, 1, 1, false)?;
        let proj = ConvBlock::load(vb.pp("proj"), dim, dim, 1, 1, 1, false)?;
        let pe = ConvBlock::load(vb.pp("pe"), dim, dim, 3, 1, dim, false)?;
        let scale = (key_dim as f64).powf(-0.5);
        Ok(Self {
            qkv,
            proj,
            pe,
            num_heads,
            key_dim,
            head_dim,
            scale,
            span: tracing::span!(tracing::Level::TRACE, "attention"),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b, c, h, w) = x.dims4()?;
        let n = h * w;
        let qkv = self.qkv.forward(x)?;
        let qkv = qkv.reshape((b, self.num_heads, self.key_dim * 2 + self.head_dim, n))?;
        let q = qkv.narrow(2, 0, self.key_dim)?;
        let k = qkv.narrow(2, self.key_dim, self.key_dim)?;
        let v = qkv.narrow(2, self.key_dim * 2, self.head_dim)?;
        let attn = q.transpose(2, 3)?.matmul(&k)?.affine(self.scale, 0.0)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = v.matmul(&attn.transpose(2, 3)?)?;
        let out = out.reshape((b, c, h, w))?;
        let v_spatial = v.reshape((b, c, h, w))?;
        let pe = self.pe.forward(&v_spatial)?;
        let out = (out + pe)?;
        self.proj.forward(&out)
    }
}

// ---------------------------------------------------------------------------
// PSABlock: Attention + FFN with residual connections
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct PsaBlock {
    attn: Attention,
    ffn_0: ConvBlock,
    ffn_1: ConvBlock,
    span: tracing::Span,
}

impl PsaBlock {
    fn load(vb: VarBuilder, dim: usize, num_heads: usize) -> Result<Self> {
        let attn = Attention::load(vb.pp("attn"), dim, num_heads)?;
        let ffn_0 = ConvBlock::load(vb.pp("ffn").pp("0"), dim, dim * 2, 1, 1, 1, true)?;
        let ffn_1 = ConvBlock::load(vb.pp("ffn").pp("1"), dim * 2, dim, 1, 1, 1, false)?;
        Ok(Self {
            attn,
            ffn_0,
            ffn_1,
            span: tracing::span!(tracing::Level::TRACE, "psa-block"),
        })
    }
}

impl Module for PsaBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x = (x + self.attn.forward(x)?)?;
        let ffn_out = self.ffn_1.forward(&self.ffn_0.forward(&x)?)?;
        &x + ffn_out
    }
}

// ---------------------------------------------------------------------------
// AttnBranch: nn.Sequential(Bottleneck(e=0.5), PSABlock)
// Used by C3k2 when attn=True.
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct AttnBranch {
    bottleneck: Bottleneck,
    psa: PsaBlock,
}

impl AttnBranch {
    fn load(vb: VarBuilder, c: usize, shortcut: bool) -> Result<Self> {
        let bottleneck = Bottleneck::load(vb.pp("0"), c, c, shortcut, (3, 3), 0.5)?;
        let num_heads = (c / 64).max(1);
        let psa = PsaBlock::load(vb.pp("1"), c, num_heads)?;
        Ok(Self { bottleneck, psa })
    }
}

impl Module for AttnBranch {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.psa.forward(&self.bottleneck.forward(x)?)
    }
}

// ---------------------------------------------------------------------------
// C3k2: C2f variant with configurable branch type
// When attn=True: AttnBranch (takes priority over c3k)
// When c3k=True: C3k; Otherwise: Bottleneck(e=0.5)
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum C3k2Branch {
    Bottleneck(Box<Bottleneck>),
    C3k(Box<C3k>),
    Attn(Box<AttnBranch>),
}

impl Module for C3k2Branch {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            C3k2Branch::Bottleneck(b) => b.forward(x),
            C3k2Branch::C3k(c) => c.forward(x),
            C3k2Branch::Attn(a) => a.forward(x),
        }
    }
}

#[derive(Debug)]
struct C3k2 {
    cv1: ConvBlock,
    cv2: ConvBlock,
    branches: Vec<C3k2Branch>,
    span: tracing::Span,
}

impl C3k2 {
    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: VarBuilder,
        c_in: usize,
        c_out: usize,
        n: usize,
        is_c3k: bool,
        expansion: f64,
        shortcut: bool,
        has_attn: bool,
    ) -> Result<Self> {
        let c_hidden = (c_out as f64 * expansion) as usize;
        let cv1 = ConvBlock::load(vb.pp("cv1"), c_in, 2 * c_hidden, 1, 1, 1, true)?;
        let cv2 = ConvBlock::load(vb.pp("cv2"), (2 + n) * c_hidden, c_out, 1, 1, 1, true)?;
        let mut branches = Vec::with_capacity(n);
        for i in 0..n {
            let branch_vb = vb.pp("m").pp(i.to_string());
            let branch = if has_attn {
                C3k2Branch::Attn(Box::new(AttnBranch::load(branch_vb, c_hidden, shortcut)?))
            } else if is_c3k {
                C3k2Branch::C3k(Box::new(C3k::load(
                    branch_vb, c_hidden, c_hidden, 2, shortcut,
                )?))
            } else {
                C3k2Branch::Bottleneck(Box::new(Bottleneck::load(
                    branch_vb,
                    c_hidden,
                    c_hidden,
                    shortcut,
                    (3, 3),
                    0.5,
                )?))
            };
            branches.push(branch);
        }
        Ok(Self {
            cv1,
            cv2,
            branches,
            span: tracing::span!(tracing::Level::TRACE, "c3k2"),
        })
    }
}

impl Module for C3k2 {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let y = self.cv1.forward(x)?;
        let mut chunks = y.chunk(2, 1)?;
        for branch in &self.branches {
            let last = chunks.last().unwrap();
            chunks.push(branch.forward(last)?);
        }
        self.cv2
            .forward(&Tensor::cat(&chunks.iter().collect::<Vec<_>>(), 1)?)
    }
}

// ---------------------------------------------------------------------------
// C2PSA: Conv split → n PSABlock → Conv merge
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct C2psa {
    cv1: ConvBlock,
    cv2: ConvBlock,
    m: Vec<PsaBlock>,
    c_split: usize,
    span: tracing::Span,
}

impl C2psa {
    fn load(vb: VarBuilder, c_in: usize, c_out: usize, n: usize) -> Result<Self> {
        let c_split = c_in / 2;
        let cv1 = ConvBlock::load(vb.pp("cv1"), c_in, 2 * c_split, 1, 1, 1, true)?;
        let cv2 = ConvBlock::load(vb.pp("cv2"), 2 * c_split, c_out, 1, 1, 1, true)?;
        let num_heads = (c_split / 64).max(1);
        let mut m = Vec::with_capacity(n);
        for i in 0..n {
            m.push(PsaBlock::load(
                vb.pp("m").pp(i.to_string()),
                c_split,
                num_heads,
            )?);
        }
        Ok(Self {
            cv1,
            cv2,
            m,
            c_split,
            span: tracing::span!(tracing::Level::TRACE, "c2psa"),
        })
    }
}

impl Module for C2psa {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let y = self.cv1.forward(x)?;
        let a = y.narrow(1, 0, self.c_split)?;
        let mut b = y.narrow(1, self.c_split, self.c_split)?;
        for psa in &self.m {
            b = psa.forward(&b)?;
        }
        self.cv2.forward(&Tensor::cat(&[&a, &b], 1)?)
    }
}

// ---------------------------------------------------------------------------
// SPPF: Spatial Pyramid Pooling Fast
// cv1(act=false) → n sequential MaxPool2d → cat → cv2 [+ shortcut]
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Sppf {
    cv1: ConvBlock,
    cv2: ConvBlock,
    k: usize,
    pool_count: usize,
    has_shortcut: bool,
    span: tracing::Span,
}

impl Sppf {
    fn load(
        vb: VarBuilder,
        c_in: usize,
        c_out: usize,
        k: usize,
        pool_count: usize,
        shortcut: bool,
    ) -> Result<Self> {
        let c_hidden = c_in / 2;
        // YOLO26 SPPF: cv1 uses act=False
        let cv1 = ConvBlock::load(vb.pp("cv1"), c_in, c_hidden, 1, 1, 1, false)?;
        let cv2 = ConvBlock::load(
            vb.pp("cv2"),
            c_hidden * (pool_count + 1),
            c_out,
            1,
            1,
            1,
            true,
        )?;
        let has_shortcut = shortcut && c_in == c_out;
        Ok(Self {
            cv1,
            cv2,
            k,
            pool_count,
            has_shortcut,
            span: tracing::span!(tracing::Level::TRACE, "sppf"),
        })
    }
}

impl Module for Sppf {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x = self.cv1.forward(input)?;
        let mut pools = vec![x];
        for _ in 0..self.pool_count {
            let prev = pools.last().unwrap();
            pools.push(maxpool2d_neg_inf_pad(prev, self.k)?);
        }
        let cat = Tensor::cat(&pools.iter().collect::<Vec<_>>(), 1)?;
        let out = self.cv2.forward(&cat)?;
        if self.has_shortcut {
            &out + input
        } else {
            Ok(out)
        }
    }
}

/// MaxPool2d with -inf padding (not zero padding).
/// candle's max_pool2d has no padding parameter, so we manually pad with -inf
/// to match PyTorch's MaxPool2d(k, stride=1, padding=k//2).
fn maxpool2d_neg_inf_pad(x: &Tensor, k: usize) -> Result<Tensor> {
    let pad = k / 2;
    let (b, c, h, w) = x.dims4()?;
    let dev = x.device();
    let top = Tensor::full(f32::NEG_INFINITY, (b, c, pad, w), dev)?.contiguous()?;
    let bottom = Tensor::full(f32::NEG_INFINITY, (b, c, pad, w), dev)?.contiguous()?;
    let x = Tensor::cat(&[&top, x, &bottom], 2)?;
    let h_padded = h + 2 * pad;
    let left = Tensor::full(f32::NEG_INFINITY, (b, c, h_padded, pad), dev)?.contiguous()?;
    let right = Tensor::full(f32::NEG_INFINITY, (b, c, h_padded, pad), dev)?.contiguous()?;
    let x = Tensor::cat(&[&left, &x, &right], 3)?;
    x.max_pool2d_with_stride(k, 1)
}

// ---------------------------------------------------------------------------
// DarkNet26: Backbone (layers 0-10)
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct DarkNet26 {
    l0: ConvBlock,
    l1: ConvBlock,
    l2: C3k2,
    l3: ConvBlock,
    l4: C3k2,
    l5: ConvBlock,
    l6: C3k2,
    l7: ConvBlock,
    l8: C3k2,
    l9: Sppf,
    l10: C2psa,
    span: tracing::Span,
}

impl DarkNet26 {
    fn load(vb: VarBuilder, m: Multiples) -> Result<Self> {
        let d2 = depth(2, m); // repeat count for C3k2/C2PSA (base=2)
        // For m/l/x: all C3k2 use c3k=True (ultralytics parse_model override)
        let c3k_24 = m.c3k_all();
        let l0 = ConvBlock::load(vb.pp("0"), 3, ch(64, m), 3, 2, 1, true)?;
        let l1 = ConvBlock::load(vb.pp("1"), ch(64, m), ch(128, m), 3, 2, 1, true)?;
        let l2 = C3k2::load(
            vb.pp("2"),
            ch(128, m),
            ch(256, m),
            d2,
            c3k_24,
            0.25,
            true,
            false,
        )?;
        let l3 = ConvBlock::load(vb.pp("3"), ch(256, m), ch(256, m), 3, 2, 1, true)?;
        let l4 = C3k2::load(
            vb.pp("4"),
            ch(256, m),
            ch(512, m),
            d2,
            c3k_24,
            0.25,
            true,
            false,
        )?;
        let l5 = ConvBlock::load(vb.pp("5"), ch(512, m), ch(512, m), 3, 2, 1, true)?;
        let l6 = C3k2::load(
            vb.pp("6"),
            ch(512, m),
            ch(512, m),
            d2,
            true,
            0.5,
            true,
            false,
        )?;
        let l7 = ConvBlock::load(vb.pp("7"), ch(512, m), ch(1024, m), 3, 2, 1, true)?;
        let l8 = C3k2::load(
            vb.pp("8"),
            ch(1024, m),
            ch(1024, m),
            d2,
            true,
            0.5,
            true,
            false,
        )?;
        let l9 = Sppf::load(vb.pp("9"), ch(1024, m), ch(1024, m), 5, 3, true)?;
        let l10 = C2psa::load(vb.pp("10"), ch(1024, m), ch(1024, m), d2)?;
        Ok(Self {
            l0,
            l1,
            l2,
            l3,
            l4,
            l5,
            l6,
            l7,
            l8,
            l9,
            l10,
            span: tracing::span!(tracing::Level::TRACE, "darknet26"),
        })
    }

    /// Returns (p3, p4, p5) for neck skip connections.
    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let _enter = self.span.enter();
        let x = self.l0.forward(x)?;
        let x = self.l1.forward(&x)?;
        let x = self.l2.forward(&x)?;
        let x = self.l3.forward(&x)?;
        let p3 = self.l4.forward(&x)?;
        let x = self.l5.forward(&p3)?;
        let p4 = self.l6.forward(&x)?;
        let x = self.l7.forward(&p4)?;
        let x = self.l8.forward(&x)?;
        let x = self.l9.forward(&x)?;
        let p5 = self.l10.forward(&x)?;
        Ok((p3, p4, p5))
    }
}

// ---------------------------------------------------------------------------
// YoloV26Neck: FPN-PAN (layers 11-22)
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct YoloV26Neck {
    l13: C3k2,
    l16: C3k2,
    l17: ConvBlock,
    l19: C3k2,
    l20: ConvBlock,
    l22: C3k2,
    span: tracing::Span,
}

impl YoloV26Neck {
    fn load(vb: VarBuilder, m: Multiples) -> Result<Self> {
        let d2 = depth(2, m);
        // FPN (top-down)
        let l13 = C3k2::load(
            vb.pp("13"),
            ch(1024, m) + ch(512, m),
            ch(512, m),
            d2,
            true,
            0.5,
            true,
            false,
        )?;
        let l16 = C3k2::load(
            vb.pp("16"),
            ch(512, m) + ch(512, m),
            ch(256, m),
            d2,
            true,
            0.5,
            true,
            false,
        )?;
        // PAN (bottom-up)
        let l17 = ConvBlock::load(vb.pp("17"), ch(256, m), ch(256, m), 3, 2, 1, true)?;
        let l19 = C3k2::load(
            vb.pp("19"),
            ch(256, m) + ch(512, m),
            ch(512, m),
            d2,
            true,
            0.5,
            true,
            false,
        )?;
        let l20 = ConvBlock::load(vb.pp("20"), ch(512, m), ch(512, m), 3, 2, 1, true)?;
        let l22 = C3k2::load(
            vb.pp("22"),
            ch(512, m) + ch(1024, m),
            ch(1024, m),
            1, // always 1 repeat for layer 22
            true,
            0.5,
            true,
            true, // attn=True
        )?;
        Ok(Self {
            l13,
            l16,
            l17,
            l19,
            l20,
            l22,
            span: tracing::span!(tracing::Level::TRACE, "yolo-v26-neck"),
        })
    }

    fn forward(&self, p3: &Tensor, p4: &Tensor, p5: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let _enter = self.span.enter();
        // FPN: top-down
        let (_, _, h4, w4) = p4.dims4()?;
        let up5 = p5.upsample_nearest2d(h4, w4)?;
        let l13 = self.l13.forward(&Tensor::cat(&[&up5, p4], 1)?)?;

        let (_, _, h3, w3) = p3.dims4()?;
        let up13 = l13.upsample_nearest2d(h3, w3)?;
        let small = self.l16.forward(&Tensor::cat(&[&up13, p3], 1)?)?;

        // PAN: bottom-up
        let l17 = self.l17.forward(&small)?;
        let medium = self.l19.forward(&Tensor::cat(&[&l17, &l13], 1)?)?;

        let l20 = self.l20.forward(&medium)?;
        let large = self.l22.forward(&Tensor::cat(&[&l20, p5], 1)?)?;

        Ok((small, medium, large))
    }
}

// ---------------------------------------------------------------------------
// DetectionHead26: End-to-end detection with one2one branches + topk
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct BoxBranch {
    cv0: ConvBlock,
    cv1: ConvBlock,
    cv2: Conv2d,
}

#[derive(Debug)]
struct ClsBranch {
    dw0: ConvBlock,
    cv0: ConvBlock,
    dw1: ConvBlock,
    cv1: ConvBlock,
    cv2: Conv2d,
}

#[derive(Debug)]
struct DetectionHead26 {
    box_branches: Vec<BoxBranch>,
    cls_branches: Vec<ClsBranch>,
    strides: Vec<f32>,
    nc: usize,
    max_det: usize,
    span: tracing::Span,
}

impl DetectionHead26 {
    fn load(vb: VarBuilder, m: Multiples, nc: usize) -> Result<Self> {
        let input_channels = [ch(256, m), ch(512, m), ch(1024, m)];
        let reg_max: usize = 1;
        let c2 = 16_usize.max(input_channels[0] / 4).max(reg_max * 4);
        let c3 = input_channels[0].max(nc.min(100));
        let strides = vec![8.0, 16.0, 32.0];
        let conv2d_cfg = Conv2dConfig::default();

        let mut box_branches = Vec::new();
        let mut cls_branches = Vec::new();

        for (i, &c) in input_channels.iter().enumerate() {
            let bvb = vb.pp("one2one_cv2").pp(i.to_string());
            box_branches.push(BoxBranch {
                cv0: ConvBlock::load(bvb.pp("0"), c, c2, 3, 1, 1, true)?,
                cv1: ConvBlock::load(bvb.pp("1"), c2, c2, 3, 1, 1, true)?,
                cv2: conv2d(c2, 4 * reg_max, 1, conv2d_cfg, bvb.pp("2"))?,
            });

            let cvb = vb.pp("one2one_cv3").pp(i.to_string());
            cls_branches.push(ClsBranch {
                dw0: ConvBlock::load(cvb.pp("0").pp("0"), c, c, 3, 1, c, true)?,
                cv0: ConvBlock::load(cvb.pp("0").pp("1"), c, c3, 1, 1, 1, true)?,
                dw1: ConvBlock::load(cvb.pp("1").pp("0"), c3, c3, 3, 1, c3, true)?,
                cv1: ConvBlock::load(cvb.pp("1").pp("1"), c3, c3, 1, 1, 1, true)?,
                cv2: conv2d(c3, nc, 1, conv2d_cfg, cvb.pp("2"))?,
            });
        }

        Ok(Self {
            box_branches,
            cls_branches,
            strides,
            nc,
            max_det: 300,
            span: tracing::span!(tracing::Level::TRACE, "detection-head-26"),
        })
    }

    fn forward(&self, xs0: &Tensor, xs1: &Tensor, xs2: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let dev = xs0.device();
        let batch_size = xs0.dim(0)?;
        let features = [xs0, xs1, xs2];

        let mut all_boxes = Vec::new();
        let mut all_scores = Vec::new();
        let mut feat_sizes = Vec::new();

        for (i, feat) in features.iter().enumerate() {
            let (_, _, h, w) = feat.dims4()?;
            feat_sizes.push((h, w));

            let bx = self.box_branches[i].cv0.forward(feat)?;
            let bx = self.box_branches[i].cv1.forward(&bx)?;
            let bx = self.box_branches[i].cv2.forward(&bx)?;
            all_boxes.push(bx.reshape((batch_size, 4, h * w))?);

            let cx = self.cls_branches[i].dw0.forward(feat)?;
            let cx = self.cls_branches[i].cv0.forward(&cx)?;
            let cx = self.cls_branches[i].dw1.forward(&cx)?;
            let cx = self.cls_branches[i].cv1.forward(&cx)?;
            let cx = self.cls_branches[i].cv2.forward(&cx)?;
            all_scores.push(cx.reshape((batch_size, self.nc, h * w))?);
        }

        let boxes = Tensor::cat(&all_boxes.iter().collect::<Vec<_>>(), 2)?;
        let scores = Tensor::cat(&all_scores.iter().collect::<Vec<_>>(), 2)?;

        let (anchors, stride_tensor) = make_anchors(&feat_sizes, &self.strides, dev)?;
        let dbox = dist2bbox_xyxy(&boxes, &anchors)?;
        let dbox = dbox.broadcast_mul(&stride_tensor)?;
        let cls_scores = candle_nn::ops::sigmoid(&scores)?;

        // [B, 4+nc, N] → [B, N, 4+nc]
        let y = Tensor::cat(&[&dbox, &cls_scores], 1)?.transpose(1, 2)?;

        self.topk_postprocess(&y, batch_size)
    }

    /// Top-k selection: [B, N, 4+nc] → [B, 300, 6] = [x1, y1, x2, y2, conf, cls_id]
    fn topk_postprocess(&self, preds: &Tensor, batch_size: usize) -> Result<Tensor> {
        let n_anchors = preds.dim(1)?;
        let k = self.max_det.min(n_anchors);
        let preds = preds.to_device(&Device::Cpu)?;
        let boxes = preds.narrow(2, 0, 4)?;
        let scores = preds.narrow(2, 4, self.nc)?;

        let mut all_outputs = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let scores_2d: Vec<f32> = scores.get(b)?.flatten_all()?.to_vec1()?;
            let boxes_2d: Vec<f32> = boxes.get(b)?.flatten_all()?.to_vec1()?;

            // Find max class score and class id per anchor
            let mut scored: Vec<(usize, f32, u32)> = (0..n_anchors)
                .map(|i| {
                    let base = i * self.nc;
                    let (max_idx, &max_val) = scores_2d[base..base + self.nc]
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .unwrap();
                    (i, max_val, max_idx as u32)
                })
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let mut output = Vec::with_capacity(k * 6);
            for &(idx, score, cls_id) in scored.iter().take(k) {
                let base = idx * 4;
                output.push(boxes_2d[base]);
                output.push(boxes_2d[base + 1]);
                output.push(boxes_2d[base + 2]);
                output.push(boxes_2d[base + 3]);
                output.push(score);
                output.push(cls_id as f32);
            }
            all_outputs.push(Tensor::from_vec(output, (1, k, 6), &Device::Cpu)?);
        }
        let result = Tensor::cat(&all_outputs, 0)?;
        result.to_device(preds.device())
    }
}

/// Generate anchor points and stride tensor for all feature map scales.
fn make_anchors(
    feat_sizes: &[(usize, usize)],
    strides: &[f32],
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let mut anchor_data = Vec::new();
    let mut stride_data = Vec::new();
    for (i, &(h, w)) in feat_sizes.iter().enumerate() {
        let stride = strides[i];
        for y in 0..h {
            for x in 0..w {
                anchor_data.push(x as f32 + 0.5);
                anchor_data.push(y as f32 + 0.5);
                stride_data.push(stride);
            }
        }
    }
    let n = stride_data.len();
    let anchors = Tensor::from_vec(anchor_data, (n, 2), device)?
        .transpose(0, 1)?
        .unsqueeze(0)?; // [1, 2, N]
    let stride_tensor = Tensor::from_vec(stride_data, (1, 1, n), device)?;
    Ok((anchors, stride_tensor))
}

/// dist2bbox in xyxy mode: x1y1 = anchor - lt, x2y2 = anchor + rb
fn dist2bbox_xyxy(distance: &Tensor, anchors: &Tensor) -> Result<Tensor> {
    let lt = distance.narrow(1, 0, 2)?;
    let rb = distance.narrow(1, 2, 2)?;
    let x1y1 = anchors.broadcast_sub(&lt)?;
    let x2y2 = anchors.broadcast_add(&rb)?;
    Tensor::cat(&[&x1y1, &x2y2], 1)
}

// ---------------------------------------------------------------------------
// YoloV26: Top-level model
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct YoloV26 {
    net: DarkNet26,
    fpn: YoloV26Neck,
    head: DetectionHead26,
    span: tracing::Span,
}

impl YoloV26 {
    pub fn load(vb: VarBuilder, m: Multiples, nc: usize) -> Result<Self> {
        let vb = vb.pp("model");
        let net = DarkNet26::load(vb.clone(), m)?;
        let fpn = YoloV26Neck::load(vb.clone(), m)?;
        let head = DetectionHead26::load(vb.pp("23"), m, nc)?;
        Ok(Self {
            net,
            fpn,
            head,
            span: tracing::span!(tracing::Level::TRACE, "yolo-v26"),
        })
    }
}

impl Module for YoloV26 {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (p3, p4, p5) = self.net.forward(xs)?;
        let (small, medium, large) = self.fpn.forward(&p3, &p4, &p5)?;
        self.head.forward(&small, &medium, &large)
    }
}

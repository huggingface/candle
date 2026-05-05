// Userland prototype of bilinear-with-antialias interpolation in candle.
//
// Approximates F.interpolate(mode="bilinear", align_corners=False, antialias=True)
// from PyTorch using only existing candle tensor ops (matmul + small dense
// weight matrices built per axis). Validated against reference outputs at
// /tmp/antialias-reference/antialias_reference.safetensors.
//
// The matmul approach matches PyTorch's algorithm: PyTorch's antialiased
// bilinear is a one-pass weighted sum where each output pixel is a weighted
// average over input pixels in a window of width max(1, 1/scale), with
// triangle weights. That is exactly W_h @ input @ W_w^T where W_h, W_w are
// per-axis weight matrices. Separable.
//
// Run with:
//   cargo run --release -p candle-examples --example antialias-prototype \
//       --no-default-features --features accelerate -- \
//       /tmp/antialias-reference/antialias_reference.safetensors

use anyhow::{bail, Result};
use candle::{DType, Device, Tensor};

/// Build a per-axis weight matrix W of shape (dst, src) such that for each
/// output index `o`, W[o, i] is the triangle weight applied to input pixel `i`,
/// normalized so the row sums to 1.
///
/// Matches torch.nn.functional.interpolate(mode="bilinear", align_corners=False,
/// antialias=True) along a single axis.
fn make_weight_matrix(src: usize, dst: usize, device: &Device) -> Result<Tensor> {
    let scale = dst as f64 / src as f64;
    // Triangle filter has support 1.0 in output coords; widened to 1/scale on
    // downsampling. Upsampling falls back to standard bilinear (support = 1).
    let support = (1.0_f64 / scale).max(1.0);

    let mut weights = vec![0f32; dst * src];

    for o in 0..dst {
        // Center of output pixel `o` mapped into input coords (align_corners=False).
        let center = (o as f64 + 0.5) / scale - 0.5;

        // Range of input indices contributing to this output pixel.
        let i_min = ((center - support).floor() as i64).max(0);
        let i_max_excl = ((center + support).ceil() as i64 + 1).min(src as i64);

        let mut row_sum = 0f32;
        for i in i_min..i_max_excl {
            let dist = ((i as f64 - center).abs() / support) as f32;
            let w = (1.0 - dist).max(0.0);
            weights[o * src + i as usize] = w;
            row_sum += w;
        }
        if row_sum > 0.0 {
            for i in i_min..i_max_excl {
                weights[o * src + i as usize] /= row_sum;
            }
        }
    }

    Ok(Tensor::from_vec(weights, (dst, src), device)?)
}

/// Bilinear interpolation with antialias filter, pure candle tensor ops.
///
/// `align_corners=False` only (matches PyTorch default; matches what HF
/// transformers SigLIP2 uses). Input must be (B, C, H, W).
pub fn upsample_bilinear2d_antialias(
    input: &Tensor,
    target_h: usize,
    target_w: usize,
) -> Result<Tensor> {
    let (b, c, h, w) = input.dims4()?;
    let device = input.device();
    let dtype = input.dtype();

    let w_h = make_weight_matrix(h, target_h, device)?.to_dtype(dtype)?; // (dst_h, src_h)
    let w_w = make_weight_matrix(w, target_w, device)?.to_dtype(dtype)?; // (dst_w, src_w)
    let w_w_t = w_w.t()?.contiguous()?; // (src_w, dst_w)

    // Reshape to (B*C, H, W) so we can broadcast-matmul a single 2D weight.
    let x = input.reshape((b * c, h, w))?.contiguous()?;
    let x = w_h.broadcast_matmul(&x)?; // (B*C, dst_h, W)
    let x = x.contiguous()?;
    let x = x.broadcast_matmul(&w_w_t)?; // (B*C, dst_h, dst_w)
    Ok(x.reshape((b, c, target_h, target_w))?)
}

fn diff_stats(label: &str, expected: &Tensor, actual: &Tensor) -> Result<(f32, f32)> {
    let diff = (actual.to_dtype(DType::F32)? - expected.to_dtype(DType::F32)?)?;
    let abs = diff.abs()?;
    let max = abs.max_all()?.to_scalar::<f32>()?;
    let mean = abs.mean_all()?.to_scalar::<f32>()?;
    println!("  {:30}  max={:.3e}  mean={:.3e}", label, max, mean);
    Ok((max, mean))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let ref_path = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("/tmp/antialias-reference/antialias_reference.safetensors");

    println!("Loading reference fixtures from: {}", ref_path);
    let device = Device::Cpu;
    let buf = std::fs::read(ref_path)?;
    let tensors = candle::safetensors::load_buffer(&buf, &device)?;
    println!("Loaded {} tensors\n", tensors.len());

    // Cases the Python script generated. Pull each case's input + expected
    // and compare against our prototype.
    let cases = [
        "siglip2_16x16_to_17x15",
        "siglip2_16x16_to_19x13",
        "siglip2_16x16_to_13x17",
        "siglip2_16x16_to_8x8",
        "siglip2_16x16_to_24x24",
        "synth_8x8_to_4x4",
        "synth_8x8_to_16x16",
        "synth_4x4_to_7x9",
    ];

    let tol = 1e-4_f32;
    let mut all_pass = true;
    let mut worst_max = 0f32;

    for name in cases {
        let input_key = format!("{name}__input");
        let exp_key = format!("{name}__expected");
        let input = tensors
            .get(&input_key)
            .ok_or_else(|| anyhow::anyhow!("missing {}", input_key))?;
        let expected = tensors
            .get(&exp_key)
            .ok_or_else(|| anyhow::anyhow!("missing {}", exp_key))?;

        let (_, _, dh, dw) = expected.dims4()?;
        let actual = upsample_bilinear2d_antialias(input, dh, dw)?;

        println!("=== {name} ===");
        let (max_aa, _) = diff_stats("antialias prototype vs PT", expected, &actual)?;

        // Compare to candle's plain bilinear (no antialias) for context.
        let plain = input.upsample_bilinear2d(dh, dw, false)?;
        let _ = diff_stats("candle bilinear vs PT       ", expected, &plain)?;

        // If a "no antialias" reference exists, sanity-check candle bilinear
        // matches PT bilinear-without-antialias closely.
        let noaa_key = format!("{name}__expected_no_antialias");
        if let Some(no_aa) = tensors.get(&noaa_key) {
            let _ = diff_stats("candle bilinear vs PT (no AA)", no_aa, &plain)?;
        }

        if max_aa > tol {
            println!("  ❌ FAIL (tol={:.0e})", tol);
            all_pass = false;
        } else {
            println!("  ✅ PASS (tol={:.0e})", tol);
        }
        worst_max = worst_max.max(max_aa);
        println!();
    }

    println!("Worst max abs diff across all cases: {:.3e}", worst_max);
    if !all_pass {
        bail!("at least one case exceeded tolerance {:.0e}", tol);
    }
    println!("All {} cases pass at tol {:.0e}", cases.len(), tol);
    Ok(())
}

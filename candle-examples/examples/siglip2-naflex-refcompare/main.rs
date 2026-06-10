// Tensor-level reference comparison: load HF NaFlex preprocessor outputs +
// PyTorch model's vision pooler_output, run candle's siglip2_naflex
// VisionModel directly on the same inputs, compare pooler outputs.
//
// Reference fixtures (pixel_values, pixel_attention_mask, spatial_shapes,
// vision_pooler_output) are generated separately from PyTorch + the HF NaFlex
// preprocessor. Pass the path to the .safetensors file as the first CLI arg,
// or set the SIGLIP2_NAFLEX_REFERENCE env var. Default is a local path used
// during development that won't exist on a fresh checkout.

use anyhow::Result;
use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::siglip2_naflex;

fn cosine_sim(a: &Tensor, b: &Tensor) -> Result<f32> {
    let dot = (a * b)?.sum_all()?.to_scalar::<f32>()?;
    let na = a.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
    let nb = b.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
    Ok(dot / (na * nb))
}

fn report_comparison(label: &str, expected: &Tensor, candle: &Tensor) -> Result<()> {
    let diff = (candle.to_dtype(DType::F32)? - expected.to_dtype(DType::F32)?)?;
    let abs_diff = diff.abs()?;
    let max_diff = abs_diff.max_all()?.to_scalar::<f32>()?;
    let mean_diff = abs_diff.mean_all()?.to_scalar::<f32>()?;
    println!("\n=== {label} ===");
    println!("max abs diff: {:.6e}", max_diff);
    println!("mean abs diff: {:.6e}", mean_diff);
    let n = expected.dim(0)?;
    for row in 0..n {
        let e = expected.i((row,))?;
        let c = candle.i((row,))?;
        let cos = cosine_sim(&e, &c)?;
        let ne = e.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        let nc = c.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        println!(
            "  Row {}: cosine={:.6}  ||expected||={:.4}  ||candle||={:.4}",
            row, cos, ne, nc
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model("google/siglip2-base-patch16-naflex".to_string());
    let model_file = repo.get("model.safetensors")?;
    let config_file = repo.get("config.json")?;

    let config: siglip2_naflex::Config = serde_json::from_slice(&std::fs::read(config_file)?)?;
    let device = Device::Cpu;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };
    let vision_model =
        siglip2_naflex::VisionModel::new(&config.vision_config, vb.pp("vision_model"))?;
    println!("VisionModel loaded.");

    // Load reference outputs (variable-shape: [[17,15],[19,13]]). Path comes
    // from CLI arg or SIGLIP2_NAFLEX_REFERENCE env var; the default is a
    // dev-machine path that won't exist on a fresh checkout.
    let args: Vec<String> = std::env::args().collect();
    let ref_file = args
        .get(1)
        .cloned()
        .or_else(|| std::env::var("SIGLIP2_NAFLEX_REFERENCE").ok())
        .unwrap_or_else(|| {
            "/tmp/siglip2-practice-run/test_data_naflex/naflex_vision_reference.safetensors"
                .to_string()
        });
    let ref_buf = std::fs::read(&ref_file).map_err(|e| {
        anyhow::anyhow!(
            "failed to read reference fixtures at {}: {}. \
             Pass the path as arg 1 or set SIGLIP2_NAFLEX_REFERENCE.",
            ref_file,
            e,
        )
    })?;
    let ref_tensors = candle::safetensors::load_buffer(&ref_buf, &device)?;

    let pixel_values = ref_tensors.get("pixel_values").unwrap();
    let pixel_attention_mask = ref_tensors.get("pixel_attention_mask").unwrap();
    let spatial_shapes = ref_tensors.get("spatial_shapes").unwrap();
    let expected_pooler = ref_tensors.get("vision_pooler_output").unwrap();

    println!("\n--- TEST 1: variable-shape from HF preprocessor ---");
    println!("spatial_shapes: {:?}", spatial_shapes.to_vec2::<i64>()?);
    let candle_pooler = vision_model.forward(
        pixel_values,
        Some(pixel_attention_mask),
        Some(spatial_shapes),
        None,
    )?;
    report_comparison("variable-shape", expected_pooler, &candle_pooler)?;

    // TEST 2: re-run with the SAME pixel_values BUT pretend it's batch-uniform
    // 16x16 (which ignores the actual spatial_shapes [[17,15],[19,13]]).
    // This is a wrong test (different math) but shows the forward_uniform path runs.
    println!("\n--- TEST 2: forward_uniform 16x16 on the same pixel_values (sanity, NOT correct math) ---");
    let candle_pooler_uniform = vision_model.forward(pixel_values, None, None, Some((16, 16)))?;
    report_comparison(
        "uniform-16x16-on-padded-input",
        expected_pooler,
        &candle_pooler_uniform,
    )?;

    // TEST 3: drop the attention mask, keep the spatial_shapes
    println!("\n--- TEST 3: variable-shape WITHOUT attention mask ---");
    let candle_no_mask = vision_model.forward(pixel_values, None, Some(spatial_shapes), None)?;
    report_comparison("variable-shape-no-mask", expected_pooler, &candle_no_mask)?;

    Ok(())
}

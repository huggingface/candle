// Tensor-level reference comparison: load HF NaFlex preprocessor outputs +
// PyTorch model's vision pooler_output, run candle's siglip2_naflex
// VisionModel directly on the same inputs, compare pooler outputs.

use anyhow::Result;
use candle::{DType, Device, IndexOp, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::siglip2_naflex;

fn main() -> Result<()> {
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model("google/siglip2-base-patch16-naflex".to_string());
    let model_file = repo.get("model.safetensors")?;
    let config_file = repo.get("config.json")?;

    let config: siglip2_naflex::Config = serde_json::from_slice(&std::fs::read(config_file)?)?;
    let device = Device::Cpu;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };

    // Build VisionModel directly (need vision_model.* sub-tree of the vb)
    let vision_model =
        siglip2_naflex::VisionModel::new(&config.vision_config, vb.pp("vision_model"))?;
    println!("VisionModel loaded.");

    // Load reference outputs
    let ref_file = "/tmp/siglip2-practice-run/test_data_naflex/naflex_vision_reference.safetensors";
    let ref_buf = std::fs::read(ref_file)?;
    let ref_tensors = candle::safetensors::load_buffer(&ref_buf, &device)?;

    let pixel_values = ref_tensors
        .get("pixel_values")
        .ok_or_else(|| anyhow::anyhow!("missing pixel_values"))?;
    let pixel_attention_mask = ref_tensors
        .get("pixel_attention_mask")
        .ok_or_else(|| anyhow::anyhow!("missing pixel_attention_mask"))?;
    let spatial_shapes = ref_tensors
        .get("spatial_shapes")
        .ok_or_else(|| anyhow::anyhow!("missing spatial_shapes"))?;
    let expected_pooler = ref_tensors
        .get("vision_pooler_output")
        .ok_or_else(|| anyhow::anyhow!("missing vision_pooler_output"))?;

    println!("pixel_values shape: {:?}", pixel_values.shape());
    println!(
        "pixel_attention_mask shape: {:?}",
        pixel_attention_mask.shape()
    );
    println!(
        "spatial_shapes: {:?}",
        spatial_shapes.to_vec2::<i64>()?
    );
    println!("expected pooler shape: {:?}", expected_pooler.shape());

    println!("\nRunning candle VisionModel forward...");
    let candle_pooler =
        vision_model.forward(pixel_values, Some(pixel_attention_mask), Some(spatial_shapes), None)?;
    println!("candle pooler shape: {:?}", candle_pooler.shape());

    // Compare element-wise
    let diff = (candle_pooler.to_dtype(DType::F32)? - expected_pooler.to_dtype(DType::F32)?)?;
    let abs_diff = diff.abs()?;
    let max_diff = abs_diff.max_all()?.to_scalar::<f32>()?;
    let mean_diff = abs_diff.mean_all()?.to_scalar::<f32>()?;

    let exp_norm = expected_pooler.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    let cdl_norm = candle_pooler.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    println!(
        "\nExpected pooler L2 norms per row: {:?}",
        exp_norm.flatten_all()?.to_vec1::<f32>()?
    );
    println!(
        "Candle pooler L2 norms per row:   {:?}",
        cdl_norm.flatten_all()?.to_vec1::<f32>()?
    );

    println!("\n=== TENSOR-LEVEL COMPARISON ===");
    println!("max abs diff: {:.6e}", max_diff);
    println!("mean abs diff: {:.6e}", mean_diff);

    // First 5 values per row for both
    let exp_v: Vec<Vec<f32>> = expected_pooler.to_vec2()?;
    let cdl_v: Vec<Vec<f32>> = candle_pooler.to_vec2()?;
    println!("\n=== ROW 0 first 5 ===");
    for i in 0..5 {
        println!(
            "  [{}] expected={:.6} candle={:.6} diff={:.4e}",
            i,
            exp_v[0][i],
            cdl_v[0][i],
            (exp_v[0][i] - cdl_v[0][i]).abs()
        );
    }
    println!("\n=== ROW 1 first 5 ===");
    for i in 0..5 {
        println!(
            "  [{}] expected={:.6} candle={:.6} diff={:.4e}",
            i,
            exp_v[1][i],
            cdl_v[1][i],
            (exp_v[1][i] - cdl_v[1][i]).abs()
        );
    }

    // Cosine similarity per row (more robust than abs diff for embeddings)
    for row in 0..expected_pooler.dim(0)? {
        let e = expected_pooler.i((row,))?;
        let c = candle_pooler.i((row,))?;
        let dot = (&e * &c)?.sum_all()?.to_scalar::<f32>()?;
        let ne = e.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        let nc = c.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        let cos = dot / (ne * nc);
        println!("\nRow {} cosine similarity: {:.6}", row, cos);
    }

    Ok(())
}

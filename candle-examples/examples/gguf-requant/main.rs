//! Offline requantizer for the deploy artifact: rewrite a GGUF, optionally
//! requantizing selected tensors to a lower-bit dtype (e.g. tied lm_head Q6_K ->
//! Q4_K) and/or baking Q6_K matmul weights into the pre-packed Q6_Kx8 interleaved
//! layout (no load-time repack). Q4_K weights are left as-is - the runtime path
//! repacks/lane-rows them (#3643 + lanerow), so only Q6_K residuals are baked here.
//!
//!   gguf-requant --input m.gguf --output m-q6packed.gguf --pack
//!   gguf-requant --input m.gguf --output m-q4out.gguf --tensors token_embd,output --dtype q4k
use anyhow::Result;
use candle::quantized::{
    ggml_file, gguf_file, k_quants::BlockQ6K, repack, GgmlDType, GgmlType, QTensor,
};
use candle::Device;
use clap::Parser;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    input: String,
    #[arg(long)]
    output: String,
    /// Comma-separated substrings; any tensor whose name contains one is requantized
    /// to --dtype. Empty (default) = no requant: --pack alone only bakes Q6Kx8 and
    /// leaves embeddings/lm_head untouched. Pass e.g. --tensors token_embd,output to
    /// opt into requant (q6->q4 lm_head).
    #[arg(long, default_value = "")]
    tensors: String,
    /// Requant target dtype: q4k | q5k | q3k | q4_0.
    #[arg(long, default_value = "q4k")]
    dtype: String,
    /// Bake the pre-packed Q6_Kx8 layout for Q6_K matmul weights (attn_v/ffn_down).
    #[arg(long, default_value_t = false)]
    pack: bool,
}

fn parse_dtype(s: &str) -> Result<GgmlDType> {
    Ok(match s {
        "q4k" | "q4_k" => GgmlDType::Q4K,
        "q5k" | "q5_k" => GgmlDType::Q5K,
        "q3k" | "q3_k" => GgmlDType::Q3K,
        "q4_0" => GgmlDType::Q4_0,
        other => anyhow::bail!("unsupported dtype {other}"),
    })
}

// Matmul weight name suffixes eligible for packing.
const PACK_MATMUL: &[&str] = &[
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
];

fn is_packable_matmul(name: &str) -> bool {
    !name.contains("norm")
        && name.ends_with(".weight")
        && PACK_MATMUL.iter().any(|p| name.contains(p))
}

// Repack a Q6_K weight QTensor [n, k] into a pre-packed Q6_Kx8 QTensor (the residual
// attn_v/ffn_down weights Q4_K_M leaves at Q6_K). Q4_K is left for the runtime path.
fn pack_q6k_to_q6kx8(qt: &QTensor) -> Result<QTensor> {
    let (n, k) = qt.shape().dims2()?;
    anyhow::ensure!(qt.dtype() == GgmlDType::Q6K, "pack expects Q6_K input");
    anyhow::ensure!(n % 8 == 0 && k % 256 == 0, "pack needs n%8==0 && k%256==0");
    let nb = k / 256;
    let data = qt.data()?;
    let bs = std::mem::size_of::<BlockQ6K>();
    anyhow::ensure!(
        data.len() == n * nb * bs,
        "unexpected Q6_K byte count {} vs {}",
        data.len(),
        n * nb * bs
    );
    let mut rows = vec![BlockQ6K::zeros(); n * nb];
    // SAFETY: bytes came straight from a Q6_K QTensor (POD #[repr(C)] BlockQ6K) and the
    // length is an exact multiple of the block size; copy into an aligned Vec first.
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), rows.as_mut_ptr() as *mut u8, data.len());
    }
    let packed = repack::repack_q6k_weight(&rows, n, nb);
    let raw: &[u8] = unsafe {
        std::slice::from_raw_parts(
            packed.as_ptr() as *const u8,
            std::mem::size_of_val(packed.as_slice()),
        )
    };
    let tensor = ggml_file::qtensor_from_ggml(GgmlDType::Q6Kx8, raw, vec![n, k], &Device::Cpu)?;
    Ok(tensor)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::Cpu;
    let target = parse_dtype(&args.dtype)?;
    let pats: Vec<&str> = args.tensors.split(',').filter(|s| !s.is_empty()).collect();

    let mut f = std::fs::File::open(&args.input)?;
    let content = gguf_file::Content::read(&mut f).map_err(|e| e.with_path(&args.input))?;

    let names: Vec<String> = content.tensor_infos.keys().cloned().collect();
    let mut owned: Vec<(String, QTensor)> = Vec::with_capacity(names.len() + 1);
    let (mut before, mut after) = (0usize, 0usize);
    let mut has_output = false;

    for name in &names {
        if name == "output.weight" {
            has_output = true;
        }
        let qt = content.tensor(&mut f, name, &device)?;
        before += qt.storage_size_in_bytes();
        // Never requant norm weights (tiny F32, precision-critical).
        let is_norm = name.contains("norm");
        let qt = if !is_norm && pats.iter().any(|p| name.contains(p)) && qt.dtype() != target {
            let de = qt.dequantize(&device)?;
            let rq = QTensor::quantize(&de, target)?;
            println!("  requant {name}: {:?} -> {:?}", qt.dtype(), target);
            rq
        } else {
            qt
        };

        // Pack only Q6_K matmul weights -> Q6Kx8; Q4_K stays plain.
        let packable = args.pack
            && is_packable_matmul(name)
            && qt.dtype() == GgmlDType::Q6K
            && qt
                .shape()
                .dims2()
                .map(|(n, k)| n % 8 == 0 && k % 256 == 0)
                .unwrap_or(false);
        let qt = if packable {
            let packed = pack_q6k_to_q6kx8(&qt)?;
            println!("  pack    {name}: Q6K -> Q6Kx8");
            packed
        } else {
            qt
        };

        after += qt.storage_size_in_bytes();
        owned.push((name.clone(), qt));
    }

    // Tied embedding (no output.weight): emit a packed Q6Kx8 output.weight from a Q6_K
    // token_embd for the lm_head matmul. token_embd itself stays Q6_K (gather lookup).
    if args.pack && !has_output {
        if let Some((_, te)) = owned.iter().find(|(n, _)| n == "token_embd.weight") {
            if te.dtype() == GgmlDType::Q6K
                && te.shape().dims2().map(|(n, _)| n % 8 == 0).unwrap_or(false)
            {
                match pack_q6k_to_q6kx8(te) {
                    Ok(packed) => {
                        println!("  emit    output.weight (tied): Q6K -> Q6Kx8");
                        after += packed.storage_size_in_bytes();
                        owned.push(("output.weight".to_string(), packed));
                    }
                    Err(e) => println!("  skip tied output.weight pack: {e}"),
                }
            }
        }
    }

    let metadata: Vec<(&str, &gguf_file::Value)> = content
        .metadata
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect();
    let tensors: Vec<(&str, &QTensor)> = owned.iter().map(|(n, t)| (n.as_str(), t)).collect();

    let mut w = std::fs::File::create(&args.output)?;
    gguf_file::write(&mut w, &metadata, &tensors)?;
    println!(
        "wrote {} ({} tensors): weights {:.1}MB -> {:.1}MB",
        args.output,
        tensors.len(),
        before as f64 / 1e6,
        after as f64 / 1e6,
    );
    Ok(())
}
